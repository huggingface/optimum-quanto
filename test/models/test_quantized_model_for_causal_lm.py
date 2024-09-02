import uuid
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import delete_repo

from optimum.quanto import QModuleMixin, is_transformers_available, qint4, qint8


def quantized_model_for_causal_lm(model_id, qtype, exclude, from_config=False):
    from transformers import AutoModelForCausalLM, OPTConfig

    from optimum.quanto import QuantizedModelForCausalLM

    if from_config:
        config = OPTConfig(
            **{
                "activation_dropout": 0.0,
                "activation_function": "relu",
                "architectures": ["OPTForCausalLM"],
                "attention_dropout": 0.0,
                "bos_token_id": 2,
                "do_layer_norm_before": True,
                "dropout": 0.1,
                "eos_token_id": 2,
                "ffn_dim": 32,
                "hidden_size": 8,
                "init_std": 0.02,
                "layerdrop": 0.0,
                "max_position_embeddings": 16,
                "model_type": "opt",
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "pad_token_id": 1,
                "prefix": "</s>",
                "torch_dtype": "float16",
                "use_cache": True,
                "vocab_size": 64,
                "word_embed_proj_dim": 8,
            }
        )
        model = AutoModelForCausalLM.from_config(config).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    return QuantizedModelForCausalLM.quantize(model, weights=qtype, exclude=exclude)


def compare_models(a_model, b_model):
    # Compare tensors
    for (a_name, a_m), (b_name, b_m) in zip(a_model.named_modules(), b_model.named_modules()):
        assert a_name == b_name
        if isinstance(a_m, QModuleMixin):
            assert isinstance(b_m, QModuleMixin)
        if isinstance(b_m, QModuleMixin):
            assert isinstance(a_m, QModuleMixin)
        if isinstance(a_m, QModuleMixin):
            assert torch.equal(a_m.weight, b_m.weight)
        for (a_p_name, a_p), (b_p_name, b_p) in zip(a_m.named_parameters(), b_m.named_parameters()):
            assert a_p_name == b_p_name
            assert isinstance(a_p, torch.Tensor)
            assert torch.equal(a_p, b_p)
    # Compare model outputs
    inputs = torch.ones((1, 1), dtype=torch.int64)
    with torch.no_grad():
        output_a = a_model.forward(inputs)
        output_b = b_model.forward(inputs)
    assert torch.equal(output_a.logits, output_b.logits)
    for i, a_key_value in enumerate(output_a.past_key_values):
        b_key_value = output_b.past_key_values[i]
        for j, a_value in enumerate(a_key_value):
            assert torch.equal(a_value, b_key_value[j])


@pytest.mark.skipif(not is_transformers_available(), reason="requires transformers")
@pytest.mark.parametrize("model_id", ["facebook/opt-125m"])
@pytest.mark.parametrize("qtype", [qint4, qint8], ids=["qint4", "qint8"])
@pytest.mark.parametrize("exclude_lm_head", [True, False], ids=["full", "no_lm_head"])
def test_quantized_model_for_causal_lm_base(model_id, qtype, exclude_lm_head):
    from optimum.quanto import QuantizedModelForCausalLM

    exclude = "lm_head" if exclude_lm_head else None
    quantized = quantized_model_for_causal_lm(model_id, qtype, exclude)
    with TemporaryDirectory() as tmpdir:
        quantized.save_pretrained(tmpdir)
        requantized = QuantizedModelForCausalLM.from_pretrained(tmpdir)

    compare_models(quantized, requantized)


@pytest.mark.skipif(not is_transformers_available(), reason="requires transformers")
def test_quantized_model_for_causal_lm_sharded():
    from optimum.quanto import QuantizedModelForCausalLM

    model_id = "facebook/opt-125m"
    qtype = qint4
    quantized = quantized_model_for_causal_lm(model_id, qtype, exclude=None)
    with TemporaryDirectory() as tmpdir:
        quantized.save_pretrained(tmpdir, max_shard_size="100MB")
        requantized = QuantizedModelForCausalLM.from_pretrained(tmpdir)

    compare_models(quantized, requantized)


@pytest.mark.skipif(not is_transformers_available(), reason="requires transformers")
@pytest.mark.parametrize("in_org", [True, False], ids=["org", "user"])
def test_causal_lm_base_push_to_hub(staging, in_org):
    from optimum.quanto import QuantizedModelForCausalLM

    identifier = uuid.uuid4()

    qtype = qint4
    exclude = None
    quantized = quantized_model_for_causal_lm(None, qtype, exclude, from_config=True)

    repo_id = f"test-model-{identifier}"
    if in_org:
        quantized.push_to_hub(repo_id, token=staging["token"])
        hub_repo_id = f"{staging['user']}/{repo_id}"
    else:
        hub_repo_id = f"valid_org/{repo_id}-org"
        quantized.push_to_hub(hub_repo_id, token=staging["token"])

    requantized = QuantizedModelForCausalLM.from_pretrained(hub_repo_id, token=staging["token"])

    compare_models(quantized, requantized)

    delete_repo(hub_repo_id, token=staging["token"])


@pytest.mark.skipif(not is_transformers_available(), reason="requires transformers")
@pytest.mark.parametrize("model_id", ["facebook/opt-125m"])
@pytest.mark.parametrize("qtype", [qint4, qint8], ids=["qint4", "qint8"])
def test_quantized_model_load_state_dict_non_strict(model_id, qtype):
    # see issue #278
    quantized = quantized_model_for_causal_lm(model_id, qtype, exclude=None)
    sd = quantized.state_dict()

    # delete a key used by both qint4 and qint8 from the state dict
    key = "model.decoder.layers.0.self_attn.k_proj.weight._scale"
    del sd[key]

    # strict loading should raise a RuntimeError, which is what PyTorch does in this case
    with pytest.raises(RuntimeError, match=key):
        quantized.load_state_dict(sd)

    # non-strict loading should not raise an errror
    result = quantized.load_state_dict(sd, strict=False)
    assert result.missing_keys == [key]
