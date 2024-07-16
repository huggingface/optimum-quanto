from tempfile import TemporaryDirectory

import pytest
import torch

from optimum.quanto import QModuleMixin, is_transformers_available, qint4, qint8


def quantized_model_for_causal_lm(model_id, qtype, exclude):
    from transformers import AutoModelForCausalLM

    from optimum.quanto import QuantizedModelForCausalLM

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
            assert torch.equal(a_m.weight._data, b_m.weight._data)
            assert torch.equal(a_m.weight._scale, b_m.weight._scale)
        for (a_p_name, a_p), (b_p_name, b_p) in zip(a_m.named_parameters(), b_m.named_parameters()):
            assert a_p_name == b_p_name
            assert isinstance(a_p, torch.Tensor)
            assert torch.equal(a_p, b_p)
    # Compare model outputs
    inputs = torch.ones((1, 1), dtype=torch.int64)
    with torch.no_grad():
        output_a = a_model.forward(inputs)
        output_b = b_model.forward(inputs)
    assert torch.equal(output_a.last_hidden_state, output_b.last_hidden_state)
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
