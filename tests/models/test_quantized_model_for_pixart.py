import uuid
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import delete_repo

from optimum.quanto import QModuleMixin, is_diffusers_available, qint4, qint8


def quantized_model_for_pixart(qtype, exclude):
    from diffusers import PixArtTransformer2DModel

    from optimum.quanto import QuantizedPixArtTransformer2DModel

    init_dict = {
        "sample_size": 8,
        "num_layers": 1,
        "patch_size": 2,
        "attention_head_dim": 2,
        "num_attention_heads": 2,
        "in_channels": 4,
        "cross_attention_dim": 8,
        "out_channels": 8,
        "attention_bias": True,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 8,
        "norm_type": "ada_norm_single",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-6,
        "use_additional_conditions": False,
        "caption_channels": None,
    }
    torch.manual_seed(0)
    model = PixArtTransformer2DModel(**init_dict).eval()

    return QuantizedPixArtTransformer2DModel.quantize(model, weights=qtype, exclude=exclude)


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
        for (a_b_name, a_b), (b_b_name, b_b) in zip(a_m.named_buffers(), b_m.named_buffers()):
            assert a_b_name == b_b_name
            assert isinstance(a_b, torch.Tensor)
            assert torch.equal(a_b, b_b)

    # Compare model outputs
    hidden_states = torch.randn((1, 4, 8, 8))
    timesteps = torch.tensor([1.0])
    encoder_hidden_states = torch.randn((1, 8, 8))
    model_inputs = {
        "hidden_states": hidden_states,
        "timestep": timesteps,
        "encoder_hidden_states": encoder_hidden_states,
        "added_cond_kwargs": {"aspect_ratio": None, "resolution": None},
        "return_dict": False,
    }

    with torch.no_grad():
        output_a = a_model(**model_inputs)[0]
        output_b = b_model(**model_inputs)[0]
    assert torch.allclose(output_a, output_b, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not is_diffusers_available(), reason="requires diffusers")
@pytest.mark.parametrize("qtype", [qint4, qint8], ids=["qint4", "qint8"])
@pytest.mark.parametrize("exclude_proj_out", [True, False], ids=["without_proj_out", "with_proj_out"])
def test_quantized_model_for_pixart(qtype, exclude_proj_out):
    from optimum.quanto import QuantizedPixArtTransformer2DModel

    exclude = "proj_out" if exclude_proj_out else None
    quantized = quantized_model_for_pixart(qtype, exclude)
    with TemporaryDirectory() as tmpdir:
        quantized.save_pretrained(tmpdir)
        requantized = QuantizedPixArtTransformer2DModel.from_pretrained(tmpdir)

    compare_models(quantized, requantized)


@pytest.mark.skipif(not is_diffusers_available(), reason="requires diffusers")
@pytest.mark.parametrize("in_org", [True, False], ids=["org", "user"])
def test_push_to_hub(staging, in_org):
    from optimum.quanto import QuantizedPixArtTransformer2DModel

    identifier = uuid.uuid4()

    exclude = None
    quantized = quantized_model_for_pixart("qint8", exclude)
    repo_id = f"test-model-{identifier}"
    if in_org:
        quantized.push_to_hub(repo_id, token=staging["token"])
        hub_repo_id = f"{staging['user']}/{repo_id}"
    else:
        hub_repo_id = f"valid_org/{repo_id}-org"
        quantized.push_to_hub(hub_repo_id, token=staging["token"])

    requantized = QuantizedPixArtTransformer2DModel.from_pretrained(hub_repo_id, token=staging["token"])
    compare_models(quantized, requantized)

    delete_repo(hub_repo_id, token=staging["token"])
