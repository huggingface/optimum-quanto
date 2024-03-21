import torch
from hqq.core.quantize import BaseQuantizeConfig
from hqq.engine.hf import HQQModelForCausalLM
from transformers import AutoTokenizer


def setup(model_id: str, weights: str, activations: str, device: torch.device, group_size: int = 64):
    if activations != "none":
        raise ValueError("Activation quantization is not supported by HQQ")
    if weights == "int4":
        quant_config = BaseQuantizeConfig(nbits=4, group_size=group_size)
    elif weights == "int8":
        quant_config = BaseQuantizeConfig(nbits=8, group_size=group_size)
    else:
        raise ValueError("HQQ only supports int4 and int8 weights.")
    # Load model
    model = HQQModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    # Quantize
    model.quantize_model(quant_config=quant_config, compute_dtype=torch.float16, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer
