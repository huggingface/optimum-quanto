# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def prepare_inputs_for_generation(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values is not None:
        cache_length = past_length = past_key_values[0][0].shape[2]
        max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def setup(model_id: str, weights: str, activations: str, group_size: int = 64, version="GEMV_FAST"):
    if activations != "none":
        raise ValueError("Activation quantization is not supported by HQQ")
    if weights != "int4":
        raise ValueError("AWQ only supports int4 weights.")
    quant_config = {"zero_point": True, "q_group_size": group_size, "w_bit": 4, "version": version}
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)
    # We need to save otherwise it doesn't work
    quant_path = model_id.replace("/", "-") + f"_{group_size}_{version}"
    model.save_quantized(quant_path)
    # Reload model
    model = AutoAWQForCausalLM.from_quantized(quant_path)
    # Hack: force transformers 4.36.2 behaviour
    model.model.prepare_inputs_for_generation = prepare_inputs_for_generation
    # Hack because AWQ models are not transformers models
    model.device = next(model.parameters()).device
    return model, tokenizer
