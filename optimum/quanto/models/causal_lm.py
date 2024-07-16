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

import json
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import torch
from safetensors import safe_open

from ..nn import QModuleMixin
from ..quantize import Optimizer, freeze, qtype, quantization_map, quantize, requantize
from . import is_transformers_available


if not is_transformers_available():
    raise ImportError("QuantizedModelForCausalLM requires the transformers library")

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import get_checkpoint_shard_files, load_state_dict
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, is_accelerate_available


__all__ = ["QuantizedModelForCausalLM"]


class ShardedStateDict(Mapping):
    """A pytorch state_dict stored in multiple safetensors files

    This class implements the `collections.abc.Mapping` interface.
    It can be passed to `torch.nn.Module.load_state_dict()` to recursively
    load the module tensors.
    """

    def __init__(self, base_dir: str, tensor_index: Dict[str, str]):
        self._base_dir = base_dir
        self._index = tensor_index
        self._handles = {}

    def __iter__(self):
        yield from self._index

    def __len__(self):
        return self._index.__len__()

    def __getitem__(self, key: Any) -> Any:
        filename = self._index.__getitem__(key)
        if filename not in self._handles:
            f = safe_open(os.path.join(self._base_dir, filename), framework="pytorch")
            self._handles[filename] = f
        f = self._handles[filename]
        return f.get_tensor(key)

    def __contains__(self, key: object) -> bool:
        return self._index.__contains__(key)

    def keys(self):
        return self._index.keys()


class QuantizedModelForCausalLM:

    BASE_NAME = "quanto"

    def __init__(self, model: torch.nn.Module):
        if not isinstance(model, PreTrainedModel) or len(quantization_map(model)) == 0:
            raise ValueError("The source model must be a quantized transformers model.")
        self._wrapped = model

    def __getattr__(self, name: str) -> Any:
        """If an attribute is not found in this class, look in the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__["_wrapped"]
            return getattr(wrapped, name)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    @staticmethod
    def _qmap_name():
        return f"{QuantizedModelForCausalLM.BASE_NAME}_qmap.json"

    @classmethod
    def quantize(
        cls,
        model: PreTrainedModel,
        weights: Optional[Union[str, qtype]] = None,
        activations: Optional[Union[str, qtype]] = None,
        optimizer: Optional[Optimizer] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
    ):
        """Quantize the specified model

        By default, each layer of the model will be quantized if is quantizable.

        If include patterns are specified, the layer name must match one of them.

        If exclude patterns are specified, the layer must not match one of them.

        Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
        https://docs.python.org/3/library/fnmatch.html for more details.

        Note: quantization happens in-place and modifies the original model.

        Note that the resulting quantized model will be frozen: if you wish to do
        quantization-aware training then you should use `optimum.quanto.quantize` instead,
        and call `optimum.quanto.freeze` only after the training.

        Args:
            model (`PreTrainedModel`): the model to quantize.
            weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
            activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
            include (`Optional[Union[str, List[str]]]`):
                Patterns constituting the allowlist. If provided, layer names must match at
                least one pattern from the allowlist.
            exclude (`Optional[Union[str, List[str]]]`):
                Patterns constituting the denylist. If provided, layer names must not match
                any patterns from the denylist.
        """
        if not isinstance(model, PreTrainedModel):
            raise ValueError("The source model must be a transformers model.")
        quantize(
            model, weights=weights, activations=activations, optimizer=optimizer, include=include, exclude=exclude
        )
        freeze(model)
        return cls(model)

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        if not is_accelerate_available():
            raise ValueError("Reloading a quantized transformers model requires the accelerate library.")
        from accelerate import init_empty_weights

        if os.path.isdir(model_name_or_path):
            # Look for a quantization map
            qmap_path = os.path.join(model_name_or_path, cls._qmap_name())
            if not os.path.exists(qmap_path):
                raise ValueError(f"No quantization map found in {model_name_or_path}: is this a quantized model ?")
            with open(qmap_path, "r", encoding="utf-8") as f:
                qmap = json.load(f)
            # Create an empty model
            config = AutoConfig.from_pretrained(model_name_or_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            # Look for the index of a sharded checkpoint
            checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
            if os.path.exists(checkpoint_file):
                # Convert the checkpoint path to a list of shards
                checkpoint_file, sharded_metadata = get_checkpoint_shard_files(model_name_or_path, checkpoint_file)
                # Create a mapping for the sharded safetensor files
                state_dict = ShardedStateDict(model_name_or_path, sharded_metadata["weight_map"])
            else:
                # Look for a single checkpoint file
                checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_NAME)
                if not os.path.exists(checkpoint_file):
                    raise ValueError(f"No safetensor weights found in {model_name_or_path}.")
                # Get state_dict from model checkpoint
                state_dict = load_state_dict(checkpoint_file)
            # Requantize and load quantized weights from state_dict
            requantize(model, state_dict=state_dict, quantization_map=qmap)
            if getattr(model.config, "tie_word_embeddings", True):
                # Tie output weight embeddings to input weight embeddings
                # Note that if they were quantized they would NOT be tied
                model.tie_weights()
            # Set model in evaluation mode as it is done in transformers
            model.eval()
            return cls(model)
        else:
            raise NotImplementedError("Reloading quantized models directly from the hub is not supported yet.")

    def save_pretrained(self, save_directory: Union[str, os.PathLike], max_shard_size: Union[int, str] = "5GB"):

        model = self._wrapped
        if getattr(model.config, "tie_word_embeddings", True):
            # The original model had tied embedding inputs and outputs
            if isinstance(model.get_input_embeddings(), QModuleMixin) or isinstance(
                model.get_output_embeddings(), QModuleMixin
            ):
                # At least one of the two is quantized, so they are not tied anymore
                model.config.tie_word_embeddings = False
        self._wrapped.save_pretrained(save_directory, max_shard_size=max_shard_size, safe_serialization=True)
        # Save quantization map to be able to reload the model
        qmap_name = os.path.join(save_directory, self._qmap_name())
        qmap = quantization_map(self._wrapped)
        with open(qmap_name, "w", encoding="utf8") as f:
            json.dump(qmap, f, indent=4)
