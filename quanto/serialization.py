import os
from typing import Dict, Union

import torch
from safetensors.torch import safe_open, save_file


def safe_save(state_dict: Dict[str, Union[torch.Tensor, str]], filename: Union[str, os.PathLike]):
    """Save a quantized model state_dict to `safetensors` format

    Args:
        state_dict (`Dict[str, Union[torch.Tensor, str]]`): a quantized model state_dict obtained from `model.state_dict())`.
        filename (`Union[str, os.PathLike`): the path to the serialized state_dict.
    """
    # Split state_dict into tensors and metadata
    tensors = {}
    metadata = {}
    for name, value in state_dict.items():
        if type(value) == torch.Tensor:
            tensors[name] = value
        else:
            metadata[name] = value
    save_file(tensors, filename, metadata)


def safe_load(filename: Union[str, os.PathLike]) -> Dict[str, Union[torch.Tensor, str]]:
    """Load a quantized model state_dict from a `safetensors` file

    Args:
        filename (`Union[str, os.PathLike`): the path to the serialized state_dict.

    Returns:
        `Dict[str, Union[torch.Tensor, str]]`: a state_dict object compatible with `model.load_state_dict()`.
    """
    with safe_open(filename, framework="pt") as f:
        # Recover first metadata
        state_dict = f.metadata()
        # Extract all tensors
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
        return state_dict
