import os
from typing import Dict, Union

import torch
from safetensors.torch import safe_open, save_file


def safe_save(state_dict: Dict[str, Union[torch.Tensor, str]], filename: Union[str, os.PathLike]):
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
    with safe_open(filename, framework="pt") as f:
        # Recover first metadata
        state_dict = f.metadata()
        # Extract all tensors
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
        return state_dict
