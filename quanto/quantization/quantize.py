import torch

from .nn import QLinear


def set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        next_module = parent_module
        for idx in range(len(module_names) - 1):
            next_module_name = module_names[idx]
            if next_module_name.isnumeric():
                next_module = next_module[int(next_module_name)]
            else:
                next_module = getattr(next_module, next_module_name)
        setattr(next_module, module_names[-1], child_module)


def quantize(model):
    # Quantization happens in-place
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            qlinear = QLinear.from_module(m)
            set_module_by_name(model, name, qlinear)
