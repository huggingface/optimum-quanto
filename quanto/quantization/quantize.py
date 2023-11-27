from torch.nn import ModuleList

from .nn import QModuleMixin, quantize_module


__all__ = ["quantize", "freeze"]


def _quantize_recursive(module):
    if isinstance(module, ModuleList):
        for i, m in enumerate(module):
            module[i] = _quantize_recursive(m)
        return module
    qmodule = quantize_module(module)
    if qmodule is None:
        for name, m in module.named_children():
            qmodule = _quantize_recursive(m)
            setattr(module, name, qmodule)
        return module
    return qmodule


def quantize(model, names=None):
    # Quantization happens in-place
    for name, m in model.named_children():
        if names is not None and name not in names:
            continue
        qmodule = _quantize_recursive(m)
        setattr(model, name, qmodule)
        qmodule.name = name


def freeze(model):
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            m.freeze()
