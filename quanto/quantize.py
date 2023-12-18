from .nn import QModuleMixin, quantize_module


__all__ = ["quantize", "freeze"]


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


def quantize(model, modules=None, **kwargs):
    # Quantization happens in-place
    for name, m in model.named_modules():
        if modules is not None and m not in modules:
            continue
        qmodule = quantize_module(m, **kwargs)
        if qmodule is not None:
            set_module_by_name(model, name, qmodule)
            qmodule.name = name
            for name, param in m.named_parameters():
                # Save device memory by clearing parameters
                setattr(m, name, None)
                del param


def freeze(model):
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            m.freeze()
