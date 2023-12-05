from abc import ABC
from typing import Any, Mapping

import torch


__all__ = ["QModuleMixin", "register_qmodule", "quantize_module"]


_QMODULE_TABLE = {}


def register_qmodule(module_cls):
    """
    Used for registering a new quantized module

    The code to register a new module looks like:

    @register_qmodule(<base torch.nn.Module>)
    class MyQModule(QModuleMixin, <base torch.nn.Module>):
        <implementation>
    """

    def wrapper(cls):
        _QMODULE_TABLE[module_cls] = cls
        return cls

    return wrapper


def quantize_module(module):
    for cls in _QMODULE_TABLE:
        if isinstance(module, cls):
            return _QMODULE_TABLE[cls].from_module(module)
    return None


class QModuleMixin(ABC):
    def __init__(self, *args, **kwargs):
        # The tests below are meant to help people writing their own quantized Module class
        mro = self.__class__.__mro__
        if torch.nn.Module not in mro:
            raise TypeError("Quantized modules must inherit from a torch.nn.Module class")
        if mro.index(__class__) > mro.index(torch.nn.Module):
            raise TypeError(
                "QModuleMixin must be placed before any torch.nn.Module class in quantized module inheritance."
            )
        # This will setup the torch.nn.Module
        super().__init__(*args, **kwargs)
        self.record_scale("in_scale", torch.ones((), dtype=torch.float32))
        self.record_scale("out_scale", torch.ones((), dtype=torch.float32))
        # We need to register a state_dict pre-hook to initialize scales that have been dynamically recorded
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def record_scale(self, scale_attr, new_scale):
        if getattr(self, scale_attr, None) is None:
            self.register_buffer(scale_attr, new_scale)
        else:
            setattr(self, scale_attr, new_scale)

    def _load_state_dict_pre_hook(self, state_dict: Mapping[str, Any], prefix: str, *args, **kwargs):
        def init_scale_from_dict(state_dict, prefix, scale_attr):
            scale_key = f"{prefix}{scale_attr}"
            if scale_key in state_dict:
                self.record_scale(scale_attr, state_dict[scale_key])

        # We need to update the shapes of the scale as they are not known at initialization
        for scale_attr in ["in_scale", "out_scale"]:
            init_scale_from_dict(state_dict, prefix, scale_attr)

    @classmethod
    def from_module(cls, module: torch.nn.Module):
        raise NotImplementedError

    def freeze(self):
        # Default implementation if the quantized Module does not have any quantized weights
        pass
