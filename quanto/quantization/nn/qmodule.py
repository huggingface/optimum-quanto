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
        # We can now add our Module attributes
        self.register_buffer("in_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("out_scale", torch.ones((), dtype=torch.float32))
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict: Mapping[str, Any], prefix: str, *args, **kwargs):
        # We need to update the shapes of the scale as they are not known at initialization
        self.in_scale.resize_(state_dict[f"{prefix}in_scale"].size())
        self.out_scale.resize_(state_dict[f"{prefix}out_scale"].size())

    @classmethod
    def from_module(cls, module: torch.nn.Module):
        raise NotImplementedError

    def freeze(self):
        # Default implementation if the quantized Module does not have any quantized weights
        pass
