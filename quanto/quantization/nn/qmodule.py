from abc import ABC
from inspect import signature
from typing import Any, Mapping

import torch

from ..qtensor import QTensor


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
        # Inspect the signature of the QModule creation method
        sig = signature(cls.from_module)
        cls_params = []
        for i, param in enumerate(sig.parameters.values()):
            if i == 0:
                # Skip the first parameter which is the module
                continue
            if param.kind == param.POSITIONAL_ONLY:
                raise ValueError(f"{cls}.from_module() can only have a single positional parameter")
            cls_params.append(param.name)
        _QMODULE_TABLE[module_cls] = [cls, cls_params]
        return cls

    return wrapper


def quantize_module(module, **kwargs):
    for cls in _QMODULE_TABLE:
        if isinstance(module, cls):
            qcls, qparams = _QMODULE_TABLE[cls]
            module_kwargs = {}
            for name in qparams:
                if name in kwargs:
                    module_kwargs[name] = kwargs[name]
            return qcls.from_module(module, **module_kwargs)
    return None


class QModuleMixin(ABC):
    class ScalesMixin(object):
        """Syntactic sugar class to manipulate scales as attributes.

        Its main purpose is to return None if a scale is not set without
        overloading getattr in the module class directly to avoid interfering
        with the existing overloads.
        """

        def __init__(self, m: torch.nn.Module):
            # Avoid recursion by using parent method
            object.__setattr__(self, "_m", m)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._m, name, None)

        def __setattr__(self, name: str, value: Any) -> None:
            m = self._m
            if getattr(m, name, None) is None:
                m.register_buffer(name, value)
            else:
                setattr(m, name, value)

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
        self.scales = self.ScalesMixin(self)
        # We need to register a state_dict pre-hook to initialize scales that have been dynamically recorded
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict: Mapping[str, Any], prefix: str, *args, **kwargs):
        def init_scale_from_dict(state_dict, prefix, scale_attr):
            scale_key = f"{prefix}{scale_attr}"
            if scale_key in state_dict:
                setattr(self.scales, scale_attr, state_dict[scale_key])

        # We need to update the shapes of the scale as they are not known at initialization
        for scale_attr in ["input", "output"]:
            init_scale_from_dict(state_dict, prefix, scale_attr)

    @classmethod
    def from_module(cls, module: torch.nn.Module):
        raise NotImplementedError

    def qweight(self):
        # Default implementation for QModules that do not quantize their weights
        return None

    def freeze(self):
        qweight = self.qweight()
        if qweight is not None:
            # Replace float weights by quantized weights
            self.weight = torch.nn.Parameter(self.qweight())

    @property
    def frozen(self):
        return isinstance(self.weight, QTensor)
