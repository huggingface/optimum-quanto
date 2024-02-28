from abc import ABC
from inspect import signature
from typing import Any, Mapping, Optional

import torch

from ..tensor import QBitsTensor, QTensor, absmax_scale, qint2, qint4, qtype


__all__ = ["QModuleMixin", "register_qmodule", "quantize_module"]


_QMODULE_TABLE = {}


def register_qmodule(module_cls):
    """
    Used for registering a new quantized module.

    The QModule must implement two abstract methods:

    - qcreate: class method to instantiate a new QModule from an nn.Module, without copying its weights,
    - qforward: instance method for quantized inference.

    The code to register a new module looks like:

    ```
    @register_qmodule(<base torch.nn.Module>)
    class MyQModule(QModuleMixin, <base torch.nn.Module>):
        <implementation>

        @classmethod
        def qcreate(cls,
                    module: torch.nn.Module,
                    weights: Optional[qtype],
                    activations: Optional[qtype] = None):
            ...

        def qforward(self, input: torch.Tensor) -> torch.Tensor:
            ...
    ```

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
    def __init__(self, *args, weights: Optional[qtype] = None, activations: Optional[qtype] = None, **kwargs):
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
        self.weights = weights
        self.activations = activations
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))
        # We need to register a state_dict pre-hook to reset scales because their actual shapes and dtype are yet unknown
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict: Mapping[str, Any], prefix: str, *args, **kwargs):
        def init_scale_from_dict(state_dict, prefix, scale_attr):
            scale_key = f"{prefix}{scale_attr}"
            if scale_key in state_dict:
                setattr(self, scale_attr, state_dict[scale_key])

        # We need to update the shapes and dtypes of the scales as they are not known at initialization
        for scale_attr in ["input_scale", "output_scale"]:
            init_scale_from_dict(state_dict, prefix, scale_attr)

    @classmethod
    def from_module(
        cls, module: torch.nn.Module, weights: Optional[qtype] = None, activations: Optional[qtype] = None
    ):
        qmodule = cls.qcreate(module, weights, activations)
        if qmodule is None:
            return None
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            if module.bias is not None:
                qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    @classmethod
    def qcreate(cls, module: torch.nn.Module, weights: Optional[qtype], activations: Optional[qtype] = None):
        raise NotImplementedError

    def qweight(self):
        if self.weights is None:
            # QModule that does not quantize its weights
            return None
        if isinstance(self.weight, QTensor):
            # Frozen QModule
            return self.weight
        # Quantize dynamically the weights per-axis
        if self.weights in (qint2, qint4):
            return QBitsTensor.quantize(self.weight, qtype=self.weights, axis=0)
        elif isinstance(self.weights, qtype):
            wscale = absmax_scale(self.weight, axis=0)
            return QTensor.quantize(self.weight, qtype=self.weights, axis=0, scale=wscale)
        raise ValueError(f"Invalid quantized weights type {self.weights}")

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def maybe_requantize(t, scale):
            if t.qtype == self.activations and t.axis is None:
                return t
            return QTensor.quantize(t.dequantize(), qtype=self.activations, axis=None, scale=scale)

        if self.activations is not None and isinstance(input, QTensor):
            input = maybe_requantize(input, self.input_scale)
        output = self.qforward(input)
        if self.activations is not None:
            if isinstance(output, QTensor):
                output = maybe_requantize(output, self.output_scale)
            else:
                output = QTensor.quantize(output, qtype=self.activations, axis=None, scale=self.output_scale)
        return output

    def freeze(self):
        qweight = self.qweight()
        if qweight is not None:
            # Replace float weights by quantized weights
            self.weight = torch.nn.Parameter(self.qweight())

    @property
    def frozen(self):
        return isinstance(self.weight, QTensor)
