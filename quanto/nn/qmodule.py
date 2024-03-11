from abc import ABC
from inspect import signature
from typing import Optional

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
        self.weight_qtype = weights
        self.weight_group_size = None
        if self.weight_qtype in (qint2, qint4):
            out_features = self.weight.shape[0]
            if out_features >= 128:
                group_size = self.weight.numel() // out_features
                while group_size >= 128 and group_size % 2 == 0:
                    group_size = group_size // 2
                self.weight_group_size = group_size
        self.activation_qtype = activations
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))

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

    @property
    def qweight(self):
        """Return the module quantized weight

        When the module is frozen or does not quantize its weight parameter, it simply
        returns the weight.
        When the module is not frozen, this property is required to add the dynamic quantization
        of the weight parameter to the graph and allow gradients to be propagated to the
        underlying weight float values.
        """
        if self.weight_qtype is None:
            # QModule that does not quantize its weights
            return None
        if isinstance(self.weight, QTensor):
            # Frozen QModule
            return self.weight
        # Quantize dynamically the weights per-axis
        if self.weight_qtype in (qint2, qint4):
            return QBitsTensor.quantize(
                self.weight, qtype=self.weight_qtype, axis=0, group_size=self.weight_group_size
            )
        elif isinstance(self.weight_qtype, qtype):
            wscale = absmax_scale(self.weight, axis=0)
            return QTensor.quantize(self.weight, qtype=self.weight_qtype, axis=0, group_size=None, scale=wscale)
        raise ValueError(f"Invalid quantized weights type {self.weight_qtype}")

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def maybe_requantize(t, scale):
            if t.qtype == self.activation_qtype and t.axis is None:
                return t
            return QTensor.quantize(
                t.dequantize(), qtype=self.activation_qtype, axis=None, group_size=None, scale=scale
            )

        if self.activation_qtype is not None and isinstance(input, QTensor):
            input = maybe_requantize(input, self.input_scale)
        output = self.qforward(input)
        if self.activation_qtype is not None:
            if isinstance(output, QTensor):
                output = maybe_requantize(output, self.output_scale)
            else:
                output = QTensor.quantize(
                    output, qtype=self.activation_qtype, axis=None, group_size=None, scale=self.output_scale
                )
        return output

    def freeze(self):
        qweight = self.qweight
        if qweight is not None:
            # Replace float weights by quantized weights
            self.weight = torch.nn.Parameter(qweight)

    @property
    def frozen(self):
        return isinstance(self.weight, QTensor)
