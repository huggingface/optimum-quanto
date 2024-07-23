# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from typing import Optional, Union

import torch

from ..tensor import (
    ActivationQBytesTensor,
    Optimizer,
    QBitsTensor,
    QTensor,
    qfloat8_e4m3fn,
    qint2,
    qint4,
    qtype,
    qtypes,
    quantize_activation,
    quantize_weight,
)


__all__ = ["QModuleMixin", "register_qmodule", "quantize_module"]


_QMODULE_TABLE = {}


def register_qmodule(module_cls):
    """
    Used for registering a new quantized module.

    The QModule must implement two abstract methods:

    - qcreate: class method to instantiate a new QModule from an nn.Module, without copying its weights,
    - forward: instance method for quantized inference.

    The code to register a new module looks like:

    ```
    @register_qmodule(<base torch.nn.Module>)
    class MyQModule(QModuleMixin, <base torch.nn.Module>):
        <implementation>

        @classmethod
        def qcreate(cls,
                    module: torch.nn.Module,
                    weights: Optional[qtype],
                    activations: Optional[qtype] = None,
                    optimizer: Optional[Optimizer] = None):
            ...

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            ...
    ```

    """

    def wrapper(cls):
        _QMODULE_TABLE[module_cls] = cls
        return cls

    return wrapper


def quantize_module(
    module,
    weights: Optional[Union[qtype, str]] = None,
    activations: Optional[Union[qtype, str]] = None,
    optimizer: Optional[Optimizer] = None,
):
    for cls in _QMODULE_TABLE:
        if isinstance(module, cls):
            qcls = _QMODULE_TABLE[cls]
            return qcls.from_module(module, weights=weights, activations=activations, optimizer=optimizer)
    return None


class QModuleMixin(ABC):
    def __init__(
        self,
        *args,
        weights: Optional[Union[qtype, str]] = None,
        activations: Optional[Union[qtype, str]] = None,
        optimizer: Optional[Optimizer] = None,
        quantize_input: Optional[bool] = False,
        **kwargs,
    ):
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
        if weights is not None and not isinstance(weights, qtype):
            weights = qtypes[weights]
        if activations is not None and not isinstance(activations, qtype):
            activations = qtypes[activations]
        self.weight_qtype = weights
        self.weight_group_size = None
        if self.weight_qtype in (qint2, qint4):
            out_features = self.weight.shape[0]
            in_features = self.weight.numel() // out_features
            group_size = 128
            if in_features > group_size:
                while in_features % group_size != 0 and group_size > 32:
                    group_size -= 32
                if in_features % group_size == 0:
                    self.weight_group_size = group_size
        self.activation_qtype = activations
        self._quantize_hooks = {}
        if activations is not None:
            if quantize_input:
                self._quantize_hooks["input"] = self.register_forward_pre_hook(self.quantize_input)
            self._quantize_hooks["output"] = self.register_forward_hook(self.quantize_output)
        self.optimizer = optimizer
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))

    def disable_output_quantization(self):
        if "output" in self._quantize_hooks:
            self._quantize_hooks["output"].remove()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.weight_qtype is None or not self.frozen:
            # Save standard weight Tensor
            destination[prefix + "weight"] = self.weight if keep_vars else self.weight.detach()
        else:
            # Save QTensor using dedicated method
            self.weight.save_to_state_dict(destination, prefix + "weight.", keep_vars)
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()
        destination[prefix + "input_scale"] = self.input_scale if keep_vars else self.input_scale.detach()
        destination[prefix + "output_scale"] = self.output_scale if keep_vars else self.output_scale.detach()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight_name = prefix + "weight"
        if self.weight_qtype is not None and weight_name not in state_dict:
            # The weight Tensor is not present because it is a flattened QTensor
            weight_prefix = weight_name + "."
            if self.weight_qtype.bits == 8:
                deserialized_weight = WeightQBytesTensor.load_from_state_dict(
                    state_dict,
                    weight_prefix,
                    qtype=self.weight_qtype,
                    axis=0,
                    size=self.weight.size(),
                    stride=self.weight.stride(),
                )
            else:
                deserialized_weight = QBitsTensor.load_from_state_dict(
                    state_dict,
                    weight_prefix,
                    qtype=self.weight_qtype,
                    axis=0,
                    group_size=self.weight_group_size,
                    size=self.weight.size(),
                    stride=self.weight.stride(),
                )
                deserialized_weight = deserialized_weight.optimize()

            assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
            if assign_to_params_buffers:
                self.weight = torch.nn.Parameter(deserialized_weight)
            else:
                if type(self.weight.data) is not type(deserialized_weight):
                    # Reloading frozen weights into unfrozen module: move to the correct device and force assignment
                    self.weight = torch.nn.Parameter(deserialized_weight.to(self.weight.device))
                else:
                    # FIXME: here we should copy frozen weights into frozen module, but this leads to grad error
                    self.weight = torch.nn.Parameter(deserialized_weight.to(self.weight.device))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        weights: Optional[qtype] = None,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        qmodule = cls.qcreate(module, weights, activations, optimizer)
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

        if self.weight_qtype == qfloat8_e4m3fn and self.activation_qtype is None:
            # Marlin FP8 kernel only supports per-tensor fp8 quantization.
            axis = None
        else:
            axis = 0

        return quantize_weight(
            self.weight,
            qtype=self.weight_qtype,
            axis=axis,
            group_size=self.weight_group_size,
            optimizer=self.optimizer,
            activation_qtype=self.activation_qtype,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def quantize_input(self, module: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
        input = input[0]
        if isinstance(input, ActivationQBytesTensor):
            if input.qtype != self.activation_qtype:
                raise ValueError(
                    "Models with heterogeneous quantized activations are not supported:"
                    f" expected {self.activation_qtype.name} input but got {input.qtype.name} instead."
                )
        else:
            input = quantize_activation(input, qtype=self.activation_qtype, scale=self.input_scale)
        return input

    def quantize_output(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return quantize_activation(output, qtype=self.activation_qtype, scale=self.output_scale)

    def freeze(self):
        qweight = self.qweight
        if qweight is not None:
            # Replace float weights by quantized weights
            self.weight = torch.nn.Parameter(qweight)

    @property
    def frozen(self):
        return isinstance(self.weight, QTensor)
