from abc import ABC

import torch


__all__ = ["QModuleMixin"]


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

    @classmethod
    def from_module(cls, module: torch.nn.Module):
        raise NotImplementedError

    def freeze(self):
        # Default implementation if the quantized Module does not have any quantized weights
        pass
