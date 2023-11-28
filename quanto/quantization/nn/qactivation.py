from typing import Any

from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode


__all__ = ["QActivationWrapper"]


class QActivationWrapper(Module):
    def __init__(self, m: Module):
        super().__init__()
        self.wrapped = m

    def __getattr__(self, name: str) -> Any:
        """If an attribute is not found in this class, look in the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__["_modules"]
            return getattr(modules["wrapped"], name)

    class ActivationWrapperMode(TorchDispatchMode):
        def __init__(self, m):
            self._m = m

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs is not None else {}
            # Here we will eventually align per-axis QTensor if needed
            output = func(*args, **kwargs)
            return output

    def forward(self, *args, **kwargs):
        with self.ActivationWrapperMode(self):
            return self.wrapped(*args, **kwargs)
