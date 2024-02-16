# Quanto library python/pytorch operations

This folder contains the implementations of all `quanto_py::` operations.

This namespace corresponds to the default, python-only implementations of quanto operations.

The operations are defined in `library/ops.py`.

To provide an implementation for an operation, use the following syntax:

```python
@torch.library.impl("quanto_py::unpack", "default")
def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    ...
```

The implementation **must** support all device types. This is true if it
is a composition of built-in PyTorch operators.
