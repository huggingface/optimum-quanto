# Quanto library extensions

This folder contains the implementations of all `quanto_ext::` operations.

This namespace corresponds to the device-specifc optimized implementations of quanto operations.

Implementations can be provided as part of:

- the generic C++ pytorch extension under `cpp`,
- the CUDA extension under `cuda`,
- the Metal Performance Shader extension under `mps`.

The operations are defined in `library/ops.py`.

To provide an implementation for specific device types, use the following syntax:

```python
@torch.library.impl("quanto_ext::unpack", ["CPU", "CUDA"])
def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    return ext().unpack(t, bits)
```

Please refer to each extension folder to see how to add the actual implementation.
