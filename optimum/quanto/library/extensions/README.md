# Quanto library extensions

This folder contains device-specific `quanto::` operations.

Implementations can be provided as part of:

- the generic C++ pytorch extension under `cpp`,
- the CUDA extension under `cuda`,
- the Metal Performance Shader extension under `mps`.


To provide a device-specific implementation of an operation that already has a default implementation (such as unpack), use the following syntax:

```python
@torch.library.impl("quanto::unpack", ["CPU", "CUDA"])
def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    return ext.unpack(t, bits)
```

To declare a new device-specific operation, you need to add it to the library:

```python
torch.library.define(
    "quanto::gemm_f16i4",
    "(Tensor input,"
    " Tensor other,"
    " Tensor other_scale,"
    " Tensor other_shift,"
    " int group_size)"
    " -> Tensor",
)
```

Then you can provide its implementation:

```python
@torch.library.impl("quanto::gemm_f16i4", ["CUDA"])
def gemm_f16i4(
    input: torch.Tensor,
    other: torch.Tensor,
    scales: torch.Tensor,
    shift: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    ...
```


Please refer to each extension folder for examples.
