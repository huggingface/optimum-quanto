# Quanto generic CUDA extension

Kernels in this extension can use both the C++ and CUDA syntax.

They can use any pytorch operation defined under `aten::` or `c10::`.

To add a new implementation for an operation defined in `library./ops.py`:

- add the corresponding `.cpp` or `.cu` file to the list of sources in `__init__.py`,
- add a binding to `pybind_module.cpp`,
- provide an implementation calling the binding in `__init__.py`.
