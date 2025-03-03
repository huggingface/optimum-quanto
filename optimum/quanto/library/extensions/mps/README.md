# Quanto Metal Performance Shaders extension

To add a new implementation for an operation defined in `library./ops.py`:

- add the corresponding `.mm` file to the list of sources in `__init__.py`,
- add a binding to `pybind_module.cpp`,
- provide an implementation calling the binding in `__init__.py`.

Note: torch JIT extensions for MPS requires the xcode command-line tools.
