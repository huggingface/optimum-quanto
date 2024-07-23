import torch
from optimum.quanto.library.ext.cpp import ext as cpp_ext
from optimum.quanto.library.ext.cuda import ext as cuda_ext

assert cpp_ext.lib is not None
print("CPP extension is available")
if torch.cuda.is_available():
    assert cuda_ext.lib is not None
    print("CUDA extension is available")
