from functools import partial

import torch

from .core import QTensor


__all__ = ["get_qtensor_func", "register_qtensor_func"]


_QTENSOR_FUNC_TABLE = {}


def register_qtensor_func(funcs):
    """
    Used for registering a new __torch_dispatch__ function to QTensor

    The code to register a new function looks like:

    @register_qtensor_func(list_of_funcs)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(qfunc):
        for func in funcs:
            _QTENSOR_FUNC_TABLE[func] = partial(qfunc, func)

    return wrapper


def get_qtensor_func(func):
    return _QTENSOR_FUNC_TABLE.get(func, None)


def dequantize(*args):
    return [arg.dequantize() if isinstance(arg, QTensor) else arg for arg in args]


@register_qtensor_func([torch.nn.functional.log_softmax, torch.topk])
def unary_unsupported_op(func, t, *args, **kwargs):
    return func(t.dequantize(), *args, **kwargs)


@register_qtensor_func([torch.nn.functional.cross_entropy])
def plurary_unsupported_op(func, *args, **kwargs):
    return func(*dequantize(*args), **kwargs)


@register_qtensor_func([torch.matmul])
def matmul(func, input, other, out=None):
    if not isinstance(input, QTensor) or not isinstance(other, QTensor) or out is not None:
        return func(*dequantize(input, other), out=out)
    # For some reason the dispatched default chain of operations will lead to errors if
    # inputs are not contiguous. It is anyway better to overload directly the matmul
    # function, as we will likely have optimized implementations for integer inputs.
    # For now though, we cast int8 values to float32 to use float implementation.
    out_data = func(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)
