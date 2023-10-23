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
