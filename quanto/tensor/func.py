from functools import partial

import torch

from .core import QTensor, qfallback


__all__ = ["get_qtensor_func", "register_qtensor_func"]


_QTENSOR_FUNC_TABLE = {}


def register_qtensor_func(funcs):
    """
    Used for registering a new __torch_dispatch__ function to QTensor.

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


@register_qtensor_func(
    [
        torch.nn.functional.cross_entropy,
        torch.nn.functional.cosine_similarity,
        torch.nn.functional.layer_norm,
        torch.nn.functional.log_softmax,
        torch.topk,
    ]
)
def unsupported_op(func, *args, **kwargs):
    return qfallback(func, *args, **kwargs)


@register_qtensor_func([torch.nn.functional.linear])
def linear(func, input, other, bias=None):
    if isinstance(bias, QTensor):
        return torch.ops.aten.linear(input, other, bias=bias)
    output = torch.matmul(input, other.t())
    if bias is not None:
        output = output + bias
    return output
