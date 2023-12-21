import timeit

import torch


def mm(a, b):
    return torch._int_mm(a, b)


A = torch.randint(1, 10, [2400, 2400]).type(torch.int8).cuda()
B = torch.randint(1, 10, [2400, 2400]).type(torch.int8).cuda()
it = 100

# Warmup (slow)
mm(A, B)
# Get a reference
print(timeit.Timer(lambda: mm(A, B)).timeit(it) / it)

cmm = torch.compile(mm, backend="inductor")
# First invocation will trigger the actual compilation
cmm(A, B)
# Now compare execution time
print(timeit.Timer(lambda: cmm(A, B)).timeit(it) / it)
