import timeit

import torch


def mm(A, B):
    return torch._int_mm(A, B)


A = torch.randint(1, 10, [2400, 2400]).type(torch.int8).cuda()
B = torch.randint(1, 10, [2400, 2400]).type(torch.int8).cuda()
print(A)

# Warmup (slow)
mm(A, B)
# Average on several calls
it = 100
print(timeit.Timer(lambda: mm(A, B)).timeit(it) / it)

# Warmup (slow)
A = A.type(torch.float16)
B = B.type(torch.float16)
torch.matmul(A, B)
# Average on several calls
it = 100
print(timeit.Timer(lambda: torch.matmul(A, B)).timeit(it) / it)
