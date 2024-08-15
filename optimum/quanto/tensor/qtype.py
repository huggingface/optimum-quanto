# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch


@dataclass
class qtype:
    """A quantized type class mimicking torch dtype"""

    name: str
    is_floating_point: bool
    bits: int
    # This defines the storage dtype
    dtype: torch.dtype
    qmin: float
    qmax: float

    def __str__(self):
        return f"quanto.{self.name}"

    def __hash__(self):
        return hash(str(self))


# Integer qtypes


def qint(bits):
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    return qtype(f"qint{bits}", is_floating_point=False, bits=bits, dtype=torch.int8, qmin=qmin, qmax=qmax)


qint2 = qint(2)
qint4 = qint(4)
qint8 = qint(8)

# Float qtypes


def qfloat(dtype: torch.dtype):
    finfo = torch.finfo(dtype)
    qmin = finfo.min
    qmax = finfo.max
    return qtype(f"q{finfo.dtype}", is_floating_point=True, bits=8, dtype=dtype, qmin=qmin, qmax=qmax)


qfloat8_e4m3fn = qfloat(torch.float8_e4m3fn)
qfloat8_e4m3fnuz = qfloat(torch.float8_e4m3fnuz)
qfloat8_e5m2 = qfloat(torch.float8_e5m2)

# Alias the float8 representation that has the better support and inference efficiency
qfloat8 = qfloat8_e4m3fn

# Convenience dict to get a dtype from its name
qtypes = {name: q for (name, q) in locals().items() if isinstance(q, qtype)}

__all__ = ["qtype", "qtypes"] + [str(name) for name in qtypes.keys()]
