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

    def __str__(self):
        return f"quanto.{self.name}"

    def __hash__(self):
        return hash(str(self))


qint2 = qtype("qint2", is_floating_point=False, bits=2, dtype=torch.int8)
qint4 = qtype("qint4", is_floating_point=False, bits=4, dtype=torch.int8)
qint8 = qtype("qint8", is_floating_point=False, bits=8, dtype=torch.int8)
# Alias the float8 representation that has the better support and inference efficiency
qfloat8 = qtype("qfloat8", is_floating_point=True, bits=8, dtype=torch.float8_e4m3fn)
qfloat8_e4m3fn = qtype("qfloat8_e4m3fn", is_floating_point=True, bits=8, dtype=torch.float8_e4m3fn)
qfloat8_e5m2 = qtype("qfloat8_e5m2", is_floating_point=True, bits=8, dtype=torch.float8_e5m2)

# Convenience dict to get a dtype from its name
qtypes = {name: q for (name, q) in locals().items() if isinstance(q, qtype)}

__all__ = ["qtype", "qtypes"] + [str(name) for name in qtypes.keys()]
