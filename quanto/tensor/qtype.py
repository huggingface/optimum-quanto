from dataclasses import dataclass

import torch


__all__ = ["qtype", "qint2", "qint4", "qint8", "qint16", "qint32", "qfloat8", "qfloat8_e4m3fn", "qfloat8_e5m2"]


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
qint16 = qtype("qint16", is_floating_point=False, bits=16, dtype=torch.int16)
qint32 = qtype("qint32", is_floating_point=False, bits=32, dtype=torch.int32)
# Alias the float8 representation that has the better support and inference efficiency
qfloat8 = qtype("qfloat8", is_floating_point=True, bits=8, dtype=torch.float8_e4m3fn)
qfloat8_e4m3fn = qtype("qfloat8_e4m3fn", is_floating_point=True, bits=8, dtype=torch.float8_e4m3fn)
qfloat8_e5m2 = qtype("qfloat8_e5m2", is_floating_point=True, bits=8, dtype=torch.float8_e5m2)
