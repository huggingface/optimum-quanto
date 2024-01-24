from torch.library import Library


# This file contains the definitions of all operations under torch.ops.quanto
quanto_ops = Library("quanto", "DEF")

quanto_ops.define("unpack(Tensor self, int bits) -> Tensor")
