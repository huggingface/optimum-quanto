from torch.library import define


# This file contains the definitions of all operations under torch.ops.quanto

define("quanto::unpack", "(Tensor self, int bits) -> Tensor")
