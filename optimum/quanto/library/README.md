# Quanto operations library

This contains the `quanto::` operations, available in python under `torch.ops.quanto`.

To add a new operation:

- add a definition for the operation in `library/ops.py`,
- provide a default implementation using pytorch operators only under `library/python`,
- provide optimized kernels for all devices under `library/ext`.
