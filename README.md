# pytorch quantization toolkit

Uses a torch.Tensor subclass QuantizedTensor to dispatch aten base operations to operations using integer ops.

All operations accept QuantizedTensor with int8 data.

Most arithmetic operations return a QuantizedTensor with int32 data.

Uses quantized modules to:

- store quantized weights,
- gather input and output scales to rescale QuantizedTensor int32 data to int8.

For now only Linear is quantizable.

Eventually, the produced quantized graph should be passed to a specific inductor backend to fuse rescale into the previous operation.

Examples of fused operations can be found in https://github.com/Guangxuan-Xiao/torch-int.
