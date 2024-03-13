#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

inline  unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
#define BLOCK_SIZE 256

using namespace at;


static torch::Tensor allocate_output(const torch::Tensor& input, int bits) {
    int n_packed = 8 / bits;
    auto output_shape = input.sizes().vec();
    output_shape[0] = output_shape[0] * n_packed;
    return torch::empty(output_shape, input.options());
}

__global__ void unpack_4bit_kernel(unsigned char* input, unsigned char* output, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n) return;

	output[i]     = (input[i] & 0x0F);
	output[i + n] = (input[i] & 0xF0) >> 4;
}

static torch::Tensor unpack_4bit(const torch::Tensor& input){

	auto output = allocate_output(input, 4);

    const auto numel = input.numel();
	int blocks = cdiv(numel, BLOCK_SIZE);
	unpack_4bit_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<unsigned char>(),
        output.data_ptr<unsigned char>(),
        numel
    );

	C10_CUDA_KERNEL_LAUNCH_CHECK();

	return output;
}

__global__ void unpack_2bit_kernel(unsigned char* input, unsigned char* output, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n) return;

	output[i]       = (input[i] & 0x03);
	output[i + n]   = (input[i] & 0x0C) >> 2;
	output[i + n*2] = (input[i] & 0x30) >> 4;
	output[i + n*3] = (input[i] & 0xC0) >> 6;
}

static torch::Tensor unpack_2bit(const torch::Tensor& input){

	auto output = allocate_output(input, 2);

    const auto numel = input.numel();
	int blocks = cdiv(numel, BLOCK_SIZE);
	unpack_2bit_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<unsigned char>(),
        output.data_ptr<unsigned char>(),
        numel
    );

	C10_CUDA_KERNEL_LAUNCH_CHECK();

	return output;
}

torch::Tensor unpack(torch::Tensor &t, int bits) {
    TORCH_CHECK(t.scalar_type() == torch::kUInt8, "Unsupported data type: ", t.scalar_type());
    TORCH_CHECK(t.device().is_cuda(), "t must be a CUDA tensor.");
    TORCH_CHECK(t.is_contiguous(), "t must be contiguous.");
    switch(bits) {
      case 4:
        return unpack_4bit(t);
      case 2:
        return unpack_2bit(t);
      default:
        throw std::invalid_argument("Can only unpack 2-bit or 4-bit tensors.");
    }
}
