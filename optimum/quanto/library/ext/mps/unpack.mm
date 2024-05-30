// Copyright 2024 The HuggingFace Team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "unpack.h"
#include <torch/extension.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Defines a Metal custom kernel to mask and shift a buffer element-wise.
static char *MASK_AND_SHIFT = R"MPS_MASK&SHIFT(
#include <metal_stdlib>
using namespace metal;

[[host_name("mask_and_rshift")]]
kernel void mask_and_rshift(constant uint8_t*     input  [[buffer(0)]],
                            device   uint8_t*     output [[buffer(1)]],
                            constant uint8_t&     mask   [[buffer(2)]],
                            constant int&       shift  [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    output[index] = (input[index] & mask) >> shift;
}

)MPS_MASK&SHIFT";

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor& mask_and_shift(const torch::Tensor& input, torch::Tensor& output, uint8_t mask, int shift) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int num_threads = input.numel();

        // Load the custom mask and shift shader.
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:MASK_AND_SHIFT]
                                  options:nil
                                  error:&error];
        TORCH_CHECK(library, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        id<MTLFunction> kernel = [library newFunctionWithName:[NSString stringWithUTF8String:"mask_and_rshift"]];
        TORCH_CHECK(kernel, "Failed to create function state object for mask_and_rshift");

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:kernel error:&error];
        TORCH_CHECK(pso, error.localizedDescription.UTF8String);

        // This is required if torch already encoded something in the command buffer
        torch::mps::synchronize();

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
        TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serial_queue = torch::mps::get_dispatch_queue();

        dispatch_sync(serial_queue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
            TORCH_CHECK(compute_encoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [compute_encoder setComputePipelineState:pso];
            [compute_encoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
            [compute_encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:1];
            [compute_encoder setBytes:&mask length:sizeof(uint8_t) atIndex:2];
            [compute_encoder setBytes:&shift length:sizeof(int) atIndex:3];

            MTLSize grid_size = MTLSizeMake(num_threads, 1, 1);

            // Calculate a thread group size.
            NSUInteger thread_group_size = pso.maxTotalThreadsPerThreadgroup;
            if (thread_group_size > num_threads) {
                thread_group_size = num_threads;
            }
            MTLSize mtl_size = MTLSizeMake(thread_group_size, 1, 1);

            // Encode the compute command.
            [compute_encoder dispatchThreads:grid_size
                      threadsPerThreadgroup:mtl_size];

            [compute_encoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });

        torch::mps::synchronize();
    }

    return output;
}

torch::Tensor unpack_4bit(const torch::Tensor &input) {

    torch::Tensor output = torch::empty_like(input);
    mask_and_shift(input, output, 0x0F, 0);
    torch::Tensor output1 = torch::empty_like(input);
    mask_and_shift(input, output1, 0xF0, 4);
    return torch::cat({output, output1}, 0);
}

torch::Tensor unpack_2bit(const torch::Tensor &input) {

    torch::Tensor output = torch::empty_like(input);
    mask_and_shift(input, output, 0x03, 0);
    torch::Tensor output1 = torch::empty_like(input);
    mask_and_shift(input, output1, 0x0C, 2);
    torch::Tensor output2 = torch::empty_like(input);
    mask_and_shift(input, output2, 0x30, 4);
    torch::Tensor output3 = torch::empty_like(input);
    mask_and_shift(input, output3, 0xC0, 6);
    return torch::cat({output, output1, output2, output3}, 0);
}

// C++ op dispatching the Metal unpack operation.
torch::Tensor unpack(const torch::Tensor &input, int bits) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Check the supported data types for soft shrink.
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "Unsupported data type: ", input.scalar_type());

    switch(bits) {
      case 4:
        return unpack_4bit(input);
      case 2:
        return unpack_2bit(input);
      default:
        throw std::invalid_argument("Can only unpack 2-bit or 4-bit tensors.");
    }
}
