#include "quantize.h"
#include <torch/extension.h>


template <typename T>
torch::Tensor quantize_symmetric_per_tensor(const torch::Tensor& input, const torch::Tensor& scale) {
    torch::Tensor output = torch::empty_like(input, c10::TensorOptions(c10::kChar).dtype(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto qdata = reinterpret_cast<int8_t*>(output.data_ptr());
    auto numel = input.numel();
    const T* const data = input.data_ptr<T>();
    float float_scale = scale.data_ptr<T>()[0];
    float inv_scale = float_scale == 0 ? 1.0f : 1.0f / float_scale;
    for (const auto i : c10::irange(numel)) {
        int64_t qvalue = lrintf(std::nearbyint(data[i] * inv_scale));
        qvalue = std::max<int64_t>(-127LL, std::min<int64_t>(qvalue, 127LL));
        qdata[i] = static_cast<int8_t>(qvalue);
    }
    return output;
}


int get_scale_axis(const torch::Tensor& scale) {
    int axis = -1;
    auto scale_dims = scale.sizes();
    for (int i = 0; i < scale_dims.size(); ++i) {
        if (scale_dims[i] != 1) {
            axis = i;
        }
    }
    return axis;
}


torch::Tensor quantize_symmetric_char(const torch::Tensor& input,
                                      const torch::Tensor& scale) {
    int axis = get_scale_axis(scale);
    if (axis == -1) {
        auto scale_dtype = scale.dtype();
        if (scale_dtype == at::ScalarType::Float) {
            return quantize_symmetric_per_tensor<float>(input, scale);
        }
        if (scale_dtype == at::ScalarType::Half) {
            return quantize_symmetric_per_tensor<at::Half>(input, scale);
        }
        TORCH_CHECK(false, "Unsupported scale dtype:", scale_dtype)
    }
    TORCH_CHECK(false, "symmetric per-axis is not supported")
}


torch::Tensor quantize_symmetric(const torch::Tensor& input,
                                 const torch::Tensor& scale,
                                 at::ScalarType dtype) {
    bool scalar_scale = (scale.sizes().size() == 0);
    bool broadcastable_scale = (input.sizes().size() == scale.sizes().size());
    TORCH_CHECK(scalar_scale || broadcastable_scale,
                "Quantization scale must be scalar or broadcastable to the base tensor.")
    TORCH_CHECK((scale.dtype() == at::ScalarType::Float) || (scale.dtype() == at::ScalarType::Half),
                "Quantization scale must be float or float16.")
    if (dtype == at::ScalarType::Char) {
        return quantize_symmetric_char(input, scale);
    }
    TORCH_CHECK_NOT_IMPLEMENTED(false, "quantize_symmetric not supported for ", dtype)
}
