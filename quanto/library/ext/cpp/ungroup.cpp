#include "ungroup.h"
#include <torch/extension.h>

torch::Tensor ungroup(torch::Tensor &grouped, int axis, torch::IntArrayRef orig_shape){
    if (grouped.sizes() == orig_shape){
        return grouped;
    }
    if (axis == 0) {
        return torch::reshape(grouped, orig_shape);
    }
    int64_t group_size = (axis == -1) ? grouped.size(0) : grouped.size(-1);
    int64_t axis_dim = (axis == -1) ? orig_shape.back() : orig_shape[axis];
    // Calculate the number of groups per axis
    int64_t groups_per_axis = grouped.numel() / axis_dim / group_size;

    torch::Tensor ungrouped = grouped.reshape({group_size, axis_dim, groups_per_axis});
    ungrouped = ungrouped.transpose(1, 2);
    ungrouped = ungrouped.transpose(0, 1);

    // Reshape to the original shape
    return ungrouped.reshape(orig_shape);
}