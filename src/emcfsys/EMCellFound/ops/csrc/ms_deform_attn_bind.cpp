#include <torch/extension.h>

// These functions are defined by the bundled CUDA kernel source.  This small
// binding deliberately exposes only the operator used by EMCellFound rather
// than compiling MMCV's complete operator registry.
torch::Tensor ms_deform_attn_cuda_forward(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_loc,
    const torch::Tensor& attn_weight,
    int im2col_step);

void ms_deform_attn_cuda_backward(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_loc,
    const torch::Tensor& attn_weight,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_value,
    torch::Tensor& grad_sampling_loc,
    torch::Tensor& grad_attn_weight,
    int im2col_step);

torch::Tensor ms_deform_attn_forward(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_loc,
    const torch::Tensor& attn_weight,
    int im2col_step) {
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  return ms_deform_attn_cuda_forward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
      im2col_step);
}

void ms_deform_attn_backward(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_loc,
    const torch::Tensor& attn_weight,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_value,
    torch::Tensor& grad_sampling_loc,
    torch::Tensor& grad_attn_weight,
    int im2col_step) {
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  ms_deform_attn_cuda_backward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
      grad_output, grad_value, grad_sampling_loc, grad_attn_weight,
      im2col_step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward);
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward);
}
