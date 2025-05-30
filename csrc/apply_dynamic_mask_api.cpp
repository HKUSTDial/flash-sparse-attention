#include <torch/extension.h>

// 声明CUDA函数
torch::Tensor apply_dynamic_mask_cuda(
    const torch::Tensor& zero_hold_states,
    const int keep_window_size,
    const bool is_causal);

// 从Python调用的主API函数
torch::Tensor apply_dynamic_mask(
    const torch::Tensor& zero_hold_states,
    const torch::Tensor& causal_mask,  // 保留此参数以兼容Python接口，但不会使用
    const int keep_window_size = 2048,
    const bool is_causal = true) {
    
    // 忽略causal_mask参数，只转发其他参数到CUDA实现
    return apply_dynamic_mask_cuda(
        zero_hold_states,
        keep_window_size,
        is_causal
    );
}

// 定义Python模块及其函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_dynamic_mask", &apply_dynamic_mask, 
          "Apply dynamic mask to attention mechanism",
          py::arg("zero_hold_states"),
          py::arg("causal_mask"),
          py::arg("keep_window_size") = 2048,
          py::arg("is_causal") = true);
}