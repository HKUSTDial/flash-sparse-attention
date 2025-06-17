#include <torch/extension.h>

// 声明CUDA函数
std::vector<torch::Tensor> apply_dynamic_mask_attention_cuda(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& zoh_states,
    const torch::Tensor& active_mask,
    float scale,
    int keep_window_size,
    bool is_causal,
    bool return_softmax);

// 主API函数，从Python调用 - 移除了冗余的causal_mask参数
std::vector<torch::Tensor> apply_dynamic_mask_attention(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& zoh_states,
    const torch::Tensor& active_mask,
    float scale = 1.0f,
    int keep_window_size = 2048,
    bool is_causal = true,
    bool return_softmax = false) {
    
    // 验证所有张量都在CUDA上
    TORCH_CHECK(query_states.is_cuda(), "query_states必须是CUDA张量");
    TORCH_CHECK(key_states.is_cuda(), "key_states必须是CUDA张量");
    TORCH_CHECK(value_states.is_cuda(), "value_states必须是CUDA张量");
    TORCH_CHECK(zoh_states.is_cuda(), "zoh_states必须是CUDA张量");
    TORCH_CHECK(active_mask.is_cuda(), "active_mask必须是CUDA张量");
    
    // 所有张量必须在同一设备上
    TORCH_CHECK(query_states.device() == key_states.device(), "所有张量必须在同一设备上");
    TORCH_CHECK(query_states.device() == value_states.device(), "所有张量必须在同一设备上");
    TORCH_CHECK(query_states.device() == zoh_states.device(), "所有张量必须在同一设备上");
    TORCH_CHECK(query_states.device() == active_mask.device(), "所有张量必须在同一设备上");
    
    // 转发到CUDA实现
    return apply_dynamic_mask_attention_cuda(
        query_states, 
        key_states, 
        value_states,
        zoh_states,
        active_mask,
        scale,
        keep_window_size,
        is_causal,
        return_softmax
    );
}

// 定义Python模块和函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_dynamic_mask_attention", &apply_dynamic_mask_attention, 
          py::arg("query_states"),
          py::arg("key_states"),
          py::arg("value_states"),
          py::arg("zoh_states"),
          py::arg("active_mask"),
          py::arg("scale") = 1.0f,
          py::arg("keep_window_size") = 2048,
          py::arg("is_causal") = true,
          py::arg("return_softmax") = false,
          "使用动态掩码计算注意力");
}