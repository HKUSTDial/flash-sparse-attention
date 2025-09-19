/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashDynamicMaskAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params params)

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_kernel, bool Is_causal, bool Has_mask, bool Has_bias, bool Is_even_M, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
       FLASH_NAMESPACE::compute_dq_dk_dv<Kernel_traits, Is_causal, Has_mask, Has_bias, Is_even_M, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel, bool Is_causal, bool Has_mask, bool Has_bias, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_causal, Has_mask, Has_bias, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}


template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    FLASH_NAMESPACE::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    FLASH_NAMESPACE::clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    FLASH_NAMESPACE::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    FLASH_NAMESPACE::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Has_mask, bool Has_bias>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        int num_sm = get_num_sm(get_current_device());
        gridDimx = (num_sm + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_causal, Has_mask, Has_bias, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap>;
                if (smem_size_dq_dk_dv >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel_traits, bool Is_causal, bool Has_mask, bool Has_bias>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_causal, Has_mask, Has_bias>(params, stream);
#endif
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    );
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 104 * 1024) {             // H100 and A100
        // 104KB, 1 CTAs in A100, 2 CTAs in H100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else {                                            // sm86 and sm89
        // 96KB, 1 CTAs in sm86 and sm 89.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, true, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    );
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 144 * 1024) {             // H100 and A100
        // In fwd, multi-CTA configurations are faster, but in bwd, their speeds are very close.
        // 56KB, 1 CTAs in sm86 and sm 89, 2 CTAs in A100, 4 CTAs in H100.
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
        // 72KB, 1 CTAs in sm86 and sm 89, 2 CTAs in A100, 3 CTAs in H100.
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
        // 144KB, N/A CTAs in sm86 and sm 89, 1 CTAs in A100, 1 CTAs in H100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else {                                            // sm86 and sm89
        // 88KB, 1 CTAs in sm86 and sm 89.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    }
    // M=128, N=64 is quite slow, I think because we need to read/write dQaccum twice as many times
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    );
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 116 * 1024) {             // H100 and A100
        // 116KB, 1 CTAs in A100, 1 CTAs in H100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else {                                            // sm86 and sm89
        // 76KB, 1 CTAs in sm86 and sm 89.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 2, 4, 4, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    );
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 144 * 1024) {             // H100 and A100
        // 144KB, 1 CTAs in A100, 1 CTAs in H100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else {                                            // sm86 and sm89
        // 80KB, 1 CTAs in sm86 and sm 89.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    );
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 136 * 1024) {             // H100 and A100
        // 136KB, 1 CTAs in A100, 1 CTAs in H100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else {                                            // sm86 and sm89
        // 96KB, 1 CTAs in sm86 and sm 89.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    );
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 176 * 1024) {             // H100
        // 176KB, 1 CTAs in H100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else if (max_smem_per_block >= 144 * 1024) {      // A100
        // 144KB, 1 CTAs in A100.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    } else {                                            // sm86 and sm89
        // 96KB, 1 CTAs in sm86 and sm 89.
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, Is_causal, Has_mask, Has_bias>(params, stream);
    }
}

} // namespace FLASH_NAMESPACE
