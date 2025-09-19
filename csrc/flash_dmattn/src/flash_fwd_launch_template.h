#define FLASH_ATTENTION_ENABLE_BF16
/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

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
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_causal, bool Has_mask, bool Has_bias, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_causal, Has_mask, Has_bias, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Has_mask, bool Has_bias, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn_splitkv<Kernel_traits, Is_causal, Has_mask, Has_bias, Is_even_MN, Is_even_K, Is_softcap, Split>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    FLASH_NAMESPACE::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Has_mask, bool Has_bias>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    const size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d (includes mask memory)\n", int(smem_size));

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, Has_mask, Has_bias, IsEvenMNConst && IsEvenKConst && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst, Is_softcap, ReturnSoftmaxConst && !Is_softcap>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    // printf("run_flash_fwd: smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal, bool Has_mask, bool Has_bias>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Has_mask, Has_bias, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    // printf("run_flash_splitkv_fwd: smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

template<typename T, int Headdim, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    constexpr static int kBlockN = Has_mask || Has_bias
        ? 64
        : Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, Has_mask, Has_bias, T>, Is_causal, Has_mask, Has_bias>(params, stream);
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
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
    if constexpr (Has_mask && Has_bias) {
        if (max_smem_per_block >= 112 * 1024) {
            // 28KB, 5 CTAs in A100, 8 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, true, T>, Is_causal, true, true>(params, stream);
        } else {
            // 24KB, 4 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, true, T>, Is_causal, true, true>(params, stream);
        }
    } else if constexpr (Has_mask && !Has_bias) {
        // 20KB, 5 CTAs in sm86 and sm 89, 8 CTAs in A100, 11 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, false, T>, Is_causal, true, false>(params, stream);
    } else if constexpr (!Has_mask && Has_bias) {
        // 56KB, 1 CTAs in sm86 and sm 89, 2 CTAs in A100, 4 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, false, true, T>, Is_causal, false, true>(params, stream);
    } else {
        // 24KB, 4 CTAs in sm86 and sm 89, 6 CTAs in A100, 9 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
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
    if constexpr (Has_mask && Has_bias) {
        if (max_smem_per_block >= 160 * 1024) {
            // 40KB, 4 CTAs in A100, 5 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, true, T>, Is_causal, true, true>(params, stream);
        } else {
            // 32KB, 3 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, true, T>, Is_causal, true, true>(params, stream);
        }
    } else if constexpr (Has_mask && !Has_bias) {
        // 32KB, 3 CTAs in sm86 and sm 89, 5 CTAs in A100, 7 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, false, T>, Is_causal, true, false>(params, stream);
    } else if constexpr (!Has_mask && Has_bias) {
        // 48KB, 2 CTAs in sm86 and sm 89, 3 CTAs in A100, 4 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, false, true, T>, Is_causal, false, true>(params, stream);
    } else {
        // 48KB, 2 CTAs in sm86 and sm 89, 3 CTAs in A100, 4 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
    } 
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
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
    if constexpr (Has_mask && Has_bias) {
        if (max_smem_per_block >= 156 * 1024) {
            // 52KB, 3 CTAs in A100, 4 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, true, T>, Is_causal, true, true>(params, stream);
        } else {
            // 40KB, 2 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, true, T>, Is_causal, true, true>(params, stream);
        }
    } else if constexpr (Has_mask && !Has_bias) {
        // 44KB, 2 CTAs in sm86 and sm 89, 3 CTAs in A100, 5 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, false, T>, Is_causal, true, false>(params, stream);
    } else if constexpr (!Has_mask && Has_bias) {
        // 44KB, 2 CTAs in sm86 and sm 89, 3 CTAs in A100, 5 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, true, T>, Is_causal, false, true>(params, stream);
    } else {
        // 48KB, 2 CTAs in sm86 and sm 89, 3 CTAs in A100, 4 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
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
    if constexpr (Has_mask && Has_bias) {
        if (max_smem_per_block >= 128 * 1024) {
            // 64KB, 2 CTAs in A100, 3 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, true, T>, Is_causal, true, true>(params, stream);
        } else {
            // 48KB, 2 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, true, T>, Is_causal, true, true>(params, stream);
        }
    } else if constexpr (Has_mask && !Has_bias) {
        if (max_smem_per_block >= 112 * 1024) {
            // 56KB, 2 CTAs in A100, 4 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, false, T>, Is_causal, true, false>(params, stream);
        } else {
            // 40KB, 2 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, false, T>, Is_causal, true, false>(params, stream);
        }
    } else if constexpr (!Has_mask && Has_bias) {
        // 80KB, 2 CTAs in A100, 2 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, true, T>, Is_causal, false, true>(params, stream);
    } else {
        if (max_smem_per_block >= 128 * 1024) {
            // 64KB, 2 CTAs in A100, 3 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
            return;
        } else {
            // 48KB, 2 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
            return;
        }
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
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
    if constexpr (Has_mask && Has_bias) {
        // 88KB, 1 CTAs in sm86 and sm 89, 1 CTAs in A100, 2 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, true, T>, Is_causal, true, true>(params, stream);
    } else if constexpr (Has_mask && !Has_bias) {
        // 80KB, 1 CTAs in sm86 and sm 89, 2 CTAs in A100, 2 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, false, T>, Is_causal, true, false>(params, stream);
    } else if constexpr (!Has_mask && Has_bias) {
        // 80KB, 1 CTAs in sm86 and sm 89, 2 CTAs in A100, 2 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, true, T>, Is_causal, false, true>(params, stream);
    } else {
        // 72KB, 1 CTAs in sm86 and sm 89, 2 CTAs in A100, 3 CTAs in H100.
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
    }
}

template<typename T, bool Is_causal, bool Has_mask, bool Has_bias>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
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
    if constexpr (Has_mask && Has_bias) {
        if (max_smem_per_block >= 112 * 1024) {
            // 112KB, 1 CTAs in A100, 2 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, true, T>, Is_causal, true, true>(params, stream);
        } else {
            // 80KB, 1 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, true, T>, Is_causal, true, true>(params, stream);
        }
    } else if constexpr (Has_mask && !Has_bias) {
        if (max_smem_per_block >= 104 * 1024) {
            // 104KB, 1 CTAs in A100, 2 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, true, false, T>, Is_causal, true, false>(params, stream);
        } else {
            // 72KB, 1 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, true, false, T>, Is_causal, true, false>(params, stream);
        }
    } else if constexpr (!Has_mask && Has_bias) {
        if (max_smem_per_block >= 104 * 1024) {
            // 104KB, 1 CTAs in A100, 2 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, true, T>, Is_causal, false, true>(params, stream);
        } else {
            // 72KB, 1 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, false, true, T>, Is_causal, false, true>(params, stream);
        }
    } else {
        if (max_smem_per_block >= 128 * 1024) {
            // 128KB, 1 CTAs in A100, 1 CTAs in H100.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
        } else {
            // 96KB, 1 CTAs in sm86 and sm 89.
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, false, false, T>, Is_causal, false, false>(params, stream);
        }
    }
}

} // namespace FLASH_NAMESPACE
