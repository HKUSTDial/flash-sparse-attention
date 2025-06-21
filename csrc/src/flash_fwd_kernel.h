/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <ATen/cuda/detail/UnpackRaw.cuh> // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;    // query_block_len
    constexpr int kBlockN = Kernel_traits::kBlockN;    // key_block_len
    constexpr int kHeadDim = Kernel_traits::kHeadDim;  // head_dim
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    FLASH_NAMESPACE::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    // Check if there are any queries to process in the block
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // Compute the actual range of N blocks to process
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = std::min(
            n_block_max,
            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN)
        );
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
            make_shape(binfo.actual_seqlen_q, params.h, params.d), make_stride(params.o_row_stride, params.o_head_stride, _1{})
        );
        Tensor gO = local_tile(
            mO(_, bidh, _),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_coord(m_block, 0)
        );  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    // Golobal memory tensor configuration
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{})
    );
    Tensor gQ = local_tile(
        mQ(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0)
    );  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{})
    );          
    Tensor gK = local_tile(
        mK(_, bidh / params.h_h_k_ratio, _),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0)
    );  // (kBlockN, kHeadDim, nblocksN)                 
    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{})
    );           
    Tensor gV = local_tile(
        mV(_, bidh / params.h_h_k_ratio, _),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0)
    );  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.p_ptr) + row_offset_p),
        Shape<Int<kBlockM>, Int<kBlockN>>{},
        make_stride(params.seqlen_k_rounded, _1{})
    );
    Tensor mZOH = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.zoh_ptr) + binfo.zoh_offset(params.zoh_batch_stride, bidb)),
        make_shape(params.h_k, binfo.actual_seqlen_q, binfo.actual_seqlen_k),
        make_stride(params.zoh_head_stride, params.zoh_row_stride, _1{})
    );
    Tensor gZOH = local_tile(
        mZOH(bidh / params.h_h_k_ratio, _, _),
        Shape<Int<kBlockM>, Int<kBlockN>>{},
        make_coord(m_block, _)
    );  // (kBlockM, kBlockN, nblocksN)
    Tensor mActiveMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.active_mask_ptr) + binfo.active_mask_offset(params.active_mask_batch_stride, bidb)),
        make_shape(params.h_k, binfo.actual_seqlen_q, binfo.actual_seqlen_k),
        make_stride(params.active_mask_head_stride, params.active_mask_row_stride, _1{})
    );
    Tensor gActiveMask = local_tile(
        mActiveMask(bidh / params.h_h_k_ratio, _, _),
        Shape<Int<kBlockM>, Int<kBlockN>>{},
        make_coord(m_block, _)
    );  // (kBlockM, kBlockN, nblocksN)

    // Shared memory layout configuration
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element *>(smem_)),
        typename Kernel_traits::SmemLayoutQ{}
    );
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(
        sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
        typename Kernel_traits::SmemLayoutKV{}
    );
    Tensor sV = make_tensor(
        sK.data() + size(sK),
        typename Kernel_traits::SmemLayoutKV{}
    );
    Tensor sVt = make_tensor(
        sV.data(),
        typename Kernel_traits::SmemLayoutVtransposed{}
    );
    Tensor sVtNoSwizzle = make_tensor(
        sV.data().get(),
        typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{}
    );
    Tensor sZOH = make_tensor(
        sV.data() + size(sV),
        typename Kernel_traits::SmemLayoutZOH{}
    );
    Tensor sActiveMask = make_tensor(
        sZOH.data() + size(sZOH),
        typename Kernel_traits::SmemLayoutActiveMask{}
    );

    // Golobal to Shared Memory operation
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyZOH gmem_tiled_copy_ZOH;
    auto gmem_thr_copy_ZOH = gmem_tiled_copy_ZOH.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyActiveMask gmem_tiled_copy_AM;
    auto gmem_thr_copy_AM = gmem_tiled_copy_AM.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tZOHgZOH = gmem_thr_copy_ZOH.partition_S(gZOH);  // (ZOHCPY, ZOHCPY_M, ZOHCPY_N, nblocksN)
    Tensor tZOHsZOH = gmem_thr_copy_ZOH.partition_D(sZOH);
    Tensor tAMgAM = gmem_thr_copy_AM.partition_S(gActiveMask);  // (AMCPY, AMCPY_M, AMCPY_N, nblocksN)
    Tensor tAMsAM = gmem_thr_copy_AM.partition_D(sActiveMask);

    // Matrix Multiply Accumulate
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);                                         // (MMA,MMA_M,MMA_K)
    Tensor tSrK = thr_mma.partition_fragment_B(sK);                                         // (MMA,MMA_N,MMA_K)
    Tensor tSrZOH = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});   // (MMA, MMA_M, MMA_N)
    Tensor tSrAM = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});    // (MMA, MMA_M, MMA_N)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);                              // (MMA, MMA_K,MMA_N)
    Tensor tSgS  = thr_mma.partition_C(gP);
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});   // MMA, MMA_M, MMA_K

    // Copy Atom retiling
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_ZOH = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_ZOH = smem_tiled_copy_ZOH.get_thread_slice(tidx);
    Tensor tSsZOH = smem_thr_copy_ZOH.partition_S(sZOH);
    auto smem_tiled_copy_AM = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_AM = smem_tiled_copy_AM.get_thread_slice(tidx);
    Tensor tSsAM = smem_thr_copy_AM.partition_S(sActiveMask);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));                     // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));                    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cZOH = make_identity_tensor(make_shape(size<0>(sZOH), size<1>(sZOH)));               // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cAM = make_identity_tensor(make_shape(size<0>(sActiveMask), size<1>(sActiveMask)));  // (BLK_M,BLK_N) -> (blk_m,blk_n)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);        // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);     // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tZOHcZOH = gmem_thr_copy_ZOH.partition_S(cZOH);  // (ZOHCPY, ZOHCPY_M, ZOHCPY_N) -> (blk_m, blk_n)
    Tensor tAMcAM = gmem_thr_copy_AM.partition_S(cAM);      // (AMCPY, AMCPY_M, AMCPY_N) -> (blk_m, blk_n)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    // Prologue
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
        binfo.actual_seqlen_q - m_block * kBlockM
    );
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // if (cute::thread(1, 0)) { print(tQsQ); }
    // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // if (cute::thread0()) { print(sQNoSwizzle); }

    // If share Q and K smem, wait and sync
    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }
    // Reverse iteration over N blocks
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV,
        tKgK(_, _, _, n_block),
        tKsK, tKVcKV, tKVpKV,
        binfo.actual_seqlen_k - n_block * kBlockN
    );
    FLASH_NAMESPACE::copy_ZOH<Is_even_MN>(
        gmem_tiled_copy_ZOH,
        tZOHgZOH(_, _, _, n_block),
        tZOHsZOH, 
        tZOHcZOH,
        binfo.actual_seqlen_q - m_block * kBlockM,
        binfo.actual_seqlen_k - n_block * kBlockN
    );
    FLASH_NAMESPACE::copy_ZOH<Is_even_MN>(
        gmem_tiled_copy_AM,
        tAMgAM(_, _, _, n_block),
        tAMsAM, 
        tAMcAM,
        binfo.actual_seqlen_q - m_block * kBlockM,
        binfo.actual_seqlen_k - n_block * kBlockN
    );
    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

    // Init dynamic mask processor
    FLASH_NAMESPACE::DynamicMask<Is_causal, Kernel_traits::kNThreads> dynamic_mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.keep_window_size
    );

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Copy ZOH and ActiveMask from smem to registers
        Tensor tSrZOH_copy_view = smem_thr_copy_ZOH.retile_D(tSrZOH);
        cute::copy(smem_tiled_copy_ZOH, tSsZOH, tSrZOH_copy_view);
        Tensor tSrAM_copy_view = smem_thr_copy_AM.retile_D(tSrAM);
        cute::copy(smem_tiled_copy_AM, tSsAM, tSrAM_copy_view);

        // Advance gV
        if (masking_step > 0) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence();

        // Use sparse general matrix multiplication
        FLASH_NAMESPACE::sparse_gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s,
            tSrQ,
            tSrK, tSsQ, tSsK, tSrAM,        // Active key indices for sparse K matrix multiplication
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        // Scale attention scores and apply dynamic mask
        dynamic_mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, tSrZOH, tSrAM, params.scale_softmax,
            n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            FLASH_NAMESPACE::copy_ZOH</*Is_even_MN=*/true>(
                gmem_tiled_copy_ZOH,
                tZOHgZOH(_, _, _, n_block - 1),
                tZOHsZOH, 
                tZOHcZOH,
                binfo.actual_seqlen_q - m_block * kBlockM,
                binfo.actual_seqlen_k - (n_block - 1) * kBlockN
            );
            FLASH_NAMESPACE::copy_ZOH</*Is_even_MN=*/true>(
                gmem_tiled_copy_AM,
                tAMgAM(_, _, _, n_block - 1),
                tAMsAM, 
                tAMcAM,
                binfo.actual_seqlen_q - m_block * kBlockM,
                binfo.actual_seqlen_k - (n_block - 1) * kBlockN
            );
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, acc_o)
            : softmax.template softmax</*Is_first=*/false, /*Check_inf=*/true>(acc_s, acc_o);
        // masking_step == 0
        //     ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, acc_o, params.scale_softmax_log2)
        //     : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(acc_s, acc_o, params.scale_softmax_log2);
        
        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        // Use sparse general matrix multiplication with register accumulation for V as well
        FLASH_NAMESPACE::sparse_gemm_rs(
            acc_o,
            tOrP, tOrVt, tOsVt,
            tSrAM,        // Apply the same mask for sparse V matrix multiplication
            tiled_mma, smem_tiled_copy_V, smem_thr_copy_V
        );
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Copy ZOH and ActiveMask from smem to registers
        Tensor tSrZOH_copy_view = smem_thr_copy_ZOH.retile_D(tSrZOH);
        cute::copy(smem_tiled_copy_ZOH, tSsZOH, tSrZOH_copy_view);
        Tensor tSrAM_copy_view = smem_thr_copy_AM.retile_D(tSrAM);
        cute::copy(smem_tiled_copy_AM, tSsAM, tSrAM_copy_view);

        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        FLASH_NAMESPACE::sparse_gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s,
            tSrQ,
            tSrK, tSsQ, tSsK, tSrAM,        // Active key indices for sparse K matrix multiplication
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        // Scale attention scores and apply dynamic mask
        dynamic_mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, tSrZOH, tSrAM, params.scale_softmax,
            n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            FLASH_NAMESPACE::copy_ZOH</*Is_even_MN=*/true>(
                gmem_tiled_copy_ZOH,
                tZOHgZOH(_, _, _, n_block - 1),
                tZOHsZOH, 
                tZOHcZOH,
                binfo.actual_seqlen_q - m_block * kBlockM,
                binfo.actual_seqlen_k - (n_block - 1) * kBlockN
            );
            FLASH_NAMESPACE::copy_ZOH</*Is_even_MN=*/true>(
                gmem_tiled_copy_AM,
                tAMgAM(_, _, _, n_block - 1),
                tAMsAM, 
                tAMcAM,
                binfo.actual_seqlen_q - m_block * kBlockM,
                binfo.actual_seqlen_k - (n_block - 1) * kBlockN
            );
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        softmax.template softmax</*Is_first=*/false, /*Check_inf=*/true>(acc_s, acc_o);
        // softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

        // Use sparse general matrix multiplication with register accumulation for V as well
        FLASH_NAMESPACE::sparse_gemm_rs(
            acc_o,
            tOrP, tOrVt, tOsVt,
            tSrAM,        // Apply the same mask for sparse V matrix multiplication
            tiled_mma, smem_tiled_copy_V, smem_thr_copy_V
        );
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{})
    );
    Tensor gO = local_tile(
        mO(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0)
    );  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    FLASH_NAMESPACE::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}

}  // namespace FLASH_NAMESPACE