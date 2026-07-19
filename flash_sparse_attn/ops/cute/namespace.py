# Copyright (c) 2026, Jingze Shi.

from typing import Any, NamedTuple


class FlashFwdMmaParamsSm80(NamedTuple):
    thr_mma_qk: Any
    thr_mma_pv: Any
    tSrQ: Any
    tSrK: Any
    tOrVt: Any
    acc_O: Any


class FlashFwdSmemCopyParamsSm80(NamedTuple):
    smem_thr_copy_Q: Any
    smem_thr_copy_K: Any
    smem_thr_copy_V: Any
    tSsQ: Any
    tSsK: Any
    tOsVt: Any


class FlashBwdMmaParamsSm80(NamedTuple):
    thr_mma_sdp: Any
    thr_mma_dkv: Any
    thr_mma_dq: Any
    tSrQ: Any
    tSrK: Any
    tdPrdO: Any
    tdPrV: Any
    tdVrP: Any
    tdVrdO: Any
    tdKrdS: Any
    tdKrQ: Any
    tdQrdS: Any
    tdQrK: Any
    acc_dK: Any
    acc_dV: Any


class FlashBwdSmemCopyParamsSm80(NamedTuple):
    smem_thr_copy_QdO: Any
    smem_thr_copy_KV: Any
    smem_thr_copy_PdSt: Any
    smem_thr_copy_QdOt: Any
    smem_thr_copy_dS: Any
    smem_thr_copy_Kt: Any
    r2s_thr_copy_PdS: Any
    tSsQ: Any
    tSsK: Any
    tdPsdO: Any
    tdPsV: Any
    tSsLSEMma: Any
    tSsdPsumMma: Any
    tPsP: Any
    tdSsdS: Any
    tdVsPt: Any
    tdVsdOt: Any
    tdKsdSt: Any
    tdKsQt: Any
    tdQsdS: Any
    tdQsKt: Any


class FlashBwdGmemCopyParamsSm80(NamedTuple):
    gmem_thr_copy_dQaccum: Any
    tdQgdQaccum: Any
