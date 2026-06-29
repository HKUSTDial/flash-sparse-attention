import torch
import triton


def _get_max_shared_mem():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(
        props, "shared_memory_per_block_optin", props.shared_memory_per_block
    )


def _smem_bytes_fwd(tile_m, tile_n, tile_k, num_stages, dtype_bytes):
    resident = tile_m * tile_k * dtype_bytes
    pipelined = 2 * tile_n * tile_k * dtype_bytes * num_stages
    return resident + pipelined


def _smem_bytes_bwd(tile_m, tile_n, tile_k, num_stages, dtype_bytes):
    resident = 2 * tile_n * tile_k * dtype_bytes
    pipelined = 2 * tile_m * tile_k * dtype_bytes * num_stages
    return resident + pipelined


def _smem_bytes_dec(tile_m, tile_n, tile_k, num_stages, dtype_bytes):
    resident = tile_m * tile_k * dtype_bytes
    pipelined = 2 * tile_n * tile_k * dtype_bytes * num_stages
    return resident + pipelined


def _prune_fwd_configs(configs, named_args, **kwargs):
    tile_k = kwargs.get("TILE_K", named_args.get("TILE_K", 128))
    dtype_bytes = named_args["Q"].element_size()
    max_smem = _get_max_shared_mem() - 4 * 1024
    pruned = []
    for cfg in configs:
        tm, tn, ns = cfg.kwargs["TILE_M"], cfg.kwargs["TILE_N"], cfg.num_stages
        if _smem_bytes_fwd(tm, tn, tile_k, ns, dtype_bytes) <= max_smem:
            pruned.append(cfg)
    if not pruned:
        pruned = [
            min(
                configs,
                key=lambda c: _smem_bytes_fwd(
                    c.kwargs["TILE_M"],
                    c.kwargs["TILE_N"],
                    tile_k,
                    c.num_stages,
                    dtype_bytes,
                ),
            )
        ]
    return pruned


def _prune_bwd_configs(configs, named_args, **kwargs):
    tile_k = kwargs.get("TILE_K", named_args.get("TILE_K", 128))
    dtype_bytes = named_args["Q"].element_size()
    max_smem = _get_max_shared_mem() - 4 * 1024
    pruned = []
    for cfg in configs:
        tm, tn, ns = cfg.kwargs["TILE_M"], cfg.kwargs["TILE_N"], cfg.num_stages
        if _smem_bytes_bwd(tm, tn, tile_k, ns, dtype_bytes) <= max_smem:
            pruned.append(cfg)
    if not pruned:
        pruned = [
            min(
                configs,
                key=lambda c: _smem_bytes_bwd(
                    c.kwargs["TILE_M"],
                    c.kwargs["TILE_N"],
                    tile_k,
                    c.num_stages,
                    dtype_bytes,
                ),
            )
        ]
    return pruned


def _prune_dec_configs(configs, named_args, **kwargs):
    tile_k = kwargs.get("TILE_K", named_args.get("TILE_K", 128))
    seqlen_q = kwargs.get("seqlen_q", named_args.get("seqlen_q", None))
    dtype_bytes = named_args["Q"].element_size()
    max_smem = _get_max_shared_mem() - 4 * 1024
    pruned = []
    for cfg in configs:
        tm, tn, ns = cfg.kwargs["TILE_M"], cfg.kwargs["TILE_N"], cfg.num_stages
        if seqlen_q is not None and tm > seqlen_q:
            continue
        if _smem_bytes_dec(tm, tn, tile_k, ns, dtype_bytes) <= max_smem:
            pruned.append(cfg)
    if not pruned:
        pruned = [
            min(
                configs,
                key=lambda c: _smem_bytes_dec(
                    c.kwargs["TILE_M"],
                    c.kwargs["TILE_N"],
                    tile_k,
                    c.num_stages,
                    dtype_bytes,
                ),
            )
        ]
    return pruned


def get_fwd_dense_autotune_configs(tile_n=None):
    configs = []
    tile_ns = [tile_n] if tile_n is not None else [32, 64, 128]
    for tile_m in [64, 128, 256]:
        for tile_n in tile_ns:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_fwd_sparse_autotune_configs(tile_n=None):
    configs = []
    tile_ns = [tile_n] if tile_n is not None else [32, 64, 128]
    for tile_m in [64, 128, 256]:
        for tile_n in tile_ns:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_fwd_gated_autotune_configs(tile_n=None):
    configs = []
    tile_ns = [tile_n] if tile_n is not None else [32, 64, 128]
    for tile_m in [64, 128, 256]:
        for tile_n in tile_ns:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_bwd_dense_autotune_configs():
    configs = []
    for tile_m in [32, 64, 128]:
        for tile_n in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_bwd_sparse_autotune_configs():
    configs = []
    for tile_m in [32, 64, 128]:
        for tile_n in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_bwd_gated_autotune_configs():
    configs = []
    for tile_m in [32, 64, 128]:
        for tile_n in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_dec_dense_autotune_configs(tile_n=None):
    configs = []
    tile_ns = [tile_n] if tile_n is not None else [64, 128, 256]
    for tile_m in [16, 32, 64]:
        for tile_n in tile_ns:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_dec_sparse_autotune_configs(tile_n=None):
    configs = []
    tile_ns = [tile_n] if tile_n is not None else [64, 128, 256]
    for tile_m in [16, 32, 64]:
        for tile_n in tile_ns:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_dec_gated_autotune_configs(tile_n=None):
    configs = []
    tile_ns = [tile_n] if tile_n is not None else [64, 128, 256]
    for tile_m in [16, 32, 64]:
        for tile_n in tile_ns:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def make_fwd_dense_autotuned_kernel(jit_kernel, tile_n=None):
    configs = get_fwd_dense_autotune_configs(tile_n=tile_n)
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_CAUSAL", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_fwd_configs},
    )(jit_kernel)


def make_fwd_sparse_autotuned_kernel(jit_kernel, tile_n=None):
    configs = get_fwd_sparse_autotune_configs(tile_n=tile_n)
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_CAUSAL", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_fwd_configs},
    )(jit_kernel)


def make_fwd_gated_autotuned_kernel(jit_kernel, tile_n=None):
    configs = get_fwd_gated_autotune_configs(tile_n=tile_n)
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_CAUSAL", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_fwd_configs},
    )(jit_kernel)


def make_bwd_dense_autotuned_kernel(jit_kernel):
    configs = get_bwd_dense_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_CAUSAL", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_bwd_configs},
    )(jit_kernel)


def make_bwd_sparse_autotuned_kernel(jit_kernel):
    configs = get_bwd_sparse_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_CAUSAL", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_bwd_configs},
    )(jit_kernel)


def make_bwd_gated_autotuned_kernel(jit_kernel):
    configs = get_bwd_gated_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_CAUSAL", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_bwd_configs},
    )(jit_kernel)


def make_dec_dense_autotuned_kernel(jit_kernel, tile_n=None):
    configs = get_dec_dense_autotune_configs(tile_n=tile_n)
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_dec_configs},
    )(jit_kernel)


def make_dec_sparse_autotuned_kernel(jit_kernel, tile_n=None):
    configs = get_dec_sparse_autotune_configs(tile_n=tile_n)
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_dec_configs},
    )(jit_kernel)


def make_dec_gated_autotuned_kernel(jit_kernel, tile_n=None):
    configs = get_dec_gated_autotune_configs(tile_n=tile_n)
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K", "IS_LOCAL"],
        prune_configs_by={"early_config_prune": _prune_dec_configs},
    )(jit_kernel)


class AutotunedKernel:
    STRIP_KWARGS = {"TILE_M", "TILE_N", "num_warps", "num_stages", "num_ctas"}

    def __init__(self, autotuned_kernel):
        self._autotuned = autotuned_kernel

    def __getitem__(self, grid):
        autotuned = self._autotuned

        class _Launcher:
            def __call__(_, *args, **kwargs):
                for key in AutotunedKernel.STRIP_KWARGS:
                    kwargs.pop(key, None)
                return autotuned[grid](*args, **kwargs)

        return _Launcher()

    def __getattr__(self, name):
        return getattr(self._autotuned, name)
