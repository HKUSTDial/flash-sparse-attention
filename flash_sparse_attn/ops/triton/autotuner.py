import triton


def get_fwd_dense_autotune_configs():
    configs = []
    for tile_m in [64, 128, 256]:
        for tile_n in [32, 64, 128]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_fwd_sparse_autotune_configs():
    configs = []
    for tile_m in [64, 128, 256]:
        for tile_n in [32, 64, 128]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_fwd_gated_autotune_configs():
    configs = []
    for tile_m in [64, 128, 256]:
        for tile_n in [32, 64, 128]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
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
                for num_stages in [1, 2]:
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
                for num_stages in [1, 2]:
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
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_dec_dense_autotune_configs():
    configs = []
    for tile_m in [16, 32, 64]:
        for tile_n in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_dec_sparse_autotune_configs():
    configs = []
    for tile_m in [16, 32, 64]:
        for tile_n in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def get_dec_gated_autotune_configs():
    configs = []
    for tile_m in [16, 32, 64]:
        for tile_n in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"TILE_M": tile_m, "TILE_N": tile_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                            num_ctas=1,
                        )
                    )
    return configs


def make_fwd_dense_autotuned_kernel(jit_kernel):
    configs = get_fwd_dense_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_fwd_sparse_autotuned_kernel(jit_kernel):
    configs = get_fwd_sparse_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_fwd_gated_autotuned_kernel(jit_kernel):
    configs = get_fwd_gated_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_bwd_dense_autotuned_kernel(jit_kernel):
    configs = get_bwd_dense_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_bwd_sparse_autotuned_kernel(jit_kernel):
    configs = get_bwd_sparse_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_bwd_gated_autotuned_kernel(jit_kernel):
    configs = get_bwd_gated_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_dec_dense_autotuned_kernel(jit_kernel):
    configs = get_dec_dense_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_dec_sparse_autotuned_kernel(jit_kernel):
    configs = get_dec_sparse_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
    )(jit_kernel)


def make_dec_gated_autotuned_kernel(jit_kernel):
    configs = get_dec_gated_autotune_configs()
    return triton.autotune(
        configs=configs,
        key=["SEQLEN_Q_CACHE", "SEQLEN_K_CACHE", "TILE_K"],
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
