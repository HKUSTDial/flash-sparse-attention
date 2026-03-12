import triton

from flash_sparse_attn.ops.triton import utils


def get_fwd_launch_config(
    is_split_kv,
    pack_gqa,
    qheads_per_kvhead,
    tile_k,
) -> tuple[int, int, int, int, int]:
    """
    Get launch configuration for forward kernel based on input parameters and device architecture.

    :param is_split_kv: Whether the attention is split KV
    :param pack_gqa: Whether GQA packing is used
    :param qheads_per_kvhead: Number of query heads per key/value head
    :param tile_k: Tile size in the K dimension

    :return launch_config: Tuple of (tile_m, tile_n, num_warps, num_stages, num_ctas) for launching the kernel
    """
    device = utils.get_device()
    arch = utils.get_arch(device)

    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    # NOTE: Setting num_ctas=2 for the forward kernel can trigger Triton's PlanCTA assertion
    # Setting num_ctas=1 for now to avoid this issue, but we may want to revisit this in the future
    if device.type == "cuda":
        # If split KV, we set tile_m based on qheads_per_kvhead to ensure good occupancy
        if is_split_kv:
            if pack_gqa and qheads_per_kvhead > 1:
                tile_m = triton.next_power_of_2(qheads_per_kvhead)
            else:
                tile_m = 1
        else:
            # will be set based on architecture and tile_k
            tile_m = None

        # For A100
        if arch // 10 == 8:
            if not is_split_kv:
                if tile_k <= 64:
                    return (128, 128, 4, 1, 1)
                elif tile_k <= 128:
                    return (128, 64, 4, 1, 1)
                elif tile_k <= 256:
                    return (64, 64, 4, 1, 1)
                else:
                    return (64, 64, 4, 1, 1)
            else:
                if tile_k <= 64:
                    return (tile_m, 256, 4, 1, 1)
                elif tile_k <= 128:
                    return (tile_m, 128, 4, 1, 1)
                elif tile_k <= 256:
                    return (tile_m, 64, 4, 1, 1)
                else:
                    return (tile_m, 64, 4, 1, 1)

        # For H100
        elif arch // 10 == 9:
            if not is_split_kv:
                if tile_k <= 64:
                    return (256, 128, 4, 1, 1)
                elif tile_k <= 128:
                    return (128, 128, 4, 1, 1)
                elif tile_k <= 256:
                    return (128, 64, 4, 1, 1)
                else:
                    return (128, 64, 4, 1, 1)
            else:
                if tile_k <= 64:
                    return (tile_m, 256, 4, 1, 1)
                elif tile_k <= 128:
                    return (tile_m, 128, 4, 1, 1)
                elif tile_k <= 256:
                    return (tile_m, 64, 4, 1, 1)
                else:
                    return (tile_m, 64, 4, 1, 1)

        # For B200
        elif arch // 10 == 10:
            # TODO: Tune launch config for SM 100
            if not is_split_kv:
                return (64, 64, 4, 1, 1)
            else:
                return (tile_m, 64, 4, 1, 1)

        # For RTX Pro 6000
        elif arch // 10 == 12:
            if not is_split_kv:
                if tile_k <= 64:
                    return (128, 64, 4, 1, 1)
                elif tile_k <= 128:
                    return (64, 64, 4, 1, 1)
                elif tile_k <= 256:
                    return (32, 32, 4, 1, 1)
                else:
                    return (32, 32, 4, 1, 1)
            else:
                if tile_k <= 64:
                    return (tile_m, 128, 4, 1, 1)
                elif tile_k <= 128:
                    return (tile_m, 64, 4, 1, 1)
                elif tile_k <= 256:
                    return (tile_m, 32, 4, 1, 1)
                else:
                    return (tile_m, 32, 4, 1, 1)
        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")


def get_fwd_sparse_launch_config(
    is_split_kv,
    pack_gqa,
    qheads_per_kvhead,
    tile_k,
) -> tuple[int, int, int, int, int]:
    """
    Get launch configuration for forward sparse kernel based on input parameters and device architecture.

    :param is_split_kv: Whether the attention is split KV
    :param pack_gqa: Whether GQA packing is used
    :param qheads_per_kvhead: Number of query heads per key/value head
    :param tile_k: Tile size in the K dimension

    :return launch_config: Tuple of (tile_m, tile_n, num_warps, num_stages, num_ctas) for launching the kernel
    """
    device = utils.get_device()
    arch = utils.get_arch(device)

    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    # NOTE: Setting num_ctas=2 for the forward sparse kernel can trigger Triton's PlanCTA assertion
    # Setting num_ctas=1 for now to avoid this issue, but we may want to revisit this in the future
    if device.type == "cuda":
        # If split KV, we set tile_m based on qheads_per_kvhead to ensure good occupancy
        if is_split_kv:
            if pack_gqa and qheads_per_kvhead > 1:
                tile_m = triton.next_power_of_2(qheads_per_kvhead)
            else:
                tile_m = 1
        else:
            # will be set based on architecture and tile_k
            tile_m = None

        # For A100
        if arch // 10 == 8:
            if not is_split_kv:
                if tile_k <= 64:
                    return (128, 128, 4, 1, 1)
                elif tile_k <= 128:
                    return (128, 64, 4, 1, 1)
                elif tile_k <= 256:
                    return (64, 64, 4, 1, 1)
                else:
                    return (64, 64, 4, 1, 1)
            else:
                if tile_k <= 64:
                    return (tile_m, 256, 4, 1, 1)
                elif tile_k <= 128:
                    return (tile_m, 128, 4, 1, 1)
                elif tile_k <= 256:
                    return (tile_m, 64, 4, 1, 1)
                else:
                    return (tile_m, 64, 4, 1, 1)

        # For H100
        elif arch // 10 == 9:
            if not is_split_kv:
                if tile_k <= 64:
                    return (256, 128, 4, 1, 1)
                elif tile_k <= 128:
                    return (128, 128, 4, 1, 1)
                elif tile_k <= 256:
                    return (128, 64, 4, 1, 1)
                else:
                    return (128, 64, 4, 1, 1)
            else:
                if tile_k <= 64:
                    return (tile_m, 256, 4, 1, 1)
                elif tile_k <= 128:
                    return (tile_m, 128, 4, 1, 1)
                elif tile_k <= 256:
                    return (tile_m, 64, 4, 1, 1)
                else:
                    return (tile_m, 64, 4, 1, 1)

        # For B200
        elif arch // 10 == 10:
            # TODO: Tune launch config for SM 100
            if not is_split_kv:
                return (64, 64, 4, 1, 1)
            else:
                return (tile_m, 64, 4, 1, 1)

        # For RTX Pro 6000
        elif arch // 10 == 12:
            if not is_split_kv:
                if tile_k <= 64:
                    return (128, 64, 4, 1, 1)
                elif tile_k <= 128:
                    return (64, 64, 4, 1, 1)
                elif tile_k <= 256:
                    return (32, 32, 4, 1, 1)
                else:
                    return (32, 32, 4, 1, 1)
            else:
                if tile_k <= 64:
                    return (tile_m, 128, 4, 1, 1)
                elif tile_k <= 128:
                    return (tile_m, 64, 4, 1, 1)
                elif tile_k <= 256:
                    return (tile_m, 32, 4, 1, 1)
                else:
                    return (tile_m, 32, 4, 1, 1)
        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")


def get_bwd_launch_config(
    tile_k,
) -> tuple[int, int, int, int, int]:
    """
    Get launch configuration for backward kernel based on input parameters and device architecture.

    :param tile_k: Tile size in the K dimension

    :return launch_config: Tuple of (tile_m, tile_n, num_warps, num_stages, num_ctas) for launching the kernel
    """
    device = utils.get_device()
    arch = utils.get_arch(device)

    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    # NOTE: Setting num_ctas=2 for the backward kernel can trigger Triton's PlanCTA assertion
    # Setting num_ctas=1 for now to avoid this issue, but we may want to revisit this in the future
    if device.type == "cuda":
        # For A100
        if arch // 10 == 8:
            if tile_k <= 64:
                return (128, 128, 4, 1, 1)
            elif tile_k <= 128:
                return (128, 64, 4, 1, 1)
            elif tile_k <= 256:
                return (64, 64, 4, 1, 1)
            else:
                return (64, 64, 4, 1, 1)

        # For H100
        elif arch // 10 == 9:
            if tile_k <= 64:
                return (256, 128, 4, 1, 1)
            elif tile_k <= 128:
                return (128, 128, 4, 1, 1)
            elif tile_k <= 256:
                return (128, 64, 4, 1, 1)
            else:
                return (128, 64, 4, 1, 1)

        # For B200
        elif arch // 10 == 10:
            # TODO: Tune launch config for SM 100
            return (64, 64, 4, 1, 1)

        # For RTX Pro 6000
        elif arch // 10 == 12:
            if tile_k <= 64:
                return (64, 128, 8, 1, 1)
            elif tile_k <= 128:
                return (64, 64, 8, 1, 1)
            elif tile_k <= 256:
                return (32, 32, 4, 1, 1)
            else:
                return (32, 32, 4, 1, 1)
        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")


def get_bwd_sparse_launch_config(
    tile_k,
) -> tuple[int, int, int, int, int]:
    """
    Get launch configuration for backward sparse kernel based on input parameters and device architecture.

    :param tile_k: Tile size in the K dimension

    :return launch_config: Tuple of (tile_m, tile_n, num_warps, num_stages, num_ctas) for launching the kernel
    """
    device = utils.get_device()
    arch = utils.get_arch(device)

    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    # NOTE: Setting num_ctas=2 for the backward sparse kernel can trigger Triton's PlanCTA assertion
    # Setting num_ctas=1 for now to avoid this issue, but we may want to revisit this in the future
    if device.type == "cuda":
        # For A100
        if arch // 10 == 8:
            if tile_k <= 64:
                return (128, 128, 4, 1, 1)
            elif tile_k <= 128:
                return (128, 64, 4, 1, 1)
            elif tile_k <= 256:
                return (64, 64, 4, 1, 1)
            else:
                return (64, 64, 4, 1, 1)

        # For H100
        elif arch // 10 == 9:
            if tile_k <= 64:
                return (256, 128, 4, 1, 1)
            elif tile_k <= 128:
                return (128, 128, 4, 1, 1)
            elif tile_k <= 256:
                return (128, 64, 4, 1, 1)
            else:
                return (128, 64, 4, 1, 1)

        # For B200
        elif arch // 10 == 10:
            # TODO: Tune launch config for SM 100
            return (64, 64, 4, 1, 1)

        # For RTX Pro 6000
        elif arch // 10 == 12:
            if tile_k <= 64:
                return (64, 128, 8, 1, 1)
            elif tile_k <= 128:
                return (64, 64, 8, 1, 1)
            elif tile_k <= 256:
                return (32, 32, 4, 1, 1)
            else:
                return (32, 32, 4, 1, 1)
        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")


def get_fwd_combine_launch_config(
    tile_k,
) -> tuple[int, int, int, int]:
    """
    Get launch configuration for forward combine kernel based on input parameters and device architecture.

    :param tile_k: Tile size in the K dimension

    :return launch_config: Tuple of (tile_m, num_warps, num_stages, num_ctas) for launching the kernel
    """
    device = utils.get_device()
    arch = utils.get_arch(device)

    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    # NOTE: Setting num_ctas=2 for the forward kernel can trigger Triton's PlanCTA assertion
    # Setting num_ctas=1 for now to avoid this issue, but we may want to revisit this in the future
    if device.type == "cuda":
        # For A100
        if arch // 10 == 8:
            tile_m = 4 if tile_k % 128 == 0 else (8 if tile_k % 64 == 0 else 16)
            return (tile_m, 4, 1, 1)

        # For H100
        elif arch // 10 == 9:
            tile_m = 8 if tile_k % 128 == 0 else (16 if tile_k % 64 == 0 else 32)
            return (tile_m, 4, 1, 1)

        # For B200
        elif arch // 10 == 10:
            tile_m = 16 if tile_k % 128 == 0 else (32 if tile_k % 64 == 0 else 64)
            return (tile_m, 4, 1, 1)

        # For RTX Pro 6000
        elif arch // 10 == 12:
            tile_m = 4 if tile_k % 128 == 0 else (8 if tile_k % 64 == 0 else 16)
            return (tile_m, 4, 1, 1)

        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")
