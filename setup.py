# Copyright (c) 2025, Jingze Shi.

import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_dmattn"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
# Also useful when user only wants Triton/Flex backends without CUDA compilation
FORCE_BUILD = os.getenv("FLASH_DMATTN_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_DMATTN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_DMATTN_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# Auto-detect if user wants only Triton/Flex backends based on pip install command
# This helps avoid unnecessary CUDA compilation when user only wants Python backends
def should_skip_cuda_build():
    """Determine if CUDA build should be skipped based on installation context."""
    
    if SKIP_CUDA_BUILD:
        return True
    
    if FORCE_BUILD:
        return False  # User explicitly wants to build, respect that
    
    # Check command line arguments for installation hints
    if len(sys.argv) > 1:
        install_args = ' '.join(sys.argv)
        
        # Check if Triton or Flex extras are requested
        has_triton_or_flex = 'triton' in install_args or 'flex' in install_args
        has_all_or_dev = 'all' in install_args or 'dev' in install_args

        if has_triton_or_flex and not has_all_or_dev:
            print("Detected Triton/Flex-only installation. Skipping CUDA compilation.")
            print("Set FLASH_DMATTN_FORCE_BUILD=TRUE to force CUDA compilation.")
            return True
    
    return False

# Update SKIP_CUDA_BUILD based on auto-detection
SKIP_CUDA_BUILD = should_skip_cuda_build()

@functools.lru_cache(maxsize=None)
def cuda_archs():
    return os.getenv("FLASH_DMATTN_CUDA_ARCHS", "80;86;89;90;100;120").split(";")


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
else:
    assert (
        os.path.exists("csrc/cutlass/include/cutlass/cutlass.h")
    ), "csrc/cutlass is missing, please use source distribution or git clone"

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("flash_dmattn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "Flash Dynamic Mask Attention is only supported on CUDA 11.7 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )

    if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")

    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8") and "86" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_86,code=sm_86")
        if bare_metal_version >= Version("11.8") and "89" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_89,code=sm_89")
        if bare_metal_version >= Version("11.8") and "90" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8") and "100" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    # "--ptxas-options=-v",
    # "--ptxas-options=-O2",
    # "-lineinfo",
    # "-DFLASHATTENTION_DISABLE_BACKWARD",
    # "-DFLASHATTENTION_DISABLE_SOFTCAP",
    # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
    ]

    compiler_c17_flag=["-O3", "-std=c++17"]
    # Add Windows-specific flags
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]

    ext_modules.append(
        CUDAExtension(
            name="flash_dmattn_cuda",
            sources=[
                "csrc/flash_api.cpp",
                # Forward kernels - regular
                "csrc/src/instantiations/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim256_bf16_sm80.cu",
                # Forward kernels - causal
                "csrc/src/instantiations/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_hdim256_bf16_causal_sm80.cu",
                # Forward kernels - split
                "csrc/src/instantiations/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim256_bf16_sm80.cu",
                # Forward kernels - split causal
                "csrc/src/instantiations/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
                # Backward kernels - regular
                "csrc/src/instantiations/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim256_bf16_sm80.cu",
                # Backward kernels - causal
                "csrc/src/instantiations/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/src/instantiations/flash_bwd_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                "cxx": compiler_c17_flag,
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            include_dirs=[
                Path(this_dir) / "csrc",
                Path(this_dir) / "csrc" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "flash_dmattn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_DMATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, (os.cpu_count() or 1) // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension}
    if ext_modules
    else {},
)
