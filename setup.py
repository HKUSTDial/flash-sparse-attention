import os
import platform
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取CUDA主目录
CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
if not os.path.exists(CUDA_HOME):
    # 尝试标准位置
    if os.path.exists('/usr/local/cuda'):
        CUDA_HOME = '/usr/local/cuda'
    elif platform.system() == 'Windows':
        # Windows上尝试默认位置
        for cuda_version in range(12, 9, -1):  # 尝试CUDA 12至10
            cuda_path = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_version}.0"
            if os.path.exists(cuda_path):
                CUDA_HOME = cuda_path
                break

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义所有包含路径
include_dirs = [
    os.path.join(CUDA_HOME, 'include'),
    os.path.join(current_dir, 'csrc'),                    # 项目源目录
    os.path.join(current_dir, 'csrc/cutlass/include'),    # CUTLASS头文件
    os.path.join(current_dir, 'csrc/src'),                # 项目源代码子目录
    os.path.join(current_dir, 'fcsrc'),                   # Flash attention 源目录
    os.path.join(current_dir, 'fcsrc/src'),               # Flash attention 源代码子目录
]

# 禁用警告的编译标志
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_70,code=sm_70',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_86,code=compute_86',
        '--use_fast_math',
        '--expt-relaxed-constexpr',           # 允许在constexpr中使用更多功能
        '--extended-lambda',                   # 支持更高级的lambda功能
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
        # 抑制特定警告
        '-Xcudafe', '--diag_suppress=177',
        '-Xcudafe', '--diag_suppress=550',
    ]
}

# 源文件列表
sources = [
    'csrc/apply_dynamic_mask_api.cpp',
    'csrc/apply_dynamic_mask_kernel.cu',
    'csrc/apply_dynamic_mask_attention_api.cpp',
    'csrc/apply_dynamic_mask_attention_kernel.cu',
    'fcsrc/apply_attention_api.cpp',
    'fcsrc/apply_attention_kernel.cu',
]

# 创建扩展
ext_modules = [
    CUDAExtension(
        name='flash_dma_cpp',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
]

# 设置包
setup(
    name='flash_dma',
    version='0.1',
    description='Dynamic Mask Attention and Standard Attention for PyTorch',
    author='AI Assistant',
    author_email='example@example.com',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.10.0',
    ],
    python_requires='>=3.7',
)