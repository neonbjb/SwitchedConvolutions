from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='switched_conv_cuda_naive',
    ext_modules=[
        CUDAExtension('switched_conv_cuda_naive', [
            'switched_conv_cuda_naive.cpp',
            'switched_conv_cuda_kernel_naive.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)