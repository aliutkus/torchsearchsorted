import torch
from setuptools import setup
import torch.utils.cpp_extension as cpp

# In any case, include the CPU version
modules = [cpp.CppExtension(
    'torchsearchsorted.cpu',
    ['torchsearchsorted/cpu/searchsorted_cpu_wrapper.cpp'])]

# if CUDA is available, add the cuda extension
if torch.cuda.is_available():
    modules += [cpp.CUDAExtension(
        'torchsearchsorted.cuda',
        ['torchsearchsorted/cuda/searchsorted_cuda_wrapper.cpp',
         'torchsearchsorted/cuda/searchsorted_cuda_kernel.cu'])]

# Now proceed to setup
setup(
    name='torchsearchsorted',
    version='1.0',
    description='A searchsorted implementation for pytorch',
    keywords='searchsorted',

    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=['torchsearchsorted'],
    ext_modules=modules,
    cmdclass={
        'build_ext': cpp.BuildExtension
    })
