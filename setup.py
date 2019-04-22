import torch
from setuptools import setup
import torch.utils.cpp_extension as cpp

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

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
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='searchsorted',

    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=['torchsearchsorted torch'],
    ext_modules=modules,
    cmdclass={
        'build_ext': cpp.BuildExtension
    })
