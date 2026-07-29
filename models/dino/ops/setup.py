# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
import glob
from pathlib import Path

import torch

import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

CUDA_HOME = TORCH_CUDA_HOME.strip() if TORCH_CUDA_HOME is not None else None
cpp_extension.CUDA_HOME = CUDA_HOME

if os.name == "nt":
    _orig_check_cuda_version = cpp_extension._check_cuda_version

    def _check_cuda_version_windows(compiler_name, compiler_version):
        # PyTorch 2.0.x on Windows may resolve nvcc without the .exe suffix,
        # which causes CreateProcess([full_path_without_ext, ...]) to fail.
        return None

    cpp_extension._check_cuda_version = _check_cuda_version_windows


class BuildExtensionWithAbsoluteMSVC(BuildExtension):
    def build_extensions(self):
        compiler = self.compiler
        if os.name == "nt" and compiler is not None:
            if hasattr(compiler, "initialize") and not getattr(compiler, "initialized", False):
                compiler.initialize()
            vc_tools = os.environ.get("VCToolsInstallDir")
            sdk_bin = os.environ.get("WindowsSdkBinPath")
            sdk_version = os.environ.get("WindowsSDKVersion", "").strip("\\/")
            if vc_tools:
                msvc_bin = Path(vc_tools) / "bin" / "Hostx64" / "x64"
                tool_map = {
                    "cc": msvc_bin / "cl.exe",
                    "linker": msvc_bin / "link.exe",
                    "lib": msvc_bin / "lib.exe",
                }
                for attr, exe_path in tool_map.items():
                    if exe_path.exists():
                        setattr(compiler, attr, str(exe_path))
            if sdk_bin and sdk_version:
                sdk_x64_bin = Path(sdk_bin) / sdk_version / "x64"
                sdk_tool_map = {
                    "rc": sdk_x64_bin / "rc.exe",
                    "mt": sdk_x64_bin / "mt.exe",
                }
                for attr, exe_path in sdk_tool_map.items():
                    if exe_path.exists():
                        setattr(compiler, attr, str(exe_path))
        super().build_extensions()

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []



    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-allow-unsupported-compiler",
        ]
        host_compiler = os.environ.get("CUDAHOSTCXX")
        if host_compiler:
            host_compiler = host_compiler.strip()
            extra_compile_args["nvcc"] += ["-ccbin", host_compiler]
    else:
        raise NotImplementedError('Cuda is not availabel')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtensionWithAbsoluteMSVC.with_options(use_ninja=False)},
)
