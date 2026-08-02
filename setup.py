"""Optional CUDA extension build for release wheels.

The extension is built when PyTorch and a CUDA toolkit are available during
installation.  Source installs without a CUDA toolchain remain installable;
the model operator can then use its project-owned lazy build or PyTorch
fallback path.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup


def _cuda_extensions():
    if os.environ.get("EMCFSYS_BUILD_CUDA", "auto").lower() in {
        "0",
        "false",
        "no",
    }:
        return [], {}
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError:
        return [], {}
    if CUDA_HOME is None:
        return [], {}
    if not os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip():
        if not torch.cuda.is_available():
            return [], {}
        capability = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    if os.name == "nt":
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
        os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")

    csrc = Path("src") / "emcfsys" / "EMCellFound" / "ops" / "csrc"
    extension = CUDAExtension(
        name="emcfsys.EMCellFound.ops._ms_deform_attn",
        sources=[
            (csrc / "ms_deform_attn_bind.cpp").as_posix(),
            (csrc / "ms_deform_attn_cuda.cu").as_posix(),
        ],
        include_dirs=[csrc.as_posix()],
        extra_compile_args={"cxx": ["/O2"] if os.name == "nt" else ["-O3"], "nvcc": ["-O3"]},
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = _cuda_extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
