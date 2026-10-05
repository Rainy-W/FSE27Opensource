"""Shared runtime utilities."""

from __future__ import annotations

import os
import random


def set_random_seed(seed: int = 42, *, strict_determinism: bool = False) -> None:
    """Set RNG seeds and, optionally, require deterministic CUDA execution."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            if strict_determinism and cuda_available:
                workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if workspace_config not in {":16:8", ":4096:8"}:
                    raise RuntimeError(
                        "Strict determinism requires CUBLAS_WORKSPACE_CONFIG to be set before "
                        "the Python process starts (use :4096:8 or :16:8)."
                    )
                if torch.cuda.is_initialized():
                    raise RuntimeError(
                        "Strict determinism must be configured before CUDA initialization. "
                        "Restart with CUBLAS_WORKSPACE_CONFIG already in the environment."
                    )
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("highest")
                # Force the deterministic math SDPA backend instead of Flash/xFormers-style kernels.
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(False)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
            torch.use_deterministic_algorithms(True, warn_only=not strict_determinism)
    except ImportError:
        pass
