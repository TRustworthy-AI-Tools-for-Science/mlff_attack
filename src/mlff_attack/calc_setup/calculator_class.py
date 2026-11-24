"""Base abstractions and shared helpers for calculator setup."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from mlff_attack.random_seed import set_random_seed


def dtype_from_string(dtype_str):
    """Convert a dtype string into a torch dtype."""
    dtype_str = str(dtype_str).strip().lower()
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError("dtype_str must be 'float32' or 'float64'")


def cast_torch_modules_dtype(obj, dtype):
    """Update any PyTorch modules found inside a value to use the requested dtype."""
    seen = set()
    changed = False

    def visit(value):
        nonlocal changed

        if value is None:
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, torch.nn.Module):
            value.to(dtype=dtype)
            changed = True
            return

        if isinstance(value, (str, bytes, int, float, bool, Path)):
            return

        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return

        if hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item)

    visit(obj)
    return changed


class MLFFCalc(ABC):
    """Base class for setting up MLFF calculators."""

    def __init__(
        self,
        model_path,
        device="cpu",
        dtype_str="float64",
        seed=None,
        verbose=False,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype_str
        self.seed = seed
        self.verbose = verbose

    def set_seed(self):
        if self.seed is not None:
            set_random_seed(self.seed)

    @abstractmethod
    def setup(self, atoms):
        """Attach the calculator and return atoms."""
        raise NotImplementedError
