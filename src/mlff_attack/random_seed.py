"""Allow for a manual controllable random seed so that MACE, UMA, and PGD runs are reproducible."""

import os
import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set process-level random seeds used by Python, NumPy, and Torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False