"""
Projected Gradient Descent (PGD) attack implementation for MLFF models.

This module implements the PGD attack specifically for MACE models,
extending the base MLFFAttack class.
"""

from typing import Optional, Callable, Any
import numpy as np
import torch
from datetime import datetime

from .mlff_attack_class import MLFFAttack


class BIM_MACE(MLFFAttack):
    def __init__(
        self,
        model: Any,
        epsilon: float,
        alpha: float,
        num_iter: int,
        device: str = 'cpu',
        track_history: bool = True
    ):
        """Initialize the BIM attack.
        Args:
            model: MLFF model with calculator interface
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of attack iterations
            device: Device for PyTorch computations
            track_history: Whether to track attack progression
        """
        super().__init__(model, epsilon, device, track_history)
        self.alpha = alpha
        self.num_iter = num_iter    