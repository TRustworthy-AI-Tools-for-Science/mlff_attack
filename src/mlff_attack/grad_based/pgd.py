"""Projected Gradient Descent (PGD) attack implementation for MLFF models.

This module implements the PGD update used by Madry et al.:
take a signed gradient step and project the perturbed input back into the
allowed epsilon-neighborhood of the original input.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np

from mlff_attack.grad_based.fgsm import FGSM_MACE

logger = logging.getLogger(__name__)

class PGD_MACE(FGSM_MACE):
    """Projected Gradient Descent attack for MACE force field models."""

    def __init__(
        self,
        model: Any,
        epsilon: float,
        alpha: float,
        num_iter: int,
        device: str = 'cpu',
        track_history: bool = True,
        target_energy: Optional[float] = None,
        random_start: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the PGD attack.

        Args:
            model: MLFF model with calculator interface
            epsilon: Maximum per-atom perturbation magnitude in Angstroms
            alpha: Step size for each iteration
            num_iter: Number of attack iterations
            device: Device for PyTorch computations
            track_history: Whether to track attack progression
            target_energy: Optional target energy objective
            random_start: Whether to initialize inside the epsilon ball
            rng: Optional NumPy random generator for deterministic starts
        """
        super().__init__(
            model=model,
            epsilon=epsilon,
            device=device,
            track_history=track_history,
            target_energy=target_energy,
        )
        self.alpha = alpha
        self.num_iter = num_iter
        self.random_start = random_start
        self.rng = rng if rng is not None else np.random.default_rng()

    def compute_gradient(
        self,
        atoms: Any,
        loss_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute gradient of loss with respect to atomic positions.

        Args:
            atoms: ASE Atoms object or equivalent structure
            loss_fn: Optional custom loss function (default: maximize energy)

        Returns:
            Gradient array with shape (n_atoms, 3)
        """
        return super().compute_gradient(atoms, loss_fn=loss_fn)


    def attack_step(
        self,
        atoms: Any,
        step: int = 0,
        loss_fn: Optional[Callable] = None,
    ) -> Any:
        """Perform one step of the PGD adversarial attack.

        Args:
            atoms: Current atomic structure
            step: Current iteration number
            loss_fn: Optional custom loss function

        Returns:
            Updated atomic structure after one attack step
        """
        if loss_fn is None:
            gradients = self.compute_gradient(atoms)
        else:
            gradients = self.compute_gradient(atoms, loss_fn=loss_fn)
        perturbation = self.alpha * np.sign(gradients)

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc

        if self.track_history:
            try:
                perturbed_energy = perturbed_atoms.get_potential_energy()
                forces = perturbed_atoms.get_forces()
                max_force = np.max(np.linalg.norm(forces, axis=1))

                self.attack_history['energies'].append(perturbed_energy)
                self.attack_history['max_forces'].append(max_force)
                self.attack_history['perturbations'].append(perturbation.copy())
                self.attack_history['gradients'].append(gradients.copy())
            except (ValueError, RuntimeError):
                pass

        return perturbed_atoms

    def _random_start(self, atoms: Any) -> Any:
        """Return a copy of atoms randomly initialized inside the epsilon ball."""
        directions = self.rng.normal(size=atoms.get_positions().shape)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(
            directions,
            norms,
            out=np.zeros_like(directions),
            where=norms > 0,
        )

        radii = self.rng.random((len(atoms), 1)) ** (1.0 / 3.0)
        perturbation = self.epsilon * radii * directions

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc
        return perturbed_atoms

    def attack(
        self,
        atoms: Any,
        n_steps: Optional[int] = None,
        clip: bool = True,
        random_start: Optional[bool] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Any:
        """Execute the full PGD attack over the specified number of iterations.

        Args:
            atoms: Input atomic structure with attached calculator
            n_steps: Optional override for ``num_iter``
            clip: Whether to project perturbations back into the epsilon ball
            random_start: Optional override for random initialization
            loss_fn: Optional custom loss function

        Returns:
            Final perturbed atomic structure after all attack iterations
        """
        self._original_positions = atoms.get_positions().copy()
        self._reset_history()

        perturbed_atoms = atoms.copy()
        perturbed_atoms.calc = atoms.calc

        use_random_start = self.random_start if random_start is None else random_start
        if use_random_start:
            perturbed_atoms = self._random_start(perturbed_atoms)
            if clip:
                self._clip_perturbations(perturbed_atoms)

        total_steps = self.num_iter if n_steps is None else n_steps
        for step in range(total_steps):
            perturbed_atoms = self.attack_step(
                perturbed_atoms,
                step=step,
                loss_fn=loss_fn,
            )
            if clip:
                self._clip_perturbations(perturbed_atoms)
                if self.target_energy is not None:
                    try:
                        current_energy = perturbed_atoms.get_potential_energy()
                        energy_diff = abs(current_energy - self.target_energy)
                        if energy_diff < 0.01:  # Within 0.01 eV of target
                            logger.info(
                                "Target energy reached at step %s: %.4f eV "
                                "(target: %.4f eV)",
                                step + 1,
                                current_energy,
                                self.target_energy,
                            )
                            break
                    except (ValueError, RuntimeError):
                        pass
        self._perturbed_positions = perturbed_atoms.get_positions().copy()
        return perturbed_atoms

    def _reset_history(self) -> None:
        """Clear tracked attack history."""
        if self.attack_history is None:
            return
        for key in self.attack_history:
            self.attack_history[key] = []
