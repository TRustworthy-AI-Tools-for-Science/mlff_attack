"""PGD attack for generic ASE calculators."""

from typing import Any, Callable, Optional

import numpy as np

from mlff_attack.grad_based.mlff_attack_class import MLFFAttack


class PGD_ASE(MLFFAttack):
    """PGD attack using ASE calculator forces."""

    def __init__(
        self,
        model: Any,
        epsilon: float,
        alpha: float,
        num_iter: int,
        device: str = "cpu",
        track_history: bool = True,
        target_energy: Optional[float] = None,
    ):
        super().__init__(
            model=model,
            epsilon=epsilon,
            device=device,
            track_history=track_history,
        )
        self.alpha = alpha
        self.num_iter = num_iter
        self.target_energy = target_energy
        self._last_energy = None
        self._last_gradients = None

    def compute_gradient(
        self,
        atoms: Any,
        loss_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Compute position gradient from ASE forces.

        ASE forces are -dE/dR, so dE/dR is -forces.
        """
        if loss_fn is not None:
            raise NotImplementedError(
                "Custom torch loss_fn is not supported for ASE-force attacks."
            )

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)

        if self.target_energy is None:
            gradients = -forces
        else:
            gradients = -2.0 * (energy - self.target_energy) * forces

        self._last_energy = energy
        self._last_gradients = gradients
        return gradients

    def attack_step(
        self,
        atoms: Any,
        step: int = 0,
        loss_fn: Optional[Callable] = None,
    ) -> Any:
        """Perform one PGD step."""
        gradients = self.compute_gradient(atoms, loss_fn=loss_fn)
        direction = 1.0 if self.target_energy is None else -1.0
        perturbation = direction * self.alpha * np.sign(gradients)

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc

        self._record_history(perturbed_atoms, perturbation, gradients)
        return perturbed_atoms

    def attack(
        self,
        atoms: Any,
        n_steps: Optional[int] = None,
        clip: Optional[bool] = None,
        random_start=None,
        loss_fn: Optional[Callable] = None,
    ) -> Any:
        """Execute PGD attack."""
        if clip is None:
            clip = True
        elif clip is False:
            raise ValueError("PGD requires clipping; clip=False is not allowed.")

        self.reset()
        self._original_positions = atoms.get_positions().copy()

        perturbed_atoms = atoms.copy()
        perturbed_atoms.calc = atoms.calc

        total_steps = self.num_iter if n_steps is None else n_steps
        for step in range(total_steps):
            perturbed_atoms = self.attack_step(
                perturbed_atoms,
                step=step,
                loss_fn=loss_fn,
            )
            self._clip_perturbations(perturbed_atoms)

        self._perturbed_positions = perturbed_atoms.get_positions().copy()
        return perturbed_atoms

    def _record_history(
        self,
        atoms: Any,
        perturbation: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        """Record attack history if enabled."""
        if not self.track_history:
            return

        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        except (ValueError, RuntimeError):
            return

        self.attack_history["energies"].append(energy)
        self.attack_history["max_forces"].append(max_force)
        self.attack_history["perturbations"].append(perturbation.copy())
        self.attack_history["gradients"].append(gradients.copy())