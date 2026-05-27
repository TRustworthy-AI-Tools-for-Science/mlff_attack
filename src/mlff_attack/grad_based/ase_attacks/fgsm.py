"""FGSM attack for generic ASE calculators."""

from typing import Any, Optional

import numpy as np

from mlff_attack.grad_based.mlff_attack_class import MLFFAttack


class FGSM_ASE(MLFFAttack):
    """FGSM / iterative FGSM attack using ASE calculator forces."""

    def __init__(
        self,
        model: Any,
        epsilon: float = 0.01,
        device: str = "cpu",
        track_history: bool = True,
        target_energy: Optional[float] = None,
    ):
        super().__init__(model, epsilon, device, track_history)
        self.target_energy = target_energy
        self._last_energy = None
        self._last_gradients = None

    def compute_gradient(self, atoms: Any, loss_fn=None) -> np.ndarray:
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

    def attack_step(self, atoms: Any, step: int = 0, n_steps: int = 1) -> Any:
        gradients = self.compute_gradient(atoms)
        step_size = self.epsilon / n_steps
        direction = 1.0 if self.target_energy is None else -1.0
        perturbation = direction * step_size * np.sign(gradients)

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc

        self._record_history(perturbed_atoms, perturbation, gradients)
        return perturbed_atoms

    def attack(self, atoms: Any, n_steps: int = 1, clip: Optional[bool] = None) -> Any:
        if clip is None:
            clip = False

        self.reset()
        self._original_positions = atoms.get_positions().copy()

        perturbed_atoms = atoms.copy()
        perturbed_atoms.calc = atoms.calc

        for step in range(n_steps):
            perturbed_atoms = self.attack_step(
                perturbed_atoms,
                step=step,
                n_steps=n_steps,
            )
            if clip:
                self._clip_perturbations(perturbed_atoms)

        self._perturbed_positions = perturbed_atoms.get_positions().copy()
        return perturbed_atoms

    def _record_history(self, atoms, perturbation, gradients) -> None:
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