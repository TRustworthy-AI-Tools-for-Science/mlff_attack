"""Projected Gradient Descent (PGD) attack implementation for MLFF models.

This module implements the PGD update used by Madry et al.:
take a signed gradient step and project the perturbed input back into the
allowed epsilon-neighborhood of the original input.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np

import torch
from mace.data import AtomicData, config_from_atoms

from mlff_attack.grad_based.mlff_attack_class import MLFFAttack

logger = logging.getLogger(__name__)

class PGD_MACE(MLFFAttack):
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
            epsilon: Maximum absolute displacement per coordinate under L-infinity
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
        )
        self.alpha = alpha
        self.num_iter = num_iter
        self.target_energy = target_energy
        self._last_energy = None
        self._last_gradients = None
        self.random_start = random_start
        self.rng = rng if rng is not None else np.random.default_rng()

    def _forward_pass_with_gradients(self, atoms: Any) -> tuple:
        calc = atoms.calc
        model = calc.models[0]
        positions_np = atoms.get_positions()

        config = config_from_atoms(atoms)
        atomic_data = AtomicData.from_config(
            config, z_table=calc.z_table, cutoff=calc.r_max
        )
        batch = atomic_data.to_dict()

        model_dtype = next(model.parameters()).dtype
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                if torch.is_floating_point(batch[key]):
                    batch[key] = batch[key].to(model_dtype)

        if "batch" not in batch:
            batch["batch"] = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        if "ptr" not in batch:
            batch["ptr"] = torch.tensor([0, len(atoms)], dtype=torch.long, device=self.device)

        positions = torch.tensor(
            positions_np, dtype=model_dtype, device=self.device, requires_grad=True
        )
        batch["positions"] = positions

        if "natoms" in batch:
            natoms_val = batch["natoms"]
            if natoms_val.dim() == 0:
                batch["natoms"] = torch.tensor(
                    [len(atoms), len(atoms)], dtype=torch.long, device=self.device
                )
            elif natoms_val.dim() == 1 and len(natoms_val) < 2:
                batch["natoms"] = torch.tensor(
                    [len(atoms), len(atoms)], dtype=torch.long, device=self.device
                )
        else:
            batch["natoms"] = torch.tensor(
                [len(atoms), len(atoms)], dtype=torch.long, device=self.device
            )

        if hasattr(calc, "head") and calc.head is not None:
            if hasattr(calc, "heads") and calc.heads is not None:
                head_idx = calc.heads.index(calc.head) if calc.head in calc.heads else 0
            else:
                head_idx = 0
            batch["head"] = torch.full(
                (len(atoms),), head_idx, dtype=torch.long, device=self.device
            )
        elif "head" not in batch:
            batch["head"] = torch.zeros(len(atoms), dtype=torch.long, device=self.device)

        model.eval()

        with torch.enable_grad():
            positions.requires_grad_(True)
            batch["positions"] = positions
            output = model(batch, training=False, compute_force=False)

            energy = output["energy"]
            if energy.dim() > 0:
                energy = energy.sum()

            forces = -torch.autograd.grad(
                outputs=energy,
                inputs=positions,
                retain_graph=True,
                create_graph=False,
            )[0]

        return energy, forces, positions

    def compute_gradient(
        self,
        atoms: Any,
        loss_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        energy, _forces, positions = self._forward_pass_with_gradients(atoms)

        if loss_fn is not None:
            loss = loss_fn(energy)
        elif self.target_energy is not None:
            # Try to reach target energy (minimize squared error)
            loss = -(energy - self.target_energy) ** 2
        else:
            loss = energy

        loss.backward()
        grad_positions = positions.grad
        if grad_positions is None:
            raise RuntimeError(
                "Gradient did not flow to positions; check model differentiability."
            )

        self._last_energy = energy.item()
        self._last_gradients = grad_positions.detach().cpu().numpy()

        return self._last_gradients

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
        direction = -1 if self.target_energy is not None else 1
        perturbation = direction * self.alpha * np.sign(gradients)

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
        """Return a copy of atoms randomly initialized inside the L-infinity epsilon box."""
        perturbation = self.rng.uniform(
            low=-self.epsilon,
            high=self.epsilon,
            size=atoms.get_positions().shape,
        )

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc
        return perturbed_atoms

    def attack(
        self,
        atoms: Any,
        n_steps: Optional[int] = None,
        clip: Optional[bool] = None,
        random_start: Optional[bool] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Any:
        """Execute the full PGD attack over the specified number of iterations.

        Args:
            atoms: Input atomic structure with attached calculator
            n_steps: Optional override for ``num_iter``
            clip: Whether to project perturbations within the epsilon ball, by default and always True
            random_start: Optional override for random initialization
            loss_fn: Optional custom loss function

        Returns:
            Final perturbed atomic structure after all attack iterations
        """
        if clip is None:
            clip = True
        elif clip is False:
            raise ValueError("PGD requires clipping; clip=False is not allowed.")

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
