"""Projected Gradient Descent (PGD) attack implementation for MLFF models."""

import logging
from typing import Any, Callable, Optional

import numpy as np

from mlff_attack.grad_based.mlff_attack_class import MLFFAttack

logger = logging.getLogger(__name__)


class PGD_ASE(MLFFAttack):
    """Projected Gradient Descent attack for ASE-compatible MLFF calculators."""

    def __init__(
        self,
        model: Any,
        epsilon: float,
        alpha: float,
        num_iter: int,
        device: str = "cpu",
        track_history: bool = True,
        target_energy: Optional[float] = None,
        random_start: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the unified PGD attack.

        Parameters
        ----------
        model : Any
            ASE-compatible calculator. This may be a MACE calculator, UMA
            calculator, or another calculator that provides energies and forces.
        epsilon : float
            Maximum absolute displacement per coordinate under L-infinity.
        alpha : float
            Signed-gradient step size for each PGD iteration.
        num_iter : int
            Default number of PGD iterations.
        device : str, optional
            Device for MACE torch computations, by default "cpu".
        track_history : bool, optional
            Whether to track energies, forces, perturbations, and gradients,
            by default True.
        target_energy : Optional[float], optional
            Optional target energy. If None, maximize energy, by default None.
        random_start : bool, optional
            Whether to initialize randomly inside the epsilon box, by default
            True.
        rng : Optional[np.random.Generator], optional
            Optional NumPy random generator for deterministic random starts, by
            default None.
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

    def _uses_mace_autograd(self, atoms: Any) -> bool:
        """Return True when the attached calculator looks like a MACE calculator.

        MACE calculators expose ``models``, ``z_table``, and ``r_max``. UMA does
        not expose this same batch-building interface, so UMA falls through to
        the generic ASE-force backend.
        """
        calc = getattr(atoms, "calc", None)
        return all(hasattr(calc, attr) for attr in ("models", "z_table", "r_max"))

    def _forward_pass_with_gradients(self, atoms: Any) -> tuple:
        """Perform a differentiable forward pass through a MACE model.

        This method is only used for MACE calculators. It recreates the MACE
        atomic batch, replaces the position tensor with a gradient-enabled
        tensor, evaluates the model energy, and computes forces via torch
        autograd.

        Parameters
        ----------
        atoms : Any
            ASE Atoms object with a MACE calculator attached.

        Returns
        -------
        tuple
            ``(energy, forces, positions)`` where ``energy`` is a scalar torch
            tensor, ``forces`` has shape ``(n_atoms, 3)``, and ``positions`` is
            the gradient-enabled torch position tensor.
        """
        import torch
        from mace.data import AtomicData, config_from_atoms

        calc = atoms.calc
        model = calc.models[0]
        positions_np = atoms.get_positions()

        config = config_from_atoms(atoms)
        atomic_data = AtomicData.from_config(
            config,
            z_table=calc.z_table,
            cutoff=calc.r_max,
        )
        batch = atomic_data.to_dict()

        model_dtype = next(model.parameters()).dtype
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                if torch.is_floating_point(batch[key]):
                    batch[key] = batch[key].to(model_dtype)

        if "batch" not in batch:
            batch["batch"] = torch.zeros(
                len(atoms), dtype=torch.long, device=self.device
            )
        if "ptr" not in batch:
            batch["ptr"] = torch.tensor(
                [0, len(atoms)], dtype=torch.long, device=self.device
            )

        positions = torch.tensor(
            positions_np,
            dtype=model_dtype,
            device=self.device,
            requires_grad=True,
        )
        batch["positions"] = positions

        if "natoms" in batch:
            natoms_val = batch["natoms"]
            if natoms_val.dim() == 0:
                batch["natoms"] = torch.tensor(
                    [len(atoms), len(atoms)],
                    dtype=torch.long,
                    device=self.device,
                )
            elif natoms_val.dim() == 1 and len(natoms_val) < 2:
                batch["natoms"] = torch.tensor(
                    [len(atoms), len(atoms)],
                    dtype=torch.long,
                    device=self.device,
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
            batch["head"] = torch.zeros(
                len(atoms), dtype=torch.long, device=self.device
            )

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
        """Compute the gradient of the attack loss with respect to positions."""
        if self._uses_mace_autograd(atoms):
            energy, _forces, positions = self._forward_pass_with_gradients(atoms)

            if loss_fn is not None:
                loss = loss_fn(energy)
            elif self.target_energy is not None:
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
        """Perform one PGD signed-gradient step."""
        gradients = self.compute_gradient(atoms, loss_fn=loss_fn)
        direction = -1.0 if self.target_energy is not None else 1.0
        perturbation = direction * self.alpha * np.sign(gradients)

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc

        self._record_history(perturbed_atoms, perturbation, gradients)
        return perturbed_atoms

    def _random_start(self, atoms: Any) -> Any:
        """Return atoms randomly initialized inside the L-infinity epsilon box."""
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
        """Execute PGD over multiple iterations."""
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
            self._clip_perturbations(perturbed_atoms)

        total_steps = self.num_iter if n_steps is None else n_steps
        for step in range(total_steps):
            perturbed_atoms = self.attack_step(
                perturbed_atoms,
                step=step,
                loss_fn=loss_fn,
            )
            self._clip_perturbations(perturbed_atoms)

            if self.target_energy is not None:
                try:
                    current_energy = perturbed_atoms.get_potential_energy()
                    energy_diff = abs(current_energy - self.target_energy)
                    if energy_diff < 0.01:
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

    def _record_history(
        self,
        atoms: Any,
        perturbation: np.ndarray,
        gradients: np.ndarray,
    ) -> None:
        """Record energy, force, perturbation, and gradient history."""
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

    def _reset_history(self) -> None:
        """Clear tracked attack history."""
        if self.attack_history is None:
            return
        for key in self.attack_history:
            self.attack_history[key] = []