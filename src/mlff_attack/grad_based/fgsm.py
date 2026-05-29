"""Fast Gradient Sign Method (FGSM) attack implementation for MLFF models."""

from datetime import datetime
import logging
from typing import Any, Callable, Optional

import numpy as np

from mlff_attack.grad_based.mlff_attack_class import MLFFAttack

logger = logging.getLogger(__name__)


class FGSM_ASE(MLFFAttack):
    """FGSM attack implementation for ASE-compatible MLFF calculators.

    The Fast Gradient Sign Method computes the gradient of the loss with respect
    to atomic positions and perturbs them in the direction that maximizes the
    loss. For targeted attacks, the sign is reversed so the perturbation moves
    toward the target energy objective.

    MACE calculators expose enough torch internals to compute exact autograd
    gradients. UMA calculators expose the standard ASE calculator API, so this
    class computes gradients from forces instead.

    Attributes
    ----------
    model : Any
        ASE-compatible calculator attached to atoms.
    epsilon : float
        Maximum perturbation size in Angstroms. For iterative FGSM, each step
        uses ``epsilon / n_steps`` before optional clipping.
    device : str
        Device for torch computations when the MACE autograd backend is used.
    target_energy : float or None
        Optional target energy. If None, the attack maximizes energy.
    """

    def __init__(
        self,
        model: Any,
        epsilon: float = 0.01,
        device: str = "cpu",
        track_history: bool = True,
        target_energy: Optional[float] = None,
    ):
        """Initialize the unified FGSM attack.

        Parameters
        ----------
        model : Any
            ASE-compatible calculator. This may be a MACE calculator, UMA
            calculator, or another calculator that provides energies and forces.
        epsilon : float, optional
            Perturbation size in Angstroms, by default 0.01.
        device : str, optional
            Device for MACE torch computations, by default "cpu".
        track_history : bool, optional
            Whether to track energies, forces, perturbations, and gradients,
            by default True.
        target_energy : Optional[float], optional
            Optional target energy. If None, maximize energy, by default None.
        """
        super().__init__(model, epsilon, device, track_history)
        self.target_energy = target_energy
        self._last_energy = None
        self._last_gradients = None
        self._initial_energy = None

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
        """Compute the gradient of the attack loss with respect to positions.

        Parameters
        ----------
        atoms : Any
            ASE Atoms object with an attached calculator.
        loss_fn : Optional[Callable], optional
            Optional custom torch loss function. This is supported only for the
            MACE autograd backend. UMA/generic ASE force attacks do not expose a
            torch energy tensor, by default None.

        Returns
        -------
        np.ndarray
            Gradient array with shape ``(n_atoms, 3)``.
        """
        if self._uses_mace_autograd(atoms):
            energy, _forces, positions = self._forward_pass_with_gradients(atoms)

            if loss_fn is not None:
                loss = loss_fn(energy)
            elif self.target_energy is not None:
                loss = -((energy - self.target_energy) ** 2)
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

    def attack_step(self, atoms: Any, step: int = 0, n_steps: int = 1) -> Any:
        """Perform one FGSM step.

        Parameters
        ----------
        atoms : Any
            Current atomic structure with calculator attached.
        step : int, optional
            Current step index, by default 0.
        n_steps : int, optional
            Total number of FGSM steps. The per-step displacement is
            ``epsilon / n_steps``, by default 1.

        Returns
        -------
        Any
            New Atoms object with perturbed positions and the same calculator.
        """
        gradients = self.compute_gradient(atoms)
        step_size = self.epsilon / n_steps
        direction = -1.0 if self.target_energy is not None else 1.0
        perturbation = direction * step_size * np.sign(gradients)

        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(atoms.get_positions() + perturbation)
        perturbed_atoms.calc = atoms.calc

        self._record_history(perturbed_atoms, perturbation, gradients)
        return perturbed_atoms

    def attack(self, atoms: Any, n_steps: int = 1, clip: Optional[bool] = None) -> Any:
        """Execute FGSM or iterative FGSM.

        Parameters
        ----------
        atoms : Any
            Input atomic structure with calculator attached.
        n_steps : int, optional
            Number of attack steps. Use 1 for FGSM and greater than 1 for
            iterative FGSM, by default 1.
        clip : Optional[bool], optional
            Whether to clip total coordinate displacement to ``epsilon``. If
            None, defaults to False for FGSM, by default None.

        Returns
        -------
        Any
            Final perturbed Atoms object.
        """
        if clip is None:
            clip = False

        self.reset()
        self._original_positions = atoms.get_positions().copy()
        try:
            self._initial_energy = atoms.get_potential_energy()
        except (ValueError, RuntimeError):
            self._initial_energy = None

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
        """Record energy, force, perturbation, and gradient history.

        Parameters
        ----------
        atoms : Any
            Perturbed atoms for the current step.
        perturbation : np.ndarray
            Position perturbation applied during this step.
        gradients : np.ndarray
            Gradients used to choose the perturbation direction.
        """
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

    def save_perturbation(
        self,
        filepath: str,
        atoms_original: Optional[Any] = None,
        atoms_perturbed: Optional[Any] = None,
        include_metadata: bool = True,
    ) -> None:
        """Save perturbation data and optional metadata to a compressed NPZ file.

        Parameters
        ----------
        filepath : str
            Output path for the ``.npz`` file.
        atoms_original : Optional[Any], optional
            Original Atoms object used to save symbols, cell, and PBC, by default
            None.
        atoms_perturbed : Optional[Any], optional
            Perturbed Atoms object used to save final energy metadata, by default
            None.
        include_metadata : bool, optional
            Whether to include attack parameters and energies, by default True.
        """
        if self._original_positions is None or self._perturbed_positions is None:
            raise ValueError("No attack has been performed yet")

        data = {
            "original_positions": self._original_positions,
            "perturbed_positions": self._perturbed_positions,
            "displacement": self._perturbed_positions - self._original_positions,
        }

        if atoms_original is not None:
            data["chemical_symbols"] = np.array(
                atoms_original.get_chemical_symbols(), dtype="U2"
            )
            data["cell"] = atoms_original.get_cell().array
            data["pbc"] = atoms_original.get_pbc()

        if include_metadata:
            data["epsilon"] = self.epsilon
            data["device"] = self.device
            data["target_energy"] = (
                self.target_energy if self.target_energy is not None else np.nan
            )
            data["timestamp"] = datetime.now().isoformat()

            if atoms_original is not None and hasattr(atoms_original, "calc"):
                try:
                    data["energy_original"] = atoms_original.get_potential_energy()
                except (ValueError, RuntimeError):
                    pass

            if atoms_perturbed is not None and hasattr(atoms_perturbed, "calc"):
                try:
                    data["energy_perturbed"] = atoms_perturbed.get_potential_energy()
                    if "energy_original" in data:
                        data["energy_change"] = (
                            data["energy_perturbed"] - data["energy_original"]
                        )
                except (ValueError, RuntimeError):
                    pass

        if self.track_history and self.attack_history:
            for key, value in self.attack_history.items():
                if value:
                    data[f"history_{key}"] = np.array(value)

        np.savez_compressed(filepath, **data)

    def get_attack_summary(self) -> dict:
        """Return summary statistics for the most recent attack.

        Returns
        -------
        dict
            Perturbation statistics plus available energy and force history.
        """
        summary = self.get_perturbation_stats()

        if self.track_history and self.attack_history:
            if self.attack_history["energies"]:
                summary["initial_energy"] = self._initial_energy
                summary["final_energy"] = self.attack_history["energies"][-1]
                if self._initial_energy is not None:
                    summary["energy_change"] = (
                        summary["final_energy"] - self._initial_energy
                    )

            if self.attack_history["max_forces"]:
                summary["final_max_force"] = self.attack_history["max_forces"][-1]

        summary["target_energy"] = self.target_energy
        summary["n_iterations"] = (
            len(self.attack_history["energies"]) if self.track_history else 0
        )

        return summary