"""
Fast Gradient Sign Method (FGSM) attack implementation for MLFF models.

This module implements the FGSM attack specifically for MACE models,
extending the base MLFFAttack class.
"""

from typing import Optional, Callable, Any
import numpy as np
import torch
from datetime import datetime

from .mlff_attack_class import MLFFAttack
from mace.data import AtomicData, config_from_atoms


class FGSM_MACE(MLFFAttack):
    """FGSM attack implementation for MACE force field models.
    
    The Fast Gradient Sign Method computes the gradient of the loss with respect
    to atomic positions and perturbs them in the direction that maximizes the loss.
    
    Attributes:
        model: MACE calculator attached to atoms
        epsilon: Step size for perturbation (Angstroms)
        device: Device for PyTorch computations
        target_energy: Optional target energy for the attack
    """
    
    def __init__(
        self,
        model: Any,
        epsilon: float = 0.01,
        device: str = 'cpu',
        track_history: bool = True,
        target_energy: Optional[float] = None
    ):
        """Initialize FGSM attack for MACE models.
        
        Parameters
        ----------
        model : Any
            MACE calculator (will be attached to atoms)
        epsilon : float, optional
            Perturbation step size in Angstroms, by default 0.01
        device : str, optional
            Device for computations ('cpu' or 'cuda'), by default 'cpu'
        track_history : bool, optional
            Whether to track attack progression, by default True
        target_energy : Optional[float], optional
            Optional target energy (if None, maximize energy), by default None
        """
        super().__init__(model, epsilon, device, track_history)
        self.target_energy = target_energy
        self._last_energy = None
        self._last_gradients = None
    
    def _forward_pass_with_gradients(self, atoms: Any) -> tuple:
        """Perform forward pass through MACE model with gradient tracking.
        
        This uses the calculator's internal method to prepare the batch,
        then replaces positions with a gradient-enabled version.
        
        Parameters
        ----------
        atoms : Any
            ASE Atoms object with MACE calculator attached
        
        Returns
        -------
        tuple
            A tuple containing:
            
            - energy : torch.Tensor
                Total energy (scalar, requires_grad=True)
            - forces : torch.Tensor
                Forces on atoms (shape [n_atoms, 3], with gradients)
            - positions : torch.Tensor
                Position tensor (requires_grad=True)
        """
        calc = atoms.calc
        model = calc.models[0]
        # model = self.model
        # calc = atoms.calc
        
        # Save original positions
        positions_np = atoms.get_positions()
        
        # Create configuration from atoms
        config = config_from_atoms(atoms)
        
        # Create AtomicData with the calculator's settings
        atomic_data = AtomicData.from_config(
            config, z_table=calc.z_table, cutoff=calc.r_max
        )
        
        # Convert to dict
        batch = atomic_data.to_dict()
        
        # Move everything to the right device first
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        
        # Add batch indexing if not present (on correct device)
        if "batch" not in batch:
            batch["batch"] = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        if "ptr" not in batch:
            batch["ptr"] = torch.tensor([0, len(atoms)], dtype=torch.long, device=self.device)
        
        # Replace positions with gradient-enabled version
        positions = torch.tensor(
            positions_np, dtype=torch.float32 if self.device == "cpu" else torch.float64, device=self.device, requires_grad=True
        )
        batch["positions"] = positions
        
        # Check and fix natoms if present
        if "natoms" in batch:
            natoms_val = batch["natoms"]
            # Ensure it's 1D with 2 elements
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
        
        # Add head field if present in calculator (for multi-head models)
        if hasattr(calc, 'head') and calc.head is not None:
            # Map head name to index
            if hasattr(calc, 'heads') and calc.heads is not None:
                head_idx = calc.heads.index(calc.head) if calc.head in calc.heads else 0
            else:
                head_idx = 0
            batch["head"] = torch.full(
                (len(atoms),), head_idx, dtype=torch.long, device=self.device
            )
        elif "head" not in batch:
            batch["head"] = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        
        # Forward pass - disable computing forces in the model output
        model.eval()
        
        with torch.enable_grad():
            positions.requires_grad_(True)
            batch["positions"] = positions
            
            # Forward pass through model
            output = model(batch, training=False, compute_force=False)
            
            # Extract energy
            energy = output["energy"]
            if energy.dim() > 0:
                energy = energy.sum()
            
            # Compute forces as negative gradient
            forces = -torch.autograd.grad(
                outputs=energy,
                inputs=positions,
                retain_graph=True,
                create_graph=False
            )[0]
        
        return energy, forces, positions
    
    def compute_gradient(
        self,
        atoms: Any,
        loss_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute gradient of loss with respect to atomic positions.
        
        Parameters
        ----------
        atoms : Any
            ASE Atoms object with MACE calculator
        loss_fn : Optional[Callable], optional
            Optional custom loss function. If None, uses default
            (maximize energy or target energy loss), by default None
        
        Returns
        -------
        np.ndarray
            Gradient array with shape (n_atoms, 3)
        """
        # Forward pass with gradients
        energy, forces, positions = self._forward_pass_with_gradients(atoms)
        
        # Define loss
        if loss_fn is not None:
            loss = loss_fn(energy)
        elif self.target_energy is not None:
            # Try to reach target energy
            loss = (energy - self.target_energy) ** 2
        else:
            # Maximize energy (adversarial attack)
            loss = -energy
        
        # Backprop to get gradients w.r.t. positions
        loss.backward()
        grad_positions = positions.grad
        
        # Store for history tracking
        self._last_energy = energy.item()
        self._last_gradients = grad_positions.detach().cpu().numpy()
        
        return self._last_gradients
    
    def attack_step(
        self,
        atoms: Any,
        step: int = 0
    ) -> Any:
        """Perform one step of FGSM attack.
        
        Parameters
        ----------
        atoms : Any
            Current atomic structure with MACE calculator
        step : int, optional
            Current iteration number, by default 0
        
        Returns
        -------
        Any
            Updated atoms object with perturbed positions
        """
        # Compute gradients
        gradients = self.compute_gradient(atoms)
        
        # FGSM: perturbation is epsilon * sign of gradient
        perturbation = self.epsilon * np.sign(gradients)
        
        # Apply perturbation
        current_positions = atoms.get_positions()
        perturbed_positions = current_positions + perturbation
        
        # Create new atoms with perturbed positions
        perturbed_atoms = atoms.copy()
        perturbed_atoms.set_positions(perturbed_positions)
        
        # Ensure calculator is attached
        perturbed_atoms.calc = atoms.calc
        
        # Track history if enabled
        if self.track_history:
            try:
                perturbed_energy = perturbed_atoms.get_potential_energy()
                forces = perturbed_atoms.get_forces()
                max_force = np.max(np.linalg.norm(forces, axis=1))
                
                self.attack_history['energies'].append(perturbed_energy)
                self.attack_history['max_forces'].append(max_force)
                self.attack_history['perturbations'].append(perturbation.copy())
                self.attack_history['gradients'].append(gradients.copy())
            except Exception:
                pass  # Skip if energy calculation fails
        
        return perturbed_atoms
    
    def attack(
        self,
        atoms: Any,
        n_steps: int = 1,
        clip: bool = True
    ) -> Any:
        """Execute FGSM attack.
        
        For standard FGSM, n_steps=1. For iterative FGSM (I-FGSM), use n_steps>1.
        
        Parameters
        ----------
        atoms : Any
            Input atomic structure with MACE calculator
        n_steps : int, optional
            Number of attack iterations (1 for FGSM, >1 for I-FGSM), by default 1
        clip : bool, optional
            Whether to clip perturbations to epsilon bound, by default True
        
        Returns
        -------
        Any
            Perturbed atoms object
        """
        # Store original positions
        if self._original_positions is None:
            self._original_positions = atoms.get_positions().copy()
        
        # Execute attack
        perturbed_atoms = self.attack_step(atoms, n_steps)

        self._perturbed_positions = perturbed_atoms.get_positions().copy()
        
        return perturbed_atoms
    
    def save_perturbation(
        self,
        filepath: str,
        atoms_original: Optional[Any] = None,
        atoms_perturbed: Optional[Any] = None,
        include_metadata: bool = True
    ) -> None:
        """Save perturbation data with additional FGSM-specific information.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        atoms_original : Optional[Any], optional
            Optional original atoms (for saving chemical symbols, cell), by default None
        atoms_perturbed : Optional[Any], optional
            Optional perturbed atoms, by default None
        include_metadata : bool, optional
            Whether to include attack parameters, by default True
        """
        if self._original_positions is None or self._perturbed_positions is None:
            raise ValueError("No attack has been performed yet")
        
        # Build data dictionary
        data = {
            'original_positions': self._original_positions,
            'perturbed_positions': self._perturbed_positions,
            'displacement': self._perturbed_positions - self._original_positions,
        }
        
        # Add atomic structure information if provided
        if atoms_original is not None:
            data['chemical_symbols'] = np.array(
                atoms_original.get_chemical_symbols(), dtype='U2'
            )
            data['cell'] = atoms_original.get_cell().array
            data['pbc'] = atoms_original.get_pbc()
        
        # Add metadata
        if include_metadata:
            data['epsilon'] = self.epsilon
            data['device'] = self.device
            data['target_energy'] = self.target_energy if self.target_energy else np.nan
            data['timestamp'] = datetime.now().isoformat()
            
            # Add energy information if available
            if atoms_original is not None and hasattr(atoms_original, 'calc'):
                try:
                    data['energy_original'] = atoms_original.get_potential_energy()
                except Exception:
                    pass
            
            if atoms_perturbed is not None and hasattr(atoms_perturbed, 'calc'):
                try:
                    data['energy_perturbed'] = atoms_perturbed.get_potential_energy()
                    if 'energy_original' in data:
                        data['energy_change'] = (
                            data['energy_perturbed'] - data['energy_original']
                        )
                except Exception:
                    pass
        
        # Add history if tracked
        if self.track_history and self.attack_history:
            for key, value in self.attack_history.items():
                if value:
                    data[f'history_{key}'] = np.array(value)
        
        # Save
        np.savez_compressed(filepath, **data)
    
    def get_attack_summary(self) -> dict:
        """Get a summary of the attack results.
        
        Returns
        -------
        dict
            Dictionary with attack summary statistics
        """
        summary = self.get_perturbation_stats()
        
        if self.track_history and self.attack_history:
            if self.attack_history['energies']:
                summary['initial_energy'] = self.attack_history['energies'][0]
                summary['final_energy'] = self.attack_history['energies'][-1]
                summary['energy_change'] = (
                    summary['final_energy'] - summary['initial_energy']
                )
            
            if self.attack_history['max_forces']:
                summary['final_max_force'] = self.attack_history['max_forces'][-1]
        
        summary['target_energy'] = self.target_energy
        summary['n_iterations'] = len(self.attack_history['energies']) if self.track_history else 0
        
        return summary
