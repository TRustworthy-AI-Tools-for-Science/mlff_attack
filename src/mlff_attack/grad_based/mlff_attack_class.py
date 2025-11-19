from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import numpy as np


class MLFFAttack(ABC):
    """Base class for adversarial attacks on Machine Learning Force Fields.
    
    This abstract class defines the interface for implementing various attack
    strategies on MLFF models like MACE, ALIGNN, etc.
    
    Attributes:
        model: The MLFF model (e.g., MACE calculator, ALIGNN ForceField)
        epsilon: Maximum perturbation magnitude (in Angstroms for position attacks)
        device: Device for computations ('cpu', 'cuda', etc.)
        attack_history: Dictionary storing attack trajectory information
    """
    
    def __init__(
        self, 
        model: Any, 
        epsilon: float,
        device: str = 'cpu',
        track_history: bool = True
    ):
        """Initialize the attack.
        
        Parameters
        ----------
        model : Any
            MLFF model with calculator interface
        epsilon : float
            Maximum perturbation magnitude
        device : str, optional
            Device for PyTorch computations, by default 'cpu'
        track_history : bool, optional
            Whether to track attack progression, by default True
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.track_history = track_history
        self.attack_history = {
            'energies': [],
            'max_forces': [],
            'perturbations': [],
            'gradients': []
        } if track_history else None
        self._original_positions = None
        self._perturbed_positions = None
    
    @abstractmethod
    def compute_gradient(
        self, 
        atoms: Any,
        loss_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute gradient of loss with respect to atomic positions.
        
        Parameters
        ----------
        atoms : Any
            ASE Atoms object or equivalent structure
        loss_fn : Optional[Callable], optional
            Optional custom loss function (default: maximize energy), by default None
            
        Returns
        -------
        np.ndarray
            Gradient array with shape (n_atoms, 3)
        """
        pass
    
    @abstractmethod
    def attack_step(
        self,
        atoms: Any,
        step: int = 0
    ) -> Any:
        """Perform one step of the adversarial attack.
        
        Parameters
        ----------
        atoms : Any
            Current atomic structure
        step : int, optional
            Current iteration number, by default 0
            
        Returns
        -------
        Any
            Updated atoms object with perturbed positions
        """
        pass
    
    def attack(
        self, 
        atoms: Any,
        n_steps: int = 1,
        clip: bool = True
    ) -> Any:
        """Execute the complete attack.
        
        Parameters
        ----------
        atoms : Any
            Input atomic structure to attack
        n_steps : int, optional
            Number of attack iterations, by default 1
        clip : bool, optional
            Whether to clip perturbations to epsilon bound, by default True
            
        Returns
        -------
        Any
            Perturbed atoms object
        """
        if self._original_positions is None:
            self._original_positions = atoms.get_positions().copy()
        
        perturbed_atoms = atoms.copy()
        
        for step in range(n_steps):
            perturbed_atoms = self.attack_step(perturbed_atoms, step)
            
            if clip:
                self._clip_perturbations(perturbed_atoms)
        
        self._perturbed_positions = perturbed_atoms.get_positions().copy()
        return perturbed_atoms
    
    def _clip_perturbations(self, atoms: Any) -> None:
        """Ensure perturbations stay within epsilon bound.
        
        Parameters
        ----------
        atoms : Any
            Atoms object to clip (modified in-place)
        """
        if self._original_positions is None:
            return
            
        current_pos = atoms.get_positions()
        perturbations = current_pos - self._original_positions
        
        # Clip per-atom displacement magnitude
        magnitudes = np.linalg.norm(perturbations, axis=1, keepdims=True)
        mask = magnitudes > self.epsilon
        
        if np.any(mask):
            perturbations[mask.squeeze()] *= (self.epsilon / magnitudes[mask])
            atoms.set_positions(self._original_positions + perturbations)
    
    def save_perturbation(
        self, 
        filepath: str,
        include_metadata: bool = True
    ) -> None:
        """Save perturbation data to file.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        include_metadata : bool, optional
            Whether to include attack parameters, by default True
        """
        if self._original_positions is None or self._perturbed_positions is None:
            raise ValueError("No attack has been performed yet")
        
        data = {
            'original_positions': self._original_positions,
            'perturbed_positions': self._perturbed_positions,
            'perturbations': self._perturbed_positions - self._original_positions,
        }
        
        if include_metadata:
            data['epsilon'] = self.epsilon
            data['device'] = self.device
            
        if self.track_history and self.attack_history:
            for key, value in self.attack_history.items():
                if value:  # Only save non-empty lists
                    data[f'history_{key}'] = np.array(value)
        
        np.savez(filepath, **data)
    
    def load_perturbation(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load perturbation data from file.
        
        Parameters
        ----------
        filepath : str
            Input file path (.npz format)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing loaded data
        """
        data = np.load(filepath)
        self._original_positions = data['original_positions']
        self._perturbed_positions = data['perturbed_positions']
        return {key: data[key] for key in data.files}
    
    def reset(self) -> None:
        """Reset attack state."""
        self._original_positions = None
        self._perturbed_positions = None
        if self.attack_history:
            for key in self.attack_history:
                self.attack_history[key] = []
    
    def get_perturbation_stats(self) -> Dict[str, float]:
        """Get statistics about the current perturbation.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with perturbation statistics
        """
        if self._original_positions is None or self._perturbed_positions is None:
            return {}
        
        perturbations = self._perturbed_positions - self._original_positions
        magnitudes = np.linalg.norm(perturbations, axis=1)
        
        return {
            'mean_displacement': float(np.mean(magnitudes)),
            'max_displacement': float(np.max(magnitudes)),
            'std_displacement': float(np.std(magnitudes)),
            'total_atoms': len(magnitudes),
            'atoms_perturbed': int(np.sum(magnitudes > 1e-6))
        }