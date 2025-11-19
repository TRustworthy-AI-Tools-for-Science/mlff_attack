#!/usr/bin/env python3
"""
MACE relaxation functionality.
"""

from pathlib import Path
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from mace.calculators import mace as mace_calculator
import mace
import torch

def load_structure(input_path):
    """Load structure from input file.
    
    Parameters
    ----------
    input_path : str or Path
        Path to input structure file (CIF, POSCAR, etc.)
        
    Returns
    -------
    ase.Atoms or None
        ASE Atoms object, or None if loading fails
    """
    try:
        atoms = read(input_path)
        print(f"[INFO] Loaded structure: {input_path}")
        print(f"[INFO] Number of atoms: {len(atoms)}")
        print(f"[INFO] Chemical formula: {atoms.get_chemical_formula()}")
        return atoms
    except Exception as e:
        print(f"[ERROR] Failed to load structure from {input_path}: {e}")
        return None


def setup_calculator(atoms, model_path, device="cuda", dtype_str="float64"):
    """Initialize and attach MACE calculator to atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object
    model_path : str or Path or MACECalculator
        Path to MACE model file or existing MACECalculator instance
    device : str, optional
        Device to use (cuda or cpu), by default "cuda"
    dtype_str : str, optional
        Data type for calculations ("float32" or "float64"), by default "float64"
        
    Returns
    -------
    ase.Atoms or None
        ASE Atoms object with calculator attached, or None if setup fails
    """
    try:


        if isinstance(model_path, mace.calculators.mace.MACECalculator):
            print("model is already a MACECalculator")
            atoms.calc = model_path
        else:
            if dtype_str == "float32":
                dtype = torch.float32
            else:
                dtype = "float64"
            
            print(f"[INFO] Loading MACE model: {model_path} on {device}")
            atoms.calc = mace_calculator.MACECalculator(
                model_paths=model_path, 
                device=device, 
                default_dtype=dtype
            )
        return atoms
    except Exception as e:
        print(f"[ERROR] Failed to setup MACE calculator: {e}")
        return None


def get_optimizer_class(optimizer_name):
    """Get the ASE optimizer class from name.
    
    Parameters
    ----------
    optimizer_name : str
        Name of optimizer ("BFGS" or "LBFGS")
        
    Returns
    -------
    type
        ASE Optimizer class
    """
    optimizers = {
        "BFGS": BFGS,
        "LBFGS": LBFGS
    }
    return optimizers.get(optimizer_name, LBFGS)


def run_relaxation(atoms, traj_path, fmax=0.01, max_steps=300, optimizer="LBFGS"):
    """Run structural relaxation.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object with calculator attached
    traj_path : str or Path
        Path to save trajectory file
    fmax : float, optional
        Force convergence criterion (eV/Å), by default 0.01
    max_steps : int, optional
        Maximum number of optimization steps, by default 300
    optimizer : str, optional
        Name of optimizer to use ("BFGS" or "LBFGS"), by default "LBFGS"
        
    Returns
    -------
    bool
        True if relaxation completed successfully, False otherwise
    """
    try:
        print(f"[INFO] Starting relaxation with {optimizer} optimizer")
        print(f"[INFO] Convergence criterion: fmax = {fmax} eV/Å")
        print(f"[INFO] Maximum steps: {max_steps}")
        
        opt_cls = get_optimizer_class(optimizer)
        opt = opt_cls(atoms, trajectory=str(traj_path), logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        
        # Get final forces
        final_forces = atoms.get_forces()
        max_force = max([sum(f**2)**0.5 for f in final_forces])
        
        converged = max_force < fmax
        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"[INFO] Relaxation {status} after {opt.nsteps} steps")
        print(f"[INFO] Final maximum force: {max_force:.6f} eV/Å")
        
        return True
    except Exception as e:
        print(f"[ERROR] Relaxation failed: {e}")
        return False


def save_results(atoms, output_dir, base_name="relaxed"):
    """Save relaxed structure to output files.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object to save
    output_dir : str or Path
        Output directory path
    base_name : str, optional
        Base name for output files, by default "relaxed"
        
    Returns
    -------
    Path or None
        Path to saved CIF file, or None if saving fails
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cif_path = output_dir / f"{base_name}.cif"
        write(cif_path, atoms)
        print(f"[INFO] Saved relaxed structure to: {cif_path}")
        
        return cif_path
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return None
