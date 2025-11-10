#!/usr/bin/env python3
"""
MACE relaxation functionality.
"""

from pathlib import Path
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from mace.calculators import mace


def load_structure(input_path):
    """
    Load structure from input file.
    
    Args:
        input_path: Path to input structure file (CIF, POSCAR, etc.)
        
    Returns:
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


def setup_calculator(atoms, model_path, device="cuda", dtype="float64"):
    """
    Initialize and attach MACE calculator to atoms object.
    
    Args:
        atoms: ASE Atoms object
        model_path: Path to MACE model file
        device: Device to use (cuda or cpu)
        dtype: Data type for calculations
        
    Returns:
        ASE Atoms object with calculator attached, or None if setup fails
    """
    try:
        print(f"[INFO] Loading MACE model: {model_path} on {device}")
        atoms.calc = mace.MACECalculator(
            model_paths=model_path, 
            device=device, 
            default_dtype=dtype
        )
        return atoms
    except Exception as e:
        print(f"[ERROR] Failed to setup MACE calculator: {e}")
        return None


def get_optimizer_class(optimizer_name):
    """
    Get the ASE optimizer class from name.
    
    Args:
        optimizer_name: Name of optimizer ("BFGS" or "LBFGS")
        
    Returns:
        Optimizer class
    """
    optimizers = {
        "BFGS": BFGS,
        "LBFGS": LBFGS
    }
    return optimizers.get(optimizer_name, LBFGS)


def run_relaxation(atoms, traj_path, fmax=0.01, max_steps=300, optimizer="LBFGS"):
    """
    Run structural relaxation.
    
    Args:
        atoms: ASE Atoms object with calculator attached
        traj_path: Path to save trajectory file
        fmax: Force convergence criterion (eV/Å)
        max_steps: Maximum number of optimization steps
        optimizer: Name of optimizer to use
        
    Returns:
        bool: True if relaxation completed successfully, False otherwise
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
    """
    Save relaxed structure to output files.
    
    Args:
        atoms: ASE Atoms object to save
        output_dir: Output directory path
        base_name: Base name for output files
        
    Returns:
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
