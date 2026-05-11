#!/usr/bin/env python3
"""
Example script demonstrating how to use the FGSM_MACE attack class.

This script shows:
1. Basic FGSM attack (single step)
2. Iterative FGSM (I-FGSM) attack (multiple steps)
3. Using history tracking and statistics
4. Saving and loading perturbations
"""

from pathlib import Path
import logging
logger = logging.getLogger(__name__)
import ase
from ase.io import read, write
from mlff_attack.relaxation import setup_calculator, load_structure
from mlff_attack.grad_based.fgsm import FGSM_MACE
from mlff_attack.attacks import visualize_perturbation
from ase import build
from mace.calculators import mace_mp


def basic_fgsm_example():
    """Example 1: Basic FGSM attack (single step)."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Basic FGSM Attack (Single Step)")
    logger.info("=" * 70)
    
    # Load structure
    # atoms = load_structure("initial_cifs/chemistry_value_isovalent_0_05_18_traj.cif")
    atoms = build.molecule("H2O")  # Using a simple molecule for demonstration
    
    # Setup MACE calculator
    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    device = "cpu"
    atoms = setup_calculator(atoms, model, device)

    # Get original energy
    orig_energy = atoms.get_potential_energy()
    logger.info(f"\nOriginal energy: {orig_energy:.4f} eV")
    
    # Create FGSM attack
    fgsm = FGSM_MACE(
        model=atoms.calc,
        epsilon=0.05,  # 0.05 Angstrom perturbation
        device=device,
        track_history=True,
        target_energy=None  # Maximize energy
    )
    
    # Execute attack (single step)
    perturbed_atoms = fgsm.attack(atoms, n_steps=1, clip=True)
    
    # Get results
    pert_energy = perturbed_atoms.get_potential_energy()
    logger.info(f"Perturbed energy: {pert_energy:.4f} eV")
    logger.info(f"Energy change: {pert_energy - orig_energy:+.4f} eV")
    
    # Print statistics
    logger.info("\nPerturbation Statistics:")
    stats = fgsm.get_perturbation_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Get attack summary
    logger.info("\nAttack Summary:")
    summary = fgsm.get_attack_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Save perturbed structure
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    write(output_dir / "fgsm_perturbed.cif", perturbed_atoms)
    
    # Save perturbation data
    fgsm.save_perturbation(
        output_dir / "fgsm_perturbation.npz",
        atoms_original=atoms,
        atoms_perturbed=perturbed_atoms
    )
    logger.info(f"\nSaved outputs to {output_dir}/")
    
    # Visualize
    visualize_perturbation(atoms, perturbed_atoms, epsilon=0.05, outdir=output_dir)
    
    return atoms, perturbed_atoms, fgsm


def iterative_fgsm_example():
    """Example 2: Iterative FGSM (I-FGSM) attack."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Iterative FGSM (I-FGSM) Attack (5 Steps)")
    logger.info("=" * 70)
    
    # Load structure
    atoms = build.molecule("H2O")  # Using a simple molecule for demonstration
    
    # Setup MACE calculator
    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    device = "cpu"
    atoms = setup_calculator(atoms, model, device)
    
    orig_energy = atoms.get_potential_energy()
    logger.info(f"\nOriginal energy: {orig_energy:.4f} eV")
    
    # Create I-FGSM attack (smaller epsilon per step)
    attack = FGSM_MACE(
        model=atoms.calc,
        epsilon=0.01,  # Smaller step size
        device=device,
        track_history=True,
        target_energy=None
    )
    
    # Execute iterative attack (5 steps with clipping)
    n_steps = 5
    perturbed_atoms = attack.attack(atoms, n_steps=n_steps, clip=True)
    
    pert_energy = perturbed_atoms.get_potential_energy()
    logger.info(f"Perturbed energy: {pert_energy:.4f} eV")
    logger.info(f"Energy change: {pert_energy - orig_energy:+.4f} eV")
    
    # Show energy progression
    logger.info(f"\nEnergy progression over {n_steps} steps:")
    for i, energy in enumerate(attack.attack_history['energies'], 1):
        logger.info(f"  Step {i}: {energy:.4f} eV")
    
    # Print statistics
    stats = attack.get_perturbation_stats()
    logger.info(f"\nFinal displacement: {stats['max_displacement']:.4f} Å (max)")
    
    # Save outputs
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    write(output_dir / "ifgsm_perturbed.cif", perturbed_atoms)
    attack.save_perturbation(
        output_dir / "ifgsm_perturbation.npz",
        atoms_original=atoms,
        atoms_perturbed=perturbed_atoms
    )
    logger.info(f"\nSaved outputs to {output_dir}/")
    
    return atoms, perturbed_atoms, attack


def targeted_attack_example():
    """Example 3: Targeted attack (reach specific energy)."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Targeted Energy Attack")
    logger.info("=" * 70)
    
    # Load structure
    atoms = build.molecule("H2O")  # Using a simple molecule for demonstration
    
    # Setup MACE calculator
    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    device = "cpu"
    atoms = setup_calculator(atoms, model, device)
    
    orig_energy = atoms.get_potential_energy()
    target_energy = orig_energy + 2.0  # Try to increase energy by 2 eV
    
    logger.info(f"\nOriginal energy: {orig_energy:.4f} eV")
    logger.info(f"Target energy: {target_energy:.4f} eV")
    
    # Create targeted attack
    attack = FGSM_MACE(
        model=atoms.calc,
        epsilon=0.02,
        device=device,
        track_history=True,
        target_energy=target_energy  # Specify target
    )
    
    # Execute attack with multiple iterations
    perturbed_atoms = attack.attack(atoms, n_steps=10, clip=True)
    
    pert_energy = perturbed_atoms.get_potential_energy()
    logger.info(f"Perturbed energy: {pert_energy:.4f} eV")
    logger.info(f"Distance to target: {abs(pert_energy - target_energy):.4f} eV")
    
    # Save outputs
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)

    write(output_dir / "targeted_perturbed.cif", perturbed_atoms)
    
    return atoms, perturbed_atoms, attack


def load_and_analyze_example():
    """Example 4: Load saved perturbation and analyze."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Load and Analyze Saved Perturbation")
    logger.info("=" * 70)
    
    # Load perturbation data
    attack = FGSM_MACE(
        model=None,  # Don't need model for loading
        epsilon=0.05,
        device="cpu"
    )
    
    data = attack.load_perturbation("example_outputs/fgsm_perturbation.npz")
    
    logger.info("\nLoaded perturbation data:")
    for key in data.keys():
        logger.info(f"  {key}")
    
    # Analyze
    if 'energy_original' in data and 'energy_perturbed' in data:
        logger.info(f"\nEnergy change: {data['energy_change']:.4f} eV")
    
    stats = attack.get_perturbation_stats()
    logger.info("\nDisplacement statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    import sys
    
    # Change to repository root if needed
    repo_root = Path(__file__).parent.parent.parent
    if (repo_root / "initial_cifs").exists():
        import os
        os.chdir(repo_root)
    
    # Run examples
    try:
        # Example 1: Basic FGSM
        atoms, perturbed, attack = basic_fgsm_example()
        
        # Example 2: Iterative FGSM
        atoms2, perturbed2, attack2 = iterative_fgsm_example()
        
        # Example 3: Targeted attack
        atoms3, perturbed3, attack3 = targeted_attack_example()
        
        # Example 4: Load and analyze
        load_and_analyze_example()
        
        logger.info("\n" + "=" * 70)
        logger.info("All examples completed successfully!")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        logger.info("Make sure you're running this from the repository root directory")
        logger.info("and that the required files exist.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.logger.info_exc()
        sys.exit(1)
