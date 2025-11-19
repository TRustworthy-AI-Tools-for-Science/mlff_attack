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
from ase.io import read, write
from mlff_attack.relaxation import setup_calculator, load_structure
from mlff_attack.grad_based.fgsm import FGSM_MACE
from mlff_attack.attacks import visualize_perturbation


def basic_fgsm_example():
    """Example 1: Basic FGSM attack (single step)."""
    print("=" * 70)
    print("EXAMPLE 1: Basic FGSM Attack (Single Step)")
    print("=" * 70)
    
    # Load structure
    atoms = load_structure("initial_cifs/chemistry_value_isovalent_0_05_18_traj.cif")
    
    # Setup MACE calculator
    model_path = "mace-mpa-0-medium.model"
    device = "cpu"
    atoms = setup_calculator(atoms, model_path, device)

    # Get original energy
    orig_energy = atoms.get_potential_energy()
    print(f"\nOriginal energy: {orig_energy:.4f} eV")
    
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
    print(f"Perturbed energy: {pert_energy:.4f} eV")
    print(f"Energy change: {pert_energy - orig_energy:+.4f} eV")
    
    # Print statistics
    print("\nPerturbation Statistics:")
    stats = fgsm.get_perturbation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get attack summary
    print("\nAttack Summary:")
    summary = fgsm.get_attack_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
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
    print(f"\nSaved outputs to {output_dir}/")
    
    # Visualize
    visualize_perturbation(atoms, perturbed_atoms, epsilon=0.05, outdir=output_dir)
    
    return atoms, perturbed_atoms, fgsm


def iterative_fgsm_example():
    """Example 2: Iterative FGSM (I-FGSM) attack."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Iterative FGSM (I-FGSM) Attack (5 Steps)")
    print("=" * 70)
    
    # Load structure
    atoms = read("initial_cifs/chemistry_value_isovalent_0_05_18_traj.cif")
    
    # Setup MACE calculator
    model_path = "mace-mpa-0-medium.model"
    device = "cpu"
    atoms = setup_calculator(atoms, model_path, device)
    
    orig_energy = atoms.get_potential_energy()
    print(f"\nOriginal energy: {orig_energy:.4f} eV")
    
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
    print(f"Perturbed energy: {pert_energy:.4f} eV")
    print(f"Energy change: {pert_energy - orig_energy:+.4f} eV")
    
    # Show energy progression
    print(f"\nEnergy progression over {n_steps} steps:")
    for i, energy in enumerate(attack.attack_history['energies'], 1):
        print(f"  Step {i}: {energy:.4f} eV")
    
    # Print statistics
    stats = attack.get_perturbation_stats()
    print(f"\nFinal displacement: {stats['max_displacement']:.4f} Å (max)")
    
    # Save outputs
    output_dir = Path("example_outputs")
    write(output_dir / "ifgsm_perturbed.cif", perturbed_atoms)
    attack.save_perturbation(
        output_dir / "ifgsm_perturbation.npz",
        atoms_original=atoms,
        atoms_perturbed=perturbed_atoms
    )
    print(f"\nSaved outputs to {output_dir}/")
    
    return atoms, perturbed_atoms, attack


def targeted_attack_example():
    """Example 3: Targeted attack (reach specific energy)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Targeted Energy Attack")
    print("=" * 70)
    
    # Load structure
    atoms = read("initial_cifs/chemistry_value_isovalent_0_05_18_traj.cif")
    
    # Setup calculator
    model_path = "mace-mpa-0-medium.model"
    device = "cpu"
    atoms = setup_calculator(atoms, model_path, device)
    
    orig_energy = atoms.get_potential_energy()
    target_energy = orig_energy + 2.0  # Try to increase energy by 2 eV
    
    print(f"\nOriginal energy: {orig_energy:.4f} eV")
    print(f"Target energy: {target_energy:.4f} eV")
    
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
    print(f"Perturbed energy: {pert_energy:.4f} eV")
    print(f"Distance to target: {abs(pert_energy - target_energy):.4f} eV")
    
    # Save outputs
    output_dir = Path("example_outputs")
    write(output_dir / "targeted_perturbed.cif", perturbed_atoms)
    
    return atoms, perturbed_atoms, attack


def load_and_analyze_example():
    """Example 4: Load saved perturbation and analyze."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Load and Analyze Saved Perturbation")
    print("=" * 70)
    
    # Load perturbation data
    attack = FGSM_MACE(
        model=None,  # Don't need model for loading
        epsilon=0.05,
        device="cpu"
    )
    
    data = attack.load_perturbation("example_outputs/fgsm_perturbation.npz")
    
    print("\nLoaded perturbation data:")
    for key in data.keys():
        print(f"  {key}")
    
    # Analyze
    if 'energy_original' in data and 'energy_perturbed' in data:
        print(f"\nEnergy change: {data['energy_change']:.4f} eV")
    
    stats = attack.get_perturbation_stats()
    print("\nDisplacement statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


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
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure you're running this from the repository root directory")
        print("and that the required files exist.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
