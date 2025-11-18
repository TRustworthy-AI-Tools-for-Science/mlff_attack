#!/usr/bin/env python3
"""
CLI entry point for MACE single structure attack.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import argparse


import matplotlib.pyplot as plt
from mlff_attack.relaxation import (
    load_structure,
    setup_calculator,
)
from mlff_attack.attacks import make_attack, visualize_perturbation

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform adversarial attack on atomic structures using MACE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="initial_cifs/chemistry_value_isovalent_0_05_18_traj.cif",
        help="Path to input CIF file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mace-mpa-0-medium.model",
        help="Path to MACE model file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on"
    )
    
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Perturbation step size in Angstroms"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CIF file (default: auto-generated from input with '_perturbed' suffix)"
    )
    
    parser.add_argument(
        "--target-energy",
        type=float,
        default=None,
        help="Target energy for attack (if None, maximize energy)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate visualization plot"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Skip visualization plot generation"
    )
    
    return parser.parse_args()



def main():

    # Parse command line arguments
    args = parse_args()
    
    # Override configuration with command line arguments
    input_cif = args.input
    model_path = args.model
    device = args.device
    epsilon = args.epsilon
    target_energy = args.target_energy
    
    # Determine output path
    if args.output is not None:
        output_cif = Path(args.output)
    else:
        output_cif = Path(input_cif).with_name(Path(input_cif).stem + "_perturbed.cif")

    # Load structure
    print(f"\nLoading structure from: {input_cif}")
    atoms = load_structure(input_cif)
    if atoms is None:
        raise RuntimeError(f"Failed to load structure from {input_cif}")
    print(f"   Loaded {len(atoms)} atoms: {atoms.get_chemical_formula()}")

    # Generate perturbed structure
    print(f"\nGenerating perturbed structure with epsilon={epsilon} Å")
    output_file, perturbed_atoms = make_attack(
        atoms=atoms,
        model_path=model_path,
        device=device,
        epsilon=epsilon,
        target_energy=target_energy,
        output_cif=output_cif
    )

    # Visualize perturbation
    if args.visualize:
        print(f"\nVisualizing perturbation")
        # Store output filename in atoms info for visualization
        perturbed_atoms.info['filename'] = str(output_cif)
        fig = visualize_perturbation(atoms, perturbed_atoms, epsilon=epsilon, outdir=Path(output_cif).parent)
        plt.close(fig)

    if output_file:
        return 0
    else:
        return 1

    
if __name__ == "__main__":
    exit(main())
