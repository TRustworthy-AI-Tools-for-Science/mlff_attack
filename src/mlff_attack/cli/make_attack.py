#!/usr/bin/env python3
"""
CLI entry point for MACE single structure attack.
"""

import argparse
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mlff_attack.attacks import make_attack, visualize_perturbation
from mlff_attack.relaxation import load_structure

logger = logging.getLogger(__name__)


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
        "--outdir",
        type=str,
        default=None,
        help=(
            "Path to output directory "
            "(default: auto-generated from input with '_perturbed' suffix)"
        )
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

    parser.add_argument(
        "--type",
        type=str,
        default="fgsm",
        choices=["fgsm", "FGSM", "pgd", "PGD"],
        help="Type of adversarial attack to perform"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Perturbation step size in Angstroms"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="PGD step size. If not provided, use epsilon / n_steps",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=1,
        help="Number of attack iterations",
    )

    parser.add_argument(
        "--target-energy",
        type=float,
        default=None,
        help="Target energy for attack (if None, maximize energy)"
    )

    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        choices=["true", "True", "false", "False"],
        help="Clip total perturbation displacements to epsilon (true or false)"
    )

    return parser.parse_args()


def main():
    """Run the MACE adversarial attack CLI."""
    # Parse command line arguments
    args = parse_args()
    attack_type = args.type.lower()

    if attack_type != "pgd" and args.alpha is not None:
        raise SystemExit("--alpha can only be used with --type pgd")

    # Override configuration with command line arguments
    input_cif = args.input
    model_path = args.model
    device = args.device
    epsilon = args.epsilon
    alpha = args.alpha
    n_steps = args.n_steps
    target_energy = args.target_energy
    clip = args.clip
    if clip is not None:
        clip = clip in ("true", "True")

    # Determine output path
    if args.outdir is not None:
        output_cif = Path(args.outdir) / (Path(input_cif).stem + "_perturbed.cif")
    else:
        output_cif = Path(input_cif).with_name(Path(input_cif).stem + "_perturbed.cif")

    output_cif.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_cif.parent / "make_attack.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
    )

    # Load structure
    logger.info("\nLoading structure from: %s", input_cif)
    atoms = load_structure(input_cif)
    if atoms is None:
        raise RuntimeError(f"Failed to load structure from {input_cif}")
    logger.info("   Loaded %s atoms: %s", len(atoms), atoms.get_chemical_formula())

    # Generate perturbed structure
    logger.info("\nGenerating perturbed structure with epsilon=%s Å", epsilon)
    output_file, perturbed_atoms, _attack_details = make_attack(
        atoms=atoms,
        model_path=model_path,
        device=device,
        output_cif=output_cif,
        attack_type=attack_type,
        epsilon=epsilon,
        alpha=alpha,
        n_steps=n_steps,
        target_energy=target_energy,
        clip=clip
    )

    # Visualize perturbation
    if args.visualize:
        logger.info("\nVisualizing perturbation")
        # Store output filename in atoms info for visualization
        perturbed_atoms.info['filename'] = str(output_cif)
        fig = visualize_perturbation(
            atoms,
            perturbed_atoms,
            epsilon=epsilon,
            outdir=Path(output_cif).parent,
        )
        plt.close(fig)

    if output_file:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
