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
        required=True,
        help="Path to input CIF file"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to MACE (include .model) or UMA (omit .pt) file"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on"
    )

    parser.add_argument(
        "--outdir",
        required=True,
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
        required=True,
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
        nargs="?",
        const="true",
        default=None,
        choices=["true", "True", "false", "False"],
        help=(
            "Clip total perturbation displacements to epsilon. "
            "Use --clip or --clip true to enable, --clip false to disable."
        ),
    )

    parser.add_argument(
        "--mace-head",
        default=None,
        choices=["omat_pbe", "omol", "spice_wB97M", "rgd1_b3lyp", "oc20_usemppbe", "matpes_r2scan"],
        help="Only used with MACE-MH model",
    )

    parser.add_argument(
        "--uma-task",
        default=None,
        choices=["oc20", "oc22", "oc25", "omat", "omol", "odac", "omc"],
        help="Only used with UMA model",
    )

    parser.add_argument(
        "--uma-charge",
        type=int,
        default=None,
        help="Molecular charge only used with UMA model and --uma-task omol",
    )

    parser.add_argument(
        "--uma-spin",
        type=int,
        default=None,
        help="Spin multiplicity only used with UMA model and --uma-task omol",
    )

    return parser.parse_args()


def main():
    """Run the adversarial attack CLI."""
    # Parse command line arguments
    args = parse_args()
    attack_type = args.type.lower()

    if attack_type != "pgd" and args.alpha is not None:
        raise SystemExit("--alpha can only be used with --type pgd")

    model_name = Path(args.model).name.lower()
    is_mace_mh = model_name.startswith("mace-mh")
    if model_name.startswith("uma"):
        calculator = "uma"
    elif model_name.startswith("mace"):
        calculator = "mace"
    else:
        raise SystemExit(
            "--model basename must start with 'uma' for UMA or 'mace' for MACE"
        )
    
    if calculator == "mace":
        if args.uma_task is not None:
            raise SystemExit("--uma-task can only be used with UMA")
        if args.uma_charge is not None:
            raise SystemExit("--uma-charge can only be used with UMA")
        if args.uma_spin is not None:
            raise SystemExit("--uma-spin can only be used with UMA")
        if args.mace_head is None and is_mace_mh:
            args.mace_head = "omat_pbe"

    if args.mace_head is not None and not is_mace_mh:
        raise SystemExit("--mace-head can only be used with MACE-MH models")

    if calculator == "uma":
        if args.mace_head is not None:
            raise SystemExit("--mace-head can only be used with MACE-MH models")
        if args.uma_task is None:
            args.uma_task = "omat"
        if args.uma_charge is None:
            args.uma_charge = 0
        if args.uma_spin is None:
            args.uma_spin = 1

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
        logger.info("[ERROR] Failed to load structure from %s", input_cif)
        return 1

    mace_head = args.mace_head
    uma_task = args.uma_task
    uma_charge = args.uma_charge
    uma_spin = args.uma_spin

    if calculator == "uma":
        atoms.info["charge"] = uma_charge
        atoms.info["spin"] = uma_spin

    logger.info("   Loaded %s atoms: %s", len(atoms), atoms.get_chemical_formula())

    # Generate perturbed structure
    logger.info("\nGenerating perturbed structure with epsilon=%s Å", epsilon)
    try:
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
            clip=clip,
            calculator=calculator,
            mace_head=mace_head,
            uma_task=uma_task,
            uma_charge=uma_charge,
            uma_spin=uma_spin,
        )
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        logger.info("[ERROR] Failed to generate attack. Run calc-single first and use the same model to generate attack: %s", exc)
        return 1

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
