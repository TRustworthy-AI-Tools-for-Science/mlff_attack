#!/usr/bin/env python3
"""
CLI entry point for MACE or UMA single structure relaxation.
"""
import json
import argparse
import logging
from pathlib import Path
import sys

from mlff_attack.relaxation import (
    load_structure,
    setup_calculator,
    run_relaxation,
    save_results,
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for MACE or UMA single structure relaxation."""
    parser = argparse.ArgumentParser(
        description="Relax a single CIF with MACE or UMA."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CIF file",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to MACE (include .model) or UMA (omit .pt) file",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence criterion (eV/Angstrom)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum relaxation steps",
    )

    parser.add_argument(
        "--optimizer",
        default="LBFGS",
        choices=["BFGS", "LBFGS"],
        help="ASE optimizer",
    )

    parser.add_argument(
        "--task",
        default=None,
        choices=["oc20", "oc22", "oc25", "omat", "omol", "odac", "omc"],
        help="Only used with UMA model",
    )

    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Molecular charge only used with UMA model and --task omol",
    )

    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        help="Spin multiplicity only used with UMA model and --task omol",
    )

    args = parser.parse_args()
    
    if args.model.startswith("uma"):
        calculator = "uma"
    elif args.model.startswith("mace"):
        calculator = "mace"
    else:
        raise SystemExit(
            "--model must start with 'uma' for UMA or 'mace' for MACE"
        )
    
    if calculator == "uma":
        if args.task is None:
            args.task = "omat"
        if args.charge is None:
            args.charge = 0
        if args.spin is None:
            args.spin = 1

    if calculator == "mace":
        if args.task is not None:
            parser.error("--task can only be used with UMA")
        if args.charge is not None:
            parser.error("--charge can only be used with UMA")
        if args.spin is not None:
            parser.error("--spin can only be used with UMA")

    # Setup output paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=outdir / "calc_single.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
    )
    traj_path = outdir / "relaxed.traj"

    # Load structure
    atoms = load_structure(args.input)
    if atoms is None:
        logger.info("[ERROR] Failed to load input structure for %s.", args.input)
        return 1

    metadata_path = outdir / "metadata.json"
    metadata = {
        "calculator": calculator,
        "fmax": args.fmax,
        "task": args.task,
        "charge": args.charge,
        "spin": args.spin,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("[INFO] Saved required calculation metadata for following visualizations and attacks to: %s", metadata_path)

    # Setup calculator
    atoms = setup_calculator(
        atoms,
        args.model,
        args.device,
        calculator=calculator,
        uma_task=args.task,
        uma_charge=args.charge,
        uma_spin=args.spin,
    )

    if atoms is None:
        logger.info(
            "[ERROR] Failed to setup %s calculator with model %s.",
            calculator.upper(),
            args.model,
        )
        return 1

    # Run relaxation
    success = run_relaxation(
        atoms=atoms,
        traj_path=traj_path,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
    )

    if not success:
        logger.info("[ERROR] Relaxation failed.")
        return 1

    # Save results
    cif_path = save_results(atoms, outdir)
    if cif_path is None:
        return 1

    logger.info(
        "[DONE] Relaxation complete. Trajectory -> %s, CIF -> %s",
        traj_path,
        cif_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
