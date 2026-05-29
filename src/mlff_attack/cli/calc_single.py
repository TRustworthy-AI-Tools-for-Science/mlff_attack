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

REPO_ROOT = Path(__file__).resolve().parents[3]


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

    args = parser.parse_args()
    
    model_name = Path(args.model).name.lower()
    is_mace_mh = model_name.startswith("mace-mh")

    if model_name.startswith("uma"):
        calculator = "uma"
    elif model_name.startswith("mace"):
        calculator = "mace"
    else:
        raise SystemExit(
            "--model must start with 'uma' for UMA or 'mace' for MACE"
        )

    if calculator == "mace":
        if args.uma_task is not None:
            parser.error("--uma-task can only be used with UMA")
        if args.uma_charge is not None:
            parser.error("--uma-charge can only be used with UMA")
        if args.uma_spin is not None:
            parser.error("--uma-spin can only be used with UMA")
        if args.mace_head is None and is_mace_mh:
            args.mace_head = "omat_pbe"

    if args.mace_head is not None and not is_mace_mh:
        parser.error("--mace-head can only be used with MACE-MH models")

    if calculator == "uma":
        if args.uma_task is None:
            args.uma_task = "omat"
        if args.uma_charge is None:
            args.uma_charge = 0
        if args.uma_spin is None:
            args.uma_spin = 1

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

    metadata = {
        "input": str(Path(args.input).resolve()),
        "model": Path(args.model).name,
        "calculator": calculator,
        "fmax": args.fmax,
        "mace_head": args.mace_head,
        "uma_task": args.uma_task,
        "uma_charge": args.uma_charge,
        "uma_spin": args.uma_spin,
    }
    root_metadata_path = REPO_ROOT / "previous_calculation.json"
    root_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("[INFO] Saved required calculation metadata for following visualizations and attacks to: %s", root_metadata_path)

    # Setup calculator
    atoms = setup_calculator(
        atoms,
        args.model,
        args.device,
        calculator=calculator,
        mace_head=args.mace_head,
        uma_task=args.uma_task,
        uma_charge=args.uma_charge,
        uma_spin=args.uma_spin,
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
