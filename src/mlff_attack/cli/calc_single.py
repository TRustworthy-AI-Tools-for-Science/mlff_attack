#!/usr/bin/env python3
"""
CLI entry point for MACE, UMA, CHGNet, or MTP single-structure relaxation.
"""

import argparse
import logging
from pathlib import Path
import sys

from mlff_attack.calculators import infer_calculator_type
from mlff_attack.relaxation import (
    load_structure,
    setup_calculator,
    run_relaxation,
    save_results,
)

from mlff_attack.random_seed import set_random_seed

logger = logging.getLogger(__name__)


def main():
    """Main entry point for MACE, UMA, CHGNet, or MTP relaxation."""
    parser = argparse.ArgumentParser(
        description="Relax a single CIF with MACE, UMA, CHGNet, or MTP."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CIF file",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="MACE model path, UMA model name, CHGNet model name, or MTP .almtp path",
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
        "--dtype",
        dest="dtype_str",
        default="float64",
        choices=["float32", "float64"],
        help="Floating point precision to request for calculator setup",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MACE/UMA/CHGNet/MTP calculator setup",
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

    try:
        calculator = infer_calculator_type(args.model)
    except ValueError as exc:
        parser.error(str(exc))

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

    elif calculator == "uma":
        if args.uma_task is None:
            args.uma_task = "omat"
        if args.uma_charge is None:
            args.uma_charge = 0
        if args.uma_spin is None:
            args.uma_spin = 1

    elif calculator == "chgnet":
        if args.mace_head is not None:
            parser.error("--mace-head can only be used with MACE-MH models")
        if args.uma_task is not None:
            parser.error("--uma-task can only be used with UMA")
        if args.uma_charge is not None:
            parser.error("--uma-charge can only be used with UMA")
        if args.uma_spin is not None:
            parser.error("--uma-spin can only be used with UMA")

    elif calculator == "mtp":
        if args.device != "cpu":
            parser.error("MTP only supports --device cpu")
        if args.dtype_str != "float64":
            parser.error("MTP only supports --dtype float64")
        if args.mace_head is not None:
            parser.error("--mace-head can only be used with MACE-MH models")
        if args.uma_task is not None:
            parser.error("--uma-task can only be used with UMA")
        if args.uma_charge is not None:
            parser.error("--uma-charge can only be used with UMA")
        if args.uma_spin is not None:
            parser.error("--uma-spin can only be used with UMA")

    # Setup output paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=outdir / "calc_single.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
    )
    logger.info("Calculator: %s", calculator.upper())
    if args.mace_head is not None:
        logger.info("MACE-MH head: %s", args.mace_head)
    if calculator == "uma":
        logger.info("UMA task: %s", args.uma_task)
        logger.info("UMA charge: %s", args.uma_charge)
        logger.info("UMA spin: %s", args.uma_spin)

    traj_path = outdir / "relaxed.traj"

    # Load structure
    atoms = load_structure(args.input)
    if atoms is None:
        logger.error("[error] Failed to load input structure for %s.", args.input)
        return 1

    atoms.info["fmax"] = args.fmax

    logger.info("Dtype: %s", args.dtype_str)
    logger.info("Random seed: %s", args.seed)
    set_random_seed(args.seed)

    # Setup calculator
    atoms = setup_calculator(
        atoms,
        args.model,
        args.device,
        dtype_str=args.dtype_str,
        seed=args.seed,
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
        logger.error("[error] Relaxation failed.")
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
