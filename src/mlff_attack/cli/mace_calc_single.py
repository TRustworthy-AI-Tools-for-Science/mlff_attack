#!/usr/bin/env python3
"""
CLI entry point for MACE single structure relaxation.
"""

import argparse
from pathlib import Path
from mlff_attack.relaxation import (
    load_structure,
    setup_calculator,
    run_relaxation,
    save_results
)


def main():
    """Main entry point for MACE single structure relaxation."""
    parser = argparse.ArgumentParser(description="Relax a single CIF with MACE.")
    parser.add_argument("--input", required=True, help="Input CIF file")
    parser.add_argument("--model", required=True, help="Path to MACE model file (.model)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence criterion (eV/Å)")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum relaxation steps")
    parser.add_argument("--optimizer", default="LBFGS", choices=["BFGS", "LBFGS"], help="ASE optimizer")
    args = parser.parse_args()

    # Setup output paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    traj_path = outdir / "relaxed.traj"

    # Load structure
    atoms = load_structure(args.input)
    if atoms is None:
        return 1

    # Setup calculator
    atoms = setup_calculator(atoms, args.model, args.device)
    if atoms is None:
        return 1

    # Run relaxation
    success = run_relaxation(
        atoms=atoms,
        traj_path=traj_path,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer
    )
    if not success:
        return 1

    # Save results
    cif_path = save_results(atoms, outdir)
    if cif_path is None:
        return 1

    print(f"[DONE] Relaxation complete. Trajectory → {traj_path}, CIF → {cif_path}")
    return 0


if __name__ == "__main__":
    exit(main())
