#!/usr/bin/env python3
"""
CLI entry point for trajectory visualization.
"""

import argparse
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from mlff_attack.visualization import (
    load_trajectory,
    create_visualization
)


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "visualize_traj.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    """Main entry point for trajectory visualization."""
    parser = argparse.ArgumentParser(description="Visualize MACE relaxation trajectory.")
    parser.add_argument("--traj", required=True, help="Path to trajectory file (.traj)")
    parser.add_argument("--outdir", default=".", help="Output directory for plots (default: current directory)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format for plots")
    args = parser.parse_args()

    # Load trajectory
    traj = load_trajectory(args.traj)
    if traj is None:
        logger.error(f"Error: Could not load trajectory from {args.traj}")
        return 1

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    output_file = create_visualization(
        traj=traj,
        traj_path=Path(args.traj),
        outdir=outdir,
        output_format=args.format,
        show=args.show
    )
    
    if output_file:
        logger.info(f"Visualization saved to {output_file}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
