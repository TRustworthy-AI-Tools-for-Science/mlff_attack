#!/usr/bin/env python3
"""
MACE and UMA relaxation functionality.
"""

import logging
from pathlib import Path
import torch

from ase.io import read, write
from ase.optimize import BFGS, LBFGS

logger = logging.getLogger(__name__)


def load_structure(input_path):
    """Load structure from input file.

    Parameters
    ----------
    input_path : str or Path
        Path to input structure file (CIF, POSCAR, etc.)

    Returns
    -------
    ase.Atoms or None
        ASE Atoms object, or None if loading fails
    """
    try:
        atoms = read(input_path)
        logger.info("[INFO] Loaded structure: %s", input_path)
        logger.info("[INFO] Number of atoms: %s", len(atoms))
        logger.info("[INFO] Chemical formula: %s", atoms.get_chemical_formula())
        return atoms
    except (OSError, ValueError, RuntimeError) as exc:
        logger.info("[ERROR] Failed to load structure from %s: %s", input_path, exc)
        return None


def setup_calculator(
    atoms,
    model_path,
    device="cpu",
    dtype_str="float64",
    verbose=False,
    calculator=None,
    uma_task="omat",
    uma_charge=None,
    uma_spin=None,
):
    """Initialize and attach MACE or UMA calculator to atoms object.

    This supports two model types:

    - MACE models: pass a local model path, like "mace-mpa-0-medium.model"
    - UMA models: pass a UMA model name, like "uma-s-1p1"

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object
    model_path : str or Path or MACECalculator
        Path to MACE model file or existing MACECalculator instance, or a UMA model name
    device : str, optional
        Device to use (cuda or cpu), by default "cpu"
    dtype_str : str, optional
        Data type for MACE calculations ("float32" or "float64"), by default "float64"
    verbose : bool, optional
        Whether to print detailed information, by default False

    Returns
    -------
    ase.Atoms or None
        ASE Atoms object with calculator attached, or None if setup fails
    """
    try:
        model_id = str(model_path)
        if calculator not in {None, "mace", "uma"}:
            logger.info(
                "[ERROR] Invalid calculator '%s'. Use 'mace' or 'uma'.",
                calculator,
            )
            return None

        use_uma = calculator == "uma" or (
            calculator is None and model_id.startswith("uma-")
        )

        # if use UMA
        if use_uma:
            try:
                from fairchem.core import pretrained_mlip, FAIRChemCalculator
            except ImportError:
                logger.info(
                    "[ERROR] UMA requires fairchem-core. Install it with: pip install -e \".[uma]\""
                )
                return None

            valid_uma_tasks = {"oc20", "oc22", "oc25", "omat", "omol", "odac", "omc"}
            if uma_task not in valid_uma_tasks:
                logger.info(
                    "[ERROR] Invalid UMA uma_task '%s'. Choose one of: %s",
                    uma_task,
                    ", ".join(sorted(valid_uma_tasks)),
                )
                return None

            if uma_task == "omol":
                atoms.info["charge"] = uma_charge if uma_charge is not None else 0
                atoms.info["spin"] = uma_spin if uma_spin is not None else 1

            if verbose:
                logger.info(
                    "[INFO] Loading UMA model: %s on %s with uma_task=%s",
                    model_id,
                    device,
                    uma_task,
                )

            predictor = pretrained_mlip.get_predict_unit(model_id, device=device)
            atoms.calc = FAIRChemCalculator(predictor, task_name=uma_task)
            return atoms

        # if use MACE
        try:
            import mace
            from mace.calculators import mace as mace_calculator
        except ImportError:
            logger.info(
                "[ERROR] MACE requires mace-torch. Install it with: pip install -e \".[mace]\""
            )
            return None

        if isinstance(model_path, mace.calculators.mace.MACECalculator):
            if verbose:
                logger.info("[INFO] Model is already a MACECalculator")
            atoms.calc = model_path
        else:
            # Patch to prevent atoms and models from having different datatypes
            if dtype_str == "float32":
                dtype = torch.float32
            else:
                dtype = torch.float64

            if verbose:
                logger.info("[INFO] Loading MACE model: %s on %s", model_path, device)
            atoms.calc = mace_calculator.MACECalculator(
                model_paths=model_path,
                device=device,
                default_dtype=dtype
            )
        return atoms

    except (OSError, ValueError, RuntimeError) as exc:
        logger.info("[ERROR] Failed to setup calculator: %s", exc)
        return None


def get_optimizer_class(optimizer_name):
    """Get the ASE optimizer class from name.

    Parameters
    ----------
    optimizer_name : str
        Name of optimizer ("BFGS" or "LBFGS")

    Returns
    -------
    type
        ASE Optimizer class
    """
    optimizers = {"BFGS": BFGS, "LBFGS": LBFGS}
    return optimizers.get(optimizer_name, LBFGS)


def run_relaxation(
    atoms,
    traj_path,
    *,
    fmax=0.01,
    max_steps=300,
    optimizer="LBFGS",
    verbose=True,
    checkpoint_interval=None,
    checkpoint_dir=None,
):
    """Run structural relaxation.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object with calculator attached
    traj_path : str or Path
        Path to save trajectory file
    fmax : float, optional
        Force convergence criterion (eV/Å), by default 0.01
    max_steps : int, optional
        Maximum number of optimization steps, by default 300
    optimizer : str, optional
        Name of optimizer to use ("BFGS" or "LBFGS"), by default "LBFGS"
    verbose : bool, optional
        Whether to print detailed information, by default True
    checkpoint_interval : int, optional
        Save checkpoint every N steps. If None, no checkpoints are saved.
    checkpoint_dir : str or Path, optional
        Directory to save checkpoints. If None, uses same directory as traj_path.

    Returns
    -------
    bool
        True if relaxation completed successfully, False otherwise
    """
    try:
        logger.info("[INFO] Starting relaxation with %s optimizer", optimizer)
        logger.info("[INFO] Convergence criterion: fmax = %s eV/Å", fmax)
        logger.info("[INFO] Maximum steps: %s", max_steps)

        # Setup checkpoint directory
        if checkpoint_interval is not None:
            if checkpoint_dir is None:
                checkpoint_dir = (
                    Path(traj_path).parent / f"{Path(traj_path).stem}_ckpts"
                )
            else:
                checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                logger.info(
                    "[INFO] Checkpoints will be saved every %s steps to %s",
                    checkpoint_interval,
                    checkpoint_dir,
                )

        opt_cls = get_optimizer_class(optimizer)
        opt = opt_cls(atoms, trajectory=str(traj_path), logfile=None)

        # Run relaxation with checkpointing
        if checkpoint_interval is not None:
            for step in range(0, max_steps, checkpoint_interval):
                steps_to_run = min(checkpoint_interval, max_steps - step)
                opt.run(fmax=fmax, steps=steps_to_run)

                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{opt.nsteps}.cif"
                write(checkpoint_path, atoms)
                if verbose:
                    forces = atoms.get_forces()
                    max_force = max(sum(force**2)**0.5 for force in forces)
                    logger.info(
                        "[INFO] Checkpoint saved at step %s, max force: %.6f eV/Å",
                        opt.nsteps,
                        max_force,
                    )

                # Check if converged
                final_forces = atoms.get_forces()
                max_force = max(sum(force**2)**0.5 for force in final_forces)
                if max_force < fmax:
                    break
        else:
            opt.run(fmax=fmax, steps=max_steps)

        # Get final forces
        final_forces = atoms.get_forces()
        max_force = max(sum(force**2)**0.5 for force in final_forces)

        converged = max_force < fmax
        status = "CONVERGED" if converged else "NOT CONVERGED"
        if verbose:
            logger.info("[INFO] Relaxation %s after %s steps", status, opt.nsteps)
            logger.info("[INFO] Final maximum force: %.6f eV/Å", max_force)

        return True
    except (OSError, ValueError, RuntimeError) as exc:
        logger.info("[ERROR] Relaxation failed: %s", exc)
        return False


def save_results(atoms, output_dir, base_name="relaxed"):
    """Save relaxed structure to output files.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object to save
    output_dir : str or Path
        Output directory path
    base_name : str, optional
        Base name for output files, by default "relaxed"

    Returns
    -------
    Path or None
        Path to saved CIF file, or None if saving fails
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cif_path = output_dir / f"{base_name}.cif"
        write(cif_path, atoms)
        logger.info("[INFO] Saved relaxed structure to: %s", cif_path)

        return cif_path
    except (OSError, ValueError, RuntimeError) as exc:
        logger.info("[ERROR] Failed to save results: %s", exc)
        return None
