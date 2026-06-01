"""Calculator setup helpers for MACE and UMA."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def mace_calculator(
    atoms,
    model_path,
    device="cpu",
    dtype_str="float64",
    verbose=False,
    mace_head=None,
):
    try:
        import mace
        from mace.calculators import mace as mace_calculator # TODO: import at the top if venv is fixed - DC
    except ImportError:
        logger.info(
            "[ERROR] MACE requires mace-torch. Install it with: pip install -e \".[mace]\""
        )
        return None

    if isinstance(model_path, mace.calculators.mace.MACECalculator):
        if verbose:
            logger.info("[INFO] Model is already a MACECalculator")
        if mace_head is not None:
            if not hasattr(model_path, "heads") or model_path.heads is None:
                logger.error("[error] mace_head can only be used with MACE-MH models")
                return None
            if mace_head not in model_path.heads:
                logger.info(
                    "[ERROR] Invalid MACE-MH head '%s'. Choose one of: %s",
                    mace_head,
                    ", ".join(model_path.heads),
                )
                return None
            model_path.head = mace_head
        atoms.calc = model_path
    else:
        # Patch to prevent atoms and models from having different datatypes
        if dtype_str == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float64

        model_id = str(model_path)
        model_name = Path(model_id).name.lower()
        if mace_head is not None and not model_name.startswith("mace-mh"):
            logger.error("[error] mace_head can only be used with MACE-MH models")
            return None

        if verbose:
            logger.info("[INFO] Loading MACE model: %s on %s", model_path, device)

        if mace_head is None:
            atoms.calc = mace_calculator.MACECalculator(
                model_paths=model_path,
                device=device,
                default_dtype=dtype,
            )
        else:
            atoms.calc = mace_calculator.MACECalculator(
                model_paths=model_path,
                device=device,
                default_dtype=dtype,
                head=mace_head,
            )

    return atoms

def uma_calculator(
    atoms,
    model_path,
    device="cpu",
    verbose=False,
    uma_task="omat",
    uma_charge=None,
    uma_spin=None,
):
    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator # TODO: import at the top if venv is fixed - DC
    except ImportError:
        logger.error(
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
    
    model_id = str(model_path)

    if verbose:
        logger.info(
            "[INFO] Loading UMA model: %s on %s with uma_task=%s",
            model_id,
            device,
            uma_task,
        )

    if uma_charge is None:
        uma_charge = 0
    if uma_spin is None:
        uma_spin = 1
    atoms.info["charge"] = uma_charge
    atoms.info["spin"] = uma_spin

    predictor = pretrained_mlip.get_predict_unit(model_id, device=device)
    atoms.calc = FAIRChemCalculator(predictor, task_name=uma_task)
    return atoms