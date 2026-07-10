"""Calculator factory for MACE, UMA, CHGNet, and MTP."""

from pathlib import Path

from mlff_attack.calc_setup import (
    MACECalcSetup,
    UMACalcSetup,
    CHGNetCalcSetup,
    MTPCalcSetup,
)


def infer_calculator_type(model_path):
    """Infer calculator type from a complete model filename or model name."""

    model_name = Path(str(model_path)).name.lower()

    if model_name.endswith(".model"):
        return "mace"

    if model_name.endswith(".almtp"):
        return "mtp"

    if model_name.endswith(".pt") or model_name.startswith("uma-"):
        return "uma"

    if model_name in {"chgnet-0.2.0", "chgnet-0.3.0", "chgnet-r2scan"}:
        return "chgnet"

    raise ValueError(
        "--model must be a complete model filename: "
        "MACE '.model', MTP '.almtp', UMA '.pt' or 'uma-*', "
        "or CHGNet 'chgnet-0.2.0', 'chgnet-0.3.0', 'chgnet-r2scan'"
    )


def calculator(
    atoms,
    model_path,
    calculator_type=None,
    device="cpu",
    dtype_str="float64",
    seed=None,
    verbose=False,
    mace_head=None,
    uma_task="omat",
    uma_charge=None,
    uma_spin=None,
):
    """Setup any supported MLFF calculator."""

    if calculator_type is None:
        calculator_type = infer_calculator_type(model_path)

    calculator_type = calculator_type.lower()

    if calculator_type == "mace":
        setup = MACECalcSetup(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            verbose=verbose,
            mace_head=mace_head,
        )

    elif calculator_type == "uma":
        setup = UMACalcSetup(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            verbose=verbose,
            uma_task=uma_task,
            uma_charge=uma_charge,
            uma_spin=uma_spin,
        )

    elif calculator_type == "chgnet":
        setup = CHGNetCalcSetup(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            verbose=verbose,
        )

    elif calculator_type == "mtp":
        setup = MTPCalcSetup(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            verbose=verbose,
        )

    else:
        raise ValueError(
            "calculator_type must be 'mace', 'uma', 'chgnet', or 'mtp'"
        )

    return setup.setup(atoms)