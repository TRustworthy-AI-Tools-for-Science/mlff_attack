"""UMA calculator setup."""

import logging

import torch

from mlff_attack.calc_setup.calculator_class import (
    MLFFCalc,
    cast_torch_modules_dtype,
    dtype_from_string,
)

logger = logging.getLogger(__name__)


class UMACalcSetup(MLFFCalc):
    """Setup class for UMA."""

    def __init__(
        self,
        model_path,
        device="cpu",
        dtype_str="float64",
        seed=None,
        verbose=False,
        uma_task="omat",
        uma_charge=None,
        uma_spin=None,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            verbose=verbose,
        )
        self.uma_task = uma_task
        self.uma_charge = uma_charge
        self.uma_spin = uma_spin

    def setup(self, atoms):
        self.set_seed()

        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError:
            logger.error(
                "[ERROR] UMA requires fairchem-core. Install it with: "
                'pip install -e ".[uma]"'
            )
            return None

        valid_uma_tasks = {"oc20", "oc22", "oc25", "omat", "omol", "odac", "omc"}
        if self.uma_task not in valid_uma_tasks:
            logger.info(
                "[ERROR] Invalid UMA uma_task '%s'. Choose one of: %s",
                self.uma_task,
                ", ".join(sorted(valid_uma_tasks)),
            )
            return None

        model_id = str(self.model_path)

        if self.verbose:
            logger.info(
                "[INFO] Loading UMA model: %s on %s with uma_task=%s",
                model_id,
                self.device,
                self.uma_task,
            )

        if self.uma_charge is None:
            self.uma_charge = 0
        if self.uma_spin is None:
            self.uma_spin = 1

        atoms.info["charge"] = self.uma_charge
        atoms.info["spin"] = self.uma_spin

        dtype = dtype_from_string(self.dtype_str)
        previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            predictor = pretrained_mlip.get_predict_unit(
                model_id,
                device=self.device,
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)

        changed_dtype = cast_torch_modules_dtype(predictor, dtype)
        if self.verbose:
            logger.info(
                "[INFO] UMA dtype requested: %s; cast torch modules: %s",
                self.dtype_str,
                changed_dtype,
            )

        atoms.calc = FAIRChemCalculator(predictor, task_name=self.uma_task)

        self.set_seed()
        return atoms
