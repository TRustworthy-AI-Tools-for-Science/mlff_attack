"""MACE calculator setup."""

import logging
from pathlib import Path

from mlff_attack.calc_setup.calculator_class import MLFFCalc, dtype_from_string

logger = logging.getLogger(__name__)


class MACECalcSetup(MLFFCalc):
    """Setup class for MACE."""

    def __init__(
        self,
        model_path,
        device="cpu",
        dtype_str="float64",
        seed=None,
        verbose=False,
        mace_head=None,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            verbose=verbose,
        )
        self.mace_head = mace_head

    def setup(self, atoms):
        self.set_seed()

        try:
            import mace
            from mace.calculators import mace as mace_module
        except ImportError:
            logger.info(
                "[ERROR] MACE requires mace-torch. Install it with: "
                'pip install -e ".[mace]"'
            )
            return None

        if isinstance(self.model_path, mace.calculators.mace.MACECalculator):
            if self.verbose:
                logger.info("[INFO] Model is already a MACECalculator")

            if self.mace_head is not None:
                if not hasattr(self.model_path, "heads") or self.model_path.heads is None:
                    logger.error("[error] mace_head can only be used with MACE-MH models")
                    return None

                if self.mace_head not in self.model_path.heads:
                    logger.info(
                        "[ERROR] Invalid MACE-MH head '%s'. Choose one of: %s",
                        self.mace_head,
                        ", ".join(self.model_path.heads),
                    )
                    return None

                self.model_path.head = self.mace_head

            atoms.calc = self.model_path

        else:
            dtype = dtype_from_string(self.dtype_str)

            model_id = str(self.model_path)
            model_name = Path(model_id).name.lower()
            if self.mace_head is not None and not model_name.startswith("mace-mh"):
                logger.error("[error] mace_head can only be used with MACE-MH models")
                return None

            if self.verbose:
                logger.info(
                    "[INFO] Loading MACE model: %s on %s",
                    self.model_path,
                    self.device,
                )

            if self.mace_head is None:
                atoms.calc = mace_module.MACECalculator(
                    model_paths=self.model_path,
                    device=self.device,
                    default_dtype=dtype,
                )
            else:
                atoms.calc = mace_module.MACECalculator(
                    model_paths=self.model_path,
                    device=self.device,
                    default_dtype=dtype,
                    head=self.mace_head,
                )

        self.set_seed()
        return atoms
