"""CHGNet calculator setup."""

import logging
from pathlib import Path

from mlff_attack.calc_setup.calculator_class import MLFFCalc, dtype_from_string

logger = logging.getLogger(__name__)


class CHGNetCalcSetup(MLFFCalc):
    """Setup class for CHGNet."""

    def setup(self, atoms):
        self.set_seed()

        try:
            from chgnet.model.dynamics import CHGNetCalculator
            from chgnet.model.model import CHGNet
        except ImportError:
            logger.error(
                "[ERROR] CHGNet requires chgnet. "
                'Install it with: pip install -e ".[chgnet]"'
            )
            return None

        dtype = dtype_from_string(self.dtype_str)

        if isinstance(self.model_path, CHGNetCalculator):
            if self.verbose:
                logger.info("[INFO] Reusing existing CHGNetCalculator")
            model = self.model_path.model
        else:
            model_id = str(self.model_path)
            model_file = Path(model_id)

            if self.verbose:
                logger.info(
                    "[INFO] Loading CHGNet model: %s on %s with dtype %s",
                    model_id,
                    self.device,
                    self.dtype_str,
                )

            if model_file.is_file():
                model = CHGNet.from_file(str(model_file))
            else:
                model_name = Path(model_id).name.lower()

                if model_name.startswith("chgnet-"):
                    model_name = model_name.removeprefix("chgnet-")

                valid_models = {"0.2.0", "0.3.0", "r2scan"}
                if model_name not in valid_models:
                    logger.error(
                        "[ERROR] Invalid CHGNet model '%s'. Choose one of: %s",
                        model_name,
                        ", ".join(sorted(valid_models)),
                    )
                    return None

                model = CHGNet.load(
                    model_name=model_name,
                    use_device=self.device,
                    verbose=self.verbose,
                )

        model.to(device=self.device, dtype=dtype)

        atom_reference = getattr(model, "composition_model", None)
        atom_reference_layer = getattr(atom_reference, "fc", None)

        if atom_reference_layer is not None:
            old_hook = getattr(atom_reference_layer, "_mlff_dtype_hook", None)
            if old_hook is not None:
                old_hook.remove()

            def match_atom_reference_dtype(layer, inputs):
                composition_features = inputs[0]
                composition_features = composition_features.to(layer.weight)
                return (composition_features,)

            new_hook = atom_reference_layer.register_forward_pre_hook(
                match_atom_reference_dtype
            )
            atom_reference_layer._mlff_dtype_hook = new_hook

        model.graph_converter.forward.__func__.__globals__["TORCH_DTYPE"] = dtype
        model.forward.__func__.__globals__["TORCH_DTYPE"] = dtype

        atoms.calc = CHGNetCalculator(
            model=model,
            use_device=self.device,
        )

        self.set_seed()
        return atoms
