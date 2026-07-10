"""Calculator setup classes for supported MLFF models."""

from mlff_attack.calc_setup.chgnet import CHGNetCalcSetup
from mlff_attack.calc_setup.mace import MACECalcSetup
from mlff_attack.calc_setup.mtp import MTPCalcSetup, MTPCalculator
from mlff_attack.calc_setup.uma import UMACalcSetup

__all__ = [
    "CHGNetCalcSetup",
    "MACECalcSetup",
    "MTPCalcSetup",
    "MTPCalculator",
    "UMACalcSetup",
]
