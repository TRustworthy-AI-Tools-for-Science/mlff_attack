"""Backend-independent tests verifying untargeted and targeted attack signs."""

import numpy as np
import pytest

from mlff_attack.grad_based.fgsm import FGSM_ASE
from mlff_attack.grad_based.pgd import PGD_ASE


class _MockAtoms:
    """Minimal Atoms stand-in for testing FGSM_ASE / PGD_ASE gradient signs."""

    def __init__(self, positions: np.ndarray, calc):
        self._positions = np.asarray(positions, dtype=float).copy()
        self.calc = calc

    def __len__(self) -> int:
        return len(self._positions)

    def copy(self) -> "_MockAtoms":
        return _MockAtoms(self._positions, self.calc)

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()

    def set_positions(self, new_positions: np.ndarray) -> None:
        self._positions = np.asarray(new_positions, dtype=float).copy()

    def get_potential_energy(self) -> float:
        return self.calc.get_energy(self._positions)

    def get_forces(self) -> np.ndarray:
        return self.calc.get_forces(self._positions)


class _HarmonicCalculator:
    """Exact harmonic potential E(q) = E0 + 0.5 * k * ||q||^2, F(q) = -k * q."""

    def __init__(self, e0: float = -10.0, k: float = 4.0):
        self.e0 = float(e0)
        self.k = float(k)

    def get_energy(self, pos: np.ndarray) -> float:
        return self.e0 + 0.5 * self.k * float(np.sum(pos**2))

    def get_forces(self, pos: np.ndarray) -> np.ndarray:
        return -self.k * np.asarray(pos, dtype=float)


class _ScalarGradVar:
    """Tiny scalar autograd node for testing _forward_pass_with_gradients logic."""

    def __init__(self, value: float, grad_wrt_pos: np.ndarray, pos_holder: "_PosHolder"):
        self._value = float(value)
        self._grad_wrt_pos = np.asarray(grad_wrt_pos, dtype=float)
        self._pos_holder = pos_holder

    def item(self) -> float:
        return self._value

    def __sub__(self, other: float) -> "_ScalarGradVar":
        return _ScalarGradVar(self._value - float(other), self._grad_wrt_pos, self._pos_holder)

    def __pow__(self, power: float) -> "_ScalarGradVar":
        assert power == 2
        chain = 2.0 * self._value * self._grad_wrt_pos
        return _ScalarGradVar(self._value**2, chain, self._pos_holder)

    def __neg__(self) -> "_ScalarGradVar":
        return _ScalarGradVar(-self._value, -self._grad_wrt_pos, self._pos_holder)

    def backward(self) -> None:
        self._pos_holder.grad = _DetachableArray(self._grad_wrt_pos)


class _DetachableArray:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr, dtype=float)

    def detach(self) -> "_DetachableArray":
        return self

    def cpu(self) -> "_DetachableArray":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr.copy()


class _PosHolder:
    def __init__(self):
        self.grad = None


def _patch_autograd_backend(attack_obj, calc: _HarmonicCalculator) -> None:
    """Force attack_obj to take the MACE-autograd codepath using exact chain rule."""
    attack_obj._uses_mace_autograd = lambda _atoms: True

    def _fake_forward(atoms):
        pos = atoms.get_positions()
        e_val = calc.get_energy(pos)
        f_val = calc.get_forces(pos)
        grad_e = -f_val
        holder = _PosHolder()
        return _ScalarGradVar(e_val, grad_e, holder), f_val, holder

    attack_obj._forward_pass_with_gradients = _fake_forward


@pytest.mark.parametrize("attack_cls", [FGSM_ASE, PGD_ASE])
def test_untargeted_and_targeted_signs_match_across_backends(attack_cls):
    calc = _HarmonicCalculator(e0=-10.0, k=4.0)
    q0 = np.array([[0.3, -0.2, 0.1], [-0.4, 0.25, -0.15]])
    atoms = _MockAtoms(q0, calc)
    e_init = atoms.get_potential_energy()

    def _make(target_energy):
        if attack_cls is FGSM_ASE:
            return FGSM_ASE(calc, epsilon=0.02, target_energy=target_energy)
        return PGD_ASE(
            calc,
            epsilon=0.02,
            alpha=0.01,
            num_iter=2,
            target_energy=target_energy,
            random_start=False,
        )

    # 1. Untargeted (maximize energy): both backends must match & increase E
    ase_untargeted = _make(target_energy=None)
    mace_untargeted = _make(target_energy=None)
    _patch_autograd_backend(mace_untargeted, calc)

    g_ase = ase_untargeted.compute_gradient(atoms)
    g_mace = mace_untargeted.compute_gradient(atoms)
    assert np.allclose(g_ase, g_mace)

    pert_ase = ase_untargeted.attack(atoms, n_steps=1, clip=True)
    pert_mace = mace_untargeted.attack(atoms, n_steps=1, clip=True)
    assert np.allclose(pert_ase.get_positions(), pert_mace.get_positions())
    assert pert_mace.get_potential_energy() > e_init

    # 2. Targeted ABOVE initial energy AND BELOW initial energy:
    # both backends must produce identical gradients and reduce |E - E_target|!
    for target_e in (e_init + 0.5, e_init - 0.3):
        ase_targeted = _make(target_energy=target_e)
        mace_targeted = _make(target_energy=target_e)
        _patch_autograd_backend(mace_targeted, calc)

        gt_ase = ase_targeted.compute_gradient(atoms)
        gt_mace = mace_targeted.compute_gradient(atoms)
        assert np.allclose(gt_ase, gt_mace)

        out_ase = ase_targeted.attack(atoms, n_steps=1, clip=True)
        out_mace = mace_targeted.attack(atoms, n_steps=1, clip=True)
        assert np.allclose(out_ase.get_positions(), out_mace.get_positions())
        assert abs(out_mace.get_potential_energy() - target_e) < abs(e_init - target_e)
