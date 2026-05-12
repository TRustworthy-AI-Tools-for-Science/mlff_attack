# refactored TestPGD_MACE from test_attacks.py to test_pgd.py - DC
import pytest
import os
import torch
import numpy as np

from ase import build
from mace.calculators import mace_mp

from mlff_attack.grad_based.pgd import PGD_MACE
from mlff_attack.relaxation import setup_calculator
from mlff_attack.attacks import save_perturbation, load_perturbation
from pathlib import Path

def create_dummy_atoms():
    """Create a dummy ASE Atoms object for testing."""
    return build.molecule("H2O")


def dummy_model():
    import mace
    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    model.models = [m.to(dtype=torch.float32) for m in model.models]  # Ensure model tensors use float32
    assert isinstance(model, mace.calculators.mace.MACECalculator)
    return model


def test_init():
    model = dummy_model()
    attack = PGD_MACE(model, epsilon=0.1, alpha=0.01, num_iter=10)
    assert attack.model == model
    assert attack.epsilon == 0.1
    assert attack.alpha == 0.01
    assert attack.num_iter == 10


def test_make_attack():
    from mlff_attack.attacks import make_attack

    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    output_path, perturbed_atoms, attack_details = make_attack(
        model_path=model,
        device="cpu",
        atoms=atoms,
        epsilon=0.1,
        target_energy=None,
        output_cif="perturbed_structure.cif",
        attack_type="pgd",
        n_steps=1,
        clip=True,
    )

    assert Path(output_path).exists()
    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert attack_details is not None
    assert 'energies' in attack_details
    assert 'max_forces' in attack_details
    assert 'perturbations' in attack_details
    assert 'gradients' in attack_details


def test_attack_iterations():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    pgd = PGD_MACE(atoms.calc, epsilon=0.1, alpha=0.01, num_iter=5, device="cpu")
    num_iter = 5
    perturbed_atoms = pgd.attack(atoms)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(pgd.attack_history["perturbations"]) == pgd.num_iter
    assert len(pgd.attack_history["gradients"]) == pgd.num_iter


def test_epsilon_ball_bounds_displacement():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    attack = PGD_MACE(atoms.calc, epsilon=0.2, alpha=0.1, num_iter=3, device="cpu")

    perturbed_atoms = attack.attack(atoms)

    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    displacement_magnitudes = np.linalg.norm(displacement, axis=1)

    assert np.all(displacement_magnitudes <= attack.epsilon + 1e-6)