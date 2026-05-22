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
from mlff_attack.attacks import make_attack
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
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    output_cif = "perturbed_structure.cif"

    output_path, perturbed_atoms, attack_details = make_attack(
        atoms=atoms,
        model_path=model,
        device="cpu",
        output_cif=output_cif,
        attack_type="pgd",
        epsilon=0.1,
        n_steps=2,
        target_energy=None,
        clip=True,
    )

    assert Path(output_path).exists()
    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert attack_details is not None
    assert 'energies' in attack_details
    assert 'max_forces' in attack_details
    assert 'perturbations' in attack_details
    assert 'gradients' in attack_details

    os.remove(output_cif)


def test_attack_iterations():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    pgd = PGD_MACE(atoms.calc, device="cpu", epsilon=0.1, alpha=0.01, num_iter=5)
    num_iter = 5
    perturbed_atoms = pgd.attack(atoms)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(pgd.attack_history["perturbations"]) == pgd.num_iter
    assert len(pgd.attack_history["gradients"]) == pgd.num_iter


def test_make_attack_defaults_clipping():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.2
    pgd = PGD_MACE(atoms.calc, device="cpu", epsilon=epsilon, alpha=0.3, num_iter=3)
    fake_gradients = lambda a: np.ones_like(a)
    atoms_positions = atoms.get_positions()
    temp_g = fake_gradients(atoms_positions)
    pgd.compute_gradient = lambda a: temp_g

    assert pgd.alpha > epsilon

    perturbed_atoms = pgd.attack(atoms)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()

    assert np.all(np.abs(displacement) <= epsilon + 1e-6)


def test_epsilon_bounds_displacement_with_L_infinity():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.2
    pgd = PGD_MACE(atoms.calc, device="cpu", epsilon=epsilon, alpha=0.3, num_iter=3)

    original_positions = atoms.get_positions().copy()
    unclipped_displacement = np.full_like(original_positions, 2 * epsilon)

    assert np.any(np.abs(unclipped_displacement) > epsilon + 1e-6)

    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(original_positions + unclipped_displacement)

    pgd._original_positions = original_positions
    pgd._clip_perturbations(perturbed_atoms)

    displacement = perturbed_atoms.get_positions() - original_positions

    assert np.all(np.abs(displacement) <= epsilon + 1e-6)

def test_target_energy():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    target_energy = atoms.get_potential_energy()

    pgd = PGD_MACE(atoms.calc, device="cpu", epsilon=0.001, alpha=0.2, num_iter=3, target_energy=target_energy)
    perturbed_atoms = pgd.attack(atoms, n_steps=3, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert pgd.target_energy == target_energy


def test_compute_gradient():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    pgd = PGD_MACE(atoms.calc, device="cpu", epsilon=0.1, alpha=0.01, num_iter=3)
    gradients = pgd.compute_gradient(atoms)

    assert gradients.shape == atoms.get_positions().shape
    assert np.all(np.isfinite(gradients))

def test_random_start_is_within_epsilon():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.1
    pgd = PGD_MACE(atoms.calc, epsilon=epsilon, alpha=0.01, num_iter=3, device="cpu")

    perturbed_atoms = pgd._random_start(atoms)

    displacement = perturbed_atoms.get_positions() - atoms.get_positions()

    assert not np.allclose(perturbed_atoms.get_positions(), atoms.get_positions(), atol=1e-6)
    assert np.all(np.abs(displacement) <= epsilon + 1e-6)

def test_attack_step():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    alpha = 0.01
    pgd = PGD_MACE(atoms.calc, epsilon=0.1, alpha=alpha, num_iter=3, device="cpu")
    fake_gradients = lambda a: np.ones_like(a)
    atoms_positions = atoms.get_positions()
    temp_g = fake_gradients(atoms_positions)
    pgd.compute_gradient = lambda a: temp_g

    perturbed_atoms = pgd.attack_step(atoms)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    expected_displacement = alpha * np.sign(temp_g)

    assert np.allclose(displacement, expected_displacement, atol=1e-6)