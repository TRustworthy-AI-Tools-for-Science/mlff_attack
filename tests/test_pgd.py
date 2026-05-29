# refactored TestPGD_ASE from test_attacks.py to test_pgd.py - DC
import pytest
import torch
import numpy as np

from ase import build

from mlff_attack.grad_based.pgd import PGD_ASE
from mlff_attack.relaxation import setup_calculator
from mlff_attack.attacks import make_attack
from pathlib import Path
import os


def create_dummy_atoms():
    """Create a dummy ASE Atoms object for testing."""
    return build.molecule("H2O")


def dummy_mace_model():
    mace = pytest.importorskip(
        "mace",
        reason="test_pgd.py MACE tests require mace-torch dependencies",
    )
    from mace.calculators import mace_mp

    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    model.models = [m.to(dtype=torch.float32) for m in model.models]
    assert isinstance(model, mace.calculators.mace.MACECalculator)
    return model


def dummy_uma_atoms():
    pytest.importorskip(
        "fairchem.core",
        reason="test_pgd.py UMA tests require fairchem-core / UMA dependencies",
    )
    atoms = setup_calculator(
        create_dummy_atoms(),
        "uma-s-1p1",
        device="cpu",
        calculator="uma",
        uma_task="omat",
        uma_charge=0,
        uma_spin=1,
    )
    assert atoms is not None
    return atoms


def test_init_mace():
    model = dummy_mace_model()
    attack = PGD_ASE(model, epsilon=0.1, alpha=0.01, num_iter=10)

    assert attack.model == model
    assert attack.epsilon == 0.1
    assert attack.alpha == 0.01
    assert attack.num_iter == 10


def test_init_uma():
    atoms = dummy_uma_atoms()
    attack = PGD_ASE(atoms.calc, epsilon=0.1, alpha=0.01, num_iter=10)

    assert attack.model == atoms.calc
    assert attack.epsilon == 0.1
    assert attack.alpha == 0.01
    assert attack.num_iter == 10


def test_make_attack_mace():
    model = dummy_mace_model()
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
        calculator="mace",
    )

    assert Path(output_path).exists()
    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert attack_details is not None
    assert 'energies' in attack_details
    assert 'max_forces' in attack_details
    assert 'perturbations' in attack_details
    assert 'gradients' in attack_details

    os.remove(output_cif)


def test_make_attack_uma():
    pytest.importorskip(
        "fairchem.core",
        reason="test_pgd.py UMA tests require fairchem-core / UMA dependencies",
    )
    atoms = create_dummy_atoms()
    output_cif = "perturbed_structure.cif"

    output_path, perturbed_atoms, attack_details = make_attack(
        atoms=atoms,
        model_path="uma-s-1p1",
        device="cpu",
        output_cif=output_cif,
        attack_type="pgd",
        epsilon=0.1,
        n_steps=2,
        target_energy=None,
        clip=True,
        calculator="uma",
        uma_task="omat",
        uma_charge=0,
        uma_spin=1,
    )

    assert Path(output_path).exists()
    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert attack_details is not None
    assert 'energies' in attack_details
    assert 'max_forces' in attack_details
    assert 'perturbations' in attack_details
    assert 'gradients' in attack_details

    os.remove(output_cif)


def test_attack_iterations_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=0.1, alpha=0.01, num_iter=5)
    num_iter = 5
    perturbed_atoms = pgd.attack(atoms)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(pgd.attack_history["perturbations"]) == pgd.num_iter
    assert len(pgd.attack_history["gradients"]) == pgd.num_iter


def test_attack_iterations_uma():
    atoms = dummy_uma_atoms()
    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=0.1, alpha=0.01, num_iter=5)
    perturbed_atoms = pgd.attack(atoms)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(pgd.attack_history["perturbations"]) == pgd.num_iter
    assert len(pgd.attack_history["gradients"]) == pgd.num_iter


def test_make_attack_defaults_clipping():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.2
    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=epsilon, alpha=0.3, num_iter=3)
    fake_gradients = lambda a: np.ones_like(a)
    atoms_positions = atoms.get_positions()
    temp_g = fake_gradients(atoms_positions)
    pgd.compute_gradient = lambda a, loss_fn=None: temp_g

    assert pgd.alpha > epsilon

    perturbed_atoms = pgd.attack(atoms)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()

    assert np.all(np.abs(displacement) <= epsilon + 1e-6)


def test_epsilon_bounds_displacement_with_L_infinity():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.2
    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=epsilon, alpha=0.3, num_iter=3)

    original_positions = atoms.get_positions().copy()
    unclipped_displacement = np.full_like(original_positions, 2 * epsilon)

    assert np.any(np.abs(unclipped_displacement) > epsilon + 1e-6)

    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(original_positions + unclipped_displacement)

    pgd._original_positions = original_positions
    pgd._clip_perturbations(perturbed_atoms)

    displacement = perturbed_atoms.get_positions() - original_positions

    assert np.all(np.abs(displacement) <= epsilon + 1e-6)


def test_target_energy_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    target_energy = atoms.get_potential_energy()

    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=0.001, alpha=0.2, num_iter=3, target_energy=target_energy)
    perturbed_atoms = pgd.attack(atoms, n_steps=3, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert pgd.target_energy == target_energy


def test_target_energy_uma():
    atoms = dummy_uma_atoms()

    target_energy = atoms.get_potential_energy()

    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=0.001, alpha=0.2, num_iter=3, target_energy=target_energy)
    perturbed_atoms = pgd.attack(atoms, n_steps=3, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert pgd.target_energy == target_energy


def test_compute_gradient_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=0.1, alpha=0.01, num_iter=3)
    gradients = pgd.compute_gradient(atoms)

    assert gradients.shape == atoms.get_positions().shape
    assert np.all(np.isfinite(gradients))


def test_compute_gradient_uma():
    atoms = dummy_uma_atoms()

    pgd = PGD_ASE(atoms.calc, device="cpu", epsilon=0.1, alpha=0.01, num_iter=3)
    gradients = pgd.compute_gradient(atoms)

    assert gradients.shape == atoms.get_positions().shape
    assert np.all(np.isfinite(gradients))


def test_random_start_is_within_epsilon():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    epsilon = 0.1

    pgd = PGD_ASE(
        atoms.calc,
        epsilon=epsilon,
        alpha=0.01,
        num_iter=3,
        device="cpu",
        rng=np.random.default_rng(0),
    )
    perturbed_atoms = pgd._random_start(atoms)

    displacement = perturbed_atoms.get_positions() - atoms.get_positions()

    assert not np.allclose(perturbed_atoms.get_positions(), atoms.get_positions(), atol=1e-6)
    assert np.all(np.abs(displacement) <= epsilon + 1e-6)


def test_attack_step():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    alpha = 0.01
    pgd = PGD_ASE(atoms.calc, epsilon=0.1, alpha=alpha, num_iter=3, device="cpu")
    fake_gradients = lambda a: np.ones_like(a)
    atoms_positions = atoms.get_positions()
    temp_g = fake_gradients(atoms_positions)
    pgd.compute_gradient = lambda a, loss_fn=None: temp_g

    perturbed_atoms = pgd.attack_step(atoms)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    expected_displacement = alpha * np.sign(temp_g)

    assert np.allclose(displacement, expected_displacement, atol=1e-6)
