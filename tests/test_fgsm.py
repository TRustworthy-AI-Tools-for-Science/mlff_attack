# refactored TestFGSM_MACE from test_attacks.py to test_fgsm.py - DC
import pytest
import os
import torch
import numpy as np
import mlff_attack.grad_based.fgsm as fgsm_module

from ase import build
from mace.calculators import mace_mp

from mlff_attack.grad_based.fgsm import FGSM_MACE
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


def test_make_attack():
    from mlff_attack.attacks import make_attack

    atoms = create_dummy_atoms()
    model = dummy_model()
    atoms = setup_calculator(atoms, model, device="cpu", dtype_str="float32")

    output_path, perturbed_atoms, attack_details = make_attack(
        model_path=model,
        device="cpu",
        atoms=atoms,
        epsilon=0.1,
        target_energy=None,
        output_cif="perturbed_structure.cif",
        attack_type="fgsm",
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


def test_save_load_perturbation():
    cwd = os.path.dirname(os.path.realpath(__file__))

    atoms = create_dummy_atoms()
    atoms_perturbed = create_dummy_atoms()
    epsilon = 0.1
    energy_original = 0.0
    energy_perturbed = 1.0
    gradients = np.random.rand(len(atoms), 3)
    save_path = os.path.join(cwd, "test_perturbation")
    metadata = {'test_key': 0.0}

    save_path = save_perturbation(
        atoms,
        atoms_perturbed,
        epsilon,
        energy_original,
        energy_perturbed,
        gradients,
        save_path,
        metadata
    )

    save_file = os.path.join(cwd, "test_perturbation.npz")
    assert Path(save_file).exists()
    with np.load(save_file, allow_pickle=True) as data: # fixed permission error - DC
        assert 'positions_original' in data
        assert 'positions_perturbed' in data
        assert 'epsilon' in data
        assert 'energy_original' in data
        assert 'energy_perturbed' in data
        assert 'gradients' in data
        assert 'meta_test_key' in data   

    loaded_data = load_perturbation(save_file)
    assert loaded_data['atoms_original'] == atoms
    assert loaded_data['atoms_perturbed'] == atoms_perturbed
    assert loaded_data['epsilon'] == epsilon
    assert loaded_data['energy_original'] == energy_original
    assert loaded_data['energy_perturbed'] == energy_perturbed
    assert np.allclose(loaded_data['gradients'], gradients)
    assert loaded_data['metadata']['test_key'] == metadata['test_key']

    # Clean up
    os.remove(save_file)


def test_init():
    model = dummy_model()
    attack = FGSM_MACE(model, epsilon=0.1)
    assert attack.epsilon == 0.1
    assert attack.model == model


def test_attack_basic():
    atoms = setup_calculator(create_dummy_atoms(), dummy_model(), device="cpu", dtype_str="float32")

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.1, device="cpu")
    perturbed_atoms = fgsm.attack(atoms, n_steps=1, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert not np.array_equal(perturbed_atoms.get_positions(), atoms.get_positions())


def test_epsilon_scaling():
    model = dummy_model()
    
    fgsm = FGSM_MACE(model, epsilon=0.5)
    atoms = create_dummy_atoms()
    atoms = setup_calculator(atoms, model, device="cpu")

    perturbed_atoms = fgsm.attack(atoms)
    perturbation = perturbed_atoms.get_positions() - atoms.get_positions()
    assert np.all(np.abs(perturbation) <= 0.5 + 1e-6)


def test_attack_step_scales_epsilon_by_n_steps():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.1
    n_steps = 5
    fgsm = FGSM_MACE(atoms.calc, epsilon=0.1, device="cpu")

    perturbed_atoms = fgsm.attack_step(atoms, step=0, n_steps=n_steps)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    step_size = epsilon / n_steps

    assert np.max(np.abs(displacement)) <= step_size + 1e-6


def test_n_steps():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    fgsm = FGSM_MACE(atoms.calc, epsilon=0.5, device="cpu")
    n_steps = 3
    perturbed_atoms = fgsm.attack(atoms, n_steps, clip=False)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(fgsm.attack_history["perturbations"]) == n_steps
    assert len(fgsm.attack_history["gradients"]) == n_steps
    

def test_displacements_are_clipped_to_epsilon():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.01, device="cpu")
    n_steps = 5

    perturbed_atoms = fgsm.attack(atoms, n_steps, clip=True)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    displacement_magnitudes = np.linalg.norm(displacement, axis=1)

    assert np.all(displacement_magnitudes <= fgsm.epsilon + 1e-6)


def test_displacements_are_not_clipped_to_epsilon():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.01, device="cpu")
    n_steps = 5

    perturbed_atoms = fgsm.attack(atoms, n_steps, clip=False)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    displacement_magnitudes = np.linalg.norm(displacement, axis=1)

    assert np.max(displacement_magnitudes) > fgsm.epsilon + 1e-6


def test_target_energy():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    target_energy = atoms.get_potential_energy()

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.001, device="cpu", target_energy=target_energy)
    perturbed_atoms = fgsm.attack(atoms, n_steps=3, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert fgsm.target_energy == target_energy


def test_forward_pass():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.1, device="cpu")
    energy, forces, positions = fgsm._forward_pass_with_gradients(atoms)

    assert energy is not None
    assert forces.shape == atoms.get_positions().shape
    assert positions.shape == atoms.get_positions().shape


def test_compute_gradient():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.1, device="cpu")
    gradients = fgsm.compute_gradient(atoms)

    assert gradients.shape == atoms.get_positions().shape
    assert np.all(np.isfinite(gradients))


def test_compute_gradient_uses_loss_function():
    model = dummy_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_MACE(atoms.calc, epsilon=0.1, device="cpu")

    default_gradients = fgsm.compute_gradient(atoms)
    custom_gradients = fgsm.compute_gradient(atoms, loss_fn=lambda energy: energy)

    assert np.allclose(custom_gradients, -default_gradients, atol=1e-5)