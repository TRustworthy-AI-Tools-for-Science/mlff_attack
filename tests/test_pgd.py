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
    attack = PGD_MACE(model, epsilon=0.1, alpha=0.01, num_iter=10)
    assert attack.epsilon == 0.1
    assert attack.alpha == 0.01
    assert attack.num_iter == 10


def test_attack_iterations():
    model = dummy_model()
    
    attack = PGD_MACE(model, epsilon=0.1, alpha=0.01, num_iter=5)
    atoms = create_dummy_atoms()
    atoms.calc = [model]

    perturbed = attack.attack(atoms)
    assert perturbed.shape == atoms.get_positions().shape


def test_projection_bounds():
    model = dummy_model()
    
    attack = PGD_MACE(model, epsilon=0.2, alpha=0.1, num_iter=3)
    positions = torch.zeros((5, 3), requires_grad=True)
    
    perturbed = attack.attack(positions)
    perturbation = perturbed - positions
    assert torch.all(torch.abs(perturbation) <= 0.2 + 1e-6)