# refactored TestFGSM_ASE from test_attacks.py to test_fgsm.py - DC
import pytest
import os
import torch
import numpy as np
import mlff_attack.grad_based.fgsm as fgsm_module

from ase import build

from mlff_attack.grad_based.fgsm import FGSM_ASE
from mlff_attack.attacks import make_attack
from mlff_attack.relaxation import setup_calculator
from mlff_attack.attacks import save_perturbation, load_perturbation
from pathlib import Path


def create_dummy_atoms():
    """Create a dummy ASE Atoms object for testing."""
    return build.molecule("H2O")


def dummy_mace_model():
    mace = pytest.importorskip(
        "mace",
        reason="test_fgsm.py MACE tests require mace-torch dependencies",
    )
    from mace.calculators import mace_mp

    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    model.models = [m.to(dtype=torch.float32) for m in model.models]
    assert isinstance(model, mace.calculators.mace.MACECalculator)
    return model


def dummy_uma_atoms():
    pytest.importorskip(
        "fairchem.core",
        reason="test_fgsm.py UMA tests require fairchem-core / UMA dependencies",
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
    attack = FGSM_ASE(model, epsilon=0.1)
    assert attack.model == model
    assert attack.epsilon == 0.1


def test_init_uma():
    atoms = dummy_uma_atoms()
    attack = FGSM_ASE(atoms.calc, epsilon=0.1)
    assert attack.model == atoms.calc
    assert attack.epsilon == 0.1


def test_make_attack_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")
    output_cif = "perturbed_structure.cif"

    output_path, perturbed_atoms, attack_details = make_attack(
        atoms=atoms,
        model_path=model,
        device="cpu",
        output_cif=output_cif,
        attack_type="fgsm",
        epsilon=0.1,
        n_steps=1,
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


def test_make_attack_mace_mh():
    model = dummy_mace_model()
    model.heads = ["omat_pbe", "omol"]
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32", calculator="mace", mace_head="omat_pbe")
    output_cif = "perturbed_structure.cif"

    output_path, perturbed_atoms, attack_details = make_attack(
        atoms=atoms,
        model_path=model,
        device="cpu",
        output_cif=output_cif,
        attack_type="fgsm",
        epsilon=0.1,
        n_steps=1,
        target_energy=None,
        clip=True,
        calculator="mace",
        mace_head="omat_pbe",
    )

    assert Path(output_path).exists()
    assert atoms.calc.head == "omat_pbe"
    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert attack_details is not None
    assert 'energies' in attack_details
    assert 'max_forces' in attack_details
    assert 'perturbations' in attack_details
    assert 'gradients' in attack_details

    os.remove(output_cif)


def test_make_attack_uma():
    from mlff_attack.attacks import make_attack

    pytest.importorskip(
        "fairchem.core",
        reason="test_fgsm.py UMA tests require fairchem-core / UMA dependencies",
    )
    atoms = create_dummy_atoms()
    output_cif = "perturbed_structure.cif"

    output_path, perturbed_atoms, attack_details = make_attack(
        atoms=atoms,
        model_path="uma-s-1p1",
        device="cpu",
        output_cif=output_cif,
        attack_type="fgsm",
        epsilon=0.1,
        n_steps=1,
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
    with np.load(save_file, allow_pickle=True) as data:
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

    os.remove(save_file)


def test_attack_basic():
    atoms = setup_calculator(create_dummy_atoms(), dummy_mace_model(), device="cpu", dtype_str="float32")

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")
    perturbed_atoms = fgsm.attack(atoms, n_steps=1, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert not np.array_equal(perturbed_atoms.get_positions(), atoms.get_positions())


def test_attack_basic_uma():
    atoms = dummy_uma_atoms()

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")
    perturbed_atoms = fgsm.attack(atoms, n_steps=1, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert not np.array_equal(perturbed_atoms.get_positions(), atoms.get_positions())


def test_epsilon_scaling_mace():
    model = dummy_mace_model()

    fgsm = FGSM_ASE(model, epsilon=0.5)
    atoms = create_dummy_atoms()
    atoms = setup_calculator(atoms, model, device="cpu")

    perturbed_atoms = fgsm.attack(atoms)
    perturbation = perturbed_atoms.get_positions() - atoms.get_positions()
    assert np.all(np.abs(perturbation) <= 0.5 + 1e-6)


def test_epsilon_scaling_uma():
    atoms = dummy_uma_atoms()

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.5, device="cpu")
    perturbed_atoms = fgsm.attack(atoms)
    perturbation = perturbed_atoms.get_positions() - atoms.get_positions()
    assert np.all(np.abs(perturbation) <= 0.5 + 1e-6)


def test_attack_step_scales_epsilon_by_n_steps_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.1
    n_steps = 5
    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")

    perturbed_atoms = fgsm.attack_step(atoms, step=0, n_steps=n_steps)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    step_size = epsilon / n_steps

    assert np.all(np.abs(displacement) <= step_size + 1e-6)


def test_attack_step_scales_epsilon_by_n_steps_uma():
    atoms = dummy_uma_atoms()

    epsilon = 0.1
    n_steps = 5
    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")

    perturbed_atoms = fgsm.attack_step(atoms, step=0, n_steps=n_steps)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    step_size = epsilon / n_steps

    assert np.all(np.abs(displacement) <= step_size + 1e-6)


def test_n_steps_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.5, device="cpu")
    n_steps = 3
    perturbed_atoms = fgsm.attack(atoms, n_steps=3)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(fgsm.attack_history["perturbations"]) == n_steps
    assert len(fgsm.attack_history["gradients"]) == n_steps


def test_n_steps_uma():
    atoms = dummy_uma_atoms()

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.5, device="cpu")
    n_steps = 3
    perturbed_atoms = fgsm.attack(atoms, n_steps=3)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert len(fgsm.attack_history["perturbations"]) == n_steps
    assert len(fgsm.attack_history["gradients"]) == n_steps


def test_displacements_are_clipped_to_epsilon():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.01
    fgsm = FGSM_ASE(atoms.calc, epsilon=epsilon, device="cpu")
    unclipped_displacement = np.full_like(atoms.get_positions(), 2 * epsilon)

    assert np.any(np.abs(unclipped_displacement) > epsilon + 1e-6)

    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(atoms.get_positions() + unclipped_displacement)

    fgsm._original_positions = atoms.get_positions()
    fgsm._clip_perturbations(perturbed_atoms)
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()

    assert np.all(np.abs(displacement) <= epsilon + 1e-6)


def test_displacements_are_not_clipped_to_epsilon():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    epsilon = 0.01
    fgsm = FGSM_ASE(atoms.calc, device="cpu", epsilon=epsilon)
    unclipped_displacement = np.full_like(atoms.get_positions(), 2 * epsilon)

    assert np.any(np.abs(unclipped_displacement) > epsilon + 1e-6)

    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(atoms.get_positions() + unclipped_displacement)

    fgsm._original_positions = atoms.get_positions()
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()

    assert np.all(np.abs(displacement) > epsilon + 1e-6)


def test_target_energy_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    target_energy = atoms.get_potential_energy()

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.001, device="cpu", target_energy=target_energy)
    perturbed_atoms = fgsm.attack(atoms, n_steps=3, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert fgsm.target_energy == target_energy


def test_target_energy_uma():
    atoms = dummy_uma_atoms()

    target_energy = atoms.get_potential_energy()

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.001, device="cpu", target_energy=target_energy)
    perturbed_atoms = fgsm.attack(atoms, n_steps=3, clip=True)

    assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
    assert fgsm.target_energy == target_energy


def test_forward_pass():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")
    energy, forces, positions = fgsm._forward_pass_with_gradients(atoms)

    assert energy is not None
    assert forces.shape == atoms.get_positions().shape
    assert positions.shape == atoms.get_positions().shape


def test_compute_gradient_mace():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")
    gradients = fgsm.compute_gradient(atoms)

    assert gradients.shape == atoms.get_positions().shape
    assert np.all(np.isfinite(gradients))


def test_compute_gradient_uma():
    atoms = dummy_uma_atoms()

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")
    gradients = fgsm.compute_gradient(atoms)

    assert gradients.shape == atoms.get_positions().shape
    assert np.all(np.isfinite(gradients))


def test_compute_gradient_uses_loss_function():
    model = dummy_mace_model()
    atoms = setup_calculator(create_dummy_atoms(), model, device="cpu", dtype_str="float32")

    fgsm = FGSM_ASE(atoms.calc, epsilon=0.1, device="cpu")

    default_gradients = fgsm.compute_gradient(atoms)
    
    energy_loss = lambda energy: energy
    loss_function = energy_loss
    custom_gradients = fgsm.compute_gradient(atoms, loss_fn=loss_function)

    assert np.allclose(custom_gradients, default_gradients, atol=1e-5)
