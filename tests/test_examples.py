import pytest
import shutil
import os
from mlff_attack.examples.example_fgsm_attack import basic_fgsm_example
from mlff_attack.examples.example_fgsm_attack import iterative_fgsm_example
from mlff_attack.examples.example_fgsm_attack import targeted_attack_example
from mlff_attack.examples.example_fgsm_attack import load_and_analyze_example

from ase.atoms import Atoms
from mlff_attack.grad_based.fgsm import FGSM_MACE

@pytest.mark.smoke
def test_example_fgsm_attack():
    """Test example FGSM attack runs without error."""
    atoms, perturbed, attack = basic_fgsm_example()

    assert isinstance(atoms, Atoms)
    assert isinstance(perturbed, Atoms)
    assert isinstance(attack, FGSM_MACE)
    assert atoms.get_positions().shape == perturbed.get_positions().shape

    # Clean up after test
    if os.path.exists("example_outputs"):
        shutil.rmtree("example_outputs", ignore_errors=True)
    assert not os.path.exists("example_outputs")

@pytest.mark.smoke
def test_example_iterative_fgsm_attack():
    """Test example iterative FGSM attack runs without error."""
    atoms, perturbed, attack = iterative_fgsm_example()

    assert isinstance(atoms, Atoms)
    assert isinstance(perturbed, Atoms)
    assert isinstance(attack, FGSM_MACE)
    assert atoms.get_positions().shape == perturbed.get_positions().shape

    # Clean up after test
    if os.path.exists("example_outputs"):
        shutil.rmtree("example_outputs", ignore_errors=True)
    assert not os.path.exists("example_outputs")

@pytest.mark.smoke
def test_example_targeted_attack():
    """Test example targeted FGSM attack runs without error."""
    atoms, perturbed, attack = targeted_attack_example()

    assert isinstance(atoms, Atoms)
    assert isinstance(perturbed, Atoms)
    assert isinstance(attack, FGSM_MACE)
    assert atoms.get_positions().shape == perturbed.get_positions().shape

    # Clean up after test
    if os.path.exists("example_outputs"):
        shutil.rmtree("example_outputs", ignore_errors=True)
    assert not os.path.exists("example_outputs")


