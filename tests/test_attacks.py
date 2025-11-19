import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from mlff_attack.grad_based.fgsm import FGSM_MACE
from mlff_attack.grad_based.pgd import PGD_MACE
# from mlff_attack.grad_based.bim import BIM_MACE

from mace.calculators import mace_mp
from mlff_attack.relaxation import setup_calculator

from ase import Atoms
from ase import build
from ase.io import write, read
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

class TestFGSM_MACE:
    def test_init(self):
        model = dummy_model()
        attack = FGSM_MACE(model, epsilon=0.1)
        assert attack.epsilon == 0.1
        assert attack.model == model

    def test_attack_basic(self):
        atoms = setup_calculator(create_dummy_atoms(), dummy_model(), device="cpu", dtype_str="float32")

        fgsm = FGSM_MACE(atoms.calc, epsilon=0.1, device="cpu")
        perturbed_atoms = fgsm.attack(atoms, n_steps=1, clip=True)

        assert perturbed_atoms.get_positions().shape == atoms.get_positions().shape
        assert not np.array_equal(perturbed_atoms.get_positions(), atoms.get_positions())

    def test_epsilon_scaling(self):
        model = dummy_model()
        
        fgsm = FGSM_MACE(model, epsilon=0.5)
        atoms = create_dummy_atoms()
        atoms = setup_calculator(atoms, model, device="cpu")

        perturbed_atoms = fgsm.attack(atoms)
        perturbation = perturbed_atoms.get_positions() - atoms.get_positions()
        assert np.all(np.abs(perturbation) <= 0.5 + 1e-6)
    
@pytest.mark.skip(reason="Test not yet implemented")
class TestPGD:
    def test_init(self):
        model = dummy_model()
        attack = PGD_MACE(model, epsilon=0.1, alpha=0.01, num_iter=10)
        assert attack.epsilon == 0.1
        assert attack.alpha == 0.01
        assert attack.num_iter == 10

    def test_attack_iterations(self):
        model = dummy_model()
        
        attack = PGD_MACE(model, epsilon=0.1, alpha=0.01, num_iter=5)
        atoms = create_dummy_atoms()
        atoms.calc = [model]

        perturbed = attack.attack(atoms)
        assert perturbed.shape == atoms.get_positions().shape

    def test_projection_bounds(self):
        model = dummy_model()
        
        attack = PGD_MACE(model, epsilon=0.2, alpha=0.1, num_iter=3)
        positions = torch.zeros((5, 3), requires_grad=True)
        
        perturbed = attack.attack(positions)
        perturbation = perturbed - positions
        assert torch.all(torch.abs(perturbation) <= 0.2 + 1e-6)

# @pytest.mark.skip(reason="Test not yet implemented")
# class TestBIM:
#     def test_init(self):
#         model = Mock()
#         attack = BIM(model, epsilon=0.1, alpha=0.01, num_iter=10)
#         assert attack.epsilon == 0.1
#         assert attack.alpha == 0.01
#         assert attack.num_iter == 10

#     def test_attack_basic(self):
#         model = Mock()
#         model.return_value = {'energy': torch.tensor([1.0], requires_grad=True)}
        
#         attack = BIM(model, epsilon=0.1, alpha=0.01, num_iter=5)
#         positions = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
        
#         perturbed = attack.attack(positions)
#         assert perturbed.shape == positions.shape
#         assert not torch.equal(perturbed, positions)

#     def test_iterative_refinement(self):
#         model = Mock()
#         call_count = 0
        
#         def mock_forward(*args, **kwargs):
#             nonlocal call_count
#             call_count += 1
#             return {'energy': torch.tensor([1.0], requires_grad=True)}
        
#         model.side_effect = mock_forward
        
#         attack = BIM(model, epsilon=0.1, alpha=0.01, num_iter=7)
#         positions = torch.zeros((2, 3), requires_grad=True)
        
#         perturbed = attack.attack(positions)
#         assert call_count == 7