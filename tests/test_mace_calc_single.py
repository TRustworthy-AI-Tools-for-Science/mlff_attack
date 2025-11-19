import pytest
from mlff_attack import relaxation
import mace
from mace.calculators import mace_mp
from ase import build
from ase import Atoms
from pathlib import Path

def test_load_mace_model():
    calc = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    assert isinstance(calc, mace.calculators.mace.MACECalculator)

def test_load_structure():
    """Test loading a structure file."""
    from ase import Atoms
    from ase.io import write, read
    from pathlib import Path

    # Create a temporary structure file
    struct_file = Path(__file__).parent / "data" / "sample_struct.xyz"
    struct_file.parent.mkdir(parents=True, exist_ok=True)
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0.76, 0.58, 0], [-0.76, 0.58, 0]])
    write(struct_file, atoms)

    loaded_atoms = relaxation.load_structure(str(struct_file))
    assert loaded_atoms is not None
    assert len(loaded_atoms) == len(atoms)

    # Cleanup
    struct_file.unlink()
    try:
        struct_file.parent.rmdir()
    except OSError:
        pass

def test_setup_calculator():
    """Test setting up MACE calculator."""

    # Create a dummy atoms object
    atoms = build.molecule("H2O")
                
    # Use a non-existent model path for testing
    model_path = Path(__file__).parent / "data" / "non_existent_model.pth"

    atoms_with_calc = relaxation.setup_calculator(atoms, str(model_path), device="cpu", dtype_str="float32")
    assert atoms_with_calc is None  # Should fail to load model

    model = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')

    atoms_with_mace = relaxation.setup_calculator(atoms, model, device="cpu", dtype_str="float32")

    assert atoms_with_mace is not None
    

def test_get_optimizer_class():
    """Test getting optimizer class."""
    opt_class = relaxation.get_optimizer_class("BFGS")
    assert opt_class is not None

    opt_class = relaxation.get_optimizer_class("LBFGS")
    assert opt_class is not None

def test_run_relaxation():
    """Test running relaxation (mocked)."""
    from ase import Atoms
    from unittest.mock import patch, MagicMock
    from pathlib import Path

    atoms = build.molecule("H2O")
    atoms.calc = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')
    
    traj_path = Path(__file__).parent / "data" / "test_traj.traj"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    with patch('mlff_attack.relaxation.get_optimizer_class') as mock_get_opt:
        # Create a mock optimizer instance
        mock_opt_instance = MagicMock()
        mock_opt_instance.run = MagicMock()
        mock_opt_instance.nsteps = 5
        
        # Create a mock optimizer class that returns the instance when called
        mock_opt_class = MagicMock(return_value=mock_opt_instance)
        mock_get_opt.return_value = mock_opt_class
        
        success = relaxation.run_relaxation(atoms, str(traj_path), optimizer="BFGS", fmax=0.01, max_steps=10)
        
        # Verify the optimizer class getter was called with correct optimizer name
        mock_get_opt.assert_called_once_with("BFGS")
        
        # Verify the optimizer class was instantiated
        mock_opt_class.assert_called_once()
        
        # Verify the optimizer's run method was called
        mock_opt_instance.run.assert_called_once_with(fmax=0.01, steps=10)
        
        # The function should return True on success
        assert success == True
    
    # Cleanup
    if traj_path.exists():
        traj_path.unlink()
    try:
        traj_path.parent.rmdir()
    except OSError:
        pass




def test_save_results():
    """Test saving results to files."""
    from ase import Atoms
    from pathlib import Path

    atoms = Atoms('H2O', positions=[[0, 0, 0], [0.76, 0.58, 0], [-0.76, 0.58, 0]])

    output_dir = Path(__file__).parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    relaxation.save_results(atoms, output_dir, base_name="test_traj")

    traj_file = output_dir / "test_traj.cif"

    assert traj_file.exists()

    # Cleanup
    traj_file.unlink()
    try:
        output_dir.rmdir()
        output_dir.parent.rmdir()
    except OSError:
        pass
