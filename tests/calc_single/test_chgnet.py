import pytest
import torch

chgnet = pytest.importorskip(
    "chgnet",
    reason="test_chgnet.py requires CHGNet dependencies",
)

from ase import Atoms, build
from ase.io import write
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from mlff_attack import relaxation
from mlff_attack.random_seed import set_random_seed
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_load_chgnet_dependency():
    assert chgnet is not None
    assert CHGNet is not None
    assert CHGNetCalculator is not None


def test_load_structure():
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


def test_setup_chgnet_calculator():
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms_with_chgnet = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float32",
        calculator="chgnet",
    )

    assert atoms_with_chgnet is not None
    assert atoms_with_chgnet.calc is not None
    assert isinstance(atoms_with_chgnet.calc, CHGNetCalculator)


def test_setup_chgnet_calculator_rejects_invalid_model():
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms_with_chgnet = relaxation.setup_calculator(
        atoms,
        "chgnet-invalid",
        device="cpu",
        dtype_str="float32",
        calculator="chgnet",
    )

    assert atoms_with_chgnet is None


def test_setup_chgnet_calculator_rejects_invalid_dtype():
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms_with_chgnet = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float16",
        calculator="chgnet",
    )

    assert atoms_with_chgnet is None


def test_setup_chgnet_calculator_rejects_mace_head():
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms_with_chgnet = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float32",
        calculator="chgnet",
        mace_head="omat_pbe",
    )

    assert atoms_with_chgnet is None


def test_setup_chgnet_calculator_rejects_uma_charge_and_spin():
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms_with_chgnet = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float32",
        calculator="chgnet",
        uma_charge=0,
        uma_spin=1,
    )

    assert atoms_with_chgnet is None


def test_chgnet_energy_and_forces():
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float32",
        calculator="chgnet",
    )

    assert atoms is not None
    assert atoms.calc is not None

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert torch.as_tensor(energy).ndim == 0
    assert torch.is_floating_point(torch.as_tensor(energy))
    assert torch.isfinite(torch.as_tensor(energy))
    assert forces.shape == (len(atoms), 3)


def test_get_optimizer_class():
    opt_class = relaxation.get_optimizer_class("BFGS")
    assert opt_class is not None

    opt_class = relaxation.get_optimizer_class("LBFGS")
    assert opt_class is not None


def test_run_relaxation():
    atoms = build.bulk("Si", "diamond", a=5.43)
    atoms = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float32",
        calculator="chgnet",
    )

    assert atoms is not None
    assert atoms.calc is not None

    traj_path = Path(__file__).parent / "data" / "test_chgnet_traj.traj"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    with patch(
        "mlff_attack.relaxation.get_optimizer_class"
    ) as mock_get_optimizer:
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.run = MagicMock()
        mock_optimizer_instance.nsteps = 5

        mock_optimizer_class = MagicMock(
            return_value=mock_optimizer_instance
        )
        mock_get_optimizer.return_value = mock_optimizer_class

        success = relaxation.run_relaxation(
            atoms,
            str(traj_path),
            optimizer="BFGS",
            fmax=0.01,
            max_steps=10,
        )

        mock_get_optimizer.assert_called_once_with("BFGS")
        mock_optimizer_class.assert_called_once()
        mock_optimizer_instance.run.assert_called_once_with(
            fmax=0.01,
            steps=10,
        )

        assert success is True

    if traj_path.exists():
        traj_path.unlink()

    try:
        traj_path.parent.rmdir()
    except OSError:
        pass


def test_saving_results_to_files():
    atoms = build.bulk("Si", "diamond", a=5.43)

    output_dir = Path(__file__).parent / "data" / "chgnet_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    relaxation.save_results(
        atoms,
        output_dir,
        base_name="test_chgnet",
    )

    output_file = output_dir / "test_chgnet.cif"
    assert output_file.exists()

    output_file.unlink()
    try:
        output_dir.rmdir()
        output_dir.parent.rmdir()
    except OSError:
        pass


@pytest.mark.parametrize(
    ("dtype_str", "expected_dtype"),
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
    ],
)
def test_dtype_toggle(dtype_str, expected_dtype):
    atoms = build.bulk("Si", "diamond", a=5.43)

    atoms = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str=dtype_str,
        calculator="chgnet",
    )

    assert atoms is not None
    assert atoms.calc is not None
    assert next(atoms.calc.model.parameters()).dtype == expected_dtype


def test_seed_sets_torch_rng():
    seed = 43

    set_random_seed(seed)
    first_draw = torch.rand(5)

    set_random_seed(seed)
    second_draw = torch.rand(5)

    assert torch.initial_seed() == seed
    assert torch.allclose(first_draw, second_draw)

    atoms = build.bulk("Si", "diamond", a=5.43)
    atoms = relaxation.setup_calculator(
        atoms,
        "chgnet-0.3.0",
        device="cpu",
        dtype_str="float32",
        seed=seed,
        calculator="chgnet",
    )

    assert atoms is not None
    assert atoms.calc is not None
    assert torch.initial_seed() == seed