import shutil

import numpy as np
import pytest
import torch

mlp_path = shutil.which("mlp")
if mlp_path is None:
    pytest.skip(
        "test_mtp.py requires the MLIP-3 'mlp' executable. "
        "Switch to env-mtp.",
        allow_module_level=True,
    )

from ase import Atoms, build
from ase.io import write

from mlff_attack import relaxation
from mlff_attack.calc_setup.mtp import MTPCalculator
from mlff_attack.random_seed import set_random_seed
from pathlib import Path


def test_load_mtp_dependency():
    import mlff_attack.calc_setup.mtp as mtp

    assert mtp is not None
    assert MTPCalculator is not None


def test_load_structure():
    struct_file = Path(__file__).parent / "data" / "sample_mtp_struct.xyz"
    struct_file.parent.mkdir(parents=True, exist_ok=True)

    atoms = Atoms(
        "Li2",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    write(struct_file, atoms)

    loaded_atoms = relaxation.load_structure(str(struct_file))

    assert loaded_atoms is not None
    assert len(loaded_atoms) == len(atoms)

    struct_file.unlink()
    try:
        struct_file.parent.rmdir()
    except OSError:
        pass


def test_setup_mtp_calculator():
    atoms = build.bulk("Li", "bcc", a=3.5)

    atoms_with_mtp = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cpu",
        dtype_str="float64",
        calculator="mtp",
    )

    assert atoms_with_mtp is not None
    assert atoms_with_mtp.calc is not None
    assert isinstance(atoms_with_mtp.calc, MTPCalculator)


def test_setup_mtp_calculator_rejects_float32():
    atoms = build.bulk("Li", "bcc", a=3.5)

    atoms_with_mtp = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cpu",
        dtype_str="float32",
        calculator="mtp",
    )

    assert atoms_with_mtp is None


def test_setup_mtp_calculator_rejects_cuda():
    atoms = build.bulk("Li", "bcc", a=3.5)

    atoms_with_mtp = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cuda",
        dtype_str="float64",
        calculator="mtp",
    )

    assert atoms_with_mtp is None


def test_mtp_energy_and_forces():
    atoms = build.bulk("Li", "bcc", a=3.5)

    atoms = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cpu",
        dtype_str="float64",
        calculator="mtp",
    )

    assert atoms is not None
    assert atoms.calc is not None

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (len(atoms), 3)
    assert forces.dtype == np.float64
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))


def test_dtype_toggle():
    atoms = build.bulk("Li", "bcc", a=3.5)

    atoms_float64 = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cpu",
        dtype_str="float64",
        calculator="mtp",
    )

    assert atoms_float64 is not None
    assert atoms_float64.calc is not None

    atoms_float32 = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cpu",
        dtype_str="float32",
        calculator="mtp",
    )

    assert atoms_float32 is None


def test_seed_sets_torch_rng():
    seed = 43

    set_random_seed(seed)
    first_draw = torch.rand(5)

    set_random_seed(seed)
    second_draw = torch.rand(5)

    assert torch.initial_seed() == seed
    assert torch.allclose(first_draw, second_draw)

    set_random_seed(seed)
    atoms = build.bulk("Li", "bcc", a=3.5)

    atoms = relaxation.setup_calculator(
        atoms,
        "pot.almtp",
        device="cpu",
        dtype_str="float64",
        seed=seed,
        calculator="mtp",
    )

    assert atoms is not None
    assert atoms.calc is not None
    assert torch.initial_seed() == seed
