import pytest
import subprocess
import os
import sys
from pathlib import Path

@pytest.mark.cli
def test_cli_mace_calc_single():
    # Define input parameters
    input_cif = 'does_not_exist.cif'  # Intentionally incorrect path
    model_path = 'does_not_exist.model'  # Intentionally incorrect path
    outdir = "tests/output/mace_calc_single_test"

    # Ensure output directory is clean
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)

    # Construct command
    cmd = [
        "python",
        "src/mlff_attack/cli/calc_single.py",
        "--input", input_cif,
        "--model", model_path,
        "--outdir", outdir,
        "--device", "cpu",
        "--fmax", "0.02",
        "--max-steps", "100",
        "--optimizer", "LBFGS"
    ]

    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check that the command executed successfully
    assert result.returncode == 1, f"CLI failed with error: {result.stderr}"

    # Check that output files are created
    traj_path = Path(outdir) / "relaxed.traj"
    cif_path = Path(outdir) / "relaxed.cif"
    assert not traj_path.exists(), "Trajectory file should not have been created."
    assert not cif_path.exists(), "Relaxed CIF file should not have been created."

    # Clean up after test
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)


def test_cli_uma_calc_single():
    # Define input parameters
    input_cif = 'does_not_exist.cif'  # Intentionally incorrect path
    model_path = 'does_not_exist.model'  # Intentionally incorrect path
    outdir = "tests/output/uma_calc_single_test"

    # Ensure output directory is clean
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)

    # Construct command
    cmd = [
        "python",
        "src/mlff_attack/cli/calc_single.py",
        "--input", input_cif,
        "--model", model_path,
        "--outdir", outdir,
        "--device", "cpu",
        "--uma-task", "omat",
        "--fmax", "0.02",
        "--max-steps", "100",
        "--optimizer", "LBFGS"
    ]

    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check that the command executed successfully
    assert result.returncode == 1, f"CLI failed with error: {result.stderr}"

    # Check that output files are created
    traj_path = Path(outdir) / "relaxed.traj"
    cif_path = Path(outdir) / "relaxed.cif"
    assert not traj_path.exists(), "Trajectory file should not have been created."
    assert not cif_path.exists(), "Relaxed CIF file should not have been created."

    # Clean up after test
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)


def test_cli_calc_single_accepts_head():
    cmd = [
        sys.executable,
        "src/mlff_attack/cli/calc_single.py",
        "--input", "does_not_exist.cif",
        "--model", "mace-mh-1.model",
        "--outdir", "tests/output/mace_mh_calc_single_test",
        "--mace-head", "omat_pbe",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 1
    assert "unrecognized arguments" not in result.stderr


def test_cli_calc_single_rejects_head():
    cmd = [
        sys.executable,
        "src/mlff_attack/cli/calc_single.py",
        "--input", "does_not_exist.cif",
        "--model", "mace_sample.model",
        "--outdir", "tests/output/mace_mh_calc_single_test",
        "--mace-head", "omat_pbe",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--mace-head can only be used with MACE-MH models" in result.stderr


@pytest.mark.cli
def test_cli_make_attack():
    # Define input parameters
    input_cif = 'does_not_exist.cif'  # Intentionally incorrect path
    model_path = 'does_not_exist.model'  # Intentionally incorrect path
    outdir = "tests/output/make_attack_test"

    # Ensure output directory is clean
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)

    # Construct command
    cmd = [
        "python",
        "src/mlff_attack/cli/make_attack.py",
        "--input", input_cif,
        "--model", model_path,
        "--outdir", outdir,
        "--device", "cpu",
        "--type", "fgsm",
        "--epsilon", "0.05"
    ]

    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check that the command executed successfully
    assert result.returncode == 1, f"CLI failed with error: {result.stderr}"

    # Check that output files are created
    perturbed_cif_path = Path(outdir) / "perturbed.cif"
    perturbation_npz_path = Path(outdir) / "perturbation.npz"
    assert not perturbed_cif_path.exists(), "Perturbed CIF file should not have been created."
    assert not perturbation_npz_path.exists(), "Perturbation NPZ file should not have been created."

    # Clean up after test
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)


def test_cli_mace_rejects_uma_args():
    cmd = [
        sys.executable,
        "src/mlff_attack/cli/calc_single.py",
        "--input", "does_not_exist.cif",
        "--model", "mace_sample.model",
        "--outdir", "tests/output/mace_rejects_uma_charge_test",
        "--uma-task", "omat",
        "--uma-charge", "0",
        "--uma-spin", "0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--uma-task can only be used with UMA" in result.stderr


def test_cli_uma_rejects_head():
    cmd = [
        sys.executable,
        "src/mlff_attack/cli/calc_single.py",
        "--input", "does_not_exist.cif",
        "--model", "uma_sample",
        "--outdir", "tests/output/mace_rejects_uma_charge_test",
        "--mace-head", "omat_pbe"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--mace-head can only be used with MACE-MH models" in result.stderr


@pytest.mark.cli
def test_cli_fgsm_accepts_all_arguments():
    cmd = [
        "python",
        "src/mlff_attack/cli/make_attack.py",
        "--input", "does_not_exist.cif",
        "--model", "does_not_exist.model",
        "--device", "cpu",
        "--dtype", "float32",
        "--seed", "43",
        "--epsilon", "0.05",
        "--outdir", "tests/output/make_attack_test",
        "--target-energy", "1.5",
        "--type", "fgsm",
        "--n-steps", "3",
        "--clip",
        "--no-visualize",
    ]

    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Check that the command executed successfully
    assert result.returncode == 1, f"CLI failed with error: {result.stderr}"
    assert "unrecognized arguments" not in result.stderr


def test_cli_fgsm_rejects_alpha():
    cmd = [
        sys.executable,
        "src/mlff_attack/cli/make_attack.py",
        "--input", "does_not_exist.cif",
        "--model", "does_not_exist.model",
        "--device", "cpu",
        "--epsilon", "0.05",
        "--alpha", "0.01",
        "--outdir", "tests/output/make_attack_test",
        "--type", "fgsm",
        "--no-visualize",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--alpha can only be used with --type pgd" in result.stderr


def test_cli_pgd_accepts_all_arguments():
    cmd = [
        "python",
        "src/mlff_attack/cli/make_attack.py",
        "--input", "does_not_exist.cif",
        "--model", "does_not_exist.model",
        "--device", "cpu",
        "--dtype", "float32",
        "--seed", "43",
        "--epsilon", "0.05",
        "--alpha", "0.01",
        "--outdir", "tests/output/make_attack_test",
        "--target-energy", "1.5",
        "--type", "pgd",
        "--n-steps", "3",
        "--clip",
        "--no-visualize",
    ]

    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Check that the command executed successfully
    assert result.returncode == 1, f"CLI failed with error: {result.stderr}"
    assert "unrecognized arguments" not in result.stderr


@pytest.mark.cli
def test_cli_visualize_traj():
    # Define input parameters
    perturbation_npz = 'does_not_exist.traj'  # Intentionally incorrect path
    output_plot = "tests/output/visualize_traj_test"

    # Ensure output directory is clean
    outdir = Path(output_plot).parent
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)

    # Construct command
    cmd = [
        "python",
        "src/mlff_attack/cli/visualize_traj.py",
        "--traj", perturbation_npz,
        "--outdir", outdir,
        "--format", "png"
    ]

    # Run the CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check that the command executed successfully
    assert result.returncode == 1, f"CLI failed with error: {result.stderr}"

    # Check that output plot is created
    assert not Path(output_plot).exists(), "Visualization plot should not have been created."

    # Clean up after test
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)

def test_cli_calc_single_accepts_seed_and_dtype():
    cmd = [
        sys.executable,
        "src/mlff_attack/cli/calc_single.py",
        "--input", "does_not_exist.cif",
        "--model", "mace_sample.model",
        "--outdir", "tests/output/calc_single_seed_dtype_test",
        "--device", "cpu",
        "--dtype", "float32",
        "--seed", "43",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 1
    assert "unrecognized arguments" not in result.stderr