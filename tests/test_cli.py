import pytest
import subprocess
import os
from pathlib import Path

@pytest.mark.cli
def test_cli_mace_calc_single():
    """Test the MACE single structure relaxation CLI."""


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
        "src/mlff_attack/cli/mace_calc_single.py",
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

@pytest.mark.cli
def test_cli_make_attack():
    """Test make-attack CLI."""

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


@pytest.mark.cli
def test_cli_visualize_traj():
    """Test visualize-traj CLI."""

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