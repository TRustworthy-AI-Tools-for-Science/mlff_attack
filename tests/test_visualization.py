"""Unit tests for visualization functions in mlff_attack.cli.visualize_traj."""
import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("Agg")

from mlff_attack import __version__
from mlff_attack import visualization
from pathlib import Path

def create_example_traj():
    """Create a simple example trajectory for testing."""
    from ase import Atoms
    from ase.io import Trajectory
    from pathlib import Path

    # Create the data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    traj_file = data_dir / "sample_traj.xyz"
    traj = Trajectory(str(traj_file), "w")
    for i in range(5):
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0.76 + i*0.1, 0.58, 0], [-0.76 - i*0.1, 0.58, 0]])
        atoms.set_cell([10, 10, 10])
        
        # Add fake energy and forces for testing
        from ase.calculators.singlepoint import SinglePointCalculator
        energy = -10.0 - i * 0.5  # Decreasing energy (relaxation)
        forces = [[0.1/(i+1), 0.05/(i+1), 0.02/(i+1)] for _ in range(len(atoms))]  # Decreasing forces
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc
        
        traj.write(atoms)
    traj.close()
    
    return str(traj_file)

def clean_up_example_traj(traj_file):
    """Remove the example trajectory file after testing."""
    p = Path(traj_file)
    if p.exists():
        p.unlink()
        try:
            p.parent.rmdir()
        except OSError:
            pass

def test_load_trajectory():
    """Test loading a trajectory file."""

    traj_file = create_example_traj()

    traj = visualization.load_trajectory(traj_file)
    assert traj is not None
    assert len(traj) > 0
    assert all(hasattr(atoms, 'get_potential_energy') for atoms in traj)

    # cleanup created trajectory file and data directory if empty
    clean_up_example_traj(traj_file)

def test_extract_trajectory_data():
    """Test extracting data from trajectory."""
    
    traj_file = create_example_traj()
    traj = visualization.load_trajectory(traj_file)
    steps, energies, max_forces, volumes = visualization.extract_trajectory_data(traj)
    
    assert len(steps) == len(traj)
    assert len(energies) == len(traj)
    assert len(max_forces) == len(traj)
    assert len(volumes) == len(traj)

    # cleanup created trajectory file and data directory if empty
    clean_up_example_traj(traj_file)

def test_calculate_statistics():
    """Test calculation of statistics from trajectory data."""
    
    traj_file = create_example_traj()
    traj = visualization.load_trajectory(traj_file)
    steps, energies, max_forces, volumes = visualization.extract_trajectory_data(traj)
    
    fmax = 0.03 # added fmax test - DC
    stats = visualization.calculate_statistics(energies, max_forces, volumes, fmax)
    
    assert 'initial_energy' in stats
    assert 'final_energy' in stats
    assert 'energy_change' in stats

    assert 'initial_force' in stats
    assert 'final_force' in stats
    assert 'converged' in stats
    assert stats['converged'] == (stats['final_force'] < fmax) # test convergence result - DC

    assert 'initial_volume' in stats
    assert 'final_volume' in stats
    assert 'volume_change_percent' in stats

    # cleanup created trajectory file and data directory if empty
    clean_up_example_traj(traj_file)


def test_converged_when_final_force_below_fmax():
    """Verify relaxation summary in visualize-traj displays "converged" using custom fmax - DC"""
    energies = [3, 2, 1]
    max_forces = [0.1, 0.05, 0.015]
    volumes = [100, 100, 100]
    fmax = 0.02 # final force (0.015) < 0.02 so converged
    stats = visualization.calculate_statistics(energies, max_forces, volumes, fmax)

    fig, ax = plt.subplots()
    summary_text = visualization.plot_summary(ax, stats, len(energies) - 1, fmax)
    plt.close(fig)

    assert "Converged" in summary_text
    assert "Yes" in summary_text

def test_not_converged_when_final_force_above_fmax():
    """Test that relaxation summary in visualize-traj displays "not converged" using custom fmax - DC"""
    energies = [3, 2, 1]
    max_forces = [0.1, 0.05, 0.015]
    volumes = [100.0, 100.0, 100.0]
    fmax = 0.001 # final force (0.015) > 0.001 so not converged
    stats = visualization.calculate_statistics(energies, max_forces, volumes, fmax)
    
    fig, ax = plt.subplots()
    summary_text = visualization.plot_summary(ax, stats, len(energies) - 1, fmax)
    plt.close(fig)

    assert stats["converged"] is False
    assert f"Converged (fmax<{fmax}): No\n\n" in summary_text

def test_plot_displays_custom_fmax_line():
    """Test that the plot displays a horizontal line at the custom fmax value - DC"""
    steps = [0, 1, 2]
    max_forces = [0.1, 0.05, 0.015]
    fmax = 0.03

    fig, ax = plt.subplots()
    visualization.plot_forces(ax, steps, max_forces, fmax)
    plt.close(fig)

    # Assert that the y = custom fmax
    lines = ax.get_lines()
    fmax_line = lines[-1]

    assert list(fmax_line.get_ydata()) == [fmax, fmax]
    assert fmax_line.get_label() == f"fmax={fmax:g} eV/Å"

def test_plot_energy():
    """Verify plot_energy draws a line through steps and energies - DC"""
    steps = [0, 1, 2]
    energies = [3, 2, 1]

    fig, ax = plt.subplots()
    visualization.plot_energy(ax, steps, energies)
    plt.close(fig)

    lines = ax.get_lines()
    energy_line = lines[0]

    assert list(energy_line.get_ydata()) == energies


def test_plot_volume():
    """Verify plot_volume draws a line through steps and volumes - DC"""
    steps = [0, 1, 2]
    volumes = [100, 200, 300]

    fig, ax = plt.subplots()
    visualization.plot_volume(ax, steps, volumes)
    plt.close(fig)
    
    lines = ax.get_lines()
    volume_line = lines[0]

    assert list(volume_line.get_ydata()) == volumes

    
def test_plot_noise():
    """Verify plot_noise draws a line through freq and spectrum - DC"""
    freq = [0, 1, 2]
    spectrum = [0, 1, 2]

    fig, ax = plt.subplots()
    visualization.plot_noise(ax, freq, spectrum)
    plt.close(fig)

    lines = ax.get_lines()
    noise_line = lines[0]

    assert list(noise_line.get_ydata()) == spectrum


def test_calculate_noise_spectrum():
    max_forces = [0.1, 0.05, 0.025]

    freq, spectrum = visualization.calculate_noise_spectrum(max_forces)

    assert len(freq) == len(spectrum)
    assert len(freq) > 0
    assert np.all(spectrum >= 0)


def test_create_visualization():
    """Verify create_visualization saves a relaxation plot - DC"""
    traj_file = create_example_traj()
    traj = visualization.load_trajectory(traj_file)
    outdir = Path(traj_file).parent

    output_file = visualization.create_visualization(
        traj,
        Path(traj_file),
        outdir,
        save_to_csv=False,
    )

    assert output_file is not None
    assert Path(output_file).exists()

    Path(output_file).unlink()
    clean_up_example_traj(traj_file)