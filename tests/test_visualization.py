"""Unit tests for visualization functions in mlff_attack.cli.visualize_traj."""
import pytest
from mlff_attack import __version__
from mlff_attack import visualization

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
    from pathlib import Path
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
    
    stats = visualization.calculate_statistics(energies, max_forces, volumes)
    
    assert 'initial_energy' in stats
    assert 'final_energy' in stats
    assert 'energy_change' in stats

    assert 'initial_force' in stats
    assert 'final_force' in stats
    assert 'converged' in stats

    assert 'initial_volume' in stats
    assert 'final_volume' in stats
    assert 'volume_change_percent' in stats

    # cleanup created trajectory file and data directory if empty
    clean_up_example_traj(traj_file)
