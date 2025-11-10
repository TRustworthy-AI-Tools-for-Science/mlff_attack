#!/usr/bin/env python3
"""
Trajectory visualization functionality.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from ase.io import read
import numpy as np


def load_trajectory(traj_path):
    """
    Load a trajectory file and validate it exists.
    
    Args:
        traj_path: Path to the trajectory file
        
    Returns:
        list: List of ASE Atoms objects, or None if loading fails
    """
    traj_path = Path(traj_path)
    if not traj_path.exists():
        print(f"[ERROR] Trajectory file not found: {traj_path}")
        return None

    print(f"[INFO] Reading trajectory: {traj_path}")
    try:
        traj = read(traj_path, index=":")
        print(f"[INFO] Trajectory contains {len(traj)} frames")
        return traj
    except Exception as e:
        print(f"[ERROR] Failed to read trajectory: {e}")
        return None


def extract_trajectory_data(traj):
    """
    Extract energy, force, and volume data from trajectory.
    
    Args:
        traj: List of ASE Atoms objects
        
    Returns:
        tuple: (steps, energies, max_forces, volumes)
    """
    steps = list(range(len(traj)))
    energies = []
    max_forces = []
    volumes = []
    
    for atoms in traj:
        # Energy
        try:
            energy = atoms.get_potential_energy()
            energies.append(energy)
        except Exception:
            energies.append(np.nan)
        
        # Forces
        try:
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            max_forces.append(max_force)
        except Exception:
            max_forces.append(np.nan)
        
        # Volume
        volume = atoms.get_volume()
        volumes.append(volume)
    
    return steps, energies, max_forces, volumes


def calculate_statistics(energies, max_forces, volumes):
    """
    Calculate summary statistics from trajectory data.
    
    Args:
        energies: List of energies
        max_forces: List of maximum forces
        volumes: List of volumes
        
    Returns:
        dict: Dictionary of statistics
    """
    stats = {}
    
    # Energy statistics
    initial_energy = energies[0] if not np.isnan(energies[0]) else None
    final_energy = energies[-1] if not np.isnan(energies[-1]) else None
    stats['initial_energy'] = initial_energy
    stats['final_energy'] = final_energy
    stats['energy_change'] = final_energy - initial_energy if (initial_energy is not None and final_energy is not None) else None
    
    # Force statistics
    initial_force = max_forces[0] if not np.isnan(max_forces[0]) else None
    final_force = max_forces[-1] if not np.isnan(max_forces[-1]) else None
    stats['initial_force'] = initial_force
    stats['final_force'] = final_force
    stats['converged'] = final_force < 0.01 if final_force is not None else None
    
    # Volume statistics
    initial_volume = volumes[0]
    final_volume = volumes[-1]
    stats['initial_volume'] = initial_volume
    stats['final_volume'] = final_volume
    stats['volume_change_percent'] = ((final_volume - initial_volume) / initial_volume) * 100
    
    return stats


def plot_energy(ax, steps, energies):
    """Plot energy evolution."""
    if not all(np.isnan(energies)):
        ax.plot(steps, energies, 'b-o', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Total Energy')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Energy data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Total Energy')


def plot_forces(ax, steps, max_forces):
    """Plot force convergence."""
    if not all(np.isnan(max_forces)):
        ax.plot(steps, max_forces, 'r-o', markersize=4)
        ax.axhline(y=0.01, color='g', linestyle='--', label='fmax=0.01 eV/Å')
        ax.axhline(y=0.05, color='orange', linestyle='--', label='fmax=0.05 eV/Å')
        ax.set_xlabel('Step')
        ax.set_ylabel('Max Force (eV/Å)')
        ax.set_title('Maximum Force')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Force data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Maximum Force')


def plot_volume(ax, steps, volumes):
    """Plot volume evolution."""
    ax.plot(steps, volumes, 'g-o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Volume (Å³)')
    ax.set_title('Cell Volume')
    ax.grid(True, alpha=0.3)


def plot_summary(ax, stats, n_frames):
    """Plot summary statistics."""
    ax.axis('off')
    
    # Create summary text
    summary_text = "Relaxation Summary\n" + "="*40 + "\n\n"
    summary_text += f"Total steps: {n_frames}\n\n"
    
    if stats['initial_energy'] is not None and stats['final_energy'] is not None:
        summary_text += f"Initial energy: {stats['initial_energy']:.6f} eV\n"
        summary_text += f"Final energy: {stats['final_energy']:.6f} eV\n"
        summary_text += f"Energy change: {stats['energy_change']:.6f} eV\n\n"
    else:
        summary_text += "Energy: Not available\n\n"
    
    if stats['initial_force'] is not None and stats['final_force'] is not None:
        summary_text += f"Initial max force: {stats['initial_force']:.6f} eV/Å\n"
        summary_text += f"Final max force: {stats['final_force']:.6f} eV/Å\n"
        converged = "Yes" if stats['converged'] else "No"
        summary_text += f"Converged (fmax<0.01): {converged}\n\n"
    else:
        summary_text += "Forces: Not available\n\n"
    
    summary_text += f"Initial volume: {stats['initial_volume']:.2f} Å³\n"
    summary_text += f"Final volume: {stats['final_volume']:.2f} Å³\n"
    summary_text += f"Volume change: {stats['volume_change_percent']:+.2f}%"
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return summary_text


def create_visualization(traj, traj_path, outdir, output_format='png', show=False):
    """
    Create visualization plots for trajectory data.
    
    Args:
        traj: List of ASE Atoms objects
        traj_path: Path object for the trajectory file
        outdir: Output directory for plots
        output_format: Format for output plots (png, pdf, svg)
        show: Whether to show plots interactively
        
    Returns:
        str: Path to saved plot, or None if failed
    """
    # Extract data
    steps, energies, max_forces, volumes = extract_trajectory_data(traj)
    
    # Calculate statistics
    stats = calculate_statistics(energies, max_forces, volumes)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Relaxation Trajectory: {traj_path.name}', fontsize=14, fontweight='bold')
    
    # Create plots
    plot_energy(axes[0, 0], steps, energies)
    plot_forces(axes[0, 1], steps, max_forces)
    plot_volume(axes[1, 0], steps, volumes)
    summary_text = plot_summary(axes[1, 1], stats, len(traj))
    
    plt.tight_layout()
    
    # Save figure
    output_file = outdir / f"relaxation_analysis.{output_format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {output_file}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary to console
    print("\n" + "="*50)
    print(summary_text)
    print("="*50)
    
    return str(output_file)
