#!/usr/bin/env python3
"""
Trajectory visualization functionality.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read

logger = logging.getLogger(__name__)


def load_trajectory(traj_path):
    """Load a trajectory file and validate it exists.

    Parameters
    ----------
    traj_path : str or Path
        Path to the trajectory file

    Returns
    -------
    list of ase.Atoms or None
        List of ASE Atoms objects, or None if loading fails
    """
    traj_path = Path(traj_path)
    if not traj_path.exists():
        logger.info("[ERROR] Trajectory file not found: %s", traj_path)
        return None

    logger.info("[INFO] Reading trajectory: %s", traj_path)
    try:
        traj = read(traj_path, index=":")
        if len(traj) == 0:
            logger.error("[ERROR] Trajectory contains no frames: %s", traj_path)
            return None
        logger.info("[INFO] Trajectory contains %s frames", len(traj))
        return traj
    except (OSError, ValueError, RuntimeError) as exc:
        logger.info("[ERROR] Failed to read trajectory: %s", exc)
        return None


def extract_trajectory_data(traj):
    """Extract energy, force, and volume data from trajectory.

    Parameters
    ----------
    traj : list of ase.Atoms
        List of ASE Atoms objects

    Returns
    -------
    tuple
        A tuple containing:

        - steps : list of int
            Step numbers
        - energies : list of float
            Energies at each step
        - max_forces : list of float
            Maximum forces at each step
        - volumes : list of float
            Volumes at each step
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
        except (ValueError, RuntimeError):
            energies.append(np.nan)

        # Forces
        try:
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            max_forces.append(max_force)
        except (ValueError, RuntimeError):
            max_forces.append(np.nan)

        # Volume
        volume = atoms.get_volume()
        volumes.append(volume)

    return steps, energies, max_forces, volumes


def calculate_noise_spectrum(max_forces):
    """Calculate noise spectrum from maximum forces.

    Parameters
    ----------
    max_forces : list of float
        List of maximum forces

    Returns
    -------
    tuple
        A tuple containing:

        - frequencies : np.ndarray
            Frequency array
        - spectrum : np.ndarray
            Power spectrum
    """
    forces_array = np.array(max_forces)
    n = len(forces_array)
    freq = np.fft.rfftfreq(n)
    spectrum = np.abs(np.fft.rfft(forces_array - np.mean(forces_array)))**2
    return freq, spectrum

def calculate_statistics(energies, max_forces, volumes, fmax=0.01):
    """Calculate summary statistics from trajectory data.

    Parameters
    ----------
    energies : list of float
        List of energies
    max_forces : list of float
        List of maximum forces
    volumes : list of float
        List of volumes

    Returns
    -------
    dict
        Dictionary of statistics including initial/final energies, forces, volumes,
        convergence status, and change percentages
    """
    stats = {}

    # Energy statistics
    initial_energy = energies[0] if not np.isnan(energies[0]) else None
    final_energy = energies[-1] if not np.isnan(energies[-1]) else None
    stats['initial_energy'] = initial_energy
    stats['final_energy'] = final_energy
    stats['energy_change'] = (
        final_energy - initial_energy
        if (initial_energy is not None and final_energy is not None)
        else None
    )

    # Force statistics
    initial_force = max_forces[0] if not np.isnan(max_forces[0]) else None
    final_force = max_forces[-1] if not np.isnan(max_forces[-1]) else None
    stats['initial_force'] = initial_force
    stats['final_force'] = final_force
    stats['converged'] = final_force < fmax if final_force is not None else None
    # Volume statistics
    initial_volume = volumes[0]
    final_volume = volumes[-1]
    stats['initial_volume'] = initial_volume
    stats['final_volume'] = final_volume
    stats['volume_change_percent'] = ((final_volume - initial_volume) / initial_volume) * 100

    return stats


def plot_energy(ax, steps, energies):
    """Plot energy evolution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on
    steps : list of int
        Step numbers
    energies : list of float
        Energy values at each step
    """
    if not all(np.isnan(energies)):
        ax.plot(steps, energies, 'b-o', markersize=4, label="Energy")
        ax.legend()
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Total Energy')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, 'Energy data not available',
            ha='center', va='center', transform=ax.transAxes
        )
        ax.set_title('Total Energy')


def plot_forces(ax, steps, max_forces, fmax=0.01):
    """Plot force convergence.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on
    steps : list of int
        Step numbers
    max_forces : list of float
        Maximum force values at each step
    """
    if not all(np.isnan(max_forces)):
        ax.plot(steps, max_forces, 'r-o', markersize=4, label="Max Force")
        default_fmax_values = [0.01, 0.05]
        for value in default_fmax_values:
            color = 'orange' if fmax == value else 'g'
            ax.axhline(
                y=value,
                color=color,
                linestyle='--',
                label=f'fmax={value:g} eV/Å'
            )
        if fmax not in default_fmax_values:
            ax.axhline(
                y=fmax,
                color='orange',
                linestyle='--',
                label=f'fmax={fmax:g} eV/Å'
            )
        ax.set_xlabel('Step')
        ax.set_ylabel('Max Force (eV/Å)')
        ax.set_title('Maximum Force')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, 'Force data not available',
            ha='center', va='center', transform=ax.transAxes
        )
        ax.set_title('Maximum Force')


def plot_volume(ax, steps, volumes):
    """Plot volume evolution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on
    steps : list of int
        Step numbers
    volumes : list of float
        Volume values at each step
    """
    ax.plot(steps, volumes, 'g-o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Volume (Å³)')
    ax.set_title('Cell Volume')
    ax.grid(True, alpha=0.3)


def plot_summary(ax, stats, n_frames, fmax=0.01):
    """Plot summary statistics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on
    stats : dict
        Dictionary of statistics from calculate_statistics
    n_frames : int
        Number of frames in trajectory

    Returns
    -------
    str
        Formatted summary text
    """
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
        summary_text += f"Converged (fmax<{fmax}): {converged}\n\n"
    else:
        summary_text += "Forces: Not available\n\n"

    summary_text += f"Initial volume: {stats['initial_volume']:.2f} Å³\n"
    summary_text += f"Final volume: {stats['final_volume']:.2f} Å³\n"
    summary_text += f"Volume change: {stats['volume_change_percent']:+.2f}%"

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

    return summary_text


def plot_noise(ax, freq, spectrum):
    """Plot noise spectrum of maximum forces.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on
    freq : np.ndarray
        Frequency array
    spectrum : np.ndarray
        Power spectrum values
    """

    if not all(np.isnan(spectrum)):
        ax.plot(freq, spectrum, 'm-', label="Noise Spectrum")
        ax.legend()
        ax.set_xlabel('Frequency (1/steps)')
        ax.set_ylabel('Power Spectrum')
        ax.set_title('Noise Spectrum of Max Forces')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, 'Force data not available',
            ha='center', va='center', transform=ax.transAxes
        )
        ax.set_title('Noise Spectrum of Max Forces')


def create_visualization(
    traj,
    traj_path,
    outdir,
    *,
    output_format='png',
    show=False,
    save_to_csv=True,
    fmax=0.01
):
    """Create visualization plots for trajectory data.

    Parameters
    ----------
    traj : list of ase.Atoms
        List of ASE Atoms objects
    traj_path : Path
        Path object for the trajectory file
    outdir : Path
        Output directory for plots
    output_format : str, optional
        Format for output plots (png, pdf, svg), by default 'png'
    show : bool, optional
        Whether to show plots interactively, by default False
    save_to_csv : bool, optional
        Whether to save extracted data to CSV files, by default True

    Returns
    -------
    str or None
        Path to saved plot, or None if failed
    """
    # Extract data
    steps, energies, max_forces, volumes = extract_trajectory_data(traj)
    fmax = float(traj[-1].info.get("fmax", fmax))

    # Calculate statistics
    stats = calculate_statistics(energies, max_forces, volumes, fmax)

    # Calculate noise spectrum
    freq, spectrum = calculate_noise_spectrum(max_forces)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Relaxation Trajectory: {traj_path.name}', fontsize=14, fontweight='bold')

    # Create plots
    plot_energy(axes[0, 0], steps, energies)
    plot_forces(axes[0, 1], steps, max_forces, fmax)
    # plot_volume(axes[1, 0], steps, volumes)
    plot_noise(axes[1, 0], freq, spectrum)

    summary_text = plot_summary(axes[1, 1], stats, len(traj) - 1, fmax)

    plt.tight_layout()

    # Save figure
    output_file = outdir / f"relaxation_analysis.{output_format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info("[INFO] Plot saved to: %s", output_file)

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    # Print summary to console
    logger.info("\\n%s", "=" * 50)
    logger.info(summary_text)
    logger.info("%s", "=" * 50)

    # Save data to CSV if requested
    if save_to_csv:
        data = {
            'Step': steps,
            'Energy (eV)': energies,
            'Max Force (eV/Å)': max_forces,
            'Volume (Å³)': volumes
        }
        df = pd.DataFrame(data)
        csv_file = outdir / "relaxation_data.csv"
        df.to_csv(csv_file, index=False)
        logger.info("[INFO] Data saved to CSV: %s", csv_file)

        noise = {
            'Frequency (1/steps)': freq,
            'Power Spectrum': spectrum
        }
        df_noise = pd.DataFrame(noise)
        csv_noise_file = outdir / "noise_spectrum.csv"
        df_noise.to_csv(csv_noise_file, index=False)
        logger.info("[INFO] Noise spectrum data saved to CSV: %s", csv_noise_file)

    return str(output_file)
