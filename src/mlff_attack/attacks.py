#!/usr/bin/env python3

"""
Contains implementation for FGSM, I-FGSM, and PGD attacks on MLFF models.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from ase.io import read, write
from mlff_attack.relaxation import setup_calculator

def fgsm_attack(model, data, epsilon):
    """
    Perform Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: The machine learning force field model
        data: Input data to attack
        epsilon: Perturbation magnitude
        
    Returns:

        perturbed_data: The adversarially perturbed data
    """
    

    pass

def ifgsm_attack(model, data, epsilon, alpha, num_iterations):
    """
    Perform Iterative Fast Gradient Sign Method (I-FGSM) attack.
    
    Args:
        model: The machine learning force field model
        data: Input data to attack
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        num_iterations: Number of iterations to perform
        
    Returns:
        perturbed_data: The adversarially perturbed data
    """

    pass

def pgd_attack(model, data, epsilon, alpha, num_iterations):
    """
    Perform Projected Gradient Descent (PGD) attack.
    
    Args:
        model: The machine learning force field model
        data: Input data to attack
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        num_iterations: Number of iterations to perform
        
    Returns:
        perturbed_data: The adversarially perturbed data
    """

    pass


def make_attack(model_path, device, atoms, epsilon, target_energy, output_cif):
    """
    Perform an adversarial attack on the given atomic structure using a MACE model.

    Args:

        model_path: Path to the MACE model file
        device: Device to run the model on ("cpu" or "cuda")
        atoms: ASE Atoms object representing the structure to attack
        epsilon: Perturbation step size in Angstroms
        target_energy: Target energy for the attack (if None, maximize energy)
        output_cif: Path to save the perturbed CIF file
    Returns:
        output_cif: Path to the saved perturbed CIF file
    """

    # Setup calculator
    print(f"\nSetting up MACE calculator")
    print(f"   Model: {model_path}")
    print(f"   Device: {device}")
    atoms = setup_calculator(atoms, model_path, device)
    if atoms is None:
        raise RuntimeError("Failed to set up calculator")
    
    # Perform adversarial attack
    print(f"\nPerforming adversarial attack")
    print(f"   Epsilon: {epsilon} Å")
    if target_energy is not None:
        print(f"   Target energy: {target_energy} eV")
    else:
        print(f"   Mode: Maximize energy")
    
    perturbed_atoms, orig_energy, pert_energy, grads = adversarial_attack_step(
        atoms, device=device, epsilon=epsilon, target_energy=target_energy
    )
    
    energy_change = pert_energy - orig_energy
    print(f"   Original energy:  {orig_energy:.4f} eV")
    print(f"   Perturbed energy: {pert_energy:.4f} eV")
    print(f"   Energy change:    {energy_change:+.4f} eV")
    
    # Calculate displacement statistics
    displacement = perturbed_atoms.get_positions() - atoms.get_positions()
    displacement_mag = np.linalg.norm(displacement, axis=1)
    print(f"   Mean displacement: {displacement_mag.mean():.4f} Å")
    print(f"   Max displacement:  {displacement_mag.max():.4f} Å")
    
    # Save perturbed structure
    print(f"\nSaving perturbed structure to: {output_cif}")
    write(output_cif, perturbed_atoms)
    print(f"   Successfully saved!")

    return str(output_cif), perturbed_atoms


def forward_pass_with_gradients(atoms, device="cpu"):
    """
    Perform a forward pass through the MACE model with gradient tracking.
    
    This uses the calculator's internal method to prepare the batch,
    then replaces positions with a gradient-enabled version.
    
    Args:
        atoms: ASE Atoms object with MACE calculator attached
        device: Device to run on ("cpu" or "cuda")
    
    Returns:
        energy: Total energy (scalar, requires_grad=True)
        forces: Forces on atoms (shape [n_atoms, 3], with gradients)
        positions: Position tensor (requires_grad=True)
    """

    
    calc = atoms.calc
    model = calc.models[0]
    
    # Save original positions
    positions_np = atoms.get_positions()
    
    # Use the calculator's internal method to prepare the batch
    # Call the calculator to get the batch structure, then modify positions
    from mace.data import AtomicData, config_from_atoms
    
    # Create configuration from atoms
    config = config_from_atoms(atoms)
    
    # Create AtomicData with the calculator's settings
    atomic_data = AtomicData.from_config(
        config, z_table=calc.z_table, cutoff=calc.r_max
    )
    
    # Convert to dict
    batch = atomic_data.to_dict()
    
    # Move everything to the right device first
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device)
    
    # Add batch indexing if not present (on correct device)
    if "batch" not in batch:
        batch["batch"] = torch.zeros(len(atoms), dtype=torch.long, device=device)
    if "ptr" not in batch:
        batch["ptr"] = torch.tensor([0, len(atoms)], dtype=torch.long, device=device)
    
    # Now replace positions with gradient-enabled version
    positions = torch.tensor(positions_np, dtype=torch.float64, device=device, requires_grad=True)
    batch["positions"] = positions
    
    # Check and fix natoms if present
    if "natoms" in batch:
        natoms_val = batch["natoms"]
        # Ensure it's 1D with 2 elements
        if natoms_val.dim() == 0:
            batch["natoms"] = torch.tensor([len(atoms), len(atoms)], dtype=torch.long, device=device)
        elif natoms_val.dim() == 1 and len(natoms_val) < 2:
            batch["natoms"] = torch.tensor([len(atoms), len(atoms)], dtype=torch.long, device=device)
    else:
        batch["natoms"] = torch.tensor([len(atoms), len(atoms)], dtype=torch.long, device=device)
    
    # Add head field if present in calculator (for multi-head models)
    if hasattr(calc, 'head') and calc.head is not None:
        # Map head name to index
        if hasattr(calc, 'heads') and calc.heads is not None:
            head_idx = calc.heads.index(calc.head) if calc.head in calc.heads else 0
        else:
            head_idx = 0
        # Head should be an array with one value per atom in the batch
        batch["head"] = torch.full((len(atoms),), head_idx, dtype=torch.long, device=device)
    elif "head" not in batch:
        # Default head is 0
        batch["head"] = torch.zeros(len(atoms), dtype=torch.long, device=device)
    
    # Forward pass - disable computing forces in the model output
    model.eval()
    
    # We need to recompute with fresh gradients
    # Don't use model's internal force computation
    with torch.enable_grad():
        # Ensure positions require grad
        positions.requires_grad_(True)
        
        # Update batch with fresh positions
        batch["positions"] = positions
        
        # Forward pass through model
        output = model(batch, training=False, compute_force=False)
        
        # Extract energy
        energy = output["energy"]
        # Ensure energy is a scalar for gradient computation
        if energy.dim() > 0:
            energy = energy.sum()
        
        # Compute forces as negative gradient
        forces = -torch.autograd.grad(
            outputs=energy,
            inputs=positions,
            retain_graph=True,
            create_graph=False
        )[0]
    
    return energy, forces, positions


def backprop_step(energy, positions, loss_fn=None):
    """
    Perform backpropagation to compute gradients.
    
    Args:
        energy: Energy tensor from forward pass (requires_grad=True)
        positions: Position tensor from forward pass (requires_grad=True)
        loss_fn: Optional loss function to apply to energy before backprop.
                 If None, backprop directly on energy.
    
    Returns:
        grad_positions: Gradient of loss w.r.t. positions (dL/dr)
    """
    # Apply loss function if provided
    if loss_fn is not None:
        loss = loss_fn(energy)
    else:
        loss = energy
    
    # Perform backpropagation
    loss.backward(retain_graph=True)
    
    # Get gradients w.r.t. positions
    grad_positions = positions.grad
    
    return grad_positions


def save_perturbation(atoms_original, atoms_perturbed, epsilon, energy_original, 
                     energy_perturbed, gradients, save_path, metadata=None):
    """
    Save perturbation data to a file for later analysis.
    
    Args:
        atoms_original: Original ASE Atoms object
        atoms_perturbed: Perturbed ASE Atoms object
        epsilon: Step size used for perturbation
        energy_original: Original energy (eV)
        energy_perturbed: Perturbed energy (eV)
        gradients: Gradients used for perturbation (torch tensor or numpy array)
        save_path: Path to save the data (will save as .npz file)
        metadata: Optional dictionary with additional metadata
    """
    import numpy as np
    from datetime import datetime
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert gradients to numpy if needed
    if torch.is_tensor(gradients):
        gradients = gradients.cpu().numpy()
    
    # Prepare data dictionary
    data = {
        'positions_original': atoms_original.get_positions(),
        'positions_perturbed': atoms_perturbed.get_positions(),
        'displacement': atoms_perturbed.get_positions() - atoms_original.get_positions(),
        'chemical_symbols': np.array(atoms_original.get_chemical_symbols(), dtype='U2'),
        'cell': atoms_original.get_cell().array,
        'pbc': atoms_original.get_pbc(),
        'epsilon': epsilon,
        'energy_original': energy_original,
        'energy_perturbed': energy_perturbed,
        'energy_change': energy_perturbed - energy_original,
        'gradients': gradients,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add metadata if provided
    if metadata is not None:
        for key, value in metadata.items():
            # Convert to numpy-compatible types
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            data[f'meta_{key}'] = value
    
    # Save to npz file
    np.savez_compressed(save_path, **data)
    print(f"Saved perturbation data to {save_path}")
    
    return save_path


def load_perturbation(load_path):
    """
    Load perturbation data from a saved file.
    
    Args:
        load_path: Path to the saved .npz file
    
    Returns:
        dict: Dictionary containing all saved perturbation data with keys:
            - atoms_original: Reconstructed original ASE Atoms object
            - atoms_perturbed: Reconstructed perturbed ASE Atoms object
            - displacement: Position displacement array
            - epsilon: Perturbation step size
            - energy_original: Original energy
            - energy_perturbed: Perturbed energy
            - energy_change: Energy change
            - gradients: Gradient array
            - timestamp: Save timestamp
            - metadata: Dictionary of any additional metadata
    """
    import numpy as np
    from ase import Atoms
    
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"File not found: {load_path}")
    
    # Load data
    data = np.load(load_path, allow_pickle=True)
    
    # Reconstruct original atoms
    atoms_original = Atoms(
        symbols=data['chemical_symbols'].tolist(),
        positions=data['positions_original'],
        cell=data['cell'],
        pbc=data['pbc']
    )
    
    # Reconstruct perturbed atoms
    atoms_perturbed = Atoms(
        symbols=data['chemical_symbols'].tolist(),
        positions=data['positions_perturbed'],
        cell=data['cell'],
        pbc=data['pbc']
    )
    
    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            metadata[key[5:]] = data[key].item() if data[key].ndim == 0 else data[key]
    
    result = {
        'atoms_original': atoms_original,
        'atoms_perturbed': atoms_perturbed,
        'displacement': data['displacement'],
        'epsilon': float(data['epsilon']),
        'energy_original': float(data['energy_original']),
        'energy_perturbed': float(data['energy_perturbed']),
        'energy_change': float(data['energy_change']),
        'gradients': data['gradients'],
        'timestamp': str(data['timestamp']),
        'metadata': metadata
    }
    
    print(f"Loaded perturbation data from {load_path}")
    print(f"  Timestamp: {result['timestamp']}")
    print(f"  Atoms: {len(atoms_original)}")
    print(f"  Epsilon: {result['epsilon']:.4f} Å")
    print(f"  Energy change: {result['energy_change']:+.4f} eV")
    
    return result


def adversarial_attack_step(atoms, device="cpu", epsilon=0.01, target_energy=None):
    """
    Perform one step of adversarial attack on atomic positions.
    
    Args:
        atoms: ASE Atoms object with MACE calculator attached
        device: Device to run on
        epsilon: Step size for perturbation
        target_energy: Optional target energy for attack. If None, maximize energy.
    
    Returns:
        perturbed_atoms: Atoms with perturbed positions
        energy: Original energy
        perturbed_energy: Energy after perturbation
        grad_positions: Gradients used for perturbation
    """
    # Forward pass with gradients
    energy, forces, positions = forward_pass_with_gradients(atoms, device=device)
    
    # Define loss (maximize or target energy)
    if target_energy is not None:
        # Try to reach target energy
        loss = (energy - target_energy) ** 2
    else:
        # Maximize energy (for adversarial attack)
        loss = -energy
    
    # Backprop to get gradients w.r.t. positions
    loss.backward()
    grad_positions = positions.grad
    
    # Compute perturbation (gradient ascent to maximize energy)
    perturbation = epsilon * torch.sign(grad_positions)
    
    # Apply perturbation
    perturbed_positions = positions.detach() + perturbation
    
    # Create new atoms with perturbed positions
    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(perturbed_positions.cpu().numpy())
    
    # Set calculator on perturbed atoms
    perturbed_atoms.calc = atoms.calc
    
    # Compute energy of perturbed structure
    perturbed_energy_np = perturbed_atoms.get_potential_energy()
    
    return perturbed_atoms, energy.item(), perturbed_energy_np, grad_positions.detach()




# %%

def visualize_perturbation(atoms_before, atoms_after, epsilon=0.01, outdir=None):
    """
    Visualize the difference between original and perturbed atomic structures.
    
    Args:
        atoms_before: Original ASE Atoms object
        atoms_after: Perturbed ASE Atoms object
        epsilon: Perturbation magnitude used
        outdir: Optional output directory to save plots
    
    Returns:
        fig: Matplotlib figure
    """
    import numpy as np
    
    # Get positions
    pos_before = atoms_before.get_positions()
    pos_after = atoms_after.get_positions()
    
    # Calculate displacement
    displacement = pos_after - pos_before
    displacement_mag = np.linalg.norm(displacement, axis=1)
    
    # Create figure - large 3D plot with statistics on the side
    fig = plt.figure(figsize=(18, 10))
    
    # Main 3D plot showing all atoms and displacements
    ax_3d = plt.subplot(1, 2, 1, projection='3d')
    
    # Get atom symbols for coloring
    symbols = atoms_before.get_chemical_symbols()
    unique_symbols = list(set(symbols))
    
    # Create color map for different elements
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('tab10')
    symbol_colors = {sym: cmap(i / len(unique_symbols)) for i, sym in enumerate(unique_symbols)}
    colors = [symbol_colors[sym] for sym in symbols]
    
    # Plot all original atoms (blue/transparent)
    ax_3d.scatter(pos_before[:, 0], pos_before[:, 1], pos_before[:, 2], 
                  c=colors, s=100, alpha=0.3, marker='o', 
                  edgecolors='blue', linewidth=1, label='Original')
    
    # Plot all perturbed atoms (red/solid)
    ax_3d.scatter(pos_after[:, 0], pos_after[:, 1], pos_after[:, 2], 
                  c=colors, s=100, alpha=0.8, marker='o', 
                  edgecolors='red', linewidth=1.5, label='Perturbed')
    
    # Draw displacement vectors for all atoms
    # Scale arrows for visibility
    arrow_scale = 1.0 if displacement_mag.max() < 1.0 else 0.5
    ax_3d.quiver(pos_before[:, 0], pos_before[:, 1], pos_before[:, 2],
                 displacement[:, 0], displacement[:, 1], displacement[:, 2],
                 length=arrow_scale, normalize=False, alpha=0.6, 
                 arrow_length_ratio=0.2, color='black', linewidth=0.5)
    
    # Set labels and title
    ax_3d.set_xlabel('X (Å)', fontsize=12)
    ax_3d.set_ylabel('Y (Å)', fontsize=12)
    ax_3d.set_zlabel('Z (Å)', fontsize=12)
    ax_3d.set_title(f'Atomic Displacement Visualization (ε={epsilon} Å)', fontsize=14, fontweight='bold')
    
    # Add legend for atom types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=symbol_colors[sym], 
                                  markersize=10, label=sym, markeredgecolor='black')
                      for sym in unique_symbols]
    legend_elements.extend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, label='Original', markeredgecolor='blue', markeredgewidth=2, alpha=0.3),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, label='Perturbed', markeredgecolor='red', markeredgewidth=2)
    ])
    ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Make the plot look better
    ax_3d.grid(True, alpha=0.3)
    
    # Right side - statistics panel
    ax_stats = plt.subplot(1, 2, 2)
    ax_stats.axis('off')
    
    # Calculate statistics
    try:
        energy_before = atoms_before.get_potential_energy()
        energy_after = atoms_after.get_potential_energy()
        delta_E = energy_after - energy_before
        energy_text = f"""Energy:
  Original:  {energy_before:.4f} eV
  Perturbed: {energy_after:.4f} eV
  ΔE:        {delta_E:+.4f} eV
"""
    except Exception as e:
        energy_text = f"Energy: Calculation failed\n"
    
    # Per-element statistics
    element_stats = "\nPer-Element Displacement:\n"
    for sym in unique_symbols:
        mask = np.array([s == sym for s in symbols])
        mean_disp = displacement_mag[mask].mean()
        max_disp = displacement_mag[mask].max()
        element_stats += f"  {sym:>2s}: mean={mean_disp:.4f} Å, max={max_disp:.4f} Å\n"
    
    stats_text = f"""PERTURBATION ANALYSIS
{'='*40}

System:
  Total atoms:     {len(atoms_before)}
  Element types:   {', '.join(unique_symbols)}
  Epsilon:         {epsilon} Å

{energy_text}
Displacement Statistics:
  Mean:      {displacement_mag.mean():.4f} Å
  Median:    {np.median(displacement_mag):.4f} Å
  Max:       {displacement_mag.max():.4f} Å
  Min:       {displacement_mag.min():.4f} Å
  Std Dev:   {displacement_mag.std():.4f} Å
  
Component Mean Displacement:
  X: {displacement[:, 0].mean():+.4f} ± {displacement[:, 0].std():.4f} Å
  Y: {displacement[:, 1].mean():+.4f} ± {displacement[:, 1].std():.4f} Å
  Z: {displacement[:, 2].mean():+.4f} ± {displacement[:, 2].std():.4f} Å
{element_stats}
Displacement Distribution:
  < 0.01 Å:  {np.sum(displacement_mag < 0.01)} atoms ({100*np.sum(displacement_mag < 0.01)/len(atoms_before):.1f}%)
  < 0.05 Å:  {np.sum(displacement_mag < 0.05)} atoms ({100*np.sum(displacement_mag < 0.05)/len(atoms_before):.1f}%)
  < 0.10 Å:  {np.sum(displacement_mag < 0.10)} atoms ({100*np.sum(displacement_mag < 0.10)/len(atoms_before):.1f}%)
  ≥ 0.10 Å:  {np.sum(displacement_mag >= 0.10)} atoms ({100*np.sum(displacement_mag >= 0.10)/len(atoms_before):.1f}%)
"""
    
    ax_stats.text(0.05, 0.98, stats_text, transform=ax_stats.transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2, pad=1))
    
    # Adjust layout manually to avoid tight_layout warning with mixed axes
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
    
    # Save figure - use the perturbed atoms file path if available
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Try to determine filename from atoms_after if it has a file path
        if hasattr(atoms_after, 'info') and 'filename' in atoms_after.info:
            base_name = Path(atoms_after.info['filename']).stem
        else:
            base_name = 'perturbation_analysis'
        
        save_path = outdir / f'{base_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig