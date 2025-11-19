# FGSM Attack Class Documentation

## Overview

The `FGSM_MACE` class provides a clean, object-oriented interface for performing Fast Gradient Sign Method attacks on MACE force field models. It extends the abstract `MLFFAttack` base class.

## Architecture

```
MLFFAttack (Abstract Base Class)
    ↓
FGSM_MACE (MACE-specific implementation)
```

### Key Features

1. **Automatic gradient computation** through MACE models
2. **History tracking** for analysis (energies, forces, perturbations)
3. **Perturbation clipping** to enforce epsilon bounds
4. **Flexible attack modes**: maximize energy or target specific energy
5. **Save/load functionality** for reproducibility
6. **Built-in statistics** and visualization support

## Quick Start

### Basic FGSM Attack (Single Step)

```python
from ase.io import read
from mlff_attack.relaxation import setup_calculator
from mlff_attack.grad_based.fgsm import FGSM_MACE

# Load structure and setup calculator
atoms = read("structure.cif")
atoms = setup_calculator(atoms, "mace-model.model", device="cpu")

# Create attack
attack = FGSM_MACE(
    model=atoms.calc,
    epsilon=0.05,  # Max perturbation in Angstroms
    device="cpu",
    track_history=True
)

# Execute attack
perturbed_atoms = attack.attack(atoms, n_steps=1, clip=True)

# Get results
stats = attack.get_perturbation_stats()
summary = attack.get_attack_summary()
```

### Iterative FGSM (I-FGSM)

```python
# Smaller epsilon, multiple steps
attack = FGSM_MACE(
    model=atoms.calc,
    epsilon=0.01,
    device="cpu"
)

# 10 iterations with clipping
perturbed_atoms = attack.attack(atoms, n_steps=10, clip=True)

# View energy progression
for i, energy in enumerate(attack.attack_history['energies']):
    print(f"Step {i+1}: {energy:.4f} eV")
```

### Targeted Attack

```python
# Try to reach specific energy
target_energy = original_energy + 2.0

attack = FGSM_MACE(
    model=atoms.calc,
    epsilon=0.02,
    device="cpu",
    target_energy=target_energy  # Minimize (E - target)^2
)

perturbed_atoms = attack.attack(atoms, n_steps=10, clip=True)
```

## Class Reference

### `FGSM_MACE`

**Constructor Parameters:**
- `model`: MACE calculator attached to atoms
- `epsilon`: Perturbation step size (Angstroms)
- `device`: PyTorch device ('cpu' or 'cuda')
- `track_history`: Enable history tracking (default: True)
- `target_energy`: Optional target energy for attack (default: None = maximize)

**Key Methods:**

#### `attack(atoms, n_steps=1, clip=True)`
Execute the attack.
- `atoms`: Input structure with MACE calculator
- `n_steps`: Number of iterations (1 for FGSM, >1 for I-FGSM)
- `clip`: Enforce epsilon bound
- Returns: Perturbed atoms object

#### `compute_gradient(atoms, loss_fn=None)`
Compute gradient of loss w.r.t. positions.
- `atoms`: Current structure
- `loss_fn`: Optional custom loss function
- Returns: Gradient array (n_atoms, 3)

#### `attack_step(atoms, step=0)`
Perform one attack iteration.
- `atoms`: Current structure
- `step`: Iteration number
- Returns: Updated atoms

#### `save_perturbation(filepath, atoms_original=None, atoms_perturbed=None, include_metadata=True)`
Save perturbation data.
- `filepath`: Output path (.npz)
- `atoms_original`: Original structure (for saving chemical info)
- `atoms_perturbed`: Perturbed structure
- `include_metadata`: Include attack parameters

#### `load_perturbation(filepath)`
Load saved perturbation.
- `filepath`: Input path (.npz)
- Returns: Dictionary with all data

#### `get_perturbation_stats()`
Get displacement statistics.
- Returns: Dict with mean/max/std displacement, etc.

#### `get_attack_summary()`
Get complete attack summary.
- Returns: Dict with energies, forces, displacements

#### `reset()`
Reset attack state for reuse.

## Saved Data Format

When you call `save_perturbation()`, the following is saved to a `.npz` file:

### Always Saved:
- `positions_original`: Original atomic positions
- `positions_perturbed`: Perturbed positions
- `displacement`: Position change vectors

### With `atoms_original` provided:
- `chemical_symbols`: Element symbols
- `cell`: Unit cell
- `pbc`: Periodic boundary conditions

### With `include_metadata=True`:
- `epsilon`: Perturbation magnitude
- `device`: Device used
- `target_energy`: Target energy (if specified)
- `timestamp`: ISO format timestamp
- `energy_original`: Original energy
- `energy_perturbed`: Perturbed energy
- `energy_change`: Energy difference

### With `track_history=True`:
- `history_energies`: Energy at each step
- `history_max_forces`: Max force at each step
- `history_perturbations`: Perturbation vectors at each step
- `history_gradients`: Gradients at each step

## Advanced Usage

### Custom Loss Function

```python
def custom_loss(energy):
    """Custom loss: penalize deviations from -300 eV"""
    return (energy + 300.0) ** 2

attack = FGSM_MACE(model=calc, epsilon=0.05, device="cpu")
gradients = attack.compute_gradient(atoms, loss_fn=custom_loss)
```

### Manual Step-by-Step Control

```python
attack = FGSM_MACE(model=calc, epsilon=0.05, device="cpu")

# Initialize
attack._original_positions = atoms.get_positions().copy()

# Manually control iterations
perturbed = atoms.copy()
for i in range(10):
    perturbed = attack.attack_step(perturbed, step=i)
    
    # Check convergence
    if abs(perturbed.get_potential_energy() - target) < 0.1:
        break
    
    # Apply clipping
    attack._clip_perturbations(perturbed)

attack._perturbed_positions = perturbed.get_positions().copy()
```

### Analyze History

```python
attack = FGSM_MACE(model=calc, epsilon=0.01, device="cpu", track_history=True)
perturbed = attack.attack(atoms, n_steps=20)

# Plot energy evolution
import matplotlib.pyplot as plt
plt.plot(attack.attack_history['energies'])
plt.xlabel('Step')
plt.ylabel('Energy (eV)')
plt.show()

# Analyze gradient magnitudes
import numpy as np
grad_mags = [np.linalg.norm(g) for g in attack.attack_history['gradients']]
print(f"Mean gradient magnitude: {np.mean(grad_mags):.4f}")
```

## Comparison: Old vs New API

### Old (Functional API)
```python
# Old way - functional
from mlff_attack.attacks import adversarial_attack_step

perturbed, orig_e, pert_e, grads = adversarial_attack_step(
    atoms, device="cpu", epsilon=0.05, target_energy=None
)
```

### New (Object-Oriented API)
```python
# New way - OOP
from mlff_attack.grad_based.fgsm import FGSM_MACE

attack = FGSM_MACE(model=atoms.calc, epsilon=0.05, device="cpu")
perturbed = attack.attack(atoms, n_steps=1)
summary = attack.get_attack_summary()
```

**Benefits of New API:**
- Reusable attack objects
- Built-in history tracking
- Automatic perturbation clipping
- Cleaner code organization
- Easier to extend (just subclass `MLFFAttack`)

## Examples

See `/home/ashley/attack_mlff/mlff_attack/example_notebooks/example_fgsm_attack.py` for complete working examples:

1. **Basic FGSM** - Single-step attack
2. **Iterative FGSM** - Multi-step attack with clipping
3. **Targeted Attack** - Reach specific energy value
4. **Save/Load** - Persistence and analysis

Run with:
```bash
cd /home/ashley/attack_mlff
python mlff_attack/example_notebooks/example_fgsm_attack.py
```

## Future Extensions

The `MLFFAttack` base class makes it easy to implement other attack methods:

- **PGD** (Projected Gradient Descent)
- **C&W** (Carlini & Wagner)
- **Momentum-based attacks**
- **Model-specific optimizations** for ALIGNN, CHGNet, etc.

Each would just need to implement:
- `compute_gradient()`
- `attack_step()`

All other functionality (history, clipping, save/load, stats) is inherited!
