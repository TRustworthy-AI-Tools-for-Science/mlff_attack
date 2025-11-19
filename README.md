# mlff_attack

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://trustworthy-ai-tools-for-science.github.io/mlff_attack/)



Attacks against MLFF Models - A Python package for testing and analyzing Machine Learning Force Fields models through adversarial attacks.

## Installation

### From source (development mode)

```bash
# Clone the repository
git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
cd mlff_attack

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Running MACE calculations

After installation, you can use the `mace-calc-single` command:

```bash
mace-calc-single --input structure.cif --model mace-model.model --outdir results/
```

#### Command-line options

- `--input`: Input CIF file (required)
- `--model`: Path to MACE model file (.model) (required)
- `--outdir`: Output directory (required)
- `--device`: Device to use (cuda or cpu, default: cuda)
- `--fmax`: Force convergence criterion in eV/Å (default: 0.01)
- `--max-steps`: Maximum relaxation steps (default: 300)
- `--optimizer`: ASE optimizer to use (BFGS or LBFGS, default: LBFGS)

### Visualizing trajectories

After running a calculation, you can visualize the relaxation trajectory:

```bash
visualize-traj --traj results/relaxed.traj --outdir results/
```

This will generate a comprehensive plot showing:
- Energy evolution during relaxation
- Maximum force convergence
- Volume changes
- Summary statistics

#### Visualization options

- `--traj`: Path to trajectory file (.traj) (required)
- `--outdir`: Output directory for plots (default: current directory)
- `--show`: Show plots interactively
- `--format`: Output format for plots (png, pdf, or svg, default: png)

### Running Attacks

The `make-attack` command allows you to perform adversarial attacks on MLFF models. Supported attack types include FGSM and PGD.

```bash
make-attack --type <attack_type> --input <input_file> --model <model_file> --outdir <output_directory>
```

#### Command-line options

- `--type`: Type of attack to perform (e.g., `fgsm`, `pgd`) (required).
- `--input`: Path to the input structure file (CIF format) (required).
- `--model`: Path to the MACE model file (.model) (required).
- `--outdir`: Directory to save the results (required).
- `--epsilon`: Perturbation step size for the attack (default: 0.01).
- `--n-steps`: Number of attack iterations (default: 1 for FGSM, >1 for PGD).
- `--clip`: Whether to clip perturbations to the epsilon bound (default: True).
- `--device`: Device to use for computations (cuda or cpu, default: cuda).

#### Example usage

```bash
# Perform an FGSM attack
make-attack --type fgsm --input structure.cif --model mace-model.model --outdir perturbed_structure.cif --epsilon 0.1

# Perform a PGD attack with 10 steps
make-attack --type pgd --input structure.cif --model mace-model.model --outdir perturbed_structure.cif --epsilon 0.1 --n-steps 10
```

### Example workflow

```bash
# Run MACE relaxation
mace-calc-single --input structure.cif --model mace-model.model --outdir output/

# Visualize the results
visualize-traj --traj output/relaxed.traj --outdir output/ --show

# Generate an attack
make-attack --type fgsm --input structure.cif --outdir perturbed_structure.cif

# Run MACE relaxation on perturbed structure
mace-calc-single --input perturbed_structure.cif --model mace-model.model --outdir output_perturbed/

# Visualize the results of the attack
visualize-traj --traj output_perturbed/relaxed.traj --outdir output_perturbed/

```

## Requirements

- Python >= 3.10
- ase >= 3.22.0
- mace-torch >= 0.3.0
- torch >= 2.0.0

## License

See LICENSE file for details.
