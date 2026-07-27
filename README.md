# mlff_attack

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://trustworthy-ai-tools-for-science.github.io/mlff_attack/)

Attacks against MLFF Models - A Python package for testing and analyzing Machine Learning Force Fields models through adversarial attacks.

### Attacks Implemented

| Attack Name | Paper | 
| --- | --- |
| Fast Gradient Sign Method (FGSM) | [link](https://arxiv.org/abs/1412.6572) |
| Iterative Fast Gradient Sign Method (I-FGSM) | [link](https://arxiv.org/abs/1607.02533) | 
| Projected Gradient Descent (PGD) | [link](https://arxiv.org/abs/1706.06083) |

### Models Supported

| Model Name | Paper |
| --- | --- |
| Message Passing Atomic Cluster Expansion (MACE) | [link](https://arxiv.org/abs/2206.07697) |
| Multi-Head Message Passing Atomic Cluster Expansion (MACE-MH) | [link](https://arxiv.org/pdf/2510.25380) |
| Universal Model for Atoms (UMA) | [link](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/) |
| Crystal Hamiltonian Graph Neural Network (CHGNet) | [link](https://chgnet.lbl.gov/) |
| Moment Tensor Potential (MTP) | [link](https://github.com/gitliwq/LiCOHPF_database_1) |

## Installation

### Install from source (development mode)

```bash
# Clone the repository
git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
cd mlff_attack

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install targeted MLFF support

Each MLFF model should be installed in separate Python environments because their dependencies can conflict:
* MACE support
* UMA support must be in Python <= 3.12 and requires access through Hugging Face
* CHGNet support must be in Python <= 3.12
* MTP support uses the external MLIP-3 `mlp` executable

```bash
git switch mlff-<model>
bash setup.sh
```

> **Reminder to activate environment:** Before running calculations, make sure to activate the correct MLFF environment for the targeted model.

## Usage

### Running calculations

After installation, you can use the `calc-single` commands for MLFF calculations:

```bash
# MACE
calc-single --input <structure>.cif --model mace-<model>.model --outdir <output_directory>

# MACE-MH
calc-single --input <structure>.cif --model mace-mh-<model>.model --outdir <output_directory> --mace-head <head-name>

# UMA
calc-single --input <structure>.cif --model uma-<variant> --outdir <output_directory> --uma-task <task-name> --uma-charge <charge> --uma-spin <spin>

# CHGNet
calc-single --input <structure>.cif --model chgnet-<type> --outdir <output_directory>

# MTP
calc-single --input <structure>.xyz --model <type>.almtp --outdir <output_directory>
```

#### Command-line options

- `--input`: Input CIF file (required).
- `--model`: MLFF model filename (required).
- `--outdir`: Output directory (required).
- `--device`: Device to use (cuda or cpu, default: cpu).
- `--seed`: Random seed for MACE/UMA calculator setup (default: `42`).
- `--dtype`: Data type for calculations ("float32" or "float64") (default: float64).
- `--fmax`: Force convergence criterion in eV/Å (default: 0.01).
- `--max-steps`: Maximum relaxation steps (default: 300).
- `--optimizer`: ASE optimizer to use (BFGS or LBFGS, default: LBFGS).
- `--mace-head`: MACE-MH head, only for MACE-MH (default: `omat_pbe`).
- `--uma-task`: UMA task/domain, only for UMA (default: `omat`).
- `--uma-charge`: Molecular charge, only for UMA.
- `--uma-spin`: Spin multiplicity, only for UMA.

### Visualizing trajectories

After running a calculation, you can visualize the relaxation trajectory:

```bash
visualize-traj --traj <output_directory>/relaxed.traj --outdir <output_directory>
```

This will generate a comprehensive plot showing:
- Energy evolution during relaxation
- Maximum force convergence
- Volume changes
- Noise spectrum of maximum forces
- Summary statistics

#### Visualization options

- `--traj`: Path to trajectory file (.traj) (required).
- `--outdir`: Output directory for plots (default: current directory).
- `--show`: Show plots interactively.
- `--format`: Output format for plots (png, pdf, or svg, default: png).

### Running Attacks

The `make-attack` command allows you to perform adversarial attacks on MLFF models. Supported attack types include FGSM and PGD.

```bash
make-attack --type <attack_type> --input <input_file> --model <model_file> --outdir <output_directory>
```

#### Command-line options

- `--type`: Type of attack to perform, either `fgsm` or `pgd` (required).
- `--input`: Path to the input structure file (CIF format) (required).
- `--model`: MACE model path, UMA model name, CHGNet model name (required).
- `--device`: Device to use for computations (cuda or cpu, default: cpu).
- `--dtype`: Data type for calculations (`float32` or `float64`, default: `float64`).
- `--seed`: Random seed for MACE/UMA calculator setup and PGD random start (default: `42`).
- `--outdir`: Directory to save the results (required).
- `--visualize`: Generate perturbation visualization plot (default: enabled).
- `--no-visualize`: Skip perturbation visualization plot generation.
- `--epsilon`: Perturbation step size for the attack (default: 0.05).
- `--alpha`: PGD step size only valid with `--type pgd` (default: epsilon / n_steps).
- `--n-steps`: Number of attack iterations (default: 1 for FGSM, >1 for PGD).
- `--target-energy`: Target energy in eV (default: maximize the predicted energy).
- `--clip`: Whether to clip perturbations to the epsilon bound. Pass `true` or `false`. If omitted, FGSM defaults to `false`; PGD defaults to `true` and rejects `false`.
- `--mace-head`: MACE-MH head, only for MACE-MH (default: `omat_pbe`).
- `--uma-task`: UMA task/domain, only for UMA (default: `omat`).
- `--uma-charge`: Molecular charge, only for UMA.
- `--uma-spin`: Spin multiplicity, only for UMA.

#### Example usage


```bash
# Perform an FGSM attack
make-attack --type fgsm --input structure.cif --model mace-model.model --outdir output_perturbed/ --epsilon 0.1

# Perform an I-FGSM attack
make-attack --type fgsm --input structure.cif --model mace-model.model --outdir output_perturbed/ --epsilon 0.1 --n-steps 10

# Perform a PGD attack
make-attack --type pgd --input structure.cif --model mace-model.model --outdir output_perturbed/ --epsilon 0.1 --alpha 0.01 --n-steps 10

```

### Example workflow

```bash
# Run MACE relaxation
calc-single --input structure.cif --model mace-model.model --outdir output/

# Visualize the results
visualize-traj --traj output/relaxed.traj --outdir output/ --show

# Generate an attack
make-attack --type fgsm --input structure.cif --model mace-model.model --outdir output_perturbed/

# Run MACE relaxation on perturbed structure
calc-single --input structure_perturbed.cif --model mace-model.model --outdir output_perturbed/

# Visualize the results of the attack
visualize-traj --traj output_perturbed/relaxed.traj --outdir output_perturbed/

```

## Requirements

### Base package

- Python >= 3.10
- ase >= 3.22.0
- torch >= 2.0.0
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- pandas
- seaborn
- spglib
- mp_api
- ipywidgets
- jupyterlab

### MACE support

- mace-torch >= 0.3.0
- `.model` file

### UMA support

- fairchem-core >= 2.0.0
- huggingface_hub
- `.pt` file

### CHGNet support

- chgnet >= 0.4.2

### MTP support

- MLIP-3 `mlp` executable
- WSL Ubuntu or Linux recommended
- Conda environment recommended
- CPU and float64 only
- `.almtp` and `.almtp.elements` file

## License

See LICENSE file for details.

## Citation
If you use this library in your research, please consider citing:

```bibtex
@software{mlff_attack,
  title = {MLFF Attack: A library for attacking MLFF models},
  author = {Ashley S. Dale AND Hao Wan},
  url = {https://github.com/Trustworthy-AI-Tools-for-Science/mlff_attack},
  year = {2025}
}
```
