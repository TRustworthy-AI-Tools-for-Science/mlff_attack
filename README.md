# mlff_attack

Attacks against MLFF Models - A Python package for testing and analyzing Machine Learning Force Fields models.

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

### Using pip

```bash
pip install mlff_attack
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

### Example workflow

```bash
# Run MACE relaxation
mace-calc-single --input structure.cif --model mace-model.model --outdir output/

# Visualize the results
visualize-traj --traj output/relaxed.traj --outdir output/ --show
```

## Requirements

- Python >= 3.8
- ase >= 3.22.0
- mace-torch >= 0.3.0
- torch >= 2.0.0

## License

See LICENSE file for details.
