.. mlff_attack documentation master file, created by
   sphinx-quickstart on Tue Nov 18 23:09:54 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mlff_attack
===========

**mlff_attack** is a Python package for testing and analyzing Machine Learning Force Fields (MLFF) models through adversarial attacks.

Overview
--------

This package provides tools for:

- **Adversarial Attacks**: Implement various attack methods (FGSM, PGD) on MLFF models
- **MACE / UMA Integration**: Supports MACE, MACE-MH, and Meta UMA force field calculators
- **Structure Relaxation**: Tools for performing geometry optimization with MLFF models
- **Visualization**: Comprehensive plotting and analysis of attack results and trajectories

Key Features
------------

* **FGSM (Fast Gradient Sign Method)**: Single-step adversarial perturbations
* **PGD (Projected Gradient Descent)**: Iterative attack with bounded perturbations
* **Attack history tracking**: Monitor energy changes, forces, and perturbations during attacks
* **Structure analysis**: Visualize perturbation effects on atomic structures
* **Trajectory visualization**: Analyze relaxation convergence and energy landscapes

Installation
------------

``uv`` is the recommended package manager. MACE and UMA **require separate
environments** due to an incompatible ``e3nn`` version pin — see
:doc:`installation` for the full explanation and two-venv setup.

Quick start (MACE):

.. code-block:: bash

   git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
   cd mlff_attack
   uv sync --extra mace --extra dev

Quick start (UMA):

.. code-block:: bash

   uv sync --extra uma --extra dev
   hf auth login

See :doc:`installation` for pip instructions, the ``[notebooks]`` extra,
and details on running both backends side-by-side.

Quick Start
-----------

Performing an FGSM attack:

.. code-block:: python

   from ase.io import read
   from mlff_attack.attacks import make_attack
   
   # Load structure
   atoms = read('structure.cif')
   
   # Perform attack
   output_path, perturbed_atoms, attack_details = make_attack(
       model_path='mace-model.model',
       device='cuda',
       atoms=atoms,
       epsilon=0.1,
       target_energy=None,
       output_cif='perturbed.cif',
       attack_type='fgsm'
   )

Using the class-based API:

.. code-block:: python

   from mlff_attack.grad_based.fgsm import FGSM_ASE
   from mlff_attack.relaxation import setup_calculator
   
   # Setup calculator
   atoms = setup_calculator(atoms, 'mace-model.model', device='cuda')
   
   # Create attack
   attack = FGSM_ASE(
       model=atoms.calc,
       epsilon=0.1,
       device='cuda',
       track_history=True
   )
   
   # Execute attack
   perturbed_positions = attack.attack(atoms, n_steps=1, clip=True)
   
   # Get summary
   summary = attack.get_attack_summary()

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   attacks
   examples
   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
