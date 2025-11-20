Examples
========

This page provides practical examples of using ``mlff_attack`` for various adversarial attack scenarios.

Basic FGSM Attack
-----------------

The simplest way to perform an FGSM attack using the high-level API:

.. code-block:: python

   from ase.io import read
   from mlff_attack.attacks import make_attack
   
   # Load your structure
   atoms = read('structure.cif')
   
   # Perform FGSM attack
   output_path, perturbed_atoms, attack_details = make_attack(
       model_path='mace-mpa-0-medium.model',
       device='cuda',
       atoms=atoms,
       epsilon=0.1,
       target_energy=None,
       output_cif='perturbed_structure.cif',
       attack_type='fgsm'
   )
   
   print(f"Perturbed structure saved to: {output_path}")

PGD Attack with Multiple Steps
-------------------------------

For iterative attacks with stronger perturbations:

.. code-block:: python

   from ase.io import read
   from mlff_attack.attacks import make_attack
   
   atoms = read('structure.cif')
   
   # Perform iterative FGSM attack (I-FGSM) with 10 iterations
   output_path, perturbed_atoms, attack_details = make_attack(
       model_path='mace-mpa-0-medium.model',
       device='cuda',
       atoms=atoms,
       epsilon=0.05,
       target_energy=None,
       output_cif='perturbed_ifgsm.cif',
       attack_type='fgsm',
       n_steps=10,
       clip=False
   )

Using the Class-Based API
--------------------------

For more control over the attack process:

.. code-block:: python

   from ase.io import read
   from mlff_attack.grad_based.fgsm import FGSM_MACE
   from mlff_attack.relaxation import setup_calculator
   
   # Load structure and setup calculator
   atoms = read('structure.cif')
   atoms = setup_calculator(atoms, 'mace-mpa-0-medium.model', device='cuda')
   
   # Create FGSM attack instance
   fgsm_attack = FGSM_MACE(
       model=atoms.calc,
       epsilon=0.1,
       device='cuda',
       track_history=True
   )
   
   # Execute attack (returns perturbed atoms object)
   perturbed_atoms = fgsm_attack.attack(atoms, n_steps=1, clip=True)
   
   # Access attack history
   print("Energies:", fgsm_attack.attack_history['energies'])
   print("Max forces:", fgsm_attack.attack_history['max_forces'])

Targeted Energy Attack
----------------------

Perform an attack aiming for a specific energy value:

.. code-block:: python

   from ase.io import read
   from mlff_attack.attacks import make_attack
   from mlff_attack.relaxation import setup_calculator
   
   atoms = read('structure.cif')
   
   # Get initial energy
   atoms = setup_calculator(atoms, 'mace-mpa-0-medium.model', device='cuda')
   initial_energy = atoms.get_potential_energy()
   target_energy = initial_energy + 1.0  # Target 1 eV higher
   
   # Perform targeted attack
   output_path, perturbed_atoms, attack_details = make_attack(
       model_path='mace-mpa-0-medium.model',
       device='cuda',
       atoms=atoms,
       epsilon=0.01,
       target_energy=target_energy,
       output_cif='targeted_attack.cif',
       attack_type='fgsm',
       n_steps=100
   )

Tracking Attack Progress
-------------------------

Monitor the attack progress with detailed history tracking:

.. code-block:: python

   from ase.io import read
   from mlff_attack.grad_based.fgsm import FGSM_MACE
   from mlff_attack.relaxation import setup_calculator
   import matplotlib.pyplot as plt
   
   atoms = read('structure.cif')
   atoms = setup_calculator(atoms, 'mace-mpa-0-medium.model', device='cuda')
   
   # Create attack with history tracking
   attack = FGSM_MACE(
       model=atoms.calc,
       epsilon=0.05,
       device='cuda',
       track_history=True
   )
   
   # Execute iterative attack
   perturbed_atoms = attack.attack(atoms, n_steps=20, clip=False)
   
   # Get attack summary
   summary = attack.get_attack_summary()
   print(f"Initial Energy: {summary['initial_energy']:.3f} eV")
   print(f"Final Energy: {summary['final_energy']:.3f} eV")
   print(f"Energy Change: {summary['energy_change']:.3f} eV")
   print(f"Max Displacement: {summary['max_displacement']:.3f} Å")
   
   # Plot energy evolution
   plt.figure(figsize=(10, 6))
   plt.plot(attack.attack_history['energies'])
   plt.xlabel('Attack Step')
   plt.ylabel('Energy (eV)')
   plt.title('Energy Evolution During Attack')
   plt.savefig('attack_energy.png')

Visualizing Perturbations
--------------------------

Visualize the perturbations applied to the structure:

.. code-block:: python

   from ase.io import read
   from mlff_attack.attacks import make_attack, visualize_perturbation
   
   # Perform attack
   atoms = read('structure.cif')
   output_path, perturbed_atoms, attack_details = make_attack(
       model_path='mace-mpa-0-medium.model',
       device='cuda',
       atoms=atoms,
       epsilon=0.1,
       attack_type='fgsm',
       output_cif='perturbed.cif'
   )
   
   # Visualize the perturbation
   fig = visualize_perturbation(
       atoms_before=atoms,
       atoms_after=perturbed_atoms,
       epsilon=0.1,
       outdir='./'
   )

Saving and Loading Perturbations
---------------------------------

Save perturbations for later analysis or reuse:

.. code-block:: python

   from ase.io import read
   from mlff_attack.attacks import make_attack, save_perturbation, load_perturbation
   
   # Perform attack
   atoms = read('structure.cif')
   output_path, perturbed_atoms, attack_details = make_attack(
       model_path='mace-mpa-0-medium.model',
       device='cuda',
       atoms=atoms,
       epsilon=0.1,
       attack_type='fgsm',
       output_cif='perturbed.cif'
   )
   
   # Save perturbation with full metadata
   save_perturbation(
       atoms_original=atoms,
       atoms_perturbed=perturbed_atoms,
       epsilon=0.1,
       energy_original=atoms.get_potential_energy(),
       energy_perturbed=perturbed_atoms.get_potential_energy(),
       gradients=attack_details['gradients'][-1],
       save_path='perturbation_data.npz'
   )
   
   # Later, load perturbation data
   loaded_data = load_perturbation('perturbation_data.npz')
   print(f"Energy change: {loaded_data['energy_change']:.4f} eV")
   print(f"Displacement: {loaded_data['displacement']}")

Complete Workflow Example
--------------------------

A complete workflow from attack to relaxation analysis:

.. code-block:: python

   from ase.io import read, write
   from mlff_attack.attacks import make_attack
   from mlff_attack.relaxation import run_relaxation, load_structure, setup_calculator
   from mlff_attack.visualization import create_visualization
   
   # 1. Load original structure
   original_atoms = read('structure.cif')
   
   # 2. Perform adversarial attack
   output_cif, perturbed_atoms, attack_details = make_attack(
       model_path='mace-mpa-0-medium.model',
       device='cuda',
       atoms=original_atoms,
       epsilon=0.1,
       attack_type='pgd',
       n_steps=10,
       output_cif='perturbed.cif'
   )
   
   # 3. Relax original structure
   original_atoms = setup_calculator(original_atoms, 'mace-mpa-0-medium.model')
   original_results = run_relaxation(
       atoms=original_atoms,
       fmax=0.01,
       max_steps=300,
       optimizer='LBFGS',
       trajectory_file='original_relaxed.traj'
   )
   
   # 4. Relax perturbed structure
   perturbed_atoms = setup_calculator(perturbed_atoms, 'mace-mpa-0-medium.model')
   perturbed_results = run_relaxation(
       atoms=perturbed_atoms,
       fmax=0.01,
       max_steps=300,
       optimizer='LBFGS',
       trajectory_file='perturbed_relaxed.traj'
   )
   
   # 5. Visualize and compare trajectories
   create_visualization(
       'original_relaxed.traj',
       output_dir='analysis/',
       show_plots=False
   )
   
   create_visualization(
       'perturbed_relaxed.traj',
       output_dir='analysis/',
       show_plots=False
   )
   
   # 6. Compare final energies
   print(f"Original final energy: {original_results['final_energy']:.3f} eV")
   print(f"Perturbed final energy: {perturbed_results['final_energy']:.3f} eV")
   print(f"Energy difference: {abs(perturbed_results['final_energy'] - original_results['final_energy']):.3f} eV")

Batch Processing Multiple Structures
-------------------------------------

Process multiple structures in a batch:

.. code-block:: python

   from pathlib import Path
   from ase.io import read
   from mlff_attack.attacks import make_attack
   
   # Directory containing CIF files
   input_dir = Path('initial_cifs/')
   output_dir = Path('perturbed_cifs/')
   output_dir.mkdir(exist_ok=True)
   
   # Process each structure
   for cif_file in input_dir.glob('*.cif'):
       print(f"Processing {cif_file.name}...")
       
       atoms = read(cif_file)
       output_path, perturbed_atoms, attack_details = make_attack(
           model_path='mace-mpa-0-medium.model',
           device='cuda',
           atoms=atoms,
           epsilon=0.1,
           attack_type='fgsm',
           output_cif=str(output_dir / f'perturbed_{cif_file.name}')
       )
       
       print(f"  Saved to {output_path}")

Using CLI Commands
------------------

Examples using the command-line interface:

**FGSM Attack:**

.. code-block:: bash

   make-attack --type fgsm --input structure.cif --model mace-model.model \\
               --outdir perturbed.cif --epsilon 0.1

**PGD Attack:**

.. code-block:: bash

   make-attack --type pgd --input structure.cif --model mace-model.model \\
               --outdir perturbed.cif --epsilon 0.05 --n-steps 10

**MACE Relaxation:**

.. code-block:: bash

   mace-calc-single --input structure.cif --model mace-model.model \\
                    --outdir results/ --fmax 0.01 --max-steps 300

**Trajectory Visualization:**

.. code-block:: bash

   visualize-traj --traj results/relaxed.traj --outdir results/ --show --format png
