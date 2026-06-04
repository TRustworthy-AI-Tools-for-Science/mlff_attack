Installation
============

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

**Python 3.10 – 3.12** is required. Python 3.12 is recommended.

`uv <https://docs.astral.sh/uv/>`_ is the recommended package manager because
it enforces the lockfile, handles the MACE / UMA environment conflict
automatically, and creates reproducible environments in seconds.

Install ``uv`` once if you don't already have it:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
   cd mlff_attack

MACE environment (recommended default)
---------------------------------------

.. code-block:: bash

   uv sync --extra mace --extra dev

This creates ``.venv`` in the repository root and installs:

- the base ``mlff_attack`` package
- ``mace-torch >= 0.3.0`` (with ``e3nn == 0.4.4``)
- development and testing tools

Verify the installation:

.. code-block:: bash

   uv run python -c "from mlff_attack.grad_based.fgsm import FGSM_ASE; print('MACE ready')"
   uv run make-attack --help

UMA environment
---------------

.. code-block:: bash

   uv sync --extra uma --extra dev
   hf auth login        # authenticate with Hugging Face to download UMA weights

Verify the installation:

.. code-block:: bash

   uv run python -c "import fairchem; print('UMA ready')"
   uv run calc-single --help

.. note::

   UMA models are gated on Hugging Face. You must run ``hf auth login`` and
   accept the model licence on the Hub before weights can be downloaded.

.. _mace-uma-conflict:

MACE / UMA environment conflict
---------------------------------

**MACE and UMA cannot be installed in the same Python environment.**

The root cause is an incompatible ``e3nn`` version requirement:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Package
     - e3nn required
     - Notes
   * - ``mace-torch >= 0.3``
     - ``== 0.4.4`` (exact pin)
     - Uses internal code-generation APIs removed in e3nn 0.5
   * - ``fairchem-core >= 1.0``
     - ``>= 0.5``
     - All 1.x and 2.x releases require the newer API

Attempting to override the pin installs successfully but raises a
``ValueError`` inside ``e3nn`` at model-load time, making MACE unusable.
The ``pyproject.toml`` declares these extras as ``conflicts`` in
``[tool.uv]`` so ``uv`` rejects the combination with a clear error
rather than producing a broken environment.

**Using both backends on the same machine** requires two separate virtual
environments:

.. code-block:: bash

   # MACE environment (default .venv)
   uv sync --extra mace --extra dev

   # UMA environment (separate directory)
   uv venv .venv-uma
   uv sync --extra uma --extra dev --venv .venv-uma

Activate whichever environment matches the model you are using:

.. code-block:: bash

   source .venv/bin/activate         # MACE / MACE-MH models
   source .venv-uma/bin/activate     # UMA models

The conflict is tracked upstream:

- mace-torch e3nn upgrade: https://github.com/ACEsuit/mace/issues
- fairchem-core e3nn lower-bound: https://github.com/facebookresearch/fairchem/issues

Once ``mace-torch`` releases a version that works with ``e3nn >= 0.5``,
remove the ``conflicts`` block from ``[tool.uv]`` in ``pyproject.toml``
and run ``uv sync --extra mace --extra uma`` to use a single environment.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Contents
   * - ``mace``
     - ``mace-torch >= 0.3`` — MACE and MACE-MH calculator support
   * - ``uma``
     - ``fairchem-core >= 2.0, < 3.0`` + ``huggingface_hub`` — UMA calculator support
   * - ``dev``
     - pytest, pytest-cov, black, flake8, mypy, pylint, sphinx, ipython
   * - ``notebooks``
     - jupyterlab, ipywidgets, ipykernel — for running the example notebooks

Install multiple compatible extras together:

.. code-block:: bash

   uv sync --extra mace --extra notebooks --extra dev

Installing with pip (existing conda / venv users)
--------------------------------------------------

If you manage your own environment and prefer ``pip``:

.. code-block:: bash

   # activate your environment first, then:
   pip install -e ".[mace,dev]"    # MACE environment
   # — or —
   pip install -e ".[uma,dev]"     # UMA environment (separate environment required)

.. warning::

   ``pip`` does not enforce the ``conflicts`` declaration in
   ``pyproject.toml``, so it will silently install both ``mace-torch``
   and ``fairchem-core`` without error. The resulting environment will
   load ``e3nn 0.6`` (satisfying fairchem-core) and MACE model loading
   will fail at runtime with a ``ValueError``.

Verifying the full test suite
------------------------------

.. code-block:: bash

   # MACE environment
   uv run pytest --no-header -q

   # UMA-specific tests only
   source .venv-uma/bin/activate
   pytest tests/test_uma_calc_single.py -v

Dependencies
------------

Base package (always installed):

- ``ase >= 3.22.0``
- ``torch >= 2.0.0``
- ``numpy >= 1.20.0``
- ``scipy >= 1.7.0``
- ``matplotlib >= 3.5.0``
- ``pandas >= 1.3.0``
- ``tqdm >= 4.60.0``
- ``seaborn``, ``spglib``, ``mp_api``
