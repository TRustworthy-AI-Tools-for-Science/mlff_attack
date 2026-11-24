Installation
============

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

**Python 3.10 -- 3.12** is required (Python 3.12 is recommended).

Clone the repository:

.. code-block:: bash

   git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
   cd mlff_attack

Each MLFF backend should be installed in its **own virtual environment** to
avoid dependency conflicts (see :ref:`mace-uma-conflict` below).

Automated setup scripts (Linux / WSL Ubuntu)
--------------------------------------------

Setup scripts for each backend are provided in ``scripts/``:

.. code-block:: bash

   bash scripts/setup_mace.sh    # creates .venv-mace
   bash scripts/setup_uma.sh     # creates .venv-uma
   bash scripts/setup_chgnet.sh  # creates .venv-chgnet
   bash scripts/setup_mtp.sh     # creates conda env env-mtp and builds MLIP-3 mlp

Activate the corresponding environment before running commands:

.. code-block:: bash

   source .venv-mace/bin/activate
   source .venv-uma/bin/activate
   source .venv-chgnet/bin/activate
   conda activate env-mtp

Manual installation with pip
----------------------------

.. code-block:: bash

   # MACE environment
   python3 -m venv .venv-mace
   source .venv-mace/bin/activate
   pip install -e ".[mace,dev]"

   # UMA environment (separate environment required)
   python3 -m venv .venv-uma
   source .venv-uma/bin/activate
   pip install -e ".[uma,dev]"
   hf auth login

   # CHGNet environment
   python3 -m venv .venv-chgnet
   source .venv-chgnet/bin/activate
   pip install -e ".[chgnet,dev]"

.. note::

   UMA models are gated on Hugging Face. You must run ``hf auth login`` and
   accept the model license on the Hub before weights can be downloaded.

Installing with uv
------------------

If you use `uv <https://docs.astral.sh/uv/>`_, ``pyproject.toml`` declares
the ``mace`` and ``uma`` extras as conflicting so ``uv`` prevents accidentally
mixing incompatible ``e3nn`` versions:

.. code-block:: bash

   # MACE environment (.venv)
   uv sync --extra mace --extra dev

   # UMA environment (.venv-uma)
   uv venv .venv-uma
   UV_PROJECT_ENVIRONMENT=.venv-uma uv sync --extra uma --extra dev

.. _mace-uma-conflict:

MACE / UMA environment conflict
-------------------------------

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
   * - ``fairchem-core >= 2.0``
     - ``>= 0.5``
     - Requires the newer e3nn API

Attempting to install both in one environment via ``pip`` can silently install
``e3nn >= 0.5`` and cause MACE model loading to fail at runtime with a
``ValueError``. Maintain separate virtual environments when using both models.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Contents
   * - ``mace``
     - ``mace-torch >= 0.3`` -- MACE and MACE-MH calculator support
   * - ``uma``
     - ``fairchem-core >= 2.0, < 3.0`` + ``huggingface_hub`` -- UMA calculator support
   * - ``chgnet``
     - ``chgnet >= 0.4.2`` -- CHGNet calculator support
   * - ``notebooks``
     - ``jupyterlab``, ``ipywidgets``, ``ipykernel`` -- running example notebooks
   * - ``dev``
     - ``pytest``, ``pytest-cov``, ``black``, ``flake8``, ``mypy``, ``pylint``, ``sphinx``, ``furo``

Verifying the test suite
------------------------

.. code-block:: bash

   # Core + MACE tests (in MACE environment)
   pytest

   # Backend-specific calculator tests
   pytest tests/calc_single/test_mace.py -v
   pytest tests/calc_single/test_uma.py -v
   pytest tests/calc_single/test_chgnet.py -v
   pytest tests/calc_single/test_mtp.py -v
