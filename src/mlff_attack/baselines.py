"""Non-adversarial perturbation baselines and VASP DFT reference loaders.

This module provides control perturbations to contrast against gradient-based
adversarial attacks (FGSM, I-FGSM, PGD), together with VASP DFT parsing
utilities for MLFF--DFT comparison:

1. **Random isotropic / box-bounded baselines**:
   :func:`random_uniform_box_perturbation` and
   :func:`random_gaussian_perturbation` matching the :math:`L_\\infty` budget
   :math:`\\epsilon`.
2. **Iso-energy Contour Exploration baseline**:
   :func:`initial_velocities`, :func:`out_of_plane_angle`,
   :func:`parse_betas`, :func:`beta_tag`, and :func:`run_contour_exploration`.
3. **VASP DFT reference loaders**:
   :func:`read_dft_structure`, :func:`read_outcar_force_blocks`,
   :func:`parse_perturbed_structure_name`, and :func:`load_dft_manifest`.

Attribution
-----------
Contour exploration helpers (:func:`parse_betas`, :func:`beta_tag`,
:func:`initial_velocities`, :func:`out_of_plane_angle`,
:func:`run_contour_exploration`) and VASP DFT loaders
(:func:`read_dft_structure`, :func:`read_outcar_force_blocks`,
:func:`parse_perturbed_structure_name`) are adapted from
``declanwhchan/mlff_attack_data_collection``
(https://github.com/declanwhchan/mlff_attack_data_collection),
MIT License, Copyright (c) 2026 Declan Chan.
"""

import csv
from io import StringIO
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mlff_attack import metrics as mlff_metrics

DFT_SOURCE_MODELS: Dict[str, str] = {
    "mace": "mace_mh",
    "mace_mh": "mace_mh",
    "uma": "uma",
    "chgnet": "chgnet",
    "mtp": "mtp",
}


# ---------------------------------------------------------------------------
# 1. Random Non-Adversarial Perturbation Controls
# ---------------------------------------------------------------------------


def _apply_position_delta(atoms_or_positions: Any, delta: np.ndarray) -> Any:
    """Apply ``delta`` to an ASE ``Atoms`` copy (preserving ``calc``) or array."""
    if hasattr(atoms_or_positions, "get_positions"):
        perturbed = atoms_or_positions.copy()
        perturbed.set_positions(atoms_or_positions.get_positions() + delta)
        if hasattr(atoms_or_positions, "calc"):
            perturbed.calc = atoms_or_positions.calc
        return perturbed
    return np.asarray(atoms_or_positions, dtype=float) + delta


def random_uniform_box_perturbation(
    atoms_or_positions: Any,
    epsilon: float,
    seed: Optional[int] = 42,
) -> Any:
    """Apply an i.i.d. uniform coordinate perturbation inside ``[-epsilon, epsilon]``.

    Serves as a non-adversarial :math:`L_\\infty`-matched random baseline
    against FGSM, I-FGSM, and PGD attacks.

    Parameters
    ----------
    atoms_or_positions : ase.Atoms or np.ndarray
        Input structure or ``(N, 3)`` atomic positions in Angstroms.
    epsilon : float
        Maximum per-coordinate perturbation magnitude in Angstroms.
    seed : int, optional
        Random seed for reproducibility, by default ``42``.

    Returns
    -------
    ase.Atoms or np.ndarray
        Perturbed copy matching the input type.
    """
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    positions = (
        np.asarray(atoms_or_positions.get_positions(), dtype=float)
        if hasattr(atoms_or_positions, "get_positions")
        else np.asarray(atoms_or_positions, dtype=float)
    )
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-float(epsilon), float(epsilon), size=positions.shape)
    return _apply_position_delta(atoms_or_positions, delta)


def random_gaussian_perturbation(
    atoms_or_positions: Any,
    epsilon: float,
    seed: Optional[int] = 42,
    clip: bool = True,
) -> Any:
    """Apply isotropic Gaussian coordinate noise with standard deviation ``epsilon``.

    Parameters
    ----------
    atoms_or_positions : ase.Atoms or np.ndarray
        Input structure or ``(N, 3)`` atomic positions in Angstroms.
    epsilon : float
        Standard deviation (and optional :math:`L_\\infty` clip bound) in Å.
    seed : int, optional
        Random seed for reproducibility, by default ``42``.
    clip : bool, optional
        If True, clip each coordinate displacement to ``[-epsilon, epsilon]``,
        by default True.

    Returns
    -------
    ase.Atoms or np.ndarray
        Perturbed copy matching the input type.
    """
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    positions = (
        np.asarray(atoms_or_positions.get_positions(), dtype=float)
        if hasattr(atoms_or_positions, "get_positions")
        else np.asarray(atoms_or_positions, dtype=float)
    )
    rng = np.random.default_rng(seed)
    delta = rng.normal(loc=0.0, scale=float(epsilon), size=positions.shape)
    if clip:
        delta = np.clip(delta, -float(epsilon), float(epsilon))
    return _apply_position_delta(atoms_or_positions, delta)


# ---------------------------------------------------------------------------
# 2. Iso-Energy Contour Exploration Baseline
# ---------------------------------------------------------------------------


def parse_betas(
    raw: Optional[Union[str, Sequence[float]]],
    default: Sequence[float] = (0.1, 0.05, 0.0),
) -> List[float]:
    """Parse ``parallel_drift`` (``beta``) values from CLI strings or sequences.

    Parameters
    ----------
    raw : str, sequence of float, or None
        Comma-separated string (e.g. ``"0.2, 0.1, 0"``), iterable of floats,
        or ``None`` to use ``default``.
    default : sequence of float, optional
        Fallback values when ``raw`` is None or blank.

    Returns
    -------
    list of float
        Parsed beta values.
    """
    if raw is None:
        return [float(val) for val in default]
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if not tokens:
            return [float(val) for val in default]
        return [float(token) for token in tokens]
    return [float(val) for val in raw]


def beta_tag(beta: float) -> str:
    """Format a deterministic directory/file tag for contour ``beta`` values.

    Examples
    --------
    >>> beta_tag(0.1)
    'beta_100'
    >>> beta_tag(0.05)
    'beta_050'
    """
    return f"beta_{int(round(float(beta) * 1000)):03d}"


def initial_velocities(atoms: Any, seed: Optional[int] = 42) -> np.ndarray:
    """Assign repeatable, zero-mean Gaussian initial velocities to ``atoms``.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose velocities are initialized in-place.
    seed : int, optional
        Random seed, by default ``42``.

    Returns
    -------
    np.ndarray
        Assigned velocity array of shape ``(N, 3)``.
    """
    rng = np.random.default_rng(seed)
    velocities = rng.normal(size=(len(atoms), 3))
    velocities -= velocities.mean(axis=0)
    atoms.set_velocities(velocities)
    return velocities


def out_of_plane_angle(
    prev_prev_positions: Optional[np.ndarray],
    prev_positions: Optional[np.ndarray],
    positions: np.ndarray,
) -> float:
    """Compute turning angle (degrees) between consecutive displacement steps.

    Parameters
    ----------
    prev_prev_positions : np.ndarray or None
        Positions at step ``t - 2``.
    prev_positions : np.ndarray or None
        Positions at step ``t - 1``.
    positions : np.ndarray
        Positions at step ``t``.

    Returns
    -------
    float
        Turning angle in degrees in ``[0, 180]``, or ``np.nan`` if undefined.
    """
    if prev_prev_positions is None or prev_positions is None:
        return float("nan")

    vec_a = (
        np.asarray(prev_positions, dtype=float)
        - np.asarray(prev_prev_positions, dtype=float)
    ).reshape(-1)
    vec_b = (
        np.asarray(positions, dtype=float) - np.asarray(prev_positions, dtype=float)
    ).reshape(-1)

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")

    dot = float(np.clip(np.dot(vec_a / norm_a, vec_b / norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def run_contour_exploration(
    atoms: Any,
    steps: int = 100,
    beta: float = 0.05,
    maxstep: float = 0.01,
    energy_target: Optional[float] = None,
    angle_limit: float = 20.0,
    seed: int = 42,
    trajectory_path: Optional[Union[str, Path]] = None,
    compute_topology: bool = False,
    compute_symmetry: bool = False,
) -> Tuple[Any, List[Dict[str, float]], Dict[str, Any]]:
    """Run ASE ``ContourExploration`` as an iso-energy non-adversarial baseline.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure with an ASE calculator attached (not modified in-place).
    steps : int, optional
        Number of contour-exploration steps, by default ``100``.
    beta : float, optional
        ``parallel_drift`` parameter passed to ASE ``ContourExploration``,
        by default ``0.05``.
    maxstep : float, optional
        Maximum step size in Angstroms, by default ``0.01``.
    energy_target : float, optional
        Target potential energy in eV. If None, uses the initial energy of
        ``atoms``.
    angle_limit : float, optional
        Maximum turning angle limit in degrees, by default ``20.0``.
    seed : int, optional
        Base random seed for velocities and contour RNG, by default ``42``.
    trajectory_path : str or Path, optional
        Optional ``.traj`` file path to record contour frames.
    compute_topology : bool, optional
        If True, include neighbor-Jaccard / coordination / RDF metrics between
        initial and final structures, by default False.
    compute_symmetry : bool, optional
        If True, include ``spglib`` symmetry-retention metrics between initial
        and final structures, by default False.

    Returns
    -------
    tuple of (ase.Atoms, list of dict, dict)
        ``(final_atoms, history_rows, summary)``.
    """
    from ase.md.contour_exploration import ContourExploration

    if getattr(atoms, "calc", None) is None:
        raise ValueError("atoms must have an ASE calculator attached.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    initial_atoms = atoms.copy()
    initial_atoms.calc = atoms.calc
    work_atoms = atoms.copy()
    work_atoms.calc = atoms.calc

    derived_seed = int(seed) + int(round(float(beta) * 1000))
    initial_velocities(work_atoms, seed=derived_seed)

    initial_energy = float(work_atoms.get_potential_energy())
    target_e = initial_energy if energy_target is None else float(energy_target)
    initial_positions = work_atoms.get_positions().copy()
    initial_forces = np.asarray(work_atoms.get_forces(), dtype=float).copy()

    traj_arg = None
    if trajectory_path is not None:
        traj_out = Path(trajectory_path)
        traj_out.parent.mkdir(parents=True, exist_ok=True)
        traj_arg = str(traj_out)

    dyn = ContourExploration(
        work_atoms,
        maxstep=float(maxstep),
        parallel_drift=float(beta),
        energy_target=target_e,
        angle_limit=float(angle_limit),
        rng=np.random.default_rng(derived_seed),
        trajectory=traj_arg,
        logfile=None,
        loginterval=1,
    )

    history: List[Dict[str, float]] = []
    prev_prev_pos = None
    prev_pos = None

    for step in range(steps + 1):
        energy = float(work_atoms.get_potential_energy())
        forces = np.asarray(work_atoms.get_forces(), dtype=float)
        cur_pos = work_atoms.get_positions()
        disp = np.linalg.norm(cur_pos - initial_positions, axis=1)
        f_delta = np.linalg.norm(forces - initial_forces, axis=1)

        history.append(
            {
                "step": step,
                "energy_ev": energy,
                "energy_deviation_ev": energy - target_e,
                "energy_deviation_mev_per_atom": (
                    (energy - target_e) * 1000.0 / len(work_atoms)
                ),
                "step_size_a": float(getattr(dyn, "step_size", np.nan)),
                "curvature_1_per_a": float(getattr(dyn, "curvature", np.nan)),
                "mean_displacement_a": float(np.mean(disp)),
                "max_displacement_a": float(np.max(disp)),
                "mean_force_delta_ev_a": float(np.mean(f_delta)),
                "out_of_plane_angle_deg": out_of_plane_angle(
                    prev_prev_pos, prev_pos, cur_pos
                ),
            }
        )

        if step == steps:
            break

        prev_prev_pos = None if prev_pos is None else prev_pos.copy()
        prev_pos = cur_pos.copy()
        dyn.run(1)

    final_forces = np.asarray(work_atoms.get_forces(), dtype=float)
    summary: Dict[str, Any] = {
        "steps": int(steps),
        "beta": float(beta),
        "energy_target_ev": target_e,
        "initial_energy_ev": initial_energy,
        "final_energy_ev": float(history[-1]["energy_ev"]),
        **mlff_metrics.displacement_metrics(initial_positions, work_atoms),
        **mlff_metrics.force_change_metrics(initial_forces, final_forces),
    }

    if compute_topology:
        summary.update(
            mlff_metrics.topology_change_metrics(initial_atoms, work_atoms)
        )
    if compute_symmetry:
        summary.update(
            mlff_metrics.symmetry_change_metrics(initial_atoms, work_atoms)
        )

    return work_atoms, history, summary


# ---------------------------------------------------------------------------
# 3. VASP DFT Reference Loaders
# ---------------------------------------------------------------------------


def read_dft_structure(path: Union[str, Path], index: int = -1) -> Any:
    """Read a DFT structure file, including POSCAR files saved with a ``.cif`` suffix.

    Parameters
    ----------
    path : str or Path
        Path to a CIF, VASP POSCAR/CONTCAR, or trajectory file.
    index : int, optional
        Frame index passed to ``ase.io.read``, by default ``-1``.

    Returns
    -------
    ase.Atoms
        Parsed ASE ``Atoms`` object.
    """
    from ase.io import read as ase_read

    file_path = Path(path)
    try:
        return ase_read(file_path, index=index)
    except Exception as original_error:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            counts = lines[6].split()
            symbols = [token.rstrip("/") for token in lines[5].split()]
            valid = bool(counts) and all(int(token) >= 0 for token in counts)
        except Exception:
            raise original_error
        if not valid or len(symbols) != len(counts) or any(not s for s in symbols):
            raise original_error
        lines[5] = "  " + "  ".join(symbols)
        return ase_read(
            StringIO("\n".join(lines) + "\n"),
            format="vasp",
            index=index,
        )


def read_outcar_force_blocks(
    path: Union[str, Path],
    expected_atoms: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse first and final complete ``TOTAL-FORCE (eV/Angst)`` blocks from VASP OUTCAR.

    Parameters
    ----------
    path : str or Path
        Path to a VASP ``OUTCAR`` file.
    expected_atoms : int
        Number of atoms expected in each force block.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, int)
        ``(first_forces, final_forces, block_count)`` where each force array
        has shape ``(expected_atoms, 3)`` in eV/Å.
    """
    if expected_atoms <= 0:
        raise ValueError("expected_atoms must be positive.")

    blocks: List[np.ndarray] = []
    rows: Optional[List[List[float]]] = None

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "TOTAL-FORCE (eV/Angst)" in line:
                rows = []
                continue
            if rows is None:
                continue
            fields = line.split()
            try:
                values = [float(value) for value in fields[:6]]
            except (ValueError, TypeError):
                if not rows:
                    continue
                if len(rows) == expected_atoms:
                    blocks.append(np.asarray(rows, dtype=float))
                rows = None
                continue
            if len(values) != 6:
                continue
            rows.append(values[3:6])
            if len(rows) == expected_atoms:
                blocks.append(np.asarray(rows, dtype=float))
                rows = None

    if not blocks:
        raise ValueError(f"No complete TOTAL-FORCE blocks found in {path}.")
    return blocks[0], blocks[-1], len(blocks)


def _parse_epsilon_token(token: str) -> float:
    """Decode filename epsilon tokens like ``'005' -> 0.05`` or ``'2' -> 2.0``."""
    token_str = str(token)
    return float(f"0.{token_str[1:]}") if token_str.startswith("0") else float(token_str)


def parse_perturbed_structure_name(name: str) -> Optional[Dict[str, Any]]:
    """Parse canonical perturbed DFT/MLFF structure identifiers.

    Recognizes names of the form
    ``<model>_<material>_<attack>_eps<token>[_steps<n>]_perturbed``, e.g.
    ``uma_berlinite_alpo4_ifgsm_eps03_perturbed``.

    Parameters
    ----------
    name : str
        Structure folder or file stem.

    Returns
    -------
    dict or None
        Dictionary with ``source_model``, ``material_slug``, ``attack_label``,
        ``epsilon``, and ``n_steps``, or ``None`` if the name does not match.
    """
    match = re.match(
        r"^(mace_mh|mace|uma|chgnet|mtp)_(.+)_"
        r"(fgsm|ifgsm|pgd)_eps([0-9]+)"
        r"(?:_steps([0-9]+))?_perturbed$",
        str(name).strip().lower(),
    )
    if match is None:
        return None
    model, material, attack, epsilon_token, steps = match.groups()
    return {
        "source_model": DFT_SOURCE_MODELS[model],
        "material_slug": material,
        "attack_label": {
            "fgsm": "FGSM",
            "ifgsm": "I-FGSM",
            "pgd": "PGD",
        }[attack],
        "epsilon": _parse_epsilon_token(epsilon_token),
        "n_steps": int(steps) if steps else None,
    }


def load_dft_manifest(
    manifest_path: Union[str, Path],
    delivery_group: Optional[str] = "FORCE_CONVERGED_CLEAN",
) -> List[Dict[str, Any]]:
    """Load a fixed-cell VASP DFT delivery manifest CSV with parsed metadata.

    Parameters
    ----------
    manifest_path : str or Path
        Path to ``preliminary_manifest.csv`` or ``combined_manifest.csv``.
    delivery_group : str or None, optional
        If provided, filter rows to this ``delivery_group`` (case-insensitive),
        by default ``"FORCE_CONVERGED_CLEAN"``. Pass ``None`` to return all rows.

    Returns
    -------
    list of dict
        Manifest rows enriched with parsed ``source_model``, ``material_slug``,
        ``attack_label``, ``parsed_epsilon``, and ``parsed_n_steps`` fields
        when ``structure_name`` matches the canonical naming pattern.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"DFT manifest not found: {path}")

    target_group = delivery_group.upper() if delivery_group is not None else None
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if target_group is not None:
                group = str(row.get("delivery_group", "")).strip().upper()
                if group != target_group:
                    continue
            enriched: Dict[str, Any] = dict(row)
            parsed = parse_perturbed_structure_name(str(row.get("structure_name", "")))
            if parsed is not None:
                enriched["source_model"] = parsed["source_model"]
                enriched["material_slug"] = parsed["material_slug"]
                enriched["attack_label"] = parsed["attack_label"]
                enriched["parsed_epsilon"] = parsed["epsilon"]
                enriched["parsed_n_steps"] = parsed["n_steps"]
            records.append(enriched)

    return records


__all__ = [
    "DFT_SOURCE_MODELS",
    "beta_tag",
    "initial_velocities",
    "load_dft_manifest",
    "out_of_plane_angle",
    "parse_betas",
    "parse_perturbed_structure_name",
    "random_gaussian_perturbation",
    "random_uniform_box_perturbation",
    "read_dft_structure",
    "read_outcar_force_blocks",
    "run_contour_exploration",
]
