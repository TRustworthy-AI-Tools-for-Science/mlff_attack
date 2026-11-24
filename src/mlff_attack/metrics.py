"""Metrics for evaluating MLFF adversarial attack and relaxation outputs.

Attribution
-----------
Structural topology metrics (neighbor-edge Jaccard distance, coordination
changes, periodic-safe RDF L1 distance), multi-tolerance ``spglib`` symmetry
retention metrics, normalized epsilon, RMS force, and delta-force angle metrics
are adapted from ``declanwhchan/mlff_attack_data_collection``
(https://github.com/declanwhchan/mlff_attack_data_collection),
MIT License, Copyright (c) 2026 Declan Chan.
"""

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

RDF_METHOD = "ase_total_rdf_l1_v1"
DEFAULT_SYMPRECS: Tuple[float, ...] = (1e-2, 1e-3, 1e-4)


def calculate_spectral_flatness(signal: Sequence[float]) -> float:
    """Calculate the spectral flatness of a 1D signal.

    Parameters
    ----------
    signal : array-like
        Input 1D signal array.

    Returns
    -------
    float
        Spectral flatness value in ``[0, 1]``, or ``np.nan`` on invalid input.
    """
    try:
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("Input signal must be a non-empty 1D array.")
        shifted = np.abs(arr) + 1e-10
        geometric_mean = float(np.exp(np.mean(np.log(shifted))))
        arithmetic_mean = float(np.mean(shifted))
        spectral_flatness = geometric_mean / arithmetic_mean
    except ValueError as exc:
        logger.error("[ERROR] Failed to calculate spectral flatness: %s", exc)
        spectral_flatness = float("nan")

    return spectral_flatness


def normalized_epsilon(
    epsilon: float,
    reference: Union[float, np.ndarray, Any],
) -> float:
    """Express attack budget ``epsilon`` as a percentage of minimum lattice length.

    Computes :math:`\\epsilon_{\\%} = 100 \\, \\epsilon / L_{\\min}`, where
    :math:`L_{\\min}` is either a positive scalar reference length or the
    shortest lattice vector norm ``min(||a||, ||b||, ||c||)``.

    Parameters
    ----------
    epsilon : float
        Nominal perturbation budget in Angstroms.
    reference : float, np.ndarray, or ase.Atoms
        Reference length in Angstroms, a ``(3, 3)`` cell matrix, or an ASE
        ``Atoms`` object with a non-degenerate unit cell.

    Returns
    -------
    float
        Normalized perturbation percentage, or ``np.nan`` if invalid.
    """
    try:
        eps_val = float(epsilon)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(eps_val):
        return float("nan")

    if hasattr(reference, "cell"):
        lengths = np.asarray(reference.cell.lengths(), dtype=float)
        positive = lengths[lengths > 0]
        ref_length = float(np.min(positive)) if positive.size else float("nan")
    elif np.ndim(reference) == 0:
        try:
            ref_length = float(reference)
        except (TypeError, ValueError):
            ref_length = float("nan")
    else:
        cell = np.asarray(reference, dtype=float)
        if cell.shape != (3, 3):
            return float("nan")
        lengths = np.linalg.norm(cell, axis=1)
        positive = lengths[lengths > 0]
        ref_length = float(np.min(positive)) if positive.size else float("nan")

    if not np.isfinite(ref_length) or ref_length <= 0.0:
        return float("nan")

    return float(100.0 * eps_val / ref_length)


def _extract_positions(obj: Any) -> np.ndarray:
    """Return an ``(N, 3)`` float position array from ``Atoms`` or array-like."""
    if hasattr(obj, "get_positions"):
        positions = np.asarray(obj.get_positions(), dtype=float)
    else:
        positions = np.asarray(obj, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected positions with shape (N, 3), got {positions.shape}."
        )
    return positions


def displacement_metrics(
    before: Union[np.ndarray, Any],
    after: Union[np.ndarray, Any],
) -> Dict[str, float]:
    """Summarize per-atom Euclidean displacements between two structures.

    Parameters
    ----------
    before : ase.Atoms or np.ndarray
        Reference structure or ``(N, 3)`` coordinates in Angstroms.
    after : ase.Atoms or np.ndarray
        Target structure or ``(N, 3)`` coordinates in Angstroms.

    Returns
    -------
    dict
        Summary statistics: ``mean_displacement``, ``median_displacement``,
        ``max_displacement``, and ``rmsd`` (all in Angstroms).
    """
    pos_before = _extract_positions(before)
    pos_after = _extract_positions(after)
    if pos_before.shape != pos_after.shape:
        raise ValueError(
            f"Position shapes do not match: {pos_before.shape} vs {pos_after.shape}."
        )
    if pos_before.shape[0] == 0:
        return {
            "mean_displacement": float("nan"),
            "median_displacement": float("nan"),
            "max_displacement": float("nan"),
            "rmsd": float("nan"),
        }

    displacements = np.linalg.norm(pos_after - pos_before, axis=1)
    return {
        "mean_displacement": float(np.mean(displacements)),
        "median_displacement": float(np.median(displacements)),
        "max_displacement": float(np.max(displacements)),
        "rmsd": float(np.sqrt(np.mean(displacements**2))),
    }


def rms_force(forces: np.ndarray) -> float:
    """Compute root-mean-square atomic force norm ``sqrt(mean(||F_i||^2))``.

    Non-finite rows are ignored; returns ``np.nan`` if no valid rows exist.

    Parameters
    ----------
    forces : array-like
        Atomic force vectors with shape ``(N, 3)`` in eV/Å.

    Returns
    -------
    float
        RMS atomic force magnitude in eV/Å, or ``np.nan``.
    """
    vectors = np.asarray(forces, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        return float("nan")
    squared = np.sum(vectors**2, axis=1)
    squared = squared[np.isfinite(squared)]
    return float(np.sqrt(np.mean(squared))) if squared.size else float("nan")


def force_angle_changes(
    forces_before: np.ndarray,
    forces_after: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """Compute per-atom angular change between two force-vector arrays.

    Pairs where either force vector has zero (or non-finite) norm yield
    ``np.nan``.

    Parameters
    ----------
    forces_before : array-like
        Reference forces with shape ``(N, 3)``.
    forces_after : array-like
        Target forces with shape ``(N, 3)``.
    degrees : bool, optional
        Return angles in degrees if True, radians if False, by default True.

    Returns
    -------
    np.ndarray
        Per-atom angle array of shape ``(N,)``.
    """
    before = np.asarray(forces_before, dtype=float)
    after = np.asarray(forces_after, dtype=float)
    if before.ndim != 2 or before.shape[1] != 3 or before.shape != after.shape:
        raise ValueError(
            f"Expected matching (N, 3) force arrays, got {before.shape} and {after.shape}."
        )

    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    denom = before_norm * after_norm

    angles = np.full(before.shape[0], np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 0.0)
    if np.any(valid):
        cos_theta = np.clip(
            np.sum(before[valid] * after[valid], axis=1) / denom[valid],
            -1.0,
            1.0,
        )
        raw_angles = np.arccos(cos_theta)
        angles[valid] = np.degrees(raw_angles) if degrees else raw_angles

    return angles


def force_change_metrics(
    forces_before: np.ndarray,
    forces_after: np.ndarray,
) -> Dict[str, float]:
    """Summarize force-vector magnitude and directional changes between two states.

    Parameters
    ----------
    forces_before : array-like
        Reference forces with shape ``(N, 3)`` in eV/Å.
    forces_after : array-like
        Target forces with shape ``(N, 3)`` in eV/Å.

    Returns
    -------
    dict
        Contains ``delta_force_mean``, ``delta_force_median``,
        ``delta_force_max``, ``rms_force_before``, ``rms_force_after``,
        ``rms_force_difference``, ``delta_force_angle_mean_deg``, and
        ``delta_force_angle_median_deg``.
    """
    before = np.asarray(forces_before, dtype=float)
    after = np.asarray(forces_after, dtype=float)
    if before.ndim != 2 or before.shape[1] != 3 or before.shape != after.shape:
        raise ValueError(
            f"Expected matching (N, 3) force arrays, got {before.shape} and {after.shape}."
        )

    delta_norms = np.linalg.norm(after - before, axis=1)
    finite_deltas = delta_norms[np.isfinite(delta_norms)]
    angles = force_angle_changes(before, after, degrees=True)
    finite_angles = angles[np.isfinite(angles)]

    rms_before = rms_force(before)
    rms_after = rms_force(after)
    rms_diff = (
        abs(rms_after - rms_before)
        if np.isfinite(rms_before) and np.isfinite(rms_after)
        else float("nan")
    )

    return {
        "delta_force_mean": (
            float(np.mean(finite_deltas)) if finite_deltas.size else float("nan")
        ),
        "delta_force_median": (
            float(np.median(finite_deltas)) if finite_deltas.size else float("nan")
        ),
        "delta_force_max": (
            float(np.max(finite_deltas)) if finite_deltas.size else float("nan")
        ),
        "rms_force_before": rms_before,
        "rms_force_after": rms_after,
        "rms_force_difference": rms_diff,
        "delta_force_angle_mean_deg": (
            float(np.mean(finite_angles)) if finite_angles.size else float("nan")
        ),
        "delta_force_angle_median_deg": (
            float(np.median(finite_angles)) if finite_angles.size else float("nan")
        ),
    }


def _atom_signature(symbols: Sequence[str], index: int) -> str:
    return f"{symbols[index]}{index}"


def neighbor_edge_set(
    atoms: Any,
    cutoffs: Optional[Sequence[float]] = None,
) -> Set[Tuple[str, str]]:
    """Build the undirected neighbor-graph edge set for an ASE ``Atoms`` object.

    Uses ``ase.neighborlist.natural_cutoffs`` by default.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    cutoffs : sequence of float, optional
        Per-atom covalent cutoff radii. If None, uses ASE ``natural_cutoffs``.

    Returns
    -------
    set of tuple of (str, str)
        Sorted undirected edge tuples such as ``("C0", "C1")``.
    """
    from ase.neighborlist import natural_cutoffs, neighbor_list

    symbols = atoms.get_chemical_symbols()
    if cutoffs is None:
        cutoffs = natural_cutoffs(atoms)

    try:
        i_list, j_list = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        i_list, j_list = [], []

    edges: Set[Tuple[str, str]] = set()
    for i_idx, j_idx in zip(i_list, j_list):
        i_int = int(i_idx)
        j_int = int(j_idx)
        if i_int == j_int:
            continue
        a_idx, b_idx = sorted((i_int, j_int))
        edges.add((_atom_signature(symbols, a_idx), _atom_signature(symbols, b_idx)))

    return edges


def edge_jaccard_distance(
    before_edges: Iterable[Tuple[str, str]],
    after_edges: Iterable[Tuple[str, str]],
) -> float:
    """Compute Jaccard dissimilarity between two neighbor-edge sets.

    Returns ``0.0`` when both edge sets are empty or identical, and ``1.0``
    when they share no edges (and at least one is non-empty).

    Parameters
    ----------
    before_edges : iterable
        Reference edge collection.
    after_edges : iterable
        Target edge collection.

    Returns
    -------
    float
        Jaccard distance in ``[0, 1]``.
    """
    before_set = set(before_edges)
    after_set = set(after_edges)
    union = before_set | after_set
    if not union:
        return 0.0
    intersection = before_set & after_set
    return float(1.0 - len(intersection) / len(union))


def coordination_by_atom(
    edges: Iterable[Tuple[str, str]],
    atoms: Any,
) -> Dict[str, int]:
    """Count per-atom coordination numbers from an undirected edge set.

    Parameters
    ----------
    edges : iterable of (str, str)
        Undirected neighbor edges formatted by :func:`neighbor_edge_set`.
    atoms : ase.Atoms
        Atomic structure providing atom symbols and indices.

    Returns
    -------
    dict of str to int
        Mapping from atom signature (e.g. ``"C0"``) to coordination count.
    """
    symbols = atoms.get_chemical_symbols()
    counts = {_atom_signature(symbols, index): 0 for index in range(len(atoms))}

    for node_a, node_b in edges:
        counts[node_a] = counts.get(node_a, 0) + 1
        counts[node_b] = counts.get(node_b, 0) + 1

    return counts


def rdf_analysis_cell(atoms: Any, r_max: float = 6.0) -> Any:
    """Repeat periodic unit cells until ASE can safely compute RDF to ``r_max``.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure.
    r_max : float, optional
        Maximum radial distance in Angstroms, by default 6.0.

    Returns
    -------
    ase.Atoms
        Expanded supercell (or copy if non-periodic).
    """
    expanded = atoms.copy()

    if not np.any(expanded.pbc):
        return expanded

    cell = np.asarray(expanded.cell, dtype=float)
    volume = abs(float(np.linalg.det(cell)))

    if volume <= 1e-12:
        raise ValueError("RDF requires a cell with nonzero volume.")

    repeats = [1, 1, 1]
    for axis in range(3):
        if not expanded.pbc[axis]:
            continue
        other_axes = [index for index in range(3) if index != axis]
        face_area = float(
            np.linalg.norm(np.cross(cell[other_axes[0]], cell[other_axes[1]]))
        )
        if face_area <= 1e-12:
            raise ValueError("RDF requires valid periodic cell vectors.")
        cell_height = volume / face_area
        repeats[axis] = max(
            1,
            int(np.ceil((2.0 * r_max + 1e-8) / cell_height)),
        )

    return expanded.repeat(tuple(repeats))


def rdf_values(
    atoms: Any,
    r_max: float = 6.0,
    bins: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ASE's solid-state radial distribution function ``g(r)``.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    r_max : float, optional
        Maximum radial cutoff in Angstroms, by default 6.0.
    bins : int, optional
        Number of radial histogram bins, by default 60.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(rdf, radii)`` arrays each of shape ``(bins,)``.
    """
    from ase.geometry.rdf import get_rdf

    analysis_atoms = rdf_analysis_cell(atoms, r_max=r_max)
    rdf, radii = get_rdf(
        analysis_atoms,
        rmax=r_max,
        nbins=bins,
        no_dists=False,
    )

    rdf_arr = np.asarray(rdf, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)

    if rdf_arr.shape != (bins,) or radii_arr.shape != (bins,):
        raise ValueError("ASE returned an unexpected RDF shape.")
    if not np.all(np.isfinite(rdf_arr)):
        raise ValueError("ASE returned non-finite RDF values.")

    return rdf_arr, radii_arr


def rdf_l1_distance(
    before_atoms: Any,
    after_atoms: Any,
    r_max: float = 6.0,
    bins: int = 60,
) -> float:
    """Integrated L1 distance between two radial distribution functions.

    Computes :math:`d_{\\mathrm{RDF}} = \\sum_k |g_{\\mathrm{after}}(r_k) -
    g_{\\mathrm{before}}(r_k)| \\, \\Delta r`.

    Parameters
    ----------
    before_atoms : ase.Atoms
        Reference structure.
    after_atoms : ase.Atoms
        Target structure.
    r_max : float, optional
        Maximum radial cutoff in Angstroms, by default 6.0.
    bins : int, optional
        Number of histogram bins, by default 60.

    Returns
    -------
    float
        Non-negative integrated RDF L1 distance.
    """
    before_rdf, before_radii = rdf_values(before_atoms, r_max=r_max, bins=bins)
    after_rdf, after_radii = rdf_values(after_atoms, r_max=r_max, bins=bins)

    if not np.allclose(before_radii, after_radii):
        raise ValueError("Before and after RDF grids do not match.")

    dr = float(r_max) / float(bins)
    return float(np.sum(np.abs(before_rdf - after_rdf)) * dr)


def topology_change_metrics(
    before_atoms: Any,
    after_atoms: Any,
    r_max: float = 6.0,
    bins: int = 60,
) -> Dict[str, Any]:
    """Compute neighbor-graph, coordination, and RDF topology change metrics.

    Parameters
    ----------
    before_atoms : ase.Atoms
        Reference structure.
    after_atoms : ase.Atoms
        Target structure.
    r_max : float, optional
        Maximum RDF radial cutoff in Angstroms, by default 6.0.
    bins : int, optional
        Number of RDF histogram bins, by default 60.

    Returns
    -------
    dict
        Dictionary of neighbor-edge counts, ``neighbor_jaccard_distance``,
        ``coordination_change_mean``, ``coordination_change_max``,
        ``rdf_l1_distance``, and ``rdf_method``.
    """
    before_edges = neighbor_edge_set(before_atoms)
    after_edges = neighbor_edge_set(after_atoms)

    added_edges = after_edges - before_edges
    removed_edges = before_edges - after_edges

    before_coord = coordination_by_atom(before_edges, before_atoms)
    after_coord = coordination_by_atom(after_edges, after_atoms)

    coordination_changes = [
        abs(after_coord.get(atom, 0) - before_coord.get(atom, 0))
        for atom in sorted(set(before_coord) | set(after_coord))
    ]

    return {
        "neighbor_edges_before": len(before_edges),
        "neighbor_edges_after": len(after_edges),
        "neighbor_edges_added": len(added_edges),
        "neighbor_edges_removed": len(removed_edges),
        "neighbor_edge_change_count": len(added_edges) + len(removed_edges),
        "neighbor_jaccard_distance": edge_jaccard_distance(
            before_edges,
            after_edges,
        ),
        "coordination_change_mean": (
            float(np.mean(coordination_changes)) if coordination_changes else 0.0
        ),
        "coordination_change_max": (
            float(np.max(coordination_changes)) if coordination_changes else 0.0
        ),
        "rdf_l1_distance": rdf_l1_distance(
            before_atoms,
            after_atoms,
            r_max=r_max,
            bins=bins,
        ),
        "rdf_method": RDF_METHOD,
    }


def symmetry_signature(
    atoms: Any,
    symprec: float = 1e-3,
) -> Optional[Dict[str, int]]:
    """Return space-group number, symmetry operation count, and unique sites.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic crystal structure.
    symprec : float, optional
        Symmetry tolerance in Angstroms passed to ``spglib``, by default 1e-3.

    Returns
    -------
    dict or None
        Dictionary with keys ``number``, ``operations``, and ``unique_sites``,
        or ``None`` if ``spglib`` cannot determine symmetry.
    """
    import spglib

    cell = (
        atoms.cell.array,
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=symprec,
        angle_tolerance=-1.0,
    )
    if dataset is None:
        return None

    if isinstance(dataset, dict):
        number = int(dataset["number"])
        rotations = dataset["rotations"]
        equivalent_atoms = dataset["equivalent_atoms"]
    else:
        number = int(dataset.number)
        rotations = dataset.rotations
        equivalent_atoms = dataset.equivalent_atoms

    return {
        "number": number,
        "operations": len(rotations),
        "unique_sites": int(len(np.unique(equivalent_atoms))),
    }


def symmetry_change_metrics(
    initial: Any,
    current: Any,
    symprecs: Sequence[float] = DEFAULT_SYMPRECS,
) -> Dict[str, float]:
    """Compare crystallographic symmetry across multiple ``spglib`` tolerances.

    Parameters
    ----------
    initial : ase.Atoms
        Reference crystal structure.
    current : ase.Atoms
        Target crystal structure.
    symprecs : sequence of float, optional
        Symmetry tolerances in Angstroms, by default ``(1e-2, 1e-3, 1e-4)``.

    Returns
    -------
    dict
        Contains ``space_group_change_fraction``,
        ``symmetry_operation_retention``, and ``unique_site_change``.
    """
    initial_data = [symmetry_signature(initial, value) for value in symprecs]
    current_data = [symmetry_signature(current, value) for value in symprecs]
    pairs = [
        (before, after)
        for before, after in zip(initial_data, current_data)
        if before is not None and after is not None
    ]
    if not pairs:
        return {
            "space_group_change_fraction": float("nan"),
            "symmetry_operation_retention": float("nan"),
            "unique_site_change": float("nan"),
        }

    return {
        "space_group_change_fraction": float(
            np.mean([before["number"] != after["number"] for before, after in pairs])
        ),
        "symmetry_operation_retention": float(
            np.median(
                [
                    min(after["operations"], before["operations"])
                    / before["operations"]
                    for before, after in pairs
                ]
            )
        ),
        "unique_site_change": float(
            np.median(
                [
                    after["unique_sites"] - before["unique_sites"]
                    for before, after in pairs
                ]
            )
        ),
    }


def relaxation_stability(energies: Sequence[float]) -> float:
    """Fraction of relaxation steps where potential energy increased.

    Computes :math:`n(\\Delta E > 0) / (N - 1)` across consecutive finite
    energies. Lower values indicate smoother, more monotonic descent.

    Parameters
    ----------
    energies : sequence of float
        Energy trajectory values in eV.

    Returns
    -------
    float
        Fraction in ``[0, 1]``, or ``np.nan`` if fewer than 2 finite steps.
    """
    arr = np.asarray(energies, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.ndim != 1 or arr.size < 2:
        return float("nan")
    diffs = np.diff(arr)
    return float(np.mean(diffs > 0.0))


def force_coherence(
    force_magnitudes: Sequence[float],
    displacements: Sequence[float],
) -> float:
    """Pearson correlation between per-atom force magnitudes and displacements.

    Computes :math:`\\rho = \\operatorname{corr}(|F_i|, \\Delta R_i)`.

    Parameters
    ----------
    force_magnitudes : sequence of float
        Per-atom force norms (or mean norms) of shape ``(N,)``.
    displacements : sequence of float
        Per-atom displacement magnitudes of shape ``(N,)``.

    Returns
    -------
    float
        Pearson correlation coefficient in ``[-1, 1]``, or ``np.nan`` if fewer
        than 2 valid atoms or zero variance.
    """
    f_arr = np.asarray(force_magnitudes, dtype=float)
    d_arr = np.asarray(displacements, dtype=float)
    if f_arr.ndim != 1 or d_arr.ndim != 1 or f_arr.size != d_arr.size:
        raise ValueError(
            "force_magnitudes and displacements must be 1D arrays of equal length."
        )
    valid = np.isfinite(f_arr) & np.isfinite(d_arr)
    f_valid = f_arr[valid]
    d_valid = d_arr[valid]
    if f_valid.size < 2:
        return float("nan")
    if np.std(f_valid) == 0.0 or np.std(d_valid) == 0.0:
        return float("nan")
    return float(np.corrcoef(f_valid, d_valid)[0, 1])


__all__ = [
    "DEFAULT_SYMPRECS",
    "RDF_METHOD",
    "calculate_spectral_flatness",
    "coordination_by_atom",
    "displacement_metrics",
    "edge_jaccard_distance",
    "force_angle_changes",
    "force_change_metrics",
    "force_coherence",
    "neighbor_edge_set",
    "normalized_epsilon",
    "rdf_analysis_cell",
    "rdf_l1_distance",
    "rdf_values",
    "relaxation_stability",
    "rms_force",
    "symmetry_change_metrics",
    "symmetry_signature",
    "topology_change_metrics",
]
