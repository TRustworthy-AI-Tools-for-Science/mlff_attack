"""Potential energy surface (PES) and model loss-landscape diagnostics.

This module connects input-space adversarial perturbations to:

1. **Predicted PES geometry in configuration space** (:math:`\\mathbf{R}`-space):
   1D/2D PES slices along adversarial and orthogonal relaxation directions,
   PES roughness/smoothness metrics, finite-difference atomic Hessians
   :math:`H_q[E_\\theta]`, perturbation-Hessian mode alignment, and
   :math:`\\Gamma`-point phonon stability.
2. **Model loss-landscape geometry in parameter space** (:math:`\\theta`-space)
   and **cross-space coupling** (:math:`\\theta \\leftrightarrow \\mathbf{R}`):
   filter-normalized random directions (Li et al., 2018), 1D/2D parameter loss
   slices, local parameter sharpness, and parameter sensitivity of predicted
   energies and forces projected onto adversarial perturbation directions.

Attribution
-----------
The 2D attack/orthogonal-relaxation PES basin projection
(:func:`build_attack_relaxation_basis`, :func:`project_onto_basis`,
:func:`evaluate_pes_slice_2d`, :func:`plot_pes_basin_2d`) and
:math:`\\Gamma`-point phonon stability calculation (:func:`gamma_phonon_modes`)
are adapted from ``declanwhchan/mlff_attack_data_collection``
(https://github.com/declanwhchan/mlff_attack_data_collection),
MIT License, Copyright (c) 2026 Declan Chan.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _extract_positions(obj: Any) -> np.ndarray:
    """Return an ``(N, 3)`` float coordinate array from ``Atoms`` or array."""
    if hasattr(obj, "get_positions"):
        positions = np.asarray(obj.get_positions(), dtype=float)
    else:
        positions = np.asarray(obj, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected positions with shape (N, 3), got {positions.shape}."
        )
    return positions


def _unit_vector(vector: np.ndarray, name: str = "direction") -> np.ndarray:
    """Flatten and normalize a vector to unit Euclidean norm."""
    flat = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{name} has near-zero or non-finite norm ({norm}).")
    return flat / norm


def _eval_energy(
    reference: Any,
    positions: np.ndarray,
    calculator: Optional[Any] = None,
    energy_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> float:
    """Evaluate potential energy at ``positions`` using ``energy_fn`` or ASE."""
    if energy_fn is not None:
        return float(energy_fn(positions))

    atoms = reference.copy()
    atoms.set_positions(positions)
    calc = calculator if calculator is not None else getattr(reference, "calc", None)
    if calc is None:
        raise ValueError(
            "Provide either `energy_fn`, `calculator`, or an Atoms object with "
            "an attached calculator."
        )
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def _eval_forces(
    reference: Any,
    positions: np.ndarray,
    calculator: Optional[Any] = None,
    forces_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Evaluate atomic forces at ``positions`` using ``forces_fn`` or ASE."""
    if forces_fn is not None:
        forces = np.asarray(forces_fn(positions), dtype=float)
        if forces.shape != positions.shape:
            raise ValueError(
                f"Expected forces shape {positions.shape}, got {forces.shape}."
            )
        return forces

    atoms = reference.copy()
    atoms.set_positions(positions)
    calc = calculator if calculator is not None else getattr(reference, "calc", None)
    if calc is None:
        raise ValueError(
            "Provide either `forces_fn`, `calculator`, or an Atoms object with "
            "an attached calculator."
        )
    atoms.calc = calc
    return np.asarray(atoms.get_forces(), dtype=float)


# ---------------------------------------------------------------------------
# 1. Configuration-Space (PES) Slicing & Smoothness
# ---------------------------------------------------------------------------


def build_attack_relaxation_basis(
    reference: Any,
    perturbed: Any,
    final: Optional[Any] = None,
    fallback_direction: Optional[np.ndarray] = None,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct an orthonormal 2D basis for a local PES basin slice.

    ``u1`` is the unit vector along the adversarial perturbation
    ``perturbed - reference``. ``u2`` is obtained via Gram-Schmidt
    orthogonalization of ``final - reference`` (or ``fallback_direction`` if
    ``final`` is omitted or collinear with ``u1``) against ``u1``.

    Parameters
    ----------
    reference : ase.Atoms or np.ndarray
        Pre-attack reference structure or ``(N, 3)`` positions.
    perturbed : ase.Atoms or np.ndarray
        Adversarially perturbed structure or ``(N, 3)`` positions.
    final : ase.Atoms or np.ndarray, optional
        Post-relaxation structure or ``(N, 3)`` positions.
    fallback_direction : np.ndarray, optional
        Optional direction (shape ``(N, 3)`` or ``(3N,)``) used to form ``u2``
        when ``final`` is omitted or collinear with ``u1``.
    tol : float, optional
        Minimum Euclidean norm threshold, by default ``1e-8``.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Orthonormal 1D unit vectors ``(u1, u2)`` of shape ``(3N,)``.
    """
    ref_pos = _extract_positions(reference)
    pert_pos = _extract_positions(perturbed)
    if ref_pos.shape != pert_pos.shape:
        raise ValueError(
            f"Reference and perturbed shapes do not match: {ref_pos.shape} vs {pert_pos.shape}."
        )

    d1 = (pert_pos - ref_pos).reshape(-1)
    norm_d1 = float(np.linalg.norm(d1))
    if norm_d1 < tol:
        raise ValueError(
            "Perturbed structure has near-zero displacement from reference."
        )
    u1 = d1 / norm_d1

    candidate = None
    if final is not None:
        final_pos = _extract_positions(final)
        if final_pos.shape != ref_pos.shape:
            raise ValueError(
                f"Reference and final shapes do not match: {ref_pos.shape} vs {final_pos.shape}."
            )
        d2 = (final_pos - ref_pos).reshape(-1)
        d2 = d2 - float(np.dot(d2, u1)) * u1
        if float(np.linalg.norm(d2)) >= tol:
            candidate = d2

    if candidate is None and fallback_direction is not None:
        d2 = np.asarray(fallback_direction, dtype=float).reshape(-1)
        if d2.shape != u1.shape:
            raise ValueError(
                f"fallback_direction size {d2.size} does not match 3N={u1.size}."
            )
        d2 = d2 - float(np.dot(d2, u1)) * u1
        if float(np.linalg.norm(d2)) >= tol:
            candidate = d2

    if candidate is None:
        raise ValueError(
            "Relaxation displacement is collinear with the attack direction "
            "(or no `final`/`fallback_direction` was provided), so no "
            "independent 2D basin slice exists."
        )

    u2 = candidate / float(np.linalg.norm(candidate))
    return u1, u2


def project_onto_basis(
    structures: Sequence[Any],
    reference: Any,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    """Project atomic structures onto the 2D ``(u1, u2)`` PES slice.

    Parameters
    ----------
    structures : sequence of (ase.Atoms or np.ndarray)
        Structures to project.
    reference : ase.Atoms or np.ndarray
        Origin structure corresponding to ``(0, 0)``.
    u1, u2 : np.ndarray
        Orthonormal 1D basis vectors of shape ``(3N,)``.

    Returns
    -------
    np.ndarray
        Projected coordinates of shape ``(len(structures), 2)``.
    """
    ref_pos = _extract_positions(reference)
    u1_flat = np.asarray(u1, dtype=float).reshape(-1)
    u2_flat = np.asarray(u2, dtype=float).reshape(-1)

    if len(structures) == 0:
        return np.empty((0, 2), dtype=float)

    coords = []
    for item in structures:
        disp = (_extract_positions(item) - ref_pos).reshape(-1)
        coords.append((float(np.dot(disp, u1_flat)), float(np.dot(disp, u2_flat))))
    return np.asarray(coords, dtype=float)


def pes_roughness(
    alphas: Sequence[float],
    energies: Sequence[float],
) -> Dict[str, float]:
    """Quantify smoothness / roughness of a 1D PES profile.

    For a perfectly harmonic quadratic basin :math:`E(\\alpha) = E_0 + \\frac{1}{2}k\\alpha^2`,
    the second derivative :math:`E''(\\alpha) = k` is constant, so
    ``curvature_tv`` (total variation of curvature) is identically zero.
    Non-smooth neural-network PES artifacts produce oscillating curvature and
    elevated ``curvature_tv`` and ``curvature_std``.

    Parameters
    ----------
    alphas : sequence of float
        Monotonically increasing 1D coordinate grid.
    energies : sequence of float
        Potential energies evaluated at ``alphas`` (in eV).

    Returns
    -------
    dict
        Contains ``mean_abs_curvature``, ``rms_curvature``,
        ``max_abs_curvature``, ``curvature_std``, and ``curvature_tv``.
    """
    alpha_arr = np.asarray(alphas, dtype=float)
    energy_arr = np.asarray(energies, dtype=float)
    valid = np.isfinite(alpha_arr) & np.isfinite(energy_arr)
    alpha_arr = alpha_arr[valid]
    energy_arr = energy_arr[valid]

    if alpha_arr.size < 3:
        return {
            "mean_abs_curvature": float("nan"),
            "rms_curvature": float("nan"),
            "max_abs_curvature": float("nan"),
            "curvature_std": float("nan"),
            "curvature_tv": float("nan"),
        }

    # Exact 3-point central second derivative on interior points (handles
    # uniform and non-uniform grids without one-sided boundary leakage)
    h_left = alpha_arr[1:-1] - alpha_arr[:-2]
    h_right = alpha_arr[2:] - alpha_arr[1:-1]
    slope_left = (energy_arr[1:-1] - energy_arr[:-2]) / h_left
    slope_right = (energy_arr[2:] - energy_arr[1:-1]) / h_right
    interior = 2.0 * (slope_right - slope_left) / (h_left + h_right)

    return {
        "mean_abs_curvature": float(np.mean(np.abs(interior))),
        "rms_curvature": float(np.sqrt(np.mean(interior**2))),
        "max_abs_curvature": float(np.max(np.abs(interior))),
        "curvature_std": float(np.std(interior)),
        "curvature_tv": (
            float(np.sum(np.abs(np.diff(interior)))) if interior.size >= 2 else 0.0
        ),
    }


def evaluate_pes_profile_1d(
    reference: Any,
    direction: np.ndarray,
    alphas: Sequence[float],
    calculator: Optional[Any] = None,
    energy_fn: Optional[Callable[[np.ndarray], float]] = None,
    forces_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    compute_forces: bool = False,
) -> Dict[str, Any]:
    """Evaluate a 1D PES slice ``E(R_0 + alpha * d_hat) - E(R_0)``.

    Parameters
    ----------
    reference : ase.Atoms or np.ndarray
        Reference structure or ``(N, 3)`` positions.
    direction : np.ndarray
        Direction vector of shape ``(N, 3)`` or ``(3N,)`` (normalized internally).
    alphas : sequence of float
        1D displacement amplitudes along ``d_hat`` (in Å).
    calculator : Any, optional
        ASE calculator (if omitted, uses ``reference.calc``).
    energy_fn : callable, optional
        Optional analytical callback ``energy_fn(positions) -> float``.
    forces_fn : callable, optional
        Optional analytical callback ``forces_fn(positions) -> (N, 3) array``.
    compute_forces : bool, optional
        If True, also record force norms and directional slope ``-F · d_hat``,
        by default False.

    Returns
    -------
    dict
        Contains ``alphas``, ``direction``, ``reference_energy``,
        ``relative_energies``, ``roughness``, and (when ``compute_forces=True``)
        ``directional_slopes`` and ``max_forces``.
    """
    ref_pos = _extract_positions(reference)
    u_dir = _unit_vector(direction, name="direction")
    if u_dir.size != ref_pos.size:
        raise ValueError(
            f"Direction size {u_dir.size} does not match 3N={ref_pos.size}."
        )
    u_matrix = u_dir.reshape(ref_pos.shape)
    alpha_arr = np.asarray(alphas, dtype=float)

    ref_energy = _eval_energy(
        reference, ref_pos, calculator=calculator, energy_fn=energy_fn
    )
    rel_energies = np.full(alpha_arr.shape, np.nan, dtype=float)
    directional_slopes = np.full(alpha_arr.shape, np.nan, dtype=float)
    max_forces = np.full(alpha_arr.shape, np.nan, dtype=float)

    for idx, alpha in enumerate(alpha_arr):
        pos = ref_pos + float(alpha) * u_matrix
        try:
            rel_energies[idx] = (
                _eval_energy(reference, pos, calculator=calculator, energy_fn=energy_fn)
                - ref_energy
            )
            if compute_forces:
                forces = _eval_forces(
                    reference, pos, calculator=calculator, forces_fn=forces_fn
                )
                directional_slopes[idx] = -float(np.sum(forces * u_matrix))
                max_forces[idx] = float(np.max(np.linalg.norm(forces, axis=1)))
        except Exception:
            continue

    result: Dict[str, Any] = {
        "alphas": alpha_arr,
        "direction": u_dir,
        "reference_energy": ref_energy,
        "relative_energies": rel_energies,
        "roughness": pes_roughness(alpha_arr, rel_energies),
    }
    if compute_forces:
        result["directional_slopes"] = directional_slopes
        result["max_forces"] = max_forces
    return result


def evaluate_pes_slice_2d(
    reference: Any,
    u1: np.ndarray,
    u2: np.ndarray,
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    calculator: Optional[Any] = None,
    energy_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """Evaluate a 2D PES basin slice ``E(R_0 + x u1 + y u2) - E(R_0)``.

    Parameters
    ----------
    reference : ase.Atoms or np.ndarray
        Origin structure or ``(N, 3)`` positions.
    u1, u2 : np.ndarray
        Orthonormal 1D basis vectors of shape ``(3N,)``.
    x_grid, y_grid : sequence of float
        1D coordinate arrays along ``u1`` and ``u2``.
    calculator : Any, optional
        ASE calculator (if omitted, uses ``reference.calc``).
    energy_fn : callable, optional
        Optional analytical callback ``energy_fn(positions) -> float``.

    Returns
    -------
    dict
        Contains ``x``, ``y``, ``reference_energy``, and ``relative_energies``
        with shape ``(len(y_grid), len(x_grid))``.
    """
    ref_pos = _extract_positions(reference)
    u1_mat = np.asarray(u1, dtype=float).reshape(ref_pos.shape)
    u2_mat = np.asarray(u2, dtype=float).reshape(ref_pos.shape)

    x_arr = np.asarray(x_grid, dtype=float)
    y_arr = np.asarray(y_grid, dtype=float)
    ref_energy = _eval_energy(
        reference, ref_pos, calculator=calculator, energy_fn=energy_fn
    )
    energies = np.full((y_arr.size, x_arr.size), np.nan, dtype=float)

    for iy, y_val in enumerate(y_arr):
        for ix, x_val in enumerate(x_arr):
            pos = ref_pos + float(x_val) * u1_mat + float(y_val) * u2_mat
            try:
                energies[iy, ix] = (
                    _eval_energy(
                        reference, pos, calculator=calculator, energy_fn=energy_fn
                    )
                    - ref_energy
                )
            except Exception:
                continue

    return {
        "x": x_arr,
        "y": y_arr,
        "reference_energy": ref_energy,
        "relative_energies": energies,
    }


def plot_pes_basin_2d(
    pes_slice: Dict[str, Any],
    endpoints: Optional[np.ndarray] = None,
    relaxation_path: Optional[np.ndarray] = None,
    dft_final: Optional[np.ndarray] = None,
    title: str = "Local MLFF PES Basin",
    percentile_cap: float = 92.0,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot a 2D PES contour map with attack and relaxation overlays.

    Adapted from ``declanwhchan/mlff_attack_data_collection`` (MIT License,
    Copyright (c) 2026 Declan Chan).

    Parameters
    ----------
    pes_slice : dict
        Output dictionary from :func:`evaluate_pes_slice_2d`.
    endpoints : np.ndarray, optional
        Array of shape ``(3, 2)`` containing projected ``(reference, perturbed,
        final)`` points.
    relaxation_path : np.ndarray, optional
        Array of shape ``(T, 2)`` containing projected relaxation trajectory
        frames.
    dft_final : np.ndarray, optional
        Optional ``(2,)`` projected DFT final structure coordinate.
    title : str, optional
        Figure title.
    percentile_cap : float, optional
        Upper percentile cap on relative energies for contour readability,
        by default ``92.0``.
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure object.
    """
    import matplotlib.pyplot as plt

    x_arr = np.asarray(pes_slice["x"], dtype=float)
    y_arr = np.asarray(pes_slice["y"], dtype=float)
    energies = np.asarray(pes_slice["relative_energies"], dtype=float)
    finite = energies[np.isfinite(energies)]
    if finite.size == 0:
        raise ValueError("PES slice contains no finite energies to plot.")

    cap = float(np.percentile(finite, percentile_cap))
    display = np.minimum(energies, cap)
    xx, yy = np.meshgrid(x_arr, y_arr)

    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    contour = ax.contourf(xx, yy, display, levels=64, cmap="magma")
    ax.contour(xx, yy, display, levels=8, colors="white", alpha=0.18, linewidths=0.4)

    if endpoints is not None and len(endpoints) >= 2:
        pts = np.asarray(endpoints, dtype=float)
        ax.plot(
            pts[:2, 0],
            pts[:2, 1],
            color="#A9A9A9",
            alpha=0.8,
            linewidth=1.0,
            linestyle="--",
            zorder=3,
            label="attack displacement",
        )
        labels = ("reference", "perturbed", "final relaxed")
        markers = ("o", "^", "s")
        colors = ("#35B7EB", "#FFB000", "#00A878")
        for idx in range(min(len(pts), 3)):
            ax.scatter(
                pts[idx, 0],
                pts[idx, 1],
                marker=markers[idx],
                s=68,
                color=colors[idx],
                edgecolor="white",
                linewidth=0.85,
                zorder=5,
                label=labels[idx],
            )

    if relaxation_path is not None and len(relaxation_path) >= 2:
        path_pts = np.asarray(relaxation_path, dtype=float)
        ax.plot(
            path_pts[:, 0],
            path_pts[:, 1],
            color="#D4D4D8",
            alpha=0.9,
            linewidth=1.2,
            zorder=4,
            label="relaxation trajectory",
        )

    if dft_final is not None:
        dft_pt = np.asarray(dft_final, dtype=float).reshape(-1)
        if dft_pt.size == 2 and np.all(np.isfinite(dft_pt)):
            ax.scatter(
                dft_pt[0],
                dft_pt[1],
                marker="s",
                s=74,
                color="#7C3AED",
                edgecolor="white",
                linewidth=0.9,
                zorder=7,
                label="DFT final",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Attack direction ($\AA\sqrt{\mathrm{atoms}}$)", fontsize=12)
    ax.set_ylabel(
        r"Orthogonal-to-attack direction ($\AA\sqrt{\mathrm{atoms}}$)", fontsize=12
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    cbar = fig.colorbar(contour, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Relative energy (eV)", fontsize=11)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 2. Local PES Curvature: Atomic Hessian & Gamma-Point Phonons
# ---------------------------------------------------------------------------


def compute_pes_hessian(
    reference: Any,
    delta: float = 1e-3,
    calculator: Optional[Any] = None,
    forces_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    symmetrize: bool = True,
) -> np.ndarray:
    """Compute the ``(3N, 3N)`` PES position Hessian via central force differences.

    Evaluates :math:`H_{jk} = \\frac{\\partial^2 E}{\\partial q_j \\partial q_k}
    \\approx -\\frac{F_j(\\mathbf{R} + \\delta \\mathbf{e}_k) -
    F_j(\\mathbf{R} - \\delta \\mathbf{e}_k)}{2\\delta}` in eV/Å².

    Parameters
    ----------
    reference : ase.Atoms or np.ndarray
        Structure or ``(N, 3)`` atomic positions.
    delta : float, optional
        Finite-difference step size in Angstroms, by default ``1e-3``.
    calculator : Any, optional
        ASE calculator (if omitted, uses ``reference.calc``).
    forces_fn : callable, optional
        Optional callback ``forces_fn(positions) -> (N, 3) array``.
    symmetrize : bool, optional
        Symmetrize as ``0.5 * (H + H.T)``, by default True.

    Returns
    -------
    np.ndarray
        Position Hessian matrix of shape ``(3N, 3N)`` in eV/Å².
    """
    if delta <= 0.0:
        raise ValueError("delta must be strictly positive.")

    ref_pos = _extract_positions(reference)
    n_dof = ref_pos.size
    hessian = np.zeros((n_dof, n_dof), dtype=float)

    for idx in range(n_dof):
        atom_idx, coord_idx = divmod(idx, 3)

        pos_plus = ref_pos.copy()
        pos_plus[atom_idx, coord_idx] += delta
        f_plus = _eval_forces(
            reference, pos_plus, calculator=calculator, forces_fn=forces_fn
        ).reshape(-1)

        pos_minus = ref_pos.copy()
        pos_minus[atom_idx, coord_idx] -= delta
        f_minus = _eval_forces(
            reference, pos_minus, calculator=calculator, forces_fn=forces_fn
        ).reshape(-1)

        hessian[:, idx] = -(f_plus - f_minus) / (2.0 * delta)

    if symmetrize:
        hessian = 0.5 * (hessian + hessian.T)
    return hessian


def hessian_spectrum(
    hessian: np.ndarray,
    masses: Optional[Sequence[float]] = None,
    zero_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Compute eigenspectrum and curvature summary of a PES Hessian matrix.

    If ``masses`` (shape ``(N,)``) is provided, diagonalizes the mass-weighted
    dynamical matrix :math:`D = M^{-1/2} H M^{-1/2}`.

    Parameters
    ----------
    hessian : np.ndarray
        Symmetric ``(3N, 3N)`` position Hessian matrix.
    masses : sequence of float, optional
        Optional per-atom masses of length ``N`` for mass-weighting.
    zero_tol : float, optional
        Threshold separating near-zero (e.g. translational) eigenvalues,
        by default ``1e-6``.

    Returns
    -------
    dict
        Contains ``eigenvalues``, ``eigenvectors``, ``n_negative``, ``n_zero``,
        ``n_positive``, ``min_eigenvalue``, ``max_eigenvalue``, ``trace``,
        ``frobenius_norm``, and ``condition_number``.
    """
    h_mat = np.asarray(hessian, dtype=float)
    if h_mat.ndim != 2 or h_mat.shape[0] != h_mat.shape[1]:
        raise ValueError(f"Expected square Hessian matrix, got {h_mat.shape}.")
    h_sym = 0.5 * (h_mat + h_mat.T)

    if masses is not None:
        m_arr = np.asarray(masses, dtype=float).reshape(-1)
        if m_arr.size * 3 != h_sym.shape[0]:
            raise ValueError(
                f"Expected {h_sym.shape[0] // 3} masses, got {m_arr.size}."
            )
        if np.any(m_arr <= 0.0):
            raise ValueError("All atomic masses must be positive.")
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(m_arr), 3)
        target = (inv_sqrt_m[:, None] * h_sym) * inv_sqrt_m[None, :]
    else:
        target = h_sym

    eigenvalues, eigenvectors = np.linalg.eigh(target)
    nonzero_abs = np.abs(eigenvalues[np.abs(eigenvalues) > zero_tol])
    cond = (
        float(np.max(nonzero_abs) / np.min(nonzero_abs))
        if nonzero_abs.size
        else float("nan")
    )

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "n_negative": int(np.sum(eigenvalues < -zero_tol)),
        "n_zero": int(np.sum(np.abs(eigenvalues) <= zero_tol)),
        "n_positive": int(np.sum(eigenvalues > zero_tol)),
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "trace": float(np.sum(eigenvalues)),
        "frobenius_norm": float(np.linalg.norm(target, ord="fro")),
        "condition_number": cond,
    }


def perturbation_hessian_alignment(
    hessian: np.ndarray,
    direction: np.ndarray,
    zero_tol: float = 1e-6,
) -> Dict[str, float]:
    """Measure how a perturbation direction aligns with local PES Hessian modes.

    Computes the Rayleigh quotient (directional curvature)
    :math:`\\kappa_d = \\hat{\\mathbf{d}}^\\top H \\hat{\\mathbf{d}}`, its
    relative position within ``[lambda_min, lambda_max]``, and the squared
    projection weight on the softest, stiffest, and negative-curvature
    eigenspaces.

    Parameters
    ----------
    hessian : np.ndarray
        ``(3N, 3N)`` PES Hessian matrix.
    direction : np.ndarray
        Perturbation vector of shape ``(N, 3)`` or ``(3N,)``.
    zero_tol : float, optional
        Threshold for negative-curvature subspace, by default ``1e-6``.

    Returns
    -------
    dict
        Contains ``directional_curvature``, ``relative_curvature_position``,
        ``softest_mode_alignment``, ``stiffest_mode_alignment``, and
        ``negative_subspace_weight``.
    """
    spec = hessian_spectrum(hessian, zero_tol=zero_tol)
    u_dir = _unit_vector(direction, name="direction")
    eigvals = spec["eigenvalues"]
    eigvecs = spec["eigenvectors"]

    if u_dir.size != eigvals.size:
        raise ValueError(
            f"Direction size {u_dir.size} does not match Hessian size {eigvals.size}."
        )

    projections = eigvecs.T @ u_dir
    weights = projections**2
    rayleigh = float(np.sum(weights * eigvals))

    span = float(eigvals[-1] - eigvals[0])
    rel_pos = (
        float((rayleigh - eigvals[0]) / span) if span > zero_tol else float("nan")
    )
    neg_mask = eigvals < -zero_tol

    return {
        "directional_curvature": rayleigh,
        "relative_curvature_position": rel_pos,
        "softest_mode_alignment": float(abs(projections[0])),
        "stiffest_mode_alignment": float(abs(projections[-1])),
        "negative_subspace_weight": float(np.sum(weights[neg_mask])),
    }


def gamma_phonon_modes(
    atoms: Any,
    calculator: Optional[Any] = None,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    delta: float = 0.01,
    cache_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Calculate signed :math:`\\Gamma`-point phonon energies (in meV) via ASE.

    Negative values indicate imaginary (unstable/saddle) modes. Adapted from
    ``declanwhchan/mlff_attack_data_collection`` (MIT License, Copyright (c)
    2026 Declan Chan).

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to evaluate.
    calculator : Any, optional
        ASE calculator (if omitted, uses ``atoms.calc``).
    supercell : tuple of (int, int, int), optional
        Supercell repetition tuple, by default ``(1, 1, 1)``.
    delta : float, optional
        Displacement amplitude in Angstroms, by default ``0.01``.
    cache_dir : str or Path, optional
        Directory prefix for ASE ``Phonons`` cache files. If None, uses a
        temporary directory that is cleaned up automatically.

    Returns
    -------
    np.ndarray
        Signed :math:`\\Gamma`-point phonon energies in meV.
    """
    import tempfile
    from ase.phonons import Phonons

    calc = calculator if calculator is not None else getattr(atoms, "calc", None)
    if calc is None:
        raise ValueError(
            "Provide `calculator` or an Atoms object with an attached calculator."
        )

    work_atoms = atoms.copy()
    work_atoms.calc = calc

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        phonons = Phonons(
            work_atoms,
            calc,
            supercell=supercell,
            delta=delta,
            name=str(cache_path),
        )
        phonons.run()
        phonons.read(acoustic=True)
        frequencies, _ = phonons.band_structure(
            np.array([[0.0, 0.0, 0.0]]), modes=True, verbose=False
        )
        return np.asarray(frequencies[0] * 1000.0, dtype=float)

    with tempfile.TemporaryDirectory() as tmpdir:
        phonons = Phonons(
            work_atoms,
            calc,
            supercell=supercell,
            delta=delta,
            name=str(Path(tmpdir) / "phonon"),
        )
        phonons.run()
        phonons.read(acoustic=True)
        frequencies, _ = phonons.band_structure(
            np.array([[0.0, 0.0, 0.0]]), modes=True, verbose=False
        )
        return np.asarray(frequencies[0] * 1000.0, dtype=float)


# ---------------------------------------------------------------------------
# 3. Model Loss Landscape (Parameter Space) & Parameter-PES Coupling
# ---------------------------------------------------------------------------


def _trainable_params(model: Any) -> List[Any]:
    """Return floating-point parameters of a PyTorch ``nn.Module``."""
    params = [
        p for p in model.parameters() if p.requires_grad and p.is_floating_point()
    ]
    if not params:
        params = [p for p in model.parameters() if p.is_floating_point()]
    if not params:
        raise ValueError("Model has no floating-point parameters.")
    return params


def filter_normalized_direction(
    model: Any,
    seed: Optional[int] = None,
) -> List[Any]:
    """Sample a filter-normalized random direction in parameter space.

    Implements the filter-wise normalization of Li et al. (NeurIPS 2018),
    scaling each random tensor slice to match the Frobenius/Euclidean norm of
    the corresponding parameter filter so scale invariance does not distort
    loss-landscape curvature comparisons.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model whose parameter shapes and norms define the direction.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of torch.Tensor
        Direction tensors aligned with the model's floating-point parameters.
    """
    import torch

    params = _trainable_params(model)
    rng = np.random.default_rng(seed)
    direction = []

    for param in params:
        noise_np = rng.standard_normal(size=tuple(param.shape))
        d_tensor = torch.as_tensor(
            noise_np, dtype=param.dtype, device=param.device
        )
        if param.ndim >= 2:
            # Normalize per output filter (dim 0) to match parameter filter norm
            flat_p = param.detach().reshape(param.shape[0], -1)
            flat_d = d_tensor.reshape(param.shape[0], -1)
            p_norm = torch.linalg.norm(flat_p, dim=1, keepdim=True)
            d_norm = torch.linalg.norm(flat_d, dim=1, keepdim=True).clamp(min=1e-12)
            d_tensor = (flat_d * (p_norm / d_norm)).reshape(param.shape)
        else:
            p_norm = torch.linalg.norm(param.detach())
            d_norm = torch.linalg.norm(d_tensor).clamp(min=1e-12)
            d_tensor = d_tensor * (p_norm / d_norm)
        direction.append(d_tensor)

    return direction


def _apply_parameter_offset(
    params: Sequence[Any],
    base_weights: Sequence[Any],
    offsets: Sequence[Tuple[float, Sequence[Any]]],
) -> None:
    """Set each parameter tensor to ``base + sum(coeff * d)`` in-place."""
    import torch

    with torch.no_grad():
        for idx, (param, base) in enumerate(zip(params, base_weights)):
            updated = base.clone()
            for coeff, direction in offsets:
                if coeff != 0.0:
                    updated.add_(direction[idx], alpha=float(coeff))
            param.copy_(updated)


def evaluate_loss_landscape_1d(
    model: Any,
    loss_fn: Callable[[], float],
    direction: Sequence[Any],
    alphas: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate a 1D parameter-space loss slice ``L(theta_0 + alpha * d)``.

    Original model weights are guaranteed to be restored even if ``loss_fn``
    raises an exception.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    loss_fn : callable
        Zero-argument callable returning the scalar loss (or energy/force
        error) evaluated with ``model``'s current parameters.
    direction : sequence of torch.Tensor
        Direction tensors from :func:`filter_normalized_direction`.
    alphas : sequence of float
        1D perturbation coefficients along ``direction``.

    Returns
    -------
    dict
        Contains ``alphas``, ``losses``, ``base_loss``, and ``roughness``.
    """
    params = _trainable_params(model)
    base_weights = [p.detach().clone() for p in params]
    alpha_arr = np.asarray(alphas, dtype=float)
    losses = np.full(alpha_arr.shape, np.nan, dtype=float)

    try:
        base_loss = float(loss_fn())
        for idx, alpha in enumerate(alpha_arr):
            _apply_parameter_offset(
                params, base_weights, ((float(alpha), direction),)
            )
            losses[idx] = float(loss_fn())
    finally:
        _apply_parameter_offset(params, base_weights, ())

    return {
        "alphas": alpha_arr,
        "base_loss": base_loss,
        "losses": losses,
        "roughness": pes_roughness(alpha_arr, losses),
    }


def evaluate_loss_landscape_2d(
    model: Any,
    loss_fn: Callable[[], float],
    dir1: Sequence[Any],
    dir2: Sequence[Any],
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate a 2D parameter-space loss surface ``L(theta_0 + a d1 + b d2)``.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    loss_fn : callable
        Zero-argument callable returning scalar loss under current parameters.
    dir1, dir2 : sequence of torch.Tensor
        Filter-normalized parameter directions.
    alpha_grid, beta_grid : sequence of float
        1D grids along ``dir1`` and ``dir2``.

    Returns
    -------
    dict
        Contains ``alpha``, ``beta``, ``base_loss``, and ``losses`` with shape
        ``(len(beta_grid), len(alpha_grid))``.
    """
    params = _trainable_params(model)
    base_weights = [p.detach().clone() for p in params]
    a_arr = np.asarray(alpha_grid, dtype=float)
    b_arr = np.asarray(beta_grid, dtype=float)
    losses = np.full((b_arr.size, a_arr.size), np.nan, dtype=float)

    try:
        base_loss = float(loss_fn())
        for ib, beta in enumerate(b_arr):
            for ia, alpha in enumerate(a_arr):
                _apply_parameter_offset(
                    params,
                    base_weights,
                    ((float(alpha), dir1), (float(beta), dir2)),
                )
                losses[ib, ia] = float(loss_fn())
    finally:
        _apply_parameter_offset(params, base_weights, ())

    return {
        "alpha": a_arr,
        "beta": b_arr,
        "base_loss": base_loss,
        "losses": losses,
    }


def parameter_sharpness(
    model: Any,
    loss_fn: Callable[[], float],
    rho: float = 1e-2,
    n_directions: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Estimate local parameter-space sharpness and curvature around ``theta_0``.

    Samples ``n_directions`` filter-normalized directions ``d_k`` and evaluates
    symmetric perturbations :math:`\\theta_0 \\pm \\rho d_k` to compute:

    - worst-case sharpness :math:`\\max_k \\max(\\mathcal{L}(\\theta_0 \\pm \\rho d_k)) - \\mathcal{L}(\\theta_0)`
    - central-difference directional curvature
      :math:`(\\mathcal{L}_+ + \\mathcal{L}_- - 2\\mathcal{L}_0) / \\rho^2`

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    loss_fn : callable
        Zero-argument scalar loss callable.
    rho : float, optional
        Parameter perturbation radius, by default ``1e-2``.
    n_directions : int, optional
        Number of random filter-normalized directions, by default ``10``.
    seed : int, optional
        Base random seed, by default ``42``.

    Returns
    -------
    dict
        Contains ``base_loss``, ``rho``, ``max_sharpness``, ``mean_sharpness``,
        ``mean_curvature``, ``max_curvature``, and ``curvatures``.
    """
    if rho <= 0.0:
        raise ValueError("rho must be strictly positive.")
    if n_directions <= 0:
        raise ValueError("n_directions must be a positive integer.")

    params = _trainable_params(model)
    base_weights = [p.detach().clone() for p in params]
    sharpness_vals = []
    curvatures = []

    try:
        base_loss = float(loss_fn())
        for idx in range(n_directions):
            direction = filter_normalized_direction(model, seed=seed + idx)
            _apply_parameter_offset(params, base_weights, ((rho, direction),))
            loss_plus = float(loss_fn())
            _apply_parameter_offset(params, base_weights, ((-rho, direction),))
            loss_minus = float(loss_fn())

            sharpness_vals.append(max(loss_plus, loss_minus) - base_loss)
            curvatures.append((loss_plus + loss_minus - 2.0 * base_loss) / (rho**2))
    finally:
        _apply_parameter_offset(params, base_weights, ())

    sharp_arr = np.asarray(sharpness_vals, dtype=float)
    curv_arr = np.asarray(curvatures, dtype=float)
    return {
        "base_loss": base_loss,
        "rho": float(rho),
        "max_sharpness": float(np.max(sharp_arr)),
        "mean_sharpness": float(np.mean(sharp_arr)),
        "mean_curvature": float(np.mean(curv_arr)),
        "max_curvature": float(np.max(curv_arr)),
        "curvatures": curv_arr,
    }


def pes_parameter_coupling(
    model: Any,
    eval_energy_and_forces: Callable[[], Tuple[float, np.ndarray]],
    direction_r: np.ndarray,
    rho: float = 1e-3,
    n_directions: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Quantify coupling between model parameter perturbations and PES forces.

    Perturbs model parameters along ``n_directions`` filter-normalized
    directions :math:`\\theta_0 \\pm \\rho d_k` and computes finite-difference
    parameter sensitivities of:

    - predicted energy :math:`|\\partial E_\\theta / \\partial \\alpha_k|`
    - full force field :math:`\\|\\partial \\mathbf{F}_\\theta / \\partial \\alpha_k\\|_F`
    - force projected onto the adversarial perturbation direction
      :math:`\\hat{\\mathbf{d}}_R`:
      :math:`|\\langle \\partial \\mathbf{F}_\\theta / \\partial \\alpha_k, \\hat{\\mathbf{d}}_R \\rangle|`
      (a finite-difference probe of the mixed derivative
      :math:`\\nabla_\\theta \\nabla_{\\mathbf{R}} E_\\theta \\cdot \\hat{\\mathbf{d}}_R`)
    - cosine alignment between :math:`\\partial \\mathbf{F}_\\theta / \\partial \\alpha_k`
      and :math:`\\hat{\\mathbf{d}}_R`.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch MLFF model.
    eval_energy_and_forces : callable
        Zero-argument callable returning ``(energy, forces)`` where ``forces``
        has shape ``(N, 3)`` evaluated with ``model``'s current parameters.
    direction_r : np.ndarray
        Input-space (e.g. adversarial) perturbation vector of shape ``(N, 3)``
        or ``(3N,)``.
    rho : float, optional
        Parameter-space finite-difference step size, by default ``1e-3``.
    n_directions : int, optional
        Number of filter-normalized parameter directions, by default ``5``.
    seed : int, optional
        Base random seed, by default ``42``.

    Returns
    -------
    dict
        Summary statistics and per-direction sensitivity arrays linking
        parameter-space perturbations to configuration-space PES response.
    """
    if rho <= 0.0:
        raise ValueError("rho must be strictly positive.")
    if n_directions <= 0:
        raise ValueError("n_directions must be a positive integer.")

    u_r = _unit_vector(direction_r, name="direction_r")
    params = _trainable_params(model)
    base_weights = [p.detach().clone() for p in params]

    energy_sens = []
    force_sens = []
    adv_proj_sens = []
    adv_cos_align = []

    try:
        base_energy, base_forces = eval_energy_and_forces()
        base_forces = np.asarray(base_forces, dtype=float)
        if base_forces.size != u_r.size:
            raise ValueError(
                f"direction_r size {u_r.size} does not match forces size {base_forces.size}."
            )

        for idx in range(n_directions):
            direction_theta = filter_normalized_direction(model, seed=seed + idx)
            _apply_parameter_offset(params, base_weights, ((rho, direction_theta),))
            e_plus, f_plus = eval_energy_and_forces()
            _apply_parameter_offset(params, base_weights, ((-rho, direction_theta),))
            e_minus, f_minus = eval_energy_and_forces()

            de_dalpha = (float(e_plus) - float(e_minus)) / (2.0 * rho)
            df_dalpha = (
                np.asarray(f_plus, dtype=float) - np.asarray(f_minus, dtype=float)
            ).reshape(-1) / (2.0 * rho)

            df_norm = float(np.linalg.norm(df_dalpha))
            proj = float(abs(np.dot(df_dalpha, u_r)))
            cos_align = proj / df_norm if df_norm > 1e-12 else float("nan")

            energy_sens.append(abs(de_dalpha))
            force_sens.append(df_norm)
            adv_proj_sens.append(proj)
            adv_cos_align.append(cos_align)
    finally:
        _apply_parameter_offset(params, base_weights, ())

    e_arr = np.asarray(energy_sens, dtype=float)
    f_arr = np.asarray(force_sens, dtype=float)
    p_arr = np.asarray(adv_proj_sens, dtype=float)
    c_arr = np.asarray(adv_cos_align, dtype=float)
    finite_cos = c_arr[np.isfinite(c_arr)]

    return {
        "base_energy": float(base_energy),
        "rho": float(rho),
        "mean_energy_sensitivity": float(np.mean(e_arr)),
        "mean_force_sensitivity": float(np.mean(f_arr)),
        "mean_adversarial_force_sensitivity": float(np.mean(p_arr)),
        "max_adversarial_force_sensitivity": float(np.max(p_arr)),
        "mean_adversarial_cosine_alignment": (
            float(np.mean(finite_cos)) if finite_cos.size else float("nan")
        ),
        "energy_sensitivities": e_arr,
        "force_sensitivities": f_arr,
        "adversarial_force_sensitivities": p_arr,
        "adversarial_cosine_alignments": c_arr,
    }


__all__ = [
    "build_attack_relaxation_basis",
    "compute_pes_hessian",
    "evaluate_loss_landscape_1d",
    "evaluate_loss_landscape_2d",
    "evaluate_pes_profile_1d",
    "evaluate_pes_slice_2d",
    "filter_normalized_direction",
    "gamma_phonon_modes",
    "hessian_spectrum",
    "parameter_sharpness",
    "perturbation_hessian_alignment",
    "pes_parameter_coupling",
    "pes_roughness",
    "plot_pes_basin_2d",
    "project_onto_basis",
]
