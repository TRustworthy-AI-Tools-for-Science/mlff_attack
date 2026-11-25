"""Potential energy surface (PES) and model loss-landscape diagnostics.

This module connects input-space adversarial perturbations to:

1. **Predicted PES geometry in configuration space** (:math:`\\mathbf{R}`-space):
   1D/2D PES slices along adversarial and orthogonal relaxation directions,
   PES roughness/smoothness metrics, finite-difference atomic Hessians
   :math:`H_q[E_\\theta]`, perturbation-Hessian mode alignment, and
   :math:`\\Gamma`-point phonon stability.
2. **Model loss-landscape geometry in parameter space** (:math:`\\theta`-space)
   and **cross-space coupling** (:math:`\\theta \\leftrightarrow \\mathbf{R}`):
   thin adapters delegating directly to
   `oxford-loss-landscapes <https://github.com/TRustworthy-AI-Tools-for-Science/oxford-loss-landscapes>`_
   (``random_line``, ``random_plane``, ``hessian_plane``,
   ``min_max_hessian_eigs_vrpca``, ``ModelWrapper``, and ``ModelParameters``
   filter/layer/model normalization), plus parameter-to-PES force coupling.

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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

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
# 3. Model Loss Landscape (oxford-loss-landscapes integration) & PES Coupling
# ---------------------------------------------------------------------------


def _require_oxford_loss_landscapes():
    """Import ``oxford_loss_landscapes`` or raise an actionable ImportError."""
    try:
        import oxford_loss_landscapes as oll
    except ImportError as exc:
        raise ImportError(
            "Parameter-space loss landscape analysis requires "
            "`oxford-loss-landscapes`. Install it with:\n"
            '    pip install -e ".[landscape]"\n'
            "or directly from GitHub:\n"
            "    pip install git+https://github.com/"
            "TRustworthy-AI-Tools-for-Science/oxford-loss-landscapes.git"
        ) from exc
    return oll


def extract_torch_module(calculator_or_module: Any) -> Any:
    """Extract the underlying PyTorch ``nn.Module`` from an MLFF calculator.

    Supports:
    - MACE calculators (``calc.models[0]``)
    - UMA / ``FAIRChemCalculator`` (``calc.predictor.model``)
    - CHGNet calculators (``calc.model``)
    - Raw ``torch.nn.Module`` or ``oxford_loss_landscapes.ModelWrapper``

    Parameters
    ----------
    calculator_or_module : Any
        ASE calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.

    Returns
    -------
    Any
        Underlying PyTorch module (or ``ModelWrapper`` if already wrapped).
    """
    if hasattr(calculator_or_module, "get_module_parameters"):
        return calculator_or_module
    if hasattr(calculator_or_module, "models") and calculator_or_module.models:
        return calculator_or_module.models[0]
    predictor = getattr(calculator_or_module, "predictor", None)
    if predictor is not None and hasattr(predictor, "model"):
        return predictor.model
    if hasattr(calculator_or_module, "model") and hasattr(
        calculator_or_module.model, "parameters"
    ):
        return calculator_or_module.model
    if hasattr(calculator_or_module, "parameters"):
        return calculator_or_module
    raise TypeError(
        "Could not extract a PyTorch module from object of type "
        f"{type(calculator_or_module).__name__}."
    )


def invalidate_calculator_cache(calculator: Any) -> None:
    """Clear cached ASE calculator results after in-place weight changes.

    ASE calculators cache energies and forces in ``calc.results`` keyed only
    by atomic positions and cell (``check_state(atoms)``). When perturbing
    MLFF parameters in-place with fixed atomic coordinates, the cache must be
    invalidated before every evaluation so the perturbed weights are actually
    used.

    Parameters
    ----------
    calculator : Any
        ASE calculator instance (no-op if ``None``).
    """
    if calculator is None:
        return
    reset_fn = getattr(calculator, "reset", None)
    if callable(reset_fn):
        reset_fn()
    results = getattr(calculator, "results", None)
    if isinstance(results, dict):
        results.clear()
    if hasattr(calculator, "atoms"):
        calculator.atoms = None


def _is_ase_calculator_like(obj: Any) -> bool:
    """Return True if ``obj`` looks like an ASE calculator rather than a raw nn.Module."""
    return (
        hasattr(obj, "get_potential_energy")
        or hasattr(obj, "calculate")
        or isinstance(getattr(obj, "results", None), dict)
    )


def _snapshot_wrapper_params(wrapper: Any) -> list:
    """Clone current wrapper parameters for safe in-place restoration."""
    if hasattr(wrapper, "snapshot_parameters"):
        return wrapper.snapshot_parameters()
    return [p.detach().clone() for p in wrapper.get_module_parameters()]


def _restore_wrapper_params(wrapper: Any, snapshot: list) -> None:
    """Restore wrapper parameters from ``snapshot`` and invalidate calculator cache."""
    if hasattr(wrapper, "restore_parameters"):
        wrapper.restore_parameters(snapshot)
    else:
        import torch

        params = wrapper.get_module_parameters()
        with torch.no_grad():
            for cur_p, saved_p in zip(params, snapshot):
                cur_p.copy_(saved_p)
    if hasattr(wrapper, "invalidate_cache"):
        wrapper.invalidate_cache()


def wrap_mlff_model(
    calculator_or_module: Any,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    trainable_only: bool = False,
) -> Any:
    """Wrap an MLFF calculator or PyTorch module for ``oxford_loss_landscapes``.

    Returns an ``MLFFModelWrapper`` (a subclass of
    :class:`oxford_loss_landscapes.ModelWrapper`) that:

    - extracts the underlying PyTorch ``nn.Module`` from MACE, UMA, or CHGNet
      calculators,
    - keeps a reference to the original ASE ``calculator`` and exposes
      :meth:`invalidate_cache` so fixed-structure evaluations never return
      stale cached energies/forces,
    - filters out non-floating-point parameters (and frozen parameters when
      ``trainable_only=True``), and
    - restores parameters in-place via :meth:`snapshot_parameters` /
      :meth:`restore_parameters` without ``copy.deepcopy`` (which fails on
      ``e3nn`` / TorchScript modules and detaches ASE calculators).

    Parameters
    ----------
    calculator_or_module : Any
        MACE/UMA/CHGNet calculator, ``torch.nn.Module``, or ``ModelWrapper``.
    forward_fn : callable, optional
        Optional ``forward_fn(model, x)`` for custom forward passes.
    trainable_only : bool, optional
        If True, restrict parameter perturbations to tensors with
        ``requires_grad=True``, by default False.

    Returns
    -------
    oxford_loss_landscapes.ModelWrapper
        Wrapped model ready for ``oxford_loss_landscapes`` routines.
    """
    import torch

    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.model_interface.model_parameters import ModelParameters

    if isinstance(calculator_or_module, oll.ModelWrapper):
        return calculator_or_module

    module = extract_torch_module(calculator_or_module)
    if isinstance(module, oll.ModelWrapper):
        return module

    calculator = (
        calculator_or_module
        if (
            calculator_or_module is not module
            or _is_ase_calculator_like(calculator_or_module)
        )
        else None
    )

    class MLFFModelWrapper(oll.ModelWrapper):
        """ASE/MLFF-aware ``oxford_loss_landscapes.ModelWrapper``."""

        def __init__(
            self,
            torch_module: Any,
            calc: Optional[Any],
            fwd_fn: Optional[Callable[[Any, Any], Any]],
            only_trainable: bool,
        ):
            super().__init__([torch_module])
            self.model = torch_module
            self.calculator = calc
            self.forward_fn = fwd_fn
            self.trainable_only = bool(only_trainable)

        def invalidate_cache(self) -> None:
            invalidate_calculator_cache(self.calculator)

        def get_module_parameters(
            self, trainable_only: Optional[bool] = None
        ) -> Any:
            use_trainable = (
                self.trainable_only if trainable_only is None else bool(trainable_only)
            )
            params = [
                p
                for mod in self.modules
                for p in mod.parameters()
                if p.is_floating_point() and (not use_trainable or p.requires_grad)
            ]
            return ModelParameters(params)

        def snapshot_parameters(self) -> list:
            return [p.detach().clone() for p in self.get_module_parameters()]

        def restore_parameters(self, snapshot: list) -> None:
            params = self.get_module_parameters()
            with torch.no_grad():
                for param, saved in zip(params, snapshot):
                    param.copy_(saved)
            self.invalidate_cache()

        def forward(self, x: Any) -> Any:
            if self.forward_fn is not None:
                return self.forward_fn(self.model, x)
            return self.model(x)

    return MLFFModelWrapper(module, calculator, forward_fn, trainable_only)


def make_mlff_metric(
    eval_fn: Callable[[Any], float],
    calculator: Optional[Any] = None,
) -> Any:
    """Adapt an MLFF evaluation callable into an ``oxford_loss_landscapes.Metric``.

    Automatically invalidates the ASE calculator cache before every evaluation
    so weight perturbations are never masked by stale cached energies/forces.

    Parameters
    ----------
    eval_fn : callable
        Callable accepting an ``oxford_loss_landscapes.ModelWrapper`` and
        returning a scalar ``float`` loss, energy, or force-error metric.
    calculator : Any, optional
        Optional ASE calculator whose cache should be invalidated prior to
        each evaluation (in addition to ``model_wrapper.invalidate_cache()``).

    Returns
    -------
    oxford_loss_landscapes.metrics.metric.Metric
        Metric instance usable with ``random_line``, ``random_plane``,
        ``planar_interpolation``, ``linear_interpolation``, and
        ``hessian_plane``.
    """
    _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    class _MLFFCallableMetric(OllMetric):
        def __init__(self, fn: Callable[[Any], float], calc: Optional[Any]):
            super().__init__()
            self._fn = fn
            self._calculator = calc

        def __call__(self, model_wrapper: Any) -> float:
            invalidate_calculator_cache(self._calculator)
            if hasattr(model_wrapper, "invalidate_cache"):
                model_wrapper.invalidate_cache()
            return float(self._fn(model_wrapper))

    return _MLFFCallableMetric(eval_fn, calculator)


def _as_atoms_sequence(atoms_or_list: Any) -> list:
    """Normalize a single ASE ``Atoms`` or sequence of ``Atoms`` into a list."""
    if hasattr(atoms_or_list, "get_positions"):
        return [atoms_or_list]
    return list(atoms_or_list)


def _resolve_metric_calculator(
    atoms: Any,
    model_wrapper: Any,
    explicit_calculator: Optional[Any],
) -> Any:
    calc = (
        explicit_calculator
        if explicit_calculator is not None
        else getattr(model_wrapper, "calculator", None)
    )
    if calc is None:
        calc = getattr(atoms, "calc", None)
    if calc is None:
        raise ValueError(
            "Provide `calculator`, wrap an ASE calculator with `wrap_mlff_model`, "
            "or attach a calculator to the input `Atoms` objects."
        )
    return calc


def make_energy_loss_metric(
    atoms_list: Any,
    target_energies: Union[float, Sequence[float]],
    calculator: Optional[Any] = None,
) -> Any:
    """Build an ``oxford_loss_landscapes.Metric`` computing energy MSE over structures.

    Parameters
    ----------
    atoms_list : ase.Atoms or sequence of ase.Atoms
        Structure(s) to evaluate at each parameter-space grid point.
    target_energies : float or sequence of float
        Reference/target potential energy (eV) for each structure.
    calculator : Any, optional
        ASE calculator (defaults to ``model_wrapper.calculator`` or ``atoms.calc``).

    Returns
    -------
    oxford_loss_landscapes.metrics.metric.Metric
        Metric returning mean squared energy error :math:`\\frac{1}{M}\\sum_i (E_\\theta(\\mathbf{R}_i) - E_i^*)^2`.
    """
    structures = [a.copy() for a in _as_atoms_sequence(atoms_list)]
    targets = np.asarray(target_energies, dtype=float).reshape(-1)
    if len(structures) != targets.size:
        raise ValueError(
            f"Number of structures ({len(structures)}) must match "
            f"target_energies length ({targets.size})."
        )

    def _eval(model_wrapper: Any) -> float:
        sq_errors = []
        for atoms, target_e in zip(structures, targets):
            calc = _resolve_metric_calculator(atoms, model_wrapper, calculator)
            invalidate_calculator_cache(calc)
            atoms.calc = calc
            pred_e = float(atoms.get_potential_energy())
            sq_errors.append((pred_e - float(target_e)) ** 2)
        return float(np.mean(sq_errors))

    return make_mlff_metric(_eval, calculator=calculator)


def make_force_loss_metric(
    atoms_list: Any,
    target_forces: Union[np.ndarray, Sequence[np.ndarray]],
    calculator: Optional[Any] = None,
) -> Any:
    """Build an ``oxford_loss_landscapes.Metric`` computing force MSE over structures.

    Parameters
    ----------
    atoms_list : ase.Atoms or sequence of ase.Atoms
        Structure(s) to evaluate at each parameter-space grid point.
    target_forces : np.ndarray or sequence of np.ndarray
        Reference force array ``(N, 3)`` (or one per structure) in eV/Å.
    calculator : Any, optional
        ASE calculator (defaults to ``model_wrapper.calculator`` or ``atoms.calc``).

    Returns
    -------
    oxford_loss_landscapes.metrics.metric.Metric
        Metric returning mean squared force component error across all atoms.
    """
    structures = [a.copy() for a in _as_atoms_sequence(atoms_list)]
    if len(structures) == 1 and isinstance(target_forces, np.ndarray) and target_forces.ndim == 2:
        force_targets = [np.asarray(target_forces, dtype=float)]
    else:
        force_targets = [np.asarray(f, dtype=float) for f in target_forces]
    if len(structures) != len(force_targets):
        raise ValueError(
            f"Number of structures ({len(structures)}) must match "
            f"target_forces count ({len(force_targets)})."
        )

    def _eval(model_wrapper: Any) -> float:
        sq_errors = []
        for atoms, target_f in zip(structures, force_targets):
            calc = _resolve_metric_calculator(atoms, model_wrapper, calculator)
            invalidate_calculator_cache(calc)
            atoms.calc = calc
            pred_f = np.asarray(atoms.get_forces(), dtype=float)
            sq_errors.append(float(np.mean((pred_f - target_f) ** 2)))
        return float(np.mean(sq_errors))

    return make_mlff_metric(_eval, calculator=calculator)


def make_adversarial_gap_metric(
    atoms_reference: Any,
    atoms_perturbed: Any,
    calculator: Optional[Any] = None,
) -> Any:
    """Build a ``Metric`` tracking the adversarial energy gap over parameter space.

    Evaluates :math:`\\Delta E_\\theta = E_\\theta(\\mathbf{R}_{\\mathrm{adv}}) - E_\\theta(\\mathbf{R}_0)`
    as model weights :math:`\\theta` move along random or Hessian directions,
    directly linking input-space adversarial perturbations to parameter-space
    loss-landscape geometry.

    Parameters
    ----------
    atoms_reference : ase.Atoms
        Unperturbed reference structure :math:`\\mathbf{R}_0`.
    atoms_perturbed : ase.Atoms
        Adversarially perturbed structure :math:`\\mathbf{R}_{\\mathrm{adv}}`.
    calculator : Any, optional
        ASE calculator (defaults to ``model_wrapper.calculator`` or ``atoms_reference.calc``).

    Returns
    -------
    oxford_loss_landscapes.metrics.metric.Metric
        Metric returning :math:`E_\\theta(\\mathbf{R}_{\\mathrm{adv}}) - E_\\theta(\\mathbf{R}_0)` in eV.
    """
    ref_copy = atoms_reference.copy()
    pert_copy = atoms_perturbed.copy()

    def _eval(model_wrapper: Any) -> float:
        calc = _resolve_metric_calculator(ref_copy, model_wrapper, calculator)
        invalidate_calculator_cache(calc)
        ref_copy.calc = calc
        e_ref = float(ref_copy.get_potential_energy())
        invalidate_calculator_cache(calc)
        pert_copy.calc = calc
        e_pert = float(pert_copy.get_potential_energy())
        return e_pert - e_ref

    return make_mlff_metric(_eval, calculator=calculator)


def make_mace_loss_callable(
    atoms_or_list: Any,
    calculator: Any,
    target_energies: Optional[Union[float, Sequence[float]]] = None,
    target_forces: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    force_weight: float = 0.0,
    device: Optional[Any] = None,
) -> Tuple[Callable[[Optional[Sequence[int]]], Any], int]:
    """Build a differentiable MACE ``loss_callable`` for VR-PCA Hessian analysis.

    Converts one or more ASE ``Atoms`` structures into MACE ``AtomicData`` graph
    batches using the calculator's ``r_max``, ``z_table``, and ``head`` settings,
    returning ``(loss_callable, dataset_size)`` suitable for
    :func:`evaluate_hessian_loss_plane` and :func:`compute_parameter_hessian_eigs`.

    Parameters
    ----------
    atoms_or_list : ase.Atoms or sequence of ase.Atoms
        Input structure(s).
    calculator : Any
        A ``MACECalculator`` instance (with ``calculator.models[0]``).
    target_energies : float or sequence of float, optional
        Reference potential energies (eV). If both ``target_energies`` and
        ``target_forces`` are omitted, returns the sum of predicted energies
        (useful for inspecting the Hessian of :math:`E_\\theta` itself).
    target_forces : np.ndarray or sequence of np.ndarray, optional
        Reference forces ``(N, 3)`` (eV/Å) when ``force_weight > 0``.
    force_weight : float, optional
        Weight multiplying the force MSE term, by default ``0.0``.
    device : torch.device or str, optional
        Target PyTorch device (defaults to the MACE model's parameter device).

    Returns
    -------
    tuple of (callable, int)
        ``(loss_callable, dataset_size)`` where ``loss_callable(batch_indices)``
        returns a scalar differentiable PyTorch loss tensor.
    """
    import torch
    from mace import data as mace_data
    from mace.tools import torch_geometric

    model = extract_torch_module(calculator)
    first_param = next(model.parameters())
    target_device = torch.device(device) if device is not None else first_param.device
    target_dtype = first_param.dtype

    structures = _as_atoms_sequence(atoms_or_list)
    n_samples = len(structures)
    if n_samples == 0:
        raise ValueError("atoms_or_list must contain at least one structure.")

    r_max = float(getattr(calculator, "r_max", model.r_max.item()))
    z_table = getattr(calculator, "z_table", None)
    if z_table is None:
        from mace.tools import AtomicNumberTable

        z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers.tolist()])

    import inspect

    heads = getattr(model, "heads", ["Default"])
    head_name = getattr(calculator, "head", None) or (heads[0] if heads else "Default")
    from_config_params = inspect.signature(mace_data.AtomicData.from_config).parameters

    data_list = []
    for atoms in structures:
        cfg = mace_data.config_from_atoms(atoms)
        if hasattr(cfg, "head"):
            cfg.head = head_name
        fc_kwargs: Dict[str, Any] = {"z_table": z_table, "cutoff": r_max}
        if "heads" in from_config_params:
            fc_kwargs["heads"] = heads
        atomic_data = mace_data.AtomicData.from_config(cfg, **fc_kwargs)
        data_list.append(atomic_data)

    e_targets = (
        torch.as_tensor(target_energies, dtype=target_dtype, device=target_device).reshape(-1)
        if target_energies is not None
        else None
    )
    if e_targets is not None and e_targets.numel() != n_samples:
        raise ValueError(
            f"target_energies length ({e_targets.numel()}) must match "
            f"number of structures ({n_samples})."
        )

    f_targets = None
    if force_weight > 0.0 and target_forces is not None:
        if n_samples == 1 and isinstance(target_forces, np.ndarray) and target_forces.ndim == 2:
            raw_f = [target_forces]
        else:
            raw_f = list(target_forces)
        if len(raw_f) != n_samples:
            raise ValueError(
                f"target_forces count ({len(raw_f)}) must match "
                f"number of structures ({n_samples})."
            )
        f_targets = [
            torch.as_tensor(arr, dtype=target_dtype, device=target_device)
            for arr in raw_f
        ]

    compute_force = bool(force_weight > 0.0 and f_targets is not None)

    def loss_callable(batch_indices: Optional[Sequence[int]] = None) -> Any:
        if batch_indices is None:
            idxs = list(range(n_samples))
        else:
            idxs = [int(i) for i in batch_indices]

        batch = torch_geometric.batch.Batch.from_data_list(
            [data_list[i] for i in idxs]
        ).to(target_device)
        batch_dict = batch.to_dict()
        for key, val in batch_dict.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                batch_dict[key] = val.to(dtype=target_dtype)

        # Pass training=compute_force so MACE retains create_graph=True on
        # out["forces"] = -grad_q E_theta(q) when differentiating force losses wrt theta.
        out = model(batch_dict, training=compute_force, compute_force=compute_force)
        pred_e = out["energy"].reshape(-1)

        if e_targets is None and not compute_force:
            return pred_e.sum()

        loss = torch.zeros((), dtype=target_dtype, device=target_device)
        if e_targets is not None:
            loss = loss + torch.mean((pred_e - e_targets[idxs]) ** 2)
        if compute_force and f_targets is not None:
            ref_f = torch.cat([f_targets[i] for i in idxs], dim=0)
            loss = loss + float(force_weight) * torch.mean((out["forces"] - ref_f) ** 2)
        return loss

    return loss_callable, n_samples


def _normalize_oll_direction(
    direction: Any,
    ref_point: Any,
    normalization: Optional[str],
) -> Any:
    """Apply ``oxford_loss_landscapes`` parameter normalization in-place."""
    if normalization == "filter":
        direction.filter_normalize_(ref_point)
    elif normalization == "layer":
        direction.layer_normalize_(ref_point)
    elif normalization == "model":
        direction.model_normalize_(ref_point)
    elif normalization is None:
        pass
    else:
        raise ValueError(
            "normalization must be 'filter', 'layer', 'model', or None."
        )
    return direction


def _call_oll_with_restore(fn: Callable[..., Any], *args: Any, restore_model: bool, **kwargs: Any) -> Any:
    """Call an ``oxford_loss_landscapes`` function passing ``restore_model`` if supported."""
    import inspect

    sig = inspect.signature(fn)
    if "restore_model" in sig.parameters:
        kwargs["restore_model"] = restore_model
    return fn(*args, **kwargs)


def evaluate_loss_landscape_1d(
    model: Any,
    eval_fn: Callable[[Any], float],
    distance: float = 0.1,
    steps: int = 51,
    normalization: Optional[str] = "filter",
    deepcopy_model: bool = False,
    restore_model: bool = True,
    trainable_only: bool = False,
) -> Dict[str, Any]:
    """Evaluate a 1D random parameter-space loss slice via ``oxford_loss_landscapes``.

    Delegates directly to :func:`oxford_loss_landscapes.random_line` and
    computes curvature roughness along the slice. Defaults to
    ``deepcopy_model=False, restore_model=True`` so ASE calculators and
    ``e3nn`` / TorchScript MLFF modules are perturbed safely in-place and
    restored identically afterwards.

    Parameters
    ----------
    model : Any
        MLFF calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.
    eval_fn : callable
        Callable ``eval_fn(model_wrapper) -> float`` (or an existing
        ``oxford_loss_landscapes.metrics.metric.Metric``).
    distance : float, optional
        Perturbation distance passed to ``oll.random_line``, by default ``0.1``.
    steps : int, optional
        Number of grid points along the line, by default ``51``.
    normalization : {'filter', 'layer', 'model', None}, optional
        Direction normalization scheme, by default ``'filter'``.
    deepcopy_model : bool, optional
        Deep-copy the model before perturbing weights, by default False
        (recommended for MLFF calculators / ``e3nn`` modules).
    restore_model : bool, optional
        Restore original parameters in-place when ``deepcopy_model=False``,
        by default True.
    trainable_only : bool, optional
        Perturb only parameters with ``requires_grad=True``, by default False.

    Returns
    -------
    dict
        Contains ``alphas``, ``losses``, ``distance``, ``steps``, and
        ``roughness``.
    """
    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    wrapper = wrap_mlff_model(model, trainable_only=trainable_only)
    metric = eval_fn if isinstance(eval_fn, OllMetric) else make_mlff_metric(eval_fn)
    snapshot = _snapshot_wrapper_params(wrapper) if (not deepcopy_model and restore_model) else None
    try:
        losses = np.asarray(
            _call_oll_with_restore(
                oll.random_line,
                wrapper,
                metric=metric,
                distance=distance,
                steps=steps,
                normalization=normalization,
                deepcopy_model=deepcopy_model,
                restore_model=restore_model,
            ),
            dtype=float,
        )
    finally:
        if snapshot is not None:
            _restore_wrapper_params(wrapper, snapshot)
    alphas = np.linspace(-float(distance), float(distance), losses.size)
    return {
        "alphas": alphas,
        "losses": losses,
        "distance": float(distance),
        "steps": int(steps),
        "roughness": pes_roughness(alphas, losses),
    }


def evaluate_loss_landscape_2d(
    model: Any,
    eval_fn: Callable[[Any], float],
    distance: float = 1.0,
    steps: int = 21,
    normalization: Optional[str] = "filter",
    deepcopy_model: bool = False,
    restore_model: bool = True,
    trainable_only: bool = False,
    use_ray: bool = False,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a 2D random parameter-space loss surface via ``oxford_loss_landscapes``.

    Delegates directly to :func:`oxford_loss_landscapes.random_plane`.
    Defaults to ``deepcopy_model=False, restore_model=True`` for MLFF safety.

    Parameters
    ----------
    model : Any
        MLFF calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.
    eval_fn : callable
        Callable ``eval_fn(model_wrapper) -> float`` or ``Metric``.
    distance : float, optional
        Maximum perturbation distance along each axis, by default ``1.0``.
    steps : int, optional
        Grid resolution (``steps x steps``), by default ``21``.
    normalization : {'filter', 'layer', 'model', None}, optional
        Direction normalization scheme, by default ``'filter'``.
    deepcopy_model : bool, optional
        Deep-copy the model before perturbing weights, by default False.
    restore_model : bool, optional
        Restore original parameters in-place when ``deepcopy_model=False``,
        by default True.
    trainable_only : bool, optional
        Perturb only parameters with ``requires_grad=True``, by default False.
    use_ray : bool, optional
        Enable Ray parallel evaluation in ``oxford_loss_landscapes``,
        by default False.
    num_workers : int, optional
        Number of Ray workers when ``use_ray=True``.

    Returns
    -------
    dict
        Contains ``alpha``, ``beta``, ``losses`` (shape ``(steps, steps)``),
        ``distance``, and ``steps``.
    """
    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    wrapper = wrap_mlff_model(model, trainable_only=trainable_only)
    metric = eval_fn if isinstance(eval_fn, OllMetric) else make_mlff_metric(eval_fn)
    snapshot = _snapshot_wrapper_params(wrapper) if (not deepcopy_model and restore_model) else None
    try:
        losses = np.asarray(
            _call_oll_with_restore(
                oll.random_plane,
                wrapper,
                metric=metric,
                distance=distance,
                steps=steps,
                normalization=normalization,
                deepcopy_model=deepcopy_model,
                use_ray=use_ray,
                num_workers=num_workers,
                restore_model=restore_model,
            ),
            dtype=float,
        )
    finally:
        if snapshot is not None:
            _restore_wrapper_params(wrapper, snapshot)
    axis = np.linspace(-float(distance), float(distance), int(steps))
    return {
        "alpha": axis,
        "beta": axis,
        "losses": losses,
        "distance": float(distance),
        "steps": int(steps),
    }


def evaluate_hessian_loss_plane(
    model: Any,
    eval_fn: Callable[[Any], float],
    criterion: Optional[Any] = None,
    inputs: Any = None,
    targets: Any = None,
    distance: float = 1.0,
    steps: int = 21,
    deepcopy_model: bool = False,
    restore_model: bool = True,
    trainable_only: bool = False,
    vrpca_config: Optional[Any] = None,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    loss_callable: Optional[Callable[[Optional[Sequence[int]]], Any]] = None,
    dataset_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a 2D loss plane along the dominant parameter Hessian eigenvector.

    Delegates directly to :func:`oxford_loss_landscapes.hessian_plane`, which
    computes the top parameter-Hessian eigenvector via VR-PCA and pairs it
    with a random orthogonal direction. Supports MLFF graph/dict batches via
    ``forward_fn`` or ``loss_callable`` (e.g. from :func:`make_mace_loss_callable`).

    Parameters
    ----------
    model : Any
        MLFF calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.
    eval_fn : callable
        Callable ``eval_fn(model_wrapper) -> float`` or ``Metric``.
    criterion : callable, optional
        Differentiable PyTorch loss criterion (optional if ``loss_callable`` is set).
    inputs, targets : Any, optional
        Inputs and targets passed to ``oll.hessian_plane``.
    distance : float, optional
        Maximum distance from center along each axis, by default ``1.0``.
    steps : int, optional
        Grid resolution (``steps x steps``), by default ``21``.
    deepcopy_model : bool, optional
        Deep-copy the model before perturbing weights, by default False.
    restore_model : bool, optional
        Restore original parameters in-place when ``deepcopy_model=False``,
        by default True.
    trainable_only : bool, optional
        Perturb only parameters with ``requires_grad=True``, by default False.
    vrpca_config : VRPCAConfig, optional
        Optional ``oxford_loss_landscapes.VRPCAConfig`` instance.
    forward_fn : callable, optional
        Custom ``forward_fn(model, batch)`` for graph/dict MLFF inputs.
    loss_callable : callable, optional
        Differentiable ``loss_callable(batch_indices) -> scalar Tensor`` (e.g.
        from :func:`make_mace_loss_callable`).
    dataset_size : int, optional
        Dataset size paired with ``loss_callable`` or dict/graph inputs.

    Returns
    -------
    dict
        Contains ``alpha``, ``beta``, ``losses``, ``distance``, and ``steps``.
    """
    import inspect

    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    wrapper = wrap_mlff_model(model, forward_fn=forward_fn, trainable_only=trainable_only)
    metric = eval_fn if isinstance(eval_fn, OllMetric) else make_mlff_metric(eval_fn)

    sig = inspect.signature(oll.hessian_plane)
    extra_kwargs: Dict[str, Any] = {}
    if "forward_fn" in sig.parameters and forward_fn is not None:
        extra_kwargs["forward_fn"] = forward_fn
    if "loss_callable" in sig.parameters and loss_callable is not None:
        extra_kwargs["loss_callable"] = loss_callable
    if "dataset_size" in sig.parameters and dataset_size is not None:
        extra_kwargs["dataset_size"] = dataset_size

    snapshot = _snapshot_wrapper_params(wrapper) if (not deepcopy_model and restore_model) else None
    try:
        losses = np.asarray(
            _call_oll_with_restore(
                oll.hessian_plane,
                wrapper,
                metric=metric,
                criterion=criterion,
                inputs=inputs,
                targets=targets,
                distance=distance,
                steps=steps,
                deepcopy_model=deepcopy_model,
                vrpca_config=vrpca_config,
                restore_model=restore_model,
                **extra_kwargs,
            ),
            dtype=float,
        )
    finally:
        if snapshot is not None:
            _restore_wrapper_params(wrapper, snapshot)
    axis = np.linspace(-float(distance), float(distance), int(steps))
    return {
        "alpha": axis,
        "beta": axis,
        "losses": losses,
        "distance": float(distance),
        "steps": int(steps),
    }


def compute_parameter_hessian_eigs(
    model: Any,
    criterion: Optional[Any] = None,
    inputs: Any = None,
    targets: Any = None,
    vrpca_config: Optional[Any] = None,
    compute_min: bool = False,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    loss_callable: Optional[Callable[[Optional[Sequence[int]]], Any]] = None,
    dataset_size: Optional[int] = None,
) -> Any:
    """Estimate min/max parameter-space Hessian eigenpairs via ``oxford_loss_landscapes``.

    Delegates directly to :func:`oxford_loss_landscapes.min_max_hessian_eigs_vrpca`,
    forwarding ``forward_fn``, ``loss_callable``, and ``dataset_size`` for
    graph/dict MLFF models.
    """
    import inspect

    oll = _require_oxford_loss_landscapes()
    module = extract_torch_module(model)
    if isinstance(module, oll.ModelWrapper):
        module = module.modules[0]

    sig = inspect.signature(oll.min_max_hessian_eigs_vrpca)
    extra_kwargs: Dict[str, Any] = {}
    if "compute_min" in sig.parameters:
        extra_kwargs["compute_min"] = compute_min
    if "forward_fn" in sig.parameters and forward_fn is not None:
        extra_kwargs["forward_fn"] = forward_fn
    if "loss_callable" in sig.parameters and loss_callable is not None:
        extra_kwargs["loss_callable"] = loss_callable
    if "dataset_size" in sig.parameters and dataset_size is not None:
        extra_kwargs["dataset_size"] = dataset_size

    return oll.min_max_hessian_eigs_vrpca(
        net=module,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=vrpca_config,
        **extra_kwargs,
    )


def parameter_sharpness(
    model: Any,
    eval_fn: Callable[[Any], float],
    rho: float = 1e-2,
    n_directions: int = 10,
    normalization: Optional[str] = "filter",
    seed: int = 42,
    trainable_only: bool = False,
) -> Dict[str, Any]:
    """Estimate local parameter-space sharpness using ``oxford_loss_landscapes`` directions.

    Samples ``n_directions`` normalized parameter directions via
    :func:`oxford_loss_landscapes.model_interface.model_parameters.rand_n_like`
    and evaluates symmetric perturbations :math:`\\theta_0 \\pm \\rho d_k`,
    invalidating any ASE calculator cache before every evaluation and restoring
    the original weights in a ``finally`` block.

    Parameters
    ----------
    model : Any
        MLFF calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.
    eval_fn : callable
        Callable ``eval_fn(model_wrapper) -> float``.
    rho : float, optional
        Perturbation scale, by default ``1e-2``.
    n_directions : int, optional
        Number of random parameter directions, by default ``10``.
    normalization : {'filter', 'layer', 'model', None}, optional
        Direction normalization passed to ``ModelParameters``, by default
        ``'filter'``.
    seed : int, optional
        Random seed for PyTorch direction sampling, by default ``42``.
    trainable_only : bool, optional
        Perturb only parameters with ``requires_grad=True``, by default False.

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

    import torch

    _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.model_interface.model_parameters import (
        ModelParameters,
        rand_n_like,
    )

    wrapper = wrap_mlff_model(model, trainable_only=trainable_only)
    params = wrapper.get_module_parameters()
    base_snapshot_list = _snapshot_wrapper_params(wrapper)
    base_snapshot = ModelParameters(base_snapshot_list)

    sharpness_vals = []
    curvatures = []

    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        try:
            if hasattr(wrapper, "invalidate_cache"):
                wrapper.invalidate_cache()
            base_loss = float(eval_fn(wrapper))
            for _ in range(n_directions):
                direction = rand_n_like(params)
                _normalize_oll_direction(direction, base_snapshot, normalization)
                step_vec = direction * float(rho)

                params.add_(step_vec)
                if hasattr(wrapper, "invalidate_cache"):
                    wrapper.invalidate_cache()
                loss_plus = float(eval_fn(wrapper))

                params.sub_(step_vec * 2.0)
                if hasattr(wrapper, "invalidate_cache"):
                    wrapper.invalidate_cache()
                loss_minus = float(eval_fn(wrapper))

                # Restore exact baseline parameters before next direction
                _restore_wrapper_params(wrapper, base_snapshot_list)

                sharpness_vals.append(max(loss_plus, loss_minus) - base_loss)
                curvatures.append(
                    (loss_plus + loss_minus - 2.0 * base_loss) / (rho**2)
                )
        finally:
            _restore_wrapper_params(wrapper, base_snapshot_list)

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
    eval_energy_and_forces: Callable[[Any], Tuple[float, np.ndarray]],
    direction_r: np.ndarray,
    rho: float = 1e-3,
    n_directions: int = 5,
    normalization: Optional[str] = "filter",
    seed: int = 42,
    trainable_only: bool = False,
) -> Dict[str, Any]:
    """Quantify coupling between parameter perturbations and PES forces.

    Uses ``oxford_loss_landscapes`` ``ModelParameters`` direction sampling and
    normalization to perturb :math:`\\theta_0 \\pm \\rho d_k`, invalidating any
    ASE calculator cache before each evaluation, then measures:

    - predicted energy sensitivity :math:`|\\partial E_\\theta / \\partial \\alpha_k|`
    - full force sensitivity :math:`\\|\\partial \\mathbf{F}_\\theta / \\partial \\alpha_k\\|_F`
    - adversarial-direction force sensitivity
      :math:`|\\langle \\partial \\mathbf{F}_\\theta / \\partial \\alpha_k, \\hat{\\mathbf{d}}_R \\rangle|`
      (finite-difference probe of
      :math:`\\nabla_\\theta \\nabla_{\\mathbf{R}} E_\\theta \\cdot \\hat{\\mathbf{d}}_R`)
    - cosine alignment between :math:`\\partial \\mathbf{F}_\\theta / \\partial \\alpha_k`
      and :math:`\\hat{\\mathbf{d}}_R`.

    Parameters
    ----------
    model : Any
        MLFF calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.
    eval_energy_and_forces : callable
        Callable ``eval_energy_and_forces(model_wrapper) -> (energy, forces)``
        returning scalar energy and ``(N, 3)`` forces.
    direction_r : np.ndarray
        Input-space (e.g. adversarial) perturbation vector of shape ``(N, 3)``
        or ``(3N,)``.
    rho : float, optional
        Parameter-space finite-difference step size, by default ``1e-3``.
    n_directions : int, optional
        Number of normalized parameter directions, by default ``5``.
    normalization : {'filter', 'layer', 'model', None}, optional
        Direction normalization passed to ``ModelParameters``, by default
        ``'filter'``.
    seed : int, optional
        Random seed, by default ``42``.
    trainable_only : bool, optional
        Perturb only parameters with ``requires_grad=True``, by default False.

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

    import torch

    _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.model_interface.model_parameters import (
        ModelParameters,
        rand_n_like,
    )

    u_r = _unit_vector(direction_r, name="direction_r")
    wrapper = wrap_mlff_model(model, trainable_only=trainable_only)
    params = wrapper.get_module_parameters()
    base_snapshot_list = _snapshot_wrapper_params(wrapper)
    base_snapshot = ModelParameters(base_snapshot_list)

    energy_sens = []
    force_sens = []
    adv_proj_sens = []
    adv_cos_align = []

    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        try:
            if hasattr(wrapper, "invalidate_cache"):
                wrapper.invalidate_cache()
            base_energy, base_forces = eval_energy_and_forces(wrapper)
            base_forces = np.asarray(base_forces, dtype=float)
            if base_forces.size != u_r.size:
                raise ValueError(
                    f"direction_r size {u_r.size} does not match forces size {base_forces.size}."
                )

            for _ in range(n_directions):
                direction_theta = rand_n_like(params)
                _normalize_oll_direction(
                    direction_theta, base_snapshot, normalization
                )
                step_vec = direction_theta * float(rho)

                params.add_(step_vec)
                if hasattr(wrapper, "invalidate_cache"):
                    wrapper.invalidate_cache()
                e_plus, f_plus = eval_energy_and_forces(wrapper)

                params.sub_(step_vec * 2.0)
                if hasattr(wrapper, "invalidate_cache"):
                    wrapper.invalidate_cache()
                e_minus, f_minus = eval_energy_and_forces(wrapper)

                _restore_wrapper_params(wrapper, base_snapshot_list)

                de_dalpha = (float(e_plus) - float(e_minus)) / (2.0 * rho)
                df_dalpha = (
                    np.asarray(f_plus, dtype=float)
                    - np.asarray(f_minus, dtype=float)
                ).reshape(-1) / (2.0 * rho)

                df_norm = float(np.linalg.norm(df_dalpha))
                proj = float(abs(np.dot(df_dalpha, u_r)))
                cos_align = proj / df_norm if df_norm > 1e-12 else float("nan")

                energy_sens.append(abs(de_dalpha))
                force_sens.append(df_norm)
                adv_proj_sens.append(proj)
                adv_cos_align.append(cos_align)
        finally:
            _restore_wrapper_params(wrapper, base_snapshot_list)

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
    "compute_parameter_hessian_eigs",
    "compute_pes_hessian",
    "evaluate_hessian_loss_plane",
    "evaluate_loss_landscape_1d",
    "evaluate_loss_landscape_2d",
    "evaluate_pes_profile_1d",
    "evaluate_pes_slice_2d",
    "extract_torch_module",
    "gamma_phonon_modes",
    "hessian_spectrum",
    "invalidate_calculator_cache",
    "make_adversarial_gap_metric",
    "make_energy_loss_metric",
    "make_force_loss_metric",
    "make_mace_loss_callable",
    "make_mlff_metric",
    "parameter_sharpness",
    "perturbation_hessian_alignment",
    "pes_parameter_coupling",
    "pes_roughness",
    "plot_pes_basin_2d",
    "project_onto_basis",
    "wrap_mlff_model",
]
