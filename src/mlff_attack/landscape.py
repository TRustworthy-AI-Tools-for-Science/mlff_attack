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


def wrap_mlff_model(
    calculator_or_module: Any,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> Any:
    """Wrap an MLFF calculator or PyTorch module for ``oxford_loss_landscapes``.

    Parameters
    ----------
    calculator_or_module : Any
        MACE/UMA/CHGNet calculator, ``torch.nn.Module``, or ``ModelWrapper``.
    forward_fn : callable, optional
        Optional ``forward_fn(model, x)`` for ``GeneralModelWrapper``.

    Returns
    -------
    oxford_loss_landscapes.ModelWrapper
        Wrapped model ready for ``oxford_loss_landscapes`` routines.
    """
    oll = _require_oxford_loss_landscapes()
    module = extract_torch_module(calculator_or_module)
    if isinstance(module, oll.ModelWrapper):
        return module
    if forward_fn is not None:
        return oll.GeneralModelWrapper(module, [module], forward_fn)
    return oll.SimpleModelWrapper(module)


def make_mlff_metric(eval_fn: Callable[[Any], float]) -> Any:
    """Adapt an MLFF evaluation callable into an ``oxford_loss_landscapes.Metric``.

    Parameters
    ----------
    eval_fn : callable
        Callable accepting an ``oxford_loss_landscapes.ModelWrapper`` (or zero
        arguments if wrapped via ``lambda _wrapper: ...``) and returning a
        scalar ``float`` loss, energy, or force-error metric.

    Returns
    -------
    oxford_loss_landscapes.metrics.metric.Metric
        Metric instance usable with ``random_line``, ``random_plane``,
        ``planar_interpolation``, ``linear_interpolation``, and
        ``hessian_plane``.
    """
    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    class _MLFFCallableMetric(OllMetric):
        def __init__(self, fn: Callable[[Any], float]):
            super().__init__()
            self._fn = fn

        def __call__(self, model_wrapper: Any) -> float:
            return float(self._fn(model_wrapper))

    return _MLFFCallableMetric(eval_fn)


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


def evaluate_loss_landscape_1d(
    model: Any,
    eval_fn: Callable[[Any], float],
    distance: float = 0.1,
    steps: int = 51,
    normalization: Optional[str] = "filter",
    deepcopy_model: bool = True,
) -> Dict[str, Any]:
    """Evaluate a 1D random parameter-space loss slice via ``oxford_loss_landscapes``.

    Delegates directly to :func:`oxford_loss_landscapes.random_line` and
    computes curvature roughness along the slice.

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
        Deep-copy the model before perturbing weights, by default True.

    Returns
    -------
    dict
        Contains ``alphas``, ``losses``, ``distance``, ``steps``, and
        ``roughness``.
    """
    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    wrapper = wrap_mlff_model(model)
    metric = eval_fn if isinstance(eval_fn, OllMetric) else make_mlff_metric(eval_fn)
    losses = np.asarray(
        oll.random_line(
            wrapper,
            metric=metric,
            distance=distance,
            steps=steps,
            normalization=normalization,
            deepcopy_model=deepcopy_model,
        ),
        dtype=float,
    )
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
    deepcopy_model: bool = True,
    use_ray: bool = False,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a 2D random parameter-space loss surface via ``oxford_loss_landscapes``.

    Delegates directly to :func:`oxford_loss_landscapes.random_plane`.

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
        Deep-copy the model before perturbing weights, by default True.
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

    wrapper = wrap_mlff_model(model)
    metric = eval_fn if isinstance(eval_fn, OllMetric) else make_mlff_metric(eval_fn)
    losses = np.asarray(
        oll.random_plane(
            wrapper,
            metric=metric,
            distance=distance,
            steps=steps,
            normalization=normalization,
            deepcopy_model=deepcopy_model,
            use_ray=use_ray,
            num_workers=num_workers,
        ),
        dtype=float,
    )
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
    criterion: Any,
    inputs: Any,
    targets: Any,
    distance: float = 1.0,
    steps: int = 21,
    deepcopy_model: bool = True,
    vrpca_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate a 2D loss plane along the dominant parameter Hessian eigenvector.

    Delegates directly to :func:`oxford_loss_landscapes.hessian_plane`, which
    computes the top parameter-Hessian eigenvector via VR-PCA and pairs it
    with a random orthogonal direction.

    Parameters
    ----------
    model : Any
        MLFF calculator, PyTorch ``nn.Module``, or ``ModelWrapper``.
    eval_fn : callable
        Callable ``eval_fn(model_wrapper) -> float`` or ``Metric``.
    criterion : callable
        Differentiable PyTorch loss criterion for Hessian computation.
    inputs, targets : Any
        Inputs and targets passed to ``oll.hessian_plane``.
    distance : float, optional
        Maximum distance from center along each axis, by default ``1.0``.
    steps : int, optional
        Grid resolution (``steps x steps``), by default ``21``.
    deepcopy_model : bool, optional
        Deep-copy the model before perturbing weights, by default True.
    vrpca_config : VRPCAConfig, optional
        Optional ``oxford_loss_landscapes.VRPCAConfig`` instance.

    Returns
    -------
    dict
        Contains ``alpha``, ``beta``, ``losses``, ``distance``, and ``steps``.
    """
    oll = _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.metrics.metric import Metric as OllMetric

    module = extract_torch_module(model)
    metric = eval_fn if isinstance(eval_fn, OllMetric) else make_mlff_metric(eval_fn)
    losses = np.asarray(
        oll.hessian_plane(
            module,
            metric=metric,
            criterion=criterion,
            inputs=inputs,
            targets=targets,
            distance=distance,
            steps=steps,
            deepcopy_model=deepcopy_model,
            vrpca_config=vrpca_config,
        ),
        dtype=float,
    )
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
    criterion: Any,
    inputs: Any,
    targets: Any,
    vrpca_config: Optional[Any] = None,
) -> Any:
    """Estimate min/max parameter-space Hessian eigenpairs via ``oxford_loss_landscapes``.

    Delegates directly to :func:`oxford_loss_landscapes.min_max_hessian_eigs_vrpca`.
    """
    oll = _require_oxford_loss_landscapes()
    module = extract_torch_module(model)
    if isinstance(module, oll.ModelWrapper):
        module = module.modules[0]
    return oll.min_max_hessian_eigs_vrpca(
        net=module,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=vrpca_config,
    )


def parameter_sharpness(
    model: Any,
    eval_fn: Callable[[Any], float],
    rho: float = 1e-2,
    n_directions: int = 10,
    normalization: Optional[str] = "filter",
    seed: int = 42,
) -> Dict[str, Any]:
    """Estimate local parameter-space sharpness using ``oxford_loss_landscapes`` directions.

    Samples ``n_directions`` normalized parameter directions via
    :func:`oxford_loss_landscapes.model_interface.model_parameters.rand_n_like`
    and evaluates symmetric perturbations :math:`\\theta_0 \\pm \\rho d_k`,
    restoring the original weights in a ``finally`` block.

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

    import copy
    import torch

    _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.model_interface.model_parameters import rand_n_like

    wrapper = wrap_mlff_model(model)
    params = wrapper.get_module_parameters()
    base_snapshot = copy.deepcopy(params)

    sharpness_vals = []
    curvatures = []

    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        try:
            base_loss = float(eval_fn(wrapper))
            for _ in range(n_directions):
                direction = rand_n_like(params)
                _normalize_oll_direction(direction, base_snapshot, normalization)
                step_vec = direction * float(rho)

                params.add_(step_vec)
                loss_plus = float(eval_fn(wrapper))

                params.sub_(step_vec * 2.0)
                loss_minus = float(eval_fn(wrapper))

                # Restore exact baseline parameters before next direction
                with torch.no_grad():
                    for cur_p, base_p in zip(params, base_snapshot):
                        cur_p.copy_(base_p)

                sharpness_vals.append(max(loss_plus, loss_minus) - base_loss)
                curvatures.append(
                    (loss_plus + loss_minus - 2.0 * base_loss) / (rho**2)
                )
        finally:
            with torch.no_grad():
                for cur_p, base_p in zip(params, base_snapshot):
                    cur_p.copy_(base_p)

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
) -> Dict[str, Any]:
    """Quantify coupling between parameter perturbations and PES forces.

    Uses ``oxford_loss_landscapes`` ``ModelParameters`` direction sampling and
    normalization to perturb :math:`\\theta_0 \\pm \\rho d_k`, then measures:

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

    import copy
    import torch

    _require_oxford_loss_landscapes()
    from oxford_loss_landscapes.model_interface.model_parameters import rand_n_like

    u_r = _unit_vector(direction_r, name="direction_r")
    wrapper = wrap_mlff_model(model)
    params = wrapper.get_module_parameters()
    base_snapshot = copy.deepcopy(params)

    energy_sens = []
    force_sens = []
    adv_proj_sens = []
    adv_cos_align = []

    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        try:
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
                e_plus, f_plus = eval_energy_and_forces(wrapper)

                params.sub_(step_vec * 2.0)
                e_minus, f_minus = eval_energy_and_forces(wrapper)

                with torch.no_grad():
                    for cur_p, base_p in zip(params, base_snapshot):
                        cur_p.copy_(base_p)

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
            with torch.no_grad():
                for cur_p, base_p in zip(params, base_snapshot):
                    cur_p.copy_(base_p)

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
    "make_mlff_metric",
    "parameter_sharpness",
    "perturbation_hessian_alignment",
    "pes_parameter_coupling",
    "pes_roughness",
    "plot_pes_basin_2d",
    "project_onto_basis",
    "wrap_mlff_model",
]
