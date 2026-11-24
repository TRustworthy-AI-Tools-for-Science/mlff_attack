"""Unit tests for mlff_attack.landscape.

Tests cover 1D/2D PES slicing, PES smoothness/roughness, finite-difference
position Hessians, perturbation-Hessian alignment, 2D basin plotting, and
PyTorch parameter-space loss-landscape & PES coupling probes.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlff_attack import landscape


def _harmonic_potential(stiffness: np.ndarray, origin: np.ndarray):
    """Return exact (energy_fn, forces_fn) for E(R) = 0.5 (R-R0)^T K (R-R0)."""
    k_mat = 0.5 * (stiffness + stiffness.T)
    r0 = origin.reshape(-1)

    def energy_fn(positions: np.ndarray) -> float:
        disp = positions.reshape(-1) - r0
        return 0.5 * float(disp @ k_mat @ disp)

    def forces_fn(positions: np.ndarray) -> np.ndarray:
        disp = positions.reshape(-1) - r0
        return -(k_mat @ disp).reshape(positions.shape)

    return energy_fn, forces_fn


def test_build_attack_relaxation_basis_and_projection():
    ref = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    perturbed = np.array([[0.3, 0.0, 0.0], [1.5, 0.0, 0.0]])
    final = np.array([[0.65, 0.2, 0.0], [1.5, -0.2, 0.0]])

    u1, u2 = landscape.build_attack_relaxation_basis(ref, perturbed, final)
    assert np.isclose(np.linalg.norm(u1), 1.0)
    assert np.isclose(np.linalg.norm(u2), 1.0)
    assert np.isclose(np.dot(u1, u2), 0.0)

    pts = landscape.project_onto_basis([ref, perturbed, final], ref, u1, u2)
    assert pts.shape == (3, 2)
    assert np.allclose(pts[0], [0.0, 0.0])
    assert np.allclose(pts[1], [0.3, 0.0])
    assert np.isclose(pts[2, 0], 0.65)
    assert pts[2, 1] > 0.0


def test_build_basis_uses_fallback_when_collinear():
    ref = np.zeros((2, 3))
    perturbed = np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    collinear_final = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    fallback = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="collinear"):
        landscape.build_attack_relaxation_basis(ref, perturbed, collinear_final)

    u1, u2 = landscape.build_attack_relaxation_basis(
        ref, perturbed, collinear_final, fallback_direction=fallback
    )
    assert np.isclose(np.dot(u1, u2), 0.0)
    assert np.isclose(np.linalg.norm(u2), 1.0)


def test_pes_profile_1d_and_roughness():
    ref = np.zeros((2, 3))
    stiffness = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    energy_fn, forces_fn = _harmonic_potential(stiffness, ref)

    direction = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    alphas = np.linspace(-0.5, 0.5, 41)
    profile = landscape.evaluate_pes_profile_1d(
        ref,
        direction,
        alphas,
        energy_fn=energy_fn,
        forces_fn=forces_fn,
        compute_forces=True,
    )

    assert np.allclose(profile["relative_energies"], 0.5 * 2.0 * alphas**2)
    assert np.allclose(profile["directional_slopes"], 2.0 * alphas)
    # Quadratic potential has constant curvature = 2.0 and near-zero TV
    assert np.isclose(profile["roughness"]["mean_abs_curvature"], 2.0, atol=1e-2)
    assert profile["roughness"]["curvature_tv"] < 1e-6

    # Adding oscillatory roughness increases curvature total variation
    rough_energies = profile["relative_energies"] + 0.05 * np.sin(25.0 * alphas)
    rough_stats = landscape.pes_roughness(alphas, rough_energies)
    assert rough_stats["curvature_tv"] > 1.0


def test_pes_slice_2d_and_plot(tmp_path):
    ref = np.zeros((2, 3))
    stiffness = np.diag([2.0, 6.0, 4.0, 4.0, 4.0, 4.0])
    energy_fn, _ = _harmonic_potential(stiffness, ref)

    perturbed = np.array([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    final = np.array([[0.1, 0.4, 0.0], [0.0, 0.0, 0.0]])
    u1, u2 = landscape.build_attack_relaxation_basis(ref, perturbed, final)

    x_grid = np.linspace(-0.2, 0.5, 9)
    y_grid = np.linspace(-0.5, 0.5, 11)
    pes_slice = landscape.evaluate_pes_slice_2d(
        ref, u1, u2, x_grid, y_grid, energy_fn=energy_fn
    )
    assert pes_slice["relative_energies"].shape == (11, 9)

    xx, yy = np.meshgrid(x_grid, y_grid)
    expected = 0.5 * (2.0 * xx**2 + 6.0 * yy**2)
    assert np.allclose(pes_slice["relative_energies"], expected)

    endpoints = landscape.project_onto_basis([ref, perturbed, final], ref, u1, u2)
    out_file = tmp_path / "basin.png"
    fig = landscape.plot_pes_basin_2d(
        pes_slice,
        endpoints=endpoints,
        relaxation_path=endpoints[1:],
        dft_final=np.array([0.05, 0.02]),
        save_path=out_file,
    )
    plt.close(fig)
    assert out_file.is_file()


def test_compute_pes_hessian_spectrum_and_alignment():
    ref = np.array([[0.0, 0.0, 0.0], [1.2, 0.3, -0.4]])
    stiffness = np.diag([-1.5, 0.0, 2.0, 5.0, 8.0, 10.0])
    _, forces_fn = _harmonic_potential(stiffness, ref)

    hessian = landscape.compute_pes_hessian(ref, delta=1e-3, forces_fn=forces_fn)
    assert np.allclose(hessian, stiffness, atol=1e-8)

    spec = landscape.hessian_spectrum(hessian, zero_tol=1e-6)
    assert spec["n_negative"] == 1
    assert spec["n_zero"] == 1
    assert spec["n_positive"] == 4
    assert np.isclose(spec["min_eigenvalue"], -1.5)
    assert np.isclose(spec["max_eigenvalue"], 10.0)
    assert np.isclose(spec["trace"], np.sum(np.diag(stiffness)))

    # Direction aligned with stiffest mode (index 5, eigenvalue 10.0)
    stiff_dir = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    align_stiff = landscape.perturbation_hessian_alignment(hessian, stiff_dir)
    assert np.isclose(align_stiff["directional_curvature"], 10.0)
    assert np.isclose(align_stiff["stiffest_mode_alignment"], 1.0)
    assert np.isclose(align_stiff["negative_subspace_weight"], 0.0)

    # Direction aligned with unstable/saddle mode (index 0, eigenvalue -1.5)
    saddle_dir = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    align_saddle = landscape.perturbation_hessian_alignment(hessian, saddle_dir)
    assert np.isclose(align_saddle["directional_curvature"], -1.5)
    assert np.isclose(align_saddle["softest_mode_alignment"], 1.0)
    assert np.isclose(align_saddle["negative_subspace_weight"], 1.0)

    # Mass-weighted spectrum
    mw_spec = landscape.hessian_spectrum(hessian, masses=[4.0, 1.0])
    assert np.isclose(mw_spec["min_eigenvalue"], -1.5 / 4.0)
    assert np.isclose(mw_spec["max_eigenvalue"], 10.0)


def test_loss_landscape_sharpness_and_pes_coupling():
    torch = pytest.importorskip("torch")

    # Tiny differentiable quadratic surrogate E_theta(R) = 0.5 * sum((W @ R)^2)
    class ToyPotential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(6, dtype=torch.float64) * 2.0)

        def energy_and_forces(self, r_vec):
            r_t = torch.tensor(
                r_vec.reshape(-1), dtype=torch.float64, requires_grad=True
            )
            energy = 0.5 * torch.sum((self.weight @ r_t) ** 2)
            forces = -torch.autograd.grad(energy, r_t)[0]
            return float(energy.item()), forces.detach().numpy().reshape(-1, 3)

    model = ToyPotential()
    r0 = np.array([[0.5, -0.2, 0.1], [0.3, 0.4, -0.6]])
    base_weight_copy = model.weight.detach().clone()

    def loss_fn():
        e_val, _ = model.energy_and_forces(r0)
        return (e_val - 1.0) ** 2

    d1 = landscape.filter_normalized_direction(model, seed=10)
    d2 = landscape.filter_normalized_direction(model, seed=11)

    slice_1d = landscape.evaluate_loss_landscape_1d(
        model, loss_fn, d1, alphas=np.linspace(-0.05, 0.05, 9)
    )
    assert slice_1d["losses"].shape == (9,)
    assert np.all(np.isfinite(slice_1d["losses"]))
    assert torch.allclose(model.weight, base_weight_copy)

    slice_2d = landscape.evaluate_loss_landscape_2d(
        model, loss_fn, d1, d2, alpha_grid=[-0.02, 0.0, 0.02], beta_grid=[-0.02, 0.0, 0.02]
    )
    assert slice_2d["losses"].shape == (3, 3)
    assert np.isclose(slice_2d["losses"][1, 1], slice_2d["base_loss"])
    assert torch.allclose(model.weight, base_weight_copy)

    sharp = landscape.parameter_sharpness(model, loss_fn, rho=1e-2, n_directions=4, seed=42)
    assert np.isfinite(sharp["max_sharpness"])
    assert np.isfinite(sharp["mean_curvature"])
    assert torch.allclose(model.weight, base_weight_copy)

    adv_dir = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coupling = landscape.pes_parameter_coupling(
        model,
        lambda: model.energy_and_forces(r0),
        direction_r=adv_dir,
        rho=1e-3,
        n_directions=4,
        seed=7,
    )
    assert coupling["mean_energy_sensitivity"] > 0.0
    assert coupling["mean_force_sensitivity"] > 0.0
    assert coupling["mean_adversarial_force_sensitivity"] > 0.0
    assert 0.0 <= coupling["mean_adversarial_cosine_alignment"] <= 1.0
    assert torch.allclose(model.weight, base_weight_copy)
