"""Unit tests for mlff_attack.metrics.

Topology tests are adapted from ``declanwhchan/mlff_attack_data_collection``
(MIT License, Copyright (c) 2026 Declan Chan).
"""

import numpy as np
import pytest

from mlff_attack import metrics


def test_calculate_spectral_flatness():
    signal = [1, 2, 3, 4, 5]
    sf = metrics.calculate_spectral_flatness(signal)
    assert sf is not None
    assert 0 <= sf <= 1

    constant_signal = [5, 5, 5, 5, 5]
    sf_constant = metrics.calculate_spectral_flatness(constant_signal)
    assert sf_constant is not None
    assert np.round(sf_constant, 2) == 1.0

    empty_signal = []
    sf_empty = metrics.calculate_spectral_flatness(empty_signal)
    assert np.isnan(sf_empty)

    non_1d_signal = [[1, 2], [3, 4]]
    sf_non_1d = metrics.calculate_spectral_flatness(non_1d_signal)
    assert np.isnan(sf_non_1d)


def test_normalized_epsilon():
    assert np.isclose(metrics.normalized_epsilon(0.05, 5.0), 1.0)
    cell = np.diag([4.0, 5.0, 10.0])
    assert np.isclose(metrics.normalized_epsilon(0.2, cell), 5.0)
    assert np.isnan(metrics.normalized_epsilon(0.1, 0.0))
    assert np.isnan(metrics.normalized_epsilon(np.nan, 4.0))


def test_displacement_metrics():
    before = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    after = np.array([[0.0, 0.0, 0.0], [1.0, 0.3, 0.4]])
    summary = metrics.displacement_metrics(before, after)
    assert np.isclose(summary["mean_displacement"], 0.25)
    assert np.isclose(summary["median_displacement"], 0.25)
    assert np.isclose(summary["max_displacement"], 0.5)
    assert np.isclose(summary["rmsd"], np.sqrt(0.125))


def test_rms_force_and_force_change_metrics():
    before = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    after = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])

    assert np.isclose(metrics.rms_force(before), np.sqrt(2.5))
    angles = metrics.force_angle_changes(before, after, degrees=True)
    assert np.allclose(angles, [90.0, 0.0])

    summary = metrics.force_change_metrics(before, after)
    assert np.isclose(summary["delta_force_max"], np.sqrt(2.0))
    assert np.isclose(summary["delta_force_angle_median_deg"], 45.0)
    assert np.isclose(summary["rms_force_difference"], 0.0)


def test_force_angle_changes_handles_zero_norm():
    before = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    after = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    angles = metrics.force_angle_changes(before, after, degrees=True)
    assert np.isnan(angles[0])
    assert np.isclose(angles[1], 180.0)


def test_edge_jaccard_distance_sets():
    assert metrics.edge_jaccard_distance(set(), set()) == 0.0
    assert metrics.edge_jaccard_distance({("C0", "C1")}, {("C0", "C1")}) == 0.0
    assert metrics.edge_jaccard_distance({("C0", "C1")}, set()) == 1.0
    assert np.isclose(
        metrics.edge_jaccard_distance(
            {("C0", "C1"), ("C1", "C2")},
            {("C0", "C1"), ("C2", "C3")},
        ),
        2.0 / 3.0,
    )


def test_relaxation_stability():
    assert metrics.relaxation_stability([-1.0, -2.0, -3.0, -4.0]) == 0.0
    assert np.isclose(metrics.relaxation_stability([-1.0, -0.5, -2.0, -1.5]), 2.0 / 3.0)
    assert np.isnan(metrics.relaxation_stability([-1.0]))


def test_force_coherence():
    assert np.isclose(metrics.force_coherence([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]), 1.0)
    assert np.isclose(metrics.force_coherence([1.0, 2.0, 3.0], [0.3, 0.2, 0.1]), -1.0)
    assert np.isnan(metrics.force_coherence([1.0, 1.0], [0.1, 0.2]))


def _carbon_pair(distance: float):
    ase = pytest.importorskip("ase")
    return ase.Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )


def test_identical_structures_have_zero_topology_change():
    atoms = _carbon_pair(distance=1.2)
    summary = metrics.topology_change_metrics(atoms, atoms.copy())

    assert summary["neighbor_jaccard_distance"] == 0.0
    assert summary["coordination_change_mean"] == 0.0
    assert summary["coordination_change_max"] == 0.0
    assert summary["rdf_l1_distance"] == 0.0


def test_removing_only_bond_has_jaccard_distance_one():
    bonded = _carbon_pair(distance=1.2)
    separated = _carbon_pair(distance=3.0)
    summary = metrics.topology_change_metrics(bonded, separated)

    assert summary["neighbor_edges_before"] == 1
    assert summary["neighbor_edges_after"] == 0
    assert summary["neighbor_edges_added"] == 0
    assert summary["neighbor_edges_removed"] == 1
    assert summary["neighbor_edge_change_count"] == 1
    assert summary["neighbor_jaccard_distance"] == 1.0
    assert summary["coordination_change_max"] == 1.0


def test_standard_rdf_is_finite_and_changes_on_displacement():
    ase = pytest.importorskip("ase")
    before = ase.Atoms(
        "C4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.0, 1.2],
        ],
        cell=[15.0, 15.0, 15.0],
        pbc=True,
    )
    rdf, radii = metrics.rdf_values(before)
    assert len(rdf) == 60
    assert len(radii) == 60
    assert np.all(np.isfinite(rdf))
    assert np.all(rdf >= 0.0)
    assert np.all(np.diff(radii) > 0.0)

    after = before.copy()
    after.positions[1, 0] = 1.6
    distance = metrics.rdf_l1_distance(before, after)
    assert np.isfinite(distance)
    assert distance > 0.0


def test_symmetry_change_metrics():
    ase_build = pytest.importorskip("ase.build")
    pytest.importorskip("spglib")

    si_bulk = ase_build.bulk("Si", "diamond", a=5.43, cubic=True)
    unchanged = metrics.symmetry_change_metrics(si_bulk, si_bulk.copy())
    assert unchanged["space_group_change_fraction"] == 0.0
    assert unchanged["symmetry_operation_retention"] == 1.0
    assert unchanged["unique_site_change"] == 0.0

    perturbed = si_bulk.copy()
    perturbed.positions[0] += np.array([0.2, 0.05, -0.1])
    changed = metrics.symmetry_change_metrics(si_bulk, perturbed)
    assert changed["space_group_change_fraction"] == 1.0
    assert changed["symmetry_operation_retention"] < 1.0
    assert changed["unique_site_change"] > 0.0
