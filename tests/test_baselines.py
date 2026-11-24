"""Unit tests for mlff_attack.baselines.

Contour helper tests are adapted from ``declanwhchan/mlff_attack_data_collection``
(MIT License, Copyright (c) 2026 Declan Chan).
"""

import numpy as np
import pytest

from mlff_attack import baselines


def test_parse_betas_uses_configured_defaults():
    assert baselines.parse_betas(None, [0.1, 0.05, 0.0]) == [0.1, 0.05, 0.0]


def test_parse_betas_accepts_comma_separated_values():
    assert baselines.parse_betas("0.2, 0.1, 0", []) == [0.2, 0.1, 0.0]


def test_beta_tag_is_stable():
    assert baselines.beta_tag(0.1) == "beta_100"
    assert baselines.beta_tag(0.05) == "beta_050"
    assert baselines.beta_tag(0.0) == "beta_000"


def test_out_of_plane_angle_for_right_angle_is_90_degrees():
    first = np.array([[0.0, 0.0, 0.0]])
    second = np.array([[1.0, 0.0, 0.0]])
    third = np.array([[1.0, 1.0, 0.0]])

    angle = baselines.out_of_plane_angle(first, second, third)
    assert np.isclose(angle, 90.0)
    assert np.isnan(baselines.out_of_plane_angle(None, second, third))


def test_random_uniform_and_gaussian_perturbations():
    positions = np.zeros((10, 3))

    u1 = baselines.random_uniform_box_perturbation(positions, epsilon=0.05, seed=42)
    u2 = baselines.random_uniform_box_perturbation(positions, epsilon=0.05, seed=42)
    assert np.allclose(u1, u2)
    assert np.max(np.abs(u1 - positions)) <= 0.05 + 1e-12
    assert np.max(np.abs(u1 - positions)) > 0.0

    g1 = baselines.random_gaussian_perturbation(
        positions, epsilon=0.05, seed=7, clip=True
    )
    assert np.max(np.abs(g1 - positions)) <= 0.05 + 1e-12


def test_initial_velocities_are_repeatable_and_centered():
    ase = pytest.importorskip("ase")
    first = ase.Atoms("H4", positions=np.zeros((4, 3)))
    second = first.copy()

    baselines.initial_velocities(first, seed=42)
    baselines.initial_velocities(second, seed=42)

    assert np.allclose(first.get_velocities(), second.get_velocities())
    assert np.allclose(first.get_velocities().mean(axis=0), 0.0)


def test_parse_perturbed_structure_name():
    parsed = baselines.parse_perturbed_structure_name(
        "uma_berlinite_alpo4_ifgsm_eps03_perturbed"
    )
    assert parsed == {
        "source_model": "uma",
        "material_slug": "berlinite_alpo4",
        "attack_label": "I-FGSM",
        "epsilon": 0.3,
        "n_steps": None,
    }

    parsed_steps = baselines.parse_perturbed_structure_name(
        "mace_reo3_pgd_eps005_steps10_perturbed"
    )
    assert parsed_steps == {
        "source_model": "mace_mh",
        "material_slug": "reo3",
        "attack_label": "PGD",
        "epsilon": 0.05,
        "n_steps": 10,
    }

    assert baselines.parse_perturbed_structure_name("unrecognized_name") is None


def test_read_outcar_force_blocks(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "\n".join(
            [
                " POSITION                                       TOTAL-FORCE (eV/Angst)",
                " -----------------------------------------------------------------------------------",
                "      0.00000      0.00000      0.00000         0.100000     -0.200000      0.300000",
                "      1.00000      0.00000      0.00000        -0.100000      0.200000     -0.300000",
                " -----------------------------------------------------------------------------------",
                " some intermediate text",
                " POSITION                                       TOTAL-FORCE (eV/Angst)",
                " -----------------------------------------------------------------------------------",
                "      0.01000      0.00000      0.00000         0.010000     -0.020000      0.030000",
                "      0.99000      0.00000      0.00000        -0.010000      0.020000     -0.030000",
                " -----------------------------------------------------------------------------------",
            ]
        ),
        encoding="utf-8",
    )

    first, final, count = baselines.read_outcar_force_blocks(outcar, expected_atoms=2)
    assert count == 2
    assert np.allclose(first, [[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    assert np.allclose(final, [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]])


def test_load_dft_manifest_filters_and_enriches(tmp_path):
    manifest = tmp_path / "preliminary_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "structure_name,delivery_group,final_energy_eV,final_fmax_eV_A",
                "uma_reo3_fgsm_eps05_perturbed,FORCE_CONVERGED_CLEAN,-30.5,0.02",
                "mace_reo3_fgsm_eps2_perturbed,FORCE_CONVERGED_FLAGGED,-25.1,0.03",
            ]
        ),
        encoding="utf-8",
    )

    clean = baselines.load_dft_manifest(manifest, delivery_group="FORCE_CONVERGED_CLEAN")
    assert len(clean) == 1
    assert clean[0]["source_model"] == "uma"
    assert clean[0]["material_slug"] == "reo3"
    assert clean[0]["attack_label"] == "FGSM"
    assert np.isclose(clean[0]["parsed_epsilon"], 0.5)

    all_rows = baselines.load_dft_manifest(manifest, delivery_group=None)
    assert len(all_rows) == 2
