"""
Tests for sona.fields.csv_field.CSVFieldProvider.

Strategy: write a CSV of a KNOWN analytic function (not the synthetic
two-coil field -- that's tested separately, in test_synthetic_field.py,
against Biot-Savart). Using a simple closed-form Bz(z) here isolates the
interpolation/derivative/unit-conversion machinery from any question
about whether the coil physics itself is right.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sona.fields.csv_field import CSVFieldProvider


def _write_tanh_field_csv(path, z_cm, B0_G=20.0, L_cm=5.0):
    """
    Bz(z) = B0 * tanh(z/L)  -- a smooth, analytically known reversal
    shape with a simple closed-form derivative:  dBz/dz = (B0/L) * sech^2(z/L)
    Written in (cm, G) to also exercise the unit conversion.
    """
    Bz_G = B0_G * np.tanh(z_cm / L_cm)
    pd.DataFrame({"z": z_cm, "Bz": Bz_G}).to_csv(path, index=False)


@pytest.fixture
def tanh_csv(tmp_path):
    path = tmp_path / "tanh_field.csv"
    z_cm = np.linspace(-30, 30, 121)   # 0.5 cm spacing, realistic density
    _write_tanh_field_csv(path, z_cm)
    return str(path), z_cm


def test_Bz_matches_analytic_function(tanh_csv):
    path, z_cm = tanh_csv
    provider = CSVFieldProvider(path, z_unit="cm", b_unit="G")

    # check at OFF-GRID points, in meters/Tesla (SI) as the interface expects
    z_test_cm = np.array([-27.3, -10.1, -0.05, 0.0, 0.05, 12.7, 28.8])
    z_test_m = z_test_cm * 1e-2

    B0_T = 20.0 * 1e-4   # 20 G -> T
    L_m = 5.0 * 1e-2     # 5 cm -> m
    expected = B0_T * np.tanh(z_test_m / L_m)

    got = provider.Bz(z_test_m)
    # atol set relative to the field scale (B0=20 G=2e-3 T): near the
    # zero-crossing rtol alone is meaningless since values -> 0
    assert np.allclose(got, expected, rtol=2e-4, atol=1e-7)


def test_dBz_dz_matches_analytic_derivative(tanh_csv):
    path, z_cm = tanh_csv
    provider = CSVFieldProvider(path, z_unit="cm", b_unit="G")

    z_test_m = np.array([-0.1, -0.02, 0.0, 0.03, 0.15])
    B0_T = 20.0 * 1e-4
    L_m = 5.0 * 1e-2
    expected = (B0_T / L_m) / np.cosh(z_test_m / L_m) ** 2

    got = provider.dBz_dz(z_test_m)
    # derivative of an interpolant is inherently less precise than the
    # function itself -- looser tolerance is appropriate and expected
    assert np.allclose(got, expected, rtol=5e-3, atol=1e-12)


def test_Br_derived_from_Bz_via_maxwell_relation(tanh_csv):
    path, z_cm = tanh_csv
    provider = CSVFieldProvider(path, z_unit="cm", b_unit="G")

    r = 0.005  # 5 mm
    z = np.array([-0.05, 0.0, 0.07])
    expected = -(r / 2.0) * provider.dBz_dz(z)
    got = provider.Br(r, z)
    assert np.allclose(got, expected)


def test_Br_with_supplied_column_scales_linearly_with_r(tmp_path):
    z_cm = np.linspace(-20, 20, 81)
    Bz_G = 15.0 * np.tanh(z_cm / 4.0)
    # synthetic Br column "measured" at r_ref = 3 mm
    dBz_dz_approx = np.gradient(Bz_G, z_cm)   # G/cm, rough, just for test data
    r_ref_cm = 0.3
    Br_at_ref_G = -(r_ref_cm / 2.0) * dBz_dz_approx

    path = tmp_path / "field_with_br.csv"
    pd.DataFrame({"z": z_cm, "Bz": Bz_G, "Br": Br_at_ref_G}).to_csv(path, index=False)

    provider = CSVFieldProvider(
        str(path), br_col="Br", z_unit="cm", b_unit="G",
        br_reference_radius_m=r_ref_cm * 1e-2,
    )

    z_test = np.array([-0.05, 0.0, 0.06])
    Br_at_ref = provider.Br(r_ref_cm * 1e-2, z_test)
    Br_at_double = provider.Br(2 * r_ref_cm * 1e-2, z_test)
    assert np.allclose(Br_at_double, 2 * Br_at_ref)


def test_out_of_range_z_raises(tanh_csv):
    path, z_cm = tanh_csv
    provider = CSVFieldProvider(path, z_unit="cm", b_unit="G")
    z_min, z_max = provider.z_range()
    with pytest.raises(ValueError):
        provider.Bz(z_max + 1.0)
    with pytest.raises(ValueError):
        provider.Bz(z_min - 1.0)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"position": [1, 2, 3, 4], "field": [1, 2, 3, 4]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        CSVFieldProvider(str(path))  # default column names 'z','Bz' won't be found


def test_br_col_requires_reference_radius(tmp_path):
    z_cm = np.linspace(-10, 10, 21)
    path = tmp_path / "field.csv"
    pd.DataFrame({"z": z_cm, "Bz": z_cm, "Br": z_cm}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        CSVFieldProvider(str(path), br_col="Br")  # missing br_reference_radius_m


def test_unit_conversion_mm_and_mT(tmp_path):
    """Same physics, different units -- should give identical SI results."""
    z_cm = np.linspace(-10, 10, 41)
    Bz_G = 10.0 * np.tanh(z_cm / 3.0)

    path_cm_G = tmp_path / "cm_G.csv"
    pd.DataFrame({"z": z_cm, "Bz": Bz_G}).to_csv(path_cm_G, index=False)
    p1 = CSVFieldProvider(str(path_cm_G), z_unit="cm", b_unit="G")

    path_mm_mT = tmp_path / "mm_mT.csv"
    pd.DataFrame({"z": z_cm * 10, "Bz": Bz_G * 0.1}).to_csv(path_mm_mT, index=False)
    p2 = CSVFieldProvider(str(path_mm_mT), z_unit="mm", b_unit="mT")

    z_test_m = np.array([-0.03, 0.0, 0.05])
    assert np.allclose(p1.Bz(z_test_m), p2.Bz(z_test_m), rtol=1e-10)