"""
Physics-level tests for scripts/generate_synthetic_field.py -- checking
that the two-opposed-coil model actually has the properties a Sona field
must have, independent of the CSV/interpolation machinery (that's
covered in test_csv_field.py).
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import generate_synthetic_field as gsf


def test_zero_crossing_at_midpoint_by_symmetry():
    """Equal and opposite coils at +/- d/2 must give exactly Bz(0)=0."""
    assert gsf.two_coil_Bz(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-30)


def test_antisymmetric_about_midpoint():
    """Bz(-z) == -Bz(z) -- required by the +I/-I coil symmetry."""
    z = np.linspace(0.01, 0.25, 30)
    assert np.allclose(gsf.two_coil_Bz(-z), -gsf.two_coil_Bz(z), rtol=1e-10)


def test_sign_reverses_across_crossing():
    z_neg, z_pos = np.array([-0.02]), np.array([0.02])
    assert gsf.two_coil_Bz(z_neg)[0] < 0
    assert gsf.two_coil_Bz(z_pos)[0] > 0


def test_field_decays_far_from_coils():
    """Far from both coils, |Bz| should be small compared to its peak."""
    z_far = np.array([0.29])
    z_near_peak = np.array([gsf.COIL_SEPARATION_M / 2])
    assert abs(gsf.two_coil_Bz(z_far)[0]) < 0.1 * abs(gsf.two_coil_Bz(z_near_peak)[0])


def test_generated_csv_roundtrips_through_provider(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sona.fields.csv_field import CSVFieldProvider

    out = tmp_path / "test_field.csv"
    gsf.generate(str(out))

    provider = CSVFieldProvider(
        str(out), br_col="Br", z_unit="cm", b_unit="G",
        br_reference_radius_m=gsf.BR_REFERENCE_RADIUS_M,
    )
    z0 = np.array([0.0])
    assert provider.Bz(z0)[0] == pytest.approx(0.0, abs=1e-9)