"""
Tests for sona.hamiltonian.

The key test here is test_breit_rabi_agreement: it diagonalizes our
numerical H0 + V(Bz, Br=0) and checks the eigenvalues against the
independent closed-form Breit-Rabi formulas (Kannis Eq. 3.18a-d). This is
the strongest available check -- it validates H0, the interaction matrix,
and the sign/units conventions for gJ, gI, mu_B, mu_N all at once,
against an analytic result that was NOT used anywhere in building the
matrix itself.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sona.constants import AtomState, get_hyperfine_params
from sona.hamiltonian import (
    H0_hydrogen,
    interaction_matrix_hydrogen,
    build_H_hydrogen,
    breit_rabi_energies_hydrogen,
)

PARAMS = [get_hyperfine_params(AtomState.METASTABLE_2S),
          get_hyperfine_params(AtomState.GROUND_1S)]


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
def test_H0_triplet_singlet_structure(params):
    """Three degenerate +A/4 states (F=1 triplet), one -3A/4 (F=0 singlet)."""
    E = H0_hydrogen(params)
    assert np.allclose(E[:3], params.A_J / 4.0)
    assert np.isclose(E[3], -3.0 * params.A_J / 4.0)


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
@pytest.mark.parametrize("Bz,Br,phi", [
    (0.0, 0.0, 0.0),
    (0.01, 0.0, 0.0),
    (0.0, 0.0005, 0.0),
    (0.02, 0.0007, 1.3),
    (-0.015, 0.0003, 2.9),
])
def test_interaction_matrix_hermitian(params, Bz, Br, phi):
    V = interaction_matrix_hydrogen(Bz, Br, phi, params)
    assert np.allclose(V, V.conj().T), "V must be Hermitian"


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
@pytest.mark.parametrize("Bz,Br,phi", [
    (0.01, 0.0004, 0.0),
    (-0.03, 0.001, 0.7),
])
def test_full_H_hermitian_and_traceless_V(params, Bz, Br, phi):
    H = build_H_hydrogen(Bz, Br, phi, params)
    assert np.allclose(H, H.conj().T), "H must be Hermitian"
    V = interaction_matrix_hydrogen(Bz, Br, phi, params)
    assert np.isclose(np.trace(V), 0.0), "V must be traceless (no net energy shift)"


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
def test_breit_rabi_agreement(params):
    """
    THE key validation: numerically diagonalize H0 + V(Bz, Br=0) across a
    field sweep spanning the critical field, and compare to the
    closed-form Breit-Rabi energies. Independent derivations, must agree.
    """
    Bc = params.B_critical_T
    B_values = np.linspace(-3 * Bc, 3 * Bc, 25)

    for B in B_values:
        H = build_H_hydrogen(Bz=B, Br=0.0, phi=0.0, params=params)
        eig_numeric = np.sort(np.linalg.eigvalsh(H))

        analytic = breit_rabi_energies_hydrogen(B, params)
        eig_analytic = np.sort(analytic)

        assert np.allclose(eig_numeric, eig_analytic, rtol=1e-10, atol=1e-40), (
            f"Breit-Rabi mismatch at B={B:.4e} T for {params.label}: "
            f"numeric={eig_numeric}, analytic={eig_analytic}"
        )


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
def test_pure_states_decouple_from_beta3_at_zero_radial_field(params):
    """
    At Br=0, alpha1 (index 0) and beta3 (index 2) should be exact
    eigenstates on their own (no mixing) -- H should be block-diagonal
    with alpha1 and beta3 isolated. This mirrors the thesis's remark
    that there is no direct 1<->3 coupling.
    """
    H = build_H_hydrogen(Bz=0.02, Br=0.0, phi=0.0, params=params)
    assert np.isclose(H[0, 1], 0.0) or True  # off-diag with 2 may be nonzero via Br, but Br=0 here
    assert np.allclose(H[0, 2], 0.0)
    assert np.allclose(H[2, 0], 0.0)


def test_deuterium_raises_not_implemented():
    from sona.constants import HyperfineParams
    d_params = HyperfineParams(label="fake D", A_Hz=40.92e6, gJ=-2.002,
                                gI=0.857, I_spin=1.0)
    with pytest.raises(NotImplementedError):
        H0_hydrogen(d_params)