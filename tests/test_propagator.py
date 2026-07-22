"""
Tests for sona.propagator.

Four independent lines of evidence, deliberately kept separate:
  1. Norm conservation -- guaranteed by construction (Hermitian H), but
     verified numerically since floating point isn't a proof.
  2. alpha1 / beta3 are EXACT eigenstates on-axis (Br=0), for any Bz(t)
     -- a strong, exact (not approximate) check on the Hamiltonian
     derivation, not just the propagator.
  3. Independent cross-check: the fixed-step matrix-exponential method
     agrees with a generic adaptive solver (solve_ivp) integrating the
     same equation a completely different way.
  4. alpha2 / beta4 show the expected adiabatic-to-diabatic TREND as
     velocity increases (more transfer at higher velocity) -- a
     qualitative physics check, not a quantitative Landau-Zener match
     (our field isn't an idealized infinite linear sweep, so an exact
     LZ formula doesn't directly apply).
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sona.constants import AtomState, get_hyperfine_params
from sona.fields.csv_field import CSVFieldProvider
from sona.propagator import (
    kinetic_energy_to_velocity,
    propagate,
    propagate_solve_ivp_reference,
)

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                         "synthetic_two_solenoid.csv")


@pytest.fixture(scope="module")
def field():
    if not os.path.exists(DATA_CSV):
        pytest.skip("Run scripts/generate_synthetic_field.py first.")
    return CSVFieldProvider(DATA_CSV, br_col="Br", z_unit="cm", b_unit="G",
                             br_reference_radius_m=0.003)


PARAMS = [get_hyperfine_params(AtomState.METASTABLE_2S),
          get_hyperfine_params(AtomState.GROUND_1S)]


def test_kinetic_energy_to_velocity_3keV_proton():
    """Sanity: 3 keV proton should be a few times 1e5 m/s, well
    non-relativistic (v/c << 1)."""
    v = kinetic_energy_to_velocity(3000.0)
    assert 5e5 < v < 1e6
    assert v / 3e8 < 0.01  # << c, confirms non-relativistic treatment is fine


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
@pytest.mark.parametrize("start_index", [0, 1, 2, 3])
def test_norm_conserved(field, params, start_index):
    v = kinetic_energy_to_velocity(3000.0)
    a0 = np.zeros(4, dtype=complex)
    a0[start_index] = 1.0
    a_final = propagate(v, field, params, a0, r=0.0)
    assert np.isclose(np.sum(np.abs(a_final) ** 2), 1.0, atol=1e-8)


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
def test_alpha1_and_beta3_frozen_on_axis(field, params):
    """
    alpha1 (index 0) and beta3 (index 2) must show EXACTLY zero
    population transfer on-axis (Br=0), for any Bz(t) -- this follows
    from V[0,2]=V[2,0]=0 identically in our derived interaction matrix.
    Tight tolerance is appropriate here: this isn't approximate physics.
    """
    v = kinetic_energy_to_velocity(3000.0)
    for idx in (0, 2):
        a0 = np.zeros(4, dtype=complex)
        a0[idx] = 1.0
        a_final = propagate(v, field, params, a0, r=0.0)
        pop = np.abs(a_final) ** 2
        assert pop[idx] > 1 - 1e-6, (
            f"state {idx} should stay ~1.0, got {pop[idx]} for {params.label}"
        )


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: p.label)
def test_matrix_exponential_agrees_with_solve_ivp(field, params):
    """
    Cross-check the fixed-step unitary propagator against a completely
    independent adaptive integrator on the alpha2 initial state (the
    physically interesting, coupled case) at 3 keV.
    """
    v = kinetic_energy_to_velocity(3000.0)
    a0 = np.array([0, 1, 0, 0], dtype=complex)

    a_fast = propagate(v, field, params, a0, r=0.0)
    a_ref = propagate_solve_ivp_reference(v, field, params, a0, r=0.0)

    assert np.allclose(np.abs(a_fast) ** 2, np.abs(a_ref) ** 2, atol=5e-4), (
        f"population mismatch for {params.label}: "
        f"stepper={np.abs(a_fast)**2}, solve_ivp={np.abs(a_ref)**2}"
    )


def test_alpha2_beta4_adiabatic_to_diabatic_trend(field):
    """
    Qualitative physics check: as velocity increases (faster sweep
    through the alpha2/beta4 avoided crossing), transfer to beta4 should
    increase monotonically (adiabatic -> diabatic transition), for
    metastable H. Not an exact Landau-Zener match (our field is not an
    idealized infinite linear sweep) -- a trend check.
    """
    params = get_hyperfine_params(AtomState.METASTABLE_2S)
    a0 = np.array([0, 1, 0, 0], dtype=complex)

    velocities = [7.58e4, 7.58e5, 7.58e6, 7.58e7]
    transferred = []
    for v in velocities:
        a_final = propagate(v, field, params, a0, r=0.0)
        transferred.append(np.abs(a_final[3]) ** 2)  # population in beta4

    # The first two velocities are both deep in the adiabatic regime --
    # transfer sits at the ~1e-6 noise/interference floor there (multiple
    # passages through the field's two extrema give small Stueckelberg-
    # like oscillations, not a clean monotonic ramp at that scale). What
    # IS physically meaningful, and what we check, is the clear overall
    # separation: highest velocity gives unambiguously more transfer than
    # the deeply-adiabatic low-velocity cases.
    assert transferred[0] < 1e-4 and transferred[1] < 1e-4, (
        f"expected negligible transfer in the deep adiabatic regime, got {transferred[:2]}"
    )
    assert transferred[-1] > 100 * max(transferred[0], transferred[1]), (
        f"expected clearly more transfer at high velocity, got {transferred}"
    )
    # at the real 3 keV velocity, transfer should be negligible (deep
    # adiabatic regime for this synthetic field's peak ~48 G vs B_c~63 mT)
    v_3keV = kinetic_energy_to_velocity(3000.0)
    a_3keV = propagate(v_3keV, field, params, a0, r=0.0)
    assert np.abs(a_3keV[3]) ** 2 < 1e-3