"""
propagator.py
==============
Propagates the hydrogen hyperfine-state amplitude vector a(t) through a
field trajectory Bz(z), Br(r,z), converting spatial position to time via
z = v*t for a particle moving at constant velocity v.

Method: piecewise-constant unitary propagation (short-time matrix
exponential). Over each small step dt, H(t) is evaluated at the step
midpoint and treated as constant; the exact propagator for a constant
Hermitian H is

    a(t+dt) = exp(-i H dt / hbar) a(t)

computed via eigendecomposition of H (H is Hermitian, so this is exact
and numerically cheap for a 4x4 matrix). Because H is Hermitian at every
step, this propagator is unitary to machine precision REGARDLESS of step
size -- norm conservation is not something to check after the fact and
hope holds, it's guaranteed by construction. See
propagate_and_check_norm() and tests/test_propagator.py.

An independent cross-check against a generic adaptive ODE solver
(scipy.integrate.solve_ivp) is provided in
propagate_solve_ivp_reference() -- if the two methods disagree, that's
the thing to investigate before trusting either.

On-axis vs off-axis
--------------------
This module is written generally (radius r is a parameter of
propagate()), but for now we are only validating the r=0 (on-axis, Br=0)
case. At Br=0:
    - alpha1 (index 0) and beta3 (index 2) are EXACT eigenstates at every
      instant, for any Bz(t) -- V[0,2]=V[2,0]=0 identically and neither
      couples to anything else without Br. A particle prepared in either
      state shows zero population transfer, always. This is a strong,
      exact correctness check on the Hamiltonian (see
      test_alpha1_beta3_are_frozen_on_axis).
    - alpha2 (index 1) and beta4 (index 3) ARE coupled via Bz alone
      (V[1,3] != 0 with no Br needed) -- this is the real on-axis
      physics: the avoided crossing visible as the two curved branches
      in the Breit-Rabi diagram. Slow field reversal drives adiabatic
      following (population tends to swap diabatic identity across the
      crossing); fast reversal tends to preserve the initial diabatic
      state. See test_alpha2_beta4_adiabatic_vs_diabatic_trend.
"""

import numpy as np
from scipy.integrate import solve_ivp

from sona.constants import HyperfineParams, HBAR, E_CHARGE, M_PROTON
from sona.hamiltonian import build_H_hydrogen


def kinetic_energy_to_velocity(KE_eV: float, mass_kg: float = M_PROTON) -> float:
    """
    Non-relativistic KE -> speed [m/s]. Valid here: even at a few keV,
    v/c ~ 1e-3 for a proton, so relativistic corrections are irrelevant
    (gamma - 1 ~ 3e-6 at 3 keV).

    Example: kinetic_energy_to_velocity(3000.0) -> proton speed at 3 keV.
    """
    KE_J = KE_eV * E_CHARGE
    return np.sqrt(2.0 * KE_J / mass_kg)


def _unitary_step(H: np.ndarray, dt: float) -> np.ndarray:
    """exp(-i H dt / hbar) via eigendecomposition. H must be Hermitian."""
    eigvals, eigvecs = np.linalg.eigh(H)
    phase = np.exp(-1j * eigvals * dt / HBAR)
    return (eigvecs * phase) @ eigvecs.conj().T


def estimate_dt(field, v_mps: float, params: HyperfineParams, r: float = 0.0,
                 phi: float = 0.0, oversample: int = 30, n_sample: int = 400) -> float:
    """
    Pick a fixed step size dt that resolves the fastest energy splitting
    the trajectory will encounter, oversampled by `oversample`.

    Scans the field's z-range, builds H at each sample point, and uses
    the largest |Ek - Ej| found anywhere along the trajectory (a
    conservative, not adaptive, choice -- simple and safe for a first
    implementation).
    """
    z_min, z_max = field.z_range()
    z_samples = np.linspace(z_min, z_max, n_sample)
    max_gap = 0.0
    for z in z_samples:
        Bz = float(field.Bz(np.array([z]))[0])
        Br = float(field.Br(r, np.array([z]))[0]) if r > 0 else 0.0
        H = build_H_hydrogen(Bz, Br, phi, params)
        eigvals = np.linalg.eigvalsh(H)
        gap = eigvals[-1] - eigvals[0]
        max_gap = max(max_gap, gap)

    if max_gap <= 0.0:
        # Degenerate/trivial field (e.g. Bz=0 everywhere) -- fall back to
        # the hyperfine splitting A, which is always present.
        max_gap = params.A_J

    omega_max = max_gap / HBAR
    return (2.0 * np.pi) / (oversample * omega_max)


def propagate(v_mps: float, field, params: HyperfineParams,
              initial_state: np.ndarray, r: float = 0.0, phi: float = 0.0,
              dt: float = None, oversample: int = 30,
              return_trajectory: bool = False):
    """
    Propagate a single particle at radius r, speed v_mps, through the
    field's full z-range, using piecewise-constant unitary steps.

    Parameters
    ----------
    v_mps : float          particle speed [m/s]
    field : FieldProvider   e.g. a CSVFieldProvider
    params : HyperfineParams
    initial_state : (4,) complex array, should be normalized
    r : float               radius [m]; r=0 means on-axis, Br forced to 0
    phi : float              azimuthal angle [rad]; populations don't depend on it
    dt : float or None       fixed step [s]; if None, computed via estimate_dt
    oversample : int         passed to estimate_dt if dt is None
    return_trajectory : bool  if True, also return arrays of t, z, |a_k|^2

    Returns
    -------
    a_final : (4,) complex ndarray
    trajectory : dict (only if return_trajectory=True), keys 't','z','pop'
    """
    initial_state = np.asarray(initial_state, dtype=complex)
    if not np.isclose(np.sum(np.abs(initial_state) ** 2), 1.0, atol=1e-9):
        raise ValueError("initial_state must be normalized (sum |a_k|^2 = 1).")

    z_min, z_max = field.z_range()
    if dt is None:
        dt = estimate_dt(field, v_mps, params, r=r, phi=phi, oversample=oversample)

    t_span = (z_max - z_min) / v_mps
    n_steps = int(np.ceil(t_span / dt))
    dt = t_span / n_steps  # exact fit, avoids overshooting z_max

    a = initial_state.copy()

    if return_trajectory:
        t_arr = np.empty(n_steps + 1)
        z_arr = np.empty(n_steps + 1)
        pop_arr = np.empty((n_steps + 1, len(initial_state)))
        t_arr[0], z_arr[0] = 0.0, z_min
        pop_arr[0] = np.abs(a) ** 2

    for i in range(n_steps):
        t_mid = (i + 0.5) * dt
        z_mid = z_min + v_mps * t_mid
        Bz = float(field.Bz(np.array([z_mid]))[0])
        Br = float(field.Br(r, np.array([z_mid]))[0]) if r > 0 else 0.0

        H_mid = build_H_hydrogen(Bz, Br, phi, params)
        U = _unitary_step(H_mid, dt)
        a = U @ a

        if return_trajectory:
            t_arr[i + 1] = (i + 1) * dt
            z_arr[i + 1] = z_min + v_mps * t_arr[i + 1]
            pop_arr[i + 1] = np.abs(a) ** 2

    if return_trajectory:
        return a, {"t": t_arr, "z": z_arr, "pop": pop_arr}
    return a


def propagate_solve_ivp_reference(v_mps: float, field, params: HyperfineParams,
                                   initial_state: np.ndarray, r: float = 0.0,
                                   phi: float = 0.0, rtol: float = 1e-10,
                                   atol: float = 1e-12) -> np.ndarray:
    """
    Independent cross-check: integrate the same i*hbar*da/dt = H(t) a
    with a generic adaptive solver (solve_ivp, DOP853), rather than the
    fixed-step matrix-exponential method. solve_ivp doesn't accept
    complex state vectors directly, so the 4 complex amplitudes are
    packed into 8 real numbers and unpacked afterward.
    """
    initial_state = np.asarray(initial_state, dtype=complex)
    z_min, z_max = field.z_range()
    t_span = (0.0, (z_max - z_min) / v_mps)

    def rhs(t, y):
        a = y[:4] + 1j * y[4:]
        z = z_min + v_mps * t
        Bz = float(field.Bz(np.array([z]))[0])
        Br = float(field.Br(r, np.array([z]))[0]) if r > 0 else 0.0
        H = build_H_hydrogen(Bz, Br, phi, params)
        da_dt = (H @ a) / (1j * HBAR)
        return np.concatenate([da_dt.real, da_dt.imag])

    y0 = np.concatenate([initial_state.real, initial_state.imag])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=rtol, atol=atol)
    yf = sol.y[:, -1]
    return yf[:4] + 1j * yf[4:]