"""
hamiltonian.py
==============
Builds the atomic Hamiltonian H(t) = H0 + V(r,t) for the Sona-transition
propagator, in the FIXED B=0 eigenbasis of the field-free hyperfine
Hamiltonian H0 = A * I.J.

Basis (hydrogen, J = I = 1/2)
------------------------------
The four basis states are the B=0 limits of the field-dependent hyperfine
eigenstates alpha1, alpha2, beta3, beta4 (Kannis Table 3.1 / Table 6.1):

    |1> = |mJ=+1/2, mI=+1/2>                                  (alpha1 @ B=0)
    |2> = (|mJ=-1/2,mI=+1/2> + |mJ=+1/2,mI=-1/2>) / sqrt(2)    (alpha2 @ B=0)
    |3> = |mJ=-1/2, mI=-1/2>                                  (beta3  @ B=0)
    |4> = (|mJ=-1/2,mI=+1/2> - |mJ=+1/2,mI=-1/2>) / sqrt(2)    (beta4  @ B=0)

States 1,2,3 are the F=1 triplet (E=+A/4); state 4 is the F=0 singlet
(E=-3A/4) -- the familiar hydrogen 21 cm hyperfine structure. This basis
is FIXED -- it does not change as the field changes. Everything the field
does is packed into V(r,t).

H0
--
Diagonal in this basis: H0 = diag(A/4, A/4, A/4, -3A/4). Field-independent.

V(r,t)
------
The Zeeman interaction -(gJ mu_B J + gI mu_N I).B, with
B = Bz(t) zhat + Br(r,t) rhat(phi), expressed as matrix elements in the
basis above. Derived here directly from the J+/J-/I+/I- raising and
lowering algebra on the uncoupled |mJ,mI> states (spin-1/2 ladder
operators have no sqrt prefactor, which keeps this clean), then rotated
into the fixed basis.

This was cross-checked two independent ways against Kannis (2023):
    1. Matrix form -- every one of the 16 entries reproduces Eq. 6.23.
    2. ODE form -- inserting these matrix elements into Eq. 6.16 and
       adding the phase factors e^{i*omega_kj*t} reproduces Eq. 6.24a-d
       term for term.
    3. (Implemented as an automatic test, not just a claim.) Diagonalizing
       H0 + V(Bz, Br=0) at any field must reproduce the closed-form
       Breit-Rabi energies E1..E4(B), Eq. 3.18a-d -- see
       breit_rabi_energies_hydrogen() below and
       tests/test_hamiltonian.py.

Populations are independent of the azimuthal angle phi (Kannis Sec. 6.1);
it appears only as an overall phase. Callers may safely pass phi=0.

Deuterium (I=1, 6-state, Kannis Eq. 6.25 / Sec. 6.1.2) is NOT yet
implemented -- functions below raise NotImplementedError for I != 1/2
rather than silently returning wrong-sized matrices.
"""

import numpy as np

from sona.constants import HyperfineParams, MU_B, MU_N

SQRT2 = np.sqrt(2.0)

# Basis label reference, in the fixed B=0 eigenbasis (hydrogen):
BASIS_LABELS_H = ("alpha1", "alpha2", "beta3", "beta4")


def _check_is_hydrogen(params: HyperfineParams) -> None:
    if params.I_spin != 0.5:
        raise NotImplementedError(
            f"I_spin={params.I_spin} (deuterium-like) is not implemented. "
            "This module currently supports hydrogen (I=1/2) only -- see "
            "Kannis Eq. 6.25 for the deuterium H0 this would need."
        )


def H0_hydrogen(params: HyperfineParams) -> np.ndarray:
    """
    Field-free hyperfine Hamiltonian, diagonal in the fixed basis
    (alpha1, alpha2, beta3, beta4) @ B=0.

    Returns
    -------
    (4,) real ndarray of eigenenergies [J], order matching BASIS_LABELS_H.
    """
    _check_is_hydrogen(params)
    A = params.A_J
    return np.array([A / 4.0, A / 4.0, A / 4.0, -3.0 * A / 4.0])


def interaction_matrix_hydrogen(Bz: float, Br: float, phi: float,
                                 params: HyperfineParams) -> np.ndarray:
    """
    <k|V(r,t)|j> for hydrogen -- 4x4 Hermitian, complex128, in the same
    fixed basis as H0_hydrogen.

    Parameters
    ----------
    Bz, Br : float   Tesla. Longitudinal / radial field components.
    phi    : float   radians. Azimuthal angle; populations don't depend
                      on it (pass 0.0 unless you have a specific reason).
    params : HyperfineParams

    Returns
    -------
    (4,4) complex128 ndarray.
    """
    _check_is_hydrogen(params)

    alpha = params.gJ * MU_B   # electron magnetic-moment coefficient
    beta = params.gI * MU_N    # nuclear magnetic-moment coefficient
    ephi_m = np.exp(-1j * phi)
    ephi_p = np.exp(1j * phi)

    V = np.zeros((4, 4), dtype=complex)

    # -- diagonal (Bz only) --
    V[0, 0] = -Bz / 2.0 * (alpha + beta)
    V[1, 1] = 0.0
    V[2, 2] = Bz / 2.0 * (alpha + beta)
    V[3, 3] = 0.0

    # -- (1,2) / (2,1): couples alpha1 <-> alpha2 via Br, (alpha+beta) --
    V[0, 1] = -(Br / (2.0 * SQRT2)) * (alpha + beta) * ephi_m
    V[1, 0] = np.conj(V[0, 1])

    # -- (1,3) / (3,1): no direct coupling --
    V[0, 2] = 0.0
    V[2, 0] = 0.0

    # -- (1,4) / (4,1): couples alpha1 <-> beta4 via Br, (alpha-beta) --
    V[0, 3] = -(Br / (2.0 * SQRT2)) * (alpha - beta) * ephi_m
    V[3, 0] = np.conj(V[0, 3])

    # -- (2,3) / (3,2): couples alpha2 <-> beta3 via Br, (alpha+beta) --
    V[1, 2] = -(Br / (2.0 * SQRT2)) * (alpha + beta) * ephi_m
    V[2, 1] = np.conj(V[1, 2])

    # -- (2,4) / (4,2): couples alpha2 <-> beta4 via Bz, (alpha-beta) --
    V[1, 3] = Bz / 2.0 * (alpha - beta)
    V[3, 1] = V[1, 3]   # real -> Hermitian conjugate is itself

    # -- (3,4) / (4,3): couples beta3 <-> beta4 via Br, (alpha-beta) --
    V[2, 3] = (Br / (2.0 * SQRT2)) * (alpha - beta) * ephi_p
    V[3, 2] = np.conj(V[2, 3])

    return V


def build_H_hydrogen(Bz: float, Br: float, phi: float,
                      params: HyperfineParams) -> np.ndarray:
    """
    Full Schrödinger-picture Hamiltonian H(t) = H0 + V(r,t), 4x4 complex128,
    ready to hand to the propagator (matrix exponential or solve_ivp).
    """
    H0 = np.diag(H0_hydrogen(params)).astype(complex)
    V = interaction_matrix_hydrogen(Bz, Br, phi, params)
    return H0 + V


def breit_rabi_energies_hydrogen(B: float, params: HyperfineParams) -> np.ndarray:
    """
    Closed-form Breit-Rabi energies E1..E4(B) for hydrogen, on-axis field
    only (Kannis Eq. 3.18a-d). Used as an independent analytic oracle to
    validate build_H_hydrogen(Bz=B, Br=0, ...) -- NOT used by the
    propagator itself.

    Returns
    -------
    (4,) real ndarray: (E1, E2, E3, E4) matching (alpha1, alpha2, beta3,
    beta4) at high field, in Joules.
    """
    _check_is_hydrogen(params)
    A = params.A_J
    alpha = params.gJ * MU_B
    beta = params.gI * MU_N

    E1 = A / 4.0 - 0.5 * (alpha + beta) * B
    E3 = A / 4.0 + 0.5 * (alpha + beta) * B
    disc = np.sqrt(A ** 2 + (alpha - beta) ** 2 * B ** 2)
    E2 = -A / 4.0 + 0.5 * disc
    E4 = -A / 4.0 - 0.5 * disc
    return np.array([E1, E2, E3, E4])