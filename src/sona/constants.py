"""
constants.py
============
Universal physical constants, plus the atom-specific hyperfine parameters
that the rest of the simulation depends on.

Why this file exists on its own
--------------------------------
The propagator, the Hamiltonian, the field providers -- none of them care
*which* hydrogen level they're simulating. The physics only differs through
one number, the hyperfine constant A (and, derived from it, the critical
field B_c = A / (|g_J| mu_B + g_I mu_N)):

    metastable 2S1/2 (FZJ, Kannis/Engels validation)   A ~ 177.6 MHz
    ground-state 1S1/2 (BNL OPPIS)                     A ~ 1420.4 MHz

Everything downstream should ask this module for a HyperfineParams object
rather than hard-coding A anywhere. Swapping metastable <-> ground state
is then a one-line change: pass a different AtomState to
get_hyperfine_params().

References
----------
Kannis (2023) thesis, Table 5.2 (A values), Sec. 3.1.1 (B_c definition).
Hydrogen ground-state hyperfine (21 cm line): A/h = 1420.405751 MHz
(CODATA / standard reference value).
"""

from dataclasses import dataclass
from enum import Enum
import math

# ---------------------------------------------------------------------------
# Universal physical constants (SI units)
# ---------------------------------------------------------------------------
H_PLANCK   = 6.62607015e-34        # J s        Planck constant
HBAR       = H_PLANCK / (2.0 * math.pi)   # J s
MU_B       = 9.2740100783e-24      # J/T        Bohr magneton
MU_N       = 5.0507837461e-27      # J/T        nuclear magneton
E_CHARGE   = 1.602176634e-19       # C
M_ELECTRON = 9.1093837015e-31      # kg
M_PROTON   = 1.67262192369e-27     # kg

# g-factors (dimensionless)
G_J_S = -2.00231930436   # electron Lande g-factor for an S state (J=1/2, L=0)
G_I_P = 5.5856946893     # proton g-factor
G_I_D = 0.8574382338     # deuteron g-factor


# ---------------------------------------------------------------------------
# Atom / level selector
# ---------------------------------------------------------------------------
class AtomState(Enum):
    """Which hydrogen level we're simulating."""
    METASTABLE_2S = "metastable_2S1/2"   # FZJ Lamb-shift setup (Kannis/Engels)
    GROUND_1S     = "ground_1S1/2"       # BNL RHIC OPPIS


@dataclass(frozen=True)
class HyperfineParams:
    """
    Everything the Hamiltonian needs that depends on WHICH level is being
    simulated. Immutable on purpose -- build a new one rather than mutating.
    """
    label:  str
    A_Hz:   float   # hyperfine constant A / h   [Hz]
    gJ:     float   # electron g-factor
    gI:     float   # nuclear g-factor (proton or deuteron)
    I_spin: float   # nuclear spin quantum number (1/2 for H, 1 for D)

    @property
    def A_J(self) -> float:
        """Hyperfine constant A in Joules (energy units, not A/h)."""
        return self.A_Hz * H_PLANCK

    @property
    def B_critical_T(self) -> float:
        """
        Critical field B_c [Tesla] -- the field scale separating the
        low-field (hyperfine-dominated) region from the high-field
        (Zeeman-dominated, Paschen-Back) region.

            B_c = A / (|g_J| mu_B + g_I mu_N)

        Kannis quotes B_c = 6.34 mT for metastable H; this formula
        reproduces that value from A_Hz = 177.556 MHz.
        """
        denom = abs(self.gJ) * MU_B + self.gI * MU_N
        return self.A_J / denom


# ---------------------------------------------------------------------------
# The swappable presets
# ---------------------------------------------------------------------------
HYPERFINE_PRESETS = {
    AtomState.METASTABLE_2S: HyperfineParams(
        label="metastable 2S1/2 (H)",
        A_Hz=177.556e6,          # Kannis (2023), Table 5.2
        gJ=G_J_S,
        gI=G_I_P,
        I_spin=0.5,
    ),
    AtomState.GROUND_1S: HyperfineParams(
        label="ground state 1S1/2 (H)",
        A_Hz=1420.405751e6,      # hydrogen 21 cm hyperfine transition
        gJ=G_J_S,
        gI=G_I_P,
        I_spin=0.5,
    ),
}


def get_hyperfine_params(state: AtomState = AtomState.METASTABLE_2S) -> HyperfineParams:
    """
    Single entry point for atom-dependent parameters.

    Example
    -------
    >>> from sona.constants import AtomState, get_hyperfine_params
    >>> p = get_hyperfine_params(AtomState.GROUND_1S)
    >>> p.A_Hz, p.B_critical_T
    """
    return HYPERFINE_PRESETS[state]


if __name__ == "__main__":
    # Quick sanity print -- run `python -m sona.constants` from src/.
    for state in AtomState:
        p = get_hyperfine_params(state)
        print(f"{p.label:28s}  A = {p.A_Hz/1e6:9.3f} MHz   "
              f"B_c = {p.B_critical_T*1e3:7.3f} mT")
