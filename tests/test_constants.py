"""
Sanity checks for sona.constants -- these are the numbers everything else
gets built on, so they're worth pinning down with a real test rather than
eyeballing a printout.

Reference values from Kannis (2023):
    metastable 2S1/2:  A = 177.556 MHz,  B_c = 6.34 mT
    ground 1S1/2:      B_c = 50.76 mT  (Sec. 4.1.1)
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sona.constants import AtomState, get_hyperfine_params


def test_metastable_B_critical():
    p = get_hyperfine_params(AtomState.METASTABLE_2S)
    assert math.isclose(p.B_critical_T * 1e3, 6.34, rel_tol=0.02)


def test_ground_B_critical():
    p = get_hyperfine_params(AtomState.GROUND_1S)
    assert math.isclose(p.B_critical_T * 1e3, 50.76, rel_tol=0.02)


def test_ground_over_metastable_ratio_matches_A_ratio():
    """B_c scales linearly with A since g-factors are shared -- the ratio
    should match the A ratio (~8x) regardless of the exact g-factor values
    used, which is a stronger check than either absolute value alone."""
    meta = get_hyperfine_params(AtomState.METASTABLE_2S)
    gnd = get_hyperfine_params(AtomState.GROUND_1S)
    ratio_A = gnd.A_Hz / meta.A_Hz
    ratio_Bc = gnd.B_critical_T / meta.B_critical_T
    assert math.isclose(ratio_A, ratio_Bc, rel_tol=1e-9)


def test_swap_is_one_line():
    """The whole point of this module: swapping cases changes nothing
    except the AtomState argument."""
    a = get_hyperfine_params(AtomState.METASTABLE_2S)
    b = get_hyperfine_params(AtomState.GROUND_1S)
    assert a.gJ == b.gJ and a.gI == b.gI  # only A differs, as expected
    assert a.A_Hz != b.A_Hz
