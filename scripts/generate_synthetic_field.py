"""
generate_synthetic_field.py
============================
Generates a synthetic "two opposed solenoids" magnetic field map and
writes it as a CSV in the same (z, Bz[, Br]) format CSVFieldProvider
reads -- so this script is a stand-in for OPERA2D until real exports
exist, exercising the exact same code path.

Physics: on-axis field of two circular current loops of radius `a`,
separated by distance `d`, carrying equal and opposite current (Engels
Fig. 2: "two opposing solenoids"). A single loop's on-axis field is

    B(z) = mu0 * N * I * a^2 / (2 * (a^2 + (z-z0)^2)^(3/2))

Placing one coil at +d/2 with +I and the other at -d/2 with -I gives a
field that is positive far downstream, negative far upstream, and
crosses zero at the midpoint by symmetry -- exactly the Sona reversal
shape, and it produces the near-sinusoidal profile through the
zero-crossing that Engels describes.

This is a genuine physical model (not an arbitrary curve), it just
isn't yet calibrated to a real OPPIS coil geometry -- that calibration
is what OPERA2D will eventually provide.

Run directly to (re)generate data/raw/synthetic_two_solenoid.csv:
    python scripts/generate_synthetic_field.py
"""

import os
import numpy as np
import pandas as pd

MU0 = 4 * np.pi * 1e-7   # T m / A

# ---- coil geometry / current -- edit to explore different configurations ----
COIL_RADIUS_M   = 0.05     # a  [m]
COIL_SEPARATION_M = 0.06   # d  [m]  -- try e.g. 0.006 vs 0.06 to mimic
                            #           Kannis's "minimum" vs "maximum" cases
CURRENT_A       = 5.0      # I  [A]
N_TURNS         = 100

Z_RANGE_M       = (-0.30, 0.30)
N_POINTS        = 241       # ~2.5 mm spacing

BR_REFERENCE_RADIUS_M = 0.003   # 3 mm, matching Kannis's measurement radius


def _single_loop_Bz(z, z0, current_signed):
    a = COIL_RADIUS_M
    return (MU0 * N_TURNS * current_signed * a ** 2
            / (2.0 * (a ** 2 + (z - z0) ** 2) ** 1.5))


def two_coil_Bz(z):
    """On-axis Bz(z) [T] for two opposed coils."""
    z_plus = COIL_SEPARATION_M / 2.0
    z_minus = -COIL_SEPARATION_M / 2.0
    return (_single_loop_Bz(z, z_plus, +CURRENT_A)
             + _single_loop_Bz(z, z_minus, -CURRENT_A))


def two_coil_dBz_dz(z, h=1e-7):
    """Central-difference derivative -- fine for generating test data
    (the CSVFieldProvider's own PCHIP derivative is what the propagator
    actually uses; this is only to populate an optional Br column)."""
    return (two_coil_Bz(z + h) - two_coil_Bz(z - h)) / (2 * h)


def generate(path, include_br_column=True):
    z = np.linspace(*Z_RANGE_M, N_POINTS)
    Bz = two_coil_Bz(z)

    data = {"z": z * 1e2, "Bz": Bz * 1e4}   # -> cm, G (matches MagScan convention)

    if include_br_column:
        r_ref = BR_REFERENCE_RADIUS_M
        dBz_dz = two_coil_dBz_dz(z)
        Br_at_ref = -(r_ref / 2.0) * dBz_dz
        data["Br"] = Br_at_ref * 1e4   # -> G

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    Bz_max_G = np.max(np.abs(Bz)) * 1e4
    print(f"[generate_synthetic_field] wrote {path}")
    print(f"  coil radius a = {COIL_RADIUS_M*100:.1f} cm, "
          f"separation d = {COIL_SEPARATION_M*100:.1f} cm, "
          f"I = {CURRENT_A} A, N = {N_TURNS}")
    print(f"  z range: {Z_RANGE_M[0]*100:.0f} to {Z_RANGE_M[1]*100:.0f} cm, "
          f"{N_POINTS} points")
    print(f"  max |Bz| = {Bz_max_G:.3f} G")


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..",
                             "data", "raw", "synthetic_two_solenoid.csv")
    generate(out_path)