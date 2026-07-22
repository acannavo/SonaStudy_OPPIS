"""
csv_field.py
============
Generic CSV-based magnetic field provider -- the ONE field provider both
the simplified two-solenoid test geometry and the full OPPIS geometry
use. Whether the CSV came from a synthetic test generator, an OPERA2D
export, or (eventually) a wrapped MagScan_Ana4 measurement, the
propagator only ever sees Bz(z), dBz_dz(z), Br(r,z) through this class.
Swapping geometries is "point this at a different CSV", not a code
change.

Expected CSV format
--------------------
A header row followed by numeric columns:

    z, Bz[, Br]

Column names are configurable (see __init__) rather than guessed, since
OPERA2D and lab-measurement exports don't agree on naming.

Units are configurable per axis and converted to SI (meters, Tesla) on
load. Defaults (cm, G) match the existing MagScan_Ana4 convention used
for OPPIS field scans.

Interpolation uses a monotonic cubic Hermite spline (PCHIP) -- the same
family Kannis (2023) used ("Hermite" in Mathematica) -- which avoids the
ringing a plain cubic spline can introduce near a sharp field reversal.
dBz/dz comes from the spline's exact analytic derivative, not a finite
difference: this removes step-size-dependent noise right at the
zero-crossing, where the gradient matters most (compare to
MagScan_Ana4.field_derivative, which uses a forward difference).

Br handling
-----------
If no Br column is supplied, Br is derived from the interpolated Bz via
the Maxwell relation (Kannis Eq. 3.30):

    Br(r, z) = -(r/2) * dBz/dz(z)

If a Br column IS supplied, it is assumed to have been measured/
simulated at one fixed reference radius (br_reference_radius_m) and is
scaled linearly with r for other radii -- valid near the axis, where Br
is proportional to r.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from sona.fields.base import FieldProvider

_Z_UNIT_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
_B_UNIT_TO_T = {"T": 1.0, "mT": 1e-3, "G": 1e-4, "kG": 1e-1}


class CSVFieldProvider(FieldProvider):
    """Load a (z, Bz[, Br]) field map from CSV and interpolate it."""

    def __init__(self, path, z_col="z", bz_col="Bz", br_col=None,
                 z_unit="cm", b_unit="G", br_reference_radius_m=None):
        """
        Parameters
        ----------
        path : str
            Path to the CSV file.
        z_col, bz_col, br_col : str
            Column header names in the CSV. br_col=None (default) means
            "no Br column in this file -- derive Br from Bz instead".
        z_unit : {'m', 'cm', 'mm'}
        b_unit : {'T', 'mT', 'G', 'kG'}
        br_reference_radius_m : float
            Required only if br_col is given: the radius [m] at which
            the Br column was measured/simulated.
        """
        if z_unit not in _Z_UNIT_TO_M:
            raise ValueError(f"z_unit must be one of {list(_Z_UNIT_TO_M)}")
        if b_unit not in _B_UNIT_TO_T:
            raise ValueError(f"b_unit must be one of {list(_B_UNIT_TO_T)}")
        if br_col is not None and br_reference_radius_m is None:
            raise ValueError(
                "br_reference_radius_m is required when br_col is given -- "
                "Br is only meaningful at the radius it was recorded at."
            )

        df = pd.read_csv(path)
        needed_cols = [z_col, bz_col] + ([br_col] if br_col else [])
        for col in needed_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {path}. "
                    f"Available columns: {list(df.columns)}"
                )

        df = df.sort_values(z_col).reset_index(drop=True)

        z_m = df[z_col].to_numpy(dtype=float) * _Z_UNIT_TO_M[z_unit]
        Bz_T = df[bz_col].to_numpy(dtype=float) * _B_UNIT_TO_T[b_unit]

        if len(z_m) < 4:
            raise ValueError("Need at least 4 points to build a PCHIP interpolant.")
        if not np.all(np.diff(z_m) > 0):
            raise ValueError(
                f"'{z_col}' must be strictly increasing after sorting -- "
                "check for duplicate z values in the CSV."
            )

        self._z_min, self._z_max = float(z_m[0]), float(z_m[-1])
        self._Bz_spline = PchipInterpolator(z_m, Bz_T, extrapolate=False)
        self._dBz_spline = self._Bz_spline.derivative()

        self._br_col_given = br_col is not None
        if self._br_col_given:
            Br_T = df[br_col].to_numpy(dtype=float) * _B_UNIT_TO_T[b_unit]
            self._Br_spline = PchipInterpolator(z_m, Br_T, extrapolate=False)
            self._r_ref = br_reference_radius_m
        else:
            self._Br_spline = None
            self._r_ref = None

    def z_range(self):
        return (self._z_min, self._z_max)

    def _check_bounds(self, z):
        z = np.asarray(z, dtype=float)
        if np.any(z < self._z_min) or np.any(z > self._z_max):
            raise ValueError(
                f"z outside field-map range [{self._z_min:.4g}, "
                f"{self._z_max:.4g}] m."
            )
        return z

    def Bz(self, z):
        z = self._check_bounds(z)
        return self._Bz_spline(z)

    def dBz_dz(self, z):
        z = self._check_bounds(z)
        return self._dBz_spline(z)

    def Br(self, r, z):
        z = self._check_bounds(z)
        if self._br_col_given:
            return self._Br_spline(z) * (r / self._r_ref)
        return -(r / 2.0) * self._dBz_spline(z)