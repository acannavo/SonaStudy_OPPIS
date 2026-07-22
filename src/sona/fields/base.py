"""
base.py
=======
Common interface every magnetic-field provider implements, so the
propagator and sweep code never need to know whether the field came from
a synthetic test profile, an OPERA2D export (toy two-solenoid geometry
or the full OPPIS model), or -- eventually -- a measured OPPIS scan
wrapped from MagScan_Ana4.

All providers work in SI units (meters, Tesla) once loaded, regardless
of what units the source file used.
"""

import numpy as np


class FieldProvider:
    """Abstract interface. Concrete providers implement these methods."""

    def Bz(self, z):
        """Longitudinal field [T] at position(s) z [m]."""
        raise NotImplementedError

    def dBz_dz(self, z):
        """dBz/dz [T/m] at position(s) z [m]."""
        raise NotImplementedError

    def Br(self, r, z):
        """Radial field [T] at radius r [m], position(s) z [m]."""
        raise NotImplementedError

    def z_range(self):
        """(z_min, z_max) [m] over which this provider is valid."""
        raise NotImplementedError