"""
base.py
=======
STUB -- defines the interface, not yet implemented.

The common contract every field provider (two_solenoid, measured_map)
must satisfy, so sweep.py and propagator.py never need to know which one
they're talking to:

    class FieldProvider:
        def Bz(self, z: np.ndarray) -> np.ndarray: [Tesla]
        def dBz_dz(self, z: np.ndarray) -> np.ndarray: [Tesla / m]
        def Br(self, r: float, z: np.ndarray) -> np.ndarray:
            # via Br = -(r/2) * dBz/dz, OR measured Br if available
        def scale(self, current: float) -> "FieldProvider":
            # returns a rescaled copy for a different coil current

Depends on: numpy only
"""

raise NotImplementedError("fields/base.py not yet built -- see README build order")
