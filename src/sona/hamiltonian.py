"""
hamiltonian.py
==============
STUB -- next module to build.

Will contain:
    - H0: the fixed, B=0 eigenbasis of A * I.J  (Kannis Eq. 6.17 / Table 6.1
      for H; Eq. 6.25 / Table 6.2 for D), returned as eigenenergies
      (E1..E4) given a HyperfineParams.
    - interaction_matrix(Bz, Br, params): the 4x4 (H) or 6x6 (D) matrix
      <k|V(r,t)|j> from Eq. 6.23, built from the instantaneous field
      components.
    - build_H(Bz, Br, params): H0 + V, ready to hand to the propagator.

Depends only on: sona.constants
"""

raise NotImplementedError("hamiltonian.py not yet built -- see README build order")
