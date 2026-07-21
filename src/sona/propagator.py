"""
propagator.py
=============
STUB.

Will contain the matrix-exponential (short-time unitary) stepper:

    a(t + dt) = expm(-1j * H(t + dt/2) * dt / hbar) @ a(t)

Exactly unitary at every step by construction (H Hermitian), independent
of step size -- norm conservation is not something to check after the
fact, it's guaranteed. A generic scipy.integrate.solve_ivp path will be
added alongside as an independent cross-check, not a replacement.

Self-tests to live in tests/test_propagator.py:
    - norm conservation: sum(|a_k|^2) == 1 at every step
    - adiabatic limit: slow / non-reversing field leaves population on
      the initial eigenstate
    - agreement between the matrix-exponential and solve_ivp propagators

Depends on: sona.constants, sona.hamiltonian
"""

raise NotImplementedError("propagator.py not yet built -- see README build order")
