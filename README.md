# Sona Transition Simulation

Numerical propagation of hydrogen (H) hyperfine-substate populations through
a Sona transition unit, for two physically distinct cases sharing one code
path:

- **metastable 2S1/2** — validation against Kannis (2023) / Engels (2024),
  FZJ Lamb-shift polarimeter data.
- **ground-state 1S1/2** — the BNL RHIC OPPIS, using measured magnetic field
  maps processed by `MagScan_Ana4.py`.

## Status

- `constants.py` — done, tested (4 tests).
- `hamiltonian.py` — done, tested (21 tests, including a numeric-vs-analytic
  Breit-Rabi cross-check across both atom states).
- Everything else is still a stub — see each file's docstring for what it
  will contain and in what order we're building it.

## Setup (Windows / VS Code)

Create the virtual environment **outside** this OneDrive-synced folder to
avoid sync lag / file locks:

```powershell
python -m venv C:\Users\acannavo\venvs\sona
C:\Users\acannavo\venvs\sona\Scripts\activate
pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → point at
`C:\Users\acannavo\venvs\sona\Scripts\python.exe`.

## Layout

```
src/sona/
    constants.py       physical constants + swappable atom parameters
    hamiltonian.py      H0 (fixed B=0 eigenbasis) + V(r,t) (Eq. 6.23)
    propagator.py       matrix-exponential stepper, unitary by construction
    beam.py             radial / Gaussian beam averaging
    sweep.py            current-sweep orchestration -> |c_k|^2(I)
    fields/
        base.py            common field-provider interface
        two_solenoid.py    Phase 1: analytic two-coil field
        measured_map.py    Phase 2: wraps MagScan_Ana4 output

tests/              pytest: norm conservation, adiabatic limit, etc.
scripts/            runnable drivers (e.g. Kannis reproduction)
data/raw/           input CSVs (gitignored)
data/processed/     cached/derived data (gitignored)
results/figures/    output plots (gitignored)
```

## Build order

1. `constants.py` — done.
2. `hamiltonian.py` — done. H0 (fixed B=0 eigenbasis) + interaction matrix
   V(r,t), derived independently from J/I ladder operators and verified
   against Kannis Eq. 6.23/6.24, plus a numeric-vs-closed-form Breit-Rabi
   agreement test (Eq. 3.18a-d) across a field sweep for both atom states.
3. `fields/two_solenoid.py` — analytic Bz(z), Br(r,z).
4. `propagator.py` — single-particle stepper + self-tests.
5. `beam.py`, `sweep.py` — beam averaging + current sweep.
6. Reproduce Kannis Fig. 6.10 → validation checkpoint.
7. `fields/measured_map.py` — swap in real OPPIS data.