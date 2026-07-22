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
- `fields/` — done, tested (13 tests). Generic CSVFieldProvider handles
  synthetic test data now, OPERA2D exports (any geometry) later.
- `propagator.py` — done, tested (18 tests). Matrix-exponential stepper
  (unitary by construction) + solve_ivp cross-check + kinetic-energy ->
  velocity helper. On-axis (r=0, Br=0) validated two ways:
    - alpha1 / beta3 shown to be EXACT eigenstates (zero transfer, any
      Bz(t)) -- confirms the Hamiltonian derivation, not just the stepper.
    - alpha2 / beta4 show genuine adiabatic <-> diabatic crossing physics
      as velocity is scanned.
  **Finding:** at the real 3 keV OPPI proton velocity, the synthetic
  coil's peak field (~48 G) is below B_c (63 mT metastable), so the
  on-axis alpha2/beta4 crossing is deeply adiabatic -- negligible
  population transfer. On-axis alone will not reproduce Sona
  oscillations at realistic energy; the interesting physics needs the
  off-axis Br coupling (alpha1<->alpha2, alpha1<->beta4, etc.), which
  needs a finite beam radius.
- **Off-axis exploration** (`scripts/inspect_offaxis.py`, exploratory --
  not part of the tested core, no assertions) -- confirmed the above:
  starting in alpha1 at r=0 it's frozen (matches the on-axis proof);
  turning on r makes the off-diagonal alpha1<->alpha2 and alpha1<->beta4
  matrix elements nonzero and population visibly transfers, growing with
  r. An extended (unphysically large radius) scan shows genuine
  Rabi-oscillation behavior -- population returns toward alpha1 and
  ripples at larger r -- confirming the coupling mechanism works as
  designed, though real Sona oscillations are seen sweeping *coil
  current* at a fixed physical beam radius, which is what `sweep.py`
  will do.
- Everything else is still a stub — see each file's docstring for what it
  will contain and in what order we're building it.

## Setup (Windows / VS Code / Anaconda)

Using Anaconda rather than a bare venv avoids OneDrive sync issues
automatically, since conda environments live under your Anaconda
install (e.g. `C:\Users\<you>\AppData\Local\anaconda3\envs\`), outside
any OneDrive-synced folder.

From an **Anaconda Prompt**:

```powershell
conda create -n sona python=3.11
conda activate sona
cd "C:\Users\<you>\OneDrive - Brookhaven National Laboratory\SonaSimulation"
pip install -r requirements.txt
```

First time using `conda activate` in a regular PowerShell terminal
(rather than Anaconda Prompt)? Run `conda init powershell` once, then
close and reopen the terminal.

In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the
`sona` environment (path ends in `envs\sona\python.exe`).

## Layout

```
src/sona/
    constants.py       physical constants + swappable atom parameters
    hamiltonian.py      H0 (fixed B=0 eigenbasis) + V(r,t) (Eq. 6.23)
    propagator.py       matrix-exponential stepper, unitary by construction
    beam.py             radial / Gaussian beam averaging
    sweep.py            current-sweep orchestration -> |c_k|^2(I)
    fields/
        base.py            FieldProvider interface: Bz(z), dBz_dz(z), Br(r,z)
        csv_field.py        CSVFieldProvider -- the one implementation,
                             used for synthetic test fields, OPERA2D
                             exports (any geometry), and eventually
                             wrapped MagScan_Ana4 measurements

tests/              pytest: norm conservation, adiabatic limit, etc.
scripts/            runnable drivers
    generate_synthetic_field.py   synthetic two-opposed-coil test field
                                   (on-axis Biot-Savart), stand-in for
                                   OPERA2D until real exports exist
    inspect_offaxis.py            EXPLORATORY, not tested -- prints the
                                   Hamiltonian/eigenstates at a field
                                   point and plots field shape (Bz, Br)
                                   and off-axis trajectories, for
                                   building intuition. Run any time:
                                       python scripts/inspect_offaxis.py
data/raw/           input CSVs (gitignored -- regenerate via scripts/)
data/processed/     cached/derived data (gitignored)
results/figures/    output plots (gitignored)
```

## Build order

1. `constants.py` — done.
2. `hamiltonian.py` — done. H0 (fixed B=0 eigenbasis) + interaction matrix
   V(r,t), derived independently from J/I ladder operators and verified
   against Kannis Eq. 6.23/6.24, plus a numeric-vs-closed-form Breit-Rabi
   agreement test (Eq. 3.18a-d) across a field sweep for both atom states.
3. `fields/` — done. Generic CSVFieldProvider (z, Bz[, Br]) with PCHIP
   interpolation and an exact analytic derivative for dBz/dz; Br derived
   via the Maxwell relation if not supplied. Validated against a known
   analytic tanh-field CSV (interpolation correctness) and a physically
   real synthetic two-opposed-coil Biot-Savart field (Sona-reversal
   symmetry checks). Same code path will read future OPERA2D exports.
4. `propagator.py` — done. Matrix-exponential stepper + solve_ivp cross-
   check, validated on-axis (r=0): alpha1/beta3 exactly frozen, alpha2/
   beta4 show real adiabatic-to-diabatic crossing physics vs velocity.
   Finding: on-axis alone is too adiabatic at 3 keV to reproduce Sona
   oscillations -- off-axis (Br) coupling is essential.
   Off-axis (r>0) explored via `scripts/inspect_offaxis.py`: confirms
   alpha1 population transfers once Br != 0, growing with r, with
   genuine Rabi-oscillation structure at larger (unphysical) r. Uses the
   same `propagate()` -- no new production code needed for r>0, since it
   was written generally from the start.
5. `beam.py`, `sweep.py` — beam averaging + current sweep -- next.
6. Reproduce Kannis Fig. 6.10 → validation checkpoint.
7. Swap in real OPERA2D / OPPIS field data.