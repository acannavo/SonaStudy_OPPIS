"""
inspect_offaxis.py
===================
EXPLORATORY / throwaway script -- NOT part of the tested core (no
assertions, nothing else imports this). Purpose: let a human actually
SEE what the simulation is doing at each stage -- the raw Hamiltonian
matrix, its eigenstates, and off-axis trajectories -- rather than only
trusting pytest's green checkmarks.

Prints:
    - H0 eigenenergies and the full H matrix (on-axis vs off-axis) at a
      representative field point, so you can see the off-diagonal
      couplings switch on with r.
    - Eigenvalues/eigenvectors at that point.

Saves three figures to results/figures/:
    - offaxis_trajectories.png       alpha1 populations vs z, several r
    - exit_population_vs_radius.png  exit populations vs r (physical range)
    - exit_population_vs_radius_extended.png
          same, extended well past physical beam sizes -- shows the
          Rabi-like oscillatory return, illustrating the MECHANISM
          (population oscillates with coupling strength) without
          claiming this is what a real Kannis/Engels current-scan
          oscillation looks like -- that requires sweeping the FIELD
          (sweep.py, not yet built), at a fixed physical beam radius.

Run: python scripts/inspect_offaxis.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sona.constants import AtomState, get_hyperfine_params, E_CHARGE
from sona.hamiltonian import build_H_hydrogen, H0_hydrogen, BASIS_LABELS_H
from sona.fields.csv_field import CSVFieldProvider
from sona.propagator import kinetic_energy_to_velocity, propagate

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                         "synthetic_two_solenoid.csv")


def print_matrix_snapshot(params, field):
    z_probe = -0.04  # near the first coil's field extremum
    Bz = float(field.Bz(np.array([z_probe]))[0])
    r = 0.003  # 3 mm, Kannis's measurement radius
    Br = float(field.Br(r, np.array([z_probe]))[0])

    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(f"Basis order: {BASIS_LABELS_H}")
    print("H0 (B=0) eigenenergies [ueV]:", H0_hydrogen(params) / E_CHARGE * 1e6)
    print()
    print(f"At z={z_probe*100:.1f} cm, r={r*1000:.1f} mm:  "
          f"Bz={Bz*1e4:.3f} G, Br={Br*1e4:.3f} G")
    print()
    H_on = build_H_hydrogen(Bz, 0.0, 0.0, params) / E_CHARGE * 1e6
    H_off = build_H_hydrogen(Bz, Br, 0.0, params) / E_CHARGE * 1e6
    print("H [micro-eV], on-axis (Br=0):")
    print(H_on)
    print("H [micro-eV], off-axis (r=3mm) -- note new nonzero off-diagonals:")
    print(H_off)
    print()
    eigvals, eigvecs = np.linalg.eigh(build_H_hydrogen(Bz, Br, 0.0, params))
    print("Eigenvalues at this point [ueV]:", eigvals / E_CHARGE * 1e6)
    print("Eigenvectors (columns, in alpha1/alpha2/beta3/beta4 basis):")
    print(eigvecs)
    print()


def plot_offaxis_trajectories(params, field, v):
    a0 = np.array([1, 0, 0, 0], dtype=complex)
    radii_mm = [0.0, 1.0, 2.0, 3.0, 5.0]
    fig, axes = plt.subplots(1, len(radii_mm), figsize=(16, 3.2), sharey=True)
    for ax, r_mm in zip(axes, radii_mm):
        r = r_mm * 1e-3
        a_final, traj = propagate(v, field, params, a0, r=r, return_trajectory=True)
        for k, lab in enumerate(["a1", "a2", "b3", "b4"]):
            ax.plot(traj["z"] * 100, traj["pop"][:, k], label=lab, lw=1.3)
        ax.set_title(f"r = {r_mm:.0f} mm")
        ax.set_xlabel("z (cm)")
    axes[0].set_ylabel("population")
    axes[0].legend(fontsize=8)
    fig.suptitle("alpha1 initial state -- off-axis trajectories at 3 keV")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "offaxis_trajectories.png"), dpi=140)
    plt.close(fig)


def plot_exit_population_vs_radius(params, field, v, r_max_mm, filename, note=""):
    a0 = np.array([1, 0, 0, 0], dtype=complex)
    radii_mm = np.linspace(0, r_max_mm, 160)
    pops = np.zeros((len(radii_mm), 4))
    for i, r_mm in enumerate(radii_mm):
        a_final = propagate(v, field, params, a0, r=r_mm * 1e-3)
        pops[i] = np.abs(a_final) ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lab in enumerate(["alpha1", "alpha2", "beta3", "beta4"]):
        ax.plot(radii_mm, pops[:, k], label=lab)
    ax.set_xlabel("beam radius r (mm)" + (f"  [{note}]" if note else ""))
    ax.set_ylabel("exit population")
    ax.set_title("alpha1 initial state, exit populations vs radius (3 keV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=140)
    plt.close(fig)


def plot_field_shape(field, radii_mm=(1.0, 3.0, 5.0)):
    """Bz(z) and Br(r,z) for a few radii, over the field map's full range."""
    z_min, z_max = field.z_range()
    z = np.linspace(z_min, z_max, 600)
    Bz_G = field.Bz(z) * 1e4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    ax1.plot(z * 100, Bz_G, color="tab:blue")
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_ylabel("Bz (G)")
    ax1.set_title("Longitudinal field Bz(z)")

    for r_mm in radii_mm:
        r = r_mm * 1e-3
        Br_G = field.Br(r, z) * 1e4
        ax2.plot(z * 100, Br_G, label=f"r = {r_mm:.0f} mm")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_ylabel("Br (G)")
    ax2.set_xlabel("z (cm)")
    ax2.set_title("Radial field Br(r,z) at a few radii")
    ax2.legend(fontsize=8)

    fig.suptitle("Synthetic two-solenoid field shape")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "field_shape_Bz_Br.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    params = get_hyperfine_params(AtomState.METASTABLE_2S)
    field = CSVFieldProvider(DATA_CSV, br_col="Br", z_unit="cm", b_unit="G",
                              br_reference_radius_m=0.003)
    v = kinetic_energy_to_velocity(3000.0)

    print_matrix_snapshot(params, field)
    plot_field_shape(field)
    plot_offaxis_trajectories(params, field, v)
    plot_exit_population_vs_radius(params, field, v, r_max_mm=10,
                                    filename="exit_population_vs_radius.png",
                                    note="physical beam-radius range")
    plot_exit_population_vs_radius(params, field, v, r_max_mm=40,
                                    filename="exit_population_vs_radius_extended.png",
                                    note="unphysically large -- exploratory, shows Rabi-like return")
    print(f"Saved figures to {FIG_DIR}")