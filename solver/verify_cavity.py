"""
M1 VERIFICATION -- lid-driven cavity vs Ghia, Ghia & Shin (1982).
================================================================================
Ground truth that ships with the code. Ghia et al. tabulated the steady cavity
velocity on the two geometric centrelines for several Reynolds numbers using a
128x128 grid. We reproduce their Re=100 (and optionally Re=400/1000) values and
report the error. If our solver disagrees, the solver is wrong -- we fix the
code, never the reference.

Reference: U. Ghia, K. N. Ghia, C. T. Shin, "High-Re Solutions for Incompressible
Flow Using the Navier-Stokes Equations and a Multigrid Method", J. Comput. Phys.
48, 387-411 (1982), Tables I & II.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ns_solver import solve_cavity, centreline_u, centreline_v

# --- Ghia Table I: u on the vertical centreline x=0.5 (y, u) -----------------
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                   0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                   0.9688, 0.9766, 1.0000])
GHIA_U = {
    100:  np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                    -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                    0.68717, 0.73722, 0.78871, 0.84123, 1.00000]),
    400:  np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                    -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                    0.55892, 0.61756, 0.68439, 0.75837, 1.00000]),
    1000: np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                    -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                    0.46604, 0.51117, 0.57492, 0.65928, 1.00000]),
}

# --- Ghia Table II: v on the horizontal centreline y=0.5 (x, v) --------------
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                   0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                   0.9609, 0.9688, 1.0000])
GHIA_V = {
    100:  np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                    0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                    -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]),
    400:  np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
                    0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
                    -0.22847, -0.19254, -0.15663, -0.12146, 0.00000]),
    1000: np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                    0.33075, 0.32235, 0.02526, -0.45418, -0.52357, -0.54053,
                    -0.44307, -0.41257, -0.36169, -0.30719, 0.00000]),
}


def verify(Re=100, nx=128, ny=128):
    Re = int(Re)
    sol = solve_cavity(Re=Re, nx=nx, ny=ny, tol=1e-6, verbose=True)

    y, u = centreline_u(sol)
    x, v = centreline_v(sol)
    u_at_ghia = np.interp(GHIA_Y, y, u)
    v_at_ghia = np.interp(GHIA_X, x, v)

    eu = u_at_ghia - GHIA_U[Re]
    ev = v_at_ghia - GHIA_V[Re]
    rms_u = np.sqrt(np.mean(eu**2))
    rms_v = np.sqrt(np.mean(ev**2))
    max_u = np.abs(eu).max()
    max_v = np.abs(ev).max()

    print("\n--- u on vertical centreline x=0.5  (Ghia vs computed) ---")
    print("    y       Ghia      ours      diff")
    for k in range(len(GHIA_Y)):
        print(f"  {GHIA_Y[k]:.4f}  {GHIA_U[Re][k]:+.5f}  "
              f"{u_at_ghia[k]:+.5f}  {eu[k]:+.5f}")
    print(f"  RMS(u) = {rms_u:.4e}   max|du| = {max_u:.4e}")

    print("\n--- v on horizontal centreline y=0.5  (Ghia vs computed) ---")
    print("    x       Ghia      ours      diff")
    for k in range(len(GHIA_X)):
        print(f"  {GHIA_X[k]:.4f}  {GHIA_V[Re][k]:+.5f}  "
              f"{v_at_ghia[k]:+.5f}  {ev[k]:+.5f}")
    print(f"  RMS(v) = {rms_v:.4e}   max|dv| = {max_v:.4e}")

    # u-range amplitude as a single headline number
    print(f"\n  min u (Ghia {GHIA_U[Re].min():+.4f}) vs ours {u.min():+.4f}")
    print(f"  max v (Ghia {GHIA_V[Re].max():+.4f}) vs ours {v.max():+.4f}")

    _plot(sol, Re, y, u, x, v, u_at_ghia, v_at_ghia, rms_u, rms_v)
    return sol, rms_u, rms_v


def _plot(sol, Re, y, u, x, v, u_at_ghia, v_at_ghia, rms_u, rms_v):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(u, y, "-", lw=2, label="this solver")
    ax[0].plot(GHIA_U[Re], GHIA_Y, "o", ms=6, mfc="none", mec="crimson",
               mew=1.6, label="Ghia et al. 1982")
    ax[0].set_xlabel("u  (x-velocity)"); ax[0].set_ylabel("y")
    ax[0].set_title(f"u on vertical centreline x=0.5\nRMS err = {rms_u:.2e}")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(x, v, "-", lw=2, label="this solver")
    ax[1].plot(GHIA_X, GHIA_V[Re], "o", ms=6, mfc="none", mec="crimson",
               mew=1.6, label="Ghia et al. 1982")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("v  (y-velocity)")
    ax[1].set_title(f"v on horizontal centreline y=0.5\nRMS err = {rms_v:.2e}")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(f"Lid-driven cavity, Re={Re}, grid {sol['nx']}x{sol['ny']} "
                 f"-- verification vs Ghia 1982", fontweight="bold")
    fig.tight_layout()
    out = f"output/cavity_ghia_Re{Re}.png"
    fig.savefig(out, dpi=130)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    Re = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    verify(Re=Re, nx=nx, ny=nx)
