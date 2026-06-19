"""
M2 VERIFICATION -- steady wake behind a cylinder (Re=20, 40).
================================================================================
Below Re~47 the wake is steady: two symmetric standing vortices form behind the
cylinder. The length of that recirculation bubble, L_w/D (rear of cylinder to
where the centreline flow turns forward again), is a classic benchmark:

    Re=20  ->  L_w/D ~ 0.93
    Re=40  ->  L_w/D ~ 2.24
    (Coutanceau & Bouard 1977; Tritton 1959 -- widely reproduced.)

This is a clean, steady, grid-converged check that complements the unsteady
Strouhal verification (verify_cylinder.py) -- a different physical regime.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cylinder import simulate

LIT = {20: 0.93, 40: 2.24}


def bubble_length(sol):
    """Recirculation length L_w/D measured on the wake centreline."""
    grid = sol["grid"]
    nx, ny = grid.nx, grid.ny
    uc = 0.5 * (sol["u"][:-1, :] + sol["u"][1:, :])     # u at cell centres
    # centreline rows straddling yc
    j = np.argmin(np.abs(grid.yp - sol["yc"]))
    uline = 0.5 * (uc[:, j] + uc[:, min(j + 1, ny - 1)])
    x = grid.xp
    rear = sol["xc"] + 0.5 * sol["D"]
    # search downstream of the cylinder for the first u>0 crossing
    mask = x > rear + 0.02
    xs, us = x[mask], uline[mask]
    # find first index where u goes from <=0 to >0
    Lw = 0.0
    neg_seen = us[0] < 0
    for k in range(1, len(us)):
        if us[k - 1] < 0 <= us[k]:
            # linear interp of zero crossing
            xz = xs[k - 1] + (xs[k] - xs[k - 1]) * (-us[k - 1]) / (us[k] - us[k - 1])
            Lw = xz - rear
            break
    return Lw / sol["D"], (x, uline)


def run():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    print("  Re   L_w/D (ours)   L_w/D (lit)   rel.err")
    for ax, Re in zip(axes, (20, 40)):
        # low Re => low cell-Peclet => a coarse grid is plenty (and fast)
        nc = 24 if Re == 20 else 30
        sol = simulate(Re=Re, ncells_per_D=nc, Lx=14, Ly=10, xc=4, yc=5,
                       t_end=60, verbose=False)
        LwD, (x, uline) = bubble_length(sol)
        err = (LwD - LIT[Re]) / LIT[Re] * 100
        print(f"  {Re:3d}   {LwD:8.3f}      {LIT[Re]:6.2f}      {err:+5.1f}%")

        rear = sol["xc"] + 0.5 * sol["D"]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot((x - rear) / sol["D"], uline, lw=2)
        ax.axvline(LwD, color="crimson", ls="--",
                   label=f"L_w/D = {LwD:.2f}\n(lit {LIT[Re]:.2f})")
        ax.axvline(LIT[Re], color="0.4", ls=":")
        ax.set_xlim(0, 4); ax.set_xlabel("distance behind cylinder  (x−rear)/D")
        ax.set_ylabel("centreline u / U")
        ax.set_title(f"Re={Re} steady wake"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Steady recirculation bubble vs literature "
                 "(Coutanceau & Bouard 1977)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/cylinder_steady.png", dpi=130)
    print("  saved output/cylinder_steady.png")


if __name__ == "__main__":
    run()
