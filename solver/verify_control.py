"""
M6 -- ACTIVE FLOW CONTROL: switch the vortex street OFF by spinning the cylinder.
================================================================================
The "we don't just predict the problem, we solve it" piece -- and the tie to
LBB's Structural Control & Health Monitoring. A steadily ROTATING cylinder
reorganises its wake; above a critical rotation rate alpha = omega*R/U the
von Karman shedding is SUPPRESSED -- the wake goes steady, the oscillating lift
(and any vortex-induced vibration it would drive) collapses to ~0.

Ground truth: Mittal & Kumar (J. Fluid Mech. 476, 2003, Re=200) find the wake
steady (no shedding) for 1.91 < alpha < 4.34; Tokumaru & Dimotakis (1991) show
rotary control reorganises/suppresses the street. We reproduce the suppression
and locate the threshold by the collapse of the lift-oscillation amplitude.

Engineering point: this is WHY chimneys wear helical strakes and risers get
fairings -- break the coherent shedding, kill the vibration. Here we do it
actively, by rotation, and verify the threshold.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cylinder import simulate, vorticity, strouhal

ALPHAS = [0.0, 1.0, 1.8, 2.6]      # tip-speed ratios to sweep


def run(alphas=ALPHAS, Re=100, ncells_per_D=48, Lx=15, Ly=8, t_end=80):
    sols, stats = [], []
    for a in alphas:
        print(f"\n=== alpha = {a} ===")
        s = simulate(Re=Re, alpha=a, ncells_per_D=ncells_per_D,
                     Lx=Lx, Ly=Ly, xc=4, yc=Ly / 2, t_end=t_end, verbose=True)
        half = slice(len(s["t"]) // 2, None)
        Cl_amp = (s["Cl"][half].max() - s["Cl"][half].min()) / 2.0
        Cl_mean = s["Cl"][half].mean()                 # Magnus lift
        Cd_mean = s["Cd"][half].mean()
        St, _ = strouhal(s["t"], s["Cl"], s["U_inf"], s["D"]) if Cl_amp > 0.02 else (0.0, 0.0)
        shedding = Cl_amp > 0.05
        print(f"  Cl_amp={Cl_amp:.3f}  Cl_mean(Magnus)={Cl_mean:+.2f}  "
              f"Cd={Cd_mean:.2f}  St={St:.3f}  shedding={'YES' if shedding else 'NO'}")
        sols.append(s); stats.append((a, Cl_amp, Cl_mean, Cd_mean, St, shedding))

    print("\n  alpha   Cl_amp   shedding")
    for a, ca, cm, cd, st, sh in stats:
        print(f"  {a:4.1f}   {ca:.3f}    {'YES' if sh else 'NO  (suppressed)'}")
    crit = next((a for a, ca, *_ in stats if ca < 0.05), None)
    print(f"\n  suppression sets in by alpha ~ {crit}  "
          f"(lit: steady wake for 1.91<alpha<4.34, Mittal & Kumar 2003)")

    _plot(sols, stats)
    return sols, stats


def _plot(sols, stats):
    n = len(sols)
    fig = plt.figure(figsize=(12, 2.2 * n + 1.4))
    gs = fig.add_gridspec(n, 2, width_ratios=[3, 1.25], hspace=0.35, wspace=0.18)
    lim = 3.0
    for i, (s, st) in enumerate(zip(sols, stats)):
        g = s["grid"]; w = vorticity(s["u"], s["v"], g)
        Xp, Yp = np.meshgrid(g.xp, g.yp, indexing="ij")
        inside = (Xp - s["xc"])**2 + (Yp - s["yc"])**2 < (0.5 * s["D"])**2
        wm = np.ma.array(w, mask=inside)
        ax = fig.add_subplot(gs[i, 0])
        ax.pcolormesh(g.xp, g.yp, wm.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                      shading="auto")
        th = np.linspace(0, 2 * np.pi, 50)
        ax.fill(s["xc"] + 0.5 * s["D"] * np.cos(th),
                s["yc"] + 0.5 * s["D"] * np.sin(th), color="0.2", zorder=5)
        if st[0] > 0:   # arrow hint of spin
            ax.annotate("", xy=(s["xc"] + 0.6, s["yc"] + 0.6),
                        xytext=(s["xc"] - 0.6, s["yc"] + 0.6),
                        arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
        ax.set_aspect("equal"); ax.set_xlim(g.xp[0], g.xp[-1])
        verdict = "SHEDDING" if st[5] else "SUPPRESSED"
        ax.set_title(f"alpha = {st[0]}   ->   {verdict}   "
                     f"(Cl_amp = {st[1]:.3f})", fontsize=10,
                     color="crimson" if st[5] else "green", loc="left")
        ax.set_ylabel("y/D")
        if i == n - 1:
            ax.set_xlabel("x/D")

    # Cl_amp vs alpha collapse
    ax2 = fig.add_subplot(gs[:, 1])
    A = [s[0] for s in stats]; CA = [s[1] for s in stats]
    ax2.plot(A, CA, "o-", lw=2, ms=9, color="navy")
    ax2.axvspan(1.91, 4.34, color="0.85", label="steady wake\n(Mittal & Kumar)")
    ax2.axhline(0.05, color="0.6", ls=":", lw=1)
    ax2.set_xlabel("rotation rate  alpha = omega R / U")
    ax2.set_ylabel("lift oscillation amplitude  Cl_amp")
    ax2.set_title("shedding collapses\nwhen you spin it")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle("Active flow control: spin the cylinder, switch the vortex "
                 "street OFF  (Re=100)", fontweight="bold", y=0.995)
    fig.savefig("output/control_suppression.png", dpi=125, bbox_inches="tight")
    print("  saved output/control_suppression.png")


if __name__ == "__main__":
    run()
