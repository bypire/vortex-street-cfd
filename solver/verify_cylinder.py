"""
M3 VERIFICATION -- vortex shedding past a cylinder at Re=100.
================================================================================
Two independent checks against the literature:

  (1) Strouhal number St = f*D/U of the periodic wake, vs Williamson's
      experimental law  St = 0.198 (1 - 19.7/Re)   [C. Williamson, ARFM 1996].
  (2) Drag coefficient Cd (mean) and lift amplitude Cl, vs the accepted
      laminar-shedding range Cd ~ 1.33, Cl_amp ~ 0.33 at Re=100.

Note: a finite-width domain confines the flow (blockage), which raises St, Cd
and Cl above the unbounded reference. A wide domain (blockage ~8%) is used and
the residual offset quantified; the blockage sweep shows St -> the unbounded
value as the walls recede.

Reference plot: output/cylinder_Re100.png (vorticity snapshot + force history).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cylinder import simulate, vorticity, strouhal


def williamson_St(Re):
    return 0.198 * (1.0 - 19.7 / Re)


def run(Re=100, ncells_per_D=40, Lx=16, Ly=12, t_end=110):
    xc, yc = 4.0, Ly / 2
    sol = simulate(Re=Re, ncells_per_D=ncells_per_D, Lx=Lx, Ly=Ly,
                   xc=xc, yc=yc, t_end=t_end, verbose=True)

    t, Cd, Cl = sol["t"], sol["Cd"], sol["Cl"]
    St, fpk = strouhal(t, Cl, sol["U_inf"], sol["D"])
    half = slice(len(t) // 2, None)
    Cd_mean = Cd[half].mean()
    Cl_amp = (Cl[half].max() - Cl[half].min()) / 2.0
    blockage = sol["D"] / Ly

    print("\n================  Re=%d  ================" % Re)
    print(f"  blockage D/Ly        = {blockage*100:.1f} %")
    print(f"  Strouhal  St (ours)  = {St:.4f}")
    print(f"  Strouhal  St (Will.) = {williamson_St(Re):.4f}  (unbounded)")
    print(f"  mean Cd   (ours)     = {Cd_mean:.3f}   (lit ~1.33, unbounded)")
    print(f"  Cl amplitude (ours)  = {Cl_amp:.3f}   (lit ~0.33, unbounded)")

    _plot(sol, St, Cd_mean, Cl_amp, blockage)
    return sol, St, Cd_mean, Cl_amp


def _plot(sol, St, Cd_mean, Cl_amp, blockage):
    grid = sol["grid"]
    w = vorticity(sol["u"], sol["v"], grid)
    # mask the cylinder for a clean image
    Xp, Yp = np.meshgrid(grid.xp, grid.yp, indexing="ij")
    inside = (Xp - sol["xc"])**2 + (Yp - sol["yc"])**2 < (0.5 * sol["D"])**2
    wm = np.ma.array(w, mask=inside)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.22)

    # --- vorticity snapshot (the von Karman street) ---
    ax0 = fig.add_subplot(gs[0, :])
    lim = 3.0
    pc = ax0.pcolormesh(grid.xp, grid.yp, wm.T, cmap="RdBu_r",
                        vmin=-lim, vmax=lim, shading="auto")
    th = np.linspace(0, 2*np.pi, 60)
    ax0.fill(sol["xc"] + 0.5*sol["D"]*np.cos(th),
             sol["yc"] + 0.5*sol["D"]*np.sin(th), color="0.2", zorder=5)
    ax0.set_aspect("equal"); ax0.set_xlim(grid.xp[0], grid.xp[-1])
    ax0.set_title(f"von Karman vortex street -- vorticity, Re={int(sol['Re'])}",
                  fontweight="bold")
    ax0.set_xlabel("x / D"); ax0.set_ylabel("y / D")
    fig.colorbar(pc, ax=ax0, label="vorticity", shrink=0.85, pad=0.01)

    # --- lift / drag history ---
    t = sol["t"]; i0 = len(t) // 3
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t[i0:], sol["Cl"][i0:], color="crimson", lw=1.2, label="C_L (lift)")
    ax1.plot(t[i0:], sol["Cd"][i0:], color="navy", lw=1.2, label="C_D (drag)")
    ax1.set_xlabel("time  t U/D"); ax1.set_ylabel("force coefficient")
    ax1.set_title("periodic lift & drag"); ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # --- spectrum of lift ---
    ax2 = fig.add_subplot(gs[1, 1])
    s = sol["Cl"][len(t)//2:] - sol["Cl"][len(t)//2:].mean()
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(len(s), dt) * sol["D"] / sol["U_inf"]
    amp = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    ax2.plot(freqs, amp, color="darkgreen")
    ax2.axvline(St, color="crimson", ls="--",
                label=f"St = {St:.3f}")
    ax2.axvline(williamson_St(sol["Re"]), color="0.4", ls=":",
                label=f"Williamson {williamson_St(sol['Re']):.3f}")
    ax2.set_xlim(0, 0.6); ax2.set_xlabel("St = f D / U")
    ax2.set_ylabel("lift spectrum"); ax2.set_title("shedding frequency")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f"Flow past a cylinder, Re={int(sol['Re'])}, "
                 f"blockage {blockage*100:.0f}%  --  "
                 f"St={St:.3f}, Cd={Cd_mean:.2f}, Cl_amp={Cl_amp:.2f}",
                 fontweight="bold", y=0.98)
    out = f"output/cylinder_Re{int(sol['Re'])}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    Re = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(Re=Re)
