"""
Vortex-induced vibration: the lock-in resonance.

Demonstrates and checks the synchronisation / lock-in: when the shedding
frequency approaches the structure's natural frequency, the spring-mounted
cylinder's response amplitude peaks sharply.

We sweep the reduced velocity U_r = U/(f_n D): far from resonance the amplitude
stays small; near U_r ~ 5-6 (f_n ~ shedding St) it locks in and the amplitude
jumps to O(0.5 D), the accepted VIV peak for low mass-damping
(cf. Khalak & Williamson 1999). Ground truth: the response must peak at lock-in
and the oscillation frequency must capture f_n inside the band.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fsi import simulate_viv

# domain shared by all runs (kept modest so the sweep is affordable)
COMMON = dict(Re=100, Lx=14, Ly=8, ncells_per_D=40, m_star=10.0, zeta=0.01)


def run(Ur_list=(4.0, 5.5, 9.0), t_end=140):
    results = []
    resonant = None
    for Ur in Ur_list:
        rec = 30 if abs(Ur - 5.5) < 1e-6 else 0
        s = simulate_viv(Ur=Ur, t_end=t_end, record_every=rec,
                         verbose=True, **COMMON)
        results.append((Ur, s["A_D"], s["f_n"], s["St"]))
        print(f"  -> Ur={Ur:4.1f}  A/D={s['A_D']:.3f}  "
              f"f_n={s['f_n']:.3f}  response St={s['St']:.3f}")
        if rec:
            resonant = s

    print("\n  Ur     A/D     f_n     responseSt")
    for Ur, A, fn, St in results:
        print(f"  {Ur:4.1f}  {A:.3f}  {fn:.3f}  {St:.3f}")

    _plot(results, resonant)
    return results, resonant


def _plot(results, resonant):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # response amplitude vs reduced velocity (lock-in curve)
    Ur = [r[0] for r in results]; A = [r[1] for r in results]
    ax[0].plot(Ur, A, "o-", lw=2, ms=9, color="crimson")
    ax[0].axhspan(0.5, 0.6, color="0.8", alpha=0.5,
                  label="VIV peak band (lit ~0.5-0.6 D)")
    ax[0].set_xlabel("reduced velocity  U_r = U / (f_n D)")
    ax[0].set_ylabel("response amplitude  A / D")
    ax[0].set_title("lock-in: amplitude peaks at resonance")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # resonant time history (build-up to limit cycle)
    if resonant is not None:
        t, y, D = resonant["t"], resonant["y"], resonant["D"]
        ax[1].plot(t, y / D, color="navy", lw=1)
        ax[1].set_xlabel("time  t U / D"); ax[1].set_ylabel("y / D")
        ax[1].set_title(f"resonant build-up  (U_r={resonant['Ur']}, "
                        f"f_n={resonant['f_n']:.3f})")
        ax[1].grid(alpha=0.3)

    fig.suptitle("Vortex-induced vibration of a spring-mounted cylinder "
                 "(Re=100, m*=10, zeta=1%)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/viv_lockin.png", dpi=130)
    print("  saved output/viv_lockin.png")


if __name__ == "__main__":
    run()
