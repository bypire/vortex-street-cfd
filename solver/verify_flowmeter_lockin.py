"""
The vortex flow-meter mis-reads when the shedder body vibrates (lock-in).

A vortex flow-meter reads the flow speed from the shedding frequency, assuming
f = St0 * U / D with St0 ~ const. That assumption holds only for a RIGID shedder.
If the body can vibrate and the flow puts the shedding near the body's natural
frequency, the wake LOCKS IN: the shedding frequency is captured by the
structure (f -> f_n) and stops tracking the flow. So the meter -- looking only at
frequency -- reports a flow speed set by the STRUCTURE, not the FLOW.

Experiment (all at the SAME flow, U=U_inf=1): sweep the reduced velocity
U_r = U/(f_n D) by varying the mount stiffness, i.e. scan a structure across the
lock-in band. For each, measure the response frequency and form the meter's
"apparent U" = St_response / St0. It should read 1.0 (the true, unchanged flow);
inside the lock-in band it does NOT -- the error is the failure envelope.

The true flow never changes (U=1). Lock-in band ~ U_r in [5,8] (Khalak &
Williamson 1999), which is why real vortex meters specify a maximum vibration
and avoid resonance of the shedder bar.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cylinder import simulate, strouhal
from fsi import simulate_viv

UR_SWEEP = [3.5, 5.0, 5.5, 8.0]   # scan structures across lock-in (brackets band)
COMMON = dict(Re=100, Lx=14, Ly=8, ncells_per_D=40, m_star=10.0, zeta=0.01)


def run():
    # baseline: RIGID cylinder -> the calibration St0 the meter trusts
    print("=== rigid baseline (St0) ===")
    s0 = simulate(Re=100, ncells_per_D=40, Lx=14, Ly=8, xc=4, yc=4,
                  t_end=80, verbose=False)
    St0, _ = strouhal(s0["t"], s0["Cl"], s0["U_inf"], s0["D"])
    print(f"  St0 (rigid) = {St0:.4f}  -> a rigid meter reads U correctly")

    rows = []
    for Ur in UR_SWEEP:
        print(f"\n=== flexible shedder, U_r = {Ur} (same flow U=1) ===")
        s = simulate_viv(Ur=Ur, t_end=100, verbose=False, **COMMON)
        St_resp, _ = strouhal(s["t"], s["Cl"], s["U_inf"], s["D"])
        apparent_U = St_resp / St0            # what the frequency-only meter reports
        err = (apparent_U - 1.0) * 100.0
        rows.append(dict(Ur=Ur, A_D=s["A_D"], St_resp=St_resp,
                         apparent_U=apparent_U, err=err))
        print(f"  A/D={s['A_D']:.3f}  St_resp={St_resp:.4f}  "
              f"apparent U={apparent_U:.3f}  (true 1.000)  err={err:+.1f}%")

    print("\n  U_r    A/D    apparentU   err     (true U = 1.000)")
    for r in rows:
        print(f"  {r['Ur']:4.1f}  {r['A_D']:.3f}   {r['apparent_U']:.3f}    "
              f"{r['err']:+5.1f}%")
    worst = max(rows, key=lambda r: abs(r["err"]))
    print(f"\n  worst meter error {worst['err']:+.1f}% at U_r={worst['Ur']} "
          f"(A/D={worst['A_D']:.2f}) -- the flow never changed; the VIBRATION did.")

    _plot(rows, St0)
    return rows, St0


def _plot(rows, St0):
    Ur = np.array([r["Ur"] for r in rows])
    appU = np.array([r["apparent_U"] for r in rows])
    A = np.array([r["A_D"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax[0].axhline(1.0, color="green", lw=2, label="true flow speed (constant)")
    ax[0].axvspan(5, 8, color="#ffe2e2", label="lock-in band")
    ax[0].plot(Ur, appU, "o-", color="crimson", lw=2, ms=9,
               label="flow-meter reading")
    ax[0].set_xlabel("reduced velocity  U_r = U / (f_n D)   (stiffer  ->  softer)")
    ax[0].set_ylabel("apparent flow speed  /  true")
    ax[0].set_title("the meter reads the STRUCTURE, not the flow")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].plot(Ur, A, "s-", color="navy", lw=2, ms=8)
    ax[1].axvspan(5, 8, color="#ffe2e2")
    ax[1].set_xlabel("reduced velocity  U_r")
    ax[1].set_ylabel("vibration amplitude  A / D")
    ax[1].set_title("...and it fails where the body vibrates most")
    ax[1].grid(alpha=0.3)

    fig.suptitle("Failure envelope: a vortex flow-meter lies under "
                 "vortex-induced vibration (lock-in)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/flowmeter_lockin_failure.png", dpi=130)
    print("  saved output/flowmeter_lockin_failure.png")


if __name__ == "__main__":
    run()
