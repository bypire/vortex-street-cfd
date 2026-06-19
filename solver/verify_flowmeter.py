"""
Inverse check -- vortex flow-meter round-trip (known U in -> recovered U out).

A simulate-then-recover round-trip with quantified uncertainty.

  1. forward: run the solver at several true speeds U (-> Reynolds numbers),
     read off each shedding frequency -> CALIBRATE St(Re) from OUR OWN solver.
  2. inverse: hand the (noisy) wake signal back, recover U via St(Re) -- report
     true U vs recovered U, % error, and a 95% CI from sensor-noise Monte-Carlo.

Ground truth: the known input U (round-trip error) and the Williamson 1988 law
(plotted alongside; the gap is our reported blockage, and it CANCELS in the
round-trip because we invert with our own calibration -- exactly like a real
meter calibrated per device).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from cylinder import simulate, strouhal
from flowmeter import calibrate_st, st_of_Re, recover_with_uncertainty

# physical scaling so that Re = U * D / nu = 100 * U  (clean mapping)
D_PHYS, NU_PHYS = 0.1, 1.0e-3
U_TRUE = [0.7, 1.0, 1.3, 1.6]                  # -> Re = 70, 100, 130, 160
WILLIAMSON = np.array([-3.3265, 0.1816, 1.6e-4])


def williamson_St(Re):
    return st_of_Re(Re, WILLIAMSON)


def run(noise_frac=0.05):
    runs = []
    for U in U_TRUE:
        Re = U * D_PHYS / NU_PHYS
        print(f"\n=== forward run: true U={U} m/s -> Re={Re:.0f} ===")
        sol = simulate(Re=Re, ncells_per_D=40, Lx=14, Ly=8, xc=4, yc=4,
                       t_end=78, verbose=True)
        St, _ = strouhal(sol["t"], sol["Cl"], sol["U_inf"], sol["D"])
        # physical time so the lift signal oscillates at the physical f = St*U/D
        t_phys = sol["t"] * D_PHYS / U
        runs.append(dict(U=U, Re=Re, St=St, t=t_phys, Cl=sol["Cl"]))
        print(f"  measured St(solver) = {St:.4f}   (Williamson {williamson_St(Re):.4f})")

    # --- calibrate St(Re) from our own solver -----------------------------
    Re_arr = np.array([r["Re"] for r in runs])
    St_arr = np.array([r["St"] for r in runs])
    coef = calibrate_st(Re_arr, St_arr)
    print(f"\n  calibrated St(Re) = {coef[0]:+.3f}/Re {coef[1]:+.4f} {coef[2]:+.2e}*Re")

    # --- inverse round-trip with sensor-noise UQ --------------------------
    print(f"\n  noise = {noise_frac*100:.0f}% of signal amplitude, "
          f"Monte-Carlo 95% CI")
    print("  true U   recovered U      95% CI          err")
    table = []
    for r in runs:
        res = recover_with_uncertainty(r["t"], r["Cl"], D_PHYS, NU_PHYS, coef,
                                       noise_frac=noise_frac, n_mc=300)
        Uhat = res["U_mean"]; lo, hi = res["ci"]
        err = (Uhat - r["U"]) / r["U"] * 100
        print(f"  {r['U']:.2f}     {Uhat:.3f}        "
              f"[{lo:.3f}, {hi:.3f}]     {err:+.1f}%")
        table.append(dict(U=r["U"], Re=float(r["Re"]), St=float(r["St"]),
                          Uhat=float(Uhat), lo=float(lo), hi=float(hi),
                          err=float(err)))

    _plot(runs, coef, table)
    _export(coef, table, noise_frac)
    return runs, coef, table


def _plot(runs, coef, table):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # St(Re): our solver, our calibration, Williamson
    Re_arr = np.array([r["Re"] for r in runs])
    St_arr = np.array([r["St"] for r in runs])
    Re_grid = np.linspace(60, 175, 100)
    ax[0].plot(Re_arr, St_arr, "o", ms=9, color="navy", label="our solver")
    ax[0].plot(Re_grid, st_of_Re(Re_grid, coef), "-", color="navy",
               label="our calibration")
    ax[0].plot(Re_grid, williamson_St(Re_grid), "--", color="0.5",
               label="Williamson 1988 (unbounded)")
    ax[0].set_xlabel("Reynolds number"); ax[0].set_ylabel("Strouhal St")
    ax[0].set_title("calibration: St(Re)\n(gap = reported blockage)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # round-trip: true vs recovered, with CI
    U = np.array([t["U"] for t in table])
    Uhat = np.array([t["Uhat"] for t in table])
    lo = np.array([t["lo"] for t in table]); hi = np.array([t["hi"] for t in table])
    ax[1].plot([0.6, 1.7], [0.6, 1.7], "-", color="0.7", label="perfect (y=x)")
    ax[1].errorbar(U, Uhat, yerr=[Uhat - lo, hi - Uhat], fmt="o", ms=8,
                   color="crimson", capsize=4, label="recovered U ± 95% CI")
    ax[1].set_xlabel("true flow speed U  [m/s]")
    ax[1].set_ylabel("recovered flow speed  [m/s]")
    ax[1].set_title("round-trip: known U in -> recovered U out")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_aspect("equal")

    fig.suptitle("Vortex flow-meter -- the inverse problem "
                 "(read the flow speed off the wake)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/flowmeter_roundtrip.png", dpi=130)
    print("\n  saved output/flowmeter_roundtrip.png")


def _export(coef, table, noise_frac):
    data = dict(coef=list(map(float, coef)), noise_frac=noise_frac,
                D=D_PHYS, nu=NU_PHYS, table=table)
    with open("output/flowmeter_data.js", "w") as f:
        f.write("const FLOWMETER_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("  saved output/flowmeter_data.js")


if __name__ == "__main__":
    run()
