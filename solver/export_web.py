"""
Export baked vortex-street frames for the web demo.
================================================================================
Runs the cylinder simulation at one or more Reynolds numbers, records the
vorticity field over ~2 shedding periods in the wake window, quantises each
frame to int8, and writes output/cylinder_data.js as

    const CYLINDER_DATA = { cases: { "100": {...}, ... }, ... }

The web page decodes the base64 Int8Array and paints it to a <canvas> with a
blue-white-red colormap, looping -- a live von Karman street, no server, opens
by double-click (same injected-data pattern as the VBI project).
"""

import base64
import json
import sys
import numpy as np

from cylinder import simulate, vorticity, strouhal


# wake window we actually show (crop tightly around the street)
WIN_X = (2.0, 15.0)
WIN_Y0_FRAC = 0.12        # crop a margin off top/bottom of the (wide) domain
NX_OUT, NY_OUT = 220, 90  # output raster
VORT_CLIP = 3.0           # +/- clip for the colormap / quantisation


def bake_case(Re, ncells_per_D=40, Lx=16, Ly=10, t_end=72,
              n_frames=56, periods=2.2):
    xc, yc = 4.0, Ly / 2
    St_guess = max(0.16, 0.198 * (1 - 19.7 / Re)) if Re > 47 else 0.16
    T_shed = 1.0 / St_guess
    rec_window = periods * T_shed

    sol = simulate(Re=Re, ncells_per_D=ncells_per_D, Lx=Lx, Ly=Ly,
                   xc=xc, yc=yc, t_end=t_end, verbose=True)
    grid = sol["grid"]
    dt = sol["dt"]

    # re-march the last rec_window saving frames (cheap: reuse final state)
    # -- simpler: re-run and record_every; but to avoid a second sim we instead
    #    recompute vorticity stroboscopically from a short continuation.
    from cylinder import make_poisson, cylinder_masks, step
    poisson = make_poisson(grid)
    chi_u, chi_v = cylinder_masks(grid, xc, yc, sol["D"])
    u, v = sol["u"].copy(), sol["v"].copy()
    nu = sol["nu"]; U = sol["U_inf"]

    n_total = int(rec_window / dt)
    stride = max(1, n_total // n_frames)
    frames = []
    # crop indices
    ix = np.where((grid.xp >= WIN_X[0]) & (grid.xp <= WIN_X[1]))[0]
    jy = np.where((grid.yp >= Ly * WIN_Y0_FRAC) &
                  (grid.yp <= Ly * (1 - WIN_Y0_FRAC)))[0]
    x0, x1, y0, y1 = ix[0], ix[-1] + 1, jy[0], jy[-1] + 1

    for n in range(n_total):
        u, v, p, Fx, Fy = step(u, v, poisson, grid, dt, nu,
                               chi_u, chi_v, 1e-4, U, kick=None)
        if n % stride == 0 and len(frames) < n_frames:
            w = vorticity(u, v, grid)[x0:x1, y0:y1]
            # resample to NX_OUT x NY_OUT
            wq = _resample(w, NX_OUT, NY_OUT)
            q = np.clip(wq / VORT_CLIP, -1, 1)
            frames.append(np.round(q * 127).astype(np.int8))
    St, _ = strouhal(sol["t"], sol["Cl"], U, sol["D"])

    arr = np.stack(frames, 0)                       # (n_frames, NX_OUT, NY_OUT)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")

    # cylinder position inside the cropped/resampled window, in raster coords
    cx = (xc - WIN_X[0]) / (WIN_X[1] - WIN_X[0]) * NX_OUT
    cyc = (yc - Ly * WIN_Y0_FRAC) / (Ly * (1 - 2 * WIN_Y0_FRAC)) * NY_OUT
    crad = (0.5 * sol["D"]) / (WIN_X[1] - WIN_X[0]) * NX_OUT

    half = slice(len(sol["t"]) // 2, None)
    return {
        "Re": int(Re),
        "St": round(float(St), 4),
        "Cd": round(float(sol["Cd"][half].mean()), 3),
        "Cl_amp": round(float((sol["Cl"][half].max() -
                               sol["Cl"][half].min()) / 2), 3),
        "blockage": round(float(sol["D"] / Ly), 3),
        "nframes": len(frames), "nx": NX_OUT, "ny": NY_OUT,
        "clip": VORT_CLIP,
        "cyl": {"x": round(cx, 1), "y": round(cyc, 1), "r": round(crad, 1)},
        "frames_b64": b64,
    }


def _resample(field, nx_out, ny_out):
    """Nearest-neighbour resample of a 2D array to (nx_out, ny_out)."""
    nx, ny = field.shape
    xi = (np.linspace(0, nx - 1, nx_out)).astype(int)
    yi = (np.linspace(0, ny - 1, ny_out)).astype(int)
    return field[np.ix_(xi, yi)]


if __name__ == "__main__":
    Res = [int(a) for a in sys.argv[1:]] or [100]
    cases = {}
    for Re in Res:
        print(f"\n=== baking Re={Re} ===")
        cases[str(Re)] = bake_case(Re)
    data = {"cases": cases, "win_x": WIN_X, "note":
            "vorticity quantised to int8 in [-clip,clip]; "
            "von Karman street baked from the MAC/Chorin NS solver"}
    out = "output/cylinder_data.js"
    with open(out, "w") as f:
        f.write("const CYLINDER_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    import os
    print(f"\nwrote {out}  ({os.path.getsize(out)/1024:.0f} kB)  "
          f"cases={list(cases)}")
