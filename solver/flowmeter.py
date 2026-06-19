"""
Vortex flow-meter -- the inverse problem: read the flow speed off the wake.

A bluff body in a flow sheds vortices at f = St(Re) * U / D. A vortex flow-meter
inserts such a body on purpose and reads U straight off the shedding frequency.
The same is done here, backwards, on the solver: known U -> solver sheds at f ->
invert -> recover U, with the uncertainty quantified by adding sensor noise
(Monte-Carlo -> 95% CI).

The Strouhal relation is calibrated from the solver's own runs (St carries the
solver's blockage offset), as a real meter is calibrated per device -- so the
round-trip is self-consistent and the offset cancels. Inverting with an
unbounded law would bias U by the blockage.

numpy only. Re-uses cylinder.strouhal for the frequency pick.
"""

import numpy as np
from cylinder import strouhal


# ---------------------------------------------------------------------------
# Calibration: St(Re) = a/Re + b + c*Re   (Williamson's functional form)
# ---------------------------------------------------------------------------
def calibrate_st(Re_arr, St_arr):
    """Least-squares fit of St = a/Re + b + c*Re to measured (Re, St) points."""
    Re = np.asarray(Re_arr, float); St = np.asarray(St_arr, float)
    A = np.column_stack([1.0 / Re, np.ones_like(Re), Re])
    coef, *_ = np.linalg.lstsq(A, St, rcond=None)
    return coef                                    # (a, b, c)


def st_of_Re(Re, coef):
    a, b, c = coef
    return a / Re + b + c * Re


# ---------------------------------------------------------------------------
# The inversion: measured shedding frequency f -> flow speed U
# ---------------------------------------------------------------------------
def recover_speed(f, D, nu, coef, U0=None):
    """Solve  f*D/U = St(Re),  Re = U*D/nu  for U, by fixed-point iteration.
    St varies only weakly with Re, so this converges in a handful of steps."""
    St_nom = coef[1] if coef[1] > 0 else 0.18
    U = U0 if U0 is not None else f * D / St_nom
    for _ in range(60):
        Re = U * D / nu
        St = st_of_Re(Re, coef)
        U_new = f * D / St
        if abs(U_new - U) < 1e-10:
            U = U_new; break
        U = 0.5 * (U + U_new)                      # damped -> robust
    return U


def freq_from_signal(t, signal, D):
    """Dominant shedding frequency f (= St*U/D) of a wake/lift signal.
    Independent of U -- that's the whole point of the inverse."""
    _, f_peak = strouhal(t, signal, U_inf=1.0, D=D)   # St return unused here
    return f_peak


# ---------------------------------------------------------------------------
# Monte-Carlo recovery under sensor noise -> point estimate + 95% CI
# ---------------------------------------------------------------------------
def recover_with_uncertainty(t, signal, D, nu, coef, noise_frac=0.05,
                             n_mc=300, seed=0):
    """Add band-limited Gaussian sensor noise to the signal, recover U each
    time, and report mean +/- 95% CI -- the UQ rhyme with Bayesian B-WIM."""
    rng = np.random.default_rng(seed)
    amp = np.std(signal)
    Us = []
    for _ in range(n_mc):
        noisy = signal + rng.normal(0.0, noise_frac * amp, size=len(signal))
        f = freq_from_signal(t, noisy, D)
        Us.append(recover_speed(f, D, nu, coef))
    Us = np.array(Us)
    return dict(U_mean=Us.mean(), U_std=Us.std(),
                ci=(np.percentile(Us, 2.5), np.percentile(Us, 97.5)),
                samples=Us)


if __name__ == "__main__":
    # self-test of the calibration+inversion math on synthetic data
    true_coef = np.array([-3.3265, 0.1816, 1.6e-4])     # Williamson 1988
    D, nu = 1.0, 0.01
    for U in (0.8, 1.0, 1.3):
        Re = U * D / nu
        f = st_of_Re(Re, true_coef) * U / D
        Uhat = recover_speed(f, D, nu, true_coef)
        print(f"  U={U:.2f} -> Re={Re:.0f} St={st_of_Re(Re,true_coef):.4f} "
              f"f={f:.4f} -> recovered U={Uhat:.4f}  err={(Uhat-U)/U*100:+.2f}%")
