"""
Flow past a circular cylinder -- external flow, immersed boundary, MAC/Chorin.
================================================================================
Same incompressible Navier-Stokes core as the cavity (staggered MAC grid, Chorin
projection), but now an *external* flow: uniform stream comes in from the left,
leaves on the right, and a solid cylinder sits in the way. Above a critical
Reynolds number the wake goes unsteady and sheds the famous von Karman vortex
street -- alternating vortices that would shake a real structure.

Immersed boundary  = volume penalization (Angot/Brinkman). The solid is a region
where we drive the velocity to zero through a penalty term -(chi/eta)*u added to
the momentum equation, treated IMPLICITLY (pointwise) so it costs no extra
stability. A bonus: the penalty reaction integrated over the solid IS the
hydrodynamic force on the cylinder -> drag and lift fall straight out.

Boundary conditions:
  left   : inflow, u = U_inf, v = 0
  right  : outflow, zero-gradient (+ global mass-flux correction)
  top/bot: free-slip (symmetry) -- mimics an unbounded stream, no wall layers
  cylinder: no-slip, enforced by penalization

Pure numpy for the physics; the constant pressure Laplacian is assembled once
(sparse) and LU-factored, exactly as in ns_solver.py.
"""

import numpy as np
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------------------
# Grid + masks
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        self.dx, self.dy = Lx / nx, Ly / ny
        # face / centre coordinates
        self.xu = np.arange(nx + 1) * self.dx                 # u x-faces
        self.yu = (np.arange(ny) + 0.5) * self.dy             # u y-centres
        self.xv = (np.arange(nx) + 0.5) * self.dx             # v x-centres
        self.yv = np.arange(ny + 1) * self.dy                 # v y-faces
        self.xp = (np.arange(nx) + 0.5) * self.dx             # p centres
        self.yp = (np.arange(ny) + 0.5) * self.dy


def cylinder_masks(grid, xc, yc, D):
    """Boolean masks (1 inside solid) at u-points and v-points."""
    R = 0.5 * D
    Xu, Yu = np.meshgrid(grid.xu, grid.yu, indexing="ij")
    Xv, Yv = np.meshgrid(grid.xv, grid.yv, indexing="ij")
    chi_u = ((Xu - xc) ** 2 + (Yu - yc) ** 2 < R ** 2).astype(float)
    chi_v = ((Xv - xc) ** 2 + (Yv - yc) ** 2 < R ** 2).astype(float)
    return chi_u, chi_v


def make_poisson(grid):
    """Fast pressure-Poisson solver lap(p)=rhs with homogeneous Neumann BC on a
    uniform cell-centred grid, via the discrete cosine transform (DCT-II). The
    5-point Neumann Laplacian is diagonalised exactly by the DCT, so each solve
    is two FFTs -- O(N log N), no factorisation, and trivially handles the
    singular (pure-Neumann) null space by zeroing the constant mode. Returns a
    callable poisson(rhs2d) -> p2d with mean(p)=0."""
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
    i = np.arange(nx); j = np.arange(ny)
    lam_x = 2.0 * (np.cos(np.pi * i / nx) - 1.0) / dx**2
    lam_y = 2.0 * (np.cos(np.pi * j / ny) - 1.0) / dy**2
    denom = lam_x[:, None] + lam_y[None, :]
    denom[0, 0] = 1.0                          # avoid /0 at the constant mode

    def poisson(rhs):
        rhat = dctn(rhs, type=2, norm="ortho")
        phat = rhat / denom
        phat[0, 0] = 0.0                       # fix the additive constant
        return idctn(phat, type=2, norm="ortho")

    return poisson


# ---------------------------------------------------------------------------
# One projection step (external flow + penalized cylinder)
# ---------------------------------------------------------------------------
def step(u, v, poisson, grid, dt, nu, chi_u, chi_v, eta, U_inf, kick=None,
         u_rot=None, v_rot=None):
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy

    # ---- u predictor (interior x-faces 1..nx-1); free-slip top/bottom ------
    up = np.empty((nx + 1, ny + 2))
    up[:, 1:-1] = u
    up[:, 0] = u[:, 0]            # free-slip: du/dy = 0 at bottom
    up[:, -1] = u[:, -1]         # free-slip at top
    u_P = u[1:nx, :]
    u_E = u[2:nx + 1, :]; u_W = u[0:nx - 1, :]
    u_N = up[1:nx, 2:ny + 2]; u_S = up[1:nx, 0:ny]
    v_at_u = 0.25 * (v[0:nx - 1, 0:ny] + v[1:nx, 0:ny]
                     + v[0:nx - 1, 1:ny + 1] + v[1:nx, 1:ny + 1])
    conv_u = u_P * (u_E - u_W) / (2 * dx) + v_at_u * (u_N - u_S) / (2 * dy)
    diff_u = nu * ((u_E - 2 * u_P + u_W) / dx**2 + (u_N - 2 * u_P + u_S) / dy**2)
    u_star = u.copy()
    u_star[1:nx, :] = u_P + dt * (-conv_u + diff_u)
    u_star[0, :] = U_inf                       # inflow
    u_star[nx, :] = u_star[nx - 1, :]          # outflow: zero-gradient
    # global mass-flux correction so inflow == outflow (Neumann compatibility)
    Qin = U_inf * grid.Ly
    s = u_star[nx, :].sum()
    u_star[nx, :] += (Qin / dy - s) / ny

    # ---- v predictor (interior y-faces 1..ny-1); inflow/outflow on x -------
    vp = np.empty((nx + 2, ny + 1))
    vp[1:-1, :] = v
    vp[0, :] = -v[0, :]          # inflow: v = 0  (reflect)
    vp[-1, :] = v[-1, :]        # outflow: dv/dx = 0
    v_P = v[:, 1:ny]
    v_N = v[:, 2:ny + 1]; v_S = v[:, 0:ny - 1]
    v_E = vp[2:nx + 2, 1:ny]; v_W = vp[0:nx, 1:ny]
    u_at_v = 0.25 * (u[0:nx, 0:ny - 1] + u[1:nx + 1, 0:ny - 1]
                     + u[0:nx, 1:ny] + u[1:nx + 1, 1:ny])
    conv_v = u_at_v * (v_E - v_W) / (2 * dx) + v_P * (v_N - v_S) / (2 * dy)
    diff_v = nu * ((v_E - 2 * v_P + v_W) / dx**2 + (v_N - 2 * v_P + v_S) / dy**2)
    v_star = v.copy()
    v_star[:, 1:ny] = v_P + dt * (-conv_v + diff_v)
    v_star[:, 0] = 0.0; v_star[:, ny] = 0.0    # no penetration top/bottom

    # ---- symmetry-breaking kick (brief, LOCAL: a small transverse jet just
    #      behind the cylinder to trigger shedding; removed after startup) -----
    if kick is not None:
        v_star[kick] += 0.1 * U_inf

    # ---- penalize the solid (implicit, pointwise) -------------------------
    # drive the fluid to the solid's surface velocity: 0 if stationary, or the
    # solid-body rotation field (u_rot,v_rot) for a spinning cylinder (control).
    beta = dt / eta
    if u_rot is None:
        u_pen = u_star / (1.0 + beta * chi_u)
        v_pen = v_star / (1.0 + beta * chi_v)
        Fx = (chi_u * u_pen / eta).sum() * dx * dy
        Fy = (chi_v * v_pen / eta).sum() * dx * dy
    else:
        u_pen = (u_star + beta * chi_u * u_rot) / (1.0 + beta * chi_u)
        v_pen = (v_star + beta * chi_v * v_rot) / (1.0 + beta * chi_v)
        Fx = (chi_u * (u_pen - u_rot) / eta).sum() * dx * dy
        Fy = (chi_v * (v_pen - v_rot) / eta).sum() * dx * dy

    # ---- pressure projection ---------------------------------------------
    div = ((u_pen[1:nx + 1, :] - u_pen[0:nx, :]) / dx
           + (v_pen[:, 1:ny + 1] - v_pen[:, 0:ny]) / dy)
    p = poisson(div / dt)

    u_new = u_pen.copy(); v_new = v_pen.copy()
    u_new[1:nx, :] -= dt * (p[1:nx, :] - p[0:nx - 1, :]) / dx
    v_new[:, 1:ny] -= dt * (p[:, 1:ny] - p[:, 0:ny - 1]) / dy
    # re-apply hard BCs after projection
    u_new[0, :] = U_inf
    u_new[nx, :] = u_new[nx - 1, :]
    v_new[:, 0] = 0.0; v_new[:, ny] = 0.0
    return u_new, v_new, p, Fx, Fy


# ---------------------------------------------------------------------------
# Vorticity (for visualisation): omega = dv/dx - du/dy at cell corners->centres
# ---------------------------------------------------------------------------
def vorticity(u, v, grid):
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
    # du/dy at cell centres
    uc = 0.5 * (u[0:nx, :] + u[1:nx + 1, :])        # u at p-centres (nx,ny)
    vc = 0.5 * (v[:, 0:ny] + v[:, 1:ny + 1])        # v at p-centres
    dudy = np.zeros((nx, ny)); dvdx = np.zeros((nx, ny))
    dudy[:, 1:-1] = (uc[:, 2:] - uc[:, :-2]) / (2 * dy)
    dvdx[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2 * dx)
    return dvdx - dudy


# ---------------------------------------------------------------------------
# Time-marching driver
# ---------------------------------------------------------------------------
def simulate(Re=100.0, D=1.0, U_inf=1.0,
             Lx=16.0, Ly=8.0, xc=4.0, yc=4.0, ncells_per_D=48,
             eta=1e-4, cfl=0.4, t_end=120.0, alpha=0.0,
             probe=None, record_every=0, verbose=True):
    """March flow past a cylinder. Returns dict with time series of Fx/Fy (->
    Cd/Cl), an optional v-probe signal (for Strouhal), and optional saved frames.

    alpha = rotation rate (tip-speed ratio) omega*R/U_inf. alpha=0 is the plain
    stationary cylinder; alpha>~2 is the rotary-control case that SUPPRESSES the
    von Karman street (Tokumaru & Dimotakis 1991; Mittal & Kumar 2003)."""
    nx = int(round(Lx / D * ncells_per_D))
    ny = int(round(Ly / D * ncells_per_D))
    grid = Grid(Lx, Ly, nx, ny)
    nu = U_inf * D / Re

    dx = grid.dx
    dt = min(cfl * dx / U_inf, cfl * 0.25 * dx**2 / nu)
    Pe = U_inf * dx / nu                   # cell Peclet -- central diff wants < 2

    chi_u, chi_v = cylinder_masks(grid, xc, yc, D)
    poisson = make_poisson(grid)

    # rotary control: solid-body rotation field omega x r inside the cylinder
    omega = alpha * U_inf / (0.5 * D)
    if alpha != 0.0:
        u_rot = -omega * (grid.yu - yc)[None, :] * np.ones((nx + 1, 1))
        v_rot = omega * (grid.xv - xc)[:, None] * np.ones((1, ny + 1))
    else:
        u_rot = v_rot = None

    # one-sided transverse jet just behind the cylinder -> triggers shedding
    Xv, Yv = np.meshgrid(grid.xv, grid.yv, indexing="ij")
    kick_mask = ((Xv > xc + 0.3 * D) & (Xv < xc + 1.0 * D)
                 & (Yv > yc) & (Yv < yc + 0.4 * D))

    u = np.full((nx + 1, ny), U_inf)
    v = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))

    if probe is None:
        probe = (xc + 3 * D, yc)          # 3 diameters behind the cylinder
    ip = int(np.argmin(np.abs(grid.xv - probe[0])))
    jp = int(np.argmin(np.abs(grid.yv - probe[1])))

    nsteps = int(t_end / dt)
    q = 0.5 * U_inf**2 * D                 # dynamic pressure * D (rho=1), for Cd/Cl

    if verbose:
        print(f"[cylinder] Re={Re:g} grid={nx}x{ny} D={D} cells/D={ncells_per_D} "
              f"nu={nu:.4g} dt={dt:.3e} steps={nsteps} cellPe={Pe:.2f} alpha={alpha:g}")

    ts, Cds, Cls, vprobe = [], [], [], []
    frames = []
    for n in range(1, nsteps + 1):
        t = n * dt
        kick = kick_mask if (0.3 < t < 0.6) else None   # brief startup nudge
        u, v, p, Fx, Fy = step(u, v, poisson, grid, dt, nu,
                               chi_u, chi_v, eta, U_inf, kick=kick,
                               u_rot=u_rot, v_rot=v_rot)
        ts.append(t); Cds.append(Fx / q); Cls.append(Fy / q)
        vprobe.append(v[ip, jp])
        if record_every and n % record_every == 0:
            frames.append(vorticity(u, v, grid).copy())
        if verbose and n % 1000 == 0:
            print(f"  step {n:6d}  t={t:7.2f}  Cd={Fx/q:6.3f}  Cl={Fy/q:+6.3f}")

    return dict(grid=grid, u=u, v=v, p=p, Re=Re, D=D, U_inf=U_inf, nu=nu, dt=dt,
                xc=xc, yc=yc, alpha=alpha, t=np.array(ts), Cd=np.array(Cds),
                Cl=np.array(Cls), vprobe=np.array(vprobe), frames=frames,
                probe=(ip, jp))


def strouhal(t, signal, U_inf, D, t_start_frac=0.5):
    """Dominant frequency of a wake signal (e.g. lift or probe-v) via FFT over
    the second half (developed shedding). Returns (St, f_peak)."""
    t = np.asarray(t); s = np.asarray(signal)
    i0 = int(len(t) * t_start_frac)
    tt, ss = t[i0:], s[i0:] - np.mean(s[i0:])
    dt = tt[1] - tt[0]
    freqs = np.fft.rfftfreq(len(tt), dt)
    amp = np.abs(np.fft.rfft(ss * np.hanning(len(ss))))
    k = np.argmax(amp[1:]) + 1
    # parabolic interpolation around the peak for sub-bin frequency accuracy
    if 1 <= k < len(amp) - 1:
        a, b, c = amp[k - 1], amp[k], amp[k + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    f_peak = (k + delta) * df
    return f_peak * D / U_inf, f_peak


if __name__ == "__main__":
    import time
    t0 = time.time()
    sol = simulate(Re=100, ncells_per_D=40, Lx=12, Ly=6, xc=3, yc=3,
                   t_end=80, verbose=True)
    St, f = strouhal(sol["t"], sol["Cl"], sol["U_inf"], sol["D"])
    half = slice(len(sol["Cd"]) // 2, None)
    Cd_mean = sol["Cd"][half].mean()
    Cl_amp = (sol["Cl"][half].max() - sol["Cl"][half].min()) / 2
    print(f"\n  wall time {time.time()-t0:.0f} s")
    print(f"  Strouhal St = {St:.3f}   (lit ~0.164 at Re=100, Williamson)")
    print(f"  mean Cd = {Cd_mean:.3f}  (lit ~1.33)   Cl amp = {Cl_amp:.3f} "
          f"(lit ~0.33)")
