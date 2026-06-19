"""
Vortex-induced vibration (VIV) -- a spring-mounted cylinder in the wind.

The cylinder moves transversely on a spring + damper (1 structural DOF), coupled
two-way to the flow:

    fluid : Navier-Stokes (as in cylinder.py), but the immersed solid now moves
            and the penalization drives the fluid to the *cylinder's* velocity
    solid : m y'' + c y' + k y = F_lift(t)        (the shed vortices force it)

When the shedding frequency f_s = St U/D approaches the structure's natural
frequency f_n = (1/2pi) sqrt(k/m), the response locks in (synchronization) and
the amplitude blows up -- the mechanism that topples chimneys and galloped the
Tacoma deck's cousins. We show the resonant build-up and the lock-in amplitude
response vs reduced velocity U_r = U/(f_n D), comparable to Khalak & Williamson.

Reuses the verified solver core (make_poisson, vorticity) from cylinder.py.
"""

import numpy as np
from cylinder import Grid, make_poisson, vorticity, strouhal


def step_fsi(u, v, poisson, grid, dt, nu, geom, struct, U_inf):
    """One coupled step. geom/struct are mutable dicts carrying the moving
    cylinder state. Returns Fy (lift on the cylinder)."""
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
    xc = geom["xc"]; yc = geom["yc0"] + struct["y"]
    R = 0.5 * geom["D"]; eta = geom["eta"]
    v_solid = struct["yd"]                       # cylinder transverse velocity

    # current masks at the moved position (cheap: reuse precomputed coords)
    chi_u = ((geom["Xu"] - xc) ** 2 + (geom["Yu"] - yc) ** 2 < R ** 2)
    chi_v = ((geom["Xv"] - xc) ** 2 + (geom["Yv"] - yc) ** 2 < R ** 2)

    # ---- predictor (free-slip top/bottom, inflow/outflow) -- as cylinder.step
    up = np.empty((nx + 1, ny + 2)); up[:, 1:-1] = u
    up[:, 0] = u[:, 0]; up[:, -1] = u[:, -1]
    u_P = u[1:nx, :]; u_E = u[2:nx + 1, :]; u_W = u[0:nx - 1, :]
    u_N = up[1:nx, 2:ny + 2]; u_S = up[1:nx, 0:ny]
    v_at_u = 0.25 * (v[0:nx - 1, 0:ny] + v[1:nx, 0:ny]
                     + v[0:nx - 1, 1:ny + 1] + v[1:nx, 1:ny + 1])
    conv_u = u_P * (u_E - u_W) / (2 * dx) + v_at_u * (u_N - u_S) / (2 * dy)
    diff_u = nu * ((u_E - 2 * u_P + u_W) / dx**2 + (u_N - 2 * u_P + u_S) / dy**2)
    u_star = u.copy(); u_star[1:nx, :] = u_P + dt * (-conv_u + diff_u)
    u_star[0, :] = U_inf; u_star[nx, :] = u_star[nx - 1, :]
    u_star[nx, :] += (U_inf * grid.Ly / dy - u_star[nx, :].sum()) / ny

    vp = np.empty((nx + 2, ny + 1)); vp[1:-1, :] = v
    vp[0, :] = -v[0, :]; vp[-1, :] = v[-1, :]
    v_P = v[:, 1:ny]; v_N = v[:, 2:ny + 1]; v_S = v[:, 0:ny - 1]
    v_E = vp[2:nx + 2, 1:ny]; v_W = vp[0:nx, 1:ny]
    u_at_v = 0.25 * (u[0:nx, 0:ny - 1] + u[1:nx + 1, 0:ny - 1]
                     + u[0:nx, 1:ny] + u[1:nx + 1, 1:ny])
    conv_v = u_at_v * (v_E - v_W) / (2 * dx) + v_P * (v_N - v_S) / (2 * dy)
    diff_v = nu * ((v_E - 2 * v_P + v_W) / dx**2 + (v_N - 2 * v_P + v_S) / dy**2)
    v_star = v.copy(); v_star[:, 1:ny] = v_P + dt * (-conv_v + diff_v)
    v_star[:, 0] = 0.0; v_star[:, ny] = 0.0

    # ---- penalize toward the MOVING solid: u->0, v->v_solid ----------------
    beta = dt / eta
    u_pen = u_star / (1.0 + beta * chi_u)
    v_pen = (v_star + beta * chi_v * v_solid) / (1.0 + beta * chi_v)
    Fy = (chi_v * (v_pen - v_solid) / eta).sum() * dx * dy   # lift on cylinder

    # ---- projection -------------------------------------------------------
    div = ((u_pen[1:nx + 1, :] - u_pen[0:nx, :]) / dx
           + (v_pen[:, 1:ny + 1] - v_pen[:, 0:ny]) / dy)
    p = poisson(div / dt)
    u[:] = u_pen; v[:] = v_pen
    u[1:nx, :] -= dt * (p[1:nx, :] - p[0:nx - 1, :]) / dx
    v[:, 1:ny] -= dt * (p[:, 1:ny] - p[:, 0:ny - 1]) / dy
    u[0, :] = U_inf; u[nx, :] = u[nx - 1, :]; v[:, 0] = 0.0; v[:, ny] = 0.0
    return Fy


def simulate_viv(Re=100, D=1.0, U_inf=1.0, m_star=10.0, zeta=0.01, Ur=5.5,
                 Lx=15, Ly=9, xc=4.0, ncells_per_D=40, eta=1e-4, cfl=0.4,
                 t_end=170, record_every=0, verbose=True):
    """Spring-mounted cylinder. f_n is set from the reduced velocity
    Ur = U/(f_n D) -> f_n = U/(Ur D). Returns y(t), Fy(t), amplitude, frames."""
    nx = int(round(Lx / D * ncells_per_D)); ny = int(round(Ly / D * ncells_per_D))
    grid = Grid(Lx, Ly, nx, ny); nu = U_inf * D / Re
    dx = grid.dx; dt = min(cfl * dx / U_inf, cfl * 0.25 * dx**2 / nu)
    poisson = make_poisson(grid)
    yc0 = Ly / 2

    Xu, Yu = np.meshgrid(grid.xu, grid.yu, indexing="ij")
    Xv, Yv = np.meshgrid(grid.xv, grid.yv, indexing="ij")
    geom = dict(xc=xc, yc0=yc0, D=D, eta=eta, Xu=Xu, Yu=Yu, Xv=Xv, Yv=Yv)

    # structure: fluid-added-mass-free definition. m = m_star * rho * pi R^2 (per
    # unit span, rho=1). f_n from reduced velocity; k,c follow.
    R = 0.5 * D
    m = m_star * np.pi * R**2
    f_n = U_inf / (Ur * D)
    omega_n = 2 * np.pi * f_n
    k = m * omega_n**2
    c = 2 * zeta * np.sqrt(k * m)
    struct = dict(y=0.0, yd=0.0)

    u = np.full((nx + 1, ny), U_inf); v = np.zeros((nx, ny + 1))

    nsteps = int(t_end / dt)
    if verbose:
        print(f"[VIV] Re={Re} Ur={Ur} m*={m_star} zeta={zeta} f_n={f_n:.4f} "
              f"grid={nx}x{ny} dt={dt:.2e} steps={nsteps}")

    ts, ys, Fys = [], [], []
    frames = []
    # one-sided kick early to start shedding
    kick_mask = ((Xv > xc + 0.3 * D) & (Xv < xc + D) & (Yv > yc0) & (Yv < yc0 + 0.4 * D))
    for n in range(1, nsteps + 1):
        t = n * dt
        Fy = step_fsi(u, v, poisson, grid, dt, nu, geom, struct, U_inf)
        if 0.3 < t < 0.6:
            v[kick_mask] += 0.1 * U_inf
        # structural update (symplectic Euler): m y'' + c y' + k y = Fy
        acc = (Fy - c * struct["yd"] - k * struct["y"]) / m
        struct["yd"] += dt * acc
        struct["y"] += dt * struct["yd"]
        ts.append(t); ys.append(struct["y"]); Fys.append(Fy)
        if record_every and n % record_every == 0:
            frames.append((struct["y"], vorticity(u, v, grid).copy()))
        if verbose and n % 2000 == 0:
            print(f"  t={t:7.1f}  y/D={struct['y']/D:+.3f}  Fy={Fy:+.3f}")

    ts = np.array(ts); ys = np.array(ys); Fys = np.array(Fys)
    half = ts > ts[-1] * 0.6
    A_D = (ys[half].max() - ys[half].min()) / 2 / D
    St, _ = strouhal(ts, Fys, U_inf, D)
    return dict(grid=grid, t=ts, y=ys, Fy=Fys, A_D=A_D, f_n=f_n, Ur=Ur,
                St=St, Re=Re, D=D, U_inf=U_inf, yc0=yc0, xc=xc, frames=frames, dt=dt)


if __name__ == "__main__":
    s = simulate_viv(Re=120, Ur=5.5, t_end=160, verbose=True)
    print(f"\n  lock-in amplitude A/D = {s['A_D']:.3f}  (f_n={s['f_n']:.3f}, "
          f"response St={s['St']:.3f})")
