"""
Incompressible Navier-Stokes -- 2D, staggered MAC grid, Chorin projection.
================================================================================

Primitive variables (u, v, p). The whole difficulty of *incompressible* flow is
the pressure-velocity coupling: pressure has no evolution equation of its own --
it is the Lagrange multiplier that enforces div(u) = 0 at every instant. We
resolve it with Chorin's fractional-step (projection) method:

    1. predictor : u* = u + dt * ( -(u.grad)u + nu * lap(u) )     (ignore pressure)
    2. pressure  : lap(p) = div(u*) / dt                          (enforce div u = 0)
    3. corrector : u^{n+1} = u* - dt * grad(p)                    (now div u = 0)

The pressure Poisson solve (step 2) is an elliptic, symmetric-positive system,
the same family as a FEM stiffness matrix.

Staggered (Marker-and-Cell) layout -- avoids odd/even pressure decoupling:

      p : cell centres            shape (nx,   ny)
      u : vertical   cell faces   shape (nx+1, ny)     u[0,:] = left wall, u[nx,:] = right wall
      v : horizontal cell faces   shape (nx,   ny+1)   v[:,0] = bottom wall, v[:,ny] = top wall

Physics is plain numpy. Only the constant pressure Laplacian is assembled once
as a sparse matrix and LU-factored, then reused every step.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Pressure Poisson operator (constant -> assemble + factor ONCE)
# ---------------------------------------------------------------------------
def build_pressure_laplacian(nx, ny, dx, dy):
    """
    5-point Laplacian on cell centres with homogeneous Neumann BC (dp/dn = 0 at
    every wall -- consistent with the no-penetration velocity correction). The
    pure-Neumann system is singular (p defined up to a constant), so we pin one
    reference cell (index 0) to make it invertible; only grad(p) matters.
    Returns a factorized solver object with a .solve(rhs_flat) method.
    """
    N = nx * ny
    idx = np.arange(N).reshape(nx, ny)        # idx[i,j] -> flat row

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    inv_dx2, inv_dy2 = 1.0 / dx**2, 1.0 / dy**2
    for i in range(nx):
        for j in range(ny):
            r = idx[i, j]
            diag = 0.0
            # x-neighbours (Neumann: skip the missing neighbour, drop its term)
            if i > 0:
                add(r, idx[i - 1, j], inv_dx2); diag -= inv_dx2
            if i < nx - 1:
                add(r, idx[i + 1, j], inv_dx2); diag -= inv_dx2
            # y-neighbours
            if j > 0:
                add(r, idx[i, j - 1], inv_dy2); diag -= inv_dy2
            if j < ny - 1:
                add(r, idx[i, j + 1], inv_dy2); diag -= inv_dy2
            add(r, r, diag)

    L = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Pin reference cell 0 -> Dirichlet p=0 there (removes the null space).
    L = L.tolil()
    L[0, :] = 0.0
    L[0, 0] = 1.0
    L = L.tocsc()
    return spla.splu(L)


# ---------------------------------------------------------------------------
# Ghost-cell helpers (no-slip walls, optional moving lid on top)
# ---------------------------------------------------------------------------
def _pad_u(u, u_top, u_bot):
    """u is (nx+1, ny). Add bottom/top ghost rows enforcing wall tangential
    velocity by linear extrapolation: (u_interior + u_ghost)/2 = u_wall."""
    nxp1, ny = u.shape
    up = np.empty((nxp1, ny + 2))
    up[:, 1:-1] = u
    up[:, 0] = 2.0 * u_bot - u[:, 0]      # below bottom wall
    up[:, -1] = 2.0 * u_top - u[:, -1]    # above top wall
    return up


def _pad_v(v, v_left, v_right):
    """v is (nx, ny+1). Add left/right ghost cols enforcing wall tangential v."""
    nx, nyp1 = v.shape
    vp = np.empty((nx + 2, nyp1))
    vp[1:-1, :] = v
    vp[0, :] = 2.0 * v_left - v[0, :]
    vp[-1, :] = 2.0 * v_right - v[-1, :]
    return vp


# ---------------------------------------------------------------------------
# One projection step (explicit predictor + pressure projection)
# ---------------------------------------------------------------------------
def projection_step(u, v, p_solver, dx, dy, dt, nu,
                    u_top=1.0, u_bot=0.0, v_left=0.0, v_right=0.0):
    """Advance (u, v) one step. Returns new (u, v, p). Central differences for
    both convection and diffusion (2nd order -- matches Ghia's discretisation;
    fine for the low-Re laminar regime we verify in)."""
    nx = u.shape[0] - 1
    ny = u.shape[1]

    # ----- predictor for u (interior x-faces i = 1..nx-1) ------------------
    up = _pad_u(u, u_top, u_bot)                  # (nx+1, ny+2)
    u_P = u[1:nx, :]
    u_E = u[2:nx + 1, :]
    u_W = u[0:nx - 1, :]
    u_N = up[1:nx, 2:ny + 2]
    u_S = up[1:nx, 0:ny]
    # v averaged onto the u-face
    v_at_u = 0.25 * (v[0:nx - 1, 0:ny] + v[1:nx, 0:ny]
                     + v[0:nx - 1, 1:ny + 1] + v[1:nx, 1:ny + 1])
    conv_u = u_P * (u_E - u_W) / (2 * dx) + v_at_u * (u_N - u_S) / (2 * dy)
    diff_u = nu * ((u_E - 2 * u_P + u_W) / dx**2 + (u_N - 2 * u_P + u_S) / dy**2)

    u_star = u.copy()
    u_star[1:nx, :] = u_P + dt * (-conv_u + diff_u)
    u_star[0, :] = 0.0; u_star[nx, :] = 0.0       # no-penetration left/right

    # ----- predictor for v (interior y-faces j = 1..ny-1) ------------------
    vp = _pad_v(v, v_left, v_right)               # (nx+2, ny+1)
    v_P = v[:, 1:ny]
    v_N = v[:, 2:ny + 1]
    v_S = v[:, 0:ny - 1]
    v_E = vp[2:nx + 2, 1:ny]
    v_W = vp[0:nx, 1:ny]
    # u averaged onto the v-face
    u_at_v = 0.25 * (u[0:nx, 0:ny - 1] + u[1:nx + 1, 0:ny - 1]
                     + u[0:nx, 1:ny] + u[1:nx + 1, 1:ny])
    conv_v = u_at_v * (v_E - v_W) / (2 * dx) + v_P * (v_N - v_S) / (2 * dy)
    diff_v = nu * ((v_E - 2 * v_P + v_W) / dx**2 + (v_N - 2 * v_P + v_S) / dy**2)

    v_star = v.copy()
    v_star[:, 1:ny] = v_P + dt * (-conv_v + diff_v)
    v_star[:, 0] = 0.0; v_star[:, ny] = 0.0       # no-penetration bottom/top

    # ----- pressure Poisson: lap(p) = div(u*)/dt ---------------------------
    div = ((u_star[1:nx + 1, :] - u_star[0:nx, :]) / dx
           + (v_star[:, 1:ny + 1] - v_star[:, 0:ny]) / dy)
    rhs = (div / dt).reshape(-1).copy()
    rhs[0] = 0.0                                   # match the pinned reference cell
    p = p_solver.solve(rhs).reshape(nx, ny)

    # ----- corrector: project velocity to divergence-free ------------------
    u_new = u_star.copy()
    v_new = v_star.copy()
    u_new[1:nx, :] -= dt * (p[1:nx, :] - p[0:nx - 1, :]) / dx
    v_new[:, 1:ny] -= dt * (p[:, 1:ny] - p[:, 0:ny - 1]) / dy

    return u_new, v_new, p


# ---------------------------------------------------------------------------
# Driver: lid-driven cavity to steady state
# ---------------------------------------------------------------------------
def solve_cavity(Re=100.0, nx=128, ny=128, L=1.0, U=1.0,
                 cfl=0.4, tol=1e-6, max_steps=200000, verbose=True):
    """
    Lid-driven cavity on [0,L]^2: top lid moves at +U, other 3 walls no-slip.
    March to steady state (stop when max velocity change/step < tol).
    Returns dict with u, v, p, grid, and a convergence history.
    """
    dx = L / nx
    dy = L / ny
    nu = U * L / Re

    # explicit stability: convective CFL and viscous diffusion number
    dt_conv = cfl * min(dx, dy) / U
    dt_visc = cfl * 0.25 * min(dx, dy)**2 / nu
    dt = min(dt_conv, dt_visc)

    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))

    p_solver = build_pressure_laplacian(nx, ny, dx, dy)

    if verbose:
        print(f"[cavity] Re={Re:g} grid={nx}x{ny} nu={nu:.4g} dt={dt:.3e} "
              f"(conv {dt_conv:.2e} / visc {dt_visc:.2e})")

    hist = []
    for step in range(1, max_steps + 1):
        u_old = u
        u, v, p = projection_step(u, v, p_solver, dx, dy, dt, nu, u_top=U)
        if step % 50 == 0 or step == 1:
            change = np.abs(u - u_old).max() / dt      # ~ d|u|/dt, steadiness
            hist.append((step, step * dt, change))
            if verbose and (step % 1000 == 0 or step == 1):
                print(f"  step {step:6d}  t={step*dt:7.3f}  resid={change:.3e}")
            if change < tol:
                if verbose:
                    print(f"  converged at step {step}, t={step*dt:.3f}, "
                          f"resid={change:.3e}")
                break

    return dict(u=u, v=v, p=p, dx=dx, dy=dy, nx=nx, ny=ny, L=L, U=U,
                Re=Re, nu=nu, dt=dt, steps=step, hist=hist)


# ---------------------------------------------------------------------------
# Sampling helpers: velocity on the geometric centrelines (for Ghia compare)
# ---------------------------------------------------------------------------
def centreline_u(sol):
    """u along the vertical centreline x = L/2, as function of y in [0,L].
    Returns (y, u). u lives on x-faces; x=L/2 is exactly face nx/2 (nx even)."""
    nx, ny, dy = sol["nx"], sol["ny"], sol["dy"]
    yc = (np.arange(ny) + 0.5) * dy
    if nx % 2 == 0:
        u_line = sol["u"][nx // 2, :]
    else:
        u_line = 0.5 * (sol["u"][nx // 2, :] + sol["u"][nx // 2 + 1, :])
    # include wall endpoints: bottom wall u=0, top lid u=U (moving lid)
    y = np.concatenate(([0.0], yc, [sol["L"]]))
    u = np.concatenate(([0.0], u_line, [sol["U"]]))
    return y, u


def centreline_v(sol):
    """v along the horizontal centreline y = L/2, as function of x in [0,L]."""
    nx, ny, dx = sol["nx"], sol["ny"], sol["dx"]
    xc = (np.arange(nx) + 0.5) * dx
    if ny % 2 == 0:
        v_line = sol["v"][:, ny // 2]
    else:
        v_line = 0.5 * (sol["v"][:, ny // 2] + sol["v"][:, ny // 2 + 1])
    x = np.concatenate(([0.0], xc, [sol["L"]]))
    v = np.concatenate(([0.0], v_line, [0.0]))
    return x, v


if __name__ == "__main__":
    sol = solve_cavity(Re=100, nx=64, ny=64, tol=1e-6)
    y, u = centreline_u(sol)
    print("centreline u sample (y, u):")
    for k in range(0, len(y), 8):
        print(f"  y={y[k]:.3f}  u={u[k]:+.4f}")
