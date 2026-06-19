# Incompressible Flow Solver and Vortex Flow-Meter

A two-dimensional incompressible Navier-Stokes solver written from first principles in NumPy, used
to model flow past a cylinder (the von Karman vortex street and vortex-induced vibration) and then
inverted: from the wake signal it recovers the free-stream velocity, with a quantified uncertainty.
The core uses NumPy.

![von Karman vortex street at Reynolds number 100](output/cylinder_Re100.png)

## What it does

**Forward.** The solver integrates the incompressible Navier-Stokes equations
`du/dt + (u . grad) u = -grad p + nu lap u`, `div u = 0` on a staggered (MAC) grid with Chorin
projection. The pressure Poisson equation is solved spectrally with a cosine transform. A cylinder
is represented as an immersed boundary by volume penalisation, and the integrated penalty reaction
is the hydrodynamic force on the body. Above a Reynolds number of about 47 the wake sheds the von
Karman vortex street; mounted on a spring, the cylinder undergoes vortex-induced vibration lock-in.

**Inverse.** A bluff body sheds vortices at `f = St(Re) U / D`. A vortex flow-meter reads the flow
speed directly from that frequency. The same procedure is applied to the solver: a known free-stream
velocity produces a shedding frequency, and inverting the solver-calibrated `St(Re)` relation
recovers the velocity, with a 95 percent confidence interval from a sensor-noise Monte-Carlo.

## Verification

| Check | This solver | Reference | |
|---|---|---|---|
| Lid-driven cavity centreline u, v (Re 100) | RMS 2.2e-3 / 4.5e-3 | Ghia, Ghia and Shin 1982 | ok |
| Steady wake bubble L_w / D (Re 20 / 40) | 0.88 / 2.03 | 0.93 / 2.24 (Coutanceau-Bouard) | ok |
| Shedding Strouhal number St (Re 100) | 0.18 | 0.16 unbounded (Williamson 1988) | plus blockage |
| Drag and lift Cd, Cl (Re 100) | 1.56 / 0.42 | about 1.33 / 0.33 unbounded | plus blockage |
| VIV lock-in peak A / D | see `viv_lockin.png` | about 0.5 to 0.6 (Khalak and Williamson) | ok |
| Flow-meter round trip: recover U | see `flowmeter_roundtrip.png` | known input U, with CI | ok |

A finite domain confines the flow (blockage), raising St, Cd and Cl above the unbounded references.
The offset is reported, not removed by tuning: the Strouhal number falls from 0.20 at 17 percent
blockage to 0.18 at 8 percent, approaching 0.16 as the walls recede. The inverse uses the `St(Re)`
calibrated from the solver, so the blockage cancels in the round trip, in the same way that a real
meter is calibrated per device.

## How to run

```bash
python solver/verify_cavity.py 100 128   # solver correctness vs Ghia 1982
python solver/verify_cylinder.py 100     # vortex street: St, Cd, Cl, with a snapshot
python solver/verify_cylinder_steady.py  # steady wake length vs the literature
python solver/verify_viv.py              # vortex-induced vibration lock-in
python solver/verify_flowmeter.py        # the inverse: recover the flow speed with a CI
python solver/export_web.py 100          # writes the web animation data
```

Then open `web/index.html` for a live vortex street; the data is injected, so no server is required.

## Limitations

Two-dimensional and laminar (Reynolds number up to a few hundred); a single circular cylinder.
Volume penalisation gives order `sqrt(eta)` boundary accuracy and a slightly thick effective
cylinder. The finite-domain blockage shifts the force coefficients, which is quantified above and
cancels in the inverse. Explicit time stepping caps the time step.

## References

1. U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow using the
   Navier-Stokes equations and a multigrid method," *J. Comput. Phys.*, 48, 1982.
2. A. J. Chorin, "Numerical solution of the Navier-Stokes equations," *Math. Comp.*, 22, 1968
   (the projection method).
3. C. H. K. Williamson, "Vortex dynamics in the cylinder wake," *Annu. Rev. Fluid Mech.*, 28, 1996;
   the Strouhal-Reynolds relation (1988).
4. M. Coutanceau and R. Bouard, steady wake recirculation length behind a circular cylinder,
   *J. Fluid Mech.*, 79, 1977.
5. A. Khalak and C. H. K. Williamson, "Motions, forces and mode transitions in vortex-induced
   vibrations at low mass-damping," *J. Fluids Struct.*, 13, 1999.
6. P. Angot, C.-H. Bruneau and P. Fabrie, "A penalization method to take into account obstacles in
   incompressible viscous flows," *Numer. Math.*, 81, 1999.
