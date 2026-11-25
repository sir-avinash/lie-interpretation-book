---
name: ch12-variational-integrator-design
description: Derive, implement and validate a variational integrator for a mechanical system on R^n, S^2, SO(3) or a product of them, and choose a transcription (collocation in a chart, flat outputs, or variational constraints) for trajectory optimization. Use when a simulation must hold energy and momentum over long horizons or coarse steps, when an integrator that claims to be structure-preserving must be tested, or when discrete dynamics are needed as equality constraints of a planner.
---

# Design and validate a variational integrator, and pick a transcription

Follows Chapter 12. Notation: time step `h`, knots `k = 0..N`; `R` in SO(3) maps body to world; step rotation `Upsilon_k = R_k^\top R_{k+1} = \exp(h\,\Omega_k^\wedge)` with step velocity `Omega_k`; discrete momentum `p_k` (body frame on SO(3), world frame on S^2); `\bar J = \tfrac12\operatorname{tr}(J) I - J`; total body torque `tau_k` at `R_k`; gravity along `-e3`. Code names: `hat`, `vee`, `expm_so3`, `log_so3`.

## When to use

- Long simulations (many periods), coarse steps, or any use where a leak of energy or momentum changes the qualitative answer.
- A planner needs the dynamics as algebraic constraints between knots.
- Do not use it for a short, accurate trajectory with cheap force evaluations: a fourth-order Runge-Kutta method with the exponential update is more accurate there (0.00067° against 0.68° after 1 s at `h = 0.02` s on the 3D pendulum).

## Procedure: derive

1. Write the Lagrangian `L` on the configuration manifold (Chapter 4). List its symmetries: each one is a momentum the integrator must conserve exactly.
2. Choose the discrete Lagrangian `L_h(x_k, x_{k+1})`, an approximation of the action over one step.
   - Vector `x`: `L_h = \frac{m}{2h}\|x_{k+1}-x_k\|^2 - \frac h2(\text{potential at } x_k + \text{potential at } x_{k+1})`.
   - `R` in SO(3): `L_h = \frac1h \operatorname{tr}\big((I - \Upsilon_k)\bar J\big) - \frac h2(\text{potential at } R_k + \text{potential at } R_{k+1})`.
   - `q` in S^2: `L_h = \frac{m l^2}{2h}\|q_{k+1}-q_k\|^2 - \frac h2(\text{potential at } q_k + \text{potential at } q_{k+1})`.
   - Product manifold: add the terms factor by factor.
3. Vary the interior knots so that they stay on the manifold, ends fixed: `\delta x_k`; `\delta R_k = R_k \eta_k^\wedge`, which gives `\delta\Upsilon_k = -\eta_k^\wedge\Upsilon_k + \Upsilon_k\eta_{k+1}^\wedge`; `\delta q_k = \xi_k \times q_k`.
4. Read off the discrete momentum from the two slopes of `L_h`: `p_k = -\partial L_h/\partial x_k`, `p_{k+1} = \partial L_h/\partial x_{k+1}` (on SO(3) and S^2, the coefficients of `\eta_k`, `\xi_k`). Use `\operatorname{tr}(x^\wedge A) = -x\cdot(A - A^\top)^\vee`.
5. Add non-conservative forces by the discrete virtual work `\frac h2\tau_k\cdot\eta_k + \frac h2\tau_{k+1}\cdot\eta_{k+1}` per step.
6. The step is: solve the first slope equation for the next configuration, evaluate the second.

## Results to reuse

Rigid body or 3D pendulum on SO(3), with `\theta_k = h\|\Omega_k\|`:

```
p_k + (h/2) tau_k = (sin(theta_k)/theta_k) J Omega_k
                    + ((1 - cos(theta_k))/theta_k^2) h Omega_k x J Omega_k     (solve for Omega_k)
R_{k+1} = R_k expm_so3(h Omega_k)
p_{k+1} = expm_so3(-h Omega_k) (p_k + (h/2) tau_k) + (h/2) tau_{k+1}
```

Solve the first line by Newton's method on `Omega_k` in R^3, starting from `J^{-1}(p_k + (h/2) tau_k)`, with the Jacobian `J + (h/2)(hat(Omega_k) J - hat(J Omega_k))`. Expect 5 to 7 iterations to a relative tolerance of `1e-14`.

World-frame form, valid at any `h`: `R_{k+1} p_{k+1} = R_k p_k + (h/2)(R_k tau_k + R_{k+1} tau_{k+1})`.

Spherical pendulum on S^2 (explicit), `p_k` the angular momentum about the pivot in the world frame:

```
pbar    = p_k - (h/2) m g l  q_k x e3
q_{k+1} = (h/(m l^2)) pbar x q_k + sqrt(1 - |h pbar/(m l^2)|^2) q_k
p_{k+1} = pbar - (h/2) m g l  q_{k+1} x e3
```

Particle (Störmer-Verlet): `m(x_{k+1} - 2x_k + x_{k-1}) = h^2 F(x_k)`.

## Checks that must pass

1. **Stationarity.** Along a computed discrete trajectory, the finite-difference slope of `L_h(x_{k-1}, x_k) + L_h(x_k, x_{k+1})` under the manifold variation of `x_k` is zero (below `1e-8` with a step of `1e-6`).
2. **Momentum definition.** The finite-difference slopes of `L_h` in its two slots equal `-p_k` and `p_{k+1}`.
3. **Implicit residual.** The solved step satisfies its equation to `1e-13`.
4. **Manifold.** `\|R^\top R - I\|` and `\|q\|^2 - 1` stay at rounding level (`1e-13`) over the whole run.
5. **Momentum (discrete Noether).** Every momentum that belongs to a symmetry of `L_h` is constant to rounding: `R_k p_k` for a free body, `e_3\cdot R_k p_k` for the 3D pendulum, `e_3\cdot p_k` for the spherical pendulum.
6. **Energy on an uncontrolled run.** Run at least 100 periods with no input and no damping. The energy error must stay in a band, the band of the last tenth must equal the band of the first tenth, and halving `h` must divide it by 4.
7. **Order.** Against a reference at a much smaller step, halving `h` divides the error of the configuration by 4.
8. **Continuous model.** The continuous equations the scheme is meant to approximate agree with a second derivation (geomech `check_eom` below `1e-5`).

## Choosing a transcription for trajectory optimization

| Situation | Use | Constraint between knots |
| :-- | :-- | :-- |
| System has flat outputs | Collocate the chain of integrators of the flat output, recover the rest algebraically | trapezoid `x_{k+1} - x_k - (h/2)(xdot_k + xdot_{k+1}) = 0` |
| Rotations stay well inside one turn, Euclidean solver | Chart `R = R_0 \exp(\eta^\wedge)`, `q = \exp(\xi^\wedge) q_0`, with exact kinematics `\dot\eta = \operatorname{dexp}_\eta^{-1}\Omega` | trapezoid or Hermite-Simpson on `(eta, Omega)` |
| Large rotations, flips, long horizons, or a solver that steps by a retraction | Variational constraints, `Omega_k` (or `Upsilon_k`) as a variable | below |

Variational constraints at each interior knot, with `Upsilon_k = \exp(h\,\Omega_k^\wedge)`:

```
log_so3(R_{k+1}^T R_k expm_so3(h Omega_k)) = 0
vee(Upsilon_k Jbar - Jbar Upsilon_k^T)
  - Upsilon_{k-1}^T vee(Upsilon_{k-1} Jbar - Jbar Upsilon_{k-1}^T) - h^2 tau_k = 0
```

Hermite-Simpson: `x_{k+1/2} = (x_k + x_{k+1})/2 + (h/8)(xdot_k - xdot_{k+1})`, defect `x_{k+1} - x_k - (h/6)(xdot_k + 4 xdot_{k+1/2} + xdot_{k+1}) = 0`.

After solving, always:

1. Check the largest constraint residual and `\|R_k^\top R_k - I\|` at every knot.
2. Replay the inputs through an integrator at least 100 times finer and report the miss at the final knot. At `h = 0.1` s a second-order transcription of a 150° turn misses by about 1°.
3. For a variational transcription, check the momentum balance `R_N p_N - R_0 p_0 = \sum (h/2)(R_k tau_k + R_{k+1} tau_{k+1})`.

## Failure modes

- **Explicit step rotation.** Computing `Upsilon_k = \exp(h\,(J^{-1}p_k)^\wedge)` from the current momentum, instead of solving the implicit line, keeps the manifold and the momentum and lets the energy drift in proportion to `h` (+108 J in 60 s at `h = 0.02` s on the spinning 3D pendulum). Check 6 catches it. Checks 4 and 5 do not.
- **Entrywise collocation of `R`.** Knot matrices leave SO(3) by `O(h^2\|\Omega\|^2)`, and end conditions on nine entries are dependent. Do not add `R^\top R = I` to repair it. Change the transcription.
- **First-order chart kinematics.** Using `\dot\eta \approx \Omega` plans for a different body: a 15° miss on a 150° turn. Use `dexp^{-1}`.
- **Chart edge.** `\operatorname{dexp}_\eta` is singular at `\|\eta\| = 2\pi`. Re-center `R_0` or switch to variational constraints.
- **Variable step.** Changing `h` between steps destroys the energy band. Keep `h` fixed, or use an integrator built for adaptive steps.
- **Step too large on S^2.** The square root needs `\|h\,\bar p/(m l^2)\| < 1`.
- **Expecting energy conservation with forcing.** With applied torques or drag only the momentum balance is exact.
- **Impacts and slack cables** are outside these formulas.
