---
name: ch04-manifold-dynamics-modeling
description: Derive, validate and simulate coordinate-free equations of motion for a mechanical system whose configuration is a product of R^3, S^2 and SO(3) (rigid bodies, pendulums, quadrotors, cable-suspended loads, legs). Use when a new robot model is needed without Euler angles, when an existing model must be checked, or when a simulation drifts off its manifold.
---

# Derive and validate equations of motion on a manifold

Follows Chapter 4. Notation: `R` in SO(3) maps body to world, `R_dot = R hat(Omega)` with `Omega` in the body frame; `q` in S^2 with `q_dot = omega x q`, `omega . q = 0`; gravity along `-e3`; body torque `tau`, thrust `f`.

## When to use

- A new system needs a dynamic model and some part of it rotates through large angles.
- An Euler-angle model shows `1/cos`, `tan`, or a mass matrix whose condition number grows with attitude.
- A hand-derived or published model must be checked before it is used for control.
- A simulated attitude stops being a rotation, or a simulated cable changes length.

## Procedure

1. **Choose the configuration manifold.** One factor per part: `R^3` for a position, `SO(3)` for a rigid body's attitude, `S^2` for a massless link or cable direction, `R` for a length or joint angle. Write every other position through these, for example `x_Q = x_L - l q`. Count degrees of freedom (3, 3, 2, 1 per factor) and inputs.

2. **Write the Lagrangian** `L = T - V` in the velocities `x_dot`, `omega` (or `q_dot`), `Omega`. Rigid body: `T = 0.5 m |x_dot|^2 + 0.5 Omega . J Omega`, `V = m g e3 . x`.

3. **Write the virtual work** of every applied force, using the variations

   ```
   delta x = free            delta q = xi x q,  xi . q = 0            delta R = R hat(eta)
   delta q_dot = xi_dot x q + xi x q_dot       delta Omega = eta_dot + Omega x eta
   ```

   - Body torque `tau`: `dW = tau . eta`.
   - World-frame force `F` at the body point `x + R r`: `dW = F . delta x + eta . (r x R^T F)`.
   - World-frame force `F` at the point `x - l q`: `dW = F . (delta x - l xi x q)`.

4. **Apply Lagrange-d'Alembert**, `delta int L dt + int dW dt = 0`: vary, integrate by parts, collect the coefficient of each variation.
   - Coefficient of `delta x`: a full vector equation.
   - Coefficient of `eta`: a full vector equation. The term `Omega x J Omega` comes from `Omega x eta` in `delta Omega`.
   - Coefficient of `xi`: only the part perpendicular to `q` is an equation. Project with `I - q q^T` and add `q . omega_dot = 0`.

   If the free-body diagram is simple, derive by Newton-Euler instead and eliminate the constraint forces by the same projection. Use `q . q_ddot = -|q_dot|^2` and `q x q_ddot = omega_dot`.

5. **Compare with the reference forms.**

   ```
   spherical pendulum     m l^2 omega_dot = -m g l q x e3
                          tension T = m l |q_dot|^2 - m g e3 . q
   rigid body             m x_ddot = F - m g e3
                          J Omega_dot + Omega x J Omega = tau + r x R^T F
   3D pendulum            J Omega_dot + Omega x J Omega = m g (R^T e3) x rho + tau      (J about the pivot)
   quadrotor              m x_ddot = f R e3 - m g e3
                          J Omega_dot + Omega x J Omega = tau
   quadrotor with load    (m_Q + m_L)(xddot_L + g e3) = (q . f R e3 - m_Q l |q_dot|^2) q
                          m_Q l omega_dot = -q x f R e3
                          J Omega_dot + Omega x J Omega = tau
                          tension T = m_L / (m_Q + m_L) * (m_Q l |q_dot|^2 - q . f R e3)
   ```

6. **Derive a second time with geomech** and compare numerically.

   ```python
   from geomech import check_eom, compute_eom, to_standard_form
   from geomech.codegen import StateLayout, collect_params, load_module, module_source
   from geomech.simulation import accelerations

   eom = compute_eom(L, dW, variables)
   err = check_eom(L, dW, variables, eom, inputs=inputs)        # require err < 1e-5
   sf = to_standard_form(eom, variables, inputs)
   layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
   plant = load_module(module_source(sf, layout, lagrangian=L), "plant")
   a_ref = accelerations(plant, state, u, params)               # tangent-plane solve included
   ```

   Evaluate your hand-written accelerations at 10 or more random states on the manifold (`|q| = 1`, `omega . q = 0`, `R` a rotation) with random inputs. They must match `a_ref` to `1e-8`. geomech's signs: each equation is `E = -(Euler-Lagrange) + forcing = 0`, the standard-form mass block is minus the physical inertia, and S^2 equations are returned unprojected.

7. **Check conservation in simulation** with all inputs zero: total energy; `|q| = 1` and `omega . q = 0`; `R^T R = I`. Add the momentum the symmetries give: all of `R J Omega` for a free body, `e3 . R J Omega` for a body pivoting under gravity, `e3 . omega` for a spherical pendulum. With inputs, check that the energy change equals the integral of the input power.

8. **Integrate on the manifold.** First order: `R <- R expm(h hat(Omega))`, `q <- expm(h hat(omega)) q`. Higher order: run the Runge-Kutta stages on a rotation vector `theta` with `R = R_k expm(hat(theta))`, `theta_dot = Omega + 0.5 theta x Omega + (1/12) theta x (theta x Omega)`, and apply the exponential at the end of the step.

## Checks that must pass

| Check | Pass condition |
| :-- | :-- |
| Variation formulas | finite-difference `delta q`, `delta q_dot`, `delta Omega` match step 3 to `1e-5` |
| geomech derivation | `check_eom < 1e-5` |
| Hand form against geomech | accelerations agree to `1e-8` at random states and inputs |
| Tangency | every S^2 acceleration satisfies `q . omega_dot = 0` |
| Energy, no inputs | relative drift below `1e-6` with a tight-tolerance integrator |
| Momentum | the conserved components named in step 7 hold to the same tolerance |
| Work-energy, with inputs | energy change equals integrated input power to `1e-5` |
| On-manifold | `norm(R^T R - I) < 1e-10`, `abs(norm(q) - 1) < 1e-10` after a long run |
| Rigid-body sanity | a spin about the intermediate principal axis flips; spins about the other two do not |

`verify/ch04.py` implements all of these and is the reference.

## Failure modes

- **Solving an S^2 equation without projecting.** The unprojected 3 x 3 solve gives `q . omega_dot != 0` and the state leaves the sphere.
- **Varying with `q + eps dq` or `R + eps dR`.** The varied curve leaves the manifold and the derived equation contains a spurious constraint force.
- **Dropping `Omega x eta` from `delta Omega`.** The gyroscopic term `Omega x J Omega` disappears.
- **Mixing frames in the virtual work.** A world-frame force must be rotated: the moment is `r x R^T F`, not `r x F`.
- **Wrong inertia.** For a body on a pivot `J` is about the pivot, not about the center of mass.
- **A cable that pushes.** The taut-cable equations stay smooth when the tension formula goes negative. Monitor its sign and switch models at zero.
- **Reading manifold preservation as accuracy.** An exponential update keeps `R` a rotation. It does not conserve energy. Check the energy drift separately.
- **Sign conventions when comparing.** Compare accelerations, not printed equations: geomech's `E` and mass blocks carry a minus sign.
