---
name: ch20-legged-geometric-modeling
description: Model and control a legged robot on a product of SO(3)s. Choose the factors, derive and check the stance dynamics, derive and check the impact map at touchdown, track a motion primitive with a geometric law on each factor, integrate with a step that stays on the manifold, and test the hybrid loop over many steps. Use for a biped or quadruped reduced model whose bodies each carry an attitude, when turning, leaning or a variable inertia matter. Not for contact-rich whole-body models with friction cones.
---

# A legged robot on a product of SO(3)s

## When to use

- A reduced legged model with one attitude per body (legs, torso) and a few lengths, in stance on a foot that does not slip.
- You need one model for standing, walking and turning, with no singular posture.
- Not for underactuated point-foot gaits, and not for contact planning. This procedure assumes an ankle that supplies torque.

## Conventions (fixed)

- World frame with `e3` up, gravity along `-e3`. `R` maps body to world, `R_dot = R hat(Omega)`.
- Stance leg `R1`, torso `RP`, swing leg `R2`, stance leg length `l`, slides `slide` (math: `r_i`). Both leg axes point from foot to hip: hip `x_P = l R1 e3` from the stance foot, swing foot `x_c = x_P - l_0 R2 e3`.
- Inputs `u = (f, tau_1, tau_P, tau_2, f_s)`: leg force, ankle torque, hip torque between stance leg and torso, hip torque between torso and swing leg, slide forces, each torque in the frame of the body it acts on.
- Errors take the actual configuration first: `e_R(R, Rd)`. Time step `h` is also the sampling period.
- A source that puts `e3` along gravity: flip the sign of every gravity term before comparing.

## Procedure

Do the steps in order. Do not start a step before the check of the previous one passes.

1. **Factors** (`ch02-manifold-kinematics`). One SO(3) per rigid body, one R per length. Count dimensions twice: the manifold, and the degrees of freedom that enter the stance dynamics. A leg balanced about the hip loses its length from the dynamics.
2. **Stance dynamics** (`ch04-manifold-dynamics-modeling`, `ch05-geomech-symbolic-derivation`). Build the Lagrangian mass by mass, not from a reduced kinetic energy. For a configuration-dependent inertia `J(l) = J_0 + m l^2 Pi`, `Pi = I - e3 e3^T`, expect the term `J_dot Omega` on the left:
   `J(l) Omega_dot + Omega x J(l) Omega + J_dot Omega = m g l (R^T e3) x e3 + tau`.
   Check: `check_eom < 1e-5`, and hand-written accelerations equal the symbolic model at random states to `1e-9`.
3. **Floating base.** Add the hip position `x_P`. Write the inertia `D` of the velocity `(Omega_1, Omega_P, r_dot, Omega_2, v_P)` and the Jacobian `A` of the swing foot, `A = [0 0 0  l_0 R2 hat(e3)  I]`.
   Check: `D` equals the mass matrix of a second symbolic derivation.
4. **Impact map.** Solve `D (v+ - v-) = A^T p_c`, `A v+ = 0`. If every body has its center of mass at the hip, only the landing leg and the hip change:
   `J(l_0) Omega_2+ = J_0 Omega_2- + m l_0 e3 x R2^T v_P-`,  `v_P+ = l_0 R2 (Omega_2+ x e3)`,  `p_c = m (v_P+ - v_P-)`.
   Then trade the names of the legs: `(l, l_dot, R1, Omega1, R2, Omega2) -> (l_0, 0, R2, Omega_2+, R1, Omega_1-)`.
   Checks, at random states: closed form equals the linear system; angular momentum about the landing foot, summed mass by mass, is conserved; energy lost equals `(v+ - v-) . D (v+ - v-) / 2 >= 0`; the relabeled stance state reproduces `v_P+`; a second landing changes nothing; trading names twice is the identity.
5. **Motion primitive.** One closed-form reference per factor over a step of duration `t_s`: leg angles, leg length (lengthen in mid-step for foot clearance), torso attitude, slides. Give velocities and accelerations in closed form.
   Check: the reference swing foot is above the ground for `0 < t < t_s` and on it at `t_s`.
6. **Tracking law** (`ch06-geometric-tracking-control`). Per body, a net torque, then stack: `tau_2 = n_2`, `tau_P = n_P + RP^T R2 tau_2`, `tau_1 = n_1 + R1^T RP tau_P`. Two equivalent forms:
   - eq6-law with the terms of a varying inertia: `n = -k_R e_R - k_Omega e_Omega + Omega x J Omega - J (Omega x R^T Rd Omegad - R^T Rd Omegad_dot) + J_dot Omega - J_dot e_Omega / 2`.
   - with the error `R Rd^T` and `Psi = tr(D (I - R Rd^T))`: `n = -e - k_Omega (Omega - Omegad) + J Omegad_dot + Omegad x J Omega + J_dot (Omega + Omegad) / 2`, `e = sum_i d_i (R^T e_i) x (Rd^T e_i)`.
   Cancel the gravity torque on the stance leg. Lengths: PD with feedforward.
   Check: the rate of the Lyapunov function along the closed loop equals `-sum k_Omega |velocity error|^2` minus the damping of the lengths, by finite differences.
7. **Integrate** (`ch12-variational-integrator-design`). Exponential update per factor. Test energy on an uncontrolled run first: constant leg force `f`, conserved quantity `energy - f l`.
8. **Hybrid loop.** Step the stance dynamics, detect the swing foot crossing the ground from above (only after mid-step), bisect for the touchdown time, apply the impact map, move the stance foot, restart the primitive.
9. **Test over many steps.** Record per step: step length, step time, errors at start and at touchdown, foot clearance, impulse. Record along the motion: ground force `m (x_P_ddot + g e3)`, its ratio horizontal over vertical, joint torques.
   Checks: vertical force and vertical impulse positive; force and impulse ratios below the friction coefficient; step-to-step change shrinking to a tolerance; `R^T R = I` to rounding error; results stable when `h` is halved.
10. **Local optimal control** (`ch07-variation-lqr-design`), optional. Linear state `(dl, dr, eta_1, eta_P, eta_2)` and velocity variations. Kinematic rows `eta_dot = dOmega - Omega x eta`. Validate against the symbolic linearization, then integrate the Riccati equation backward over one stance phase.

## Failure modes

- **Gravity sign.** A source with `e3` down, mixed with figures that draw it up. Symptom: the leg falls upward. Flip every gravity term.
- **Foot on the wrong side.** `x_P + l_0 R2 e3` with a plain exchange of the legs gives a new stance leg whose kinematics imply minus the hip velocity. Use `x_P - l_0 R2 e3`.
- **Right side of the impact system.** It is `+D v-`. With `-D v-` a velocity that satisfies the constraint comes back reversed.
- **Missing half-rate term.** eq6-law alone on a body with varying inertia: the Lyapunov rate gains `e_Omega . J_dot e_Omega / 2`.
- **Weights on world axes.** `tr(D (I - R Rd^T))` with unequal `d_i` resists rotation about world axes with `d_2 + d_3`, `d_1 + d_3`, `d_1 + d_2`. The gait then depends on the heading. Test at two headings.
- **Reference that stops at touchdown.** Every step starts with a velocity error. Expect a sawtooth in the Lyapunov function.
- **High gains.** The friction ratio rises before the torque bound is reached. Monitor it.
- **Early touchdown.** Tracking error can scuff the swing foot. Detect touchdown only after mid-step, and report the smallest clearance.
- **Explicit exponential update.** Stays on the group and gains energy in proportion to `h`. Test energy open loop.
- **Twisting torque at a point foot.** A turn asks the ankle for a torque about the vertical. A point foot has none.

## What this procedure does not give

Hybrid stability of the gait, underactuated stance, friction-cone enforcement, a planner, an estimator in the loop. For the estimator see `ch14-invariant-filter-design` (contact-aided filter). For planning see `ch13-manifold-mpc-design` and `ch10-flatness-trajectory-design`.
