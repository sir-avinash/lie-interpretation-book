---
name: ch18-cable-suspended-load
description: Control a quadrotor with a cable-suspended load end to end on R^3 x S^2 x SO(3). Plan the load's path in flat outputs and check feasibility, recover the vehicle's state and inputs, track with the load cascade and flat feedforward, linearize on the product manifold and validate it, adapt to an unknown load mass and a gust as a lumped force on the thrust vector, protect the taut-cable assumption with a cone barrier and a tension floor, estimate the cable direction with a variation-based EKF on S^2, and test the blocks together with an ablation. Use when asked to fly, simulate, plan for or stabilize a load hanging on a cable under one multicopter, when a load swings at set-down, sags, or the cable goes slack, or as the checklist for taking any robot through the whole recipe.
---

# A quadrotor with a cable-suspended load, end to end

## When to use

- One multicopter carries a point mass on a cable attached at its center of mass, and the load's motion is what matters.
- You need the order of work, the check at each stage, and the per-chapter skill that holds the details.
- Not for a slack cable (hybrid model, not covered), several vehicles (`ch04-manifold-dynamics-modeling`, eq4-multi) or a cable attached off the center of mass (the attitude then feels the load and the flat map changes).

## Conventions (fixed)

- World frame, `e3` up, gravity along `-e3`. `R` maps body to world, `R_dot = R hat(Omega)`.
- Cable direction `q` in S^2 points from the quadrotor to the load: `x_Q = x_L - l q`, `q_dot = omega x q`, `omega . q = 0`.
- Tension `T > 0` when the cable pulls: `-T q = m_L (x_L_ddot + g e3)`.
- Thrust vector `F = f R e3`. The cascade's request is `F_d`.
- Believed load mass `m_L0` (subscript 0: what the controller believes). True mass `m_L`.
- Variations `xi` (S^2, `xi . q = 0`), `eta` (SO(3)). Estimates `qhat`, `omegahat`. Innovation `ytilde` = predicted minus measured.
- Code names: `hat`, `vee`, `expm_so3`, `log_so3`, `e_q(q, qd)`, `e_omega(q, omega, omegad)`, `e_R(R, Rd)`, `e_Omega(R, Omega, Rd, Omegad)`.
- Model library: `robots.get("quadrotor_load")` in `case_studies/robots`. Reference implementation of everything below: `verify/ch18.py`.

## The model

```
(m_Q + m_L)(x_L_ddot + g e3) = (q . F - m_Q l |q_dot|^2) q        F = f R e3
m_Q l omega_dot = -q x F
J Omega_dot + Omega x J Omega = tau
T = m_L / (m_Q + m_L) * (m_Q l |q_dot|^2 - q . F)
```

Two facts to use everywhere: the load accelerates only along `q`, and the cable equation does not contain `m_L`. The cable equation is the spherical pendulum of mass `m_Q` pushed by `m_Q g e3 - f R e3`, so every S^2 result for a pendulum applies with that substitution.

## Procedure

1. **Model and check** (`ch02-manifold-kinematics`, `ch04-manifold-dynamics-modeling`, `ch05-geomech-symbolic-derivation`).
   Take the model from the library. Derive it again with geomech: `check_eom < 1e-5`, accelerations equal at random states to `1e-9`. An external force `F_e` on the quadrotor adds to `F`. Integrate with the library `step`.
   Check: `| |q| - 1 |` and `R^T R - I` at rounding level over the whole run.

2. **Plan the load** (`ch10-flatness-trajectory-design`).
   Flat output: `x_L(t)` and heading `psi`. Curve through waypoints minimizing the squared sixth derivative, pieces of degree 11, at rest up to the fifth derivative at both ends.
   ```
   T = m_L0 |x_L_ddot + g e3|      q = -(x_L_ddot + g e3) / |x_L_ddot + g e3|      x_Q = x_L - l q
   F = m_Q (x_Q_ddot + g e3) + m_L0 (x_L_ddot + g e3)
   ```
   then `R`, `Omega`, `Omega_dot`, `tau` from `F` as for a bare quadrotor. State needs five derivatives of `x_L`, the torque six.
   Checks before flying: cable angle inside the cone with margin for estimation error, `T > T_min`, thrust and torque inside limits, the course does not double back (waypoint times), and open-loop flight on the model reproduces the curve (start the test at the instant the course begins: the torque jumps there).
   If a check fails, stretch the time or move a waypoint. Expect thrust and the cone to bind before the tension.

3. **Track** (`ch06-geometric-tracking-control`).
   ```
   F_L = -k_x e_x - k_v e_v + (m_Q + m_L0)(x_d_ddot + g e3)
   A   = F_L + m_Q l |q_dot|^2 q            qd = -A / |A|
   omega_dot = (I - q q^T) omegad_dot - (q . omegad) q_dot - k_q e_q - k_omega e_omega
   F_d = (A . q) q + m_Q l q x omega_dot
   f = F_d . R e3,  Rd from F_d and the heading (eq6-Rd),  tau = attitude law (eq6-law) with Omegad, Omegad_dot
   ```
   `omegad`, `omegad_dot`, `Omegad`, `Omegad_dot` come from the flat map of step 2.
   Checks: on the reference the controller returns the flat inputs to rounding. With zero feedforward the error is large (tens of centimeters on a 6 m course): the cable feedforward matters most. Keep the attitude loop faster than the cable loop, and the cable loop faster than the position loop.

4. **Linearize and validate** (`ch07-variation-lqr-design`).
   State `s = (dx_L, dv_L, xi, domega, eta, dOmega)`, input `(df, dtau)`. Cable block: the pendulum block (thm7-s2) for mass `m_Q` and force `m_Q g e3 - F`. Attitude block: thm7-so3. Coupling: `dF = R e3 df - f R hat(e3) eta`. Load row:
   ```
   (m_Q + m_L) dv_L_dot = q ((q x F) . xi) - c q x xi - 2 m_Q l (omega . domega) q + q (q . dF),   c = q . F - m_Q l |omega|^2
   ```
   Constraints: `q . xi = 0`, `(q x omega) . xi + q . domega = 0`.
   Validate three ways: (a) against geomech `linearize` on the constraint subspace, projecting the S^2 rows by `P = I - q q^T` and adding `(xi x q)(q . E_xi)` for the variation of the projection; (b) finite differences of perturbed open-loop flights on the manifold, error falling 100 times when eps falls 10 times; (c) reachable dimension 16 on `ker C` (build the Krylov space by orthogonalization, not by powers of `A`).
   LQR: integrate the Riccati equation along the reference, feed `(e_x, e_v, e_q, omega - omegad, e_R, e_Omega + Omegad x e_R)`. Expect faster recovery than the cascade from small errors and failure from large swings (about 50° in the worked case).

5. **Uncertainty** (`ch08-adaptive-safety-design`).
   Lumped force on the thrust vector: `Delta = F_e - (m_L - m_L0)(x_L_ddot + g e3)`. Without adaptation the load sags by `(m_L - m_L0) g / k_x`.
   Predictor for `v_L` and `omega` driven by `F + Deltahat`, adaptation direction `y = -(q q^T) e_v_tilde + l e_omega_tilde x q`, projection, low-pass, and `F_d <- F_d - Delta_f`.
   No adaptation is needed on SO(3) when the cable is at the center of mass.
   Checks: `Delta_f` converges to the true `Delta` in hover; prediction error shrinks as `k_Delta` grows, up to the stability limit of the sampled update (`k_Delta h^2` small against the believed total mass).
   A force on the load itself is not matched.

6. **Safety** (`ch08-adaptive-safety-design`).
   Cone: `H = q . q_c - cos(theta_max)`, `H_dot = omega . (q x q_c)`, `H_ddot = -|omega|^2 (q . q_c) - (I - q q^T) q_c . F / (m_Q l)`. Enforce `H_ddot + (alpha_1 + alpha_2) H_dot + alpha_1 alpha_2 H >= 0`.
   Tension: `q . F <= m_Q l |omega|^2 - (m_Q + m_L0) T_min / m_L0`.
   Filter: the `F` closest to `F_d` satisfying both (QP, 3 unknowns, 2 constraints). Then enforce the tension exactly as a floor on the thrust: `f >= ((m_Q + m_L0) T_min / m_L0 - m_Q l |omega|^2) / (-q . R e3)` when `q . R e3 < 0`.
   Checks: a plan that leaves the cone is held near the edge (expect a fraction of a degree of overshoot from the attitude loop and input limits); a descent faster than free fall keeps `T >= T_min`.
   The cone guarantee is for the thrust vector as input. The attitude loop delivers it late.

7. **Estimate** (`ch14-invariant-filter-design`, Part B).
   Variation-based EKF on S^2 for `(qhat, omegahat)`, model = cable equation with the applied `f R e3 + Delta_f`, reading `y = l q + n_y`, `C_y = [-l hat(qhat), 0]`, `ytilde = l qhat - y`, correction by the map `(+)`, covariance moved and projected by `Pi_C`. Load state: `x_L = x_Q + l qhat`, `v_L = v_Q + l omegahat x qhat`.
   Checks: `|qhat| = 1`, `qhat . omegahat = 0`, `C Sigma = 0` at rounding level; convergence from 40° wrong within a few readings; NEES near 1 in the four constrained dimensions.

8. **Fly it all, then ablate.**
   Report tracking error, set-down error and cable angle, clearance, smallest `T` and `H`, estimation error, input peaks against limits, manifold residual. Remove one block at a time: flat feedforward, adaptation, filter, estimate, and the cascade itself (a controller on the bare quadrotor).

## Checks that must pass

- `check_eom < 1e-5`; linearization validated three ways (step 4).
- Plan feasible on paper with margin; open-loop error at rounding level.
- Full flight: `H > 0`, `T > T_min`, no saturation, manifold residual below `1e-12`.
- Ablation shows what each block buys. A block that changes nothing is reported as such.

## Failure modes

- Flying the vehicle along the load's course one cable length higher: the load lags and arrives swinging.
- Dropping `omegad`: the cable law's damping fights the reference. A faster attitude loop does not fix it.
- A course whose waypoint times make it double back.
- Comparing geomech's S^2 rows without the variation of the projection: differences of order one.
- Rank tests on powers of `A`: badly scaled, wrong rank.
- `k_Delta` too large for the sampling period.
- Filter plus missing feedforward on an infeasible plan: the composed loop can diverge although each block passed alone.
- Trusting the filter in place of a plan: it keeps the cable safe and gives up tracking.
- A tension floor computed with a believed mass larger than the true one delivers less than `T_min`.

## What is not guaranteed

Stability results exist per loop (attitude, cable direction, load position) and for the barrier with the thrust vector as input. Nothing here proves stability or safety of the composed system, of the adaptation law, or with the estimator in the loop. Treat the full flight as evidence and keep margins.
