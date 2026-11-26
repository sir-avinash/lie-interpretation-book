---
name: ch19-collaborative-transport
description: Take a team of quadrotors that carries one rigid payload on cables, or a flexible cable between them, through the whole recipe (manifold, model with a symbolic check, grasp map and tension distribution, flat plan, tracking cascade, variation-based LQR, pair constraints and separation barriers, integration on the manifold), and measure what each block buys with an ablation table. Use when modeling, planning or controlling several aerial vehicles joined by cables, when choosing how a payload wrench is shared among cables, or when a team that tracks well must also keep its vehicles apart and its cables taut.
---

# Several quadrotors, one payload

## When to use

- `n` quadrotors carry a rigid payload on `n` cables, each attached at the vehicle's center of mass and at a known point `r_i` of the payload.
- Two or more quadrotors carry a flexible cable or hose, modeled as a chain of links.
- Not for a single vehicle with a cable-suspended load (`ch18-cable-suspended-load`), and not when a cable is expected to go slack: every formula below assumes `T_i > 0`.

## Conventions (fixed)

- World frame with `e3` up, gravity along `-e3`. `R` maps body to world, `R_dot = R hat(Omega)`.
- Cable direction `q_i` points from quadrotor `i` to its attachment on the payload: `x_i = x_L + R_L r_i - l_i q_i`. Tension `T_i > 0` when the cable pulls. `T_i q_i` is the tension vector.
- Chain: joint 0 at a quadrotor, link `i` from joint `i - 1` to joint `i`, `x_i = x_{i-1} + l_i q_i`.
- Inputs `u = (f, tau)` per vehicle. Thrust vector `F_i = f_i R_i e3`.
- Variations `xi_i` on each `S^2`, `eta` on each `SO(3)`. Gains `k_x`, `k_v`, `k_R`, `k_Omega`, `k_q`, `k_omega`. Barrier function `H_ij`, rates `alpha_1`, `alpha_2`, smallest distance `r_min`.
- New here: grasp map `G_L`, cable forces `lam_i = -R_L^T T_i q_i` (payload frame), payload torque `tau_L`.

## Procedure

Do the steps in order. Do not start a step before the check of the previous one passes.

1. **Manifold** (`ch02-manifold-kinematics`). `SE(3) x (S^2 x SO(3))^n`, dimension `6 + 5n`, with `4n` inputs. A cable of `n` links between `n_Q` quadrotors: `R^3 x (S^2)^n x SO(3)^{n_Q}`, dimension `3 (n_Q + 1) + 2n`.
   Check: `robot.dim`, and `robot.on_manifold(state)` at rounding level for a random state.

2. **Model, derived twice** (`ch04-manifold-dynamics-modeling`, `ch05-geomech-symbolic-derivation`). Take `multi_quadrotor_payload` or `quadrotor_cable` from `case_studies/robots`. Derive it again with geomech from `robot.lagrangian()`.
   Check: `check_eom < 1e-5` for every `n` you use. Tensions: `T_i = q_i . (m_i (a_i + g e3) - f_i R_i e3)`, and the payload obeys `m_L (x_L_ddot + g e3) = -sum_i T_i q_i`.

3. **Grasp map.** `G_L = [[I ... I], [hat(r_1) ... hat(r_n)]]`, `6 x 3n`, with `G_L lam = (R_L^T F_L, tau_L)`.
   Check: `rank G_L`. Two cables give rank 5: the payload turns freely about the line through the attachments, and no controller below changes that. Three or more attachments not on a line give rank 6 and `3n - 6` internal forces (`ker G_L`).

4. **Distribute the tensions.** `lam = pinv(G_L) W + N c`, `N` an orthonormal basis of `ker G_L`. Choose `c` by a small QP: stay close to the planned `lam`, subject to `e3 . lam_i >= T_min` and, for each pair, `|r_i - r_j| + e_ij . (l_i lam_i / T_i - l_j lam_j / T_j) >= d_min` with `e_ij` the unit vector from attachment `j` to attachment `i`.
   Check: `G_L lam = W` to rounding error; every `e3 . lam_i > 0`; the minimum-norm solution in hover has vertical cables, so compare the distance between attachments with `r_min` before trusting it.

5. **Plan in flat outputs** (`ch10-flatness-trajectory-design`). Flat output: payload pose, one heading per vehicle, and `3n - 6` internal-force coordinates. Per cable: `T_i q_i = -R_L lam_i`, `x_i = x_L + R_L r_i - l_i q_i`, `f_i R_i e3 = m_i (x_i_ddot + g e3) + R_L lam_i`, then the quadrotor's map. The payload pose needs six derivatives and the internal force four.
   Check: flat state and inputs satisfy the model to `1e-10`; open loop over a few seconds is fourth order in `h`; planned thrusts inside the limits; planned tensions above `T_min`; planned distances against `r_min`.

6. **Tracking cascade** (`ch06-geometric-tracking-control`). Payload wrench: `F_L = m_L (-k_x e_x - k_v e_v + g e3 + x_d_ddot)` and the attitude law for `(R_L, J_L)`. Then step 4, then `q_i,d = -R_L lam_i / |lam_i|`, the sphere law for each cable, the thrust vector
   `F_i = m_i (a_att,i + g e3 + l_i |omega_i|^2 q_i - l_i omega_i_dot x q_i) + (q_i q_i^T) R_L lam_i`
   with `a_att,i` the commanded acceleration of the attachment, and the attitude law for each vehicle with `Rd` from `F_i`.
   Loop speeds: payload position and attitude slowest (2 rad/s in the chapter), cables faster (7 rad/s), vehicle attitudes fastest (16 to 22 rad/s).
   Check: on the reference the cascade returns the flat inputs; on a plan that respects every constraint the payload tracks to millimeters. If raising one gain loses tracking, speed up the loop inside it.

7. **Linearize and compare** (`ch07-variation-lqr-design`). Linear state: `(dx_L, xi_i, eta_L, eta_i)` and their velocity variations, with two constraints per cable. Take differences of the model along variations in `ker C`, or use geomech's `linearize`.
   Check: controllable on `ker C` (rank equals `2 (6 + 5n) `). Against geomech: equal at states at rest. At moving states compare tangent parts only, and add `(xi x q)(q . E_xi)` to the tangent part of the residual `M_lin s_dot - A_lin s - B_lin du` of each direction (that residual is `-(xi x q)(q . E_xi)`), with `E_xi = M v_dot + b + G u` from geomech's standard form: geomech varies `E_xi` as printed, and the equation of motion is only its tangent part (same term as in the `ch18-cable-suspended-load` skill). Nothing is fitted. Along a moving reference use the linearization you have tested against perturbed simulations.

8. **Keep apart, keep taut** (`ch08-adaptive-safety-design`). After the cascade, filter the thrust vectors: smallest change such that `H_ij_ddot + (alpha_1 + alpha_2) H_ij_dot + alpha_1 alpha_2 H_ij >= 0` for `H_ij = |x_i - x_j|^2 - r_min^2`, and `T_i >= T_min`. Accelerations and tensions are affine in the thrust vectors: read both maps from the model.
   Use both layers. The pair constraint of step 4 moves the cables in `ker G_L` and costs the payload nothing, but it is kinematic and overshoots when it first becomes active. The barrier sees velocities and catches the overshoot, at the price of a wrench error.
   Check: `H_ij >= 0` and `T_i >= T_min` over the flight; the QP feasible at every step; the model's `H_ij_ddot` equal to a finite difference.

9. **Integrate on the manifold** (`ch12-variational-integrator-design` for long runs). Use the library's `step`.
   Check: manifold residual at rounding level; with the motors off, energy drift falling like `h^4`.

10. **Ablate.** Fly the mission with everything on. Then remove one block at a time: minimum-norm tensions, no pair constraint, no barrier, no flat feedforward, a weak vehicle, a cable of the wrong length, an unknown push. Tabulate payload error, payload tilt, smallest tension, smallest distance, smallest `H_ij`, peak thrust.

## Failure modes

- **Rank 5.** With two cables one rotation of the payload is uncontrollable. Count before designing.
- **Minimum norm.** It minimizes tension, not risk: in hover it hangs every vehicle straight above its attachment.
- **Loop speeds.** In the chapter's mission a payload attitude loop at 6 rad/s under a cable loop at 7 rad/s loses tracking, and 4 rad/s is marginal. A cable loop at 14 rad/s under vehicle loops at 16 to 22 rad/s loses tracking, and 10 rad/s works.
- **A barrier cannot see what the model lacks.** An unknown push on a vehicle is invisible until the distance has changed. Telling the filter about the push helps and does not suffice: the thrust must first be aimed.
- **Tuned for three is not tuned for four.** More vehicles on the same payload sit closer. Re-plan the spread for each `n`.
- **Slack.** `T_i <= 0` anywhere invalidates the model, the flat map and the linearization at once.
- **Centralized.** Every step above uses the state of the whole team.

## Reference

Chapter 19 of *A Manifold Introduction to Robotics and Control*; `verify/ch19.py` holds a tested implementation of every step, and `notebooks/ch19_case_transport.ipynb` a self-contained one.
