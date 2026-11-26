---
name: ch17-quadrotor-end-to-end
description: Take one vehicle through the whole recipe in order (manifold, errors, model with a symbolic check, flat plan, tracking cascade, optional LQR or error-state MPC, L1 adaptive control, safety filter, invariant EKF, integration on the manifold), connect the blocks in one simulation, and measure what each block buys with an ablation table. Use when assembling a full flight stack for a quadrotor or another vehicle from the per-chapter skills, or when a stack built from verified blocks misbehaves once the blocks are connected.
---

# One vehicle, end to end

## When to use

- You have, or can build, each block from the per-chapter skills in `skills/`, and need them to work together in closed loop on an estimated state.
- A stack whose blocks pass their own tests fails when connected (typical: adaptation and a safety filter fight each other).
- Not a substitute for the block skills. Each step below names the skill that holds the details.

## Conventions (fixed)

- `R` maps body to world, `R_dot = R hat(Omega)`. Errors take the actual configuration first: `e_R(R, Rd)`, `e_Omega(R, Omega, Rd, Omegad)`.
- Input `u = (f, tau)`. What the tracker proposes is `(f_d, tau_d)`. What reaches the rotors is `(f, tau)`. The override is `dtau = tau - tau_d`.
- Estimates carry a hat appended: `Rhat`, `vhat`, `xhat`. Covariance `Sigma`, innovation `ytilde`, Kalman gain `Ktilde`.
- Gains `k_x`, `k_v`, `k_R`, `k_Omega`, adaptation gain `k_Delta`, cutoff `k_f`, ball `Delta_max`. Barrier function `H`, rates `alpha_1`, `alpha_2`. Time step `h` is also the sampling period.

## Procedure

Do the steps in order. Do not start a step before the check of the previous one passes.

1. **Manifold and state** (`ch02-manifold-kinematics`). Write the configuration as a product of R^n, S^2 and SO(3) factors. Store rotation matrices and unit vectors, not charts.
   Check: `on_manifold(state)` at rounding level for a random state.

2. **Errors** (`ch03-geometric-error-design`). Take `Psi_R = 0.5 tr(I - Rd^T R)`, `e_R = 0.5 vee(Rd^T R - R^T Rd)`, `e_Omega = Omega - R^T Rd Omegad`. Use the same construction for tracking (vehicle against reference), prediction (state predictor against vehicle) and estimation (`Xhat X^-1`).
   Check: `|e_R| = sin(theta)` at a known error angle `theta`. Note the angle at which your mission starts: beyond 90° the restoring torque fades.

3. **Model, derived twice** (`ch04-manifold-dynamics-modeling`, `ch05-geomech-symbolic-derivation`). Take the model from `case_studies/robots` or write it with the inertia on the left. Derive it again with geomech from the Lagrangian.
   Check: `check_eom < 1e-5`, and hand-written accelerations equal the derived ones at 20 random states to `1e-9`.
   Write down separately what the controller believes (`m`, `J_0`) and what is true (`J`, center-of-mass offset `rho`, external force and torque). A thrust `f` that misses the center of mass by `rho` adds the body torque `f e3 x rho`.

4. **Plan in flat outputs** (`ch10-flatness-trajectory-design`). Minimum snap through the waypoints, rest at both ends. Read `f_d`, tilt, `Omegad`, `tau_d` from the flat map.
   Check: thrust and torque inside the limits with margin for feedback and for the safety filter; flat-map inputs fed open loop to the believed model arrive at the target (error at integrator level). At the two ends of the curve only the snap jumps: evaluate the one-sided limit there, not zero.

5. **Tracking cascade** (`ch06-geometric-tracking-control`). `F_d = -k_x e_x - k_v e_v + m g e3 + m xd_ddot`, `f = F_d . R e3` clipped at zero, `Rd` from `F_d` and the heading, then
   `tau = -k_R e_R - k_Omega e_Omega + Omega x J_0 Omega - J_0 (Omega x R^T Rd Omegad - R^T Rd Omegad_dot)`.
   Feedforward `Omegad`, `Omegad_dot`: from the flat map of the reference. Then the attitude loop must be faster than the position loop: scale the attitude gains by `(kappa^2, kappa)` with `kappa` about 2.
   Check, with nothing unknown and the true state: tracking error with flat feedforward several times smaller than with zero feedforward, and recovery from the worst initial attitude of the mission.

6. **Local optimality, optional** (`ch07-variation-lqr-design`, `ch13-manifold-mpc-design`). Compare the PD gains with an LQR gain for the linearized attitude dynamics. Use error-state MPC only where an input bound is active for a long time and the stage references are known over the horizon.
   Check: with bounds inactive the MPC torque equals the LQR torque.

7. **Uncertainty** (`ch08-adaptive-safety-design`). L1 adaptive control on the attitude loop: state predictor, adaptation law with projection, low-pass filter. Start the predictor at the estimate when the controller is switched on. Choose `Delta_max` above the largest torque you expect, `k_f` from the slowest actuator and the largest delay.
   Check, without the safety filter: filtered estimate `Delta_f` at hover equals the known offset torque; prediction error stays small.

8. **Safety filter** (`ch08-adaptive-safety-design`). Augmented barrier function for each obstacle, QP on `(f, tau)` with `H1_dot + alpha_2 H1 >= 0` and `0 <= f <= f_max`. Clip the torque after it.
   Check: `H1_dot = a + b_f f + b_tau . tau` against finite differences along the model; QP feasible at every step of a test flight.

9. **Connect adaptation and safety. Two wires, both or neither.**
   - Tell the predictor. After the safety filter and the actuator limits, compute `dtau = tau - tau_d` and add `Xi R_p^T R dtau` to the predictor's feedback torque `tau_fb_p`, with `Xi = J_0 R_p^T R J_0^-1 R^T R_p`.
   - Inform the barrier. In the barrier's model use `Omega_dot = J_0^-1 (tau + Delta_f - Omega x J_0 Omega)`.
   Check: prediction error with the safety filter active no larger than without it; smallest `H` over the flight nonnegative. If only the second wire is present the result is worse than with none.
   No theorem of the book covers the connected loop. Treat the result as a test, not a proof.

10. **Estimation** (`ch14-invariant-filter-design`). Right-invariant EKF on SE_2(3) with IMU propagation and landmark (or position) updates. It needs no model of the vehicle, so run it before the motors start.
    Check alone, against the truth: error converged; innovation weighted by its covariance near 1 per dimension; NEES near 1. Expect a NEES far above 1 during the transient from a large initial error: do not use the covariance as a release criterion until it has settled.
    Then close the loop: controller, predictor and safety filter read `Rhat`, `vhat`, `xhat`, and `Omega` from the latest gyroscope reading.
    Check: repeat the flight with the truth in the loop. The differences should be at the size of the estimation error.

11. **Integration** (`ch04-manifold-dynamics-modeling`, `ch12-variational-integrator-design` for long runs). One step = Runge-Kutta stages on increments, then `R_{k+1} = R_k expm_so3(h Omega_k)`. Same update for the estimate and the predictor.
    Check: manifold residual at rounding level over the whole mission; with the motors off, energy constant and its drift falling by 16 when `h` is halved.

12. **Ablation.** Fly the mission with everything on, then with one block changed at a time: truth for estimate, no feedforward, no adaptation, no safety filter, each wire removed, both removed, slower attitude gains. Report tracking error outside the detour, hover error, closest approach, smallest `H`, peak inputs against limits.
    Check: every row is either explained or investigated. A block that changes nothing is a finding. Report it.

## Failure modes

- **Predictor not told.** The safety filter overrides the torque, predictor and vehicle part, the estimate runs to the edge of its ball, and the tracker asks for a large torque that is released when the filter lets go. Symptom: prediction error jumps by orders of magnitude when the filter becomes active.
- **Barrier informed by an untold predictor.** `Delta_f` contains the filter's own override. The vehicle enters the obstacle.
- **Barrier not informed.** Its model lacks the unknown torque. `H` dips below zero by about (error in `H_ddot`) / (`alpha_1 alpha_2`).
- **Flat feedforward with a slow attitude loop.** Tracking error of the order of a meter. Raise `kappa` or differentiate the feedback too.
- **Recovery depends on the throw.** From near a half turn `|e_R|` is small. Test several initial attitudes and report the slowest. Almost-global stability promises no rate.
- **Torque limit unknown to the filter.** A clipped torque breaks the barrier condition for as long as it is clipped. Log how long.
- **Force disturbances.** `Delta` is a torque. A steady force `F` leaves a position error of about `F / k_x`.
- **One seed.** Repeat the table over many seeds of the sensor noise before any claim about reliability.

## Reference implementation

`verify/ch17.py` (`Mission`, `Plan`, `fly`, `summary`) and the notebook `notebooks/ch17_case_quadrotor.ipynb`, which is self-contained. Change the fields of `Mission` to change waypoints, obstacle, gust, package, sensors and limits.
