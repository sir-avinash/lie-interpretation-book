---
name: ch10-flatness-trajectory-design
description: Use differential flatness to design trajectories and feedforward for an underactuated system whose state contains attitudes in SO(3) or directions in S^2, such as a quadrotor, a quadrotor with a cable-suspended load, or quadrotors carrying a cable of several links. Gives the procedure for finding and checking a flat output, the coordinate-free flat maps with the number of derivatives each needs, a minimum-snap (or higher) spline through waypoints as one linear solve, the feasibility and singularity checks, and the open-loop and closed-loop validation. Use when asked for a dynamically feasible reference with its inputs, for the feedforward Omegad and Omegad_dot of a tracking controller, for an initial guess of a trajectory optimizer, or when a planned path makes a vehicle or a cable misbehave.
---

# Trajectory design with flat outputs

## When to use

- You need a full reference (state and input as functions of time) for a system with fewer inputs than degrees of freedom, and you want it without integrating or iterating.
- You need the feedforward of a tracking controller (`Rd`, `Omegad`, `Omegad_dot`, `f_d`, `tau_d`), or a dynamically consistent initial guess for an optimizer.
- Do not use it as a controller: it has no feedback (use the tracking, robust or MPC skills). Do not use it for input bounds as constraints (check after the fact, or optimize). Do not use it through a slack cable or through zero thrust.

## Conventions

World frame with `e3` up, gravity along `-e3`. `R_dot = R hat(Omega)`, `Omega` in the body frame. Cable direction `q` from the quadrotor to the load, `x_Q = x_L - l q`, `q_dot = omega x q`. Thrust `f` along `R e3`, torque `tau`. `x^(k)` is the k-th time derivative. The heading `psi` is the angle for which `R e2` is perpendicular to `Rz(psi) e1`.

## Procedure

1. **Count.** A flat output has as many components as there are inputs. Quadrotor: 4. Quadrotor with a cable-suspended load: 4. Cable of `n` links with `n_Q` quadrotors: `4 n_Q`.

2. **Find a candidate.** There is no general recipe. Look for the point farthest from the actuators whose free-body equation contains a single unknown force. For a mass on a cable it is the mass. For a rigid body swung by a force at a pivot it is the center of oscillation, at `l = |rho| + J/(m |rho|)` from the pivot. Add one heading per vehicle. For each extra vehicle on the same structure add the tension vector of one adjacent link.

3. **Write the map by climbing from the output to the actuators.** Each step is algebra on a free-body equation.

   Quadrotor, flat output `(x, psi)`:
   ```
   F  = m (x_ddot + g e3)                  # = f R e3
   f  = |F|
   R e3 = F / f
   R e2 = (R e3 x Rz(psi) e1) / |R e3 x Rz(psi) e1|
   R e1 = R e2 x R e3
   f_dot  = R e3 . F_dot
   Omega_1 = -(R e2 . F_dot) / f
   Omega_2 =  (R e1 . F_dot) / f
   Omega_3 = (Omega_1 (R e3 . Rz(psi) e1) + psi_dot (R e2 . Rz(psi) e2)) / (R e1 . Rz(psi) e1)
   Omega_1_dot = -(R e2 . F_ddot + 2 f_dot Omega_1) / f + Omega_2 Omega_3
   Omega_2_dot =  (R e1 . F_ddot - 2 f_dot Omega_2) / f - Omega_1 Omega_3
   Omega_3_dot = time derivative of the Omega_3 line (brings in psi_ddot)
   tau = J Omega_dot + Omega x J Omega
   ```
   with `F_dot = m x^(3)`, `F_ddot = m x^(4)`. Needs `x` up to order 4 and `psi` up to order 2.

   Quadrotor with a cable-suspended load, flat output `(x_L, psi)`:
   ```
   T = m_L |x_L_ddot + g e3|
   q = -(x_L_ddot + g e3) / |x_L_ddot + g e3|
   x_Q = x_L - l q
   omega = q x q_dot
   F = m_Q (x_Q_ddot + g e3) + m_L (x_L_ddot + g e3)      # = f R e3
   ```
   then the quadrotor lines with this `F`. Needs `x_L` up to order 6: `q, T, x_Q` order 2, `omega` 3, `f, R` 4, `Omega` 5, `tau` 6.

   Cable of `n` links, joint 0 the quadrotor, joint `n` the free end, link `i` from joint `i - 1` to joint `i`, flat output `(x_n, psi)`:
   ```
   for i = n, ..., 1:
       T_i q_i = T_{i+1} q_{i+1} - m_i (x_i_ddot + g e3)     # T_{n+1} = 0
       x_{i-1} = x_i - l_i q_i
   F = (m_Q + m_0)(x_0_ddot + g e3) - T_1 q_1
   ```
   Needs `x_n` up to order `2 n + 4`. A further quadrotor at joint `k` adds its heading and the tension vector `T_k q_k` as flat outputs (order `2 k + 2`), and its thrust vector is `m_k (x_k_ddot + g e3) + T_k q_k - T_{k+1} q_{k+1}`.

4. **Differentiate exactly.** Never difference a sampled curve six times. Either use the closed forms above, or carry every quantity as a truncated Taylor series (products by the Cauchy product, quotients, square roots and unit vectors by their recurrences) and read the derivatives off the coefficients.

5. **Check the map against the model, twice.**
   - Algebra: at 20 random times on random smooth curves, put the recovered state and input into the equations of motion. The residual must be at rounding level (1e-10 or better).
   - Simulation: integrate the model open loop with the recovered input, evaluated at each stage of the integrator, not held over the step. The flat output must be recovered to integration accuracy, and the error must fall with the order of the integrator when the step is halved.
   - Ladder: perturb one derivative of the flat output at a time and confirm that each quantity depends on exactly the orders you counted.

6. **Draw the curve.** The state must be continuous: the flat output needs continuous derivatives up to the order of the last state (3 for the quadrotor, 5 for the load, `2 n + 3` for `n` links). Minimize the next derivative, where the torque appears. Through waypoints at times `t_0 < ... < t_f`:
   ```
   minimize   integral of |x^(r)|^2        r = 4 quadrotor (snap), r = 6 load
   subject to x(t_i) given, x^(1..r-1) = 0 at t_0 and t_f,
              continuity of x^(0..r-1) at interior waypoints
   ```
   with one polynomial of degree `2 r - 1` per interval and per axis. Cost `c^T P c`, constraints `A c = b`: solve the KKT system `[[2 P, A^T], [A, 0]] [c; nu_r] = [0; b]`. Use time normalized to each interval. Sinusoids are the other safe choice.

7. **Check feasibility on the map.** Along the curve compute `f`, tilt `acos(R e3 . e3)`, `|Omega|`, `|tau|`, every tension `T`. Compare with the actuators. If a bound is violated, stretch the time (the k-th derivative scales with the k-th power of the speed) or move a waypoint, and repeat.

8. **Check the singularities.** Require a margin on `f > 0`, on every `T > 0`, and on `R e1 . Rz(psi) e1 != 0`. Near `f = 0` the angular velocity grows like `|F_dot| / f`.

9. **Use it.** Feedforward for a tracker: `Rd`, `Omegad`, `Omegad_dot`, `f_d`, `tau_d` from the map along the reference. Reference with input for a linearization or an MPC. Initial guess for a transcription: its dynamics residual is zero.

10. **Validate closed loop.** Start with an offset and confirm convergence. The flat feedforward of the reference makes zero error a solution of the closed loop. It does not contain the derivatives of the feedback part of `F_d`, so the attitude loop must still be faster than the position loop.

## Checks that must pass

1. Residual of the flat map in the model below 1e-10 at 20 random points, for every system and every size used.
2. Open-loop simulation recovers the flat output, with the error falling at the order of the integrator.
3. Derivative ladder as counted: 2, 3, 4 for the quadrotor's `(f, R)`, `Omega`, `tau`; 2, 3, 4, 5, 6 for the load's `(q, T, x_Q)`, `omega`, `(f, R)`, `Omega`, `tau`; `2 n + 4` for the torque under `n` links.
4. `T` from the map equals the model's tension formula. For the load: `T = m_L/(m_Q + m_L) (m_Q l |q_dot|^2 - q . f R e3)`.
5. Spline: passes through the waypoints, rest conditions met, the KKT solution equals the solution of the square system that imposes continuity up to order `2 r - 2`, and random feasible perturbations raise the cost.
6. Thrust, tilt, torque and all tensions inside their bounds along the whole curve, with margin.
7. Closed loop from an offset converges with the flat feedforward. If it diverges, scale the attitude gains by `(kappa^2, kappa)` with `kappa > 1`, or add the derivatives of the feedback part of `F_d`.

## Failure modes

- **Treating the feedforward as a controller.** Open loop, a tilt error of 1° at release puts a 4.34 kg quadrotor 2 m off after 5 s, and a 1% mass error 1.2 m.
- **Flat feedforward with a slow attitude loop.** With the published gains `k_x = 16 m`, `k_v = 5.6 m`, `k_R = 8.81` N·m, `k_Omega = 2.54` N·m·s the cascade diverges with the flat feedforward of the reference as it does with none. With `kappa = 1.5` it converges to zero error, where zero feedforward leaves 1.9 cm.
- **A curve that is not smooth enough.** A minimum-snap curve used for the load is at rest only up to the jerk: the map then starts the vehicle tilted and turning. A piecewise path makes the generated state jump at the corners.
- **Noise in the flat output.** A 1 mm ripple at 2 Hz on a load path multiplies the torque by more than 50. Build the curve from polynomials or sinusoids, never from a filtered measurement.
- **Zero thrust.** `F = 0` is free fall. Past it the map returns an inverted vehicle. A load whirled fast enough reaches it before the cable goes slack, through the centripetal term `m_Q l |q_dot|^2`.
- **Zero tension.** `x_L_ddot = -g e3`. The model is no longer valid: the cable is slack and the system is hybrid.
- **Heading singularity.** Thrust axis along `Rz(psi) e1`. Choose another `psi`.
- **Input limits.** They are not in the map. Check them (step 7) or optimize with them.
- **Wrong sign of `q`.** With `q` from the quadrotor to the load, `-T q = m_L (x_L_ddot + g e3)`. The opposite sign puts the quadrotor below the load.
- **Assuming flatness.** Not every system is flat. A cart with a pendulum and one input is not. A cable attached away from the center of mass changes the map. Re-derive and re-check (step 5) whenever the model changes.
