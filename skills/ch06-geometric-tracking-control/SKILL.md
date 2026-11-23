---
name: ch06-geometric-tracking-control
description: Design, tune and validate a geometric tracking controller (PD with feedforward built from e_R, e_Omega, e_q, e_omega) for a system whose state contains attitudes in SO(3) or directions in S^2, including underactuated cascades such as a quadrotor or a quadrotor with a cable-suspended load. Gives the control laws, the Lyapunov function with its explicit gain condition and sublevel set, the equilibrium analysis behind an almost-global claim, and the simulation tests that must pass. Use when asked for an attitude or direction tracking controller, a stability proof or region of attraction for one, a position-attitude cascade, or when an Euler-angle controller fails at large errors.
---

# Geometric tracking control

## When to use

- The plant has a rigid body `R` in SO(3) with body torque, or a direction `q` in S^2 with a force or moment, possibly inside a cascade (position, direction, attitude).
- You need a statement about large errors: a region with exponential convergence, or an almost-global claim.
- Do not use it for optimal gains near a reference (use the variation-based LQR skill), or for model uncertainty, input limits and safety constraints (robust and safe control).

## Conventions

Actual before desired. `R_dot = R hat(Omega)`, `q_dot = omega x q` with `omega . q = 0`. Errors:

| Space | Error function | Error vector | Velocity error |
| :-- | :-- | :-- | :-- |
| SO(3) | `Psi_R = 1/2 tr(I - Rd^T R)` | `e_R = 1/2 vee(Rd^T R - R^T Rd)` | `e_Omega = Omega - R^T Rd Omegad` |
| S^2 | `Psi_q = 1 - q . qd` | `e_q = qd x q` | `e_omega = omega - (I - q q^T) omegad` |
| R^3 | `1/2 |x - xd|^2` | `e_x = x - xd` | `e_v = v - xd_dot` |

Identities used in every proof: `|e_R|^2 = Psi_R (2 - Psi_R)`, `e_R_dot = C_R e_Omega` with `C_R = 1/2 (tr(R^T Rd) I - R^T Rd)` and `|C_R|_2 <= 1`, `Psi_R_dot = e_R . e_Omega`. The same three hold on S^2 with `e_q`, `e_omega`.

## Procedure

1. **Write the model and check it.** Equations of motion in `R`, `Omega`, `q`, `omega`. Derive them a second time with geomech and require `check_eom < 1e-5`.

2. **Attitude loop.** For `J Omega_dot + Omega x J Omega = tau`:
   ```
   tau = -k_R e_R - k_Omega e_Omega + Omega x J Omega
         - J (Omega x (R^T Rd Omegad) - R^T Rd Omegad_dot)
   ```
   Closed loop: `J e_Omega_dot = -k_R e_R - k_Omega e_Omega`. It does not contain `Rd(t)`. Add any other known torque (gravity of a pendulum) to the cancellation.

3. **Direction loop.** For `q_dot = omega x q` with a controllable tangential acceleration, command
   ```
   omega_dot = (I - q q^T) omegad_dot - (q . omegad) q_dot - k_q e_q - k_omega e_omega
   ```
   For a pendulum with a force on the bob: `F = m g e3 + m l omega_dot x q`. Closed loop: `e_omega_dot = -k_q e_q - k_omega e_omega + (q_dot . omegad) q`.

4. **Cascade (thrust along one body axis).**
   ```
   F_d = -k_x e_x - k_v e_v + m g e3 + m xd_ddot
   f   = F_d . R e3
   Rd e3 = F_d / |F_d|
   Rd e2 = (Rd e3 x Rz(psi_d) e1) / |Rd e3 x Rz(psi_d) e1|
   Rd e1 = Rd e2 x Rd e3
   ```
   Then step 2 with this `Rd`. `Omegad`, `Omegad_dot` are derivatives of `Rd` along the closed loop: they need `F_d_dot`, `F_d_ddot`, which the model gives from the state (acceleration and jerk). The position error obeys `m e_v_dot = -k_x e_x - k_v e_v + Delta`, `|Delta| = |F_d| sin(tilt)`, and `|Delta| <= |F_d| |e_R|` when `Psi_R < 1`.

5. **Deeper cascade (cable-suspended load, `q` from quadrotor to load).**
   ```
   F_L = -k_x e_x - k_v e_v + (m_Q + m_L)(xd_ddot + g e3)
   A   = F_L + m_Q l |q_dot|^2 q
   qd  = -A / |A|
   F   = (A . q) q + m_Q l q x (-k_q e_q - k_omega e_omega)
   ```
   Feed `F` to step 4 in place of `F_d`. The load receives `(q q^T) F_L`. The cable obeys `omega_dot = -k_q e_q - k_omega e_omega`.

6. **Choose gains.** Per axis near the target: `J_i eta_ddot + k_Omega eta_dot + k_R eta = 0`. Place poles there, or take the gains from an LQR. Order the loops: position slowest, then direction, then attitude. If any feedforward (`Omegad`, `omegad`) is dropped, the inner loop must be faster than the outer by a clear factor. Scale attitude gains as `(kappa^2 k_R, kappa k_Omega)` to speed the loop up by `kappa`.

7. **State the guarantee.** With `V0 = 1/2 e_Omega . J e_Omega + k_R Psi_R`:
   - `V0_dot = -k_Omega |e_Omega|^2`. If `V0(0) < 2 k_R` then `Psi_R(t) <= Psi_max = V0(0)/k_R < 2` for all `t`.
   - For `0 < c < min( sqrt(k_R J_min)/J_max , 4 k_R k_Omega/(k_Omega^2 + 4 k_R J_max) )`, `V = V0 + c e_R . J e_Omega` satisfies `z^T W1 z <= V <= z^T W2 z`, `V_dot <= -z^T W3 z`, `z = (|e_R|, |e_Omega|)`:
     ```
     W1 = 1/2 [[k_R, -c J_max], [-c J_max, J_min]]
     W2 = 1/2 [[2 k_R/(2 - Psi_max), c J_max], [c J_max, J_max]]
     W3 = [[c k_R, -c k_Omega/2], [-c k_Omega/2, k_Omega - c J_max]]
     ```
     Rate: `gamma = eigmin(W3)/eigmax(W2)`. It is a floor, often orders of magnitude below the measured rate, and it goes to zero as `Psi_max -> 2`.
   - On S^2 with `|omegad| <= omega_max`: same with `J_min = J_max = 1`, `k_R -> k_q`, `c < min( sqrt(k_q), 4 k_q k_omega/((k_omega + omega_max)^2 + 4 k_q) )`, and `k_omega + omega_max` in the off-diagonal of `W3`.
   - Almost-global: claim it only with a weighted error function, `Psi_R = tr(D (I - Rd^T R))`, `e_R = vee(D Rd^T R - R^T Rd D)`, `D = diag(d1, d2, d3)` with distinct positive entries. Then there are four isolated equilibria, three of them unstable, and the starts that converge to those have measure zero. With equal weights state the sublevel-set result instead.
   - Cascade with exact feedforward and `F_d` never zero: attitude error is unaffected by the position loop, and all errors converge if `V0(0) < 2 k_R`. Without exact feedforward, say that stability rests on time-scale separation.

## Checks that must pass

1. Closed-loop identity: at 20 random states, `J e_Omega_dot + k_R e_R + k_Omega e_Omega` is zero to 1e-10, with `e_Omega_dot = Omega_dot + Omega x (R^T Rd Omegad) - R^T Rd Omegad_dot`.
2. Autonomy: the same initial `(Rd^T R, e_Omega)` against a fixed target and against a tumbling reference gives the same error history (1e-6).
3. `V0` never increases along any simulated run. If it does, a feedforward term or a sign is wrong.
4. The three matrix inequalities hold at a few thousand random states and gains, and `V(t) <= V(0) exp(-gamma t)` along a run.
5. Half turns: start exactly at `Psi_R = 2`, at rest. The body must stay. Start at 179°, 179.9°, 179.99°: the settling time must grow by `ln(10)/lambda` per decade, `J_i lambda^2 + k_Omega lambda - k_R = 0`.
6. Dense starts: random axes, angles from 5° to 179.999°, all converge.
7. Cascade: with exact feedforward the attitude error history equals the stand-alone attitude loop (1e-4). Then remove the feedforward and confirm the loop still tracks. If it diverges, raise `kappa`.
8. Record peak thrust and torque of the worst start and compare with the actuators.

## Failure modes

- **Naive velocity error.** `Omega - Omegad` instead of `e_Omega`. Passes at hover, fails check 2.
- **Missing transport term.** Leaving `J (Omega x R^T Rd Omegad)` out of the torque. `V0` then increases when the reference turns.
- **No feedforward and no separation.** `Omegad = 0` with attitude poles as slow as the position loop. The published gains `k_x = 16 m`, `k_v = 5.6 m`, `k_R = 8.81`, `k_Omega = 2.54` for `J = diag(0.0820, 0.0845, 0.1377)` track to 1e-10 m with exact feedforward and diverge without it. `kappa = 2` restores stability.
- **Quoting the Lyapunov rate as performance.** It is a lower bound for the whole sublevel set.
- **Claiming "global".** Impossible for a continuous law. Claiming "almost-global" from tests on a grid of round angles is also wrong: such grids land on half turns.
- **`F_d` near zero.** `Rd` is undefined. Bound the reference acceleration and the position error, or freeze `Rd`.
- **Negative thrust.** `F_d . R e3 < 0` beyond 90° of tilt. Clip at zero and expect altitude loss.
- **Slack cable.** The load model assumes tension. Monitor `T = m_L/(m_Q + m_L) (m_Q l |q_dot|^2 - q . f R e3) > 0`.
- **Lingering.** With the trace error the restoring torque is `k_R sin(theta)`. If recovery from near half a turn matters, switch to the logarithm error `log_so3(Rd^T R)` with the same gains, and add hysteresis at 180°.
- **Exact output linearization of `e_R`.** Needs `C_R^{-1}`, and `det C_R = 1/2 cos(theta)(1 + cos(theta))` vanishes at 90° and 180°. Linearize `e_Omega` instead, as in step 2.
