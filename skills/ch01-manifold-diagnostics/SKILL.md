---
name: ch01-manifold-diagnostics
description: Audit an attitude or direction controller for Euler-angle singularities, frame-dependent feedback and quaternion unwinding, and replace its error terms with errors defined on SO(3) or S^2. Use when reviewing or debugging an existing attitude, thrust-direction or cable-direction controller, or before flying one through large angles.
---

# Diagnose a controller's attitude representation

Follows Chapter 1. Notation: `R` in SO(3) maps body to world, `R_dot = R hat(Omega)` with `Omega` in the body frame; `q` in S^2 with `q_dot = omega x q`; unit quaternion `(q0, qv)`.

## When to use

- A controller subtracts Euler angles, or feeds back on Euler-angle rates.
- A controller feeds back on the vector part of a quaternion.
- A vehicle behaves differently after the world frame or the mocap origin was changed.
- A vehicle rotates the long way round after a large disturbance, or chatters near 180° of error.

## Procedure

1. **Find the representation.** List every place the attitude or direction enters the feedback law, and what is subtracted from what.

2. **Three numbers: look for the singularity.** For yaw-pitch-roll, `Omega = E(theta, phi) eta_dot` and `det E = -cos(theta)`. Any `1/cos` or `tan` of an angle in the kinematics or in the mass matrix is the same defect. Evaluate the condition number of `E` along the intended trajectories. If the pitch (or the corresponding angle for another order) can approach 90°, the representation must go. Changing the angle order moves the singularity and does not remove it.

3. **Three numbers: test invariance.** Pick `R`, `Rd` and a world-frame change `Q`. Compute the commanded torque for `(R, Rd)` and for `(Q R, Q Rd)`. If they differ, the law depends on the choice of world frame. `eta - eta_d` always fails this test.

4. **Four numbers: test for unwinding.** If the law is `tau = -kp qv - kd Omega`, simulate from one physical attitude with both quaternion signs. If the angle turned differs, the law unwinds. If the law uses `sign(q0)`, it is discontinuous at 180° of error: test with measurement noise at 179° and look for chatter. The remedy within quaternions is a hysteresis switch, which makes the controller hybrid.

5. **Set expectations.** No continuous time-invariant feedback is globally asymptotically stabilizing on SO(3) or S^2. Reject any claim of global stability from a continuous law. The achievable target is almost-global: one or more extra equilibria, all unstable, placed far from the target.

6. **Replace the error terms.**

   On SO(3):

   ```
   Psi_R   = 0.5 * trace(I - Rd^T R)
   e_R     = 0.5 * vee(Rd^T R - R^T Rd)
   e_Omega = Omega - R^T Rd Omegad
   ```

   On S^2 (fixed `qd`):

   ```
   Psi_q = 1 - q . qd
   e_q   = qd x q
   ```

   If the relative rotation `Rd^T R` is a turn by `theta` about `u`, then `Psi_R = 1 - cos(theta)` and `e_R = sin(theta) u`. Likewise `Psi_q = 1 - cos(theta)` and `|e_q| = sin(theta)`.

7. **Integrate on the manifold.** Advance attitudes with `R <- R expm(h hat(Omega))`, never with `R <- R + h R_dot`.

## Checks that must pass

| Check | Pass condition |
| :-- | :-- |
| Invariance | `e_R(Q R, Q Rd) == e_R(R, Rd)` to machine precision for random `Q` |
| Angle form | `Psi_R == 1 - cos(theta)` and `norm(e_R) == sin(theta)` |
| Compatibility | finite-difference `d/dt Psi_R` equals `e_R . e_Omega` to `1e-7` |
| Tangency on S^2 | `e_q . q == 0` |
| Sphere compatibility | finite-difference `d/dt Psi_q` equals `e_q . omega` to `1e-7` |
| Bad point | `e_R == 0` only at `R == Rd` and at half turns, where `Psi_R == 2` |
| On-manifold | `norm(R^T R - I) < 1e-10` after a long simulation |

`verify/ch01.py` implements all of these and is the reference.

## Failure modes

- **The restoring torque fades near 180°.** `norm(e_R) = sin(theta)` shrinks past 90° and is zero at a half turn. A vehicle flipped nearly upside down recovers slowly. If large-angle recovery matters, use an error function whose slope does not vanish there (Chapter 6).
- **Subtracting angular velocities directly.** `Omega - Omegad` compares vectors expressed in different frames. Always transport first: `R^T Rd Omegad`.
- **Renormalising instead of integrating properly.** Projecting `R` back to SO(3) after an Euler step hides the error and biases the motion.
- **Trusting noiseless simulation.** The `sign(q0)` patch looks perfect without noise.
