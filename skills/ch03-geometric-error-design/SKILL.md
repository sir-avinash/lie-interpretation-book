---
name: ch03-geometric-error-design
description: Design an error function, error vector, transport map and velocity error for a tracking problem whose state lives on SO(3), S^2, R^3 or a product of them, and verify the design by finite differences. Use when writing a geometric tracking controller, a Lyapunov or barrier function on a manifold, an invariant estimator, or the error state for a linearization or an MPC.
---

# Design a geometric tracking error

Follows Chapter 3. Notation: `R` in SO(3) maps body to world, `Rdot = R hat(Omega)`, `Omega` in the body frame. `q` in S^2, `qdot = omega x q`, `omega . q = 0`, so `omega = q x qdot`. Reference quantities carry the subscript `d`.

## When to use

- A controller needs proportional and derivative terms for an attitude, a direction, or a product of these.
- A velocity error is formed by subtracting two angular velocities measured in different frames.
- A linearization or MPC needs an error state that has no singularities.
- An estimator needs an error whose dynamics do not depend on the state.

## Procedure

1. **Factor the configuration space.** Write it as a product of R^3, S^2 and SO(3) factors. Design each factor separately. The total error function is a weighted sum of the factor error functions.

2. **Fix the variation on each factor.**

   ```
   R^3:    x_eps = x + eps dx                  dx free
   SO(3):  R_eps = R exp(eps hat(eta))         dR = R hat(eta)
   S^2:    q_eps = exp(eps hat(xi)) q          dq = xi x q,   xi . q = 0
   ```

   Velocity variations:

   ```
   SO(3):  dOmega = eta_dot + Omega x eta
   S^2:    d(qdot) = xi_dot x q + xi x qdot
           domega  = xi_dot - 2 omega x xi          (xi . q = 0 along the whole curve)
   ```

   Do not use `domega = xi_dot - omega x xi`: it is only the tangential part of `domega`.

3. **Choose the error function Psi.** It must be nonnegative and vanish only at the target.

   ```
   R^3:    Psi = 0.5 |x - xd|^2
   S^2:    Psi = 1 - q . qd
   SO(3):  Psi = trace(G (I - Rd^T R)),   G = diag(g1, g2, g3) > 0     (G = I/2 is the standard choice)
   ```

   Alternatives on SO(3), for a relative rotation `Rd^T R` by angle `theta` about `u`:

   ```
   trace:       Psi = 1 - cos(theta)            e = sin(theta) u            smooth, e -> 0 at 180 deg
   logarithm:   Psi = 0.5 theta^2               e = theta u = log(Rd^T R)   e jumps at 180 deg
   quaternion:  Psi = 4 (1 - cos(theta/2))      e = 2 sin(theta/2) u        e jumps at 180 deg
   ```

   No choice is smooth, never fading and free of extra critical points. State which property is given up.

4. **Derive the error vector as the slope of Psi.** Vary the first argument and read off `e` from `dPsi = e . (variation vector)`.

   ```
   R^3:    e_x = x - xd
   S^2:    e_q = qd x q
   SO(3):  e_R = vee(G Rd^T R - R^T Rd G)          (G = I/2:  e_R = 0.5 vee(Rd^T R - R^T Rd))
   ```

   Useful identity: `trace(A hat(eta)) = -vee(A - A^T) . eta`.

5. **Choose the transport map and the velocity error.**

   ```
   R^3:    e_v     = v - vd
   S^2:    e_omega = omega - (I - q q^T) omega_d         equivalently  e_qdot = qdot - (qd x qd_dot) x q = e_omega x q
   SO(3):  e_Omega = Omega - R^T Rd Omega_d
   ```

   On SO(3) the relative attitude obeys `d/dt (Rd^T R) = (Rd^T R) hat(e_Omega)`, so `e_Omega = 0` exactly when the relative attitude is constant.

6. **Error dynamics, if the design needs them.**

   ```
   SO(3), G = I/2:  e_R_dot = 0.5 (trace(R^T Rd) I - R^T Rd) e_Omega
   S^2:             e_q_dot = (q . qd) e_omega + (q . omega_d) q x e_q - (qd . omega) q
   ```

7. **On a matrix Lie group, pick the invariance.** With `Xdot = X hat(xi)`:

   ```
   left-invariant error   E = Xd^-1 X     unchanged by (X, Xd) -> (L X, L Xd)     change of world frame
   right-invariant error  E = X Xd^-1     unchanged by (X, Xd) -> (X L, Xd L)     change of body frame
   left error dynamics    E_dot = E hat(xi - Ad_{E^-1} xi_d)                      no X on the right-hand side
   Lie-algebra error      E = exp(hat(psi)),  psi_dot = -ad_{xi_d} psi + (xi - xi_d)   to first order
   ```

8. **Relate errors to variations for linearization.** For `R = Rd exp(eps hat(eta))`:

   ```
   e_R     = eps eta + O(eps^3)
   e_Omega = eps eta_dot + O(eps^2)
   eps dOmega = e_Omega + Omega_d x e_R + O(eps^2)
   ```

   On S^2: `e_q = eps xi + O(eps^3)`, `e_omega = eps (xi_dot - omega_d x xi) + O(eps^2)`.

## Checks that must pass

Run all of them numerically at random states before using the design. `verify/ch03.py` contains a reference implementation of each.

1. **Slope test.** For a random variation vector, the central difference of `Psi` along the varied configuration equals `e . (variation vector)` to 1e-7.
2. **Compatibility test.** Move both the state and the reference along arbitrary smooth curves. The central difference of `Psi` in time equals `e . e_vel` to 1e-7. Use a moving reference: a fixed reference does not test the transport map.
3. **Frozen-error test.** Move the state and the reference rigidly together. The velocity error must be zero. `Omega - Omega_d` fails this.
4. **Invariance test.** Apply a common change of world frame to state and reference. `Psi`, `e` and `e_vel` must not change.
5. **Tangency.** `e_q . q = 0` and `e_omega . q = 0`.
6. **Critical points.** Find every zero of `e`. Confirm the target is the only minimum and that every other zero is a saddle or a maximum of `Psi`. For `G` with distinct weights the zeros are the target and the three half turns about the body axes, with `Psi = 2(g2+g3), 2(g1+g3), 2(g1+g2)`.
7. **First-order test.** Halve `eps` repeatedly. `|e - eps (variation)|` must fall by a factor of eight.
8. **Variation test,** when a new velocity variation formula is used: build a family of curves, difference the velocity across the family, and compare.

## Failure modes

- **Subtracting velocities across frames.** `Omega - Omega_d` is nonzero for two bodies that move as one. With `G = I/2` it still passes the compatibility test, which hides the mistake. It fails with unequal weights and fails the frozen-error test.
- **Testing with a fixed reference.** Every transport map passes.
- **Fading restoring action.** With the trace error, the proportional action is `sin(theta)`. Recovery time from near 180 degrees grows without bound: `ln(10)/lambda` seconds per factor of ten in closeness, where `lambda` is the unstable closed-loop rate at 180 degrees.
- **Discontinuous error at 180 degrees.** Logarithm and quaternion errors flip sign there. With noise the command chatters unless a hysteresis is added.
- **Equal weights.** Every half turn is a critical point, a two-dimensional set. Use distinct weights if isolated critical points are needed.
- **Half-strength gyroscopic terms on S^2.** Caused by `domega = xi_dot - omega x xi`. Use `xi_dot - 2 omega x xi`.
- **Treating e_Omega as dOmega.** They differ by `Omega_d x e_R` at first order whenever the reference is turning.
- **Expecting the transport map to be parallel transport.** The S^2 map scales the component across the connecting great circle by `cos(theta)`. That is compatible and intended.
