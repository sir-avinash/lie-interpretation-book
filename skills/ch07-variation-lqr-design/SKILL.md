---
name: ch07-variation-lqr-design
description: Design a variation-based LQR for a system whose state lives on SO(3), S^2 or a product of these with R^n. Obtain the linearization along a reference with geomech or by hand, check it by finite differences of perturbed trajectories, test controllability on the constraint subspace, solve the Riccati equation, implement the feedback with geometric errors, and measure the region of attraction. Use when asked for an LQR, a linearization, a time-varying gain or a "small-error model" of a rigid body, pendulum, cable-suspended load or similar system, or when an Euler-angle LQR misbehaves near a singularity.
---

# Variation-based LQR design

## When to use

- The plant state contains an attitude `R` in SO(3) or a direction `q` in S^2, and you need a linear model about an equilibrium or a trajectory.
- The reference passes near a chart singularity, or the target is one (nose-up hover, inverted pendulum).
- Do not use it to claim large-error or almost-global behavior. The result is first order.

## Conventions

| Manifold | Kinematics | Variation | Linear state | Constraint |
| :-- | :-- | :-- | :-- | :-- |
| SO(3) | `R_dot = R hat(Omega)`, `Omega` in the body frame | `dR = Rd hat(eta)` | `(eta, dOmega)`, `dOmega = Omega - Omegad` | none |
| S^2 | `q_dot = omega x q`, `omega . q = 0` | `dq = xi x qd` | `(xi, domega)`, `domega = omega - omegad` | `qd . xi = 0`, `qd . domega = xi . (omegad x qd)` |
| R^n | `x_dot = v` | `dx` | `(dx, dv)` | none |

Kinematic rows: `eta_dot = -Omegad x eta + dOmega` and `xi_dot = omegad x xi + (I - qd qd^T) domega`.

## Procedure

1. **Write the model.** Equations of motion in `R`, `Omega`, `q`, `omega`, with every input declared. State the reference `(xd(t), u_d(t))` and confirm it satisfies the equations (residual below 1e-8).

2. **Obtain `A(t)`, `B(t)`.**
   - With geomech: `eom = compute_eom(L, dOmega, variables)`, require `check_eom(...) < 1e-5`, then `lin = linearize(eom, variables, inputs)` and generate code with `module_source(lin, layout)`. geomech returns `M d(dx)/dt = A dx + B du`. Use `A = solve(M_lin, A_lin)`, `B = solve(M_lin, B_lin)`. `M` is the physical inertia (positive). If an S^2 block of `M_lin` is singular, project with the tangent basis first (step 4).
   - By hand: vary each equation with `d(R^T a) = hat(Rd^T a) eta`, `d(q) = xi x qd`, the product rule, and the kinematic rows above.
   - Rigid body `J Omega_dot + Omega x J Omega = m g (R^T e3) x rho + tau`:
     `A = [[-hat(Omegad), I], [-m g J^-1 hat(rho) hat(Rd^T e3), J^-1 (hat(J Omegad) - hat(Omegad) J)]]`, `B = [0; J^-1]`.
   - Spherical pendulum `m l^2 omega_dot = -m g l q x e3 + l q x F`:
     `A = [[hat(omegad), I - qd qd^T], [hat(Fd - m g e3) hat(qd) / (m l), 0]]`, `B = [0; hat(qd) / (m l)]`.

3. **Check the linearization by finite differences (mandatory).**
   - Simulate the nonlinear plant from the reference state and from two neighbors at `+eps` and `-eps`: `Rd exp(eps hat(eta0))`, `Omegad + eps dW0`, `exp(eps hat(xi0)) qd`, and `omega` re-projected onto the tangent plane of the perturbed `q`. Perturb the input by `eps du(t)`.
   - Read the centered difference as `eta = vee(Rd^T dR)`, `xi = qd x dq`, plain differences for velocities.
   - Integrate `s_dot = A(t) s + B(t) du(t)` from the same initial `s`. Require agreement that improves by 100 when `eps` shrinks by 10.
   - On S^2, compare two derivations only on the constraint subspace. 3x3 blocks that multiply `xi` or `domega` are not unique off it (geomech prints `q q^T hat(omega)` where the hand derivation gives `hat(omega)`).

4. **Reduce a constrained state.** For each S^2 factor build `C(t) = [[qd^T, 0], [(qd x omegad)^T, qd^T]]`. Check `C B = 0` and `C A + dC/dt = 0`. At an equilibrium take an orthonormal tangent basis `T` (3x2) of each S^2 factor, stack `Z = blkdiag(T, T, I, ...)`, and use `A_r = Z^T A Z`, `B_r = Z^T B`. Never pass an unreduced S^2 system, or a linearization in the nine entries of `R`, to an algebraic Riccati solver: its uncontrollable modes at zero make the solver fail.

5. **Test controllability where the state lives.**
   - Equilibrium: `rank ctrb(A_r, B_r) = dim A_r`. The unreduced rank equals `dim ker C`, not the full dimension. That is expected.
   - Moving reference: `rank [Gamma_0, Gamma_1, ...] = dim ker C(t)` with `Gamma_0 = B`, `Gamma_{k+1} = dGamma_k/dt - A Gamma_k`. The sign is minus. Confirm with the controllability Gramian restricted to `ker C(t_0)` if in doubt.

6. **Solve the Riccati equation.**
   - Equilibrium: `P = solve_continuous_are(A_r, B_r, Q, R)`, `K = R^-1 B_r^T P Z^T`.
   - Trajectory: integrate `-P_dot = A^T P + P A - P B R^-1 B^T P + Q` backwards from `P(t_f)`, tabulate `K(t) = R^-1 B(t)^T P(t)`. Check the residual of the differential equation on the stored grid, and that a constant `(A, B)` reproduces the algebraic solution.

7. **Implement the feedback with geometric errors.**
   - SO(3): `e_R = 0.5 vee(Rd^T R - R^T Rd)`, `e_Omega = Omega - R^T Rd Omegad`, and
     `tau = taud(t) - K(t) [e_R; e_Omega + Omegad x e_R]`.
     To first order `e_R = eta` and `e_Omega = dOmega - Omegad x eta = eta_dot`. The correction `Omegad x e_R` is required whenever the reference is turning. `Omega - Omegad` may be used in place of `e_Omega + Omegad x e_R`.
   - S^2 at an equilibrium: `e_q = qd x q`, `F = Fd - K [e_q; omega]`.
   - S^2 along a moving reference: `e_omega = omega - (I - q q^T) omegad` equals `(I - qd qd^T) domega` to first order, so feed `domega = e_omega + qd (e_q . (omegad x qd))`.
   - Never feed differences of Euler angles, quaternion components or matrix entries.

8. **Validate the closed loop on the nonlinear plant.**
   - Cost test: for a small initial error the measured cost divided by `s_0^T P(t_0) s_0` must tend to 1. A ratio near 2, or any ratio that does not tend to 1, means the fed-back state is not the design state (usually a missing `Omegad x e_R`).
   - Invariance test: for a free rigid body the closed-loop error history at a hold must not depend on the hold attitude.
   - Region of attraction: sweep the initial error angle to 180°, about several axes, and record settle time or failure. Repeat with input limits. Report the boundary as measured, not as guaranteed.

## Checks that must pass

- `check_eom < 1e-5` for a geomech model.
- Finite-difference mismatch of the linearization shrinks like `eps^2`.
- `C B = 0`, `C A + dC/dt = 0` for every S^2 factor.
- Controllability rank equals the dimension of the constraint subspace.
- Riccati residual small; algebraic and backward-integrated solutions agree for constant `(A, B)`.
- Closed-loop cost ratio tends to 1 for small errors.

## Failure modes

| Symptom | Cause | Action |
| :-- | :-- | :-- |
| Riccati solver raises "failed to find a finite solution" | unreduced S^2 state or embedded `R` entries | reduce to `ker C` (step 4) |
| Rank test says uncontrollable, rank equals `dim ker C` | test applied in the ambient space | judge on the reduced pair |
| LTV rank test passes on a system that is not controllable | plus sign in `Gamma_{k+1}` | use `dGamma_k/dt - A Gamma_k` |
| Stable loop, cost about twice the prediction along a fast reference | `e_Omega` fed where `dOmega` was designed | add `Omegad x e_R`, or feed `Omega - Omegad` |
| Linear prediction wrong at first order on S^2 | initial `domega` violates `qd . domega = xi . (omegad x qd)` | start in `ker C` |
| Recovery fails beyond a cone of initial errors | `B` depends on the state and the gain is fixed | report the cone; reschedule the gain or use a nonlinear geometric law |
| Recovery slow near 180° | `e_R` has length `sin(theta)` | expected; the logarithm error does not fade but is discontinuous at 180° |
| Euler-angle LQR tumbles near a target at high pitch | chart singularity | replace with this procedure |
