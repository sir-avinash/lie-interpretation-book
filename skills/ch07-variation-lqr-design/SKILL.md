---
name: ch07-variation-lqr-design
description: Design a variation-based LQR for a system whose state lives on SO(3), S^2 or a product of these with R^n. Obtain the linearization along a reference with geomech or by hand, check it by finite differences of perturbed trajectories, test controllability on the constraint subspace, solve the Riccati equation, implement the feedback with geometric errors, and measure the region of attraction. Use when asked for an LQR, a linearization, a time-varying gain or a "small-error model" of a rigid body, pendulum, cable-suspended load or similar system, or when an Euler-angle LQR misbehaves near a singularity.
---

# Variation-based LQR design

## When to use

- The plant state contains an attitude `R` in SO(3) or a direction `q` in S^2, and you need a linear model about an equilibrium or a trajectory.
- The reference passes near a chart singularity, or the set-point is one (nose-up hover, inverted pendulum).
- Do not use it to claim large-error or almost-global behaviour. The result is first order.

## Conventions

| Manifold | Kinematics | Variation | Linear state | Constraint |
| :-- | :-- | :-- | :-- | :-- |
| SO(3) | `R' = R hat(W)`, W in body frame | `dR = R_d hat(eta)` | `(eta, dW)`, `dW = W - W_d` | none |
| S^2 | `q' = w x q`, `w . q = 0` | `dq = xi x q_d` | `(xi, dw)`, `dw = w - w_d` | `q_d . xi = 0`, `q_d . dw = xi . (w_d x q_d)` |
| R^n | `x' = v` | `dx` | `(dx, dv)` | none |

Kinematic rows: `eta' = -W_d x eta + dW` and `xi' = w_d x xi + (I - q_d q_d^T) dw`.

## Procedure

1. **Write the model.** Equations of motion in `R`, `W`, `q`, `w`, with every input declared. State the reference `(x_d(t), u_d(t))` and confirm it satisfies the equations (residual below 1e-8).

2. **Obtain `A(t)`, `B(t)`.**
   - With geomech: `eom = compute_eom(L, dW, variables)`, require `check_eom(...) < 1e-5`, then `lin = linearize(eom, variables, inputs)` and generate code with `module_source(lin, layout)`. geomech returns `M d(dx)/dt = A dx + B du`. Use `A = solve(M_lin, A_lin)`, `B = solve(M_lin, B_lin)`. `M` is the physical inertia (positive). If an S^2 block of `M_lin` is singular, project with the tangent basis first (step 4).
   - By hand: vary each equation with `d(R^T a) = hat(R_d^T a) eta`, `d(q) = xi x q_d`, the product rule, and the kinematic rows above.
   - Rigid body `J W' = J W x W + m g (R^T e3) x rho + tau`:
     `A = [[-hat(W_d), I], [-m g J^-1 hat(rho) hat(R_d^T e3), J^-1 (hat(J W_d) - hat(W_d) J)]]`, `B = [0; J^-1]`.
   - Spherical pendulum `m l w' = q x (F - m g e3)`:
     `A = [[hat(w_d), I - q_d q_d^T], [hat(F_d - m g e3) hat(q_d) / (m l), 0]]`, `B = [0; hat(q_d) / (m l)]`.

3. **Check the linearization by finite differences (mandatory).**
   - Simulate the nonlinear plant from the reference state and from two neighbours at `+eps` and `-eps`: `R_d exp(eps hat(eta0))`, `W_d + eps dW0`, `exp(eps hat(xi0)) q_d`, and `w` re-projected onto the tangent plane of the perturbed `q`. Perturb the input by `eps du(t)`.
   - Read the centred difference as `eta = vee(R_d^T dR)`, `xi = q_d x dq`, plain differences for velocities.
   - Integrate `s' = A(t) s + B(t) du(t)` from the same initial `s`. Require agreement that improves by 100 when `eps` shrinks by 10.
   - On S^2, compare two derivations only on the constraint subspace. 3x3 blocks that multiply `xi` or `dw` are not unique off it (geomech prints `q q^T hat(w)` where the hand derivation gives `hat(w)`).

4. **Reduce a constrained state.** For each S^2 factor build `C(t) = [[q_d^T, 0], [(q_d x w_d)^T, q_d^T]]`. Check `C B = 0` and `C A + dC/dt = 0`. At an equilibrium take an orthonormal tangent basis `T` (3x2) of each S^2 factor, stack `N = blkdiag(T, T, I, ...)`, and use `A_r = N^T A N`, `B_r = N^T B`. Never pass an unreduced S^2 system, or a linearization in the nine entries of `R`, to an algebraic Riccati solver: its uncontrollable modes at zero make the solver fail.

5. **Test controllability where the state lives.**
   - Equilibrium: `rank ctrb(A_r, B_r) = dim A_r`. The unreduced rank equals `dim ker C`, not the full dimension. That is expected.
   - Moving reference: `rank [M_0, M_1, ...] = dim ker C(t)` with `M_0 = B`, `M_{k+1} = dM_k/dt - A M_k`. The sign is minus. Confirm with the controllability Gramian restricted to `ker C(t_0)` if in doubt.

6. **Solve the Riccati equation.**
   - Equilibrium: `P = solve_continuous_are(A_r, B_r, Q, R)`, `K = R^-1 B_r^T P N^T`.
   - Trajectory: integrate `-P' = A^T P + P A - P B R^-1 B^T P + Q` backwards from `P(T)`, tabulate `K(t) = R^-1 B(t)^T P(t)`. Check the residual of the differential equation on the stored grid, and that a constant `(A, B)` reproduces the algebraic solution.

7. **Implement the feedback with geometric errors.**
   - SO(3): `e_R = 0.5 vee(R_d^T R - R^T R_d)`, `e_W = W - R^T R_d W_d`, and
     `tau = tau_d(t) - K(t) [e_R; e_W + W_d x e_R]`.
     To first order `e_R = eta` and `e_W = dW - W_d x eta = eta'`. The correction `W_d x e_R` is required whenever the reference is turning. `W - W_d` may be used in place of `e_W + W_d x e_R`.
   - S^2 at an equilibrium: `e_q = q_d x q`, `F = F_d - K [e_q; w]`.
   - S^2 along a moving reference: `e_w = w - (I - q q^T) w_d` equals `(I - q_d q_d^T) dw` to first order, so feed `dw = e_w + q_d (e_q . (w_d x q_d))`.
   - Never feed differences of Euler angles, quaternion components or matrix entries.

8. **Validate the closed loop on the nonlinear plant.**
   - Cost test: for a small initial error the measured cost divided by `s_0^T P(t_0) s_0` must tend to 1. A ratio near 2, or any ratio that does not tend to 1, means the fed-back state is not the design state (usually a missing `W_d x e_R`).
   - Invariance test: for a free rigid body the closed-loop error history at a hold must not depend on the hold attitude.
   - Region of attraction: sweep the initial error angle to 180 degrees, about several axes, and record settle time or failure. Repeat with input limits. Report the boundary as measured, not as guaranteed.

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
| LTV rank test passes on a system that is not controllable | plus sign in `M_{k+1}` | use `dM_k/dt - A M_k` |
| Stable loop, cost about twice the prediction along a fast reference | `e_W` fed where `dW` was designed | add `W_d x e_R`, or feed `W - W_d` |
| Linear prediction wrong at first order on S^2 | initial `dw` violates `q_d . dw = xi . (w_d x q_d)` | start in `ker C` |
| Recovery fails beyond a cone of initial errors | `B` depends on the state and the gain is fixed | report the cone; reschedule the gain or use a nonlinear geometric law |
| Recovery slow near 180 degrees | `e_R` has length `sin(theta)` | expected; the logarithmic error does not fade but is discontinuous at 180 degrees |
| Euler-angle LQR tumbles near a set-point at high pitch | chart singularity | replace with this procedure |
