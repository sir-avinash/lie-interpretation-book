---
name: ch13-manifold-mpc-design
description: Formulate error-state MPC (a convex QP) or a direct transcription with an on-manifold Newton, SQP or interior-point method for a system whose state lives on SO(3), SE(3), S^2 or a product of these with R^n. Choose the error, linearize the error dynamics about the reference and check them by finite differences, build and check the QP, apply steps with a retraction, handle bounds and keep-out constraints, and validate in closed loop. Use when asked for MPC, trajectory tracking with input limits, trajectory optimization, or a "solver on the manifold" for a rigid body, quadrotor, marine vehicle or similar system, or when an optimizer is being fed rotation-matrix entries, Euler angles or quaternions with constraints.
---

# MPC and trajectory optimization on a Lie group

## When to use

- Tracking a reference `(Rd(t), Omegad(t), tau_d(t))` with input bounds: use error-state MPC (Part A).
- No reference yet, or state constraints that the reference violates: use direct transcription with on-manifold steps (Part B), then track its result with Part A.
- Do not use Part A when the error can reach half a turn, or when the reference ignores the dynamics badly. Do not expect Part B to find a global minimum.

## Rules that hold in both parts

- The solver's unknown for an attitude is a step `eta` in R^3 (for a pose, a step in R^6). It is never the nine entries of `R`, Euler angles, or a quaternion with a norm constraint.
- Update with a retraction: `R <- R expm_so3(eta)`. Vectors are updated by addition.
- Gradient and Hessian of a function `Phi` on SO(3) are the coefficients of `Phi(R expm_so3(eta)) = Phi(R) + grad . eta + 1/2 eta^T Hess eta + O(|eta|^3)`. Check both by finite differences of `Phi` along `eta`.
- The Hessian is the symmetric part of the derivative of the gradient. The derivative of the gradient is not symmetric away from a stationary point.
- Derivative of the exponential: `expm_so3(x + dx) = expm_so3(x) expm_so3(dexp(x) dx)` to first order, with `dexp(x) = I - (1 - cos th)/th^2 hat(x) + (th - sin th)/th^3 hat(x)^2`, `th = |x|`.

## Part A. Error-state MPC

1. **Choose the error.** Left-invariant error and its logarithm: `chi = log_so3(Rd^T R)`, with `dOmega = Omega - Omegad` and `dtau = tau - tau_d`. The error state is `s = (chi, dOmega)`. On SE(3): `chi = log(Xd^-1 X)` in R^6, `dzeta = zeta - zeta_d`, twists ordered `(V, Omega)`.

2. **Linearize the error dynamics about the reference.**
   - Rigid body `J Omega_dot + Omega x J Omega = tau`:
     `A = [[-hat(Omegad), I], [0, J^-1 (hat(J Omegad) - hat(Omegad) J)]]`, `B = [0; J^-1]`.
   - SE(3), `M zeta_dot = ad(zeta)^T M zeta + u` with `M = blkdiag(m I, J)` and `ad(zeta) = [[hat(Omega), hat(V)], [0, hat(Omega)]]`:
     `A = [[-ad(zeta_d), I], [0, A22]]`, `B = [0; M^-1]`, `A22 = M^-1 [[-m hat(Omegad), m hat(Vd)], [0, hat(J Omegad) - hat(Omegad) J]]`.
   - Other forces (drag, added mass): linearize them about the reference twist and add `M^-1` times the result to `A22`.
   - `A` must contain the reference velocity and not the reference attitude or pose. If `Rd` appears, a term is not left-invariant (gravity on an offset center of mass, a world-frame force): keep it, and note that the model must then be rebuilt along the reference.

3. **Check the linearization (mandatory).** Simulate the nonlinear plant from `Rd expm_so3(+eps chi0)`, `Omegad + eps dOmega0` with input `tau_d + eps dtau(t)`, and the same with `-eps`. Read `s_fd = (log_so3(Rd^T R+) - log_so3(Rd^T R-), Omega+ - Omega-) / (2 eps)`. Integrate `s_dot = A(t) s + B dtau(t)`. Require a mismatch that falls by 100 when `eps` falls by 10. Also turn `R` and `Rd` by the same rotation on the left and confirm that `s`, `A`, `B` do not change.

4. **Sample.** With the sampling period `h` and the input held: `[A_k, B_k] = first rows of expm([[A, B], [0, 0]] h)`, with `A` evaluated at the middle of the step. Check against an integration of the linear model.

5. **Build the QP.** Minimize `1/2 s_N^T P_f s_N + h/2 sum_{k=1}^{N-1} s_k^T Q s_k + h/2 sum_{k=0}^{N-1} dtau_k^T R_u dtau_k` subject to `s_{k+1} = A_k s_k + B_k dtau_k` and `-tau_max <= tau_d,k + dtau_k <= tau_max`. Eliminate the states: with `U = (dtau_0, ..., dtau_{N-1})` the cost is a quadratic form in `(s_0, U)` with blocks `P_UU`, `P_Us`, and the problem is `min 1/2 U^T P_UU U + (P_Us s_0) . U`, `U_min <= U <= U_max`. Take `P_f` from the discrete Riccati equation of the sampled model at rest.

6. **Check the QP.**
   - The condensed cost equals the cost of a rollout for random `s_0`, `U`.
   - `P_UU` is positive definite.
   - With the bounds removed and the Riccati `P_f`, the first block of `-P_UU^-1 P_Us` equals the LQR gain for every `N`.
   - The solver's answer matches an independent solver (for a box: `scipy.optimize.lsq_linear` on the Cholesky factor of `P_UU`) and satisfies the optimality conditions: gradient zero on free entries, nonnegative at lower bounds, nonpositive at upper bounds.

7. **Solve and apply.** For box bounds use projected Newton: hold entries on a bound whose gradient pushes outward, take a Newton step in the rest, clip, backtrack. Warm-start from the previous solution shifted by one step. Apply `tau = tau_d,0 + dtau_0`, then recompute `s_0` from the new measurement with the logarithm.

8. **Validate in closed loop on the nonlinear plant.** Record the largest applied input (must respect the bound at every step), the error angle `|log_so3(Rd^T R)|`, and the solve time against `h`. Compare with the same regulator clipped after the fact. Report both.

## Part B. Direct transcription with on-manifold steps

1. **Discretize on the group.** Kinematics by the exponential step, written as a residual in the Lie algebra: `r_k = log_so3(R_{k+1}^T R_k expm_so3(h Omega_k)) = 0`. Momentum: `J (Omega_{k+1} - Omega_k) - h (J Omega_k x Omega_k + tau_k) = 0`. No constraint of the form `R^T R = I` is needed.

2. **Take derivatives along the steps.** With `r_k` at its current value:
   `d r_k = dexp_inv(r_k) (expm_so3(-h Omega_k) eta_k + h dexp(h Omega_k) dOmega_k) - dexp_inv(-r_k) eta_{k+1}`.
   For a keep-out cone `q_c . R e1 <= cos(theta_max)` with axis `q_c` and half-angle `theta_max`: gradient `e1 x R^T q_c`, Hessian `1/2 (e1 q_c^T R + R^T q_c e1^T) - (q_c . R e1) I`.
   Check every Jacobian against centered differences through the retraction.

3. **Solve the optimality conditions (interior-point method).** With slack variables `w > 0`, `c(x) + w = 0`, multipliers `nu_r`, `nu_c > 0` and the barrier parameter `mu`:
   `grad Phi + A_r^T nu_r + A_c^T nu_c = 0`, `r = 0`, `c + w = 0`, `w_i nu_c,i = mu`.
   Newton step: `[[Hess + A_c^T diag(nu_c/w) A_c, A_r^T], [A_r, 0]] [dx; dnu_r] = residuals`, where `Hess` is the Hessian of `Phi + nu_r . r + nu_c . c` along the steps. Then recover `dw`, `dnu_c`, shorten the step so that `w` and `nu_c` stay positive, backtrack, retract `x`, add the rest. Reduce `mu` when the residual is below `10 mu`. If the matrix does not have as many negative eigenvalues as equality constraints, add a multiple of the identity to `Hess`.

4. **Use the structure.** Order unknowns and multipliers by stage. The matrix is then banded and the cost of a step is linear in the horizon.

5. **Validate.** At the solution: residual of the optimality conditions, feasibility, nonnegative multipliers, complementarity, positive definite reduced Hessian on the null space of the active constraints. Every iterate must have `|R^T R - I|` at rounding level. Confirm the cost with an independent solver on a different formulation (for example SLSQP on a shooting form). Roll out the inputs through the discrete dynamics and compare.

## Failure modes

- **Half a turn.** `log_so3` jumps at an error of 180°. Errors of 179° and 181° about one axis give opposite corrections. Keep the reference close, or commit to a direction with extra logic.
- **Velocity errors.** A pure attitude error is propagated exactly by the linear model. Velocity errors are not: expect prediction errors of tens of percent beyond about 30° with a matching rate error.
- **A convex QP with a wrong model.** An Euler-angle model near its singularity gives a QP that is solved exactly and tumbles the vehicle.
- **Additive updates.** `R + R hat(eta)` leaves the group at the first step. Project if you must add: projection after a tangent step is also a retraction.
- **Local minima.** A keep-out region on a sphere has two sides. Try initial guesses on both and compare costs.
- **State constraints on a linearized prediction** can be violated by the plant. Input bounds enforced on the applied input cannot.
- **Twist ordering.** Papers that order twists `(omega, v)` have the blocks of `ad` swapped relative to `(V, Omega)`.
- **Weighted costs.** The gradient of `1/2 chi^T P chi` is `P chi` exactly only on SO(3) with `P` a multiple of the identity. Otherwise it holds to first order.
