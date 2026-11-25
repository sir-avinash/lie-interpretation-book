---
name: ch14-invariant-filter-design
description: Design and validate an invariant extended Kalman filter (InEKF) for a system whose state is a matrix Lie group such as SO(3), SE(3), SE_2(3) or SE_K(3) (attitude, pose, IMU navigation, contact-aided legged odometry), or a variation-based EKF when the state is a sphere S^2 and not a group. Test the group-affine property, choose the left- or right-invariant error from the measurement form, derive and check the error matrices by finite differences, implement the correction on the group, and test consistency (NEES) and convergence from poor initial estimates against a conventional EKF. Use when asked for an attitude or navigation filter, IMU plus landmark, GPS or leg-kinematics fusion, a filter that diverges or is overconfident after a bad initialization, or an EKF whose state must stay on a manifold.
---

# Invariant filter design on a Lie group, and the variation-based EKF on a sphere

## When to use

- The state is a group element `X` and the model is driven by measured velocities or accelerations (gyroscope, IMU, odometry): Part A.
- The state contains a direction `q` in S^2 (pendulum, cable): Part B.
- Do not expect a gain when the initial estimate is good: from a small initial error an InEKF and a conventional error-state EKF perform alike. The gain is in the transient from a poor estimate and in a linearization that does not move.

## Conventions (fixed)

- A hat marks an estimate (`Xhat`, `Rhat`, `qhat`, `omegahat`), never the hat map, which is the function `hat` or a postfix `^`. The truth has no mark. Errors are the estimate relative to the truth, with the estimate in the slot of the actual configuration:
  `E_r = Xhat X^-1 = exp(chi_r^)`, `E_l = X^-1 Xhat = exp(chi^)`.
- SE_2(3): `X = [R v x; 0 1 0; 0 0 1]`, vector of the algebra `chi = (chi_v, chi_x, chi_R)` (linear parts in column order, rotation last). `exp(chi^)` has rotation `expm_so3(chi_R)` and columns `J chi_v`, `J chi_x`, with `J = I + (1 - cos th)/th^2 hat(chi_R) + (th - sin th)/th^3 hat(chi_R)^2`, `th = |chi_R|`.
- Adjoint on SE_2(3): `Ad_X = [[R, 0, hat(v) R], [0, R, hat(x) R], [0, 0, R]]`. `chi_r = Ad_X chi`.
- Symbols: covariance `Sigma`, process noise `Sigma_n`, measurement noise `Sigma_y`, innovation `ytilde` (predicted minus measured), its covariance `Sigma_ytilde`, Kalman gain `Ktilde`, measurement matrix `C_y`. A correction subtracts.
- Code names: `hat`, `vee`, `expm_so3`, `log_so3`.

## Part A. Invariant EKF

1. **Write the model as `X_dot = f_u(X)` and test the group-affine property** at random `X1`, `X2`, `u`:
   `f_u(X1 X2) = f_u(X1) X2 + X1 f_u(X2) - X1 f_u(I) X2`.
   Require a residual at rounding level. IMU kinematics `R_dot = R hat(Omega)`, `v_dot = R a_s - g e3`, `x_dot = v` pass. A linear drag `-c1 v` passes. A drag along a body axis fails. Biases fail.
   If it fails, find the smallest part that breaks it (usually biases or parameters) and treat that part as appended vector states (step 8).

2. **Choose the error from the measurements.**
   - Known world vector seen in the body frame, `[y; 0; 1] = X^-1 d + noise` (landmark, leg kinematics, magnetometer, gravity direction): right-invariant error, correction multiplies on the left.
   - Body point seen in the world frame, `y = X d + noise` (position fix): left-invariant error, correction multiplies on the right.
   - Mixed sets: pick the form of the measurement that matters most, and linearize the others at the estimate.

3. **Derive the error matrix `A`** as the first-order part of the autonomous error equation:
   - right: `f_u(exp(chi^)) - exp(chi^) f_u(I) = (A chi)^ + O(|chi|^2)`;
   - left: `f_u(exp(chi^)) - f_u(I) exp(chi^) = (A chi)^ + O(|chi|^2)`.
   IMU on SE_2(3), right-invariant: `A = [[0, 0, -g hat(e3)], [I, 0, 0], [0, 0, 0]]`, constant, `A^3 = 0`,
   `exp(A h) = [[I, 0, -g h hat(e3)], [h I, I, -(1/2) g h^2 hat(e3)], [0, 0, I]]`.
   IMU on SE_2(3), left-invariant: `A = [[-hat(Omega), 0, -hat(a_s)], [I, -hat(Omega), 0], [0, 0, -hat(Omega)]]`.
   With a fixed contact point `x_c` as a further column (SE_3(3), order `(chi_v, chi_x, chi_c, chi_R)`): the same `A` with a zero row and column for `chi_c`.

4. **Check `A` (mandatory).** Two checks, both must pass:
   - finite differences of the expression in step 3 at `chi = eps * random`;
   - log-linear property at a LARGE error (1 to 2 rad): propagate truth and estimate with the same inputs, and require `E_r(t) = exp((Phi(t) chi_r(0))^)` to integration accuracy, on two different trajectories. If this holds only for small errors, the system is not group-affine or `A` is wrong.

5. **Derive `C_y` and check it by finite differences.**
   - Landmark at known `x_m`, reading `y = R^T (x_m - x) + n_y`: `d = (x_m, 0, 1)`, `ytilde = Rhat y + xhat - x_m`, `C_y = [0, I, -hat(x_m)]`.
   - Leg kinematics to a contact point, `y = R^T (x_c - x) + n_y`: `d = (0, 0, 1, -1)`, `ytilde = Rhat y - (xhat_c - xhat)`, `C_y = [0, I, -I, 0]`.
   - Check: set `Xhat = exp(eps chi^) X`, noise zero, and require `ytilde / eps -> C_y chi`.
   `C_y` must not contain the estimate. If it does, the measurement is not of the invariant form for this error.

6. **Implement the filter.**
   - Propagate the estimate with the readings. Covariance, with sensor noise in the body frame (`n_k = h (n_a, 0, n_g)`, covariance `Sigma_n`):
     `Sigma <- exp(A h) (Sigma + Ad_Xhat Sigma_n Ad_Xhat^T) exp(A h)^T`.
   - Gain: `Sigma_ytilde = C_y Sigma C_y^T + Rhat Sigma_y Rhat^T`, `Ktilde = Sigma C_y^T Sigma_ytilde^-1`.
   - Correct (right-invariant): `Xhat <- exp(-(Ktilde ytilde)^) Xhat`,
     `Sigma <- (I - Ktilde C_y) Sigma (I - Ktilde C_y)^T + Ktilde Rhat Sigma_y Rhat^T Ktilde^T`.
   - Left-invariant: `Xhat <- Xhat exp(-(Ktilde ytilde)^)`, and body-frame process noise enters without the adjoint.
   - Initial covariance: if it is stated for a left or body-frame error (independent attitude, velocity, position), convert it: a covariance `Sigma` of `chi` is `Ad_Xhat Sigma Ad_Xhat^T` for `chi_r`.

7. **Validate.**
   - One correction in components equals the matrix formula `exp(-(Ktilde ytilde)^) Xhat`.
   - Dead reckoning Monte Carlo without measurements: the sample covariance of `chi_r = log(Xhat X^-1)` matches the propagated `Sigma` (standard deviations within sampling error).
   - Consistency: NEES `chi_r . Sigma^-1 chi_r / dim`, averaged over runs and time after convergence, near 1.
   - Convergence from poor starts: sweep the initial attitude error (for example 5°, 60°, 120°, 170°), 100 or more runs, against a conventional error-state EKF given the same readings and the same initial covariance in its own coordinates. Report median and 90th percentile of the settling time and the NEES. Expect equal performance at 5°.
   - Observability: with constant `A`, `C_y` the rank of `[C_y; C_y Phi; C_y Phi^2; ...]` gives the unobservable directions once, for all trajectories. IMU plus one contact: rank 8 of 12 (yaw and absolute position unobservable). Do not judge convergence on unobservable states.

8. **Biases and other non-group states ("imperfect" InEKF).** Append the bias error `(bhat_g - b_g, bhat_a - b_a)` to `chi_r` as vector states. With `bhat_g - b_g` and `bhat_a - b_a` the bias errors, a bias error enters the error equation as `-Ad_Xhat (bhat_a - b_a, 0, bhat_g - b_g)`: the new columns of `A` contain `Rhat`, `hat(vhat) Rhat`, `hat(xhat) Rhat`. The group block of `A` is unchanged. The log-linear property no longer holds exactly: rerun step 7 and expect the advantage to shrink but not vanish.

## Part B. Variation-based EKF on S^2

State `(q, omega)` with `|q| = 1`, `omega . q = 0`, estimate `(qhat, omegahat)`. Error: the linear state `s = (xi, delta omega)` with `C s = 0`, `C = [[q^T, 0], [(q x omega)^T, q^T]]`.

1. **Operators.** `(q, omega) (+) s = (expm_so3(xi) q, expm_so3(xi) (omega + (I - q q^T) delta omega))`. The inverse `(-)`: `xi = angle * (q1 x q2)/|q1 x q2|`, tangent part of `delta omega` is `expm_so3(xi)^T omega2 - omega1`, part along `q1` is `-(omega1 . (xi x q1))`. Check that `(-)` inverts `(+)`.
2. **Projector onto ker C.** `Pi_C = [[I - q q^T, 0], [-q (q x omega)^T, I - q q^T]]`. Check `C Pi_C = 0`, `Pi_C Pi_C = Pi_C`.
3. **Propagate.** Integrate the dynamics, put `q` back to unit length and `omega` into the tangent plane. With the variation-based linearization `A` at the estimate (free spherical pendulum: `A = [[hat(omega), I - q q^T], [-(g/l) hat(e3) hat(q), 0]]`), `A_k = exp(A h)`: `Sigma <- Pi_C (A_k Sigma A_k^T + Sigma_n) Pi_C^T` with `Pi_C` at the new estimate.
4. **Correct.** For `y = l q + n_y`: `ytilde = l qhat - y`, `C_y = [-l hat(qhat), 0]`, `Ktilde = Sigma C_y^T (C_y Sigma C_y^T + Sigma_y)^-1`, `s+ = -Ktilde ytilde`, estimate `<- estimate (+) s+`. Never invert `Sigma`: it has rank 4.
5. **Move the covariance.** After the covariance update, with `T = blkdiag(expm_so3(xi), expm_so3(xi))` for the `xi` of `s+`: `Sigma <- Pi_C T Sigma T^T Pi_C^T`, `Pi_C` at the new estimate.
6. **Validate.** `|qhat| - 1` and `qhat . omegahat` at rounding level; `C Sigma` at rounding level with no extra constraint update; the transition matrix against finite differences of the flow read through `(-)`; NEES with the pseudo-inverse of `Sigma`, divided by 4, near 1; convergence from random starts drawn from the initial covariance.
7. `A` and `C_y` depend on the estimate here. This filter keeps the state on the manifold. It is not invariant.

## Failure modes

- **Correction on the wrong side.** `Xhat exp(.)` in a right-invariant filter diverges even from a 2° error when the vehicle is far from the origin. The side of the correction is the side of the error.
- **Initial covariance in the wrong coordinates.** "Heading uncertain, position certain" needs the correlated block `hat(x) R` in right-invariant coordinates.
- **Sign of the innovation.** With `ytilde` predicted minus measured, the correction is `exp(-(Ktilde ytilde)^)`. Papers that use measured minus predicted correct with `exp(+.)`. Mixing the two makes the filter diverge at once.
- **Checking `A` only at small errors.** Every Jacobian passes that test. Step 4 at a large error is what separates an invariant error from a conventional one.
- **Claiming exactness with noise or biases.** Noise enters through `Ad_Xhat`, biases through estimate-dependent columns. With odometry noise the cloud is only approximately Gaussian in exponential coordinates (second-order terms show first in the smallest variance).
- **Errors near half a turn.** The logarithm is not unique there, and one Gaussian cannot represent two candidate attitudes. Expect slow convergence from near half a turn in every filter, and do not use `log_so3` blindly at an angle of pi.
- **Subtracting and adding on S^2.** A filter on the six numbers `(q, omega)` returns estimates with `|q|` a few percent from 1 and `q . omega` far from 0.
- **Inverting a rank-deficient covariance** on S^2. Use the gain formula above or a 4x4 covariance in a basis of `ker C`.
