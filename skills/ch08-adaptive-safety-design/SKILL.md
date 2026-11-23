---
name: ch08-adaptive-safety-design
description: Add geometric L1 adaptation and a barrier-function safety filter to a geometric tracking controller on SO(3), S^2 or a product of these with R^n. Covers how model error enters the error dynamics, the state predictor on the manifold, the adaptation law with projection, the low-pass filter and how to choose its cutoff, barrier functions for cones, tilt limits and obstacles, the relative-degree-two condition, and the CBF-CLF-QP with relaxed tracking. Use when a geometric controller shows a steady offset or a residual error under payload, wind or inertia mismatch, or when a state constraint (cable cone, tilt limit, keep-out region, minimum separation) must hold even if the reference violates it.
---

# Adaptive and safety-critical geometric control

## When to use

- A geometric PD controller with feedforward tracks well in simulation with the right model and leaves an offset or a wobble on the vehicle.
- A constraint on a direction, an attitude or a position must never be violated, and the reference may violate it.
- Do not use the adaptive part for unmatched uncertainty (errors that do not enter where the input does). Do not call the result safe without the checks in "Failure modes".

## Conventions

| Item | Form |
| :-- | :-- |
| Kinematics | `R_dot = R hat(Omega)`, `Omega` in the body frame. `q_dot = omega x q`, `omega . q = 0` |
| Errors | `e_R(R, Rd) = vee(Rd^T R - R^T Rd) / 2`, `e_Omega = Omega - R^T Rd Omegad`, `Psi_R = tr(I - Rd^T R) / 2` |
| Believed inertia | `J0`. True inertia `J`. Unknown body torque `Delta_e` |
| Lumped uncertainty | `Delta`, defined by `J0 e_Omega_dot = mu + Delta` |
| Predictor | `Rp`, `Omegap`. Estimate `Dhat`. Filtered estimate `Df` |
| Barrier function | `H >= 0` is the safe set. `H1 = H_dot + alpha1 H` |

Formulas below are in plain LaTeX.

## Part A. Adaptation

1. **Write the baseline and find where the uncertainty enters.** With

   $$\tau = \mu + \Omega\times J_0\Omega - J_0\big(\Omega^\wedge R^\top R_d\,\Omega_d - R^\top R_d\,\dot\Omega_d\big), \qquad \mu = -k_R\,e_R - k_\Omega\,e_\Omega,$$

   and the plant $J\dot\Omega + \Omega\times J\Omega = \tau + \Delta_e$, the error dynamics are

   $$J_0\,\dot e_\Omega = \mu + \Delta, \qquad \Delta = (J_0 J^{-1} - I)\mu + J_0 J^{-1}(\Delta_m + \Delta_e),$$

   $$\Delta_m = -\Omega\times(J - J_0)\Omega + (J - J_0)\big(\Omega^\wedge R^\top R_d\,\Omega_d - R^\top R_d\,\dot\Omega_d\big).$$

   For another system, repeat this: substitute the control law that uses the believed parameters into the true equations of motion and collect everything except `mu`. Proceed only if the remainder multiplies the same input matrix as `mu` (matched).

2. **Predict the damage before fixing it.** A constant torque leaves the offset $\sin\theta = \|\Delta_e\| / k_R$. If $\|\Delta_e\| \ge k_R$ the plain controller has no equilibrium and tumbles. Simulate both cases first.

3. **Build the state predictor on the manifold.** $\dot R_p = R_p\,\Omega_p^\wedge$, $J_0\,\dot\Omega_p + \Omega_p\times J_0\Omega_p = \tau_p$, with $\tau_p$ of the same form as $\tau$, evaluated at $(R_p, \Omega_p)$ with input $\mu_p$. Start it at the measured state.

4. **Measure the prediction error with the geometric errors**, predictor first, plant second:

   $$\tilde\Psi = \tfrac12\operatorname{tr}(I - R^\top R_p), \qquad \tilde e_R = \tfrac12(R^\top R_p - R_p^\top R)^\vee, \qquad \tilde e_\Omega = \Omega_p - R_p^\top R\,\Omega .$$

   Never subtract rotation matrices or body rates entry by entry.

5. **Inputs.** With $U = J_0 R_p^\top R\,J_0^{-1} R^\top R_p$ and $e_R^p, e_\Omega^p$ the predictor's tracking errors,

   $$\mu_p = -k_R\,e_R^p - k_\Omega\,e_\Omega^p + U\hat\Delta - \Delta_f,$$

   $$\mu = J_0 R^\top R_p J_0^{-1}\big(-k_R\,e_R^p - k_\Omega\,e_\Omega^p - \Delta_f + \tilde k_R\,\tilde e_R + \tilde k_\Omega\,\tilde e_\Omega\big) + J_0 R^\top R_p\,\tilde e_\Omega^\wedge R_p^\top R\,e_\Omega .$$

   Result to check: $J_0\,\dot{\tilde e}_\Omega = -\tilde k_R\,\tilde e_R - \tilde k_\Omega\,\tilde e_\Omega + U\tilde\Delta$ with $\tilde\Delta = \hat\Delta - R_p^\top R\,\Delta$.

6. **Adaptation law with projection.**

   $$\dot{\hat\Delta} = \gamma\,\mathrm{Proj}(\hat\Delta, y), \qquad y = -U^\top\big(\tilde e_\Omega + c\,J_0^{-1}\tilde e_R\big),$$

   $$\mathrm{Proj}(\hat\Delta, y) = y - \frac{\nabla f\,(\nabla f\cdot y)}{\|\nabla f\|^2}\,f \ \text{ if } f > 0 \text{ and } y\cdot\nabla f > 0, \ \text{ else } y, \qquad f = \frac{1.1\,\|\hat\Delta\|^2 - \Delta_{\max}^2}{0.1\,\Delta_{\max}^2}.$$

   Choose `Delta_max` as the largest uncertainty you are prepared to cancel, with the true one inside `0.95 Delta_max`. In a sampled implementation, after each step rescale `Dhat` back to norm `Delta_max` if it overshot.

7. **Filter.** $\dot\Delta_f = \kappa(\hat\Delta - \Delta_f)$. Only `Df` enters the plant torque.

8. **Integrate the fast loop semi-implicitly.** Each period: compute the torque from the current state, advance `Dhat`, then advance the predictor with the new `Dhat`, then the filter. This is stable for `sqrt(gamma / lambda_min(J0)) h < 2`. An explicit Euler step on both is unstable at useful gains.

9. **Tune.**
   - `gamma`: as large as step 8 allows. It sets how closely the plant follows the predictor (prediction error proportional to `1 / gamma`). It does not set the delay margin.
   - `kappa`: from the slowest actuator and the largest loop delay, not from the tracking plot. Measure the delay margin for each candidate. Lower `kappa` if the margin is short. Expect a residual error for disturbances faster than `kappa`.

### Checks for Part A

- `J0 e_Omega_dot - mu - Delta` is zero at random states (below 1e-9).
- The prediction-error identity of step 5 holds at random states, with unequal gains (below 1e-9).
- `Dtilde . (Proj(Dhat, y) - y) <= 0` on random samples with the true value inside `0.95 Delta_max`.
- With a constant torque, the plain controller settles at `asin(|Delta_e| / k_R)` and the adaptive one at zero, with `Df` equal to `Delta`.
- Doubling `gamma` halves the peak prediction error rate.
- The largest input delay the loop survives falls as `kappa` rises and is the same for different `gamma`.
- Reference numbers (vehicle of the chapter, 0.5 kg camera 8 cm off-center, `k_R = 2`, `gamma = 4000`, `kappa = 20` rad/s): offset 11.3° without adaptation; peak 2.8° and zero offset with it; delay margin 46 ms without, 41, 23 and 6 ms at `kappa = 3`, 30 and 300 rad/s.

## Part B. Safety filter

1. **Write the safe set as `H >= 0` with objects on the manifold.**
   - Cone for a direction: $H = q\cdot q_c - \cos\theta_{\max}$.
   - Tilt limit for a body: $H = e_3\cdot R\,e_3 - \cos\theta_{\max}$.
   - Keep-out ball for a point: $H = \|x - x_o\|^2 - r_o^2$.
   - Separation of two vehicles: $H = \|x_1 - x_2\|^2 - r_{\min}^2$.

2. **Differentiate with the kinematics until the input appears.**
   - Cone: $\dot H = (\omega - \omega_c)\cdot(q\times q_c)$. For the pendulum $m l^2\dot\omega = -m g l\,q\times e_3 + l\,q\times F$ and a fixed axis,
     $$\ddot H = -\|\omega\|^2 (q\cdot q_c) + (I - q q^\top) q_c\cdot\frac{F - m g\,e_3}{m\,l}.$$
     A moving axis adds $2\,\dot q\cdot\dot q_c + q\cdot\ddot q_c$.
   - Tilt: $\dot H = \Omega\cdot(e_3\times R^\top e_3)$ and
     $$\ddot H = \dot\Omega\cdot(e_3\times R^\top e_3) - \|\Omega\|^2 (e_3\cdot R^\top e_3) + (\Omega\cdot R^\top e_3)(\Omega\cdot e_3).$$
   - Check both against finite differences along a simulated motion (below 1e-5).

3. **Relative degree one:** require $\dot H \ge -\alpha H$. **Relative degree two:** set $H_1 = \dot H + \alpha_1 H$ and require

   $$\ddot H + (\alpha_1 + \alpha_2)\,\dot H + \alpha_1\alpha_2\,H \ge 0 .$$

   The initial state must satisfy $H \ge 0$ and $H_1 \ge 0$.

4. **Check that every input reaches the condition with a usable sign.** For a quadrotor and a keep-out ball, the torque is absent from $\ddot H$ and the thrust enters through $(x - x_o)\cdot R\,e_3$, of either sign. Use the augmented barrier function

   $$H(x, R) = \|x - x_o\|^2 - \beta\,r_o^2 - \sigma(p), \qquad p = (x - x_o)\cdot R\,e_3,$$

   with $\sigma' < 0$ and $|\sigma| \le \sigma_0 < (\beta - 1)\,r_o^2$. Then the thrust is multiplied by $(2p - \sigma'(p))/m$ and the torque by $-\sigma'(p)\,J^{-1}(e_3\times R^\top(x - x_o))$ in $\dot H_1$. Wu's third condition, $2p - \sigma'(p) > 0$ on $(-\sqrt{\beta}\,r_o, 0)$, needs $\beta > 2$.

5. **Choose the program.**
   - Filter after an existing tracker: $u = \arg\min \|u - u_d\|^2$ subject to the barrier condition. With one constraint $a\cdot u \ge b$ the solution is $u = u_d + \max(0,\ b - a\cdot u_d)\,a/\|a\|^2$.
   - CBF-CLF-QP: minimize $\tfrac12\|F_{\mathrm{fb}}\|^2 + \tfrac12 k_w w^2$ subject to $\dot V + k_V V \le w$ and $\dot H_1 + \alpha_2 H_1 \ge 0$. The barrier constraint is hard. Only the tracking constraint has a slack.

6. **Simulate with a reference that violates the constraint.** The state must slide along the boundary and rejoin the reference.

### Checks for Part B

- `min H >= -1e-6` over the whole run at the design sample time, and the program feasible at every step.
- The closed-form filter equals a QP solver on random states.
- The slack returns to near zero when the reference is safe again.
- Reference numbers (pendulum `m = 0.5` kg, `l = 0.8` m, cone 30°, reference out to 48°, `alpha1 = alpha2 = 6`): tracker alone reaches `H = -0.197`; with the program `min H = 3e-7`.

## Failure modes

- **Wrong model inside the barrier condition.** An unmodeled force shifts the boundary by about `(error in H_ddot) / (alpha1 alpha2)`. With a 1 N unknown wind the pendulum above sits at `H = -0.032`. Tighten the set by that bound, or raise the gains, and test with the wrong mass.
- **Feeding the adaptive estimate to the barrier.** It helps and is not covered by a guarantee: the theorem bounds the prediction error, and `Df` lags by design. Keep a margin.
- **Input limits.** The condition can demand more than the actuators have (6.2 N for a 4.9 N bob arriving at 29° at 1.5 rad/s). Add the limits to the program and count infeasible steps. Zero is the only acceptable count.
- **Sampling.** At slow sample rates the input chatters at the boundary (force jumps of 18 N at 20 ms against 0.013 N at 2 ms in the pendulum example). Run the program at the attitude-loop rate.
- **Measure-zero sets.** The input drops out of the cone condition at `q = ±q_c` and out of the augmented barrier when the thrust axis is parallel to `x - x_o`. A reference aimed exactly at the center of an obstacle stalls. Offset it.
- **Cutoff set by the tracking plot.** Raising `kappa` spends delay margin. With a 20 ms motor lag a cutoff of 3000 rad/s is unstable.
- **Projection ball too small.** The estimate is clipped and an offset returns (35° in the chapter's example with `Delta_max = 1` N·m against 2.16 N·m).
- **Plain barrier on an underactuated vehicle.** The program becomes infeasible when the thrust axis points toward the obstacle. In the chapter's example it fails 1.88 m from the center and the vehicle hits.
