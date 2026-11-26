---
title: Error-state MPC and on-manifold solvers
short: MPC on manifolds
id: ch-mpc
summary: What an optimizer's variables should be when the state is a rotation, how tracking on a group becomes a quadratic program, and how Newton and interior-point methods work on the manifold itself.
notebook: notebooks/ch13_error_state_mpc.ipynb
verify: verify/ch13.py
epigraph: "Plans are worthless, but planning is everything."
epigraph_source: "Dwight D. Eisenhower, quoting an Army saying, 14 November 1957"
---

Model predictive control (MPC) is a simple loop. Solve an optimization problem over the next $N$ steps. Apply the first input. Measure, and solve again. The window of $N$ steps moves along with you: a **receding horizon**. The loop needs a solver, and every textbook solver expects its unknowns to be a vector in $\R^n$: it adds a step to the current guess and tries again.

Now the state is an attitude. What do you hand the solver?

Hand it the nine entries of each rotation matrix. With $N = 25$ that is 225 numbers and 150 nonlinear equations, six per step, only to say "this is a rotation". The solver adds a step, and its iterates stop being rotations. Halfway between two rotation matrices 84° apart is a matrix with determinant 0.55.

Hand it yaw, pitch and roll. There are no constraints now. There is a singularity at 90° of pitch ([](#sec-euler)), and the solver will find it.

Hand it a unit quaternion. That is one constraint per step and two quaternions for each attitude ([](#sec-quaternions)), so every minimum appears twice.

Each choice describes the attitude with numbers and then moves the numbers. [](#fig13-opening) shows what goes wrong, and what to do instead. Do not move the description. Ask the optimizer for a small rotation, which is three free numbers with no constraint, and apply that rotation to the attitude.

![The puzzle on a sphere. An optimizer at $q$ wants to move toward the mean of four readings. Added to $q$ as a vector, its step (terracotta) ends outside the sphere. Used as a rotation, the same step (indigo) ends on the sphere, 2.5° from the mean.](figures/ch13-opening.svg){#fig13-opening}

## A step that stays on the sphere {#sec13-newton}

Start with the smallest problem that has the difficulty in it. Four sensors report the direction of a cable, $q_1, \dots, q_4 \in S^2$, and they disagree. You want one direction $q$ that is as close as possible to all of them:

$$
\Phi(q) = \tfrac12 \sum_i \theta_i^2, \qquad \theta_i = \text{the angle between } q \text{ and } q_i .
$$

::: {.thought #thought13-average name="The average that is underground"}
Three friends stand in Quito, Nairobi and Singapore, and want to meet at their average position. Average the three position vectors from the center of the Earth. The result is a point 4300 km below the surface: the average of unit vectors is shorter than a unit vector. You could push that point up to the surface. But why should the place directly above it be the fairest meeting point, measured along the ground?

Now do it the way a traveler would. Stand anywhere. Each friend's city lies in some compass direction at some distance. Average those three arrows on your map. Walk that way, that far. Look again. The arrows are shorter. Repeat. Every position you visit is on the surface, because walking cannot take you anywhere else.
:::

The arrows on the map are tangent vectors, and the walk is the exponential map of [](#sec2-exp). In the language of [](#def3-variation), a move of $q$ is a small rotation $q \mapsto \exp(\xi^\wedge)\,q$ with $\xi \cdot q = 0$. Expand the cost along it. The first-order coefficient is the gradient,

$$
\operatorname{grad}\Phi(q) = \sum_i \frac{\theta_i}{\sin\theta_i}\; q_i \times q .
$$

Each term is the error vector $e_q = q_d \times q$ of [](#def-error-s2) with $q_i$ as the target, stretched so that its length is the angle and not its sine. The cost is a sum of squared angles, and its gradient is a sum of errors.

For the four readings of [](#fig13-opening) the plain average has length 0.82. Start Newton's method at the first reading instead. The distance to the minimizer goes 29°, 0.41°, 0.00003°. The figure starts 58° away to make the step visible: 58°, 2.5°, 0.0011°, then $2\times10^{-10}$ degrees. Added to the starting direction as a vector, that first step of 60° would land at a distance 1.45 from the center.

The same construction works on the rotation group.

::: {.definition #def13-newton name="Gradient, Hessian and Newton step on $\SO(3)$"}
For a cost $\Phi$ on $\SO(3)$, the **gradient** $\operatorname{grad}\Phi(R) \in \R^3$ and the **Hessian** $\operatorname{Hess}\Phi(R) \in \R^{3\times3}$, symmetric, are the coefficients of
$$
\Phi\big(R\exp(\eta^\wedge)\big) = \Phi(R) + \operatorname{grad}\Phi\cdot\eta + \tfrac12\,\eta\T \operatorname{Hess}\Phi\;\eta + O(\|\eta\|^3). \label{eq13-expansion}
$$
The **Newton step** solves a $3\times3$ linear system, and the update is a product of rotations:
$$
\operatorname{Hess}\Phi\;\eta = -\operatorname{grad}\Phi, \qquad R \leftarrow R\,\exp(\eta^\wedge). \label{eq13-newton}
$$
:::

The unknown $\eta$ is the variation of [](#def3-variation), used at finite size. It has three components and no constraint, and the update cannot leave the group.

::: {.example #ex13-mean name="The mean of five attitudes"}
Five attitude estimates $R_1, \dots, R_5$ are scattered up to 91° from their mean. Average their entries and the result has determinant 0.08. It is not an attitude.

Minimize $\Phi(R) = \tfrac12\sum_i \|\chi_i\|^2$ instead, with the logarithm errors $\chi_i = \log(R_i\T R)^\vee$ of [](#sec3-errors). Write $\theta_i = \|\chi_i\|$ and $a_i = \chi_i/\theta_i$. Then
$$
\operatorname{grad}\Phi = \sum_i \chi_i, \qquad
\operatorname{Hess}\Phi = \sum_i \Big[\tfrac{\theta_i}{2}\cot\tfrac{\theta_i}{2}\; I + \big(1 - \tfrac{\theta_i}{2}\cot\tfrac{\theta_i}{2}\big)\, a_i a_i\T\Big]. \label{eq13-mean}
$$
Start at $R_1$, 72° from the answer. Newton's method gives 1.6°, then 0.00014°, then $10^{-12}$ degrees ([](#fig13-newton)). Each step at least squares the error. Replace the Newton step by the gradient step $\eta = -\tfrac15\operatorname{grad}\Phi$ and each iteration divides the error by about 7: steady, and far slower.
:::

![Three ways to use the same tangent-space information on the mean of [](#ex13-mean). Left: distance to the minimizer. Newton's method squares the error with either retraction. Gradient steps shrink it by a constant factor. Right: add the step to the matrix, $R + R\,\eta^\wedge$, and the iterate leaves the group at the first step and never returns. The exponential update stays on it to rounding error.](figures/ch13-newton.svg){#fig13-newton}

The gradient in [](#eq13-mean) is again a sum of errors. That is the observation of Teng et al. (2022): design the cost as a quadratic in the Lie algebra, and its gradient is the tracking error itself.^[Exactly so on $\SO(3)$ with the plain dot product, as here. With a weight, $\tfrac12\chi\T P\chi$, or on $\SE(3)$, the gradient equals $P\chi$ to first order only. At 60° of error and $P = \operatorname{diag}(1,2,3)$ the two differ by 13%.]

The exponential is not the only way back to the group.

::: {.definition #def13-retraction name="Retraction"}
A **retraction** is a rule that moves a point of a manifold along a tangent vector and lands on the manifold, and that agrees with the exponential map to first order. On $\SO(3)$, both $R\exp(\eta^\wedge)$ and "the rotation nearest to $R + R\,\eta^\wedge$" are retractions.
:::

So adding the step to the entries and projecting is not wrong. It is a retraction, and in [](#fig13-newton) it converges at the same order. What made it work is that the step was computed in the tangent space. The method that fails is the one that treats nine entries as free and never projects. On a group the exponential is the natural choice: it is a matrix product, and second-order expansions along it are short.

An optimizer, then, should work in $\eta$. MPC needs more than a point. It needs a model that predicts, written in the same kind of variable.

## The error is the variable {#sec13-error}

Take the rigid body of [](#eq7-body) with $\rho = 0$, and a reference $(R_d, \Omega_d, \tau_d)$ that satisfies the same equations. Measure the body against the reference with the left-invariant error of [](#def3-invariant-errors) and its logarithm:

$$
E_\ell = R_d\T R = \exp(\chi^\wedge), \qquad \delta\Omega = \Omega - \Omega_d, \qquad \delta\tau = \tau - \tau_d .
$$

The pair $s = (\chi, \delta\Omega)$ is the **error state**. It is the linear state of [](#sec7-variation) with $\chi$ in place of $\eta$, and a controller can compute it from measurements.

How does $\chi$ move? [](#eq3-left-error-dynamics) gives $\dot E_\ell$. Turning that into $\dot\chi$ needs the derivative of the exponential,

$$
\exp\big((x + \delta x)^\wedge\big) = \exp(x^\wedge)\,\exp\big((\operatorname{dexp}_x\,\delta x)^\wedge\big) + O(\|\delta x\|^2), \label{eq13-dexp}
$$

which on $\SO(3)$ is $\operatorname{dexp}_x = I - \tfrac{1-\cos\theta}{\theta^2}\,x^\wedge + \tfrac{\theta - \sin\theta}{\theta^3}\,(x^\wedge)^2$ with $\theta = \|x\|$. The result is exact at any size of error:

$$
\dot\chi = \operatorname{dexp}_\chi^{-1}\big(\Omega - \exp(-\chi^\wedge)\,\Omega_d\big). \label{eq13-exact}
$$

Look for $R$ or $R_d$ on the right-hand side. Neither is there. Now keep first-order terms: $\operatorname{dexp}_\chi^{-1} = I + \tfrac12\chi^\wedge + \dots$ and $\exp(-\chi^\wedge)\,\Omega_d = \Omega_d - \chi\times\Omega_d + \dots$. Vary the momentum equation as in [](#thm7-so3).

::: {.proposition #thm13-so3 name="Error-state model of the rigid body"}
To first order in $s = (\chi, \delta\Omega)$,
$$
\ddt\begin{bmatrix}\chi \\ \delta\Omega\end{bmatrix} =
\underbrace{\begin{bmatrix} -\Omega_d^\wedge & I \\[2pt] 0 & J^{-1}\big((J\Omega_d)^\wedge - \Omega_d^\wedge J\big) \end{bmatrix}}_{A(t)}
\begin{bmatrix}\chi \\ \delta\Omega\end{bmatrix} +
\underbrace{\begin{bmatrix} 0 \\ J^{-1} \end{bmatrix}}_{B} \delta\tau . \label{eq13-so3}
$$
The first row is [](#eq3-lie-algebra-error). The whole system is [](#eq7-so3-lin) with $\rho = 0$, reached by the road of [](#sec7-lie).
:::

Along a tumbling reference with a full torque history, perturbed nonlinear trajectories match [](#eq13-so3) to $2.6\times10^{-7}$ for a perturbation of $10^{-4}$, and the mismatch grows a hundredfold when the perturbation grows tenfold. The matrices agree with the symbolic linearization of [](#ch-symbolic) entry by entry.

::: {.thought #thought13-turn name="Turn the lab"}
Fly a flip facing north and record every matrix your controller uses. Now turn the whole lab to face east and fly the flip again: every attitude, actual and desired, is multiplied on the left by the same rotation. Which recorded numbers change?

In [](#eq13-so3), none. The reference enters through $\Omega_d$, which the turn does not touch, and the error $R_d\T R$ is left-invariant. A model written in yaw, pitch and roll has matrices that depend on the heading, and on the pitch, and they blow up near 90° of pitch. Along the flip of [](#ex7-flip) the largest entry of the Euler-angle $A(t)$ is 478. The largest entry of [](#eq13-so3) is 7.8.
:::

This buys three things. The matrices can be tabulated before the flight, as functions of time alone. A reference that turns at a constant rate, or hovers, has one constant model. And nothing has to be linearized again when the measured attitude changes, because the measured attitude appears only in the initial error $s_0$.

**On $\SE(3)$.** The same steps work for a pose $X$ with body twist $\zeta = (V, \Omega)$, where $\dot X = X\,\zeta^\wedge$ as in [](#eq2-se3-kin). Write the Lie-algebra adjoint as a matrix, in the ordering of this book:

$$
\ad_\zeta = \begin{bmatrix} \Omega^\wedge & V^\wedge \\ 0 & \Omega^\wedge \end{bmatrix}, \qquad
M\,\dot\zeta = \ad_\zeta\T M\,\zeta + u, \qquad M = \begin{bmatrix} m I & 0 \\ 0 & J \end{bmatrix}. \label{eq13-ep}
$$

The second equation is the rigid body in the body frame: $m(\dot V + \Omega\times V)$ equals the force and $J\dot\Omega + \Omega\times J\Omega$ equals the torque. The wrench $u$ stacks force and torque in the same order as the twist.

::: {.proposition #thm13-se3 name="Error-state model on $\SE(3)$"}
With $\chi = \log(X_d^{-1}X)^\vee \in \R^6$, $\delta\zeta = \zeta - \zeta_d$ and $\delta u = u - u_d$, to first order
$$
\ddt\begin{bmatrix}\chi \\ \delta\zeta\end{bmatrix} =
\begin{bmatrix} -\ad_{\zeta_d} & I_6 \\[2pt] 0 & A_{22} \end{bmatrix}
\begin{bmatrix}\chi \\ \delta\zeta\end{bmatrix} +
\begin{bmatrix} 0 \\ M^{-1} \end{bmatrix} \delta u, \label{eq13-se3}
$$
$$
A_{22} = M^{-1}\begin{bmatrix} -m\,\Omega_d^\wedge & m\,V_d^\wedge \\ 0 & (J\Omega_d)^\wedge - \Omega_d^\wedge J \end{bmatrix}.
$$
The reference pose $X_d$ does not appear.
:::

The finite-difference test on a tumbling, translating body agrees to $5\times10^{-8}$ and is second order. Jang, Teng and Ghaffari (2023) derive the same structure for a marine vehicle, whose $M$ includes added mass and whose forces include drag, which they linearize about the reference twist. They order the twist angular part first, so their $\ad$ has the blocks of [](#eq13-ep) swapped.

We have a linear model in the right variable. Now add the horizon.

## A quadratic program {#sec13-qp}

Sample with the sampling period $h$. Hold the torque over each step and solve [](#eq13-so3) exactly over it, with $A$ frozen at the middle of the step. That gives a discrete model,

$$
s_{k+1} = A_k\,s_k + B_k\,\delta\tau_k, \qquad k = 0, \dots, N-1, \label{eq13-sampled}
$$

where $A_k$ and $B_k$ are the sampled counterparts of $A$ and $B$. The actuators have limits, which give a bound $|\tau_i| \le \tau_{\max}$ on each axis. The bound is on the torque, not on the correction, so it moves with the feedforward $\tau_d$.

::: {.definition #def13-mpc name="Error-state MPC"}
At each sampling instant, compute $s_0 = \big(\log(R_d\T R)^\vee,\ \Omega - \Omega_d\big)$ from the measured state and solve, with the weights $Q$, $R_u$, $P_f$ of [](#sec7-lqr),
$$
\begin{aligned}
\min_{\delta\tau_0, \dots, \delta\tau_{N-1}}\quad & \tfrac12\, s_N\T P_f\, s_N + \tfrac{h}{2}\sum_{k=1}^{N-1} s_k\T Q\,s_k + \tfrac{h}{2}\sum_{k=0}^{N-1} \delta\tau_k\T R_u\,\delta\tau_k \\
\text{subject to}\quad & s_{k+1} = A_k s_k + B_k\,\delta\tau_k, \qquad -\tau_{\max} \le \tau_{d,k} + \delta\tau_k \le \tau_{\max} .
\end{aligned} \label{eq13-mpc}
$$
Apply $\tau = \tau_{d,0} + \delta\tau_0$ for one period. Discard the rest and repeat.
:::

Eliminate the states with [](#eq13-sampled). Stack the corrections into one vector $U = (\delta\tau_0, \dots, \delta\tau_{N-1})$. Every $s_k$ is linear in $s_0$ and $U$, so the cost is a quadratic form in the pair $(s_0, U)$. Call its blocks $P_{UU}$ and $P_{Us}$, after the matrix $P$ that gives the optimal cost in [](#sec7-lqr). The term without $U$ does not matter, and [](#eq13-mpc) becomes

$$
\min_U\ \tfrac12\, U\T P_{UU}\, U + (P_{Us}\, s_0)\cdot U \qquad \text{subject to} \qquad U_{\min} \le U \le U_{\max}. \label{eq13-qp}
$$

This is a quadratic program, a QP, like the safety filter [](#eq8-safety-filter) but with $3N$ unknowns. It is the program that [](#sec7-lie) announced. $P_{UU}$ is positive definite because $R_u$ is, so the problem is strictly convex: one minimum, no other stationary point, and solvers that find it reliably. $P_{UU}$ and $P_{Us}$ are built from $A_k$, $B_k$ and the weights. By [](#thought13-turn) they depend on time and not on the state.

The bounds are a box, so a small solver is enough. Hold the entries of $U$ that sit on a bound and want to push through it. Take a Newton step in the others. Clip the result to the box, shorten the step if the cost went up, and repeat. This is the projected Newton method of Bertsekas (1982). On 40 test problems with a third of the bounds active, it agrees with SciPy's bounded least-squares solver to $10^{-13}$, and the optimality conditions hold to $10^{-14}$.

**What it has to do with LQR.** Remove the bounds and the minimizer of [](#eq13-qp) is $U = -P_{UU}^{-1}P_{Us}\,s_0$, linear in the error. If $P_f$ is the Riccati solution of the sampled regulator, its first block is exactly $-K s_0$ with the LQR gain $K$ of [](#eq7-riccati) in sampled form, for any horizon, even $N = 1$. That is the principle of optimality: $P_f$ already accounts for everything after the horizon. With a cruder terminal weight, $P_f = hQ$, the first move differs from LQR by 76% in norm at $N = 5$, by 1.5% at $N = 25$ and by $5\times10^{-8}$ at $N = 100$. Error-state MPC is the regulator of [](#sec7-lqr) with a finite horizon and the ability to see a bound coming. Whether that ability matters is a question for an experiment.

## A flip with too little torque {#sec13-flight}

Take the flip of [](#ex7-flip): one turn in 1.5 s about an axis 5° from the pitch axis, with $J = \operatorname{diag}(0.03, 0.05, 0.07)$ kg·m² and the weights $Q = \operatorname{diag}(100\,I, I)$, $R_u = 20\,I$ of [](#sec7-worked). Its feedforward torque peaks at 0.80 N·m. Now suppose the motors deliver only $\tau_{\max} = 0.6$ N·m. The reference was planned without knowing that, which is how references are often planned.

::: {.example #ex13-flip name="Three controllers, one bound"}
Sample at 50 Hz, $h = 0.02$ s, with a horizon of $N = 25$ steps, half a second. The QP has 75 unknowns. Start 20° from the reference and simulate the nonlinear body. Compare three controllers ([](#fig13-tracking)).

*Saturated LQR.* The regulator of [](#ex7-flip) in sampled form, fed with $(\chi, \delta\Omega)$, with its torque clipped to the bound afterwards.

*Error-state MPC.* [](#def13-mpc).

*Euler-angle MPC.* The same QP with the model and the error written in yaw, pitch and roll.

All three respect the bound at every step ([](#tbl13-flip)). The clipped regulator learns of the shortage when it happens. It falls 46° behind, and it is still on the bound a quarter of a second after the reference has stopped. The MPC sees the shortage half a second ahead. It pushes harder before the peak and brakes earlier, and its error stays under 6°.
:::

::: {.table #tbl13-flip}

| | Worst error after 0.5 s | Error at 2.2 s | Steps on the bound |
| :-- | :-: | :-: | :-: |
| Saturated LQR | 46.3° | 1.75° | 66% |
| Error-state MPC | 5.9° | 0.04° | 52% |
| Euler-angle MPC | 10.7° | 0.19° | 50% |

: Tracking the flip of [](#ex7-flip) with 0.6 N·m of torque where the reference asks for 0.80.

:::

![Receding horizon on the flip of [](#ex13-flip). Left: the thrust axis $R\,e_3$ (indigo) chases the reference (dashed), and the plan for the next 25 steps (teal) is redrawn at every sample. Right: the pitch torque. The reference asks for more than the bound allows (shaded). The plan stays inside and makes up the difference before and after.](figures/ch13-horizon-strip.png){#fig13-horizon anim="figures/ch13-horizon.gif"}

![The flip of [](#ex13-flip). Top: attitude error of the three controllers. Bottom: pitch torque of the saturated LQR and of the error-state MPC against the feedforward and the bound.](figures/ch13-tracking.svg){#fig13-tracking}

Two honest notes on the table.

The Euler-angle MPC does well. Its worst error is under twice that of the error-state MPC. The flip passes the chart singularity in a few hundredths of a second, too briefly for the model to do harm, as [](#sec7-worked) found for LQR. Ask it to linger and the result changes. Hold the nose-up attitude of [](#ex7-hold) at 85° of pitch, from a 20° error, with the same bound. The error-state MPC is within 1° after 0.54 s and never uses more than 0.41 N·m. The Euler-angle MPC tumbles to 177° of error, sits on the bound for 99% of the steps, and is still 76° off after 3 s. Its QP is convex and solved exactly. The model inside it is wrong.

The solve time is small because the problem is small. The projected Newton solver needs 1.3 iterations on average and at most 3, warm-started from the previous plan shifted by one step. Even in plain NumPy every solve finishes well inside the sampling period of 20 ms. Jang, Teng and Ghaffari (2023) report the same pattern at scale. With $N = 100$ on a surface vessel in simulation, their QP took about 50 ms on average. Two nonlinear MPC baselines took 460 to 950 ms. They also report what the linearization costs: overshoot and slower convergence from large initial errors, and slightly larger tracking error than nonlinear MPC. Their controller constrains inputs only.

## What is convex, and what is not {#sec13-convex}

The QP [](#eq13-qp) is convex. The problem you care about is not. Its states live on $\SO(3)$, which is not a convex set. Its dynamics are nonlinear equality constraints. Its natural cost, $\|\log(R_d\T R)\|^2$, has a crease at half a turn. The QP is a convex model of that problem, valid near the reference, and the reference had to come from somewhere.

How near is near? One part of the model is better than first order.

Give the body a pure attitude error, with $\delta\Omega = 0$ and the feedforward torque. Both bodies then have the same angular velocity forever, since the momentum equation does not contain $R$. Then [](#eq13-exact) reduces to $\dot\chi = -\Omega_d\times\chi$ exactly. The linear model propagates a 120° attitude error as accurately as a 10° one: the prediction half a second ahead is off by 0.09% in both cases, all of it from sampling.

The model breaks through velocity. Add a velocity error of $2\ \text{s}^{-1}$ times the angle and the same prediction is off by 1.5% at 2°, 7.4% at 10°, 22% at 30° and 41% at 60°. The neglected terms are the ones dropped from [](#eq13-exact): the curvature of the group, through $\operatorname{dexp}^{-1}$, and the quadratic gyroscopic term.

::: {.warning #warn13-log name="The logarithm jumps at half a turn"}
At an error of 179° about some axis, $\chi$ is a vector of length 3.12 along that axis. At 181° it is a vector of length 3.12 along the opposite one. The QP is handed opposite initial errors and returns opposite torques. A body near the cut locus of [](#sec7-local), in noise, can be told to go one way and then the other. Error-state MPC inherits every large-error behavior of the logarithm error, and [](#thm-obstruction) says no continuous fix exists. Keep the reference close enough that this cannot happen, or add logic that commits to a direction.
:::

So error-state MPC needs a good reference, one that respects the dynamics, avoids obstacles and stays within what the actuators can do, at least roughly. Producing one is a harder problem. It is not convex, and the solver itself has to work on the manifold.

## From tracking to planning {#sec13-plan}

Turn a satellite from one attitude to another in 2 s with as little torque as possible. A camera along the body axis $e_1$ must never point within $\theta_{\max} = 30^\circ$ of the sun, which lies in the direction $q_c$. This is the cone of [](#eq8-cone) with the inequality reversed: there a direction had to stay inside, and here the camera axis $R\,e_1$ must stay outside. The shortest turn sweeps the camera through that cone, 8.5° from its axis.

No reference exists yet. The unknowns are the trajectory itself: attitudes $R_k$, angular velocities $\Omega_k$ and torques $\tau_k$ at $N = 20$ steps of $h = 0.1$ s. This is **direct transcription**: discretize first, then optimize over states and inputs together, with the dynamics as constraints.

$$
\begin{aligned}
\min\quad & \Phi = \tfrac{h}{2}\sum_{k=0}^{N-1} \|\tau_k\|^2 \\
\text{subject to}\quad & \log\!\big(R_{k+1}\T\, R_k \exp(h\,\Omega_k^\wedge)\big)^\vee = 0, \\
& J(\Omega_{k+1} - \Omega_k) - h\,\big(J\Omega_k\times\Omega_k + \tau_k\big) = 0, \\
& |\tau_{k,i}| \le \tau_{\max}, \qquad q_c\cdot R_k e_1 \le \cos\theta_{\max} ,
\end{aligned} \label{eq13-dto}
$$

with both ends fixed and at rest. The first constraint is the exponential step [](#eq2-exp-step), written as "the mismatch is the identity" and brought to $\R^3$ by the logarithm. Because the step is a product of rotations, an exact solution of the constraint is a trajectory of rotations. No equation of the form $R\T R = I$ appears anywhere.

Three choices make this an optimization problem on a manifold.

*The variables are group elements.* Each $R_k$ is stored as a rotation matrix. The problem has $9N - 6 = 174$ degrees of freedom: three per attitude, three per velocity, three per torque.

*The steps are in the Lie algebra.* The solver's unknown at each iteration is $(\eta_k, \delta\Omega_k, \delta\tau_k)$, and it updates $R_k \leftarrow R_k\exp(\eta_k^\wedge)$ as in [](#eq13-newton). Vectors are updated by addition.

*Derivatives are taken along those steps.* Perturb the attitudes in the first constraint and collect first-order terms. With $r_k$ the current value of that constraint,

$$
\delta r_k = \operatorname{dexp}_{r_k}^{-1}\Big(\exp(-h\,\Omega_k^\wedge)\,\eta_k + h\,\operatorname{dexp}_{h\Omega_k}\,\delta\Omega_k\Big) - \operatorname{dexp}_{-r_k}^{-1}\,\eta_{k+1} . \label{eq13-jac}
$$

On a trajectory that satisfies the dynamics, $r_k = 0$ and this reads $\eta_{k+1} = \exp(-h\Omega_k^\wedge)\,\eta_k + h\,\operatorname{dexp}_{h\Omega_k}\delta\Omega_k$. Divide by $h$ and let $h \to 0$: it is $\dot\eta = -\Omega\times\eta + \delta\Omega$, the first row of [](#eq13-so3). The Jacobian of a transcription on a group is the error-state model, one step at a time.

The keep-out cone shows what second-order information on a group looks like. The sun direction seen from the body is $R\T q_c$. Expanding $q_c\cdot R\exp(\eta^\wedge)e_1$ as in [](#eq13-expansion) gives

$$
\operatorname{grad} = e_1\times R\T q_c, \qquad \operatorname{Hess} = \tfrac12\big(e_1\,q_c\T R + R\T q_c\,e_1\T\big) - (q_c\cdot R\,e_1)\,I . \label{eq13-cone}
$$

The Hessian is not zero, although the constraint is linear in the entries of $R$. The curvature of the group has moved into it.

Equality constraints can be handled by Newton's method on the optimality conditions. Inequalities need one more idea.

## Walls made of logarithms {#sec13-ipm}

Minimize $x$ subject to $x \ge 0$. The answer is $x = 0$, on the boundary, and that is the trouble: at the boundary the problem stops being smooth. Which constraints are active at the solution? With 139 inequalities there are $2^{139}$ possible answers.

::: {.thought #thought13-wall name="A wall you can lean on"}
A ball rolls down a ramp toward a wall at $x = 0$. Replace the hard wall with a spring that pushes back with force $\mu/x$: weak far away, unbounded at the wall. The ball settles where the slope of the ramp, which is 1, balances the spring: $x = \mu$. It never touches the wall. Now soften the spring. As $\mu \to 0$ the resting point slides to the wall, smoothly, from the inside.
:::

In formulas, replace the constraint by a **logarithmic barrier**: minimize $x - \mu\ln x$, where $\mu > 0$ is the **barrier parameter**. The barrier function of [](#sec8-barrier) shares the picture of a wall that cannot be crossed, and nothing else: that one constrains a motion, and this one is a term in a cost. The minimizer is $x(\mu) = \mu$. The curve of minimizers as $\mu$ decreases is the **central path**, and following it to $\mu = 0$ is an **interior-point method**. The smooth problems on the way never ask which constraints are active.

The practical form is **primal-dual**. Call the spring force $\nu$, so that $\nu = \mu/x$, and solve for the position and the force together: $1 - \nu = 0$ and $x\,\nu = \mu$. Newton's method on this pair is much better behaved than on $1 - \mu/x = 0$, and $\nu$ becomes the Lagrange multiplier in the limit.

For a general problem, collect the unknowns in a point $x$ (attitudes and vectors), write the equalities as $r(x) = 0$ and the inequalities as $c(x) \le 0$, and add slack variables $w > 0$ with $c(x) + w = 0$. With multipliers $\nu_r$ for the equalities and $\nu_c > 0$ for the inequalities, the conditions for a point of the central path are

$$
\operatorname{grad}\Phi + A_r\T \nu_r + A_c\T \nu_c = 0, \qquad r = 0, \qquad c + w = 0, \qquad w_i\,\nu_{c,i} = \mu, \label{eq13-kkt}
$$

where $A_r$ and $A_c$ are the Jacobians of $r$ and $c$ along the steps, as in [](#eq13-jac) and [](#eq13-cone). At $\mu = 0$ these are the Karush-Kuhn-Tucker conditions, the **KKT conditions**, of the original problem. Newton's method on [](#eq13-kkt), after eliminating the slack and inequality steps, solves

$$
\begin{bmatrix} \operatorname{Hess} + A_c\T \operatorname{diag}(\nu_c/w)\,A_c & A_r\T \\ A_r & 0 \end{bmatrix}
\begin{bmatrix} \delta x \\ \delta\nu_r \end{bmatrix} = \text{residuals}, \label{eq13-newton-kkt}
$$

with $\operatorname{Hess}$ the Hessian of $\Phi + \nu_r\cdot r + \nu_c\cdot c$ along the steps, in the sense of [](#eq13-expansion).

This is the structure of LieIPM (Teng et al., 2026). Only two things differ from an interior-point method in $\R^n$. The primal step $\delta x$ is applied by a retraction, $R_k \leftarrow R_k\exp(\eta_k^\wedge)$. And the entries of $A_r$, $A_c$ and the Hessian are the coefficients of expansions like [](#eq13-expansion). The multipliers and slacks are ordinary vectors and are updated as in any textbook. LieIPM discretizes the rigid body with a variational integrator and makes the rotation over one step a variable in its own right. Every constraint is then a product of group elements, and its first- and second-order expansions follow in closed form from the series for $\log(\exp(x^\wedge)\exp(y^\wedge))$, with Lie brackets as the second-order terms. [](#eq12-constraints) writes that discretization out, and [](#sec12-transcribe) compares it with collocation on one problem.

Ordered by stage, the matrix of [](#eq13-newton-kkt) is banded ([](#fig13-ipm)): each constraint couples one step to the next and nothing else. For [](#eq13-dto) it is $294\times294$ with 3.4% of its entries nonzero and a half bandwidth of 17. The cost of a Newton step grows in proportion to the horizon.

::: {.example #ex13-cone name="Around the cone"}
We implemented [](#eq13-kkt) and [](#eq13-newton-kkt) for the problem [](#eq13-dto), with $\tau_{\max} = 0.15$ N·m. To be precise about what the code is: a primal-dual interior-point method with slack variables; the exponential retraction for attitudes; first derivatives in closed form; the Hessian from symmetrized differences of those derivatives; a step length that keeps $w$ and $\nu_c$ positive, then backtracking on the norm of the residual of [](#eq13-kkt); a shift added to the Hessian when the matrix has the wrong signs of eigenvalues; $\mu$ reduced when the residual falls below $10\mu$; dense linear algebra. It has no line-search filter, no restoration phase and no sparse factorization.

Start from the shortest turn, which violates the cone at 9 of its 19 interior steps. The solver converges in 20 iterations to a residual below $10^{-8}$ ([](#fig13-cone), [](#fig13-ipm)). Every attitude at every iteration is a rotation to $10^{-15}$. The camera axis dips under the cone and grazes it at one step. Nine torque bounds are active. The detour costs 3.4% more effort than the same turn with no cone. SciPy's SLSQP, given the problem in a different form (torques only, the dynamics rolled out), returns the same cost to $10^{-7}$.
:::

![The interior-point method at work on [](#ex13-cone). The first guess sweeps the camera axis $R_k e_1$ through the keep-out cone. Early iterates, with a large barrier parameter $\mu$, stay well clear of it. As $\mu$ shrinks the path tightens until it grazes the cone.](figures/ch13-cone-strip.png){#fig13-cone anim="figures/ch13-cone.gif"}

![Left: the KKT residual of [](#ex13-cone) and the barrier parameter at each iteration. Right: the nonzero entries of the matrix in [](#eq13-newton-kkt), with unknowns ordered by stage.](figures/ch13-ipm.svg){#fig13-ipm}

Follow the central path slowly and the wall softens as [](#thought13-wall) promised. At $\mu = 10^{-2}$ the camera axis keeps 14° of margin from the cone. At $10^{-4}$ it keeps 4.7°, at $10^{-6}$ 0.015°. At fixed $\mu$ the last Newton residuals are $10^{-3}$, $2\times10^{-4}$, $2\times10^{-6}$, $7\times10^{-11}$: faster than linear, as Newton's method should be.

What the full method achieves is reported by Teng et al. (2026). They plan for a quadrotor with motor-force limits, modeled as a full rigid body, over 21 knots. With variational-integrator dynamics and no added noise, LieIPM converged on 999 of 1000 random problems in 22 iterations on average. IPOPT and SNOPT, given quaternion formulations of the same problems, converged on 862 and 920, in 316 and 227 iterations on average, and took 10 to 20 times longer in wall-clock time. The comparison is not one-sided. With an explicit Euler integrator, which lets the quaternion drift off the unit sphere, the shooting solver Crocoddyl converged on all 1000 in about a millisecond, faster than LieIPM. On hardware, LieIPM ran inside a 100 Hz loop with an 8.5 ms time limit and flew a quadrotor through flips. The authors prove fast local convergence under standard conditions and state that they do not claim convergence from arbitrary initial guesses.

## Limits {#sec13-limits}

**Local minima and initialization.** Bulge the initial guess of [](#ex13-cone) the other way and the same solver converges, just as cleanly, to a path over the cone. It costs 16% more. Both pass the second-order test for a minimum. The solver cannot tell you which one you have. A cone on a sphere has two sides, and no local method sees both. The answer depends on the initial guess.

**Time.** A QP with 75 unknowns is solved in a fraction of a millisecond. A nonlinear program is solved in "a few iterations, usually". Real-time use needs a time limit and a plan for when it is hit, as in the hardware experiments above.

**Warm starts.** The QP solver starts each solve from the previous plan, shifted by a step, and usually finds it nearly optimal. An interior-point method gains less from a warm start: the old solution sits on the boundary, and the method wants to start inside.

**Linearization.** Error-state MPC is exactly as local as [](#sec7-local) says, plus the jump of the logarithm at half a turn. With no bounds and a Riccati terminal weight it *is* the LQR of [](#def7-feedback), fed with the error state.

**Guarantees.** Nothing here proves that a constraint will hold on the real system. The bound in [](#ex13-flip) held because the QP enforces it on the input directly. A state constraint enforced on a linearized prediction can be violated by the plant. A safety filter, the program [](#eq8-safety-filter), takes the opposite position: it makes no plan, and changes whatever input it is given by the smallest amount that keeps one inequality true at the next instant. The two are complements. MPC looks ahead with an approximate model. A filter looks at the present with a certificate.

## Exercises {#sec13-exercises .unnumbered}

::: {.exercise #exr13-sphere}
For the mean of directions in [](#sec13-newton), show that the Hessian of $\tfrac12\theta_i^2$ along $q \mapsto \exp(\xi^\wedge)q$ has eigenvalue 1 for $\xi$ along $q_i \times q$ and $\theta_i\cot\theta_i$ for $\xi$ perpendicular to it in the tangent plane. What happens at $\theta_i = 90^\circ$, and beyond? Run Newton's method in the notebook with one reading 150° from the others.
:::

::: {.exercise #exr13-hess}
Derive [](#eq13-mean) from [](#eq13-dexp). You will need $\chi\T\operatorname{dexp}_\chi^{-1} = \chi\T$ and the fact that the Hessian of [](#eq13-expansion) is the symmetric part of the derivative of the gradient. Check in the notebook that the derivative of the gradient is itself not symmetric, and explain where the antisymmetric part comes from.
:::

::: {.exercise #exr13-se3}
Derive the block $A_{22}$ in [](#eq13-se3) by varying the second equation of [](#eq13-ep). Then add linear drag, a force $-c_1 V$ and a torque $-c_2\,\Omega$ with positive constants $c_1, c_2$, and write the new $A_{22}$. Does the reference pose appear?
:::

::: {.exercise #exr13-lqr}
Prove that the first block of $-P_{UU}^{-1}P_{Us}$ in [](#eq13-qp) equals the LQR gain when $P_f$ solves the discrete Riccati equation. Then, in the notebook, set $P_f = 0$ and find the shortest horizon for which the closed loop of [](#ex13-flip) still tracks.
:::

::: {.exercise #exr13-bound}
In [](#ex13-flip), lower $\tau_{\max}$ from 0.8 to 0.4 N·m in steps and record the worst error of the three controllers. At what bound does the saturated LQR match the MPC? At what bound does the MPC lose the reference, and what does the logarithm do when it does?
:::

::: {.exercise #exr13-right}
Repeat [](#thm13-so3) with the right-invariant error $E_r = R\,R_d\T$ and world-frame angular velocities. Which quantities appear in the linear model now? Use [](#thought3-remount) to predict the answer before you compute it.
:::

::: {.exercise #exr13-path}
In the notebook, follow the central path of [](#ex13-cone) from the guess that bulges over the cone. Plot cost against $\mu$ for both paths. Then move $q_c$ toward the equator until the two minima cost the same. What does the solver do from the shortest turn?
:::

## Notes and sources {#sec13-notes .unnumbered}

Optimization on manifolds with retractions, and the convergence theory of Newton's method in that setting, are set out by Absil, Mahony and Sepulchre (2008) and Boumal (2023). The quadratic cost in the Lie algebra, its gradient and the linearized error dynamics are from Teng et al. (2022), and the convex error-state MPC on $\SE(3)$ is from Jang, Teng and Ghaffari (2023). Kotaru (2022, ch. 6) runs an MPC of the same kind on the variation-based linearization of a quadrotor with a cable-suspended load, with a Riccati terminal weight. The exact linearity of the logarithm of an invariant error, met in [](#sec13-convex) for a pure attitude error, is due to Barrau and Bonnabel (2017), and [](#thm14-loglinear) states it for estimation. The projected Newton method is that of Bertsekas (1982). Interior-point methods are treated by Nocedal and Wright (2006), and the line-search implementation that LieIPM follows is that of Wächter and Biegler (2006). LieIPM, its closed-form derivatives and its benchmarks are from Teng et al. (2026). Kotaru (2022, ch. 5) transcribes a quadrotor with a cable-suspended load by writing each attitude as a fixed rotation times $\exp(\eta^\wedge)$ and each direction as $\exp(\xi^\wedge)$ applied to a fixed direction, with $\eta$ and $\xi$ as Euclidean decision variables, and handles the taut and slack cable with a complementarity constraint, which is beyond this chapter ([](#sec12-collocation)). Wu (2017) solves a QP at every instant to enforce safety constraints, the approach of [](#sec8-barrier). Siravuru (2019, ch. 5) derives a variational integrator for the Reaction Mass Biped, which [](#sec12-biped) examines.

- P.-A. Absil, R. Mahony and R. Sepulchre, [*Optimization Algorithms on Matrix Manifolds*](https://press.princeton.edu/absil), Princeton University Press, 2008.
- A. Barrau and S. Bonnabel, ["The invariant extended Kalman filter as a stable observer"](https://doi.org/10.1109/TAC.2016.2594085), *IEEE Transactions on Automatic Control* 62(4), 2017.
- D. P. Bertsekas, ["Projected Newton methods for optimization problems with simple constraints"](https://doi.org/10.1137/0320018), *SIAM Journal on Control and Optimization* 20(2), 1982.
- N. Boumal, [*An Introduction to Optimization on Smooth Manifolds*](https://doi.org/10.1017/9781009166164), Cambridge University Press, 2023.
- J. Jang, S. Teng and M. Ghaffari, ["Convex geometric trajectory tracking using Lie algebraic MPC for autonomous marine vehicles"](https://doi.org/10.1109/LRA.2023.3328450), *IEEE Robotics and Automation Letters* 8(12), 2023. arXiv:2305.09009.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- J. Nocedal and S. J. Wright, [*Numerical Optimization*](https://doi.org/10.1007/978-0-387-40065-5), 2nd ed., Springer, 2006.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- S. Teng, W. Clark, A. Bloch, R. Vasudevan and M. Ghaffari, ["Lie algebraic cost function design for control on Lie groups"](https://arxiv.org/abs/2204.09177), arXiv:2204.09177, 2022.
- S. Teng, R. Zhang, T.-Y. Lin, W. A. Clark, M. Mueller, R. Vasudevan, M. Ghaffari and K. Sreenath, ["LieIPM: Lie group interior point method for direct trajectory optimization of rigid bodies"](https://arxiv.org/abs/2606.10579), arXiv:2606.10579, 2026.
- A. Wächter and L. T. Biegler, ["On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming"](https://doi.org/10.1007/s10107-004-0559-y), *Mathematical Programming* 106(1), 2006.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
