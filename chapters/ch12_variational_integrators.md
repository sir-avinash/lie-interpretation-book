---
title: Variational integrators and collocation
short: Variational integrators
id: ch-integrators
summary: Discretize Hamilton's principle instead of the equations of motion, and the integrator keeps the momentum, the manifold and, very nearly, the energy. The same discrete equations serve as constraints for trajectory optimization.
notebook: notebooks/ch12_variational_integrators.ipynb
verify: verify/ch12.py
---

Hang the 3D pendulum of [](#eq4-pendulum) from its pivot, tip it over by two radians, spin it at six radians per second and let go. It has 7.39 J of energy above rest and nothing to lose it to. Now simulate it for an hour, twice.

The first simulation uses the best method of [](#sec4-integrate): fourth-order Runge-Kutta with the exponential update, at $h = 0.08$ s. It keeps $R$ on $\SO(3)$ to fourteen digits. After ten minutes it has lost 2.3 J. After an hour it has lost 5.57 J, and the pendulum circles on a steady cone 17° from the vertical. The swing is gone.

The second simulation uses a second-order method at $h = 0.02$ s, so that both evaluate the gravity torque equally often. After one second its attitude is off by 0.68°. At equal step sizes fourth-order Runge-Kutta is a thousand times more accurate. Yet after an hour its energy has never been more than 0.027 J from the true value, the same band as in the first minute, and the pendulum is still swinging ([](#fig12-hour)).

![The puzzle of this chapter. The same 3D pendulum, simulated for an hour by two methods that evaluate the gravity torque equally often. Watch the bars: they show the energy above rest. Left: fourth-order Runge-Kutta with the exponential update, $h = 0.08$ s. Right: the second-order variational integrator of [](#sec12-lgvi), $h = 0.02$ s. Both keep $R$ on $\SO(3)$ to rounding. Only one of them still has a swinging pendulum.](figures/ch12-hour-strip.png){#fig12-hour anim="figures/ch12-hour.gif"}

How can the less accurate method be the better one? And if staying on the manifold is not what protects the energy, as [](#fig4-integrators) already showed, what does?

## Three ways to take a step {#sec12-three}

Strip the problem down. A planar pendulum has one angle $\theta$, measured from the hanging position, and momentum $p = m l^2\dot\theta$. Its equations of motion are $\dot\theta = p/(m l^2)$ and $\dot p = -m g l\sin\theta$. Take $m = 1$ kg and $l = 1$ m, release it from rest at 60°, and step with $h = 0.05$ s. The energy above rest is 4.905 J and the period is 2.15 s.

Here are three steps. All three are consistent with the equations. They differ only in *where* the right-hand sides are evaluated.

$$
\begin{aligned}
\textbf{explicit Euler:}\quad & \theta_{k+1} = \theta_k + \frac{h}{m l^2}\,p_k, & p_{k+1} &= p_k - h\,m g l\sin\theta_k ,\\
\textbf{implicit Euler:}\quad & \theta_{k+1} = \theta_k + \frac{h}{m l^2}\,p_{k+1}, & p_{k+1} &= p_k - h\,m g l\sin\theta_{k+1} ,\\
\textbf{Störmer-Verlet:}\quad & \theta_{k+1} = \theta_k + \frac{h}{m l^2}\Big(p_k - \tfrac h2\,m g l\sin\theta_k\Big), & p_{k+1} &= p_k - \tfrac h2\,m g l\,\big(\sin\theta_k + \sin\theta_{k+1}\big) .
\end{aligned} \label{eq12-three}
$$

Run each for ten seconds ([](#fig12-three)). Explicit Euler pumps energy in: the pendulum swings higher each time, goes over the top at 5.4 s, and ends with 28.5 J. Implicit Euler drains it: 0.05 J is left and the swing has shrunk to 8.5°. Störmer-Verlet stays between 4.877 and 4.905 J. Run it for 20 000 s, which is 400 000 steps, and the band is the same 0.028 J. Halve the step and the band shrinks by a factor of 4.00.

![Three pendulums released from 60° and stepped with $h = 0.05$ s by the three rules of [](#eq12-three). The bars show the energy, with the true value marked. Explicit Euler gains energy and goes over the top. Implicit Euler loses it and hangs still. Störmer-Verlet keeps swinging to the same height.](figures/ch12-three-strip.png){#fig12-three anim="figures/ch12-three.gif"}

Plot $p$ against $\theta$ and the difference becomes a picture ([](#fig12-phase)). The true motion is a closed curve. Explicit Euler spirals out, implicit Euler spirals in, and the Störmer-Verlet points lie on a closed curve that is not quite the true one.

![The same three runs in the $(\theta, p)$ plane. The thin gray curve is the true motion. Explicit Euler spirals outward and leaves over the top. Implicit Euler spirals into the origin. The Störmer-Verlet points sit on a closed curve a little inside the true one.](figures/ch12-phase.svg){#fig12-phase}

::: {.thought #thought12-area name="A drop of ink in the phase plane"}
Do not start one pendulum. Start a small blob of them: a patch of initial conditions in the $(\theta, p)$ plane. Let every point of the patch take one step. What happens to the area of the patch?

For small swings, where $\sin\theta \approx \theta$, each step is a $2\times2$ matrix and the area is multiplied by its determinant. For explicit Euler the determinant is $1 + h^2 g/l = 1.0245$. For implicit Euler it is the reciprocal. For Störmer-Verlet it is exactly 1.

The true pendulum has no friction, and a blob of true pendulums keeps its area forever (this is Liouville's theorem). A method that inflates the blob by 2.45% per step has multiplied it by 127 after ten seconds. No trajectory inside it can stay on a closed curve.
:::

A step that preserves area in the phase plane is called **symplectic**. With more degrees of freedom the definition asks for more than volume, but the idea is the same.^[A map $(x, p) \mapsto (x', p')$ with Jacobian $A$ is symplectic when $A\T \Lambda_n A = \Lambda_n$, where $\Lambda_n$ is the block matrix with $I$ and $-I$ off the diagonal. For one degree of freedom this says $\det A = 1$.] For the small swings of the thought experiment you can also say exactly what Störmer-Verlet does to the energy. It conserves, to rounding, the quantity

$$
\frac{p^2}{2 m l^2} + \tfrac12\,m g l\,\Big(1 - \frac{h^2 g}{4 l}\Big)\,\theta^2 . \label{eq12-modified}
$$

This is the energy of a slightly different pendulum, one with gravity reduced by the fraction $h^2 g/4l$. The method does not solve our problem approximately. It solves a neighboring problem exactly. That is why its error does not accumulate.

Why should one way of shuffling the evaluation points have this property and the others not? The third rule of [](#eq12-three) looks like a trick. It is not. It is what Hamilton's principle says, once time is made of steps.

## Discretize the principle, not the equation {#sec12-discrete}

[](#sec4-hamilton) derived the equations of motion in three moves from one statement: the action $S = \int L\,\dd t$ is stationary among paths with the same endpoints. Every integrator we have met takes the output of that derivation, a differential equation, and approximates it. Go one step earlier instead, and approximate the action.

A discrete path is a list of configurations $x_0, x_1, \dots, x_N$ at the times $t_k = k h$. Between two neighbors we know nothing, so we guess: the path is a straight line, covered at constant speed. The action of that piece is then a function of its two ends.

::: {.definition #def12-discrete name="Discrete Lagrangian and action sum"}
A **discrete Lagrangian** $L_h(x_k, x_{k+1})$ approximates the action of the true motion from $x_k$ to $x_{k+1}$ over one time step $h$. The **action sum** of a discrete path is
$$
S_h = \sum_{k=0}^{N-1} L_h(x_k, x_{k+1}) . \label{eq12-action-sum}
$$
The **discrete Hamilton's principle** says: a discrete motion is a discrete path whose action sum is stationary when the interior points $x_1, \dots, x_{N-1}$ are varied and the ends $x_0$, $x_N$ are held.
:::

For a particle of mass $m$ under a force $F(x)$ that comes from a potential, the straight-line guess with the potential averaged over the two ends gives

$$
L_h(x_k, x_{k+1}) = \frac{m}{2h}\,\|x_{k+1} - x_k\|^2 - \frac h2\,\big(\text{potential at } x_k + \text{potential at } x_{k+1}\big) . \label{eq12-Lh-particle}
$$

Now carry out the principle. The point $x_k$ appears in exactly two terms of the sum, the step that arrives and the step that leaves. Setting the derivative of those two terms to zero gives the **discrete Euler-Lagrange equations**

$$
\frac{\partial}{\partial x_k}\Big[L_h(x_{k-1}, x_k) + L_h(x_k, x_{k+1})\Big] = 0, \qquad k = 1, \dots, N-1 . \label{eq12-del}
$$

For [](#eq12-Lh-particle) this reads $m\,(x_{k+1} - 2x_k + x_{k-1}) = h^2 F(x_k)$. No limit was taken and no differential equation was approximated. Two neighbors determine the third, and the path can be marched forward.

::: {.thought #thought12-knot name="Bump one knot"}
Return to the ball of [](#thought4-ball): 1 kg, thrown to land one second later. Describe its flight by six heights, $h = 0.2$ s apart. The discrete Euler-Lagrange equations give the heights 0, 0.785, 1.177, 1.177, 0.785, 0 m. Check them against the parabola: every one lies on it exactly ([](#fig12-path)).

Now raise the third height by 10 cm and leave the others alone. Two straight pieces get steeper, and one point sits higher. The first-order changes cancel, as they did for the continuous path, and the action sum rises by $m\epsilon^2/h = 0.05$ J·s. Lower the same height by 10 cm and it rises by the same amount. The discrete path sits at the bottom of a bowl too. The bowl has four dimensions, one per interior height.
:::

![The discrete Hamilton's principle for a thrown ball. Left: the discrete path, six heights joined by straight pieces, with its third height moved up and down. The gray curve is the parabola. Right: the action sum as that one height is moved. It is stationary where the height lies on the parabola.](figures/ch12-path.svg){#fig12-path}

A march from two positions is awkward to start. A mechanical system is started from a position and a momentum. The principle supplies the momentum as well. Define the **discrete momentum** at $x_k$ by the two slopes of the discrete Lagrangian:

$$
p_k = -\frac{\partial L_h}{\partial x_k}(x_k, x_{k+1}), \qquad p_{k+1} = \frac{\partial L_h}{\partial x_{k+1}}(x_k, x_{k+1}) . \label{eq12-legendre}
$$

Equation [](#eq12-del) says that the two definitions agree at every $x_k$ between the ends: the momentum with which one step arrives is the momentum with which the next one leaves. So a step is: solve the first equation of [](#eq12-legendre) for $x_{k+1}$, then evaluate the second. An integrator built this way is a **variational integrator**.

Do it for the pendulum, with $L_h = \tfrac{m l^2}{2h}(\theta_{k+1} - \theta_k)^2 + \tfrac h2\,m g l\,(\cos\theta_k + \cos\theta_{k+1})$. The first equation gives $p_k = \tfrac{m l^2}{h}(\theta_{k+1} - \theta_k) + \tfrac h2\,m g l\sin\theta_k$. The second gives $p_{k+1} = \tfrac{m l^2}{h}(\theta_{k+1} - \theta_k) - \tfrac h2\,m g l\sin\theta_{k+1}$. Solve the first for $\theta_{k+1}$ and substitute. The result is the third line of [](#eq12-three), term for term. Störmer-Verlet is the discrete Hamilton's principle for the simplest discrete Lagrangian there is.

That explains where the rule comes from. It does not yet explain the closed curve in [](#fig12-phase).

## What a discrete principle conserves {#sec12-conserve}

[](#sec4-conservation) asked of every model what it must conserve. Ask the same of the discrete model. Three facts hold for every variational integrator, whatever the system and whatever the step size.

::: {.theorem #thm12-vi name="Structure of a variational integrator"}
Let $(x_k, p_k) \mapsto (x_{k+1}, p_{k+1})$ be the step defined by [](#eq12-legendre) for a discrete Lagrangian $L_h$.

1. *The step is symplectic.*
2. *Discrete Noether theorem.* Suppose a one-parameter family of transformations $x \mapsto g_\epsilon(x)$, with $g_0$ the identity, leaves the discrete Lagrangian unchanged: $L_h(g_\epsilon(x_k), g_\epsilon(x_{k+1})) = L_h(x_k, x_{k+1})$. Write $\delta x_k$ for the derivative of $g_\epsilon(x_k)$ at $\epsilon = 0$. Then $p_k\cdot\delta x_k$ has the same value at every $k$.
3. *Order.* If $L_h$ differs from the action of the true motion between its two arguments by $O(h^{r+1})$, the step has order $r$.
:::

::: {.proof}
For the second claim, differentiate the invariance at $\epsilon = 0$. The chain rule and [](#eq12-legendre) give $-p_k\cdot\delta x_k + p_{k+1}\cdot\delta x_{k+1} = 0$. The first and third claims are proved by Marsden and West (2001).
:::

The second claim deserves a second look, because it is exact. If the discrete Lagrangian does not change when the whole system is turned about an axis, the angular momentum about that axis is conserved to rounding error, at any step size, forever. For a particle in a central force stepped by Störmer-Verlet, the angular momentum after 100 000 steps is off by $3\times10^{-14}$.

Energy is missing from the list, and it has to be. The symmetry behind energy is a shift in time, and a fixed grid of times $t_k$ has broken it. What survives is the phenomenon of [](#eq12-modified). For a symplectic method of order $r$ with a fixed step, applied to a system with an analytic energy function, the numerical solution conserves a *modified* energy up to terms that are exponentially small in $1/h$. As long as the solution stays in a bounded region, the true energy therefore stays within $O(h^r)$ of its initial value over times of order $e^{c/h}$, for some constant $c > 0$ that depends on the system (Hairer, Lubich and Wanner 2006, Theorem IX.8.1). The energy oscillates in a band. It does not drift.

Now the puzzle of the opening can be stated precisely. Fourth-order Runge-Kutta is not symplectic. Its energy error per step is tiny, $O(h^5)$, but nothing stops the errors from adding up, and in this example they all have the same sign. Störmer-Verlet has a larger error at every instant, $O(h^2)$, and it stays that size. Over a short horizon the first is better. Over a long one the second wins, and the crossover comes sooner the larger the step.

So far $x$ has been a vector. Our pendulum's configuration is a rotation matrix.

## The rigid body, one rotation at a time {#sec12-lgvi}

Take the 3D pendulum: attitude $R \in \SO(3)$, inertia $J$ about the pivot, center of mass at $R\rho$, and

$$
L = \tfrac12\,\Omega\cdot J\Omega - m g\,e_3\cdot R\rho .
$$

A discrete path is a list of attitudes $R_0, \dots, R_N$. The first thing to decide is what replaces the velocity. On flat ground it was the difference $x_{k+1} - x_k$. Two rotations cannot be subtracted, but one can be divided by the other.

::: {.definition #def12-step name="Step rotation"}
The **step rotation** of the step from $R_k$ to $R_{k+1}$ is
$$
\Upsilon_k = R_k\T R_{k+1} \in \SO(3), \qquad\text{so that}\qquad R_{k+1} = R_k\,\Upsilon_k . \label{eq12-step}
$$
Its rotation vector divided by $h$ is the **step velocity** $\Omega_k$: $\ \Upsilon_k = \exp(h\,\Omega_k^\wedge)$.
:::

The step velocity is the constant body angular velocity that would carry $R_k$ to $R_{k+1}$ in the time $h$. The update $R_{k+1} = R_k\exp(h\,\Omega_k^\wedge)$ is the exponential step [](#eq2-exp-step) again, and the kinematic constraint of [](#eq13-dto). What is new is that nobody hands us $\Omega_k$. The principle has to produce it.

*The discrete Lagrangian.* Write the kinetic energy with matrices. With the symmetric matrix $\bar J = \tfrac12\tr(J)\,I - J$,

$$
\tfrac12\,\Omega\cdot J\Omega = \tfrac12\tr\!\big(\Omega^\wedge\,\bar J\,(\Omega^\wedge)\T\big) .
$$

Over one step, replace $\Omega^\wedge = R\T\dot R$ by the difference quotient $(\Upsilon_k - I)/h$, as the straight line did on flat ground. Since $\Upsilon_k$ is a rotation, $\tfrac12\tr\big((\Upsilon_k - I)\bar J(\Upsilon_k - I)\T\big) = \tr\big((I - \Upsilon_k)\bar J\big)$, and

$$
L_h(R_k, R_{k+1}) = \frac1h\,\tr\!\big((I - \Upsilon_k)\,\bar J\big) - \frac h2\,m g\,e_3\cdot\big(R_k\rho + R_{k+1}\rho\big) . \label{eq12-Lh-so3}
$$

*Vary.* The interior attitudes are varied the way [](#sec4-so3) varied a curve, by rotating them: $\delta R_k = R_k\,\eta_k^\wedge$ with $\eta_0 = \eta_N = 0$. The step rotation then varies as

$$
\delta\Upsilon_k = -\eta_k^\wedge\,\Upsilon_k + \Upsilon_k\,\eta_{k+1}^\wedge . \label{eq12-var-step}
$$

This is the discrete form of $\delta\Omega = \dot\eta + \Omega\times\eta$ in [](#eq4-var-omega). The two terms do not combine into a difference $\eta_{k+1} - \eta_k$, because $\Upsilon_k$ stands between them. Rotations do not commute, and that will again be the source of the gyroscopic term.

*Collect.* Two identities do the bookkeeping: $\tr(x^\wedge A) = -x\cdot(A - A\T)^\vee$ for any matrix $A$, and the variation of the potential from [](#eq4-pendulum), $\delta\big(m g\,e_3\cdot R\rho\big) = -\tau\cdot\eta$ with the gravity torque $\tau = m g\,(R\T e_3)\times\rho$. Inserting [](#eq12-var-step) into [](#eq12-Lh-so3),

$$
\delta L_h(R_k, R_{k+1}) = -\eta_k\cdot\Big[\tfrac1h\big(\Upsilon_k\bar J - \bar J\,\Upsilon_k\T\big)^\vee - \tfrac h2\,\tau_k\Big] + \eta_{k+1}\cdot\Big[\tfrac1h\,\Upsilon_k\T\big(\Upsilon_k\bar J - \bar J\,\Upsilon_k\T\big)^\vee + \tfrac h2\,\tau_{k+1}\Big] .
$$

The first bracket is the discrete momentum $p_k$ of [](#eq12-legendre), now a vector in the body frame. The second is $p_{k+1}$. No integration by parts is needed. Its discrete counterpart is a shift of the index, which is already done.

*Use that $\eta_k$ is arbitrary.* The first bracket is an equation for the step rotation. Insert $\Upsilon_k = \exp(h\,\Omega_k^\wedge)$ and Rodrigues' formula ([](#prop2-rodrigues)), and it becomes an equation between vectors.

::: {.proposition #thm12-lgvi name="Lie group variational integrator for the rigid body"}
Let $\tau_k$ be the total torque in the body frame at $R_k$: the gravity torque $m g\,(R_k\T e_3)\times\rho$ plus any applied torque. The discrete Euler-Lagrange equations of [](#eq12-Lh-so3) are
$$
\begin{aligned}
p_k + \tfrac h2\,\tau_k &= \frac{\sin\theta_k}{\theta_k}\,J\Omega_k + \frac{1 - \cos\theta_k}{\theta_k^2}\;h\,\Omega_k\times J\Omega_k , \qquad \theta_k = h\,\|\Omega_k\| ,\\
R_{k+1} &= R_k\exp\!\big(h\,\Omega_k^\wedge\big) ,\\
p_{k+1} &= \exp\!\big(-h\,\Omega_k^\wedge\big)\big(p_k + \tfrac h2\,\tau_k\big) + \tfrac h2\,\tau_{k+1} .
\end{aligned} \label{eq12-lgvi}
$$
The first line is solved for the step velocity $\Omega_k$. The other two are explicit.
:::

Read the three lines as kick, turn, kick. Half the torque impulse is added to the momentum. The body then turns freely by the step rotation. The body-frame momentum is carried along by $\Upsilon_k\T$, because the same world-frame vector has new components in the turned body. Then the other half of the impulse arrives. An applied torque enters as in [](#def4-dalembert), with the virtual work of each step split between its two ends, $\tfrac h2\,\tau_k\cdot\eta_k + \tfrac h2\,\tau_{k+1}\cdot\eta_{k+1}$.

For small $h$ the first line reads $p_k + \tfrac h2\tau_k = J\Omega_k + \tfrac h2\,\Omega_k\times J\Omega_k + O(h^2)$. Substitute into the third and divide by $h$: it is $J\dot\Omega + \Omega\times J\Omega = \tau$. The gyroscopic term came from carrying the momentum through the step rotation, with nothing added by hand. To first order in $h$ the three lines reduce to the two dynamics constraints of [](#eq13-dto).

Multiply the third line by $R_{k+1}$:

$$
R_{k+1}\,p_{k+1} = R_k\,p_k + \tfrac h2\,\big(R_k\tau_k + R_{k+1}\tau_{k+1}\big) . \label{eq12-world}
$$

This is Newton-Euler in the world frame, [](#eq4-newton-euler), integrated by the trapezoidal rule. With no torque, the world-frame angular momentum $R_k p_k$ is the same vector at every step. That is the discrete Noether theorem at work: the kinetic part of [](#eq12-Lh-so3) does not change when $R_k$ and $R_{k+1}$ are both multiplied on the left by any rotation. Gravity breaks that symmetry except for rotations about $e_3$, and what remains is that $e_3\cdot R_k p_k$ is conserved, exactly as for the continuous pendulum ([](#exr4-heavy-top)).

*Solving the implicit line.* The unknown $\Omega_k$ is a vector in $\R^3$, the Lie algebra, so Newton's method needs no special care. Start from $\Omega_k = J^{-1}(p_k + \tfrac h2\tau_k)$ and use the Jacobian of the second-order expansion, $J + \tfrac h2\big(\Omega_k^\wedge J - (J\Omega_k)^\wedge\big)$. Over the hour of the opening this took 5.6 iterations per step on average and never more than 7, to a relative tolerance of $10^{-14}$. Each iteration is a $3\times3$ solve and needs no new torque evaluation.

::: {.example #ex12-hour name="The hour, explained"}
For the pendulum of the opening, $J = \operatorname{diag}(0.20, 0.25, 0.10)$ kg·m², $m = 1$ kg, $\rho = -0.4\,e_3$ m, started at $R_0 = \exp(2\,e_1^\wedge)$ with $\Omega = (0, 0.5, 6)$ rad/s. Stepping [](#eq12-lgvi) at $h = 0.02$ s for 180 000 steps:

- $\|R\T R - I\| = 7\times10^{-14}$ at the end;
- $e_3\cdot R_k p_k$ never moves by more than $2\times10^{-14}$ N·m·s;
- the energy $\tfrac12\,p_k\cdot J^{-1}p_k + m g\,e_3\cdot R_k\rho$ stays within 0.0271 J of its initial value, against 0.0269 J in the first minute.

At $h = 0.04$, 0.02 and 0.01 s the band is 0.109, 0.027 and 0.0067 J: it scales as $h^2$ ([](#fig12-energy)). Runge-Kutta with the exponential update at $h = 0.08$ s also stays on $\SO(3)$, but it loses 5.57 J, and $e_3\cdot R J\Omega$, which nothing can change, moves by 0.74 N·m·s.
:::

![Energy error of the 3D pendulum over one hour, on logarithmic axes. The two Runge-Kutta runs drift in proportion to time until, at $h = 0.08$ s, there is no swing left to lose. The variational integrator at $h = 0.02$ s stays inside a band that is as wide after an hour as after a minute.](figures/ch12-energy.svg){#fig12-energy}

The comparison has to be read honestly, in both directions. [](#tbl12-accuracy) gives the attitude error after one second. At equal steps Runge-Kutta wins by three orders of magnitude, and at $h = 0.02$ s its energy drift after the full hour, 0.016 J, is still inside the variational band. If you need an accurate trajectory over a few seconds and can afford the evaluations, use the fourth-order method. The variational integrator is for long horizons, coarse steps, and any use in which a model that leaks energy or momentum gives a qualitatively wrong answer.

::: {.table #tbl12-accuracy}

| $h$ (s) | 0.08 | 0.04 | 0.02 | 0.01 |
| :-- | :-- | :-- | :-- | :-- |
| Runge-Kutta, exponential update | 7.9° | 0.012° | 0.00067° | 0.00004° |
| Variational, [](#eq12-lgvi) | 5.2° | 2.8° | 0.68° | 0.17° |

: Attitude error of the 3D pendulum after 1 s. The measured orders are 4.06 and 2.01.

:::

*The free body.* Without gravity [](#eq12-lgvi) is the discrete rigid body of Moser and Veselov (1991), and it does something no general theorem promises. Tumble the book of [](#ex4-book) about its middle axis with $h = 0.01$ s for 100 s. Its spin reverses 65 times, the first time at 0.88 s as in [](#ex4-book). The world-frame momentum $R_k p_k$ holds to $10^{-16}$ N·m·s, and the energy to a relative $6\times10^{-12}$: rounding, not a band. Every $p_k$ lies on the curve of [](#fig4-polhodes) that the true motion follows. The whole error of the method is in *when* the body gets to each point. This is a special property of the free rigid body, whose motion has enough conserved quantities to pin the curve down. Add gravity and the energy goes back to oscillating.

*The sphere.* The same recipe works on $S^2$. For the spherical pendulum of [](#eq4-sph), take $L_h = \tfrac{m l^2}{2h}\|q_{k+1} - q_k\|^2 - \tfrac h2\,m g l\,e_3\cdot(q_k + q_{k+1})$ and vary by $\delta q_k = \xi_k\times q_k$ as in [](#eq4-var-s2). The discrete momentum $p_k$ is the angular momentum about the pivot, in the world frame, with $p_k\cdot q_k = 0$, and the step is explicit:

$$
\begin{aligned}
\bar p_k &= p_k - \tfrac h2\,m g l\; q_k\times e_3 ,\\
q_{k+1} &= \frac{h}{m l^2}\,\bar p_k\times q_k + \sqrt{1 - \Big\|\frac{h}{m l^2}\,\bar p_k\Big\|^2}\;\, q_k ,\\
p_{k+1} &= \bar p_k - \tfrac h2\,m g l\; q_{k+1}\times e_3 .
\end{aligned} \label{eq12-s2}
$$

Kick, turn, kick again. The second line has unit length by Pythagoras, since $\bar p_k\times q_k$ is perpendicular to $q_k$. Over 100 000 steps at $h = 0.01$ s, $\|q\|^2 - 1$ stays below $10^{-13}$, $e_3\cdot p_k$ does not change in a single bit, and the energy stays within 0.0012 J of 5.79 J.

One body, one sphere. A legged robot has three bodies.

## A biped made of three rotations {#sec12-biped}

Siravuru's Reaction Mass Biped ([](#sec4-products)) has a stance leg, a torso and a swing leg, each with an attitude in $\SO(3)$, plus the leg length and the positions of three masses that slide inside the torso. Chapter 5 of his thesis carries the recipe of this chapter through for that product of manifolds. Each body gets its own step rotation (written $F_{1k}$, $F_{Pk}$, $F_{2k}$ there), the discrete Lagrangian is $h$ times the Lagrangian evaluated at the step velocities, the three attitudes are varied by $\delta R_k = R_k\,\eta_k^\wedge$ and the lengths in the ordinary way, and the joint torques enter through the discrete virtual work. Setting the coefficient of each variation to zero gives one momentum equation per body. For the stance leg, with the configuration-dependent inertia $J(l)$ of [](#ex4-leg), it reads

$$
J(l_{k+1})\,\Omega_{k+1} = \Upsilon_k\T\,J(l_k)\,\Omega_k + h\,\tau_{k+1}, \qquad \Upsilon_k = \exp\!\big(h\,\Omega_k^\wedge\big) , \label{eq12-rmb}
$$

where $\tau_{k+1}$ collects gravity, the ankle torque and the reaction of the hip. The torso and swing-leg equations have the same form. You can see [](#eq12-world) in it: momentum carried through the step rotation, plus an impulse.

The thesis runs this integrator at $h = 10^{-3}$ s inside the closed loop of a walking controller. It reports that $\|I - R\T R\|$ stays within $10^{-12}$ for the legs, orders of magnitude better than an adaptive Runge-Kutta method, and that tracking errors and energies follow the continuous model at least as accurately.

One detail matters to anyone who reuses the scheme. In [](#eq12-rmb) the step rotation is computed from the current velocity, so the update is explicit. That comes from varying the step rotation as $\delta\exp(h\,\Omega_k^\wedge) = h\,\delta\Omega_k^\wedge\exp(h\,\Omega_k^\wedge)$, which holds to first order in $h$. The exact variation carries the derivative of the exponential, [](#eq13-dexp), and leads to an implicit equation, as [](#eq12-lgvi) did.

::: {.warning #warn12-explicit name="On the group and momentum-preserving is not yet variational"}
We ran the explicit update [](#eq12-rmb) on the uncontrolled 3D pendulum of [](#ex12-hour), with fixed inertia. It keeps $R$ on $\SO(3)$ to $10^{-14}$ and $e_3\cdot R_k p_k$ to $10^{-15}$, as it must: both follow from the form of the update alone. But the energy grows steadily: by 108 J in one minute at $h = 0.02$ s, by 50 J at $h = 0.01$ s, and by 2.5 J at $h = 10^{-3}$ s. The drift is proportional to $h$. Replace the explicit step rotation by the solution of the implicit equation, changing nothing else, and the energy stays inside a band of 0.26 J.

For the free body the same explicit update keeps $\|p_k\|$ exactly and still walks across the level curves of the energy, gaining energy at every step ([](#fig12-sphere)). Inside a feedback loop that dissipates energy, over a few seconds, the drift is hidden. In an open-loop simulation or a long-horizon plan it is not. Test energy on an uncontrolled run before you trust any integrator that claims to be variational.
:::

![Body angular momentum of the book of [](#ex4-book) on its sphere, with no torque, stepped at $h = 0.02$ s. Both methods keep the length of the momentum to rounding, so both stay on the sphere. The variational points lie on the closed curve of constant energy through the start. The explicit exponential update crosses the level curves and winds in toward $e_1$, the axis of smallest inertia, where the energy is largest.](figures/ch12-sphere.svg){#fig12-sphere}

An integrator marches forward from an initial state. A planner does not know the initial input, the final state is given, and everything in between is unknown at once.

## Dynamics as constraints: collocation {#sec12-collocation}

[](#sec13-plan) introduced direct transcription: make the states and inputs at a grid of times the unknowns of an optimization problem, and impose the dynamics as equality constraints between neighbors. The grid points are called **knots**. The solver does not march. It adjusts all knots together until every constraint holds. So the question changes from "how do I take a step?" to "what equation should tie $x_k$ to $x_{k+1}$?"

**Collocation** answers with a polynomial. Take a scalar model that assigns a slope $\dot x$ to every state and input, and write $\dot x_k$ for that slope at the knot values $(x_k, u_k)$. Join $x_k$ and $x_{k+1}$ by a polynomial and demand that it obey the model at chosen points, the collocation points. What is left over is the **defect**, and the constraint is that the defect vanishes.

The simplest choice is a parabola whose slope is right at both ends. That is **trapezoidal collocation**:

$$
x_{k+1} - x_k - \tfrac h2\,\big(\dot x_k + \dot x_{k+1}\big) = 0 . \label{eq12-trap}
$$

One step up, a cubic matches the values and slopes at both ends. Its midpoint is $x_{k+1/2} = \tfrac12(x_k + x_{k+1}) + \tfrac h8(\dot x_k - \dot x_{k+1})$. Evaluate the model there, ask the cubic's slope to agree, and you get **Hermite-Simpson collocation** ([](#fig12-colloc)):

$$
x_{k+1} - x_k - \tfrac h6\,\big(\dot x_k + 4\,\dot x_{k+1/2} + \dot x_{k+1}\big) = 0 . \label{eq12-hs}
$$

Insert the true solution of $\dot x = x(1 - x)$ with $h = 0.1$: the trapezoidal defect is $4\times10^{-6}$ and falls as $h^3$, the Hermite-Simpson defect is $9\times10^{-10}$ and falls as $h^5$. These are the local errors of a second- and a fourth-order method.

![Collocation on a scalar model. Left: the cubic through two knots with the model's slopes at both ends. At the midpoint the model asks for one slope and the cubic has another. The difference is the Hermite-Simpson defect. Right: when the knot values lie on a solution, the defect is zero to $O(h^5)$ and the cubic lies on top of it.](figures/ch12-colloc.svg){#fig12-colloc}

Now put a rotation matrix where $x$ was. Every formula above adds states and multiplies them by numbers.

::: {.thought #thought12-chord name="Halfway between two attitudes"}
A body points along $e_1$ at one knot and along $e_2$ at the next. Where does the interpolating polynomial put it halfway?

Entry by entry, at $\tfrac12(e_1 + e_2)$. That vector has length $0.707$. The body's nose has shrunk by 29% ([](#fig12-chord)). Do the same with the two rotation matrices, 90° apart: their average has singular values 1, 0.707 and 0.707 and determinant $\tfrac12$. It is not a rotation, and no torque made it so. The straight line between two points of a curved set leaves the set.
:::

![Entrywise interpolation on the sphere. The chord between two knots $q_k$ and $q_{k+1}$ cuts through the inside: its midpoint has length 0.707. The arc on the sphere is what a step $\exp(h\,\omega_k^\wedge)\,q_k$ follows.](figures/ch12-chord.svg){#fig12-chord}

At the knots things look better than they are. The trapezoidal constraint applied to the nine entries of $R$ returns a rotation at $R_{k+1}$ only if $\Omega_k = \Omega_{k+1}$. Otherwise each knot matrix is stretched by an amount of order $h^2\|\Omega\|^2$. People then add $R_k\T R_k = I$ as six more constraints per knot, and the solver has to hold the trajectory on the manifold by force.

The sources of this book use two remedies.

*Optimize in a chart.* Kotaru (2022, ch. 5) writes each attitude and each cable direction through the exponential,

$$
R = R_0\,\exp(\eta^\wedge), \qquad q = \exp(\xi^\wedge)\,q_0 , \label{eq12-chart}
$$

with $\eta, \xi \in \R^3$ as the decision variables. Whatever values the solver tries, $R$ is a rotation and $q$ a unit vector, and any collocation scheme for vectors applies. The price is in the kinematics. The thesis uses $\dot\eta \approx \Omega$ and $\dot\xi \approx \omega$ and says plainly that these are first-order approximations, which a tracking controller must absorb. The exact relation is $\dot\eta = \operatorname{dexp}_\eta^{-1}\,\Omega$, and it fails where the chart does, at a full turn $\|\eta\| = 2\pi$.

*Do not collocate the rotations at all.* For the quadrotor with a cable-suspended load, the same chapter then goes further. The load position is a flat output ([](#ch-flatness)): the cable direction, the attitude and the inputs are all functions of $x_L$ and its derivatives. So the planner collocates only a chain of integrators, the derivatives of $x_L$ up to the sixth, with the trapezoidal rule [](#eq12-trap), and recovers the cable from one algebraic constraint per knot, $-T\,q = m_L(\ddot x_L + g\,e_3)$. A complementarity condition between the tension and the slack length lets the cable go slack without a prescribed sequence of modes, which is beyond this chapter ([](#warn4-slack)). Obstacles enter as smooth constraints obtained by duality. The thesis reports plans with 50 to 100 knots, solved with the solver IPOPT in under a second for hover-to-hover flight, waypoints and a payload throw, and flown on a 0.825 kg quadrotor with a 65 g load on a 1.1 m cable. With random box obstacles and an initial guess from a graph search, 20 of 20 trials succeed up to six obstacles, and 10 of 20 with twelve.

Flatness is a gift that few systems offer, and a single chart has an edge. The third remedy needs neither.

## Dynamics as constraints: the variational way {#sec12-transcribe}

Look at [](#eq12-lgvi) with the eyes of an optimizer. It is already a set of algebraic equations between neighboring knots. Use it as the constraint.

Make the step velocity $\Omega_k$ an unknown in its own right, next to $R_k$ and $\tau_k$. Eliminating the momentum between two consecutive steps of [](#eq12-lgvi) leaves, at every interior knot,

$$
\begin{aligned}
\log\!\big(R_{k+1}\T R_k\exp(h\,\Omega_k^\wedge)\big)^\vee &= 0 ,\\
\big(\Upsilon_k\bar J - \bar J\,\Upsilon_k\T\big)^\vee - \Upsilon_{k-1}\T\big(\Upsilon_{k-1}\bar J - \bar J\,\Upsilon_{k-1}\T\big)^\vee - h^2\,\tau_k &= 0 , \qquad \Upsilon_k = \exp\!\big(h\,\Omega_k^\wedge\big) .
\end{aligned} \label{eq12-constraints}
$$

The first line is the kinematic constraint of [](#eq13-dto), unchanged. The second replaces its explicit Euler line. This is the discretization of LieIPM (Teng et al. 2026), which keeps the step rotation itself as the variable and writes the second line as $\Upsilon_{k+1}\bar J - \bar J\,\Upsilon_{k+1}\T = \bar J\,\Upsilon_k - \Upsilon_k\T\bar J + h^2\tau^\wedge$. Three things recommend it. Every constraint is a product of group elements, so a solver that steps by a retraction ([](#def13-retraction)) never leaves the manifold and no constraint is spent on staying there. There is no chart and so no edge. And every candidate the constraints accept obeys the momentum balance [](#eq12-world) exactly, at any $h$.

::: {.example #ex12-opt name="One turn, four transcriptions"}
Turn the body of [](#ex12-hour), without gravity, from rest at $R = I$ to rest at 150° about the axis $(1,1,1)/\sqrt3$, in $t_f = 2$ s, with $N = 20$ steps of $h = 0.1$ s. Minimize $\tfrac h2\sum\|\tau_k\|^2$ subject to $|\tau_{k,i}| \le \tau_{\max} = 0.5$ N·m. We transcribed the dynamics four ways and gave each to the same general-purpose sequential quadratic programming solver (SciPy's SLSQP) with the same initial guess: a constant-rate turn and zero torque. Then we flew each torque sequence, joined linearly between knots, through Runge-Kutta at $h/200$, and measured how far from the target it ends ([](#tbl12-opt), [](#fig12-opt)).
:::

::: {.table #tbl12-opt}

| Transcription | Iterations | Cost | $\max_k\|R_k\T R_k - I\|$ | Miss on replay |
| :-- | :-- | :-- | :-- | :-- |
| Trapezoid on the entries of $R$ | 188 | 0.243 | 0.012 | 0.74° |
| Trapezoid in the chart, $\dot\eta \approx \Omega$ | 53 | 0.205 | $10^{-15}$ | 14.8° |
| Trapezoid in the chart, exact kinematics | 62 | 0.161 | $10^{-15}$ | 0.45° |
| Variational, [](#eq12-constraints) | 60 | 0.157 | $10^{-15}$ | 1.44° |

: A 150° rest-to-rest turn transcribed four ways. The bound on the torque is active in all four solutions.

:::

Read the table without favoritism. The entrywise transcription works worst. Its knot matrices are off the group by 1.2%. Six of its 264 equality constraints are dependent at the solution, because nine end conditions pin down three degrees of freedom. The solver needs three times the iterations and stops at a visibly worse cost. The first-order chart is fast and misses by 15°: its plan is a plan for a different body, as its author warned. The exact chart and the variational constraints agree on the cost to 3%, take about the same number of iterations, and both miss by about a degree, which is what second order at $h = 0.1$ s buys. On this problem the variational plan is *not* the more accurate of the two.

![The torques found by the exact-chart and the variational transcriptions, and the attitude error when each plan is replayed by a fine integrator. Both plans saturate the bound at the start and at the end. Both drift from their own knots by about a degree.](figures/ch12-opt.svg){#fig12-opt}

What the variational transcription buys is structure, not digits. Its solution satisfies the world-frame momentum balance [](#eq12-world) to $10^{-15}$, at this step size or any other. It has 123 unknowns where the chart has 189. And it has no chart to re-center when a motion goes through a full turn. At scale these are the properties that matter: Teng et al. (2026) report that LieIPM, on this discretization, converged on 999 of 1000 random quadrotor problems where general-purpose solvers on quaternion models converged on 862 and 920 ([](#sec13-ipm)).

## What this does not give you {#sec12-limits}

**Implicit steps cost.** [](#eq12-lgvi) needs a Newton solve per body per step. It is cheap, and it is not free. Variants built on the Cayley map in place of the exponential avoid the trigonometric functions.

**Second order, unless built otherwise.** The discrete Lagrangians of this chapter make second-order methods. Higher order comes from better quadrature of the action or from composing steps, at the price of more stages.

**Fixed step.** The energy argument of [](#sec12-conserve) assumes a constant $h$. Change the step from one step to the next with a standard error controller and the modified energy changes with it: the band becomes a drift again. Adaptive variational integrators exist, and they have to be built on purpose.

**Forces and dissipation.** With an applied torque or drag there is no energy to conserve and the map is no longer symplectic. What survives is a correct accounting: the momentum changes by exactly the impulses in [](#eq12-world). That is still more than a general method promises, and it is why forced variational integrators behave well in control.

**Impacts and contact.** A foot that strikes the ground, or a cable that snaps taut, makes the velocity jump. The discrete principle extends to that case by letting the time of impact be varied too. This chapter does not cover it.

**The knots are not the trajectory.** A transcription constrains the knots. What the system does between them is decided by the real dynamics, and [](#tbl12-opt) shows how far apart the two can be. Replay every plan with an integrator finer than the one that produced it.

A plan is a reference. [](#ch-mpc) is about following one.

## Exercises {#sec12-exercises .unnumbered}

::: {.exercise #exr12-symplectic-euler}
Take the one-sided discrete Lagrangian $L_h = \tfrac{m l^2}{2h}(\theta_{k+1} - \theta_k)^2 + h\,m g l\cos\theta_{k+1}$ for the planar pendulum. Derive its step from [](#eq12-legendre). Show that it preserves area and is first order. In the notebook, compare its energy band with that of Störmer-Verlet at $h = 0.05$ s.
:::

::: {.exercise #exr12-modified}
For the linear oscillator $\ddot\theta = -(g/l)\,\theta$, write the Störmer-Verlet step as a $2\times2$ matrix. Prove that it conserves [](#eq12-modified). For which step sizes is the conserved quantity positive definite, and what happens to the numerical solution beyond that?
:::

::: {.exercise #exr12-vector-form}
Starting from $\tfrac1h\big(\Upsilon_k\bar J - \bar J\,\Upsilon_k\T\big)^\vee = p_k + \tfrac h2\tau_k$ with $\Upsilon_k = \exp(h\,\Omega_k^\wedge)$, use Rodrigues' formula and the identity $x^\wedge\bar J + \bar J x^\wedge = (Jx)^\wedge$ to derive the first line of [](#eq12-lgvi). Then show that $\big(\bar J\,\Upsilon - \Upsilon\T\bar J\big)^\vee = \Upsilon\T\big(\Upsilon\bar J - \bar J\,\Upsilon\T\big)^\vee$.
:::

::: {.exercise #exr12-noether}
Use [](#eq12-world) to show that the Lie group variational integrator conserves $e_3\cdot R_k p_k$ for the 3D pendulum. Then obtain the same result from the discrete Noether theorem of [](#thm12-vi), with $g_\epsilon(R) = \exp(\epsilon\,e_3^\wedge)\,R$. What is $\eta_k$ for this family?
:::

::: {.exercise #exr12-s2}
Derive [](#eq12-s2) from the discrete Lagrangian on the sphere given in [](#sec12-lgvi) and the variation $\delta q_k = \xi_k\times q_k$. Show that $e_3\cdot p_k$ is conserved exactly. What limits the step size?
:::

::: {.exercise #exr12-explicit}
In the notebook, implement the explicit update [](#eq12-rmb) for the 3D pendulum of [](#ex12-hour). Measure the energy after one minute at four step sizes and confirm that the drift is proportional to $h$. Then repeat for a pendulum released from rest, with no spin. What changes, and why does a spin make the difference visible?
:::

::: {.exercise #exr12-transcribe}
Repeat [](#ex12-opt) with $N = 40$. By what factor does the miss on replay fall for the exact-chart and the variational transcriptions? Then ask the chart transcription for a target $350^\circ$ away about the same axis, with a constant-rate initial guess the long way round, and describe what the solver does.
:::

## Notes and sources {#sec12-notes .unnumbered}

Discrete mechanics, the discrete Noether theorem, symplecticity and the order of variational integrators are set out by Marsden and West (2001). Hairer, Lubich and Wanner (2006) is the reference for symplectic methods, Störmer-Verlet and the long-time behavior of the energy through backward error analysis. The discrete rigid body with the trace Lagrangian is due to Moser and Veselov (1991). Lee, McClamroch and Leok (2005) and Lee, Leok and McClamroch (2007) built Lie group variational integrators for the 3D pendulum and for interacting rigid bodies, including the vector form of the implicit equation and its solution by Newton's method, and Lee, Leok and McClamroch (2018) extend the construction to $S^2$ and to multibody systems. The variational discretization of the Reaction Mass Biped, its explicit update and its numerical tests are from Siravuru (2019, ch. 5). Trapezoidal and Hermite-Simpson collocation follow the tutorial of Kelly (2017), and the method goes back to Hargraves and Paris (1987). The exponential-chart decision variables, the flat-output collocation with complementarity constraints for a quadrotor with a cable-suspended load, and the experiments are from Kotaru (2022, ch. 5). The use of the variational rigid-body equations as constraints of a direct transcription, with the step rotation as a variable, is from Teng et al. (2026).

- E. Hairer, C. Lubich and G. Wanner, [*Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*](https://doi.org/10.1007/3-540-30666-8), 2nd ed., Springer, 2006.
- C. R. Hargraves and S. W. Paris, ["Direct trajectory optimization using nonlinear programming and collocation"](https://doi.org/10.2514/3.20223), *Journal of Guidance, Control, and Dynamics* 10(4), 1987.
- M. Kelly, ["An introduction to trajectory optimization: how to do your own direct collocation"](https://doi.org/10.1137/16M1062569), *SIAM Review* 59(4), 2017.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- T. Lee, N. H. McClamroch and M. Leok, ["A Lie group variational integrator for the attitude dynamics of a rigid body with applications to the 3D pendulum"](https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2005_lgvi.pdf), *IEEE Conference on Control Applications*, 2005.
- T. Lee, M. Leok and N. H. McClamroch, ["Lie group variational integrators for the full body problem"](https://doi.org/10.1016/j.cma.2007.01.017), *Computer Methods in Applied Mechanics and Engineering* 196(29-30), 2007.
- T. Lee, M. Leok and N. H. McClamroch, [*Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*](https://doi.org/10.1007/978-3-319-56953-6), Springer, 2018.
- J. E. Marsden and M. West, ["Discrete mechanics and variational integrators"](https://www.cambridge.org/core/journals/acta-numerica/article/discrete-mechanics-and-variational-integrators/C8F45478A9290DEC24E63BB7FBE3CEB5), *Acta Numerica* 10, 2001.
- J. Moser and A. P. Veselov, ["Discrete versions of some classical integrable systems and factorization of matrix polynomials"](https://doi.org/10.1007/BF02352494), *Communications in Mathematical Physics* 139(2), 1991.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- S. Teng, R. Zhang, T.-Y. Lin, W. A. Clark, M. Mueller, R. Vasudevan, M. Ghaffari and K. Sreenath, ["LieIPM: Lie group interior point method for direct trajectory optimization of rigid bodies"](https://arxiv.org/abs/2606.10579), arXiv:2606.10579, 2026.
