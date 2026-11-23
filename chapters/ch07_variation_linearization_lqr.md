---
title: Variation-based linearization and LQR
short: Linearization and LQR
id: ch-linearization
summary: What a small deviation of a rotation is, how to linearize about any trajectory without a singularity, and how far the resulting LQR can be trusted.
notebook: notebooks/ch07_variation_lqr_sim.ipynb
verify: verify/ch07.py
---

A tail-sitter drone takes off like a rocket, nose pointing at the sky, then tips over and flies like an aeroplane. You want a linear quadratic regulator for the nose-up hover. LQR needs a model $\dot x = Ax + Bu$. So what is $x$?

Try yaw, pitch and roll. Nose-up is 90° of pitch, where the matrix $E$ of [](#eq-euler-jacobian) has no inverse. There is no $A$. Back the set-point off to 85° and $A$ exists, with entries of size $1/\cos 85° = 11.5$. At 89.5° they are 115.

Try the nine entries of $R$ instead. Now $x$ has twelve components, $A$ and $B$ exist at every attitude, and the design fails anyway. The controllability matrix has rank 6. Six of the twelve directions are changes that would make $R$ stop being a rotation. No torque produces them, the Riccati solver gives up, and $x_d + \delta x$ is not an attitude.

Both attempts ask how the *numbers that label* an attitude change. The useful question is how the attitude itself changes. The answer is a vector with three components and no singularity anywhere. This chapter builds linear models out of it, puts LQR on top, and then finds out where the result stops being true.

## A small deviation of a direction {#sec7-variation}

Start one dimension down. A pendulum rod, a cable or a thrust axis is a direction $q \in S^2$, and it moves as $\dot q = \omega \times q$ with $\omega \cdot q = 0$. Let $q_d(t)$ be a reference motion and $q(t)$ a neighbour ([](#fig7-variation)).

The difference $q - q_d$ is a poor state. It has three components for two degrees of freedom, and $q_d$ plus a small multiple of it is off the sphere. Chapter [](#ch-variations) gave the right object. A nearby direction is the reference turned by a small rotation, $q = \exp(\epsilon\,\xi^\wedge)\,q_d$, and to first order

$$
\delta q = \xi \times q_d, \qquad \xi \cdot q_d = 0. \label{eq7-var-s2}
$$

The vector $\xi$ is the axis and angle of that small rotation. Its part along $q_d$ would spin $q_d$ about itself and move nothing, so we set it to zero. What remains lies in the tangent plane at $q_d$: two numbers, as there should be.

![A free swing of a spherical pendulum (indigo) and a neighbour released 20° away (terracotta), both simulated. At each instant the neighbour is the reference turned about $\xi$. Both $\xi$ and the displacement $\delta q = \xi \times q_d$ lie in the tangent plane at $q_d(t)$, at right angles to each other.](figures/ch07-variation.svg){#fig7-variation}

The velocity deviation is simpler to define and stranger to live with. Set $\delta\omega = \omega - \omega_d$ to first order. Both are vectors in $\R^3$, so the subtraction is legal.

::: {.thought #thought7-planes name="Two tangent planes"}
Hold two pencils by their ends so that they point in different directions, and twirl each one sideways. Each angular velocity is perpendicular to its own pencil. So $\omega_d$ lies in the plane perpendicular to $q_d$, and $\omega$ lies in the plane perpendicular to $q$. Those are different planes.

Carry $\omega$ over to $q_d$ and compare ([](#fig7-planes)). It sticks out of the plane there. The difference $\delta\omega$ therefore has a component along $q_d$, and that component is not free. It is fixed by how far apart the pencils are and how fast the reference is turning.
:::

![Why $\delta\omega$ is constrained. The reference spins with $\omega_d$, tangent at $q_d$. The neighbour spins with $\omega$, tangent at $q$. Redrawn at $q_d$ (dashed), $\omega$ leaves the tangent plane, so $\delta\omega = \omega - \omega_d$ has a component along $q_d$ (teal).](figures/ch07-two-planes.svg){#fig7-planes}

To find that component, vary $q \cdot \omega = 0$ with [](#eq7-var-s2):

$$
q_d \cdot \delta\omega = \xi \cdot (\omega_d \times q_d). \label{eq7-var-omega}
$$

One more formula and the toolkit is complete. Vary the kinematics $\dot q = \omega \times q$. The left side is $\tfrac{\dd}{\dd t}(\xi \times q_d)$, the right side is $\delta\omega \times q_d + \omega_d \times (\xi \times q_d)$, and the Jacobi identity collapses the difference to $(\dot\xi - \delta\omega - \omega_d \times \xi) \times q_d = 0$. The bracket is therefore along $q_d$, and keeping $\xi \cdot q_d = 0$ for all time fixes its size:

$$
\dot\xi = \omega_d \times \xi + (I - q_d q_d\T)\,\delta\omega . \label{eq7-xi-dot}
$$

Read it as two pieces. The tangential part of $\delta\omega$ changes $\xi$ within the plane. The term $\omega_d \times \xi$ points along $q_d$ and tilts $\xi$ so that it stays in the plane as the plane moves.

We now have a candidate state, $s = (\xi, \delta\omega)$, and two constraints on it. Does it produce a linear model?

## The spherical pendulum, by hand {#sec7-s2}

Put a mass $m$ at the end of a massless rod of length $l$, pivoted at the origin, and push on the mass with a force $F \in \R^3$. Chapter [](#ch-dynamics) derives the equations of motion:

$$
\dot q = \omega \times q, \qquad m l\,\dot\omega = q \times (F - m g\,e_3). \label{eq7-pendulum}
$$

Let $(q_d, \omega_d, F_d)$ be any solution. Vary the second equation. The product rule gives $m l\,\delta\dot\omega = \delta q \times (F_d - m g e_3) + q_d \times \delta F$. Substitute $\delta q = \xi \times q_d$ and use $(\xi \times q) \times a = a^\wedge q^\wedge \xi$. Together with [](#eq7-xi-dot) this is a linear time-varying system.

::: {.proposition #thm7-s2 name="Linearized spherical pendulum"}
Along any solution of [](#eq7-pendulum),
$$
\ddt\begin{bmatrix}\xi \\ \delta\omega\end{bmatrix} =
\underbrace{\begin{bmatrix} \omega_d^\wedge & I - q_d q_d\T \\[2pt] \tfrac{1}{ml}(F_d - m g e_3)^\wedge q_d^\wedge & 0 \end{bmatrix}}_{A(t)}
\begin{bmatrix}\xi \\ \delta\omega\end{bmatrix} +
\underbrace{\begin{bmatrix} 0 \\[2pt] \tfrac{1}{ml}\,q_d^\wedge \end{bmatrix}}_{B(t)} \delta F, \label{eq7-s2-lin}
$$
with the state confined to
$$
C(t)\,s = 0, \qquad C = \begin{bmatrix} q_d\T & 0 \\ (q_d \times \omega_d)\T & q_d\T \end{bmatrix}. \label{eq7-s2-constraint}
$$
:::

No angle appears. $A$ and $B$ are polynomials in the reference, defined at every point of the sphere, including straight up and straight down where spherical coordinates fail.

::: {.remark}
Wu and Sreenath (2015) and the symbolic engine of Chapter [](#ch-symbolic) print the upper-left block as $q_d q_d\T \omega_d^\wedge$. On the constraint set the two agree, because $\omega_d \times \xi$ is along $q_d$ when both vectors are tangent. Off it they differ. Blocks that multiply $\xi$ or $\delta\omega$ are unique only as maps on the constraint set, so compare two derivations there and nowhere else.
:::

::: {.example #ex7-s2-check name="Checking the linearization"}
Take $m = 0.5$ kg, $l = 0.8$ m and a reference released 140° from the hanging position and pushed by a time-varying force. Perturb the initial state by $\pm\epsilon$ along a random admissible $s$, perturb the force too, and simulate the nonlinear pendulum three times. The centred difference of the two neighbours, read through [](#eq7-var-s2), is a finite-difference $s(t)$. Integrate [](#eq7-s2-lin) from the same start. With $\epsilon = 10^{-4}$ the two differ by $3.6\times10^{-8}$ over 1.5 s, and by a hundred times more with $\epsilon = 10^{-3}$, as a second-order error should. The symbolic derivation matches the lower blocks of $A$ and $B$ entry by entry, to $10^{-14}$.
:::

So far $s$ has been an infinitesimal. For control we need to know what it says about a real error.

Near the reference the linear state *is* the geometric error of [](#def-error-s2). Put $q = \exp(\xi^\wedge) q_d$ into $e_q = q_d \times q$ and expand: $e_q = \xi + O(\|\xi\|^3)$. So integrating [](#eq7-s2-lin) predicts $e_q(t)$.

[](#fig7-linear) tests the prediction on a free swing. Release a neighbour 1° away and after 1.2 s the linear model is off by 1.6% of the error. At 10° it is off by 16%, at 30° by 48%. The mismatch grows in proportion to the initial angle. That is what "first order" means, and it is the only promise a linearization makes.

![What the linear state predicts. Left: the error $\|e_q(t)\|$ between two simulated swings, divided by the initial angle, against the linear prediction from [](#eq7-s2-lin). Right: the relative error of the prediction at 1.2 s grows in proportion to the initial angle.](figures/ch07-linear-vs-nonlinear.svg){#fig7-linear}

The experiment also has a trap in it. The neighbour's $\omega$ must be perpendicular to the neighbour's $q$, so its initial $\delta\omega$ has to satisfy [](#eq7-var-omega). Start the linear model from $(\xi, 0)$ instead and the prediction is wrong by 3.5% however small the perturbation. The constraints are part of the model.

## A state that is not free {#sec7-constrained}

The state $s$ has six components, and the pendulum has four degrees of freedom in $(q, \omega)$. The two constraints of [](#eq7-s2-constraint) remove the excess. They hold at the start by construction. Do they keep holding?

They do, for every input. A direct computation with [](#eq7-s2-lin) gives $CB = 0$ and $CA + \dot C = 0$, so

$$
\ddt\big(C(t)\,s(t)\big) = 0 . \label{eq7-invariant}
$$

A solution that starts in the **constraint subspace** $\ker C(t_0)$ stays in $\ker C(t)$. That is reassuring. It is also a warning.

::: {.thought #thought7-broom name="The broom that cannot be balanced"}
Balance a broom upright on your palm. Everyone knows this can be done: a sideways push on the base controls the lean. Now ask the textbook. Linearize about upright, $q_d = e_3$ and $F_d = 0$, form $[B\ \ AB\ \cdots\ A^5B]$ and compute its rank. You get 4. The state has 6 components. The rank test says the broom is uncontrollable.

The test is not wrong. It was asked the wrong question. By [](#eq7-invariant) no input can change $Cs$, so two directions of $\R^6$ are unreachable. But those two directions are not states of a pendulum. One is a rotation vector along the rod. The other is an angular velocity along the rod. A broom cannot be in them, so there is nothing to control.
:::

Controllability has to be judged where the state lives.

::: {.definition #def7-constrained name="Controllability on the constraint subspace"}
The system [](#eq7-s2-lin) with [](#eq7-s2-constraint) is **controllable on the constraint subspace** if every state in $\ker C(t_0)$ can be steered to every state in $\ker C(t_1)$. If the columns of $N$ are an orthonormal basis of a constant $\ker C$, this is ordinary controllability of the reduced pair $(N\T A N,\ N\T B)$. Along a moving reference a sufficient test is
$$
\operatorname{rank}\,[\,M_0\ \ M_1\ \cdots\,] = \dim\ker C(t) \ \text{ for all } t, \qquad M_0 = B, \quad M_{k+1} = \dot M_k - A M_k . \label{eq7-ltv-rank}
$$
:::

For the pendulum, [](#eq7-ltv-rank) can be done on paper. $M_0$ has the block $q_d^\wedge$ in its lower half, and $M_1 = \dot B - AB$ has $-q_d^\wedge$ in its upper half. Each has rank 2, in different rows, so the rank is 4 at every point of every reference. The linearized pendulum is controllable along all of its trajectories, a result due to Wu and Sreenath. Mind the minus sign in $M_{k+1}$. With a plus the test can pass on a system that is not controllable ([](#exr7-sign)).

::: {.example #ex7-upright name="LQR for the upright pendulum"}
At $q_d = e_3$ the tangent plane is spanned by $e_1, e_2$, so $N$ selects the first two components of $\xi$ and of $\delta\omega$. The reduced system is two copies of an inverted pendulum,
$$
N\T A N = \begin{bmatrix} 0 & I_2 \\ \tfrac{g}{l} I_2 & 0 \end{bmatrix}, \qquad \tfrac{g}{l} = 12.26\ \text{s}^{-2},
$$
with rank 4 of 4. Weight the reduced state with $Q = \operatorname{diag}(10, 10, 1, 1)$ and the force with $0.1\,I$, and solve the algebraic Riccati equation. Written back in three dimensions with the error of [](#def-error-s2), the control is
$$
F = k_p\; e_3 \times e_q + k_d\; e_3 \times \omega, \qquad k_p = 16.04\ \text{N}, \quad k_d = 4.78\ \text{N\,s},
$$
with closed-loop poles at $-3.17$ and $-8.77\ \text{s}^{-1}$, each double. Released 30° from upright, the nonlinear pendulum is back within 1° in 1.26 s.
:::

::: {.warning name="Do not solve a Riccati equation on the unreduced state"}
The six-state pair $(A, B)$ of the upright pendulum has two uncontrollable modes at zero. No stabilising solution of the algebraic Riccati equation exists, and a standard solver raises an error. Reduce to $\ker C$ first. The nine-entry attitude model of the opening failed for the same reason, with six such modes instead of two.
:::

The sphere charges two constraints for a singularity-free model. The rotation group charges nothing.

## The rigid body {#sec7-so3}

Take a rigid body with inertia $J$ about a fixed pivot, centre of mass at $\rho$ in the body frame, and body torque $\tau$. With $\rho = 0$ this is the attitude of a multirotor. Chapter [](#ch-dynamics) gives

$$
\dot R = R\,\Omega^\wedge, \qquad J\dot\Omega = J\Omega \times \Omega + m g\,(R\T e_3) \times \rho + \tau . \label{eq7-body}
$$

A neighbouring attitude is the reference turned by a small body-frame rotation, $R = R_d \exp(\epsilon\,\eta^\wedge)$. From Chapter [](#ch-variations),

$$
\delta R = R_d\,\eta^\wedge, \qquad \delta\Omega = \dot\eta + \Omega_d \times \eta . \label{eq7-var-so3}
$$

Any $\eta \in \R^3$ is allowed and any $\delta\Omega \in \R^3$ is allowed. The reason is that $\Omega$ is measured in the body's own frame, so every angular velocity lives in the same $\R^3$. There is no second tangent plane to fall out of.

Vary the second equation of [](#eq7-body) term by term. The gyroscopic term gives $(J\Omega_d)^\wedge\delta\Omega - \Omega_d^\wedge J\,\delta\Omega$. For gravity, $\delta(R\T e_3) = -\eta^\wedge R_d\T e_3 = (R_d\T e_3)^\wedge \eta$.

::: {.proposition #thm7-so3 name="Linearized rigid body"}
Along any solution of [](#eq7-body),
$$
\ddt\begin{bmatrix}\eta \\ \delta\Omega\end{bmatrix} =
\begin{bmatrix} -\Omega_d^\wedge & I \\[2pt] -m g\,J^{-1}\rho^\wedge (R_d\T e_3)^\wedge & J^{-1}\big((J\Omega_d)^\wedge - \Omega_d^\wedge J\big) \end{bmatrix}
\begin{bmatrix}\eta \\ \delta\Omega\end{bmatrix} +
\begin{bmatrix} 0 \\ J^{-1} \end{bmatrix} \delta\tau, \label{eq7-so3-lin}
$$
with no constraint on the state. Since $[\,B\ \ {-AB}\,]$ has rank 6, the system is controllable along every reference.
:::

The same finite-difference test as [](#ex7-s2-check), on a tumbling body with a full inertia matrix, agrees to $9\times10^{-6}$ at $\epsilon = 10^{-4}$ with second-order convergence. The symbolic derivation agrees entry by entry.

::: {.thought #thought7-anywhere name="The same nudge at every attitude"}
Hold a book at rest, flat on your palm, and twist it 5° about its spine. Now hold it upright like a tail-sitter and twist it 5° about its spine again. Your wrist cannot tell the two apart. Nor can [](#eq7-so3-lin): for $\rho = 0$ and $\Omega_d = 0$ the reference attitude $R_d$ does not appear in $A$ or $B$ at all. One linear model, and so one LQR gain, serves every attitude you could hold. This is the left-invariance that Chapter [](#ch-why) asked for.
:::

Now the dictionary between the linear state and what a controller can compute. Insert $R = R_d\exp(\epsilon\eta^\wedge)$ and $\Omega = \Omega_d + \epsilon\,\delta\Omega$ into the errors of [](#def-error-so3).

::: {.proposition #thm7-dictionary name="What the linear state means"}
To first order in the deviation,
$$
e_R = \eta, \qquad e_\Omega = \delta\Omega - \Omega_d \times \eta = \dot\eta . \label{eq7-dictionary}
$$
:::

The first is no surprise. The second deserves a second look. The transported error $e_\Omega$ is not $\delta\Omega$. It is $\dot\eta$. The two differ by $\Omega_d \times \eta$, which is first order whenever the reference is turning. This is the transport map doing its job: $e_\Omega$ compares the two angular velocities in one frame, and $\delta\Omega$ subtracts their components in two.

::: {.table #tbl7-dictionary}

| | Linear state | Constraint | First-order image of |
| :-- | :-: | :-: | :-: |
| $S^2$ | $(\xi,\ \delta\omega)$ | $q_d\cdot\xi = 0$, [](#eq7-var-omega) | $(e_q,\ \omega - \omega_d)$ |
| $\SO(3)$ | $(\eta,\ \delta\Omega)$ | none | $(e_R,\ e_\Omega + \Omega_d \times e_R)$ |

: The linearized states of this chapter and the measurable quantities that equal them to first order.

:::

## LQR on the tangent space {#sec7-lqr}

You know the regulator. For $\dot s = A(t)s + B(t)\,\delta u$ and the cost

$$
\tfrac12\, s(T)\T P_T\, s(T) + \tfrac12\int_0^T \big(s\T Q\,s + \delta u\T R_u\,\delta u\big)\,\dd t,
$$

integrate the Riccati equation backwards from $P(T) = P_T$,

$$
-\dot P = A\T P + P A - P B R_u^{-1} B\T P + Q, \qquad K(t) = R_u^{-1} B(t)\T P(t), \label{eq7-riccati}
$$

and apply $\delta u = -K(t)\,s$. At an equilibrium $A$ and $B$ are constant, $\dot P = 0$, and [](#eq7-riccati) becomes the algebraic Riccati equation. The optimal cost from $s_0$ at time $t_0$ is $\tfrac12 s_0\T P(t_0)\,s_0$.

The part that is new is the last step. The plant state is $(R, \Omega)$, not $s$. The gain must be fed something computed from the manifold-valued state that equals $s$ to first order. [](#tbl7-dictionary) says what.

::: {.definition #def7-feedback name="Variation-based LQR"}
With $K(t)$ from [](#eq7-riccati) and [](#eq7-so3-lin), and the errors of [](#def-error-so3),
$$
\tau = \tau_d(t) - K(t)\begin{bmatrix} e_R \\ e_\Omega + \Omega_d \times e_R \end{bmatrix}. \label{eq7-feedback}
$$
For a direction held at an equilibrium $q_d$, with $K$ designed on $\ker C$ and the error of [](#def-error-s2), $F = F_d - K\,(e_q,\ \omega)$.
:::

Nothing in [](#eq7-feedback) is a difference of coordinates.^[Since $\delta\Omega$ is the plain difference $\Omega - \Omega_d$ of body-frame components, you may feed that instead. The form with $e_\Omega$ shows how to convert a law written in the transported errors used elsewhere in this book.] The errors are built from $R_d\T R$, so they pass the two-labs test of Chapter [](#ch-why), and they are defined at every attitude short of half a turn.

## Nose up, and through a flip {#sec7-worked}

Now the tail-sitter. Take $J = \operatorname{diag}(0.03, 0.05, 0.07)$ kg·m², $\rho = 0$, $Q = \operatorname{diag}(100\,I, I)$ and $R_u = 20\,I$.

::: {.example #ex7-hold name="Holding any attitude"}
At rest, [](#eq7-so3-lin) is three decoupled double integrators and the Riccati equation solves in closed form:
$$
K = \begin{bmatrix} k_R I & \operatorname{diag}(k_{\Omega,i}) \end{bmatrix}, \qquad k_R = \sqrt{100/20} = 2.24, \qquad k_{\Omega,i} = \sqrt{1/20 + 2 J_i k_R} = (0.43,\ 0.52,\ 0.60).
$$
The same $K$ holds the body level, nose-up or inverted.

For comparison, design an LQR the usual way: linearize the same dynamics in yaw, pitch, roll and $\Omega$ about the hold point, use the same $Q$ and $R_u$, and feed back wrapped angle differences. Start both loops 20° from the target about the axis $(1,1,1)$.

Level, the two are hard to tell apart. Peak torque is 0.76 N·m for the geometric loop and 0.81 for the Euler loop.

At 85° of pitch the geometric loop does exactly what it did before, to the last digit. The Euler loop rolls the body almost all the way over ([](#fig7-hold)). The attitude error climbs from 20° to 177.8° before it recovers. Peak torque is 14.2 N·m, 19 times higher, and the effort $\int\|\tau\|^2\dd t$ is 327 times higher. At 88° of pitch a 5° error is enough to set it off.
:::

![Holding a nose-up attitude at 85° of pitch from a 20° initial error, simulated. Left: the path traced by the body's second axis. The geometric LQR moves it straight to the target. The Euler-angle LQR sends it round the sphere. Right: the attitude error angle of both loops.](figures/ch07-hold.svg){#fig7-hold}

The cause is the one from Chapter [](#ch-why). Near the pole of the chart a small rotation is a large change of yaw and roll. The Euler regulator sees a large error, where the body has a small one, and spends torque on it.

::: {.example #ex7-flip name="Tracking a flip"}
Now a reference that moves. The body makes one full turn in 1.5 s about an axis tilted 5° from its pitch axis, with a smooth angle profile. The rate peaks at 7.85 rad/s, 450° per second, and the pitch reaches 85° on the way up and $-85°$ on the way down. Evaluate [](#eq7-so3-lin) along it, integrate [](#eq7-riccati) backwards from the hover solution, and close the loop with [](#eq7-feedback).

[](#fig7-gain) shows the gain. In $(\eta, \delta\Omega)$ its norm stays between 3.83 and 4.09 for the whole flip. The reference enters $A(t)$ only through $\Omega_d$, gently. Do the same design in Euler angles and $A(t)$ has entries up to 873, and the gain swells to 18.0 each time the reference nears the pole.

Start the body 15° off. The geometric loop ends the flip 0.007° from the reference. Its peak torque is 0.86 N·m, against 0.80 for the feedforward alone.
:::

![Riccati gain along the flip of [](#ex7-flip). Top: the pitch angle of the reference passes within 5° of the chart singularity twice. Bottom: the norm of the LQR gain, designed in the variation coordinates and in Euler angles.](figures/ch07-gain-history.svg){#fig7-gain}

Honesty requires one more number. The Euler-angle LQR also tracks this flip, ending 0.005° off. The reference is within 10° of the pole for less than 0.05 s per pass, too briefly for the swollen gain to do harm. A chart singularity hurts where you linger near it, as in [](#ex7-hold), and it forbids a reference that passes through it. It does not ruin every manoeuvre that goes near.

::: {.warning name="Feed the gain the state it was designed for"}
LQR has a built-in test: for small errors the closed-loop cost must equal $\tfrac12 s_0\T P(t_0)\,s_0$. Inject an attitude error mid-flip, at $t_0 = 0.45$ s, where $\Omega_d$ is 5.5 rad/s. With [](#eq7-feedback) the ratio of measured to predicted cost is 1.0000 for a 1° error, 1.006 for 15° and 1.064 for 45°. Replace the velocity entry by $e_\Omega$ alone, as the feedback is sometimes written, and the ratio is 2.00: twice the optimal cost, however small the error. That loop is stable. It is not the regulator you designed. By [](#eq7-dictionary), $e_\Omega$ is $\dot\eta$, and the gain was computed for $\delta\Omega$. At a hover the two agree and the slip is invisible.
:::

A ratio of 1.064 at 45° says the linear model is still good a long way out. How far does that go?

## How local is local? {#sec7-local}

LQR theory gives no answer. It guarantees a neighbourhood and says nothing about its size. The honest move is to measure it ([](#fig7-large)).

Take the upright pendulum of [](#ex7-upright) first. The gain was computed at $q_d = e_3$, where a horizontal force is perpendicular to the rod. Tilt the rod by $\theta$ and the same force has a lever arm shorter by $\cos\theta$. The feedback torque is $l\,k_p \sin\theta\cos\theta$. Gravity's is $m g l \sin\theta$. Released from rest, the pendulum comes back only if

$$
\cos\theta_0 > \frac{m g}{k_p}, \qquad \theta_0 < 72.2° .
$$

Simulation agrees to the half degree. Beyond that the regulator pushes too weakly and the pendulum falls. Here $B$ depends on the state, the gain does not know it, and the region of attraction is a cone.

The attitude hold of [](#ex7-hold) is the opposite case. Torque acts in the body frame, the lower block of $B$ is $J^{-1}$ everywhere, and in simulation the loop recovers from every initial error we tried, up to 179.99°. But look at the time it takes. The feedback uses $e_R$, whose length is $\sin\theta$ ([](#warn-sine)). It fades as the error approaches half a turn. At exactly 180° the regulator applies no torque at all, so the settling time grows without bound as the start approaches it.

![Where the regulators stop working, from simulation. Left: time for the upright pendulum to settle from rest. With $e_q$ it falls beyond 72.2°. With the logarithmic error it falls beyond 77.2°. Right: the attitude hold recovers from any error below 180°, slowly near half a turn with $e_R$, quickly with the logarithmic error.](figures/ch07-large-errors.svg){#fig7-large}

Swap $e_R$ for the rotation vector $\log(R_d\T R)^\vee$, whose length is $\theta$ itself, and keep the same $K$. To first order nothing has changed: same linearization, same gain, same poles. Far from the reference the two loops differ. The logarithmic one does not fade near half a turn, and it moves the pendulum's boundary out to 77.2°. In exchange it is discontinuous at 180°, the **cut locus**, where the shortest rotation back is not unique.

Mueller (2018) makes this point for multicopters. He compares four attitude controllers that are identical to first order and shows that they recover differently from large disturbances, and that the $e_R$ form can linger near half a turn for an arbitrarily long time. Linearization cannot see any of this. Two designs with the same $A$, $B$ and $K$ are one design as far as this chapter's theory goes.

So be clear about what has been built. A variation-based LQR is valid at every reference, with no singular attitude, and it is optimal to first order around each one. It promises nothing far away. No almost-global claim follows, and by [](#thm-obstruction) no global one is possible. The large-error behaviour we saw belongs to these two examples and to the error function, not to LQR.

## The same linear system from the Lie algebra {#sec7-lie}

There is a second road to [](#eq7-so3-lin), and it generalises well. For a state $X$ in a matrix Lie group with body velocity $\zeta$, form the left-invariant error $X_d^{-1}X$ and take its logarithm $\psi$. On $\SO(3)$ that is $\psi = \log(R_d\T R)^\vee$. Differentiate and keep first-order terms:

$$
\dot\psi = -\ad_{\zeta_d}\psi + \zeta - \zeta_d . \label{eq7-lie}
$$

On $\SO(3)$ the operator $\ad_\Omega$ is $\Omega^\wedge$, and [](#eq7-lie) is the first row of [](#eq7-so3-lin) with $\psi$ in place of $\eta$. The two are the same vector to first order. On $\SE(3)$ the operator is a $6\times 6$ matrix built from the linear and angular reference velocities, and position and attitude errors come out coupled the right way.

This form shows why the idea matters beyond LQR. Given the reference, [](#eq7-lie) is linear in the unknowns $(\psi, \zeta)$. So is the linearized momentum equation. Discretise over a horizon, add a quadratic cost in $\psi$ and linear bounds on the inputs, and the tracking problem is a quadratic program. It is convex, and fast reliable solvers exist for it. The same problem posed in $R$ and $\Omega$ is not convex. Teng and co-authors (2022) design the cost in the Lie algebra and show that the logarithmic error keeps its convergence rate out to large angles, which is the right-hand panel of [](#fig7-large) seen from another side. Jang, Teng and Ghaffari (2023) build the convex predictive controller on $\SE(3)$ for marine vehicles. The chapter on error-state MPC and on-manifold solvers takes it from there.

## Exercises {#sec7-exercises .unnumbered}

::: {.exercise #exr7-derive}
Derive [](#eq7-var-omega) and [](#eq7-xi-dot). Then show from [](#eq7-s2-lin) that $CB = 0$ and $CA + \dot C = 0$, which proves [](#eq7-invariant). Does the proof use $\omega_d \cdot q_d = 0$?
:::

::: {.exercise #exr7-sign}
For $\dot x = A(t)x + B(t)u$ with transition matrix $\Phi$, show that $\tfrac{\partial}{\partial\tau}\big(\Phi(t_0,\tau)B(\tau)\big) = \Phi(t_0,\tau)\,(\dot B - AB)$, and explain why this puts a minus sign in [](#eq7-ltv-rank). Then take constant $A$ and $B(t) = e^{At}b$. Show that the system is not controllable, that [](#eq7-ltv-rank) says so, and that the test with $\dot M_k + A M_k$ does not.
:::

::: {.exercise #exr7-horizontal}
Restrict the pendulum's force to the horizontal plane, $F = (F_1, F_2, 0)$. At which $q_d$ does $B$ lose rank? Use $M_1$ to decide whether a reference that passes through such a point is still controllable on the constraint subspace.
:::

::: {.exercise #exr7-s2-velocity}
For a moving reference on the sphere, the transported velocity error is $e_\omega = \omega - (I - qq\T)\,\omega_d$. Show that to first order $e_\omega = (I - q_d q_d\T)\,\delta\omega$, and write $\delta\omega$ in terms of $e_q$ and $e_\omega$. What must replace $(e_q, \omega)$ in [](#def7-feedback)?
:::

::: {.exercise #exr7-euler}
In the notebook, repeat [](#ex7-hold) for hold pitches from 0° to 89° and plot the peak torque of both loops. For a hold at 80°, find the smallest initial error that makes the Euler loop tumble. Does the answer depend on the axis of the error?
:::

::: {.exercise #exr7-roa}
Derive the bound $\cos\theta_0 > m g/k_p$ of [](#sec7-local). Then limit the force to $\|F\| \le F_{\max}$ and predict the new boundary. Check it in the notebook. What happens to the boundary as you raise the weight on $\xi$ in $Q$?
:::

::: {.exercise #exr7-pendulum3d}
Set $\rho = (0, 0, -0.5)$ m and $m = 1$ kg in [](#eq7-so3-lin). Find the hanging and inverted equilibria, the eigenvalues of $A$ at each, and an LQR for the inverted one. Which mode of $A$ does gravity leave at zero, and why?
:::

## Notes and sources {#sec7-notes .unnumbered}

Variation-based linearization on $S^2$ and $\SO(3)$, the constrained state and the controllability results of [](#sec7-constrained) are due to Wu and Sreenath (2015), and are developed for quadrotors with suspended loads in Wu (2017). The variation formulas are those of Lee, Leok and McClamroch (2018), and the error functions those of Bullo and Lewis (2005). The transported velocity error on the sphere in [](#exr7-s2-velocity) is used by Sreenath, Lee and Kumar (2013). The rank test [](#eq7-ltv-rank) is that of Silverman and Meadows (1967). Anderson and Moore (1990) is the standard reference for the regulator. The comparison with Euler-angle designs follows the spirit of Siravuru (2019, ch. 4). The Lie-algebraic route is from Teng et al. (2022) and Jang, Teng and Ghaffari (2023), and Teng et al. (2026) use Lie-group derivatives of the same kind inside a constrained trajectory optimizer.

- B. D. O. Anderson and J. B. Moore, *Optimal Control: Linear Quadratic Methods*, Prentice Hall, 1990.
- F. Bullo and A. D. Lewis, *Geometric Control of Mechanical Systems*, Springer, 2005.
- J. Jang, S. Teng and M. Ghaffari, "Convex geometric trajectory tracking using Lie algebraic MPC for autonomous marine vehicles," *IEEE Robotics and Automation Letters*, 2023. arXiv:2305.09009.
- T. Lee, M. Leok and N. H. McClamroch, *Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*, Springer, 2018.
- M. W. Mueller, "Multicopter attitude control for recovery from large disturbances," arXiv:1802.09143, 2018.
- L. M. Silverman and H. E. Meadows, "Controllability and observability in time-variable linear systems," *SIAM Journal on Control* 5(1), 1967.
- A. Siravuru, *Geometric Control and Learning for Dynamic Legged Robots*, PhD thesis, Carnegie Mellon University, 2019.
- K. Sreenath, T. Lee and V. Kumar, "Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load," *IEEE Conference on Decision and Control*, 2013.
- S. Teng, W. Clark, A. Bloch, R. Vasudevan and M. Ghaffari, "Lie algebraic cost function design for control on Lie groups," arXiv:2204.09177, 2022.
- S. Teng, R. Zhang, T.-Y. Lin, W. A. Clark, M. Mueller, R. Vasudevan, M. Ghaffari and K. Sreenath, "LieIPM: Lie group interior point method for direct trajectory optimization of rigid bodies," arXiv:2606.10579, 2026.
- G. Wu, *Safety-critical Geometric Control Design with Application to Aerial Transportation*, PhD thesis, Carnegie Mellon University, 2017.
- G. Wu and K. Sreenath, "Variation-based linearization of nonlinear systems evolving on $\SO(3)$ and $S^2$," *IEEE Access* 3, 2015.
