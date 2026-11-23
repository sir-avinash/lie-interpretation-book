---
title: Variations, errors and transport
short: Variations and errors
id: ch-variations
summary: How to nudge a configuration, measure how far off it is, and compare two velocities that live in different places.
notebook: notebooks/ch03_variations_transport_maps.ipynb
verify: verify/ch03.py
---

A reference satellite spins steadily about the vertical at one radian per second. Yours copies it exactly: same axis, same rate. The only difference is that your body frame was bolted on a quarter turn away, so your attitude error is a constant 90°. What is your angular velocity error?

Both gyros are honest. The reference gyro reads $\Omega_d = (0, 0, 1)$. Yours reads $\Omega = (0, 1, 0)$, because the vertical happens to be your second body axis ([](#fig3-frames)). Subtract and you get an error of 1.41 rad/s. Yet the two bodies turn together like two parts of one rigid object. Nothing between them is changing. The sensible answer is zero.

![Two frames that spin together about the vertical. They share one motion, but each gyro reports it along its own axes, so the readings differ. Subtracting the readings compares components taken in two different frames.](figures/ch03-frames.svg){#fig3-frames}

[](#ch-why) asked what "error" means for a configuration on a curved space. This is the same question for velocities, and it is the subtler one. A velocity is attached to the place where it was measured. On a flat space every place shares one set of axes, so you subtract without thinking. On a curved space you have to carry one velocity over to the other before you compare, and there is more than one way to carry it. To sort this out we need three things: a way to nudge a configuration, a way to measure how far off it is, and a way to move a velocity from one place to another.

## Nudging a configuration {#sec3-variation}

Start where nothing can go wrong. A position $x \in \R^3$ is nudged by adding a small vector, $x_\epsilon = x + \epsilon\,\delta x$. Any $\delta x$ will do. The result is still a position.

Try that with a rotation. Is $R + \epsilon M$ a rotation for an arbitrary matrix $M$? Test it against [](#def-so3):

$$
(R + \epsilon M)\T (R + \epsilon M) = I + \epsilon\,(R\T M + M\T R) + \epsilon^2 M\T M .
$$

The first-order term must vanish, so $R\T M$ must be skew-symmetric. That is six conditions on nine numbers. Three free numbers remain, and the hat map of [](#def-hat) packs them into a vector: $M = R\,\eta^\wedge$. A nudge of a rotation is not an arbitrary matrix. It is a small rotation, and $\eta$ is its axis and angle in the body frame.

::: {.thought #thought3-pointer name="Nudging a pointer"}
Hold a pen at arm's length and call its direction $q$. Now disturb it slightly. You cannot push the tip along the pen: that would stretch it. Every allowed nudge moves the tip sideways, and every sideways nudge is a small rotation about some axis $\xi$, so $\delta q = \xi \times q$. Twirling the pen about its own length does not move the tip at all. So the part of $\xi$ along $q$ does nothing, and we may as well set it to zero: $\xi \cdot q = 0$.
:::

::: {.definition #def3-variation name="Variation"}
Embed a curve in a one-parameter family of curves that stay on the manifold, with $\epsilon = 0$ giving the curve itself. Its **variation** is the first-order gap to its neighbours, $\delta(\cdot) = \frac{\dd}{\dd\epsilon}\big|_{\epsilon = 0}$. On the three spaces of this book,
$$
\begin{aligned}
x_\epsilon &= x + \epsilon\,\delta x, & \delta x &\in \R^3, \\
R_\epsilon &= R\,\exp(\epsilon\,\eta^\wedge), & \delta R &= R\,\eta^\wedge, \quad \eta \in \R^3, \\
q_\epsilon &= \exp(\epsilon\,\xi^\wedge)\,q, & \delta q &= \xi \times q, \quad \xi \cdot q = 0 .
\end{aligned}
$$
:::

Put $\delta R = R\,\eta^\wedge$ next to the kinematics $\dot R = R\,\Omega^\wedge$ of [](#eq-kinematics). They have the same shape. A variation is a velocity in a direction that is not time. The vectors $\eta(t)$ and $\xi(t)$ may change along the curve, so the family is a ribbon of curves, as in [](#fig3-variation).

![A variation on the sphere. The reference curve $q(t)$ sits inside a family of nearby curves $q_\epsilon(t)$. At each instant the gap opens along $\delta q = \xi \times q$, which is tangent to the sphere. The axis $\xi$ lies in the tangent plane too, at right angles to $\delta q$.](figures/ch03-variation.svg){#fig3-variation}

The ribbon has two directions: along the curve, in $t$, and across it, in $\epsilon$. We know how the configuration changes in each. What happens to the velocity when we step across?

## Nudging a velocity {#sec3-velocity}

For a position the answer is immediate. Derivatives in $t$ and in $\epsilon$ commute, so $\delta(\dot x) = \frac{\dd}{\dd t}(\delta x)$. The variation of the velocity is the velocity of the variation.

The matrices $R_\epsilon(t)$ obey the same rule, $\delta(\dot R) = \frac{\dd}{\dd t}(\delta R)$. But $\dot R$ is not what a gyro measures. The body rate is $\Omega^\wedge = R\T \dot R$, and the frame $R$ moves when we step across the ribbon. Work out both sides:

$$
\delta(\dot R) = \delta(R\,\Omega^\wedge) = R\,\eta^\wedge\Omega^\wedge + R\,\delta\Omega^\wedge,
\qquad
\ddt(\delta R) = \ddt(R\,\eta^\wedge) = R\,\Omega^\wedge\eta^\wedge + R\,\dot\eta^\wedge .
$$

Set them equal and cancel $R$. What is left is $\delta\Omega^\wedge = \dot\eta^\wedge + \Omega^\wedge\eta^\wedge - \eta^\wedge\Omega^\wedge$, and the commutator of two hats is the hat of the cross product.

::: {.proposition #thm3-delta-omega name="Variation of the body rate"}
For $\dot R = R\,\Omega^\wedge$ and $\delta R = R\,\eta^\wedge$,
$$
\delta\Omega = \dot\eta + \Omega \times \eta . \label{eq3-delta-Omega}
$$
:::

The term $\dot\eta$ is what you would have guessed. The term $\Omega \times \eta$ is the opening puzzle in miniature. Freeze $\eta$, so the neighbour is the same body with its frame remounted by a small fixed rotation. The body still spins as before, but its gyro axes have turned, so the reading turns with them.

::: {.example #ex3-remount name="A slightly remounted gyro"}
Take $\Omega = (0, 0, 1)$ rad/s and remount the frame by $\eta = (0.1, 0, 0)$, held constant. Equation [](#eq3-delta-Omega) predicts a change of $\Omega \times \eta = (0, 0.1, 0)$. The exact reading is $\exp(-\eta^\wedge)\,\Omega = (0,\ 0.0998,\ 0.9950)$. Grow $\eta$ to a quarter turn and the reading becomes $(0, 1, 0)$: the satellite of [](#fig3-frames).
:::

This small term has large consequences. The rotational kinetic energy is $\tfrac12\,\Omega \cdot J\Omega$, and its variation is

$$
\delta\!\left(\tfrac12\,\Omega\cdot J\Omega\right) = J\Omega \cdot \delta\Omega = J\Omega\cdot\dot\eta - (\Omega \times J\Omega)\cdot \eta .
$$

There is the $\Omega \times J\Omega$ of Euler's equations. It is not a fictitious force added by hand. It is the price of measuring velocity in a frame that the variation moves. [](#ch-dynamics) turns this line into equations of motion.

Now the sphere. The rule for $\dot q$ is again safe, because $q$ is an ordinary vector of $\R^3$:

$$
\delta(\dot q) = \dot\xi \times q + \xi \times \dot q . \label{eq3-delta-qdot}
$$

The angular velocity needs care. Our convention is $\dot q = \omega \times q$ with $\omega \cdot q = 0$, which makes $\omega = q \times \dot q$ unique. Vary that product, use [](#eq3-delta-qdot), and keep $\xi \cdot q = 0$ at every instant.

::: {.proposition #thm3-delta-omega-s2 name="Variation of the angular velocity on the sphere"}
For $\omega = q \times \dot q$ and $\delta q = \xi \times q$ with $\xi \cdot q = 0$ along the whole curve,
$$
\delta\omega = \dot\xi - 2\,\omega \times \xi
= \underbrace{\dot\xi - \omega \times \xi}_{\text{tangent to the sphere}} \; - \underbrace{\omega \times \xi}_{\text{along } q} . \label{eq3-delta-omega}
$$
:::

Two facts drive the proof, which is [](#exr3-s2-variation). Both $\omega$ and $\xi$ lie in the tangent plane, so $\omega \times \xi$ points along $q$. And differentiating $\xi \cdot q = 0$ in time gives $q \cdot \dot\xi = q \cdot (\omega \times \xi)$, so $\dot\xi$ is forced to leave the tangent plane. Notice that $\delta\omega$ itself is not tangent. Its component along $q$ is what keeps the neighbour's angular velocity perpendicular to the neighbour's $q$.

::: {.warning #warn3-s2-convention name="Three formulas for one variation"}
You will meet [](#eq3-delta-omega) in other clothes, and one of them is a trap.

Lee, Leok and McClamroch (2018) write $\delta\omega = -\omega \times \xi + (I - q q\T)\dot\xi$. It is the same vector, because $q\,(q \cdot \dot\xi) = \omega \times \xi$.

Wu and Sreenath (2015) keep $\delta\omega$ as a variable of its own and impose $q \cdot \delta\omega = -(\xi \times q)\cdot\omega$. That constraint is exactly the component along $q$ above.

The trap is $\delta\omega = \dot\xi - \omega \times \xi$. It is only the tangential part. It gives correct equations whenever $\delta\omega$ is dotted with something tangent, which covers every simple pendulum. It gives gyroscopic terms at half strength otherwise. The symbolic engine of [](#ch-symbolic) applies the full rule $\dot\xi - 2\,\omega\times\xi$.

Do not borrow the sign from $\SO(3)$ either. There $\Omega$ and $\eta$ are body-frame vectors and the cross term enters with a plus. On the sphere $\omega$ and $\xi$ are world-frame axes, and the sign flips ([](#exr3-world-frame)).
:::

We can now nudge configurations and velocities. Next we need a number that says how far a configuration is from where it should be.

## Heights and slopes {#sec3-errors}

[](#ch-why) introduced the attitude error of [](#def-error-so3) as a height $\Psi_R$ and its slope $e_R$. With variations in hand, that recipe becomes a definition that works on any manifold.

::: {.definition #def3-error-vector name="Error function and error vector"}
An **error function** $\Psi(q, q_d) \ge 0$ vanishes only at $q = q_d$. Its **error vector** $e$ is its slope under a variation of the first argument:
$$
\delta\Psi = e_R \cdot \eta \ \text{ on } \SO(3), \qquad \delta\Psi = e_q \cdot \xi \ \text{ on } S^2, \qquad \delta\Psi = e_x \cdot \delta x \ \text{ on } \R^3 .
$$
:::

On $\R^3$ the height $\tfrac12\|x - x_d\|^2$ has slope $x - x_d$. Nothing new. On the sphere, $\Psi_q = 1 - q\cdot q_d$ gives $\delta\Psi_q = -(\xi \times q)\cdot q_d = \xi \cdot (q_d \times q)$, so $e_q = q_d \times q$, as in [](#def-error-s2).

On $\SO(3)$ let us do slightly more than [](#ch-why) did and weight the axes. Take $G = \operatorname{diag}(g_1, g_2, g_3)$ with positive entries and

$$
\Psi_R = \tr\!\big(G\,(I - R_d\T R)\big). \label{eq3-weighted-psi}
$$

Vary $R$: $\delta\Psi_R = -\tr(G R_d\T R\,\eta^\wedge)$. For any matrix $A$, $\tr(A\,\eta^\wedge) = -(A - A\T)^\vee\cdot\eta$. So

$$
e_R = \big(G R_d\T R - R\T R_d\, G\big)^\vee . \label{eq3-weighted-eR}
$$

With $G = \tfrac12 I$ these are the functions of [](#eq-error-so3). Near the target, $e_R \approx (\tr(G)\,I - G)\,\eta$, so the stiffness about body axis 1 is $g_2 + g_3$, and so on. The weights also decide where the unavoidable extra equilibria of [](#thm-obstruction) sit.

::: {.example #ex3-weights name="Where the slope vanishes"}
Take $G = \operatorname{diag}(0.9, 0.6, 0.3)$. The slope [](#eq3-weighted-eR) vanishes when $G R_d\T R$ is symmetric. That happens at the target and at the three half turns about the body axes, and nowhere else. The heights there are $0$, $1.8$, $2.4$ and $3.0$: a minimum, two saddles and a maximum.

With equal weights the three half turns are no longer special. Every half turn about every axis has $e_R = 0$ and $\Psi_R = 2$, a whole two-dimensional family of critical points. Unequal weights cut that family down to three isolated points, the fewest that topology allows.
:::

The trace is not the only height. Suppose the relative rotation $R_d\T R$ is a turn by $\theta$ about a unit axis $u$. Three choices are in common use, scaled here to have the same stiffness near the target:

$$
\begin{array}{lll}
\text{trace:} & \Psi = 1 - \cos\theta, & e = \sin\theta\; u, \\[2pt]
\text{logarithm:} & \Psi = \tfrac12\theta^2, & e = \theta\, u = \log(R_d\T R)^\vee, \\[2pt]
\text{quaternion:} & \Psi = 4\,(1 - \cos\tfrac{\theta}{2}) = 4\,(1 - |q_0|), & e = 2\sin\tfrac{\theta}{2}\; u = 2\operatorname{sgn}(q_0)\, q_v .
\end{array}
$$

Here $(q_0, q_v)$ is the unit quaternion of $R_d\T R$. [](#fig3-heights) plots all three around one full turn.

![Three attitude error functions for a rotation by $\theta$ about a fixed axis. Left: the heights. Right: their slopes, which are the restoring actions a proportional controller applies. Only the trace error is smooth at 180°, and it pays by vanishing there. The other two never fade, and pay with a jump.](figures/ch03-heights.svg){#fig3-heights}

The right-hand panel is [](#thm-obstruction) drawn as a graph. Go once round the axis. The restoring action is positive just after $0°$ and negative just before $360°$. If it is continuous it must cross zero in between: that is the trace error, whose fading torque [](#ch-why) warned about. If it never crosses zero it must jump: that is the other two, with a kink in the height at $180°$ where the shortest way home changes side.

::: {.example #ex3-linger name="The cost of the last degree"}
Rotate about one axis with unit inertia: $\ddot\theta = -k_p\, e(\theta) - k_d\,\dot\theta$, with $k_p = 4$ and $k_d = 2\sqrt2$. Start at rest, $179°$ from the target. The logarithm law is inside $5°$ for good after 2.8 s and the quaternion law after 2.9 s. The trace law needs 6.1 s ([](#fig3-linger)).

Start at $179.9°$ instead. The first two laws do not notice. The trace law needs 8.4 s. Near half a turn it sees $\ddot\phi \approx k_p\phi - k_d\dot\phi$ for the small angle $\phi = \pi - \theta$, an unstable equilibrium with growth rate $\lambda = 1.04\ \text{s}^{-1}$. Every factor of ten closer to $180°$ costs $\ln 10/\lambda = 2.2$ s more. There is no upper bound.
:::

![Simulated recovery from rest near half a turn, same gains, three error vectors. The trace error lingers, and lingers 2.2 s longer for each factor of ten by which the start approaches 180°.](figures/ch03-linger.svg){#fig3-linger}

Mueller (2018) makes this comparison for multicopters and adds an idea worth keeping. Split the relative rotation into a **tilt**, the shortest rotation that aligns the thrust axes, and a yaw about the thrust axis. The tilt angle $\rho_r$ is the angle between $R e_3$ and $R_d e_3$, so its height $1 - \cos\rho_r$ is exactly the sphere error $\Psi_q$ of the thrust direction. The angles obey $\cos\tfrac{\theta}{2} = \cos\tfrac{\rho_r}{2}\cos\tfrac{\rho_y}{2}$, so the tilt never exceeds the total. A multicopter translates by tilting, so the tilt matters more than the yaw. Mueller's controller adds to the logarithm law a second restoring action that acts on the tilt alone and is proportional to the tilt angle. Teng and co-authors (2022) reach the logarithm from another side: they define the cost as $\tfrac12\psi\T P\psi$ in $\psi = \log(R_d\T R)^\vee$, and the same definition works on any matrix Lie group.

None of these is free. Smoothness, a restoring action that never fades, and a single critical point cannot all be had. Choose knowing which one you gave up.

A proportional term alone does not make a controller. We need a derivative term, and that brings back the two gyros.

## Carrying a velocity {#sec3-transport}

::: {.thought #thought3-globe name="Which way is north?"}
One walker stands on the equator and another at 60° north on the same meridian. Each walks north at one metre per second. Do they have the same velocity? As arrows in space they do not: the two "norths" are 60° apart. To compare them you must bring one arrow to the other, and the natural way is to slide it along the surface without turning it.

Now test "without turning". Stand at the pole with an arrow pointing down the Greenwich meridian. Walk down that meridian to the equator, keeping the arrow straight ahead. Walk a quarter of the way round the equator. The arrow now points sideways, due south, and you do not turn it. Walk back up to the pole. You are home. Your arrow points along the 90° meridian ([](#fig3-globe)). It has turned a quarter turn, and you never turned it.
:::

![Sliding an arrow around one eighth of the sphere without ever rotating it relative to the surface. It returns rotated by 90°, the area enclosed by the path. The result of carrying a vector between two points depends on the route.](figures/ch03-globe.svg){#fig3-globe}

So "the same vector, somewhere else" has no meaning on a curved space until you say how it was carried. A rule that carries velocities from the reference to the body is a **transport map**. Since no rule is forced on us, we choose one that makes the control problem easy. The requirement is modest.

::: {.definition #def3-transport name="Compatible transport and velocity error"}
Let $\Psi$ be an error function with error vector $e$. A transport map $\mathcal{T}$, taking velocities at $q_d$ to velocities at $q$, is **compatible** with $\Psi$ if, along every pair of motions,
$$
\ddt\Psi = e \cdot e_{\text{vel}}, \qquad e_{\text{vel}} = (\text{velocity at } q) - \mathcal{T}\,(\text{velocity at } q_d) . \label{eq3-compatible}
$$
The vector $e_{\text{vel}}$ is the **velocity error**.
:::

Compatibility says that the height changes only through the velocity error, exactly as $\frac{\dd}{\dd t}\tfrac12\|x - x_d\|^2 = (x - x_d)\cdot(\dot x - \dot x_d)$ on a flat space. It is the property that makes $\Psi$ plus a kinetic term into a Lyapunov function.

**On $\SO(3)$.** Differentiate the relative rotation $R_d\T R$, using $\dot R = R\,\Omega^\wedge$ and $\dot R_d = R_d\,\Omega_d^\wedge$:

$$
\ddt\big(R_d\T R\big) = \big(R_d\T R\big)\, e_\Omega^\wedge, \qquad e_\Omega = \Omega - R\T R_d\,\Omega_d . \label{eq3-error-kinematics}
$$

The relative rotation obeys the kinematics [](#eq-kinematics), with $e_\Omega$ as its body rate. The transport map is $\Omega_d \mapsto R\T R_d\,\Omega_d$. Read it from the right: $R_d\,\Omega_d$ is the reference's angular velocity in the world frame, and $R\T$ expresses it along the body's axes. It answers the question "what would my gyro read if I moved exactly as the reference does?" For the two satellites, $R\T R_d\,\Omega_d = (0, 1, 0) = \Omega$, and $e_\Omega = 0$.

::: {.proposition #thm3-so3-compat name="Attitude error dynamics"}
For any weight $G$, the functions [](#eq3-weighted-psi) and [](#eq3-weighted-eR) satisfy $\dot\Psi_R = e_R \cdot e_\Omega$. For $G = \tfrac12 I$,
$$
\dot e_R = C\, e_\Omega, \qquad C = \tfrac12\big(\tr(R\T R_d)\, I - R\T R_d\big), \label{eq3-eR-dot}
$$
with $C = I$ at zero error and $\|C\| \le 1$ everywhere.
:::

::: {.proof}
By [](#eq3-error-kinematics), the relative rotation is varied by $\eta = e_\Omega\,\dd t$ in time $\dd t$, so $\dot\Psi_R = e_R\cdot e_\Omega$ by [](#def3-error-vector). The second claim follows by differentiating $e_R$ with [](#eq3-error-kinematics) and using $\big(A\,x^\wedge + x^\wedge A\T\big)^\vee = (\tr(A)\,I - A\T)\,x$.
:::

::: {.remark #rem3-not-unique name="What compatibility does not decide"}
With equal weights, $e_R$ points along the axis of $R_d\T R$, and a rotation leaves its own axis alone. So $e_R \cdot (R\T R_d\,\Omega_d) = e_R\cdot\Omega_d$, and the naive difference passes the test too: $\dot\Psi_R = e_R\cdot(\Omega - \Omega_d)$. The scalar condition [](#eq3-compatible) does not single out one transport map. What singles out $e_\Omega$ is the stronger statement [](#eq3-error-kinematics): it vanishes exactly when the relative attitude stands still. The naive difference reported 1.41 rad/s for two bodies moving as one, and with unequal weights it fails [](#eq3-compatible) as well.
:::

**On the sphere.** The reference direction moves as $\dot q_d = \omega_d \times q_d$. The simplest way to carry that motion to $q$ is to turn $q$ about the same axis at the same rate:

$$
\mathcal{T}\,\dot q_d = (q_d \times \dot q_d)\times q = \omega_d \times q, \qquad e_{\dot q} = \dot q - \omega_d \times q . \label{eq3-transport-s2}
$$

In terms of angular velocities, with $\omega_d$ projected onto the tangent plane at $q$ so that the convention $\omega \cdot q = 0$ survives,

$$
e_\omega = \omega - (I - q q\T)\,\omega_d, \qquad e_{\dot q} = e_\omega \times q, \qquad \ddt\Psi_q = e_q \cdot e_\omega . \label{eq3-e-omega}
$$

This is the pair used by Wu (2017), and it completes [](#eq-error-s2), which held only for a fixed target.

![Velocity error on the sphere. The reference velocity $\dot q_d$ lives in the tangent plane at $q_d$ and cannot be subtracted from $\dot q$ directly. The transport map [](#eq3-transport-s2) carries it to $q$. It agrees with parallel transport along the great circle (dashed) for the component along the circle, and shrinks the component across it by $\cos\theta$.](figures/ch03-transport.svg){#fig3-transport}

::: {.example #ex3-transport-s2 name="A transport map that forgets"}
Put $q_d = e_3$ and $q = e_1$, a quarter circle apart. If the reference moves toward $q$, $\dot q_d = e_1$, then $\mathcal{T}\dot q_d = -e_3$: the body is asked to move away along the same great circle, which keeps the angle fixed. Parallel transport gives the same answer. If the reference moves sideways, $\dot q_d = e_2$, then $\mathcal{T}\dot q_d = 0$, while parallel transport gives $e_2$. In general the sideways part is scaled by $\cos\theta$.

This is not a defect. A sideways step of $q_d$ does not change the angle to $q$ at first order, so $\Psi_q$ cannot see it, and any scaling of that part is compatible. A transport map need not preserve lengths and need not be parallel transport. It has to be compatible, and cheap to compute.
:::

The slope vector itself moves as

$$
\dot e_q = (q\cdot q_d)\, e_\omega + (q\cdot\omega_d)\; q \times e_q - (q_d\cdot\omega)\, q . \label{eq3-eq-dot}
$$

The first term is the counterpart of [](#eq3-eR-dot), with $\cos\theta$ in the role of $C$. The second turns $e_q$ within the tangent plane when the reference spins about $q$. The third keeps $e_q$ tangent as $q$ moves.

On the sphere we had to choose a transport map. On $\SO(3)$ one seemed to fall out of the algebra. That is because $\SO(3)$ is a group, and a group comes with its own ways to move things around. It comes with two.

## Two errors on a group {#sec3-invariant}

Let $X$ and $X_d$ be elements of a matrix Lie group: attitudes, or poses in $\SE(3)$. There are two ways to form "$X$ relative to $X_d$".

::: {.definition #def3-invariant-errors name="Left- and right-invariant error"}
$$
E_\ell = X_d^{-1} X, \qquad E_r = X\, X_d^{-1} .
$$
The **left-invariant error** $E_\ell$ is unchanged when both are multiplied on the left, $(X, X_d) \mapsto (LX, LX_d)$. The **right-invariant error** $E_r$ is unchanged when both are multiplied on the right, $(X, X_d) \mapsto (XL, X_dL)$.
:::

::: {.thought #thought3-remount name="Two labs, one technician"}
Recall the two labs of [](#ch-why), whose motion-capture frames disagree. Changing the world frame multiplies every attitude on the left. The error $R_d\T R$ does not notice. That was the invariance we demanded of a controller.

Now a technician unbolts the flight computer and remounts it rotated. Every attitude is multiplied on the right. The error $R\,R_d\T$ does not notice that. It is the rotation, seen from the world, that takes the reference onto the body, and the world does not care how the computer is mounted. The left-invariant error does notice: its angle is the same, but its axis is now expressed along the new body axes.
:::

Each error pairs with one kind of velocity. With body-frame velocities, $\dot X = X\,\xi^\wedge$ and $\dot X_d = X_d\,\xi_d^\wedge$, the left-invariant error obeys

$$
\dot E_\ell = E_\ell\,\big(\xi - \Ad_{E_\ell^{-1}}\,\xi_d\big)^\wedge . \label{eq3-left-error-dynamics}
$$

On $\SO(3)$ this is [](#eq3-error-kinematics), and it identifies the transport map $R\T R_d$ as the adjoint of the inverse error. Look at what is missing from the right-hand side. The state $X$ does not appear. The error evolves by a law that involves only the error and the two velocities. Wherever the robot is, however it is oriented, the same error with the same velocities changes in the same way.

::: {.example #ex3-dead-reckoning name="Dead reckoning with a wrong start"}
An attitude estimator integrates a gyro, $\dot{\hat R} = \hat R\,\Omega^\wedge$, while the true attitude obeys $\dot R = R\,\Omega^\wedge$ with the same $\Omega(t)$. The estimate starts $5°$ wrong. The body then tumbles for ten seconds.

The right-invariant error does not move: $\frac{\dd}{\dd t}(\hat R R\T) = \hat R\,\Omega^\wedge R\T - \hat R\,\Omega^\wedge R\T = 0$. The left-invariant error $R\T\hat R$ keeps its $5°$ angle while its axis turns with the body. Its rate, $E_\ell\,\Omega^\wedge - \Omega^\wedge E_\ell$, involves the gyro signal but not the attitude. Compare the difference of yaw, pitch and roll between estimate and truth. In the simulation its norm wanders between $4.6°$ and $64°$, although nothing about the actual discrepancy changes. The Euler-angle error has dynamics that depend on where the body happens to be. The invariant errors do not.
:::

That is the idea behind invariant filtering. Barrau and Bonnabel (2017) show that for a whole class of systems, which includes inertial navigation, the invariant estimation error obeys an equation that does not involve the estimate, and that its logarithm obeys a linear equation exactly. Hartley and co-authors (2018) built a legged-robot state estimator on this.

Controllers use the same fact. Write the left-invariant error as $E_\ell = \exp(\psi^\wedge)$. To first order in $\psi$,

$$
\dot\psi = -\ad_{\xi_d}\psi + (\xi - \xi_d), \label{eq3-lie-algebra-error}
$$

where on $\SO(3)$ the operator $\ad_{\Omega_d}$ is the cross product with $\Omega_d$. This is a linear system whose coefficients depend on the reference velocity alone. Teng and co-authors (2022) and Jang, Teng and Ghaffari (2023) build quadratic costs and convex model predictive controllers on it. One linear model serves along the whole trajectory, where a model in coordinates would have to be recomputed at every attitude. The chapters on filtering on Lie groups and on error-state MPC return to both uses.

Equation [](#eq3-lie-algebra-error) is linear in $\psi$, and $\psi$ is a small rotation vector. So is the variation $\eta$. Are they the same thing?

## Errors are variations, to first order {#sec3-first-order}

Place the body near the reference, $R = R_d\exp(\epsilon\,\eta^\wedge)$. Then $R_d\T R$ is a turn by $\epsilon\|\eta\|$ about $\eta$, and [](#eq-error-angle) gives the error vector exactly:

$$
e_R = \sin(\epsilon\|\eta\|)\,\frac{\eta}{\|\eta\|} = \epsilon\,\eta + O(\epsilon^3).
$$

For $\eta = (0.1, 0, 0.2)$ and $\epsilon = 1$, an error of $12.8°$, this gives $e_R = (0.0992, 0, 0.1983)$. The error vector falls short of the variation by $0.5\%$ at $10°$ and by $4.5\%$ at $30°$. The same holds on the sphere: $q = \exp(\epsilon\,\xi^\wedge)\,q_d$ gives $e_q = \sin(\epsilon\|\xi\|)\,\xi/\|\xi\|$.

The velocity needs a second look. Transport does part of the job that the cross term does in [](#eq3-delta-Omega). Expanding $e_\Omega = \Omega - \exp(-\epsilon\,\eta^\wedge)\,\Omega_d$ with $\Omega = \Omega_d + \epsilon\,\delta\Omega$ gives

$$
e_\Omega = \epsilon\,\dot\eta + O(\epsilon^2), \qquad\text{so}\qquad \epsilon\,\delta\Omega = e_\Omega + \Omega_d \times e_R + O(\epsilon^2). \label{eq3-first-order}
$$

The velocity error matches $\dot\eta$, not $\delta\Omega$. The two differ by $\Omega_d \times \eta$, which is first order whenever the reference is turning. On the sphere, likewise, $e_\omega = \epsilon\,(\dot\xi - \omega_d\times\xi) + O(\epsilon^2)$, the tangential part of $\epsilon\,\delta\omega$ in [](#eq3-delta-omega).

::: {.table #tbl3-first-order}

| $\epsilon$ | $\lVert e_R - \epsilon\eta\rVert$ | $\lVert e_\Omega - \epsilon\dot\eta\rVert$ | $\lVert e_\Omega - \epsilon\,\delta\Omega\rVert$ |
| :-: | :-: | :-: | :-: |
| 0.4 | $6.8\times10^{-3}$ | $3.8\times10^{-2}$ | $2.3\times10^{-1}$ |
| 0.2 | $8.6\times10^{-4}$ | $9.6\times10^{-3}$ | $1.1\times10^{-1}$ |
| 0.1 | $1.1\times10^{-4}$ | $2.4\times10^{-3}$ | $5.4\times10^{-2}$ |
| 0.05 | $1.3\times10^{-5}$ | $6.0\times10^{-4}$ | $2.7\times10^{-2}$ |

: Errors against variations along a tumbling reference, for one $\eta(t)$ with $\|\eta\| = 0.86$. Halving $\epsilon$ divides the first column by eight and the second by four. The third only halves: $e_\Omega$ is not $\epsilon\,\delta\Omega$.

:::

This is what makes linearization on a manifold practical. [](#ch-linearization) linearizes the dynamics in the variables $(\eta, \delta\Omega)$, which have no singularities and need no coordinates. A controller cannot measure $\eta$. It can compute $e_R$ and $e_\Omega$ from the state, and [](#eq3-first-order) converts one pair into the other. For a reference at rest, such as a hover, the two pairs coincide.

::: {.table #tbl3-summary}

| | $\R^3$ | $S^2$ | $\SO(3)$ |
| :-- | :-: | :-: | :-: |
| Kinematics | $\dot x = v$ | $\dot q = \omega\times q$, $\ \omega\cdot q = 0$ | $\dot R = R\,\Omega^\wedge$ |
| Variation | $\delta x$ | $\delta q = \xi\times q$, $\ \xi\cdot q = 0$ | $\delta R = R\,\eta^\wedge$ |
| Velocity variation | $\delta v = \frac{\dd}{\dd t}\delta x$ | $\delta\omega = \dot\xi - 2\,\omega\times\xi$ | $\delta\Omega = \dot\eta + \Omega\times\eta$ |
| Error function | $\tfrac12\lVert x - x_d\rVert^2$ | $1 - q\cdot q_d$ | $\tfrac12\tr(I - R_d\T R)$ |
| Error vector | $x - x_d$ | $q_d\times q$ | $\tfrac12(R_d\T R - R\T R_d)^\vee$ |
| Transported reference | $v_d$ | $(I - qq\T)\,\omega_d$ | $R\T R_d\,\Omega_d$ |
| Velocity error | $v - v_d$ | $\omega - (I - qq\T)\,\omega_d$ | $\Omega - R\T R_d\,\Omega_d$ |
| Near the reference | $e_x = \epsilon\,\delta x$ | $e_q \approx \epsilon\,\xi$ | $e_R \approx \epsilon\,\eta$ |

: The toolkit of this chapter. On a product of these spaces, apply each column to its factor and add the error functions.

:::

## Exercises {#sec3-exercises .unnumbered}

::: {.exercise #exr3-skew}
Show that $R\exp(\epsilon\,\eta^\wedge)$ is a rotation for every $\epsilon$, and that $R + \epsilon R\,\eta^\wedge$ is not. How large is $\|R_\epsilon\T R_\epsilon - I\|$ in the second case? Why does the definition of a variation not care?
:::

::: {.exercise #exr3-world-frame}
Use world-frame quantities instead: $\dot R = w^\wedge R$ and $\delta R = \zeta^\wedge R$. Show that $\delta w = \dot\zeta - w\times\zeta$. Compare the sign with [](#eq3-delta-Omega) and explain it using [](#ex3-remount).
:::

::: {.exercise #exr3-s2-variation}
Prove [](#eq3-delta-omega) from $\omega = q\times\dot q$ and [](#eq3-delta-qdot). Then check it by finite differences in the notebook: build a curve $q(t)$, a field $\xi(t)$ with $\xi\cdot q = 0$, the family $q_\epsilon(t)$, and difference $\omega_\epsilon$ in $\epsilon$. Repeat with a $\xi$ that is not perpendicular to $q$. Which formula survives?
:::

::: {.exercise #exr3-weights}
For $\Psi_R$ in [](#eq3-weighted-psi) with $g_1 > g_2 > g_3 > 0$, show that $e_R = 0$ exactly at the four attitudes of [](#ex3-weights). Expand $\Psi_R$ to second order in $\eta$ at each and show the Hessians are $\operatorname{diag}(d_2 + d_3,\ d_1 + d_3,\ d_1 + d_2)$, where $d_i$ are the diagonal entries of $G R_d\T R$. Classify the four points.
:::

::: {.exercise #exr3-linger}
Reproduce [](#ex3-linger). Then give all three laws an initial angular velocity that carries them through $180°$, and add measurement noise. Which law chatters, and where?
:::

::: {.exercise #exr3-transport}
On the sphere, let $Q$ be the rotation about $q_d\times q$ that takes $q_d$ to $q$, so that $\omega_d\mapsto Q\,\omega_d$ is parallel transport along the great circle. Show that it is compatible with $\Psi_q$. Describe all linear maps that are. Then find the transport map compatible with $\Psi = \tfrac12\theta^2$, where $\theta$ is the angle between $q$ and $q_d$.
:::

::: {.exercise #exr3-right-error}
Derive the dynamics of the right-invariant error $E_r = R\,R_d\T$ in terms of the world-frame rates $w = R\,\Omega$ and $w_d = R_d\,\Omega_d$. Which velocity error appears? Redo the opening puzzle with $E_r$.
:::

::: {.exercise #exr3-first-order}
Derive [](#eq3-first-order). Then take a reference that spins at constant $\Omega_d$ and a constant offset $\eta$. Compute $e_R$, $e_\Omega$ and $\delta\Omega$ and say which of them a derivative term should act on.
:::

## Notes and sources {#sec3-notes .unnumbered}

Variations on $\SO(3)$ and $S^2$ in the form used here, including [](#eq3-delta-Omega) and the expression for $\delta\omega$, are developed by Lee, Leok and McClamroch (2018). Error functions, compatible transport maps and the weighted trace function come from Bullo and Lewis (2005). Lee, Leok and McClamroch (2010) brought $e_R$ and $e_\Omega$ to quadrotors, and [](#eq3-eR-dot) appears there and in Siravuru (2019). The sphere errors [](#eq3-e-omega) and the general statement of compatibility follow Wu (2017). The first-order link between errors and variations is the basis of Wu and Sreenath (2015). The comparison of attitude errors, the lingering of the trace error and tilt prioritisation are from Mueller (2018). Left- and right-invariant errors and their autonomous dynamics are due to Barrau and Bonnabel (2017). Their use for legged robots is from Hartley et al. (2018). The Lie-algebraic tracking error [](#eq3-lie-algebra-error) is from Teng et al. (2022). Teng et al. (2026) use the same variations as search directions for trajectory optimisation on Lie groups.

- A. Barrau and S. Bonnabel, "The invariant extended Kalman filter as a stable observer," *IEEE Transactions on Automatic Control* 62(4), 2017.
- F. Bullo and A. D. Lewis, *Geometric Control of Mechanical Systems*, Springer, 2005.
- R. Hartley, M. Ghaffari Jadidi, J. W. Grizzle and R. M. Eustice, "Contact-aided invariant extended Kalman filtering for legged robot state estimation," arXiv:1805.10410, 2018.
- J. Jang, S. Teng and M. Ghaffari, "Convex geometric trajectory tracking using Lie algebraic MPC for autonomous marine vehicles," arXiv:2305.09009, 2023.
- T. Lee, M. Leok and N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on SE(3)," *IEEE Conference on Decision and Control*, 2010.
- T. Lee, M. Leok and N. H. McClamroch, *Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*, Springer, 2018.
- M. W. Mueller, "Multicopter attitude control for recovery from large disturbances," arXiv:1802.09143, 2018.
- A. Siravuru, *Geometric Control and Learning for Dynamic Legged Robots*, PhD thesis, Carnegie Mellon University, 2019.
- S. Teng, W. Clark, A. Bloch, R. Vasudevan and M. Ghaffari, "Lie algebraic cost function design for control on Lie groups," arXiv:2204.09177, 2022.
- S. Teng, R. Zhang, T.-Y. Lin, W. A. Clark, M. Mueller, R. Vasudevan, M. Ghaffari and K. Sreenath, "LieIPM: Lie group interior point method for direct trajectory optimization of rigid bodies," arXiv:2606.10579, 2026.
- G. Wu, *Safety-critical Geometric Control Design with Application to Aerial Transportation*, PhD thesis, Carnegie Mellon University, 2017.
- G. Wu and K. Sreenath, "Variation-based linearization of nonlinear systems evolving on SO(3) and S²," *IEEE Access* 3, 2015.
