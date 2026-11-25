---
title: Dynamics without coordinates
short: Dynamics
id: ch-dynamics
summary: Hamilton's principle carried out on the sphere and on the rotation group, and the equations of motion of pendulums, rigid bodies, quadrotors and cable-suspended loads that come out of it.
notebook: notebooks/ch04_lagrange_dalembert_sim.ipynb
verify: verify/ch04.py
epigraph: "No figures will be found in this work. The methods I present require … solely algebraic operations subject to a regular and uniform procedure."
epigraph_source: "Joseph-Louis Lagrange, *Mécanique analytique* (1788), preface"
---

Pin a rigid body at one point and let it swing under gravity. This is the 3D pendulum. Write its equations of motion twice.

First with yaw, pitch and roll. Substitute $\Omega = E(\theta,\phi)\,\dot\Theta$ from [](#eq-euler-jacobian) into the kinetic energy $\tfrac12\Omega\cdot J\Omega$, apply the Euler-Lagrange equations, and expand. For a general inertia matrix you get three coupled equations with 105 terms between them, each a product of sines and cosines. Their mass matrix is $E\T J E$, and it has to be inverted at every time step.

Then with the rotation matrix. The whole model is

$$
J\dot\Omega + \Omega\times J\Omega = m g\,(R\T e_3)\times\rho, \qquad \dot R = R\,\Omega^\wedge ,
$$

where $\rho$ is the body-frame vector from the pivot to the center of mass. One line. The mass matrix is $J$, a constant.

![The puzzle of this chapter. A 3D pendulum is released and swings twice to within about a degree of the attitude where its pitch is 90°, with the body axis $R\,e_1$ vertical. The body does nothing special there. The condition number of the Euler-angle mass matrix $E\T J E$ spikes at each pass. That of $J$ stays at 2.5.](figures/ch04-pendulum-strip.png){#fig4-pendulum anim="figures/ch04-pendulum.gif"}

Where both work they give the same motion. But take a body with $J = \operatorname{diag}(0.20, 0.25, 0.10)$ kg·m² and look at how hard each mass matrix is to invert ([](#fig4-euler-cond)). The condition number of $J$ is 2.5 at every attitude. The condition number of $E\T J E$ is 2.5 when the body is level, 25 at 60° of pitch, 974 at 85°, and $2.4\times10^4$ at 89°. At 90° the matrix is singular and cannot be inverted at all.

![Condition number of the mass matrix of a 3D pendulum against pitch angle, at 20° of roll. In Euler angles it grows like $1/\cos^2\theta$. In the body frame it is the constant 2.5.](figures/ch04-euler-cond.svg){#fig4-euler-cond}

Nothing happens to the pendulum at 90° of pitch ([](#fig4-pendulum)). So where did the hundred terms go, and why does the short equation never blow up? To answer, we need to derive equations of motion without ever choosing a chart.

## Hamilton's principle, on flat ground {#sec4-hamilton}

Start where there is nothing to fix. Throw a ball of mass $m$. Its position is $x \in \R^3$, its **Lagrangian** is kinetic minus potential energy,

$$
L(x,\dot x) = \tfrac12 m\,\dot x\cdot\dot x - m g\,e_3\cdot x ,
$$

and the **action** of a path $x(t)$ over $[0,t_f]$ is the number $S = \int_0^{t_f} L\,\dd t$.

::: {.definition #def4-hamilton name="Hamilton's principle"}
Among all paths with the same endpoints $x(0)$ and $x(t_f)$, the motion is a path whose action is stationary: $\delta S = 0$ for every variation that vanishes at the endpoints.
:::

A variation ([](#def3-variation)) is a one-parameter family of nearby paths, $x_\epsilon(t) = x(t) + \epsilon\,\delta x(t)$ with $\delta x(0) = \delta x(t_f) = 0$, and $\delta S$ is the derivative of $S$ along the family at $\epsilon = 0$. Three moves turn the principle into an equation.

*Vary.* Differentiate under the integral. Since $\epsilon$ and $t$ are independent, $\delta\dot x = \ddt\delta x$, and

$$
\delta S = \int_0^{t_f} \big(m\,\dot x\cdot\delta\dot x - m g\,e_3\cdot\delta x\big)\,\dd t .
$$

*Integrate by parts*, to move the time derivative off $\delta x$. The boundary term vanishes because $\delta x$ does:

$$
\delta S = -\int_0^{t_f} \big(m\ddot x + m g\,e_3\big)\cdot\delta x\;\dd t .
$$

*Use that $\delta x$ is arbitrary.* The bracket must then vanish at every instant, and $m\ddot x = -m g\,e_3$.

::: {.thought #thought4-ball name="Bump the parabola"}
Throw a 1 kg ball so that it lands one second later. Imagine it had flown a path 10 cm higher in the middle, between the same two points: add $\epsilon\sin(\pi t/t_f)\,e_3$ with $\epsilon = 0.1$ m. It moves faster, and it sits higher. Which effect wins?

Neither, to first order. The kinetic and potential changes cancel exactly, and what is left is second order: the action rises by $m\epsilon^2\pi^2/(4t_f) = 0.0247$ J·s. Bump it downward by 10 cm and the action rises by the same amount. The parabola sits at the bottom of a bowl.
:::

Every step used the fact that $x + \epsilon\,\delta x$ is again a valid position. On a curved configuration space that is false, and the first move has to change.

## The same three moves on a sphere {#sec4-sphere}

A spherical pendulum is a bob of mass $m$ on a rigid massless rod of length $l$. The bob sits at $l q$ with $q \in S^2$, and

$$
L(q,\dot q) = \tfrac12 m l^2\,\dot q\cdot\dot q - m g l\,e_3\cdot q .
$$

Try $q + \epsilon\,\delta q$ and the bob leaves the sphere: the rod has stretched. A variation must be a family of curves that all stay on $S^2$. The simplest way to stay on a sphere is to rotate. Pick any vector field $\xi(t)$ along the path and set

$$
q_\epsilon(t) = \exp\!\big(\epsilon\,\xi(t)^\wedge\big)\,q(t)
\qquad\Longrightarrow\qquad
\delta q = \xi\times q, \qquad \delta\dot q = \dot\xi\times q + \xi\times\dot q . \label{eq4-var-s2}
$$

This is the variation on the sphere of [](#def3-variation), with the velocity rule [](#eq3-delta-qdot). Each $q_\epsilon(t)$ is $q(t)$ turned by the angle $\epsilon\|\xi\|$ about $\xi$, so its length is still one ([](#fig4-variation)). A component of $\xi$ along $q$ turns $q$ about itself and does nothing, so we take $\xi\cdot q = 0$. The fixed endpoints become $\xi(0) = \xi(t_f) = 0$.

![Hamilton's principle on the sphere. The indigo path $q(t)$ is a computed swing of a 1 kg bob on a 1 m rod. The teal path is $q(t)$ turned about $\xi(t)$ by the angle $\epsilon\|\xi\|$, so it stays on the sphere and keeps its endpoints. The arrows show how far five of its points have moved. Right: the action of the varied path, computed for each $\epsilon$. It is stationary at $\epsilon = 0$.](figures/ch04-variation-strip.png){#fig4-variation anim="figures/ch04-variation.gif"}

Now the three moves. Varying and integrating by parts work exactly as before and give

$$
\delta S = -\int_0^{t_f} \big(m l^2\ddot q + m g l\,e_3\big)\cdot\delta q\;\dd t
= -\int_0^{t_f} \xi\cdot\Big(q\times\big(m l^2\ddot q + m g l\,e_3\big)\Big)\,\dd t ,
$$

using $y\cdot(\xi\times q) = \xi\cdot(q\times y)$ for any vector $y$. The third move is where the sphere shows. $\xi$ is arbitrary, but only among vectors perpendicular to $q$. The vector $q\times(\cdots)$ is already perpendicular to $q$, so it must vanish:

$$
q\times\big(m l^2\ddot q + m g l\,e_3\big) = 0 . \label{eq4-sph-cross}
$$

Read this carefully. It does not say that $m l^2\ddot q + m g l\,e_3$ is zero. It says that vector is parallel to $q$. Only its tangent part must vanish:

$$
(I - q q\T)\big(m l\,\ddot q + m g\,e_3\big) = 0 . \label{eq4-sph-proj}
$$

The part along $q$ is fixed by the constraint alone. Differentiate $q\cdot q = 1$ twice to get the centripetal acceleration $q\cdot\ddot q = -\|\dot q\|^2$. Together:

$$
\ddot q = -\|\dot q\|^2\,q - \frac{g}{l}\,(I - q q\T)\,e_3 . \label{eq4-sph-qdd}
$$

With the angular velocity of [](#eq2-s2-kinematics), $\dot q = \omega\times q$ and $\omega\cdot q = 0$, one finds $q\times\ddot q = \dot\omega$, and [](#eq4-sph-cross) becomes

$$
m l^2\,\dot\omega = -m g l\; q\times e_3, \qquad \dot q = \omega\times q . \label{eq4-sph}
$$

No angles, and nothing special about the bottom or the top of the sphere.

What is the component along $q$ that the principle refused to constrain? Draw the free-body diagram ([](#fig4-forces)). Gravity $-m g\,e_3$ acts on the bob, and the rod pulls along itself with some tension $T$. Newton says $m l\,\ddot q = -m g\,e_3 - T q$. Project onto the tangent plane and the tension disappears, leaving [](#eq4-sph-proj). Project onto $q$ and you get the tension:

$$
T = m l\,\|\dot q\|^2 - m g\,e_3\cdot q . \label{eq4-tension}
$$

Hamilton's principle never needed $T$. A variation that stays on the sphere does no work against the rod, so the rod's force drops out unnamed.

![The spherical pendulum as a free-body diagram. Gravity splits into a part in the tangent plane $T_qS^2$, which moves the bob, and a part along the rod, which the tension absorbs. Equation [](#eq4-sph-proj) is the tangent part.](figures/ch04-pendulum-forces.svg){#fig4-forces}

::: {.example #ex4-tension name="Three times the weight"}
A 1 kg bob on a 1 m rod is released from rest with the rod horizontal, $q = e_1$. At that instant [](#eq4-tension) gives $T = 0$: the bob is in free fall. At the bottom of the swing $q = -e_3$ and $\|\dot q\|^2 = 2g/l$ by energy conservation, so $T = 3mg = 29.4$ N. Integrating [](#eq4-sph) reproduces it to ten digits.
:::

::: {.warning #warn4-projection name="An equation on the sphere is only its tangent part"}
Any equation of motion for $q \in S^2$ is meaningful only after projection by $I - q q\T$. Solve the unprojected three equations for $\dot\omega$ and you can get a component along $q$: the state walks off the sphere. Symbolic tools return the equation as derived and leave the projection to you. Solve the tangent part together with $q\cdot\dot\omega = 0$.
:::

The sphere was easy because $\delta\dot q$ was just the time derivative of $\delta q$. On the rotation group the velocity we use is $\Omega$, and that changes the first move.

## Rotations: where the gyroscopic term comes from {#sec4-so3}

A free rigid body has attitude $R \in \SO(3)$ and Lagrangian $L = \tfrac12\,\Omega\cdot J\Omega$, with $\Omega^\wedge = R\T\dot R$ as in [](#eq-kinematics). Vary the attitude by rotating it, as in [](#def3-variation), so that the family stays on the manifold:

$$
R_\epsilon(t) = R(t)\exp\!\big(\epsilon\,\eta(t)^\wedge\big), \qquad \delta R = R\,\eta^\wedge , \qquad \eta(0) = \eta(t_f) = 0 .
$$

The Lagrangian is written in $\Omega$, so we need $\delta\Omega$. [](#thm3-delta-omega) derives it. The calculation is two lines: vary $\Omega^\wedge = R\T\dot R$ and use $\delta\dot R = \dot R\,\eta^\wedge + R\,\dot\eta^\wedge$,

$$
\delta\Omega^\wedge = -\eta^\wedge\,\Omega^\wedge + \Omega^\wedge\eta^\wedge + \dot\eta^\wedge
\qquad\Longrightarrow\qquad
\delta\Omega = \dot\eta + \Omega\times\eta . \label{eq4-var-omega}
$$

For the ball, $\delta\dot x = \ddt\delta x$ and nothing else. Here there is an extra term $\Omega\times\eta$. It is the commutator of two rotations, the one the body is making and the one we vary by. It is there because rotations do not commute.

Now the three moves:

$$
\delta S = \int_0^{t_f} J\Omega\cdot\big(\dot\eta + \Omega\times\eta\big)\,\dd t
= -\int_0^{t_f} \eta\cdot\big(J\dot\Omega + \Omega\times J\Omega\big)\,\dd t .
$$

The first term was integrated by parts. The second used $J\Omega\cdot(\Omega\times\eta) = -\eta\cdot(\Omega\times J\Omega)$. Since $\eta$ is an arbitrary vector in $\R^3$, with no constraint this time,

$$
J\dot\Omega + \Omega\times J\Omega = 0 . \label{eq4-euler-free}
$$

This is Euler's equation, and the derivation shows where its nonlinear term is born. The integration by parts produced $J\dot\Omega$. The commutator in [](#eq4-var-omega) produced $\Omega\times J\Omega$. A body whose rotations commuted would have no gyroscopic term at all.

A free body is not yet a robot. We need forces.

## Forces and torques {#sec4-forces}

Thrust, motor torque and drag do not come from a potential. They enter through the work they do on a variation.

::: {.definition #def4-dalembert name="Lagrange-d'Alembert principle"}
Let $\delta W$ be the **virtual work**: the work the applied forces would do on the variation. The motion satisfies
$$
\delta\!\int_0^{t_f} L\,\dd t + \int_0^{t_f} \delta W\,\dd t = 0 \label{eq4-dalembert}
$$
for every variation that stays on the manifold and vanishes at the endpoints.
:::

The job is to write $\delta W$ for each actuator. Two cases cover almost everything.

A torque $\tau$ in the body frame does work on a body-frame rotation $\eta$: $\delta W = \tau\cdot\eta$. That puts $\tau$ on the right of Euler's equation.

A force $F$ in the world frame, applied at the body point $x + R r$, does work on the displacement of that point. Vary the point: $\delta(x + R r) = \delta x + R\,(\eta\times r)$. Then

$$
\delta W = F\cdot\delta x + F\cdot R\,(\eta\times r) = F\cdot\delta x + \eta\cdot\big(r\times R\T F\big) . \label{eq4-virtual-work}
$$

The coefficient of $\delta x$ is the force. The coefficient of $\eta$ is the moment arm crossed with the force, both in the body frame. The rule "$r\times F$" has been derived, frames included.

::: {.proposition #thm4-rigid-body name="Rigid body with a force and a torque"}
For a body of mass $m$ and inertia $J$ about its center of mass $x$, with $L = \tfrac12 m\|\dot x\|^2 + \tfrac12\Omega\cdot J\Omega - m g\,e_3\cdot x$ and the virtual work [](#eq4-virtual-work) plus $\tau\cdot\eta$,
$$
m\ddot x = F - m g\,e_3, \qquad J\dot\Omega + \Omega\times J\Omega = \tau + r\times R\T F, \qquad \dot R = R\,\Omega^\wedge . \label{eq4-rigid-body}
$$
:::

The 3D pendulum of the opening is now a short exercise. Take $J$ about the pivot. Gravity acts at the center of mass $R\rho$, with potential $m g\,e_3\cdot R\rho$, whose variation is $m g\,\eta\cdot(\rho\times R\T e_3)$. Therefore

$$
J\dot\Omega + \Omega\times J\Omega = m g\,(R\T e_3)\times\rho + \tau . \label{eq4-pendulum}
$$

That resolves the puzzle. The physics of the pendulum is the three terms of [](#eq4-pendulum). The other hundred terms of the Euler-angle model are $E$, its derivative and their products with $J$: a description of how the chart is bent. The blow-up in [](#fig4-euler-cond) is $E$ losing rank: $\det(E\T J E) = \cos^2\theta\,\det J$. The body-frame velocity $\Omega$ is defined at every attitude, so there was never anything to invert.

You may know these equations from another route. **Newton-Euler** mechanics says that force is the rate of change of linear momentum, and moment about the center of mass is the rate of change of angular momentum, both in the world frame. The world-frame angular momentum is $R J\Omega$. Differentiate it with $\dot R = R\,\Omega^\wedge$:

$$
\ddt\big(R J\Omega\big) = R\big(J\dot\Omega + \Omega\times J\Omega\big) = R\,\tau . \label{eq4-newton-euler}
$$

The same equation, and the same origin for the gyroscopic term: it comes from differentiating $R$. The two methods always agree. Newton-Euler is quicker when the free-body diagram is simple. Lagrange-d'Alembert is safer when constraint forces are many, because it never introduces them. We use each where it is shorter.^[[](#ch-symbolic) automates the variational route with the [`geomech`](https://github.com/vkotaru/geomech) library. Every equation of motion in this chapter was derived a second time that way and compared numerically.]

## A sanity check you can throw in the air {#sec4-conservation}

Before trusting a model, ask what it must conserve. With no torque, [](#eq4-newton-euler) says the world-frame angular momentum $R J\Omega$ is constant, all three components. The energy $\tfrac12\Omega\cdot J\Omega$ is constant too. The body-frame momentum $J\Omega$ is *not* constant: only its length is, because $R$ is a rotation. These are tests a simulation passes or fails. They also predict something surprising.

::: {.thought #thought4-book name="Toss a book"}
Put a rubber band round a paperback so it stays shut. It has three natural axes: through the covers, along the spine, and across the width. Toss it spinning about the axis through the covers. It spins cleanly. Toss it spinning along the spine. Clean again. Now toss it spinning about the axis across the width. However carefully you throw, it comes down having flipped over.

Nothing pushed it. There is no torque in the air. Euler's equation [](#eq4-euler-free) is doing this by itself ([](#fig4-book)).
:::

![The three tosses, computed from Euler's equation for the book of [](#ex4-book) at two revolutions per second. Left: the book in the world frame. The dashed arrow is its angular momentum $R J\Omega$, which never moves. Right: the body angular momentum $J\Omega$ on its sphere. Spun about $e_3$ or $e_1$ the book keeps its axis. Spun about $e_2$ it turns over, and then comes back.](figures/ch04-book-strip.png){#fig4-book anim="figures/ch04-book.gif"}

Here is the reason. In the body frame, $J\Omega$ moves on a sphere, since its length is conserved. It must also stay on a level set of the energy. Draw those level sets on the sphere ([](#fig4-polhodes), left). Around the axes of smallest and largest inertia they are closed loops: a spin that starts near those axes stays near. At the intermediate axis the level sets cross. A spin that starts there is carried along the crossing to the opposite side of the sphere, and back.

![The intermediate-axis instability, computed from Euler's equation for a 20 × 13 × 3 cm book. Left: paths of the body angular momentum on its sphere. They circle $e_1$ and $e_3$ and cross at $\pm e_2$. The heavy curve is the simulated spin that starts 1% away from the middle axis. It leaves $e_2$ (hidden), passes $-e_2$ and returns. Right: the spin component along the starting axis. About $e_2$ it reverses every 1.5 s.](figures/ch04-polhodes.svg){#fig4-polhodes}

::: {.example #ex4-book name="How fast it flips"}
Model the book as a uniform 0.3 kg block, 20 × 13 × 3 cm. Its principal inertias are $(J_1, J_2, J_3) = (0.445,\ 1.02,\ 1.42)\times10^{-3}$ kg·m², with $e_2$ across the width. Linearize [](#eq4-euler-free) about a steady spin $\Omega_2$ about $e_2$. The wobble in $\Omega_1$ and $\Omega_3$ grows like $e^{\lambda t}$ with

$$
\lambda = \Omega_2\sqrt{\frac{(J_2 - J_1)(J_3 - J_2)}{J_1 J_3}} . \label{eq4-growth}
$$

At two revolutions per second, $\lambda = 7.6$ s⁻¹: the wobble multiplies by $e$ every 0.13 s. Start the simulation with 1% of the spin on each of the other axes. $\Omega_2$ first reverses at 0.88 s and then every 1.5 s ([](#fig4-polhodes), right). Through all of it the energy and all three components of $R J\Omega$ hold to better than one part in $10^{8}$. The same 1% error on a spin about $e_1$ or $e_3$ never grows beyond 2%. For those axes the product under the root changes sign and the wobble oscillates.
:::

This is the tennis racket theorem, or Dzhanibekov effect, after the cosmonaut who saw a wing nut do it in orbit. For us it is a check: a rigid-body simulation that does not flip the book is wrong.

## Robots as products {#sec4-products}

Real robots have several parts, each on its own manifold. The configuration space is the product, the Lagrangian is a sum, and the variations are the ones we already have, one per factor.

**Quadrotor, on $\SE(3)$.** Four rotors produce a total thrust $f$ along the body axis $R e_3$ and a body torque $\tau$. The thrust acts at the center of mass, so $F = f R e_3$ and $r = 0$ in [](#thm4-rigid-body):

$$
m\ddot x = f R e_3 - m g\,e_3, \qquad J\dot\Omega + \Omega\times J\Omega = \tau . \label{eq4-quadrotor}
$$

Six degrees of freedom, four inputs. The force can only point along $R e_3$, so to accelerate sideways the vehicle must first rotate.

**Quadrotor with a cable-suspended load, on $\R^3\times S^2\times\SO(3)$.** Hang a point mass $m_L$ from the quadrotor (mass $m_Q$, inertia $J$) on a massless cable of length $l$. As in [](#sec2-products), $q \in S^2$ points from the quadrotor to the load, so that $x_Q = x_L - l q$ ([](#fig4-quad-load)). The Lagrangian and virtual work are

$$
L = \tfrac12 m_Q\|\dot x_L - l\dot q\|^2 + \tfrac12 m_L\|\dot x_L\|^2 + \tfrac12\Omega\cdot J\Omega - m_Q g\,e_3\cdot(x_L - l q) - m_L g\,e_3\cdot x_L ,
$$
$$
\delta W = f R e_3\cdot\big(\delta x_L - l\,\xi\times q\big) + \tau\cdot\eta .
$$

![A quadrotor carrying a point-mass load on a cable. The configuration is the load position $x_L$, the cable direction $q \in S^2$ and the attitude $R$. Thrust acts along $R e_3$. The tension $T$ is internal and does not appear in the equations of motion.](figures/ch04-quad-load.svg){#fig4-quad-load}

The variational route works, with $\delta x_L$, $\xi$ and $\eta$ varied independently ([](#exr4-load)). The free-body route is shorter. The load feels gravity and the cable, $m_L(\ddot x_L + g e_3) = -T q$. The quadrotor feels gravity, thrust and the cable, $m_Q(\ddot x_Q + g e_3) = f R e_3 + T q$. Add them, substitute $\ddot x_Q = \ddot x_L - l\ddot q$, and split along and across $q$ exactly as for the pendulum:

$$
\begin{aligned}
(m_Q + m_L)\big(\ddot x_L + g\,e_3\big) &= \big(q\cdot f R e_3 - m_Q l\,\|\dot q\|^2\big)\,q ,\\
m_Q l\,\dot\omega &= -\,q\times f R e_3 , \qquad \dot q = \omega\times q ,\\
J\dot\Omega + \Omega\times J\Omega &= \tau , \qquad \dot R = R\,\Omega^\wedge .
\end{aligned} \label{eq4-load}
$$

Read three things off. First, the right side of the load equation is a multiple of $q$: measured against free fall, the load can only accelerate along the cable. Second, the load and the cable see the quadrotor only through the vector $f R e_3$, and the attitude equation does not feel the load at all, because the cable is attached at the center of mass. Torque turns the body, the body aims the thrust, the thrust swings the cable, the cable pulls the load. Controllers for this system, with four inputs for eight degrees of freedom, are built on that cascade. Third, gravity is absent from the cable equation.

::: {.thought #thought4-cut name="Cut the motors"}
The quadrotor is flying with the load swinging beneath it. Cut the thrust to zero. What does the cable do?

Set $f = 0$ in [](#eq4-load). Then $\dot\omega = 0$: the cable keeps turning at whatever constant rate it had. Gravity does not swing it down, because gravity accelerates both ends equally. The pair has become a dumbbell tumbling in free fall, its center of mass on a parabola ([](#fig4-cut)). A pendulum needs something to hang from, and a falling quadrotor is not it.
:::

The tension is still available. From the load's free-body equation,

$$
T = \frac{m_L}{m_Q + m_L}\big(m_Q l\,\|\dot q\|^2 - q\cdot f R e_3\big) . \label{eq4-load-tension}
$$

For $m_Q = 1.5$ kg and $m_L = 0.3$ kg in hover, $q = -e_3$, $R = I$ and $f = (m_Q + m_L)g = 17.7$ N, which gives $T = m_L g = 2.94$ N as it should.

![Cut the motors, simulated with [](#eq4-load) for the same two masses and $l = 1$ m. The quadrotor flies level at hover thrust with the load swinging, and the thrust is cut as the cable passes the vertical. The cable keeps turning at a constant rate: the gray lines show it every 0.14 s. The center of mass leaves its level line for a parabola. The tension drops from 4.3 N to 1.4 N and stays there, positive because the cable is still turning.](figures/ch04-cut-strip.png){#fig4-cut anim="figures/ch04-cut.gif"}

::: {.warning #warn4-slack name="A cable cannot push"}
Equations [](#eq4-load) assume a taut cable. They stay smooth when [](#eq4-load-tension) goes negative, and will then simulate a cable that pushes. Monitor the sign of $T$. When it reaches zero the load is in free fall, and the model must switch to two separate bodies until the cable snaps taut again. The flat map of the load fails at the same instant, and well before it the map asks for torques no vehicle has ([](#sec10-limits)).
:::

**More factors.** The recipe does not change with size. Let $n$ quadrotors carry one point mass, each on its own cable $q_i$ of length $l_i$. The configuration space is $\R^3\times(S^2\times\SO(3))^n$ and the same free-body argument gives

$$
\Big(m_L I + \sum_{i} m_i\,q_i q_i\T\Big)\big(\ddot x_L + g\,e_3\big) = \sum_{i}\big(q_i\cdot f_i R_i e_3 - m_i l_i\|\dot q_i\|^2\big)\,q_i , \label{eq4-multi}
$$
$$
m_i l_i\,\dot\omega_i = q_i\times\Big(m_i\big(\ddot x_L + g\,e_3\big) - f_i R_i e_3\Big), \qquad J_i\dot\Omega_i + \Omega_i\times J_i\Omega_i = \tau_i .
$$

With one quadrotor this collapses to [](#eq4-load). Each quadrotor adds inertia to the load only along its own cable, through $m_i q_i q_i\T$. Wu (2017) carries the derivation through for a rigid-body payload, on $\SE(3)\times(S^2\times\SO(3))^n$.

Legged robots fit the same mold. Siravuru's Reaction Mass Biped has two telescoping legs and a torso with movable masses, on a product of three copies of $\SO(3)$ with two leg lengths and three mass positions, of dimension fourteen ([](#tbl2-robots)). In stance the length of the swing leg drops out of the dynamics, which leaves thirteen ([](#sec16-legs)). Its simplest piece already shows something new.

::: {.example #ex4-leg name="A telescoping leg"}
A leg pivots about its foot with attitude $R$ and inertia $J_0$, and carries a mass $m$ at distance $l$ along its axis $R e_3$. A force $f$ extends it and a torque $\tau$ acts at the ankle. The configuration space is $\R\times\SO(3)$, and with $\Pi = I - e_3 e_3\T$, the projection off the leg axis,

$$
\begin{aligned}
m\ddot l &= m l\;\Omega\cdot \Pi\,\Omega - m g\,e_3\cdot R e_3 + f ,\\
J(l)\,\dot\Omega + \Omega\times J(l)\,\Omega + 2 m l\,\dot l\,\Pi\,\Omega &= m g l\,(R\T e_3)\times e_3 + \tau ,
\qquad J(l) = J_0 + m l^2\, \Pi .
\end{aligned} \label{eq4-leg}
$$

The inertia now depends on the configuration, and its rate $\dot J = 2 m l\,\dot l\,\Pi$ appears as a new term: extend the leg and the rotation slows, like a skater opening her arms. The variation $\delta l$ is the ordinary one and $\delta R = R\,\eta^\wedge$ is that of [](#sec4-so3). Nothing else was needed.
:::

## Staying on the manifold while integrating {#sec4-integrate}

The naive integration step treats the nine entries of $R$ as nine numbers: $R_{k+1} = R_k + h\,R_k\Omega_k^\wedge$. [](#sec2-integration) showed what happens:

$$
R_{k+1}\T R_{k+1} = I - h^2\,(\Omega_k^\wedge)^2 \neq I ,
$$

so every step stretches the matrix a little, and it is no longer a rotation. The remedy is the exponential step [](#eq2-exp-step), which moves along the manifold with the device we used to build variations:

$$
R_{k+1} = R_k\exp\!\big(h\,\Omega_k^\wedge\big), \qquad q_{k+1} = \exp\!\big(h\,\omega_k^\wedge\big)\,q_k . \label{eq4-lie-euler}
$$

Each update is a rotation, so $R_{k+1} \in \SO(3)$ and $\|q_{k+1}\| = 1$ to rounding error, for any step size. Higher-order methods run their Runge-Kutta stages on a rotation vector inside the step and apply its exponential at the end.

[](#fig4-integrators) compares the two fourth-order versions on the pendulum [](#eq4-pendulum), swinging and spinning for 60 s with $h = 0.02$ s. Classical RK4 on the entries of $R$ ends with $\|R\T R - I\| = 4.6\times10^{-4}$ and growing. The exponential version holds $10^{-14}$.

![A 3D pendulum integrated for 60 s at $h = 0.02$ s by two fourth-order methods. Left: the entrywise method drifts off $\SO(3)$, the exponential update does not. Right: the energy error of the two methods is nearly the same.](figures/ch04-integrators.svg){#fig4-integrators}

Now look at the right panel. The energy drifts in both: by $2.9\times10^{-4}$ J for the naive method and $2.2\times10^{-4}$ J for the exponential one, out of 7.4 J. Staying on the manifold is not the same as conserving energy. Integrators that do both are built by discretizing Hamilton's principle itself, with an action sum in place of the action integral. [](#sec12-lgvi) builds one for this pendulum and runs it for an hour.

We now have models that are short, global and checkable. What do you do when the system is too large to derive by hand? [](#ch-symbolic) answers that.

## Exercises {#sec4-exercises .unnumbered}

::: {.exercise #exr4-variation-omega}
For the variation [](#eq4-var-s2) on the sphere, show that the angular velocity $\omega = q\times\dot q$ varies as in [](#eq3-delta-omega), $\delta\omega = \dot\xi - 2\,\omega\times\xi$, and that the last term equals $2(\xi\cdot\dot q)\,q$. Check that $\delta(\omega\cdot q) = 0$. Then confirm the formula by finite differences in the notebook.
:::

::: {.exercise #exr4-sph-omega}
Starting from $\dot q = \omega\times q$ with $\omega\cdot q = 0$, prove $q\cdot\dot\omega = 0$ and $q\times\ddot q = \dot\omega$, and derive [](#eq4-sph) from [](#eq4-sph-cross). Show that the energy and $e_3\cdot\omega$ are conserved.
:::

::: {.exercise #exr4-heavy-top}
Apply the three moves to $L = \tfrac12\Omega\cdot J\Omega - m g\,e_3\cdot R\rho$ and derive [](#eq4-pendulum). Show that $e_3\cdot R J\Omega$ is conserved when $\tau = 0$. Why are the other two components of $R J\Omega$ not?
:::

::: {.exercise #exr4-growth}
Linearize Euler's equation about a steady spin on each principal axis and derive [](#eq4-growth). Show that the spin is unstable only about the intermediate axis. In the notebook, measure the growth rate for the book of [](#ex4-book) and compare.
:::

::: {.exercise #exr4-load}
Derive [](#eq4-load) from the Lagrange-d'Alembert principle, using the Lagrangian and virtual work given in [](#sec4-products) and the variations $\delta x_L$, $\delta q = \xi\times q$, $\delta R = R\,\eta^\wedge$. Where does the term $m_Q l\|\dot q\|^2$ come from? Then show that with $f = 0$ and $\tau = 0$ the total energy is conserved.
:::

::: {.exercise #exr4-offset}
The thrust of a quadrotor acts at the body point $r$, measured from the center of mass. Use [](#thm4-rigid-body) to write its equations of motion. For $r = (0.01, 0, 0)$ m and a 1 kg vehicle, what constant torque must the rotors supply in hover?
:::

::: {.exercise #exr4-integrator}
Implement both first-order steps, the naive one and [](#eq4-lie-euler), for the free rigid body. Verify that after $N$ naive steps at constant $\Omega$ the error $\|R\T R - I\|$ is close to $N h^2\|\Omega\|^2$ in the spectral norm. Then tumble the book of [](#ex4-book) with each and plot $\|R J\Omega - R_0 J\Omega_0\|$.
:::

## Notes and sources {#sec4-notes .unnumbered}

Variations on $S^2$ and $\SO(3)$ are treated as in Lee, Leok and McClamroch (2018), which works through pendulums, rigid bodies and multibody systems in this style. The comparison of Euler-angle and $\SO(3)$ models of the 3D pendulum, with the conditioning of the Euler mass matrix, is from Siravuru (2019, ch. 4). He uses the ZXY angle order, singular at 90° of roll; we used the ZYX order of [](#eq-zyx). The Reaction Mass Biped is his chapter 5, which also carries out the derivation in discrete time. Sreenath, Lee and Kumar (2013) derived the quadrotor with a cable-suspended load, written there on $\SE(3)\times S^2$. The form [](#eq4-load), the tension [](#eq4-load-tension) and the multi-quadrotor payload models are in Wu (2017, ch. 3). Teng et al. (2026) write rigid-body dynamics on $\SE(3)$ in Euler-Poincaré form for trajectory optimization. Marsden and Ratiu (1999) is the standard reference for the free rigid body and the stability of its steady spins.

- T. Lee, M. Leok and N. H. McClamroch, [*Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*](https://doi.org/10.1007/978-3-319-56953-6), Springer, 2018.
- J. E. Marsden and T. S. Ratiu, [*Introduction to Mechanics and Symmetry*](https://doi.org/10.1007/978-0-387-21792-5), 2nd ed., Springer, 1999.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- K. Sreenath, T. Lee and V. Kumar, ["Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load"](https://doi.org/10.1109/CDC.2013.6760219), *IEEE Conference on Decision and Control*, 2013.
- S. Teng, R. Zhang, T.-Y. Lin, W. A. Clark, M. Mueller, R. Vasudevan, M. Ghaffari and K. Sreenath, ["LieIPM: Lie group interior point method for direct trajectory optimization of rigid bodies"](https://arxiv.org/abs/2606.10579), arXiv:2606.10579, 2026.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
