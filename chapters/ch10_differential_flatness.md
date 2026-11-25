---
title: Differential flatness
short: Differential flatness
id: ch-flatness
summary: Draw a smooth curve for the load and a heading, and the whole state and all four inputs follow by algebra and differentiation. Why that works, how many derivatives it costs, and where it breaks.
notebook: notebooks/ch10_differential_flatness.ipynb
verify: verify/ch10.py
epigraph: "If you cannot solve the proposed problem try to solve first some related problem."
epigraph_source: "George Pólya, *How to Solve It* (1945)"
---

A quadrotor carries a small load on a one-meter cable. You want the load, not the vehicle, to trace a figure-eight in the air. The vehicle has four inputs: one thrust and three torques. The pair has eight degrees of freedom, so sixteen numbers of state. Which thrust and which torques, as functions of time, make the load do that?

It looks like a hard problem. The equations of motion [](#eq4-load) tell you what the load does when you choose the inputs. You want the reverse. The obvious plan is to guess inputs, integrate, compare and correct, and to hope the corrections converge. A long chain stands between the torque and the load: torque turns the body, the body aims the thrust, the thrust swings the cable, the cable pulls the load.

For this machine you do not solve anything. You differentiate. Write down the curve you want for the load. Take its derivatives, up to the sixth. The cable direction, the quadrotor's position, its attitude, its angular velocity, the thrust and the torque all follow by algebra, one after another ([](#fig10-opening)). No integration, no iteration, no guess. This chapter explains why that is possible, what it costs, and where it stops being possible.

![The puzzle. The load is asked to trace the indigo figure-eight, once every 6.3 s. Everything else in the picture was computed from that curve and its first six derivatives: the cable at seven instants, the quadrotor's path (gray), its tilt. Nothing was integrated to draw it.](figures/ch10-opening.svg){#fig10-opening}

## The weight on a string {#sec10-hand}

Start with something you can hold.

::: {.thought #thought10-hand name="Where must the hand be?"}
Tie a weight to a string of length $l$ and hold the other end. You want the weight to slide back and forth along a level line, 30 cm each way. Where must your hand be at each instant?

Ask what the weight feels. Two forces act on it: gravity and the string. Call the weight's position $x_L$. Newton's law for it is $m_L\ddot x_L = -m_L g\,e_3 + (\text{string})$, so the string must supply

$$
\text{string force} = m_L\big(\ddot x_L + g\,e_3\big).
$$

You chose $x_L(t)$, so you know $\ddot x_L(t)$, so you know this vector. It is the force the weight is missing: what its motion needs beyond what gravity gives. A string can only pull along itself. So the string points along the missing force, and your hand is one string length away in that direction:

$$
x_Q = x_L + l\,\frac{\ddot x_L + g\,e_3}{\|\ddot x_L + g\,e_3\|} . \label{eq10-hand}
$$

Try it with a real string. Move the weight slowly and your hand travels with it, a little less far. With $l = 1$ m and two radians per second, the weight reaches 30 cm when the hand is at 18 cm. Speed up to the pendulum's own rate, $\sqrt{g/l} = 3.13$ rad/s, and your hand hardly moves at all: 1.3 cm. That is what a pendulum is. Go faster still, 4.5 rad/s, and your hand must move *against* the weight: it is at $-23$ cm when the weight is at $+30$ cm ([](#fig10-hand)). You can feel all three.
:::

![The weight is asked to slide along the level indigo line, $x_L = 0.3\cos(\text{rate}\cdot t)$ m, at three rates. The terracotta arrow is the missing force $m_L(\ddot x_L + g\,e_3)$, drawn from the weight. The string lies along it, and the hand (hollow marker) is where [](#eq10-hand) puts it. The thick bar above each hand shows how far it travels. Slow: the hand moves with the weight, less far. At the pendulum's own rate it is nearly still. Fast: it moves opposite.](figures/ch10-hand-strip.png){#fig10-hand anim="figures/ch10-hand.gif"}

Look at what [](#eq10-hand) did. The hand has to obey a differential equation for the weight to follow a path. We never solved it. We read the answer off the path and its second derivative. The price is the two derivatives, and one condition: the missing force must not vanish. If the weight is in free fall, $\ddot x_L = -g\,e_3$, there is no direction to point the string along.

Now name the hand $x_Q$, as we already did, and let a quadrotor be the hand. The quadrotor is itself a body with a path to follow, and it cannot produce any force it likes. So the same question comes back one level up: given the path of the quadrotor, what must its rotors do?

## The point that sees everything {#sec10-flat}

First the smallest case, to fix the idea.

::: {.example #ex10-cart name="A cart"}
A cart of mass $m$ on a rail is pushed by a force $u$: $m\ddot x = u$. Choose any twice-differentiable $x(t)$. The state is $(x, \dot x)$ and the input is $u = m\ddot x$. Nothing to solve. Two things made it trivial. The quantity we chose, the position, determines the whole state through its derivatives. And there is one input for one chosen function, so no freedom is left over and no constraint is left unmet.
:::

A system with this property is called flat, and the chosen quantity a flat output. The cart hides the idea because its flat output is the obvious one. The weight on the string does not: the hand is where the force acts, yet the right thing to choose is the weight, the far end.

::: {.definition #def10-flat name="Flat output"}
A system with $n$ inputs $u$ is **differentially flat** if there is a set of $n$ functions of the state, the inputs and finitely many derivatives of the inputs, written $\varphi$ and called a **flat output**, such that the state and the inputs can be written as functions of $\varphi$ and finitely many of its time derivatives:

$$
\text{state} = \text{a function of } \big(\varphi, \dot\varphi, \dots, \varphi^{(k)}\big), \qquad u = \text{a function of } \big(\varphi, \dot\varphi, \dots, \varphi^{(k)}\big). \label{eq10-flat}
$$

The pair of functions in [](#eq10-flat) is the **flat map**. We write $\varphi^{(k)}$ for the $k$-th time derivative, and call $x^{(3)}$ the **jerk** and $x^{(4)}$ the **snap** of a position $x$.
:::

The notion is due to Fliess, Lévine, Martin and Rouchon (1995). The statement above is the one Kotaru (2022, Definition 2.2) takes from Murray, Rathinam and Sluis (1995). Read what it says. Away from singularities, every smooth curve $\varphi(t)$ is the shadow of exactly one motion of the system, and the motion is reconstructed from the shadow without integrating.

Read also what it does not say. It asserts that a map exists. It does not tell you how to find $\varphi$. It is local: the formulas may divide by something, and where that something is zero the map has a singularity. And it says nothing about stability, disturbances or actuator limits. We will meet all four.

A flat output need not be a coordinate you already use.

::: {.example #ex10-stick name="The sweet spot of a stick"}
Replace the string and weight by a rigid stick, swung in a vertical plane by one end. Your hand applies a force there and no torque. Which point of the stick can you steer along any path?

Let the stick have mass $m$, its center of mass at distance $\|\rho\|$ from the hand, and moment of inertia $J$ about the center of mass for rotation in the plane. A force across the stick at the hand pushes the center of mass one way and spins the stick the other way. At the point a distance $J/(m\|\rho\|)$ beyond the center of mass the two effects cancel. That point, at distance

$$
l = \|\rho\| + \frac{J}{m\,\|\rho\|} \label{eq10-oscillation}
$$

from the hand, feels only the part of your force that lies along the stick. Measured against free fall, it can only accelerate along the stick: it behaves like the weight on a string of length $l$. So [](#eq10-hand) applies with this $l$, and this point is a flat output. It is the **center of oscillation**. For a uniform stick of length 0.9 m it lies at two thirds of the length, 0.6 m from the hand. Ball players know it as the sweet spot of a bat.

A simulation confirms it. Move the hand by [](#eq10-hand) with $l = 0.6$ m and the point at 0.6 m follows a prescribed curve to $3\times10^{-12}$ m over five seconds. Apply the same recipe to the tip, with $l = 0.9$ m, and the tip misses its curve by 4 cm.
:::

The flat output of the stick is a point you have to find. It is not the hand, where the input acts, and not the center of mass, which the equations of motion are written for. That is typical. Now the quadrotor.

## The quadrotor {#sec10-quad}

The quadrotor of [](#eq4-quadrotor) has four inputs, $f$ and $\tau$, so a flat output has four components. Take the position $x$ and a **heading** angle $\psi$. The heading is the angle that [](#eq6-Rd) calls $\psi_d$, read off the actual attitude: the second body axis $R e_2$ is perpendicular to $R_z(\psi)\,e_1$. For a level vehicle it is the yaw angle.

Climb from the bottom, as with the string. The translational equation $m\ddot x = f R e_3 - m g\,e_3$ says that the thrust vector is the missing force:

$$
F = f R e_3 = m\big(\ddot x + g\,e_3\big). \label{eq10-quad-force}
$$

Its length is the thrust and its direction is the third body axis. The heading fixes the rotation about that axis. This is the construction [](#eq6-Rd), with $F$ in place of $F_d$:

$$
f = \|F\|, \qquad R e_3 = \frac{F}{f}, \qquad R e_2 = \frac{R e_3\times R_z(\psi)\,e_1}{\|R e_3\times R_z(\psi)\,e_1\|}, \qquad R e_1 = R e_2\times R e_3 . \label{eq10-quad-map}
$$

So the attitude is a function of the acceleration and the heading. [](#sec6-cascade) used this as a controller: it built the attitude reference from the force it wanted. Here it is a fact about every motion of the vehicle.

The angular velocity needs one more derivative. Differentiate $F = f R e_3$ with $\dot R = R\,\Omega^\wedge$:

$$
\dot F = \dot f\,R e_3 + f\,R\,(\Omega\times e_3).
$$

Since $\Omega\times e_3 = (\Omega_2, -\Omega_1, 0)$, taking the dot product with each body axis gives

$$
\dot f = R e_3\cdot\dot F, \qquad \Omega_1 = -\frac{R e_2\cdot\dot F}{f}, \qquad \Omega_2 = \frac{R e_1\cdot\dot F}{f} . \label{eq10-quad-Omega}
$$

The thrust vector cannot see a spin about its own axis, so $\Omega_3$ comes from the heading. Differentiate $R e_2\cdot R_z(\psi)\,e_1 = 0$:

$$
\Omega_3 = \frac{\Omega_1\,\big(R e_3\cdot R_z(\psi)\,e_1\big) + \dot\psi\,\big(R e_2\cdot R_z(\psi)\,e_2\big)}{R e_1\cdot R_z(\psi)\,e_1} . \label{eq10-quad-Omega3}
$$

One more derivative of $F$ gives the angular acceleration:

$$
\dot\Omega_1 = -\frac{R e_2\cdot\ddot F + 2\dot f\,\Omega_1}{f} + \Omega_2\Omega_3, \qquad
\dot\Omega_2 = \frac{R e_1\cdot\ddot F - 2\dot f\,\Omega_2}{f} - \Omega_1\Omega_3 , \label{eq10-quad-Omega-dot}
$$

and $\dot\Omega_3$ is the time derivative of [](#eq10-quad-Omega3), which brings in $\ddot\psi$ and nothing new from $x$. The notebook writes it out. The torque is then Euler's equation read from right to left, $\tau = J\dot\Omega + \Omega\times J\Omega$. For the bare quadrotor $\dot F = m\,x^{(3)}$ and $\ddot F = m\,x^{(4)}$.

::: {.proposition #thm10-quad name="Flat map of the quadrotor"}
Along any motion of [](#eq4-quadrotor) with $f > 0$ and $R e_1\cdot R_z(\psi)\,e_1 \neq 0$, the state $(x, \dot x, R, \Omega)$ and the inputs $(f, \tau)$ are given by [](#eq10-quad-force) to [](#eq10-quad-Omega-dot) in terms of $x$, its derivatives up to the snap $x^{(4)}$, and $\psi$, $\dot\psi$, $\ddot\psi$. The thrust and the attitude need the acceleration, the angular velocity the jerk, the torque the snap.
:::

No Euler angles appear, and none are needed: the map produces the rotation matrix column by column. Its two conditions are honest singularities. The second, a thrust axis lying along the heading direction, is avoided by choosing $\psi$. The first cannot be chosen away, and [](#sec10-limits) returns to it.

We check this map, and every map that follows, in two ways. First, algebra: put the recovered state and inputs into the library's model and compare accelerations. At twenty random instants on random smooth curves the residual is $9\times10^{-16}$. Second, simulation: feed $f(t)$ and $\tau(t)$ to the model open loop, with no feedback. On the reference of [](#sec6-cascade), a circle of radius 1.5 m flown at 1 rad/s with a vertical wobble, the thrust ranges from 37.9 to 48.2 N around the hover value of 42.6 N, the tilt reaches 9.9°, and after ten seconds the simulated vehicle is $9\times10^{-10}$ m from the curve. Halving the time step divides that error by sixteen, as it should for a fourth-order integrator: it is the integrator's error, not the map's.

So the quadrotor answers the question that the string left open. Stack the two.

## The quadrotor with a cable-suspended load {#sec10-load}

The flat output is the load position $x_L$ together with the quadrotor's heading $\psi$. The load is the weight and the quadrotor is the hand.

**Rung one: the cable.** The load's free-body equation from [](#sec4-products) is $m_L(\ddot x_L + g\,e_3) = -T q$, with $q$ pointing from the quadrotor to the load. So

$$
T = m_L\,\|\ddot x_L + g\,e_3\|, \qquad q = -\frac{\ddot x_L + g\,e_3}{\|\ddot x_L + g\,e_3\|}, \qquad x_Q = x_L - l\,q . \label{eq10-load-map}
$$

This is [](#eq10-hand) with the book's sign for $q$. The tension it gives agrees with [](#eq4-load-tension) to rounding error at every state we tested. The cable's angular velocity is $\omega = q\times\dot q$.

**Rung two: the thrust.** The quadrotor feels gravity, thrust and the cable, $m_Q(\ddot x_Q + g\,e_3) = f R e_3 + T q$. Substituting the first rung,

$$
F = f R e_3 = m_Q\big(\ddot x_Q + g\,e_3\big) + m_L\big(\ddot x_L + g\,e_3\big). \label{eq10-load-force}
$$

**Rung three: the attitude and the torque.** With this $F$, equations [](#eq10-quad-map) to [](#eq10-quad-Omega-dot) apply unchanged.

Now count, honestly ([](#fig10-ladder)). The cable direction $q$ contains $\ddot x_L$. The quadrotor position $x_Q$ contains $q$, so its acceleration contains $\ddot q$, which contains $x_L^{(4)}$. That is the order at which the thrust and the attitude appear. The angular velocity needs $\dot F$, so $x_L^{(5)}$. The torque needs $\ddot F$, so $x_L^{(6)}$.

::: {.proposition #thm10-load name="Flat map of the quadrotor with a cable-suspended load"}
Along any motion of [](#eq4-load) with $T > 0$, $f > 0$ and $R e_1\cdot R_z(\psi)\,e_1 \neq 0$, the state and the inputs are given by [](#eq10-load-map), [](#eq10-load-force) and [](#eq10-quad-map) to [](#eq10-quad-Omega-dot) in terms of $x_L$ and its first six derivatives, and $\psi$, $\dot\psi$, $\ddot\psi$. The state needs derivatives up to the fifth. The sixth enters only the torque.
:::

![The ladder of derivatives. Each quantity sits on the rung of the highest derivative of the flat output it depends on. The left ladder is the bare quadrotor of [](#thm10-quad), the right one the quadrotor with a cable-suspended load of [](#thm10-load). The cable costs two derivatives, and the script that checks this chapter confirms each rung by perturbing one derivative at a time.](figures/ch10-ladder.svg){#fig10-ladder}

The flatness of this system was established by Sreenath, Lee and Kumar (2013), and it is the property that the planners of Kotaru (2022) and the references of Wu (2017) are built on.

::: {.example #ex10-eight name="The figure-eight"}
Take $m_Q = 1.5$ kg, $m_L = 0.3$ kg, $l = 1$ m, the worked example of [](#sec4-products), and ask the load to follow

$$
x_L(t) = \big(1.2\sin t,\; 0.6\sin 2t,\; 0.15\cos 2t\big)\ \text{m}, \qquad \psi = 0 .
$$

One lap takes 6.28 s. The flat map says: the tension stays between 2.76 and 3.14 N around the load's weight of 2.94 N. The cable swings up to 14.5° from the vertical, and the quadrotor is up to 25 cm ahead of the load horizontally. The thrust stays between 16.1 and 18.6 N around the hover value of 17.7 N, the vehicle tilts at most 10.5°, and the torque never exceeds 0.016 N·m.

Now fly it. Start the library's model in the state the map gives at $t = 0$ and apply $f(t)$ and $\tau(t)$ open loop for two laps ([](#fig10-eight)). With a 5 ms time step the load ends $9\times10^{-8}$ m from where it should be, and the error falls by sixteen when the step is halved.
:::

![The figure-eight of [](#ex10-eight), flown open loop on the model of [](#eq4-load). The dashed gray curve with the hollow marker is the curve drawn for the load. The solid indigo trace is the simulated load, driven only by the thrust and torque that the flat map computed. The terracotta arrow is the missing force $m_L(\ddot x_L + g\,e_3)$: the cable lies along it at every instant.](figures/ch10-eight-strip.png){#fig10-eight anim="figures/ch10-eight.gif"}

The cable cost two derivatives and one new condition, $T > 0$. A longer chain should cost more of both.

## Cables with many links {#sec10-chain}

A real cable bends. Kotaru (2022, ch. 7) models it as a chain of $n$ massless links joined by spherical joints, with a point mass at each joint, the model of [](#sec5-scaling). Number the joints from the top: joint 0 is the quadrotor, joint $n$ is the free end, and link $i$ has length $l_i$ and direction $q_i$ from joint $i-1$ to joint $i$, so that $x_i = x_{i-1} + l_i q_i$. Joint $i$ carries the mass $m_i$.^[The thesis numbers the joints from the free end, so its link directions point the other way. We keep the book's convention: a cable direction points away from the quadrotor.]

The hand-and-weight argument applies at the free end without change. The last mass feels gravity and the last link only, so the last link points along that mass's missing force. One joint up, the mass feels gravity and two links, and one of those is now known. Work upward:

$$
T_i\,q_i = T_{i+1}\,q_{i+1} - m_i\big(\ddot x_i + g\,e_3\big), \qquad x_{i-1} = x_i - l_i\,q_i, \qquad i = n, n-1, \dots, 1, \label{eq10-chain}
$$

with $T_{n+1} = 0$. Summed up, the tension vector of link $i$ balances the missing forces of all the masses below it. Each link direction is the unit vector of the tension vector, each tension is its length, and the next joint up follows. At the top, the thrust vector is $F = (m_Q + m_0)(\ddot x_0 + g\,e_3) - T_1 q_1$, and the quadrotor's map takes over ([](#fig10-chain)).

![The recursion [](#eq10-chain) on a cable of three links. Only the curve of the free end is given. The missing force of the lowest mass fixes the last link, which places the next joint. The terracotta arrow is the tension vector of the link being placed. At the top, what remains is the thrust. Then time runs, open loop on the library's model, and the whole cable follows from the one curve.](figures/ch10-chain-strip.png){#fig10-chain anim="figures/ch10-chain.gif"}

Each link costs two derivatives, exactly as the single cable did. So the position of the quadrotor contains $x_n^{(2n)}$, the thrust $x_n^{(2n+2)}$, and the torque $x_n^{(2n+4)}$. With one link, six. With three, ten.

The thesis proves more than this, for several quadrotors along one cable.

::: {.lemma #thm10-chain name="Kotaru (2022), Lemma 7.1"}
Consider a chain of $n$ links carried by $n_Q$ quadrotors, attached at their centers of mass to joints of the chain, with a quadrotor at one end. Call the other end the start. Then the following is a flat output: the position of the start, the heading of every quadrotor, and, for each quadrotor other than the one at the end, the tension vector of the link that leaves it on the side away from the start. The state and the inputs need $2n + 4$ derivatives of the position of the start, two of each heading, and $2k + 2$ derivatives of the tension vector of a quadrotor that sits $k$ joints from the end.
:::

In our numbering the end is joint 0, the start is joint $n$, and a quadrotor at joint $k$ contributes $T_k q_k$. The start may be free, as above, or carry a quadrotor itself. The count works out: $3 + n_Q + 3(n_Q - 1) = 4 n_Q$, the number of inputs. The proof is the recursion above. At a joint that carries a quadrotor, the link above is not determined by the masses below, because the quadrotor's thrust is also unknown. Prescribing that link's tension vector closes the gap, and the thrust follows as the difference. A corollary covers a cable tethered to the ground. The thesis then linearizes the model about a trajectory generated this way and tracks it in simulation with a finite-horizon LQR, for up to ten links and three quadrotors. An earlier paper (Kotaru, Wu and Sreenath, 2018) treats a load on a flexible cable under one quadrotor. Wu (2017) generates the reference for seven quadrotors carrying a rigid payload in the same spirit, from the pose of the payload, the yaw angles of the quadrotors and the tension vectors of four of the cables.

We reproduced the recursion on the library's `quadrotor_cable` model. For one quadrotor and $n = 1, 2, 3$ links, the recovered state and inputs satisfy the model to $8\times10^{-15}$ or better, and each quantity depends on exactly the derivatives counted above. For $n = 1$ the result is [](#thm10-load). With three links of 0.4 m, masses of 0.2, 0.2 and 0.3 kg and a 5 ms time step, an open-loop lap of a figure-eight brings the free end back to its curve within $2\times10^{-5}$ m, and sixteen times closer with half the step. That flight uses derivatives up to the tenth. The cable bends by at most 7° between its first and last link. With two quadrotors on a two-link cable, one at the top and the other at the middle joint or at the start, the position of the start, the two headings and one tension vector determine everything, with a residual below $10^{-12}$.

A tenth derivative should make you uneasy. Where does a curve come from that has one?

## From flat outputs to trajectories {#sec10-trajectories}

Flatness turns trajectory design into curve drawing. Any curve will do, as long as it is smooth enough. "Enough" is set by the ladder: the state must be continuous, so the flat output needs continuous derivatives up to the order at which the last state appears. That is the jerk for the quadrotor and the fifth derivative for the load. The next derivative is the torque, which may jump but should stay small.

That last remark is the idea of Mellinger and Kumar (2011). To pass through given points at given times $t_0 < t_1 < \dots < t_f$, choose the curve that minimizes the integral of the squared snap:

$$
\min_{x(\cdot)}\ \int_{t_0}^{t_f}\|x^{(4)}(t)\|^2\,\dd t \quad\text{subject to}\quad x(t_i)\ \text{given},\quad \dot x = \ddot x = x^{(3)} = 0\ \text{at } t_0 \text{ and } t_f . \label{eq10-minsnap}
$$

Snap is the derivative the torque appears at, so this is the gentlest curve for the rotors. Represent $x$ on each interval by a polynomial of degree seven, with coefficients $c_0, \dots, c_7 \in \R^3$. The cost is a quadratic form in the coefficients, and the conditions are linear in them, including continuity of $x$ up to $x^{(3)}$ at the interior points. So [](#eq10-minsnap) is a quadratic program with equality constraints only, and its KKT conditions, as in [](#eq13-kkt), are one linear system. The three axes decouple.

::: {.example #ex10-waypoints name="Five waypoints"}
Five points span 6 m, with heights between 0 and 1.2 m, and times in proportion to the distances between them. For the 4.34 kg quadrotor and $t_f = 6$ s the minimum-snap curve asks for a thrust between 0.94 and 1.19 times the hover thrust, a tilt of at most 25° and a torque below 0.73 N·m ([](#fig10-waypoints)). Flown open loop, the vehicle arrives $1\times10^{-7}$ m from the last point.

Two checks on the solver. We imposed continuity up to the third derivative. The solution turns out continuous up to the sixth, and only the seventh derivative jumps. So the same curve solves a square linear system with no optimization at all, and the two solutions agree to $10^{-12}$ in the coefficients. And twenty random perturbations that respect the constraints all raise the cost.

Now ask for the same flight in 4 s. Every $k$-th derivative grows by $1.5^k$ and the cost by $1.5^7$. The tilt reaches 53° and the thrust 1.56 times hover. Nothing in [](#eq10-minsnap) knows about the motors.
:::

![The minimum-snap curve of [](#ex10-waypoints) through five waypoints, for $t_f = 6$ s. Left: the curve with the quadrotor drawn at equal time intervals, each attitude computed from the curve by [](#eq10-quad-map). Right: the thrust and tilt that the flat map reports, for $t_f = 6$ s (indigo) and $t_f = 4$ s (terracotta). Feasibility is checked after the fact, on these curves.](figures/ch10-waypoints.svg){#fig10-waypoints}

For the load the same reasoning points two rungs higher. The torque appears at the sixth derivative, so minimize that, with pieces of degree eleven and rest conditions up to the fifth derivative. Through the same five points in 8 s, the tension stays between 2.72 and 3.50 N and the cable swings up to 37°. The rest conditions matter. A minimum-snap curve for the load is at rest only up to its jerk. Its fourth derivative is free at $t_0$, and the flat map then starts the vehicle tilted by 9° and turning at 0.6 rad/s. A vehicle waiting in hover cannot follow that curve from its first instant.

The flat map is also a cheap feasibility check. Thrust, tilt, tension and torque along a candidate curve are algebra. If a bound is violated, stretch the time or move a waypoint, and check again.

## What flatness gives a controller {#sec10-feedforward}

A flat trajectory is an exact solution of the model, state and input together. Three users in this book want one.

**The cascade.** [](#warn6-feedforward) showed the tracking controller diverging when $\Omega_d$ and $\dot\Omega_d$ were set to zero, and asked where they should come from. The flat map supplies them along the reference: $R_d$, $\Omega_d$, $\dot\Omega_d$, and the inputs $f_d$, $\tau_d$ as well, from $x_d$ and $\psi_d$ and their derivatives. On the reference itself, the library's controller with its exact feedforward returns exactly these inputs.

We reran that experiment: the same circle, Lee's gains, a 5 cm offset at the start ([](#fig10-feedforward)). With zero feedforward the error passes 5 m after 14.6 s, as before. With the exact feedforward, differentiated along the closed loop, it decays to $2\times10^{-10}$ m. With the flat feedforward of the reference it also diverges, after 9.2 s. That is not a contradiction. The reference is now an exact solution of the closed loop. But the attitude target $R_d$ is still built from $F_d$, which contains the feedback terms, and the flat feedforward knows nothing about their derivatives. With an attitude loop no faster than the position loop, that omission is enough. Speed the attitude loop up by $\kappa = 1.5$ and the picture changes. With the flat feedforward the error decays toward zero: it is below $2\times10^{-5}$ m over the last five of twenty seconds and still falling. With zero feedforward it settles at 1.9 cm. So flatness supplies the part of the feedforward that belongs to the reference, and removes the steady error. It does not remove the need for a fast inner loop, unless you also differentiate the feedback.

![The experiment of [](#warn6-feedforward), rerun with three feedforwards. Left: Lee's gains. Right: attitude gains scaled by $(\kappa^2, \kappa)$ with $\kappa = 1.5$. Terracotta: $\Omega_d = \dot\Omega_d = 0$. Indigo: the flat feedforward of the reference. Teal: the exact feedforward along the closed loop. The flat feedforward makes zero error a solution. Whether the loop converges to it still depends on the attitude loop being fast enough.](figures/ch10-feedforward.svg){#fig10-feedforward}

**Linearization and MPC.** [](#ch-linearization) and [](#ch-mpc) both need a reference with its input, $(R_d, \Omega_d, \tau_d)$ in [](#def7-feedback) and the stage references of [](#def13-mpc). A flat trajectory provides them consistently, so the linearized error dynamics have no residual forcing term.

**Optimizers.** A direct transcription ([](#sec13-plan)) or a collocation ([](#sec12-collocation)) converges faster from a guess that already satisfies the dynamics. A flat trajectory does, to rounding error. Kotaru (2022, ch. 5) goes one step further for the load, as [](#sec12-collocation) describes: the decision variables are the flat output and its six derivatives, the dynamics are a chain of integrators, and the rest of the state is recovered by [](#eq10-load-map).

## Where it stops {#sec10-limits}

**It is a feedforward, not a controller.** Open loop means no correction. Release the quadrotor of [](#thm10-quad) on the circle with a tilt error of 1° and after five seconds it is 2.1 m from the reference. Give the model 1% more mass than the map assumed and it is 1.2 m off. Start the load's cable 2° from where the map puts it and the load wanders 19 cm from the figure-eight within two laps. The feedback of [](#ch-tracking) and [](#ch-robust) is not optional.

**Zero thrust.** The map divides by $f = m\|\ddot x + g\,e_3\|$. In free fall the thrust vanishes and its direction is undefined.

::: {.thought #thought10-freefall name="Let go"}
Hold the weight on its string and let the weight fall freely for a moment, by dropping your hand with it. Where must the hand be?

Anywhere within reach. The string is slack and transmits nothing, so the weight's motion carries no information about the hand. That is what a singularity of a flat map is: a moment when the flat output stops seeing part of the system. Just before that moment the string is nearly slack, and a tiny sideways acceleration of the weight swings the string direction through a large angle. The hand must then move very fast to stay where the formula puts it.
:::

The numbers follow the thought. Let a quadrotor bob vertically with a slight sideways sway, and raise the bob until the peak downward acceleration is a fraction of $g$. At half of $g$ the map asks for at most 0.33 rad/s of angular velocity. At 0.9 it asks for 1.6 rad/s, and at 0.99 for 16 rad/s and 31 N·m: the angular velocity grows like jerk over thrust, [](#eq10-quad-Omega). Past free fall the map does not fail. It returns an inverted vehicle, pushing down.

**Zero tension, and zero thrust again.** The load's map divides by $T$ as well, and [](#warn4-slack) is the same statement seen from the model's side: [](#eq4-load) is valid only while the cable is taut. Fly the figure-eight of [](#ex10-eight) faster and watch which condition fails first ([](#fig10-limits)). At 1.5 times the rate the torque has grown from 0.016 to 1.2 N·m. At 1.72 times the rate the map breaks, and not at the cable. The quadrotor is being whirled around the load, the centripetal term $m_Q l\,\|\dot q\|^2$ of [](#eq4-load) uses up the whole thrust, and at the crossing point $f = 0$. Beyond that rate the map asks for a vehicle that is inverted as it passes the crossing. The cable itself stays taut much longer. Its tension reaches zero at 4.04 times the rate, where the load is in free fall at the crossing, and just below that, at 3.9, the map asks for 18,000 N·m. A slack cable makes the system hybrid. Sreenath, Michael and Kumar (2013) treat it as a differentially-flat hybrid system, and Kotaru (2022, ch. 5) plans through both modes with a complementarity condition. Both are outside this chapter.

![The figure-eight of [](#ex10-eight) flown at increasing rates. Top: the smallest tension and the smallest thrust over a lap, each as a fraction of its value in hover. The thrust reaches zero first, at 1.72 rad/s. The tension reaches zero at 4.04 rad/s. Bottom: the largest torque the flat map asks for, on a logarithmic scale. It is unbounded at each singularity and already useless well before.](figures/ch10-limits.svg){#fig10-limits}

**High derivatives amplify everything.** Add a ripple of 1 mm at 2 Hz to the load's curve. You could not see it on a plot. Its sixth derivative is a hundred times that of the figure-eight itself, and the largest torque rises from 0.016 to 0.88 N·m. A curve for the load must be smooth to the sixth derivative by construction: a polynomial or a sum of sinusoids, never a filtered measurement. Kotaru (2022, ch. 6) reports a related effect in experiments: at the corners of a path the states generated by flatness jump, and the payload jerks.

**Input limits are not in the map.** A curve can be perfectly smooth and ask for more thrust than the motors have. Check after the fact, as in [](#ex10-waypoints), or put the bounds into an optimization ([](#ch-mpc)).

**The model decides.** Flatness is a property of a model. Add rotor drag and the formulas change, though Faessler, Franchi and Scaramuzza (2018) show that a quadrotor with linear rotor drag is still flat and use the result for feedforward. Attach the cable away from the center of mass and the attitude starts to feel the load, so the ladder of [](#thm10-load) no longer applies as written.

::: {.warning #warn10-recipe name="Not every system is flat, and nobody hands you the flat output"}
Flatness is a gift. With a single input it is the same as being exactly linearizable by static feedback, a classical result of the flatness literature (see Fliess et al., 1995). The cart with a pendulum fails that test: the script for this chapter checks that the distribution involved is not involutive. Give the same pendulum a hand that moves in two directions and it is flat, by [](#ex10-stick). For several inputs no general test is known to us, and no general recipe for finding a flat output. What exists is a catalog of examples (Murray, Rathinam and Sluis, 1995) and a habit worth learning from the cases above: look for the point farthest from the actuators, whose equation of motion contains a single unknown force.
:::

## Exercises {#sec10-exercises .unnumbered}

::: {.exercise #exr10-hand}
For small horizontal motions of the weight, linearize [](#eq10-hand) and show that the hand is at $x_L + (l/g)\,\ddot x_L$ horizontally. For a sinusoid this is $x_L$ times one minus the squared rate times $l/g$. Explain the three regimes of [](#fig10-hand). Then use the exact formula to find how far the hand dips below its rest height at the turning points.
:::

::: {.exercise #exr10-oscillation}
Derive [](#eq10-oscillation) from the planar equations of a rigid stick with a force applied at one end. Show that for any other point of the stick the acceleration, measured against free fall, has a component across the stick that depends on the hand's force. Why does that rule the point out as a flat output of this construction?
:::

::: {.exercise #exr10-Omega3}
Differentiate [](#eq10-quad-Omega3) to obtain $\dot\Omega_3$ and check it against a finite difference in the notebook. Then show that the condition $R e_1\cdot R_z(\psi)\,e_1 \ne 0$ fails exactly when the thrust axis lies in the vertical plane through the heading direction and is horizontal, and propose a heading that avoids it for a given acceleration.
:::

::: {.exercise #exr10-count}
Two quadrotors carry a point-mass load on two cables, the model [](#eq4-multi) with $n = 2$. There are eight inputs. Propose eight flat outputs, using [](#thm10-chain) as a guide, and write the flat map. How many derivatives of each do you need? Check your map against `robots.get("multi_quadrotor_load", n=2)`.
:::

::: {.exercise #exr10-minjerk}
Replace the snap in [](#eq10-minsnap) by the jerk, with pieces of degree five. In the notebook, fly both curves through the waypoints of [](#ex10-waypoints) and compare the torque. At which instants does the minimum-jerk curve ask for an infinite torque, and why does the ladder of [](#fig10-ladder) predict it?
:::

::: {.exercise #exr10-kappa}
This is the second half of [](#exr6-feedforward). With the flat feedforward of the reference, find to one decimal place the smallest $\kappa$ for which the cascade of [](#sec10-feedforward) converges from the 5 cm offset. Compare it with the smallest $\kappa$ for zero feedforward. Then add the derivative of the feedback part of $F_d$ to the feedforward, one term at a time, and watch the threshold move.
:::

::: {.exercise #exr10-slack}
For the figure-eight of [](#ex10-eight) flown at rate 3.9, plot the cable direction near the crossing point. How fast does $q$ turn? Now lower the crossing: replace $0.15\cos 2t$ by $-0.15\cos 2t$, so that the load passes the crossing at the bottom of its vertical motion. Where is the tension smallest now, and at what rate does it vanish?
:::

## Notes and sources {#sec10-notes .unnumbered}

Differential flatness was introduced by Fliess, Lévine, Martin and Rouchon (1995), with a series of mechanical examples. The center of oscillation as the flat output of a body swung by a pivot is a standard example of that literature. Murray, Rathinam and Sluis (1995) collected the flat mechanical systems known at the time. Mellinger and Kumar (2011) used the flatness of the quadrotor, with position and yaw as flat outputs, to generate minimum-snap trajectories through waypoints. The form given here builds the rotation matrix with the construction of Lee, Leok and McClamroch (2010) that [](#eq6-Rd) records, and never names an Euler angle. Sreenath, Lee and Kumar (2013) established the flatness of the quadrotor with a cable-suspended load with the load position and yaw as flat outputs, and Sreenath, Michael and Kumar (2013) extended it to the hybrid system with a slack cable. The chain of links, its recursion and [](#thm10-chain) are from Kotaru (2022, ch. 7 and Appendix A.2), following Kotaru, Wu and Sreenath (2018) and Kotaru and Sreenath (2019). Kotaru (2022, ch. 5 and 6) uses flatness for collocation and for the references of the payload experiments, and Wu (2017) generates references from flat outputs throughout. Faessler, Franchi and Scaramuzza (2018) show that flatness survives linear rotor drag. The feedforward comparison of [](#sec10-feedforward) and the numerical checks are ours.

- M. Faessler, A. Franchi and D. Scaramuzza, ["Differential flatness of quadrotor dynamics subject to rotor drag for accurate tracking of high-speed trajectories"](https://doi.org/10.1109/LRA.2017.2776353), *IEEE Robotics and Automation Letters* 3(2), 2018.
<!-- terms:off -->
- M. Fliess, J. Lévine, P. Martin and P. Rouchon, ["Flatness and defect of non-linear systems: introductory theory and examples"](https://doi.org/10.1080/00207179508921959), *International Journal of Control* 61(6), 1995.
<!-- terms:on -->
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- P. Kotaru and K. Sreenath, ["Multiple quadrotors carrying a flexible hose: dynamics, differential flatness, control"](https://arxiv.org/abs/1911.12650), arXiv:1911.12650, 2019.
- P. Kotaru, G. Wu and K. Sreenath, ["Differential-flatness and control of quadrotor(s) with a payload suspended through flexible cable(s)"](https://arxiv.org/abs/1711.04895), *Indian Control Conference*, 2018.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- D. Mellinger and V. Kumar, ["Minimum snap trajectory generation and control for quadrotors"](https://doi.org/10.1109/ICRA.2011.5980409), *IEEE International Conference on Robotics and Automation*, 2011.
- R. M. Murray, M. Rathinam and W. Sluis, ["Differential flatness of mechanical control systems: a catalog of prototype systems"](https://murray.cds.caltech.edu/Differential_Flatness_of_Mechanical_Control_Systems:_A_Catalog_of_Prototype_Systems), *ASME International Mechanical Engineering Congress and Exposition*, 1995.
- K. Sreenath, T. Lee and V. Kumar, ["Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load"](https://doi.org/10.1109/CDC.2013.6760219), *IEEE Conference on Decision and Control*, 2013.
- K. Sreenath, N. Michael and V. Kumar, ["Trajectory generation and control of a quadrotor with a cable-suspended load: a differentially-flat hybrid system"](https://doi.org/10.1109/ICRA.2013.6631275), *IEEE International Conference on Robotics and Automation*, 2013.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
