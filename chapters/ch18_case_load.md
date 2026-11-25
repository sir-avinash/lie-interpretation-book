---
title: "Case study: a quadrotor with a cable-suspended load"
short: Quadrotor with a load
id: ch-case-load
summary: One robot through the whole recipe. A load on a cable is planned, tracked, linearized, protected against an unknown mass and a slack cable, and estimated, in one flight over a wall.
notebook: notebooks/ch18_case_load.ipynb
verify: verify/ch18.py
---

A quadrotor of 1.5 kg has picked up a package on a 1 m cable. It must carry the package across a room, over a wall 1 m high, and set it down on a platform, 6 s after it starts. Here is the obvious way to do it. Draw a smooth course for the package. Fly the quadrotor along the same course, one cable length higher. Let the package hang. Will that work?

Try it on the model of [](#eq4-load), with the vehicle held on its course by the position loop [](#eq6-Fd). The quadrotor does what it is told. The package does not. It trails as the vehicle pulls away and swings out to 48° from the vertical. At the appointed instant it is 27 cm from the platform, moving at 0.8 m/s, with the cable 18° off the vertical.

The obvious plan treats the package as luggage. The equations say otherwise. The package is the thing being flown, and the quadrotor is the hand that flies it ([](#thought10-hand)). This chapter flies it properly: one robot, taken through every step of the recipe of [](#sec16-recipe), in one flight ([](#fig18-mission)).

![The mission, drawn from the plan of [](#sec18-plan). The indigo curve is the course drawn for the load: up from the floor, over the wall, down to the platform. The quadrotor is drawn once a second where the flat map puts it. It leads the load while the load speeds up and trails it while the load slows down. The cone under the third pose is the set the cable must stay in.](figures/ch18-mission.svg){#fig18-mission}

## The mission {#sec18-mission}

Here are the numbers, once. They are the defaults of the robot model library ([](#sec16-library)), so they agree with [](#sec4-products) and [](#ex10-eight).

- **The robot.** Quadrotor mass $m_Q = 1.5$ kg, cable length $l = 1$ m, and the inertia matrix of the library, with principal moments near 0.02, 0.025 and 0.04 kg·m². The manifest says the load weighs $m_{L,0} = 0.3$ kg. The subscript 0 marks what the controller believes, as in $J_0$ ([](#sec8-baseline)).
- **The course.** The load starts 0.3 m above the floor, hanging at rest. It must pass over a wall that stands 2.5 m away, 30 cm above its top, and come to rest 5 m away on a platform, 0.5 m up. The course swings wide on both sides of the straight line, is 6.7 m long, and is to take 6 s.
- **What the controller does not know.** The package weighs $m_L = 0.45$ kg, half as much again as the manifest says. A crosswind pushes on the quadrotor with up to 2 N for 2.5 s in the middle of the flight.
- **What it can see.** The quadrotor knows its own position, velocity, attitude and angular velocity. Of the cable it has only a reading of where the load hangs below it, $l\,q$ plus noise of 3.2 cm on each coordinate, a hundred times a second.
- **What it must respect.** The thrust is limited to 35.3 N, twice the believed hover thrust, and each torque component to 1 N·m. The cable must stay taut, with a tension of at least $T_{\min} = 0.5$ N. And it must stay inside a cone of half-angle $\theta_{\max} = 25^\circ$ about the downward vertical, so that the load stays in view of the camera that reads it.

The controller runs at 500 Hz, so $h = 2$ ms.

This robot is the right second case study for three reasons. It has four inputs and eight degrees of freedom, so half of what it does is reached only through the other half. The thing you care about, the load, is two steps removed from the things you command: torque turns the body, the body aims the thrust, the thrust swings the cable, and only the cable moves the load. And the model rests on an assumption, a taut cable, that the controller itself must keep true ([](#warn2-taut), [](#warn4-slack)).

::: {.thought #thought18-first name="Which way first?"}
The package hangs at rest and must start moving forward. Before you read on, decide what the quadrotor does first.

The load feels two forces: its weight, and the cable. The cable can only pull along itself. So for the load to accelerate forward the cable must lean, with the quadrotor ahead of the load. To get ahead the quadrotor must accelerate forward, and to do that it must first pitch its nose down. So the first thing that happens is a torque. The load learns about it four integrations later. That count is why the load's course will have to be smooth to the sixth derivative.
:::

## The manifold and its errors {#sec18-manifold}

Step 1 of the recipe is to name the manifold. [](#ex2-quadload) did: $\R^3\times S^2\times\SO(3)$. Read it factor by factor ([](#fig2-quadload)). The load position $x_L \in \R^3$. The cable direction $q \in S^2$, pointing from the quadrotor to the load. The attitude $R \in \SO(3)$. The quadrotor's position is not a coordinate. It is $x_Q = x_L - l\,q$.

The middle factor is a claim about the world. It says the load is exactly one cable length from the quadrotor, always. A slack cable is not a point of this manifold, and [](#warn2-taut) is where the book first says so. Keep that in mind: two sections of this chapter exist only to keep the claim true.

Step 2 is to choose errors, and there is nothing to design. [](#tbl3-summary) supplies them per factor. For the load, $e_x = x_L - x_d$ and $e_v = v_L - \dot x_d$. For the cable, the direction error and the velocity error of [](#eq3-e-omega),

$$
e_q = q_d\times q, \qquad e_\omega = \omega - (I - q q\T)\,\omega_d ,
$$

where the projection is the transport map [](#eq3-transport-s2) at work: $\omega_d$ is tangent at $q_d$, and must be carried to $q$ before it can be subtracted. For the attitude, $e_R$ and $e_\Omega$ of [](#def-error-so3). Eight error components for eight degrees of freedom, and no angle anywhere.

## Dynamics you can trust {#sec18-dynamics}

Steps 3 and 4. The equations of motion are [](#eq4-load),

$$
\begin{aligned}
(m_Q + m_L)\big(\ddot x_L + g\,e_3\big) &= \big(q\cdot f R e_3 - m_Q l\,\|\dot q\|^2\big)\,q ,\\
m_Q l\,\dot\omega &= -\,q\times f R e_3 , \qquad \dot q = \omega\times q ,\\
J\dot\Omega + \Omega\times J\Omega &= \tau , \qquad \dot R = R\,\Omega^\wedge ,
\end{aligned}
$$

and the tension is [](#eq4-load-tension),

$$
T = \frac{m_L}{m_Q + m_L}\big(m_Q l\,\|\dot q\|^2 - q\cdot f R e_3\big) .
$$

Two things in these lines will be used again and again. The load can only accelerate along the cable: the right side of the first line is a multiple of $q$. And the second line does not contain $m_L$. The cable swings the same way whatever hangs on it, because the load pulls along the cable and that pull has no moment about it.

We do not derive the equations again. [](#ch-symbolic) derived them by machine and settled a sign that a printed source had wrong ([](#eq5-wu), [](#ex5-sign)). The library carries the result, and this chapter repeats the check of [](#sec5-verify) before using it: the symbolic derivation from the library's Lagrangian passes with an error of $4\times10^{-8}$, against the bar of $10^{-5}$.

The wind needs one more term. A force $F_e$ on the quadrotor acts at the point where the thrust acts, so it adds to the thrust vector: replace $f R e_3$ by $f R e_3 + F_e$ in the first two lines. We derived that model a second time as well, with $F_e$ in the virtual work, and the two agree to $10^{-13}$ at random states.

For integration the library's `step` is used throughout ([](#sec4-integrate)): fourth-order stages on one increment per factor, and the exponential update [](#eq4-lie-euler) at the end. Over every flight in this chapter $\big|\|q\| - 1\big|$ and the entries of $R\T R - I$ stay below $10^{-13}$.

So the model can be trusted as far as its assumption goes. What course should the load take?

## Planning the load, not the vehicle {#sec18-plan}

Step 8, taken early, because everything after it needs a reference. The load position and the heading are a flat output ([](#thm10-load)). Draw a curve $x_L(t)$, and the rest follows by [](#eq10-load-map) and [](#eq10-load-force):

$$
T = m_L\,\|\ddot x_L + g\,e_3\|, \qquad q = -\frac{\ddot x_L + g\,e_3}{\|\ddot x_L + g\,e_3\|}, \qquad x_Q = x_L - l\,q ,
$$
$$
F = f R e_3 = m_Q\big(\ddot x_Q + g\,e_3\big) + m_L\big(\ddot x_L + g\,e_3\big),
$$

and then the attitude, the angular velocity and the torque from $F$ by [](#eq10-quad-map) to [](#eq10-quad-Omega-dot). The state needs five derivatives of $x_L$ and the torque needs the sixth ([](#fig10-ladder)).

So the curve is the one [](#sec10-trajectories) recommends for a load: through five waypoints, minimizing the integral of the squared sixth derivative, with pieces of degree eleven that are at rest up to the fifth derivative at both ends. The sixth is free there, so the torque jumps, by 0.08 N·m, at the instant the course begins. The heading is held at $\psi_d = 0$.

Now read the feasibility off the flat map, before anything flies ([](#fig18-plan)). With the believed masses and 6 s for the course:

- the cable leans at most 21.3° from the vertical, inside the 25° cone;
- the tension stays between 2.68 and 3.36 N around the believed weight of 2.94 N;
- the thrust stays between 15.4 and 20.2 N, the tilt below 11.2°, the torque below 0.08 N·m;
- the quadrotor is up to 36 cm from the point above the load, ahead of it or behind;
- the load clears the wall by 26 cm at the lowest point of its crossing.

Flown open loop on the library model from the instant the course begins, with nothing but the thrust and torque of the map, the load arrives $2\times10^{-9}$ m from the platform. The plan is an exact solution of the model.

![What the flat map says about two plans, before any flight. Indigo: the course in 6 s. Terracotta: the same course in 5 s. Top: the angle of the cable from the vertical, with the edge of the cone. Bottom: the tension and the thrust, each as a fraction of its believed hover value. The faster plan leaves the cone and is still far from a slack cable.](figures/ch18-plan.svg){#fig18-plan}

Ask for the same course in 5 s and the map answers at once: 31.1° of swing, outside the cone. That is how a constraint should be met first, at the drawing board, by stretching the time. We keep the 5 s plan for [](#sec18-safety), where something else has to stop it.

The same figure shows which constraint is close. Even on the 5 s plan the tension never drops below 90% of the load's weight. [](#sec10-limits) found the same on the figure-eight: on a smooth course the cable is the last thing to give, and the thrust and the cone go first.

A plan is a feedforward. [](#sec10-limits) flew one open loop with a cable 2° off and lost 19 cm in two laps. Something must close the loop.

## Tracking a moving load {#sec18-track}

Step 5. The cascade of [](#sec6-cascade) climbs the ladder from the bottom. With $F_L = -k_x e_x - k_v e_v + (m_Q + m_{L,0})(\ddot x_d + g\,e_3)$ the force wanted on the pair, [](#eq6-load) reads

$$
\begin{aligned}
q_d &= -\frac{F_L + m_Q l\,\|\dot q\|^2 q}{\|F_L + m_Q l\,\|\dot q\|^2 q\|}, \\[4pt]
f R e_3 \;&\leftarrow\; \big(F_L\cdot q + m_Q l\,\|\dot q\|^2\big)\,q \;+\; m_Q l\; q\times\big({-k_q}\,e_q - k_\omega\,e_\omega\big).
\end{aligned}
$$

Call the thrust vector on the right $F_d$. The attitude loop turns it into $R_d$ by [](#eq6-Rd), into a thrust $f = F_d\cdot R e_3$, and into a torque by [](#eq6-law).

The library's controller is this law for a fixed target, with $\omega_d = 0$ and $\Omega_d = 0$. For a moving reference two things are added, and the flat map supplies both. The cable law becomes the full direction law [](#eq6-law-s2),

$$
\dot\omega = (I - q q\T)\,\dot\omega_d - (q\cdot\omega_d)\,\dot q - k_q\,e_q - k_\omega\,e_\omega ,
$$

with $\omega_d$ and $\dot\omega_d$ of the reference cable, and this $\dot\omega$ takes the place of the bracket $-k_q e_q - k_\omega e_\omega$ in [](#eq6-load). The attitude law [](#eq6-law) gets $\Omega_d$ and $\dot\Omega_d$ of the reference vehicle. On the reference itself the controller then returns exactly the thrust and torque of the flat map, to $10^{-14}$.

The gains: $k_x = 7.2$ N/m and $k_v = 5.04$ N·s/m put the position loop at 2 rad/s, and $k_q = 25$ s⁻², $k_\omega = 7$ s⁻¹ put the cable loop at 5 rad/s, as in [](#sec6-cascade). For the attitude, $k_R = 8$ N·m and $k_\Omega = 1.2$ N·m·s, which gives a slowest pole of 7.6 s⁻¹. That respects what [](#sec6-cascade) and [](#sec10-feedforward) found: the load cascade diverges when the attitude loop is no faster than the loops outside it, and feedforward from the reference does not remove that need.

With the believed mass, no wind and the true state, this controller follows the course to 0.4 mm at worst.

Now take the flat feedforward away: $\omega_d = \dot\omega_d = 0$ and $\Omega_d = \dot\Omega_d = 0$, with everything else as before. The worst error is 87 cm, the cable swings to 48°, and the load strikes the wall, 1 cm below its top ([](#fig18-feedforward)). Restore the cable's two terms alone and the worst error is 8 cm. So nearly all of the damage is done in the cable loop.

A line of arithmetic says why. Without $\omega_d$ the term $-k_\omega e_\omega$ is a brake on any motion of the cable. The reference cable turns at up to 0.81 rad/s, so the brake must be balanced by a direction error of about $k_\omega/k_q \times 0.81 = 0.23$ rad, or 13°. A thrust of 17.7 N aimed 13° wrong is 4 N in the wrong direction, and the position loop needs 55 cm of error to push back that hard. A faster attitude loop does not help here. The lag is in the middle loop.

![The same course, the same gains, with and without the flat feedforward. The dashed gray curve is the reference of the load. Indigo: the cascade with $\omega_d$, $\dot\omega_d$, $\Omega_d$ and $\dot\Omega_d$ from the flat map. Terracotta: the same cascade with all four set to zero. Watch the terracotta cable hang back at the start and swing through at the end.](figures/ch18-feedforward-strip.png){#fig18-feedforward anim="figures/ch18-feedforward.gif"}

The cascade works, and its gains were chosen by rule of thumb, loop by loop. Is there a way to choose all of them at once?

## Linearizing on the whole manifold {#sec18-linear}

Step 6, and the first thing in this chapter that the book has not done before. [](#ch-linearization) linearized a pendulum and a rigid body. This robot is both, coupled.

Start from an observation. The cable equation $m_Q l\,\dot\omega = -q\times f R e_3$ is the pendulum [](#eq7-pendulum) with $m_Q$ for the mass and a force on its bob of

$$
F = m_Q g\,e_3 - f R e_3 . \label{eq18-pendulum}
$$

In words: seen from the load, the quadrotor is the bob, and the thrust is what pushes it. Every result of the book about a pendulum on $S^2$ therefore applies to the cable after this substitution. This section uses it once, and the sections on safety and on estimation use it again.

The linear state stacks one variation per factor, in the notation of [](#tbl7-dictionary):

$$
s = (\delta x_L,\ \delta v_L,\ \xi,\ \delta\omega,\ \eta,\ \delta\Omega) \in \R^{18}, \qquad \delta u = (\delta f,\ \delta\tau) .
$$

Vary [](#eq4-load) along a solution. Write $F_d = f_d R_d e_3$ for the thrust vector of the reference and $c = q_d\cdot F_d - m_Q l\,\|\omega_d\|^2$ for the bracket of the first line. Everything on the right below is evaluated on the reference:

$$
\begin{aligned}
\delta\dot x_L &= \delta v_L ,\\
(m_Q + m_L)\,\delta\dot v_L &= q_d\,\big((q_d\times F_d)\cdot\xi\big) - c\;q_d\times\xi - 2 m_Q l\,(\omega_d\cdot\delta\omega)\,q_d + q_d\,(q_d\cdot\delta F) ,\\
\dot\xi &= \omega_d\times\xi + (I - q_d q_d\T)\,\delta\omega ,\\
m_Q l\;\delta\dot\omega &= -F_d^\wedge\, q_d^\wedge\,\xi - q_d\times\delta F ,\\
\dot\eta &= -\Omega_d\times\eta + \delta\Omega ,\\
J\,\delta\dot\Omega &= \big((J\Omega_d)^\wedge - \Omega_d^\wedge J\big)\,\delta\Omega + \delta\tau ,
\end{aligned} \label{eq18-linear}
$$

with the two factors joined by the variation of the thrust vector,

$$
\delta F = R_d e_3\,\delta f - f_d\,R_d\,e_3^\wedge\,\eta , \label{eq18-coupling}
$$

and with the state confined by [](#eq7-s2-constraint), applied to its cable part:

$$
q_d\cdot\xi = 0, \qquad (q_d\times\omega_d)\cdot\xi + q_d\cdot\delta\omega = 0 . \label{eq18-constraint}
$$

Lines three and four are [](#eq7-s2-lin) for the pendulum [](#eq18-pendulum). Lines five and six are [](#eq7-so3-lin) with $\rho = 0$. The code takes both blocks from the library as they stand. Only the second line and [](#eq18-coupling) are new, and each is one application of the product rule. The system has 18 components, 2 constraints and 16 free directions, twice the degrees of freedom.

A linearization is a claim, so we tested it three ways.

*Against the symbolic derivation.* The library of [](#ch-symbolic) linearizes the same Lagrangian. Its rows and ours agree to $10^{-15}$ at random states, for random $s$ in the constraint subspace, after one correction that is worth knowing. An equation on the sphere holds only after projection by $I - q q\T$ ([](#secA-conventions)). The symbolic rows for the cable are the variation of the unprojected equation $E_\xi$. The projection varies too, and contributes $-(\xi\times q)(q\cdot E_\xi)$. For the lone pendulum of [](#ex7-s2-check) that term vanishes. Here it does not, and without it the cable rows differ by amounts of order one.

*Against finite differences on the manifold.* Perturb the reference state at $t = 3$ s by $\pm\epsilon s_0$, moving each factor along its own exponential, and perturb the inputs too. Fly both neighbors open loop for one second and difference them, as in [](#ex7-s2-check). With $\epsilon = 10^{-3}$ the result differs from the solution of [](#eq18-linear) by $2\times10^{-6}$ of its size, and with $\epsilon = 10^{-4}$ by $2\times10^{-8}$: second order, as it should be.

*Controllability where the state lives.* At hover the rank test on the 18 numbers reports 16, the broom of [](#thought7-broom) again. On the constraint subspace the reduced pair of [](#def7-constrained) has rank 16 of 16. Along the moving reference we integrated the reachability Gramian from 3 s to 5 s: sixteen nonzero singular values, and the seventeenth is $10^{-13}$ of the first. The linearized robot is controllable along this course, with four inputs for sixteen states.

Kotaru (2022, ch. 6) prints the linearization of the first four lines, with the thrust vector as the input, and uses it for predictive control of the load and the cable. Our blocks agree with his on the constraint subspace. The attitude is what [](#eq18-linear) adds.

Now design on it. Integrate the Riccati equation [](#eq7-riccati) backwards along the reference, with weights 40 and 8 on $\delta x_L$ and $\delta v_L$, 20 and 2 on $\xi$ and $\delta\omega$, 10 and 0.2 on $\eta$ and $\delta\Omega$, and 0.5 and 30 on $\delta f$ and $\delta\tau$. Feed the gain what [](#tbl7-dictionary) says equals $s$ to first order, as in [](#def7-feedback):

$$
\begin{bmatrix} f \\ \tau \end{bmatrix} = \begin{bmatrix} f_d \\ \tau_d \end{bmatrix} - K(t)\,\big(e_x,\ e_v,\ e_q,\ \omega - \omega_d,\ e_R,\ e_\Omega + \Omega_d\times e_R\big). \label{eq18-lqr}
$$

One gain matrix of 4 by 18 replaces three pairs of scalar gains.

![The load's distance from its reference after a bad start: the load 37 cm off and the cable swung away from the vertical by 10°, 30° and 50°. Indigo: the cascade. Teal: the LQR of [](#eq18-lqr). The LQR is back within 5 cm in half the time from 10° and 30°. From 50° it leaves the picture.](figures/ch18-lqr.svg){#fig18-lqr}

[](#fig18-lqr) is the honest comparison. On the reference the two are equally good: each follows the course to 0.4 mm at worst. From a start with the load 37 cm off and the cable swung 10° away, the LQR is back within 5 cm after 1.7 s and the cascade after 3.8 s. From 30° it is 2.0 s against 4.2 s. From 50° the LQR saturates the torque, loses the vehicle and never comes back. The cascade recovers in 5.0 s. That is the trade of [](#sec7-local), on a larger system: the linear design is better where it is valid, and the geometric one is valid much farther out. The mission below uses the cascade, because what happens next is not small.

## What you did not pick up {#sec18-uncertain}

Step 7, first half. The package weighs 0.45 kg, not 0.3 kg, and the wind blows.

::: {.thought #thought18-heavier name="A heavier package"}
You carry a bucket on a rope and someone drops a stone into it. What changes: how the rope swings, or how high the bucket hangs?

Your arm sags. The swing does not change at all. The second line of [](#eq4-load) said so: the cable equation does not contain $m_L$. The extra weight pulls along the rope, and along the rope is exactly where the first line says a load can be pushed or pulled. So an unknown load mass does not disturb the cable loop or the attitude loop. It disturbs one scalar, the force along the cable.
:::

Make that exact. Let the true equations have $m_L$ and $F_e$, and let the controller's model have $m_{L,0}$ and no wind. The true motion is the motion of the believed model with thrust vector $f R e_3 + \Delta$, where

$$
\Delta = F_e - (m_L - m_{L,0})\big(\ddot x_L + g\,e_3\big) . \label{eq18-lumped}
$$

This is the lumped uncertainty of [](#eq8-lumped) for this robot. There it was a torque in the body frame. Here it is a force in the world frame, and it enters exactly where the thrust vector enters, so it is matched in the sense of [](#sec8-uncertainty) for the two outer loops: if you knew $\Delta$ you would subtract it from $F_d$. The second term is along $q$, as the bucket said. In hover it is 1.47 N straight down.

What does it do to the errors? The position loop balances a constant $\Delta$ with a constant error, $e_x = \Delta/k_x$. For the package that is $0.15\times9.81/7.2 = 20.4$ cm of sag. The simulation gives 20.6 cm, and the load crosses the wall 9 cm above it where the plan had 26.

The attitude needs no adaptation. The cable is attached at the center of mass, so neither the load nor the wind exerts a torque on the body, and the inertia is known. The adaptive law of [](#sec8-l1) on $\SO(3)$ has nothing to estimate here. Its architecture is what we borrow.

**A predictor.** Copy the believed model of the two outer loops, driven by the thrust vector that was applied and by the estimate, with gains $\tilde k_v = \tilde k_\omega = 10$ s⁻¹ on the prediction errors $\tilde e_v = v_p - v_L$ and $\tilde e_\omega = \omega_p - \omega$:

$$
\begin{aligned}
(m_Q + m_{L,0})\big(\dot v_p + g\,e_3\big) &= \big(q\cdot(f R e_3 + \hat\Delta) - m_Q l\,\|\dot q\|^2\big)\,q - (m_Q + m_{L,0})\,\tilde k_v\,\tilde e_v ,\\
m_Q l\,\dot\omega_p &= -\,q\times(f R e_3 + \hat\Delta) - m_Q l\,\tilde k_\omega\,\tilde e_\omega .
\end{aligned} \label{eq18-predictor}
$$

Subtract the true equations. With $\tilde\Delta = \hat\Delta - \Delta$, the prediction errors obey $(m_Q + m_{L,0})\,\dot{\tilde e}_v = (q\cdot\tilde\Delta)\,q - (m_Q + m_{L,0})\,\tilde k_v\,\tilde e_v$ and $m_Q l\,\dot{\tilde e}_\omega = -q\times\tilde\Delta - m_Q l\,\tilde k_\omega\,\tilde e_\omega$. They are driven by the estimation error and by nothing else, as in [](#eq8-pred-dyn). No transport map is needed for $\tilde\Delta$ this time: both forces are in the world frame.

**The adaptation law and the filter.** These are [](#eq8-adapt) and [](#eq8-filter) unchanged, with the projection [](#eq8-proj) onto a ball of 6 N:

$$
\dot{\hat\Delta} = k_\Delta\,\operatorname{Proj}\big(\hat\Delta,\ y\big), \qquad y = -(q\,q\T)\,\tilde e_v + l\;\tilde e_\omega\times q, \qquad \dot\Delta_f = k_f\,(\hat\Delta - \Delta_f) , \label{eq18-adapt}
$$

with $k_\Delta = 2000$ and $k_f = 6$ rad/s, and the cascade asks for $F_d - \Delta_f$. The direction $y$ is chosen as in [](#sec8-l1), so that the estimation error cancels from the rate of $\tfrac12(m_Q + m_{L,0})\|\tilde e_v\|^2 + \tfrac12 m_Q l^2\|\tilde e_\omega\|^2 + \tfrac{1}{2k_\Delta}\|\tilde\Delta\|^2$. The script checks that cancellation at random states.

Be exact about what carries over from [](#ch-robust). [](#thm8-l1) is a theorem about the attitude predictor on $\SO(3)$, and it does not cover this one. What we have is the computation just described, which is the first step of that proof, for a different predictor. If $\Delta$ and its rate are bounded, the same argument would bound these prediction errors, but $\Delta$ contains $\ddot x_L$, and we have not proved that bound. And as there, nothing here is a statement about the tracking error, or about the loop with the filter in it. So we measure.

The measurement is kind ([](#fig18-adapt)). With the true state fed back, the error over the course drops from 20.5 cm to 3.0 cm RMS, the load ends 5 mm from the target, and the filtered estimate ends within 0.007 N of the true $\Delta$. During the gust it trails the truth by up to 0.37 N of the 2 N: a cutoff of 6 rad/s passes a pulse of 2.5 s only in part, which is the trade of [](#warn8-cutoff). The prediction error shrinks as the gain grows, while the gain is moderate. Its largest value is $7.2\times10^{-3}$, $1.9\times10^{-3}$ and $7.8\times10^{-4}$ m/s for $k_\Delta = 125$, 500 and 2000. At $k_\Delta = 8000$ it is worse again, $5\times10^{-3}$ m/s. The update is an explicit step of 2 ms, and at that gain the step is at the edge of its own stability. A sampled adaptation law has a largest useful gain, and here it is not far above the one we use.

![Left: the load's distance from its reference over the mission with the true state fed back, without adaptation (terracotta) and with it (indigo). The shaded band is the gust. Right: the vertical component of the lumped uncertainty [](#eq18-lumped), dashed gray, and its filtered estimate. The estimate finds the extra weight in the first second.](figures/ch18-adapt.svg){#fig18-adapt}

One kind of wind is not matched. A force on the load itself has a part across the cable, which accelerates the load directly and which no thrust vector can cancel at that instant. [](#exr18-wind-on-load) asks what remains of the scheme then.

## Keeping the cable honest {#sec18-safety}

Step 7, second half. Two constraints, and both are about the factor $S^2$: the cable must stay inside its cone, and it must stay a cable.

**The cone.** The barrier function is [](#eq8-cone) with $q_c = -e_3$,

$$
H(q) = q\cdot q_c - \cos\theta_{\max} ,
$$

and $\dot H = \omega\cdot(q\times q_c)$ by [](#eq8-cone-dot). Its second derivative is [](#eq8-cone-ddot) for the pendulum [](#eq18-pendulum):

$$
\ddot H = -\|\omega\|^2\,(q\cdot q_c) - \big(I - q q\T\big)q_c\cdot\frac{F}{m_Q\,l} , \qquad F = f R e_3 . \label{eq18-cone}
$$

The thrust vector appears in the second derivative, so the condition is [](#eq8-rd2), $\ddot H + (\alpha_1 + \alpha_2)\dot H + \alpha_1\alpha_2 H \ge 0$, and it is linear in $F$. We use $\alpha_1 = \alpha_2 = 12$ s⁻¹.

**The tension.** By [](#eq4-load-tension) the tension is linear in the thrust vector as well, with no derivative at all. Asking for $T \ge T_{\min}$ is

$$
q\cdot F \;\le\; m_Q l\,\|\omega\|^2 - \frac{m_Q + m_{L,0}}{m_{L,0}}\,T_{\min} . \label{eq18-taut}
$$

**The filter.** Both conditions go into the program [](#eq8-safety-filter): of all thrust vectors that satisfy [](#eq8-rd2) and [](#eq18-taut), take the one closest to the $F_d$ that the cascade asked for. It is a quadratic program in three unknowns with two constraints, and it replaces $F_d$ before the attitude loop sees it. The estimate $\Delta_f$ is added to $F$ in both conditions, with the warning of [](#sec8-limits) that no theorem covers that step.

There is a difference between the two constraints, and it matters. The thrust $f$ is a real input, and [](#eq18-taut) contains it directly. So the tension can be enforced exactly, at the attitude the vehicle has now, as a floor on the thrust:

$$
f \;\ge\; \frac{(m_Q + m_{L,0})\,T_{\min}/m_{L,0} - m_Q l\,\|\omega\|^2}{-\,q\cdot R e_3} \qquad\text{whenever } q\cdot R e_3 < 0 . \label{eq18-floor}
$$

The cone is not like that. The direction of the thrust vector is not an input. It is the attitude, which the torque reaches two integrations later. So [](#thm8-rd2) applies to the cable driven by a thrust vector, and the vehicle only approximates that thrust vector.

Now the two tests that the mission's own plan would never give.

*A hurried course.* Fly the 5 s plan, which leans the cable to 31.1°, with the true state fed back ([](#fig18-cone)). Without the filter the cable follows the plan out to 30.1°, with $H$ as low as $-0.041$. With the filter it stops at 25.6°. That is 0.6° outside: the smallest $H$ is $-0.004$, and an input is at its limit for 4% of the flight. The barrier held as well as the attitude loop could deliver its thrust vector, and no better.

Look at the price. The load can no longer keep up with a reference that needs 30°. It falls 54 cm behind, and at the planned instant of arrival it is 33 cm from the platform with the cable at 13°. Three seconds later it is 3 cm from the target. This is what [](#sec8-cone) promised and no more: safety while the reference is unsafe, and nothing about tracking. A filter is the last line of defense. The first is the plan.

![The cable direction seen from below, on the 5 s plan. The ink circle is the edge of the cone, $H = 0$, and the forbidden side is shaded. Dashed gray: the cable direction of the reference, which leaves the cone twice. Terracotta: the cable without the filter. Indigo: with it. The indigo path turns back at the edge while the terracotta one follows the reference out.](figures/ch18-cone-strip.png){#fig18-cone anim="figures/ch18-cone.gif"}

*A descent that is too steep.* Ask the load to drop 1.5 m in 1.1 s. That curve accelerates downward at up to 1.42 times gravity, which no cable can do: it would have to push. The flat map does not object. Past free fall it returns a cable that points up and a tension that is positive again, as [](#thought10-freefall) warned. Without the filter the thrust is cut to zero, the tension with it, and from that instant the simulation is of a robot that does not exist ([](#warn4-slack)). With the floor [](#eq18-floor) the thrust never drops below 3.4 N and the tension never below 0.79 N. That is more than the 0.5 N asked for, because the load is heavier than the formula believes. A lighter load than believed would get less, and [](#exr18-lighter) asks how much. The load falls as fast as a taut cable allows, is 25 cm behind at worst, and catches up.

::: {.warning #warn18-composition name="Blocks that pass alone can fail together"}
Both tests above fed the filter the true state. On the estimate of [](#sec18-estimate) the hurried course goes a little worse: the cable reaches 25.7° and the load falls 60 cm behind. Condition [](#eq8-rd2) multiplies $\dot H$, and so the estimated angular velocity, by $\alpha_1 + \alpha_2 = 24$ s⁻¹, and near the edge the filter answers every outward jitter of the estimate with thrust.

One combination failed outright. Take the flat feedforward away and keep the filter, on the hurried course with the true state. The cascade without $\omega_d$ lags, the filter acts for 60% of the flight, an input sits on its limit for more than half of it, and the vehicle is lost: the simulation was stopped with the load 20 m from its reference. Each block had passed its own test. Size the cone for the estimator's error, test the blocks together, and do not let a filter stand in for a plan.
:::

Both constraints are now watched. But the watchman has been reading the true cable direction, which no sensor provides.

## Estimating the cable {#sec18-estimate}

Step 9. The quadrotor's own state is taken as known here. [](#def14-inekf) is the filter that would supply it, and [](#ch-case-quadrotor) flies with one. What is left is the factor that is not a group: the cable direction $q$ and its angular velocity $\omega$, from a reading $y = l\,q + n_y$ of the load's offset.

That is the setting of [](#sec14-s2), and [](#def14-vekf) applies after one change. Its model was a free pendulum. Ours is the pendulum [](#eq18-pendulum), pushed by the thrust that the controller itself applied, and its matrix $A$ is the cable block of [](#eq18-linear). Everything else is as defined there: the estimate $(\hat q, \hat\omega)$ on the sphere and its tangent plane, a covariance $\Sigma$ of rank four with $C\,\Sigma = 0$, the correction $s^+ = -\tilde K\tilde y$ with $\tilde y = l\,\hat q - y$ applied by $\oplus$ of [](#eq14-oplus), and the covariance carried to the new tangent plane and projected by $\Pi_C$ of [](#eq14-E).

Two facts make this estimator easy to live with. The cable equation does not contain the load mass, so the estimator's model is right even though the manifest is wrong. And the wind, which the model does not contain, enters through the thrust vector, so the filtered estimate $\Delta_f$ can be added to the thrust the estimator is told about. We do that.

The load's state then follows from the kinematics of [](#sec18-manifold): $\hat x_L = x_Q + l\,\hat q$ and $\hat v_L = v_Q + l\,\hat\omega\times\hat q$.

Run on the mission, with the controller still fed the truth, the estimator is within 0.57° RMS in direction and 0.04 rad/s in angular velocity ([](#fig18-estimate)). The readings themselves are 1.8° wrong per coordinate. Started 40° wrong, the estimate is within 4.1° after its first reading and within 2° after 0.03 s. $\|\hat q\| - 1$, $\hat q\cdot\hat\omega$ and $C\,\Sigma$ never exceed $10^{-15}$. Before trusting our code we ran it on the pendulum of [](#ex14-pendulum), where a constant thrust vector $m_Q g\,e_3$ makes the cable a free pendulum: it reproduces the estimates of that chapter's script to $10^{-15}$.

Its consistency is less tidy than its accuracy. The NEES in the four constrained dimensions should be 1. It is 0.62 with the truth in the loop, so the filter is cautious, and 0.90 when the controller flies on the estimate. The process noise was set by hand to cover the gust, and a number set by hand is not a noise model.

![The estimator on the mission. Top: the angle between the estimated and the true cable direction, started 40° wrong. The gray dots are the raw readings. Bottom: the error of the estimated angular velocity, with the gust shaded. The reading has no information on $\omega$ except through the model, and the model does not know the wind.](figures/ch18-estimate.svg){#fig18-estimate}

Every block is now in place. Do they work when each feeds the next?

## The whole flight {#sec18-flight}

Everything on: the 6 s plan, the cascade with flat feedforward, adaptation, the cone and tension filter, the controller fed by the estimator, the heavier package, the gust, the noisy readings ([](#fig18-flight)).

![The mission flown with every block on. The dashed gray curve and hollow marker are the reference of the load. The indigo trace is the load. The thin gray line under the quadrotor is the true cable and the indigo line the estimated one: at this scale they coincide. The bars show the tension against its floor $T_{\min}$ and the barrier $H$ of the cone against zero. Time runs at the rate printed in the frame.](figures/ch18-flight-strip.png){#fig18-flight anim="figures/ch18-flight.gif"}

- **Tracking.** The load follows its course to 3.9 cm RMS and 8.4 cm at worst, and clears the wall by 27 cm.
- **Set-down.** At the planned instant the load is 3.8 cm from the platform, moving at 0.10 m/s, with the cable 1.8° from the vertical. Over the next 2.5 s the cable never passes 2.2°, and the load ends 0.5 cm from the target.
- **The cable.** The tension stays between 4.0 and 5.0 N: the package weighs 4.4 N. The largest swing is 21.1°, and the smallest value of $H$ is 0.027. The cable never came within 3.9° of the edge of the cone.
- **The estimate.** 0.67° RMS in direction, 0.07 rad/s in angular velocity.
- **The inputs.** Thrust between 16.6 and 21.9 N of the 35.3 N allowed, torque at most 0.33 N·m of 1 N·m, no saturation.
- **The manifold.** $\big|\|q\| - 1\big| \le 3\times10^{-15}$, and no entry of $R\T R - I$ exceeds $2\times10^{-14}$, after 4500 steps.

Now remove one block at a time and fly the same mission ([](#tbl18-ablation)).

::: {.table #tbl18-ablation}

| Mission flown with | Error, RMS (cm) | Error, worst (cm) | Over the wall (cm) | Off the platform at 6.5 s (cm) | Cable at 6.5 s | Largest swing | Smallest $H$ | Smallest $T$ (N) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| Everything on | 3.9 | 8.4 | 27 | 3.8 | 1.8° | 21.1° | 0.027 | 4.0 |
| No flat feedforward | 56.7 | 116.6 | 49 | 40.3 | 17.1° | 31.4° | −0.052 | 0.8 |
| No adaptation | 20.4 | 23.3 | 11 | 22.0 | 2.7° | 18.9° | 0.040 | 4.0 |
| No cone and tension filter | 3.9 | 8.4 | 27 | 3.8 | 1.8° | 21.1° | 0.027 | 4.0 |
| Truth in place of the estimate | 3.0 | 6.1 | 26 | 3.0 | 1.3° | 21.2° | 0.026 | 4.0 |
| Load treated as a disturbance | 43.5 | 79.4 | 27 | 26.8 | 17.8° | 48.2° | −0.240 | 3.8 |

: The mission with one block removed at a time. Each row is one flight with the same package, gust and readings. The last row is the controller of the opening: [](#eq6-Fd) on the bare quadrotor, no adaptation and no filter.

:::

Read it row by row.

*No flat feedforward* is the worst flight in the table, worse than ignoring the cable altogether. The cascade without $\omega_d$ lags its reference, the cable leaves the cone, the filter acts during 17% of the flight, and an input is on its limit for 11% of it. The plan of [](#ch-flatness) is not a convenience for this robot. It is what makes the cascade of [](#ch-tracking) a tracking controller.

*No adaptation* costs the 20 cm of sag that [](#sec18-uncertain) computed. The load clears the wall, by 11 cm where 26 were planned, and is set down 22 cm from the platform.

*No filter* is the same flight, digit for digit. The filter never acted: the plan kept the cable 3.9° inside the cone and the tension eight times above its floor. On a mission whose plan respects the constraints the filter is insurance, and [](#sec18-safety) showed both what the insurance pays and what it costs.

*Truth in place of the estimate* is the price of not seeing the cable: 3.9 cm against 3.0 cm, and a torque that jitters up to 0.33 N·m where the truth needs 0.08 N·m.

*Load treated as a disturbance* is the opening of this chapter. The quadrotor is well controlled and the load is not controlled at all.

## Honest limits {#sec18-limits}

**The cable was taut because we kept it so.** Every equation in this chapter lives on $S^2$. When the tension reaches zero the load is a projectile and the quadrotor is a quadrotor, on $\R^3\times\R^3\times\SO(3)$, until the cable snaps taut with an impact. That is a hybrid system. Sreenath, Michael and Kumar (2013) treat it as a differentially-flat hybrid system, and Kotaru (2022, ch. 5) plans through both modes with a complementarity condition, by the direct collocation of [](#sec12-collocation). We named the hybrid model and did not build it. Our floor [](#eq18-floor) avoids the second mode. It cannot fly through it.

**Six derivatives of a plan are six derivatives of its flaws.** The reference came from a polynomial, so its sixth derivative is exact. A reference filtered from measurements, or replanned in flight with a corner in it, is another matter: [](#sec10-limits) put a ripple of 1 mm on a load's curve and the torque grew fifty-fold. Kotaru (2022, ch. 6) reports the same from experiments, where the states generated by flatness jump at the corners of a path and the load is jolted.

**The guarantees are per block.** [](#thm6-s2) is about a direction tracked by a force. [](#thm8-rd2) is about a cable driven by a thrust vector. [](#def14-vekf) comes with no stability theorem at all. Nothing proved in this book covers the cascade with its feedforward taken from a plan, the adaptation law [](#eq18-adapt), the barrier acting through an attitude loop, or any two of them together, and [](#warn18-composition) showed a composition that failed. Sreenath, Lee and Kumar (2013) state stability results for the three loops of the cascade in turn, and Wu (2017) and Kotaru (2022) justify stacking them by a time-scale argument ([](#sec6-cascade)). The flight of [](#sec18-flight) is evidence. It is not a proof.

**The model is kind.** There are no motor dynamics, no drag on the load, no delay between reading and thrust, and the cable is attached exactly at the center of mass. [](#fig8-filter) showed how little delay an adaptive loop tolerates when its cutoff is high. The barrier knows no input limits ([](#sec8-limits)), and the hurried course ran into them. The error-state MPC of [](#def13-mpc) carries the limits inside its optimization and would be the next thing to try on [](#eq18-linear).

**What experiments say.** The simulations here are ours. Kotaru (2022) flew this system. He reports that on hardware the cable gains $k_q$, $k_\omega$ decide whether the loop is stable, with the attitude loop simply run fast, that the variation-based linearization is observed to have a larger region of attraction, and that the filter on $S^2$ was validated on a pendulum and on a quadrotor with a cable-suspended load ([](#sec14-s2)). Those are his statements about his vehicle. Ours are about a model.

## Exercises {#sec18-exercises .unnumbered}

::: {.exercise #exr18-course}
Change the mission in the notebook. Move the wall to 1.5 m from the start and raise it to 1.4 m, and choose waypoints that clear it by 30 cm. Find the shortest travel time for which the flat map keeps the cable inside the 25° cone and the thrust below its limit. Which of the two binds first, and where on the course?
:::

::: {.exercise #exr18-mass}
Fly the mission with the adaptation off for $m_L = 0.15$, 0.3, 0.45 and 0.6 kg. Check that the load's final height error is $(m_L - m_{L,0})\,g/k_x$ each time. Then turn the adaptation on and plot $\Delta_f\cdot e_3$ against time. From its final value, how well could you weigh the package in flight?
:::

::: {.exercise #exr18-lighter}
The tension constraint [](#eq18-taut) uses the believed load mass. Show from [](#eq4-load-tension) that when [](#eq18-taut) holds with equality the true tension is $T_{\min}\,\dfrac{m_L\,(m_Q + m_{L,0})}{m_{L,0}\,(m_Q + m_L)}$. Evaluate it for the package of this chapter, and for one of 0.1 kg. Fly the steep descent of [](#sec18-safety) with both and compare. What value of $T_{\min}$ would you set if the load could be anything from 0.1 to 0.6 kg?
:::

::: {.exercise #exr18-wind-on-load}
Let the wind push on the load with a force $F_e$ as well. Show that the first line of [](#eq4-load) gains the term $(I - q q\T)F_e/m_L$ on the acceleration of the load, and the cable equation the term $(m_Q/m_L)\,q\times F_e$. Which part of this is matched, in the sense of [](#sec18-uncertain)? Give the load a steady 0.5 N of crosswind in the notebook. At what angle does the cable settle, and what does the adaptation law make of it?
:::

::: {.exercise #exr18-rank}
Carry out the rank test [](#eq7-ltv-rank) at hover by hand for [](#eq18-linear). The matrices are constant, so $\Gamma_{k+1} = -A\,\Gamma_k$. Which columns appear at each step, and at which step does the load position first appear? Compare with the ladder of [](#fig10-ladder).
:::

::: {.exercise #exr18-margin}
Repeat the hurried course of [](#sec18-safety) on the estimate, with the filter's cone tightened to 23°, 22° and 21° while the true limit stays at 25°. Find the largest filter cone for which the true cable stays inside 25°. Then triple the noise of the reading and find it again. State a rule for the margin in terms of the estimator's error.
:::

::: {.exercise #exr18-lqr}
In the notebook, replace the cascade by the LQR [](#eq18-lqr) in the full mission, with the adaptation off. How large is the sag now, and why is it not $\Delta/k_x$? Then find, to the nearest 5°, the initial swing from which the LQR no longer recovers, with the torque limit at 1 N·m and at 3 N·m. Is the limit of the linear design set by the linearization or by the actuator?
:::

## Notes and sources {#sec18-notes .unnumbered}

The model [](#eq4-load) and the cascade [](#eq6-load) are due to Sreenath, Lee and Kumar (2013), who also established that the load position and the heading are a flat output. Wu (2017, ch. 3) derives the model in the form used here and designs controllers on it, and his chapter 6 has the barrier functions on manifolds that [](#sec18-safety) applies, following Wu and Sreenath (2015). The relative-degree-two condition in its linear form is the exponential barrier of Nguyen and Sreenath (2016). Kotaru (2022) treats the same robot with experiments in view: the cascade with the feedforward terms of the cable (ch. 6), the variation-based linearization of the load and cable with the thrust vector as input and a predictive controller on it (ch. 6), the filter on $S^2$ (ch. 4), planning by direct collocation with the slack mode included (ch. 5), and L1 adaptive control on $\SO(3)$ (ch. 3), after Hovakimyan and Cao (2010). The variation-based linearization on $S^2$ and $\SO(3)$ is from Wu and Sreenath (2015). The curve through the waypoints follows Mellinger and Kumar (2011), two derivatives higher.

New in this chapter, and checked only by the script that accompanies it: the linearization [](#eq18-linear) on the full product manifold with its three validations, the predictor [](#eq18-predictor) and the adaptation law [](#eq18-adapt) for a lumped force, the tension constraint [](#eq18-taut) with the thrust floor [](#eq18-floor), the filter of [](#def14-vekf) driven by the applied thrust, and every number of the mission. None of these comes with a proof for the closed loop. Every model was derived a second time with `geomech` (Kotaru, 2026).

- N. Hovakimyan and C. Cao, *L1 Adaptive Control Theory: Guaranteed Robustness with Fast Adaptation*, SIAM, 2010.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- P. Kotaru, [*geomech: symbolic equations of motion on smooth manifolds*](https://github.com/vkotaru/geomech), version 1.1.0, 2026.
- D. Mellinger and V. Kumar, ["Minimum snap trajectory generation and control for quadrotors"](https://doi.org/10.1109/ICRA.2011.5980409), *IEEE International Conference on Robotics and Automation*, 2011.
- Q. Nguyen and K. Sreenath, ["Exponential control barrier functions for enforcing high relative-degree safety-critical constraints"](https://ieeexplore.ieee.org/document/7524935), *American Control Conference*, 2016.
- K. Sreenath, T. Lee and V. Kumar, ["Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load"](https://doi.org/10.1109/CDC.2013.6760219), *IEEE Conference on Decision and Control*, 2013.
- K. Sreenath, N. Michael and V. Kumar, ["Trajectory generation and control of a quadrotor with a cable-suspended load: a differentially-flat hybrid system"](https://doi.org/10.1109/ICRA.2013.6631275), *IEEE International Conference on Robotics and Automation*, 2013.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
- G. Wu and K. Sreenath, ["Safety-critical and constrained geometric control synthesis using control Lyapunov and control barrier functions for systems evolving on manifolds"](https://hybrid-robotics.berkeley.edu/publications/ACC2015_Geometric_CBFCLF.pdf), *American Control Conference*, 2015.
- G. Wu and K. Sreenath, ["Variation-based linearization of nonlinear systems evolving on SO(3) and S²"](https://doi.org/10.1109/ACCESS.2015.2477880), *IEEE Access* 3, 2015.
