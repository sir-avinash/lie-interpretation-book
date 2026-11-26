---
title: How it all comes together
short: Case studies
id: ch-cases
summary: Every robot in the book is the same few factors assembled the same way, and every technique applies to each by one recipe. A library of models, and a map of where each is treated.
verify: verify/robots.py
epigraph: "Nature uses only the longest threads to weave her patterns, so each small piece of her fabric reveals the organization of the entire tapestry."
epigraph_source: "Richard Feynman, *The Character of Physical Law* (1965)"
---

Here are three machines. A biped stands on one foot, with a torso that carries sliding masses. A quadrotor carries a load on a cable. A second quadrotor carries its load on a cable that sags, so we model the cable as five short links. Suppose you have written a simulator for the first. How much new code does the second need? And the third?

Count what the three have in common before you answer. The biped is three rigid bodies and a few lengths. The quadrotor with its load is a position, a cable direction and an attitude. The quadrotor with the sagging cable is a position, five directions and an attitude. Every item on those lists is one of three things: a vector, a direction, or an attitude ([](#fig16-gallery)).

![Seven systems of this book, each moving under its own equations of motion. A sphere marks a factor $S^2$: the direction of a rod, a cable or one link of a cable. A triad marks a factor $\SO(3)$: an attitude. Read the line under each machine against the picture and count. Nothing else appears.](figures/ch16-gallery-strip.png){#fig16-gallery anim="figures/ch16-gallery.gif"}

So the answer is: an equation of motion, and nothing else. The integration step, the check that the state is still on the manifold, the energy audit and the symbolic verification are written once, for the three kinds of factor, and every machine in the figure uses the same ones. This chapter collects the robots of the book in one library built that way, and maps each of them to the places where the earlier chapters treat it.

## One recipe {#sec16-recipe}

Each chapter so far taught one move and tried it on whichever robot showed it best. Put the moves in a row and they form a pipeline ([](#fig16-pipeline)).

![The recipe. Each box is one move, labeled with the chapter that teaches it. The top row is modeling, and ends in equations you can trust. The bottom row is what you do with them. Every arrow carries the same objects: points of the configuration manifold, and variations of them.](figures/ch16-pipeline.svg){#fig16-pipeline}

1. **Name the manifold.** List the parts. A free position is a factor $\R^3$, a cable or a rod is a direction in $S^2$, a rigid body is an attitude in $\SO(3)$ ([](#sec2-products)).
2. **Choose errors.** Each factor comes with an error function, an error vector and a transport map ([](#tbl3-summary)). You do not design them again.
3. **Derive the dynamics.** Write the Lagrangian and the virtual work, vary each factor with its own variation, and read off one equation per factor ([](#def4-dalembert)).
4. **Check them.** Derive the equations a second time by machine and compare at random states ([](#sec5-verify)).
5. **Track.** A PD law on the error vector and the velocity error, plus feedforward, on each factor ([](#eq6-law), [](#eq6-law-s2)). When the force must be aimed, nest the loops ([](#sec6-cascade)).
6. **Linearize.** Vary the equations along a solution to get a linear system in the variations, and design an LQR on it ([](#sec7-variation), [](#sec7-lqr)).
7. **Protect.** Estimate what the model gets wrong, and filter the input so the state never leaves the safe set ([](#sec8-l1), [](#sec8-barrier)).
8. **Plan.** Optimize over the error state, with steps taken by a retraction ([](#def13-mpc), [](#sec13-plan)), or plan in flat outputs ([](#sec10-trajectories)). Discretize the variational principle itself when the simulation must run long ([](#sec12-lgvi)), and use the discrete equations as constraints of a planner ([](#sec12-transcribe)).
9. **Estimate.** Put the uncertainty in the tangent space and let the error live on the group ([](#def14-gaussian), [](#def14-inekf)). On a sphere, keep the estimate on the manifold and the covariance in the constraint subspace ([](#def14-vekf)).

The steps never ask which robot you have. They ask which factors it has.

::: {.thought #thought16-count name="Count the factors"}
Before reading on, write down the configuration manifold of each of these: a satellite with a camera on a two-axis gimbal, a crane with a load on a cable, a quadrotor carrying a pole by a ball joint at its center. Then count the degrees of freedom and the inputs.

The satellite is an attitude and the camera's pointing direction, $\SO(3)\times S^2$, with five degrees of freedom. The crane is a trolley position and a cable direction, $\R\times S^2$, with three. The quadrotor with a pole is $\R^3\times S^2\times\SO(3)$, the manifold of the quadrotor with a cable-suspended load: only the mass distribution along the direction differs. Two machines with the same manifold share everything in the recipe except the Lagrangian.
:::

## The library {#sec16-library}

The folder `case_studies/robots` holds one Python module per robot. Each module is the single place where that model is written down, in the notation of [](#app-notation) with the code names of [](#tblA-code), and each has the same small interface:

```python
import robots
robot = robots.get("quadrotor_load")
robot.manifold, robot.dim             # "R^3 x S^2 x SO(3)", 8
robot.params                          # m_Q, m_L, l, J, g
state = robot.State()                 # x_L, v_L, q, omega, R, Omega
robot.dynamics(state, u)              # accelerations, one per factor
robot.step(state, u, h)               # one step that stays on the manifold
robot.energy(state)                   # kinetic plus potential
robot.on_manifold(state)              # largest constraint violation
robot.lagrangian()                    # the model handed to the symbolic check
robot.labels                          # {"equations": "eq4-load", ...}
```

The module of a robot contains its parameters, its state and its equations of motion. It contains no integrator. `step` is written once: it runs the stages of a fourth-order Runge-Kutta method on one increment per factor and applies the exponential update [](#eq4-lie-euler) at the end, so $\|q\| = 1$ and $R\T R = I$ hold to rounding error for any step size ([](#sec4-integrate)). The symbolic check is also written once. For every robot it derives the equations of motion from `lagrangian()` with the library of [](#ch-symbolic), requires the check of [](#sec5-verify) to pass, and compares the result with the hand-written `dynamics` at random states. Across the whole library the two agree to $4\times10^{-13}$ or better. Where [](#ch-dynamics) to [](#ch-robust) verified a function of their own for the same robot, the library is compared with that function too, so the code here and the equations in the book are provably the same.

::: {.warning #warn16-energy name="On the manifold is not the same as right"}
`on_manifold` returning $10^{-15}$ tells you that the state is a point of the manifold. It does not tell you that it is the right point. A sign error in an equation of motion gives a perfectly valid, perfectly wrong trajectory ([](#ex5-sign)). Check the energy with the inputs off, check a momentum that the model conserves, and check the equations against a second derivation. The library does all three for every robot, and you should do the same for yours.
:::

## The map {#sec16-map}

[](#tbl16-map) and [](#tbl16-plan) are the index of the book by robot, in two parts with the same rows. Read along a row to follow one machine through the recipe. Read down a column to see one move applied to different manifolds. An empty cell means the book does not treat that robot at that stage. It does not mean the stage does not apply.

::: {.table #tbl16-map}

| Robot | Manifold | Dynamics | Symbolic check | Tracking | Linearization | Uncertainty, safety |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Spherical pendulum | [](#eq2-s2-kinematics), [](#def-error-s2) | [](#eq4-sph), [](#eq4-tension) | [](#sec5-pipeline) | [](#eq6-law-s2), [](#thm6-s2) | [](#thm7-s2), [](#ex7-upright) | [](#sec8-cone), [](#ex8-cone) |
| Rigid body, 3D pendulum | [](#def-so3), [](#def-error-so3) | [](#eq4-euler-free), [](#eq4-pendulum) | [](#exr5-rigid-pendulum) | [](#eq6-law), [](#thm6-exp), [](#tbl20-deadline) | [](#thm7-so3), [](#ex7-flip) | [](#sec8-l1), [](#eq8-tilt) |
| Quadrotor | [](#sec2-se3) | [](#eq4-quadrotor) | [](#sec17-model) | [](#thm6-cascade), [](#ex6-inverted), [](#sec17-cascade) | [](#thm7-so3) (attitude), [](#sec17-local) | [](#sec8-quad), [](#ex8-obstacle), [](#sec17-safety) |
| Quadrotor with a cable-suspended load | [](#fig2-quadload), [](#ex2-quadload) | [](#eq4-load), [](#eq4-load-tension) | [](#eq5-wu), [](#ex5-sign), [](#ex5-conservation) | [](#eq6-load), [](#sec18-track) | [](#eq18-linear), [](#eq18-constraint), [](#eq18-lqr) | [](#eq18-adapt), [](#eq18-cone), [](#eq18-floor) |
| Several quadrotors, one load or payload | [](#tbl2-robots), [](#sec19-manifold), [](#eq19-kin) | [](#eq4-multi), [](#eq19-free) | [](#sec5-scaling), [](#sec19-model) | [](#sec19-cascade), [](#eq19-thrust), [](#tbl19-order) | [](#sec19-lqr), [](#tbl19-lqr) | [](#sec8-quad), [](#sec19-safety), [](#eq19-qp) |
| Flexible cable |  | [](#eq5-chain) | [](#sec5-scaling), [](#fig5-chain) | [](#sec19-cable), [](#fig19-cable) | [](#sec19-cable) |  |
| Reaction Mass Biped | [](#tbl2-robots) | [](#ex4-leg), [](#eq20-stance), [](#eq20-impact) | [](#sec20-model), [](#sec20-impact) | [](#sec6-legs), [](#sec20-walk), [](#tbl20-compare) | [](#sec20-recipe) |  |
| Cassie | [](#tbl2-robots), [](#sec20-cassie) |  |  |  |  |  |

: Where the book treats each robot: model and control.

:::

::: {.table #tbl16-plan}

| Robot | Planning, integration | Estimation |
| :-- | :-- | :-- |
| Spherical pendulum | [](#eq12-s2) | [](#def14-vekf), [](#ex14-pendulum) |
| Rigid body, 3D pendulum | [](#thm13-so3), [](#ex13-flip), [](#thm12-lgvi), [](#ex12-opt) | [](#ex3-dead-reckoning), [](#thm14-loglinear) |
| Quadrotor | [](#thm13-se3), [](#thm10-quad), [](#ex10-waypoints) | [](#def14-inekf), [](#sec14-worked), [](#sec17-estimation) |
| Quadrotor with a cable-suspended load | [](#thm10-load), [](#ex10-eight), [](#sec12-collocation) | [](#sec18-estimate) |
| Several quadrotors, one load or payload | [](#sec19-plan), [](#eq19-flat), [](#sec19-integration) |  |
| Flexible cable | [](#thm10-chain), [](#eq10-chain), [](#sec19-cable) |  |
| Reaction Mass Biped | [](#sec12-biped), [](#eq12-rmb), [](#eq20-primitive) |  |
| Cassie |  | [](#sec14-legged) |

: Where the book treats each robot: planning, integration and estimation.

:::

The pendulums and the rigid body filled their rows first. They are where each technique is first shown, because one factor is enough to show it. The large systems inherit the technique factor by factor. The sections below say what has to be added for each, and the case studies of [](#ch-case-quadrotor) to [](#ch-case-legged) fill most of the lower rows. The cells still empty are the honest state of things. No estimator is built for a team, a sagging cable or a walking biped. Nothing protects the biped or the cable against uncertainty. Of Cassie the book has only the manifold and the filter that was tested on it.

## Pendulums {#sec16-pendulums}

A point mass on a massless rod, hinged at a fixed point. Its configuration is one direction $q \in S^2$, from the pivot to the bob. The equation of motion is [](#eq4-sph),

$$
m l^2\,\dot\omega = -m g l\; q\times e_3, \qquad \dot q = \omega\times q ,
$$

and the rod's tension is [](#eq4-tension). Nothing here is hard, which is the point. The pendulum is the smallest system that has a curved configuration space and no group structure, so it is where the direction error, the tracking law [](#eq6-law-s2), the constrained linear state of [](#def7-constrained) and the cone barrier of [](#sec8-cone) are each introduced. It is also the cable of every aerial system below, before anything is attached to it.

```python
pend = robots.get("spherical_pendulum")                  # m = 1 kg, l = 1 m
state = pend.State(q=np.array([1.0, 0.0, 0.0]), omega=np.zeros(3))
swing = pend.rollout(state, pend.zero_input(), 1e-3, 1000)
print(max(pend.tension(s) for s in swing))   # 29.43 N: three times the weight
print(max(pend.on_manifold(s) for s in swing))           # 2e-15
```

Released from the horizontal, the bob pulls hardest at the bottom, with three times its weight. That is the worked example of [](#sec4-sphere), reproduced in three lines.

## The rigid body {#sec16-body}

One attitude $R \in \SO(3)$. Hinged at a pivot, with inertia $J$ about the pivot and center of mass at $R\rho$, the body obeys [](#eq4-pendulum),

$$
J\dot\Omega + \Omega\times J\Omega = m g\,(R\T e_3)\times\rho + \tau .
$$

Set $\rho = 0$ and it is Euler's equation [](#eq4-euler-free) with a torque: the attitude dynamics of every vehicle in this book. The library has both, as `pendulum_3d` and `rigid_body`. Siravuru (2019, ch. 4) uses the 3D pendulum to compare a geometric controller with one written in Euler angles, a comparison that [](#sec6-fl) repeats.

What makes it hard is the topology and not the equation. No continuous controller stabilizes an attitude from everywhere ([](#thm-obstruction)), and a guaranteed rate holds only on a sublevel set ([](#thm6-exp)). The attitude law [](#eq6-law) and the linearization [](#eq7-so3-lin) are exposed by the library as `tracking_controller` and `linearize`, and both are checked against the functions that verified those chapters.

```python
book = robots.get("rigid_body")   # the paperback of Chapter 4
spin = 2 * np.pi * 2.0                                   # two turns per second
state = book.State(R=np.eye(3), Omega=spin * np.array([0.01, 1.0, 0.01]))
flight = book.rollout(state, book.zero_input(), 1e-3, 6000)
flip = next(k for k, s in enumerate(flight) if s.Omega[1] < 0) * 1e-3
print(flip)   # 0.88 s: the first half flip
print(np.ptp([book.momentum(s) for s in flight], axis=0).max())    # 4e-12
```

This is the paperback of [](#sec4-conservation), spun about its intermediate axis. It reverses after 0.88 s, as it did there, and the world-frame angular momentum $R J\Omega$ does not move.

## The quadrotor {#sec16-quadrotor}

A rigid body that is free to translate, with a thrust $f$ along its own third axis and a body torque $\tau$. The manifold is $\SE(3)$, stored as a position and an attitude, and the equations are [](#eq4-quadrotor),

$$
m\ddot x = f R e_3 - m g\,e_3, \qquad J\dot\Omega + \Omega\times J\Omega = \tau .
$$

Six degrees of freedom and four inputs. The force can only point along $R e_3$, so the vehicle must turn before it can accelerate sideways. That one fact shapes everything done with it. Tracking becomes a cascade: a desired force [](#eq6-Fd), an attitude built from it [](#eq6-Rd), and the attitude law inside ([](#thm6-cascade)). Safety needs a barrier that knows about the attitude, [](#eq8-aug), because the plain one cannot use the torque. Lee, Leok and McClamroch (2010) designed the cascade, and the default parameters of the library are their vehicle.

```python
quad = robots.get("quadrotor")                           # the vehicle of Chapter 6
state = quad.State()                                     # at rest at the origin
for k in range(4000):
    ref = robots.quadrotor.circle_reference(k * 2e-3)
    state = quad.step(state, quad.tracking_controller(state, ref), 2e-3)
print(np.linalg.norm(state.x - ref[0]))                  # 3 mm after 8 s
```

The vehicle starts at rest at the origin, 1.5 m from the circle of [](#sec6-cascade), and is on it to 3 mm eight seconds later.

## The quadrotor with a cable-suspended load {#sec16-load}

Hang a point mass $m_L$ from the quadrotor's center of mass on a massless cable of length $l$. Read the manifold $\R^3\times S^2\times\SO(3)$ factor by factor: the load position $x_L$, the cable direction $q$ from the quadrotor to the load, and the attitude $R$. The quadrotor's position is not a coordinate. It is $x_Q = x_L - l\,q$. The equations of motion are [](#eq4-load),

$$
\begin{aligned}
(m_Q + m_L)\big(\ddot x_L + g\,e_3\big) &= \big(q\cdot f R e_3 - m_Q l\,\|\dot q\|^2\big)\,q ,\\
m_Q l\,\dot\omega &= -\,q\times f R e_3 , \qquad \dot q = \omega\times q ,\\
J\dot\Omega + \Omega\times J\Omega &= \tau , \qquad \dot R = R\,\Omega^\wedge .
\end{aligned}
$$

Eight degrees of freedom, four inputs. Torque turns the body, the body aims the thrust, the thrust swings the cable, the cable pulls the load. Two things make this system hard. The first is that ladder: four of its degrees of freedom are reached only through the others ([](#fig10-ladder)), and the cascade [](#eq6-load) of Sreenath, Lee and Kumar (2013) has three nested loops whose rates must be separated. The second is the factor $S^2$ itself, which is an assumption. It says the cable is taut ([](#warn2-taut)). The equations stay smooth when the tension [](#eq4-load-tension) goes negative, and will then simulate a cable that pushes ([](#warn4-slack)).

Wu (2017, ch. 3) derives this model and designs controllers on it. [](#ch-symbolic) opens with his equations and uses them to show what a symbolic check catches.

```python
params = robots.quadrotor_load.chapter6_params()         # 4.34 kg carrying 0.5 kg
robot = robots.get("quadrotor_load", params=params)
swing = robots.expm_so3(np.radians(40) * robots.E1) @ -robots.E3
state, target = robot.State(q=swing), np.array([1.5, 0.0, 0.5])
T_min = np.inf
for k in range(6000):
    u = robot.tracking_controller(state, target)         # eq6-load, eq6-Rd, eq6-law
    T_min = min(T_min, robot.tension(state, u))          # eq4-load-tension
    state = robot.step(state, u, 2e-3)
print(np.linalg.norm(state.x_L - target), T_min)         # 1.0 mm, 4.17 N
print(robot.on_manifold(state))                          # 3e-14
```

This is the flight studied in [](#sec6-cascade): the cable starts 40° off the vertical and the load is sent to a target 1.6 m away. The library reproduces that flight: the load is 1.0 mm from the target after 12 s, and the tension never falls below 4.17 N, the 4.2 N quoted there. The listing watches the tension because the model cannot.

## Several quadrotors and flexible cables {#sec16-teams}

One quadrotor lifts little. Let $n$ of them share a point-mass load, each on its own cable $q_i$. The manifold is $\R^3\times(S^2\times\SO(3))^n$ and the equations are [](#eq4-multi),

$$
\Big(m_L I + \sum_{i} m_i\,q_i q_i\T\Big)\big(\ddot x_L + g\,e_3\big) = \sum_{i}\big(q_i\cdot f_i R_i e_3 - m_i l_i\|\dot q_i\|^2\big)\,q_i ,
$$
$$
m_i l_i\,\dot\omega_i = q_i\times\Big(m_i\big(\ddot x_L + g\,e_3\big) - f_i R_i e_3\Big), \qquad J_i\dot\Omega_i + \Omega_i\times J_i\Omega_i = \tau_i .
$$

With one quadrotor they collapse to [](#eq4-load), and the library checks that they do. Replace the point mass by a rigid payload and the manifold gains the payload's attitude: $\SE(3)\times(S^2\times\SO(3))^n$, of dimension $6 + 5n$ ([](#tbl2-robots)). Wu (2017, ch. 3) derives that model, and Kotaru (2022) reports experiments in which more than one quadrotor carries and controls a payload on cables. The book states the rigid-payload model in words and derives it by machine ([](#sec5-scaling)). The library writes it out as the Lagrange-d'Alembert principle delivers it, one equation per variation, and solves the result as a linear system in the accelerations.

A real cable sags. Kotaru (2022, ch. 7) models it as a chain of $n$ short links with a point mass at each joint, and attaches quadrotors to some of the joints. Hang the chain from a fixed pivot instead and it is the chain of [](#sec5-scaling), on $(S^2)^n$, whose equation for link $i$ is [](#eq5-chain),

$$
\sum_{j=1}^{n} M_{ij}\,\Big[-\,q_i\times(\dot\omega_j\times q_j) + (\omega_j\cdot\omega_j)\; q_i\times q_j\Big]
\;-\; \Big(\sum_{k=i}^{n} m_k\Big)\, g\, l_i\; q_i\times e_3 \;=\; 0,
\qquad
M_{ij} = \Big(\sum_{k=\max(i,j)}^{n} m_k\Big)\, l_i\, l_j .
$$

Let the first joint move and carry quadrotors, and the same matrix $M_{ij}$ appears with one more row and column for the position of that joint. With a single link under a single quadrotor the model is [](#eq4-load) again, written from the quadrotor's end. The library checks that too.

What makes these systems hard is the count. A cable of $n$ links between $n_Q$ quadrotors has $3(n_Q + 1) + 2n$ degrees of freedom and $4 n_Q$ inputs. Kotaru shows that it is nevertheless differentially flat when a quadrotor holds the end of the cable ([](#thm10-chain)), derives its variation-based linearization, and tracks trajectories in simulation with a finite-horizon LQR on up to ten links. Each cable is also one more tension that must stay positive, and several vehicles must keep clear of each other ([](#sec8-quad)).

```python
team = robots.get("multi_quadrotor_payload", n=3)        # three quadrotors
state = team.random_state(np.random.default_rng(1), speed=0.5)
E0 = team.energy(state)
for k in range(1000):
    state = team.step(state, team.zero_input(), 1e-3)    # motors off: free fall
print(team.manifold, team.dim)   # R^3 x SO(3) x (S^2 x SO(3))^3 21
print(abs(team.energy(state) - E0) / E0)                 # 6e-14
print(team.on_manifold(state))                           # 9e-15
```

```python
cable = robots.get("quadrotor_cable", n=5, attach=(0, 5))  # five links
state = cable.random_state(np.random.default_rng(2), speed=0.5)
E0 = cable.energy(state)
for k in range(1000):
    state = cable.step(state, cable.zero_input(), 1e-3)
print(cable.manifold, cable.dim)   # R^3 x (S^2)^5 x SO(3)^2 19
print(abs(cable.energy(state) - E0) / abs(E0))           # 3e-11
```

Twenty-one and nineteen degrees of freedom, with the motors off. Neither module has an integrator of its own, and neither state has left its manifold.

## Legged robots {#sec16-legs}

A leg that pivots about its foot is a 3D pendulum. Let it telescope and it is [](#ex4-leg), on $\R\times\SO(3)$: with $\Pi = I - e_3 e_3\T$,

$$
\begin{aligned}
m\ddot l &= m l\;\Omega\cdot \Pi\,\Omega - m g\,e_3\cdot R e_3 + f ,\\
J(l)\,\dot\Omega + \Omega\times J(l)\,\Omega + 2 m l\,\dot l\,\Pi\,\Omega &= m g l\,(R\T e_3)\times e_3 + \tau ,
\qquad J(l) = J_0 + m l^2\, \Pi .
\end{aligned}
$$

This is [](#eq4-leg), and its new feature is the inertia. $J(l)$ depends on the configuration, and its rate appears in the equation as a term of its own.

Siravuru's Reaction Mass Biped (2019, ch. 5) is built around that feature. On top of the telescoping stance leg sits a torso with its own attitude, carrying three pairs of masses that slide along its axes, and a swing leg with a third attitude. Every mass is balanced about the hip. The consequence is striking: the mass matrix is block diagonal, and the stance leg obeys [](#eq4-leg) exactly, with the hip torque added on the right. The torso obeys an equation of the same shape, with an inertia that depends on the positions of the sliding masses in the way $J(l)$ depends on $l$. The library implements this stance model in full: thirteen degrees of freedom, all actuated. The configuration manifold of [](#tbl2-robots) has dimension fourteen. The fourteenth is the length of the swing leg, which does not enter the stance dynamics because the swing leg is balanced about the hip, and the library leaves it out. The impact at touchdown and the walking controller of the thesis ([](#sec6-legs)) are in a module of their own next to it, with which [](#ch-case-legged) makes the biped walk. The thesis integrates the model with the variational scheme of [](#sec12-biped).

```python
biped = robots.get("reaction_mass_biped")
p, I = biped.params, np.eye(3)
zero = np.zeros(3)
state = biped.State(l=0.5, l_dot=0.0, R1=I, Omega1=zero, RP=I,
                    OmegaP=np.array([0.0, 0.0, 3.0]), R2=I, Omega2=zero,
                    slide=np.full(3, 0.05), slide_dot=zero)
u = (p.m * p.g, zero, zero, zero, zero)                  # the leg holds the weight
momentum = lambda s: s.RP @ biped.torso_inertia(s.slide) @ s.OmegaP
mom0 = momentum(state)
for k in range(800):
    state = biped.step(state, u, 1e-3)
print(state.slide[0], state.OmegaP[2])                   # 0.26 m, 2.21 rad/s
print(np.abs(momentum(state) - mom0).max())              # 7e-14
```

The torso spins at 3 rad/s with its masses free to slide. They are thrown outward, 5 cm to 26 cm in 0.8 s, and the spin drops to 2.21 rad/s while the torso's angular momentum does not change in the twelfth digit ([](#fig16-skater)). A walking robot can use that: it can change how fast it turns without pushing on the ground.

![A torso that changes its own inertia, simulated with the stance model of the Reaction Mass Biped. The torso starts spinning at 3 rad/s about the vertical with its three pairs of reaction masses close to the hip. No force holds them, so they slide outward. The bars show the spin rate falling and the angular momentum staying where it was.](figures/ch16-skater-strip.png){#fig16-skater anim="figures/ch16-skater.gif"}

The same thesis (ch. 6) models Cassie with a floating pelvis in $\SE(3)$, one rotation matrix per leg and two leg lengths, fourteen degrees of freedom of which eight are actuated, and uses the model to plan and track turns in place on a pair of Hovershoes. The library carries the floating-base part of that model, derived from its Lagrangian and checked against the symbolic derivation. It does not carry the ground contact, which is what makes Cassie a walking robot and not a falling one, and the book does not state these equations. The estimation side of legs is further along: a foot on the ground joins the state as one more column of a group element, [](#eq14-contact), and the filter built on it was tested on Cassie ([](#sec14-legged)).

## Adding your own robot {#sec16-add}

The recipe, as a checklist. Each step names the skill that carries it out. The skills are the folders under `skills/`, and `case_studies/README.md` has the same list with the files to copy.

1. **Manifold and state.** Write the configuration as a product of $\R^n$, $S^2$ and $\SO(3)$ factors. Name the fields with the code names of [](#secA-conventions) and [](#tblA-code). Skill: `ch02-manifold-kinematics`. If a chart is tempting, run `ch01-manifold-diagnostics` on it first.
2. **Errors.** Take the error function, error vector and transport map of each factor from [](#tbl3-summary). Skill: `ch03-geometric-error-design`.
3. **Equations of motion.** Lagrangian, virtual work, one variation per factor. Write the result with the inertia on the left. Skill: `ch04-manifold-dynamics-modeling`.
4. **Module.** Copy the module of the closest robot. Fill in the parameters, the state, the list of factors, `dynamics`, `energy` and `lagrangian`. Do not write an integrator.
5. **Verify.** Two lines in `verify/robots.py` derive the model again and run the conservation checks. Add one more: a smaller model that yours must collapse to. Skill: `ch05-geomech-symbolic-derivation`.
6. **Track and linearize.** Add `tracking_controller` and `linearize` when you have them, each citing the law it implements. Skills: `ch06-geometric-tracking-control`, `ch07-variation-lqr-design`.
7. **Protect and plan.** Skills: `ch08-adaptive-safety-design`, `ch13-manifold-mpc-design`, `ch10-flatness-trajectory-design` when the robot has a flat output, `ch12-variational-integrator-design` for long simulations and for transcriptions.
8. **Estimate.** Skill: `ch14-invariant-filter-design`.
9. **Map.** Add a row to [](#tbl16-map) and [](#tbl16-plan), with an empty cell wherever you have not done the work.

Step 5 is the one people skip, and the one that pays. Every model in the library was written by hand and every one was compared with a derivation it could not have influenced. One printed source had a sign that did not survive this comparison: the centrifugal term on the reaction masses, which must push them outward.

Four of these robots deserve more than a section. The quadrotor, the quadrotor with a cable-suspended load, collaborative transport and legged robots are each taken end to end, from the manifold to a flight or a walk, in the case studies that follow.

## Exercises {#sec16-exercises .unnumbered}

::: {.exercise #exr16-second-load}
Add a second load. Hang a point mass from the first one on a second cable, so that the configuration manifold is $\R^3\times S^2\times S^2\times\SO(3)$. Write the Lagrangian, derive the equations of motion by hand, and add the robot to the library. It must collapse to [](#eq4-load) when the second mass is zero. Which existing model is it already, with which arguments?
:::

::: {.exercise #exr16-two-links}
Model the cable of the quadrotor with a cable-suspended load as two links of half the length, with a small mass at the middle joint. Fly both models with the same thrust and torque history, starting from hover with a pitch pulse. How far apart are the two loads after 3 s, and how does the gap depend on the middle mass? What does the two-link model do that the one-link model cannot?
:::

::: {.exercise #exr16-verify}
Take a robot of your own, or the crane of [](#thought16-count): a trolley on a rail with a load on a cable, on $\R\times S^2$. Carry it through steps 1 to 5 of [](#sec16-add). Report the result of the symbolic check, the agreement between your `dynamics` and the derived model, and the energy drift at two step sizes. By what factor should the drift fall when the step is halved?
:::

::: {.exercise #exr16-plant}
Plant an error in `quadrotor_load`: flip the sign of the term $m_Q l\,\|\dot q\|^2$ in [](#eq4-load). Which of the library's checks fail, and which still pass? Explain why the manifold residual cannot see the error, and find an initial state from which the energy check cannot see it either ([](#ex5-sign)).
:::

::: {.exercise #exr16-tension}
For three quadrotors carrying a point-mass load, find thrusts that hold the system in equilibrium with the cables spread 20° from the vertical. Compute the three tensions from Newton's law for each quadrotor. Then reduce one thrust slowly in simulation and find where the first tension reaches zero. What should a simulator do at that instant ([](#warn4-slack))?
:::

::: {.exercise #exr16-skater}
In the run of [](#fig16-skater), apply forces to the reaction masses that pull them back to 5 cm. How much work do the forces do, and where does it go? Then use the masses as an actuator: design forces that bring the spin rate of the torso from 3 rad/s to 2.5 rad/s and hold it there, with no torque at the hip.
:::

## Notes and sources {#sec16-notes .unnumbered}

The models collected here come from three theses and the papers behind them. The quadrotor with a cable-suspended load, the point-mass load carried by several quadrotors and the rigid payload are derived by Wu (2017, ch. 3), following Sreenath, Lee and Kumar (2013) for the single vehicle. Kotaru (2022) treats the same systems with an eye to experiments, and in ch. 7 and its appendix B models the flexible cable as a chain of links carried by quadrotors, with its variation-based linearization. The Reaction Mass Biped and the reduced geometric model of Cassie are due to Siravuru (2019, ch. 5 and 6). His eq. (5.14) prints the centrifugal term on the reaction masses with the opposite sign to the one its own Lagrangian gives. The library uses the sign of the Lagrangian. The quadrotor and its tracking controller are those of Lee, Leok and McClamroch (2010), and the chain of spherical pendulums and the variational method throughout follow Lee, Leok and McClamroch (2018). Every model was derived a second time with `geomech` (Kotaru, 2026). The parameter values of the Cassie model in the library are placeholders, and its ground contact is not modeled.

- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- P. Kotaru, [*geomech: symbolic equations of motion on smooth manifolds*](https://github.com/vkotaru/geomech), version 1.1.0, 2026.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- T. Lee, M. Leok and N. H. McClamroch, [*Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*](https://doi.org/10.1007/978-3-319-56953-6), Springer, 2018.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- K. Sreenath, T. Lee and V. Kumar, ["Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load"](https://doi.org/10.1109/CDC.2013.6760219), *IEEE Conference on Decision and Control*, 2013.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
