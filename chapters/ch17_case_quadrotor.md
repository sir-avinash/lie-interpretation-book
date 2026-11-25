---
title: "Case study: a quadrotor, end to end"
short: Quadrotor
id: ch-case-quadrotor
summary: One quadrotor is taken through the whole recipe in a single mission, from a bad estimate and an upside-down throw to a hover at the target, and each block is removed in turn to see what it bought.
notebook: notebooks/ch17_case_quadrotor.ipynb
verify: verify/ch17.py
---

You have four blocks, and each one has passed its own tests. A tracking controller that rights a quadrotor from upside down ([](#ex6-inverted)). An adaptive law that cancels a torque nobody modeled ([](#ex8-l1)). A safety filter that steers a quadrotor around a ball ([](#ex8-obstacle)). A filter that finds the attitude from an estimate that starts 170° wrong ([](#tbl14-mc)). Connect the output of each to the input of the next. Strap a package to one arm, throw the vehicle into the air, and send it through a ball it was not told about. Does it fly?

We tried. With the blocks wired exactly as their chapters leave them, the vehicle meets the ball, rolls to 148° and falls through the floor. Nothing in any block was wrong. Two wires were missing, and no single chapter could have drawn them, because each of them joins two chapters.

This chapter flies one mission, on one vehicle, in one simulation, with every block of [](#fig16-pipeline) connected ([](#fig17-mission)). At each stage it says where the idea was taught, what it becomes for this vehicle, and what the simulation measured. Then it removes the blocks one at a time.

![The mission. The vehicle is carried from the pad while its filter, which starts 140° wrong, reads four landmarks. It is thrown upside down, rights itself, and flies a minimum-snap course through four waypoints to a target. A ball sits on the course, a crosswind blows across it, and a package of unknown position hangs off one arm.](figures/ch17-mission.svg){#fig17-mission}

## The mission {#sec17-mission}

The vehicle is the quadrotor of the library ([](#sec16-quadrotor)), with the default parameters that [](#ch-tracking) and [](#ch-flatness) used: mass 4.34 kg and inertia $J_0 = \operatorname{diag}(0.0820,\ 0.0845,\ 0.1377)$ kg·m². We give its rotors a total thrust of at most $f_{\max} = 100$ N and a torque of at most $\tau_{\max} = 8$ N·m about each axis. These two limits are our assumption, not Lee's.

A package of 0.8 kg is strapped on at $(0.20,\ -0.12,\ -0.05)$ m in the body frame. Its weight is known, so the controller uses the total mass $m = 5.14$ kg and a hover thrust of 50.4 N. Its position is not known. The center of mass moves by $\rho = (31,\ -19,\ -8)$ mm, the thrust no longer passes through it, and the principal inertias rise by 14%, 34% and 27%. The controller still believes $J_0$.

Time runs from the moment the filter is switched on, with the time step $h = 5$ ms for the simulation, the IMU and the controller alike.

- **0 to 6 s.** An operator carries the vehicle from its pad to the release point, motors off. The filter believes the vehicle is level, at rest, at the origin, with heading zero. In truth it is 2.06 m away with a heading of 140°.
- **6 s.** The operator throws it upward at 3 m/s with a flip. At release the thrust axis points 151° from the vertical and the body turns at 4.1 rad/s.
- **6 to 8.5 s.** The reference waits at the first waypoint, $(0, 0, 3)$ m.
- **8.5 to 16 s.** The reference flies a course of 13.6 m through four waypoints to the target $(9, 0, 2)$ m, while the heading turns by 90°. A ball of radius $r_o = 0.6$ m sits on the course at $x_o = (4.2,\ 0.9,\ 3.5)$ m. The planner does not know about it. From 9.5 to 15 s a crosswind pushes with 3 N, and a gust torque of up to 0.58 N·m acts on the body.
- **16 to 18 s.** Hover at the target.

"End to end" means that nothing in this list is simulated apart from the rest. The controller never sees the true state. It sees the estimate, which comes from an IMU and from readings of four landmarks. The safety filter acts on what the controller asks for, the rotors clip what the safety filter returns, and the estimate feeds all of them.

## A state that is a group element {#sec17-state}

The first move of the recipe is to name the manifold ([](#sec2-products)). A quadrotor is a position and an attitude, $(x, R) \in \SE(3)$, with the product [](#eq2-se3). The library stores it as it is: three numbers for $x$, nine for $R$, and the velocities $v$ and $\Omega$ with $\dot R = R\,\Omega^\wedge$ as in [](#eq-kinematics). Twelve numbers hold a configuration with six degrees of freedom, and nobody chooses a chart.

The filter needs one more column. Its state is attitude, velocity and position in one matrix $X \in \SE_2(3)$, as in [](#eq14-X). So the mission carries two group elements, the vehicle's and the filter's, and three things that must stay on a group: $R$, the estimate $\hat R$, and the attitude $R_p$ of the state predictor that [](#sec17-uncertainty) adds.

Would a chart have done? The throw answers that. Yaw, pitch and roll lose a direction at 90° of pitch ([](#sec-euler)), and a vehicle released at 151° of tilt is on the far side of any chart centered at hover. The mission starts where charts end.

## Three errors, chosen once {#sec17-errors}

The second move is to take the errors from [](#tbl3-summary). For an attitude they are the error function, the error vector and the velocity error of [](#eq-error-so3),

$$
\Psi_R = \tfrac12\tr\big(I - R_d\T R\big), \qquad e_R = \tfrac12\big(R_d\T R - R\T R_d\big)^\vee, \qquad e_\Omega = \Omega - R\T R_d\,\Omega_d .
$$

The mission uses this one construction three times. Between the vehicle and its reference it is the tracking error. Between the state predictor and the vehicle it is the prediction error [](#eq8-pred-error). Between the estimate and the truth it is the right-invariant error $E_r = \hat X X^{-1}$ of [](#eq14-errors), whose logarithm $\chi_r$ the filter's covariance describes. Nothing was designed for this chapter.

At release the numbers are these. The attitude the cascade asks for is 146.8° from the attitude the vehicle has, so $\Psi_R = 1.84$ out of a possible 2. By [](#eq-error-angle) the error vector has length $\sin 146.8^\circ = 0.55$. That is the trap of [](#warn-sine): the restoring torque $k_R\,\|e_R\|$ is weaker here than it would be at 90°. Keep it in mind. It returns in [](#sec17-limits).

## The model, and a second opinion {#sec17-model}

The equations of motion are [](#eq4-quadrotor),

$$
m\ddot x = f R e_3 - m g\,e_3, \qquad J\dot\Omega + \Omega\times J\Omega = \tau ,
$$

and the simulation takes them from the library, `robots.get("quadrotor")`. Before trusting a mission to them we follow [](#sec5-verify): derive the model a second time by machine from the library's Lagrangian, and compare. The check of the derivation returns $4\times10^{-8}$, well below the $10^{-5}$ that [](#ch-symbolic) requires. At twenty random states and inputs, with the inertia of the loaded vehicle, the accelerations of the two models agree to $7\times10^{-15}$.

The true vehicle differs from the believed one in three ways, and we add all three to the library's model in one place. The thrust acts at the geometric center, which is at $-\rho$ from the center of mass, so it produces the torque $f\,e_3\times\rho$. In hover that is $(0.94,\ 1.57,\ 0)$ N·m, 1.83 N·m in all. The inertia is that of the loaded vehicle. And the gust adds a force in the world frame and a torque in the body frame. With all three set to zero the modified model returns the library's accelerations exactly.

::: {.thought #thought17-scale name="Weigh it, then balance it"}
You strap a package to a quadrotor. A kitchen scale tells you its mass in ten seconds. Finding where its center of mass now sits, to the millimeter, takes a jig and an afternoon. Which of the two errors can the controller least afford?

Do the sum. A mass error of 0.8 kg is a missing force of 7.8 N, straight down, and the position loop of [](#eq6-Fd) would sag by 11 cm under it. An offset of 31 mm is a torque of 1.6 N·m, and the attitude loop would lean 3° into it and drift sideways. Both matter. But the first is cheap to measure and the second is not. So the mission assumes the first is known and leaves the second to [](#sec17-uncertainty).
:::

## A plan in flat outputs {#sec17-plan}

The quadrotor is differentially flat, with the position and the heading as flat output ([](#thm10-quad)). So the reference is a curve, and the curve is the minimum-snap problem [](#eq10-minsnap) through the five waypoints, with times in proportion to the distances between them: one linear system per axis, as in [](#ex10-waypoints). The heading $\psi_d$ turns from 0 to 90° along a smooth fifth-order polynomial.

The flat map [](#eq10-quad-map) then reads the feasibility off the curve ([](#fig17-plan)). The thrust $f_d$ stays between 0.94 and 1.18 times the hover thrust. The tilt reaches 26°, the angular velocity 1.0 rad/s, and the torque $\tau_d$ 0.63 N·m against a limit of 8. The speed peaks at 3.4 m/s. Fed open loop to the believed model, the inputs of the flat map bring the vehicle to the target with an error of $3\times10^{-7}$ m, so the plan is an exact solution of that model, state and input together.

![What the flat map says about the plan. Left: the thrust and the tilt along the course, for the course of 7.5 s that is flown (indigo) and for the same waypoints in 5 s (terracotta). Right: the distance from the reference to the center of the ball. The reference is inside the ball where the curve is terracotta.](figures/ch17-plan.svg){#fig17-plan}

Ask for the same course in 5 s and the map answers at once: a tilt of 51°, 1.56 times the hover thrust, 3.0 N·m. That course is still inside the limits, but the margin that the safety filter and the gust will need has gone. We fly the course of 7.5 s.

One thing the map cannot tell you. The reference passes 0.13 m from the center of the ball and spends 0.42 s inside it. A plan in flat outputs knows the model and nothing else ([](#sec10-limits)).

## The cascade {#sec17-cascade}

The force can only point along $R e_3$, so tracking is the cascade of [](#sec6-cascade). The position loop asks for the force [](#eq6-Fd),

$$
F_d = -k_x\,e_x - k_v\,e_v + m g\,e_3 + m\ddot x_d ,
$$

the construction [](#eq6-Rd) turns $F_d$ and $\psi_d$ into the thrust $f = F_d\cdot R e_3$ and an attitude reference $R_d$, and the attitude law [](#eq6-law) tracks $R_d$:

$$
\tau = -k_R\,e_R - k_\Omega\,e_\Omega + \Omega\times J_0\Omega - J_0\big(\Omega\times R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d\big) .
$$

Two choices remain, and [](#ch-flatness) made both for us. The law needs $\Omega_d$ and $\dot\Omega_d$. [](#thm6-cascade) wants them differentiated along the closed loop. [](#sec10-feedforward) found that the flat map of the reference is a good substitute if the attitude loop is fast enough, and that with Lee's published gains it is not: the loop then diverged. So we keep Lee's position gains, $k_x = 16 \times 4.34 = 69.4$ N/m and $k_v = 24.3$ N·s/m, and scale his attitude gains by $(\kappa^2, \kappa)$ with $\kappa = 2$, as in [](#warn6-feedforward): $k_R = 35.2$ N·m and $k_\Omega = 5.08$ N·m·s. The feedforward $\Omega_d$, $\dot\Omega_d$ comes from the flat map of [](#sec17-plan).

First the cascade alone, with nothing unknown: no package offset, no gust, no ball, and the true state in the loop. Thrown as in the mission, the vehicle cuts its thrust for 0.20 s because $F_d\cdot R e_3$ is negative, turns, and has its thrust axis within 10° of the vertical 0.50 s after the release. It never drops below the height of the release. Then the course ([](#tbl17-cascade)).

::: {.table #tbl17-cascade}

| Feedforward $\Omega_d$, $\dot\Omega_d$ | $\kappa = 2$: RMS, largest | $\kappa = 1$ (Lee's gains): RMS, largest |
| :-- | :-: | :-: |
| Zero | 5.8 cm, 11.0 cm | 90 cm, 153 cm |
| Flat map of the reference | 0.43 cm, 2.4 cm | 74 cm, 131 cm |
| Differentiated along the closed loop | 0.05 cm, 0.28 cm | 0.37 cm, 2.6 cm |

: The cascade alone on the course, with nothing unknown and the true state in the loop: position error.

:::

The table is [](#sec10-feedforward) again, on a different curve. With Lee's gains only the exact feedforward of [](#thm6-cascade) works. With the faster attitude loop the flat feedforward is thirteen times better than none. The closed-loop feedforward is better still here, and [](#sec17-all) will show why we did not rely on it: it differentiates the model, and the model is about to be wrong.

## Would an optimal controller do better? {#sec17-local}

[](#ch-linearization) and [](#ch-mpc) offer a different attitude loop. Linearize about the reference with [](#eq7-so3-lin), solve the Riccati equation [](#eq7-riccati), and feed back as in [](#eq7-feedback). Or solve the box-constrained program [](#eq13-qp) at every sample, with the bound $\tau_{\max}$ inside it ([](#def13-mpc)).

Near the reference the answer is short. At rest [](#eq7-so3-lin) is three double integrators, and the Riccati equation solves in closed form. With the weights $Q = \operatorname{diag}(1240\,I,\ 20\,I)$ and $R_u = I$, the gain is

$$
K = \begin{bmatrix} 35.2\,I & \operatorname{diag}(5.08,\ 5.09,\ 5.45) \end{bmatrix} .
$$

Those are the cascade's $k_R$ and $k_\Omega$, to within the difference between the three inertias. The PD law is an LQR for some weights, and the LQR is a PD law with a slightly stiffer yaw. From an error of 8° the two settle to within 1° in 0.28 s and 0.29 s. With the bounds inactive the predictive controller returns the LQR torque to $4\times10^{-14}$ N·m. Nothing is gained and nothing is lost.

Far from the reference the comparison is more interesting. Take the attitude and the spin at release, a target at rest, and the torque clipped at 8 N·m. The PD law of [](#eq6-law) is within 5° after 0.71 s. The LQR, which feeds back the logarithm error ([](#sec13-error)), takes 0.52 s: its restoring torque grows with the angle where that of $e_R$ fades. The predictive controller also takes 0.52 s. With 8 N·m available, clipping the LQR is already the best the bounds allow. Cut the limit to 2 N·m and prediction pays: 0.61 s against 0.91 s for the clipped LQR and 0.74 s for the PD law.

So the optimal loop is faster from far away, by using an error that [](#sec7-local) calls local: the logarithm jumps at a half turn. And it needs the stage references of [](#def13-mpc), that is $R_d$ over the next half second. In a cascade $R_d$ is built from $F_d$, which contains the position error of the next half second. The book has a linearization of the rigid body ([](#thm7-so3), [](#thm13-so3)) and of a body with a full wrench ([](#thm13-se3)). It does not have one of the quadrotor with its position loop closed. The mission keeps the PD law.

## What the controller does not know {#sec17-uncertainty}

Now the package and the gust. Collect what the controller's model gets wrong in the lumped uncertainty $\Delta$ of [](#eq8-lumped). In hover it is the offset torque, 1.83 N·m. The plain controller balances it with $k_R\,\|e_R\|$, so it leans by $\arcsin(1.83/35.2) = 3.0^\circ$ and the position loop inherits a sideways force ([](#ex8-camera)).

The remedy is the L1 adaptive controller of [](#sec8-l1): a state predictor $(R_p, \Omega_p)$ on the group, the two inputs [](#eq8-l1-law), the adaptation law [](#eq8-adapt) with the projection [](#eq8-proj), and the filter [](#eq8-filter). We keep the settings of [](#ex8-l1), $k_\Delta = 4000$ and $k_f = 20$ rad/s, with $\tilde k_R = k_R$, $\tilde k_\Omega = k_\Omega$, and widen the ball to $\Delta_{\max} = 4$ N·m. The predictor is started at the estimate at the instant of release.

Fly the mission without the ball, with and without adaptation ([](#fig17-adapt)). With the plain controller the attitude error over the course is 3.3° RMS, and the vehicle hovers 4.4 cm from the target. With adaptation it is 0.78° RMS and 0.9 cm. At the final hover the filtered estimate $\Delta_f$ reads $(0.96,\ 1.57,\ -0.01)$ N·m against a true offset torque of $(0.94,\ 1.57,\ 0)$.

![The package and the gust, without the ball. Left: attitude error for the plain controller and with adaptation. Right: two components of the filtered estimate $\Delta_f$, and the offset torque of the hover thrust (dashed). The estimate is found during the recovery, within half a second of the release.](figures/ch17-adapt.svg){#fig17-adapt}

Adaptation did not fix everything, and it could not. During the gust the position error is 3.1 cm RMS with adaptation and 5.0 cm without. A steady force of 3 N against a stiffness of $k_x = 69.4$ N/m is $4.3$ cm, and that is what remains. $\Delta$ is a torque. Nothing in the book estimates a force on the position loop ([](#exr17-force)).

## The obstacle {#sec17-safety}

Without a safety filter the vehicle follows its reference into the ball and passes 0.14 m from the center.

The filter is the one of [](#sec8-quad). The safe set is $\{H \ge 0\}$ for the augmented barrier function [](#eq8-aug),

$$
H(x, R) = \|x - x_o\|^2 - \beta\,r_o^2 - \sigma(p), \qquad p = (x - x_o)\cdot R\,e_3 ,
$$

with the values of [](#ex8-obstacle): $\beta = 1.8$ and $\sigma(p) = 0.25\tanh(-2p)$. The bound $0.25 < (\beta - 1)\,r_o^2 = 0.288$ holds, so $H \ge 0$ keeps the vehicle at least 0.63 m from the center. The filter is the program [](#eq8-safety-filter) for the pair $(f, \tau)$: stay as close as possible to what the cascade proposes, $(f_d, \tau_d)$, subject to the condition [](#eq8-rd2) with $\alpha_1 = 3$ and $\alpha_2 = 6$ s⁻¹ and to $0 \le f \le f_{\max}$. After it, the torque is clipped at $\tau_{\max}$. Write

$$
\delta\tau = \tau - \tau_d \label{eq17-dtau}
$$

for everything that happened to the torque on its way to the rotors. Both terms are known to the controller.

Now connect the blocks as their chapters leave them, and fly ([](#fig17-compose)). This is the flight of the opening. As the vehicle nears the ball the filter overrides the torque. The state predictor knows nothing of that. It is driven by its own torque and keeps following the reference, so predictor and vehicle part, and the prediction error reaches $\tilde\Psi = 0.21$. The adaptation law reads the gap as an uncertainty and pushes the estimate to the edge of its ball. The cascade, steering the vehicle toward a predictor that is somewhere else, asks for more and more torque. When the filter lets go, the vehicle rolls to 148° and falls. It is below the floor 4 m later.

::: {.thought #thought17-copilot name="The co-pilot and the hand on the stick"}
Recall the co-pilot of [](#thought8-copilot), flying a simulator beside the real aircraft and blaming every disagreement on an unknown torque. Now a safety officer reaches over and moves the stick. Nobody tells the co-pilot. What does he conclude?

That a large unknown torque has appeared. He estimates it, and asks the pilot to cancel it. The pilot is now fighting the safety officer, on the co-pilot's advice. The cure is not a better co-pilot. Tell him what the officer did, and let him move his own stick the same way.
:::

**The first wire: tell the predictor.** The override $\delta\tau$ enters the vehicle exactly where $\Delta$ does. In the derivation of [](#eq8-pred-dyn) the estimate $\hat\Delta$ enters the predictor through $\Xi\,\hat\Delta$ and cancels $\Xi\,R_p\T R\,\Delta$. Give the predictor the override in the same way: add $\Xi\,R_p\T R\,\delta\tau$ to $\tau_{\text{fb}}^p$ in [](#eq8-l1-law). Then [](#eq8-pred-dyn) holds again with nothing new on its right-hand side. In a test with a known override, the prediction error stays at rounding level, $4\times10^{-16}$, when the predictor is told. In the mission it stays below $3\times10^{-4}$ where it had reached 0.21.

With that wire the vehicle no longer falls. But the barrier function dips to $H = -0.004$. The reason is in [](#sec8-limits): the barrier trusts the model, and its model of $\dot\Omega$ lacks 1.8 N·m.

**The second wire: inform the barrier.** [](#sec8-limits) suggests feeding $\Delta_f$ to the barrier's model, and warns that no theorem covers it. Do it: use $\dot\Omega = J_0^{-1}(\tau + \Delta_f - \Omega\times J_0\Omega)$ in [](#eq8-rd2). The vehicle now passes 0.86 m from the center, and $H$ never falls below 0.20. The program is feasible at each of the 2400 steps of the flight. The filter changes the input during 1.9 s. The thrust runs from 8.6 N to the limit of 100 N, the torque sits at its limit for 0.125 s, and the tilt reaches 40°.

![Four ways to connect adaptation and the safety filter, on the same flight. Left: the barrier function $H$, which must not go below zero. Right: the prediction error $\tilde\Psi$ on a logarithmic scale. Only with both wires does $H$ stay positive. With the second wire and not the first, the vehicle enters the ball.](figures/ch17-compose.svg){#fig17-compose}

::: {.warning #warn17-order name="One wire without the other is worse than neither"}
Inform the barrier and do not tell the predictor. The estimate $\Delta_f$ then contains the filter's own override, misread as an uncertainty. The barrier's model is now wrong by what the filter itself is doing, and the two chase each other. In our flight the vehicle passes 0.23 m from the center of a ball of radius 0.6 m. That is worse than the blocks as they were, and nearly as bad as no filter. An estimate may be fed to a second block only if everything the second block does is known to the first.
:::

## Knowing where you are {#sec17-estimation}

So far "the state" has meant the estimate. Here is where it comes from. The IMU reads the body angular velocity and the specific force, and its model is [](#eq14-imu),

$$
\dot R = R\,\Omega^\wedge, \qquad \dot v = R\,a_s - g\,e_3, \qquad \dot x = v .
$$

Each reading carries noise of 0.01 rad/s and 0.1 m/s², the values of [](#sec14-worked). Fifty times a second the vehicle sees four landmarks in its body frame, a measurement of the form [](#eq14-obs), each with 0.03 m of noise. The filter is the right-invariant extended Kalman filter of [](#def14-inekf): propagation [](#eq14-propagate), innovation [](#eq14-innovation), correction [](#eq14-update) on the left.

Notice what [](#eq14-imu) does not contain: the mass, the inertia, the thrust, the gust. It is kinematics. The filter has no model of the vehicle, so nothing about the package can mislead it, and it runs while a hand carries the vehicle exactly as it runs in flight. That is why the first six seconds of the mission exist. They are the filter, evaluated alone against the truth ([](#fig17-estimate)).

![The first six seconds. The vehicle (dashed gray) is carried from the pad and thrown. The estimate (indigo) starts level at the origin, 140° and 2 m wrong, and is drawn from the filter's own state at every frame. The four dots are the landmarks.](figures/ch17-estimate-strip.png){#fig17-estimate anim="figures/ch17-estimate.gif"}

The attitude error falls from 140° to 6.5° in half a second, is below 2° for good after 1.0 s, and is 0.05° at release. The position error is below 5 cm for good after 2.6 s. So far this is [](#tbl14-mc), with better sensors.

The covariance is a different story. The NEES, which is 1 for a filter whose covariance tells the truth, is 7600 at 0.5 s, 2800 at 1 s, 180 at 2 s and 6 at 4 s. During the transient from 140° the filter is far too sure of itself. The log-linear property of [](#thm14-loglinear) covers propagation. The correction [](#eq14-update) is still a linearization ([](#sec14-limits)).

**Closing the loop.** From the release on, the controller, the predictor and the safety filter all read $\hat x$, $\hat v$ and $\hat R$, and take $\Omega$ from the latest gyroscope reading. In flight the estimate is good: the attitude error is 0.06° RMS and never above 0.15°, the position error 0.7 cm RMS and never above 2.3 cm, and the NEES averages 1.8. That is mildly overconfident. The filter treats each reading as constant over a step while the body turns.

What changes when the truth is replaced by this estimate? Little. The tracking error away from the ball goes from 1.9 to 2.2 cm RMS, the clearance from 0.87 to 0.86 m, and the hover error from 0.1 to 0.9 cm. The last number is the filter's own position error: a controller cannot hold a point better than it knows it.

We also released early, after 0.9 s, the length of the flip itself. The estimate is then 2.5° and 6 cm off and its NEES is 3400. The mission still succeeds, with a clearance of 0.87 m. A few degrees are within what the cascade tolerates, and the filter finishes converging during the recovery. That is one flight, not a guarantee. No result in this book covers a controller driven by an estimate ([](#sec17-limits)).

## Staying on the manifold {#sec17-integration}

The simulation never leaves the manifold, because its step cannot. The library's `step` runs the stages of a fourth-order Runge-Kutta method on one increment per factor and ends with the exponential update [](#eq4-lie-euler), $R_{k+1} = R_k\exp(h\,\Omega_k^\wedge)$. The filter and the state predictor update their attitudes the same way. Over the 3600 steps of the mission the largest entry of $R\T R - I$ is $1\times10^{-14}$ for the vehicle and $2\times10^{-14}$ for the estimate. Push the same attitude history through the naive update of [](#sec2-integration), $R_{k+1} = R_k + h\,R_k\,\Omega_k^\wedge$, and after twelve seconds of flight $R\T R$ differs from the identity by 0.20.

A point of the manifold is not yet the right point ([](#warn16-energy)). So switch the motors off at the release and let the loaded vehicle tumble for two seconds. Its energy is constant to 3 parts in $10^{13}$, and halving the step divides the drift by 16.0, as a fourth-order method should. For a simulation of hours and not seconds you would change to the variational integrator of [](#thm12-lgvi), whose energy error stays bounded.

## Everything on {#sec17-all}

Here is the mission with every block connected ([](#fig17-flight)).

![The flight, from the release to the target. The reference is dashed gray with a hollow marker and the flown path is solid. The vehicle is drawn from the simulated state. The readouts are the barrier function $H$ and the distance from the reference. Time runs at a uniform rate.](figures/ch17-flight-strip.png){#fig17-flight anim="figures/ch17-flight.gif"}

The vehicle is upright 0.50 s after the release and within 5 cm of the waiting reference after 1.85 s. On the course, away from the ball, it tracks to 2.2 cm RMS and 4.6 cm at worst. It passes the ball at 0.86 m from the center with $H \ge 0.20$. It hovers 0.9 cm from the target. The thrust peaks at the limit of 100 N and the torque at the limit of 8 N·m, both during the detour and both for a fraction of a second.

[](#tbl17-ablation) repeats the mission with one thing changed at a time. "Tracking" is the RMS position error on the course outside the 4.4 s during which the filter is at work or the vehicle is returning from the detour. That window is taken from the first row and used for all.

::: {.table #tbl17-ablation}

| Mission | Tracking | Hover error | Closest to center | Smallest $H$ | Outcome |
| :-- | :-: | :-: | :-: | :-: | :-- |
| Everything on | 2.2 cm | 0.9 cm | 0.86 m | 0.20 | flies |
| Truth instead of estimate | 1.9 cm | 0.1 cm | 0.87 m | 0.22 | flies |
| No feedforward | 6.9 cm | 1.4 cm | 0.85 m | 0.26 | flies |
| Closed-loop feedforward | 2.3 cm | 0.7 cm | 0.78 m | 0.18 | flies |
| No adaptation | 5.1 cm | 4.4 cm | 0.68 m | −0.02 | leaves the safe set |
| No safety filter | 1.4 cm | 0.9 cm | 0.14 m | −0.57 | hits the ball |
| Predictor not told | 2.1 cm | 0.9 cm | 0.23 m | −0.49 | hits the ball |
| Barrier not informed | 2.1 cm | 0.9 cm | 0.69 m | −0.004 | leaves the safe set |
| Neither wire ([](#ch-robust) as it is) | 7.0 cm | 1.1 cm | 0.79 m | −0.10 | falls to the ground |
| Lee's attitude gains, $\kappa = 1$ | 131 cm | 239 cm | 0.89 m | 0.16 | does not track |
| Euler-angle error, no adaptation | 5.0 cm | 4.3 cm | 0.86 m | −0.09 | leaves the safe set |

: The mission with one block changed at a time. The ball has radius 0.6 m, and $H \ge 0$ keeps the vehicle 0.63 m from its center.

:::

Read it as a bill.

*The estimate* cost 0.3 cm of tracking and 0.8 cm at the target. The invariant filter earned its keep in the six seconds before the release, not after.

*The feedforward* of [](#ch-flatness) bought a factor of three in tracking. Without it the stack still flies, because $\kappa = 2$ was chosen so that it would. The closed-loop feedforward of [](#thm6-cascade), which won [](#tbl17-cascade), is no better than the flat one here, and it leaves less room at the ball. It differentiates $F_d$ along a model that no longer matches the vehicle.

*Adaptation* bought a factor of two in tracking and of five at the target. Its larger contribution is in another column. Without it the barrier's model is wrong and $H$ goes negative.

*The safety filter* is the difference between 0.14 m and 0.86 m. It is also the only block that makes tracking worse, which is its job.

*The two wires* are the difference between a flight and a fall. No earlier chapter contains them.

*The gains* matter more than any block. With Lee's attitude gains and the flat feedforward the vehicle wanders more than a meter from the course, as [](#sec10-feedforward) said it would.

*The Euler-angle row* is not what you might expect. We replaced $e_R$ in [](#eq6-law) by the difference of yaw, pitch and roll angles, mapped to the body frame by the Jacobian $E$ of [](#eq-euler-jacobian), which agrees with $e_R$ to first order. Compare it with the row above it without adaptation: the same tracking and the same hover error. It even rights the vehicle sooner, in 0.41 s against 0.50 s, because an angle difference does not fade near a half turn. In this mission the chart cost nothing we could measure, except that during the detour the Euler-angle loop took the vehicle through 163° of tilt where the geometric one stayed below 60°. The case against Euler angles is made elsewhere: at 90° of pitch ([](#sec6-fl), [](#ex13-flip)), and in the two-labs test of [](#thought-two-labs). This course never asks for either.

## Honest limits {#sec17-limits}

**The theorems are about blocks.** [](#thm6-cascade) is about the cascade with exact feedforward, the true state and unlimited inputs. [](#thm8-l1) bounds the prediction error, with no safety filter in the loop. [](#thm8-rd2) keeps $H \ge 0$ for the model, in continuous time, with unlimited torque. [](#thm14-loglinear) is about propagation without noise. Not one of them is a statement about the mission of [](#sec17-all), and this chapter has not added one. What it has is a simulation, twelve variants of it, and a table.

**The two wires are engineering.** The first restores [](#eq8-pred-dyn), which we checked numerically. The second has no theorem at all, as [](#sec8-limits) says. Together they kept $H$ positive in one flight with one package and one gust.

**Almost-global stability promises no rate.** We threw the vehicle twelve ways, with the flip about body axes 30° apart. Eleven times the full stack had it upright within 0.60 to 0.91 s, and the lowest point was 0.83 m above the ground. For the flip about the pitch axis it needed 1.1 s and hit the ground from a release at 2.2 m. That is [](#warn-sine) and [](#thm6-almost) in one number: from near a half turn the restoring torque of $e_R$ is small, the body coasts, and convergence from almost everywhere says nothing about how long it takes. The logarithm error of [](#sec17-local) and the tilt-first error of Mueller (2018) in [](#sec3-errors) are the known remedies.

**The limits are clipped, not planned for.** The safety filter knows $f_{\max}$. It does not know $\tau_{\max}$, and for 0.125 s the torque it asked for was not delivered. [](#sec8-limits) calls a barrier function that accounts for input limits an open problem.

**What is not modeled.** The rotors respond at once. There is no aerodynamic drag, no delay between the estimate and the torque, no bias in the IMU ([](#sec14-legged) says what a bias costs the filter), and no lever arm between the IMU and the center of mass. The landmarks are always visible. [](#warn8-cutoff) and the delay margins of [](#sec8-l1-sim) say which of these to worry about first: with $k_f = 20$ rad/s the adaptive loop of [](#ch-robust) tolerated some tens of milliseconds of delay, and this chapter has not measured that margin for this vehicle.

**One seed.** The noise of the sensors is one draw. Before a claim about reliability, fly the table a hundred times.

## Exercises {#sec17-exercises .unnumbered}

::: {.exercise #exr17-faster}
Shorten the course to 5 s in the notebook. Read the thrust, the tilt and the torque from the flat map first, and predict whether the full stack will fly it. Then fly it. Which limit is reached first, and for how long? Find, to half a second, the shortest course that the stack completes with $H \ge 0$.
:::

::: {.exercise #exr17-center}
Move the ball so that the reference passes exactly through its center, and then 0.5 m to one side. In the first case, which way does the vehicle go around, and what decides it ([](#sec8-limits))? In the second, how much of the detour remains? Report the clearance and the smallest $H$ for both.
:::

::: {.exercise #exr17-force}
The crosswind leaves a position error of about $3\ \text{N}/k_x$. Design an estimate of the unknown force on the position loop: a state predictor for $v$, an adaptation law and a low-pass filter, in the pattern of [](#sec8-l1). Add its output to $F_d$ in [](#eq6-Fd). What happens to $R_d$, and so to $\Omega_d$, when the estimate moves quickly? Find a cutoff that removes the error without shaking the attitude loop.
:::

::: {.exercise #exr17-wires}
Reproduce the four curves of [](#fig17-compose). Then try a third wiring: freeze the adaptation law [](#eq8-adapt) while the safety filter is changing the input, and tell the predictor nothing. Does $H$ stay positive? Compare the estimate $\Delta_f$ after the detour with the one in the text, and explain the difference.
:::

::: {.exercise #exr17-throw}
Throw the vehicle with the flip about its pitch axis, as in [](#sec17-limits). Replace $e_R$ in the attitude law by the logarithm error of [](#sec3-errors), with the same gains. Does the vehicle now recover? Repeat the twelve throws and report the slowest. What does the new law do at exactly half a turn?
:::

::: {.exercise #exr17-sensor}
Read the landmarks ten times a second with 0.1 m of noise, as in [](#sec14-worked). How long must the operator carry the vehicle before the attitude error stays below 2°? Before the NEES stays below 3? Design a rule for the release that uses only quantities the filter can compute, such as the innovation weighted by its covariance $\Sigma_{\tilde y}$, and test it on ten seeds.
:::

::: {.exercise #exr17-seeds}
Fly the first row and the "barrier not informed" row of [](#tbl17-ablation) for fifty seeds of the sensor noise. Report the distribution of the smallest $H$ for each. Is the negative value in the table typical? Then repeat with the package twice as far from the center.
:::

## Notes and sources {#sec17-notes .unnumbered}

The vehicle, its cascade and the default parameters are those of Lee, Leok and McClamroch (2010). Minimum-snap planning in flat outputs follows Mellinger and Kumar (2011). The L1 adaptive attitude controller on $\SO(3)$ is due to Kotaru (2022, ch. 3), who validated it in flight on a quadrotor with an added mass, in the architecture of Hovakimyan and Cao (2010). The augmented barrier function and safety-critical flight past obstacles are from Wu (2017, ch. 7), with barrier functions as in Ames, Xu, Grizzle and Tabuada (2017). The invariant extended Kalman filter is that of Barrau and Bonnabel (2017) in the form of Hartley et al. (2018). Throwing a multicopter to test its recovery, and the weakness of the trace error near a half turn, follow Mueller (2018). The two connections between adaptation and the safety filter, the mission and all of its numbers are ours, and they are simulation results.

- A. D. Ames, X. Xu, J. W. Grizzle and P. Tabuada, ["Control barrier function based quadratic programs for safety critical systems"](https://arxiv.org/abs/1609.06408), *IEEE Transactions on Automatic Control* 62(8), 2017. arXiv:1609.06408.
- A. Barrau and S. Bonnabel, ["The invariant extended Kalman filter as a stable observer"](https://doi.org/10.1109/TAC.2016.2594085), *IEEE Transactions on Automatic Control* 62(4), 2017.
- R. Hartley, M. Ghaffari Jadidi, J. W. Grizzle and R. M. Eustice, ["Contact-aided invariant extended Kalman filtering for legged robot state estimation"](https://doi.org/10.15607/RSS.2018.XIV.050), *Robotics: Science and Systems*, 2018. arXiv:1805.10410.
- N. Hovakimyan and C. Cao, *L1 Adaptive Control Theory: Guaranteed Robustness with Fast Adaptation*, SIAM, 2010.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- D. Mellinger and V. Kumar, ["Minimum snap trajectory generation and control for quadrotors"](https://doi.org/10.1109/ICRA.2011.5980409), *IEEE International Conference on Robotics and Automation*, 2011.
- M. W. Mueller, ["Multicopter attitude control for recovery from large disturbances"](https://arxiv.org/abs/1802.09143), arXiv:1802.09143, 2018.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
