---
title: "Case study: carrying a payload together"
short: Carrying together
id: ch-case-transport
summary: Three quadrotors carry a rigid payload on cables through a gap, and two carry a flexible cable. The mission is taken through the whole recipe, the tensions are chosen and not found, and each block is removed in turn to see what it bought.
notebook: notebooks/ch19_case_transport.ipynb
verify: verify/ch19.py
---

Three people carry a table through a doorway. You film them and measure the table's motion to the millimeter. How hard is each of them pulling?

You cannot tell. Two of them may be leaning away from each other, each pulling with twice the force the table needs, and the table moves exactly as it would if they were not. The motion fixes six numbers, a force and a torque. Three hands supply nine. The other three are a choice, and somebody has to make it.

Replace the people by quadrotors and the arms by cables, and the choice has consequences. Choose the smallest tensions and each vehicle hangs straight above its attachment, 0.69 m from its neighbors. Choose larger ones and the cables spread, the vehicles separate, and every cable pulls harder. A gap narrower than the team forces the choice to change in flight ([](#fig19-mission)).

![The mission. Three quadrotors carry a rigid payload on cables. They cruise with the cables spread, draw together to pass a gap narrower than the cruise formation, and set the payload down level at the target. The dashed curve is the path planned for the payload.](figures/ch19-mission.svg){#fig19-mission}

This chapter flies that mission with every block of [](#fig16-pipeline) connected, and then removes the blocks one at a time. The rows for several quadrotors and for the flexible cable in [](#tbl16-map) and [](#tbl16-plan) are mostly empty. Here they are filled, except for estimation.

## The mission {#sec19-mission}

The team is the library's `multi_quadrotor_payload` with three vehicles and its default parameters ([](#sec16-teams)). The payload has mass $m_L = 0.8$ kg and inertia $J_L = \operatorname{diag}(0.03,\ 0.05,\ 0.08)$ kg·m². Each vehicle has mass $m_i = 1.5$ kg, and each cable has length $l_i = 1$ m. Cable $i$ is attached at the vehicle's center of mass and at the point $\rho_i$ of the payload, a body-frame vector from the payload's center of mass.^[The library calls these points `r[i]`. In the text $r$ with a subscript is a radius, so we write $\rho_i$, the letter [](#eq4-pendulum) uses for a body-frame vector from a reference point of the body.] The three points are evenly spaced on a circle of radius 0.4 m.

We give each vehicle a thrust of at most $f_{\max} = 22$ N. That limit is our assumption. One vehicle would need $(1.5 + 0.8)\,g = 22.6$ N to hover with the payload, so one cannot carry it. Three need 17.3 N each.

Two vehicles must stay $r_{\min} = 0.8$ m apart, center to center, the value of [](#sec8-quad). Every cable must keep a tension of at least $T_{\min} = 0.5$ N. The time step is $h = 5$ ms for the simulation and the controller alike.

- **0 to 1 s.** Hover at $(0, 0, 1.6)$ m with the cables spread 20.5° from the vertical. The vehicles are 1.30 m apart.
- **1 to 9 s.** The payload travels 6 m along the first axis, at up to 2.2 m/s. From 1.5 to 4.5 s it turns by 60° about the vertical.
- **5 s.** The payload passes a gap 1.45 m wide. A rotor reaches 0.25 m from the center of its vehicle, so the cruise formation, 1.80 m across, does not fit. The planner was told about the walls and asked to keep about 10 cm from them. It draws the team together until the vehicles are 0.76 m apart. Nobody told it about $r_{\min}$.
- **7.5 to 11.5 s.** The payload descends 1 m and stops at the target $(6, 0, 0.6)$ m, level.

What is new, compared with one vehicle and one load ([](#ch-case-load))? Three things. **Redundancy**: the tensions are not determined by the motion. **Internal constraints**: the vehicles can hit each other, and each cable can go slack by itself. **Scale**: every vehicle adds five dimensions, and something in each block grows with it.

## The manifold {#sec19-manifold}

Recall the first move of the recipe: list the parts ([](#sec16-recipe), [](#sec2-products)). The payload is a rigid body, a position $x_L$ and an attitude $R_L$, in $\SE(3)$. Each cable is a direction $q_i \in S^2$, pointing from vehicle $i$ to its attachment. Each vehicle is an attitude $R_i \in \SO(3)$. The vehicle's position is not a coordinate. It follows from the rest,

$$
x_i = x_L + R_L\,\rho_i - l_i\,q_i . \label{eq19-kin}
$$

The configuration manifold is $\SE(3)\times(S^2\times\SO(3))^n$, listed in [](#tbl2-robots), with dimension $6 + 5n$: 21 for three vehicles. A configuration is stored as 48 numbers, because rotation matrices and unit vectors are stored whole. The inputs are a thrust and a torque per vehicle, $4n = 12$. Nine degrees of freedom have no input of their own.

Each factor brings its error and its variation with it ([](#tbl3-summary)): $e_R$ and $\eta$ for the four attitudes, $e_q$ and $\xi_i$ for the three cables, a plain difference for $x_L$. Nothing is designed again.

## The model, and a second opinion {#sec19-model}

[](#sec4-products) derived the team with a point-mass load, [](#eq4-multi), and stated the rigid payload in words. The library writes the rigid payload as the Lagrange-d'Alembert principle delivers it ([](#def4-dalembert)), one equation per variation, and solves the result as one linear system in the accelerations. The free-body form is shorter and is what the rest of the chapter uses. With $T_i$ the tension of cable $i$, positive when it pulls as in [](#eq4-load-tension),

$$
\begin{aligned}
m_L\big(\ddot x_L + g\,e_3\big) &= -\sum_i T_i\,q_i , \\
J_L\dot\Omega_L + \Omega_L\times J_L\Omega_L &= -\sum_i \rho_i\times R_L\T\,T_i\,q_i , \\
m_i\big(\ddot x_i + g\,e_3\big) &= f_i R_i e_3 + T_i\,q_i , \qquad J_i\dot\Omega_i + \Omega_i\times J_i\Omega_i = \tau_i .
\end{aligned} \label{eq19-free}
$$

The first two lines are a rigid body pulled by $n$ strings. The third is [](#eq4-load) seen from the vehicle, once per vehicle. At 20 random states the library's accelerations and the tensions computed from them satisfy all three lines to rounding error.

The second opinion is [](#sec5-verify). `geomech` derives the equations from the Lagrangian and the virtual work of [](#sec5-scaling), and `check_eom` returns $8\times10^{-8}$ for two vehicles, $9\times10^{-8}$ for three and $8\times10^{-8}$ for four, all below the threshold of $10^{-5}$.

::: {.warning #warn19-taut name="Three cables, three ways to go slack"}
[](#warn4-slack) applies to every cable separately. [](#eq19-free) stays smooth when one $T_i$ turns negative and then simulates a cable that pushes. A team has more ways to get there than a single vehicle: a payload that tilts unloads the cable on its high side, and a vehicle that falls behind unloads its own. Monitor every tension, and treat a mission in which any of them reaches zero as outside the model.
:::

## Who pulls how hard {#sec19-tensions}

Stack the first two lines of [](#eq19-free). The force of cable $i$ on the payload is $-T_i q_i$, and

$$
-\,G_L\begin{bmatrix} T_1 q_1 \\ \vdots \\ T_n q_n \end{bmatrix} = \begin{bmatrix} F_L \\ \tau_L \end{bmatrix},
\qquad
G_L = \begin{bmatrix} I & \cdots & I \\ \rho_1^\wedge R_L\T & \cdots & \rho_n^\wedge R_L\T \end{bmatrix}, \label{eq19-grasp}
$$

where $F_L = m_L(\ddot x_L + g\,e_3)$ is the force the payload's motion asks for and $\tau_L = J_L\dot\Omega_L + \Omega_L\times J_L\Omega_L$ the torque, in the payload frame. The pair $(F_L, \tau_L)$ is the **wrench**. The $6\times 3n$ matrix $G_L$ is the **grasp map**, the name Kotaru (2022, sec. 6.4) uses. It takes the stack of tension vectors $T_i q_i$ of [](#eq10-chain) to the wrench.

This is the opening puzzle as a linear system: six equations, $3n$ unknowns. For three attachments that are not on a line $G_L$ has rank 6, and its null space has dimension $3n - 6 = 3$. A stack of forces in the null space is an **internal force**: the cables pull against each other through the payload, and the payload feels nothing. The general solution is one particular solution plus an internal force.

::: {.thought #thought19-stick name="Two hands, one stick"}
Hold a broomstick level by two strings, one at each end. Move your hands however you like. Can you roll the stick about its own axis?

You cannot. Both strings pull at points of the axis, and a force through an axis has no moment about it. For two attachments $G_L$ has rank 5, not 6. A payload carried by two vehicles can turn freely about the line through its attachments, and no controller that acts through the cables can stop it. The count of inputs gives no warning, $4n = 8 > 6$. The rank does.
:::

The natural particular solution is the one of smallest norm, the pseudoinverse of $G_L$ applied to the wrench. Kotaru (2022, eq. 6.20) notes that it can be computed beforehand and run on hardware. In hover it gives every cable a third of the weight, $T_i = 2.62$ N, straight up. Then every vehicle sits above its attachment, and the vehicles are as far apart as the attachments: 0.69 m ([](#fig19-tensions)). The smallest tensions are 11 cm inside $r_{\min}$.

![Three answers to the same wrench, the weight of the payload in hover. Indigo arrows: the force of each cable on the payload. Teal arrows: its internal part, which sums to zero force and zero torque. Left: minimum norm. Every cable is vertical and the vehicles are closer than $r_{\min}$. Middle: the smallest internal force that keeps every pair 0.83 m apart. Right: the cruise formation of the mission.](figures/ch19-tensions.svg){#fig19-tensions}

So we choose the internal force. Let $\bar T$ be the stack of tension vectors that the plan asks for at this instant. Given the wrench, the tension distribution is a small quadratic program in the manner of [](#eq8-qp). Its equality constraint is the wrench balance [](#eq19-grasp):

$$
\begin{aligned}
\min\ & \Big\|\begin{bmatrix} T_1 q_1 \\ \vdots \\ T_n q_n \end{bmatrix} - \bar T\Big\|^2 \quad\text{subject to the wrench balance and} \\
& -e_3\cdot R_L\T\,T_i q_i \ \ge\ T_{\min} \quad\text{for every cable,} \\
& \|\rho_i - \rho_j\| + \frac{\rho_i - \rho_j}{\|\rho_i - \rho_j\|}\cdot R_L\T\big(l_j\,q_j - l_i\,q_i\big) \ \ge\ r_{\min} + 3\ \text{cm} \quad\text{for every pair.}
\end{aligned} \label{eq19-qp}
$$

The first constraint keeps every cable pulling upward in the payload frame. The left side of the second is, by [](#eq19-kin), the distance between vehicles $i$ and $j$ measured along the line that joins their attachments, which is never more than their distance. With $q_i$ replaced by $T_i q_i$ over the tension of the unconstrained solution it is linear in the unknowns. Eliminate the equality by writing the unknown as a particular solution plus an internal force, and three unknowns and six inequalities remain. That is small enough for the solver that checks [](#ex8-cone), which tries active sets in turn.

In hover the program adds an internal force of 0.21 N per cable. The cables lean 4.5° outward, the tensions rise from 2.616 to 2.624 N, and the vehicles are 0.83 m apart. The price of 14 cm of separation is 8 mN of tension. The cruise formation spends more, 0.98 N of internal force per cable for 1.30 m, because distance from a neighbor's downwash is worth something that this model does not price.

One finding belongs here. That solver returns the first active set that passes its test. When the active constraints are dependent, its linear system is singular, and it can return a point of size $10^{14}$ that passes. With three unknowns and six constraints this happens: on 2 of 200 random wrenches near hover. We skip dependent active sets, and check that the guarded solver agrees with the original on 200 well-posed problems.

## A plan in flat outputs {#sec19-plan}

[](#thm10-load) recovers one vehicle and its inputs from the curve of its load. Apply the same three rungs to each cable, with [](#eq19-grasp) in front. Choose the payload's pose $(x_L(t), R_L(t))$. Its derivatives give the wrench. Choose an internal force $T_{\text{int}}(t)$, and the tension vectors follow from the pseudoinverse plus the internal force. Then, cable by cable, [](#eq10-load-map) and [](#eq10-load-force) read

$$
q_i = \frac{T_i q_i}{\|T_i q_i\|}, \qquad x_i = x_L + R_L\,\rho_i - l_i\,q_i, \qquad f_i R_i e_3 = m_i\big(\ddot x_i + g\,e_3\big) - T_i q_i , \label{eq19-flat}
$$

and [](#eq10-quad-map) turns each thrust vector and a heading into an attitude, an angular velocity and a torque.

Count as [](#sec10-chain) does. The pose is six functions, the headings are $n$, the internal force is $3n - 6$. Together $4n$, the number of inputs. Wu (2017) generates the reference for seven vehicles this way, from the pose, the yaw angles and the tension vectors of four cables ([](#sec10-chain)). Be precise about what we claim. We do not prove that this is a flat output in the sense of [](#def10-flat). We checked the one direction a planner needs: for a pose with six derivatives and an internal force with four, the state and inputs from [](#eq19-flat) satisfy the library's model to $2\times10^{-14}$ along the whole mission, tensions included. Flown open loop for 2 s, the payload ends $1.4\times10^{-10}$ m from its curve, and 16 times closer with half the step. For two vehicles the claim fails at the first step: the wrench about the line through the attachments cannot be produced, so the pose is not free to choose.

The mission's plan is built from smooth steps, polynomials of degree 13 with six vanishing derivatives at both ends. The internal force is the spreading one: $T_{\text{int}}$ newtons per cable along $R_L\,\rho_i/\|\rho_i\|$, with $T_{\text{int}} = 0.98$ N in cruise and 0.10 N in the gap.

Now read the plan, as [](#ex10-waypoints) taught. The thrusts stay between 16.0 and 18.8 N, inside the limit. The tensions never fall below 2.59 N. The payload accelerates at up to 1.2 m/s². All feasible. And the vehicles come within 0.76 m of each other, 4 cm inside $r_{\min}$, and stay inside it for 2.6 s. The flat map reports what the plan implies. It does not object.

## The cascade {#sec19-cascade}

[](#eq6-load) climbs a ladder: load, cable, attitude. Here the ladder has one more rung at the bottom, and three copies of the rest.

**The payload.** The position law is [](#eq6-Fd) for the payload alone, and the attitude law is [](#eq6-law) for the payload:

$$
\begin{aligned}
F_L &= -k_x\,e_x - k_v\,e_v + m_L g\,e_3 + m_L\ddot x_d , \\
\tau_L &= -k_R\,e_R - k_\Omega\,e_\Omega + \Omega_L\times J_L\Omega_L - J_L\big(\Omega_L\times R_L\T R_d\,\Omega_d - R_L\T R_d\,\dot\Omega_d\big) ,
\end{aligned}
$$

with the errors of the payload against its reference. This is the pair Kotaru (2022, eq. 6.16) uses. We take $k_x = 3.2$ N/m and $k_v = 2.88$ N·s/m, a loop at 2 rad/s, and replace $k_R$, $k_\Omega$ by the matrices $4\,J_L$ and $3.6\,J_L$ so that all three axes of the payload answer at 2 rad/s as well.

**The tensions.** [](#eq19-qp) turns the wrench $(F_L, \tau_L)$ into tension vectors, close to the plan's.

**The cables.** Each tension vector has a direction $q_{i,d}$, and [](#eq6-law-s2) asks cable $i$ for the angular acceleration

$$
\dot\omega_i = (I - q_i q_i\T)\,\dot\omega_{i,d} - (q_i\cdot\omega_{i,d})\,\dot q_i - k_q\,e_q - k_\omega\,e_\omega ,
$$

with $k_q = 49$ s⁻² and $k_\omega = 11.2$ s⁻¹: 7 rad/s. The thrust vector that produces it, and delivers the tension along the cable, is the second line of [](#eq6-load) with one term more:

$$
f_i R_i e_3 \;\leftarrow\; m_i\Big(\underbrace{\ddot x_L + R_L\big(\dot\Omega_L\times\rho_i + \Omega_L\times(\Omega_L\times\rho_i)\big)}_{\text{acceleration of attachment } i} + g\,e_3 + l_i\|\dot q_i\|^2 q_i - l_i\,\dot\omega_i\times q_i\Big) - (q_i q_i\T)\,T_i q_{i,d} . \label{eq19-thrust}
$$

Substitute into the third line of [](#eq19-free) and the cable turns as asked. The new term is the acceleration of the attachment, which for a point-mass load lies along the cable and drops out. We use the acceleration that the wrench commands.

**The vehicles.** [](#eq6-Rd) turns each thrust vector into $f_i$ and an attitude reference, tracked by [](#eq6-law) with $k_R = 10$ N·m and $k_\Omega = 1$ N·m·s. The attitude loops of a vehicle then have natural frequencies between 16 and 22 rad/s.

The feedforward, $\omega_{i,d}$, $\dot\omega_{i,d}$, $\Omega_d$ and $\dot\Omega_d$ for every body, comes from the flat map of [](#sec19-plan), as [](#sec10-feedforward) recommends. On the reference itself the cascade returns the flat inputs to $4\times10^{-14}$.

Four loops are stacked: 2, 2, 7 and 16 to 22 rad/s. [](#warn6-feedforward) and [](#sec10-feedforward) found that such a stack works only if each loop is clearly faster than the one outside it. [](#tbl19-order) tests that on a plan that respects $r_{\min}$, with no protection in the loop.

::: {.table #tbl19-order}

| Change | Payload error, RMS | Outcome |
| :-- | :-: | :-- |
| None | 0.2 mm | tracks |
| Payload attitude loop at 4 rad/s | 2.8 mm | thrust saturates near the end |
| Payload attitude loop at 6 rad/s | 2.7 m | lost |
| The same at 4 rad/s, vehicle gains scaled by $\kappa = 1.5$ | 0.2 mm | tracks |
| Cable loop at 10 rad/s | 0.2 mm | tracks |
| Cable loop at 14 rad/s | 4.9 m | lost |
| The same at 14 rad/s, $\kappa = 2$ | 0.2 mm | tracks |
| No feedforward at all | 14 cm | lags |
| No feedforward for the vehicle attitudes only | 4.2 mm | tracks |

: The cascade alone, on a plan that keeps the vehicles 0.90 m apart. One thing changed per row.

:::

A cable loop at 10 rad/s works under these vehicles and one at 14 rad/s does not. The payload's attitude loop needs more room: at 4 rad/s under a cable loop of 7 it is already marginal. Each failure is repaired by the $\kappa$ of [](#warn6-feedforward), which speeds up the innermost loop. The feedforward is worth a factor of 700 here, more than it was for one vehicle, because the plan moves the cables on purpose. Wu (2017, Prop. 5.3) proves exponential tracking for a controller of this shape when the vehicles' attitude loops are fast enough, for a team large enough that the tensions along fixed cable directions span every wrench, which takes at least six cables. Neither his proposition nor [](#thm6-cascade) covers three vehicles with flat feedforward. The table is what we have.

## Would an optimal controller do better? {#sec19-lqr}

[](#ch-linearization) offers another route: linearize about the reference and let a Riccati equation choose all the gains at once. Wu (2017, ch. 4) and Kotaru (2022, ch. 7) both take it.

The linear state collects one variation per factor and its rate: $\delta x_L$, $\eta_L$, the $\xi_i$ and the $\eta_i$, then $\delta v_L$, $\delta\Omega_L$, the $\delta\omega_i$ and the $\delta\Omega_i$. Forty-eight numbers for three vehicles, with two constraints of the form [](#eq7-s2-constraint) for each cable, so the constraint subspace has dimension 42, twice the 21 degrees of freedom. The kinematic rows of $A$ are [](#eq7-xi-dot) and [](#eq7-var-so3), copied once per factor. The dynamic rows we do not write by hand. We take centered differences of the library's accelerations along each basis vector of the constraint subspace, moving the state by the variation itself: $R\exp(\epsilon\,\eta^\wedge)$ and $\exp(\epsilon\,\xi^\wedge)\,q$. No chart is involved, and the same thirty lines linearize every robot of the library.

`geomech` linearizes the same model symbolically. At states at rest the two agree to $10^{-8}$ on the constraint subspace, for two, three and four vehicles. At states in motion the rows of the cable directions differ, by entries of order one. The difference has an exact form. In the tangent part of the row of cable $i$ it is one term, $c_i\,\xi_i\times q_i$, with a single number $c_i$ per cable that vanishes at rest. With those three numbers fitted, the two linearizations agree to $2\times10^{-8}$. That is the term a projection would add if the symbolic coefficient $E_{\xi_i}$ of [](#eq5-pendulum) had a component $c_i$ along $q_i$: the equation of motion is only the tangent part of that coefficient ([](#secA-conventions)), and the variation of a tangent part is not the tangent part of the variation. We did not confirm that reading inside `geomech`. The linearization by differences is the one we tested against perturbed simulations ([](#sec19-cable)), and it is the one used along moving references.

**Controllability.** Judge it on the constraint subspace ([](#def7-constrained)). In the cruise hover the reduced pair has rank 42 of 42 for three vehicles and 52 of 52 for four. For two vehicles it has rank 30 of 32. The two missing directions are the roll of the payload about the line through its attachments, and its rate: the broomstick of [](#thought19-stick), found by a rank test.

**The gain.** Solve [](#eq7-riccati) on the reduced pair with weights of 60, 20 and 10 on positions, directions and attitudes, of 15, 2 and 1 on their rates, and $R_u = \operatorname{diag}(0.05\,I,\ 20\,I_9)$, thrusts first. The gain is a $12\times48$ matrix with closed-loop poles between $-0.94$ and $-10.8$ s⁻¹. The feedback is [](#eq7-feedback) for every factor at once: the gain is fed $e_x$, $e_q$, $e_R$ and the velocity differences.

::: {.table #tbl19-lqr}

| Start | LQR: settled, peak thrust, lowest tension | Cascade: settled, peak thrust, lowest tension |
| :-- | :-: | :-: |
| Payload 0.3 m aside | 2.0 s, 19.1 N, 2.65 N | 2.1 s, 20.7 N, 2.52 N |
| Payload 1 m aside | 2.6 s, 22 N, 2.28 N | 3.2 s, 22 N, 1.95 N |
| Payload 3 m aside | 3.2 s, 22 N, −1.4 N | 4.7 s, 22 N, −1.6 N |
| Payload turned 90° | 0.6 s, 19.0 N, 2.35 N | 4.0 s, 22 N, 1.58 N |
| One cable swung 25° | 2.0 s, 19.4 N, 2.63 N | 2.2 s, 22 N, 2.26 N |

: Return to the cruise hover. "Settled": the payload stays within 2 cm of its place.

:::

The LQR is as fast or faster in every row, with no more thrust and more tension to spare. It coordinates the vehicles through its off-diagonal gains, where the cascade waits for one loop to finish before the next can act. A payload turned by 90° shows it best: with the LQR the payload is back within 2 cm of its place after 0.6 s, with the cascade after 4.0 s. From 3 m away both controllers drive a tension negative, so both leave the model.

Why does the mission keep the cascade? The LQR has no place for $r_{\min}$ or $T_{\min}$, and its gain belongs to one reference: a time-varying gain of 576 entries along the mission, recomputed when the plan changes. The cascade exposes the tension vectors as an intermediate quantity, and [](#eq19-qp) constrains exactly those. The predictive controller of [](#def13-mpc) would give the linearization its constraints back. We did not build it for 42 states.

## Keeping apart {#sec19-safety}

The plan brings the vehicles within 0.76 m. Two blocks can object.

**In the tensions.** The pair constraint of [](#eq19-qp) moves the tension vectors inside the null space of $G_L$. The wrench is untouched.

**After the cascade.** The safety filter of [](#sec8-quad) takes the thrust vectors of [](#eq19-thrust) and changes them as little as possible, as in [](#eq8-safety-filter), so that every pair satisfies [](#eq8-rd2),

$$
\ddot H_{ij} + (\alpha_1 + \alpha_2)\,\dot H_{ij} + \alpha_1\alpha_2\,H_{ij} \ \ge\ 0, \qquad H_{ij} = \|x_i - x_j\|^2 - r_{\min}^2 ,
$$

with $\alpha_1 = \alpha_2 = 4$ s⁻¹, and so that every tension stays above $T_{\min}$. Kotaru (2022, sec. 6.5) filters commanded accelerations and treats each vehicle as a free point mass. Here the vehicles are tied to a payload, so we use the model: by [](#eq19-free) the accelerations $\ddot x_i$ and the tensions $T_i$ are affine in the stacked thrust vectors, and both maps come from one solve of the library's linear system. The model's $\ddot H_{ij}$ agrees with a second difference of $H_{ij}$ along the library's flow to $10^{-4}$. Three pairs and three tensions make six constraints on nine unknowns.

::: {.thought #thought19-doorway name="Elbows in"}
Carry the table through the doorway again. You pull your elbows in to fit. There are two ways to avoid your partner's elbow. You can agree beforehand to shift your grips, so that both of you stand a little farther apart while the table goes where it was going. Or you can wait until the elbows are about to touch and push off.

The first changes how the load is shared and the table does not notice. The second is a force nobody planned, and the table lurches. But the first is an agreement about where to stand, and standing somewhere takes time. If you arrive at the doorway too fast, only the second is quick enough.
:::

::: {.table #tbl19-protect}

| Protection | Closest pair | Smallest $H_{ij}$ | Payload error, RMS | Largest payload tilt | Filter active |
| :-- | :-: | :-: | :-: | :-: | :-: |
| None | 0.758 m | −0.066 | 0.02 cm | 0.01° | none |
| Pair constraints in [](#eq19-qp) only | 0.785 m | −0.024 | 0.33 cm | 0.6° | none |
| Safety filter only | 0.800 m | 0.000 | 1.6 cm | 5.8° | 3.1 s |
| Both | 0.819 m | 0.030 | 1.6 cm | 3.2° | 1.1 s |

: The mission with four ways to keep the vehicles apart. $H_{ij} \ge 0$ means at least 0.8 m.

:::

[](#tbl19-protect) and [](#fig19-protect) are the thought experiment in numbers. Without protection the team follows its reference inside $r_{\min}$. The pair constraints hold the vehicles at 0.83 m once they act, and the payload barely notices. But the constraint is about cable directions, and a cable has inertia: when the constraint first becomes active the cables are closing at speed, and they overshoot to 0.785 m. The safety filter sees velocities through $\dot H_{ij}$ and holds 0.800 m exactly. It does so by changing thrust vectors that were computed to deliver a wrench, and the payload tilts by 5.8°. Together, the constraint does the standing work, the filter catches the overshoot, and the filter is active for a third of the time.

![Four ways to keep three vehicles apart on the same mission. Left: the distance of the closest pair. The pair constraints alone overshoot when they first act. Right: the tilt of the payload, which the plan keeps at zero. The safety filter buys its guarantee with the payload's attitude.](figures/ch19-protect.svg){#fig19-protect}

**Does the finding of [](#sec17-safety) bite here?** [](#warn17-order) and [](#warn18-composition) are about a block that estimates and a block that overrides. This stack has no estimator, so nothing winds up, and we found no wire missing of that kind. What does carry over is the sentence under it in [](#sec8-limits): the barrier trusts the model. Push one vehicle toward its neighbor with 8 N for half a second while the team is in the gap, as Kotaru does by hand in his experiment. The filter is active and the vehicles close to 0.70 m. The push is not in the model, so the model's $\ddot H_{ij}$ contains the thrust with which the cable loop is already resisting and not the force it resists. To the filter the situation looks safe. Tell the filter the push, and the closest approach is 0.75 m: better, and still inside. The rest is the question [](#sec8-quad) ended on. The filter asks for a thrust vector, and a vehicle must first turn to deliver it, with less than 5 N of thrust to spare.

## A cable that sags {#sec19-cable}

Now the same team carries a hose. Kotaru (2022, ch. 7) models it as a chain of links with a point mass at each joint and quadrotors at some of the joints, the library's `quadrotor_cable` ([](#sec16-teams)). We take five links of 0.2 m with 50 g at each joint after the first and a vehicle at each end: $\R^3\times(S^2)^5\times\SO(3)^2$, 19 degrees of freedom, 8 inputs. `check_eom` returns $4\times10^{-7}$.

**Plan.** [](#thm10-chain) names the flat output: the position of joint 5, the two headings, and the tension vector of the link that leaves the vehicle at joint 5. We prescribe a slow figure-eight for that joint and a constant tension vector, half the cable's weight upward and 0.6 N outward. The recursion [](#eq10-chain) returns the rest. In the hanging equilibrium the vehicles are 0.72 m apart, the cable sags 0.30 m, and the link tensions are 1.15, 0.77, 0.60, 0.77 and 1.15 N. Along the moving reference the state and the inputs satisfy the library's model to $8\times10^{-13}$, using fourteen derivatives of the curve.

**Linearize.** The thesis derives $A$ and $B$ in closed form (its Appendix B.3) and observes that the constrained pair may not pass a rank test as it stands. We use the differences of [](#sec19-lqr) and test on the constraint subspace. The linear state has 48 entries and the subspace dimension 38. At the hanging equilibrium the reduced pair has rank 38 of 38: controllable. Its open loop has swing modes at 1.0, 1.3 and 1.7 rad/s and no damping. Along the moving reference the linear model predicts a perturbed simulation over 0.2 s with a relative error of $10^{-5}$ for a perturbation of $10^{-3}$, and of $10^{-7}$ for $10^{-4}$.

**Track.** At the equilibrium, [](#eq7-riccati) gives an $8\times48$ gain with poles between $-0.68$ and $-18$ s⁻¹. Start with every link 12° off and one end displaced by 8 cm: a largest joint error of 12 cm. With the gain it is below 1 cm after 1.4 s and at 0.08 mm after 6 s. With the equilibrium inputs alone the cable swings without end, and is still 8 cm off after 6 s. Along the figure-eight we integrate [](#eq7-riccati) backward over 8 s, as Kotaru does, with the linearization sampled every 20 ms. From the same kind of start the largest joint error is 0.4 mm RMS after the first three seconds ([](#fig19-cable)). Open loop it reaches 20 cm.

![Two quadrotors carry a cable of five links along a reference generated from the curve of one joint by [](#thm10-chain). Dashed gray: the reference cable and the curve. Indigo: the simulated cable, started with every link 12° off, under the finite-horizon LQR about the reference.](figures/ch19-cable-strip.png){#fig19-cable anim="figures/ch19-cable.gif"}

That is what the thesis reports too, at the same strength: simulations, for up to ten links and three vehicles, with a finite-horizon LQR about a flat reference. Its experiments (sec. 7.6) are of a different kind. Two quadrotors share one power tether, pass a window and reach over an obstacle to grasp, guided by waypoints. They show that the configuration is useful. They do not close a loop around the cable's shape.

## What a vehicle costs {#sec19-scaling}

Fly the mission with two, three and four vehicles on the same payload, each on a plan that respects $r_{\min}$ ([](#tbl19-scaling)).

::: {.table #tbl19-scaling}

| | $n = 2$ | $n = 3$ | $n = 4$ |
| :-- | :-: | :-: | :-: |
| Degrees of freedom, inputs | 16, 8 | 21, 12 | 26, 16 |
| Size of the derived equations, characters | 4261 | 6299 | 8337 |
| Rank of $G_L$, internal forces | 5, 1 | 6, 3 | 6, 6 |
| Controllable on the constraint subspace | 30 of 32 | 42 of 42 | 52 of 52 |
| Tension program: unknowns, constraints | 1, 3 | 3, 6 | 6, 10 |
| Safety filter: unknowns, constraints | 6, 3 | 9, 6 | 12, 10 |
| Payload error, RMS | 1.8 mm | 1.6 mm | 0.2 mm |
| Largest payload tilt | 1.9° | 0.25° | 0.01° |

: The same mission with two, three and four vehicles.

:::

The model costs almost nothing: the equations grow by 2038 characters per vehicle, the same piece each time, the symbolic derivation and its check take seconds, and the hand-written model is one loop longer. The constraints are what grow fast, three per vehicle now and one more per pair, $n + n(n-1)/2$. Enumerating active sets is fine for six constraints and already slow for ten. A real solver ([](#sec13-ipm)) is the next purchase.

The row to read twice is the first column. Two vehicles track the position as well as three. But the payload is 1.9° from level when the flight ends, its largest tilt of the flight, about the line the controller cannot reach. Three is the first team that controls a pose. The last column has a catch of its own: a plan that keeps four vehicles 0.97 m apart is 12 cm too wide for the gap. The table compares tracking, not passage.

Four vehicles on the same circle of attachments sit closer than three, 0.57 m when the cables are vertical. Fly them on the mission's reference, with its internal force tuned for three, and they are asked to come within 0.64 m of each other. Both protections act, the closest pair is at 0.808 m, a tension falls to 0.80 N, the thrust saturates, and the payload is 4 cm off on average. The team stays together, and its rotors overlap the walls by 5 cm. An internal force tuned for three is not tuned for four.

## Staying on the manifold {#sec19-integration}

The library's `step` advances every factor by its own exponential, [](#eq4-lie-euler), inside a fourth-order Runge-Kutta scheme. Over the 2400 steps of the mission the largest violation of $\|q_i\| = 1$, $q_i\cdot\omega_i = 0$ and $R\T R = I$, over three cables and four attitudes, is $1.2\times10^{-14}$. For the cable of [](#sec19-cable) it is $8\times10^{-15}$ over five links and two attitudes. With the motors off the team's energy drifts by 8 parts in $10^{12}$ in a second, and 15 times less with half the step ([](#warn16-energy)). The tensions of all three cables are computed at every step from the true parameters, and are a column of the next table.

## Everything on {#sec19-all}

Here is the mission with every block connected ([](#fig19-flight)).

![The mission. Dashed gray with hollow markers: the reference of the payload and of each vehicle. The bars are the three tensions, and the short line across them is $T_{\min}$. Watch the vehicles stop short of their hollow markers in the gap: the reference asks for 0.76 m and the team holds 0.82 m.](figures/ch19-flight-strip.png){#fig19-flight anim="figures/ch19-flight.gif"}

The payload follows its reference to 1.6 cm RMS and 4.1 cm at worst. It tilts by at most 3.2°. It arrives 0.1 mm from the target, level to 0.01°. The tensions stay between 2.30 and 3.07 N. The closest pair of vehicles is 0.819 m apart with $H_{ij} \ge 0.030$, and the rotors pass 5 cm from the walls. The thrust peaks at 20.0 N of 22. Both programs are feasible at every step. The tension program has active constraints for 2.8 s and the safety filter changes the thrust for 1.1 s.

[](#tbl19-ablation) repeats the mission with one thing changed.

::: {.table #tbl19-ablation}

| Mission | Payload error, RMS | Largest tilt | Lowest tension | Closest pair | Smallest $H_{ij}$ | Peak thrust | Outcome |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-- |
| Everything on | 1.6 cm | 3.2° | 2.30 N | 0.819 m | 0.030 | 20.0 N | flies |
| Minimum-norm tensions | 5.8 cm | 27° | 2.38 N | 0.765 m | −0.056 | 22 N | too close, tilted |
| No pair constraints | 1.6 cm | 5.8° | 2.53 N | 0.800 m | 0.000 | 18.8 N | flies |
| No safety filter | 0.33 cm | 0.6° | 2.43 N | 0.785 m | −0.024 | 19.1 N | too close |
| Neither | 0.02 cm | 0.01° | 2.59 N | 0.758 m | −0.066 | 18.8 N | too close |
| No tension floor in the filter | 1.6 cm | 3.2° | 2.30 N | 0.819 m | 0.030 | 20.0 N | flies |
| Filter rates 2 and 3 s⁻¹ | 3.7 cm | 8.2° | 2.40 N | 0.828 m | 0.046 | 19.8 N | flies |
| No feedforward | 14 cm | 2.3° | 2.57 N | 0.827 m | 0.044 | 18.9 N | lags |
| One cable 10% longer than modeled | 1.7 cm | 4.8° | 2.15 N | 0.819 m | 0.030 | 20.9 N | flies |
| One vehicle at 85% thrust | 59 cm | 93° | −2.4 N | 0.796 m | −0.006 | 22 N | fails |
| Unknown push of 8 N for 0.5 s | 1.8 cm | 3.2° | 2.07 N | 0.696 m | −0.156 | 22 N | too close |
| The same push, filter told | 1.8 cm | 3.2° | 2.04 N | 0.753 m | −0.072 | 22 N | too close |

: The mission with one block changed at a time. $H_{ij} \ge 0$ means the vehicles are at least 0.8 m apart.

:::

Read it as a bill.

*The tension distribution* is the block the mission cannot do without. With minimum-norm tensions the controller pulls every cable toward the vertical, the safety filter pushes back for 11.6 of 12 s, the thrust saturates, and the payload ends tilted by 20°. The filter does not rescue a distribution that asks for the wrong formation. It fights it.

*The pair constraints* cost nothing in tracking and buy half of the filter's work: with them the payload tilts 3.2° and not 5.8°.

*The safety filter* is the only block that holds $r_{\min}$ in this mission, and the only one that makes tracking worse, by a factor of five. Kotaru's remark that the added safety comes at the expense of stability ([](#sec8-quad)) has a number here. Slower rates make it act earlier and cost more.

*The tension floor* did nothing. No tension came near $T_{\min}$ in any row that flew. It is insurance we did not claim.

*The feedforward* of [](#ch-flatness) bought a factor of nine with the protection on, and a factor of 700 without it ([](#tbl19-order)).

*A cable 10 cm too long* tilts the payload by 1.6° more and changes little else. The cascade measures the payload, so a geometric error upstream of it is corrected like any disturbance.

*A weak vehicle* ends the flight. One vehicle delivering 85% of its thrust is 2.6 N short, as much as its whole cable tension. A tension turns negative, which a cable cannot do, and the payload tilts to 93°. Nothing in this stack estimates a thrust. The adaptation of [](#sec8-l1) estimates a torque on one body. An estimate of a force on a team is the block this chapter lacks most.

*A push* defeats the filter, told or not, as [](#sec19-safety) explained.

## Honest limits {#sec19-limits}

**Every cable is assumed taut.** [](#eq19-free), the flat map, the cascade and both linearizations divide by tensions or assume their sign. Two rows of [](#tbl19-ablation) and one of [](#tbl19-lqr) left that assumption. A team with one slack cable is a different system, with one constraint fewer, and switching between them makes the problem hybrid. Kotaru (2022, ch. 5 and 6) treats the slack cable for one vehicle, where grasping requires it.

**The geometry is assumed known.** The attachment points $\rho_i$, the cable lengths and the payload's inertia enter the grasp map and the feedforward. A wrong length cost little. We did not test a wrong attachment or a wrong center of mass, and a payload nobody has weighed is the common case.

**The vehicles do not feel each other's air.** There is no downwash, no drag and no rotor dynamics. $r_{\min}$ stands for all of it, as one number.

**The controller is centralized.** Every block reads the state of the whole team and writes the inputs of every vehicle. The tension program and the safety filter are solved in one place. Nothing here says what to do when a vehicle hears the others late, or not at all. The experiments in Kotaru (2022, sec. 6.6) are centralized as well, on one ground station at 200 Hz.

**The state is given.** No estimator runs. A cable direction would be estimated on the sphere ([](#def14-vekf)), and the payload's pose on the group ([](#def14-inekf)).

**The guarantees are per block.** [](#thm6-s2) is about one direction with its force applied directly. [](#thm8-rd2) keeps $H \ge 0$ for the model, in continuous time, with unlimited inputs. Wu's Prop. 5.3 needs at least six cables and attitude loops that are fast enough. Not one of them is a statement about the mission of [](#sec19-all), and this chapter has not added one. It has a simulation, twenty variants of it, and three tables.

**What was flown.** Kotaru (2022, sec. 6.6) reports two quadrotors carrying a wooden plank of 1.82 m along a square of side 1 m under the centralized controller of his sec. 6.4, with tracking plots for position and attitude, and three quadrotors with a triangular payload under the pairwise safety filter, one of them pushed by hand. He notes that the second was less accurate because its payload position was not fed back. Wu (2017, ch. 5) reports a simulation with seven quadrotors and states that the experiments with a rigid payload could not be included. A tension program with pair constraints, and a model-based filter for tied vehicles, are ours and have flown only in this chapter's simulation.

## Exercises {#sec19-exercises .unnumbered}

::: {.exercise #exr19-null}
For three attachments at the corners of a triangle, show that the null space of $G_L$ in [](#eq19-grasp) is spanned by three pairs of opposite forces, one along each edge. Which combination is the spreading force of [](#sec19-plan)? Check your basis against the notebook's.
:::

::: {.exercise #exr19-line}
Put the three attachments on a line. Compute the rank of $G_L$, and fly the mission in the notebook. Which motion of the payload is no longer controlled? Now lift one attachment 5 cm out of the line and repeat.
:::

::: {.exercise #exr19-cost}
In hover, the thrust of each vehicle is $\|m_i g\,e_3 - T_i q_i\|$. Plot it against the distance between vehicles as $T_{\text{int}}$ grows from 0 to 2 N. How much thrust does a meter of separation cost? Explain why so little, and find the payload mass at which it would cost 5%.
:::

::: {.exercise #exr19-rates}
[](#tbl19-protect) used $\alpha_1 = \alpha_2 = 4$ s⁻¹. In the notebook, find the largest rates for which the safety filter alone still holds $H_{ij} \ge 0$ on the mission's reference, and the tilt of the payload at those rates. Relate what limits you to the attitude loop of the vehicles.
:::

::: {.exercise #exr19-estimate}
Give the stack an estimate of each vehicle's thrust loss. With the model's maps from thrust vectors to accelerations, propose an estimator that uses the measured acceleration of each vehicle, and feed it to [](#eq19-thrust). Does the row "one vehicle at 85% thrust" of [](#tbl19-ablation) fly? Then feed the same estimate to the safety filter and read [](#warn17-order) again.
:::

::: {.exercise #exr19-gap}
Narrow the gap in the notebook until the mission fails with every block on. Which constraint fails first: $r_{\min}$, the walls, or $f_{\max}$? Then let the payload pitch as it passes, and try again.
:::

::: {.exercise #exr19-links}
In [](#sec19-cable), raise the number of links from 5 to 8 at the same total length and mass. Report the dimension of the constraint subspace, the rank of the reduced pair, and the highest derivative the flat map needs. What happens to the reference if the figure-eight is flown twice as fast?
:::

## Notes and sources {#sec19-notes .unnumbered}

The model of several quadrotors carrying a rigid payload on cables, and the matrix that maps cable forces to the payload's wrench, are from Wu (2017, sec. 3.2). Sreenath and Kumar (2013) treat the dynamics, control and planning of the cooperative problem with a dynamic model where earlier work was quasi-static. Lee (2014) gives a geometric controller for a rigid payload, with the cables modeled as rigid links. The controller of Wu (2017, ch. 5) tracks the payload's pose and the cable directions, with a proof by singular perturbation, and needs enough vehicles that the tensions along given directions span every wrench. Kotaru (2022, sec. 6.4) distributes the wrench over three-dimensional contact forces through the grasp map, by a quadratic program with box bounds or by the pseudoinverse, and reports experiments with two and three vehicles. The pairwise safety filter is his (sec. 6.5), in the exponential form of Nguyen and Sreenath (2016), on barrier functions in the sense of Ames, Xu, Grizzle and Tabuada (2017). The flexible cable as a chain of links, its flat output and its variation-based linearization with a finite-horizon LQR are from Kotaru (2022, ch. 7 and Appendix B), following Kotaru, Wu and Sreenath (2018) and Kotaru and Sreenath (2019). Variation-based linearization and controllability on the constraint subspace are due to Wu and Sreenath (2015). The cascade of [](#sec19-cascade) follows Sreenath, Lee and Kumar (2013) for each cable. The tension program with pair constraints, the model-based form of the safety filter, the linearization by differences along variations, the account of the term that separates it from the symbolic one, and every number in this chapter are ours. All models were derived a second time with `geomech` (Kotaru, 2026).

- A. D. Ames, X. Xu, J. W. Grizzle and P. Tabuada, ["Control barrier function based quadratic programs for safety critical systems"](https://arxiv.org/abs/1609.06408), *IEEE Transactions on Automatic Control* 62(8), 2017. arXiv:1609.06408.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- P. Kotaru, [*geomech: symbolic equations of motion on smooth manifolds*](https://github.com/vkotaru/geomech), version 1.1.0, 2026.
- P. Kotaru and K. Sreenath, ["Multiple quadrotors carrying a flexible hose: dynamics, differential flatness, control"](https://arxiv.org/abs/1911.12650), arXiv:1911.12650, 2019.
- P. Kotaru, G. Wu and K. Sreenath, ["Differential-flatness and control of quadrotor(s) with a payload suspended through flexible cable(s)"](https://arxiv.org/abs/1711.04895), *Indian Control Conference*, 2018.
- T. Lee, ["Geometric control of multiple quadrotor UAVs transporting a cable-suspended rigid body"](https://arxiv.org/abs/1403.3684), arXiv:1403.3684, 2014.
- Q. Nguyen and K. Sreenath, ["Exponential control barrier functions for enforcing high relative-degree safety-critical constraints"](https://ieeexplore.ieee.org/document/7524935), *American Control Conference*, 2016.
- K. Sreenath and V. Kumar, ["Dynamics, control and planning for cooperative manipulation of payloads suspended by cables from multiple quadrotor robots"](https://www.roboticsproceedings.org/rss09/p11.html), *Robotics: Science and Systems*, 2013.
- K. Sreenath, T. Lee and V. Kumar, ["Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load"](https://doi.org/10.1109/CDC.2013.6760219), *IEEE Conference on Decision and Control*, 2013.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
- G. Wu and K. Sreenath, ["Variation-based linearization of nonlinear systems evolving on SO(3) and S²"](https://doi.org/10.1109/ACCESS.2015.2477880), *IEEE Access* 3, 2015.
