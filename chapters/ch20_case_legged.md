---
title: "Case study: legged robots"
short: Legged robots
id: ch-case-legged
summary: A biped made of three rotation matrices is taken through the recipe, from a pendulum to a walking and turning gait with impacts, and each claim of the thesis behind it is marked as reproduced, described or absent.
notebook: notebooks/ch20_case_legged.ipynb
verify: verify/ch20.py
---

Geometric control earned its reputation in the air. A quadrotor thrown upside down rights itself ([](#ex6-inverted)), because its controller was never told that upside down is special. That is the selling point of [](#ch-tracking): no singular attitude, and convergence from almost everywhere.

Now put the same controller on a walking robot and ask what it bought. A biped is almost never upside down. If its torso is 150° from where it should be, the robot has already fallen. Its joint torques are modest. And whatever its controller wants to do, it has until the next footfall to do it, a second at most, after which an impact rewrites the velocities.

So why would a biped need any of this?

The question is not ours. Siravuru's thesis (2019), which this chapter follows, asks it in so many words at the start of its fourth chapter: almost-global attraction and freedom from singularities are "more meaningful" for aerial recoveries, and for legged robots "the use of geometric control needs further motivation". Large-angle recoveries, it says, are uncommon in walking, and impractical "due to limited control authority, finite time (to impact) and narrow region-of-attraction".

The thesis then gives three answers. A model written with rotation matrices is compact and has no singular posture, so one model covers walking, leaning and turning ([](#fig20-biped)). It keeps the coupling between the rotational degrees of freedom that planar models drop. And it handles a body whose inertia changes while it moves. This chapter tests each answer on the thesis's own robot, with the library of [](#sec16-library), and says after each test what was reproduced and what was not.

![The Reaction Mass Biped in the middle of a turn, drawn from the simulation of [](#sec20-walk). The configuration is three attitudes, one per body, and four lengths: the stance leg $l$ and the distances of the three pairs of reaction masses from the hip. No joint angle appears.](figures/ch20-biped.svg){#fig20-biped}

## The pendulum, with a deadline {#sec20-pendulum}

A leg that pivots about its foot is a 3D pendulum standing on its pivot. Fix the length $l$ in [](#eq4-leg) and it becomes [](#eq4-pendulum) with $\rho = l\,e_3$ and inertia $J_0 + m l^2\,\Pi$. The library agrees to rounding error. So the thesis starts where [](#ch-dynamics) started, and compares two feedback-linearizing controllers on that pendulum: one on Euler angles, one on $\SO(3)$.

[](#sec6-fl) reran that comparison. Its result is [](#tbl6-grid): over a grid of 208 attitudes the Euler-angle law is undefined at 16 and uses nearly four times the peak torque, and the geometric law is slower and stalls at 29 exact half turns. The grid reaches to 180°. A legged reader should ask what is left of this at the angles of walking, with a deadline.

We asked. Take the same pendulum, the same gains $k_p = 100$ and $k_d = 20$, and the same two laws, taken from the script that checked [](#tbl6-grid). Start at rest, rotated by a given angle about a random axis, 30 axes per angle. Stop at 0.35 s, the step time [](#sec6-legs) used.

::: {.table #tbl20-deadline}

| Initial error | Angle left, geometric | Angle left, Euler | Peak torque, geometric | Peak torque, Euler | Effort, Euler over geometric |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 10° | 1.4° | 1.4° | 1.6 N·m | 1.7 N·m | 1.02 |
| 30° | 4.3° | 4.1° | 4.9 N·m | 5.5 N·m | 1.12 |
| 60° | 10.2° | 8.2° | 9.0 N·m | 11.9 N·m | 1.44 |
| 90° | 20.2° | 13.1° | 9.8 N·m | 18.4 N·m | 2.43 |
| 120° | 40.5° | 19.0° | 9.3 N·m | 27.6 N·m | 5.2 |
| 150° | 86.8° | 24.1° | 6.0 N·m | 43.0 N·m | 17.2 |

: The two feedback-linearizing laws of [](#sec6-fl) on the 3D pendulum, from rest, stopped at 0.35 s. Medians over 30 random axes. Effort is $\int\|\tau\|^2\dd t$.

:::

Read the first two rows, because that is where a walking robot lives ([](#fig20-deadline)). At 10° the two laws leave the same error and ask for the same torque to within 2%. At 30° the Euler law uses 12% more effort. The thesis's conclusion, that the geometric law is the more efficient one at all sizes and not only in large recoveries, holds in every row. But at walking sizes the margin is a few percent, and nobody would choose a controller for that.

The lower rows cut both ways. From 120° the Euler law needs three times the torque, and on one axis nine times. It also gets there: 19° left against 40°. The geometric law pushes with $\|e_R\| = \sin\theta$ ([](#warn-sine)), so far from the target it is gentle and late. With a deadline, late is a cost.

![The two laws of [](#tbl20-deadline). Dots are medians over 30 random axes and bars span all of them. The geometric law behaves the same about every axis. The Euler-angle law does not. In the shaded band, where walking happens, the two are hard to tell apart.](figures/ch20-deadline.svg){#fig20-deadline}

So the first answer to the opening question is a negative one, and it is worth having. Efficiency at small angles is not the reason. The reasons must be in the model.

## A robot made of three attitudes {#sec20-model}

The **Reaction Mass Biped** is the thesis's reduced model of a humanoid. It has a telescoping stance leg, pinned at its foot. A torso sits on the hip and carries three pairs of **reaction masses**, one pair on each body axis, the two masses of a pair sliding symmetrically. A swing leg hangs from the same hip. Every mass is balanced about the hip, so the hip is the center of mass.

Its configuration manifold is the row of [](#tbl2-robots): three attitudes, two leg lengths and three slide positions, fourteen dimensions. The stance dynamics involve thirteen, because the swing leg is balanced about the hip and its length never enters ([](#sec16-legs)). We write $R_1$, $R_P$, $R_2$ for the attitudes of stance leg, torso and swing leg, $l$ for the length of the stance leg, and $r = (r_1, r_2, r_3)$ for the distances of the three pairs from the hip. These are the fields `R1`, `RP`, `R2`, `l` and `slide` of the library ([](#tblA-code)).

With $m$ the total mass, $m_r$ one reaction mass, $\Pi = I - e_3 e_3\T$ and $\Pi_i = I - e_i e_i\T$, the stance dynamics are

$$
\begin{aligned}
m\ddot l &= m l\;\Omega_1\cdot \Pi\,\Omega_1 - m g\,e_3\cdot R_1 e_3 + f ,\\
J(l)\,\dot\Omega_1 + \Omega_1\times J(l)\,\Omega_1 + \dot J\,\Omega_1 &= m g l\,(R_1\T e_3)\times e_3 + \tau_1 - R_1\T R_P\,\tau_P ,\\
J_P(r)\,\dot\Omega_P + \Omega_P\times J_P(r)\,\Omega_P + \dot J_P\,\Omega_P &= \tau_P - R_P\T R_2\,\tau_2 ,\\
2 m_r\,\ddot r_i &= 2 m_r\,r_i\;\Omega_P\cdot\Pi_i\,\Omega_P + f_{s,i} ,\\
J_0\,\dot\Omega_2 + \Omega_2\times J_0\,\Omega_2 &= \tau_2 ,
\end{aligned} \label{eq20-stance}
$$

with $J(l) = J_0 + m l^2\,\Pi$ as in [](#eq4-leg) and $J_P(r) = J_P(0) + 2 m_r \sum_i r_i^2\,\Pi_i$. The inputs are the leg force $f$, the ankle torque $\tau_1$, the two hip torques $\tau_P$ and $\tau_2$, and the three slide forces $f_s$: thirteen inputs for thirteen degrees of freedom. The first two lines are [](#eq4-leg) with the hip torque added. The torso line has the same shape, with the slides playing the part of the leg length. This is the `dynamics` of `reaction_mass_biped` in the library, restated.

Two independent checks stand behind [](#eq20-stance). The symbolic derivation of [](#ch-symbolic), run on a Lagrangian built mass by mass, passes `check_eom` with an error of $2.0\times10^{-7}$, and its accelerations equal the library's to $10^{-14}$ at random states.

The thesis writes these equations with other letters and another vertical. Its Table 5.1 defines $e_3$ along gravity, downward, while its figures draw $e_3$ up. Here $e_3$ points up and gravity acts along $-e_3$, as everywhere in this book.

::: {.table #tbl20-dictionary}

| Thesis | Here | Meaning |
| :-- | :-- | :-- |
| $\rho_1$, $\rho_0$ | $l$, $l_0$ | Length of the stance leg, and the fixed length of the swing leg |
| $s_i$ | $r_i$ | Distance of pair $i$ of reaction masses from the hip |
| $m_L$, $m_P$ | `m_leg`, $m_r$ | Mass of a leg, and one reaction mass |
| $J_{L_0}$, $J_{P_0}$, $K_1$, $K_P$ | $J_0$, $J_P(0)$, $m l^2\,\Pi$, $J_P(r) - J_P(0)$ | Inertias |
| $(e_3^\wedge)^2$ | $-\Pi$ | Minus the projection off the leg axis |
| $f_1$, $\tau_{D1}$, $\tau_{D2}$, $u_s$ | $f$, $\tau_P$, $\tau_2$, $f_s$ | Leg force, hip torques, slide forces |
| $N(s, \dot s)$, $L(s, \Omega_P)$ | $\dot J_P$, $2 m_r\,r_i\,\Omega_P\cdot\Pi_i\,\Omega_P$ | Rate of the torso inertia, centrifugal force on the slides |
| $Q = R\,(R^d)\T$ | $E_r = R\,R_d\T$ | Right-invariant attitude error ([](#def3-invariant-errors)) |
| $\Omega - \Omega^d$ | $\Omega - \Omega_d$ | Velocity error that goes with it |
| $A$, $\Phi$ | $D$, the identity | Weights of the error function |
| $T$, $\zeta_1$ | $t_s$, $2\theta_s\,e_2$ | Duration of a step, rotation of the leg over a step |
| $D_e$, $A$, $I_R$ | $D$, $A$, $p_c$ in [](#sec20-impact) | Floating-base inertia, contact Jacobian, impulse |
| $e_3$ along gravity | $e_3$ up | Flip the sign of every gravity term |

: From the symbols of Siravuru (2019, ch. 5) to those of this chapter.

:::

**What the variable inertia does.** The thesis's second and third answers both rest on $J_P(r)$. Here is its size. With the thesis's masses, the torso's inertia about its vertical axis is $J_{P,33}(0) + 4 m_r r^2$ for equal slides. Over the range the thesis uses while walking, 10 cm to 15 cm, it changes by 6.7%. That is small. Pull the masses from 25 cm in to 5 cm and it drops from 0.305 to 0.228 kg·m², a factor of 1.34.

::: {.thought #thought20-skater name="Turning faster without pushing"}
The torso spins about the vertical at 3 rad/s with its masses out. No torque acts on it. You may push the masses along their rails. Can you make the torso turn faster?

A torque about the vertical is needed to change the angular momentum $J_{P,33}\,\Omega_{P,3}$, and a force along a rail has none. So the momentum is fixed, and the spin is that momentum divided by the inertia. Pull the masses in and the spin must rise by the ratio of the inertias, 1.34. The simulation gives 4.01 rad/s, with the momentum unchanged to twelve digits ([](#fig20-skater)). The kinetic energy rose by 0.46 J. The rail forces supplied it, working against the centrifugal term of the fourth line of [](#eq20-stance).
:::

![The torso of the biped with no torque on it. Forces along the rails pull the reaction masses from 25 cm to 5 cm. The spin rises from 3.00 to 4.01 rad/s and the angular momentum does not move. [](#fig16-skater) showed the reverse: masses let go, and a torso that slows down.](figures/ch20-skater-strip.png){#fig20-skater anim="figures/ch20-skater.gif"}

That centrifugal term carries one of the corrections this book makes to the thesis. Its equation (5.14) prints the term with a minus sign. The Lagrangian gives a plus: a spinning torso throws its masses outward. The thesis's own controller, its equations (5.58) to (5.64), is derived with the plus sign, so only the display of (5.14) is affected.

So far the robot stands on one foot forever. Walking needs the other foot to land.

## Touchdown {#sec20-impact}

A step ends when the swing foot reaches the ground. The thesis models that instant in three moves. Release the stance foot, so that the model has a floating base. Let an impulse act on the swing foot that brings it to rest: the foot neither bounces nor slips. Then exchange the names of the two legs.

The floating-base model adds the hip position $x_P$ to the three attitudes and the slides. Its velocity is $(\Omega_1, \Omega_P, \dot r, \Omega_2, v_P)$ with $v_P = \dot x_P$, and its kinetic energy has the block-diagonal inertia

$$
D = \operatorname{diag}\big(J_0,\ J_P(r),\ 2 m_r I,\ J_0,\ m I\big) .
$$

Both leg axes point from foot to hip, so the swing foot is at $x_c = x_P - l_0\,R_2 e_3$. Its velocity is $A$ times the velocity vector, with $A = \begin{bmatrix} 0 & 0 & 0 & l_0 R_2\,e_3^\wedge & I\end{bmatrix}$. An impulse $p_c$ on the foot changes the momenta by $A\T p_c$, and the foot is at rest afterwards. With $-$ and $+$ for just before and just after,

$$
D\,(v^+ - v^-) = A\T p_c , \qquad A\,v^+ = 0 . \label{eq20-kkt}
$$

Before solving anything, look at $A$. Its first three blocks are zero.

::: {.thought #thought20-hip name="What does the torso feel?"}
The swing foot hits the ground. The impulse travels up the leg to the hip. How much does the torso's angular velocity change?

Not at all. The legs and the torso are joined at one point, the hip, and every body has its center of mass there. The joint is free, so it passes forces and no torque. A force through the center of mass of a body does not turn it. The torso keeps $\Omega_P$, the old stance leg keeps $\Omega_1$, and the slides keep $\dot r$. The impulse changes two things only: the velocity of the hip, and the angular velocity of the leg that landed.
:::

That leaves a three-by-three problem, and it has a closed form. Eliminate $p_c$ and $v_P^{+}$ from [](#eq20-kkt):

$$
J(l_0)\,\Omega_2^+ = J_0\,\Omega_2^- + m\,l_0\; e_3\times R_2\T v_P^{-} , \qquad
v_P^{+} = l_0\,R_2\,(\Omega_2^+\times e_3), \qquad p_c = m\,(v_P^{+} - v_P^{-}) . \label{eq20-impact}
$$

The first equation is a conservation law. Its right side is the angular momentum, about the landing foot, of the swing leg and of the whole mass moving with the hip, in the frame of the leg. Its left side is the same momentum once the leg pivots about that foot, with the inertia $J(l_0)$ of [](#eq4-leg). The impulse acts at the foot and has no moment about it.

Then the legs trade names:

$$
\big(l,\ \dot l,\ R_1,\ \Omega_1,\ R_2,\ \Omega_2\big)\ \longmapsto\ \big(l_0,\ 0,\ R_2,\ \Omega_2^+,\ R_1,\ \Omega_1^-\big) , \label{eq20-relabel}
$$

and the torso and the slides carry over. The new stance leg starts with $\dot l = 0$, because $v_P^{+}$ is perpendicular to its axis. The stance foot moves to $x_c$.

The library has no impact map, so [](#eq20-impact) is implemented in the script of this chapter, and checked five ways at 50 random states.

1. *Against the full system.* Solving the 18 equations of [](#eq20-kkt) gives the same $\Omega_2^+$, $v_P^{+}$ and $p_c$ to $10^{-14}$, and leaves $\Omega_1$, $\Omega_P$ and $\dot r$ untouched.
2. *Against a second derivation of $D$.* The symbolic library derives the floating-base model from a Lagrangian built mass by mass (`check_eom` $4.5\times10^{-8}$). Its mass matrix equals $D$ to rounding error.
3. *Momentum.* The angular momentum of the whole robot about the landing foot, summed over every mass and body, is the same before and after, to $10^{-14}$.
4. *Energy.* The kinetic energy lost equals the kinetic energy of the jump, $\tfrac12 (v^+ - v^-)\cdot D\,(v^+ - v^-)$, and is never negative.
5. *Consistency.* The relabeled stance state reproduces $v_P^{+}$. Landing again on the same foot changes nothing. Trading names twice is the identity.

![The impact at the end of a step of the periodic gait of [](#sec20-walk), computed with [](#eq20-impact). The hip arrives at 0.40 m/s, almost straight down the leg that lands. A rigid leg cannot give way along its axis, so the impulse takes nearly all of it, and 0.29 J is lost. The torso's angular velocity and the slides do not change.](figures/ch20-impact.svg){#fig20-impact}

Two details of the thesis's text needed a decision, recorded in the Notes at the end of this chapter. Its page 46 places the swing foot at $x_P + \rho_2 R_2 e_3$, while its equation (5.43) and its switching surface use $x_P - \rho_2 R_2 e_3$. We use the second, for which [](#eq20-relabel) is a plain exchange. And its equation (5.20) prints the right side of the linear system as $-D\,v^-$, where the relation stated just above it gives $+D\,v^-$.

::: {.warning #warn20-impact name="An impact map is a modeling claim too"}
[](#eq20-impact) assumes that the foot sticks, that the ground can pull as well as push for an instant, and that the old stance foot leaves without an impulse of its own. None of these is enforced. After every impact check that $p_c$ points up and lies inside the friction cone, as [](#warn2-taut) asks you to check a cable's tension.
:::

## Walking and turning {#sec20-walk}

**The reference.** The thesis prescribes each step by a **motion primitive**: a reference for every factor over one step, in closed form. In our convention, for a step of duration $t_s$ with heading $\psi$,

$$
\begin{aligned}
R_{1,d} &= R_z(\psi)\,R_y(\theta_d), & R_{2,d} &= R_z(\psi)\,R_y(-\theta_d), & \theta_d &= -\theta_s + 2\theta_s\sin\frac{\pi t}{2 t_s},\\
l_d &= l_0 + 0.1\,\sin\frac{\pi t}{t_s}, & r_{i,d} &= 0.125 + 0.025\,\sin\frac{\pi t}{t_s}, & R_{P,d} &= R_z(\psi)\,R_y(10^\circ) ,
\end{aligned} \label{eq20-primitive}
$$

with lengths in meters. The stance leg tips forward from $-\theta_s$ to $\theta_s$ while the swing leg does the opposite. The stance leg lengthens in mid-step, which lifts the swing foot by $(l_d - l_0)\cos\theta_d$ and sets it down again at $t = t_s$. This is the family of the thesis's equations (5.41) and (5.42), with its numbers $l_0 = 0.9$ m, $t_s = 1$ s, legs swinging from −15° to 15°, and its constant forward lean of the torso.

Three things in [](#eq20-primitive) are our choices, because the thesis does not print them: the "constant phase offset" that produces the swing from −15° to 15°, the lean angle, and the function $\Phi$ in its error function. So what follows reproduces the thesis's method and its parameters where given. It cannot reproduce its curves point for point.

Notice what the reference does at $t_s$. The legs stop: $\dot\theta_d(t_s) = 0$. At the start of the next step they move at $\theta_s\pi/t_s = 0.82$ rad/s. The reference is not continuous across a touchdown, and neither is the robot. Every step starts with an error.

**The law.** The thesis tracks the primitive with a law of the family of [](#eq6-law), and two of its choices are worth seeing next to the book's. It measures attitude error with the right-invariant error $E_r = R\,R_d\T$ and the weighted function

$$
\Psi = \tr\!\big(D\,(I - R\,R_d\T)\big), \qquad \dot\Psi = e\cdot(\Omega - \Omega_d), \qquad e = \sum_i d_i\,(R\T e_i)\times(R_d\T e_i) . \label{eq20-right}
$$

Compare [](#eq3-weighted-psi), which weights $R_d\T R$. With $R\,R_d\T$ the velocity error is the plain difference $\Omega - \Omega_d$: no transport map is needed ([](#def3-transport)). The price is that the weights $d_i$ now belong to the axes of the world, and we will meet that price below.

For a body with inertia $J$ that changes with the configuration, the law asks for the net torque

$$
\tau = -e - k_\Omega\,(\Omega - \Omega_d) + J\,\dot\Omega_d + \Omega_d\times J\,\Omega + \tfrac12\,\dot J\,(\Omega + \Omega_d) , \label{eq20-law}
$$

plus the cancellation of gravity on the stance leg. The lengths get a PD law with feedforward and gains $k_l$, $k_{\dot l}$, $k_r$, $k_{\dot r}$. The joint torques follow by stacking: $\tau_2$ is the net torque on the swing leg, $\tau_P$ adds $R_P\T R_2\,\tau_2$ to the net torque on the torso, and $\tau_1$ adds $R_1\T R_P\,\tau_P$ to that on the stance leg. Substituting into [](#eq20-stance) gives, for each body,

$$
J\,\ddt(\Omega - \Omega_d) = -(\Omega - \Omega_d)\times J\,\Omega - \tfrac12\,\dot J\,(\Omega - \Omega_d) - k_\Omega\,(\Omega - \Omega_d) - e ,
$$

and the sum over the three bodies of $\tfrac12(\Omega - \Omega_d)\cdot J\,(\Omega - \Omega_d) + \Psi$, plus the two PD energies of the lengths, is a Lyapunov function $V$ whose rate is $-\sum k_\Omega\|\Omega - \Omega_d\|^2$ minus the damping of the lengths. These are the thesis's equations (5.61) to (5.76). We implemented them as printed, flipped gravity, and checked two things at random states: the closed loop matches its (5.72) to (5.75) to $10^{-13}$, and $\dot V$ along the closed loop equals its (5.70) to seven digits.

The term $\tfrac12\dot J$ is the new one. [](#eq6-law) cancels $\Omega\times J\Omega$ and was written for a constant inertia. Apply it to a leg that telescopes, with $\dot J\,\Omega$ canceled as well, and [](#eq6-V0) is no longer a Lyapunov function: its rate gains $\tfrac12 e_\Omega\cdot\dot J\,e_\Omega$, of either sign. Subtract $\tfrac12\dot J\,e_\Omega$ from the torque and [](#eq6-V0-dot) is restored. We checked both statements numerically. That one term is what a configuration-dependent inertia adds to [](#ch-tracking).

The gains are those of the thesis's Table 5.3: $k_l = 256$, $k_{\dot l} = 32$, $D = \operatorname{diag}(19.2,\ 24,\ 28.8)$, $k_\Omega = 8$ and $2$ for stance and swing leg, $k_r = 19.2$, $k_{\dot r} = 6$. The table lists no damping for the torso under that name. We use its entry $L = 1.8$.

**The loop.** Simulate [](#eq20-stance) with the step of the library at $h = 1$ ms, the controller sampled at the same rate. When the swing foot comes down to the ground, apply [](#eq20-impact) and [](#eq20-relabel), move the stance foot, and start the next primitive. Nothing else. The robot has the thesis's masses: 3.68 kg in all.

It walks ([](#fig20-turn), [](#fig20-errors)). From a start on the reference, the first step is 46.6 cm, the reference's own $2 l_0\sin\theta_s$. By the fourth step the gait has settled to a step of 49.1 cm every 0.989 s, and after ten steps consecutive steps differ by less than $2\times10^{-5}$ in every quantity we compared. Within a step the stance leg is at worst 8.4° from its reference and arrives 1.2° behind. The leg length is off by at most 20 mm and the slides by 19 mm. The swing foot clears the ground by 14 mm at worst in mid-step.

At each touchdown the hip arrives at 0.40 m/s and leaves at 0.02 m/s, the impact takes 0.29 J, and $V$ jumps from 0.13 J to 1.58 J. Then it decays, sample after sample, until the next touchdown. That sawtooth is the whole gait: an impact that sets the error, and a stance phase that removes most of it.

The ground stays inside the assumptions. The vertical force stays between 29 N and 48 N, around a weight of 36 N. The ratio of horizontal to vertical force peaks at 0.25, and the impulse is 1.38 N·s upward with a ratio of 0.32. The ankle supplies up to 16 N·m.

Set this beside the thesis's Figure 5.3, as far as a plot can be read. Its length errors peak near 0.02 to 0.03 at every step, and its attitude errors between 0.1 and 0.2 rad. Ours are 0.020 m and 0.15 rad. Its ankle torque is a sawtooth within about $\pm10$ N·m, its vertical force stays a few newtons around the weight, and its force ratios stay under 0.2 on the straight walk. It reports impulses that point up and are at least 1.60 in every one of its impacts, with ratios up to 0.59 on its circle. The shapes and sizes agree. The numbers do not coincide, and with three unprinted choices they should not be expected to.

![Fourteen steps: four straight, six in which the heading of the reference jumps by 15° at each touchdown, four straight again. The dashed gray legs are the reference. The bar is the Lyapunov function of the tracking law. Watch it jump at each touchdown and drain during the step.](figures/ch20-turn-strip.png){#fig20-turn anim="figures/ch20-turn.gif"}

**The turn.** The thesis turns by changing the heading of the primitive at each impact. Do that: after four steps, add 15° to $\psi$ at each of six touchdowns, and roll the torso 20° into the turn. Each of those steps starts with 15° of error in both legs, and up to 25° in the torso. Each ends with less than 2.7° in the legs. After the sixth, the robot walks along $e_2$: the heading of its last step is 89.99°. The force ratio never exceeds 0.26, the impulse ratio 0.34, and the smallest clearance of the swing foot is 5 mm.

The turn costs torque. The ankle must twist the whole robot about the vertical at each change of heading, and its torque about $e_3$ peaks at 37 N·m, more than twice anything in the straight walk. The thesis's turning plots show the same spikes, larger. A point foot cannot supply a twisting torque at all, and we return to that in [](#sec20-limits).

Here the first two of the thesis's answers become concrete. The controller that walked straight also turned, with the same model, the same gains and no special case, through headings where a planar model has nothing to say. No attitude along the way was singular.

![The two walks as data. Top: attitude error of each body. Middle: the ratio of horizontal to vertical ground force, against the value 0.6 that the thesis assumes available. Bottom: the ankle torque. On the straight walk it acts about $e_2$ only. In the turn its component about $e_3$ spikes at each change of heading.](figures/ch20-errors.svg){#fig20-errors}

## What each choice bought {#sec20-compare}

[](#tbl20-compare) changes one thing at a time and lets the gait settle again. Each row is the last of seven or eight steps of a straight walk.

::: {.table #tbl20-compare}

| Walk | Step length | Step time | Largest leg error | Leg error at touchdown | Peak ankle torque | Peak force ratio |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| Thesis law, heading 0° | 49.1 cm | 0.989 s | 8.4° | 1.2° | 15.7 N·m | 0.22 |
| Thesis law, heading 90° | 50.6 cm | 0.982 s | 8.3° | 1.9° | 16.5 N·m | 0.23 |
| [](#eq6-law) on each factor, $k_R = 48$, any heading | 49.1 cm | 0.989 s | 8.4° | 1.2° | 15.7 N·m | 0.22 |
| The same without $-\tfrac12\dot J\,e_\Omega$ | 48.8 cm | 0.992 s | 8.6° | 0.9° | 15.6 N·m | 0.22 |
| The same without feedforward | 55.1 cm | 1.094 s | 11.7° | 4.4° | 17.6 N·m | 0.21 |
| Thesis law, twice as fast ($\kappa = 2$) | 45.7 cm | 1.006 s | 5.3° | 0.5° | 28.7 N·m | 0.55 |

: The periodic gait with one thing changed at a time. The reference step is 46.6 cm in 1 s.

:::

*The heading.* The first two rows are the same robot, the same law and the same gait, walking along $e_1$ and along $e_2$. They differ by 1.5 cm a step ([](#fig20-gait)). The cause is [](#eq20-right). A small rotation about the world axis $e_1$, $e_2$ or $e_3$ is resisted with stiffness $d_2 + d_3$, $d_1 + d_3$ or $d_1 + d_2$: 52.8, 48 and 43.2 N·m for these weights. Walking along $e_1$, the legs pitch about $e_2$ and feel 48. Walking along $e_2$, they pitch about $e_1$ and feel 52.8. A weight on world axes makes the controller depend on which way the robot faces.

*The book's error.* The third row uses [](#eq6-law) on each factor with the trace error and the scalar gain $k_R = 48$. Its gait is the same at every heading, exactly, because nothing in it refers to a world axis other than the vertical. At heading 0° it is also the same as the thesis's, to the last digit: for a planar pitch the two laws coincide. The thesis's unequal weights have a purpose of their own: they are what [](#thm6-almost) needs.

*The half-rate term.* Leaving it out moves the gait by 4 mm. A proof needs it. This gait would not miss it.

*Feedforward.* Without the reference's velocity and acceleration the robot still walks, 18% farther per step than asked and 0.1 s late, with nearly four times the error at touchdown. In a gait with a deadline, feedforward is most of the tracking.

*Faster gains.* Multiplying the stiffnesses by 4 and the dampings by 2 halves the errors, doubles the ankle torque, and raises the force ratio to 0.55, against a friction coefficient the thesis assumes to be above 0.6. The feet would slip before the motors complained. On a legged robot the ground, not the actuator, is the first bound a high gain meets.

One row is missing because it is identical to the first: hold the reaction masses still. With a torso that only leans, the blocks of [](#eq20-stance) are decoupled and the walk does not change by a micrometer. In this gait the reaction masses do nothing. They matter when the torso must turn, as [](#thought20-skater) showed, and the thesis leaves gaits that use them to future work.

![Step length over the first eight steps. The gait settles in about four steps. With the thesis's weights on world axes it settles to a different step when the robot faces along $e_2$. With scalar gains the two headings give the same points.](figures/ch20-gait.svg){#fig20-gait}

## The rest of the recipe {#sec20-recipe}

Walk along the boxes of [](#fig16-pipeline) and ask of each what the biped got.

**Errors and transport** ([](#ch-variations)). One error per factor, three copies of the same construction, and two ways to build it: $R_d\T R$ with a transport map, or $R\,R_d\T$ without one. [](#tbl20-compare) measured the difference.

**Tracking, and what an almost-global result means here** ([](#ch-tracking)). The thesis's Theorem 5.1.1 states that its laws asymptotically stabilize a reference of the form of the primitive, with almost-global convergence in the absence of control constraints and disturbances. Its proof is the invariance argument of [](#thm6-almost), run on the sum $V$. Read factor by factor: each of the three weighted error functions has four critical points, so there are 64 combinations, one of which is the reference. The exceptional set is made of the starts that converge to one of the other 63. As [](#thm6-almost) warns, this promises no rate, and the exponential bound of [](#thm6-exp) is not claimed.

It is also a statement about one stance phase. It says nothing about the sequence of stance phases and impacts, and neither do we. That the gait above settles to a periodic one is a measurement.

**Linearization** ([](#ch-linearization)). The linear state has one block per factor: $(\delta l, \delta r, \eta_1, \eta_P, \eta_2)$ and the variations of the velocities, 26 numbers. The kinematic rows are [](#eq3-delta-Omega). The dynamic rows we took by differences of [](#eq20-stance) along the variations, and the result equals the symbolic linearization of [](#ch-symbolic) in all 26 rows, to $10^{-9}$. Along the primitive it predicts a neighboring open-loop motion over 0.2 s to 0.13%. At mid-stance its largest eigenvalue is 2.9 s⁻¹: the leg is an inverted pendulum, and falls with a time constant of a third of a second.

With that linearization, the Riccati equation [](#eq7-riccati) integrated backward over one stance phase gives a time-varying gain on the error state of [](#sec13-error). Started from the state just after a periodic impact, with weights 400 on the errors of the configuration and 1 on those of the velocities and on the inputs, it holds the legs within 4.8° where the thesis law allows 8.4°, and ends at 1.0° against 1.3°. It spends less than half the input above the feedforward (37 against 96 in $\int\|\delta u\|^2\dd t$) and asks for a higher peak, 24 N·m against 16. As in [](#sec17-local), the optimal controller is better where it is valid and silent about everywhere else.

**Integration** ([](#ch-integrators)). The thesis integrates the biped with the explicit update of [](#sec12-biped). [](#warn12-explicit) showed on a pendulum that such an update stays on the group and gains energy. The full biped does the same. In 0.6 s of uncontrolled stance at $h = 1$ ms, the update of the thesis's equations (5.36) to (5.40) gains 11 mJ, and half of that at half the step. The step of the library holds the energy to $2\times10^{-12}$ J. Both keep $R\T R = I$ to $10^{-14}$. So we walked with the library's step: over all fourteen steps of the turn the attitudes stay on $\SO(3)$ to $10^{-14}$, which is what the thesis reports for its integrator ($10^{-12}$). The implicit integrator of [](#thm12-lgvi) is written for a constant inertia. Extending it to $J(l)$ and $J_P(r)$ is [](#exr20-lgvi).

**Planning** ([](#ch-flatness), [](#ch-mpc)). The primitives here are designed by hand, as the thesis says of its own. A fully actuated stance phase needs no flat output: any twice-differentiable reference is feasible, and the input follows by substitution. What planning would add is the choice of a reference that respects the friction cone, the torque bounds and the footholds, with the impact as a constraint between phases. The direct transcription of [](#sec13-plan) and [](#sec12-collocation) is the tool, and the thesis names trajectory optimization on manifolds as its first future direction. Nothing of that kind is done here.

**Estimation** ([](#ch-filtering)). Every law above used the true state. For a legged robot the estimator of this book is the contact-aided invariant filter of [](#sec14-legged), in which a foot on the ground joins the group element as one more column, as in [](#eq14-contact). It estimates a floating base from an IMU and leg kinematics, which is the missing input for a biped whose stance foot is not bolted to the floor. We have not closed that loop.

## Cassie {#sec20-cassie}

The thesis's sixth chapter moves from a model to a machine. Cassie is a biped built by Agility Robotics. The thesis models it with a floating pelvis in $\SE(3)$, one rotation matrix per leg and two leg lengths, the row of [](#tbl2-robots): fourteen degrees of freedom, eight of them actuated. The knees are prismatic and the feet massless, a deliberate trade of fidelity for a model that fits on a product of rotation groups ([](#fig20-cassie)).

What the thesis does with it, in its own terms:

- It derives the floating-base dynamics and imposes the ground contact as constraints, with multipliers that must stay positive.
- It adds a pair of Hovershoes, self-balancing wheeled platforms, one under each foot. Their internal controller is unknown, so only the structure of their dynamics is used. The no-sideways-motion condition at each foot is a nonholonomic constraint. With all constraints, fourteen degrees of freedom become ten.
- It plans a turn in place as a constant angular velocity of the pelvis about the vertical, and tracks it with a PD law on the errors $e_R$ and $e_\Omega$ of [](#ch-variations). It chooses PD over feedback linearization on purpose, trading efficiency for tolerance to the unknown model.
- It shows a simulated rest-to-rest turn of 2.5 s.
- It reports experiments it calls preliminary: Cassie standing under the geometric PD law, which tolerated pushes over a wider range of error than the Euler-angle controller with the same gains, and Cassie on Hovershoes making two turns in place at 1.5 rad/s. It notes that the robot had to be placed on the Hovershoes with great care, and that the body oscillated during the turns.

What this book has of that is little, and we will be exact about it. The library's `cassie` is the floating-base model only, built from our reading of the thesis's kinematics, with placeholder masses, and checked against the symbolic derivation ([](#sec16-legs)). It has no ground, no Hovershoes and no controller. A robot without a ground falls. So nothing about Cassie is simulated or verified in this chapter, and [](#fig20-cassie) is a drawing.

![The reduced model of Cassie in the thesis: a pelvis with position and attitude, an attitude and a length for each leg, and a Hovershoe under each foot. A schematic of the configuration manifold. The hips are drawn farther apart than they are.](figures/ch20-cassie.svg){#fig20-cassie}

## Honest limits {#sec20-limits}

**Reproduced here, with checks.** The stance dynamics [](#eq20-stance), against a symbolic derivation. The impact map [](#eq20-impact), against the full linear system, a second derivation of the inertia, and conservation of angular momentum about the landing foot. The thesis's tracking law, its closed-loop equations and the rate of its Lyapunov function. A walking gait on a straight line and through a 90° turn, with the thesis's masses, gains and step parameters, in which unilateral contact and a friction ratio of 0.6 hold throughout. The energy drift of the thesis's explicit update.

**Described from the thesis, not reproduced.** Its curves, point for point: three parameters of its primitive are not printed. Its walk toward a chosen point and its full circle of 19 steps with a 30° lean. Its comparison of integrators inside the closed loop. Its almost-global theorem, which we restate and do not prove. Everything about Cassie. Its third part, on learning gait policies and step lengths from images, is outside this book.

**Absent.** A variational integrator for the biped. A planner. An estimator in the loop. Any model of Cassie's contact.

**Not modeled at all.** *Underactuation.* The stance model has an ankle that supplies any torque, 37 N·m about the vertical in the turn. A foot of finite size bounds the two tilting torques by the position of the center of pressure, and a point foot supplies none. The thesis says this itself: its controllers apply to models with full actuation, "i.e. any flat-footed humanoid robot". Cassie is not one. *Contact.* The foot is a pin while it is on the ground and an impulse when it lands. We monitored the force and the impulse against a friction ratio and never enforced it. *The swing foot.* It is a point at the end of a leg whose length is set, not simulated. *Hybrid stability.* The gait is periodic to five digits after ten steps. We did not compute the linearization of the step-to-step map, and we prove nothing about it.

**The opening question.** At walking angles the geometric law is not better than an Euler-angle law at tracking ([](#tbl20-deadline)). What the geometry bought was structural. One set of equations, checked once, for standing, walking and turning. An impact map that is three lines because each body has its own attitude. A controller with no special case at any heading. And one caution in return: a weight attached to world axes is a hidden dependence on heading.

## Exercises {#sec20-exercises .unnumbered}

::: {.exercise #exr20-deadline}
Rerun [](#tbl20-deadline) in the notebook with the logarithm error of [](#sec3-errors) in the geometric law. Which rows change, and does the geometric law now meet the deadline from 120°? Then give every start a body rate of 2 rad/s about a random axis. Do the first two rows still agree?
:::

::: {.exercise #exr20-impact}
Derive [](#eq20-impact) from [](#eq20-kkt). Then move the center of mass of the swing leg from the hip to the middle of the leg. Which blocks of $D$ and $A$ change? Argue, without solving, whether the torso's angular velocity still survives the impact.
:::

::: {.exercise #exr20-energy}
For the planar case, with both legs in one vertical plane and $J_0 = 0$, show from [](#eq20-impact) that the fraction of kinetic energy kept through the impact is $\sin^2$ of the angle between $v_P^{-}$ and the axis of the landing leg. Evaluate it for the periodic gait and compare with the 0.29 J of [](#fig20-impact). What would a reference have to do at $t_s$ to lose less?
:::

::: {.exercise #exr20-heading}
In the notebook, walk eight steps with the thesis law at headings of 0°, 45° and 90°. Then set $D = 24\,I$ and repeat. Explain every number that changed and every one that did not. What does [](#thm6-almost) lose when the weights are equal?
:::

::: {.exercise #exr20-slip}
Make the gains faster in steps, $\kappa = 1, 1.25, 1.5, \dots$, and record the peak force ratio of the periodic gait. At what $\kappa$ does it cross 0.6? Then change the primitive so that the legs do not stop at $t_s$, by replacing $\sin(\pi t/2t_s)$ with $t/t_s$, and repeat. Which helps more, and why?
:::

::: {.exercise #exr20-lgvi}
Write the discrete Lagrangian of the telescoping leg [](#eq4-leg) with the midpoint rule for $l$ and the step rotation of [](#def12-step) for $R$. Derive its discrete Euler-Lagrange equations, as [](#thm12-lgvi) does for a constant inertia. Implement the step and repeat the energy test of [](#sec20-recipe). Compare with the explicit update [](#eq12-rmb).
:::

::: {.exercise #exr20-masses}
Design a turn that uses the reaction masses. During each step of the turn, command the horizontal pairs in while the torso yaws to the new heading and out again before touchdown. Measure the peak of $\tau_P$ about the vertical with and without the motion of the masses. Use [](#thought20-skater) to predict the ratio first.
:::

## Notes and sources {#sec20-notes .unnumbered}

The Reaction Mass Biped, its stance and floating-base models, the impact map, the motion primitives, the tracking law with its Lyapunov function and the walking and turning simulations are due to Siravuru (2019, ch. 5), and were first published by Siravuru, Viswanathan, Sreenath and Sanyal (2018). The model generalizes the reaction mass pendulum of Sanyal and Goswami, which the thesis cites for its tracking scheme. The comparison of Euler-angle and geometric control on the 3D pendulum and the question that opens this chapter are from ch. 4 of the thesis, and the geometric model of Cassie, its augmentation for Hovershoes and the turning experiments from ch. 6. The Hovershoe platform and its first controllers are described by Chen, Rogers, Zhang and Sreenath (2019). Rigid impacts with a foot that sticks, hybrid models of walking and the stability of periodic gaits are treated by Westervelt, Grizzle, Chevallereau, Choi and Morris (2007). The attitude law that the template comes from is that of Lee, Leok and McClamroch (2010), and the contact-aided filter is that of Hartley et al. (2018). The deadline experiment, the closed form [](#eq20-impact), the comparison of the two error conventions across headings, the LQR and every number in this chapter are ours, and they are simulations.

Where this chapter departs from the thesis as printed, in addition to the points recorded in [](#sec4-products), [](#sec6-fl), [](#sec12-biped) and [](#sec16-legs):

- Equation (5.20) has $-D_e\,\omega_e^-$ on its right side. The relation above it, $D\,\omega^+ - D\,\omega^- = F_{ext}$, gives $+D_e\,\omega_e^-$. With the printed sign, a velocity that already satisfies the contact constraint is returned reversed.
- Page 46 places the swing foot at $x_P + \rho_2 R_2 e_3$. Equation (5.43) and the switching surface use $x_P - \rho_2 R_2 e_3$. With the first, and a plain exchange of the legs, the new stance leg implies the opposite of the hip velocity that the impact produced. We use the second.
- Equation (5.14) has $-L(s, \Omega_P)$, as [](#sec16-legs) records. The controller derivation, equations (5.58) to (5.64), uses $+L$ and is consistent with the corrected sign.
- Table 5.3 lists a gain $L$ that the theorem does not use, and no $D_P$. We took $L$ for the torso's damping. The offset of the primitive, the lean of the torso and the function $\Phi$ are not printed, and [](#sec20-walk) states our choices.

- S. Chen, J. Rogers, B. Zhang and K. Sreenath, ["Feedback control for autonomous riding of Hovershoes by a Cassie bipedal robot"](https://arxiv.org/abs/1907.11353), arXiv:1907.11353, 2019.
- R. Hartley, M. Ghaffari Jadidi, J. W. Grizzle and R. M. Eustice, ["Contact-aided invariant extended Kalman filtering for legged robot state estimation"](https://doi.org/10.15607/RSS.2018.XIV.050), *Robotics: Science and Systems*, 2018. arXiv:1805.10410.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- A. Siravuru, S. P. Viswanathan, K. Sreenath and A. K. Sanyal, ["The Reaction Mass Biped: geometric mechanics and control"](https://doi.org/10.1007/s10846-017-0508-7), *Journal of Intelligent & Robotic Systems* 89(1-2), 2018.
- E. R. Westervelt, J. W. Grizzle, C. Chevallereau, J. H. Choi and B. Morris, [*Feedback Control of Dynamic Bipedal Robot Locomotion*](https://www.taylorfrancis.com/books/9781420053722), CRC Press, 2007.
