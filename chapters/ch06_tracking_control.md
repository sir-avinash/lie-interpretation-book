---
title: Tracking control and almost-global stability
short: Tracking control
id: ch-tracking
summary: A proportional-derivative law built from the errors of Chapter 3, a Lyapunov proof you can check line by line, and an honest account of what "from almost anywhere" means for rigid bodies, directions, quadrotors and legs.
notebook: notebooks/ch06_geometric_tracking.ipynb
verify: verify/ch06.py
epigraph: "… it is the mark of an educated man to look for precision in each class of things just so far as the nature of the subject admits."
epigraph_source: "Aristotle, *Nicomachean Ethics* I.3 (trans. W. D. Ross)"
---

A bead slides on a wire. You push it with $-k_p e - k_d \dot e$, where $e$ is its distance from the target. Start it a millimeter away or a kilometer away, at rest or flying past. It always arrives. The proof is two lines: the energy $\tfrac12\dot e^2 + \tfrac12 k_p e^2$ changes at the rate $-k_d\dot e^2$, which is never positive, and it can only stop changing at the target.

Now do the same for a tumbling quadrotor. Replace $e$ by the attitude error $e_R$ and $\dot e$ by the velocity error $e_\Omega$ of [](#def-error-so3). The law looks identical. Does it also converge from anywhere?

It cannot. [](#thm-obstruction) forbids it for every continuous law, and you already know one start that fails: half a turn away, at rest, $e_R = 0$ and nothing pushes ([](#fig6-puzzle)). So the promise has to be weaker than the bead's. How much weaker? One bad attitude? A band of them? And what replaces the two-line proof?

![The same law in two worlds. On the wire every start slides to the target. In the ball of rotations of [](#fig-so3-ball), simulated starts from 160° away also run in to the center. But the whole boundary, the half turns, is a place where the restoring torque is zero, and a start close to it leaves slowly.](figures/ch06-puzzle.svg){#fig6-puzzle}

## A PD law on the rotation group {#sec6-pd}

Take a rigid body with a torque on each axis, the attitude part of [](#eq4-rigid-body):

$$
J\dot\Omega + \Omega\times J\Omega = \tau, \qquad \dot R = R\,\Omega^\wedge .
$$

The reference is a moving attitude $R_d(t)$ with $\dot R_d = R_d\,\Omega_d^\wedge$. We want $R_d\T R \to I$. By [](#eq3-error-kinematics) the relative attitude moves with body rate $e_\Omega = \Omega - R\T R_d\,\Omega_d$, so the first task is to find out how $e_\Omega$ itself changes. Differentiate it. The transport map $R\T R_d$ moves too:

$$
\dot e_\Omega = \dot\Omega + \Omega\times\big(R\T R_d\,\Omega_d\big) - R\T R_d\,\dot\Omega_d . \label{eq6-eOmega-dot}
$$

The middle term is new. It is the rate at which the body's own rotation turns the transported reference velocity in the body frame. Multiply [](#eq6-eOmega-dot) by $J$ and substitute the dynamics. Everything that is not feedback can now be canceled by the torque, and what remains is ours to choose:

$$
\tau = \underbrace{-k_R\,e_R - k_\Omega\,e_\Omega}_{\text{PD}} \;+\; \underbrace{\Omega\times J\Omega - J\big(\Omega\times R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d\big)}_{\text{feedforward}} . \label{eq6-law}
$$

Here $k_R > 0$ and $k_\Omega > 0$ are scalar gains. The second group is the **feedforward**: it is computed from the model and the reference, not from the error. This is the attitude law of Lee, Leok and McClamroch (2010). In closed loop,

$$
J\,\dot e_\Omega = -k_R\,e_R - k_\Omega\,e_\Omega, \qquad \ddt\big(R_d\T R\big) = \big(R_d\T R\big)\,e_\Omega^\wedge . \label{eq6-closed}
$$

::: {.thought #thought6-forgot name="The error has forgotten the reference"}
Read [](#eq6-closed) and look for $R_d(t)$ on its own. It is not there. The pair $(R_d\T R,\ e_\Omega)$ obeys a differential equation with constant coefficients. Ask the body to hold still, or to follow a satellite that tumbles about three axes at once. Start both experiments with the same relative attitude and the same velocity error. The two error histories are then the same function of time.

This is the invariance of [](#sec3-invariant) at work. A statement proved for a fixed target is proved for every reference.
:::

In simulation the two histories agree to $3\times10^{-10}$, the tolerance of the integrator. Throughout the chapter we use one vehicle, the quadrotor of Lee, Leok and McClamroch (2010): $m = 4.34$ kg, $J = \operatorname{diag}(0.0820,\ 0.0845,\ 0.1377)$ kg·m², $k_R = 8.81$ N·m, $k_\Omega = 2.54$ N·m·s. Near the target $e_R \approx \eta$ ([](#sec3-first-order)), so each axis behaves like $J_i\ddot\eta_i + k_\Omega\dot\eta_i + k_R\eta_i = 0$. With these gains all three axes are overdamped, and the slowest pole is at $-3.98\ \text{s}^{-1}$.

Near the target, then, [](#eq6-closed) is the bead on the wire. What survives far from it?

## The proof, and what it needs {#sec6-lyapunov}

Copy the bead's energy. The height $\Psi_R$ of [](#def-error-so3) plays the part of $\tfrac12 e^2$:

$$
V_0 = \tfrac12\,e_\Omega\cdot J e_\Omega + k_R\,\Psi_R . \label{eq6-V0}
$$

Compatibility, the property $\dot\Psi_R = e_R\cdot e_\Omega$ of [](#eq-psi-dot), was designed for this moment. With [](#eq6-closed),

$$
\dot V_0 = e_\Omega\cdot(-k_R e_R - k_\Omega e_\Omega) + k_R\,e_R\cdot e_\Omega = -k_\Omega\|e_\Omega\|^2 \le 0 . \label{eq6-V0-dot}
$$

Two things follow at once, and one thing does not.

**A fence.** $V_0$ never increases and $\Psi_R \le V_0/k_R$. Since $\Psi_R = 2$ only at half a turn, a start with $V_0(0) < 2k_R$ can never reach half a turn. Define $\Psi_{\max} = V_0(0)/k_R$. Then $\Psi_R(t) \le \Psi_{\max} < 2$ for all time. The set $\{V_0 < 2k_R\}$ is a **sublevel set** of $V_0$: it contains every attitude short of half a turn, provided the velocity error is small enough to stay below the rim.

**Convergence.** $V_0$ decreases whenever $e_\Omega \ne 0$.

**No rate.** At a state with $e_\Omega = 0$ and $e_R \ne 0$, [](#eq6-V0-dot) reads $\dot V_0 = 0$. The body is at rest relative to the reference, 57° off, and $V_0$ cannot see that $e_R$ is about to accelerate it. A bound of the form $\dot V \le -\gamma V$, which is what exponential convergence means, is impossible for $V_0$.

The standard cure is a **cross term**. Add a small multiple of $e_R\cdot J e_\Omega$, which is negative when the velocity error points downhill:

$$
V = \tfrac12\,e_\Omega\cdot J e_\Omega + k_R\,\Psi_R + c\; e_R\cdot J e_\Omega . \label{eq6-V}
$$

Differentiate with [](#eq6-closed) and $\dot e_R = C_R\,e_\Omega$ from [](#eq3-eR-dot):

$$
\dot V = -k_\Omega\|e_\Omega\|^2 + c\;e_\Omega\cdot C_R\T J e_\Omega - c\,k_R\|e_R\|^2 - c\,k_\Omega\;e_R\cdot e_\Omega . \label{eq6-V-dot}
$$

There it is: $-c\,k_R\|e_R\|^2$, a term that is negative when the body is at rest and off target. The price is the second term, which has the wrong sign, and the fourth, which has no sign. To control them we need three facts. Write $J_{\min}$ and $J_{\max}$ for the smallest and largest eigenvalues of $J$.

1. $\|C_R\| \le 1$ in the matrix 2-norm ([](#thm3-so3-compat)), so $e_\Omega\cdot C_R\T J e_\Omega \le J_{\max}\|e_\Omega\|^2$.
2. $\|e_R\|^2 = \Psi_R\,(2 - \Psi_R)$, from [](#eq-error-angle). Hence $\tfrac12\|e_R\|^2 \le \Psi_R$ everywhere.
3. On the fenced set, $2 - \Psi_R \ge 2 - \Psi_{\max}$, so $\Psi_R \le \|e_R\|^2/(2 - \Psi_{\max})$.

Fact 3 is where half a turn enters the proof. Near half a turn the height $\Psi_R$ is large while the slope $\|e_R\|$ is nearly zero ([](#fig3-heights)). The slope controls the height only if you stay a fixed distance below the rim.

::: {.theorem #thm6-exp name="Exponential tracking on a sublevel set"}
Consider the closed loop [](#eq6-closed) and a start with $V_0(0) < 2k_R$. Let $\Psi_{\max} = V_0(0)/k_R$ and $z = (\|e_R\|,\ \|e_\Omega\|)$. Then $\Psi_R(t) \le \Psi_{\max}$ for all $t \ge 0$. For every $c$ with
$$
0 < c < \min\left\{ \frac{\sqrt{k_R\,J_{\min}}}{J_{\max}},\ \ \frac{4\,k_R\,k_\Omega}{k_\Omega^2 + 4\,k_R\,J_{\max}} \right\}, \label{eq6-c-bound}
$$
the function [](#eq6-V) satisfies $z\T W_1 z \le V \le z\T W_2 z$ and $\dot V \le -z\T W_3 z$ along the solution, with the positive definite matrices
$$
\begin{gathered}
W_1 = \tfrac12\begin{bmatrix} k_R & -c J_{\max} \\ -c J_{\max} & J_{\min}\end{bmatrix}, \qquad
W_2 = \tfrac12\begin{bmatrix} \tfrac{2k_R}{2 - \Psi_{\max}} & c J_{\max} \\ c J_{\max} & J_{\max}\end{bmatrix}, \\[6pt]
W_3 = \begin{bmatrix} c\,k_R & -\tfrac12 c\,k_\Omega \\ -\tfrac12 c\,k_\Omega & k_\Omega - c J_{\max}\end{bmatrix}.
\end{gathered} \label{eq6-W}
$$
Consequently $V(t) \le V(0)\,e^{-\gamma t}$ with $\gamma = \lambda_{\min}(W_3)/\lambda_{\max}(W_2)$, where $\lambda_{\min}$ and $\lambda_{\max}$ denote the smallest and largest eigenvalue of a symmetric matrix, and
$$
\|e_R(t)\|^2 + \|e_\Omega(t)\|^2 \le \frac{\lambda_{\max}(W_2)}{\lambda_{\min}(W_1)}\,\big(\|e_R(0)\|^2 + \|e_\Omega(0)\|^2\big)\,e^{-\gamma t}. \label{eq6-rate}
$$
:::

::: {.proof}
The fence follows from [](#eq6-V0-dot). The lower bound on $V$ uses fact 2 and $|e_R\cdot J e_\Omega| \le J_{\max}\|e_R\|\|e_\Omega\|$. The upper bound uses fact 3. The bound on $\dot V$ is [](#eq6-V-dot) with fact 1 and $|e_R\cdot e_\Omega| \le \|e_R\|\|e_\Omega\|$. $W_1$ is positive definite exactly when $c^2J_{\max}^2 < k_R J_{\min}$, and $W_3$ exactly when $c\,k_R(k_\Omega - cJ_{\max}) > \tfrac14c^2k_\Omega^2$. Those are the two entries of [](#eq6-c-bound). Then $\dot V \le -\lambda_{\min}(W_3)\|z\|^2 \le -\gamma V$.
:::

Each inequality in this proof was tested at 4000 random attitudes, velocity errors, full inertia matrices and gains. None failed, and [](#eq6-c-bound) is sharp: one tenth of a percent above either limit, $W_1$ or $W_3$ loses definiteness.

::: {.example #ex6-rate name="A floor, not a forecast"}
Start the vehicle 150° from a tumbling reference, with $e_\Omega(0) = (0.5, -0.3, 0.4)$ rad/s. Then $V_0(0) = 16.46$, just under the rim $2k_R = 17.62$, and $\Psi_{\max} = 1.869$. The limits in [](#eq6-c-bound) are 6.17 and 7.92. The best choice, $c = 0.46$, gives a guaranteed rate $\gamma = 0.034\ \text{s}^{-1}$.

Simulate ([](#fig6-lyapunov)). After a second, $V$ decays at $8.0\ \text{s}^{-1}$: twice the slowest pole, as the linearization predicts. The guarantee is more than 200 times slower than the truth.

That is not an error. The theorem must cover every start in the sublevel set, including those that graze the rim. Shrink the set and the guarantee improves: $\gamma = 0.26$ for $\Psi_{\max} = 1$ and $0.49$ for $\Psi_{\max} = 0.1$. Grow it and $\gamma$ falls in proportion to $2 - \Psi_{\max}$, reaching zero at the rim. A Lyapunov rate is a floor. Tune with the poles, and use the theorem to know where the floor exists.
:::

![The Lyapunov function of [](#ex6-rate) along a simulated run, on a logarithmic axis, with the decay the theorem guarantees. Right: the guaranteed rate against the size of the sublevel set. It vanishes as the set grows to include half a turn.](figures/ch06-lyapunov.svg){#fig6-lyapunov}

The theorem stops at the rim, $\Psi_R = 2$. What happens there, and to the starts above the fence that fly over it?

## Almost everywhere {#sec6-almost}

Find the places where the closed loop can sit still. In [](#eq6-closed), rest requires $e_\Omega = 0$ and $e_R = 0$. By [](#eq-error-angle), $e_R = \sin\theta\,a$ vanishes at $\theta = 0$, the target, and at $\theta = \pi$: every half turn, about every axis $a$. Those fill the whole boundary of the ball in [](#fig6-puzzle), a two-parameter family of equilibria.

::: {.thought #thought6-pencil name="Balancing on the rim"}
Sit the body exactly half a turn away about the axis $a$, at rest. Now nudge it by a small $\eta$. Near a half turn the error vector is $e_R \approx -(a\,a\T)\,\eta$: only the part of the nudge along $a$ counts, and the sign is reversed. Push further around the same axis and the torque pushes you on, away from the half turn. The same is true if you pull back. Either way you roll off, the long way or the short way, like a pencil on its point.

Nudge about an axis perpendicular to $a$ instead. You get a half turn about a slightly different axis. Still on the rim, still no torque. The rim is a ridge that is flat along itself and falls away on both sides.
:::

The linearization at a half turn confirms it: one eigenvalue with positive real part, two at zero, three stable. The unstable rate $\lambda$ solves $J_i\lambda^2 + k_\Omega\lambda - k_R = 0$ for a half turn about a principal axis, which gives $\lambda = 3.15$, $3.14$ and $2.99\ \text{s}^{-1}$ for our vehicle.

A family of equilibria is awkward to reason about. The weights of [](#eq3-weighted-psi) remove the awkwardness, as [](#ex3-weights) showed: with $D = \operatorname{diag}(d_1, d_2, d_3)$ and distinct entries, the slope [](#eq3-weighted-eR) vanishes at exactly four attitudes.

::: {.theorem #thm6-almost name="Almost-global tracking"}
Use the weighted error vector [](#eq3-weighted-eR) with $d_1 > d_2 > d_3 > 0$ in the law [](#eq6-law). Then the closed loop, written in the error state $(R_d\T R,\ e_\Omega)$, has exactly four equilibria: the target and the three half turns about the body axes. The target is asymptotically stable, and exponentially so near it. The three half turns are unstable, with one, two and three unstable directions. Every solution converges to one of the four, and the starts that converge to a half turn form a set of measure zero.
:::

::: {.proof}
The error system is autonomous, and $V_0$ with the weighted $\Psi_R$ still obeys [](#eq6-V0-dot), because $\dot\Psi_R = e_R\cdot e_\Omega$ holds for any weight ([](#thm3-so3-compat)). $\SO(3)$ is compact, so every sublevel set of $V_0$ is compact. LaSalle's invariance principle then sends every solution to the largest invariant set with $e_\Omega = 0$, which is the set of equilibria. There are four, they are isolated, so each solution converges to one of them. Near an equilibrium $e_R \approx \operatorname{Hess}\Psi_R\,\eta$, with $\operatorname{Hess}\Psi_R$ the diagonal Hessian of [](#exr3-weights), and the linearization is $J\ddot\eta + k_\Omega\dot\eta + k_R \operatorname{Hess}\Psi_R\,\eta = 0$. It has no eigenvalue on the imaginary axis, and as many unstable eigenvalues as the Hessian has negative entries: 0, 1, 2 and 3. The states that converge to an unstable equilibrium form its stable manifold, of dimension 5, 4 and 3 in a six-dimensional state space. Each has measure zero.
:::

This is what an almost-global result is ([](#sec-obstruction)), and no more. Three things it does not say.

It does not say the exceptional set is three points. It is three thin sheets through the state space, made of starts with just the right velocity to climb to a saddle and stop.

It gives no rate. The exponential bound belongs to [](#thm6-exp) and its sublevel set. Outside, convergence can be as slow as you like.

And with equal weights, the case of [](#thm6-exp) and of most flight code, the proof above does not apply as written, because the equilibria are not isolated. LaSalle still gives convergence to the target or to the rim, and each point of the rim still has its one unstable direction. Showing that the starts attracted to the rim form a thin set takes a longer argument, which we do not give. Lee, Leok and McClamroch (2010) state the equal-weight result in the form of [](#thm6-exp): exponential convergence from $\Psi_R(0) < 2$ with a bounded velocity error.

Now the experiment ([](#fig6-settle)). Start the vehicle at rest from 240 attitudes with random axes, from 5° up to 179.999° from the target. Every one of them converges. From 90° the error is inside 5° within 0.87 s, and from anything below 150° within 1.24 s. Beyond that the time climbs. About $e_1$ it is 2.34 s from 179°, 3.07 s from 179.9°, 3.80 s from 179.99°. Each factor of ten closer to the rim costs $\ln 10/\lambda = 0.73$ s, the same law that [](#ex3-linger) found on one axis, now with $\lambda$ set by the gains and the inertia. From exactly 180°, at rest, the body is still there after ten seconds.

![Time for the attitude error to fall inside 5° for good, against the initial error angle, for 240 simulated starts at rest. The trace error (indigo) slows as the start approaches half a turn, by 0.73 s per factor of ten. The same gains with the logarithm error (teal) do not slow.](figures/ch06-settle.svg){#fig6-settle}

So the slow region is wider than "measure zero" suggests. A start within a tenth of a degree of the rim is not exceptional, and it still waits three seconds.

## The same template on the sphere {#sec6-sphere}

A thrust axis, a cable and a pendulum rod are directions $q \in S^2$. [](#tbl3-summary) says to change column and repeat. Take the spherical pendulum with a force $F$ on the bob, [](#eq7-pendulum), and a reference $q_d(t)$ with $\dot q_d = \omega_d\times q_d$. The errors are $e_q = q_d\times q$ and $e_\omega = \omega - (I - q q\T)\,\omega_d$ of [](#eq3-e-omega). Differentiate the velocity error:

$$
\dot e_\omega = \dot\omega - (I - q q\T)\,\dot\omega_d + (q\cdot\omega_d)\,\dot q + (\dot q\cdot\omega_d)\,q . \label{eq6-eomega-dot}
$$

The last term points along $q$. It keeps $e_\omega$ tangent and does no work against tangent vectors. Ask for the rest to be a PD law:

$$
\dot\omega = (I - q q\T)\,\dot\omega_d - (q\cdot\omega_d)\,\dot q - k_q\,e_q - k_\omega\,e_\omega, \qquad
F = m g\,e_3 + m l\;\dot\omega\times q . \label{eq6-law-s2}
$$

The first equation is the angular acceleration we want. The second is the force that produces it in [](#eq7-pendulum). With $V_0 = \tfrac12\|e_\omega\|^2 + k_q\Psi_q$ and $\dot\Psi_q = e_q\cdot e_\omega$, the energy argument repeats word for word: $\dot V_0 = -k_\omega\|e_\omega\|^2$. So does the cross term, using [](#eq3-eq-dot) for $\dot e_q$, with one difference.

::: {.proposition #thm6-s2 name="Exponential tracking of a direction"}
Let $\|\omega_d(t)\| \le \omega_{\max}$. For the closed loop [](#eq6-law-s2), a start with $V_0(0) < 2k_q$ keeps $\Psi_q(t) \le \Psi_{\max} = V_0(0)/k_q < 2$. With $V = V_0 + c\,e_q\cdot e_\omega$, $z = (\|e_q\|, \|e_\omega\|)$ and
$$
0 < c < \min\left\{ \sqrt{k_q},\ \ \frac{4\,k_q\,k_\omega}{(k_\omega + \omega_{\max})^2 + 4\,k_q} \right\},
$$
the bounds of [](#thm6-exp) hold with $J_{\min} = J_{\max} = 1$, $k_R \to k_q$, and $k_\omega + \omega_{\max}$ in place of $k_\Omega$ in the off-diagonal entries of $W_3$.
:::

The difference is the term $(q\cdot\omega_d)\; q\times e_q$ in [](#eq3-eq-dot). So the error dynamics on the sphere are *not* autonomous: the reference's angular velocity appears in them, and a bound on it appears in the rate. The sphere is a homogeneous space and not a group ([](#sec2-sphere)), and this is one of the places where that shows. For a fixed target the dynamics are autonomous again. The only other equilibrium is the antipode $-q_d$. It is isolated, with two unstable directions, so the argument of [](#thm6-almost) applies directly.

[](#fig6-sphere) shows the law at work. The reference circles on a cone 35° from the vertical at 1.2 rad/s, and $k_q = 16$, $k_\omega = 5.6$. A start 60° away is inside 5° after 0.75 s. From 170° it takes 1.97 s, and from 179.9° it takes 4.16 s.

![Three simulated starts chase a reference direction that circles on a cone. The start 60° away closes in at once. The start 179.9° away sits near the antipode of the reference, which moves too, before it falls toward it.](figures/ch06-sphere-strip.png){#fig6-sphere anim="figures/ch06-sphere.gif"}

A direction with a force on it is fully actuated, as the rigid body was. Most flying machines are not.

## Cascades: when the force must be aimed {#sec6-cascade}

The quadrotor of [](#eq4-quadrotor) has four inputs for six degrees of freedom:

$$
m\ddot x = f R e_3 - m g\,e_3, \qquad J\dot\Omega + \Omega\times J\Omega = \tau .
$$

To follow a position reference $x_d(t)$ we would like to apply the force

$$
F_d = -k_x\,e_x - k_v\,e_v + m g\,e_3 + m\ddot x_d, \qquad e_x = x - x_d, \quad e_v = v - \dot x_d, \label{eq6-Fd}
$$

a PD law with gravity and feedforward: the desired force. The vehicle can only push along $R e_3$. So do the next best thing, in two parts. Apply the component of $F_d$ that the vehicle can produce now, and turn the vehicle until it can produce all of it:

$$
\begin{aligned}
f &= F_d\cdot R e_3, & R_d e_3 &= \frac{F_d}{\|F_d\|}, \\[4pt]
R_d e_2 &= \frac{R_d e_3\times R_z(\psi_d)\,e_1}{\|R_d e_3\times R_z(\psi_d)\,e_1\|}, & R_d e_1 &= R_d e_2\times R_d e_3 .
\end{aligned} \label{eq6-Rd}
$$

The third column of the attitude reference is the direction of the desired force. The desired **heading** $\psi_d$ fixes the rotation about it: the first body axis is kept as close to the heading direction $R_z(\psi_d)e_1$ as the tilt allows. The heading is not the yaw of [](#eq-zyx), though the two agree when the roll or the pitch is zero ([](#sec10-quad)). The torque is [](#eq6-law), tracking this $R_d$. Position loop, then direction, then attitude: a **cascade** ([](#fig6-cascade)).

What does the position error see? Substitute:

$$
m\,\dot e_v = -k_x e_x - k_v e_v + e_F, \qquad e_F = f R e_3 - F_d, \qquad \|e_F\| = \|F_d\|\,\sin\theta_{\text{tilt}} . \label{eq6-Delta}
$$

The bead on the wire again, driven by a disturbance, the **force error** $e_F$: the part of $F_d$ the vehicle is not yet pointing along. Its size is set by the tilt angle $\theta_{\text{tilt}}$ of [](#sec3-errors), between $R e_3$ and $R_d e_3$. Since the tilt never exceeds the total error angle, $\|e_F\| \le \|F_d\|\,\|e_R\|$ whenever $\Psi_R < 1$.

::: {.proposition #thm6-cascade name="The quadrotor cascade"}
Apply [](#eq6-Rd) and [](#eq6-law), with $\Omega_d$ and $\dot\Omega_d$ the exact derivatives of $R_d$ along the closed loop. Suppose the reference acceleration is bounded and $F_d$ does not vanish along the solution. Then the attitude error obeys [](#eq6-closed), unaffected by the position loop. If moreover $V_0(0) < 2k_R$, the errors $e_R$, $e_\Omega$, $e_x$ and $e_v$ all converge to zero exponentially.
:::

::: {.proof}
$R_d$ is differentiable where $F_d \ne 0$, so [](#eq6-closed) holds, and [](#thm6-exp) gives $\|e_R(t)\| \to 0$ exponentially, with $\Psi_R < 1$ after a finite time. In [](#eq6-Delta), $\|F_d\| \le k_x\|e_x\| + k_v\|e_v\| + \|m g\,e_3 + m\ddot x_d\|$, and $\sin\theta_{\text{tilt}}$ is at most one always and at most $\|e_R\|$ eventually. So the stable linear system for $(e_x, e_v)$ is driven by a term bounded by an exponentially decaying factor times an affine function of its own state. Gronwall's inequality keeps the state bounded and then makes it decay.
:::

The cascade is triangular: attitude drives position and nothing comes back. That is why no condition on the gains appears. Lee, Leok and McClamroch (2010) prove more with more work. They treat the position and attitude errors together, and state exponential stability of the complete dynamics when the initial attitude error is less than 90°, that is $\Psi_R(0) < 1$. When it is less than 180°, $\Psi_R(0) < 2$, they state a weaker property, almost-global exponential attractiveness: the attitude error first decays into the region of the first result, which then takes over. Both results come with conditions on the gains and on the other initial errors. Consult the paper for their exact form.

::: {.example #ex6-inverted name="Thrown upside down"}
The reference is a circle of radius 1.5 m flown at 1 rad/s with a vertical wobble, heading along the tangent. Position gains are $k_x = 16m$ and $k_v = 5.6m$. Release the vehicle at rest, 0.88 m from the reference, rolled 178° ([](#fig6-cascade)). Then $\Psi_R(0) = 1.55$. For the first 0.20 s the desired thrust $F_d\cdot R e_3$ is negative. Rotors cannot pull, so we clip it at zero and the vehicle falls while it turns. It is right side up after 0.30 s. It drops 0.94 m below the reference, the position error peaks at 1.55 m, and it is back within 5 cm after 1.94 s. The peak thrust is 144 N, more than three times the hover thrust of 42.6 N, and the peak torque is 18 N·m.
:::

![The run of [](#ex6-inverted). The vehicle starts inverted beside the reference circle, cuts its thrust, rights itself while it falls, then climbs back and locks on. The short arrow is the thrust axis $R e_3$. The long one is the force $F_d$ that the position loop asks for.](figures/ch06-cascade-strip.png){#fig6-cascade anim="figures/ch06-cascade.gif"}

Those thrust and torque peaks are a warning about the word "exact", and so is this.

::: {.warning #warn6-feedforward name="The feedforward is doing the separating"}
[](#thm6-cascade) needs $\Omega_d$ and $\dot\Omega_d$. They are derivatives of $R_d$, so of $F_d$, so they contain the vehicle's acceleration and jerk. Most flight code sets them to zero, or to the values along the reference $x_d(t)$, and counts on the attitude loop being fast. The values along the reference come from the flat map of [](#thm10-quad), and [](#sec10-feedforward) tries them on the experiment below: they remove the steady error, and they do not remove the need for a fast attitude loop.

Check whether it is. The position loop has natural frequency 4 rad/s. The slowest attitude pole is at 3.98 s⁻¹. These are Lee's published gains, and with exact feedforward they track the circle to $3\times10^{-11}$ m. Set $\Omega_d = \dot\Omega_d = 0$ and the same gains *diverge*: from a 5 cm offset the error passes 5 m after 14.6 s. Multiply the attitude gains by $(\kappa^2, \kappa)$, which speeds the attitude loop up by $\kappa$. With $\kappa = 2$ the loop is stable and the tracking error settles at 1.4 cm. With $\kappa = 4$ it is 0.7 cm.

Conversely, with exact feedforward and the attitude loop made three times *slower* than the position loop, the errors still go to zero. A triangular cascade does not care about time scales. An approximate one does. If you drop the feedforward, the stability you rely on is no longer [](#thm6-cascade). It is a singular-perturbation argument, which Wu (2017) makes for these systems: the attitude gains are scaled exactly as by our $\kappa$, and stability is proved for $\kappa$ above some threshold that the proof does not compute.
:::

**One more layer.** Hang a load from the quadrotor. In [](#eq4-load), torque turns the body, the body aims the thrust, the thrust swings the cable, and the cable pulls the load. The controller of Sreenath, Lee and Kumar (2013) climbs that ladder from the bottom. Let $F_L = -k_x e_x - k_v e_v + (m_Q + m_L)(\ddot x_d + g\,e_3)$ be the force we would like on the load-and-quadrotor pair, with $e_x = x_L - x_d$ and $e_v = v_L - \dot x_d$. The cable should point against it, allowing for the centripetal term of [](#eq4-load):

$$
\begin{aligned}
q_d &= -\frac{F_L + m_Q l\,\|\dot q\|^2 q}{\|F_L + m_Q l\,\|\dot q\|^2 q\|}, \\[4pt]
f R e_3 \;&\leftarrow\; \big(F_L\cdot q + m_Q l\,\|\dot q\|^2\big)\,q \;+\; m_Q l\; q\times\big({-k_q}\,e_q - k_\omega\,e_\omega\big).
\end{aligned} \label{eq6-load}
$$

The arrow means "ask the quadrotor for this thrust vector", by the construction [](#eq6-Rd). The part along $q$ passes straight through the cable: substituted into [](#eq4-load) it gives $(m_Q + m_L)(\ddot x_L + g\,e_3) = (q\,q\T)F_L$, the desired force projected on the cable. The part across $q$ is the sphere law [](#eq6-law-s2), turning the cable toward $q_d$: it produces $\dot\omega = -k_q e_q - k_\omega e_\omega$ exactly. Three loops, three pairs of gains, and each loop treats the one inside it as already converged.

Kotaru (2022) reports that on hardware the cable gains $k_q$, $k_\omega$ decide whether the loop is stable, and the inner loop is simply run fast. A simulation shows how little margin there is. Take $m_L = 0.5$ kg on a 1 m cable, a position loop at 2 rad/s and a cable loop at 5 rad/s, no feedforward, and move the load 1.6 m from a start with the cable swung 40°. With the bare vehicle's attitude gains, slowest pole 4.0 s⁻¹, the system diverges. With $\kappa = 2$ it converges: the load is within 5 cm after 5.3 s, the swing never exceeds 44°, and the tension stays above 4.2 N, so the cable stays taut ([](#warn4-slack)). Sreenath, Lee and Kumar (2013) state stability results for the three loops in turn: quadrotor attitude, cable direction and load position. Wu (2017) and Kotaru (2022) justify stacking them by the same time-scale argument.

## The same law as feedback linearization {#sec6-fl}

Look again at how [](#eq6-law) was found. We took an error, differentiated until the input appeared, canceled everything else, and imposed linear dynamics. That is input-output feedback linearization, with outputs of **relative degree** two: the input appears after two derivatives. Siravuru (2019, ch. 4) presents the geometric law this way, for the 3D pendulum of [](#eq4-pendulum), where gravity joins the terms to cancel:

$$
\tau = J\big({-k_p}\,e_R - k_d\,e_\Omega\big) + \Omega\times J\Omega - m g\,(R\T e_3)\times\rho - J\big(\Omega\times R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d\big),
$$

so that $\dot e_\Omega = -k_p e_R - k_d e_\Omega$, with the scalar gains $k_p$, $k_d$ of [](#eq-quat-pd). This is [](#eq6-closed) with gains scaled by the inertia, and both theorems apply with $J$ replaced by $I$.

One point deserves care. The output is $e_R$, and $\dot e_R = C_R\,e_\Omega$ by [](#eq3-eR-dot). Linearizing the output exactly, to $\ddot e_R + k_d\dot e_R + k_p e_R = 0$, needs $C_R^{-1}$. But $\det C_R = \tfrac12\cos\theta\,(1 + \cos\theta)$, which vanishes at 90° and at 180° of error. Exact output linearization of $e_R$ is a local method. The law above linearizes the velocity error instead and leaves $C_R$ in the loop, where the proof only needs $\|C_R\| \le 1$. The result is not a linear system in $e_R$. It is a system with a Lyapunov function, which is better.

The Euler-angle version is the textbook one: output $\Theta - \Theta_d$, and $\ddot\Theta = -k_p\Theta - k_d\dot\Theta$ imposed through $\Omega = E\,\dot\Theta$ of [](#eq-euler-jacobian). Siravuru compares the two over a grid of initial attitudes. We repeat the comparison on one plant with his parameters: $m = 1$ kg, $J = \operatorname{diag}(0.1625, 0.1625, 0.01)$ kg·m², $k_p = 100$ s⁻², $k_d = 20$ s⁻¹, target hanging at rest. The grid is every combination of yaw, pitch and roll in steps of 45°: 729 labels, which name only 208 distinct attitudes. The label (180°, 180°, 180°) is the target itself.

![The 208 initial attitudes of the grid, drawn in the ball of rotations. From the 16 marked in terracotta, at 90° of pitch, the Euler-angle law is undefined. From the 29 hollow ones, exact half turns, the geometric law applies no torque. Each half turn appears twice, at two opposite points of the boundary. From the rest both converge.](figures/ch06-grid.svg){#fig6-grid}

::: {.table #tbl6-grid}

| From rest, 208 attitudes | Euler-angle law | Geometric law |
| :-- | :-: | :-: |
| Law undefined | 16 | 0 |
| Converges | 192 | 179 |
| Stays put | 0 | 29 |
| Peak torque, median | 28.7 N·m | 7.5 N·m |
| Peak torque, largest | 88.7 N·m | 11.3 N·m |
| Time to 5°, median | 0.52 s | 0.68 s |
| Time to 5°, longest | 0.59 s | 0.97 s |

: Feedback linearization of a 3D pendulum, two ways, simulated on the same plant. Torque and time are over the starts from which both laws converge.

:::

The Euler law cannot start from 16 attitudes, where $E$ has no inverse. Near them it is worse than undefined. With a body rate of $(1, -1, 1)$ rad/s at 89.9° of pitch it asks for 69 N·m where the geometric law asks for 9.4. At 89.99° it asks for 830, and the geometric law still for 9.4. Everywhere else it converges, and quickly, but at nearly four times the peak torque and 4.3 times the effort $\int\|\tau\|^2\dd t$ of the geometric law.

The geometric law fails from 29 attitudes, all of them exact half turns at rest. That is the measure-zero set of [](#thm6-almost), and a grid of round numbers lands on it 29 times. Give every start the body rate $(1, -1, 1)$ rad/s and 205 converge. The three that do not are half turns about an axis perpendicular to that rate: the motion slides them along the rim to another half turn. Add 0.001 rad/s along the axis and they leave too. "Measure zero" is a statement about volume. It is not a promise that your test matrix avoids it.

And part of the geometric law's gentleness is not geometry at all. $\|e_R\|$ never exceeds 1, while $\|\Theta\|$ reaches 4.71 on this grid. Far from the target the geometric loop simply pushes less, and pays for it in time.

## More factors: legs {#sec6-legs}

A biped is a torso and two legs, each with an attitude. Siravuru (2019, ch. 5) models the Reaction Mass Biped on a product of three copies of $\SO(3)$ and a few lengths ([](#ex4-leg) is its stance leg), and tracks walking and turning gaits with exactly the template of this chapter. Each body gets its own $\Psi_R$, $e_R$ and $e_\Omega$ ([](#tbl3-summary)). The torques cancel the couplings between bodies and impose a PD law on each factor. The Lyapunov function is the sum of the three copies of [](#eq6-V0), and its derivative is the sum of three copies of [](#eq6-V0-dot). The stated result is almost-global asymptotic tracking, proved by the invariance argument of [](#thm6-almost) with a weighted error function on each factor. No exponential rate is claimed. The same thesis models Cassie on a product of rotation groups to plan and track turns in place.

Should a walking robot care about convergence from 170°? The thesis itself says: not much. Large-angle recoveries are rare in walking. A step lasts a fraction of a second, the joint torques are bounded, and the next impact resets the velocities.

A number makes the point. Take a swing leg with $J = \operatorname{diag}(0.9, 1.1, 0.4)$ kg·m², $k_R = 60$ and $k_\Omega = 12$, and a step of 0.35 s. A 10° error shrinks to 1.4° before touchdown. A 60° error still has 11° left. A 150° error has 100° left, and the robot has fallen. What legs gain from the geometric law is what [](#tbl6-grid) shows in its torque rows and what [](#thought6-forgot) explains: no singular attitude anywhere along a turn, error dynamics that do not depend on where in the gait the robot is, and a Lyapunov function to build on.

## What you are choosing {#sec6-limits}

**The error vector.** Everything slow in this chapter traces back to $\|e_R\| = \sin\theta$ ([](#warn-sine)). Replace $e_R$ by the logarithm error $\log(R_d\T R)^\vee$ of [](#sec3-errors) and keep the gains. Near the target nothing changes. From 150° our vehicle settles in 0.88 s where the trace error needs 1.25 s, and near the rim the time no longer grows ([](#fig6-settle)). Mueller (2018) compares four such laws that agree to first order, and adds tilt priority for multicopters, for which [](#eq6-Delta) is the argument: position cares about the tilt alone. The cost is at the cut locus. At 180° the logarithm law is discontinuous, its torque is at its largest (27.5 N·m at 179° for our gains, against a maximum of 8.8 for the trace error), and noise there can make it chatter. A hybrid switch with hysteresis repairs that (Mayhew, Sanfelice and Teel, 2011). Teng et al. (2022) build a quadratic Lyapunov function on the logarithm and show that it keeps its convergence rate out to large angles. None of these laws escapes [](#thm-obstruction). They move the bad set or make the law discontinuous on it.

**Saturation.** Both theorems assume the torque [](#eq6-law) is delivered. [](#ex6-inverted) asked for three times the hover thrust. With the PD part clipped at 1 N·m per axis, the recovery from 150° takes 1.27 s, hardly changed from 1.25 s. Clipped at 0.2 N·m it takes 2.99 s. No theorem here says it must converge. A controller that knows the bound in advance can plan around it ([](#ex13-flip)).

**Gains.** [](#thm6-exp) holds for any positive $k_R$ and $k_\Omega$ and tells you nothing about which to pick. The LQR of [](#sec7-lqr) does: at a hover its gain has exactly the form of [](#eq6-law), a multiple of $e_R$ and a per-axis multiple of $e_\Omega$ ([](#ex7-hold)), and along a maneuver it varies with time. Use the linearization to choose gains and this chapter to know what happens far away ([](#sec7-local)).

**The model.** Every cancellation here assumed that $J$, $m$ and the forces are known. They are not. [](#sec8-uncertainty) shows what a wrong inertia or a gust does to the error dynamics [](#eq6-closed), and [](#sec8-l1) keeps the tracking error small in spite of them. Keeping the vehicle out of places it must not go is the subject of [](#sec8-barrier). When the reference itself must be chosen subject to constraints, the tool is the direct transcription of [](#sec13-plan).

## Exercises {#sec6-exercises .unnumbered}

::: {.exercise #exr6-eomega}
Derive [](#eq6-eOmega-dot) from $e_\Omega = \Omega - R\T R_d\,\Omega_d$. Then substitute [](#eq6-law) and obtain [](#eq6-closed). Where exactly did the term $\Omega\times R\T R_d\,\Omega_d$ come from, and what would the closed loop be if you left it out of the torque?
:::

::: {.exercise #exr6-cross}
Show that no bound $\dot V_0 \le -\gamma V_0$ with $\gamma > 0$ can hold for [](#eq6-V0). Then prove the three matrix inequalities of [](#thm6-exp), and show that [](#eq6-c-bound) is exactly the condition for $W_1$ and $W_3$ to be positive definite.
:::

::: {.exercise #exr6-rate}
In the notebook, plot the guaranteed rate $\gamma$ against $c$ for $\Psi_{\max} = 1$. Then replace the cross term by $c\,e_R\cdot e_\Omega$, without $J$, and redo the bounds. Is the guarantee better or worse for our vehicle?
:::

::: {.exercise #exr6-rim}
Show that $e_R \approx -(a\,a\T)\,\eta$ near a half turn about $a$. Write the linearization of [](#eq6-closed) there and prove that it has exactly one unstable eigenvalue, the positive root of $1 = k_R\; a\cdot(\lambda^2 J + k_\Omega\lambda I)^{-1}a$. For which axis $a$ does our vehicle leave the rim fastest?
:::

::: {.exercise #exr6-sphere}
Derive [](#eq6-eomega-dot) and the closed loop of [](#eq6-law-s2). Prove [](#thm6-s2). Then find a reference $q_d(t)$ and a start for which the term $(q\cdot\omega_d)\;q\times e_q$ is as large as it can be, and check in simulation whether convergence is slower.
:::

::: {.exercise #exr6-feedforward}
Reproduce [](#warn6-feedforward) in the notebook. Find, to one decimal place, the smallest speed-up $\kappa$ for which the loop without feedforward is stable on the circle. Then replace the zero feedforward by the $\Omega_d$ and $\dot\Omega_d$ of the reference motion, computed from $x_d(t)$ alone by [](#eq10-quad-Omega) to [](#eq10-quad-Omega-dot), and find $\kappa$ again. Do not expect the threshold to drop to one: at Lee's gains this feedforward still diverges ([](#fig10-feedforward)). [](#exr10-kappa) continues this exercise.
:::

::: {.exercise #exr6-grid}
Rerun the grid of [](#tbl6-grid) with the logarithm error in the geometric law. Which rows change? Then shift the whole grid by 1° in each angle and rerun both laws. Explain every entry that moved.
:::

::: {.exercise #exr6-load}
For the load cascade [](#eq6-load), show that the across-cable part of the thrust vector produces $\dot\omega = -k_q e_q - k_\omega e_\omega$ and that the along-cable part delivers $(q\,q\T)F_L$ to the load. In the notebook, lower $k_q$ until the loop fails, and compare the three loop rates at that point.
:::

## Notes and sources {#sec6-notes .unnumbered}

The attitude law [](#eq6-law), the Lyapunov function with a cross term, the construction [](#eq6-Rd) of the attitude reference and the cascade analysis are due to Lee, Leok and McClamroch (2010). Their bounds use a cross term without the inertia. Ours, [](#eq6-c-bound) and [](#eq6-W), are derived for the form [](#eq6-V) and checked numerically. Error functions with four isolated critical points go back to Koditschek (1989), and Chaturvedi, Sanyal and McClamroch (2011) survey the almost-global stabilization of attitude built on them. Bullo and Lewis (2005) give the general theory of tracking with a compatible transport map. The sphere law and its Lyapunov function follow Wu (2017) and Sreenath, Lee and Kumar (2013), who also designed the load cascade. Kotaru (2022) describes its implementation and gain tuning on hardware. The singular-perturbation view of cascades with scaled attitude gains is from Wu (2017). The feedback-linearization reading, the grid comparison with Euler angles, the Reaction Mass Biped and the remarks on what large-angle guarantees are worth for legs are from Siravuru (2019, ch. 4 to 6). The comparison of attitude error vectors and the lingering near half a turn are from Mueller (2018), and the convergence of the logarithm error from Teng et al. (2022). The invariance principle, stable manifolds and Gronwall's inequality are in Khalil (2002).

- S. P. Bhat and D. S. Bernstein, ["A topological obstruction to continuous global stabilization of rotational motion and the unwinding phenomenon"](https://doi.org/10.1016/S0167-6911%2899%2900090-0), *Systems & Control Letters* 39(1), 2000.
- F. Bullo and A. D. Lewis, [*Geometric Control of Mechanical Systems*](https://doi.org/10.1007/978-1-4899-7276-7), Springer, 2005.
- N. A. Chaturvedi, A. K. Sanyal and N. H. McClamroch, ["Rigid-body attitude control"](https://doi.org/10.1109/MCS.2011.940459), *IEEE Control Systems Magazine* 31(3), 2011.
- H. K. Khalil, [*Nonlinear Systems*](https://www.egr.msu.edu/~khalil/NonlinearSystems/), 3rd ed., Prentice Hall, 2002.
- D. E. Koditschek, ["The application of total energy as a Lyapunov function for mechanical control systems"](https://repository.upenn.edu/ese_papers/672/), *Contemporary Mathematics* 97, 1989.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- C. G. Mayhew, R. G. Sanfelice and A. R. Teel, ["Quaternion-based hybrid control for robust global attitude tracking"](https://doi.org/10.1109/TAC.2011.2108490), *IEEE Transactions on Automatic Control* 56(11), 2011.
- M. W. Mueller, ["Multicopter attitude control for recovery from large disturbances"](https://arxiv.org/abs/1802.09143), arXiv:1802.09143, 2018.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- K. Sreenath, T. Lee and V. Kumar, ["Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load"](https://doi.org/10.1109/CDC.2013.6760219), *IEEE Conference on Decision and Control*, 2013.
- S. Teng, W. Clark, A. Bloch, R. Vasudevan and M. Ghaffari, ["Lie algebraic cost function design for control on Lie groups"](https://arxiv.org/abs/2204.09177), arXiv:2204.09177, 2022.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
