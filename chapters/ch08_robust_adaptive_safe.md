---
title: Robust, adaptive and safety-critical control
short: Robust and safe control
id: ch-robust
summary: A tracking controller assumes a perfect model and knows nothing of walls. An adaptive loop on the manifold supplies the first, a barrier function the second.
notebook: notebooks/ch08_adaptive_safety.ipynb
verify: verify/ch08.py
---

A quadrotor hovers under a geometric tracking controller. Its attitude error is zero to the last digit. Now clip a 0.5 kg camera to it, 8 cm off-center. The controller measures the attitude a thousand times a second and corrects whatever it sees. Where does the attitude error settle?

Most people say zero: that is what feedback is for. The answer is 11.3°, and it stays there for as long as you care to wait ([](#fig8-puzzle), left). Nothing is broken. The gains are fine and the code is correct. The controller was derived for a vehicle whose inertia is known and on which nothing pushes, and you have changed the vehicle.

Here is a second question for the same controller. Its reference passes through a wall. What does it do? It flies into the wall, accurately. Nowhere in a tracking law is there a symbol for the wall.

![Two things a tracking controller does not have. Left: with a camera clipped on off-center, the simulated vehicle comes to rest with its thrust axis $R\,e_3$ tilted 11.3° from the target $R_d\,e_3$. The dashed outline is the attitude it was asked to hold. Right: a reference that crosses a wall is tracked across the wall.](figures/ch08-puzzle.svg){#fig8-puzzle}

So two things are missing. One is a way to handle what the controller does not know. The other is a way to guarantee what must never happen. They are different kinds of requirement and they get different tools. Both tools are built from the error functions and transport maps of [](#ch-variations), so neither needs a coordinate.

## What the tracker assumes {#sec8-baseline}

The attitude controller of [](#ch-tracking) is a proportional and derivative law on $\SO(3)$ plus a feedforward term. With $e_R$ and $e_\Omega$ from [](#eq-error-so3), it reads

$$
\tau = \underbrace{-k_R\,e_R - k_\Omega\,e_\Omega}_{\mu} \;+\; \Omega\times J_0\Omega - J_0\big(\Omega^\wedge R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d\big). \label{eq8-pd}
$$

We write $J_0$ for the inertia the controller believes, and $\mu$ for the part of the torque that is left to design. The feedforward terms cancel the gyroscopic torque and supply the acceleration of the reference. Substitute [](#eq8-pd) into $J\dot\Omega + \Omega\times J\Omega = \tau$ of [](#eq4-quadrotor). If $J_0 = J$, everything cancels except $\mu$:

$$
J_0\,\dot e_\Omega = -k_R\,e_R - k_\Omega\,e_\Omega , \qquad \dot e_R = C_R\,e_\Omega .
$$

This is a spring and a damper on the manifold. The function $V = \tfrac12\,e_\Omega\cdot J_0\,e_\Omega + k_R\,\Psi_R + c\,e_R\cdot e_\Omega$, with a small constant $c > 0$, decreases along every solution that starts less than half a turn from the reference and not too fast, and the errors go to zero exponentially (Lee, Leok and McClamroch, 2010). Our vehicle has $J_0 = \operatorname{diag}(6.97,\ 6.21,\ 10.3)\times10^{-3}$ kg·m², $k_R = 2$ N·m and $k_\Omega = 0.25$ N·m·s.

Every step of that argument used $J_0 = J$ and a body on which only $\tau$ acts.

## What you do not know {#sec8-uncertainty}

Let the true inertia be $J \ne J_0$, and let an unknown torque $\Delta_e$ act in the body frame:

$$
J\dot\Omega + \Omega\times J\Omega = \tau + \Delta_e .
$$

Apply the same control law [](#eq8-pd), which still uses $J_0$. The cancellation is no longer exact. Collect what is left over and the error dynamics become

$$
J_0\,\dot e_\Omega = \mu + \Delta, \qquad
\Delta = \big(J_0 J^{-1} - I\big)\mu + J_0 J^{-1}\big(\Delta_m + \Delta_e\big), \label{eq8-lumped}
$$

where $\Delta_m = -\Omega\times(J - J_0)\Omega + (J - J_0)\big(\Omega^\wedge R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d\big)$ is the part of the feedforward that used the wrong inertia.

Three different mistakes, a wrong inertia, a wrong feedforward and an outside torque, arrive as one vector $\Delta$, the **lumped uncertainty**. It enters the equation at the same place as the input $\mu$. An uncertainty with that property is called **matched**: if you knew $\Delta$ you could cancel it with torque. The kinematic equation $\dot e_R = C_R\,e_\Omega$ is untouched, because no model error can change how rotations compose. The price is that $\Delta$ is not a constant. It depends on the state and, through $\mu$, on the control itself.

::: {.thought #thought8-spring name="A door on a spring"}
A door is held shut by a spring. Lean on it with a steady hand. The door opens until the spring pushes back as hard as you do, and then it stays open by that much. It cannot return to shut, because a shut door means an unstretched spring and no force to oppose you.

The term $-k_R\,e_R$ is that spring. A steady torque is balanced only by a steady error. Now recall what is odd about this spring. Its force is $k_R\|e_R\| = k_R\sin\theta$ for an error angle $\theta$ ([](#warn-sine)). It grows up to a quarter turn of error and then fades. Lean harder than $k_R$ and no angle balances you. The door swings all the way round.
:::

::: {.example #ex8-camera name="The camera"}
The camera has mass $0.5$ kg and sits at $(0.08,\ 0,\ -0.03)$ m in the body frame. In hover the thrust equals the total weight and no longer passes through the center of mass, which gives a constant body torque $\Delta_e = (0,\ 0.392,\ 0)$ N·m. The camera also raises the three principal inertias by 6%, 59% and 31%.

At rest $\mu + \Delta_e = 0$, so $k_R\,e_R = \Delta_e$ and by [](#eq-error-angle)

$$
\sin\theta = \frac{\|\Delta_e\|}{k_R} = \frac{0.392}{2}, \qquad \theta = 11.3^\circ .
$$

The simulation settles at 11.31°, about the axis of $\Delta_e$. The inertia error plays no part at rest. A vehicle tilted 11.3° accelerates sideways at $g\tan\theta = 1.96$ m/s², so the position loop inherits the problem.

Now make the torque 5.5 times larger, 2.16 N·m, which is more than $k_R$. No equilibrium exists. The simulated body tumbles, and its error passes 179.8°.
:::

A fixed offset can be removed by an integrator. But the torques that matter are not fixed. Thrust changes with every maneuver, gusts come and go, and $\Delta$ depends on the motion. Before reaching for a tool, ask what the plain controller already guarantees.

## Robustness you get for free {#sec8-free}

The Lyapunov function that proved convergence also bounds the damage. Write $z = (\|e_R\|, \|e_\Omega\|)$, and let $\lambda_{\min}$ and $\lambda_{\max}$ denote the smallest and largest eigenvalues of a symmetric matrix, with $\lambda_m = \lambda_{\min}(J_0)$ and $\lambda_M = \lambda_{\max}(J_0)$.

::: {.proposition #thm8-bound name="Ultimate bound for the plain controller"}
Consider $J_0\,\dot e_\Omega = -k_R\,e_R - k_\Omega\,e_\Omega + \Delta$ with $\dot e_R = C_R\,e_\Omega$. Choose $c > 0$ so small that
$$
W = \begin{bmatrix} \dfrac{c\,k_R}{\lambda_M} & -\dfrac{c\,k_\Omega}{2\lambda_m} \\[6pt] -\dfrac{c\,k_\Omega}{2\lambda_m} & k_\Omega - c \end{bmatrix}, \qquad
W_1 = \frac12\begin{bmatrix} k_R & -c \\ -c & \lambda_m \end{bmatrix}
$$
are positive definite, and let $W_2 = \tfrac12\begin{bmatrix} 2k_R & c \\ c & \lambda_M \end{bmatrix}$. Along any solution with $\Psi_R \le 1$ and $\|\Delta\| \le \Delta_{\max}$,
$$
z\T W_1 z \le V \le z\T W_2 z, \qquad
\dot V \le -z\T W z + \sqrt{1 + c^2/\lambda_m^2}\;\Delta_{\max}\,\|z\| , \label{eq8-bound}
$$
and therefore
$$
\limsup_{t\to\infty}\|z(t)\| \;\le\; \sqrt{\frac{\lambda_{\max}(W_2)}{\lambda_{\min}(W_1)}}\;\frac{\sqrt{1 + c^2/\lambda_m^2}}{\lambda_{\min}(W)}\;\Delta_{\max} .
$$
:::

::: {.proof}
Differentiate $V$ with $\dot\Psi_R = e_R\cdot e_\Omega$ and [](#eq3-eR-dot). The terms $\pm k_R\,e_R\cdot e_\Omega$ cancel. Bound what remains with $e_\Omega\cdot C_R\,e_\Omega \le \|e_\Omega\|^2$ and the eigenvalues of $J_0$. The term in $\Delta$ is $(e_\Omega + c\,J_0^{-1}e_R)\cdot\Delta$. The bounds on $V$ use $\tfrac12\|e_R\|^2 \le \Psi_R \le \|e_R\|^2$, valid while $\Psi_R \le 1$. Then $\dot V < 0$ whenever $\|z\|$ exceeds $\sqrt{1 + c^2/\lambda_m^2}\,\Delta_{\max}/\lambda_{\min}(W)$, so $V$ ends up below its largest value on that ball. This is the argument of Lee, Leok and McClamroch (2010) with one more term.
:::

So a bounded uncertainty gives a bounded error. Nothing diverges, as long as the solution stays within a quarter turn of the reference. That is real robustness and it cost nothing.

Now put in numbers. For our vehicle the best choice of $c$ gives a constant of 112 per N·m. For a gust of 0.030 N·m the bound is $\|z\| \le 3.36$. The simulated error for that gust is $0.033$, a hundred times smaller. Since $\|e_R\|$ never exceeds 1, the bound says something only for torques below 0.009 N·m. The camera's torque is 44 times that.

Read the proposition for what it is: a statement that small uncertainties do small harm, with a constant that no one should use for design. It does not make the error small. It does not remove the offset. And [](#ex8-camera) showed that past $k_R$ there is nothing to bound. To do better the controller must learn $\Delta$.

## A co-pilot who learns {#sec8-l1}

::: {.thought #thought8-copilot name="A nervous co-pilot and a calm hand"}
You fly with a co-pilot who has a perfect simulator of the aircraft you believe you have. He flies the simulator alongside the real thing, with the same commands. Whenever the two disagree, he concludes that some unknown torque is acting, and updates his guess. He is quick, and nervous. His guess jumps with every bump and every flicker of the gyroscope.

If his guess went straight to the controls, the motors would shake. So his corrections pass through a calm hand that follows the guess smoothly and ignores its jitter. Here is the point of the arrangement. How fast he learns and how hard the motors are worked are now set by two different people. He can be as quick as you like. Only the calm hand touches the vehicle.
:::

That is the architecture of **L1 adaptive control** (Hovakimyan and Cao, 2010), named for the $\mathcal{L}_1$ norm in which its guarantees are stated. Kotaru (2022, ch. 3) built it on $\SO(3)$. It has four parts.

**The state predictor.** The simulator is a copy of the believed dynamics, with its own attitude $R_p \in \SO(3)$ and body angular velocity $\Omega_p$:

$$
\dot R_p = R_p\,\Omega_p^\wedge, \qquad J_0\,\dot\Omega_p + \Omega_p\times J_0\Omega_p = \tau_p .
$$

It lives on the manifold too, and it is driven by a torque $\tau_p$ of the form [](#eq8-pd), evaluated at $(R_p, \Omega_p)$ with its own $\mu_p$. So its tracking errors $e_R^p$, $e_\Omega^p$, which are [](#eq-error-so3) with $(R_p, \Omega_p)$ in place of $(R, \Omega)$, obey $J_0\,\dot e_\Omega^p = \mu_p$ exactly. The $\mathcal{L}_1$ literature and Kotaru call it the reference model. We keep "reference" for $R_d$.

**The prediction error.** How far is the predictor from the vehicle? They are two attitudes, so the question was answered in [](#sec-error): use the error function, the error vector and the transport map again, with the predictor in the role of the actual attitude and the vehicle in the role of the desired one.

$$
\tilde\Psi = \tfrac12\tr\!\big(I - R\T R_p\big), \qquad
\tilde e_R = \tfrac12\big(R\T R_p - R_p\T R\big)^\vee, \qquad
\tilde e_\Omega = \Omega_p - R_p\T R\,\Omega . \label{eq8-pred-error}
$$

Subtracting matrices or body rates entry by entry would repeat the mistake of [](#fig3-frames). The uncertainty needs the same care. $\Delta$ is a torque in the body frame and its estimate $\hat\Delta$ is a torque in the predictor's frame, so the estimation error is $\tilde\Delta = \hat\Delta - R_p\T R\,\Delta$.

**The two inputs.** Choose

$$
\begin{aligned}
\mu_p &= -k_R\,e_R^p - k_\Omega\,e_\Omega^p + U\hat\Delta - \Delta_f, \qquad U = J_0\,R_p\T R\,J_0^{-1}R\T R_p ,\\
\mu &= J_0\,R\T R_p\,J_0^{-1}\big(-k_R\,e_R^p - k_\Omega\,e_\Omega^p - \Delta_f + \tilde k_R\,\tilde e_R + \tilde k_\Omega\,\tilde e_\Omega\big) + J_0\,R\T R_p\,\tilde e_\Omega^\wedge R_p\T R\,e_\Omega .
\end{aligned} \label{eq8-l1-law}
$$

Here $\Delta_f$ is the filtered estimate, defined below, and $\tilde k_R, \tilde k_\Omega > 0$ are gains on the prediction error. The predictor tracks the reference and feels the raw estimate. The vehicle is steered toward the predictor and feels only the filtered one. The matrix $U$ is the identity when $R_p = R$ or when $J_0$ is a multiple of $I$. It is there because $J_0$ is expressed in two different frames. With these choices the prediction error obeys

$$
J_0\,\dot{\tilde e}_\Omega = -\tilde k_R\,\tilde e_R - \tilde k_\Omega\,\tilde e_\Omega + U\tilde\Delta . \label{eq8-pred-dyn}
$$

Compare with [](#eq8-lumped). It is the same spring and damper on $\SO(3)$, now between predictor and vehicle, pushed by the estimation error and by nothing else. The filter has dropped out.

**The adaptation law.** The estimate moves in the direction that [](#eq8-pred-dyn) says will shrink the prediction error:

$$
\dot{\hat\Delta} = \gamma\,\operatorname{Proj}\big(\hat\Delta,\ y\big), \qquad y = -U\T\big(\tilde e_\Omega + c\,J_0^{-1}\tilde e_R\big), \label{eq8-adapt}
$$

with **adaptation gain** $\gamma > 0$. Without $\operatorname{Proj}$ this is an integrator of the prediction error with a very large gain, and an integrator can wind up without limit when the motors saturate or a sensor drops out. The projection keeps the estimate inside a ball of radius $\Delta_{\max}$:

$$
\operatorname{Proj}(\hat\Delta, y) = \begin{cases} y - \dfrac{\nabla\!f\,(\nabla\!f\cdot y)}{\|\nabla\!f\|^2}\,f & \text{if } f > 0 \text{ and } y\cdot\nabla\!f > 0,\\[6pt] y & \text{otherwise,} \end{cases} \label{eq8-proj}
$$

where $f(\hat\Delta) = \big(1.1\,\|\hat\Delta\|^2 - \Delta_{\max}^2\big)/\big(0.1\,\Delta_{\max}^2\big)$ is zero at radius $0.95\,\Delta_{\max}$ and one at $\Delta_{\max}$.

Inside radius $0.95\,\Delta_{\max}$ it does nothing. In the thin shell beyond, it scales down the outward part of $y$, to zero at the surface. It never makes the analysis worse: $\tilde\Delta\cdot\big(\operatorname{Proj}(\hat\Delta, y) - y\big) \le 0$ whenever the true uncertainty lies within $0.95\,\Delta_{\max}$. The ball is a statement of how much uncertainty you are prepared to cancel. Make it too small and the estimate is clipped: with $\Delta_{\max} = 1$ N·m against the 2.16 N·m torque of [](#ex8-camera), the simulated body settles 35° off. In a sampled implementation, follow each step of [](#eq8-adapt) by pulling the estimate back onto the ball, because a finite step can overshoot it.

**The filter.** The calm hand is a first-order low-pass with **cutoff** $\kappa$:

$$
\dot\Delta_f = \kappa\,\big(\hat\Delta - \Delta_f\big). \label{eq8-filter}
$$

::: {.theorem #thm8-l1 name="Kotaru, 2022, Theorem 3.1"}
Let the vehicle and the predictor be driven by [](#eq8-l1-law) with the adaptation law [](#eq8-adapt), and let the lumped uncertainty and its time derivative be bounded. If
$$
\tilde\Psi(0) < 2, \qquad \|\tilde e_\Omega(0)\|^2 < \frac{2}{\lambda_{\min}(J_0)}\,\tilde k_R\,\big(2 - \tilde\Psi(0)\big),
$$
then the prediction error $(\tilde e_R, \tilde e_\Omega)$ is exponentially input-to-state stable. The function $V = \tfrac12\tilde e_\Omega\cdot J_0\,\tilde e_\Omega + \tilde k_R\tilde\Psi + c\,\tilde e_R\cdot\tilde e_\Omega + \tfrac{1}{2\gamma}\|\tilde\Delta\|^2$ satisfies $\dot V + \beta V \le \beta\,\delta_V$ for constants $\beta, \delta_V > 0$, and $\delta_V$ is proportional to $1/\gamma$.
:::

The proof, in Appendix A.1 of the thesis, repeats [](#thm8-bound) for the prediction error, with the projection property removing the terms in $\tilde\Delta$. Start the predictor at the vehicle's state and the initial conditions hold trivially.

Be exact about what this says. The predictor and the vehicle come together exponentially, to within a distance that shrinks like $1/\gamma$. The theorem is about the prediction error. It does not bound the tracking error $e_R$ of the vehicle, and the filter does not appear in it. The tracking error has a second ingredient. By [](#eq8-l1-law) the predictor follows the reference while being pushed by $U\hat\Delta - \Delta_f$, the part of the estimate that the filter did not pass. A slow filter leaves more of it.

So the two knobs do different jobs. The gain $\gamma$ decides how closely the vehicle follows the predictor. The cutoff $\kappa$ decides how well the predictor follows the reference, and it alone decides what reaches the motors, because $\hat\Delta$ enters the torque only through $\Delta_f$. That separation is the central idea of $\mathcal{L}_1$ control. A classical adaptive law has one knob for both, and turning it up to learn faster feeds high-frequency torque to the vehicle.

## Does it work? {#sec8-l1-sim}

::: {.example #ex8-l1 name="The camera again, then gusts"}
Take $\gamma = 4000$, $\kappa = 20$ rad/s, $\Delta_{\max} = 3$ N·m, $\tilde k_R = k_R$ and $\tilde k_\Omega = k_\Omega$, with the controller running at 1 kHz.

*Hover.* With the camera of [](#ex8-camera), the error peaks at 2.8°, is below 0.1° after 0.49 s, and goes to zero. The filtered estimate converges to the true $\Delta$. With the 2.16 N·m torque that tumbled the plain controller, the error peaks at 15.6° and returns to zero.

*A moving reference.* Let $R_d(t)$ cone at up to 0.9 rad/s. With the camera, the plain controller's error is 11.3° RMS and the adaptive one's is 0.008°. Add a gust torque of up to 0.38 N·m with components between 1 and 7 rad/s, as in the validation of Kotaru (2022). The plain controller wanders between 9.9° and 17°. The adaptive one stays below 1.13°, at 0.59° RMS ([](#fig8-l1), left). It no longer converges to zero, and it should not: a gust at 7 rad/s is not far below the cutoff, so the filter passes only part of it.

*The theorem's quantity.* The prediction error stays below $\tilde\Psi = 10^{-8}$ throughout. Its rate $\|\tilde e_\Omega\|$ peaks at $3.1\times10^{-3}$, $7.8\times10^{-4}$, $2.0\times10^{-4}$ and $4.9\times10^{-5}$ rad/s for $\gamma = 250$, 1000, 4000 and 16000. Each factor of four in $\gamma$ divides it by four, as $\delta_V \propto 1/\gamma$ predicts.
:::

![Left: attitude error with the camera and a gust torque, for the plain controller [](#eq8-pd) and with adaptation. Right: one component of the lumped uncertainty $\Delta$ during the same flight with gyroscope noise. The raw estimate $\hat\Delta$ is fast and violent. Only the filtered $\Delta_f$ reaches the motors.](figures/ch08-l1.svg){#fig8-l1}

The right panel of [](#fig8-l1) is the co-pilot at work. With gyroscope noise of 0.02 rad/s, the raw estimate swings through 2.1 N·m around a truth that moves by 0.23. The filtered estimate tracks the truth.

Now turn the cutoff ([](#fig8-filter), left). At $\kappa = 1$ rad/s the hand is too calm: the error is 3.3° RMS, because the gusts are faster than the filter. At $\kappa = 300$ rad/s the error is 0.06°, and the rate of change of the commanded torque has grown from 12 to 104 N·m/s. The noise is going to the motors. Add a motor lag of 20 ms and a cutoff of 3000 rad/s is unstable, while 30 rad/s is unaffected.

![The filter cutoff trades tracking against robustness. Left: RMS attitude error and RMS rate of change of the commanded torque, with gyroscope noise. Right: the largest input delay for which the hover loop still settles. It falls as the cutoff rises, and does not move when the adaptation gain changes by a factor of 16.](figures/ch08-filter.svg){#fig8-filter}

The right panel is the claim of the co-pilot story, measured. Delay the torque command and find the largest delay the hover loop survives. The plain controller tolerates 46 ms. With adaptation the figure is 41 ms at $\kappa = 3$ rad/s, 23 ms at 30 and 6 ms at 300. At $\kappa = 30$ it is 23 ms for $\gamma = 1000$, 4000 and 16000 alike. The delay margin belongs to the filter. The adaptation gain is free.

::: {.warning #warn8-cutoff name="The cutoff is a robustness setting, so set it like one"}
It is tempting to raise $\kappa$ until the tracking plot looks good. Every increase spends delay margin, and the bill arrives on the day the motors are cold or the estimator is late. Choose $\kappa$ from the slowest actuator and the largest delay in the loop, and accept the tracking error that results. If that error is too large, the disturbance is too fast to cancel by feedback and needs a better model.
:::

Adaptation makes the error small. It does not make any error impossible. During the first tenth of a second the camera still tipped the vehicle by 2.8°. If 2.8° toward a wall is too much, a different kind of statement is needed.

## Never, as opposed to eventually {#sec8-barrier}

Stability says where the state ends up. Safety says where it must not be, at any time, on the way. A Lyapunov function measures progress toward a point. Safety needs a function that measures distance from a boundary, and a rule that keeps it positive.

::: {.thought #thought8-fog name="Driving at a wall in fog"}
You drive toward a wall you cannot see. A sensor reads the distance $H$. Adopt one rule: never close the gap faster than $\alpha H$. At 10 m you may approach at $10\alpha$ m/s, at 1 m at $\alpha$ m/s, at 1 mm hardly at all. Do you ever touch the wall?

You do not. The fastest approach the rule allows is $\dot H = -\alpha H$, whose solution is $H(0)\,e^{-\alpha t}$, positive forever. Any other permitted motion keeps $H$ above that curve. And the rule only forbids. Far from the wall it allows almost anything, and it never tells you to stop.
:::

::: {.lemma #thm8-comparison name="A barrier keeps its sign"}
If $\dot H(t) \ge -\alpha\,H(t)$ for all $t \ge 0$, with $\alpha > 0$, then $H(t) \ge H(0)\,e^{-\alpha t}$. In particular $H(0) \ge 0$ implies $H(t) \ge 0$ for all $t$.
:::

::: {.proof}
$\ddt\big(H e^{\alpha t}\big) = (\dot H + \alpha H)\,e^{\alpha t} \ge 0$.
:::

A function $H$ of the state whose nonnegative set $\{H \ge 0\}$ is the safe set, and for which some input always achieves $\dot H \ge -\alpha H$, is a **barrier function** (Ames, Xu, Grizzle and Tabuada, 2017). The safe set is then **forward invariant**: solutions that start in it stay in it.

Take the simplest system, a point in the plane with velocity input, $\dot x = u$. A wall with unit normal $n$ stands at distance $d$ from the origin, so $H(x) = d - n\cdot x$ and $\dot H = -n\cdot u$. A tracker proposes $u_d$. Among all velocities that satisfy the barrier condition, take the one nearest to $u_d$:

$$
u = \arg\min_{u}\ \|u - u_d\|^2 \quad\text{subject to}\quad n\cdot u \le \alpha H . \label{eq8-safety-filter}
$$

This is a quadratic program with one linear constraint. Its solution is the projection onto a half-plane ([](#fig8-wall), left),

$$
u = u_d - \max\big(0,\ n\cdot u_d - \alpha H\big)\,n .
$$

If the tracker's wish is permitted, it passes unchanged. If not, only its component toward the wall is trimmed. The filter makes the smallest change to what the tracker wanted.

::: {.example #ex8-wall name="A target behind the wall"}
The wall is at $x_1 = 1$, the point starts at the origin, and the tracker is $u_d = -(x - x_d)$ with the target $x_d = (2,\ 1.5)$ behind the wall. At $x = (0.9,\ 0)$ the tracker asks for $u_d = (1.1,\ 1.5)$ m/s. With $\alpha = 5$ s⁻¹ the allowed closing speed is $\alpha H = 0.5$ m/s, so the filter returns $u = (0.5,\ 1.5)$.

Unfiltered, the point goes straight to the target through the wall. With $\alpha = 5$ the filter is silent until $x_1 = 0.75$ and then bends the path. With $\alpha = 1$ it acts from the start. With $\alpha = 25$ it waits until $x_1 = 0.96$ ([](#fig8-wall), right). Every path ends at $(1,\ 1.5)$, the safe point nearest the target, and none touches the wall.
:::

![A point that must not cross a wall. Left: velocity space at $x = (0.9, 0)$ with $\alpha = 5$ s⁻¹. The barrier condition allows a half-plane of velocities, and the filter returns the allowed velocity nearest to the tracker's $u_d$. Right: the resulting paths for three values of $\alpha$. A larger $\alpha$ waits longer and turns harder.](figures/ch08-wall.svg){#fig8-wall}

Two things made this easy. The space was flat, and the input appeared in $\dot H$. A cable direction or an attitude is not flat, and a torque does not appear in the first derivative of anything that depends on the configuration alone.

## Barriers on the sphere {#sec8-cone}

A cable must stay clear of the propellers. A thrust axis must stay within a tilt limit. Each is a direction $q \in S^2$ that must remain within an angle $\theta_{\max}$ of an axis $q_c \in S^2$. The safe set is a cone, and its barrier function needs no coordinates:

$$
H(q) = q\cdot q_c - \cos\theta_{\max} . \label{eq8-cone}
$$

In the language of [](#def-error-s2) this is $H = (1 - \cos\theta_{\max}) - \Psi_q$ with $q_c$ as the second argument: a level set of the error function. Wu (2017) defines safe sets on any manifold this way, as $\Psi \le$ constant or $\Psi \ge$ constant.

The derivatives come from the kinematics $\dot q = \omega\times q$ alone:

$$
\dot H = \omega\cdot(q\times q_c), \qquad\text{or}\qquad \dot H = (\omega - \omega_c)\cdot(q\times q_c) \label{eq8-cone-dot}
$$

if the cone axis itself moves with angular velocity $\omega_c$. Only the relative angular velocity matters, as it should. No input has appeared. Differentiate again, for the pendulum of [](#eq7-pendulum) with a force $F$ on the bob and a fixed axis:

$$
\ddot H = -\|\omega\|^2\,(q\cdot q_c) \;+\; \big(I - q q\T\big)q_c\cdot\frac{F - m g\,e_3}{m\,l} . \label{eq8-cone-ddot}
$$

The first term is the centripetal acceleration, which always pulls $q$ off the axis it circles. The force enters through $(I - q q\T)q_c$, the direction in the tangent plane at $q$ that points toward the axis. The input appears only in the second derivative: the barrier has **relative degree two**.

The remedy is to apply the fog rule twice. Define $H_1 = \dot H + \alpha_1 H$ and ask that $H_1$, in turn, obey the rule:

$$
\dot H_1 + \alpha_2 H_1 = \ddot H + (\alpha_1 + \alpha_2)\,\dot H + \alpha_1\alpha_2\,H \;\ge\; 0 . \label{eq8-rd2}
$$

::: {.proposition #thm8-rd2 name="Safety with relative degree two"}
If $H \ge 0$ and $H_1 \ge 0$ at $t = 0$, and the input satisfies [](#eq8-rd2) for all $t \ge 0$, then $H(t) \ge 0$ for all $t \ge 0$.
:::

::: {.proof}
[](#thm8-comparison) applied to $H_1$ gives $H_1(t) \ge 0$, which reads $\dot H \ge -\alpha_1 H$. Apply the lemma again.
:::

The initial condition $H_1(0) \ge 0$ is new and it has teeth. It says the state must not start inside the cone while moving outward too fast to stop. The safe set in the state space is smaller than the cone. Wu (2017, Prop. 6.4 and Lemma 6.1) proves this result with the same $H_1$ and a reciprocal barrier $1/H_1$, which replaces the linear rate $\alpha_2 H_1$ by a cubic one. The linear form is the exponential barrier of Nguyen and Sreenath (2016). For a moving cone, [](#eq8-rd2) gains the terms $2\,\dot q\cdot\dot q_c + q\cdot\ddot q_c$ and nothing else changes.

Condition [](#eq8-rd2) is linear in $F$. So is the rate of a Lyapunov function. For tracking a reference $q_d(t)$ on the sphere take $V = \tfrac12 m l^2\|e_\omega\|^2 + k_q\Psi_q + c\,m l^2\,e_q\cdot e_\omega$, with the errors of [](#eq3-e-omega). Put both in one program, and split the force as $F = F_{\text{ff}} + F_{\text{fb}}$, where $F_{\text{ff}}$ is the feedforward that would keep a perfect track:

$$
\begin{aligned}
\min_{F_{\text{fb}},\,w}\quad & \tfrac12\|F_{\text{fb}}\|^2 + \tfrac12 k_w\,w^2\\
\text{subject to}\quad & \dot V(F) + k_V\,V \le w, \\
& \dot H_1(F) + \alpha_2 H_1 \ge 0 .
\end{aligned} \label{eq8-qp}
$$

This is the **CBF-CLF-QP** of Wu and Sreenath (2015). The two constraints are not treated alike. The barrier constraint is hard. The Lyapunov constraint carries a **slack** $w$ that the program may raise at a cost $k_w$.

Why relax tracking and not safety? Because the two can contradict each other, and you must decide in advance who wins. If the reference leaves the cone, "$V$ decreases" and "$H$ stays positive" cannot both hold. With two hard constraints the program has no solution at exactly the moment it is needed. With a slack on tracking there is always a solution: stay safe, and get as close as safety allows. Wu (2017, Prop. 6.5) proves that this controller keeps the state safe, provided the state starts in the safe set of [](#thm8-rd2), the program is feasible, and the states where a constraint loses its input form a set of measure zero. The proof promises nothing about tracking while the reference is unsafe.

::: {.example #ex8-cone name="A reference that leaves the cone"}
The pendulum of [](#ex7-s2-check) has $m = 0.5$ kg and $l = 0.8$ m. It hangs, and must stay within $\theta_{\max} = 30^\circ$ of $q_c = -e_3$. The reference $q_d(t)$ runs round a circle that reaches 48° from $q_c$. It is outside the cone for 56% of each 7 s lap. Use $\alpha_1 = \alpha_2 = 6$ s⁻¹, $k_V = 4$ s⁻¹ and $k_w = 50$.

The tracker alone follows the reference out to 48°, with $H$ as low as $-0.197$. Under [](#eq8-qp) the state tracks until the reference nears the boundary, then slides along the boundary for 40% of the lap, 18.5° behind the reference at worst, and rejoins it once it is back inside ([](#fig8-cone)). The smallest value of $H$ in the simulation is $3\times10^{-7}$, and it is positive. The slack rises to 2.6 while the reference is outside and falls back when it returns. The force never exceeds 2.5 N.

With a cone whose axis sways by 12°, the same program holds $H \ge -4\times10^{-6}$ at a 2 ms sample time. The small negative number is the sampling, which [](#sec8-limits) takes up.
:::

![The pendulum direction $q$ on the sphere, seen from below. The solid circle is the edge of the cone, $H = 0$. The dashed circle is the reference. While the reference is outside the cone the state slides along the edge, and it rejoins the reference when the reference returns.](figures/ch08-cone-strip.png){#fig8-cone anim="figures/ch08-cone.gif"}

**The same on $\SO(3)$.** A tilt limit on a rigid body is a cone for its thrust axis: $H(R) = e_3\cdot R\,e_3 - \cos\theta_{\max}$. With $\dot R = R\,\Omega^\wedge$,

$$
\dot H = \Omega\cdot\big(e_3\times R\T e_3\big), \qquad
\ddot H = \dot\Omega\cdot\big(e_3\times R\T e_3\big) - \|\Omega\|^2\,\big(e_3\cdot R\T e_3\big) + \big(\Omega\cdot R\T e_3\big)\big(\Omega\cdot e_3\big) , \label{eq8-tilt}
$$

and $\dot\Omega = J^{-1}(\tau - \Omega\times J\Omega)$ brings in the torque. Filter the torque of [](#eq8-pd) as in [](#eq8-safety-filter), with [](#eq8-rd2) as the constraint. Ask the body for a target 60° from level with a limit of 35°. It stops at 35.000°, 25° short, and holds there with $\Psi_R = 0.094$. This mirrors the 3D pendulum experiment of Wu (2017, sec. 6.5), whose constraint is a bound on $\Psi_R$: an unsafe target is approached as far as the safe set allows.

In both cases the input reached $\ddot H$. For a quadrotor that must avoid an obstacle, half of it does not.

## A vehicle that cannot push sideways {#sec8-quad}

Let a quadrotor avoid a ball of radius $r_o$ centered at $x_o$. The obvious barrier is $H = \|x - x_o\|^2 - r_o^2$. With [](#eq4-quadrotor),

$$
\ddot H = 2\|v\|^2 + 2\,(x - x_o)\cdot\Big(\frac{f}{m}\,R\,e_3 - g\,e_3\Big). \label{eq8-plain}
$$

Two things are wrong. The torque is absent: it appears only two derivatives later, so three of the four inputs cannot help. And the thrust enters through $p = (x - x_o)\cdot R\,e_3$, which can have either sign, while $f$ cannot be negative. If the thrust axis points toward the obstacle, more thrust makes things worse, and the only thrust that helps is less. A vehicle below the obstacle's center can do nothing but fall.

Wu (2017, ch. 7) repairs both defects with one idea. A vehicle whose thrust axis points at the obstacle is in more danger than one whose axis points away, at the same position. So let the barrier depend on the attitude. The **augmented barrier function** is

$$
H(x, R) = \|x - x_o\|^2 - \beta\,r_o^2 - \sigma(p), \qquad p = (x - x_o)\cdot R\,e_3 , \label{eq8-aug}
$$

with $\beta > 1$ and $\sigma$ a smooth decreasing function. When the thrust axis points at the obstacle, $p < 0$, $\sigma$ is large and the vehicle must keep farther away. Now $\dot H$ contains $\dot p$, which contains $\Omega$, and so $\dot H_1$ contains $\dot\Omega$. In $\dot H_1$ the thrust is multiplied by $\big(2p - \sigma'(p)\big)/m$ and the torque by the vector $-\sigma'(p)\,J^{-1}\big(e_3\times R\T(x - x_o)\big)$. The torque is back, and it vanishes only when the thrust axis is parallel to $x - x_o$.

Wu states three conditions on $\sigma$: it is strictly decreasing, it is bounded by $|\sigma| \le \sigma_0 < (\beta - 1)\,r_o^2$, and $2p - \sigma'(p) > 0$ for $-\sqrt{\beta}\,r_o < p < 0$. The second is what makes the construction sound: $H \ge 0$ then implies $\|x - x_o\|^2 \ge \beta r_o^2 - \sigma_0 > r_o^2$ (Wu, Lemma 7.2). Under all three, Wu's Lemma 7.4 states that the reciprocal of $H_1$ is a barrier function except on a set of states of measure zero, and that safety is guaranteed if $H_1 \ge 0$ at the start and the input limits are "sufficiently large". The exceptional states have the thrust axis pointing exactly at the obstacle from far away. The third condition is demanding: integrate it over the interval and it forces $2\sigma_0 > \beta r_o^2$, hence $\beta > 2$ ([](#exr8-sigma)).

::: {.example #ex8-obstacle name="Under the obstacle's center"}
A 1.13 kg quadrotor flies level at 2 m/s. Its reference passes 10 cm below the center of a ball of radius 0.6 m. Take $\beta = 1.8$ and Wu's sigmoid $\sigma(p) = 0.25\tanh(-2p)$, and filter the tracker's $(f, \tau)$ as in [](#eq8-safety-filter) with $0 \le f \le 30$ N.

With the plain barrier [](#eq8-plain) the program becomes infeasible 1.88 m from the center, where $p = -0.10$ m. No thrust between 0 and 30 N satisfies it, and the vehicle hits the ball. With the augmented barrier the program is feasible at every step. The vehicle tilts by up to 40°, passes over the ball no closer than 0.715 m from its center, and returns to the reference ([](#fig8-obstacle)). The thrust stays between 9.0 and 14.3 N against 11.1 N in hover.

This $\sigma$ meets Wu's first two conditions and meets the third only for $-0.21 < p < 0$. The simulation is a demonstration and not an instance of his lemma.
:::

![A quadrotor and an obstacle, seen from the side. With the plain barrier only the thrust can respond, and below the center it pushes the wrong way. With the augmented barrier the torque tilts the thrust axis away early and the vehicle goes around. The dashed circle has radius $\sqrt{\beta}\,r_o$.](figures/ch08-obstacle-strip.png){#fig8-obstacle anim="figures/ch08-obstacle.gif"}

When several quadrotors carry one cable-suspended load they must also keep clear of each other. Kotaru (2022, sec. 6.5) puts a filter of the kind [](#eq8-safety-filter) after the payload controller, on the commanded accelerations, with one barrier $H_{ij} = \|x_i - x_j\|^2 - r_{\min}^2$ for each pair and the constraint [](#eq8-rd2). He notes that the added safety comes at the expense of stability and must be tuned with care. In our simulation, two vehicles that swap places pass 0.10 m apart without the filter and never closer than 1.15 m with $r_{\min} = 0.8$ m. The filter treats each vehicle as a point that can accelerate in any direction. Whether the attitude loop delivers that acceleration is the question of this section again.

## Honest limits {#sec8-limits}

**The barrier trusts the model.** Condition [](#eq8-rd2) is evaluated with a model of $\ddot H$. If the model is wrong, the guarantee is about a system you do not have. Give the pendulum of [](#ex8-cone) a steady 1 N wind that the program does not know. The state now settles outside the cone, at $H = -0.032$, which is 3.4° too far. The size is predictable. At rest on the boundary, [](#eq8-rd2) holds with an error in $\ddot H$, so $H \approx -(\text{error in } \ddot H)/(\alpha_1\alpha_2) = -0.035$. Tripling both gains divides the violation by nine, to $-0.0034$. Tightening the cone by 5° keeps the true cone, with $H \ge 0.011$.

**Adaptive and safe.** It is natural to feed the estimate $\Delta_f$ of [](#sec8-l1) into the barrier's model. That helps, and no theorem in this chapter covers it. [](#thm8-l1) bounds the prediction error and not the estimate, and $\Delta_f$ lags the truth by design. What can be said is this: if the model of $\ddot H$ is wrong by at most some amount, the violation is at most that amount over $\alpha_1\alpha_2$. Size a margin from that bound and do not call the result a guarantee until the bound on the model error is itself established.

**Input limits.** The program [](#eq8-qp) has a solution because nothing limits $F$. Arrive at 29° moving outward at 1.5 rad/s, and [](#eq8-rd2) demands 6.2 N from a bob that weighs 4.9 N. Cap the force at half of that and the program is infeasible: no admissible input keeps the promise. A barrier constructed without the actuator limits certifies a set that the actuators cannot hold. Wu (2017) calls a barrier that accounts for input saturation an open problem, and predictive methods ([](#ch-mpc)) are one answer to it.

**Sampling.** [](#thm8-comparison) is a statement about continuous time. A sampled controller holds its input between samples. In [](#ex8-cone) the consequence is not a violation: $H$ stays positive even at 100 ms. It is chatter. At a 2 ms sample time the largest jump in force between samples is 0.013 N. At 20 ms it is 17.8 N, seven times the peak force of the smooth solution, as the state bounces off the constraint once per sample.

**Measure zero.** The force enters [](#eq8-cone-ddot) through $(I - q q\T)q_c$, which vanishes at $q = \pm q_c$. The torque enters [](#eq8-aug) through a vector that vanishes when the thrust axis points along $x - x_o$. On those sets the barrier condition cannot be enforced. A reference aimed exactly at the center of the ball leaves the vehicle no reason to pass on one side or the other. Wu calls such barriers almost-global, and the word is the one from [](#thm-obstruction) for the same reason: on a compact manifold a smooth function must have critical points, and a controller built from its gradient has nothing to say there.

::: {.warning #warn8-filter name="A safety filter is only as good as its model"}
"Provably safe" means: safe for the model, in continuous time, with unlimited inputs, from initial states with $H \ge 0$ and $H_1 \ge 0$, away from a set of measure zero. Each clause is a way to hit the wall with a proof in hand. Before flight, test the filter with the wrong mass, at the real sample rate, with the real thrust limits, starting fast and close.
:::

## Exercises {#sec8-exercises .unnumbered}

::: {.exercise #exr8-lumped}
Derive [](#eq8-lumped) from [](#eq8-pd) and $J\dot\Omega + \Omega\times J\Omega = \tau + \Delta_e$. Show that $\Delta = \Delta_e$ when $J = J_0$. For a body at rest at a fixed target, show that $\Delta$ depends on $J$ only through the transient.
:::

::: {.exercise #exr8-offset}
Show that with a constant $\Delta_e$ and a fixed target, the equilibria of the plain controller are the attitudes with $k_R\,e_R = \Delta_e$. For $\|\Delta_e\| < k_R$ there are two error angles, $\theta$ and $180^\circ - \theta$. Which one is stable? Check in the notebook.
:::

::: {.exercise #exr8-integral}
Add an integral term $-k_I\int (e_\Omega + c\,e_R)\,\dd t$ to [](#eq8-pd). Show in simulation that it removes the offset of [](#ex8-camera). Then apply the gust of [](#ex8-l1) and compare with the adaptive controller at equal delay margin.
:::

::: {.exercise #exr8-pred}
Derive [](#eq8-pred-dyn). Start from $\tilde e_\Omega = e_\Omega^p - R_p\T R\,e_\Omega$ and use $\ddt\big(R_p\T R\big) = -\tilde e_\Omega^\wedge R_p\T R$. Where is the last term of $\mu$ in [](#eq8-l1-law) used?
:::

::: {.exercise #exr8-proj}
Prove the projection property $\tilde\Delta\cdot\big(\operatorname{Proj}(\hat\Delta, y) - y\big) \le 0$ from the convexity of $f$ in [](#eq8-proj). Then find a pair $\hat\Delta$, $\Delta$ with $\|\Delta\| > \Delta_{\max}$ for which it fails.
:::

::: {.exercise #exr8-cone}
Derive [](#eq8-cone-ddot) from $\ddot q = -\|\dot q\|^2 q + (I - q q\T)(F - m g\,e_3)/(m l)$. Then derive [](#eq8-tilt). In both, find the configurations at which the input drops out of $\ddot H$, and explain why they do not matter for a cone with $\theta_{\max} < 90^\circ$.
:::

::: {.exercise #exr8-sigma}
Show that Wu's second and third conditions on $\sigma$ together require $\beta > 2$. Then show numerically that no sigmoid $k_1\tanh\!\big(k_2(p_0 - p)\big)$ satisfies both for $\beta = 4$, and construct a $\sigma$ that does.
:::

::: {.exercise #exr8-wind}
Repeat the wind experiment of [](#sec8-limits) with the wind turned on only while the state is on the boundary. Design a margin from the bound $(\text{error in }\ddot H)/(\alpha_1\alpha_2)$ and test it. What does the margin cost in tracking when there is no wind?
:::

## Notes and sources {#sec8-notes .unnumbered}

The geometric tracking controller and its Lyapunov function are from Lee, Leok and McClamroch (2010). L1 adaptive control is due to Cao and Hovakimyan and is set out in Hovakimyan and Cao (2010). The geometric version on $\SO(3)$, with the predictor, the prediction errors [](#eq8-pred-error), the control laws [](#eq8-l1-law) and [](#thm8-l1), is from Kotaru, Edmonson and Sreenath (2020) and Kotaru (2022, ch. 3), which also report flight experiments. The projection operator [](#eq8-proj) is that of Lavretsky and Gibson (2011). Barrier functions in the form used here, and the quadratic program that combines them with a Lyapunov function, are from Ames, Xu, Grizzle and Tabuada (2017). Wu and Sreenath (2015) and Wu (2017, ch. 6) defined them on manifolds, with safe sets given by error functions, time-varying constraints, and the examples on $S^2$ and $\SO(3)$ that [](#sec8-cone) follows. The exponential form of [](#eq8-rd2) is from Nguyen and Sreenath (2016). The augmented barrier function for a quadrotor is from Wu and Sreenath (2016) and Wu (2017, ch. 7). The pairwise filter for vehicles that share a load is from Kotaru (2022, sec. 6.5). Mueller (2018) treats recovery from large attitude errors, where the fading restoring torque of [](#thought8-spring) matters most.

- A. D. Ames, X. Xu, J. W. Grizzle and P. Tabuada, ["Control barrier function based quadratic programs for safety critical systems"](https://arxiv.org/abs/1609.06408), *IEEE Transactions on Automatic Control* 62(8), 2017. arXiv:1609.06408.
- N. Hovakimyan and C. Cao, *L1 Adaptive Control Theory: Guaranteed Robustness with Fast Adaptation*, SIAM, 2010.
- P. Kotaru, [*Dynamics and Control for Collaborative Aerial Manipulation*](https://vkotaru.github.io/dissertation), PhD thesis, University of California, Berkeley, 2022.
- P. Kotaru, R. Edmonson and K. Sreenath, ["Geometric L1 adaptive attitude control for a quadrotor unmanned aerial vehicle"](https://arxiv.org/abs/1910.07730), *ASME Journal of Dynamic Systems, Measurement, and Control*, 2020. arXiv:1910.07730.
- E. Lavretsky and T. E. Gibson, ["Projection operator in adaptive systems"](https://arxiv.org/abs/1112.4232), arXiv:1112.4232, 2011.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- M. W. Mueller, ["Multicopter attitude control for recovery from large disturbances"](https://arxiv.org/abs/1802.09143), arXiv:1802.09143, 2018.
- Q. Nguyen and K. Sreenath, ["Exponential control barrier functions for enforcing high relative-degree safety-critical constraints"](https://ieeexplore.ieee.org/document/7524935), *American Control Conference*, 2016.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
- G. Wu and K. Sreenath, ["Safety-critical and constrained geometric control synthesis using control Lyapunov and control barrier functions for systems evolving on manifolds"](https://hybrid-robotics.berkeley.edu/publications/ACC2015_Geometric_CBFCLF.pdf), *American Control Conference*, 2015.
- G. Wu and K. Sreenath, ["Safety-critical control of a 3D quadrotor with range-limited sensing"](https://hybrid-robotics.berkeley.edu/publications/DSCC2016_Safety_Control_3D_Quadrotor.pdf), *ASME Dynamic Systems and Control Conference*, 2016.
