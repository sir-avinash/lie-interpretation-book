---
title: Why coordinate-free?
short: Why coordinate-free?
id: ch-why
summary: Euler angles break, quaternions unwind, and no continuous controller can be global. What to do about it.
notebook: notebooks/ch01_singularities_unwinding.ipynb
verify: verify/ch01.py
epigraph: "A map is not the territory it represents …"
epigraph_source: "Alfred Korzybski, *Science and Sanity* (1933)"
---

Here is a question that sounds too easy. Your robot's heading is 350°. You want it at 10°. What is the error?

Subtract and you get 340°, so the controller swings the robot almost all the way round ([](#fig-heading)). The sensible answer is 20° the other way. You already know the patch: wrap the difference into $(-180^\circ, 180^\circ]$. But notice what the patch admits. Headings do not live on a line. They live on a circle, and subtraction is an operation for lines.

That was one angle. A drone has three, tangled together, and the patches stop working. This chapter shows three ways the tangle bites, proves that one of them cannot be patched by anyone, and then shows the way out that the rest of the book follows.

![The heading puzzle. Start at 350°, target at 10°. Watch the marker on the line and the robot on the circle together. Subtracting the two numbers treats headings as points on a line, and the robot turns 340° the long way round. On the circle the target is 20° away through north, because the two ends of the line are one heading.](figures/ch01-heading-strip.png){#fig-heading anim="figures/ch01-heading.gif"}

## The space a rotation lives in {#sec-rotation-space}

Bolt a frame to a rigid body, the **body frame**, and another to the world, the **world frame**. Stack the body's three axes, written in world coordinates, as the columns of a matrix $R$. The columns are unit length and mutually perpendicular, and they form a right-handed set. Those two facts are the whole definition.

::: {.definition #def-so3 name="Rotation group"}
$$
\SO(3) = \{\, R \in \R^{3\times 3} : R\T R = I,\ \det R = +1 \,\}.
$$
:::

Nine numbers, six constraints, three degrees of freedom. The set is closed under multiplication and inversion, so it is a group. It is also a smooth, curved, three-dimensional surface sitting inside $\R^9$, which makes it a Lie group. [](#def2-lie-group) takes that phrase apart. For now one picture is enough.

Every rotation is a turn by some angle $\theta \in [0, \pi]$ about some unit axis $a$. Plot it as the point $\theta a$ in ordinary space. All rotations then fill a solid ball of radius $\pi$ ([](#fig-so3-ball)). The center is "no rotation". Walking outward along a ray turns the body further about that axis.

![The set of all rotations, drawn as a solid ball. A point $\theta a$ is the rotation by angle $\theta$ about axis $a$. Half a turn about $a$ and half a turn about $-a$ are the same rotation, so each boundary point is glued to its antipode. A body that keeps turning about $a$ exits the ball at $\pi a$ and re-enters at $-\pi a$.](figures/ch01-so3-ball.svg){#fig-so3-ball}

The catch is at the boundary. Half a turn about $a$ leaves the body exactly where half a turn about $-a$ does. So the point $\pi a$ and the point $-\pi a$ are the *same rotation*, and the ball's surface is glued to itself antipodally. No flat space has that shape. Hold on to this picture. Every failure in this chapter is a different way of ignoring the gluing.

Now let the body move. Its angular velocity, measured in its own frame, is a vector $\Omega \in \R^3$. To write the motion of $R$ we need to turn that vector into a matrix.

::: {.definition #def-hat name="Hat and vee"}
The **hat map** sends $x \in \R^3$ to the skew-symmetric matrix that performs the cross product with $x$:
$$
x^\wedge = \begin{bmatrix} 0 & -x_3 & x_2 \\ x_3 & 0 & -x_1 \\ -x_2 & x_1 & 0 \end{bmatrix},
\qquad x^\wedge y = x \times y .
$$
The **vee map** is its inverse, $(x^\wedge)^\vee = x$. The skew-symmetric matrices form the vector space $\so(3)$.
:::

The kinematics of a rigid body are then one line. (Every symbol and convention of the book is collected in [](#app-notation).)

$$
\dot R = R\,\Omega^\wedge. \label{eq-kinematics}
$$

Look at what is absent. No trigonometry, no denominators, no special cases. Equation [](#eq-kinematics) holds at every attitude. Keep it in mind as the standard the alternatives have to meet.

## Trap one: three angles {#sec-euler}

Nine numbers for three degrees of freedom feels wasteful, so the first instinct is to use three. The aerospace favorite is yaw, pitch, roll:

$$
R = R_z(\psi)\, R_y(\theta)\, R_x(\phi). \label{eq-zyx}
$$

Differentiate [](#eq-zyx) and match it against [](#eq-kinematics). Collect the angles as $\Theta = (\psi, \theta, \phi)$. The angle rates $\dot\Theta = (\dot\psi, \dot\theta, \dot\phi)$ and the body rate are related by a matrix that depends on where you are:

$$
\Omega = E(\theta,\phi)\,\dot\Theta, \qquad
E = \begin{bmatrix} -\sin\theta & 0 & 1 \\ \cos\theta\sin\phi & \cos\phi & 0 \\ \cos\theta\cos\phi & -\sin\phi & 0 \end{bmatrix},
\qquad \det E = -\cos\theta . \label{eq-euler-jacobian}
$$

To integrate the angles you need $E^{-1}$, and $E^{-1}$ carries a factor $1/\cos\theta$. Pitch the nose straight up and it is gone.

::: {.thought #thought-gimbal name="A gimbal you can hold"}
Picture three nested rings ([](#fig-gimbals)). The outer ring swivels about a vertical axis: yaw. The middle ring pivots inside it: pitch. The inner ring pivots inside that: roll. Each ring's axle is carried by the ring outside it.

Now pitch the middle ring by 90°. The roll axle, which rides on the middle ring, swings up until it points along the yaw axle. You now have two rings that do the same thing and no ring that does the third thing. One direction of rotation has become impossible to command. Nothing is wrong with the body. The *rings* have run out of directions.
:::

![Three nested gimbals realize yaw, pitch and roll. Watch the roll axle as the middle ring pitches up: at 90° it lies along the yaw axle, and $\det E = -\cos\theta$ has reached zero. A yaw command and a roll command then carry the arrow on the body along the same arc. No ring is left to turn the body about the dashed axis.](figures/ch01-gimbals-strip.png){#fig-gimbals anim="figures/ch01-gimbals.gif"}

This is **gimbal lock**, and it is not a historical curiosity. Apollo's inertial platform had three gimbals. During Apollo 11, after Houston warned that a maneuver had come close to lock, Michael Collins asked mission control to send him a fourth gimbal for Christmas.^[The exchange is in the Apollo 11 air-to-ground record at about 105 hours mission time, and is discussed in the *Apollo Lunar Surface Journal*.]

::: {.example #ex-euler-blowup name="A calm body, a panicking chart"}
A quadrotor pitches up through a maneuver with body rate $\Omega = (0, 0.5, 0.5)$ rad/s. Its angular speed is $\|\Omega\| = 0.71$ rad/s, about 40° per second. Unhurried. Take roll $\phi = 45^\circ$ and let the pitch reach $89^\circ$. Solving [](#eq-euler-jacobian) for the yaw rate gives

$$
\dot\psi = \frac{\Omega_2 \sin\phi + \Omega_3\cos\phi}{\cos\theta} = \frac{0.707}{0.01745} = 40.5\ \text{rad/s},
$$

which is 2300° per second. The body is doing nothing of the kind ([](#fig-euler-rates)). A controller that feeds back on $\dot\psi$ sees a violent spin and responds with a violent torque.
:::

![The same motion, described two ways. The body's angular speed is constant. The yaw rate of the Euler chart grows like $1/\cos\theta$ and is unbounded at 90° of pitch.](figures/ch01-euler-rates.svg){#fig-euler-rates}

You might try a different angle order. It moves the singularity and does not remove it. The ZXY order used for legged robots locks at 90° of roll instead. This is a theorem and not bad luck: no three numbers can cover $\SO(3)$ smoothly without a singularity somewhere.^[Stuelpnagel (1964) showed that three parameters cannot give a global nonsingular parametrization of the rotation group, and that five is the least number that can give a one-to-one global one.] The ball of [](#fig-so3-ball) says why. A flat three-dimensional chart cannot reproduce that antipodal gluing.

There is a second defect, quieter and just as damaging.

::: {.thought #thought-two-labs name="Two labs, one robot"}
Two labs test the same robot with the same controller, $\tau = -K_p(\Theta - \Theta_d) - K_d\dot\Theta$. Lab A's world frame has $z$ pointing up. Lab B mounted the motion-capture system sideways, so its world $x$ points up. The robot, the target and the physical error are identical. Do the two controllers command the same torque?

They do not. Rotate the world frame by $Y$ and the attitudes become $YR$ and $YR_d$. The physical error, $R_d\T R$, is unchanged because $(YR_d)\T(YR) = R_d\T R$. But the Euler angles of $YR$ are not the Euler angles of $R$ plus a constant, so the difference $\Theta - \Theta_d$ changes. Lab B's robot behaves differently for no physical reason.
:::

A control law whose output does not depend on the choice of world frame is called **left-invariant**. Euler-angle feedback is not. We will insist on invariance from now on, because a controller that is sensitive to how you bolted down your cameras is not a law about the robot.

## Trap two: four numbers {#sec-quaternions}

Quaternions are the standard fix. A unit quaternion $\quat = (q_0, q_v)$, with scalar part $q_0$, vector part $q_v \in \R^3$ and $q_0^2 + \|q_v\|^2 = 1$, is a point on the three-sphere $S^3$. The rotation by angle $\theta$ about axis $a$ is

$$
\quat = \left(\cos\tfrac{\theta}{2},\ \sin\tfrac{\theta}{2}\,a\right), \qquad
R(\quat) = I + 2 q_0\, q_v^\wedge + 2\,(q_v^\wedge)^2 . \label{eq-quat-to-R}
$$

The kinematics $\dot\quat = \tfrac12\, \quat \otimes (0, \Omega)$ have no singularity. Four numbers and one constraint beat nine numbers and six. So what is the catch?

It is the half angle in [](#eq-quat-to-R). Turn the body through a full 360° and $\theta/2$ has only reached 180°, so $\quat$ has arrived at $-\quat$, not back where it started. Since $R(\quat)$ is quadratic in $\quat$, the two give the same matrix:

$$
R(\quat) = R(-\quat).
$$

Every attitude has two quaternion names. $S^3$ wraps around $\SO(3)$ twice, a **double cover**.

::: {.thought #thought-plate name="The plate trick"}
Balance a plate on your upturned palm. Rotate it a full turn about the vertical, keeping it level, by passing your hand under your armpit. The plate is back where it started. Your arm is not: it is twisted. Now do a second full turn in the same direction, this time swinging your hand over your head. The plate is back again, and your arm has untwisted.

Your arm is keeping track of the quaternion. One turn takes $\quat$ to $-\quat$: same plate, different arm. Two turns bring it home ([](#fig-plate)). The plate cannot tell the difference. A controller that works with $\quat$ can.
:::

![The plate trick with a ribbon in place of the arm. The plate stays level and only turns about the vertical. The lower dial shows its quaternion, which moves at half the speed of the plate. After one turn the plate is back and the ribbon holds a full twist, with the quaternion at $-\mathbf{1}$. No motion of the ribbon removes that twist while the plate keeps this attitude. During the second turn in the same direction the plate swings wide and the ribbon comes out flat, with the quaternion back at $+\mathbf{1}$.](figures/ch01-plate-strip.png){#fig-plate anim="figures/ch01-plate.gif"}

Here is how that becomes a failure. To hold the attitude at the identity, the natural quaternion law is

$$
\tau = -k_p\, q_v - k_d\,\Omega . \label{eq-quat-pd}
$$

It drives $q_v$ to zero, with $q_0 \to +1$. But $q_0 = -1$ is the same attitude, and this law treats it as a place to flee from.

::: {.example #ex-unwinding name="Ten degrees or three hundred and fifty"}
A body with inertia $J = \operatorname{diag}(0.05, 0.05, 0.10)$ kg·m² sits at rest, 10° of yaw away from the target. The estimator may report either of the two valid quaternions:

$$
\quat = (0.9962,\ 0,\ 0,\ -0.0872) \qquad\text{or}\qquad -\quat = (-0.9962,\ 0,\ 0,\ 0.0872).
$$

Apply [](#eq-quat-pd) with $k_p = 2$, $k_d = 0.6$. From $\quat$ the body turns 10° and stops. From $-\quat$ it turns 350° the other way to reach the same place ([](#fig-unwinding)), and spends 48 times the control effort $\int \|\tau\|^2\,\dd t$ doing it. Same robot, same physical state, same controller. Only the sign the estimator happened to choose differs.
:::

![Unwinding: the two simulated responses of [](#ex-unwinding). Both bodies start in the same attitude, 10° from the target. For rotations about one fixed axis the quaternion lives on a circle that the attitude traverses at half speed, so $+\mathbf{1}$ and $-\mathbf{1}$ are the same attitude. The law [](#eq-quat-pd) only accepts $+\mathbf{1}$. Watch the two dots on the circle: one is already next to $+\mathbf{1}$, the other starts next to $-\mathbf{1}$ and drags its body through 350°.](figures/ch01-unwinding-strip.png){#fig-unwinding anim="figures/ch01-unwinding.gif"}

This is **unwinding**, named and analyzed by Bhat and Bernstein (2000). The obvious patch is $\tau = -k_p \operatorname{sgn}(q_0)\, q_v - k_d\Omega$, which always takes the short way. It also makes the torque jump when $q_0$ crosses zero, at 180° of error. Sensor noise there flips the sign back and forth and the motors chatter. Fixing *that* needs memory: a hysteresis switch, which makes the controller hybrid (Mayhew, Sanfelice and Teel, 2011).

So the patch works, at the price of a discontinuity. Is there a smarter continuous law that avoids both problems? The next section shows there is not, for any representation, and the reason has nothing to do with quaternions.

## The obstruction nobody can engineer around {#sec-obstruction}

Go back to the heading on a circle. Suppose a continuous feedback law makes the target heading asymptotically stable. Just clockwise of the target, the flow must push counterclockwise. Just counterclockwise of it, the flow must push clockwise. Now walk around the circle from one side of the target to the other. The flow starts out pointing one way and ends up pointing the other. It is continuous, so somewhere it passes through zero.

That zero is a second equilibrium ([](#fig-obstruction), left). A trajectory that starts there never reaches the target. You did not design it and you cannot remove it. The intermediate value theorem put it there.

![Left: on a circle, any continuous flow with a stable target must vanish somewhere else. Right: the same on the sphere. Under $\omega = -k\,e_q$ every point flows to $q_d$ along a great circle, except the antipode $-q_d$, which is an equilibrium.](figures/ch01-obstruction.svg){#fig-obstruction}

The general statement replaces the circle with any compact configuration space.

::: {.theorem #thm-obstruction name="Bhat and Bernstein, 2000"}
Let the state space be a vector bundle over a compact manifold without boundary, for example the attitudes and angular velocities of a rigid body. Then no continuous time-invariant vector field on it has a globally asymptotically stable equilibrium.
:::

The proof idea fits in three sentences. The set of states that converge to an asymptotically stable equilibrium can be shrunk continuously to that equilibrium by running the flow, so it is *contractible*. If the equilibrium were globally stable, that set would be the whole state space. But a space built over the glued ball of [](#fig-so3-ball), or over a sphere, has loops and shells that cannot be shrunk to a point.

This changes what we should ask of a controller. Global asymptotic stability with continuous feedback is off the table for attitudes, for any representation and any cleverness. The best possible continuous result is **almost-global** stability: every initial condition converges except a thin set, of measure zero, that drains into the extra equilibria. Those equilibria are unstable, so noise knocks the state off them. A good geometric controller achieves exactly this, and says so honestly. [](#thm6-almost) is the precise statement.^[On $\SO(3)$ a smooth function with nondegenerate critical points has at least four of them. Koditschek (1989) built attitude error functions with exactly four: the target, and three half-turn attitudes that are unstable.]

Compare that with the quaternion law of [](#ex-unwinding). It also has an extra equilibrium, at $q_0 = -1$. But that point *is the target attitude*. The law has placed its unavoidable bad point on top of the place you want to be. The theorem says an unstable equilibrium must exist somewhere. It does not say you must put it there.

## The way out: measure error on the manifold {#sec-error}

The three failures have one cause. Each takes a difference of labels, of angles or of quaternion components, and treats it as the error. The remedy is to build the error from the rotations themselves.

Start with a rule the answer must obey. The physical discrepancy between attitude $R$ and target $R_d$ is the relative rotation $R_d\T R$: how the body is turned as seen from the target frame. It equals $I$ exactly when they agree, and it passes the two-labs test. So build everything from $R_d\T R$.

::: {.definition #def-error-so3 name="Attitude error"}
For $R, R_d \in \SO(3)$, the **error function**, **error vector** and **velocity error** are
$$
\Psi_R = \tfrac12 \tr\!\left(I - R_d\T R\right), \qquad
e_R = \tfrac12\left(R_d\T R - R\T R_d\right)^\vee, \qquad
e_\Omega = \Omega - R\T R_d\,\Omega_d . \label{eq-error-so3}
$$
:::

Each piece has a plain reading. If $R_d\T R$ is a turn by angle $\theta$ about axis $a$, then

$$
\Psi_R = 1 - \cos\theta, \qquad e_R = \sin\theta\; a. \label{eq-error-angle}
$$

$\Psi_R$ is a height: zero at the target, rising to 2 at half a turn away. It plays the role that $\tfrac12\|x - x_d\|^2$ plays for positions. The vector $e_R$ is its slope. Nudge the attitude by a small body-frame rotation $\eta$, so that $\delta R = R\,\eta^\wedge$, and the height changes by

$$
\delta\Psi_R = e_R \cdot \eta .
$$

So $e_R$ points uphill and $-e_R$ is the torque direction that reduces the error fastest. That is a proportional term, derived instead of guessed.

The velocity error needs one more idea. $\Omega$ is expressed in the body frame and $\Omega_d$ in the target frame. These are different frames, so subtracting the components directly compares unlike things. The matrix $R\T R_d$ re-expresses $\Omega_d$ in the body frame first. It is our first transport map, and [](#sec3-transport) explains why it has to be there. With it, the two errors fit together exactly as position and velocity error do:

$$
\ddt \Psi_R = e_R \cdot e_\Omega . \label{eq-psi-dot}
$$

::: {.example #ex-error-at-lock name="The error at gimbal lock, and at half a turn"}
Take $R_d = I$ and pitch the body to 90°, where the Euler chart broke:
$$
R = R_y(\tfrac{\pi}{2}) = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}, \qquad
\Psi_R = \tfrac12(3 - 1) = 1, \qquad e_R = (0, 1, 0).
$$
Nothing happens. The error is finite and points about the pitch axis, as it should.

Now take half a turn about the first axis, $R = \operatorname{diag}(1, -1, -1)$. Then $\Psi_R = 2$, the maximum, and $R$ is symmetric so $e_R = 0$. This is the extra equilibrium that [](#thm-obstruction) promised. It sits at the farthest possible attitude from the target, which is the right place for it.
:::

The same construction works for a direction. A cable, a thrust axis or a pendulum rod is a unit vector $q \in S^2$, moving as $\dot q = \omega \times q$.

::: {.definition #def-error-s2 name="Direction error"}
For $q, q_d \in S^2$,
$$
\Psi_q = 1 - q \cdot q_d, \qquad e_q = q_d \times q, \qquad \ddt \Psi_q = e_q \cdot \omega \quad (q_d \text{ fixed}). \label{eq-error-s2}
$$
:::

Again $\Psi_q = 1 - \cos\theta$ for the angle $\theta$ between the two directions, and $\|e_q\| = \sin\theta$. The vector $e_q$ lies in the tangent plane at $q$, and the feedback $\omega = -k\,e_q$ moves $q$ straight along the great circle toward $q_d$ ([](#fig-s2-error)). The one equilibrium besides the target is the antipode $-q_d$, and it is unstable: in terms of the angle, the closed loop is $\dot\theta = -k\sin\theta$ ([](#fig-s2-flow)).

![Error on the sphere. Level sets of $\Psi_q$ are circles of latitude about $q_d$. The error vector $e_q = q_d \times q$ lies in the tangent plane at $q$, perpendicular to the great circle through $q$ and $q_d$. The feedback $\omega = -k\,e_q$ produces the velocity $\dot q$ along that great circle.](figures/ch01-s2-error.svg){#fig-s2-error}

![The flow of $\omega = -k\,e_q$. Points released all over the sphere slide along great circles to $q_d$. Watch the ones that start near the antipode: they barely move at first, because the push is $\sin\theta$. A point placed exactly at $-q_d$ never leaves.](figures/ch01-flow-strip.png){#fig-s2-flow anim="figures/ch01-flow.gif"}

::: {.warning #warn-sine name="The restoring torque fades near half a turn"}
Equation [](#eq-error-angle) has a consequence worth knowing before you fly. The restoring torque is proportional to $\|e_R\| = \sin\theta$. It grows up to 90° of error and then *shrinks*, reaching zero at 180°. A body flipped almost upside down feels almost no restoring torque, and may linger there. Mueller (2018) analyzes this for multicopters and shows the closed loop can stay near a large error for an arbitrarily long time. It is the price of a smooth error function. [](#sec3-errors) compares the alternatives, which reshape $\Psi_R$ so that the push does not fade, and [](#fig3-heights) and [](#fig3-linger) show what each one costs. [](#sec6-almost) measures what the fading costs a tracking controller: a fixed delay for every factor of ten closer to half a turn ([](#fig6-settle)). [](#sec7-local) measures the same effect on a regulator built by linearization.
:::

## Where this leaves us {#sec-roadmap}

We now have the pattern for the whole book. Find the space the configuration lives on. Build errors, velocities and derivatives from the objects themselves. Choose coordinates only to compute, never to define.

The pieces we have used so far were pulled out of a hat: the hat map, the vector $e_R$, the transport map $R\T R_d$. The next two chapters derive them. [](#ch-manifolds) sets up the manifolds a robot needs, and [](#ch-variations) explains how to take derivatives and compare velocities on them. After that the same handful of tools gives equations of motion without angles ([](#ch-dynamics)), controllers with almost-global guarantees ([](#ch-tracking)), linearizations that are valid everywhere ([](#ch-linearization)), estimators whose error dynamics do not depend on the state, and optimizers that never leave the manifold ([](#ch-mpc)).

## Exercises {#sec-ch1-exercises .unnumbered}

::: {.exercise #exr-euler-jacobian}
Starting from [](#eq-zyx), compute $(R\T \dot R)^\vee$ and confirm [](#eq-euler-jacobian), including $\det E = -\cos\theta$. Then repeat for the order $R = R_z(\psi) R_x(\phi) R_y(\theta)$ and find where it is singular.
:::

::: {.exercise #exr-two-labs}
Make the two-labs experiment concrete. Pick $R$, $R_d$ and a world-frame change $Y = R_x(0.9)$. Compute $\Theta - \Theta_d$ before and after the change, and confirm that $e_R$ is unchanged.
:::

::: {.exercise #exr-unwinding}
Reproduce [](#ex-unwinding) in the notebook. How does the ratio of control effort between the two starts depend on $k_d$? Then implement the $\operatorname{sgn}(q_0)$ patch, add measurement noise, and start the body at 179° of error.
:::

::: {.exercise #exr-circle}
Write out the circle argument of [](#sec-obstruction) as a proof. Where does it use continuity? Give a discontinuous law on the circle that is globally asymptotically stable, and say what goes wrong with it under noise.
:::

::: {.exercise #exr-error-identities}
Prove [](#eq-error-angle) using Rodrigues' formula $R_d\T R = I + \sin\theta\, a^\wedge + (1 - \cos\theta)(a^\wedge)^2$ of [](#eq2-rodrigues). Deduce $\|e_R\|^2 = \Psi_R(2 - \Psi_R)$, then prove [](#eq-psi-dot).
:::

::: {.exercise #exr-antipode}
For the sphere, show that $\omega = -k\, e_q$ gives $\dot\theta = -k \sin\theta$, where $\theta$ is the angle between $q$ and $q_d$. Conclude that $-q_d$ is an unstable equilibrium. How long does it take to get from $\theta = 179^\circ$ to $\theta = 1^\circ$?
:::

## Notes and sources {#sec-ch1-notes .unnumbered}

The comparison of Euler-angle and geometric formulations follows Siravuru (2019, ch. 4), who carries it out for a 3D pendulum over a grid of initial attitudes. Chaturvedi, Sanyal and McClamroch (2011) is the best single survey of rigid-body attitude control and of the pitfalls described here. The error functions [](#eq-error-so3) are those of Bullo and Lewis (2005) and Lee, Leok and McClamroch (2010).

- S. P. Bhat and D. S. Bernstein, ["A topological obstruction to continuous global stabilization of rotational motion and the unwinding phenomenon"](https://doi.org/10.1016/S0167-6911%2899%2900090-0), *Systems & Control Letters* 39(1), 2000.
- F. Bullo and A. D. Lewis, [*Geometric Control of Mechanical Systems*](https://doi.org/10.1007/978-1-4899-7276-7), Springer, 2005.
- N. A. Chaturvedi, A. K. Sanyal and N. H. McClamroch, ["Rigid-body attitude control"](https://doi.org/10.1109/MCS.2011.940459), *IEEE Control Systems Magazine* 31(3), 2011.
- D. E. Koditschek, ["The application of total energy as a Lyapunov function for mechanical control systems"](https://repository.upenn.edu/ese_papers/672/), *Contemporary Mathematics* 97, 1989.
- T. Lee, M. Leok and N. H. McClamroch, ["Geometric tracking control of a quadrotor UAV on SE(3)"](https://doi.org/10.1109/CDC.2010.5717652), *IEEE Conference on Decision and Control*, 2010.
- C. G. Mayhew, R. G. Sanfelice and A. R. Teel, ["Quaternion-based hybrid control for robust global attitude tracking"](https://doi.org/10.1109/TAC.2011.2108490), *IEEE Transactions on Automatic Control* 56(11), 2011.
- M. W. Mueller, ["Multicopter attitude control for recovery from large disturbances"](https://arxiv.org/abs/1802.09143), arXiv:1802.09143, 2018.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- J. Stuelpnagel, ["On the parametrization of the three-dimensional rotation group"](https://doi.org/10.1137/1006093), *SIAM Review* 6(4), 1964.
