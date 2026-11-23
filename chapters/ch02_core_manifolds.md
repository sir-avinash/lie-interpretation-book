---
title: The manifolds robots live on
short: Manifolds
id: ch-manifolds
summary: Circles, spheres, rotations and rigid motions. What a tangent vector is, why rotations form a group and directions do not, and how to take a time step without falling off.
notebook: notebooks/ch02_lie_groups_product_manifolds.ipynb
verify: verify/ch02.py
---

A quadrotor carries a load on a cable one meter long. Your simulator stores the cable's direction as a unit vector $q$ and its angular velocity as $\omega$. The load swings steadily at 2 rad/s. Every hundredth of a second the simulator updates $q \leftarrow q + h\,(\omega \times q)$. After ten seconds, how long is the cable?

One meter, surely. The simulator disagrees. It reports $\|q\| = 1.22$, and the load hangs 22 cm further away than the cable allows.

The arithmetic is not at fault, and a smaller step only slows the growth. Each update moves $q$ in a direction that is correct at that instant, and still lands off the sphere. To see why, and to find an update that cannot do this, we need to know what kind of object $q$ is, and what kind of object its velocity is. They are not the same kind.

## Curved sets and their velocities {#sec2-tangent}

Start with the heading that opened [](#ch-why). A heading is a point on the unit circle,

$$
S^1 = \{\, q \in \R^2 : q \cdot q = 1 \,\}.
$$

Two numbers, one constraint, one degree of freedom. Let the heading move along a curve $q(t)$ that stays on the circle, and differentiate the constraint:

$$
\ddt\,(q\cdot q) = 2\, q \cdot \dot q = 0 .
$$

The velocity is perpendicular to the position. At each point the allowed velocities form a line, the tangent line there. With $\Lambda = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$, the quarter turn of the plane, each one is $\dot q = \omega\, \Lambda q$ for a single number $\omega$.

The cable direction lives one dimension up, on the sphere $S^2 = \{ q \in \R^3 : q\cdot q = 1\}$. The same differentiation gives the same conclusion, $q \cdot \dot q = 0$. The allowed velocities at $q$ now fill a plane ([](#fig2-tangent)).

![The tangent plane to the sphere at $q$. The teal curve $q(t)$ stays on the sphere, and its velocity $\dot q$ lies in the plane perpendicular to $q$. So does the angular velocity $\omega = q \times \dot q$, which is $\dot q$ turned a quarter turn about the normal.](figures/ch02-tangent.svg){#fig2-tangent}

::: {.thought #thought2-horizon name="Standing on a planet"}
Stand on a small, perfectly round planet. You can walk north, east, or any mixture. You cannot walk *up*. Every velocity available to you lies in the plane of your horizon.

Fly to another spot and the horizon tilts with you. A walker near the pole and a walker on the equator both say "I am heading north at 1 m/s". Those are different arrows in space, lying in different planes.
:::

The circle and the sphere are instances of one idea.

::: {.definition #def2-manifold name="Manifold and tangent space"}
Let $c : \R^n \to \R^k$ be a smooth function whose Jacobian $c'(x)$ has rank $k$ at every point of the set
$$
\mathcal{M} = \{\, x \in \R^n : c(x) = 0 \,\}.
$$
Then $\mathcal{M}$ is a **manifold** of dimension $n - k$: near each of its points it looks like a piece of flat $\R^{n-k}$. A **tangent vector** at $x \in \mathcal{M}$ is the velocity $\dot x(0)$ of a curve $x(t)$ that stays in $\mathcal{M}$ and passes through $x$ at $t = 0$. The tangent vectors at $x$ form the **tangent space**
$$
T_x\mathcal{M} = \{\, v \in \R^n : c'(x)\, v = 0 \,\},
$$
a vector space of dimension $n - k$.
:::

For the sphere, $c(q) = q\cdot q - 1$ and $c'(q) = 2q\T$, which is never zero on the sphere. So $n = 3$, $k = 1$, the dimension is 2, and $T_qS^2 = \{ v : q \cdot v = 0 \}$, the **tangent plane** at $q$.

The definition draws a line that code tends to blur. A configuration is a point of a curved set. A velocity is a vector in a flat space attached to that point. Adding one to the other, as the simulator did, gives something that belongs to neither.

On the sphere, every tangent vector at $q$ is a cross product:

$$
\dot q = \omega \times q, \qquad \omega \cdot q = 0, \qquad \omega = q \times \dot q . \label{eq2-s2-kinematics}
$$

The vector $\omega$ is the **angular velocity** of the direction. A component of $\omega$ along $q$ would spin the cable about its own axis, and $\omega \times q$ does not see it, so we set it to zero. Note the count: three numbers in $\omega$, one constraint, two degrees of freedom. On the sphere the constraint cannot be shaken off. [](#sec2-sphere) explains why.

Rotations are harder to picture. The same differentiation works on them, and it turns up something the sphere does not have.

## Rotations: a manifold you can multiply {#sec2-so3}

The rotation group of [](#def-so3) fits [](#def2-manifold). A matrix has nine entries. $R\T R = I$ is a symmetric matrix equation, so it is six constraints, and $9 - 6 = 3$. The condition $\det R = +1$ removes no dimension. It selects one of two disconnected pieces.

For the tangent space, differentiate $R\T R = I$ along a curve:

$$
\dot R\T R + R\T \dot R = 0 .
$$

So $R\T \dot R$ is skew-symmetric. By [](#def-hat) it is $\Omega^\wedge$ for some $\Omega \in \R^3$, and

$$
T_R\SO(3) = \{\, R\,\Omega^\wedge : \Omega \in \R^3 \,\}. \label{eq2-tangent-so3}
$$

So the kinematics $\dot R = R\,\Omega^\wedge$ of [](#eq-kinematics) are not a modeling assumption. They say that $\dot R$ is tangent to $\SO(3)$.

Compare [](#eq2-tangent-so3) with the sphere. At the identity the tangent space is $\so(3)$. At any other $R$ it is the same space multiplied by $R$. One fixed copy of $\R^3$ labels the velocities at every attitude, with no constraint on $\Omega$. Multiplication is the reason: composing with $R$ carries the neighborhood of the identity onto the neighborhood of $R$, velocities included.

::: {.definition #def2-lie-group name="Lie group and Lie algebra"}
A **Lie group** is a manifold that is also a group, with smooth multiplication and inversion. Its **Lie algebra** is the tangent space at the identity: the velocities with which a curve can leave the identity.
:::

The circle is a Lie group too. Identify the point at angle $\theta$ with the planar rotation by $\theta$ in $\SO(2)$, and multiplication adds angles. Its Lie algebra is the line $\{\omega \Lambda\}$. The three-sphere $S^3$ of unit quaternions from [](#sec-quaternions) is a third: four numbers, one constraint, quaternion multiplication, and a Lie algebra of pure quaternions $(0, x)$, $x \in \R^3$. It covers $\SO(3)$ twice, and its kinematics $\dot\quat = \tfrac12\,\quat\otimes(0,\Omega)$ are [](#eq-kinematics) one level up.

### Two velocities for one motion {#sec2-body-spatial}

There are two ways to carry a velocity back to the identity. Multiply $\dot R$ by $R\T$ on the left, or on the right:

$$
\dot R = R\,\Omega^\wedge = \omega^\wedge R, \qquad \omega = R\,\Omega . \label{eq2-spatial}
$$

$\Omega$ is the **body angular velocity**: the spin resolved along the body's own axes, which is what a gyroscope reports. $\omega$ is the **world-frame angular velocity**: the same spin resolved along the world axes. (The robotics literature also calls it the spatial angular velocity.) One physical arrow, two frames. Passing between the two forms of [](#eq2-spatial) uses an identity worth memorizing:

$$
R\, x^\wedge R\T = (R\,x)^\wedge \qquad \text{for all } R \in \SO(3),\ x \in \R^3 . \label{eq2-adjoint}
$$

The left side changes the frame of the matrix $x^\wedge$ by conjugation. The right side changes the frame of the vector $x$. They agree. Conjugation by a group element, acting on the Lie algebra, is the **adjoint**, $\Ad_R\, x^\wedge = R\,x^\wedge R\T$. On $\SO(3)$, in vector form, $\Ad_R$ is just the matrix $R$. The adjoint is the change of frame for velocities.

::: {.thought #thought2-turntable name="A phone on a turntable"}
A phone's axes are $x$ across the screen, $y$ along its long side and $z$ out of the screen. Lay it flat on a turntable that spins at 0.5 rad/s about the vertical. Its gyroscope reads $\Omega = (0, 0, 0.5)$: all of the spin is along the axis that comes out of the screen. Now stand the phone upright, long axis vertical, and spin the turntable again. The gyroscope reads $\Omega = (0, 0.5, 0)$.

A camera on the ceiling sees no difference. In the world frame the angular velocity is $\omega = (0, 0, 0.5)$ both times. The phone changed its frame, and $\Omega = R\T\omega$ changed with it.
:::

We can now describe a rotation in progress. We cannot yet say where it ends up. If the gyroscope reads a constant $\Omega$ for $t$ seconds, what is $R$?

## From a velocity to a rotation: exp and log {#sec2-exp}

With $\Omega$ constant, $\dot R = R\,\Omega^\wedge$ is a linear differential equation with constant coefficients. Its solution is a matrix exponential:

$$
R(t) = R(0)\,\exp(t\,\Omega^\wedge), \qquad \exp(A) = I + A + \tfrac{1}{2!}A^2 + \tfrac{1}{3!}A^3 + \cdots
$$

Write $t\,\Omega = \theta a$ with $a$ a unit vector. Since $(a^\wedge)^3 = -a^\wedge$, the powers cycle, and the series sums to a sine and a cosine.

::: {.proposition #prop2-rodrigues name="Exponential and logarithm on $\SO(3)$"}
For a unit vector $a$ and an angle $\theta$,
$$
\exp(\theta\, a^\wedge) = I + \sin\theta\; a^\wedge + (1 - \cos\theta)\,(a^\wedge)^2 . \label{eq2-rodrigues}
$$
This is the rotation by $\theta$ about $a$. Conversely, a rotation $R$ with $\tr R \neq -1$ is $\exp(\theta a^\wedge)$ for exactly one $\theta a$ with $\theta < \pi$, given by
$$
\theta = \arccos\!\left(\frac{\tr R - 1}{2}\right), \qquad
\theta\, a = \log(R)^\vee = \frac{\theta}{2\sin\theta}\,\bigl(R - R\T\bigr)^\vee , \label{eq2-log}
$$
with the factor $\theta / (2\sin\theta)$ read as $\tfrac12$ at $\theta = 0$.
:::

Equation [](#eq2-rodrigues) is Rodrigues' formula. The map $\exp$ wraps the flat space $\so(3)$ onto the curved group. [](#fig2-expmap) shows the same wrapping where you can see it, on the circle and the sphere. On the sphere the formula is

$$
\exp_q(v) = \cos\|v\|\; q + \sin\|v\|\; \frac{v}{\|v\|}, \qquad v \in T_qS^2 :
$$

leave $q$ in the direction $v$ and follow the great circle for a distance $\|v\|$.

![The exponential map wraps a tangent space onto the manifold, keeping lengths along each ray from the base point. Left: the tangent line at the identity of $\SO(2)$. A segment of length $\theta$ lands on the arc of length $\theta$. Right: a tangent vector $v$ at $q \in S^2$ lands on the great-circle arc of length $\|v\|$ that leaves $q$ in the direction of $v$.](figures/ch02-expmap.svg){#fig2-expmap}

::: {.example #ex2-exp name="A quarter turn about a diagonal"}
A body starts at $R = I$ and turns at $\Omega = \tfrac{\pi}{2}\,(1, 1, 0)/\sqrt2$ rad/s for one second. Its axis is $a = (1,1,0)/\sqrt2$ and its angle is $\theta = \pi/2$, so $\sin\theta = 1$ and $1 - \cos\theta = 1$. Equation [](#eq2-rodrigues) gives
$$
R = I + a^\wedge + (a^\wedge)^2 =
\begin{bmatrix} 0.5 & 0.5 & 0.707 \\ 0.5 & 0.5 & -0.707 \\ -0.707 & 0.707 & 0 \end{bmatrix}.
$$
The third column says the body's $e_3$ axis now points along $(0.707, -0.707, 0)$. Going back, $\tr R = 1$ gives $\theta = \arccos 0 = \pi/2$, and $\tfrac{\theta}{2}(R - R\T)^\vee = (1.111, 1.111, 0) = \theta a$.
:::

You have met the image of $\log$ already. It is the solid ball of [](#fig-so3-ball): each rotation plotted at its rotation vector $\theta a$. That ball *is* the Lie algebra, cut off at radius $\pi$. At $\theta = \pi$ the formula [](#eq2-log) divides by $\sin\pi = 0$, and $\exp(\pi a^\wedge) = \exp(-\pi a^\wedge)$, so the logarithm has two answers. The flat algebra matches the curved group near the identity and fails half a turn away. Any method that works in logarithm coordinates inherits that limit.

There is a second limit. If rotations behaved like translations, $\exp(x^\wedge)\exp(y^\wedge)$ would equal $\exp((x + y)^\wedge)$ and rotation vectors would add. They do not.

## Order matters: the bracket {#sec2-bracket}

::: {.thought #thought2-book name="A book, two quarter turns"}
Lay a book on the desk, cover up, spine on your left. Fix three axes to the desk: $x$ to your right, $y$ away from you, $z$ up. Turn the book 90° about $x$, then 90° about $z$. It ends up standing like a book on a shelf, cover facing right, spine upright.

Start again, and turn 90° about $z$ first, then 90° about $x$. Now the cover faces you and the spine lies level, underneath ([](#fig2-noncommute)). Same two turns, different attitude. The two results are 120° apart.
:::

![The same two quarter turns in two orders. The dark block marks the top of the front cover and the tan face is the spine. The final attitudes are 120° apart.](figures/ch02-noncommute.svg){#fig2-noncommute}

Quarter turns are large. Now go small. Rotate by an angle $\epsilon$ about a unit axis $a$, then about another unit axis $b$, then undo the first, then undo the second. If rotations commuted, you would be back at the start. Instead:

$$
\exp(\epsilon a^\wedge)\,\exp(\epsilon b^\wedge)\,\exp(-\epsilon a^\wedge)\,\exp(-\epsilon b^\wedge)
= \exp\!\bigl(\epsilon^2\,(a \times b)^\wedge + O(\epsilon^3)\bigr). \label{eq2-commutator}
$$

A rotation is left over. Its axis is $a \times b$, a direction you never turned about, and its size is second order. With $a = e_1$, $b = e_2$ and $\epsilon = 0.1$ rad (about 5.7°), the leftover is a rotation of 0.0100 rad, or 0.57°, almost exactly about $e_3$.

The leftover is measured by the matrix commutator, and on $\so(3)$ that is an old friend:

$$
[x^\wedge, y^\wedge] = x^\wedge y^\wedge - y^\wedge x^\wedge = (x \times y)^\wedge . \label{eq2-bracket}
$$

This is the **Lie bracket**. With it, $\so(3)$ is $\R^3$ with the cross product. The bracket is what survives in the flat algebra of the group's failure to commute. It is why rotation vectors do not add: for small $x$ and $y$,

$$
\log\bigl(\exp(x^\wedge)\exp(y^\wedge)\bigr)^\vee = x + y + \tfrac12\, x \times y + \cdots
$$

On $\SO(2)$ the bracket is zero. Planar rotations commute, angles add, and planar robots are spared most of this book's trouble. In three dimensions the bracket is everywhere, starting with the term $\Omega \times J\Omega$ in Euler's equation [](#eq4-euler-free), where $J$ is the inertia matrix.^[The bracket is the derivative of the adjoint: $\ddt \Ad_{\exp(t x^\wedge)}\, y\,\big|_{t=0} = x \times y$. This map is written $\ad_x\, y$.]

Rotations have a multiplication, an algebra and a bracket. The cable direction lives on a sphere. Can the sphere have them too?

## The sphere is not a group {#sec2-sphere}

It cannot, and the reason is a fact about hair.

::: {.thought #thought2-hair name="Combing a group"}
On a Lie group, pick any nonzero velocity $\zeta$ at the identity. Multiply it by each group element $X$ to get a velocity $X\zeta$ at $X$. Since $X$ is invertible, $X\zeta$ is never zero. You have covered the whole group with a smooth field of arrows that vanishes nowhere. On the circle this is the field "go counterclockwise at unit speed".

Now try it on a sphere. Comb a hairy ball flat. However you do it, you get a cowlick: a point where the hair has no direction ([](#fig2-comb)). This is the hairy ball theorem. Every continuous field of tangent vectors on $S^2$ is zero somewhere. So no multiplication on $S^2$ can make it a Lie group.^[Among spheres, only $S^0$, $S^1$ and $S^3$ are Lie groups: the signs $\pm1$, the unit complex numbers and the unit quaternions.]
:::

![Combing a group and combing a sphere. Left: on the circle, multiplication carries one arrow to every point, and none of the copies is zero. Right: every way of combing the sphere leaves a cowlick, a point where the field vanishes. Changing the combing moves the cowlicks and can merge two into one. It cannot remove the last.](figures/ch02-comb-strip.png){#fig2-comb anim="figures/ch02-comb.gif"}

The sphere has the next best thing: $\SO(3)$ **acts** on it: a rotation $R$ moves the direction $q$ to $Rq$, and some rotation takes any direction to any other. A manifold with a group acting on it this way is a **homogeneous space**. Every direction is $q = R\,e_3$ for some $R$, but not for a unique one. Rotations about $e_3$ leave $e_3$ alone, so $R$ and $R\,R_z(\alpha)$ give the same $q$. The sphere is the rotation group with a circle's worth of information forgotten, and the dimensions agree: $3 - 1 = 2$.

The kinematics come along for free. If $q = R\,b$ for a direction $b$ fixed in the body, then by [](#eq2-spatial)

$$
\dot q = \dot R\, b = \omega^\wedge R\, b = \omega \times q .
$$

That is [](#eq2-s2-kinematics), with $\omega$ the world-frame angular velocity. The component of $\omega$ along $q$ is the forgotten circle.

Here is what the missing group structure costs.

**No unconstrained velocity coordinates.** On $\SO(3)$, $\Omega \in \R^3$ is free. On $S^2$ that would need two basis vectors in every tangent plane, chosen continuously, and the hairy ball forbids even one. So we carry three numbers and the constraint $\omega\cdot q = 0$.

**No relative element.** On $\SO(3)$ the error between $R$ and $R_d$ is the group element $R_d\T R$. On $S^2$ there is no "$q_d^{-1}q$". Errors are built from dot and cross products instead, as in [](#def-error-s2).

**No free comparison of velocities.** On a group, every velocity can be carried to the identity and compared there. On the sphere, $\dot q$ and $\dot q_d$ lie in different planes, like the two walkers of [](#thought2-horizon). Comparing them needs an extra choice, which [](#sec3-transport) makes.

None of this is a defect of $S^2$ as a model. A cable has no twist, and modeling it on $\SO(3)$ would carry a degree of freedom that does nothing.

## Position and attitude together {#sec2-se3}

A rigid body has an attitude $R$ and a position $x \in \R^3$. Pack them into one matrix that maps body coordinates $y$ to world coordinates $Ry + x$:

$$
X = \begin{bmatrix} R & x \\ 0 & 1 \end{bmatrix} \in \SE(3), \qquad
X_1 X_2 = \begin{bmatrix} R_1R_2 & x_1 + R_1 x_2 \\ 0 & 1 \end{bmatrix}. \label{eq2-se3}
$$

This is the **special Euclidean group**. It stores twelve numbers under six constraints and has dimension six. Its Lie algebra $\se(3)$ consists of **twists**. We write a twist as $\zeta = (V, \Omega) \in \R^6$, linear part first, with

$$
\zeta^\wedge = \begin{bmatrix} \Omega^\wedge & V \\ 0 & 0 \end{bmatrix}, \qquad
\dot X = X\,\zeta^\wedge \iff \dot R = R\,\Omega^\wedge,\quad \dot x = R\,V . \label{eq2-se3-kin}
$$

So the body twist is $\zeta = (R\T v,\ \Omega)$: linear velocity $v = \dot x$ and angular velocity, both in body axes. The adjoint again converts it to the world frame, and is now more than a rotation:

$$
X\,\zeta^\wedge X^{-1} = (\Ad_X\, \zeta)^\wedge, \qquad
\Ad_X = \begin{bmatrix} R & x^\wedge R \\ 0 & R \end{bmatrix}. \label{eq2-ad-se3}
$$

The block $x^\wedge R$ is the lever arm. A body spinning about its own center, seen from a frame a distance $x$ away, also appears to translate.

::: {.remark name="$\SE(3)$ or $\R^3 \times \SO(3)$?"}
As sets they are the same. They differ in how elements combine: in [](#eq2-se3) the rotation $R_1$ acts on $x_2$, and in the plain product positions simply add. The quadrotor controllers in this book keep $x$ and $v$ in the world frame, which is the product structure. Estimators often prefer $\SE(3)$, because their measurements arrive in the body frame. The manifold and its dimension are the same either way.
:::

One relative deserves a paragraph. An inertial navigation state has an attitude, a velocity and a position. Put $R$ in the corner of a $5\times5$ matrix, with $v$ and $x$ as two extra columns. These matrices form a nine-dimensional Lie group, $\SE_2(3)$. Hartley et al. (2018) use it to estimate the state of the biped Cassie from an inertial unit and leg kinematics. On this group the estimation error evolves independently of the trajectory being estimated, which is the reason for the construction.

## Real robots are products {#sec2-products}

The configuration of a multi-part robot is a list: this body's attitude, that link's direction, that slider's extension. The set of such lists is a **product manifold**. Its dimension is the sum of its factors' dimensions, and a tangent vector is a list of tangent vectors, one per factor. Nothing new is needed. You stack the kinematic equations.

![A quadrotor with a cable-suspended load. Three pieces describe every configuration: the load position $x_L$, the cable direction $q$ and the quadrotor attitude $R$. The quadrotor position is not a separate coordinate. It follows from $x_Q = x_L - l\,q$.](figures/ch02-quadload.svg){#fig2-quadload}

Take the system of the opening puzzle ([](#fig2-quadload)). While the cable is taut, its configuration is

$$
(x_L,\ q,\ R) \in \R^3 \times S^2 \times \SO(3),
$$

with $q$ pointing from the quadrotor to the load, a convention kept throughout the book.^[Using the quadrotor's position as the coordinate instead gives the same manifold written as $\SE(3) \times S^2$.] That is $3 + 3 + 9 = 15$ stored numbers, $1 + 6 = 7$ constraints and 8 degrees of freedom. With four inputs, a thrust and three torques, four of them are unactuated. The kinematics are three lines we already have:

$$
\dot x_L = v_L, \qquad \dot q = \omega \times q, \qquad \dot R = R\,\Omega^\wedge .
$$

::: {.example #ex2-quadload name="Where is the quadrotor, and how fast?"}
A load at $x_L = (2, 0, 1)$ m moves at $v_L = (1, 0, 0)$ m/s under a cable of length $l = 1$ m that hangs straight down, so $q = -e_3$. The cable swings with $\omega = (0.5, 0, 0)$ rad/s. The quadrotor is at
$$
x_Q = x_L - l\,q = (2, 0, 2),
$$
and differentiating with [](#eq2-s2-kinematics),
$$
\dot x_Q = v_L - l\,(\omega \times q) = (1, 0, 0) - (0, 0.5, 0) = (1, -0.5, 0)\ \text{m/s}.
$$
No angles were introduced, and the formula holds for every cable direction, including straight up.
:::

[](#tbl2-robots) collects the configuration manifolds used in this book. Every one is assembled from the same few factors.

::: {.table #tbl2-robots}

| Robot | Configuration manifold | Numbers stored | Dimension |
| :-- | :-- | :-: | :-: |
| Heading of a planar robot | $S^1 \cong \SO(2)$ | 2 | 1 |
| Spherical pendulum, cable, thrust axis | $S^2$ | 3 | 2 |
| 3D pendulum, attitude of a satellite | $\SO(3)$ | 9 | 3 |
| Quadrotor | $\SE(3)$ | 12 | 6 |
| Quadrotor with cable-suspended load | $\R^3 \times S^2 \times \SO(3)$ | 15 | 8 |
| $n$ quadrotors carrying a rigid payload | $\SE(3) \times (S^2 \times \SO(3))^n$ | $12 + 12n$ | $6 + 5n$ |
| Reaction Mass Biped, stance phase | $\SO(3)^3 \times [0, r]^2 \times [0, r_s]^3$ | 32 | 14 |
| Cassie, reduced geometric model | $\SE(3) \times \SO(3)^2 \times \R_+^2$ | 32 | 14 |

: Configuration manifolds of the robots in this book.

:::

Wu (2017) models $n$ quadrotors carrying one rigid payload: the payload's pose, then a cable direction and an attitude per vehicle. Siravuru (2019) models the Reaction Mass Biped with three rotation matrices (stance leg, torso, swing leg), two leg lengths and three pairs of sliding reaction masses. His reduced model of Cassie has a floating pelvis in $\SE(3)$, a rotation matrix per leg and two leg lengths, with eight of the fourteen degrees of freedom actuated. No joint angle appears in any of them.

::: {.warning #warn2-taut name="A configuration manifold is a modeling claim"}
The factor $S^2$ asserts that the cable is taut. If the tension reaches zero the load is in free fall and the configuration space becomes $\R^3 \times \R^3 \times \SO(3)$. A product manifold records the constraints you assumed. It does not enforce them on the hardware. Monitor the tension, and treat slack as a different mode with its own model.
:::

Now the opening puzzle can be stated precisely. The simulator's state lives on a product of spheres and rotation groups, and its update rule was written for $\R^n$.

## Stepping without falling off {#sec2-integration}

The simulator's update is Euler's method, $q_{k+1} = q_k + h\,\dot q_k$. On the sphere, $\dot q_k$ is perpendicular to $q_k$, so by Pythagoras

$$
\|q_{k+1}\|^2 = \|q_k\|^2 + h^2\,\|\dot q_k\|^2 .
$$

The norm grows at every step, by an amount that depends only on the speed. The tangent plane touches the sphere at one point and lies outside it everywhere else, so a step along the tangent always lands outside ([](#fig2-steps)). With $\|\omega\| = 2$ rad/s and $h = 0.01$ s, each step multiplies $\|q\|^2$ by $1.0004$. A thousand steps give $1.0004^{500} = 1.22$. That is the stretched cable.

![Four steps on a circle, exaggerated. Each Euler step (terracotta) runs along the tangent and lands outside, and the radius grows from 1 to 1.38. Each exponential step (indigo) follows the circle.](figures/ch02-steps.svg){#fig2-steps}

Rotation matrices fail the same way. One Euler step $R_{k+1} = R_k + h\,R_k\Omega^\wedge$ gives

$$
R_{k+1}\T R_{k+1} = I - h^2\,(\Omega^\wedge)^2, \qquad \det R_{k+1} = 1 + h^2\|\Omega\|^2 .
$$

The matrix is no longer orthogonal. Used as a rotation, it stretches the body.

The cure is in [](#sec2-exp). Over one step, hold the velocity fixed and solve the kinematics exactly. The solution is an exponential, and exponentials land on the manifold:

$$
R_{k+1} = R_k\,\exp(h\,\Omega_k^\wedge), \qquad q_{k+1} = \exp(h\,\omega_k^\wedge)\, q_k . \label{eq2-exp-step}
$$

The first is a product of rotations, so it is a rotation. The second is a rotation applied to a unit vector, so it is a unit vector. This holds for any step size, up to rounding. On a product manifold, apply [](#eq2-exp-step) factor by factor.

::: {.example #ex2-drift name="Ten seconds of tumbling"}
A body tumbles with $\Omega(t) = (\cos t,\ \sin 0.7t,\ 0.5)$ rad/s. Integrate from $R = I$ for 10 s with $h = 0.01$ s ([](#fig2-drift), left). After the Euler steps, $\|R\T R - I\| = 0.16$ in the Frobenius norm and $\det R = 1.13$. After the exponential steps it stays below $10^{-13}$.
:::

![Left: distance from $\SO(3)$ during the tumble of [](#ex2-drift). Euler steps drift away. Exponential steps stay on the manifold to rounding error. Right: the attitude error of the exponential step, with $\Omega$ sampled at the start of each step (first order) and at the midpoint (second order).](figures/ch02-drift.svg){#fig2-drift}

::: {.warning #warn2-accuracy name="On the manifold is not the same as correct"}
The exponential step is exact only if $\Omega$ is constant across the step. In [](#ex2-drift) it is not, and the final attitude is wrong by 0.29°, although it is a perfect rotation matrix ([](#fig2-drift), right). Halving $h$ halves the error. Sampling $\Omega$ at the middle of the step instead of the start cuts it to 0.0012° at no extra cost. Staying on the manifold removes one kind of error, not all of them. Nor does repairing an Euler step by normalizing $q$ or reorthonormalizing $R$: that restores the constraint and leaves the integration error.
:::

## What we have, and what is missing {#sec2-next}

A robot's configuration is a point on a product of a few standard manifolds, and its velocity is a list of tangent vectors. The exponential turns velocities into motion, the adjoint changes their frame, and the bracket records that order matters.

All of this concerns one trajectory. Control and estimation concern two: actual and desired, or true and estimated. We cannot yet subtract a desired velocity from an actual one when they live in different tangent spaces, or differentiate a function like $\Psi_R$ whose argument is a matrix under constraints. [](#ch-variations) does both with one tool, the variation of [](#def3-variation).

## Exercises {#sec2-exercises .unnumbered}

::: {.exercise #exr2-puzzle}
Reproduce the opening puzzle. Show that with constant $\|\omega\|$ the Euler iterate satisfies $\|q_N\| = (1 + h^2\|\omega\|^2)^{N/2}$, and that for a fixed duration $t_f = Nh$ this tends to $\exp(\tfrac12 h\|\omega\|^2 t_f)$. What step size keeps the stretch below 1 mm over a 10 s flight?
:::

::: {.exercise #exr2-tangent}
Apply [](#def2-manifold) to $\SO(3)$. Write the six constraint functions and show that their Jacobian has rank six at $R = I$ by exhibiting its kernel. Why is the rank six at every other $R$ too?
:::

::: {.exercise #exr2-rodrigues}
Sum the exponential series to prove [](#eq2-rodrigues). Derive [](#eq2-log) from its trace and its skew-symmetric part. What is $R - R\T$ at $\theta = \pi$, and how would you recover the axis there? Then prove [](#eq2-adjoint) by applying both sides to a vector. Where is $\det R = +1$ used?
:::

::: {.exercise #exr2-book}
Do the book experiment of [](#thought2-book) with your hands, then with matrices. Find the axis of the 120° rotation that separates the two results. Repeat with turns of 10° and compare the leftover with [](#eq2-commutator).
:::

::: {.exercise #exr2-count}
A quadrotor carries a load on a chain of $n$ rigid links joined by ball joints. Propose a configuration manifold. Count stored numbers, constraints and dimension, and write the kinematics. How many degrees of freedom are unactuated? What does your manifold assume?
:::

::: {.exercise #exr2-midpoint}
In the notebook, measure the attitude error of the exponential step in [](#ex2-drift) for $h = 0.02, 0.01, 0.005$, sampling $\Omega$ at the start and at the midpoint of each step. Confirm orders one and two. For which motions is the start-of-step version exact?
:::

## Notes and sources {#sec2-notes .unnumbered}

The treatment of $\SO(3)$ and $\SE(3)$, their exponentials and adjoints follows Murray, Li and Sastry (1994) and Solà, Deray and Atchuthan (2018). Teng et al. (2022, 2026) and Jang, Teng and Ghaffari (2023) summarize the same preliminaries as used in control and trajectory optimization. $S^2$ as a configuration manifold, and multibody systems as products of $S^2$ and $\SO(3)$, follow Lee, Leok and McClamroch (2018). The cable-suspended load and multi-quadrotor models are from Wu (2017, ch. 3). The 3D pendulum, Reaction Mass Biped and Cassie models are from Siravuru (2019, ch. 4 to 6). $\SE_2(3)$ is due to Barrau and Bonnabel (2017) and was applied to legged robots by Hartley et al. (2018).

- A. Barrau and S. Bonnabel, ["The invariant extended Kalman filter as a stable observer"](https://doi.org/10.1109/TAC.2016.2594085), *IEEE Transactions on Automatic Control* 62(4), 2017.
- R. Hartley, M. Ghaffari Jadidi, J. W. Grizzle and R. M. Eustice, ["Contact-aided invariant extended Kalman filtering for legged robot state estimation"](https://doi.org/10.15607/RSS.2018.XIV.050), *Robotics: Science and Systems*, 2018. arXiv:1805.10410.
- J. Jang, S. Teng and M. Ghaffari, ["Convex geometric trajectory tracking using Lie algebraic MPC for autonomous marine vehicles"](https://doi.org/10.1109/LRA.2023.3328450), *IEEE Robotics and Automation Letters* 8(12), 2023. arXiv:2305.09009.
- T. Lee, M. Leok and N. H. McClamroch, [*Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*](https://doi.org/10.1007/978-3-319-56953-6), Springer, 2018.
- R. M. Murray, Z. Li and S. S. Sastry, [*A Mathematical Introduction to Robotic Manipulation*](https://doi.org/10.1201/9781315136370), CRC Press, 1994.
- A. Siravuru, [*Geometric Control and Learning for Dynamic Legged Robots*](https://kilthub.cmu.edu/articles/thesis/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962), PhD thesis, Carnegie Mellon University, 2019.
- J. Solà, J. Deray and D. Atchuthan, ["A micro Lie theory for state estimation in robotics"](https://arxiv.org/abs/1812.01537), arXiv:1812.01537, 2018.
- S. Teng, W. Clark, A. Bloch, R. Vasudevan and M. Ghaffari, ["Lie algebraic cost function design for control on Lie groups"](https://arxiv.org/abs/2204.09177), arXiv:2204.09177, 2022.
- S. Teng, R. Zhang, T.-Y. Lin, W. A. Clark, M. Mueller, R. Vasudevan, M. Ghaffari and K. Sreenath, ["LieIPM: Lie group interior point method for direct trajectory optimization of rigid bodies"](https://arxiv.org/abs/2606.10579), arXiv:2606.10579, 2026.
- G. Wu, [*Safety-critical Geometric Control Design with Application to Aerial Transportation*](https://kilthub.cmu.edu/articles/thesis/Safety-critical_Geometric_Control_Design_with_Application_to_Aerial_Transportation/6721382), PhD thesis, Carnegie Mellon University, 2017.
