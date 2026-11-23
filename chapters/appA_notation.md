---
title: Notation and glossary
short: Notation and glossary
id: app-notation
summary: Every symbol, convention and technical term of the book, with the place where each is defined.
---

This appendix is the reference for the whole book. Each symbol means one thing, each term has one name, and every chapter, figure, notebook and skill follows the tables below. Read [](#secA-conventions) once. Use [](#secA-symbols) to look up a symbol and [](#secA-glossary) to look up a word. The last column of each table points to the place where the entry is introduced.

A few letters are local and are not listed: $x, y, z$ for arbitrary vectors in an identity, $A$ for an arbitrary matrix, $i, j, k$ for indices, $n$ and $N$ for counts. So are $\Pi$ for a projection, $\beta$ and $\sigma$ in two exercises, and $c_i$ for the entries of a matrix. They are named where they are used and mean nothing beyond that sentence.

## Conventions {#secA-conventions}

**Frames.** The **world frame** does not move, with $e_3$ pointing up. The **body frame** is attached to a rigid body. A rotation matrix $R$ maps body-frame components to world-frame components.

**Gravity.** Gravity acts along $-e_3$. A mass $m$ at position $x$ has potential energy $m g\,e_3\cdot x$ and weight $-m g\,e_3$.

**Body and world velocities.** The angular velocity of a rigid body is $\Omega$, expressed in the body frame, with $\dot R = R\,\Omega^\wedge$. Its world-frame components are $\omega = R\,\Omega$. The angular velocity of a direction $q \in S^2$ is $\omega$, in the world frame, with $\dot q = \omega\times q$ and $\omega\cdot q = 0$. Positions $x$ and velocities $v = \dot x$ are in the world frame.

**Hat and vee.** The hat map is written after its argument: $x^\wedge y = x\times y$ and $(x^\wedge)^\vee = x$. On a twist, $\zeta^\wedge$ is the $4\times4$ matrix of [](#eq2-se3-kin).

**Twists.** A twist is ordered linear part first, $\zeta = (V, \Omega)$, both parts in the body frame.

**Cable direction.** For a cable-suspended load, $q$ points from the quadrotor to the load, so $x_Q = x_L - l\,q$. The cable tension $T$ is positive when the cable pulls.

**Desired quantities.** The subscript $d$ marks the configuration or motion that a controller aims for: $R_d$, $q_d$, $x_d$, $\Omega_d$, $\omega_d$. A fixed one is called the target. A moving one is called the reference. No other mark ("ref", "des", a star) is used.

**Order of arguments.** The actual configuration comes first and the desired one second, in formulas and in code: $\Psi_R$ is built from $R_d\T R$, and a function is called as `e_R(R, Rd)`.

**Variations.** A variation is written with $\delta$. On $\SO(3)$ it is $\delta R = R\,\eta^\wedge$ with $\eta$ in the body frame and unconstrained. On $S^2$ it is $\delta q = \xi\times q$ with $\xi\cdot q = 0$. The small parameter of a family of curves is $\epsilon$.

**Velocity variation on the sphere.** With $\omega\cdot q = 0$ and $\xi\cdot q = 0$ kept along the whole curve, $\delta\omega = \dot\xi - 2\,\omega\times\xi$. This vector is not tangent: its tangent part is $\dot\xi - \omega\times\xi$ and its part along $q$ is $-\omega\times\xi$. See [](#warn3-s2-convention).

**Equations on the sphere.** An equation of motion for $q \in S^2$ holds only after projection by $I - q q\T$, together with $q\cdot\dot\omega = 0$.

**Signs of derived equations.** Chapters write equations of motion with the inertia on the left, as in $J\dot\Omega + \Omega\times J\Omega = \tau$. The symbolic library of [](#ch-symbolic) returns the coefficient of each variation, $E_\xi = 0$, with the inertia term negative, and its standard form is $M\,\dot v + b + G\,u = 0$.

**Transposes and products.** The transpose is $R\T$. The dot product of two vectors is $a\cdot b$, never $a\T b$. A transpose appears only in matrix expressions such as $q q\T$ and $s\T Q\,s$. A norm is $\|x\|$. The identity is $I$, with a subscript only when its size is not three.

**Angles and units.** Angles in formulas are in radians. Degrees are used in prose and on plot axes, written 90° in text and $90^\circ$ in math. Units are SI and upright: rad/s, m/s², kg·m², N·m, N·s, s⁻¹. A percentage is written 5%.

**Spelling.** American spelling: linearization, modeling, stabilize, center, behavior, maneuver, neighbor.

**Names in code.** Notebooks, skills and scripts use `hat`, `vee`, `expm_so3`, `log_so3`, `Psi_R`, `Psi_q`, `e_R`, `e_Omega`, `e_q`, `e_omega`, and write desired quantities as `Rd`, `qd`, `Omegad`, `omegad`. Outside the book the same notation is written in plain LaTeX, for example `R_d^\top R`.

**Colors in figures.** Ink is structure. Indigo is the main object, the reference or the method that works. Terracotta is the contrasting object, the error or the method that fails. Teal is a third role. The trace error is always indigo and the logarithm error always teal.

## Symbols {#secA-symbols}

### Spaces and groups

::: {.table #tblA-spaces}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $\R^n$ | Real $n$-dimensional space | |
| $S^1$, $S^2$, $S^3$ | Circle, sphere of directions, sphere of unit quaternions | [](#sec2-tangent), [](#sec2-so3) |
| $\SO(3)$, $\SO(2)$ | Rotation groups | [](#def-so3) |
| $\so(3)$ | Skew-symmetric matrices, the Lie algebra of $\SO(3)$ | [](#def-hat) |
| $\SE(3)$, $\se(3)$ | Rigid motions and twists | [](#eq2-se3) |
| $\SE_2(3)$ | Group of attitude, velocity and position | [](#sec2-se3) |
| $\mathcal{M}$ | A generic manifold | [](#def2-manifold) |
| $T_x\mathcal{M}$, $T_qS^2$, $T_R\SO(3)$ | Tangent space at a point | [](#def2-manifold) |
| $X$, $X_d$ | Element of a matrix Lie group, and its desired value | [](#eq2-se3), [](#def3-invariant-errors) |
| $Y$ | A fixed group element that changes the frame | [](#thought-two-labs) |

: Spaces and groups.

:::

### Configuration and velocity

::: {.table #tblA-config}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $R$ | Attitude, body to world. $\hat R$ is an estimate of it | [](#def-so3) |
| $\Omega$ | Body angular velocity, $\dot R = R\,\Omega^\wedge$ | [](#eq-kinematics) |
| $q$ | Direction in $S^2$. The letter $q$ alone is always a direction | [](#eq2-s2-kinematics) |
| $\omega$ | World-frame angular velocity: of a direction ($\dot q = \omega\times q$, $\omega\cdot q = 0$) or of a body ($\omega = R\,\Omega$) | [](#eq2-s2-kinematics), [](#eq2-spatial) |
| $x$, $v$ | Position and velocity $v = \dot x$ in the world frame. In [](#eq5-standard), $v$ is the stack of all velocities | [](#sec2-se3) |
| $x_L$, $x_Q$, $v_L$ | Load position, quadrotor position, load velocity | [](#sec2-products) |
| $\zeta = (V, \Omega)$ | Twist, or body velocity on a matrix Lie group, $\dot X = X\,\zeta^\wedge$ | [](#eq2-se3-kin) |
| $\quat = (q_0, q_v)$ | Unit quaternion, scalar part first | [](#eq-quat-to-R) |
| $\Theta = (\psi, \theta, \phi)$ | Yaw, pitch and roll angles | [](#eq-zyx) |
| $E$ | Euler-angle Jacobian, $\Omega = E\,\dot\Theta$ | [](#eq-euler-jacobian) |
| $\theta$, $a$ | Angle and unit axis of a rotation. $\theta a$ is the rotation vector. Also the angle between $q$ and $q_d$. Inside Euler-angle formulas only, $\theta$ is the pitch | [](#sec-rotation-space), [](#prop2-rodrigues) |
| $e_1, e_2, e_3$ | Standard basis. $e_3$ points up | |
| $\Lambda$ | Quarter turn of the plane | [](#sec2-tangent) |
| $t$, $t_f$, $h$ | Time, final time, time step | [](#sec4-hamilton), [](#sec2-integration) |
| $l$ | A length: cable, rod or leg | [](#sec2-products) |

: Configurations and velocities.

:::

### Variations

::: {.table #tblA-variations}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $\delta$ | Variation: derivative across a family of curves at $\epsilon = 0$ | [](#def3-variation) |
| $\epsilon$ | Small parameter of the family | [](#def3-variation) |
| $\delta x$ | Variation of a position | [](#def3-variation) |
| $\eta$ | Variation on $\SO(3)$, $\delta R = R\,\eta^\wedge$ | [](#def3-variation) |
| $\xi$ | Variation on $S^2$, $\delta q = \xi\times q$, $\xi\cdot q = 0$ | [](#def3-variation) |
| $\delta\Omega$ | $\dot\eta + \Omega\times\eta$ | [](#eq3-delta-Omega) |
| $\delta\omega$ | $\dot\xi - 2\,\omega\times\xi$ | [](#eq3-delta-omega) |

: Variations.

:::

### Errors and transport

::: {.table #tblA-errors}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $R_d$, $q_d$, $x_d$, $\Omega_d$, $\omega_d$ | Desired configuration and velocity | [](#def-error-so3) |
| $\Psi$ | Error function. $\Psi_R = \tfrac12\tr(I - R_d\T R)$, $\Psi_q = 1 - q\cdot q_d$ | [](#def-error-so3), [](#def-error-s2), [](#def3-error-vector) |
| $e_R$ | Attitude error vector, $\tfrac12(R_d\T R - R\T R_d)^\vee$ | [](#eq-error-so3) |
| $e_\Omega$ | Velocity error on $\SO(3)$, $\Omega - R\T R_d\,\Omega_d$ | [](#eq-error-so3) |
| $e_q$ | Direction error vector, $q_d\times q$ | [](#eq-error-s2) |
| $e_\omega$, $e_{\dot q}$ | Velocity errors on $S^2$, $e_\omega = \omega - (I - q q\T)\,\omega_d$ | [](#eq3-e-omega), [](#eq3-transport-s2) |
| $e_x$ | Position error, $x - x_d$ | [](#def3-error-vector) |
| $D = \operatorname{diag}(d_1, d_2, d_3)$ | Weight of the attitude error function | [](#eq3-weighted-psi) |
| $C_R$ | Matrix in $\dot e_R = C_R\,e_\Omega$ | [](#eq3-eR-dot) |
| $\mathcal{T}$ | Transport map | [](#def3-transport) |
| $E_\ell$, $E_r$ | Left- and right-invariant error | [](#def3-invariant-errors) |
| $\chi$ | Logarithm of the left-invariant error, $E_\ell = \exp(\chi^\wedge)$ | [](#eq3-lie-algebra-error) |
| $\theta_{\text{tilt}}$, $\theta_{\text{yaw}}$ | Tilt and yaw parts of an attitude error | [](#sec3-errors) |

: Errors and transport.

:::

### Dynamics and inputs

::: {.table #tblA-dynamics}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $m$, $m_Q$, $m_L$ | Mass, quadrotor mass, load mass | [](#sec4-products) |
| $J$ | Inertia matrix in the body frame | [](#sec3-velocity) |
| $g$ | Gravitational acceleration, 9.81 m/s² | [](#sec4-hamilton) |
| $\rho$ | Body-frame vector from the pivot to the center of mass | [](#eq4-pendulum) |
| $L$, $S$ | Lagrangian and action | [](#sec4-hamilton) |
| $\delta W$ | Virtual work | [](#def4-dalembert) |
| $\tau$ | Torque in the body frame | [](#sec4-forces) |
| $f$ | Thrust, a scalar, acting along $R\,e_3$ | [](#eq4-quadrotor) |
| $F$ | Force in the world frame | [](#eq4-virtual-work) |
| $u$ | Generic input, for example $u = (f, \tau)$ | [](#eq5-standard) |
| $T$ | Tension of a cable or rod | [](#eq4-tension) |
| $E_\xi$, $E_\eta$, $E_{\delta x}$ | Coefficient of a variation in $\delta S$. The equation of motion is $E_\xi = 0$ | [](#eq5-pendulum) |
| $M$, $b$, $G$ | Mass matrix, velocity and gravity terms, input matrix of $M\,\dot v + b + G\,u = 0$ | [](#eq5-standard) |
| $\lambda$ | Growth rate of an instability | [](#eq4-growth) |

: Dynamics and inputs.

:::

### Linearization and control

::: {.table #tblA-control}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $s$ | Linear state, $(\xi, \delta\omega)$ on $S^2$ or $(\eta, \delta\Omega)$ on $\SO(3)$ | [](#sec7-variation) |
| $A$, $B$ | Matrices of the linearized system $\dot s = A\,s + B\,\delta u$ | [](#thm7-s2) |
| $C$ | Constraint matrix of the linear state, $C\,s = 0$ | [](#eq7-s2-constraint) |
| $Z$ | Orthonormal basis of the constraint subspace $\ker C$ | [](#def7-constrained) |
| $\Gamma_k$ | Matrices of the controllability rank test | [](#eq7-ltv-rank) |
| $Q$, $R_u$, $P_f$ | State, input and terminal weights of the LQR cost | [](#sec7-lqr) |
| $P$, $K$ | Riccati solution and feedback gain | [](#eq7-riccati) |
| $k_p$, $k_d$, $k_R$, $k_\Omega$ | Scalar proportional and derivative gains | [](#eq-quat-pd), [](#ex7-hold) |

: Linearization and control.

:::

### Operators

::: {.table #tblA-operators}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $x^\wedge$, $A^\vee$ | Hat and vee | [](#def-hat) |
| $R\T$ | Transpose | |
| $\tr$ | Trace | |
| $a\cdot b$, $a\times b$ | Dot and cross product | |
| $\|x\|$ | Euclidean norm (Frobenius norm for a matrix unless stated) | |
| $\exp$, $\log$ | Matrix exponential and logarithm. $\exp_q$ is the exponential on the sphere | [](#prop2-rodrigues) |
| $[\cdot,\cdot]$ | Lie bracket | [](#eq2-bracket) |
| $\Ad_X$, $\ad_\zeta$ | Adjoint of a group element and of a Lie algebra element | [](#eq2-adjoint), [](#eq2-ad-se3) |
| $I - q q\T$ | Projection onto the tangent plane at $q$ | [](#eq4-sph-proj) |
| $\dot{(\ )}$, $\ddt$ | Time derivative | |
| $\dd$ | Differential, as in $\dd t$ | |
| $R_x(\cdot)$, $R_y(\cdot)$, $R_z(\cdot)$ | Rotations about $e_1$, $e_2$, $e_3$ | [](#eq-zyx) |
| $\operatorname{diag}$, $\operatorname{sgn}$, $\operatorname{rank}$, $\ker$ | Diagonal matrix, sign, rank, kernel | |

: Operators.

:::

## Glossary {#secA-glossary}

**Action.** The integral of the Lagrangian along a path. [](#sec4-hamilton).

**Adjoint.** Conjugation by a group element, acting on the Lie algebra. It changes the frame of a velocity. [](#sec2-body-spatial).

**Almost-global stability.** Convergence from every initial condition except a set of measure zero. It is the best a continuous controller can do on a compact configuration space. [](#sec-obstruction).

**Attitude.** The rotation of a rigid body relative to the world frame, a point of $\SO(3)$. [](#sec-rotation-space).

**Body angular velocity.** The angular velocity $\Omega$ of a rigid body, expressed in the body frame. It is what a gyroscope reads. [](#sec2-body-spatial).

**Body frame.** The frame attached to a rigid body. [](#sec-rotation-space).

**Cable-suspended load.** A point mass hanging from a quadrotor on a massless cable. A rigid body carried by several quadrotors is a payload. [](#sec2-products).

**Compatible.** A transport map is compatible with an error function when the error function changes only through the velocity error. [](#def3-transport).

**Configuration manifold.** The set of all configurations of a system, with its curved structure. For a robot with several parts it is a product manifold. [](#sec2-products).

**Constraint subspace.** The subspace $\ker C$ in which the linear state of a system on $S^2$ lives. [](#sec7-constrained).

**Coordinate-free.** Written with the objects themselves ($R$, $q$) and not with a chart such as Euler angles. A geometric controller or model is a coordinate-free one. [](#sec-roadmap).

**Cut locus.** The configurations from which the shortest way back to the target is not unique. [](#sec7-local).

**Direction.** A unit vector $q \in S^2$: a cable, a rod, a thrust axis. [](#sec-error).

**Double cover.** The two-to-one map from unit quaternions to rotations. [](#sec-quaternions).

**Equations of motion.** The differential equations that give the accelerations of a system. Written out in words, never abbreviated. [](#ch-dynamics).

**Error function.** A nonnegative function $\Psi$ of the actual and desired configurations that vanishes only when they agree. The text also calls it a height. [](#def-error-so3), [](#def3-error-vector).

**Error vector.** The slope of the error function under a variation: $e_R$, $e_q$. [](#def3-error-vector).

**Exponential map.** The map that wraps a tangent space onto the manifold. On a matrix group it is the matrix exponential. [](#sec2-exp).

**Gimbal lock.** The loss of one direction of rotation in an Euler-angle description, at 90° of pitch for yaw, pitch and roll. [](#sec-euler).

**Half turn.** A rotation by 180°. Half a turn away from the target is as far as an attitude can be. [](#sec-rotation-space).

**Hamilton's principle.** The motion makes the action stationary among paths with the same endpoints. [](#def4-hamilton).

**Hat map, vee map.** The hat map turns a vector into the skew-symmetric matrix of its cross product. The vee map is its inverse. [](#def-hat).

**Homogeneous space.** A manifold on which a group acts so that any point can be moved to any other. The sphere is one, under rotations. [](#sec2-sphere).

**Lagrange-d'Alembert principle.** Hamilton's principle with the virtual work of the applied forces added. [](#def4-dalembert).

**Left-invariant, right-invariant.** Unchanged when both arguments are multiplied by the same group element on the left, or on the right. A left-invariant control law does not depend on the choice of world frame. [](#sec-euler), [](#def3-invariant-errors).

**Lie algebra.** The tangent space of a Lie group at the identity. [](#def2-lie-group).

**Lie bracket.** The operation on the Lie algebra that records the failure of the group to commute. On $\so(3)$ it is the cross product. [](#sec2-bracket).

**Lie group.** A manifold that is also a group, with smooth multiplication and inversion. [](#def2-lie-group).

**Linear state.** The state of a variation-based linearization: a variation and the variation of the velocity. [](#sec7-variation).

**Logarithm error.** The attitude error vector $\log(R_d\T R)^\vee$, whose length is the error angle. [](#sec3-errors).

**Manifold.** A smooth curved set that looks like flat space near each of its points. [](#def2-manifold).

**Multicopter.** A vehicle lifted by several rotors. A quadrotor is a multicopter with four.

**Pitfall.** A block that warns of a trap. It is always referred to by that name.

**Product manifold.** The set of lists of points, one from each factor. [](#sec2-products).

**Reference.** A desired motion that changes with time, marked by the subscript $d$. [](#sec3-transport).

**Restoring torque.** The proportional part of an attitude controller, $-k_p\,e$ for an error vector $e$. [](#warn-sine).

**Tangent space, tangent plane.** The vector space of velocities allowed at a point of a manifold. On the sphere it is a plane and is called the tangent plane. [](#def2-manifold).

**Target.** A desired configuration that does not move, marked by the subscript $d$. [](#sec-error).

**Trace error.** The attitude error function $\tfrac12\tr(I - R_d\T R)$ and its error vector $e_R$. [](#sec3-errors).

**Transport map.** A rule that carries a velocity at the desired configuration to a velocity at the actual one, so that the two can be subtracted. [](#def3-transport).

**Twist.** An element of $\se(3)$: a linear and an angular velocity, $\zeta = (V, \Omega)$. [](#sec2-se3).

**Unwinding.** A rotation the long way round, caused by a controller that treats $\quat$ and $-\quat$ as different attitudes. [](#sec-quaternions).

**Variation.** The first-order gap between a curve and its neighbors in a family of curves that stay on the manifold. The text also calls it a nudge. [](#def3-variation).

**Velocity error.** The actual velocity minus the transported desired velocity: $e_\Omega$, $e_\omega$. [](#def3-transport).

**Virtual work.** The work the applied forces would do on a variation. [](#def4-dalembert).

**World frame.** The frame that does not move, with $e_3$ up. [](#sec-rotation-space).

**World-frame angular velocity.** The angular velocity expressed in the world frame, $\omega$. [](#sec2-body-spatial).

## Preferred terms {#secA-preferred}

<!-- terms:off -->

::: {.table #tblA-preferred}

| We write | Not |
| :-- | :-- |
| world frame | inertial frame, spatial frame, fixed frame |
| world-frame angular velocity | spatial angular velocity |
| body frame | body-fixed frame |
| attitude | orientation |
| target (fixed), reference (moving) | goal, set-point, hold point |
| error function | navigation function, configuration error |
| velocity error | transported error |
| transport map | transport operator |
| variation | perturbation vector, infinitesimal variation |
| linear state | variation coordinates |
| logarithm error | logarithmic error, log error |
| restoring torque | restoring action |
| cable-suspended load | suspended load, slung load |
| quadrotor, multicopter | quadcopter, multirotor, UAV |
| gyroscope | gyro |
| center of mass | mass center, CoM |
| equations of motion | EOM |
| almost-global | almost global |
| coordinate-free | coordinate free |
| Pitfall | warning |
| Euler-Lagrange, Runge-Kutta | the same with an en dash |
| et al. | and co-authors |
| $R\T$ | $R^T$, $R^\top$ written out |
| $x^\wedge$ | $\hat x$, $[x]_\times$ |
| $t_f$ | $T$ for a final time |
| subscript $d$ | "ref", "des", a star |

: Preferred terms and forms.

:::

<!-- terms:on -->
