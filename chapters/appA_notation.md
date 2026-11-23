---
title: Notation and glossary
short: Notation and glossary
id: app-notation
summary: Every symbol, convention and technical term of the book, with the place where each is defined.
epigraph: "By relieving the brain of all unnecessary work, a good notation sets it free to concentrate on more advanced problems …"
epigraph_source: "Alfred North Whitehead, *An Introduction to Mathematics* (1911)"
---

This appendix is the reference for the whole book. Each symbol means one thing, each term has one name, and every chapter, figure, notebook and skill follows the tables below. Read [](#secA-conventions) once. Use [](#secA-symbols) to look up a symbol and [](#secA-glossary) to look up a word. The last column of each table points to the place where the entry is introduced.

A subscript makes a new symbol. $E$, $E_\xi$ and $E_\ell$ are three different things, and so are $f$ and $f_\Delta$, or $P$, $P_f$ and $P_{UU}$. The subscripts $\min$ and $\max$ always mark a bound or an extreme value: $\tau_{\max}$, $\theta_{\max}$, $\Delta_{\max}$, $\Psi_{\max}$, $\omega_{\max}$, $J_{\min}$. A hat marks an estimate ($\hat R$, $\hat\Delta$) and a tilde a prediction or estimation error ($\tilde e_R$, $\tilde\Delta$).

A few letters are local and are not listed in the tables: $x, y, z$ for arbitrary vectors in an identity, $A$ for an arbitrary matrix, $i, j, k$ for indices, $n$ and $N$ for counts. So are $\Pi$ for a projection, $\beta$ and $\sigma$ in two exercises of [](#ch-variations), $\Phi$ for a transition matrix in one exercise of [](#ch-linearization), and $c_i$ for the entries of a matrix. They are named where they are used and mean nothing beyond that sentence. [](#secA-local) lists the letters that keep a second, local meaning inside one section because the alphabet ran out.

## Conventions {#secA-conventions}

**Frames.** The **world frame** does not move, with $e_3$ pointing up. The **body frame** is attached to a rigid body. A rotation matrix $R$ maps body-frame components to world-frame components.

**Gravity.** Gravity acts along $-e_3$. A mass $m$ at position $x$ has potential energy $m g\,e_3\cdot x$ and weight $-m g\,e_3$.

**Body and world velocities.** The angular velocity of a rigid body is $\Omega$, expressed in the body frame, with $\dot R = R\,\Omega^\wedge$. Its world-frame components are $\omega = R\,\Omega$. The angular velocity of a direction $q \in S^2$ is $\omega$, in the world frame, with $\dot q = \omega\times q$ and $\omega\cdot q = 0$. Positions $x$ and velocities $v = \dot x$ are in the world frame.

**Hat and vee.** The hat map is written after its argument: $x^\wedge y = x\times y$ and $(x^\wedge)^\vee = x$. On a twist, $\zeta^\wedge$ is the $4\times4$ matrix of [](#eq2-se3-kin).

**Twists.** A twist is ordered linear part first, $\zeta = (V, \Omega)$, both parts in the body frame.

**Cable direction.** For a cable-suspended load, $q$ points from the quadrotor to the load, so $x_Q = x_L - l\,q$. The cable tension $T$ is positive when the cable pulls.

**Desired quantities.** The subscript $d$ marks the configuration or motion that a controller aims for: $R_d$, $q_d$, $x_d$, $\Omega_d$, $\omega_d$, and also the force $F_d$, the yaw $\psi_d$ and the input $u_d$ or $\tau_d$ that go with it. A fixed one is called the target. A moving one is called the reference. No other mark ("ref", "des", a star) is used.

**Order of arguments.** The actual configuration comes first and the desired one second, in formulas and in code: $\Psi_R$ is built from $R_d\T R$, and a function is called as `e_R(R, Rd)`. An error is the actual quantity minus the desired one: $e_x = x - x_d$, $e_F = f R\,e_3 - F_d$. For a prediction error the state predictor is the actual one and the vehicle the desired one ([](#eq8-pred-error)).

**The tracking law.** The attitude law is always written in the form of [](#eq6-law): the feedback part $-k_R\,e_R - k_\Omega\,e_\Omega$, then the feedforward $\Omega\times J\Omega - J(\Omega\times R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d)$, with the cross product written with $\times$. On the sphere the gains $k_q$, $k_\omega$ multiply the errors in the angular acceleration, as in [](#eq6-law-s2). Every chapter that restates these laws uses the same form and cites them.

**Lyapunov functions.** $V_0$ is the energy-like function without a cross term and $V$ the one with it. $W_1$ and $W_2$ bound $V$ from below and above, $W_3$ bounds its rate, all three as quadratic forms in $z = (\|e_R\|, \|e_\Omega\|)$, and $\gamma$ is the exponential rate that follows. Their entries depend on the cross term chosen and are stated with each result. The smallest and largest eigenvalues of a symmetric matrix are $\lambda_{\min}(\cdot)$ and $\lambda_{\max}(\cdot)$.

**Gains.** A gain is a $k$ with the subscript of the error or estimate it multiplies: $k_R$, $k_\Omega$, $k_q$, $k_\omega$, $k_x$, $k_v$, $k_\Delta$. The cutoff $k_f$, the rate $k_V$ and the slack weight $k_w$ follow the same pattern. A gain on a prediction error carries a tilde. Capital $K$ is a gain matrix.

**Constraints and safety.** A safe set is written $\{H \ge 0\}$ with a barrier function $H$. A cone on the sphere has axis $q_c$ and half-angle $\theta_{\max}$: a direction that must stay inside satisfies $q\cdot q_c \ge \cos\theta_{\max}$, and one that must stay outside satisfies $q\cdot q_c \le \cos\theta_{\max}$. A bound on an input is written with $\max$, as in $|\tau_i| \le \tau_{\max}$. A slack variable is $w$.

**Sampling and horizons.** The time step $h$ is also the sampling period of a controller. Stages along a horizon are indexed by $k = 0, \dots, N$, and a sampled quantity carries that index: $s_k$, $A_k$, $\tau_{d,k}$.

**Optimization.** A cost is $\Phi$. A step of an optimizer on a manifold is a variation used at finite size ($\eta$, $\xi$, $\delta\Omega$) and is applied by a retraction. $\operatorname{grad}$ and $\operatorname{Hess}$ are the first- and second-order coefficients along such a step ([](#def13-newton)). For a function on $\R^n$ they are the usual gradient and Hessian, and no other symbol is used for them.

**Variations.** A variation is written with $\delta$. On $\SO(3)$ it is $\delta R = R\,\eta^\wedge$ with $\eta$ in the body frame and unconstrained. On $S^2$ it is $\delta q = \xi\times q$ with $\xi\cdot q = 0$. The small parameter of a family of curves is $\epsilon$.

**Velocity variation on the sphere.** With $\omega\cdot q = 0$ and $\xi\cdot q = 0$ kept along the whole curve, $\delta\omega = \dot\xi - 2\,\omega\times\xi$. This vector is not tangent: its tangent part is $\dot\xi - \omega\times\xi$ and its part along $q$ is $-\omega\times\xi$. See [](#warn3-s2-convention).

**Equations on the sphere.** An equation of motion for $q \in S^2$ holds only after projection by $I - q q\T$, together with $q\cdot\dot\omega = 0$.

**Signs of derived equations.** Chapters write equations of motion with the inertia on the left, as in $J\dot\Omega + \Omega\times J\Omega = \tau$. The symbolic library of [](#ch-symbolic) returns the coefficient of each variation, $E_\xi = 0$, with the inertia term negative, and its standard form is $M\,\dot v + b + G\,u = 0$.

**Transposes and products.** The transpose is $R\T$. The dot product of two vectors is $a\cdot b$, never $a\T b$. A transpose appears only in matrix expressions such as $q q\T$ and $s\T Q\,s$. A norm is $\|x\|$. The identity is $I$, with a subscript only when its size is not three.

**Angles and units.** Angles in formulas are in radians. Degrees are used in prose and on plot axes, written 90° in text and $90^\circ$ in math. Units are SI and upright: rad/s, m/s², kg·m², N·m, N·m·s, N·s, s⁻¹. A percentage is written 5%. The natural logarithm of a number is $\ln$. $\log$ is the matrix logarithm.

**Spelling.** American spelling: linearization, modeling, stabilize, center, behavior, maneuver, neighbor.

**Names in code.** Notebooks, skills and scripts use `hat`, `vee`, `expm_so3`, `log_so3`, `Psi_R`, `Psi_q`, `e_R`, `e_Omega`, `e_q`, `e_omega`, and write desired quantities as `Rd`, `qd`, `Omegad`, `omegad`. A symbol with a Greek subscript is spelled out: `k_R`, `k_Omega`, `k_Delta`, `k_f`, `tau_fb`, `P_UU`, `nu_r`. Outside the book the same notation is written in plain LaTeX, for example `R_d^\top R`.

**Colors in figures.** Ink is structure. Indigo is the main object or the method that works. Terracotta is the contrasting object, the error or the method that fails. Teal is a third role. The trace error is always indigo and the logarithm error always teal.

**References, targets and constraints in figures.** A target is an ink dot labeled "target". A reference drawn alone is the main object and is indigo. A reference drawn next to a motion that tracks it steps back: a thin dashed gray line with a hollow marker, terracotta on the stretch where it violates a constraint. The motion that tracks it is a solid line with a filled marker, indigo for the controller that works and terracotta for the one that fails, and a plan or prediction is teal. The boundary of a constraint is ink, and the forbidden side is filled in a pale terracotta where it can be filled. The axis of a cone is ink and labeled $q_c$. A quadrotor is two ink arms along $R\,e_1$ and $R\,e_2$ with four rotor rings.

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
| $t$, $t_f$, $h$ | Time, final time, time step. $h$ is also the sampling period of a controller | [](#sec4-hamilton), [](#sec2-integration), [](#eq13-sampled) |
| $N$, $k$ | Number of steps in a horizon, and the stage index $k = 0, \dots, N$ | [](#eq13-sampled) |
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
| $e_x$, $e_v$ | Position error $x - x_d$ and velocity error $v - \dot x_d$ | [](#def3-error-vector), [](#eq6-Fd) |
| $e_F$ | Force error $f R\,e_3 - F_d$ of a vehicle that can only push along its thrust axis | [](#eq6-Delta) |
| $D = \operatorname{diag}(d_1, d_2, d_3)$ | Weight of the attitude error function | [](#eq3-weighted-psi) |
| $C_R$ | Matrix in $\dot e_R = C_R\,e_\Omega$ | [](#eq3-eR-dot) |
| $\mathcal{T}$ | Transport map | [](#def3-transport) |
| $E_\ell$, $E_r$ | Left- and right-invariant error | [](#def3-invariant-errors) |
| $\chi$ | Logarithm of the left-invariant error, $E_\ell = \exp(\chi^\wedge)$. On $\SO(3)$ it is the logarithm error | [](#eq3-lie-algebra-error) |
| $\theta_{\text{tilt}}$, $\theta_{\text{yaw}}$ | Tilt and yaw parts of an attitude error | [](#sec3-errors) |

: Errors and transport.

:::

### Dynamics and inputs

::: {.table #tblA-dynamics}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $m$, $m_Q$, $m_L$ | Mass, quadrotor mass, load mass | [](#sec4-products) |
| $J$ | Inertia matrix in the body frame. $J_0$ is the inertia a controller believes | [](#sec3-velocity), [](#sec8-baseline) |
| $J_{\min}$, $J_{\max}$ | Smallest and largest eigenvalues of the inertia matrix in a Lyapunov function | [](#sec6-lyapunov) |
| $g$ | Gravitational acceleration, 9.81 m/s² | [](#sec4-hamilton) |
| $\rho$ | Body-frame vector from the pivot to the center of mass | [](#eq4-pendulum) |
| $L$, $S$ | Lagrangian and action | [](#sec4-hamilton) |
| $\delta W$ | Virtual work | [](#def4-dalembert) |
| $\tau$ | Torque in the body frame | [](#sec4-forces) |
| $f$ | Thrust, a scalar, acting along $R\,e_3$ | [](#eq4-quadrotor) |
| $F$ | Force in the world frame. $F_d$ is the force a position loop asks for, $F_L$ the one asked for a load | [](#eq4-virtual-work), [](#eq6-Fd), [](#eq6-load) |
| $u$ | Generic input, for example $u = (f, \tau)$. $u_d$, $\tau_d$ are the input of a reference, or the input a tracker proposes | [](#eq5-standard), [](#def7-feedback), [](#eq8-safety-filter) |
| $T$ | Tension of a cable or rod | [](#eq4-tension) |
| $E_\xi$, $E_\eta$, $E_{\delta x}$ | Coefficient of a variation in $\delta S$. The equation of motion is $E_\xi = 0$ | [](#eq5-pendulum) |
| $M$, $b$, $G$ | Mass matrix, velocity and gravity terms, input matrix of $M\,\dot v + b + G\,u = 0$ | [](#eq5-standard) |
| $\lambda$ | An eigenvalue. Alone, the growth rate of an instability. $\lambda_{\min}(\cdot)$, $\lambda_{\max}(\cdot)$ are the smallest and largest eigenvalues of a symmetric matrix | [](#eq4-growth), [](#thm6-exp) |

: Dynamics and inputs.

:::

### Linearization and control

::: {.table #tblA-control}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $s$ | Linear state, $(\xi, \delta\omega)$ on $S^2$ or $(\eta, \delta\Omega)$ on $\SO(3)$. With $\chi$ in place of $\eta$ it is the error state $(\chi, \delta\Omega)$ | [](#sec7-variation), [](#sec13-error) |
| $A$, $B$ | Matrices of the linearized system $\dot s = A\,s + B\,\delta u$. $A_k$, $B_k$ are their sampled counterparts, and $A_{22}$ is the velocity block of $A$ | [](#thm7-s2), [](#eq13-sampled), [](#eq13-se3) |
| $C$ | Constraint matrix of the linear state, $C\,s = 0$ | [](#eq7-s2-constraint) |
| $Z$ | Orthonormal basis of the constraint subspace $\ker C$ | [](#def7-constrained) |
| $\Gamma_k$ | Matrices of the controllability rank test | [](#eq7-ltv-rank) |
| $Q$, $R_u$, $P_f$ | State, input and terminal weights of the LQR cost | [](#sec7-lqr) |
| $P$, $K$ | Riccati solution and feedback gain | [](#eq7-riccati) |
| $k_p$, $k_d$ | Scalar proportional and derivative gains of a generic PD law | [](#eq-quat-pd), [](#sec6-fl) |

: Linearization and control.

:::

### Tracking and stability

::: {.table #tblA-tracking}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $k_R$, $k_\Omega$ | Gains of the attitude law, on $e_R$ and $e_\Omega$ | [](#eq6-law) |
| $k_q$, $k_\omega$ | Gains of the direction law, on $e_q$ and $e_\omega$ in the angular acceleration | [](#eq6-law-s2) |
| $k_x$, $k_v$ | Gains of the position law, on $e_x$ and $e_v$ | [](#eq6-Fd) |
| $F_d$, $\psi_d$ | Desired force and desired yaw, from which the cascade builds $R_d$ | [](#eq6-Fd), [](#eq6-Rd) |
| $V_0$, $V$ | Lyapunov function without and with a cross term | [](#eq6-V0), [](#eq6-V) |
| $c$ | Coefficient of the cross term | [](#eq6-V) |
| $z$ | Vector of error norms, $(\|e_R\|, \|e_\Omega\|)$ or $(\|e_q\|, \|e_\omega\|)$ | [](#thm6-exp) |
| $W_1$, $W_2$, $W_3$ | Matrices with $z\T W_1 z \le V \le z\T W_2 z$ and $\dot V \le -z\T W_3 z$ | [](#eq6-W) |
| $\gamma$ | Guaranteed exponential rate, $\dot V \le -\gamma V$ | [](#thm6-exp) |
| $\Psi_{\max}$ | Size of a sublevel set: the largest value the error function can reach from it | [](#sec6-lyapunov) |
| $\omega_{\max}$ | Bound on the angular velocity of a reference direction | [](#thm6-s2) |
| $\kappa$ | Factor by which an inner loop is made faster, with gains scaled by $(\kappa^2, \kappa)$ | [](#warn6-feedforward) |

: Tracking and stability.

:::

### Uncertainty, adaptation and safety

::: {.table #tblA-robust}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $\tau_{\text{fb}}$ | Feedback part of the torque, the part left to design. $F_{\text{fb}}$, $F_{\text{ff}}$ are the feedback and feedforward parts of a force | [](#eq8-pd), [](#eq8-qp) |
| $\Delta$ | Lumped uncertainty, a torque in the body frame | [](#eq8-lumped) |
| $\Delta_e$, $\Delta_m$ | Unknown external torque, and the part of the feedforward that used the wrong inertia | [](#eq8-lumped) |
| $\Delta_{\max}$ | Bound on $\|\Delta\|$, and radius of the projection ball | [](#thm8-bound), [](#eq8-proj) |
| $R_p$, $\Omega_p$, $\tau_p$ | Attitude, body angular velocity and torque of the state predictor | [](#sec8-l1) |
| $e_R^p$, $e_\Omega^p$, $\tau_{\text{fb}}^p$ | Tracking errors and feedback torque of the state predictor | [](#sec8-l1) |
| $\tilde\Psi$, $\tilde e_R$, $\tilde e_\Omega$ | Prediction error: error function, error vector and velocity error between predictor and vehicle | [](#eq8-pred-error) |
| $\tilde k_R$, $\tilde k_\Omega$ | Gains on the prediction error | [](#eq8-l1-law) |
| $\hat\Delta$, $\tilde\Delta$, $\Delta_f$ | Estimate of the lumped uncertainty, its estimation error, and the filtered estimate | [](#sec8-l1), [](#eq8-filter) |
| $\Xi$ | The matrix $J_0\,R_p\T R\,J_0^{-1}R\T R_p$, which is the identity when predictor and vehicle agree | [](#eq8-l1-law) |
| $k_\Delta$ | Adaptation gain | [](#eq8-adapt) |
| $y$ | Adaptation direction, the second argument of $\operatorname{Proj}$ | [](#eq8-adapt) |
| $\operatorname{Proj}$, $f_\Delta$ | Projection operator and the function that defines its ball | [](#eq8-proj) |
| $k_f$ | Cutoff of the low-pass filter | [](#eq8-filter) |
| $V_\infty$ | Level below which a Lyapunov function with a persistent disturbance ends up | [](#thm8-l1) |
| $H$ | Barrier function. The safe set is $\{H \ge 0\}$ | [](#sec8-barrier) |
| $\alpha$, $\alpha_1$, $\alpha_2$ | Rates of the barrier condition | [](#thm8-comparison), [](#eq8-rd2) |
| $H_1$ | $\dot H + \alpha_1 H$, for a barrier function of relative degree two | [](#eq8-rd2) |
| $q_c$, $\theta_{\max}$ | Axis and half-angle of a cone on the sphere. $\omega_c$ is the angular velocity of a moving axis | [](#eq8-cone), [](#eq13-dto) |
| $w$, $k_w$ | Slack variable and its weight | [](#eq8-qp), [](#eq13-kkt) |
| $k_V$ | Rate demanded of $V$ in the program | [](#eq8-qp) |
| $x_o$, $r_o$ | Center and radius of an obstacle | [](#eq8-plain) |
| $p$, $\sigma$, $\beta$ | Position along the thrust axis $(x - x_o)\cdot R\,e_3$, and the function and factor that augment a barrier with it | [](#eq8-aug) |

: Uncertainty, adaptation and safety.

:::

### Optimization and predictive control

::: {.table #tblA-mpc}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $\Phi$ | Cost of an optimization problem | [](#sec13-newton) |
| $\delta\Omega$, $\delta\tau$, $\delta\zeta$, $\delta u$ | Deviation from the reference, $\Omega - \Omega_d$ and so on. It equals the variation to first order | [](#sec13-error), [](#tbl7-dictionary) |
| $\tau_{\max}$ | Bound on each component of the torque | [](#sec13-qp) |
| $U$ | Input corrections of all stages, stacked. $U_{\min}$, $U_{\max}$ are its bounds | [](#eq13-qp) |
| $P_{UU}$, $P_{Us}$ | Blocks of the cost of the horizon, a quadratic form in $(s_0, U)$ | [](#eq13-qp) |
| $A_r$, $A_c$ | Jacobians of the equality and inequality constraints along the steps | [](#eq13-kkt) |
| $\nu_r$, $\nu_c$ | Lagrange multipliers of the equality and inequality constraints | [](#eq13-kkt) |
| $\mu$ | Barrier parameter of an interior-point method | [](#sec13-ipm) |

: Optimization and predictive control.

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
| $\operatorname{dexp}_x$ | Derivative of the exponential at $x$ | [](#eq13-dexp) |
| $\operatorname{grad}$, $\operatorname{Hess}$ | Gradient and Hessian: first- and second-order coefficients of a function along a step | [](#def13-newton) |
| $\ln$ | Natural logarithm of a number | |
| $[\cdot,\cdot]$ | Lie bracket | [](#eq2-bracket) |
| $\Ad_X$, $\ad_\zeta$ | Adjoint of a group element and of a Lie algebra element | [](#eq2-adjoint), [](#eq2-ad-se3) |
| $I - q q\T$ | Projection onto the tangent plane at $q$ | [](#eq4-sph-proj) |
| $\dot{(\ )}$, $\ddt$ | Time derivative | |
| $\dd$ | Differential, as in $\dd t$ | |
| $R_x(\cdot)$, $R_y(\cdot)$, $R_z(\cdot)$ | Rotations about $e_1$, $e_2$, $e_3$ | [](#eq-zyx) |
| $\operatorname{diag}$, $\operatorname{sgn}$, $\operatorname{rank}$, $\ker$ | Diagonal matrix, sign, rank, kernel | |

: Operators.

:::

### Letters with a local meaning {#secA-local}

The book has more quantities than letters. Each letter below keeps its meaning from the tables everywhere except in the place named, where the second meaning is the standard one in its field.

::: {.table #tblA-local}

| Letter | Everywhere | Locally |
| :-- | :-- | :-- |
| $V$ | Lyapunov function | Linear part of a twist $\zeta = (V, \Omega)$, always inside the pair |
| $\theta$ | Angle of a rotation or between two directions | Pitch, inside Euler-angle formulas |
| $v$ | Velocity | Stack of all velocities in [](#eq5-standard) |
| $c$ | Coefficient of the cross term | Inequality constraints $c(x) \le 0$ in [](#sec13-ipm) |
| $r$ | A radius when it carries a subscript ($r_o$, $r_{\min}$) | Equality constraints $r(x) = 0$ and their value $r_k$ in [](#sec13-plan) and [](#sec13-ipm) |
| $x$ | Position | The point that collects all unknowns of an optimization problem in [](#sec13-ipm) |
| $n$, $d$ | A count, and the subscript for "desired" | Unit normal and distance of the wall in [](#sec8-barrier) |
| $p$ | Sub- or superscript of the state predictor | The scalar $(x - x_o)\cdot R\,e_3$ in [](#sec8-quad) |
| $q_i$, $R_i$ | A direction or attitude with an index is the $i$-th reading or stage | $q_0$ and $q_v$ are the parts of a quaternion, inside $\quat = (q_0, q_v)$ |
| $f$ as a subscript | Final ($t_f$, $P_f$) | Filtered ($\Delta_f$, $k_f$) |

: Letters with a second, local meaning.

:::

## Glossary {#secA-glossary}

**Action.** The integral of the Lagrangian along a path. [](#sec4-hamilton).

**Adaptation gain.** The gain $k_\Delta$ of the adaptation law. It sets how closely the vehicle follows the state predictor. [](#eq8-adapt).

**Adjoint.** Conjugation by a group element, acting on the Lie algebra. It changes the frame of a velocity. [](#sec2-body-spatial).

**Almost-global stability.** Convergence from every initial condition except a set of measure zero. It is the best a continuous controller can do on a compact configuration space. It promises no rate: an exponential rate holds only on a sublevel set. [](#sec-obstruction), [](#thm6-almost).

**Attitude.** The rotation of a rigid body relative to the world frame, a point of $\SO(3)$. [](#sec-rotation-space).

**Augmented barrier function.** A barrier function for a position constraint that also depends on the attitude, so that the torque appears in its second derivative. [](#eq8-aug).

**Barrier function.** A function $H$ of the state whose nonnegative set is the safe set, and for which some input always achieves $\dot H \ge -\alpha H$. The control literature calls it a control barrier function. [](#sec8-barrier).

**Barrier parameter.** The weight $\mu$ of the logarithmic barrier in an interior-point method. [](#sec13-ipm).

**Body angular velocity.** The angular velocity $\Omega$ of a rigid body, expressed in the body frame. It is what a gyroscope reads. [](#sec2-body-spatial).

**Body frame.** The frame attached to a rigid body. [](#sec-rotation-space).

**Bound.** The largest magnitude an input may take, written with the subscript $\max$. The physical cause is called a limit of the actuator. [](#sec13-qp).

**Cable-suspended load.** A point mass hanging from a quadrotor on a massless cable. A rigid body carried by several quadrotors is a payload. [](#sec2-products).

**Cascade.** A controller built as nested loops, each of which supplies the desired value for the next one in: position, then direction, then attitude. [](#sec6-cascade).

**CBF-CLF-QP.** A quadratic program solved at every instant, with a barrier condition as a hard constraint and the decrease of a Lyapunov function as a constraint with a slack. [](#eq8-qp).

**Central path.** The curve of minimizers of a problem with a logarithmic barrier, as the barrier parameter decreases to zero. [](#sec13-ipm).

**Compatible.** A transport map is compatible with an error function when the error function changes only through the velocity error. [](#def3-transport).

**Configuration manifold.** The set of all configurations of a system, with its curved structure. For a robot with several parts it is a product manifold. [](#sec2-products).

**Constraint subspace.** The subspace $\ker C$ in which the linear state of a system on $S^2$ lives. [](#sec7-constrained).

**Coordinate-free.** Written with the objects themselves ($R$, $q$) and not with a chart such as Euler angles. A geometric controller or model is a coordinate-free one. [](#sec-roadmap).

**Cross term.** The term $c\,e_R\cdot J e_\Omega$ added to a Lyapunov function so that its rate is negative in the error vector as well as in the velocity error. [](#eq6-V).

**Cut locus.** The configurations from which the shortest way back to the target is not unique. [](#sec7-local).

**Cutoff.** The bandwidth $k_f$ of the low-pass filter in L1 adaptive control. It sets what reaches the motors and how much delay the loop tolerates. [](#eq8-filter).

**Direct transcription.** Trajectory optimization that discretizes first and then optimizes over states and inputs together, with the dynamics as constraints. [](#sec13-plan).

**Direction.** A unit vector $q \in S^2$: a cable, a rod, a thrust axis. [](#sec-error).

**Double cover.** The two-to-one map from unit quaternions to rotations. [](#sec-quaternions).

**Equations of motion.** The differential equations that give the accelerations of a system. Written out in words, never abbreviated. [](#ch-dynamics).

**Error function.** A nonnegative function $\Psi$ of the actual and desired configurations that vanishes only when they agree. The text also calls it a height. [](#def-error-so3), [](#def3-error-vector).

**Error state.** The pair $(\chi, \delta\Omega)$: the logarithm of the left-invariant error and the deviation of the velocity from the reference. It is the linear state in a form a controller can compute. [](#sec13-error).

**Error-state MPC.** Model predictive control whose model is the linearized dynamics of the error state. Its optimization problem is a quadratic program. [](#def13-mpc).

**Error vector.** The slope of the error function under a variation: $e_R$, $e_q$. [](#def3-error-vector).

**Exponential map.** The map that wraps a tangent space onto the manifold. On a matrix group it is the matrix exponential. [](#sec2-exp).

**Exponential stability on a sublevel set.** Convergence at a guaranteed exponential rate from every start inside a sublevel set of a Lyapunov function. The rate depends on the set and vanishes as the set grows to include half a turn. [](#thm6-exp).

**Feedforward.** The part of a control law computed from the model and the reference and not from the error. One word, no hyphen. [](#eq6-law).

**Force error.** The difference $e_F = f R\,e_3 - F_d$ between the force a thrust axis delivers and the force the position loop asked for. [](#eq6-Delta).

**Forward invariant.** A set is forward invariant when solutions that start in it stay in it. [](#sec8-barrier).

**Gimbal lock.** The loss of one direction of rotation in an Euler-angle description, at 90° of pitch for yaw, pitch and roll. [](#sec-euler).

**Gradient, Hessian.** The first- and second-order coefficients of a function along a step that stays on the manifold. [](#def13-newton).

**Half turn.** A rotation by 180°. Half a turn away from the target is as far as an attitude can be. [](#sec-rotation-space).

**Hamilton's principle.** The motion makes the action stationary among paths with the same endpoints. [](#def4-hamilton).

**Hat map, vee map.** The hat map turns a vector into the skew-symmetric matrix of its cross product. The vee map is its inverse. [](#def-hat).

**Homogeneous space.** A manifold on which a group acts so that any point can be moved to any other. The sphere is one, under rotations. [](#sec2-sphere).

**Interior-point method.** A method for inequality constraints that follows the central path to the solution from inside the constraints. [](#sec13-ipm).

**KKT conditions.** The Karush-Kuhn-Tucker conditions: the first-order conditions for a constrained minimum. [](#eq13-kkt).

**L1 adaptive control.** An adaptive architecture with a state predictor, a fast adaptation law and a low-pass filter between the estimate and the input. Written L1 in prose. The script letter is kept for the $\mathcal{L}_1$ norm. [](#sec8-l1).

**Lagrange-d'Alembert principle.** Hamilton's principle with the virtual work of the applied forces added. [](#def4-dalembert).

**Left-invariant, right-invariant.** Unchanged when both arguments are multiplied by the same group element on the left, or on the right. A left-invariant control law does not depend on the choice of world frame. [](#sec-euler), [](#def3-invariant-errors).

**Lie algebra.** The tangent space of a Lie group at the identity. [](#def2-lie-group).

**Lie bracket.** The operation on the Lie algebra that records the failure of the group to commute. On $\so(3)$ it is the cross product. [](#sec2-bracket).

**Lie group.** A manifold that is also a group, with smooth multiplication and inversion. [](#def2-lie-group).

**Linear state.** The state of a variation-based linearization: a variation and the variation of the velocity. [](#sec7-variation).

**Logarithm error.** The attitude error vector $\log(R_d\T R)^\vee$, whose length is the error angle. [](#sec3-errors).

**Logarithmic barrier.** The term $-\mu\ln x$ that replaces a constraint $x \ge 0$ in a cost. It is not a barrier function. [](#sec13-ipm).

**Lumped uncertainty.** Everything a controller's model gets wrong, collected in the one vector $\Delta$ that appears in the error dynamics. [](#eq8-lumped).

**Manifold.** A smooth curved set that looks like flat space near each of its points. [](#def2-manifold).

**Matched uncertainty.** An uncertainty that enters the dynamics at the same place as the input, so that a known one could be canceled. [](#sec8-uncertainty).

**Multicopter.** A vehicle lifted by several rotors. A quadrotor is a multicopter with four.

**Pitfall.** A block that warns of a trap. It is always referred to by that name.

**Plain controller.** The tracking law of [](#eq6-law) with no adaptation and no safety filter. [](#sec8-free).

**Prediction error.** The error between the state predictor and the vehicle, measured with the error function, the error vector and the transport map. [](#eq8-pred-error).

**Product manifold.** The set of lists of points, one from each factor. [](#sec2-products).

**Quadratic program.** The minimization of a quadratic cost under linear constraints. Abbreviated QP. [](#eq8-safety-filter), [](#eq13-qp).

**Receding horizon.** Solving over the next $N$ steps, applying the first input, and solving again at the next sample. [](#def13-mpc).

**Reference.** A desired motion that changes with time, marked by the subscript $d$. [](#sec3-transport).

**Relative degree.** The number of times a function of the state must be differentiated before the input appears. [](#sec6-fl), [](#sec8-cone).

**Restoring torque.** The proportional part of an attitude controller, $-k_p\,e$ for an error vector $e$. [](#warn-sine).

**Retraction.** A rule that moves a point of a manifold along a tangent vector, lands on the manifold, and agrees with the exponential map to first order. [](#def13-retraction).

**Safe set.** The set of states a system must never leave, written $\{H \ge 0\}$. [](#sec8-barrier).

**Safety filter.** A program that changes a proposed input by the smallest amount that satisfies a barrier condition. [](#eq8-safety-filter).

**Sampling period.** The time between two updates of a sampled controller. It is the time step $h$. [](#sec13-qp).

**Slack variable.** A variable $w$ that turns an inequality into an equality, or lets a constraint be violated at a cost. A slack cable is a cable without tension. [](#eq8-qp), [](#eq13-kkt).

**State predictor.** A copy of the believed dynamics, driven by the estimate of the uncertainty and run alongside the vehicle. The L1 literature calls it the reference model. [](#sec8-l1).

**Sublevel set.** The set of states where a function is below a given value. A sublevel set of a Lyapunov function that never increases is a set no solution can leave. [](#sec6-lyapunov).

**Tangent space, tangent plane.** The vector space of velocities allowed at a point of a manifold. On the sphere it is a plane and is called the tangent plane. [](#def2-manifold).

**Target.** A desired configuration that does not move, marked by the subscript $d$. [](#sec-error).

**Trace error.** The attitude error function $\tfrac12\tr(I - R_d\T R)$ and its error vector $e_R$. [](#sec3-errors).

**Transport map.** A rule that carries a velocity at the desired configuration to a velocity at the actual one, so that the two can be subtracted. [](#def3-transport).

**Twist.** An element of $\se(3)$: a linear and an angular velocity, $\zeta = (V, \Omega)$. [](#sec2-se3).

**Unwinding.** A rotation the long way round, caused by a controller that treats $\quat$ and $-\quat$ as different attitudes. [](#sec-quaternions).

**Variation.** The first-order gap between a curve and its neighbors in a family of curves that stay on the manifold. The text also calls it a nudge. [](#def3-variation).

**Velocity error.** The actual velocity minus the transported desired velocity: $e_\Omega$, $e_\omega$. [](#def3-transport).

**Virtual work.** The work the applied forces would do on a variation. [](#def4-dalembert).

**Warm start.** Starting a solver from the solution of the previous problem. [](#sec13-flight).

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
| plain controller | nominal controller, baseline controller |
| feedforward | feed-forward, feed forward |
| sublevel set | sub-level set |
| cross term | cross-term |
| state predictor | reference model |
| L1 adaptive control | $\mathcal{L}_1$ control, L1 adaptation |
| barrier function | control barrier function, CBF |
| safe set | safe region |
| safety filter | CBF filter, CBF-QP |
| CBF-CLF-QP | CLF-CBF-QP |
| sampling period | sample time, sampling time |
| time step | timestep |
| interior-point method | interior point as a noun |
| logarithmic barrier | barrier alone, for a term in a cost |
| quadratic program, QP | quadratic programme |
| model predictive control, MPC | model-predictive control |
| $\lambda_{\min}(\cdot)$, $\lambda_{\max}(\cdot)$ | $\mu_{\min}$, $\lambda_m$, $\lambda_M$ |
| $\operatorname{grad}$, $\operatorname{Hess}$ | $\nabla$, $\nabla^2$ |
| $k_\Delta$, $k_f$ | $\gamma$ for an adaptation gain, $\kappa$ for a cutoff |
| $\tau_{\text{fb}}$ | $\mu$ for a part of the torque |
| $e_F$ | $\Delta$ for the force error |
| $W_1$, $W_2$, $W_3$ | $W$ without a subscript |
| $P_{UU}$, $P_{Us}$ | $H$, $F$ for the matrices of a quadratic program |
| $\nu_r$, $\nu_c$ | $y$, $z$, $\lambda$ for multipliers |
| $k_p$, $k_d$ | $K_p$, $K_d$ for scalar gains |

: Preferred terms and forms.

:::

<!-- terms:on -->
