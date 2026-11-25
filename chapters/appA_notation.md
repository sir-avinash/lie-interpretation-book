---
title: Notation and glossary
short: Notation and glossary
id: app-notation
summary: Every symbol, convention and technical term of the book, with the place where each is defined.
epigraph: "By relieving the brain of all unnecessary work, a good notation sets it free to concentrate on more advanced problems …"
epigraph_source: "Alfred North Whitehead, *An Introduction to Mathematics* (1911)"
---

This appendix is the reference for the whole book. Each symbol means one thing, each term has one name, and every chapter, figure, notebook and skill follows the tables below. Read [](#secA-conventions) once. Use [](#secA-symbols) to look up a symbol and [](#secA-glossary) to look up a word. The last column of each table points to the place where the entry is introduced.

A subscript makes a new symbol. $E$, $E_\xi$ and $E_\ell$ are three different things, and so are $f$ and $f_\Delta$, or $P$, $P_f$ and $P_{UU}$. The subscripts $\min$ and $\max$ always mark a bound or an extreme value: $\tau_{\max}$, $\theta_{\max}$, $\Delta_{\max}$, $\Psi_{\max}$, $\omega_{\max}$, $J_{\min}$. A hat marks an estimate, on any letter: $\hat R$, $\hat q$, $\hat\omega$, $\hat X$, $\hat\Delta$, $\hat b_g$. It never stands for the hat map, which is always written after its argument, $x^\wedge$. A tilde marks a prediction or estimation error ($\tilde e_R$, $\tilde\Delta$), the innovation $\tilde y$ of a filter, and a gain that multiplies one of these ($\tilde k_R$, $\tilde K$). A bar has no general meaning: $\bar J$ and $\bar p_k$ are two symbols of [](#ch-integrators). A superscript in parentheses counts time derivatives: $x^{(4)}$ is the fourth.

A few letters are local and are not listed in the tables: $x, y, z$ for arbitrary vectors in an identity, $A$ for an arbitrary matrix, $i, j, k$ for indices, $n$ and $N$ for counts. So are $\Pi$ for a projection, $\beta$ and $\sigma$ in two exercises of [](#ch-variations), $\Phi$ for a transition matrix in one exercise of [](#ch-linearization), and $c_i$ for the entries of a matrix or the coefficients of a polynomial. They are named where they are used and mean nothing beyond that sentence. [](#secA-local) lists the letters that keep a second, local meaning inside one section because the alphabet ran out.

## Conventions {#secA-conventions}

**Frames.** The **world frame** does not move, with $e_3$ pointing up. The **body frame** is attached to a rigid body. A rotation matrix $R$ maps body-frame components to world-frame components.

**Gravity.** Gravity acts along $-e_3$. A mass $m$ at position $x$ has potential energy $m g\,e_3\cdot x$ and weight $-m g\,e_3$.

**Body and world velocities.** The angular velocity of a rigid body is $\Omega$, expressed in the body frame, with $\dot R = R\,\Omega^\wedge$. Its world-frame components are $\omega = R\,\Omega$. The angular velocity of a direction $q \in S^2$ is $\omega$, in the world frame, with $\dot q = \omega\times q$ and $\omega\cdot q = 0$. Positions $x$ and velocities $v = \dot x$ are in the world frame.

**Hat and vee.** The hat map is written after its argument: $x^\wedge y = x\times y$ and $(x^\wedge)^\vee = x$. On a twist, $\zeta^\wedge$ is the $4\times4$ matrix of [](#eq2-se3-kin).

**Twists.** A twist is ordered linear part first, $\zeta = (V, \Omega)$, both parts in the body frame.

**Cable direction.** For a cable-suspended load, $q$ points from the quadrotor to the load, so $x_Q = x_L - l\,q$. The cable tension $T$ is positive when the cable pulls.

**Desired quantities.** The subscript $d$ marks the configuration or motion that a controller aims for: $R_d$, $q_d$, $x_d$, $\Omega_d$, $\omega_d$, and also the force $F_d$, the heading $\psi_d$ and the input $u_d$ or $\tau_d$ that go with it. A fixed one is called the target. A moving one is called the reference. No other mark ("ref", "des", a star) is used.

**Heading and yaw.** The **heading** $\psi$ of a vehicle is the angle of [](#eq6-Rd): the second body axis $R\,e_2$ is perpendicular to $R_z(\psi)\,e_1$. It is defined without Euler angles, and it is the fourth flat output of a quadrotor ([](#sec10-quad)). The **yaw** is the first Euler angle of $\Theta = (\psi, \theta, \phi)$ in [](#eq-zyx). The two share the letter because they agree whenever the roll or the pitch is zero. Otherwise they differ, by the angle whose tangent is $-\sin\theta\tan\phi$. Outside Euler-angle formulas $\psi$ is always the heading, and $\psi_d$ is the desired heading. For a robot in the plane the heading is its one angle.

**Order of arguments.** The actual configuration comes first and the desired one second, in formulas and in code: $\Psi_R$ is built from $R_d\T R$, and a function is called as `e_R(R, Rd)`. An error is the actual quantity minus the desired one: $e_x = x - x_d$, $e_F = f R\,e_3 - F_d$. For a prediction error the state predictor is the actual one and the vehicle the desired one, as in [](#eq8-pred-error). For an estimation error the estimate is the actual one and the truth the desired one: $E_r = \hat X X^{-1}$ and $E_\ell = X^{-1}\hat X$, as in [](#eq14-errors).

**The tracking law.** The attitude law is always written in the form of [](#eq6-law): the feedback part $-k_R\,e_R - k_\Omega\,e_\Omega$, then the feedforward $\Omega\times J\Omega - J(\Omega\times R\T R_d\,\Omega_d - R\T R_d\,\dot\Omega_d)$, with the cross product written with $\times$. On the sphere the gains $k_q$, $k_\omega$ multiply the errors in the angular acceleration, as in [](#eq6-law-s2). Every chapter that restates these laws uses the same form and cites them.

**Lyapunov functions.** $V_0$ is the energy-like function without a cross term and $V$ the one with it. $W_1$ and $W_2$ bound $V$ from below and above, $W_3$ bounds its rate, all three as quadratic forms in $z = (\|e_R\|, \|e_\Omega\|)$, and $\gamma$ is the exponential rate that follows. Their entries depend on the cross term chosen and are stated with each result. The smallest and largest eigenvalues of a symmetric matrix are $\lambda_{\min}(\cdot)$ and $\lambda_{\max}(\cdot)$.

**Gains.** A gain is a $k$ with the subscript of the error or estimate it multiplies: $k_R$, $k_\Omega$, $k_q$, $k_\omega$, $k_x$, $k_v$, $k_\Delta$. The cutoff $k_f$, the rate $k_V$ and the slack weight $k_w$ follow the same pattern. A gain on a prediction error carries a tilde. Capital $K$ is a gain matrix, and $\tilde K$ is the Kalman gain, which multiplies the innovation.

**Constraints and safety.** A safe set is written $\{H \ge 0\}$ with a barrier function $H$. A cone on the sphere has axis $q_c$ and half-angle $\theta_{\max}$: a direction that must stay inside satisfies $q\cdot q_c \ge \cos\theta_{\max}$, and one that must stay outside satisfies $q\cdot q_c \le \cos\theta_{\max}$. A bound on an input is written with $\max$, as in $|\tau_i| \le \tau_{\max}$. A slack variable is $w$.

**Sampling and horizons.** The time step $h$ is also the sampling period of a controller. Stages along a horizon are indexed by $k = 0, \dots, N$, and a sampled quantity carries that index: $s_k$, $A_k$, $\tau_{d,k}$.

**Discrete mechanics.** A quantity of a discrete path carries the stage index: $x_k$, $R_k$, $q_k$, $p_k$, $\tau_k$. The discrete Lagrangian and the action sum carry the subscript $h$, the time step: $L_h$, $S_h$. The literature writes $L_d$, $S_d$, but in this book the subscript $d$ always means desired. The rotation over one step is $\Upsilon_k = R_k\T R_{k+1}$, where the literature writes $F_k$: here $F$ is a force. $\Omega_k$ is the body angular velocity that carries $R_k$ to $R_{k+1}$ in the time $h$, $R_{k+1} = R_k\exp(h\,\Omega_k^\wedge)$, in [](#def12-step) and in [](#eq13-dto) alike. A grid point of a transcription is a knot, never a node.

**Chains.** The joints of a chain of links are numbered from the quadrotor or the pivot: joint 0 is at the top, link $i$ runs from joint $i-1$ to joint $i$ with direction $q_i$ and length $l_i$, so $x_i = x_{i-1} + l_i\,q_i$, and joint $i$ carries the mass $m_i$. This is the cable-direction convention applied to every link. Kotaru (2022) numbers from the free end.

**Estimation.** The true state has no mark and the estimate has a hat. The covariance $\Sigma$ belongs to the logarithm of one of the two invariant errors, $\chi_r$ or $\chi$, and [](#eq14-left-right) converts between them. The vector of an element of the Lie algebra of $\SE_2(3)$ lists the linear parts first and the rotation last, $\chi = (\chi_v, \chi_x, \chi_R)$, like a twist. The innovation is the predicted reading minus the actual one, so a correction subtracts: $\hat X^+ = \exp(-(\tilde K\tilde y)^\wedge)\,\hat X$. A right-invariant error is corrected by multiplying on the left, a left-invariant one on the right.

**Optimization.** A cost is $\Phi$. A step of an optimizer on a manifold is a variation used at finite size ($\eta$, $\xi$, $\delta\Omega$) and is applied by a retraction. $\operatorname{grad}$ and $\operatorname{Hess}$ are the first- and second-order coefficients along such a step ([](#def13-newton)). For a function on $\R^n$ they are the usual gradient and Hessian, and no other symbol is used for them.

**Variations.** A variation is written with $\delta$. On $\SO(3)$ it is $\delta R = R\,\eta^\wedge$ with $\eta$ in the body frame and unconstrained. On $S^2$ it is $\delta q = \xi\times q$ with $\xi\cdot q = 0$. The small parameter of a family of curves is $\epsilon$.

**Velocity variation on the sphere.** With $\omega\cdot q = 0$ and $\xi\cdot q = 0$ kept along the whole curve, $\delta\omega = \dot\xi - 2\,\omega\times\xi$. This vector is not tangent: its tangent part is $\dot\xi - \omega\times\xi$ and its part along $q$ is $-\omega\times\xi$. See [](#warn3-s2-convention).

**Equations on the sphere.** An equation of motion for $q \in S^2$ holds only after projection by $I - q q\T$, together with $q\cdot\dot\omega = 0$.

**Signs of derived equations.** Chapters write equations of motion with the inertia on the left, as in $J\dot\Omega + \Omega\times J\Omega = \tau$. The symbolic library of [](#ch-symbolic) returns the coefficient of each variation, $E_\xi = 0$, with the inertia term negative, and its standard form is $M\,\dot v + b + G\,u = 0$.

**Transposes and products.** The transpose is $R\T$. The dot product of two vectors is $a\cdot b$, never $a\T b$. A transpose appears only in matrix expressions such as $q q\T$ and $s\T Q\,s$. A norm is $\|x\|$. The identity is $I$, with a subscript only when its size is not three.

**Angles and units.** Angles in formulas are in radians. Degrees are used in prose and on plot axes, written 90° in text and $90^\circ$ in math. Units are SI and upright: rad/s, m/s², kg·m², N·m, N·m·s, N·s, s⁻¹. A percentage is written 5%. The natural logarithm of a number is $\ln$. $\log$ is the matrix logarithm.

**Spelling.** American spelling: linearization, modeling, stabilize, center, behavior, maneuver, neighbor.

**Names in code.** Notebooks, skills and scripts use `hat`, `vee`, `expm_so3`, `log_so3`, `Psi_R`, `Psi_q`, `e_R`, `e_Omega`, `e_q`, `e_omega`, and write desired quantities as `Rd`, `qd`, `Omegad`, `omegad`. A symbol with a Greek subscript is spelled out: `k_R`, `k_Omega`, `k_Delta`, `k_f`, `tau_fb`, `P_UU`, `nu_r`. A time derivative is `R_dot`, `x_L_ddot`. A hat, a tilde or a bar is appended without an underscore, like the `d` of a desired quantity: `Rhat`, `qhat`, `omegahat`, `Xhat`, `ytilde`, `Ktilde`, `Jbar`, `pbar`. Greek capitals are spelled out: `Sigma`, `Upsilon_k`, `Pi_C`. Outside the book the same notation is written in plain LaTeX, for example `R_d^\top R`. The robot library of [](#sec16-library) uses the same names for its parameters, states and inputs, and [](#tblA-code) lists the ones that have no symbol in the chapters.

**Colors in figures.** Ink is structure. Indigo is the main object or the method that works. Terracotta is the contrasting object, the error or the method that fails. Teal is a third role. The trace error is always indigo and the logarithm error always teal.

**References, targets and constraints in figures.** A target is an ink dot labeled "target". A reference drawn alone is the main object and is indigo. A reference drawn next to a motion that tracks it steps back: a thin dashed gray line with a hollow marker, terracotta on the stretch where it violates a constraint. The motion that tracks it is a solid line with a filled marker, indigo for the controller that works and terracotta for the one that fails, and a plan or prediction is teal. The boundary of a constraint is ink, and the forbidden side is filled in a pale terracotta where it can be filled. The axis of a cone is ink and labeled $q_c$. A quadrotor is two ink arms along $R\,e_1$ and $R\,e_2$ with four rotor rings. A true quantity drawn next to its estimate is treated like a reference next to a motion: the truth is the thin dashed gray line with a hollow marker, the estimate of the filter that works is solid indigo, and that of the filter it is compared with is solid terracotta. A curve prescribed for a flat output, drawn next to the simulated motion, is dashed gray in the same way.

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
| $\SE_2(3)$ | Group of attitude, velocity and position | [](#sec2-se3), [](#eq14-X) |
| $\SE_3(3)$, $\SE_{N+2}(3)$ | The same group with one more column for a contact position, or with $N$ of them | [](#eq14-contact) |
| $\SE(2)$ | Poses of a robot in the plane | [](#sec14-gaussian) |
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
| $R$ | Attitude, body to world. $\hat R$ is an estimate of it, $R_k$ its value at stage $k$ | [](#def-so3) |
| $\Omega$ | Body angular velocity, $\dot R = R\,\Omega^\wedge$ | [](#eq-kinematics) |
| $q$ | Direction in $S^2$. The letter $q$ alone is always a direction | [](#eq2-s2-kinematics) |
| $\omega$ | World-frame angular velocity: of a direction ($\dot q = \omega\times q$, $\omega\cdot q = 0$) or of a body ($\omega = R\,\Omega$) | [](#eq2-s2-kinematics), [](#eq2-spatial) |
| $x$, $v$ | Position and velocity $v = \dot x$ in the world frame. In [](#eq5-standard), $v$ is the stack of all velocities | [](#sec2-se3) |
| $x_L$, $x_Q$, $v_L$ | Load position, quadrotor position, load velocity | [](#sec2-products) |
| $\zeta = (V, \Omega)$ | Twist, or body velocity on a matrix Lie group, $\dot X = X\,\zeta^\wedge$ | [](#eq2-se3-kin) |
| $\quat = (q_0, q_v)$ | Unit quaternion, scalar part first | [](#eq-quat-to-R) |
| $\Theta = (\psi, \theta, \phi)$ | Yaw, pitch and roll angles | [](#eq-zyx) |
| $\psi$ | Heading of a vehicle: $R\,e_2$ is perpendicular to $R_z(\psi)\,e_1$. Equal to the yaw when the roll or the pitch is zero | [](#eq6-Rd), [](#sec10-quad) |
| $E$ | Euler-angle Jacobian, $\Omega = E\,\dot\Theta$ | [](#eq-euler-jacobian) |
| $\theta$, $a$ | Angle and unit axis of a rotation. $\theta a$ is the rotation vector. Also the angle between $q$ and $q_d$, the angle of a planar pendulum from the hanging position, and, as $\theta_k = h\,\|\Omega_k\|$, the angle of a step rotation. Inside Euler-angle formulas only, $\theta$ is the pitch | [](#sec-rotation-space), [](#prop2-rodrigues), [](#sec12-three), [](#thm12-lgvi) |
| $e_1, e_2, e_3$ | Standard basis. $e_3$ points up | |
| $\Lambda$ | Quarter turn of the plane. $\Lambda_n$ is its counterpart for $n$ degrees of freedom, the block matrix with $I$ and $-I$ off the diagonal | [](#sec2-tangent), [](#sec12-three) |
| $t$, $t_f$, $h$ | Time, final time, time step. $h$ is also the sampling period of a controller | [](#sec4-hamilton), [](#sec2-integration), [](#eq13-sampled) |
| $N$, $k$ | Number of steps in a horizon or a discrete path, and the stage index $k = 0, \dots, N$ | [](#eq13-sampled), [](#def12-discrete) |
| $l$ | A length: cable, rod or leg. $l_i$ is the length of link $i$ | [](#sec2-products), [](#sec10-chain) |
| $x_i$, $q_i$, $m_i$, $T_i$ | Position and mass of joint $i$ of a chain, direction and tension of link $i$, counted from the quadrotor | [](#sec10-chain) |
| $n_Q$ | Number of quadrotors on a chain | [](#thm10-chain) |

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
| $E_\ell$, $E_r$ | Left- and right-invariant error, $X_d^{-1}X$ and $X X_d^{-1}$. For an estimate, $X^{-1}\hat X$ and $\hat X X^{-1}$ | [](#def3-invariant-errors), [](#eq14-errors) |
| $\chi$ | Logarithm of the left-invariant error, $E_\ell = \exp(\chi^\wedge)$. On $\SO(3)$ it is the logarithm error | [](#eq3-lie-algebra-error) |
| $\chi_r$ | Logarithm of the right-invariant error, $E_r = \exp(\chi_r^\wedge)$, with $\chi_r = \Ad_X\,\chi$ | [](#eq14-errors), [](#eq14-left-right) |
| $\theta_{\text{tilt}}$, $\theta_{\text{yaw}}$ | Tilt and yaw parts of an attitude error | [](#sec3-errors) |

: Errors and transport.

:::

### Dynamics and inputs

::: {.table #tblA-dynamics}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $m$, $m_Q$, $m_L$ | Mass, quadrotor mass, load mass | [](#sec4-products) |
| $J$ | Inertia matrix in the body frame. $J_0$ is the inertia a controller believes, and in [](#eq4-leg) the inertia of a leg without its point mass | [](#sec3-velocity), [](#sec8-baseline), [](#ex4-leg) |
| $J_{\min}$, $J_{\max}$ | Smallest and largest eigenvalues of the inertia matrix in a Lyapunov function | [](#sec6-lyapunov) |
| $g$ | Gravitational acceleration, 9.81 m/s² | [](#sec4-hamilton) |
| $\rho$ | Body-frame vector from the pivot to the center of mass | [](#eq4-pendulum) |
| $L$, $S$ | Lagrangian and action | [](#sec4-hamilton) |
| $\delta W$ | Virtual work | [](#def4-dalembert) |
| $\tau$ | Torque in the body frame | [](#sec4-forces) |
| $f$ | Thrust, a scalar, acting along $R\,e_3$ | [](#eq4-quadrotor) |
| $F$ | Force in the world frame. $F_d$ is the force a position loop asks for, $F_L$ the one asked for a load. Along a motion, $F = f R\,e_3$ is the thrust vector | [](#eq4-virtual-work), [](#eq6-Fd), [](#eq6-load), [](#eq10-quad-force) |
| $u$ | Generic input, for example $u = (f, \tau)$. $u_d$, $\tau_d$ are the input of a reference, or the input a tracker proposes | [](#eq5-standard), [](#def7-feedback), [](#eq8-safety-filter) |
| $T$ | Tension of a cable or rod. $T\,q$ is the tension vector | [](#eq4-tension), [](#eq10-chain) |
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
| $F_d$, $\psi_d$ | Desired force and desired heading, from which the cascade builds $R_d$ | [](#eq6-Fd), [](#eq6-Rd) |
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

### Flatness and trajectory generation

::: {.table #tblA-flat}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $\varphi$ | Flat output. Not the roll $\phi$ | [](#def10-flat) |
| $x^{(k)}$, $\varphi^{(k)}$ | $k$-th time derivative. $x^{(3)}$ is the jerk and $x^{(4)}$ the snap | [](#def10-flat) |
| $\psi$ | Heading, the fourth flat output of a quadrotor | [](#sec10-quad) |
| $F$ | Thrust vector $f R\,e_3$, equal to the missing force $m(\ddot x + g\,e_3)$ of the vehicle | [](#eq10-quad-force) |
| $\Omega_1$, $\Omega_2$, $\Omega_3$ | Components of the body angular velocity | [](#eq10-quad-Omega) |
| $x_i$, $q_i$, $l_i$, $m_i$, $T_i$ | Joint positions, link directions, link lengths, joint masses and link tensions of a chain, counted from the quadrotor | [](#eq10-chain) |
| $n$, $n_Q$ | Number of links of a chain and number of quadrotors on it | [](#thm10-chain) |
| $t_0 < t_1 < \dots < t_f$ | Times at which a curve passes its waypoints | [](#eq10-minsnap) |
| $c_0, \dots, c_7$ | Coefficients of one polynomial piece of a curve. A local letter | [](#sec10-trajectories) |
| $f_d$, $\tau_d$ | Thrust and torque along a reference, from the flat map | [](#sec10-feedforward) |

: Flatness and trajectory generation.

:::

### Discrete mechanics and transcription

::: {.table #tblA-discrete}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $x_k$, $R_k$, $q_k$ | Configuration at the time $t_k = k h$ of a discrete path, or at knot $k$ of a transcription | [](#sec12-discrete), [](#sec12-collocation) |
| $L_h$ | Discrete Lagrangian: the action of one step as a function of its two ends | [](#def12-discrete) |
| $S_h$ | Action sum | [](#eq12-action-sum) |
| $p$, $p_k$ | Momentum, and the discrete momentum at stage $k$: in the body frame on $\SO(3)$, in the world frame on $S^2$ | [](#sec12-three), [](#eq12-legendre) |
| $\bar p_k$ | Discrete momentum after the first half kick of a step | [](#eq12-s2) |
| $\Upsilon_k$ | Step rotation $R_k\T R_{k+1}$ | [](#def12-step) |
| $\Omega_k$ | Step velocity: $\Upsilon_k = \exp(h\,\Omega_k^\wedge)$ | [](#def12-step), [](#eq13-dto) |
| $\theta_k$ | Angle of the step rotation, $h\,\|\Omega_k\|$ | [](#thm12-lgvi) |
| $\bar J$ | $\tfrac12\tr(J)\,I - J$, the matrix that writes the kinetic energy as a trace | [](#sec12-lgvi) |
| $\eta_k$, $\xi_k$ | Variations of $R_k$ and $q_k$ | [](#eq12-var-step) |
| $\tau_k$ | Total torque in the body frame at $R_k$ | [](#thm12-lgvi) |
| $g_\epsilon$ | A one-parameter family of transformations that leaves $L_h$ unchanged | [](#thm12-vi) |
| $r$ | Order of a method. A local letter | [](#thm12-vi) |
| $\Lambda_n$ | Block matrix of the symplectic condition | [](#sec12-three) |
| $\dot x_k$, $x_{k+1/2}$, $\dot x_{k+1/2}$ | Slope of the model at knot $k$, and the midpoint of the interpolating cubic with the slope there | [](#eq12-trap), [](#eq12-hs) |
| $R_0$, $q_0$ | Center of an exponential chart, $R = R_0\exp(\eta^\wedge)$, $q = \exp(\xi^\wedge)\,q_0$ | [](#eq12-chart) |

: Discrete mechanics and transcription.

:::

### Estimation

::: {.table #tblA-estimation}

| Symbol | Meaning | Defined in |
| :-- | :-- | :-- |
| $\hat X$, $\hat R$, $\hat v$, $\hat x$, $\hat q$, $\hat\omega$ | Estimates. The true state has no mark | [](#sec14-gaussian), [](#sec14-s2) |
| $E_r$, $E_\ell$ | Right- and left-invariant estimation error, $\hat X X^{-1}$ and $X^{-1}\hat X$ | [](#eq14-errors) |
| $\chi_r$, $\chi$ | Their logarithms. On $\SE_2(3)$ the parts are $(\chi_v, \chi_x, \chi_R)$, with $\chi_c$ for a contact | [](#eq14-errors), [](#eq14-X) |
| $E_R$ | Attitude part $\hat R R\T$ of the right-invariant error | [](#sec14-autonomous) |
| $\Sigma$ | Covariance of the logarithm of the error, or of the linear state $s$ | [](#def14-gaussian), [](#def14-vekf) |
| $\mathcal{N}(0, \Sigma)$ | Gaussian with zero mean and covariance $\Sigma$ | [](#def14-gaussian) |
| $a_s$ | Specific force: acceleration minus gravity, in the body frame | [](#eq14-imu) |
| $f_u$ | Dynamics on a group, $\dot X = f_u(X)$ with input $u$ | [](#def14-affine) |
| $A$ | Matrix of the error equation, $\dot\chi_r = A\,\chi_r$. The $A$ of a linearization, here of the error | [](#eq14-A-right), [](#eq14-A-left) |
| $n_g$, $n_a$, $n_y$, $n_k$ | Noise of the gyroscope, the accelerometer and a measurement, and the process noise of one step | [](#eq14-noise), [](#eq14-obs) |
| $\Sigma_g$, $\Sigma_a$, $\Sigma_y$, $\Sigma_n$ | Their covariances | [](#eq14-noise), [](#def14-inekf) |
| $b_g$, $b_a$ | Biases of the gyroscope and the accelerometer | [](#sec14-legged) |
| $y$ | A reading of a sensor | [](#eq14-obs) |
| $d$ | The known vector of a measurement $X^{-1}d$ or $X d$. A local letter | [](#eq14-obs) |
| $x_m$ | Position of a landmark | [](#eq14-obs) |
| $x_c$ | Position of a foot in contact | [](#eq14-contact) |
| $\tilde y$, $\Sigma_{\tilde y}$ | Innovation, predicted minus measured, and its covariance | [](#eq14-innovation), [](#eq14-gain) |
| $C_y$ | Measurement matrix, $\tilde y = C_y\,\chi_r$ to first order | [](#eq14-innovation) |
| $\tilde K$ | Kalman gain | [](#eq14-gain) |
| $\hat X^+$, $\Sigma^+$, $s^+$ | Estimate, covariance and correction after a measurement | [](#eq14-update), [](#def14-vekf) |
| $\oplus$, $\ominus$ | Apply a finite linear state $s$ to a state on the sphere, and its inverse | [](#eq14-oplus) |
| $\Pi_C$ | Projector onto the constraint subspace $\ker C$ at $(q, \omega)$ | [](#eq14-E) |
| $\delta\theta$ | Attitude error of a conventional filter, $\hat R = R\exp(\delta\theta^\wedge)$. Used once | [](#sec14-autonomous) |

: Estimation.

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
| $\Pi$ | A projection, named where it is used: $\Pi = I - e_3 e_3\T$ in [](#eq4-leg), $\Pi_C$ onto $\ker C$ | [](#ex4-leg), [](#eq14-E) |
| $x^{(k)}$ | $k$-th time derivative | [](#def10-flat) |
| $\oplus$, $\ominus$ | Apply a finite linear state on the sphere, and recover it | [](#eq14-oplus) |
| $\mathcal{N}(0, \Sigma)$ | Gaussian distribution | [](#def14-gaussian) |
| $\partial/\partial x$ | Partial derivative, on flat space only | [](#eq12-del) |
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
| $p$ | A momentum, and the discrete momentum $p_k$. As a sub- or superscript, the state predictor | The scalar $(x - x_o)\cdot R\,e_3$ in [](#sec8-quad) |
| $q_i$, $R_i$ | A direction or attitude with an index is the $i$-th reading or stage | $q_0$ and $q_v$ are the parts of a quaternion, inside $\quat = (q_0, q_v)$ |
| $f$ as a subscript | Final ($t_f$, $P_f$) | Filtered ($\Delta_f$, $k_f$) |
| $\psi$ | Heading of a vehicle | Yaw, inside Euler-angle formulas. The two agree when the roll or the pitch is zero |
| $\theta$ with an index | Angle of a rotation | $\theta_k$ is also the angle of the planar pendulum at stage $k$ in [](#sec12-three) and [](#sec12-discrete), and the angle of the step rotation in [](#thm12-lgvi). $\hat\theta$ is the estimate of a planar heading in [](#sec14-circle) |
| $c$ with an index | Entries of a matrix | Coefficients $c_0, \dots, c_7$ of a polynomial in [](#sec10-trajectories), a drag constant $c_1$ in [](#exr14-affine), and the constant in $e^{c/h}$ in [](#sec12-conserve) |
| $y$ | A reading of a sensor in [](#ch-filtering) | Adaptation direction in [](#eq8-adapt) |
| $A$ | Matrix of a linearized system | Jacobian of a map in the footnote of [](#sec12-three), and any matrix in an identity |
| $r$ | A radius when it carries a subscript | Order of a method in [](#thm12-vi) |
| $u$ | Input | Force on the cart in [](#ex10-cart), which is its input |
| $n$ with a subscript | A count | Noise: $n_g$, $n_a$, $n_y$, $n_k$ in [](#ch-filtering) |
| $d$ | The subscript for "desired" | The known vector of a measurement in [](#eq14-obs) |
| $N$ | Number of steps | Number of feet in contact, in $\SE_{N+2}(3)$ of [](#sec14-legged) |
| $h$ as a subscript | Time step | $L_h$, $S_h$: built for the time step $h$ |

: Letters with a second, local meaning.

:::

### Names in the robot library {#secA-code}

The modules of `case_studies/robots` name their parameters, states and inputs after the symbols above. The names below have no symbol in the chapters, or need a word of explanation.

::: {.table #tblA-code}

| Name in code | Meaning | Robot |
| :-- | :-- | :-- |
| `x_L`, `v_L`, `q`, `omega`, `R`, `Omega` | $x_L$, $v_L$, $q$, $\omega$, $R$, $\Omega$. For several vehicles the arrays are stacked, one row per vehicle | all aerial robots |
| `m_Q`, `m_L`, `l`, `J`, `g` | $m_Q$, $m_L$, $l$, $J$, $g$ | quadrotor with a cable-suspended load |
| `Rd`, `Omegad`, `Omegad_dot`, `qd`, `xd` | Desired quantities | every `tracking_controller` |
| `J_L`, `R_L`, `Omega_L`, `r` | Inertia, attitude and body angular velocity of a rigid payload, and the body-frame points `r[i]` where the cables are attached | `multi_quadrotor_payload` |
| `x_0`, `v_0` | Position and velocity of joint 0 of a chain | `quadrotor_cable` |
| `n`, `n_Q`, `attach` | Number of links, number of quadrotors $n_Q$, and the joints that carry a quadrotor | `flexible_cable`, `quadrotor_cable` |
| `l`, `l_dot` | Length of the stance leg and its rate | `telescoping_leg`, `reaction_mass_biped` |
| `R1`, `RP`, `R2`, `Omega1`, `OmegaP`, `Omega2` | Attitudes and body angular velocities of the stance leg, the torso and the swing leg | `reaction_mass_biped` |
| `slide`, `slide_dot` | Distance of each of the three pairs of reaction masses from the hip, and its rate | `reaction_mass_biped` |
| `m_leg`, `m_r`, `J_0`, `J_P` | Mass of one leg, one reaction mass, inertia of a leg, inertia of the torso | `reaction_mass_biped` |
| `tau_1`, `tau_P`, `tau_2`, `f`, `f_s` | Ankle torque, hip torque between stance leg and torso, hip torque between torso and swing leg, leg force, forces on the three pairs of reaction masses | `reaction_mass_biped` |
| `R_r`, `R_l`, `Omega_r`, `Omega_l`, `l_r`, `l_l` | Attitude, body angular velocity and length of the right and the left leg | `cassie` |
| `m_b`, `m_h`, `m_k`, `J_b`, `J_leg`, `d` | Masses of the pelvis, of a hip and of a knee, inertias of the pelvis and of a leg, half the hip width | `cassie` |
| `tau_r`, `f_r`, `tau_l`, `f_l` | Hip torque and leg force of the right and the left leg | `cassie` |
| `E1`, `E2`, `E3` | $e_1$, $e_2$, $e_3$ | all |

: Names in the robot library that have no symbol in the chapters.

:::

## Glossary {#secA-glossary}

**Action sum.** The sum $S_h$ of the discrete Lagrangians along a discrete path. It replaces the action integral. [](#def12-discrete).

**Action.** The integral of the Lagrangian along a path. [](#sec4-hamilton).

**Adaptation gain.** The gain $k_\Delta$ of the adaptation law. It sets how closely the vehicle follows the state predictor. [](#eq8-adapt).

**Adjoint.** Conjugation by a group element, acting on the Lie algebra. It changes the frame of a velocity. [](#sec2-body-spatial).

**Almost-global stability.** Convergence from every initial condition except a set of measure zero. It is the best a continuous controller can do on a compact configuration space. It promises no rate: an exponential rate holds only on a sublevel set. [](#sec-obstruction), [](#thm6-almost).

**Attitude.** The rotation of a rigid body relative to the world frame, a point of $\SO(3)$. [](#sec-rotation-space).

**Augmented barrier function.** A barrier function for a position constraint that also depends on the attitude, so that the torque appears in its second derivative. [](#eq8-aug).

**Barrier function.** A function $H$ of the state whose nonnegative set is the safe set, and for which some input always achieves $\dot H \ge -\alpha H$. The control literature calls it a control barrier function. [](#sec8-barrier).

**Barrier parameter.** The weight $\mu$ of the logarithmic barrier in an interior-point method. [](#sec13-ipm).

**Bias.** A slowly varying offset of a gyroscope or an accelerometer, $b_g$ or $b_a$. [](#sec14-legged).

**Body angular velocity.** The angular velocity $\Omega$ of a rigid body, expressed in the body frame. It is what a gyroscope reads. [](#sec2-body-spatial).

**Body frame.** The frame attached to a rigid body. [](#sec-rotation-space).

**Bound.** The largest magnitude an input may take, written with the subscript $\max$. The physical cause is called a limit of the actuator. [](#sec13-qp).

**Cable-suspended load.** A point mass hanging from a quadrotor on a massless cable. A rigid body carried by several quadrotors is a payload. The cable direction $q$ points from the quadrotor to the load. [](#sec2-products).

**Cascade.** A controller built as nested loops, each of which supplies the desired value for the next one in: position, then direction, then attitude. [](#sec6-cascade).

**CBF-CLF-QP.** A quadratic program solved at every instant, with a barrier condition as a hard constraint and the decrease of a Lyapunov function as a constraint with a slack. [](#eq8-qp).

**Center of oscillation.** The point of a body swung from a pivot that accelerates, measured against free fall, only along the line to the pivot. It is the flat output of that body. [](#ex10-stick).

**Central path.** The curve of minimizers of a problem with a logarithmic barrier, as the barrier parameter decreases to zero. [](#sec13-ipm).

**Collocation.** A transcription that joins neighboring knots by a polynomial and demands that the polynomial obey the model at chosen points. Trapezoidal collocation uses a parabola and is second order. Hermite-Simpson collocation uses a cubic and is fourth order. [](#sec12-collocation).

**Compatible.** A transport map is compatible with an error function when the error function changes only through the velocity error. [](#def3-transport).

**Concentrated Gaussian.** A distribution on a Lie group: a mean on the group and a Gaussian in the Lie algebra, wrapped onto the group by the exponential. [](#def14-gaussian).

**Configuration manifold.** The set of all configurations of a system, with its curved structure. For a robot with several parts it is a product manifold. [](#sec2-products).

**Constraint subspace.** The subspace $\ker C$ in which the linear state of a system on $S^2$ lives. [](#sec7-constrained).

**Coordinate-free.** Written with the objects themselves ($R$, $q$) and not with a chart such as Euler angles. A geometric controller or model is a coordinate-free one. [](#sec-roadmap).

**Covariance.** The matrix $\Sigma$ of second moments of an estimation error. On a group it belongs to the logarithm of one of the two invariant errors. [](#def14-gaussian).

**Cross term.** The term $c\,e_R\cdot J e_\Omega$ added to a Lyapunov function so that its rate is negative in the error vector as well as in the velocity error. [](#eq6-V).

**Cut locus.** The configurations from which the shortest way back to the target is not unique. [](#sec7-local).

**Cutoff.** The bandwidth $k_f$ of the low-pass filter in L1 adaptive control. It sets what reaches the motors and how much delay the loop tolerates. [](#eq8-filter).

**Dead reckoning.** Integrating the readings of a gyroscope, an IMU or an odometer from an estimate, with no other measurement. [](#ex3-dead-reckoning), [](#sec14-autonomous).

**Defect.** What is left when the interpolating polynomial of a collocation is put into the model. The constraint is that it vanishes. [](#sec12-collocation).

**Differentially flat.** A system whose state and inputs are functions of a flat output and finitely many of its time derivatives. Also called flat. [](#def10-flat).

**Direct transcription.** Trajectory optimization that discretizes first and then optimizes over states and inputs together, with the dynamics as constraints. The grid points are knots. [](#sec13-plan), [](#sec12-collocation), [](#sec12-transcribe).

**Direction.** A unit vector $q \in S^2$: a cable, a rod, a thrust axis. [](#sec-error).

**Discrete Euler-Lagrange equations.** The conditions for the action sum to be stationary. They tie three neighboring configurations of a discrete path. [](#eq12-del).

**Discrete Lagrangian.** A function $L_h$ of two neighboring configurations that approximates the action of the motion between them. [](#def12-discrete).

**Discrete momentum.** The momentum $p_k$ defined by the slopes of the discrete Lagrangian. [](#eq12-legendre).

**Double cover.** The two-to-one map from unit quaternions to rotations. [](#sec-quaternions).

**Equations of motion.** The differential equations that give the accelerations of a system. Written out in words, never abbreviated. [](#ch-dynamics).

**Error function.** A nonnegative function $\Psi$ of the actual and desired configurations that vanishes only when they agree. The text also calls it a height. [](#def-error-so3), [](#def3-error-vector).

**Error state.** The pair $(\chi, \delta\Omega)$: the logarithm of the left-invariant error and the deviation of the velocity from the reference. It is the linear state in a form a controller can compute. [](#sec13-error).

**Error vector.** The slope of the error function under a variation: $e_R$, $e_q$. [](#def3-error-vector).

**Error-state MPC.** Model predictive control whose model is the linearized dynamics of the error state. Its optimization problem is a quadratic program. [](#def13-mpc).

**Estimate.** What a filter believes about a quantity, marked with a hat. The quantity itself is called the truth. [](#sec14-gaussian).

**Exponential map.** The map that wraps a tangent space onto the manifold. On a matrix group it is the matrix exponential. [](#sec2-exp).

**Exponential stability on a sublevel set.** Convergence at a guaranteed exponential rate from every start inside a sublevel set of a Lyapunov function. The rate depends on the set and vanishes as the set grows to include half a turn. [](#thm6-exp).

**Extended Kalman filter.** A Kalman filter for a nonlinear model, with the covariance pushed through a linearization. Spelled out in the chapters. The skills abbreviate it EKF. [](#ch-filtering).

**Feedforward.** The part of a control law computed from the model and the reference and not from the error. One word, no hyphen. [](#eq6-law), [](#sec10-feedforward).

**Flat map.** The functions that give the state and the inputs from a flat output and its derivatives. [](#def10-flat).

**Flat output.** A set of as many functions as there are inputs, written $\varphi$, from which the state and the inputs follow without integration. [](#def10-flat).

**Force error.** The difference $e_F = f R\,e_3 - F_d$ between the force a thrust axis delivers and the force the position loop asked for. [](#eq6-Delta).

**Forward invariant.** A set is forward invariant when solutions that start in it stay in it. [](#sec8-barrier).

**Gimbal lock.** The loss of one direction of rotation in an Euler-angle description, at 90° of pitch for yaw, pitch and roll. [](#sec-euler).

**Gradient, Hessian.** The first- and second-order coefficients of a function along a step that stays on the manifold. [](#def13-newton).

**Group-affine.** Dynamics on a group that satisfy [](#eq14-affine). For such dynamics the invariant errors obey equations that involve neither the state nor the estimate. [](#def14-affine).

**Half turn.** A rotation by 180°. Half a turn away from the target is as far as an attitude can be. [](#sec-rotation-space).

**Hamilton's principle.** The motion makes the action stationary among paths with the same endpoints. [](#def4-hamilton).

**Hat map, vee map.** The hat map turns a vector into the skew-symmetric matrix of its cross product. The vee map is its inverse. [](#def-hat).

**Heading.** The angle $\psi$ about the vertical that fixes the rotation of a vehicle about its thrust axis: $R\,e_2$ is perpendicular to $R_z(\psi)\,e_1$. It equals the yaw when the roll or the pitch is zero. For a robot in the plane it is the one angle. [](#eq6-Rd), [](#sec10-quad).

**Homogeneous space.** A manifold on which a group acts so that any point can be moved to any other. The sphere is one, under rotations. [](#sec2-sphere).

**IMU.** Inertial measurement unit: a gyroscope and an accelerometer in one package. [](#sec14-autonomous).

**Innovation.** The predicted reading minus the actual one, $\tilde y$. [](#sec14-circle), [](#eq14-innovation).

**Interior-point method.** A method for inequality constraints that follows the central path to the solution from inside the constraints. [](#sec13-ipm).

**Invariant error.** The left- or right-invariant error between an estimate and the truth, $E_\ell = X^{-1}\hat X$ or $E_r = \hat X X^{-1}$. [](#eq14-errors).

**Invariant extended Kalman filter.** An extended Kalman filter whose error is an invariant error and whose corrections are applied on the group. The right-invariant one is [](#def14-inekf). The skills abbreviate it InEKF.

**Jerk, snap.** The third and fourth time derivatives of a position. [](#def10-flat).

**Kalman gain.** The matrix $\tilde K$ that turns an innovation into a correction. [](#sec14-circle), [](#eq14-gain).

**KKT conditions.** The Karush-Kuhn-Tucker conditions: the first-order conditions for a constrained minimum. [](#eq13-kkt).

**Knot.** A grid point of a transcription, where states and inputs are unknowns. [](#sec12-collocation).

**L1 adaptive control.** An adaptive architecture with a state predictor, a fast adaptation law and a low-pass filter between the estimate and the input. Written L1 in prose. The script letter is kept for the $\mathcal{L}_1$ norm. [](#sec8-l1).

**Lagrange-d'Alembert principle.** Hamilton's principle with the virtual work of the applied forces added. [](#def4-dalembert).

**Landmark.** A point at a known position $x_m$ that a vehicle observes in its body frame. [](#eq14-obs).

**Left-invariant, right-invariant.** Unchanged when both arguments are multiplied by the same group element on the left, or on the right. A left-invariant control law does not depend on the choice of world frame. [](#sec-euler), [](#def3-invariant-errors), [](#eq14-errors).

**Lie algebra.** The tangent space of a Lie group at the identity. [](#def2-lie-group).

**Lie bracket.** The operation on the Lie algebra that records the failure of the group to commute. On $\so(3)$ it is the cross product. [](#sec2-bracket).

**Lie group variational integrator.** A variational integrator whose discrete path is a list of group elements, joined by step rotations. [](#thm12-lgvi).

**Lie group.** A manifold that is also a group, with smooth multiplication and inversion. [](#def2-lie-group).

**Linear state.** The state of a variation-based linearization: a variation and the variation of the velocity. [](#sec7-variation).

**Log-linear property.** For group-affine dynamics the logarithm of an invariant error obeys a linear equation exactly, for errors of any size. [](#thm14-loglinear).

**Logarithm error.** The attitude error vector $\log(R_d\T R)^\vee$, whose length is the error angle. [](#sec3-errors).

**Logarithmic barrier.** The term $-\mu\ln x$ that replaces a constraint $x \ge 0$ in a cost. It is not a barrier function. [](#sec13-ipm).

**Lumped uncertainty.** Everything a controller's model gets wrong, collected in the one vector $\Delta$ that appears in the error dynamics. [](#eq8-lumped).

**Manifold.** A smooth curved set that looks like flat space near each of its points. [](#def2-manifold).

**Matched uncertainty.** An uncertainty that enters the dynamics at the same place as the input, so that a known one could be canceled. [](#sec8-uncertainty).

**Minimum-snap curve.** The curve through given waypoints that minimizes the integral of the squared snap. [](#eq10-minsnap).

**Missing force.** The force a body's motion needs beyond what gravity gives, $m(\ddot x + g\,e_3)$. A cable lies along the missing force of what it carries. [](#thought10-hand).

**Model predictive control.** Control by solving an optimization problem over a receding horizon at every sample. Abbreviated MPC. [](#def13-mpc).

**Multicopter.** A vehicle lifted by several rotors. A quadrotor is a multicopter with four.

**NEES.** Normalized estimation error squared: the squared error weighted by the inverse covariance and divided by the dimension. It is 1 on average for a filter whose covariance tells the truth. [](#tbl14-mc).

**Payload.** A rigid body carried by several quadrotors. A point mass on a cable is a cable-suspended load. [](#sec4-products).

**Pitfall.** A block that warns of a trap. It is always referred to by that name.

**Plain controller.** The tracking law of [](#eq6-law) with no adaptation and no safety filter. [](#sec8-baseline).

**Prediction error.** The error between the state predictor and the vehicle, measured with the error function, the error vector and the transport map. [](#eq8-pred-error).

**Product manifold.** The set of lists of points, one from each factor. [](#sec2-products).

**Quadratic program.** The minimization of a quadratic cost under linear constraints. Abbreviated QP. [](#eq8-safety-filter), [](#eq13-qp).

**Reading.** One output of a sensor, $y$. [](#eq14-obs).

**Receding horizon.** Solving over the next $N$ steps, applying the first input, and solving again at the next sample. [](#def13-mpc).

**Reference.** A desired motion that changes with time, marked by the subscript $d$. [](#sec3-transport).

**Relative degree.** The number of times a function of the state must be differentiated before the input appears. [](#sec6-fl), [](#sec8-cone).

**Restoring torque.** The proportional part of an attitude controller, $-k_p\,e$ for an error vector $e$. [](#warn-sine).

**Retraction.** A rule that moves a point of a manifold along a tangent vector, lands on the manifold, and agrees with the exponential map to first order. [](#def13-retraction).

**Robot model library.** The Python modules of `case_studies/robots`, one per robot, with one interface. [](#sec16-library).

**Safe set.** The set of states a system must never leave, written $\{H \ge 0\}$. [](#sec8-barrier).

**Safety filter.** A program that changes a proposed input by the smallest amount that satisfies a barrier condition. [](#eq8-safety-filter).

**Sampling period.** The time between two updates of a sampled controller. It is the time step $h$. [](#sec13-qp).

**Slack variable.** A variable $w$ that turns an inequality into an equality, or lets a constraint be violated at a cost. A slack cable is a cable without tension. [](#eq8-qp), [](#eq13-kkt).

**Specific force.** What an accelerometer reads: the acceleration minus gravity, in the body frame, $a_s$. [](#eq14-imu).

**State predictor.** A copy of the believed dynamics, driven by the estimate of the uncertainty and run alongside the vehicle. The L1 literature calls it the reference model. [](#sec8-l1).

**Step rotation, step velocity.** The rotation $\Upsilon_k = R_k\T R_{k+1}$ over one time step, and its rotation vector divided by $h$. [](#def12-step).

**Störmer-Verlet.** The second-order symplectic method that the simplest discrete Lagrangian produces. [](#eq12-three).

**Sublevel set.** The set of states where a function is below a given value. A sublevel set of a Lyapunov function that never increases is a set no solution can leave. [](#sec6-lyapunov).

**Symplectic.** A step that preserves area in the phase plane, and its generalization to more degrees of freedom. [](#sec12-three).

**Tangent space, tangent plane.** The vector space of velocities allowed at a point of a manifold. On the sphere it is a plane and is called the tangent plane. [](#def2-manifold).

**Target.** A desired configuration that does not move, marked by the subscript $d$. [](#sec-error).

**Tension vector.** The tension of a link times its direction, $T_i\,q_i$. [](#eq10-chain).

**Thrust vector.** The force $F = f R\,e_3$ of the rotors, in the world frame. [](#eq10-quad-force).

**Trace error.** The attitude error function $\tfrac12\tr(I - R_d\T R)$ and its error vector $e_R$. [](#sec3-errors).

**Transport map.** A rule that carries a velocity at the desired configuration to a velocity at the actual one, so that the two can be subtracted. [](#def3-transport).

**Twist.** An element of $\se(3)$: a linear and an angular velocity, $\zeta = (V, \Omega)$. [](#sec2-se3).

**Unwinding.** A rotation the long way round, caused by a controller that treats $\quat$ and $-\quat$ as different attitudes. [](#sec-quaternions).

**Variation-based extended Kalman filter.** An extended Kalman filter on the sphere whose error is the linear state $s = (\xi, \delta\omega)$ of the variation-based linearization. [](#def14-vekf).

**Variation.** The first-order gap between a curve and its neighbors in a family of curves that stay on the manifold. The text also calls it a nudge. [](#def3-variation).

**Variational integrator.** An integrator obtained from a discrete Hamilton's principle and not from the equations of motion. [](#sec12-discrete).

**Velocity error.** The actual velocity minus the transported desired velocity: $e_\Omega$, $e_\omega$. [](#def3-transport).

**Virtual work.** The work the applied forces would do on a variation. [](#def4-dalembert).

**Warm start.** Starting a solver from the solution of the previous problem. [](#sec13-flight).

**Waypoint.** A point that a planned curve must pass at a given time. [](#sec10-trajectories).

**World frame.** The frame that does not move, with $e_3$ up. [](#sec-rotation-space).

**World-frame angular velocity.** The angular velocity expressed in the world frame, $\omega$. [](#sec2-body-spatial).

**Yaw.** The first Euler angle of $\Theta = (\psi, \theta, \phi)$. Not the heading, unless the roll or the pitch is zero. [](#eq-zyx).

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
| cable-suspended load (point mass), payload (rigid body) | suspended load, slung load, payload for a point mass |
| quadrotor, multicopter | quadcopter, multirotor, UAV |
| gyroscope, accelerometer | gyro, accel |
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
| $\hat R$, $\hat q$, $\hat\omega$ for estimates | $\omega_{\hat q}$, a subscript "est", a hat for the hat map |
| truth, estimate | ground truth, actual state for the true one |
| $E_r = \hat X X^{-1}$, $E_\ell = X^{-1}\hat X$ | the truth in the first slot |
| $\chi_r$, $\chi$ | $\xi$ for the logarithm of an invariant error |
| $\Sigma$, $\Sigma_n$, $\Sigma_y$ | $P$, $Q$, $N$ for covariances |
| $\tilde K$, $C_y$ | $K$, $H$ for the Kalman gain and the measurement matrix |
| $\Pi_C$ | $E_C$ for the projector onto $\ker C$ |
| $x_m$ | $p$ for a landmark |
| specific force | proper acceleration |
| reading | measurement sample, observation |
| extended Kalman filter | EKF in the chapters |
| $L_h$, $S_h$ | $L_d$, $S_d$ |
| $\Upsilon_k$ | $F_k$ for a step rotation |
| step rotation, step velocity | relative rotation, discrete velocity |
| discrete Lagrangian, action sum | discrete action |
| knot | node, grid node, mesh point |
| $l$ for a length, $\ell$ only in $E_\ell$ | $\ell$ for a length, $\Sigma_\ell$ and other subscripts $\ell$ |
| Störmer-Verlet | leapfrog, velocity Verlet |
| Hermite-Simpson, trapezoidal collocation | the same with an en dash, trapezoid collocation |
| flat output, flat map | flat coordinates, endogenous transformation |
| differentially flat | differentially-flat, except in a title |
| heading $\psi$ | yaw, for the angle of [](#eq6-Rd) |
| jerk, snap | third and fourth derivative, once named |
| thrust vector | thrust force vector |
| minimum-snap | min-snap, minimum snap as an adjective |
| waypoint | way-point, via point |
| robot model library | model zoo, robot zoo |

: Preferred terms and forms.

:::

<!-- terms:on -->
