Chapter 7: Variation-Based Linearization, State-Constrained Controllability & LTV-LQR
Linearizing nonlinear mechanical systems evolving on curved configuration manifolds—such as $SO(3)$, $S^2$, and their products $\prod_i M_i$—using local coordinates (e.g., Euler angles or spherical coordinates) introduces representation singularities and cumbersome trigonometric expressions. Building on the coordinate-free variation framework of Wu & Sreenath (IEEE Access 2015) and its extension to multi-link cable-suspended aerial systems in Kotaru, Wu, & Sreenath (arXiv:1711.04895), this chapter develops symbolic variation-based linearization, establishes state-constrained controllability on the invariant nullspace of the tangent-bundle constraints, and synthesizes finite-horizon LTV-LQR controllers directly on the manifold.


________________


7.1 Notation and Lie-Algebra Preliminaries
We use the standard hat map $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ defined by


$$ x^\wedge y = x \times y, \qquad \forall x, y \in \mathbb{R}^3, \tag{7.1} $$


and its inverse vee map $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$ satisfying $(x^\wedge)^\vee = x$. For any $x, y \in \mathbb{R}^3$ and $R \in SO(3)$, the hat map obeys $x^\wedge y = -y^\wedge x$, $(x^\wedge)^T = -x^\wedge$, and $(R x)^\wedge = R x^\wedge R^T$.


________________


7.2 Variation-Based Linearization on Product Manifolds $\prod_i M_i$
Consider a mechanical system evolving on a product manifold $\mathcal{Q} = \prod_{i=1}^k M_i$, where each factor $M_i \in {\mathbb{R}^3, S^2, SO(3)}$, with state $x(t) \in T\mathcal{Q}$ and control input $u(t) \in \mathbb{R}^m$. Given a smooth, dynamically feasible desired trajectory $(x_d(t), u_d(t))$, we perturb the trajectory via a smooth one-parameter family $(x_\epsilon(t), u_\epsilon(t))$ with $x_0(t) = x_d(t)$ and $u_0(t) = u_d(t)$, and define the variation $\delta x = \left.\frac{d}{d\epsilon}\right|{\epsilon=0} x\epsilon$ and $\delta u = \left.\frac{d}{d\epsilon}\right|{\epsilon=0} u\epsilon$.


Manifold Factor $M_i$
	Configuration Variation
	Velocity Variation
	Algebraic Constraint $C_i(x_d) s_i = 0$
	$\mathbb{R}^3$
	$\delta p = \delta p \in \mathbb{R}^3$
	$\delta v = \delta \dot{p} \in \mathbb{R}^3$
	None ($C_i = 0$)
	$S^2$
	$\delta q = \xi \times q_d = -q_d^\wedge \xi$
	$\delta \omega \in \mathbb{R}^3$
	$\xi \cdot q_d = 0$, $(\xi \times q_d) \cdot \omega_d + q_d \cdot \delta \omega = 0$
	$SO(3)$
	$\delta R = R_d \eta^\wedge$
	$\delta \Omega = \Omega_d^\wedge \eta + \dot{\eta}$
	None ($C_i = 0$)
	

Table 7.1: Canonical tangent-space variations and algebraic constraints on elementary manifold factors $M_i \in {\mathbb{R}^3, S^2, SO(3)}$ (Wu & Sreenath, 2015).
7.2.1 Variations on $\mathbb{R}^3$
For $p \in \mathbb{R}^3$ with $\dot{p} = v$, the perturbed state $p_\epsilon = p_d + \epsilon \delta p$ yields linear kinematics without constraints:


$$ \delta \dot{p} = \delta v. \tag{7.2} $$
7.2.2 Variations on $SO(3)$
For $R \in SO(3)$ with body-frame angular velocity $\Omega \in \mathbb{R}^3$ ($\dot{R} = R \Omega^\wedge$), the manifold-preserving perturbation is $R_\epsilon(t) = R_d(t) \exp(\epsilon \eta^\wedge(t))$ with $\eta(t) \in \mathbb{R}^3$, giving


$$ \delta R = R_d \eta^\wedge. \tag{7.3} $$


Differentiating $\delta R$ with respect to time and equating it to $\delta(R \Omega^\wedge) = \delta R,\Omega_d^\wedge + R_d,\delta\Omega^\wedge$ yields $R_d(\Omega_d^\wedge \eta^\wedge + \dot{\eta}^\wedge) = R_d(\eta^\wedge \Omega_d^\wedge + \delta\Omega^\wedge)$. Using $[\Omega_d^\wedge, \eta^\wedge] = (\Omega_d^\wedge \eta)^\wedge$, we obtain


$$ \delta \Omega = \Omega_d^\wedge \eta + \dot{\eta} \quad \Longleftrightarrow \quad \dot{\eta} = -\Omega_d^\wedge \eta + \delta \Omega. \tag{7.4} $$


Because $\eta \in \mathbb{R}^3$ and $\delta \Omega \in \mathbb{R}^3$ are unconstrained, the variation state $s_{SO(3)} = [\eta^T, \delta \Omega^T]^T \in \mathbb{R}^6$ requires no algebraic constraint.
7.2.3 Variations on $S^2$
For $q \in S^2 = {q \in \mathbb{R}^3 : q^T q = 1}$ with $\dot{q} = \omega \times q$ and $q \cdot \omega = 0$, the perturbation $q_\epsilon(t) = \exp(\epsilon \xi^\wedge(t)) q_d(t)$ with $\xi \in T_{q_d} S^2$ yields


$$ \delta q = \xi \times q_d = -q_d^\wedge \xi, \qquad \xi \cdot q_d = 0. \tag{7.5} $$


Taking the variation of $q \cdot \omega = 0$ gives the velocity-level constraint


$$ (\xi \times q_d) \cdot \omega_d + q_d \cdot \delta \omega = 0 \quad \Longleftrightarrow \quad (q_d^\wedge \omega_d)^T \xi + q_d^T \delta \omega = 0. \tag{7.6} $$


Equating $\frac{d}{dt}(\delta q) = \dot{\xi} \times q_d + \xi \times (\omega_d \times q_d)$ with $\delta \dot{q} = \delta \omega \times q_d + \omega_d \times (\xi \times q_d)$ and applying the Jacobi identity gives $(\dot{\xi} - \delta \omega - \omega_d \times \xi) \times q_d = 0$. Enforcing $\frac{d}{dt}(\xi \cdot q_d) = 0$ and Eq. (7.6) uniquely determines the normal component along $q_d$, yielding


$$ \dot{\xi} = \omega_d^\wedge \xi + (I_3 - q_d q_d^T) \delta \omega. \tag{7.7} $$


Stacking $s_{S^2} = [\xi^T, \delta \omega^T]^T \in \mathbb{R}^6$, the constraints Eqs. (7.5)–(7.6) take the compact form $C_{S^2}(t) s_{S^2} = 0$, where


$$ C_{S^2}(t) = \begin{bmatrix} q_d^T & 0_{1\times 3} \ (q_d^\wedge \omega_d)^T & q_d^T \end{bmatrix} \in \mathbb{R}^{2 \times 6}. \tag{7.8} $$
7.2.4 General Constrained LTV System on $\prod_i M_i$
Assembling the variation vectors $s_i$ of all manifold factors into $s \in \mathbb{R}^n$ and linearizing the Newton–Euler equations of motion about $(x_d(t), u_d(t))$ yields the symbolic constrained linear time-varying (LTV) system (Wu & Sreenath, 2015; Kotaru, Wu, & Sreenath, 2017):


$$ \dot{s} = A(x_d(t)) s + B(x_d(t)) \delta u, \tag{7.9a} $$ $$ C(x_d(t)) s = 0. \tag{7.9b} $$


Here $A(x_d(t)) \in \mathbb{R}^{n \times n}$, $B(x_d(t)) \in \mathbb{R}^{n \times m}$, and $C(x_d(t)) \in \mathbb{R}^{p \times n}$ are expressed purely in terms of $(x_d(t), u_d(t))$ without local coordinate charts.


________________


7.3 State-Constrained Controllability
Because the variation state $s(t)$ is restricted to the time-varying nullspace $\mathcal{N}(C(x_d(t))) = {s \in \mathbb{R}^n : C(x_d(t)) s = 0}$, classical unconstrained controllability criteria cannot be applied blindly without first verifying that the constraint manifold is invariant under the linearized flow.


Def. 7.1 (State-Constrained Controllability). The constrained LTV system Eq. (7.9) on $[t_0, t_f]$ is state-constrained controllable if for every initial state $s_0 \in \mathcal{N}(C(x_d(t_0)))$ and every terminal state $s_f \in \mathcal{N}(C(x_d(t_f)))$, there exists a piecewise continuous control input $\delta u : [t_0, t_f] \to \mathbb{R}^m$ such that the solution $s(t)$ of Eq. (7.9a) with $s(t_0) = s_0$ satisfies $s(t_f) = s_f$ and $C(x_d(t)) s(t) = 0$ for all $t \in [t_0, t_f]$ (Wu & Sreenath, 2015).


Lemma 7.1 (Time-Invariance of $\mathcal{N}(C)$). Suppose that for all $t \in [t_0, t_f]$,


$$ C(t) B(t) = 0 \quad \text{and} \quad \big(C(t) A(t) + \dot{C}(t)\big) s = 0 ;;; \forall s \in \mathcal{N}(C(t)). \tag{7.10} $$


Then $\frac{d}{dt}\big(C(t) s(t)\big) \equiv 0$ along every trajectory of Eq. (7.9a). Consequently, if $s(t_0) \in \mathcal{N}(C(t_0))$, then $s(t) \in \mathcal{N}(C(t))$ for all $t \ge t_0$, regardless of the control input $\delta u(t)$.


Proof. Differentiating $C(t) s(t)$ along Eq. (7.9a) gives $\frac{d}{dt}(C s) = (CA + \dot{C}) s + CB,\delta u$. Under Eq. (7.10), $\frac{d}{dt}(Cs) = 0$ whenever $s \in \mathcal{N}(C)$, so $C(t)s(t) = C(t_0)s(t_0) = 0$ for all $t \ge t_0$. $\square$


Remark 7.1. Lemma 7.1 is the cornerstone of constrained LTV control on manifolds: once $s(t_0) \in \mathcal{N}(C(t_0))$ is satisfied by construction of the variation map, the unconstrained dynamics Eq. (7.9a) automatically preserve Eq. (7.9b) for all $\delta u(t)$.


To test controllability on $\mathcal{N}(C(t))$, define the linear differential operator


$$ s(B) = \dot{B}(t) + A(t) B(t), \qquad s^0(B) = B(t), \qquad s^k(B) = s\big(s^{k-1}(B)\big). \tag{7.11} $$


Prop. 7.1 (State-Constrained Controllability Criterion). Assume Lemma 7.1 holds and $A(t), B(t)$ are sufficiently smooth on $[t_0, t_f]$. If there exists $k \ge 1$ such that the columns of the controllability matrix


$$ \mathcal{C}_k(t) = \begin{bmatrix} B(t) & s(B)(t) & \cdots & s^{k-1}(B)(t) \end{bmatrix} \tag{7.12} $$


span $\mathcal{N}(C(t))$ (i.e., $\operatorname{rank}(\mathcal{C}_k(t)) = \dim \mathcal{N}(C(t)) = n - \operatorname{rank}(C(t))$) for all $t \in [t_0, t_f]$, then Eq. (7.9) is state-constrained controllable on $[t_0, t_f]$ (Wu & Sreenath, 2015).
7.3.1 Application to the Spherical Pendulum on $S^2$
Consider a spherical pendulum of mass $m$ and length $l$ with $q \in S^2$ and $q \cdot \omega = 0$:


$$ \dot{q} = \omega \times q, \qquad \dot{\omega} = \frac{g}{l} e_3^\wedge q + \frac{1}{m l^2}(I_3 - q q^T) u. \tag{7.13} $$


Linearizing Eq. (7.13) about $(q_d(t), \omega_d(t), u_d(t))$ using Eqs. (7.5)–(7.7) gives $s = [\xi^T, \delta \omega^T]^T \in \mathbb{R}^6$ with $C(t) = C_{S^2}(t)$ from Eq. (7.8) and


$$ A(t) = \begin{bmatrix} \omega_d^\wedge & I_3 - q_d q_d^T \ A_{21}(t) & 0_{3\times 3} \end{bmatrix}, \qquad B(t) = \begin{bmatrix} 0_{3\times 3} \ \frac{1}{m l^2}(I_3 - q_d q_d^T) \end{bmatrix}, \tag{7.14} $$


where $A_{21}(t) = -\frac{g}{l} e_3^\wedge q_d^\wedge + \frac{1}{ml^2}\big((q_d^T u_d)I_3 + q_d u_d^T\big) q_d^\wedge$. Direct differentiation confirms $C B = 0$ and $(CA + \dot{C}) s = 0$ for all $s \in \mathcal{N}(C)$ (verified numerically to $\mathcal{O}(10^{-9})$ in ch07_variation_lqr_sim.ipynb). Moreover,


$$ s(B) = \dot{B} + A B = \begin{bmatrix} \frac{1}{m l^2}(I_3 - q_d q_d^T) \ -\frac{1}{m l^2}(\dot{q}_d q_d^T + q_d \dot{q}_d^T) \end{bmatrix}. \tag{7.15} $$


Since $\operatorname{rank}(I_3 - q_d q_d^T) = 2$ and the upper and lower $3 \times 3$ blocks of $[B, s(B)]$ are block-triangular with respect to $T_{q_d}S^2$, $\operatorname{rank}([B, s(B)]) = 4 = \dim \mathcal{N}(C(t))$ for every smooth trajectory $(q_d(t), \omega_d(t))$. By Prop. 7.1, the spherical pendulum is state-constrained controllable along all feasible trajectories.
7.3.2 Application to the 3D Pendulum on $SO(3)$
Consider a 3D rigid-body pendulum with attitude $R \in SO(3)$, body angular velocity $\Omega \in \mathbb{R}^3$, inertia $J \succ 0$, mass $m$, and center-of-mass offset $\rho \in \mathbb{R}^3$:


$$ \dot{R} = R \Omega^\wedge, \qquad J \dot{\Omega} + \Omega \times J \Omega = -m g \rho^\wedge R^T e_3 + u. \tag{7.16} $$


Applying Eqs. (7.3)–(7.4) with $s = [\eta^T, \delta \Omega^T]^T \in \mathbb{R}^6$ ($C(t) = 0$) yields


$$ A(t) = \begin{bmatrix} -\Omega_d^\wedge & I_3 \ -m g J^{-1} \rho^\wedge (R_d^T e_3)^\wedge & J^{-1}\big((J\Omega_d)^\wedge - \Omega_d^\wedge J\big) \end{bmatrix}, \qquad B(t) = \begin{bmatrix} 0_{3\times 3} \ J^{-1} \end{bmatrix}. \tag{7.17} $$


Here $s(B) = \dot{B} + AB = \begin{bmatrix} J^{-1} \ J^{-1}((J\Omega_d)^\wedge - \Omega_d^\wedge J) J^{-1} \end{bmatrix}$, so $\operatorname{rank}([B, s(B)]) = 6 = \dim(\mathbb{R}^6)$ for all $t$, proving controllability along every trajectory on $SO(3)$.


________________


7.4 Finite-Horizon Riccati LTV-LQR Design
Because Lemma 7.1 guarantees that $\mathcal{N}(C(t))$ is forward-invariant under Eq. (7.9a), we can synthesize a finite-horizon Linear Quadratic Regulator directly on the unconstrained pair $(A(t), B(t))$ (Kotaru, Wu, & Sreenath, 2017). Given symmetric weighting matrices $Q(t) \succeq 0$, $P_f \succeq 0$, and $R(t) \succ 0$, we minimize


$$ J = \frac{1}{2} s(t_f)^T P_f s(t_f) + \frac{1}{2} \int_{t_0}^{t_f} \Big(s(t)^T Q(t) s(t) + \delta u(t)^T R(t) \delta u(t)\Big) dt. \tag{7.18} $$


The optimal feedback gain is obtained by integrating the Continuous-Time Differential Riccati Equation (DRE) backward from $t_f$ to $t_0$:


$$ -\dot{P}(t) = A(t)^T P(t) + P(t) A(t) - P(t) B(t) R(t)^{-1} B(t)^T P(t) + Q(t), \qquad P(t_f) = P_f, \tag{7.19} $$


yielding the LTV-LQR control law


$$ K(t) = R(t)^{-1} B(t)^T P(t), \qquad \delta u(t) = -K(t) s(t), \qquad u(t) = u_d(t) + \delta u(t). \tag{7.20} $$


To evaluate $s(t)$ from the nonlinear state $(x(t), x_d(t))$ at runtime:


* On $SO(3)$: $\eta = \big(\log(R_d^T R)\big)^\vee$ and $\delta \Omega = \Omega - \Omega_d$.
* On $S^2$: $\xi = \frac{\theta}{\sin\theta}(q_d \times q)$ with $\theta = \arccos(q_d^T q)$, and $\delta \omega = \omega - \omega_d - \big(q_d \cdot (\omega - \omega_d) + (\xi \times q_d)\cdot \omega_d\big) q_d$, which enforces $C_{S^2}(t) s(t) = 0$ to machine precision even at large initial errors.


________________


7.5 Worked Examples
Worked Example 7.1: Spherical Pendulum Tracking from $50^\circ$ Initial Error
In ch07_variation_lqr_sim.ipynb, we simulate Eq. (7.13) with $m = 1.0\text{ kg}$, $l = 1.0\text{ m}$, $g = 9.81\text{ m/s}^2$ along a time-varying conical figure-8 trajectory on $S^2$ over $t \in [0, 10]\text{ s}$, using $Q = \operatorname{diag}(40, 40, 40, 8, 8, 8)$, $R = 0.5 I_3$, and $P_f = \operatorname{diag}(50, 50, 50, 10, 10, 10)$. Starting from a $50.00^\circ$ attitude error and velocity perturbation $[0.8, -0.6, 0]^T\text{ rad/s}$:


* Constraint invariance holds with $\max_t |\frac{d}{dt}(C s)| = 1.104 \times 10^{-9}$ and $\max_t |C(t)s(t)| = 6.942 \times 10^{-16}$.
* Controllability rank is $\min_t \operatorname{rank}([B, s(B)]) = 4 = \dim \mathcal{N}(C)$.
* Closed-loop tracking converges to $0.0000^\circ$ attitude error and $1.853 \times 10^{-9}\text{ rad/s}$ velocity error (Fig. 7.1 in ch07_variation_lqr_sim.ipynb).
Worked Example 7.2: 3D Pendulum Tracking on $SO(3)$ from $60^\circ$ Initial Error
For the 3D pendulum Eq. (7.16) with $m = 1.5\text{ kg}$, $J = \operatorname{diag}(0.12, 0.15, 0.08)\text{ kg}\cdot\text{m}^2$, and $\rho = [0, 0, -0.5]^T\text{ m}$ over $t \in [0, 10]\text{ s}$, using $Q = \operatorname{diag}(50, 50, 50, 5, 5, 5)$ and $R = 0.2 I_3$:


* Controllability rank is $\min_t \operatorname{rank}([B, s(B)]) = 6$.
* Integrating the nonlinear dynamics with a geometric Lie-group Heun integrator preserves $\max_t |\det(R) - 1| = 6.883 \times 10^{-15}$.
* Starting from a $60.00^\circ$ attitude error, the LTV-LQR controller stabilizes the trajectory to $0.0004^\circ$ attitude error and $3.972 \times 10^{-4}\text{ rad/s}$ velocity error (Fig. 7.2 in ch07_variation_lqr_sim.ipynb).


________________


7.6 Exercises
1. Product Manifold Linearization ($SE(3) \times S^2$): Derive $(A(t), B(t), C(t))$ for a rigid-body quadrotor ($SE(3) = \mathbb{R}^3 \times SO(3)$) carrying a point-mass payload via a single rigid massless link ($S^2$), following Kotaru, Wu, & Sreenath (2017). Verify that $C(t) \in \mathbb{R}^{2 \times 16}$ and $\dim \mathcal{N}(C) = 14$.
2. Invariance Proof: Prove analytically that for the spherical pendulum Eq. (7.13)–(7.14), the second row of $(CA + \dot{C}) s$ vanishes identically for all $s \in \mathcal{N}(C)$ even when $u_d(t)$ has a nonzero normal component $q_d^T u_d \neq 0$.
3. Controllability with Restricted Input: Suppose the spherical pendulum is actuated only by a horizontal force $f \in \mathbb{R}^2$ in the $e_1$–$e_2$ plane so that $u = q \times [f_1, f_2, 0]^T$. When does $\operatorname{rank}([B, s(B)])$ drop below $4$?
4. Riccati Projection Comparison: In ch07_variation_lqr_sim.ipynb, compare the closed-loop trajectory when $P(t)$ is integrated directly in $\mathbb{R}^{6\times 6}$ versus projecting $P(t)$ onto $\mathcal{N}(C(t))$ via $N(t)^T P(t) N(t)$. Explain why both yield identical feedback $\delta u = -K(t) s(t)$ when $s(t) \in \mathcal{N}(C(t))$.