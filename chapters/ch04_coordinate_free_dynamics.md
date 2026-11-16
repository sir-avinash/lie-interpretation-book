Chapter 4: Coordinate-Free Dynamics: Riemannian Mechanics, Lagrange–d'Alembert & Multi-Body Formulations
Chapter Roadmap. Classical Euler–Lagrange equations written in local coordinates $q \in \mathbb{R}^n$ suffer from chart singularities (such as gimbal lock on $SO(3)$ or polar singularities on $S^2$) and obscure the geometric structure of mechanical energy and constraints. This chapter develops coordinate-free equations of motion directly on configuration manifolds: (1) Riemannian simple mechanical systems via the Levi-Civita covariant derivative, (2) the Lagrange–d'Alembert principle on $SE(3) \times S^2$ for a quadrotor with a cable-suspended load (including taut/slack hybrid transitions), (3) Newton–Euler flexible-cable chains on $SO(3) \times \mathbb{R}^3 \times (S^2)^n$ and modular Boltzmann–Hamel whole-body dynamics on $SE(3) \times \mathbb{R}^{N_T}$, and (4) worked examples and exercises.


________________


4.1 Notation and Lie-Algebraic Conventions
Throughout this chapter, we adhere to the following canonical conventions:


* Hat and Vee Maps. The hat map $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ is the Lie algebra isomorphism defined by $$x^\wedge y = x \times y, \qquad \forall x, y \in \mathbb{R}^3, \tag{4.1}$$ explicitly given for $x = [x_1, x_2, x_3]^T$ by $$x^\wedge = \begin{bmatrix} 0 & -x_3 & x_2 \ x_3 & 0 & -x_1 \ -x_2 & x_1 & 0 \end{bmatrix}, \tag{4.2}$$ with inverse $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$.
* Kinematics on $SO(3)$. For a rigid body with attitude $R \in SO(3) := {R \in \mathbb{R}^{3\times 3} \mid R^T R = I,, \det(R) = +1}$, the body-fixed angular velocity $\Omega \in \mathbb{R}^3$ satisfies $$\dot{R} = R \Omega^\wedge. \tag{4.3}$$
* Kinematics on $S^2$. For a unit direction vector $q \in S^2 := {q \in \mathbb{R}^3 \mid q \cdot q = 1}$, its angular velocity $\omega \in \mathbb{R}^3$ satisfies $$\dot{q} = \omega \times q = \omega^\wedge q, \qquad \omega \cdot q = 0. \tag{4.4}$$ The orthogonality condition $\omega \cdot q = 0$ ensures uniqueness of $\omega$ in the tangent plane $T_q S^2 \cong {v \in \mathbb{R}^3 \mid v \cdot q = 0}$.


Symbol
	Space
	Physical Meaning
	$M$, $T_q M$, $T_q^* M$
	Manifold, tangent, cotangent
	Configuration space and dual velocity/force fibers
	$\llangle \cdot, \cdot \rrangle_q$
	$T_q M \times T_q M \to \mathbb{R}$
	Inertia Riemannian metric $\langle M_q \dot{q}_1, \dot{q}_2 \rangle$
	$\nabla_X Y$
	$\Gamma(TM)$
	Levi-Civita covariant derivative compatible with $M_q$
	$R, \Omega$
	$SO(3), \mathbb{R}^3$
	Quadrotor/base attitude and body angular velocity
	$q, \omega$
	$S^2, \mathbb{R}^3$
	Cable/pendulum unit direction and angular velocity ($\omega \cdot q = 0$)
	$x_Q, v_Q, x_L, v_L$
	$\mathbb{R}^3$
	Quadrotor and load inertial positions and velocities
	$f, M$
	$\mathbb{R}, \mathbb{R}^3$
	Quadrotor scalar thrust and body-frame moment
	$g_b, \theta$
	$SE(3), \mathbb{R}^{N_T}$
	Floating-base pose and $N_T$ internal joint angles
	

Table 4.1: Summary of coordinate-free geometric and mechanical notation used in Chapter 4.


________________


4.2 Riemannian Simple Mechanical Systems
Following the geometric control formulation in Wu & Sreenath (IEEE TAC 2016), a mechanical system evolving on a smooth $m$-dimensional configuration manifold $M$ is modeled directly on the tangent bundle $TM = \bigcup_{q \in M} T_q M$ without choosing local charts.


Def. 4.1 (Simple Mechanical System on a Riemannian Manifold). A simple mechanical system is a tuple $(M, M_q, V_q, {F_i}_{i=1}^m)$ consisting of:


1. A smooth configuration manifold $M$ with state $(q, \dot{q}) \in TM$.
2. An inertia tensor $M_q : T_q M \to T_q^* M$ that induces a Riemannian metric (the kinetic-energy metric): $$\llangle \dot{q}_1, \dot{q}_2 \rrangle_q = \langle M_q \dot{q}_1, \dot{q}_2 \rangle, \qquad \forall \dot{q}_1, \dot{q}_2 \in T_q M, \tag{4.5}$$ so that the kinetic energy is $T(q, \dot{q}) = \frac{1}{2} \llangle \dot{q}, \dot{q} \rrangle_q$.
3. The unique torsion-free, metric-compatible Levi-Civita connection $\nabla$ determined by $\llangle \cdot, \cdot \rrangle_q$, which defines the geometric acceleration $\nabla_{\dot{q}} \dot{q} \in T_q M$.
4. A smooth potential energy function $V_q : M \to \mathbb{R}$ whose differential $dV_q(q) \in T_q^* M$ is a one-form acting on $X \in T_q M$ via $\langle dV_q, X \rangle_q = \left.\frac{d}{dt}\right|_{t=0} V_q(\alpha(t))$ for a curve $\alpha(0)=q, \dot{\alpha}(0)=X$.
5. A collection of input force one-forms $F_i(q, \dot{q}) \in T_q^* M$ ($i = 1, \dots, m$) paired with scalar control inputs $u_i \in \mathbb{R}$.


Prop. 4.1 (Coordinate-Free Equations of Motion). The dynamics of a simple mechanical system $(M, M_q, V_q, {F_i})$ are governed by the second-order geometric differential equation $$\nabla_{\dot{q}} \dot{q} = M_q^{-1}!\left(-dV_q(q) + \sum_{i=1}^m F_i(q, \dot{q}) u_i\right). \tag{4.6}$$


Remark 4.1 (Why Covariant Acceleration Matters). In Euclidean space $M = \mathbb{R}^n$ with constant mass $M_q = m I$, the Christoffel symbols vanish and $\nabla_{\dot{q}} \dot{q} = \ddot{q}$. On curved manifolds such as $S^2$ or $SO(3)$, $\nabla_{\dot{q}} \dot{q}$ automatically incorporates the centripetal/Coriolis curvature terms needed to keep the trajectory tangent to $M$. For example, on $S^2$ with the standard round metric $\llangle \dot{q}_1, \dot{q}_2 \rrangle = \dot{q}_1 \cdot \dot{q}2$, the covariant acceleration of a curve $q(t) \in S^2$ is the orthogonal projection of $\ddot{q}$ onto $T_q S^2$: $$\nabla{\dot{q}} \dot{q} = \ddot{q} + (\dot{q} \cdot \dot{q}) q = -q^\wedge q^\wedge \ddot{q}. \tag{4.7}$$
4.2.1 Configuration Errors and Transport Maps
To compare velocities at a current configuration $q \in M$ and a reference configuration $q_d \in M$—which live in different tangent spaces $T_q M$ and $T_{q_d} M$—we pair a smooth quadratic configuration error function $\Psi : M \times M \to [0, \infty)$ with a compatible transport map $T_{(q, q_d)} : T_{q_d} M \to T_q M$ (Wu & Sreenath, IEEE TAC 2016): $$e_q = d_1 \Psi(q, q_d), \qquad d_2 \Psi(q, q_d) = -T_{(q, q_d)}^* d_1 \Psi(q, q_d), \qquad e_{\dot{q}} = \dot{q} - T_{(q, q_d)} \dot{q}_d. \tag{4.8}$$


Manifold $M$
	Error Function $\Psi(q, q_d)$
	Configuration Error $e_q = d_1\Psi$
	Velocity Error $e_{\dot{q}}$
	$\mathbb{R}^3$
	$\frac{1}{2}|q - q_d|^2$
	$q - q_d$
	$\dot{q} - \dot{q}_d$
	$S^2$
	$1 - q \cdot q_d$
	$(q_d \times q)$ (or $q^\wedge q^\wedge q_d$ in $T_q S^2$)
	$\dot{q} - (q_d \times \dot{q}_d) \times q$
	$SO(3)$
	$\frac{1}{2}\operatorname{tr}(I - R_d^T R)$
	$\frac{1}{2}(R_d^T R - R^T R_d)^\vee$
	$\Omega - R^T R_d \Omega_d$
	

Table 4.2: Canonical configuration errors and compatible transport maps on $\mathbb{R}^3$, $S^2$, and $SO(3)$.


________________


4.3 Lagrange–d'Alembert Principle on $SE(3) \times S^2$
We now derive the coordinate-free dynamics of a quadrotor UAV carrying a point-mass payload via a massless cable, following Sreenath, Lee, & Kumar (CDC 2013).
4.3.1 Configuration Space and Kinematics (Taut Mode)
When the cable of length $l > 0$ is taut, the system has $8$ degrees of freedom and evolves on the product manifold $$Q_n = SE(3) \times S^2 \cong \mathbb{R}^3 \times SO(3) \times S^2, \tag{4.9}$$ with $4$ control inputs $(f, M) \in \mathbb{R} \times \mathbb{R}^3$ ($4$ degrees of underactuation). Using the $z$-up inertial frame ($e_3 = [0, 0, 1]^T$ pointing upward) and letting $q \in S^2$ denote the unit vector from the quadrotor center of mass $x_Q \in \mathbb{R}^3$ to the load position $x_L \in \mathbb{R}^3$, the kinematic constraint is $$x_Q = x_L - l q, \qquad v_Q = v_L - l \dot{q}. \tag{4.10}$$


The kinetic energy $T : TQ_n \to \mathbb{R}$ and potential energy $U : Q_n \to \mathbb{R}$ are $$T = \frac{1}{2} m_Q v_Q \cdot v_Q + \frac{1}{2} m_L v_L \cdot v_L + \frac{1}{2} \langle \Omega, J_Q \Omega \rangle, \tag{4.11}$$ $$U = m_Q g e_3 \cdot x_Q + m_L g e_3 \cdot x_L, \tag{4.12}$$ with Lagrangian $L = T - U$. Here $m_Q, m_L > 0$ are the quadrotor and load masses, and $J_Q \in \mathbb{R}^{3\times 3}$ is the symmetric positive-definite quadrotor inertia tensor in the body frame.
4.3.2 Manifold Variations and the Lagrange–d'Alembert Principle
To take variations directly on $SE(3) \times S^2$ without local coordinates, we define infinitesimal variations parameterized by $(\delta x_L, \eta, \xi)$ where $\eta \in \mathbb{R}^3$ and $\xi \in \mathbb{R}^3$ with $\xi \cdot q = 0$: $$\delta R = R \hat{\eta}, \qquad \delta \Omega = \hat{\Omega}\eta + \dot{\eta}, \tag{4.13}$$ $$\delta q = \xi \times q, \qquad \delta \dot{q} = \dot{\xi} \times q + \xi \times \dot{q}, \qquad \delta x_Q = \delta x_L - l \delta q. \tag{4.14}$$ All variations vanish at the endpoints $t = 0$ and $t = \tau$.


Def. 4.2 (Lagrange–d'Alembert Principle). In the presence of non-conservative actuator forces $(f, M)$, the physical trajectory on $[0, \tau]$ satisfies $$\delta \int_0^\tau L , dt + \int_0^\tau \left( \langle W_1, \hat{M} \rangle_{\mathfrak{so}(3)} + W_2 \cdot f R e_3 \right) dt = 0, \tag{4.15}$$ where $W_1 = R^T \delta R = \hat{\eta}$ and $W_2 = \delta x_Q = \delta x_L - l \delta q$ are the virtual displacement vector fields.
4.3.3 Equations of Motion in Taut Mode ($\Sigma_n$)
Evaluating Eq. (4.15) using Eqs. (4.13)–(4.14), integrating by parts, and applying the fundamental lemma of the calculus of variations yields the coordinate-free equations of motion on $SE(3) \times S^2$ (Sreenath, Lee, & Kumar, CDC 2013):


$$\dot{x}_L = v_L, \tag{4.16}$$ $$(m_Q + m_L)(\dot{v}_L + g e_3) = \big(q \cdot f R e_3 - m_Q l (\dot{q} \cdot \dot{q})\big) q, \tag{4.17}$$ $$\dot{q} = \omega \times q, \qquad (\omega \cdot q = 0), \tag{4.18}$$ $$m_Q l \dot{\omega} = -q \times f R e_3, \tag{4.19}$$ $$\dot{R} = R \Omega^\wedge, \tag{4.20}$$ $$J_Q \dot{\Omega} + \Omega \times J_Q \Omega = M. \tag{4.21}$$


Remark 4.2 (Decoupling and Second-Order Form on $S^2$). Notice from Eq. (4.19) that gravity does not directly enter the cable attitude dynamics $\dot{\omega}$, and the quadrotor attitude dynamics Eq. (4.21) are decoupled from the load. Differentiating Eq. (4.18) and substituting Eq. (4.19) gives the equivalent second-order form on $S^2$: $$m_Q l \ddot{q} + m_Q l (\dot{q} \cdot \dot{q}) q = q \times (q \times f R e_3) = -(I - q q^T) f R e_3, \tag{4.22}$$ which matches the Riemannian covariant acceleration $\nabla_{\dot{q}}\dot{q}$ in Eq. (4.7) driven by the tangential projection of the thrust vector.
4.3.4 Hybrid Dynamics: Taut ($\Sigma_n$) vs. Slack ($\Sigma_z$) Modes
When the cable tension $T := |m_L (\dot{v}_L + g e_3)|$ drops to zero, the cable goes slack and the system transitions to $Q_z = SE(3) \times \mathbb{R}^3$, where the quadrotor and load evolve as decoupled rigid-body and free-fall point-mass systems (Sreenath, Lee, & Kumar, CDC 2013): $$\dot{x}_L = v_L, \qquad m_L(\dot{v}_L + g e_3) = 0, \tag{4.23}$$ $$\dot{x}_Q = v_Q, \qquad m_Q(\dot{v}_Q + g e_3) = f R e_3, \tag{4.24}$$ $$\dot{R} = R \Omega^\wedge, \qquad J_Q \dot{\Omega} + \Omega \times J_Q \Omega = M. \tag{4.25}$$


       +---------------------------------------------------+
       |  Taut Mode Sigma_n on SE(3) x S^2                 |
       |  State: X_n = (x_L, v_L, q, omega, R, Omega)      |
       +---------------------------------------------------+
          |                                             ^
Guard S_z: T == 0                           Guard S_n: ||x_Q - x_L|| == l
Delta_{n->z} = id                           and d/dt ||x_Q - x_L|| > 0
          v                                 Delta_{z->n}: inelastic impact
       +---------------------------------------------------+
       |  Slack Mode Sigma_z on SE(3) x R^3                |
       |  State: X_z = (x_L, v_L, x_Q, v_Q, R, Omega)      |
       +---------------------------------------------------+

Fig. 4.1: Hybrid automaton switching between the taut cable mode $\Sigma_n$ on $SE(3) \times S^2$ and the slack free-fall mode $\Sigma_z$ on $SE(3) \times \mathbb{R}^3$.


________________


4.4 Multi-Body Formulations
4.4.1 Flexible-Cable Chains on $SO(3) \times \mathbb{R}^3 \times (S^2)^n$
When cable mass and deformation cannot be neglected, Kotaru, Wu, & Sreenath (arXiv:1711.04895) model the flexible cable as a finite-element chain of $n$ massless rigid links of lengths $l_i$ connected by spherical joints $q_i \in S^2$, with lumped masses $m_i$ at the end of each link ($m_n = m_L$). The configuration space is $$Q_{\text{flex}} = SO(3) \times \mathbb{R}^3 \times (S^2)^n, \tag{4.26}$$ having $6 + 2n$ degrees of freedom and $2n + 2$ degrees of underactuation. The link positions satisfy the kinematic chain $$x_i = x_{i-1} + l_i q_i, \qquad i \in {1, \dots, n}, \tag{4.27}$$ with $x_0 = x_Q$ and $x_n = x_L$. Applying the Newton–Euler method with internal cable tensions $T_i \in \mathbb{R}$ yields $$m_Q(\ddot{x}0 + g e_3) = f R e_3 + T_1 q_1, \tag{4.28}$$ $$m_j(\ddot{x}j + g e_3) = -T_j q_j + T{j+1} q{j+1}, \qquad j \in {1, \dots, n-1}, \tag{4.29}$$ $$m_n(\ddot{x}n + g e_3) = -T_n q_n, \tag{4.30}$$ $$J_Q \dot{\Omega} + \Omega \times J_Q \Omega = M. \tag{4.31}$$ Eliminating the internal tensions $T_1, \dots, T_n$ by projecting onto the tangent spaces $T{q_i} S^2$ produces a coupled block-matrix system in $(\ddot{x}_0, \ddot{q}_1, \dots, \ddot{q}n)$ directly on $Q{\text{flex}}$, which remains differentially flat with flat outputs $Y = (x_n, \psi)$ requiring $(2n + 4)$ derivatives of $x_n$.
4.4.2 Modular Boltzmann–Hamel Dynamics on $SE(3) \times \mathbb{R}^{N_T}$
For floating-base multi-legged robots with $N$ limbs and $N_T = \sum_{i=1}^N n_i$ internal joints, the configuration space is $SE(3) \times \mathbb{R}^{N_T}$ with base pose $g_b \in SE(3)$, body twist $\xi_b = [v_b^T, \Omega_b^T]^T \in \mathbb{R}^6$, and joint angles $\theta \in \mathbb{R}^{N_T}$. Using the screw-theoretic framework of Fast & Modular Whole-Body Lagrangian Dynamics (arXiv:2504.16383), we define the $SE(3)$ Adjoint and Lie-bracket operators: $$\mathbf{Ad}{g{a,b}} := \begin{bmatrix} R_{a,b} & P_{a,b}^\wedge R_{a,b} \ \mathbf{0}{3\times 3} & R{a,b} \end{bmatrix}, \qquad \mathbf{ad}\xi := \begin{bmatrix} \omega^\wedge & v^\wedge \ \mathbf{0}{3\times 3} & \omega^\wedge \end{bmatrix}. \tag{4.32}$$


Prop. 4.2 (Modular Boltzmann–Hamel Equations). The whole-body dynamics on $SE(3) \times \mathbb{R}^{N_T}$ take the singularity-free quasi-coordinate form $$M(\theta) \begin{bmatrix} \dot{\xi}b \ \ddot{\theta} \end{bmatrix} + C(\theta, \xi_b, \dot{\theta}) \begin{bmatrix} \xi_b \ \dot{\theta} \end{bmatrix} + G(g_b, \theta) = \begin{bmatrix} \mathbf{0}{6\times 1} \ \tau \end{bmatrix} + \mathcal{J}^T F_{\text{ext}}, \tag{4.33}$$ where the mass matrix decomposes additively across individual link modules $(i, j)$: $$M(\theta) = \begin{bmatrix} I_b & \mathbf{0}{6\times N_T} \ \mathbf{0}{N_T\times 6} & \mathbf{0}{N_T\times N_T} \end{bmatrix} + \sum{i=1}^N \sum_{j=1}^{n_i} M_{ij}(\theta), \tag{4.34}$$ and the contact Jacobian $\mathcal{J} \in \mathbb{R}^{6N \times (6+N_T)}$ has block rows $[\mathbf{Ad}{g{b,it}}^{-1},, \mathbf{Ad}{g{b,it}}^{-1} J_{it}]$.


Because every link $(i,j)$ contributes additively to $M(\theta)$ and $C(\theta, \xi_b, \dot{\theta})$, morphological changes—such as a locked joint or lost limb—are handled by matrix addition, subtraction, and reshaping without re-deriving symbolic derivatives.


________________


4.5 Worked Examples
Example 4.1: Unforced Conservation Laws on $SE(3) \times S^2$
Consider the unforced ($f = 0, M = 0$) quadrotor-load system in taut mode Eqs. (4.16)–(4.21).


1. Unit-norm invariance ($|q(t)| = 1$). From Eq. (4.18), $\frac{d}{dt}(q \cdot q) = 2 q \cdot (\omega \times q) = 0$. Furthermore, from Eq. (4.19) with $f = 0$, $\dot{\omega} = 0$, so $\frac{d}{dt}(\omega \cdot q) = \omega \cdot (\omega \times q) = 0$.
2. Total mechanical energy conservation. Differentiating $E = T + U$ from Eqs. (4.11)–(4.12) along trajectories of Eqs. (4.16)–(4.21) with $f = 0, M = 0$ yields $\dot{E} = 0$. Specifically, $v_Q = v_L - l(\omega \times q)$ satisfies $\dot{v}_L + g e_3 = -\frac{m_Q l}{m_Q + m_L}(\omega \cdot \omega) q$ and $\dot{v}_Q + g e_3 = \frac{m_L l}{m_Q + m_L}(\omega \cdot \omega) q$, so all internal tension work cancels identically.
Example 4.2: Hover Equilibrium and Cable Tension
At hover ($\dot{x}L = 0, \dot{v}L = 0, \omega = 0, \Omega = 0$), Eq. (4.17) gives $$(m_Q + m_L) g e_3 = (q \cdot f R e_3) q. \tag{4.35}$$ With $q{\text{eq}} = -e_3$ (load hanging directly below the quadrotor) and $R{\text{eq}} = I$, we have $q_{\text{eq}} \cdot f R_{\text{eq}} e_3 = -f$, yielding $f_{\text{eq}} = (m_Q + m_L) g$ and cable tension $T_{\text{eq}} = |m_L g e_3| = m_L g > 0$.


________________


4.6 Exercises
1. Covariant Acceleration on $S^2$. Starting from the constrained Lagrangian $L(q, \dot{q}) = \frac{1}{2} m l^2 (\dot{q} \cdot \dot{q}) - m g l (e_3 \cdot q)$ with $q \in S^2$, use the variation $\delta q = \xi \times q$ ($\xi \cdot q = 0$) to derive $\nabla_{\dot{q}}\dot{q} = \ddot{q} + (\dot{q}\cdot\dot{q})q = \frac{g}{l}(q \times (q \times e_3))$.
2. Degrees of Underactuation. Verify that a $p$-quadrotor system carrying a rigid-body payload via $p$ flexible cables of $n_i$ links each has $6 + 3p + 2\sum_{i=1}^p n_i$ DOF and $6 - p + 2\sum_{i=1}^p n_i$ degrees of underactuation (Kotaru, Wu, & Sreenath, arXiv:1711.04895).
3. Lie-Group Numerical Integration. Implement a geometric RK4 integrator for Eqs. (4.16)–(4.21) that updates $R \in SO(3)$ via $\exp(\Delta t,\Omega^\wedge)$ and $q \in S^2$ via Rodrigues' rotation $\exp(\Delta t,\omega^\wedge)q$. Verify that $|q(t)| - 1$ remains at machine precision ($\sim 10^{-15}$) and compare energy drift $\Delta E / E_0$ against naive Euler integration.
4. Modular Mass Matrix. Using Eq. (4.34) from arXiv:2504.16383, show how removing leg $k$ from a hexapod reduces to subtracting $\sum_{j=1}^{n_k} M_{kj}(\theta)$ and slicing the corresponding joint rows/columns without recomputing $\partial M / \partial \theta_i$ for the remaining legs.