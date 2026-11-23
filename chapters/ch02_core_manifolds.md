Chapter 2: Core Manifolds: $S^1/\mathrm{SO}(2)$, $S^2$, $\mathrm{SO}(3)$, $S^3$, $\mathrm{SE}(3)$ & Product Spaces
"Because globally valid coordinate charts do not exist for non-Euclidean configuration spaces, coordinate-free geometric formulations on Lie groups and homogeneous manifolds eliminate parameterization singularities and enable almost-global stability."
— Adapted from Wu & Sreenath (2015) and Kotaru, Edmonson, & Sreenath (2020)


________________


2.1 Motivation: Why Coordinate-Free Manifolds?
Robotic systems rarely evolve in flat Euclidean space $\mathbb{R}^n$. A planar mobile robot, a spherical pendulum, an agile quadrotor UAV, a cable-suspended aerial payload, and a floating-base humanoid each evolve on curved smooth manifolds or products of manifolds. Traditional parameterizations—such as Euler angles (roll–pitch–yaw) on $\mathrm{SO}(3)$ or spherical coordinates $(\theta,\phi)$ on $S^2$—suffer from coordinate singularities (e.g., gimbal lock at $\pm 90^\circ$ pitch or at the poles of $S^2$), rendering local linearizations and controllers invalid outside a single coordinate chart (Sreenath, Lee, & Kumar, 2013; Wu & Sreenath, 2015). Conversely, global multi-covering parameterizations such as unit quaternions $S^3$ introduce a two-to-one ambiguity ($q$ and $-q$ represent the same physical orientation), which can trigger the unwinding phenomenon under continuous feedback (Kotaru, Edmonson, & Sreenath, 2020).


This chapter establishes the mathematical foundations of the core manifolds used throughout A Lie-Interpretation of Robotics and Control:


1. Circle and Planar Rotation Group ($S^1 / \mathrm{SO}(2)$) — 1D periodic phase and planar orientation.
2. Two-Sphere ($S^2$) — 2D directional attitude for spherical pendulums and taut cable links.
3. Special Orthogonal Group ($\mathrm{SO}(3)$) — 3D rigid-body orientation, Lie algebra $\mathfrak{so}(3)$, matrix identities, and Rodrigues' exponential/logarithm maps.
4. Three-Sphere / Unit Quaternions ($S^3 \cong \mathrm{SU}(2)$) — Double cover of $\mathrm{SO}(3)$ and topological unwinding.
5. Special Euclidean Group ($\mathrm{SE}(3)$) — 3D rigid-body pose, twists in $\mathfrak{se}(3)$, $\exp/\log$, and the Adjoint map $\mathrm{Ad}_g$.
6. Product Manifolds in Robotics — $\mathrm{SE}(3) \times S^2$, $\mathrm{SO}(3) \times \mathbb{R}^3 \times (S^2)^n$, $\mathrm{SE}(3) \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}$, and $\mathrm{SE}(2) \times S^n$, with degrees of freedom (DoF) and degrees of underactuation (DoUA) analysis.


________________


2.2 The Hat and Vee Operators on $\mathbb{R}^3$ and $\mathfrak{so}(3)$
Before studying individual manifolds, we fix the standard cross-product isomorphism used across all three-dimensional rotational and directional kinematics.


Def. 2.1 (Hat and Vee Maps on $\mathbb{R}^3$). Let $\mathfrak{so}(3) := {S \in \mathbb{R}^{3\times 3} \mid S^\top = -S}$ denote the vector space of $3 \times 3$ real skew-symmetric matrices. The hat map $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ is the linear isomorphism defined for any $x = [x_1, x_2, x_3]^\top \in \mathbb{R}^3$ and $y \in \mathbb{R}^3$ by $x^\wedge y = x \times y$, explicitly:


$$ x^\wedge = \begin{bmatrix} 0 & -x_3 & x_2 \ x_3 & 0 & -x_1 \ -x_2 & x_1 & 0 \end{bmatrix} \in \mathfrak{so}(3), \qquad (S)^\vee = \begin{bmatrix} S_{32} \ S_{13} \ S_{21} \end{bmatrix} \in \mathbb{R}^3. \tag{Eq. 2.1} $$


The inverse vee map $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$ satisfies $(x^\wedge)^\vee = x$ for all $x \in \mathbb{R}^3$ and $(S^\vee)^\wedge = S$ for all $S \in \mathfrak{so}(3)$ (Sreenath, Lee, & Kumar, 2013; Wu & Sreenath, 2016; Seo et al., 2026).


Remark 2.1 (Notation Convention). In some references (e.g., Kotaru, Edmonson, & Sreenath, 2020), the hat map is written $(\cdot)^\times$. Throughout this textbook we uniformly use $(\cdot)^\wedge$ and $(\cdot)^\vee$, and equip $\mathfrak{so}(3)$ with the scaled Frobenius inner product $\langle \hat{x}, \hat{y} \rangle_{\mathfrak{so}(3)} := \frac{1}{2}\operatorname{tr}(\hat{x}^\top \hat{y}) = x^\top y$.


________________


2.3 Core Manifolds
2.3.1 The Circle $S^1$ and Planar Rotation Group $\mathrm{SO}(2)$
Def. 2.2 ($S^1$ and $\mathrm{SO}(2)$). The unit circle $S^1$ and the 2D special orthogonal group $\mathrm{SO}(2)$ are 1-dimensional smooth compact manifolds ($\dim = 1$):


$$ S^1 := {q \in \mathbb{R}^2 \mid q^\top q = 1}, \qquad \mathrm{SO}(2) := {R \in \mathbb{R}^{2\times 2} \mid R^\top R = I_2,\ \det(R) = +1}. \tag{Eq. 2.2} $$


Every $R \in \mathrm{SO}(2)$ has the form $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \ \sin\theta & \cos\theta \end{bmatrix}$ for $\theta \in [0, 2\pi)$, which is diffeomorphic (and Lie-group isomorphic) to $S^1$ via $R(\theta) \leftrightarrow [\cos\theta, \sin\theta]^\top$.


* Lie Algebra $\mathfrak{so}(2)$: Skew-symmetric $2\times 2$ matrices $\mathfrak{so}(2) = \left{ \Omega^\wedge_{2} := \begin{bmatrix} 0 & -\Omega \ \Omega & 0 \end{bmatrix} ,\middle|, \Omega \in \mathbb{R} \right}$, with $\dim\mathfrak{so}(2) = 1$.
* Tangent Space and Kinematics: $T_R \mathrm{SO}(2) = {R \Omega^\wedge_2 \mid \Omega \in \mathbb{R}}$, so $\dot{R} = R \Omega^\wedge_2$, and on $S^1$, $T_q S^1 = {v \in \mathbb{R}^2 \mid q^\top v = 0}$ with $\dot{q} = \Omega J_2 q$ where $J_2 = \begin{bmatrix} 0 & -1 \ 1 & 0 \end{bmatrix}$.
* Exponential and Logarithm Maps: $\exp(\Omega^\wedge_2) = R(\Omega)$ and $\log(R) = \operatorname{atan2}(R_{21}, R_{11}) J_2$.
* Robotics Context: Planar heading, single-axis revolute joints, and periodic gait phase variables (e.g., $\phi_t$ in legged locomotion; Seo et al., 2026).


________________


2.3.2 The Two-Sphere $S^2$
Def. 2.3 (Two-Sphere $S^2$). The unit two-sphere is the 2-dimensional compact embedded submanifold of $\mathbb{R}^3$ ($\dim S^2 = 2$) defined by:


$$ S^2 := {q \in \mathbb{R}^3 \mid q \cdot q = 1}. \tag{Eq. 2.3} $$


Unlike $\mathrm{SO}(2)$ and $\mathrm{SO}(3)$, $S^2$ is not a Lie group (by the hairy ball theorem, it admits no globally non-vanishing smooth tangent vector field), so it is treated as a homogeneous Riemannian manifold acted on by $\mathrm{SO}(3)$.


* Tangent Space and Kinematics: Differentiating $q \cdot q = 1$ gives $T_q S^2 = {\dot{q} \in \mathbb{R}^3 \mid q \cdot \dot{q} = 0}$. Introducing the angular velocity $\omega \in \mathbb{R}^3$ constrained to be orthogonal to $q$, the kinematics on $S^2$ take the form:


$$ \dot{q} = \omega \times q = \omega^\wedge q, \qquad \omega \cdot q = 0, \qquad \Longleftrightarrow \qquad \omega = q \times \dot{q}. \tag{Eq. 2.4a} $$


(Note: In the numbered sequence of this chapter, we reserve Eq. (2.4)–(2.6) for the core $\mathfrak{so}(3)/\mathrm{SO}(3)$ matrix identities below, and number the $S^2$ configuration-error and transport equations immediately after.)


Wait—let's keep the equation numbering strictly sequential without 2.4a so that Eq. (2.4)–(2.6) are unambiguously the matrix identities! Let's write the $S^2$ kinematics in Eq. (2.3) alongside the definition, and then Eq. (2.4)–(2.6) are the three matrix identities on $\mathrm{SO}(3)$! Let's double check that—yes, that's super clean and unambiguous.


Let's check the $S^2$ configuration error, position error, and transport map from Sreenath, Lee, & Kumar (2013) and Wu & Sreenath (2015, 2016):


* Configuration Error on $S^2$: $\Psi_q(q, q_d) = 1 - q \cdot q_d \in [0, 2]$.
* Position Error Vector: $e_q = (q_d^\wedge)^2 q = q_d^\wedge q_d^\wedge q = -(q \times (q \times q_d))$ or in Wu & Sreenath (2015), $e_q = q_d^\wedge q = q_d \times q$ (note: $q_d^\wedge q \in T_{q_d} S^2$ and $(q^\wedge)^2 q_d \in T_q S^2$; specifically in Wu & Sreenath (2016) Sec. V, $e_q = d_1 \Psi = (q^\wedge)^2 q_d$ and $T(q, q_d)\dot{q}_d = (q_d \times \dot{q}d) \times q$, giving $e{\dot{q}} = \dot{q} - (q_d \times \dot{q}d) \times q$ and $e\omega = \omega - (- (q^\wedge)^2)\omega_d$).
* Infinitesimal Variation on $S^2$: $\delta q = \xi \times q_d$ with $\xi \in \mathbb{R}^3$, $\xi \cdot q_d = 0$, and $(\xi \times q_d) \cdot \omega_d + q_d \cdot \delta\omega = 0$ (Wu & Sreenath, 2015).


________________


2.3.3 The Special Orthogonal Group $\mathrm{SO}(3)$
Def. 2.4 ($\mathrm{SO}(3)$ and Lie Algebra $\mathfrak{so}(3)$). The 3D special orthogonal group is the 3-dimensional compact connected Lie group ($\dim \mathrm{SO}(3) = 3$):


$$ \mathrm{SO}(3) := {R \in \mathbb{R}^{3\times 3} \mid R^\top R = I_3,\ \det(R) = +1}. $$


Its Lie algebra is $T_{I_3}\mathrm{SO}(3) = \mathfrak{so}(3)$ with Lie bracket $[A, B] = AB - BA$, satisfying $[x^\wedge, y^\wedge] = (x \times y)^\wedge$. The tangent space at any $R \in \mathrm{SO}(3)$ is $T_R \mathrm{SO}(3) = {R \Omega^\wedge \mid \Omega \in \mathbb{R}^3}$, yielding the body-frame rotational kinematics $\dot{R} = R \Omega^\wedge$, where $\Omega = (R^\top \dot{R})^\vee \in \mathbb{R}^3$ is the body-fixed angular velocity.
Core Matrix Identities on $\mathfrak{so}(3)$ and $\mathrm{SO}(3)$
Prop. 2.1 (Fundamental Matrix Identities). For any vectors $x, y \in \mathbb{R}^3$, any matrix $A \in \mathbb{R}^{3\times 3}$, and any rotation $R \in \mathrm{SO}(3)$, the following identities hold (Kotaru, Edmonson, & Sreenath, 2020, Appendix A):


1. Trace–Vee Identity: $$ \operatorname{tr}(x^\wedge A) = \operatorname{tr}(A x^\wedge) = \frac{1}{2}\operatorname{tr}!\bigl(x^\wedge (A - A^\top)\bigr) = -x^\top (A - A^\top)^\vee. \tag{Eq. 2.4} $$
2. Adjoint Conjugation (Equivariance of the Hat Map): $$ R x^\wedge R^\top = (R x)^\wedge, \qquad R^\top x^\wedge R = (R^\top x)^\wedge. \tag{Eq. 2.5} $$
3. Skew-Symmetric Symmetrization Identity: $$ x^\wedge A + A^\top x^\wedge = \bigl((\operatorname{tr}(A) I_3 - A) x\bigr)^\wedge. \tag{Eq. 2.6} $$


Proof Sketch. For Eq. (2.4), decompose $A = \frac{1}{2}(A + A^\top) + \frac{1}{2}(A - A^\top)$; the symmetric part is orthogonal to the skew-symmetric matrix $x^\wedge$ under the Frobenius inner product, and $\operatorname{tr}(x^\wedge S) = -2 x^\top S^\vee$ for $S \in \mathfrak{so}(3)$. For Eq. (2.5), evaluate $(R x^\wedge R^\top) y = R(x \times (R^\top y)) = (Rx) \times y = (Rx)^\wedge y$ for all $y \in \mathbb{R}^3$ using $\det(R) = +1$. For Eq. (2.6), verify the $(i,j)$ entries directly or apply both sides to an arbitrary vector $v \in \mathbb{R}^3$ using the BAC-CAB vector triple product. Additionally, $(x^\wedge)^2 = x x^\top - |x|^2 I_3$ and $(x^\wedge)^3 = -|x|^2 x^\wedge$. $\square$
Rodrigues' Exponential and Logarithm Maps on $\mathrm{SO}(3)$
Prop. 2.2 (Rodrigues' Formula on $\mathrm{SO}(3)$). Let $\phi = \theta u \in \mathbb{R}^3$ be an axis-angle vector with angle $\theta = |\phi|_2 \ge 0$ and unit axis $u = \phi/\theta \in S^2$ (for $\theta > 0$). The exponential map $\exp : \mathfrak{so}(3) \to \mathrm{SO}(3)$ is given in closed form by Rodrigues' formula:


$$ \exp(\phi^\wedge) = I_3 + \frac{\sin\theta}{\theta},\phi^\wedge + \frac{1 - \cos\theta}{\theta^2},(\phi^\wedge)^2 = I_3 + \sin\theta,u^\wedge + (1 - \cos\theta),(u^\wedge)^2. \tag{Eq. 2.7} $$


For $|\phi| \ll 1$, numerical evaluation uses the Taylor expansions $\frac{\sin\theta}{\theta} = 1 - \frac{\theta^2}{6} + \mathcal{O}(\theta^4)$ and $\frac{1 - \cos\theta}{\theta^2} = \frac{1}{2} - \frac{\theta^2}{24} + \mathcal{O}(\theta^4)$ to avoid $0/0$ cancellation.


Prop. 2.3 (Logarithm Map on $\mathrm{SO}(3)$). Given $R \in \mathrm{SO}(3)$, the rotation angle $\theta \in [0, \pi]$ satisfies $\cos\theta = \frac{\operatorname{tr}(R) - 1}{2}$. Away from the cut locus $\theta = \pi$, the inverse map $\log : \mathrm{SO}(3) \to \mathfrak{so}(3)$ is:


$$ \log(R) = \frac{\theta}{2\sin\theta},(R - R^\top), \qquad \theta = \arccos!\left(\operatorname{clip}!\left(\frac{\operatorname{tr}(R)-1}{2}, -1, 1\right)\right). \tag{Eq. 2.8} $$


At $\theta \to 0$, $\frac{\theta}{2\sin\theta} \to \frac{1}{2} + \frac{\theta^2}{12} + \mathcal{O}(\theta^4)$. At $\theta = \pi$, $\sin\theta = 0$ and $R = R^\top = -I_3 + 2 u u^\top$; the unit axis $u \in S^2$ is extracted from the column of $R + I_3$ with largest norm (or the eigenvector of $R$ with eigenvalue $+1$), up to sign $\pm u$.
Configuration Error and Transport Map on $\mathrm{SO}(3)$
Following Sreenath, Lee, & Kumar (2013) and Kotaru, Edmonson, & Sreenath (2020), given actual $(R, \Omega)$ and desired $(R_d, \Omega_d)$:


$$ \Psi_R(R, R_d) = \frac{1}{2}\operatorname{tr}(I_3 - R_d^\top R) \in [0, 2], \quad e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee, \quad e_\Omega = \Omega - R^\top R_d \Omega_d, \tag{Eq. 2.9} $$


where $\Omega_d \mapsto R^\top R_d \Omega_d$ is the transport map from $T_{R_d}\mathrm{SO}(3)$ to $T_R\mathrm{SO}(3)$ (Wu & Sreenath, 2015). Using Eq. (2.4) and Eq. (2.6), one obtains $\dot{\Psi}R = e\Omega \cdot e_R$ and $\dot{e}R = C(R_d^\top R) e\Omega$ with $C(R_d^\top R) = \frac{1}{2}(\operatorname{tr}(R_d^\top R)I_3 - R_d^\top R)$.


________________


2.3.4 The Three-Sphere $S^3$ (Unit Quaternions) and Double Cover
Def. 2.5 (Unit Quaternions $S^3$). The three-sphere $S^3 := {q \in \mathbb{R}^4 \mid |q|_2 = 1}$ ($\dim S^3 = 3$), equipped with Hamilton quaternion multiplication, forms a compact, simply connected Lie group isomorphic to $\mathrm{SU}(2)$. Writing $q = [q_w, q_v^\top]^\top$ with scalar $q_w \in \mathbb{R}$ and vector $q_v \in \mathbb{R}^3$:


$$ \pi(q) = I_3 + 2 q_w q_v^\wedge + 2 (q_v^\wedge)^2 \in \mathrm{SO}(3), \qquad \exp_{S^3}(\phi) = \begin{bmatrix} \cos(|\phi|/2) \ \frac{\sin(|\phi|/2)}{|\phi|}\phi \end{bmatrix}. \tag{Eq. 2.10} $$


Remark 2.2 (Double Cover and the Unwinding Phenomenon). The homomorphism $\pi : S^3 \to \mathrm{SO}(3)$ is a 2-to-1 covering map with $\ker(\pi) = {\pm 1}$: antipodal quaternions $q$ and $-q$ map to the exact same rotation matrix $\pi(q) = \pi(-q)$. Because $S^3$ is simply connected whereas $\mathrm{SO}(3) \cong \mathbb{R}\mathrm{P}^3$ is not, any continuous feedback law on $S^3$ that stabilizes only one equilibrium (say $+q_d$) forces trajectories starting near $-q_d$ to rotate through a full $2\pi$ excursion—the unwinding phenomenon (Wu & Sreenath, 2015; Kotaru, Edmonson, & Sreenath, 2020). Formulating control directly on $\mathrm{SO}(3)$ avoids unwinding altogether.


________________


2.3.5 The Special Euclidean Group $\mathrm{SE}(3)$
Def. 2.6 ($\mathrm{SE}(3)$ and Lie Algebra $\mathfrak{se}(3)$). The 3D special Euclidean group $\mathrm{SE}(3) := \mathrm{SO}(3) \ltimes \mathbb{R}^3$ ($\dim \mathrm{SE}(3) = 6$) represents rigid-body poses $g = (R, p)$ via $4 \times 4$ homogeneous matrices:


$$ g = \begin{bmatrix} R & p \ 0_{1\times 3} & 1 \end{bmatrix} \in \mathrm{SE}(3), \qquad g^{-1} = \begin{bmatrix} R^\top & -R^\top p \ 0_{1\times 3} & 1 \end{bmatrix}. \tag{Eq. 2.11} $$


Its Lie algebra $\mathfrak{se}(3)$ is isomorphic to $\mathbb{R}^6$ via the twist $\xi = \begin{bmatrix} v \ \omega \end{bmatrix} \in \mathbb{R}^6$ (with linear part $v \in \mathbb{R}^3$ and angular part $\omega \in \mathbb{R}^3$, following Seo et al., 2026, Eq. (2)):


$$ \xi^\wedge = \begin{bmatrix} \omega^\wedge & v \ 0_{1\times 3} & 0 \end{bmatrix} \in \mathfrak{se}(3), \qquad (\xi^\wedge)^\vee = \begin{bmatrix} v \ \omega \end{bmatrix} \in \mathbb{R}^6. \tag{Eq. 2.12} $$


* Body Twist Kinematics: For $g(t) \in \mathrm{SE}(3)$, the body-frame twist is $V^b = (g^{-1}\dot{g})^\vee = \begin{bmatrix} R^\top \dot{p} \ \Omega \end{bmatrix}$, so $\dot{g} = g (V^b)^\wedge$.


Prop. 2.4 (Exponential and Logarithm Maps on $\mathrm{SE}(3)$). For $\xi = [v^\top, \omega^\top]^\top \in \mathbb{R}^6$ with $\theta = |\omega|_2$, the exponential map $\exp : \mathfrak{se}(3) \to \mathrm{SE}(3)$ is:


$$ \exp(\xi^\wedge) = \begin{bmatrix} \exp(\omega^\wedge) & V(\omega) v \ 0_{1\times 3} & 1 \end{bmatrix}, \qquad V(\omega) := I_3 + \frac{1 - \cos\theta}{\theta^2},\omega^\wedge + \frac{\theta - \sin\theta}{\theta^3},(\omega^\wedge)^2. \tag{Eq. 2.13} $$


When $\theta = 0$, $V(0) = I_3$. Conversely, given $g = (R, p) \in \mathrm{SE}(3)$, $\omega = (\log R)^\vee$ and $v = V(\omega)^{-1} p$, where the inverse left Jacobian of $\mathrm{SO}(3)$ is:


$$ V(\omega)^{-1} = I_3 - \frac{1}{2}\omega^\wedge + \frac{1}{\theta^2}!\left(1 - \frac{\theta \sin\theta}{2(1 - \cos\theta)}\right)(\omega^\wedge)^2. \tag{Eq. 2.14} $$


Def. 2.7 (Adjoint Map on $\mathrm{SE}(3)$). For $g = (R, p) \in \mathrm{SE}(3)$, the Adjoint representation $\mathrm{Ad}_g : \mathfrak{se}(3) \to \mathfrak{se}(3)$ satisfies $(g \xi^\wedge g^{-1})^\vee = \mathrm{Ad}_g \xi$, with $6 \times 6$ matrix (Seo et al., 2026, Eq. (4)):


$$ \mathrm{Ad}g = \begin{bmatrix} R & p^\wedge R \ 0{3\times 3} & R \end{bmatrix} \in \mathbb{R}^{6\times 6}, \qquad \mathrm{Ad}{g_1 g_2} = \mathrm{Ad}{g_1}\mathrm{Ad}{g_2}, \qquad \mathrm{Ad}{g^{-1}} = \mathrm{Ad}_g^{-1}. \tag{Eq. 2.15} $$


As shown in Seo et al. (2026, Sec. III-A), $\mathrm{Ad}_g$ is essential in floating-base humanoid control for transforming end-effector twists and Jacobians between the world-aligned Cartesian frame ${C_E}$, the manipulation reference frame ${R}$ (pelvis ${P}$, heading ${H}$, or ground-attached heading ${H_0}$), and the end-effector body frame ${E}$, while preserving left-invariance of geometric admittance feedback $f_G(g_l g, g_l g_D) = f_G(g, g_D)$.


________________


Table 2.1: Summary of Core Manifolds in Robotics
Manifold
	Definition
	$\dim$
	Lie Group?
	Lie Algebra / Tangent Space
	Singularity / Cover Notes
	$S^1$
	${q \in \mathbb{R}^2 \mid q^\top q = 1}$
	1
	Yes ($\cong \mathrm{SO}(2)$)
	$T_q S^1 = {v \mid q^\top v = 0}$
	Periodic; $\pi_1(S^1) = \mathbb{Z}$
	$\mathrm{SO}(2)$
	${R \in \mathbb{R}^{2\times 2} \mid R^\top R = I_2, \det R = 1}$
	1
	Yes
	$\mathfrak{so}(2) = {\Omega^\wedge_2}$
	Singularity-free planar rotation
	$S^2$
	${q \in \mathbb{R}^3 \mid q \cdot q = 1}$
	2
	No
	$T_q S^2 = {\omega \times q \mid \omega \cdot q = 0}$
	No global non-vanishing vector field
	$\mathrm{SO}(3)$
	${R \in \mathbb{R}^{3\times 3} \mid R^\top R = I_3, \det R = 1}$
	3
	Yes
	$\mathfrak{so}(3) = {\Omega^\wedge \mid \Omega \in \mathbb{R}^3}$
	Singularity-free; cut locus at $\theta = \pi$
	$S^3$
	${q \in \mathbb{R}^4 \mid |q|_2 = 1}$
	3
	Yes ($\cong \mathrm{SU}(2)$)
	$\mathfrak{su}(2) \cong \mathbb{R}^3$
	2-to-1 double cover of $\mathrm{SO}(3)$; unwinding
	$\mathrm{SE}(3)$
	$\mathrm{SO}(3) \ltimes \mathbb{R}^3$
	6
	Yes
	$\mathfrak{se}(3) \cong \mathbb{R}^6$ (twists)
	Non-compact; semidirect product
	

________________


2.4 Product Manifolds in Robotics
Complex robotic systems couple multiple rigid bodies, directional links, and joints, so their configuration spaces $Q$ are product manifolds $Q = \mathcal{M}_1 \times \mathcal{M}_2 \times \cdots \times \mathcal{M}k$ with dimension $\dim(Q) = \sum{i=1}^k \dim(\mathcal{M}_i)$ (equal to the system's degrees of freedom, DoF). If the system has $m$ independent control actuators, the degrees of underactuation (DoUA) is:


$$ \mathrm{DoUA} = \mathrm{DoF} - m = \sum_{i=1}^k \dim(\mathcal{M}_i) - m. \tag{Eq. 2.16} $$


![Product Manifold Hierarchy](Fig. 2.1: Conceptual diagram of robotic product manifolds — from single-body $\mathrm{SE}(3)$ to $\mathrm{SE}(3) \times S^2$ cable-suspended load, $\mathrm{SO}(3) \times \mathbb{R}^3 \times (S^2)^n$ flexible-cable transport, and $\mathrm{SE}(3) \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}$ multi-UAV cooperative manipulation.)
2.4.1 Quadrotor with a Taut Cable-Suspended Load: $\mathrm{SE}(3) \times S^2$
Consider a quadrotor UAV (mass $m_Q$, inertia $J_Q$, position $x_Q \in \mathbb{R}^3$, attitude $R \in \mathrm{SO}(3)$) carrying a point-mass payload (mass $m_L$, position $x_L \in \mathbb{R}^3$) via a massless taut cable of length $l > 0$ with unit direction vector $q \in S^2$ pointing from the quadrotor to the load (Sreenath, Lee, & Kumar, 2013; Wu & Sreenath, 2015):


$$ Q = \mathrm{SE}(3) \times S^2, \qquad x_Q = x_L - l q, \qquad \dot{x}_Q = \dot{x}_L - l(\omega \times q). \tag{Eq. 2.17} $$


* DoF & DoUA: $\dim(\mathrm{SE}(3) \times S^2) = 6 + 2 = 8$ DoF. The quadrotor has $m = 4$ actuators (scalar thrust $f \in \mathbb{R}$ and body moment $M \in \mathbb{R}^3$), giving $\mathrm{DoUA} = 8 - 4 = 4$.
* Hybrid Mode: When cable tension $T = |m_L(\ddot{x}_L + g e_3)|$ drops to zero, the constraint $|x_Q - x_L| = l$ breaks and the configuration space switches to $Q_z = \mathrm{SE}(3) \times \mathbb{R}^3$ (9 DoF, 5 DoUA) (Sreenath, Lee, & Kumar, 2013).
2.4.2 Quadrotor with a Flexible $n$-Link Cable: $\mathrm{SO}(3) \times \mathbb{R}^3 \times (S^2)^n$
When cable mass is non-negligible or large deformations occur, Kotaru, Wu, & Sreenath (2017) discretize the flexible cable into $n$ rigid links with spherical joints ($q_i \in S^2$, length $l_i$, link-end mass $m_i$):


$$ Q = \mathrm{SO}(3) \times \mathbb{R}^3 \times (S^2)^n, \qquad x_i = x_{i-1} + l_i q_i \quad (i = 1, \dots, n), \tag{Eq. 2.18} $$


where $x_0$ is the quadrotor center of mass and $x_n$ is the payload position.


* DoF & DoUA: $\mathrm{DoF} = 3 + 3 + 2n = 6 + 2n$. With $m = 4$ quadrotor inputs, $\mathrm{DoUA} = (6 + 2n) - 4 = 2n + 2$. For $n = 5$ links, $\mathrm{DoF} = 16$ and $\mathrm{DoUA} = 12$ (Kotaru, Wu, & Sreenath, 2017).
2.4.3 Multi-Quadrotor Rigid-Body Load with Flexible Cables: $\mathrm{SE}(3) \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}$
For $p$ cooperating quadrotors transporting a shared rigid-body payload (pose $(R_L, x_L) \in \mathrm{SE}(3)$, attachment offset $r_j \in \mathbb{R}^3$ in load body frame) via flexible cables of $n_j$ links each (Kotaru, Wu, & Sreenath, 2017, Sec. II-C):


$$ Q = \mathrm{SE}(3) \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}, \qquad x_{j,(n_j - 1)} = x_L + R_L r_j - l_{j, n_j} q_{j, n_j}. \tag{Eq. 2.19} $$


* DoF & DoUA: $\mathrm{DoF} = 6 + 3p + 2\sum_{j=1}^p n_j$. With $m = 4p$ actuators, $\mathrm{DoUA} = 6 - p + 2\sum_{j=1}^p n_j$. For $p = 4$ quadrotors and $n_j = 5$ links each, $\mathrm{DoF} = 58$ and $\mathrm{DoUA} = 42$ (for a point-mass shared load on $\mathbb{R}^3 \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}$, $\mathrm{DoF} = 55$ and $\mathrm{DoUA} = 39$).
2.4.4 Planar Mobile-Base & Articulated Systems: $\mathrm{SE}(2) \times S^n$
A planar $n$-link robotic snake or planar floating-base manipulator evolves on $\mathrm{SE}(2) \times (S^1)^n$ (often written compactly as $\mathrm{SE}(2) \times S^n$ in Wu & Sreenath, 2015), where $(x, y, \psi) \in \mathrm{SE}(2)$ captures planar position and heading (3 DoF) and $n$ revolute joints contribute $n$ DoF ($\mathrm{DoF} = 3 + n$). Similarly, in humanoid loco-manipulation (Seo et al., 2026, Sec. III-C), the ground-attached heading frame ${H_0}$ projects the floating base onto $(x, y, \psi) \in \mathrm{SE}(2)$ to decouple pelvis roll, pitch, and heave from end-effector $\mathrm{SE}(3)$ commands.
Table 2.2: Product Manifolds, DoF, and Degrees of Underactuation
System
	Configuration Manifold $Q$
	DoF
	Actuators ($m$)
	DoUA ($\mathrm{DoF} - m$)
	Primary Reference
	Quadrotor (free)
	$\mathrm{SE}(3)$
	6
	4
	2
	Kotaru et al. (2020)
	Quadrotor + taut cable load
	$\mathrm{SE}(3) \times S^2$
	8
	4
	4
	Sreenath et al. (2013)
	Quadrotor + $n$-link flexible cable
	$\mathrm{SO}(3) \times \mathbb{R}^3 \times (S^2)^n$
	$6 + 2n$
	4
	$2n + 2$
	Kotaru et al. (2017)
	$p$ Quadrotors + rigid load + flex cables
	$\mathrm{SE}(3) \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}$
	$6 + 3p + 2\sum n_j$
	$4p$
	$6 - p + 2\sum n_j$
	Kotaru et al. (2017)
	Planar $n$-link snake / mobile base
	$\mathrm{SE}(2) \times (S^1)^n$
	$3 + n$
	$n$ (typical)
	$3$
	Wu & Sreenath (2015)
	

________________


2.5 Worked Examples
Example 2.1 (Rodrigues' Map and Matrix Identity Verification).
Let $\phi = [0.3, -0.4, 0.5]^\top \in \mathbb{R}^3$ ($\theta = |\phi|_2 = \sqrt{0.5} \approx 0.7071\text{ rad}$). Using Eq. (2.7), $R = \exp(\phi^\wedge) \in \mathrm{SO}(3)$ satisfies $R^\top R = I_3$ and $\det(R) = +1$. Applying Eq. (2.8) recovers $\log(R)^\vee = \phi$ to machine precision ($\sim 10^{-16}$). For any $x \in \mathbb{R}^3$ and $A \in \mathbb{R}^{3\times 3}$, Eq. (2.4)–(2.6) hold identically, which is verified numerically in the companion notebook ch02_lie_groups_product_manifolds.ipynb.


Example 2.2 ($\mathrm{SE}(3)$ Adjoint Twist Transformation in Humanoid Control).
In Seo et al. (2026, Eq. (9)–(11)), let $g_{C_E C_R} = (I_3, -(p_{WC_E} - p_{WC_R})) \in \mathrm{SE}(3)$ relate the world-aligned Cartesian frames of the manipulation reference and end effector. By Eq. (2.15):


$$ \mathrm{Ad}{g{C_E C_R}} = \begin{bmatrix} I_3 & -(p_{WC_E} - p_{WC_R})^\wedge \ 0_{3\times 3} & I_3 \end{bmatrix}, \qquad V_{RE}^{E} = \mathrm{Ad}{g{E C_E}} \bigl(V_{WE}^{C_E} - \mathrm{Ad}{g{C_E C_R}} V_{WR}^{C_R}\bigr). \tag{Eq. 2.20} $$


Example 2.3 (Quadrotor + Suspended Load Kinematics on $\mathrm{SE}(3) \times S^2$).
Let $l = 1.0\text{ m}$, $x_L(t) = [\cos t, \sin t, 0.5 t]^\top$, and $q(t) \in S^2$ evolve via $\dot{q} = \omega \times q$ with $\omega \cdot q = 0$. Differentiating Eq. (2.17) yields $v_Q = v_L - l(\omega \times q)$ and $\ddot{x}_Q = \ddot{x}_L - l(\dot{\omega} \times q + \omega \times (\omega \times q))$. Because $\omega \cdot q = 0$, $\omega \times (\omega \times q) = -|\omega|^2 q$, which enters the tension-coupled translational dynamics in Sreenath, Lee, & Kumar (2013, Eq. (6)).


________________


2.6 Exercises
1. ($\mathrm{SO}(2) / S^1$): Prove that $\mathrm{SO}(2)$ is abelian and diffeomorphic to $S^1$, and show that for $q \in S^1$, $\dot{q} = \Omega J_2 q$ preserves $|q|_2 = 1$.
2. ($S^2$ Tangent Space & Transport Map): Show that $T(q, q_d)\dot{q}_d := (q_d \times \dot{q}d) \times q$ maps $T{q_d}S^2$ into $T_q S^2$ and satisfies the compatibility condition $d_2 \Psi_q = -T^*(q, q_d) d_1 \Psi_q$ for $\Psi_q = 1 - q \cdot q_d$ (Wu & Sreenath, 2016).
3. (Matrix Identities): Prove Eq. (2.6), $x^\wedge A + A^\top x^\wedge = \bigl((\operatorname{tr}(A)I_3 - A)x\bigr)^\wedge$, by evaluating both sides on an arbitrary vector $v \in \mathbb{R}^3$.
4. ($S^3$ Unwinding): Construct an initial quaternion $q(0) \in S^3$ with $\pi(q(0)) = \exp(0.1 e_3^\wedge)$ that is antipodally close to $-q_d$ ($q_d = [1, 0, 0, 0]^\top$), and explain why minimizing $|q - q_d|_2^2$ causes a $354^\circ$ rotation whereas minimizing $\Psi_R(R, I_3)$ rotates by $5.7^\circ$.
5. ($\mathrm{SE}(3)$ Log Map): Derive Eq. (2.14) for $V(\omega)^{-1}$ from Eq. (2.13) using $(\omega^\wedge)^3 = -\theta^2 \omega^\wedge$.
6. (Product Manifolds): Compute the DoF and DoUA for (a) $p = 3$ quadrotors carrying a rigid-body payload via $n_j = 4$ flexible cable links each, and (b) a dual-arm humanoid with a 6-DoF floating base, two 7-DoF arms, and two 6-DoF legs ($26$ actuated joints total) walking on flat ground.