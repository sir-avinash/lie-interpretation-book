Chapter 3: Variations, Configuration Errors & Transport Maps
3.1 Motivation: Why Subtraction Fails on Curved Manifolds
In Euclidean space $\mathbb{R}^n$, comparing a current state $x \in \mathbb{R}^n$ with a desired reference $x_d \in \mathbb{R}^n$ is trivial: the configuration error is $e_x = x - x_d$, its time derivative is $\dot{e}_x = \dot{x} - \dot{x}_d$, and a small perturbation is $x^\epsilon = x_d + \epsilon \delta x$. All three operations—difference, velocity comparison, and linearization—rely on the global vector-space structure of $\mathbb{R}^n$, where tangent spaces at different points are canonically identified.


When a mechanical system evolves on a curved configuration manifold $Q$—such as the unit two-sphere $\mathbb{S}^2$, the special orthogonal group $SO(3)$, or a product manifold such as $SE(3) \times \mathbb{S}^2$ for a quadrotor carrying a cable-suspended load (Sreenath, Lee, & Kumar, 2013)—direct subtraction fails:


1. Closure failure: For $R, R_d \in SO(3)$, $R - R_d \notin SO(3)$; for $q, q_d \in \mathbb{S}^2$, $q - q_d \notin \mathbb{S}^2$.
2. Tangent-space mismatch: The body-frame angular velocity $\Omega \in \mathbb{R}^3$ (representing $R\Omega^\wedge \in T_R SO(3)$) and the desired body-frame angular velocity $\Omega_d \in \mathbb{R}^3$ (representing $R_d \Omega_d^\wedge \in T_{R_d} SO(3)$) live in different tangent spaces. Subtracting $\Omega - \Omega_d$ directly ignores the relative rotation $R^\top R_d$ between the current and desired frames.
3. Coordinate singularities: Local parameterizations such as Euler angles suffer from gimbal lock and fail to provide global, coordinate-free error metrics.


This chapter establishes the coordinate-free geometric toolkit used throughout geometric control and variation-based linearization: infinitesimal variations on manifolds, quadratic configuration error functions $\Psi(q, q_d)$, compatible transport maps $T(q, q_d)$ induced by the exterior differential relation $d_2 \Psi = -T^*(q, q_d) d_1 \Psi$, and the first-order equivalence between tracking errors and infinitesimal variations (Wu & Sreenath, 2015; Wu & Sreenath, 2016).


________________


3.2 Notation and Hat/Vee Isomorphism
Throughout this chapter, we adopt the standard Lie-algebraic isomorphism between $\mathbb{R}^3$ and $\mathfrak{so}(3) = { S \in \mathbb{R}^{3\times 3} \mid S^\top = -S }$.


Def. 3.1 (Hat and Vee Maps). The hat map $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ is the linear isomorphism defined for any $x = [x_1, x_2, x_3]^\top \in \mathbb{R}^3$ and $y \in \mathbb{R}^3$ by


$$ x^\wedge y = x \times y, \qquad x^\wedge = \begin{bmatrix} 0 & -x_3 & x_2 \ x_3 & 0 & -x_1 \ -x_2 & x_1 & 0 \end{bmatrix}. \tag{3.1} $$


Its inverse, the vee map $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$, extracts the axial vector from any skew-symmetric matrix $S \in \mathfrak{so}(3)$ such that $(x^\wedge)^\vee = x$ and $(S^\vee)^\wedge = S$.


Prop. 3.1 (Standard $\mathfrak{so}(3)$ Identities). For all $x, y \in \mathbb{R}^3$ and $R \in SO(3)$:


1. $x^\wedge y = -y^\wedge x = x \times y$,
2. $(x^\wedge)^2 y = x \times (x \times y) = (x \cdot y) x - |x|^2 y$,
3. $(R x)^\wedge = R x^\wedge R^\top$ (Adjoint invariance on $SO(3)$),
4. $\operatorname{Tr}(x^\wedge y^\wedge) = -2, x \cdot y$,
5. $\frac{1}{2}\operatorname{Tr}(x^\wedge S) = -x \cdot S^\vee$ for any $S \in \mathfrak{so}(3)$.


________________


3.3 Infinitesimal Variations on $\mathbb{R}^3$, $\mathbb{S}^2$, $SO(3)$, and Product Manifolds
To linearize nonlinear dynamics along a reference trajectory $q_d(t) \in Q$ without introducing local coordinates, we perturb the trajectory through a smooth one-parameter family of curves $q^\epsilon(t) \in Q$ parameterized by $\epsilon \in (-\epsilon_0, \epsilon_0)$ such that $q^0(t) = q_d(t)$ (Wu & Sreenath, 2015).


Def. 3.2 (Infinitesimal Variation on a Manifold). Given a smooth family of trajectories $q^\epsilon(t) \in Q$ with $q^0(t) = q_d(t)$, the infinitesimal variation along $q_d(t)$ is the tangent vector field


$$ \delta q(t) = \left.\frac{d}{d\epsilon}\right|{\epsilon=0} q^\epsilon(t) \in T{q_d(t)} Q. \tag{3.2} $$
3.3.1 Variations on $\mathbb{R}^3$
On $Q = \mathbb{R}^3$, the tangent space is canonically $T_{x_d} \mathbb{R}^3 \cong \mathbb{R}^3$. A perturbed trajectory $x^\epsilon(t) = x_d(t) + \epsilon \delta x(t) + \mathcal{O}(\epsilon^2)$ yields:


$$ \delta x \in \mathbb{R}^3, \qquad \delta v = \delta \dot{x} = \frac{d}{dt}(\delta x). \tag{3.3} $$
3.3.2 Variations on $\mathbb{S}^2$
The unit sphere $\mathbb{S}^2 = { q \in \mathbb{R}^3 \mid q \cdot q = 1 }$ is an embedded submanifold of $\mathbb{R}^3$ with tangent space $T_{q_d} \mathbb{S}^2 = { v \in \mathbb{R}^3 \mid v \cdot q_d = 0 }$. A smooth curve $q^\epsilon(t) \in \mathbb{S}^2$ can be generated via the exponential map on $SO(3)$ acting on $q_d(t)$:


$$ q^\epsilon(t) = \exp(\epsilon \xi^\wedge(t)), q_d(t), \qquad \xi(t) \cdot q_d(t) = 0. \tag{3.4} $$


Differentiating Eq. (3.4) with respect to $\epsilon$ at $\epsilon = 0$ gives the infinitesimal variation of $q_d$:


$$ \delta q = \left.\frac{d}{d\epsilon}\right|_{\epsilon=0} \exp(\epsilon \xi^\wedge), q_d = \xi^\wedge q_d = \xi \times q_d, \qquad \xi \cdot q_d = 0. \tag{3.5} $$


The orthogonality constraint $\xi \cdot q_d = 0$ removes the redundant component of $\xi$ parallel to $q_d$ (which leaves $q_d$ invariant), establishing a bijection between $\xi \in T_{q_d} \mathbb{S}^2$ and $\delta q \in T_{q_d} \mathbb{S}^2$. Differentiating $q^\epsilon \cdot q^\epsilon = 1$ confirms $\delta q \cdot q_d = 0$.


For the angular velocity $\omega^\epsilon(t) \in \mathbb{R}^3$ satisfying $\dot{q}^\epsilon = \omega^\epsilon \times q^\epsilon$ and $\omega^\epsilon \cdot q^\epsilon = 0$, differentiating $\omega^\epsilon \cdot q^\epsilon = 0$ with respect to $\epsilon$ at $\epsilon = 0$ yields the coupled variation constraint (Wu & Sreenath, 2015):


$$ \delta q \cdot \omega_d + q_d \cdot \delta\omega = 0 \quad \Longleftrightarrow \quad (\xi \times q_d) \cdot \omega_d + q_d \cdot \delta\omega = 0. \tag{3.6} $$


Furthermore, differentiating $\dot{q}^\epsilon = \omega^\epsilon \times q^\epsilon$ with respect to $\epsilon$ at $\epsilon = 0$ and noting $\delta \dot{q} = \frac{d}{dt}(\delta q)$ gives


$$ \delta \dot{q} = \dot{\xi} \times q_d + \xi \times \dot{q}_d = \delta\omega \times q_d + \omega_d \times (\xi \times q_d). \tag{3.7} $$
3.3.3 Variations on $SO(3)$
The special orthogonal group $SO(3) = { R \in \mathbb{R}^{3\times 3} \mid R^\top R = I_3,; \det(R) = 1 }$ has Lie algebra $\mathfrak{so}(3)$. Using right-trivialized (body-frame) perturbations about $R_d(t) \in SO(3)$, we write


$$ R^\epsilon(t) = R_d(t) \exp(\epsilon \eta^\wedge(t)), \qquad \eta(t) \in \mathbb{R}^3. \tag{3.8} $$


Differentiating Eq. (3.8) at $\epsilon = 0$ yields the variation on $SO(3)$:


$$ \delta R = \left.\frac{d}{d\epsilon}\right|_{\epsilon=0} R_d \exp(\epsilon \eta^\wedge) = R_d \eta^\wedge. \tag{3.9} $$


To find the body-frame angular velocity variation $\delta\Omega$, recall the kinematic equation $\dot{R}^\epsilon = R^\epsilon (\Omega^\epsilon)^\wedge$, or equivalently $(\Omega^\epsilon)^\wedge = (R^\epsilon)^\top \dot{R}^\epsilon$. Differentiating $(R^\epsilon)^\top \dot{R}^\epsilon$ at $\epsilon = 0$ and using $\delta(R^\top) = -\eta^\wedge R_d^\top$ and $\delta\dot{R} = \dot{R}_d \eta^\wedge + R_d \dot{\eta}^\wedge$:


$$ \begin{aligned} \delta \Omega^\wedge &= \delta(R^\top) \dot{R}_d + R_d^\top \delta\dot{R} \ &= -\eta^\wedge R_d^\top (R_d \Omega_d^\wedge) + R_d^\top (R_d \Omega_d^\wedge \eta^\wedge + R_d \dot{\eta}^\wedge) \ &= \Omega_d^\wedge \eta^\wedge - \eta^\wedge \Omega_d^\wedge + \dot{\eta}^\wedge \ &= (\Omega_d^\wedge \eta)^\wedge + \dot{\eta}^\wedge, \end{aligned} \tag{3.10} $$


where we used the Lie bracket identity $[a^\wedge, b^\wedge] = a^\wedge b^\wedge - b^\wedge a^\wedge = (a \times b)^\wedge = (a^\wedge b)^\wedge$. Applying $(\cdot)^\vee$ gives the exact variation of the body angular velocity (Sreenath, Lee, & Kumar, 2013; Wu & Sreenath, 2015):


$$ \delta \Omega = \Omega_d^\wedge \eta + \dot{\eta}. \tag{3.11} $$
3.3.4 Variations on Product Manifolds
Consider a composite system evolving on a product manifold $Q = Q_1 \times Q_2 \times \cdots \times Q_k$, such as $SE(3) \cong \mathbb{R}^3 \times SO(3)$ for a rigid body, or $Q = \mathbb{R}^3 \times SO(3) \times \mathbb{S}^2$ for a quadrotor with a taut cable-suspended load (Sreenath, Lee, & Kumar, 2013). Because the tangent space of a product manifold decomposes as $T_{(q_1, \dots, q_k)} Q \cong T_{q_1} Q_1 \oplus \cdots \oplus T_{q_k} Q_k$, the infinitesimal variation is simply the tuple of component variations:


$$ \delta q = (\delta x, \delta R, \delta q_{\text{load}}) = \big(\delta x,; R_d \eta^\wedge,; \xi \times q_d\big), \tag{3.12} $$


with state-space variation vector $s = [\delta x^\top, \delta v^\top, \eta^\top, \delta\Omega^\top, \xi^\top, \delta\omega^\top]^\top \in \mathbb{R}^{18}$ subject to the linear algebraic constraints $\xi \cdot q_d = 0$ and Eq. (3.6) (Wu & Sreenath, 2015).


  Tangent space T_{q_d} M           Transport T(q, q_d)          Tangent space T_q M
  -----------------------        ========================>       --------------------
  d_2 \Psi(q, q_d) \in T*_{q_d} M  <========================  d_1 \Psi(q, q_d) \in T*_q M
                                   -T*(q, q_d) d_1 \Psi = d_2 \Psi

Fig. 3.1: Duality between the velocity transport map $T(q, q_d) : T_{q_d}Q \to T_q Q$ and its cotangent pullback $T^(q, q_d) : T_q^* Q \to T_{q_d}^* Q$ induced by a configuration error function $\Psi(q, q_d)$.*


________________


3.4 Configuration Error Functions, Exterior Differentials, and Transport Maps
To construct geometric tracking controllers and Control Barrier Functions (CBFs) on manifolds (Wu & Sreenath, 2016), we pair a smooth scalar error function with a compatible velocity transport map.


Def. 3.3 (Configuration Error Function). A smooth function $\Psi : Q \times Q \to \mathbb{R}$ is a configuration error function if it is locally positive definite around the diagonal $\Delta = {(q, q) \mid q \in Q}$:


1. $\Psi(q, q_d) \ge 0$ for all $q, q_d \in Q$, with $\Psi(q, q_d) = 0 \iff q = q_d$,
2. $d_1 \Psi(q_d, q_d) = 0$ and the Hessian of $\Psi(\cdot, q_d)$ at $q = q_d$ is positive definite.


Here $d_1 \Psi(q, q_d) \in T_q^* Q$ and $d_2 \Psi(q, q_d) \in T_{q_d}^* Q$ denote the partial exterior differentials of $\Psi$ with respect to its first and second arguments, defined for all $\delta q \in T_q Q$ and $\delta q_d \in T_{q_d} Q$ by


$$ \langle d_1 \Psi(q, q_d),, \delta q \rangle = \left.\frac{d}{d\epsilon}\right|{\epsilon=0} \Psi(q^\epsilon, q_d), \qquad \langle d_2 \Psi(q, q_d),, \delta q_d \rangle = \left.\frac{d}{d\epsilon}\right|{\epsilon=0} \Psi(q, q_d^\epsilon). \tag{3.13} $$


Def. 3.4 (Compatible Transport Map and Velocity Error). A smooth linear map $T(q, q_d) : T_{q_d} Q \to T_q Q$ is a compatible transport map for $\Psi$ if its dual (adjoint) $T^(q, q_d) : T_q^ Q \to T_{q_d}^* Q$ satisfies


$$ d_2 \Psi(q, q_d) = -T^*(q, q_d), d_1 \Psi(q, q_d). \tag{3.14} $$


Given $T(q, q_d)$, the velocity error $e_{\dot{q}} \in T_q Q$ is defined by transporting $\dot{q}d \in T{q_d} Q$ into $T_q Q$ and subtracting:


$$ e_{\dot{q}} = \dot{q} - T(q, q_d) \dot{q}_d. \tag{3.15} $$


Prop. 3.2 (Energy-Dissipation / Lyapunov Derivative Identity). If $T(q, q_d)$ satisfies the compatibility condition Eq. (3.14), then along any smooth trajectories $q(t), q_d(t) \in Q$, the time derivative of $\Psi(q(t), q_d(t))$ depends solely on $d_1 \Psi$ and the velocity error $e_{\dot{q}}$:


$$ \begin{aligned} \dot{\Psi}(q, q_d) &= \langle d_1 \Psi, \dot{q} \rangle + \langle d_2 \Psi, \dot{q}_d \rangle \ &= \langle d_1 \Psi, \dot{q} \rangle - \langle T^*(q, q_d) d_1 \Psi, \dot{q}_d \rangle \ &= \langle d_1 \Psi, \dot{q} - T(q, q_d) \dot{q}d \rangle = \langle d_1 \Psi(q, q_d),, e{\dot{q}} \rangle. \end{aligned} \tag{3.16} $$


Remark 3.1. Eq. (3.16) is the geometric foundation of PD and backstepping control on manifolds: it ensures that $\dot{\Psi}$ is linear in the velocity error $e_{\dot{q}}$ without requiring $q$ and $q_d$ to share a tangent space.


________________


3.4.1 Configuration Error and Transport Map on $SO(3)$
On $SO(3)$, the standard trace-based quadratic configuration error function is


$$ \Psi_R(R, R_d) = \frac{1}{2}\operatorname{Tr}(I_3 - R_d^\top R). \tag{3.17} $$


Let us compute $d_1 \Psi_R$ and $d_2 \Psi_R$ explicitly. Using a body-frame variation $\delta R = R \zeta^\wedge \in T_R SO(3)$ ($\zeta \in \mathbb{R}^3$):


$$ \begin{aligned} \langle d_1 \Psi_R, R\zeta^\wedge \rangle &= -\frac{1}{2}\operatorname{Tr}(R_d^\top R \zeta^\wedge) \ &= -\frac{1}{4}\operatorname{Tr}\big((R_d^\top R - R^\top R_d) \zeta^\wedge\big) \ &= \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee \cdot \zeta, \end{aligned} \tag{3.18} $$


where we used $\operatorname{Tr}(M \zeta^\wedge) = \frac{1}{2}\operatorname{Tr}((M - M^\top)\zeta^\wedge)$ (since the symmetric part of $M$ is orthogonal to $\zeta^\wedge$) and Prop. 3.1(5). Identifying $\mathbb{R}^3 \cong T_R^* SO(3)$ via body-frame trivialization yields the attitude error vector:


$$ e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee. \tag{3.19} $$


Next, for a variation $\delta R_d = R_d \eta^\wedge \in T_{R_d} SO(3)$ ($\eta \in \mathbb{R}^3$), note that $\delta(R_d^\top) = -\eta^\wedge R_d^\top$. Thus:


$$ \begin{aligned} \langle d_2 \Psi_R, R_d\eta^\wedge \rangle &= \frac{1}{2}\operatorname{Tr}(\eta^\wedge R_d^\top R) \ &= \frac{1}{4}\operatorname{Tr}\big(\eta^\wedge (R_d^\top R - R^\top R_d)\big) \ &= -\frac{1}{2}(R_d^\top R - R^\top R_d)^\vee \cdot \eta \ &= -\big(R^\top R_d, e_R\big) \cdot \big(R^\top R_d, \eta\big)? \quad \text{Wait—let's check the frame!} \end{aligned} $$


Careful! In body-frame trivialization, $\dot{R} = R\Omega^\wedge$ and $\dot{R}_d = R_d \Omega_d^\wedge$. In Eq. (3.18), $d_1 \Psi_R$ in the body frame of $R$ is $e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee$. In the body frame of $R_d$, $d_2 \Psi_R$ is $-e_R$. Let's check the transport map:


$$ T(R, R_d)\Omega_d = R^\top R_d \Omega_d, \qquad e_\Omega = \Omega - R^\top R_d \Omega_d. \tag{3.20} $$


Wait: how does $T^*(R, R_d)$ act on $e_R$? Notice that $R_d^\top R$ commutes with $(R_d^\top R - R^\top R_d)$! Specifically, by Rodrigues' formula or direct expansion: $(R_d^\top R) (R_d^\top R - R^\top R_d) (R^\top R_d) = R_d^\top R - R^\top R_d$, which implies by Prop. 3.1(3) that


$$ (R_d^\top R), e_R = e_R \quad \Longleftrightarrow \quad (R^\top R_d), e_R = e_R! \tag{3.21} $$


Therefore, $T^*(R, R_d) e_R = (R^\top R_d)^\top e_R = R_d^\top R, e_R = e_R$, and Eq. (3.14) holds:


$$ d_2 \Psi_R = -e_R = -T^*(R, R_d), d_1 \Psi_R. \tag{3.22} $$


Consequently, by Prop. 3.2, we immediately obtain the fundamental $SO(3)$ time-derivative identities:


$$ \dot{\Psi}R = e_R \cdot e\Omega, \qquad \dot{e}R = C(R_d^\top R), e\Omega, \tag{3.23} $$


where $C(A) = \frac{1}{2}(\operatorname{Tr}(A) I_3 - A^\top)$ for $A = R_d^\top R \in SO(3)$ (verified in Section 3.6).


________________


3.4.2 Configuration Error and Transport Map on $\mathbb{S}^2$
On $\mathbb{S}^2$, the quadratic configuration error function is


$$ \Psi_q(q, q_d) = 1 - q \cdot q_d. \tag{3.24} $$


For $\delta q \in T_q \mathbb{S}^2$ ($\delta q \cdot q = 0$) and $\delta q_d \in T_{q_d} \mathbb{S}^2$ ($\delta q_d \cdot q_d = 0$):


$$ \langle d_1 \Psi_q, \delta q \rangle = -q_d \cdot \delta q = \big(q^\wedge{}^2 q_d\big) \cdot \delta q, \qquad \langle d_2 \Psi_q, \delta q_d \rangle = -q \cdot \delta q_d = \big((q_d^\wedge)^2 q\big) \cdot \delta q_d, \tag{3.25} $$


where we used $(q^\wedge)^2 q_d = (q \cdot q_d) q - q_d$ and $(q_d^\wedge)^2 q = (q_d \cdot q) q_d - q$. Identifying $T_{q_d}^* \mathbb{S}^2 \cong T_{q_d} \mathbb{S}^2$, we define the $\mathbb{S}^2$ configuration error vector:


$$ e_q = (q_d^\wedge)^2 q = q_d \times (q_d \times q). \tag{3.26} $$


The compatible transport map $T(q, q_d) : T_{q_d} \mathbb{S}^2 \to T_q \mathbb{S}^2$ and velocity errors are (Sreenath, Lee, & Kumar, 2013; Wu & Sreenath, 2016):


$$ T(q, q_d)\dot{q}_d = (q_d \times \dot{q}d) \times q, \qquad e{\dot{q}} = \dot{q} - (q_d \times \dot{q}d) \times q, \qquad e\omega = \omega - (-q^\wedge{}^2)\omega_d. \tag{3.27} $$


Prop. 3.3 (Compatibility on $\mathbb{S}^2$). The transport map $T(q, q_d)$ in Eq. (3.27) satisfies $d_2 \Psi_q = -T^*(q, q_d) d_1 \Psi_q$.


Proof. For any $v \in T_q \mathbb{S}^2$ and $w \in T_{q_d} \mathbb{S}^2$:


$$ \langle v, T(q, q_d) w \rangle = v \cdot ((q_d \times w) \times q) = (q \times v) \cdot (q_d \times w) = ((q \times v) \times q_d) \cdot w, $$


so $T^*(q, q_d) v = (q \times v) \times q_d$. Substituting $v = d_1 \Psi_q = q \times (q \times q_d) = (q \cdot q_d)q - q_d$, we have $q \times v = -q \times q_d = q_d \times q$, and hence


$$ -T^*(q, q_d) d_1 \Psi_q = -((q_d \times q) \times q_d) = q_d \times (q_d \times q) = (q_d^\wedge)^2 q = d_2 \Psi_q = e_q. \quad \blacksquare $$


Manifold $Q$
	Config. Error $\Psi(q, q_d)$
	Error Vector
	Transport Map $T(q, q_d) \dot{q}_d$
	Velocity Error
	$\mathbb{R}^3$
	$\frac{1}{2}|x - x_d|^2$
	$e_x = x - x_d$
	$\dot{x}_d$
	$e_v = \dot{x} - \dot{x}_d$
	$\mathbb{S}^2$
	$1 - q \cdot q_d$
	$e_q = (q_d^\wedge)^2 q$
	$(q_d \times \dot{q}_d) \times q$
	$e_{\dot{q}} = \dot{q} - (q_d \times \dot{q}d) \times q$, $e\omega = \omega - (-q^\wedge{}^2)\omega_d$
	$SO(3)$
	$\frac{1}{2}\operatorname{Tr}(I_3 - R_d^\top R)$
	$e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee$
	$R^\top R_d \Omega_d$
	$e_\Omega = \Omega - R^\top R_d \Omega_d$
	

Table 3.1: Summary of quadratic configuration error functions, error vectors, compatible transport maps, and velocity errors across $\mathbb{R}^3$, $\mathbb{S}^2$, and $SO(3)$.


________________


3.5 First-Order Equivalence Between Tracking Errors and Variations
A central bridge between nonlinear geometric control (Sreenath, Lee, & Kumar, 2013) and variation-based linearization (Wu & Sreenath, 2015) is that, to first order in $\epsilon$, the nonlinear tracking errors $(e_R, e_\Omega)$ and $(e_q, e_\omega)$ coincide with the infinitesimal variations $(\eta, \delta\Omega)$ and $(\xi, \delta\omega)$ up to known signs/projections.
3.5.1 Equivalence on $SO(3)$: $(\eta, \delta\Omega) \leftrightarrow (e_R, e_\Omega)$
Let $R^\epsilon = R_d \exp(\epsilon \eta^\wedge) = R_d (I_3 + \epsilon \eta^\wedge) + \mathcal{O}(\epsilon^2)$ and $\Omega^\epsilon = \Omega_d + \epsilon \delta\Omega + \mathcal{O}(\epsilon^2)$.


1. Attitude error:


$$ R_d^\top R^\epsilon = I_3 + \epsilon \eta^\wedge + \mathcal{O}(\epsilon^2) \implies e_R^\epsilon = \frac{1}{2}\big((I_3 + \epsilon\eta^\wedge) - (I_3 - \epsilon\eta^\wedge)\big)^\vee + \mathcal{O}(\epsilon^2) = \epsilon \eta + \mathcal{O}(\epsilon^2). \tag{3.28} $$


2. Angular velocity error: Using $(R^\epsilon)^\top R_d = I_3 - \epsilon \eta^\wedge + \mathcal{O}(\epsilon^2)$ and Eq. (3.11) ($\delta\Omega = \Omega_d^\wedge \eta + \dot{\eta}$):


$$ \begin{aligned} e_\Omega^\epsilon &= \Omega^\epsilon - (R^\epsilon)^\top R_d \Omega_d \ &= (\Omega_d + \epsilon \delta\Omega) - (I_3 - \epsilon \eta^\wedge)\Omega_d + \mathcal{O}(\epsilon^2) \ &= \epsilon (\delta\Omega + \eta^\wedge \Omega_d) + \mathcal{O}(\epsilon^2) \ &= \epsilon (\Omega_d^\wedge \eta + \dot{\eta} - \Omega_d^\wedge \eta) + \mathcal{O}(\epsilon^2) = \epsilon \dot{\eta} + \mathcal{O}(\epsilon^2). \end{aligned} \tag{3.29} $$


Thus, $\lim_{\epsilon \to 0} \frac{e_R^\epsilon}{\epsilon} = \eta$ and $\lim_{\epsilon \to 0} \frac{e_\Omega^\epsilon}{\epsilon} = \dot{\eta} = \delta\Omega - \Omega_d^\wedge \eta$ (Wu & Sreenath, 2015).
3.5.2 Equivalence on $\mathbb{S}^2$: $(\xi, \delta\omega) \leftrightarrow (e_q, e_\omega)$
Let $q^\epsilon = \exp(\epsilon \xi^\wedge) q_d = q_d + \epsilon (\xi \times q_d) + \mathcal{O}(\epsilon^2)$ with $\xi \cdot q_d = 0$, and $\omega^\epsilon = \omega_d + \epsilon \delta\omega + \mathcal{O}(\epsilon^2)$ with $\omega_d \cdot q_d = 0$.


1. Configuration error: Since $(q_d^\wedge)^2 q_d = 0$ and $(q_d^\wedge)^2 (\xi \times q_d) = q_d \times (q_d \times (\xi \times q_d)) = q_d \times \xi = -(\xi \times q_d)$ (because $\xi \cdot q_d = 0$):


$$ e_q^\epsilon = (q_d^\wedge)^2 q^\epsilon = \epsilon (q_d^\wedge)^2 (\xi \times q_d) + \mathcal{O}(\epsilon^2) = -\epsilon (\xi \times q_d) + \mathcal{O}(\epsilon^2) = -\epsilon \delta q + \mathcal{O}(\epsilon^2). \tag{3.30} $$


2. Angular velocity error: Expanding $-(q^\epsilon)^\wedge{}^2 \omega_d = (I_3 - q^\epsilon (q^\epsilon)^\top) \omega_d$ using $\omega_d \cdot q_d = 0$:


$$ \begin{aligned} -(q^\epsilon)^\wedge{}^2 \omega_d &= \omega_d - \big(q_d + \epsilon \delta q\big)\big((q_d + \epsilon \delta q) \cdot \omega_d\big) + \mathcal{O}(\epsilon^2) \ &= \omega_d - \epsilon (\delta q \cdot \omega_d) q_d + \mathcal{O}(\epsilon^2). \end{aligned} $$


Substituting into $e_\omega^\epsilon = \omega^\epsilon - (-(q^\epsilon)^\wedge{}^2 \omega_d)$ and applying Eq. (3.6) ($(\delta q \cdot \omega_d) + q_d \cdot \delta\omega = 0$):


$$ e_\omega^\epsilon = \epsilon \big(\delta\omega + (\delta q \cdot \omega_d) q_d\big) + \mathcal{O}(\epsilon^2) = \epsilon \big(\delta\omega - (q_d \cdot \delta\omega) q_d\big) + \mathcal{O}(\epsilon^2) = \epsilon (-q_d^\wedge{}^2 \delta\omega) + \mathcal{O}(\epsilon^2). \tag{3.31} $$


That is, $e_\omega^\epsilon/\epsilon$ converges to the projection of $\delta\omega$ onto $T_{q_d}\mathbb{S}^2$!


________________


3.6 Worked Derivations
Worked Derivation 3.1: Proof of $\dot{e}R = C(R_d^\top R) e\Omega$
Let $A = R_d^\top R \in SO(3)$. Using $\dot{R} = R\Omega^\wedge$ and $\dot{R}_d = R_d \Omega_d^\wedge$:


$$ \dot{A} = -\Omega_d^\wedge R_d^\top R + R_d^\top R \Omega^\wedge = A \big(\Omega^\wedge - A^\top \Omega_d^\wedge A\big) = A \big(\Omega - R^\top R_d \Omega_d\big)^\wedge = A e_\Omega^\wedge. $$


Differentiating $e_R = \frac{1}{2}(A - A^\top)^\vee$ gives


$$ \dot{e}R = \frac{1}{2}(\dot{A} - \dot{A}^\top)^\vee = \frac{1}{2}\big(A e\Omega^\wedge + e_\Omega^\wedge A^\top\big)^\vee. $$


For any $A \in SO(3)$ and $x \in \mathbb{R}^3$, expanding the $(i, j)$ entry via the Levi-Civita symbol or testing on $A = \exp(\theta u^\wedge)$ yields $\frac{1}{2}(A x^\wedge + x^\wedge A^\top)^\vee = \frac{1}{2}(\operatorname{Tr}(A) I_3 - A^\top) x$. Setting $x = e_\Omega$ gives $\dot{e}R = C(R_d^\top R) e\Omega$ with $C(A) = \frac{1}{2}(\operatorname{Tr}(A) I_3 - A^\top)$, which is symmetric-positive-definite whenever $\Psi_R < 2$ (and satisfies $|C(A)|_2 \le 1$) (Sreenath, Lee, & Kumar, 2013; Kotaru, Edmonson, & Sreenath, 2020).
Worked Derivation 3.2: Transport of Adaptive Uncertainty on $SO(3)$
In $\mathcal{L}1$ adaptive geometric attitude control (Kotaru, Edmonson, & Sreenath, 2020), the true attitude error dynamics are written in terms of $(e_R, e\Omega)$ with matched uncertainty $\theta \in \mathbb{R}^3$ and virtual control $\mu \in \mathbb{R}^3$. Because the reference model tracks $(\hat{R}, \hat{\Omega})$ while the true system is at $(R, \Omega)$, comparing $\hat{\Omega}$ and $\Omega$ requires the same transport map $T(R, \hat{R})\hat{\Omega} = R^\top \hat{R} \hat{\Omega}$, yielding the prediction error $\tilde{e}_\Omega = \Omega - R^\top \hat{R}\hat{\Omega}$.


________________


3.7 Exercises
1. Exercise 3.1 ($\mathfrak{so}(3)$ Bracket Identity). Prove directly from Eq. (3.1) that for any $a, b \in \mathbb{R}^3$, $a^\wedge b^\wedge - b^\wedge a^\wedge = (a \times b)^\wedge$.
2. Exercise 3.2 (Bounds on $|e_R|$). Show that $|e_R|^2 = (2 - \Psi_R)\Psi_R$ for $\Psi_R = \frac{1}{2}\operatorname{Tr}(I_3 - R_d^\top R)$, and deduce that $|e_R| \le 1$ globally on $SO(3)$.
3. Exercise 3.3 (Bounds on $|e_q|$). Show that $|e_q|^2 = (2 - \Psi_q)\Psi_q$ for $\Psi_q = 1 - q \cdot q_d$ on $\mathbb{S}^2$.
4. Exercise 3.4 (Variation of $\dot{q}$ on $\mathbb{S}^2$). Starting from Eq. (3.7), express $\delta\omega$ in terms of $\xi, \dot{\xi}, q_d, \omega_d$ when $\omega_d \cdot q_d = 0$ and $\delta\omega \cdot q_d = -(\xi \times q_d)\cdot \omega_d$.
5. Exercise 3.5 (Product Manifold Transport). Write out the full configuration error $\Psi(X, X_d)$, compatible transport map $T(X, X_d)$, and velocity error $e_{\dot{X}}$ for a quadrotor with a cable-suspended load evolving on $Q = \mathbb{R}^3 \times SO(3) \times \mathbb{S}^2$.


________________


References
1. Wu, G., & Sreenath, K. (2015). Variation-Based Linearization of Nonlinear Systems Evolving on $SO(3)$ and $\mathbb{S}^2$. IEEE Access. PDF
2. Wu, G., & Sreenath, K. (2016). Safety-Critical Geometric Control for Systems on Manifolds Subject to Time-Varying Constraints. IEEE Transactions on Automatic Control. PDF
3. Sreenath, K., Lee, T., & Kumar, V. (2013). Geometric Control and Differential Flatness of a Quadrotor UAV with a Cable-Suspended Load. IEEE CDC. PDF
4. Kotaru, P., Edmonson, R., & Sreenath, K. (2020). Geometric $L_1$ Adaptive Attitude Control for a Quadrotor UAV. ASME JDSMC. PDF