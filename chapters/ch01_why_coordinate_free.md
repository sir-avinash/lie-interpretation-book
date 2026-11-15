Chapter 1: Why Coordinate-Free Geometry? Singularities, Unwinding & Global Topology
Core Thesis: Control laws designed in local coordinates or on covering spaces inherit artificial singularities, break physical symmetries, and suffer from unwinding. By formulating kinematics, configuration errors, and feedback directly on the configuration manifold—$SO(3)$ for rigid-body attitude and $S^2$ for unit-vector directions—we obtain globally well-defined, singularity-free controllers with almost-global exponential stability.


________________


Learning Objectives
By the end of this chapter, you will be able to:


1. Define the Special Orthogonal Group $SO(3)$, its Lie algebra $\mathfrak{so}(3)$, and the hat/vee isomorphism $(\cdot)^\wedge, (\cdot)^\vee$.
2. Derive the ZYX Euler-angle kinematic Jacobian $E(\eta)$, prove rank deficiency at pitch $\theta = \pm \frac{\pi}{2}$ (gimbal lock), and explain why local-coordinate PD control loses feedback invariance.
3. Explain the $S^3 \to SO(3)$ double cover for unit quaternions and demonstrate how continuous quaternion feedback induces the unwinding phenomenon (and why naive discontinuous sign-switching induces chattering).
4. State the topological obstruction to continuous global stabilization on compact manifolds without boundary (Bhat & Bernstein, 2000; Milnor) and interpret almost-global exponential stability (AGES) around measure-zero antipodal equilibria.
5. Formulate and evaluate the coordinate-free configuration error functions $\Psi_R(R, R_d)$ on $SO(3)$ and $\Psi_q(q, q_d)$ on $S^2$.


________________


1.1 The Coordinate Trap: Why $\mathbb{R}^n$ Intuition Fails on Curved Spaces
Classical control theory is built on Euclidean vector spaces $\mathbb{R}^n$, where states can be subtracted component-wise ($e = x - x_d$), a single global coordinate chart covers the entire space, and linearizing around an equilibrium yields a globally valid coordinate representation. Mechanical and robotic systems—quadrotors, satellites, cable-suspended loads, robotic manipulators, and legged robots—do not live in $\mathbb{R}^n$. Their configurations evolve on smooth, curved manifolds such as the three-dimensional rotation group $SO(3)$ and the two-sphere $S^2$.


Attempting to force a curved manifold into $\mathbb{R}^n$ via local coordinates (such as Euler angles) or redundant covering spaces (such as unit quaternions) introduces pathologies that are mathematical artifacts of the representation, not physical properties of the robot.


Def. 1.1 (Special Orthogonal Group $SO(3)$ and Lie Algebra $\mathfrak{so}(3)$). The configuration space of rigid-body orientations in three-dimensional space is the Special Orthogonal Group: $$ SO(3) = \left{ R \in \mathbb{R}^{3 \times 3} ;\middle|; R^\top R = I_3, ; \det(R) = +1 \right}, \tag{1.1} $$ where $I_3$ is the $3 \times 3$ identity matrix. Its tangent space at the identity $I_3$ is the Lie algebra $\mathfrak{so}(3) = { \hat{\Omega} \in \mathbb{R}^{3 \times 3} \mid \hat{\Omega}^\top = -\hat{\Omega} }$, the vector space of $3 \times 3$ real skew-symmetric matrices.


Def. 1.2 (Hat and Vee Maps). The hat map $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ is the linear isomorphism defined such that for all $x, y \in \mathbb{R}^3$, $x^\wedge y = x \times y$: $$ x^\wedge = \begin{bmatrix} 0 & -x_3 & x_2 \ x_3 & 0 & -x_1 \ -x_2 & x_1 & 0 \end{bmatrix} \in \mathfrak{so}(3), \quad x = \begin{bmatrix} x_1 \ x_2 \ x_3 \end{bmatrix} \in \mathbb{R}^3. \tag{1.2} $$ Its inverse, the vee map $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$, extracts the unique vector $x$ from the skew-symmetric matrix $x^\wedge$, i.e., $(x^\wedge)^\vee = x$.


Using Eq. (1.2), the rotational kinematics of a rigid body with body-frame angular velocity $\Omega \in \mathbb{R}^3$ evolve on $SO(3)$ according to the left-invariant differential equation: $$ \dot{R} = R \Omega^\wedge. \tag{1.3} $$ Notice that Eq. (1.3) has no denominators, no trigonometric singularities, and preserves $R(t) \in SO(3)$ for all $t \ge 0$.


________________


1.2 Euler-Angle Singularities, Gimbal Lock, and Loss of Feedback Invariance
To avoid working with $3 \times 3$ matrices subject to $6$ orthogonality constraints, engineers frequently parameterize $R \in SO(3)$ using three minimal local coordinates, such as ZYX (yaw-pitch-roll) Euler angles $\eta = (\psi, \theta, \phi)^\top \in \mathbb{R}^3$: $$ R(\psi, \theta, \phi) = R_z(\psi) R_y(\theta) R_x(\phi), \tag{1.4} $$ where $\psi$ is yaw, $\theta$ is pitch, and $\phi$ is roll. Differentiating Eq. (1.4) with respect to time and matching $\dot{R} = R \Omega^\wedge$ from Eq. (1.3) yields the body angular velocity $\Omega = E(\theta, \phi)\dot{\eta}$: $$ \Omega = \underbrace{\begin{bmatrix} -\sin\theta & 0 & 1 \ \cos\theta \sin\phi & \cos\phi & 0 \ \cos\theta \cos\phi & -\sin\phi & 0 \end{bmatrix}}_{E(\theta, \phi)} \begin{bmatrix} \dot{\psi} \ \dot{\theta} \ \dot{\phi} \end{bmatrix}. \tag{1.5} $$ Inverting $E(\theta, \phi)$ gives the Euler-angle kinematic differential equation $\dot{\eta} = E^{-1}(\theta, \phi)\Omega$: $$ \begin{bmatrix} \dot{\psi} \ \dot{\theta} \ \dot{\phi} \end{bmatrix} = \frac{1}{\cos\theta} \begin{bmatrix} 0 & \sin\phi & \cos\phi \ 0 & \cos\theta\cos\phi & -\cos\theta\sin\phi \ \cos\theta & \sin\theta\sin\phi & \sin\theta\cos\phi \end{bmatrix} \Omega. \tag{1.6} $$


Prop. 1.1 (Rank Deficiency and Loss of Feedback Invariance at Gimbal Lock). For the ZYX Euler-angle parameterization in Eq. (1.4)–(1.6):


1. $\det(E(\theta, \phi)) = -\cos\theta$, so $E(\theta, \phi)$ becomes singular ($\operatorname{rank}(E) = 2 < 3$) whenever $\theta = \pm \frac{\pi}{2}$.
2. At $\theta = \pm \frac{\pi}{2}$, the yaw and roll axes align in inertial space, causing $\dot{\psi}$ and $\dot{\phi}$ to become collinear and rendering $E^{-1}(\theta, \phi)$ undefined.
3. A linear coordinate PD law $\tau = -K_p(\eta - \eta_d) - K_d\dot{\eta}$ is not left-invariant on $SO(3)$: its closed-loop behavior depends on the arbitrary choice of inertial reference frame and blows up as $\theta \to \pm \frac{\pi}{2}$.


Proof Sketch. Expanding $\det(E)$ from Eq. (1.5) along the third column gives $\det(E) = 1 \cdot (\cos\theta\sin\phi(-\sin\phi) - \cos\theta\cos\phi(\cos\phi)) = -\cos\theta$. When $\theta = +\frac{\pi}{2}$, the first and third columns of $E$ are $(-1, 0, 0)^\top$ and $(1, 0, 0)^\top$, which are linearly dependent. Consequently, the column space of $E$ collapses to $\operatorname{span}{e_1, (0, \cos\phi, -\sin\phi)^\top}$, a 2D subspace of $\mathbb{R}^3$. Furthermore, for two rotations $R, R_d \in SO(3)$ rotated by a common inertial offset $Q \in SO(3)$ ($R \mapsto QR, R_d \mapsto QR_d$), the physical relative error $R_d^\top R$ is invariant, whereas $\eta(QR) - \eta(QR_d) \neq \eta(R) - \eta(R_d)$ in general, breaking feedback invariance. $\blacksquare$
Worked Example 1.1: Gimbal Lock Jacobian Collapse & Rate Blow-Up
Suppose a quadrotor executes a smooth pitch-up maneuver with constant body angular velocity $\Omega = (0, 0.5, 0.5)^\top;\text{rad/s}$ as its pitch approaches $\theta = 89^\circ$ ($\epsilon = \frac{\pi}{2} - \theta = 1^\circ \approx 0.01745;\text{rad}$) with $\phi = 45^\circ$. From Eq. (1.6):


* $\cos\theta = \sin(1^\circ) \approx 0.01745$.
* $\dot{\psi} = \frac{1}{\cos\theta}(\Omega_2 \sin\phi + \Omega_3 \cos\phi) = \frac{1}{0.01745}(0.5 \cdot 0.7071 + 0.5 \cdot 0.7071) \approx 40.52;\text{rad/s}$.
* $\dot{\phi} \approx \tan\theta (\Omega_2 \sin\phi + \Omega_3 \cos\phi) \approx 40.51;\text{rad/s}$.


Even though the physical angular speed $|\Omega| = \frac{1}{\sqrt{2}} \approx 0.707;\text{rad/s}$ is completely benign, the Euler-angle rates explode past $40;\text{rad/s}$ (over $2300^\circ/\text{s}$) and the Jacobian condition number $\kappa(E) = \sigma_{\max}(E)/\sigma_{\min}(E)$ diverges as $O(1/\cos\theta)$. See Fig. 1.1 in the companion notebook ch01_singularities_unwinding.ipynb.


Representation
	Space
	Dimension
	Constraints
	Singularities?
	Uniqueness
	Global Topology
	Euler Angles $(\psi, \theta, \phi)$
	$\mathbb{R}^3$ (local)
	3
	0
	Yes ($\theta = \pm \frac{\pi}{2}$)
	Non-unique ($2\pi$ wrap + gimbal lock)
	Local chart only; fails globally
	Unit Quaternions $\mathfrak{q}$
	$S^3 \subset \mathbb{R}^4$
	4
	1 ($|\mathfrak{q}|=1$)
	No
	2-to-1 double cover ($\mathfrak{q} \equiv -\mathfrak{q}$)
	Simply connected; induces unwinding
	Rotation Matrix $R$
	$SO(3) \subset \mathbb{R}^{3\times 3}$
	9
	6 ($R^\top R = I_3, \det R=1$)
	No
	Unique (1-to-1)
	Compact Lie group ($\cong \mathbb{R}\mathbb{P}^3$)
	

Table 1.1: Comparison of standard 3D attitude representations for robotics and control.


________________


1.3 Unit Quaternions ($S^3$), Double Cover, and the Unwinding Phenomenon
To eliminate Euler-angle singularities, aerospace and robotics practitioners often turn to unit quaternions $\mathfrak{q} = (q_0, q_v^\top)^\top \in S^3 \subset \mathbb{R}^4$, where $q_0 \in \mathbb{R}$ is the scalar part, $q_v \in \mathbb{R}^3$ is the vector part, and $q_0^2 + |q_v|^2 = 1$.


Remark 1.1 (Notation Disambiguation: $S^3$ Quaternions vs. $S^2$ Unit Vectors). Throughout this textbook, we reserve $q \in S^2 \subset \mathbb{R}^3$ for physical unit-direction vectors (such as a cable-suspended load direction or spherical pendulum axis) and denote unit quaternions by $\mathfrak{q} = (q_0, q_v^\top)^\top \in S^3 \subset \mathbb{R}^4$.


Def. 1.3 (The $S^3 \to SO(3)$ Double Cover). The map $\pi : S^3 \to SO(3)$ defined by Rodrigues' formula: $$ R(\mathfrak{q}) = I_3 + 2 q_0 q_v^\wedge + 2 (q_v^\wedge)^2 \tag{1.7} $$ is a surjective Lie-group homomorphism with kernel $\ker(\pi) = {+\mathfrak{q}_I, -\mathfrak{q}_I}$, where $\mathfrak{q}_I = (1, 0, 0, 0)^\top$. Consequently, $\pi(\mathfrak{q}) = \pi(-\mathfrak{q})$ for every $\mathfrak{q} \in S^3$: $S^3$ is a double cover of $SO(3)$.


The quaternion kinematics corresponding to Eq. (1.3) are: $$ \dot{\mathfrak{q}} = \frac{1}{2} \mathfrak{q} \otimes \begin{bmatrix} 0 \ \Omega \end{bmatrix} = \frac{1}{2} \begin{bmatrix} -q_v^\top \ q_0 I_3 + q_v^\wedge \end{bmatrix} \Omega, \tag{1.8} $$ which are globally nonsingular. However, the 2-to-1 redundancy $\mathfrak{q} \equiv -\mathfrak{q}$ creates a subtle, dangerous control pathology known as unwinding, rigorously analyzed by Bhat & Bernstein (2000).
Why Unwinding Occurs
Consider stabilizing a rigid body $J\dot{\Omega} + \Omega \times J\Omega = \tau$ to the identity orientation $R_d = I_3$ using the continuous quaternion PD controller: $$ \tau = -k_p q_v - k_d \Omega, \quad k_p, k_d > 0. $$ This controller treats $+\mathfrak{q}_I = (1, 0, 0, 0)^\top$ as the target equilibrium and $-\mathfrak{q}_I = (-1, 0, 0, 0)^\top$ as an unstable equilibrium, even though both represent the exact same physical orientation $I_3$.


For a rotation by angle $\theta \in [0, 2\pi)$ about a unit axis $\hat{u} \in S^2$, the quaternion is $\mathfrak{q} = (\cos\frac{\theta}{2}, \sin\frac{\theta}{2},\hat{u})^\top$. The true geodesic angle on $SO(3)$ vs. the quaternion error angle are: $$ \theta_{SO(3)} = 2 \arccos(|q_0|), \qquad \theta_{S^3} = 2 \arccos(q_0). \tag{1.9} $$ If the initial physical orientation is only $10^\circ$ away from $I_3$, but the estimator initializes on the $-\mathfrak{q}I$ hemisphere ($\theta{S^3} = 350^\circ$, so $q_0 = \cos(175^\circ) < 0$), the continuous controller drives the robot along the long $350^\circ$ path all the way around $S^3$ back to $+\mathfrak{q}_I$, wasting massive fuel and potentially colliding with obstacles.


Why Not Simply Multiply by $\operatorname{sgn}(q_0)$? A common engineering patch is $\tau = -k_p \operatorname{sgn}(q_0) q_v - k_d \Omega$. While this chooses the shorter path in noiseless simulation, $\operatorname{sgn}(q_0)$ is discontinuous at the equator $q_0 = 0$ (corresponding to a $180^\circ$ rotation). In the presence of sensor noise near $q_0 = 0$, the control torque chatters violently between $+k_p q_v$ and $-k_p q_v$, and the closed-loop vector field fails to be continuous on $SO(3)$.
Worked Example 1.2: Unwinding Trajectory on $S^3$
Consider a rigid body with inertia $J = \operatorname{diag}(0.05, 0.05, 0.1);\text{kg}\cdot\text{m}^2$ starting at rest ($\Omega(0) = 0$) at a physical yaw of $-10^\circ$ about $\hat{u} = e_3$. The two equivalent quaternion representations are:


* $\mathfrak{q}_+(0) = (\cos(-5^\circ), 0, 0, \sin(-5^\circ))^\top \approx (+0.9962, 0, 0, -0.0872)^\top$ ($q_0 > 0$),
* $\mathfrak{q}-(0) = -\mathfrak{q}+(0) \approx (-0.9962, 0, 0, +0.0872)^\top$ ($q_0 < 0$).


Under $\tau = -k_p q_v - k_d \Omega$, $\mathfrak{q}+(0)$ rotates $+10^\circ$ to $I_3$, whereas $\mathfrak{q}-(0)$ rotates $-350^\circ$ in the opposite direction to reach $+\mathfrak{q}_I$, expending over $24\times$ the control energy $\int |\tau|^2 dt$. See Fig. 1.2 in ch01_singularities_unwinding.ipynb.


________________


1.4 Topological Obstructions to Global Stabilization on Compact Manifolds
Can any continuous, time-invariant state-feedback law globally asymptotically stabilize a rigid body on $SO(3)$ or a pendulum on $S^2$? The answer is no—and the reason is purely topological.


Prop. 1.2 (Topological Obstruction to Continuous Global Stabilization; Bhat & Bernstein, 2000). Let $M$ be a state space having the structure of a vector bundle over a compact manifold $Q$ without boundary (such as $TSO(3)$ or $TS^2$). Then no continuous, autonomous vector field on $M$ possesses a globally asymptotically stable equilibrium.


Topological Intuition. A fundamental theorem of dynamical systems (Milnor; Wilson) states that the domain of attraction of any asymptotically stable equilibrium in an $n$-dimensional manifold must be homeomorphic to $\mathbb{R}^n$, which is contractible (every loop can be shrunk to a point). However:


* $S^2$ is compact and non-contractible; moreover, by the Hairy Ball Theorem, every continuous tangent vector field on $S^2$ must vanish at least at one point (and any gradient-like potential on $S^2$ has at least one maximum/saddle in addition to its minimum).
* $SO(3) \cong \mathbb{R}\mathbb{P}^3$ is compact and non-contractible (its fundamental group is $\pi_1(SO(3)) = \mathbb{Z}_2$, meaning a $360^\circ$ rotation loop cannot be contracted to a point, whereas a $720^\circ$ loop can).


Remark 1.2 (Almost-Global Exponential Stability & Antipodal Equilibria). Because global asymptotic stability via continuous feedback is topologically impossible on $SO(3)$ and $S^2$, the strongest achievable continuous guarantee is almost-global exponential stability (AGES): every trajectory converges exponentially to the desired equilibrium except for initial conditions lying on a set of Lebesgue measure zero (specifically, the stable manifold of antipodal saddle equilibria, such as $\theta = \pi$ on $SO(3)$ or $q = -q_d$ on $S^2$). This is the theoretical foundation of geometric controllers in Sreenath, Lee, & Kumar (CDC 2013) and safety-critical geometric control in Wu & Sreenath (IEEE TAC 2016).


________________


1.5 Coordinate-Free Configuration Errors on $SO(3)$ and $S^2$
To avoid both Euler-angle singularities and quaternion unwinding, we define configuration error functions directly on the manifold itself.


Def. 1.4 (Coordinate-Free Attitude Error Function on $SO(3)$). Given a current attitude $R \in SO(3)$ and desired attitude $R_d \in SO(3)$, the configuration error function $\Psi_R : SO(3) \times SO(3) \to [0, 2]$ is: $$ \Psi_R(R, R_d) = \frac{1}{2}\operatorname{Tr}!\left(I_3 - R_d^\top R\right). \tag{1.10} $$


By Rodrigues' formula, if $R_d^\top R = \exp(\theta \hat{u})$ has geodesic angle $\theta \in [0, \pi]$, then $\operatorname{Tr}(R_d^\top R) = 1 + 2\cos\theta$, so $\Psi_R(R, R_d) = 1 - \cos\theta$. Thus $\Psi_R = 0$ if and only if $R = R_d$, and $\Psi_R = 2$ if and only if $\theta = \pi$ (a $180^\circ$ rotation).


Taking the directional variation $\delta R = R \eta^\wedge$ ($\eta \in \mathbb{R}^3$) of $\Psi_R$ yields $\delta\Psi_R = e_R \cdot \eta$, which defines the attitude error vector $e_R \in \mathbb{R}^3$ (Wu & Sreenath, IEEE Access 2015; Kotaru, Edmonson, & Sreenath, JDSMC 2020): $$ e_R(R, R_d) = \frac{1}{2}\left(R_d^\top R - R^\top R_d\right)^\vee. \tag{1.11} $$ To compare the body angular velocity $\Omega$ with the desired body angular velocity $\Omega_d$ (which live in different tangent spaces $T_R SO(3)$ and $T_{R_d} SO(3)$), we transport $\Omega_d$ to the current body frame via $R^\top R_d$: $$ e_\Omega(R, \Omega, R_d, \Omega_d) = \Omega - R^\top R_d \Omega_d. \tag{1.12} $$ Notice that Eq. (1.10)–(1.12) are left-invariant: replacing $(R, R_d)$ with $(QR, QR_d)$ for any $Q \in SO(3)$ leaves $R_d^\top R$, $\Psi_R$, $e_R$, and $e_\Omega$ completely unchanged.


Def. 1.5 (Coordinate-Free Unit-Vector Error Function on $S^2$). Given a unit vector $q \in S^2$ and desired unit vector $q_d \in S^2$ (e.g., the thrust direction of a quadrotor or cable direction of a suspended load in Sreenath, Lee, & Kumar, CDC 2013), the configuration error function $\Psi_q : S^2 \times S^2 \to [0, 2]$ is: $$ \Psi_q(q, q_d) = 1 - q \cdot q_d. \tag{1.13} $$ The corresponding tangent-space configuration error vector on $T_q S^2$ is: $$ e_q(q, q_d) = q_d \times q. \tag{1.14} $$ Again, $\Psi_q = 1 - \cos\theta_q \in [0, 2]$, vanishing uniquely at $q = q_d$ ($\theta_q = 0$) and attaining its maximum $\Psi_q = 2$ at the antipodal point $q = -q_d$ ($\theta_q = \pi$), where $e_q = 0$ forms a measure-zero unstable equilibrium.
Worked Example 1.3: Evaluating $\Psi_R$ and $e_R$ at Gimbal Lock and Antipodal Equilibria
1. At ZYX Gimbal Lock ($\theta = 90^\circ$): Let $R_d = I_3$ and $R = R_y(\frac{\pi}{2}) = \begin{bmatrix} 0 & 0 & 1 \ 0 & 1 & 0 \ -1 & 0 & 0 \end{bmatrix}$. From Eq. (1.10)–(1.11): $$ \Psi_R(R, I_3) = \frac{1}{2}\operatorname{Tr}(I_3 - R) = \frac{1}{2}(3 - 1) = 1, \qquad e_R = \frac{1}{2}\left(R - R^\top\right)^\vee = \begin{bmatrix} 0 \ 1 \ 0 \end{bmatrix}. $$ Both $\Psi_R = 1$ and $|e_R| = 1$ are completely smooth, bounded, and non-singular—in stark contrast to Eq. (1.6), which blows up.
2. At the Antipodal Saddle ($\theta = 180^\circ$): Let $R = \operatorname{diag}(1, -1, -1)$ (a $180^\circ$ rotation about $e_1$). Then $\operatorname{Tr}(R) = -1$, so $\Psi_R(R, I_3) = \frac{1}{2}(3 - (-1)) = 2$, and $R - R^\top = 0 \implies e_R = 0$. Any infinitesimal perturbation away from $\theta = \pi$ with $\Omega = 0$ immediately produces $e_R \neq 0$ and drives the system back toward $I_3$ (Fig. 1.3).


This coordinate-free foundation enables variation-based linearization (Wu & Sreenath, IEEE Access 2015), geometric $L_1$ adaptive control (Kotaru, Edmonson, & Sreenath, JDSMC 2020), and time-varying control barrier functions on manifolds (Wu & Sreenath, IEEE TAC 2016), which we develop in subsequent chapters.


________________


1.6 End-of-Chapter Exercises
1. Euler-Angle Jacobian Verification: Starting from $R(\psi, \theta, \phi) = R_z(\psi)R_y(\theta)R_x(\phi)$ in Eq. (1.4), compute $R^\top \dot{R}$ explicitly and use the vee map $(\cdot)^\vee$ to verify Eq. (1.5) and Eq. (1.6).
2. Singular Value Analysis at Gimbal Lock: Compute the singular values of $E(\theta, \phi)$ in Eq. (1.5) as a function of $\theta$. Show analytically that $\sigma_{\min}(E) \to 0$ as $\theta \to \pm \frac{\pi}{2}$ and verify numerically in ch01_singularities_unwinding.ipynb.
3. Unwinding Energy Ratio: Using Eq. (1.8) and (1.9), derive the initial quaternion $\mathfrak{q}-(0)$ for a rotation of angle $\alpha \in (0, \pi)$ about $\hat{u}$ on the negative hemisphere. Simulate both $\mathfrak{q}+(0)$ and $\mathfrak{q}_-(0)$ in ch01_singularities_unwinding.ipynb and compute the ratio of integrated squared torque $\int_0^T |\tau|^2 dt$.
4. Properties of $\Psi_R$ and $e_R$: Prove that for any $R, R_d \in SO(3)$: (a) $|e_R|^2 = \Psi_R(2 - \Psi_R)$, and (b) $\frac{d}{dt}\Psi_R(R, R_d) = e_R \cdot e_\Omega$. (Hint: Use $\operatorname{Tr}(A B^\wedge) = -\frac{1}{2}\operatorname{Tr}((A - A^\top)^\vee \cdot B^\vee)$ or the cyclic property of trace.)
5. Antipodal Equilibria on $S^2$: Using Eq. (1.13)–(1.14), find all points $q \in S^2$ where $e_q(q, q_d) = 0$. Linearize the $S^2$ kinematics $\dot{q} = \omega \times q$ around $q = -q_d$ under $\omega = -k_q e_q$ ($k_q > 0$) to prove that $q = -q_d$ is an unstable equilibrium.


________________


References
* Bhat, S. P., & Bernstein, D. S. (2000). A topological obstruction to continuous global stabilization of rotational motion and the unwinding phenomenon. Systems & Control Letters, 39(1), 63–70. PDF
* Kotaru, P., Edmonson, R., & Sreenath, K. (2020). Geometric $L_1$ Adaptive Attitude Control for a Quadrotor UAV. ASME Journal of Dynamic Systems, Measurement, and Control (JDSMC). PDF
* Sreenath, K., Lee, T., & Kumar, V. (2013). Geometric Control and Differential Flatness of a Quadrotor UAV with a Cable-Suspended Load. IEEE Conference on Decision and Control (CDC). PDF
* Wu, G., & Sreenath, K. (2015). Variation-based Linearization of Nonlinear Systems Evolving on $SO(3)$ and $S^2$. IEEE Access. PDF
* Wu, G., & Sreenath, K. (2016). Safety-Critical Geometric Control for Systems on Manifolds Subject to Time-Varying Constraints. IEEE Transactions on Automatic Control (TAC). PDF