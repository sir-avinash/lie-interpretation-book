________________


name: ch01-manifold-diagnostics description: Guide an engineer through diagnosing coordinate singularities (gimbal lock) and quaternion unwinding in an existing attitude/direction controller, and migrating to coordinate-free SO(3) and S^2 geometric control.
Manifold Diagnostics & Migration to $SO(3) / S^2$
Use this skill when auditing an existing attitude or direction-tracking controller for coordinate singularities, loss of feedback invariance, or quaternion unwinding, and when migrating the controller to coordinate-free geometric control on $SO(3)$ and $S^2$.
Diagnostic Checklist
Step 1: Audit State Representation for Coordinate Singularities
1. Check for minimal 3-parameter attitude representations (ZYX/XYZ Euler angles, Tait-Bryan angles, Rodrigues parameters).
2. Inspect kinematic Jacobians for trigonometric denominators such as $\frac{1}{\cos\theta}$ or $\tan\theta$:
   * For ZYX Euler angles $\eta = (\psi, \theta, \phi)^\top$, $\dot{\eta} = E^{-1}(\theta, \phi)\Omega$ blows up at $\theta = \pm \frac{\pi}{2}$ where $\det(E) = -\cos\theta = 0$.
   * Flag any controller computing $\tau = -K_p(\eta - \eta_d) - K_d\dot{\eta}$: this law is not left-invariant on $SO(3)$ and fails near pitch $\pm 90^\circ$.
3. Check numerical conditioning: Evaluate $\kappa(E) = \sigma_{\max}(E)/\sigma_{\min}(E)$ along expected flight/motion trajectories. If $\kappa(E) > 10$ or pitch approaches $|\theta| > 75^\circ$, migration to $SO(3)$ is mandatory.
Step 2: Diagnose $S^3$ Double-Cover & Unwinding Risks
1. Check for unit quaternion state $\mathfrak{q} = (q_0, q_v^\top)^\top \in S^3$:
   * Verify whether the control law uses $\tau = -k_p q_v - k_d \Omega$ without sign handling. If $q_0 < 0$ at initialization or after a disturbance, the robot will unwind by up to $360^\circ$ toward $+\mathfrak{q}_I$ instead of taking the short geodesic path.
   * Verify whether the control law uses a discontinuous $\operatorname{sgn}(q_0)$ switch. Flag this as a chattering risk near $q_0 = 0$ ($180^\circ$ rotation) under sensor noise.
Step 3: Verify Topological & Equilibrium Expectations
1. Confirm realistic stability claims: By Bhat & Bernstein (2000), no continuous autonomous feedback law can achieve global asymptotic stability on compact manifolds without boundary ($SO(3)$, $S^2$) or their tangent bundles.
2. Expect Almost-Global Exponential Stability (AGES): Verify that the target design achieves exponential convergence from all initial states except a measure-zero set (the stable manifold of antipodal saddle equilibria at $\theta = \pi$, where $\Psi_R = 2$ or $\Psi_q = 2$).


________________


Migration Workflow to Coordinate-Free $SO(3)$ and $S^2$
Step 4: Replace Error Metrics with Manifold-Intrinsic Definitions
1. Define the hat/vee isomorphism:
   * $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ such that $x^\wedge y = x \times y$, and $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$.
2. On $SO(3)$ (Rigid-Body Attitude):
   * Configuration error function: $$\Psi_R(R, R_d) = \frac{1}{2}\operatorname{Tr}(I_3 - R_d^\top R) \in [0, 2]$$
   * Attitude error vector: $$e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee \in \mathbb{R}^3$$
   * Angular velocity error (transported to body frame): $$e_\Omega = \Omega - R^\top R_d \Omega_d \in \mathbb{R}^3$$
   * Geometric PD + feedforward torque (Sreenath, Lee, & Kumar, CDC 2013; Kotaru, Edmonson, & Sreenath, JDSMC 2020): $$\tau = -k_R e_R - k_\Omega e_\Omega + \Omega \times J\Omega - J\left(\Omega^\wedge R^\top R_d \Omega_d - R^\top R_d \dot{\Omega}_d\right)$$
3. On $S^2$ (Unit-Vector Direction, e.g., Thrust Axis or Cable Load):
   * Configuration error function: $$\Psi_q(q, q_d) = 1 - q \cdot q_d \in [0, 2]$$
   * Tangent-space error vector: $$e_q = q_d \times q \in T_q S^2$$
4. Use Structure-Preserving Integration:
   * Integrate $\dot{R} = R\Omega^\wedge$ using the Lie-group exponential map $R_{k+1} = R_k \exp(\Delta t,\Omega_k^\wedge)$ (Rodrigues' formula) rather than component-wise Euler integration, ensuring $R^\top R = I_3$ and $\det(R) = +1$ to machine precision.
Step 5: Verification Tests
Run the following three sanity checks after migration:


1. Gimbal-Lock Pass-Through: Command a smooth trajectory through $\theta_{\text{pitch}} = \pm 90^\circ$ and verify $\Psi_R$ and $|e_R|$ remain smooth and bounded.
2. Antipodal / Sign-Invariance Check: Initialize $R(0)$ at $\pm 10^\circ$ from $R_d$ and verify identical geodesic convergence without unwinding.
3. Manifold Invariance Check: Assert $|R(t)^\top R(t) - I_3|_F < 10^{-10}$ and $|\det(R(t)) - 1| < 10^{-10}$ over the entire simulation.