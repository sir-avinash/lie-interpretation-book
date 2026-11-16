________________


name: ch07-variation-lqr-design description: Guide engineers through variation-based linearization, state-constrained controllability verification, and finite-horizon LTV-LQR synthesis on product manifolds (R^3, S^2, SO(3)).
Variation-Based Linearization & LTV-LQR Design on Product Manifolds
Use this skill when designing trajectory-tracking controllers for nonlinear mechanical systems evolving on product manifolds $\mathcal{Q} = \prod_i M_i$ with $M_i \in {\mathbb{R}^3, S^2, SO(3)}$ (e.g., 3D pendulums, spherical pendulums, quadrotors, and cable-suspended payloads), following Wu & Sreenath (IEEE Access 2015) and Kotaru, Wu, & Sreenath (arXiv:1711.04895).
Workflow
Step 1: Define the Reference Trajectory & Manifold Factors
1. Identify each configuration factor $M_i \in {\mathbb{R}^3, S^2, SO(3)}$.
2. Obtain a smooth, dynamically feasible reference trajectory $(x_d(t), u_d(t))$ (e.g., via differential flatness or spline optimization).
3. Use the canonical hat/vee convention: $x^\wedge y = x \times y$ for $x, y \in \mathbb{R}^3$.
Step 2: Construct Tangent-Space Variations & Constraints
Assemble the variation state vector $s = [s_1^T, \dots, s_k^T]^T \in \mathbb{R}^n$:


* $\mathbb{R}^3$ factor: $s_i = [\delta p^T, \delta v^T]^T \in \mathbb{R}^6$ with $\delta \dot{p} = \delta v$ (no constraint).
* $SO(3)$ factor: $s_i = [\eta^T, \delta \Omega^T]^T \in \mathbb{R}^6$ using $\delta R = R_d \eta^\wedge$ and $\delta \Omega = \Omega_d^\wedge \eta + \dot{\eta}$, so $\dot{\eta} = -\Omega_d^\wedge \eta + \delta \Omega$ (no constraint).
* $S^2$ factor: $s_i = [\xi^T, \delta \omega^T]^T \in \mathbb{R}^6$ using $\delta q = \xi \times q_d$ with $\xi \cdot q_d = 0$, $(\xi \times q_d)\cdot \omega_d + q_d \cdot \delta \omega = 0$, and $\dot{\xi} = \omega_d^\wedge \xi + (I_3 - q_d q_d^T) \delta \omega$. Append the 2 rows of $C_{S^2}(t)$ to $C(x_d(t))$.
Step 3: Derive $(A(t), B(t), C(t))$ & Verify Invariance
1. Linearize the Newton–Euler equations symbolically about $(x_d(t), u_d(t))$ to obtain $\dot{s} = A(x_d(t)) s + B(x_d(t)) \delta u$, $C(x_d(t)) s = 0$.
2. Verify time-invariance of $\mathcal{N}(C(t))$ numerically across $t \in [t_0, t_f]$:
   * Check $C(t) B(t) = 0$.
   * Check $|(C(t) A(t) + \dot{C}(t)) s| \approx 0$ for basis vectors $s \in \mathcal{N}(C(t))$.
Step 4: Verify State-Constrained Controllability
Compute $s(B) = \dot{B}(t) + A(t) B(t)$ (and higher iterates $s^k(B)$ if needed) and verify: $$\operatorname{rank}\big([B(t),, s(B)(t),, \dots]\big) = \dim \mathcal{N}(C(t)) = n - \operatorname{rank}(C(t)).$$
Step 5: Solve Backward Riccati ODE & Close the Loop
1. Choose $Q(t) \succeq 0$, $R(t) \succ 0$, and terminal cost $P_f \succeq 0$.
2. Integrate $-\dot{P}(t) = A(t)^T P(t) + P(t) A(t) - P(t) B(t) R(t)^{-1} B(t)^T P(t) + Q(t)$ backward from $t_f$ to $t_0$ with $P(t_f) = P_f$.
3. At runtime, compute $s(t)$ on the manifold (using $\log_{SO(3)}(R_d^T R)$ for $SO(3)$ and $\log_{S^2}(q_d, q)$ plus constraint-projected $\delta\omega$ for $S^2$), and apply $u(t) = u_d(t) - R(t)^{-1} B(t)^T P(t) s(t)$.
4. Integrate the nonlinear plant with a geometric Lie-group integrator (e.g., $\exp(\Delta t,\Omega^\wedge)$ on $SO(3)$) to preserve manifold invariants to machine precision.