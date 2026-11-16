________________


name: ch04-manifold-dynamics-modeling description: Guide for deriving coordinate-free equations of motion for robotic and multi-body systems evolving on product manifolds such as SE(3) x S^2, SO(3) x R^3 x (S^2)^n, and SE(3) x R^N_T.
Coordinate-Free Dynamics Modeling on Product Manifolds
Use this skill when deriving singularity-free equations of motion for rigid-body, aerial, cable-suspended, or floating-base legged robots evolving on non-Euclidean product manifolds.
Canonical Conventions
* Hat / Vee Maps: $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ with $x^\wedge y = x \times y$, and inverse $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$.
* SO(3) Kinematics: $\dot{R} = R \Omega^\wedge$, where $R \in SO(3)$ and $\Omega \in \mathbb{R}^3$ is the body-fixed angular velocity.
* S² Kinematics: $\dot{q} = \omega \times q = \omega^\wedge q$ with $\omega \cdot q = 0$, where $q \in S^2$ and $\omega \in \mathbb{R}^3$.
Step-by-Step Derivation Workflow
1. Specify the Product Configuration Manifold $Q$
Identify each physical subsystem and its intrinsic manifold:


* Rigid body pose: $SE(3) \cong \mathbb{R}^3 \times SO(3)$ (6 DOF).
* Unit-direction cable / spherical pendulum link: $S^2$ (2 DOF).
* Internal revolute/prismatic joints: $\mathbb{R}^{N_T}$ ($N_T$ DOF).
* Check total DOF and degrees of underactuation ($\text{DOuA} = \dim Q - \dim U$).
2. Construct the Kinetic-Energy Metric and Lagrangian
Define the Riemannian kinetic-energy metric $\llangle \dot{q}_1, \dot{q}_2 \rrangle = \langle M_q \dot{q}_1, \dot{q}_2 \rangle$, kinetic energy $T$, and potential energy $U$:


* Express constrained positions via kinematic chains (e.g., $x_Q = x_L - l q$ on $SE(3) \times S^2$, or $x_i = x_{i-1} + l_i q_i$ on $SO(3) \times \mathbb{R}^3 \times (S^2)^n$).
* Form $L = T - U$.
3. Apply Manifold Variations (Lagrange–d'Alembert Principle)
Never introduce Euler angles or spherical coordinates. Use intrinsic variations:


* On $SO(3)$: $\delta R = R \eta^\wedge$, $\delta \Omega = \Omega^\wedge \eta + \dot{\eta}$ ($\eta \in \mathbb{R}^3$).
* On $S^2$: $\delta q = \xi \times q$ with $\xi \cdot q = 0$, $\delta \dot{q} = \dot{\xi} \times q + \xi \times \dot{q}$.
* Substitute into $\delta \int_0^\tau L,dt + \int_0^\tau (\langle W_1, \hat{M}\rangle + W_2 \cdot F_{\text{ext}}),dt = 0$, integrate by parts, and collect independent variations $(\delta x, \eta, \xi)$.
4. Handle Hybrid Modes and Modularity
* Hybrid Guards: For cable-suspended systems, monitor tension $T = |m_L(\dot{v}_L + g e_3)|$ ($S_z: T = 0$) and cable extension ($S_n: |x_Q - x_L| = l, \frac{d}{dt}|x_Q - x_L| > 0$).
* Modular Floating-Base Systems: On $SE(3) \times \mathbb{R}^{N_T}$, assemble $M(\theta) = \operatorname{diag}(I_b, 0) + \sum_{i,j} M_{ij}(\theta)$ using $SE(3)$ Adjoint maps $\mathbf{Ad}{g{a,b}}$ so damaged or removed limbs can be dropped by matrix slicing without symbolic re-differentiation.
5. Verify with Geometric Integration
Integrate on the manifold using Lie-group / Rodrigues exponential updates:


* $R_{k+1} = R_k \exp(\Delta t , \Omega_k^\wedge)$, $q_{k+1} = \exp(\Delta t , \omega_k^\wedge) q_k$.
* Verify $|q(t)| = 1$, $R^T R = I$, $\omega \cdot q = 0$, and unforced total energy conservation $E(t) = E(0)$.
Primary References
* Wu & Sreenath (IEEE TAC 2016): https://hybrid-robotics.berkeley.edu/publications/TAC2016_Geometric_TimeVarying_CBF.pdf
* Sreenath, Lee, & Kumar (CDC 2013): https://hybrid-robotics.berkeley.edu/publications/CDC2013.pdf
* Kotaru, Wu, & Sreenath (arXiv:1711.04895): https://arxiv.org/pdf/1711.04895
* Fast & Modular Whole-Body Lagrangian Dynamics (arXiv:2504.16383): https://arxiv.org/html/2504.16383v1