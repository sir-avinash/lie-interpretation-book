________________


name: ch03-geometric-error-design description: Guides engineers in constructing compatible configuration errors, exterior differentials, transport maps, and velocity errors on Lie groups and product manifolds (R^3, S^2, SO(3), SE(3) x S^2).
Geometric Error & Transport Map Design Skill
Use this skill when designing geometric tracking controllers, Control Lyapunov Functions (CLFs), Control Barrier Functions (CBFs), or variation-based linearizations on curved configuration manifolds ($Q = \mathbb{R}^n$, $\mathbb{S}^2$, $SO(3)$, $SE(3)$, and product manifolds).
Step 1: Decompose the Configuration Manifold
Identify the product structure $Q = Q_1 \times Q_2 \times \cdots \times Q_k$ (e.g., $\mathbb{R}^3 \times SO(3) \times \mathbb{S}^2$ for an aerial vehicle with a suspended load). Because $T_q Q = \bigoplus_i T_{q_i} Q_i$, design $\Psi_i$, $T_i$, and $e_{\dot{q}_i}$ independently on each factor and sum $\Psi = \sum_i w_i \Psi_i$.
Step 2: Select a Quadratic Configuration Error $\Psi_i(q_i, q_{i,d})$
* $\mathbb{R}^n$: $\Psi_x = \frac{1}{2}|x - x_d|^2$.
* $\mathbb{S}^2$: $\Psi_q = 1 - q \cdot q_d$ ($q, q_d \in \mathbb{S}^2$).
* $SO(3)$: $\Psi_R = \frac{1}{2}\operatorname{Tr}(I_3 - R_d^\top R)$ ($R, R_d \in SO(3)$).


Verify $\Psi_i(q_i, q_{i,d}) \ge 0$, $\Psi_i = 0 \iff q_i = q_{i,d}$, and $d_1 \Psi_i(q_{i,d}, q_{i,d}) = 0$.
Step 3: Compute Exterior Differentials $d_1 \Psi_i$ and $d_2 \Psi_i$
* $\mathbb{S}^2$: $d_1 \Psi_q = q^\wedge{}^2 q_d \in T_q \mathbb{S}^2$, $d_2 \Psi_q = (q_d^\wedge)^2 q = e_q \in T_{q_d} \mathbb{S}^2$.
* $SO(3)$ (body-frame trivialization): $d_1 \Psi_R = e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee$, $d_2 \Psi_R = -e_R$.
Step 4: Construct the Compatible Transport Map $T_i(q_i, q_{i,d})$
Enforce the compatibility condition: $$d_2 \Psi_i(q_i, q_{i,d}) = -T_i^*(q_i, q_{i,d}), d_1 \Psi_i(q_i, q_{i,d}).$$


* $\mathbb{S}^2$: $T(q, q_d)\dot{q}_d = (q_d \times \dot{q}d) \times q$, giving $e{\dot{q}} = \dot{q} - (q_d \times \dot{q}d) \times q$ and $e\omega = \omega - (-q^\wedge{}^2)\omega_d$.
* $SO(3)$: $T(R, R_d)\Omega_d = R^\top R_d \Omega_d$, giving $e_\Omega = \Omega - R^\top R_d \Omega_d$.
Step 5: Verify Energy-Dissipation & Variation Equivalence
1. Confirm $\dot{\Psi}i = \langle d_1 \Psi_i, e{\dot{q}_i} \rangle$ (e.g., $\dot{\Psi}R = e_R \cdot e\Omega$, $\dot{e}R = C(R_d^\top R) e\Omega$ with $C(A) = \frac{1}{2}(\operatorname{Tr}(A)I_3 - A^\top)$).
2. For LTV linearization along a reference trajectory, verify first-order equivalence: $e_R^\epsilon/\epsilon \to \eta$, $e_\Omega^\epsilon/\epsilon \to \dot{\eta} = \delta\Omega - \Omega_d^\wedge \eta$ on $SO(3)$, and $e_q^\epsilon/\epsilon \to -\delta q = -(\xi \times q_d)$ on $\mathbb{S}^2$.