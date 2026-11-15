________________


name: ch02-manifold-kinematics description: Select the right configuration manifold and implement verified Lie group operators (hat, vee, exp, log, Adjoint) for robotic systems on S1/SO(2), S2, SO(3), S3, SE(3), and product spaces.
Chapter 2 Manifold Selection & Lie Group Operators Skill
Use this skill when modeling kinematics, dynamics, state estimation, or geometric control for a robotic system evolving on non-Euclidean manifolds or product spaces.
Step 1: Select the Configuration Manifold & Compute DoF / DoUA
1. Identify each physical degree of freedom:
   * Planar orientation / revolute joint / gait phase: $S^1 \cong \mathrm{SO}(2)$ ($\dim = 1$).
   * Unit direction / spherical pendulum / taut cable link: $S^2 := {q \in \mathbb{R}^3 \mid q \cdot q = 1}$ ($\dim = 2$, not a Lie group; $\dot{q} = \omega \times q$ with $\omega \cdot q = 0$).
   * 3D rigid-body attitude: $\mathrm{SO}(3)$ ($\dim = 3$, $\dot{R} = R \Omega^\wedge$). Prefer $\mathrm{SO}(3)$ over Euler angles (gimbal lock) and $S^3$ unit quaternions (double-cover unwinding).
   * 3D rigid-body pose: $\mathrm{SE}(3) = \mathrm{SO}(3) \ltimes \mathbb{R}^3$ ($\dim = 6$).
   * Planar rigid-body pose: $\mathrm{SE}(2) = \mathrm{SO}(2) \ltimes \mathbb{R}^2$ ($\dim = 3$, e.g., humanoid ground-attached heading frame ${H_0}$).
2. Form the product manifold $Q = \prod_{i=1}^k \mathcal{M}_i$:
   * Quadrotor + taut cable load: $\mathrm{SE}(3) \times S^2$ ($\mathrm{DoF} = 8$, $m = 4$, $\mathrm{DoUA} = 4$; switches to $\mathrm{SE}(3) \times \mathbb{R}^3$ if slack).
   * Quadrotor + $n$-link flexible cable: $\mathrm{SO}(3) \times \mathbb{R}^3 \times (S^2)^n$ ($\mathrm{DoF} = 6 + 2n$, $m = 4$, $\mathrm{DoUA} = 2n + 2$).
   * $p$ quadrotors + shared rigid-body load + flexible cables: $\mathrm{SE}(3) \times \prod_{j=1}^p \mathrm{SO}(3) \times (S^2)^{n_j}$ ($\mathrm{DoF} = 6 + 3p + 2\sum n_j$, $m = 4p$, $\mathrm{DoUA} = 6 - p + 2\sum n_j$).
   * Planar $n$-link snake / mobile base: $\mathrm{SE}(2) \times (S^1)^n$ ($\mathrm{DoF} = 3 + n$).
Step 2: Implement Verified Lie Group Operators
Enforce these exact conventions:


* $\mathfrak{so}(3)$ Hat/Vee: $x^\wedge y = x \times y$, $(x^\wedge)^\vee = x$.
* Rodrigues' $\exp$ on $\mathrm{SO}(3)$: Use Taylor series ($\frac{\sin\theta}{\theta} \approx 1 - \theta^2/6$, $\frac{1-\cos\theta}{\theta^2} \approx 1/2 - \theta^2/24$) when $\theta < 10^{-8}$.
* $\log$ on $\mathrm{SO}(3)$: Clip $\frac{\operatorname{tr}(R)-1}{2}$ to $[-1, 1]$ before $\arccos$. Handle $\theta < 10^{-8}$ via Taylor expansion and $\pi - \theta < 10^{-6}$ by extracting the largest-norm column of $R + I_3$.
* $\mathrm{SE}(3)$ Twist & $\exp/\log$: Use $\xi = [v^\top, \omega^\top]^\top \in \mathbb{R}^6$, $\xi^\wedge = \begin{bmatrix} \omega^\wedge & v \ 0 & 0 \end{bmatrix}$, and the left Jacobian $V(\omega)$ / $V(\omega)^{-1}$.
* Adjoint $\mathrm{Ad}_g$: $\mathrm{Ad}_g = \begin{bmatrix} R & p^\wedge R \ 0 & R \end{bmatrix} \in \mathbb{R}^{6\times 6}$ for $g = (R, p) \in \mathrm{SE}(3)$, satisfying $(g \xi^\wedge g^{-1})^\vee = \mathrm{Ad}_g \xi$.
Step 3: Mandatory Verification Checks
Before deploying any kinematic or control code, assert:


1. vee(hat(x)) == x and hat(x) @ y == np.cross(x, y).
2. Matrix identities: $\operatorname{tr}(x^\wedge A) = -x^\top (A - A^\top)^\vee$, $R x^\wedge R^\top = (Rx)^\wedge$, and $x^\wedge A + A^\top x^\wedge = ((\operatorname{tr}(A)I_3 - A)x)^\wedge$.
3. exp(log(R)) == R across $\theta \to 0$, general $\theta$, and $\theta = \pi$.
4. exp_se3(log_se3(g)) == g and vee_se3(g @ hat_se3(xi) @ inv(g)) == Ad_g @ xi.
5. Manifold invariants during integration: $|q|_2 = 1$ and $\omega \cdot q = 0$ on $S^2$; $R^\top R = I_3$ and $\det(R) = +1$ on $\mathrm{SO}(3)$.