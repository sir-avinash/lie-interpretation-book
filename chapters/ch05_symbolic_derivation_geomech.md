# Chapter 5: Symbolic Derivation on Manifolds with `geomech` & Modular Multi-Body Formulations

**Chapter Roadmap.** Deriving coordinate-free Lagrange–d'Alembert equations by hand for multi-body systems on $SE(3) \times (S^2)^n$ or $(SO(3))^n$ is tedious and error-prone: a single missed cross-product sign in $\delta\omega$ or integration by parts invalidates the entire dynamic model. This chapter presents automated symbolic derivation directly on smooth manifolds using the open-source [`geomech`](https://github.com/vkotaru/geomech) library (Kotaru, 2026), which implements the variational framework of Lee, Leok, & McClamroch (2018) and extends Bittner & Sreenath (2020): (1) frozen symbolic expression trees and manifold leaf types ($\mathbb{R}, \mathbb{R}^3, S^1, S^2, SO(3)$), (2) the 6-stage `compute_eom` rewrite pipeline, (3) standard-form extraction, variation-based linearization, numerical `check_eom` verification, and Lie-group RK4 (`RKMK4`) simulation, and (4) symbolic $n$-body chains via `geomech.indexed`.

---

## 5.1 Symbolic Expression Trees & Manifold Leaf Types

Unlike coordinate-based symbolic engines (such as SymPy or CasADi scalar graphs) that parameterize rotations with Euler angles or quaternions, `geomech` represents physical quantities as **typed, frozen expression trees** (`Expr`) whose leaves live directly on smooth manifolds.

| Type | Space | Tangent Vector | Variation Vector | Kinematic Relation |
| :--- | :--- | :--- | :--- | :--- |
| `Scalar("s")` | $\mathbb{R}$ | $\dot{s} \in \mathbb{R}$ | $\delta s \in \mathbb{R}$ | $\delta\dot{s} = \frac{d}{dt}(\delta s)$ |
| `Vector("x")` | $\mathbb{R}^3$ | $\dot{x} \in \mathbb{R}^3$ | $\delta x \in \mathbb{R}^3$ | $\delta\dot{x} = \frac{d}{dt}(\delta x)$ |
| `S1("q", axis=e)` | $S^1 \perp e$ | $\omega \in \mathbb{R}$ (scalar) | $\gamma \in \mathbb{R}$ (scalar) | $\dot{q} = \omega (e \times q),\ \delta q = \gamma (e \times q)$ |
| `S2("q")` | $S^2 \subset \mathbb{R}^3$ | $\omega \in T_q S^2$ ($\omega \cdot q = 0$) | $\xi \in T_q S^2$ ($\xi \cdot q = 0$) | $\dot{q} = \omega \times q,\ \delta q = \xi \times q$ |
| `SO3("R")` | $SO(3)$ | $\Omega \in \mathbb{R}^3$ (body) | $\eta \in \mathbb{R}^3$ (body) | $\dot{R} = R\Omega^\wedge,\ \delta R = R\eta^\wedge$ |

Table 5.1: Canonical `geomech` manifold leaf types, tangent vectors, and variation vectors matching the Master Notation Standard.

**Def. 5.1 (Frozen Symbolic Expression Tree).** An expression in `geomech` is an immutable directed tree built from:
1. **N-ary associative nodes:** `Add` (scalars), `VAdd` (vectors), `MAdd` (matrices).
2. **Binary algebraic nodes:** `Mul`, `Dot`, `Cross`, `SVMul`, `MVMul`, `MMMul`, `SMMul`, `VVMul`.
3. **Unary geometric and calculus nodes:** `Hat` ($(\cdot)^\wedge$), `Vee` ($(\cdot)^\vee$), `Transpose` ($(\cdot)^T$), `TimeDerivative` ($\frac{d}{dt}$), and `Variation` ($\delta$).

Because every node is frozen, rewrite passes never mutate trees in place; every transformation returns a new tree and optionally logs the fired rule name (`GEOMECH_TRACE=1`) and stage snapshot (`EOMTrace`).

---

## 5.2 The 6-Stage `compute_eom` Pipeline

Given a first-order Lagrangian $L(q, \dot{q}) = T - V$ and virtual work $\delta W$, calling `compute_eom(L, dW, variables)` executes six pure tree-rewrite stages in sequence:

$$\delta S = \delta L + \delta W \xrightarrow{\text{manifold\_rules}} \text{kinematic expansion} \xrightarrow{\text{full\_simplify}} \text{IBP} \xrightarrow{\text{expand}} \text{extract\_linear\_coeff} \xrightarrow{\text{full\_simplify}} E = 0. \tag{5.1}$$

### Stage 1 & 2: Variations and Manifold Rules

In Stage 1 (`dL`, `dW`, `dS`), `L.delta()` applies the product rule recursively. Manifold leaves expand eagerly into their variation vectors ($\delta x$, $\gamma(e \times q)$, $\xi \times q$, $R\eta^\wedge$), while tangent-vector variations ($\delta\omega$, $\delta\Omega$) remain unevaluated until Stage 2 (`apply_manifold_rules`), which substitutes the exact Lie-algebraic variation identities from Chapter 3:

$$
\begin{aligned}
S^1: \quad \delta\omega &= \dot{\gamma}, \\
S^2: \quad \delta\omega &= \dot{\xi} - 2\,\omega \times \xi \quad (\text{on } \xi \cdot q = 0), \\
SO(3): \quad \delta\Omega &= \dot{\eta} + \Omega \times \eta.
\end{aligned} \tag{5.2}
$$

**Remark 5.1 (Factor of 2 on $S^2$).** In Eq. (5.2), $\delta\omega = \dot{\xi} - 2\,\omega \times \xi$ arises because $\omega \in T_q S^2$ is constrained to remain orthogonal to $q(t)$ along the varied curve ($\omega_\epsilon \cdot q_\epsilon = 0$), whereas $\delta R = R\eta^\wedge$ on $SO(3)$ has no normal projection constraint and yields $\delta\Omega = \dot{\eta} + \Omega \times \eta$.

### Stage 3–6: Simplification, Integration by Parts, & Extraction

1. **Simplification (`full_simplify`):** Alternates `expand`, `pull` (hoisting scalar prefactors out of `Dot`/`MVMul`), `vector_rules` (applying $q \cdot q = 1$, $\omega \cdot q = 0$, $a \cdot (a \times b) = 0$, and $S^1$ axis identities), and `_eliminate` (skew-symmetry, $R^T R = I_3$, like-term merging) until a fixed point is reached.
2. **Integration by Parts (`integrate_by_parts`):** For each time-differentiated variation target ($\dot{\delta x}$, $\dot{\gamma}$, $\dot{\xi}$, $\dot{\eta}$), `collect(expr, target)` uses cyclic triple-product rotations $(a \times b) \cdot c = a \cdot (b \times c)$, matrix peeling $(Mf)\cdot w = f \cdot (M^T w)$, and rotation stripping $(Ra)\cdot(Rb) = a \cdot b$ to isolate the target as the left operand of `Dot(target, f)`. Then Hamilton's principle drops boundary terms via
   $$a\,(\dot{v} \cdot f) \longrightarrow -a\,(v \cdot \dot{f}) - \dot{a}\,(v \cdot f), \tag{5.3}$$
   where the $\dot{a}$ term is included automatically whenever $a$ is non-constant.
3. **Linear Coefficient Extraction (`extract_linear_coeff`):** After distributing products over sums (`expand`), the coefficient $E$ of each variation vector $v \in \{\delta x, \gamma, \xi, \eta\}$ is extracted so that $\delta S = \int \sum_v v \cdot E_v \, dt = 0$, yielding the coordinate-free equations of motion $E_v = 0$.

**Prop. 5.1 (Sign and Tangent-Plane Conventions in `geomech`).** For every configuration variable in `SystemVariables`, `compute_eom` returns $E_v = 0$ with sign convention
$$E_v = -\text{(Euler–Lagrange operator)} + \text{generalized forcing}. \tag{5.4}$$
For $q \in S^2$, Eq. (5.4) holds projected onto the tangent plane $T_q S^2$, i.e., $(I_3 - q q^T) E_\xi = 0$.

---

## 5.3 Standard Form, Variation-Based Linearization, & Code Generation

Once `eom = compute_eom(L, dW, variables)` is obtained, `geomech` automates the entire downstream modeling and control pipeline:

1. **Standard Manipulator Form (`to_standard_form`):** Partitions each equation into acceleration, forcing, and input terms:
   $$M(q)\,\dot{v} + f(q, v) + G(q)\,u = 0, \tag{5.5}$$
   where $M = -(\text{physical inertia})$ under Eq. (5.4) (e.g., $M[\ddot{x}] = -m I_3$, $M[\dot{\Omega}] = -J$).
2. **Variation-Based Linearization (`linearize`):** Takes the symbolic variation $\delta E$ of Eq. (5.4) directly on the manifold (the automated counterpart to Chapter 7), producing the linear time-varying (LTV) tangent-space system
   $$M(t)\,\delta\dot{x} = A(t)\,\delta x + B(t)\,\delta u, \tag{5.6}$$
   which feeds directly into `geomech.simulation.lqr` using the manifold error $\eta = (\log(R_{\text{eq}}^T R))^\vee$ on $SO(3)$.
3. **Mandatory Numerical Verification (`check_eom`):** Evaluates the symbolic equations $E_v$ against independent central finite differences of the Euler–Lagrange action at randomized manifold states $(q, v, \dot{v}, u)$. A valid derivation achieves residual $< 10^{-7}$ (any residual $> 10^{-5}$ indicates a modeling error).
4. **Codegen & Lie-Group Simulation (`geomech.codegen`, `geomech.simulation`):** `StateLayout` packs configuration and velocity states into a flat vector; `module_source` emits standalone NumPy or CasADi code; and `simulate(..., method="RKMK4")` integrates with a Runge–Kutta–Munthe–Kaas Lie-group integrator that preserves $R^T R = I_3$ and $\|q\| = 1$ to machine precision.

---

## 5.4 Symbolic $n$-Body Chains (`geomech.indexed`)

For multi-link chains, flexible cables, and multi-quadrotor teams with an arbitrary number of bodies $n$, `geomech.indexed` keeps both the body index $i$ and total count $n$ symbolic using `Family` and `Sum(j, lower, upper, body)`.

**Example 5.1 ($n$-Link Spherical Pendulum Chain).** Let $q_1, \dots, q_n \in S^2$ be unit link directions with lengths $l_i$ and point masses $m_i$ at $x_i = \sum_{j=1}^i l_j q_j$. Calling `compute_indexed_eom(L, dW, IndexedVariables(families=[q]))` derives the generic equation of the $i$-th body for all $n$ simultaneously:
$$M_i - \sum_{j=i}^{n} m_j g l_i (q_i \times e_3) + \sum_{j=1}^{n} \sum_{k=\max(i,j)}^{n} m_k l_i l_j \Big( q_i \times (q_i \times \ddot{q}_j) \Big) = 0, \tag{5.7}$$
automatically recovering the coupled inertia block $\sum_{k=\max(i,j)}^n m_k l_i l_j$ of Lee et al. (2018, §5.3) and matching the flexible-cable chain on $SO(3) \times \mathbb{R}^3 \times (S^2)^n$ from Chapter 4. Calling `eom.instantiate(n_val)` unrolls Eq. (5.7) for any concrete $n$ so `check_eom`, `to_standard_form`, and `linearize` apply identically.

---

## 5.5 Worked Examples & Exercises

**Example 5.2 (Quadrotor with Cable-Suspended Load on $SE(3) \times S^2$).** Using $x_Q = x_L - l q$ with `xL = Vector("x_L")`, `q = S2("q")`, `R = SO3("R")`, and $\delta W = (\delta x_L - l\,\xi \times q) \cdot (f R e_3) + \eta \cdot M$, `compute_eom` recovers Eqs. (4.16)–(4.21) in under a second with `check_eom` residual $\sim 10^{-8}$. Linearizing about hover ($q_{\text{eq}} = -e_3$, $f_{\text{eq}} = (m_Q + m_L)g$) via `linearize` and `lqr` yields an exponentially stabilizing manifold controller.

### Exercises
1. **Spherical Pendulum Verification:** Derive the spherical pendulum on $S^2$ in `geomech`, inspect `EOMTrace.summary()`, and verify `check_eom < 1e-6` and unforced `RKMK4` energy drift $< 10^{-6}$.
2. **Rigid Body on $SO(3)$:** Derive Euler's rigid-body equation $-J\dot{\Omega} - \Omega \times J\Omega + M = 0$ and verify that `to_standard_form` yields $M[\dot{\Omega}] = -J$.
3. **Symbolic $n$-Link Chain:** Use `geomech.indexed.compute_indexed_eom` for the $n$-link spherical chain, instantiate for $n = 3$ and $n = 5$, and verify `check_eom < 1e-6` for both.
