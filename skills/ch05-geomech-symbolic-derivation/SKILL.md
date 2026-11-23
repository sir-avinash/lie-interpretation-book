---
name: ch05-geomech-symbolic-derivation
description: Derive, verify and simulate coordinate-free equations of motion with the geomech Python package, for systems whose configuration is a product of R, R^3, S^1, S^2 and SO(3) (rigid bodies, pendulums, cable-suspended loads, chains of n links). Use when asked for the equations of motion of such a system, to check a hand-derived equation against its Lagrangian, or to turn a Lagrangian into a simulation.
---

# Symbolic derivation on manifolds with geomech

Procedure for Chapter 5. Tested with `geomech` 1.1.0. For the full API, the catalog of ready-made systems and every error message, read the library's own skill (`skills/geomech/SKILL.md` and `reference.md` in https://github.com/vkotaru/geomech).

## When to use

- The configuration is a product of $\R$, $\R^3$, $S^1$, $S^2$, $\SO(3)$. $\SE(3)$ is $\R^3 \times \SO(3)$ with the inertial velocity $\dot x$.
- The Lagrangian depends on configurations and velocities only.
- Constraints are already built into the configuration space (a taut cable is a point of $S^2$).

Do not use it for rolling constraints, closed kinematic chains, contact or impacts. Do not derive by hand and do not introduce Euler angles.

## Notation (same as the book)

| Declared as | Velocity | Variation | Variation of the velocity |
| :-- | :-- | :-- | :-- |
| `Scalar("s")`, `Vector("x")` | $\dot s$, $\dot x$ (`x.t_diff()`) | $\delta s$, $\delta x$ (`x.delta()`) | $\tfrac{d}{dt}\delta x$ |
| `S1("q", axis=e)` | $\dot q = \omega\, e \times q$, scalar $\omega$ | $\delta q = \gamma\, e \times q$ | $\delta\omega = \dot\gamma$ |
| `S2("q")` | $\dot q = \omega \times q$ | $\delta q = \xi \times q$ | $\delta\omega = \dot\xi - 2\,\omega\times\xi$ |
| `SO3("R")` | $\dot R = R\,\Omega^\wedge$ | $\delta R = R\,\eta^\wedge$ | $\delta\Omega = \dot\eta + \Omega\times\eta$ |

`q.get_tangent_vector()` is $\omega$ (or $\Omega$). `q.get_variation_vector()` is $\xi$ (or $\gamma$, $\eta$).

## Steps

1. **Install and record the version.** `pip install geomech`, then report `geomech.__version__`.
2. **Declare symbols.** Constants need `attr=["Constant"]` (case-sensitive). Inertias need `attr=["Constant", "SymmetricMatrix"]`. An $S^1$ axis needs `attr=["Constant", "UnitNorm"]`. One half is `Scalar("0.5", value=0.5, attr=["Constant"])`.
3. **Write $L$ and $\delta W$.** $L$ is kinetic minus potential energy, with gravity along $-e_3$. A force $F$ at a point contributes `Dot(delta_of_point, F)`, where the variation of the point is built term by term: $x \to$ `x.delta()`, $q \to$ `Cross(xi, q)`, $R\rho \to$ `R * Cross(eta, rho)`. A body torque $\tau$ contributes `Dot(eta, tau)`.
4. **List the configuration variables** in `SystemVariables(scalars=[...], vectors=[...], matrices=[...])`. $S^1$ and $S^2$ points go in `vectors`, $\SO(3)$ in `matrices`. Never list $\omega$ or $\Omega$.
5. **Derive.** `eom = compute_eom(L, dW, variables)`. The result is `{name: (variation, E)}`. Pass `trace=EOMTrace()` and call `trace.summary()` to see every stage.
6. **Verify.** `err = check_eom(L, dW, variables, eom, inputs=inputs)`. Report the number.
7. **Standard form.** `sf = to_standard_form(eom, variables, inputs)` gives $M(q)\,a + b(q,v) + G(q)\,u = 0$ per equation.
8. **Generate code and simulate.**

```python
from geomech.codegen import StateLayout, collect_params, load_module, module_source
from geomech.simulation import simulate

layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
plant = load_module(module_source(sf, layout, lagrangian=L), "my_system")
x0 = layout.pack(q=q0, omega_q=w0)          # slot names: layout.slot_names()
free = simulate(plant, x0, t_final, p)      # no inputs
traj = simulate(plant, x0, t_final, p, u=u) # constant array or a function
```

9. **For $n$ bodies with $n$ symbolic**, use `geomech.indexed` (`Family`, `Sum`, `IndexedVariables`, `compute_indexed_eom`). Check with `check_eom(L.instantiate(k), None, fam.members(k), eom.instantiate(k))` for several `k` (pass the instantiated virtual work in place of `None` if there are inputs) and say which.
10. **For linearization** call `linearize(eom, variables, inputs)` and follow the Chapter 7 skill.

## Checks that must pass

- `check_eom` below `1e-5`. A correct equation scores about `1e-7`. The planted sign error of the chapter scores between 0.3 and 2.8.
- Unforced simulation: relative energy drift `np.max(np.abs(E - E[0])) / abs(E[0])` below `1e-6` at the default step of 1 ms, falling by a factor of 16 when the step is halved, and `free.max_drift()` below `1e-10`.
- Forced simulation: energy is not conserved. Report the unforced run as the integrator check.
- If a hand-derived or published equation exists, compare accelerations numerically at random states (`geomech.simulation.accelerations`). Do not compare printed forms.

## Conventions to state with every result

- Each equation is $E = 0$ with $E = -(\text{Euler–Lagrange expression}) + \text{forcing}$. A point mass gives $-m\ddot x + F = 0$.
- In the standard form, $M$ is minus the physical inertia.
- An $S^2$ equation is $(I - qq^\top)E = 0$ together with $q\cdot\dot\omega = 0$. Its component along $q$ carries no information. Never solve it as a $3\times3$ system. `simulate` projects for you.

## Failure modes

| Symptom | Cause | Action |
| :-- | :-- | :-- |
| `ValueError: compute_eom: δS still contains the variation symbols '\delta{e3}' ...` | a symbol is neither constant nor listed | add `attr=["Constant"]`, or list it in `SystemVariables`, or pass `params=[...]` for a prescribed time-varying symbol |
| `ValueError: attr: unknown attribute 'constant' ...` | attribute typo | use the exact names in the message |
| `TypeError: ... is a tangent or variation vector, not a configuration variable` | $\omega$ or $\Omega$ listed as a variable | list the point `q` or `R` |
| `AlgebraicError: ... the Lagrangian depends on accelerations ...` | $L$ contains a second derivative | rewrite $L$ in velocities |
| `ValueError: to_standard_form: input 'g' is constant ...` | a constant passed as an input | inputs are non-constant symbols |
| `ValueError: x0: the S2 slot 'q' must have unit norm ...` | initial state off the manifold | normalize $q$, make $\omega \perp q$ |
| equation contains $J$ and $J^\top$ separately | `SymmetricMatrix` missing | declare it |
| extra terms that a paper does not have | terms along $q$ in an $S^2$ row, or an unsimplified but equal form | compare by value |
| `check_eom` above `1e-5` | the equation does not match $L$ | run `assert_eom` to find the row. Do not edit the equation by hand. Report the error text |
| `check_eom` passes but the model is wrong | $L$ or $\delta W$ does not describe the system | the check cannot see this. Review the energies and the points where forces act |

Quote error messages in full. Do not work around them.
