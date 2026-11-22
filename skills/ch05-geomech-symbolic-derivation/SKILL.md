---
name: ch05-geomech-symbolic-derivation
description: Derive, verify (check_eom < 1e-5), linearize, generate NumPy/CasADi code, and simulate coordinate-free multi-body dynamics on R, R^3, S^1, S^2, SO(3), and SE(3) using the geomech Python package.
---

# Chapter 5 Skill: Symbolic Manifold Dynamics with `geomech`

Use this skill to automate coordinate-free Lagrangian and Lagrange-d'Alembert derivations on smooth manifolds using [`geomech`](https://github.com/vkotaru/geomech) while preserving the textbook's Master Notation Standard (`R_dot = R Omega^`, `q_dot = omega x q`, `omega . q = 0`).

## 1. Setup & Variable Declarations

- Constants must carry `attr=["Constant"]` (case-sensitive); inertia tensors also need `"SymmetricMatrix"`.
- Configuration types: `Scalar("s")`, `Vector("x")`, `S1("q", axis=e)`, `S2("q")`, `SO3("R")`.
- Tangent & variation vectors:
  - `S2`: `omega = q.get_tangent_vector()`, `xi = q.get_variation_vector()` (`delta q = xi x q`).
  - `SO3`: `Om = R.get_tangent_vector()`, `eta = R.get_variation_vector()` (`delta R = R eta^`).
  - `S1`: scalar `omega` and `gamma` about constant unit `e = Vector("e", attr=["Constant", "UnitNorm"])`.
- Register all configuration variables in `SystemVariables(scalars=[...], vectors=[...], matrices=[...])`.

## 2. Derivation, Verification, & Standard Form

```python
eom = compute_eom(L, dW, variables, trace=EOMTrace())
err = check_eom(L, dW, variables, eom, inputs=inputs)
assert err < 1e-5, f"Verification failed: {err}"
sf = to_standard_form(eom, variables, inputs)   # M a + f + G u = 0 (M = -inertia)
lin = linearize(eom, variables, inputs)         # M dx_dot = A dx + B du
```

- **Sign convention:** `geomech` returns `E = -(Euler-Lagrange) + forcing = 0`, so `M` in `to_standard_form` is `-I_physical`.
- **Mandatory check:** Always run `check_eom` and confirm residual `< 1e-5` (typically `~1e-7` to `1e-8`). Never hand-edit derived equations.

## 3. Codegen, Simulation, & Indexed `n`-Body Chains

- **Codegen & RKMK4:** Build `layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))`, compile with `load_module(module_source(sf, layout, lagrangian=L), "name")`, and integrate with `simulate(plant, layout.pack(...), t_final, p)` (verify unforced energy drift).
- **Symbolic `n`-body models:** Use `geomech.indexed` (`Family`, `Sum`, `IndexedVariables`, `compute_indexed_eom`) when `n` is symbolic, then `.instantiate(k)` and verify `check_eom` for sample sizes.
