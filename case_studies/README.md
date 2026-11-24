# Case studies

This folder holds the robots of *A Manifold Introduction to Robotics and Control* in one place.

- `robots/` is the **robot model library**: one Python module per robot, each the single source of truth for that model. Every model is written in the book's notation (Appendix A), cites the labels of the equations it implements, and is checked against a second, symbolic derivation.
- `chapters/ch16_case_studies.md` ("How it all comes together") is the chapter that reads the library: it shows that every robot is the same few factors assembled the same way, and maps each robot to the place where the book treats it.
- `verify/robots.py` checks the whole library. `figures/src/ch16.py` draws the chapter's figures from it.

## The interface

Every robot has the same small interface, defined in `robots/core.py`.

```python
import sys; sys.path.insert(0, "case_studies")
import numpy as np
import robots

robot = robots.get("quadrotor_load")       # robots.list_robots() names them all
robot.manifold                             # "R^3 x S^2 x SO(3)"
robot.factors                              # [Factor("R", "x_L", "v_L"), Factor("S2", "q", "omega"), ...]
robot.dim                                  # 8
robot.params                               # dataclass: m_Q, m_L, l, J, g  (book symbols, SI units)
robot.labels                               # {"equations": "eq4-load", "manifold": "tbl2-robots", ...}

state = robot.State()                      # dataclass: x_L, v_L, q, omega, R, Omega  (hover)
u = robot.hover_input()                    # (f, tau)
acc = robot.dynamics(state, u)             # (v_L_dot, omega_dot, Omega_dot), one per factor
state = robot.step(state, u, 1e-3)         # one step; stays on the manifold to rounding error
robot.energy(state)                        # kinetic plus potential energy
robot.on_manifold(state)                   # largest violation of |q| = 1, q . omega = 0, R'R = I
L, dW, variables, inputs = robot.lagrangian()   # the geomech model it was checked against
```

| Member | What it is |
| :-- | :-- |
| `manifold`, `factors`, `dim`, `stored` | The configuration manifold as a string and as a list of factors (`"R"`, `"S2"`, `"SO3"`), its dimension, and how many numbers store a configuration |
| `Params`, `params` | A dataclass of parameters with the book's symbols and SI defaults taken from the chapters' worked examples |
| `State` | A dataclass with the code names of Appendix A: `x`, `v`, `R`, `Omega`, `q`, `omega`, `x_L`, `v_L` |
| `dynamics(state, u, params)` | The accelerations, one per factor, from equations written with the inertia on the left as in the chapters |
| `step(state, u, h, params)` | A fourth-order step with the exponential update `R exp(hat(theta))`, `exp(hat(phi)) q` (eq4-lie-euler). The input is held over the step |
| `rollout(state, u, h, n)` | `n` steps. `u` is a constant input or a function `u(t, state)` |
| `energy(state, params)` | Kinetic plus potential energy |
| `on_manifold(state)` | Residual of the manifold constraints |
| `random_state(rng)` | A random state on the manifold |
| `lagrangian()` | The geomech `(L, dW, variables, inputs)` used to derive the model a second time |
| `labels` | The book labels that define the model |
| `tracking_controller(...)`, `linearize(...)` | Where Chapters 6 and 7 have them, with the label each implements |

`params` can be left out everywhere: the robot then uses `robot.params`. An input is a tuple in the book's order, `u = (f, tau)`. For several vehicles `f` has shape `(n,)` and `tau` shape `(n, 3)`, and their states are stacked the same way: `q` is `(n, 3)`, `R` is `(n, 3, 3)`.

Only NumPy is needed to run a model. `geomech` is imported inside `lagrangian()` and by the verification script.

## The robots

| `robots.get(...)` | Robot | Configuration manifold | Dimension | Defined in the book | Derived in | Status |
| :-- | :-- | :-- | :-: | :-- | :-- | :-- |
| `spherical_pendulum` | Spherical pendulum | S^2 | 2 | eq4-sph, eq4-tension | sec4-sphere, eq5-pendulum | verified |
| `rigid_body` | Free rigid body | SO(3) | 3 | eq4-euler-free | sec4-so3 | verified |
| `pendulum_3d` | 3D pendulum | SO(3) | 3 | eq4-pendulum | sec4-forces | verified |
| `quadrotor` | Quadrotor | R^3 x SO(3) | 6 | eq4-quadrotor | sec4-products | verified |
| `quadrotor_load` | Quadrotor with a cable-suspended load | R^3 x S^2 x SO(3) | 8 | eq4-load, eq4-load-tension | sec4-products, eq5-wu | verified |
| `multi_quadrotor_load`, `n=` | n quadrotors, point-mass load | R^3 x (S^2 x SO(3))^n | 3 + 5n | eq4-multi | sec4-products | verified, n = 1, 2, 3 |
| `multi_quadrotor_payload`, `n=` | n quadrotors, rigid payload | SE(3) x (S^2 x SO(3))^n | 6 + 5n | tbl2-robots (stated in words) | sec5-scaling, Wu (2017, ch. 3) | verified, n = 2, 3 |
| `flexible_cable`, `n=` | Chain of n links hanging from a pivot | (S^2)^n | 2n | eq5-chain | sec5-scaling | verified, n = 1 to 5 |
| `quadrotor_cable`, `n=`, `attach=` | Cable of n links carried by quadrotors at chosen joints | R^3 x (S^2)^n x SO(3)^nQ | 3 + 2n + 3 nQ | sec16-teams (stated in words) | Kotaru (2022, ch. 7) | verified for five layouts |
| `telescoping_leg` | Telescoping stance leg | R x SO(3) | 4 | eq4-leg | ex4-leg | verified |
| `reaction_mass_biped` | Reaction Mass Biped, stance phase | R x SO(3)^3 x R^3 | 13 | eq4-leg for the leg, tbl2-robots | Siravuru (2019, ch. 5) | verified (stance dynamics only) |
| `cassie` | Cassie, reduced geometric model | SE(3) x SO(3)^2 x R^2 | 14 | tbl2-robots | Siravuru (2019, ch. 6) | floating base verified; no ground contact |

"Verified" means all of the following, in `verify/robots.py`:

1. geomech derives the equations of motion from the robot's `lagrangian()` and `check_eom` is below `1e-5`;
2. the hand-written `dynamics` equals the accelerations of the geomech model at random states, to `1e-9` or better;
3. `dynamics`, `tracking_controller` and `linearize` equal the functions that the chapters verified (taken from `verify/ch04.py` to `verify/ch08.py`) at the chapters' parameter values;
4. a free simulation with `step` stays on the manifold to rounding error, conserves energy to fourth order in the step, and conserves the momentum that the model conserves;
5. sized models pass for several sizes, and collapse to the smaller models when they should (one quadrotor, one link, attachment at the center of mass).

What is not included, by robot:

- `reaction_mass_biped`: the swing-leg length (it does not enter the stance dynamics), the floating-base model, the impact map and the walking controller of Siravuru (2019). The table of Chapter 2 counts the swing-leg length, hence its dimension 14 against 13 here.
- `cassie`: the ground-contact constraints and the Hovershoes. Parameter values are placeholders.
- `quadrotor_load`, `multi_quadrotor_*`, `quadrotor_cable`: every cable is assumed taut. The equations stay smooth when a tension goes negative. Check `tension(...)`.

## How to run

```bash
pip install numpy scipy geomech        # geomech only for the verification
python verify/robots.py                # every check of the library
python figures/src/ch16.py             # the figures of the hub chapter
```

## How to add a robot

1. **Manifold.** Write the configuration as a product of R^n, S^2 and SO(3) factors (skill `ch02-manifold-kinematics`). Choose the fields of the state with the code names of Appendix A.
2. **Model.** Derive the equations of motion with the Lagrange-d'Alembert principle (skill `ch04-manifold-dynamics-modeling`) and write them with the inertia on the left.
3. **Module.** Copy `robots/quadrotor_load.py`. Fill in `Params`, `State`, `factors`, `labels`, `dynamics`, `energy`, `zero_input` and `lagrangian`. Do not write a new integrator: `step` comes from `Robot`.
4. **Register** the class in `robots/__init__.py`.
5. **Verify.** Add a block to `verify/robots.py`: `against_geomech(robot, params)` and `conservation(robot)` are two lines (skill `ch05-geomech-symbolic-derivation`). Add a comparison with a smaller model that the new one must collapse to.
6. **Control.** Add `tracking_controller` (skill `ch06-geometric-tracking-control`) and `linearize` (skill `ch07-variation-lqr-design`) when you have them, each with the label of the law it implements.
7. **Map.** Add a row to the table above and to the map in the hub chapter.
