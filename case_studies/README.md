# Case studies

This folder holds the robots of *A Manifold Introduction to Robotics and Control* in one place.

- `robots/` is the **robot model library**: one Python module per robot, each the single source of truth for that model. Every model is written in the book's notation and code names (Appendix A, "Names in code" and the table "Names in the robot library"), cites the labels of the equations it implements, and is checked against a second, symbolic derivation.
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
robot.on_manifold(state)                   # largest violation of |q| = 1, q . omega = 0, R^T R = I
L, dW, variables, inputs = robot.lagrangian()   # the geomech model it was checked against
```

| Member | What it is |
| :-- | :-- |
| `manifold`, `factors`, `dim`, `stored` | The configuration manifold as a string and as a list of factors (`"R"`, `"S2"`, `"SO3"`), its dimension, and how many numbers store a configuration |
| `Params`, `params` | A dataclass of parameters with the book's symbols and SI defaults taken from the chapters' worked examples |
| `State` | A dataclass with the code names of Appendix A: `x`, `v`, `R`, `Omega`, `q`, `omega`, `x_L`, `v_L`. Names that have no symbol in the chapters (`slide`, `RP`, `x_0`, `R_r`, ...) are listed in the appendix table "Names in the robot library" |
| `dynamics(state, u, params)` | The accelerations, one per factor, from equations written with the inertia on the left as in the chapters |
| `step(state, u, h, params, t=None)` | A fourth-order step with the exponential update `R_k exp(hat(h Omega_k))`, `exp(hat(h omega_k)) q_k` (eq4-lie-euler). The input is held over the step. With a function `u(t, state)` and the time `t` of the start of the step, it is evaluated at each of the four stages instead |
| `rollout(state, u, h, n, hold=True, t0=0.0)` | `n` steps. `u` is a constant input or a function `u(t, state)`. With `hold=True` the function is sampled at the start of each step and held, as a sampled controller does. With `hold=False` it is evaluated at every stage, which keeps the fourth order for an input that moves within a step, such as the open-loop input of a flat map (sec10-quad). Held, a moving input makes the step first order |
| `energy(state, params)` | Kinetic plus potential energy |
| `on_manifold(state)` | Residual of the manifold constraints |
| `random_state(rng)` | A random state on the manifold |
| `lagrangian()` | The geomech `(L, dW, variables, inputs)` used to derive the model a second time |
| `labels` | The book labels that define the model |
| `tracking_controller(...)`, `linearize(...)` | Where Chapters 6, 7 and 20 have them, with the label each implements. The controller of `quadrotor` takes a moving reference with exact feedforward. The controller of `quadrotor_load` is the cascade eq6-load for a fixed target, with no feedforward, and is left that way: the library has no load controller for a moving reference. The feedforward experiment of Chapter 10 (sec10-feedforward) runs the quadrotor cascade with zero, flat and exact feedforward through a helper of its own, `cascade_run` in `verify/ch10.py`, which is not part of the library |

`params` can be left out everywhere: the robot then uses `robot.params`. An input is a tuple in the book's order, `u = (f, tau)`. For several vehicles `f` has shape `(n,)` and `tau` shape `(n, 3)`, and their states are stacked the same way: `q` is `(n, 3)`, `R` is `(n, 3, 3)`.

Only NumPy is needed to run a model. `geomech` is imported inside `lagrangian()` (and `biped_walking.floating_lagrangian()`) and by the verification script.

## The robots

| `robots.get(...)` | Robot | Configuration manifold | Dimension | Defined in the book | Derived in | Status |
| :-- | :-- | :-- | :-: | :-- | :-- | :-- |
| `spherical_pendulum` | Spherical pendulum | S^2 | 2 | eq4-sph, eq4-tension | sec4-sphere, eq5-pendulum | verified |
| `rigid_body` | Free rigid body | SO(3) | 3 | eq4-euler-free | sec4-so3 | verified |
| `pendulum_3d` | 3D pendulum | SO(3) | 3 | eq4-pendulum | sec4-forces | verified |
| `quadrotor` | Quadrotor | R^3 x SO(3) | 6 | eq4-quadrotor, flat map in thm10-quad | sec4-products | verified |
| `quadrotor_load` | Quadrotor with a cable-suspended load | R^3 x S^2 x SO(3) | 8 | eq4-load, eq4-load-tension, flat map in thm10-load | sec4-products, eq5-wu | verified |
| `multi_quadrotor_load`, `n=` | n quadrotors, point-mass load | R^3 x (S^2 x SO(3))^n | 3 + 5n | eq4-multi | sec4-products | verified, n = 1, 2, 3 |
| `multi_quadrotor_payload`, `n=` | n quadrotors, rigid payload | SE(3) x (S^2 x SO(3))^n | 6 + 5n | tbl2-robots (stated in words) | sec5-scaling, Wu (2017, ch. 3) | verified, n = 2, 3 |
| `flexible_cable`, `n=` | Chain of n links hanging from a pivot | (S^2)^n | 2n | eq5-chain | sec5-scaling | verified, n = 1 to 5 |
| `quadrotor_cable`, `n=`, `attach=` | Cable of n links carried by quadrotors at chosen joints | R^3 x (S^2)^n x SO(3)^n_Q | 3 + 2n + 3 n_Q | sec16-teams (stated in words), flat map in eq10-chain | Kotaru (2022, ch. 7) | verified for five layouts |
| `telescoping_leg` | Telescoping stance leg | R x SO(3) | 4 | eq4-leg | ex4-leg | verified |
| `reaction_mass_biped` | Reaction Mass Biped: stance dynamics, impact map, walking (`biped_walking`) | R x SO(3)^3 x R^3 | 13 | eq4-leg for the leg, tbl2-robots, eq20-stance, eq20-impact, eq20-law | Siravuru (2019, ch. 5) | verified, stance dynamics and walking |
| `cassie` | Cassie, reduced geometric model | SE(3) x SO(3)^2 x R^2 | 14 | tbl2-robots | Siravuru (2019, ch. 6) | floating base verified; no ground contact |

"Verified" means all of the following, in `verify/robots.py`:

1. geomech derives the equations of motion from the robot's `lagrangian()` and `check_eom` is below `1e-5`;
2. the hand-written `dynamics` equals the accelerations of the geomech model at random states, to `1e-9` or better;
3. `dynamics`, `tracking_controller` and `linearize` equal the functions that the chapters verified (taken from `verify/ch04.py` to `verify/ch08.py`) at the chapters' parameter values;
4. a free simulation with `step` stays on the manifold to rounding error, conserves energy to fourth order in the step, and conserves the momentum that the model conserves;
5. sized models pass for several sizes, and collapse to the smaller models when they should (one quadrotor, one link, attachment at the center of mass);
6. `rollout` with `hold=False` is fourth order for a moving input, and first order with `hold=True`;
7. for the walking biped: the impact map equals the linear system it solves, conserves angular momentum about the landing foot and never gains energy, relabeling the legs twice is the identity, `linearize` equals geomech's linearization, and a short walk reproduces the step lengths of `verify/ch20_values.json` exactly. `verify/ch20.py` checks the rest: the floating-base mass matrix against geomech, the closed loops and Lyapunov rates of both tracking laws, and the walks the chapter quotes.

## Walking the biped

`robots/biped_walking.py` adds to the stance model of `reaction_mass_biped` what Chapter 20 needs to make it walk. The robot itself reaches three of its pieces as methods.

```python
from robots import biped_walking as bw
biped = robots.get("reaction_mass_biped")
gait = bw.Gait()                                   # eq20-primitive: l_0, t_s, theta_s, ...
d = bw.primitive(0.3, 0.0, gait)                   # desired state and accelerations, a dict
state = bw.primitive_state(0.3, 0.0, gait)
u = biped.tracking_controller(state, d)            # eq20-law; BookGains() gives eq6-law per factor
A, B = biped.linearize(state, u)                   # 26 x 26 and 26 x 13, sec7-variation
new, p_c, stride = biped.impact(state, gait.l_0)   # eq20-impact, then eq20-relabel
log, steps = bw.walk([0.0] * 10)                   # sec20-walk: ten steps straight ahead
```

| Name | What it is |
| :-- | :-- |
| `impact`, `impact_matrices`, `impact_kkt`, `relabel` | The impact of a foot that lands and sticks: the closed form eq20-impact, the mass matrix `M_f` and the contact constraint matrix `C_c` of the linear system eq20-kkt, the exchange of the legs eq20-relabel |
| `hip_velocity`, `swing_foot`, `particles`, `angular_momentum_about`, `kinetic_floating`, `floating_lagrangian` | The floating-base model behind the impact, built mass by mass, and its geomech Lagrangian |
| `Gait`, `primitive`, `primitive_state` | The motion primitive eq20-primitive |
| `thesis_law`, `ThesisGains`, `thesis_V`; `book_law`, `BookGains`, `book_V0` | The thesis's law (eq20-right, eq20-law) and eq6-law on each factor with the term -(1/2) J_dot e_Omega, each with its Lyapunov function |
| `linearize_stance`, `perturbed`, `error_state` | The linearization along the variations and the error state of sec13-error |
| `walk` | The hybrid loop: step with `ReactionMassBiped.step`, find the touchdown by bisection, apply the impact, start the next primitive |
| `explicit_update` | The thesis's explicit update (its eqs. 5.36 to 5.40), kept only to show, as Chapter 20 does, that its energy drifts in proportion to h. Simulate with `step` |

What is not included, by robot:

- `reaction_mass_biped`: the swing-leg length, the floating base in flight (a foot always sticks; there is no phase with both feet off the ground), friction cones (`walk` does not enforce unilateral contact or friction: read the ground force and the impulses it logs, warn20-impact), and an implicit integrator for the variable inertias, such as the variational scheme of sec12-biped. The configuration manifold of the biped has dimension 14 (tbl2-robots). The stance dynamics involve 13 of them: the swing leg is balanced about the hip, so its length does not enter. The library implements those 13, which is the number `robot.dim` reports.
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
6. **Control.** Add `tracking_controller` (skill `ch06-geometric-tracking-control`) and `linearize` (skill `ch07-variation-lqr-design`) when you have them, each with the label of the law it implements. Skills `ch08-adaptive-safety-design`, `ch10-flatness-trajectory-design`, `ch12-variational-integrator-design`, `ch13-manifold-mpc-design` and `ch14-invariant-filter-design` cover protection, planning, long simulations and estimation.
7. **Map.** Add a row to the table above and to the map in the hub chapter.
