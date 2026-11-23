---
name: ch02-manifold-kinematics
description: Choose the configuration manifold of a robot (products of R^n, S^1, S^2, SO(3), SE(3)), write its kinematic equations, and implement a time step that stays on the manifold. Use when modeling a new system, when a simulator or estimator shows drifting unit vectors or non-orthogonal rotation matrices, or when reviewing code that adds velocities to configurations.
---

# Configuration manifold, kinematics and integrator for a new robot

## When to use

- You are about to model a robot with rotating bodies, cables, links or a floating base.
- A simulation shows `norm(q) != 1`, `R.T @ R != I`, or a cable or link that changes length.
- You are reviewing code that contains `q += h * q_dot` or `R += h * R_dot` for a direction or a rotation.

## Conventions (fixed)

- `R` in SO(3) maps body to world. `R_dot = R @ hat(Omega)`, `Omega` in the body frame. Spatial velocity `omega = R @ Omega`.
- `q` in S^2 is a unit vector. `q_dot = cross(omega, q)` with `dot(omega, q) = 0`, and `omega = cross(q, q_dot)`.
- `hat(x) @ y = cross(x, y)`, `vee(hat(x)) = x`.
- SE(3): `g = [[R, x], [0, 1]]`, body twist `xi = (R.T @ v, Omega)`, linear part first, `g_dot = g @ hat6(xi)`.

## Procedure

### 1. List the parts and assign one factor to each

| Physical part | Factor | Stored | Constraints | Dimension | Velocity |
| :-- | :-- | :-: | :-: | :-: | :-- |
| Point position, slider, prismatic joint | R^n (or an interval) | n | 0 | n | `v` |
| Planar heading, single revolute joint | S^1 = SO(2) | 2 | 1 | 1 | scalar `omega` |
| Direction with no twist: cable, thrust axis, thin rod, spherical pendulum | S^2 | 3 | 1 | 2 | `omega`, 3 numbers, `dot(omega, q) = 0` |
| Attitude of a rigid body | SO(3) | 9 | 6 | 3 | `Omega`, 3 free numbers |
| Pose of a free rigid body | SE(3), or R^3 x SO(3) | 12 | 6 | 6 | `(v, Omega)` |

Rules for the choice:

- Use S^2, not SO(3), when rotation about the part's own axis has no inertia and no effect. Otherwise use SO(3).
- Use rotation matrices or unit vectors for any joint with more than one rotational degree of freedom. Do not introduce Euler angles.
- Derived quantities are not factors. For a cable of length `l` from quadrotor to load, `x_Q = x_L - l * q` is computed, not stored.

### 2. Count

- Dimension of the product = sum of the factor dimensions = degrees of freedom.
- Stored numbers minus constraints must equal the dimension.
- Unactuated degrees of freedom = dimension minus number of independent inputs.

Reference counts: quadrotor SE(3), 6. Quadrotor with cable-suspended load R^3 x S^2 x SO(3), 8, with 4 inputs. `n` quadrotors with a rigid payload SE(3) x (S^2 x SO(3))^n, `6 + 5n`.

### 3. Write down the modeling claims

Each factor asserts a constraint. Record them next to the model: "cable taut" for S^2, "foot pinned" for a fixed base, "link rigid" for a fixed length. For each claim, name the signal that detects its violation (cable tension, contact force) and the model that replaces it.

### 4. Write the kinematics, one line per factor

```
x_dot = v
q_dot = cross(omega, q)          # dot(omega, q) = 0
R_dot = R @ hat(Omega)
```

Velocities of derived points follow by differentiating, for example `x_Q_dot = v_L - l * cross(omega, q)`.

### 5. Implement the maps

```python
def exp_so3(phi):                       # Rodrigues
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3) + hat(phi)
    K = hat(phi / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def log_so3(R):                         # valid for rotation angle < pi
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-10:
        return vee(R - R.T) / 2
    return th / (2 * np.sin(th)) * vee(R - R.T)
```

Formulas: `exp(theta * hat(u)) = I + sin(theta) hat(u) + (1 - cos(theta)) hat(u)^2`. `theta = arccos((tr R - 1) / 2)`, `log(R)^vee = theta / (2 sin(theta)) * vee(R - R.T)`.

### 6. Implement the step, factor by factor

```python
x = x + h * v                           # flat factor
q = exp_so3(h * omega) @ q              # S^2
R = R @ exp_so3(h * Omega)              # SO(3), body-frame Omega multiplies on the right
```

Sample `Omega` and `omega` at the midpoint of the step when they are available there. This raises the kinematic step from first to second order.

### 7. Convert between frames with the adjoint, never by guessing

- SO(3): `R @ hat(x) @ R.T = hat(R @ x)`. Body to world: `omega = R @ Omega`.
- SE(3): `g @ hat6(xi) @ inv(g) = hat6(Ad(g) @ xi)` with `Ad(g) = [[R, hat(x) @ R], [0, R]]` for twists ordered `(V, Omega)`.

## Checks that must pass

1. Counting: stored numbers minus constraints equals the dimension, for every factor and for the product.
2. Tangency: for random states, `dot(q, q_dot) < 1e-12` and `R.T @ R_dot` is skew-symmetric.
3. Invariants after 10^4 steps of a tumbling test motion: `abs(norm(q) - 1) < 1e-10`, `norm(R.T @ R - I) < 1e-10`, `abs(det(R) - 1) < 1e-10`.
4. Exactness: with constant `Omega`, one step of size `T` equals `T / h` steps of size `h` to rounding error.
5. Round trip: `log_so3(exp_so3(phi)) == phi` for `norm(phi) < pi`, and `exp_so3` agrees with a general matrix exponential.
6. Order: halving `h` halves the attitude error with start-of-step sampling and quarters it with midpoint sampling.
7. Adjoint: the two identities of step 7 hold numerically at random points.

## Failure modes

- **Euler step on a curved factor.** `q + h * q_dot` multiplies `norm(q)^2` by `1 + h^2 * norm(omega)^2` each step. `R + h * R @ hat(Omega)` has determinant `1 + h^2 * norm(Omega)^2`. Replace with step 6.
- **Normalizing as the only fix.** Projection restores the constraint and hides the integration error. If you must project a rotation, use the orthogonal polar factor (`U @ Vt` from the SVD), not Gram-Schmidt.
- **Wrong side.** Body-frame `Omega` multiplies on the right, `R @ exp_so3(h * Omega)`. World-frame `omega` multiplies on the left, `exp_so3(h * omega) @ R`.
- **Adding rotation vectors.** `exp(a) exp(b) != exp(a + b)`. The difference starts at `cross(a, b) / 2`.
- **Logarithm near half a turn.** `log_so3` divides by `sin(theta)` and is two-valued at `theta = pi`. Do not build a controller or a cost on `log` without handling this.
- **Treating S^2 as a group.** There is no `q_d^{-1} q` and no unconstrained 2-vector velocity valid everywhere. Keep `omega` in R^3 with `dot(omega, q) = 0`. Compare velocities at different points only through a transport map.
- **On the manifold, still wrong.** A perfect rotation matrix can be the wrong rotation. Check accuracy against a fine-step reference, not only the invariants.
- **Silent constraint violation in the hardware.** A slack cable or a lifted foot changes the configuration manifold. Switch models. Do not keep integrating the old one.
