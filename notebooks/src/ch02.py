# %% [markdown]
# # Chapter 2. The manifolds robots live on
#
# This notebook reproduces every number in the chapter and gives you starting points for the exercises.
# It needs only NumPy, SciPy and Matplotlib.
#
# Conventions: $R \in \mathrm{SO}(3)$ maps body to world, $\dot R = R\,\Omega^\wedge$ with $\Omega$ in the body frame,
# $q \in S^2$ with $\dot q = \omega \times q$ and $\omega \cdot q = 0$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True)
E1, E2, E3 = np.eye(3)
I3 = np.eye(3)


def hat(x):
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def exp_so3(phi):
    """Rodrigues' formula: exp(hat(phi))."""
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return I3 + hat(phi)
    K = hat(phi / th)
    return I3 + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    """Rotation vector of R. Valid for rotation angles below pi."""
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-10:
        return vee(R - R.T) / 2
    return th / (2 * np.sin(th)) * vee(R - R.T)


def angle(R):
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))


def Rx(a):
    return exp_so3(a * E1)


def Rz(a):
    return exp_so3(a * E3)

# %% [markdown]
# ## 1. The opening puzzle: a cable that stretches
#
# Euler steps $q \leftarrow q + h\,(\omega \times q)$ at 100 Hz, with the cable swinging at 2 rad/s.

# %%
h, w, T = 0.01, 2.0, 10.0
q, om = E1.copy(), w * E3
lengths = [1.0]
for _ in range(round(T / h)):
    q = q + h * np.cross(om, q)
    lengths.append(np.linalg.norm(q))
print(f"cable length after {T:.0f} s: {lengths[-1]:.4f}")
print(f"predicted (1 + h^2 w^2)^(N/2): {(1 + h**2 * w**2) ** (T / h / 2):.4f}")

# %% [markdown]
# ## 2. Tangent vectors
#
# The velocity of any curve on the sphere is perpendicular to $q$, and $\omega = q \times \dot q$ recovers it.

# %%
def curve(t):
    p = np.array([np.cos(t), np.sin(2 * t), 0.5 + 0.3 * t])
    return p / np.linalg.norm(p)


t0, d = 0.4, 1e-6
q0 = curve(t0)
qdot = (curve(t0 + d) - curve(t0 - d)) / (2 * d)
omega = np.cross(q0, qdot)
print("q . q_dot        =", f"{q0 @ qdot:.1e}")
print("omega x q - q_dot =", np.cross(omega, q0) - qdot)

# %% [markdown]
# The same for a rotation: $R^\top \dot R$ is skew-symmetric, so $\dot R = R\,\Omega^\wedge$.
# The spatial velocity is $\omega = R\,\Omega$, and $R\,x^\wedge R^\top = (Rx)^\wedge$.

# %%
rng = np.random.default_rng(0)
R = exp_so3(rng.normal(size=3))
Om = rng.normal(size=3)
Rdot = (R @ exp_so3(d * Om) - R @ exp_so3(-d * Om)) / (2 * d)
S = R.T @ Rdot
print("skew part check   :", f"{np.abs(S + S.T).max():.1e}")
print("body velocity     :", vee(S), "(should be", Om, ")")
print("spatial velocity  :", vee(Rdot @ R.T), "= R Omega =", R @ Om)
x = rng.normal(size=3)
print("adjoint identity  :", f"{np.abs(R @ hat(x) @ R.T - hat(R @ x)).max():.1e}")

# %% [markdown]
# **The phone on the turntable.** World angular velocity $(0, 0, 0.5)$. The gyroscope reads $\Omega = R^\top \omega$.

# %%
w_world = np.array([0.0, 0.0, 0.5])
print("phone flat    :", I3.T @ w_world)
print("phone upright :", Rx(np.pi / 2).T @ w_world)

# %% [markdown]
# ## 3. Exponential and logarithm

# %%
phi = (np.pi / 2) * np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
R_ex = exp_so3(phi)
print(R_ex)
print("matches scipy expm:", np.allclose(R_ex, expm(hat(phi))))
print("angle from trace  :", angle(R_ex), "(pi/2 =", np.pi / 2, ")")
print("log               :", log_so3(R_ex), "(theta u =", phi, ")")
u = np.array([0.0, 0.6, 0.8])
print("exp(pi u) == exp(-pi u):", np.allclose(exp_so3(np.pi * u), exp_so3(-np.pi * u)))

# %% [markdown]
# On the sphere, $\exp_q(v) = \cos\|v\|\,q + \sin\|v\|\,v/\|v\|$.

# %%
def exp_s2(q, v):
    n = np.linalg.norm(v)
    return q if n < 1e-12 else np.cos(n) * q + np.sin(n) * v / n


v = np.array([0.3, 0.4, 0.0])            # tangent at the north pole
p = exp_s2(E3, v)
print("landed at", p, " norm", np.linalg.norm(p), " arc length", np.arccos(p @ E3))

# %% [markdown]
# ## 4. Order matters
#
# The book: quarter turns about the desk's $x$ and $z$ axes, in both orders.
# Turns about fixed axes multiply on the left.

# %%
A, B = Rx(np.pi / 2), Rz(np.pi / 2)
x_then_z, z_then_x = B @ A, A @ B
print("x then z: cover normal ->", x_then_z @ E3, " spine direction ->", x_then_z @ E2)
print("z then x: cover normal ->", z_then_x @ E3, " spine direction ->", z_then_x @ E2)
rel = x_then_z.T @ z_then_x
print(f"the two results differ by {np.degrees(angle(rel)):.1f} degrees about", log_so3(rel) / angle(rel))

# %% [markdown]
# Small rotations: the leftover of $\exp(\epsilon a)\exp(\epsilon b)\exp(-\epsilon a)\exp(-\epsilon b)$
# is $\epsilon^2\, a \times b$ to leading order.

# %%
a, b = E1, E2
for eps in (0.1, 0.01):
    C = exp_so3(eps * a) @ exp_so3(eps * b) @ exp_so3(-eps * a) @ exp_so3(-eps * b)
    c = log_so3(C)
    print(f"eps = {eps}: leftover {np.linalg.norm(c):.6f} rad ({np.degrees(np.linalg.norm(c)):.3f} deg), "
          f"log / eps^2 = {c / eps**2}")
print("bracket identity:", np.allclose(hat(a) @ hat(b) - hat(b) @ hat(a), hat(np.cross(a, b))))

# %% [markdown]
# ## 5. SE(3) and its adjoint
#
# Twists are ordered $(V, \Omega)$, linear part first.

# %%
def se3(R, x):
    g = np.eye(4)
    g[:3, :3], g[:3, 3] = R, x
    return g


def hat6(xi):
    X = np.zeros((4, 4))
    X[:3, :3], X[:3, 3] = hat(xi[3:]), xi[:3]
    return X


def Ad(g):
    R, x = g[:3, :3], g[:3, 3]
    return np.block([[R, hat(x) @ R], [np.zeros((3, 3)), R]])


g = se3(R, rng.normal(size=3))
xi = rng.normal(size=6)
lhs = g @ hat6(xi) @ np.linalg.inv(g)
print("g xi^ g^-1 = (Ad_g xi)^ :", np.allclose(lhs, hat6(Ad(g) @ xi)))

# %% [markdown]
# ## 6. A product manifold: quadrotor with a cable-suspended load
#
# State $(x_L, q, R)$ on $\mathbb{R}^3 \times S^2 \times \mathrm{SO}(3)$, velocity $(v_L, \omega, \Omega)$.
# The quadrotor position follows from $x_Q = x_L - l\,q$.

# %%
l = 1.0
xL, vL = np.array([2.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
q, om = -E3, np.array([0.5, 0.0, 0.0])
print("x_Q     =", xL - l * q)
print("x_Q dot =", vL - l * np.cross(om, q))


def step(state, vel, h):
    """One step on R^3 x S^2 x SO(3): Euler on the flat factor, exponential on the curved ones."""
    xL, q, R = state
    vL, om, Om = vel
    return xL + h * vL, exp_so3(h * om) @ q, R @ exp_so3(h * Om)


state = (xL, q, I3)
for _ in range(1000):
    state = step(state, (vL, om, np.array([0.3, -0.2, 0.5])), 0.01)
print("after 1000 steps: |q| - 1 =", f"{np.linalg.norm(state[1]) - 1:.1e}",
      " |R^T R - I| =", f"{np.linalg.norm(state[2].T @ state[2] - I3):.1e}")

# %% [markdown]
# ## 7. Stepping without falling off
#
# A tumbling body, $\Omega(t) = (\cos t, \sin 0.7t, 0.5)$ rad/s, for 10 s at $h = 0.01$ s.

# %%
def Omega_of(t):
    return np.array([np.cos(t), np.sin(0.7 * t), 0.5])


def integrate(kind, h, T=10.0, midpoint=False):
    R, out = I3.copy(), [I3.copy()]
    for k in range(round(T / h)):
        Om = Omega_of((k + 0.5) * h) if midpoint else Omega_of(k * h)
        R = R + h * R @ hat(Om) if kind == "euler" else R @ exp_so3(h * Om)
        out.append(R)
    return np.array(out)


h = 0.01
t = np.arange(0, 10 + h / 2, h)
eu, ex, exm = integrate("euler", h), integrate("exp", h), integrate("exp", h, midpoint=True)
ref = integrate("exp", 1e-4, midpoint=True)[::100]
orth = lambda Rs: np.array([np.linalg.norm(M.T @ M - I3) for M in Rs])
err = lambda Rs: np.degrees([angle(A.T @ B) for A, B in zip(ref, Rs)])
print(f"Euler:       |R^T R - I| = {orth(eu)[-1]:.3f},  det R = {np.linalg.det(eu[-1]):.3f}")
print(f"exponential: |R^T R - I| = {orth(ex).max():.1e}")
print(f"attitude error, Omega at start of step: {err(ex)[-1]:.3f} deg")
print(f"attitude error, Omega at midpoint:      {err(exm)[-1]:.5f} deg")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].semilogy(t[1:], orth(eu)[1:], label="Euler step")
axs[0].semilogy(t[1:], np.maximum(orth(ex)[1:], 1e-17), label="exponential step")
axs[0].set_title("distance from SO(3)")
axs[1].semilogy(t[1:], np.maximum(err(ex)[1:], 1e-7), label="Omega at start of step")
axs[1].semilogy(t[1:], np.maximum(err(exm)[1:], 1e-7), label="Omega at midpoint")
axs[1].set_title("attitude error of the exponential step (deg)")
for ax in axs:
    ax.set_xlabel("time (s)")
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Projection is the usual patch for a drifted matrix. The nearest rotation is the orthogonal polar factor.

# %%
U, _, Vt = np.linalg.svd(eu[-1])
R_proj = U @ Vt
print("projected matrix is a rotation:", np.allclose(R_proj.T @ R_proj, I3), f"det = {np.linalg.det(R_proj):.6f}")
print(f"but it is {np.degrees(angle(ref[-1].T @ R_proj)):.3f} deg from the true attitude")

# %% [markdown]
# ## Exercise starters
#
# **Exercise 2.1.** Change `h` in section 1 and compare with $\exp(\tfrac12 h\,\|\omega\|^2 T)$.
#
# **Exercise 2.4.** Repeat the book experiment with 10° turns:

# %%
eps = np.radians(10)
C = Rx(eps) @ Rz(eps) @ Rx(-eps) @ Rz(-eps)
print("leftover:", log_so3(C), " prediction eps^2 (e1 x e3):", eps**2 * np.cross(E1, E3))

# %% [markdown]
# **Exercise 2.6.** Order of accuracy of the exponential step. Extend the table and explain the two columns.

# %%
ref_end = integrate("exp", 1e-4, midpoint=True)[-1]
print("   h      start-of-step   midpoint   (errors in degrees)")
for hh in (0.02, 0.01, 0.005):
    e1 = np.degrees(angle(ref_end.T @ integrate("exp", hh)[-1]))
    e2 = np.degrees(angle(ref_end.T @ integrate("exp", hh, midpoint=True)[-1]))
    print(f"{hh:6.3f}   {e1:10.5f}   {e2:12.6f}")
