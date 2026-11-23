# %% [markdown]
# # Chapter 5: Symbolic derivation
#
# This notebook reproduces the chapter's examples with `geomech` and gives you a place to start the exercises.
# The order is: derive the quadrotor with a cable-suspended load, check it, plant a sign error, simulate it,
# compare with the thesis equations, then watch the spherical pendulum go through the pipeline stage by stage,
# and finish with a chain of n links.

# %%
try:
    import geomech
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "geomech"])
    import geomech

import io
import logging

import matplotlib.pyplot as plt
import numpy as np

print("geomech", geomech.__version__)

# %% [markdown]
# ## 1. Seventeen lines (Section 5.1)
#
# Declare the variables by manifold, write $L$ and $\delta W$, derive, check.

# %%
from geomech import *

half = Scalar("0.5", value=0.5, attr=["Constant"])
mQ, mL, g, l = getScalars("m_Q m_L g l", attr=["Constant"])
J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
e3 = Vector("e3", attr=["Constant"])
xL, q, R = Vector("x_L"), S2("q"), SO3("R")     # load, cable direction, attitude
f, tau = Scalar("f"), Vector("\\tau")          # thrust, torque
omega, xi = q.get_tangent_vector(), q.get_variation_vector()
Omega, eta = R.get_tangent_vector(), R.get_variation_vector()

xQ = xL - q * l                                 # quadrotor: one cable length up
vQ, vL = xQ.t_diff(), xL.t_diff()
L = (half * mQ * Dot(vQ, vQ) + half * mL * Dot(vL, vL)
     + half * Dot(Omega, J * Omega) - mQ * g * Dot(e3, xQ) - mL * g * Dot(e3, xL))
dW = Dot(xL.delta() - Cross(xi, q) * l, (R * e3) * f) + Dot(eta, tau)

variables, inputs = SystemVariables(vectors=[xL, q], matrices=[R]), [f, tau]
eom = compute_eom(L, dW, variables)
err = check_eom(L, dW, variables, eom, inputs=inputs)

print_eom(eom)
print("check_eom error:", err)

# %% [markdown]
# `print_eom` prints LaTeX source, one line per variation. Each line is $\int (\text{variation}) \cdot E_\xi\,\mathrm{d}t = 0$,
# so the equation of motion is $E_\xi = 0$ with $E_\xi = -(\text{Euler-Lagrange expression}) + \text{forcing}$.
# The equation for $\xi_q$ only holds on the tangent plane at $q$.
#
# ## 2. Trust, but verify (Section 5.4)
#
# Type the load equation by hand, once with each sign of the centripetal term. `check_eom` compares each with a
# finite-difference Euler-Lagrange residual at random states.

# %%
aL, domega = xL.t_diff().t_diff(), omega.t_diff()


def load_equation(sign):            # the x_L row, typed in from a page of algebra
    return ((aL + e3 * g) * (mQ + mL) * (-1) + Cross(domega, q) * (mQ * l)
            + q * (Dot(omega, omega) * mQ * l) * sign + (R * e3) * f)


for sign in (-1, +1):
    hand = dict(eom)
    hand[str(xL.delta())] = (xL.delta(), load_equation(sign))
    print(sign, check_eom(L, dW, variables, hand, inputs=inputs))

# %%
hand[str(xL.delta())] = (xL.delta(), load_equation(+1))
try:
    assert_eom(L, dW, variables, hand, inputs=inputs)
except AssertionError as exc:
    print("assert_eom:", exc)

# %% [markdown]
# The three velocity terms in the cable equation add up to a vector along $q$, which is not part of the equation.

# %%
rng = np.random.default_rng(0)
qv = rng.normal(size=3)
qv /= np.linalg.norm(qv)
wv = np.cross(qv, rng.normal(size=3))
vv = rng.normal(size=3)
bracket = np.cross(np.cross(wv, qv), vv) - 2 * np.cross(wv, np.cross(qv, vv)) + np.cross(qv, np.cross(wv, vv))
print("bracket          :", bracket)
print("-(omega . v) q   :", -(wv @ vv) * qv)

# %% [markdown]
# ## 3. From equations to motion (Section 5.5)
#
# Standard form $M\,\dot v + b + G\,u = 0$, a generated NumPy module, and a Lie-group integrator.

# %%
from geomech.codegen import StateLayout, collect_params, load_module, module_source
from geomech.simulation import simulate

sf = to_standard_form(eom, variables, inputs)
layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
plant = load_module(module_source(sf, layout, lagrangian=L), "quadrotor_load")

p = {"m_Q": 1.5, "m_L": 0.3, "l": 1.0, "g": 9.81,
     "J": np.diag([0.02, 0.025, 0.04]), "e3": np.array([0.0, 0.0, 1.0])}
q0 = np.array([np.sin(0.8), 0.0, -np.cos(0.8)])     # cable swung out by 0.8 rad
w0 = np.cross(q0, np.cross([0.0, 0.0, 1.6], q0))    # circling about the vertical
v_cm = np.array([0.45, 0.0, 0.0])                   # center-of-mass velocity
v0 = v_cm + p["m_Q"] * p["l"] / (p["m_Q"] + p["m_L"]) * np.cross(w0, q0)
x0 = layout.pack(x_L=np.zeros(3), q=q0, R=np.eye(3), omega_q=w0, x_L_dot=v0)

hover = np.r_[(p["m_Q"] + p["m_L"]) * p["g"], 0.0, 0.0, 0.0]     # u = [f, tau]
traj = simulate(plant, x0, 6.0, p, u=hover)
free = simulate(plant, x0, 6.0, p)                  # no inputs: energy is conserved
E = free.energy()
print(np.max(np.abs(E - E[0])) / abs(E[0]), free.max_drift())

# %%
xQ_t = traj["x_L"] - p["l"] * traj["q"]
com = (p["m_Q"] * xQ_t + p["m_L"] * traj["x_L"]) / (p["m_Q"] + p["m_L"])
Ef = traj.energy()
work = hover[0] * (xQ_t[:, 2] - xQ_t[0, 2])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.4))
ax0.plot(traj["x_L"][:, 0], traj["x_L"][:, 2], label="load")
ax0.plot(xQ_t[:, 0], xQ_t[:, 2], label="quadrotor")
ax0.plot(com[:, 0], com[:, 2], "k--", lw=0.8, label="center of mass")
ax0.set_xlabel("x (m)")
ax0.set_ylabel("z (m)")
ax0.set_aspect("equal")
ax0.legend()
ax1.plot(traj.t, Ef - Ef[0], label="E(t) - E(0), thrust on")
ax1.plot(traj.t, Ef - work - Ef[0], label="E - f e3.x_Q, thrust on")
ax1.plot(free.t, E - E[0], label="E(t) - E(0), no inputs")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("J")
ax1.legend()
plt.tight_layout()
plt.show()
print("constraint drift, forced run:", traj.max_drift())

# %% [markdown]
# Compare with the hand-derived model of Wu (2017), eq. (3.4):
# $(m_Q+m_L)(\ddot x_L + g e_3) = (q\cdot fRe_3 - m_Q l\,\|\dot q\|^2)\,q$, $\ m_Q l\,\dot\omega = -q\times fRe_3$,
# $\ J\dot\Omega + \Omega\times J\Omega = \tau$.

# %%
from geomech.simulation import accelerations


def wu_accelerations(qv, wv, Rm, Om, fv, tauv):
    F = fv * Rm[:, 2]
    qd = np.cross(wv, qv)
    aL_ = (qv @ F - p["m_Q"] * p["l"] * qd @ qd) * qv / (p["m_Q"] + p["m_L"]) - p["g"] * p["e3"]
    dw = -np.cross(qv, F) / (p["m_Q"] * p["l"])
    dOm = np.linalg.solve(p["J"], tauv - np.cross(Om, p["J"] @ Om))
    return np.r_[aL_, dw, dOm]


worst = 0.0
for _ in range(50):
    qv = rng.normal(size=3)
    qv /= np.linalg.norm(qv)
    wv = np.cross(qv, rng.normal(size=3))
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Rm *= np.sign(np.linalg.det(Rm))
    Om = rng.normal(size=3)
    uv = np.r_[rng.uniform(5, 30), rng.normal(size=3)]
    state = layout.pack(x_L=rng.normal(size=3), q=qv, R=Rm, x_L_dot=rng.normal(size=3), omega_q=wv, Omega_R=Om)
    diff = accelerations(plant, state, uv, p) - wu_accelerations(qv, wv, Rm, Om, uv[0], uv[1:])
    worst = max(worst, float(np.max(np.abs(diff))))
print("largest difference from the thesis equations:", worst)

# %% [markdown]
# ## 4. One pendulum, every stage (Section 5.3)
#
# The same pipeline on the smallest curved example, with a trace that keeps every intermediate expression.

# %%
from geomech import *

half = Scalar("0.5", value=0.5, attr=["Constant"])
m, g, l = getScalars("m g l", attr=["Constant"])
e3 = Vector("e3", attr=["Constant"])
q, tau = S2("q"), Vector("\\tau")               # rod direction, torque at the pivot
omega, xi = q.get_tangent_vector(), q.get_variation_vector()

L = half * m * l * l * Dot(omega, omega) - m * g * l * Dot(q, e3)
dW = Dot(xi, tau)
variables = SystemVariables(vectors=[q])

trace = EOMTrace()
eom = compute_eom(L, dW, variables, trace=trace)

trace.summary(chars=95)

# %%
for stage in ["dS", "manifold_rules", "simplify", "ibp", "equation[\\xi_{q}]"]:
    print(f"{stage:22s}", to_latex(trace[stage]))

# %% [markdown]
# Every rewrite has a name. Turn the rule log on and count.

# %%
buf = io.StringIO()
handler = logging.StreamHandler(buf)
logging.getLogger("geomech.rules").addHandler(handler)
set_trace(True)
compute_eom(L, dW, variables)
set_trace(False)
logging.getLogger("geomech.rules").removeHandler(handler)
lines = buf.getvalue().splitlines()
print(len(lines), "rule applications. The first ten:")
for ln in lines[:10]:
    print("  ", ln[:110])

# %% [markdown]
# The tree itself: `print_tree` shows the object behind Figure 5.1.

# %%
print_tree(L, style="indent")

# %% [markdown]
# ## 5. More bodies (Section 5.6)
#
# A chain of $n$ spherical pendulums with $n$ symbolic, then instantiated for $n = 1, \dots, 6$.

# %%
from geomech.indexed import (Family, IndexedVariables, Sum, compute_indexed_eom,
                             indices, n)

i, j = indices("i j")
g, e3 = Scalar("g", attr=["Constant"]), Vector("e3", attr=["Constant"])
q = Family(S2, "q")                               # q_1, ..., q_n
m = Family(Scalar, "m", attr=["Constant"])
l = Family(Scalar, "l", attr=["Constant"])


def x(k):                                         # position of mass k
    return Sum(j, 1, k, q[j] * l[j])


v = x(i).t_diff()
L = Sum(i, 1, n, 0.5 * m[i] * v.dot(v)) - Sum(i, 1, n, m[i] * g * x(i).dot(e3))
chain = IndexedVariables(families=[q])
eom = compute_indexed_eom(L, None, chain)

for size in range(1, 7):
    members, eq = chain.members(size), eom.instantiate(size)
    print(size, check_eom(L.instantiate(size), None, members, eq))

# %%
print(eom["q"].to_latex())

# %%
size = 4
members, L4 = chain.members(size), L.instantiate(size)
sf4 = to_standard_form(eom.instantiate(size), members, [])
layout4 = StateLayout(members, [], collect_params([L4], members, []))
plant4 = load_module(module_source(sf4, layout4, lagrangian=L4), "chain4")
p4 = {"g": 9.81, "e3": np.array([0.0, 0.0, 1.0])}
p4.update({f"m_{{{k}}}": 0.25 for k in range(1, size + 1)})
p4.update({f"l_{{{k}}}": 0.25 for k in range(1, size + 1)})
x04 = layout4.pack(**{f"q_{k}": [1.0, 0.0, 0.0] for k in range(1, size + 1)},
                   **{f"omega_q_{k}": [0.0, 0.0, 0.6 * k] for k in range(1, size + 1)})
tr4 = simulate(plant4, x04, 2.0, p4)
E4 = tr4.energy()
print("energy drift:", np.max(np.abs(E4 - E4[0])) / max(abs(E4[0]), 1.0), " constraint drift:", tr4.max_drift())

joints = np.zeros((len(tr4.t), size + 1, 3))
for k in range(1, size + 1):
    joints[:, k] = joints[:, k - 1] + 0.25 * tr4[f"q_{k}"]
plt.figure(figsize=(5, 3.4))
for idx in range(0, 900, 150):
    plt.plot(joints[idx, :, 0], joints[idx, :, 2], "o-", ms=3, color=str(0.8 - 0.8 * idx / 900))
plt.gca().set_aspect("equal")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.show()

# %% [markdown]
# ## Starting points for the exercises
#
# **Exercise 5.3.** The 3D pendulum on $\mathrm{SO}(3)$. Run it with a trace and find the stage where the term from
# $\delta\Omega = \dot\eta + \Omega\times\eta$ survives.

# %%
half = Scalar("0.5", value=0.5, attr=["Constant"])
m, g = getScalars("m g", attr=["Constant"])
J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
rho, e3 = Vector("\\rho", attr=["Constant"]), Vector("e3", attr=["Constant"])
R, tau = SO3("R"), Vector("\\tau")
Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
L = half * Dot(Omega, J * Omega) - m * g * Dot(e3, R * rho)
dW = Dot(eta, tau)
variables = SystemVariables(matrices=[R])
trace = EOMTrace()
eom = compute_eom(L, dW, variables, trace=trace)
trace.summary(chars=95)
print_eom(eom)
print("check_eom error:", check_eom(L, dW, variables, eom, inputs=[tau]))

# %% [markdown]
# **Exercise 5.5.** Plant your own error. `eom` is a dictionary `{name: (variation, E)}`. Replace one `E` by an
# expression of your own and run `check_eom`. Start from the listing in part 2.
#
# **Exercise 5.6.** The library's catalog has the elastic cable ready made. Compare its Lagrangian with yours.

# %%
import geomech.systems

elastic = geomech.systems.get("quadrotor_elastic_load")
eom_e = compute_eom(elastic.L, elastic.dW, elastic.variables)
print(len(eom_e), "equations, keyed by", list(eom_e))
print("check_eom error:", check_eom(elastic.L, elastic.dW, elastic.variables, eom_e, inputs=elastic.inputs))
