"""Chapter 12: variational integrators and collocation. Every identity, derivation step and quoted number.

    python verify/ch12.py

Discrete Euler-Lagrange equations are checked against finite differences of the action sum.
The continuous equations they approximate are derived a second time with geomech (check_eom).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np
from scipy.optimize import minimize

from geomech import check_eom, compute_eom, systems, to_standard_form
from geomech.codegen import StateLayout, collect_params, load_module, module_source
from geomech.simulation import accelerations

from common import Checker, expm_so3, hat, random_rotation, vee

ck = Checker("ch12")
rng = np.random.default_rng(12)
E3 = np.array([0.0, 0.0, 1.0])
G = 9.81
I3 = np.eye(3)


def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def log_so3(R):
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return vee(R - R.T) / 2
    return th / (2 * np.sin(th)) * vee(R - R.T)


def dexpinv(th, w):
    """theta_dot for R = R_k exp(hat(theta)) and body rate w."""
    t = np.sqrt(th @ th)
    if t < 1e-6:
        return w + 0.5 * cross(th, w) + cross(th, cross(th, w)) / 12
    c = (1 - 0.5 * t / np.tan(0.5 * t)) / t**2
    return w + 0.5 * cross(th, w) + c * cross(th, cross(th, w))


# =========================================================================
# 1. The planar pendulum and four first steps  (m = 1 kg, l = 1 m)
GL = G / 1.0                                   # g / l


def pend_energy(th, w):
    """Energy above the hanging rest state, per unit m l^2."""
    return 0.5 * w * w + GL * (1 - np.cos(th))


def explicit_euler(th, w, h):
    return th + h * w, w - h * GL * np.sin(th)


def implicit_euler(th, w, h):
    th1, w1 = th, w
    for _ in range(60):
        f = th1 - th - h * (w - h * GL * np.sin(th1))
        th1 -= f / (1 + h * h * GL * np.cos(th1))
    return th1, w - h * GL * np.sin(th1)


def symplectic_euler(th, w, h):
    w1 = w - h * GL * np.sin(th)
    return th + h * w1, w1


def verlet(th, w, h):
    wh = w - 0.5 * h * GL * np.sin(th)
    th1 = th + h * wh
    return th1, wh - 0.5 * h * GL * np.sin(th1)


def run_planar(step, th0, w0, h, n):
    out = np.empty((n + 1, 2))
    out[0] = th0, w0
    th, w = th0, w0
    for k in range(n):
        th, w = step(th, w, h)
        out[k + 1] = th, w
    return out


TH0, H_P, T_P = np.radians(60.0), 0.05, 10.0
N_P = int(round(T_P / H_P))
E0_P = pend_energy(TH0, 0.0)
planar = {name: run_planar(step, TH0, 0.0, H_P, N_P) for name, step in
          [("explicit", explicit_euler), ("implicit", implicit_euler), ("symplectic", symplectic_euler), ("verlet", verlet)]}
en = {k: pend_energy(v[:, 0], v[:, 1]) for k, v in planar.items()}
ck.record("planar", {"theta0_deg": 60.0, "h": H_P, "T": T_P, "E0_J": float(E0_P),
                     "period_s": float(4 / np.sqrt(GL) * 1.6858)})    # K(sin 30 deg) = 1.6858
ck.record("planar_E_end_explicit_J", float(en["explicit"][-1]))
ck.record("planar_E_end_implicit_J", float(en["implicit"][-1]))
ck.record("planar_E_band_symplectic_J", [float(en["symplectic"].min()), float(en["symplectic"].max())])
ck.record("planar_E_band_verlet_J", [float(en["verlet"].min()), float(en["verlet"].max())])
ck.true("explicit Euler: the energy of the pendulum is larger after each of the first five seconds, and ends five times larger",
        np.all(np.diff(en["explicit"][:101:20]) > 0) and en["explicit"][-1] > 5 * E0_P, f"{E0_P:.2f} -> {en['explicit'][-1]:.1f} J")
ck.true("implicit Euler: the energy of the pendulum only decays", np.all(np.diff(en["implicit"]) < 1e-15), f"{E0_P:.2f} -> {en['implicit'][-1]:.2f} J")
ck.true("symplectic Euler: energy stays in a band", np.abs(en["symplectic"] - E0_P).max() < 0.25 * E0_P,
        f"{en['symplectic'].min():.2f} to {en['symplectic'].max():.2f} J")
ck.true("Stormer-Verlet: energy stays in a narrower band", np.abs(en["verlet"] - E0_P).max() < 0.02 * E0_P,
        f"{en['verlet'].min():.3f} to {en['verlet'].max():.3f} J")
# explicit Euler goes over the top
first_over = int(np.argmax(np.abs(planar["explicit"][:, 0]) > np.pi))
ck.true("explicit Euler pendulum goes over the top within 10 s", first_over > 0, f"at t = {first_over * H_P:.2f} s")
ck.record("planar_explicit_over_top_s", float(first_over * H_P))
ck.record("planar_implicit_amplitude_end_deg", float(np.degrees(np.abs(planar["implicit"][-40:, 0]).max())))

# linear oscillator: exact statements. theta_dd = -(g/l) theta
a2 = GL * H_P**2                                # h^2 g / l
th, w = 0.3, -0.2
E_lin = lambda th, w: 0.5 * w * w + 0.5 * GL * th * th
th1, w1 = th + H_P * w, w - H_P * GL * th
ck.close("linear oscillator, explicit Euler: energy is multiplied by 1 + h^2 g/l each step", E_lin(th1, w1), (1 + a2) * E_lin(th, w), tol=1e-13)
Mi = np.linalg.inv(np.array([[1, -H_P], [H_P * GL, 1]]))
th1, w1 = Mi @ np.array([th, w])
ck.close("linear oscillator, implicit Euler: energy is divided by 1 + h^2 g/l each step", E_lin(th1, w1), E_lin(th, w) / (1 + a2), tol=1e-13)
ck.record("planar_euler_energy_factor", float(1 + a2))
ck.record("planar_euler_energy_factor_10s", float((1 + a2) ** N_P))
Ms = np.array([[1 - a2, H_P], [-H_P * GL, 1]])     # symplectic Euler
Mv = np.array([[1 - a2 / 2, H_P], [-H_P * GL * (1 - a2 / 4), 1 - a2 / 2]])   # Verlet
ck.close("area: det of the explicit Euler step is 1 + h^2 g/l", np.linalg.det(np.array([[1, H_P], [-H_P * GL, 1]])), 1 + a2, tol=1e-13)
ck.close("area: det of the implicit Euler step is 1/(1 + h^2 g/l)", np.linalg.det(Mi), 1 / (1 + a2), tol=1e-13)
ck.close("area: det of the symplectic Euler step is 1", np.linalg.det(Ms), 1.0, tol=1e-13)
ck.close("area: det of the Verlet step is 1", np.linalg.det(Mv), 1.0, tol=1e-13)
s = np.array([th, w])
Em = lambda s: 0.5 * s[1] ** 2 + 0.5 * GL * (1 - a2 / 4) * s[0] ** 2
worst = 0.0
s_k = s.copy()
for _ in range(2000):
    s_k = Mv @ s_k
    worst = max(worst, abs(Em(s_k) - Em(s)))
ck.true("linear oscillator, Verlet: w^2/2 + (g/l)(1 - h^2 g/4l) theta^2/2 is conserved exactly", worst < 1e-13, f"err {worst:.1e}")
# nonlinear pendulum: area preservation of the Verlet step (Jacobian determinant) and not of Euler
def jac(step, th, w, h, e=1e-6):
    c1 = (np.array(step(th + e, w, h)) - np.array(step(th - e, w, h))) / (2 * e)
    c2 = (np.array(step(th, w + e, h)) - np.array(step(th, w - e, h))) / (2 * e)
    return np.column_stack([c1, c2])
dets = {n: np.linalg.det(jac(f, 1.1, -0.7, H_P)) for n, f in
        [("explicit", explicit_euler), ("implicit", implicit_euler), ("symplectic", symplectic_euler), ("verlet", verlet)]}
ck.close("nonlinear pendulum: Verlet and symplectic Euler preserve area", [dets["verlet"], dets["symplectic"]], [1, 1], tol=1e-8)
ck.true("nonlinear pendulum: explicit Euler expands area, implicit Euler shrinks it", dets["explicit"] > 1.005 and dets["implicit"] < 0.995,
        f"det {dets['explicit']:.4f}, {dets['implicit']:.4f}")
# long run of Verlet: no drift, band scales with h^2
def verlet_band(h, T):
    th, w = TH0, 0.0
    lo = hi = 0.0
    n = int(round(T / h))
    first = None
    for k in range(n):
        th, w = verlet(th, w, h)
        d = pend_energy(th, w) - E0_P
        lo, hi = min(lo, d), max(hi, d)
        if k == n // 10:
            first = max(-lo, hi)
    return max(-lo, hi), first
b1, b1_first = verlet_band(0.05, 20000.0)
b2, _ = verlet_band(0.025, 2000.0)
ck.true("Verlet over 20 000 s (400 000 steps): the energy band does not widen", b1 < 1.001 * b1_first, f"band {b1:.4f} J, first tenth {b1_first:.4f} J")
ck.true("Verlet: halving h divides the energy band by 4", 3.8 < b1 / b2 < 4.2, f"ratio {b1 / b2:.2f}")
ck.record("planar_verlet_band_20000s_J", float(b1))
ck.record("planar_verlet_band_ratio_h_halved", float(b1 / b2))

# =========================================================================
# 2. Discrete mechanics on flat ground: the ball of Chapter 4 and a particle in a plane
m_b, tf_b, N_b = 1.0, 1.0, 5
h_b = tf_b / N_b


def Lh_ball(xa, xb, h):
    """Discrete Lagrangian of a ball in gravity: height only."""
    return 0.5 * m_b * (xb - xa) ** 2 / h - 0.5 * h * m_b * G * (xa + xb)


def S_ball(x, h):
    return sum(Lh_ball(x[k], x[k + 1], h) for k in range(len(x) - 1))


t_b = np.linspace(0, tf_b, N_b + 1)
x_exact = 0.5 * G * t_b * (tf_b - t_b)            # parabola through x(0) = x(1) = 0
# discrete Euler-Lagrange: m (x_{k+1} - 2 x_k + x_{k-1}) = -h^2 m g. Solve the two-point problem.
A_b = np.zeros((N_b - 1, N_b - 1))
for i in range(N_b - 1):
    A_b[i, i] = -2
    if i > 0:
        A_b[i, i - 1] = 1
    if i < N_b - 2:
        A_b[i, i + 1] = 1
x_disc = np.concatenate([[0], np.linalg.solve(A_b, -h_b**2 * G * np.ones(N_b - 1)), [0]])
ck.close("ball: the discrete path lies on the parabola at every knot", x_disc, x_exact, tol=1e-12)
grad = np.array([(S_ball(x_disc + 1e-6 * np.eye(N_b + 1)[k], h_b) - S_ball(x_disc - 1e-6 * np.eye(N_b + 1)[k], h_b)) / 2e-6 for k in range(1, N_b)])
ck.close("ball: the action sum is stationary at the discrete path (finite differences)", grad, 0, tol=1e-8)
eps_b = 0.1
dS = S_ball(x_disc + eps_b * np.eye(N_b + 1)[2], h_b) - S_ball(x_disc, h_b)
dS_m = S_ball(x_disc - eps_b * np.eye(N_b + 1)[2], h_b) - S_ball(x_disc, h_b)
ck.close("ball: moving one knot by eps raises the action sum by m eps^2 / h, either way", [dS, dS_m], m_b * eps_b**2 / h_b, tol=1e-12)
ck.record("ball", {"m": m_b, "tf": tf_b, "N": N_b, "h": h_b, "knot_heights_m": x_disc.tolist(), "apex_m": float(G * tf_b**2 / 8)})
ck.record("ball_action_rise_knot_0.1m", float(dS))
ck.record("ball_action_sum", float(S_ball(x_disc, h_b)))
S_exact = m_b * G**2 * tf_b**3 / 24 - m_b * G**2 * tf_b**3 / 12     # int (m xd^2/2 - m g x)
ck.record("ball_action_exact", float(S_exact))
ck.true("ball: action sum is within h^2 of the action integral", abs(S_ball(x_disc, h_b) - S_exact) < 0.2, f"{S_ball(x_disc, h_b):.4f} vs {S_exact:.4f}")

# a particle in the plane with a central force: Verlet from the discrete Legendre transform
kc = 3.0                                         # potential kc |x|^4 / 4 + |x|^2 / 2


def Uc(x):
    r2 = x @ x
    return 0.25 * kc * r2 * r2 + 0.5 * r2


def Fc(x):
    return -(kc * (x @ x) + 1.0) * x


def Lh_c(xa, xb, h):
    return 0.5 * (xb - xa) @ (xb - xa) / h - 0.5 * h * (Uc(xa) + Uc(xb))


def verlet_c(x, p, h):
    ph = p + 0.5 * h * Fc(x)
    x1 = x + h * ph
    return x1, ph + 0.5 * h * Fc(x1)


h_c = 0.05
xs, ps = [np.array([1.0, 0.2])], [np.array([-0.3, 0.9])]
for k in range(6):
    x1, p1 = verlet_c(xs[-1], ps[-1], h_c)
    xs.append(x1)
    ps.append(p1)


def fd_grad(fun, x, e=1e-6):
    g = np.zeros(len(x))
    for i in range(len(x)):
        d = np.zeros(len(x))
        d[i] = e
        g[i] = (fun(x + d) - fun(x - d)) / (2 * e)
    return g


worst = max(np.abs(fd_grad(lambda y: Lh_c(xs[k - 1], y, h_c) + Lh_c(y, xs[k + 1], h_c), xs[k])).max() for k in range(1, 6))
ck.true("particle: Verlet satisfies the discrete Euler-Lagrange equations (finite differences of the action sum)", worst < 1e-8, f"residual {worst:.1e}")
k = 3
ck.close("particle: p_k = -dL_h(x_k, x_k+1)/dx_k", -fd_grad(lambda y: Lh_c(y, xs[k + 1], h_c), xs[k]), ps[k], tol=1e-8)
ck.close("particle: p_k+1 = dL_h(x_k, x_k+1)/dx_k+1", fd_grad(lambda y: Lh_c(xs[k], y, h_c), xs[k + 1]), ps[k + 1], tol=1e-8)
# symplectic in two degrees of freedom: M' Jc M = Jc
def flowmap(z):
    x1, p1 = verlet_c(z[:2], z[2:], h_c)
    return np.concatenate([x1, p1])
z0 = np.array([1.0, 0.2, -0.3, 0.9])
Mjac = np.column_stack([(flowmap(z0 + 1e-6 * np.eye(4)[i]) - flowmap(z0 - 1e-6 * np.eye(4)[i])) / 2e-6 for i in range(4)])
Jc = np.block([[np.zeros((2, 2)), I3[:2, :2]], [-I3[:2, :2], np.zeros((2, 2))]])
ck.close("particle: the Verlet step is symplectic, M' Jc M = Jc", Mjac.T @ Jc @ Mjac, Jc, tol=1e-8)
# Noether: rotation symmetry -> angular momentum exactly conserved; energy only in a band
x, p = xs[0], ps[0]
ang0, E0c = x[0] * p[1] - x[1] * p[0], 0.5 * p @ p + Uc(x)
ang_err = e_band = 0.0
for _ in range(100000):
    x, p = verlet_c(x, p, h_c)
    ang_err = max(ang_err, abs(x[0] * p[1] - x[1] * p[0] - ang0))
    e_band = max(e_band, abs(0.5 * p @ p + Uc(x) - E0c))
ck.true("particle: angular momentum conserved to rounding over 100 000 steps (discrete Noether)", ang_err < 1e-11, f"err {ang_err:.1e}")
ck.true("particle: energy is not conserved exactly, but stays in a band", 1e-6 < e_band < 0.02 * E0c, f"band {e_band:.1e} of {E0c:.2f}")

# =========================================================================
# 3. The rigid body: Lie group variational integrator
J_p = np.diag([0.20, 0.25, 0.10])                # the 3D pendulum of Chapter 4
m_p, rho_p = 1.0, np.array([0.0, 0.0, -0.4])


def Jbar_of(J):
    return 0.5 * np.trace(J) * I3 - J


Jbar_p = Jbar_of(J_p)
# kinetic energy as a trace
for _ in range(5):
    Jr = random_rotation(rng) @ np.diag(rng.uniform(0.1, 1, 3))
    Jr = Jr @ Jr.T
    Om = rng.normal(size=3)
    ck.close("kinetic energy: Om.J Om/2 = tr(hat(Om) Jbar hat(Om)')/2", 0.5 * Om @ Jr @ Om, 0.5 * np.trace(hat(Om) @ Jbar_of(Jr) @ hat(Om).T), tol=1e-12)
    U = random_rotation(rng)
    ck.close("tr((U - I) Jbar (U - I)')/2 = tr((I - U) Jbar)", 0.5 * np.trace((U - I3) @ Jbar_of(Jr) @ (U - I3).T), np.trace((I3 - U) @ Jbar_of(Jr)), tol=1e-12)
    x = rng.normal(size=3)
    A = rng.normal(size=(3, 3))
    ck.close("tr(hat(x) A) = -x . vee(A - A')", np.trace(hat(x) @ A), -x @ vee(A - A.T), tol=1e-12)
    ck.close("hat(x) Jbar + Jbar hat(x) = hat(J x)", hat(x) @ Jbar_of(Jr) + Jbar_of(Jr) @ hat(x), hat(Jr @ x), tol=1e-12)


def step_momentum(Om, h, J):
    """Right side of the implicit equation: (sin th/th) J Om + ((1 - cos th)/th^2) h Om x J Om, th = h |Om|."""
    th = h * np.sqrt(Om @ Om)
    if th < 1e-4:
        a, b = 1 - th * th / 6, 0.5 - th * th / 24
    else:
        a, b = np.sin(th) / th, (1 - np.cos(th)) / th**2
    JO = J @ Om
    return a * JO + b * h * cross(Om, JO)


for _ in range(5):
    Om, hh = rng.normal(size=3) * 3, rng.uniform(0.01, 0.3)
    U = expm_so3(hh * Om)
    ck.close("vee(U Jbar - Jbar U')/h equals the vector form in Om", vee(U @ Jbar_p - Jbar_p @ U.T) / hh, step_momentum(Om, hh, J_p), tol=1e-12)
    ck.close("vee(Jbar U - U' Jbar) = U' vee(U Jbar - Jbar U')", vee(Jbar_p @ U - U.T @ Jbar_p), U.T @ vee(U @ Jbar_p - Jbar_p @ U.T), tol=1e-12)
    ck.close("vector form to second order: J Om + (h/2) Om x J Om", step_momentum(Om, 1e-3, J_p), J_p @ Om + 0.5e-3 * cross(Om, J_p @ Om), tol=1e-4)

NEWTON_ITS = []


def solve_step(pb, h, J, tol=1e-14):
    """Solve step_momentum(Om) = pb for Om by Newton's method on the Lie algebra."""
    Om = np.linalg.solve(J, pb)
    for it in range(50):
        r = step_momentum(Om, h, J) - pb
        if np.max(np.abs(r)) < tol * (1 + np.max(np.abs(pb))):
            break
        Om = Om - np.linalg.solve(J + 0.5 * h * (hat(Om) @ J - hat(J @ Om)), r)
    NEWTON_ITS.append(it)
    return Om


def torque_pend(R):
    return m_p * G * cross(R.T @ E3, rho_p)


def torque_none(R):
    return np.zeros(3)


def lgvi_step(R, p, h, J=J_p, torque=torque_pend, tau0=0.0, tau1=0.0):
    """One step of the Lie group variational integrator. Returns R_{k+1}, p_{k+1}, Omega_k."""
    pb = p + 0.5 * h * (torque(R) + tau0)
    Om = solve_step(pb, h, J)
    U = expm_so3(h * Om)
    R1 = R @ U
    return R1, U.T @ pb + 0.5 * h * (torque(R1) + tau1), Om


def Lh_rb(Ra, Rb, h, J=J_p, pot=True):
    v = np.trace((I3 - Ra.T @ Rb) @ Jbar_of(J)) / h
    if pot:
        v -= 0.5 * h * m_p * G * (E3 @ (Ra @ rho_p) + E3 @ (Rb @ rho_p))
    return v


def energy_pend(R, p):
    return 0.5 * p @ np.linalg.solve(J_p, p) + m_p * G * E3 @ (R @ rho_p)


R0_p, Om0_p = expm_so3(np.array([2.0, 0.0, 0.0])), np.array([0.0, 0.5, 6.0])     # start of fig4-integrators
p0_p = J_p @ Om0_p
E0_3d = energy_pend(R0_p, p0_p)
E_rest = m_p * G * rho_p[2]
ck.close("3D pendulum: energy above rest matches Chapter 4", E0_3d - E_rest, 7.388210186610987, tol=1e-9)

h3 = 0.02
Rs, ps, Oms = [R0_p], [p0_p], []
for k in range(8):
    R1, p1, Om = lgvi_step(Rs[-1], ps[-1], h3)
    Rs.append(R1)
    ps.append(p1)
    Oms.append(Om)
ck.true("implicit equation solved to tolerance by Newton in a few iterations", max(NEWTON_ITS) <= 6, f"max {max(NEWTON_ITS)} iterations")
res = max(np.abs(step_momentum(Oms[k], h3, J_p) - ps[k] - 0.5 * h3 * torque_pend(Rs[k])).max() for k in range(8))
ck.true("implicit equation residual along the trajectory", res < 1e-13, f"{res:.1e}")


def fd_rot(fun, e=1e-6):
    """Derivative of fun(exp(hat(eps))) at the identity."""
    return np.array([(fun(expm_so3(e * I3[i])) - fun(expm_so3(-e * I3[i]))) / (2 * e) for i in range(3)])


worst = max(np.abs(fd_rot(lambda X: Lh_rb(Rs[k - 1], Rs[k] @ X, h3) + Lh_rb(Rs[k] @ X, Rs[k + 1], h3))).max() for k in range(1, 8))
ck.true("LGVI: the action sum is stationary under R_k -> R_k exp(eps hat(eta)) (finite differences)", worst < 1e-8, f"residual {worst:.1e}")
k = 4
ck.close("LGVI: p_k is minus the slope of L_h in its first slot", -fd_rot(lambda X: Lh_rb(Rs[k] @ X, Rs[k + 1], h3)), ps[k], tol=1e-8)
ck.close("LGVI: p_k+1 is the slope of L_h in its second slot", fd_rot(lambda X: Lh_rb(Rs[k], Rs[k + 1] @ X, h3)), ps[k + 1], tol=1e-8)
# variation of the step rotation
eta0, eta1 = rng.normal(size=3), rng.normal(size=3)
U = Rs[3].T @ Rs[4]
dU = ((Rs[3] @ expm_so3(1e-6 * eta0)).T @ (Rs[4] @ expm_so3(1e-6 * eta1)) - (Rs[3] @ expm_so3(-1e-6 * eta0)).T @ (Rs[4] @ expm_so3(-1e-6 * eta1))) / 2e-6
ck.close("variation of the step rotation: dU = -hat(eta_k) U + U hat(eta_k+1)", dU, -hat(eta0) @ U + U @ hat(eta1), tol=1e-8)

# continuous model, derived a second time with geomech
EOM_TOL = 1e-5
geomech_errors = {}


def library(name):
    s = systems.get(name)
    variables, inputs = s.variables, list(s.inputs)
    eom = compute_eom(s.L, s.dW, variables)
    err = check_eom(s.L, s.dW, variables, eom, inputs=inputs)
    sf = to_standard_form(eom, variables, inputs)
    layout = StateLayout(variables, inputs, collect_params([s.L, s.dW], variables, inputs))
    geomech_errors[name] = float(err)
    ck.true(f"geomech check_eom: {name}", err < EOM_TOL, f"err {err:.1e}")
    return load_module(module_source(sf, layout, lagrangian=s.L), name), layout, dict(s.params)


J_inv = np.linalg.inv(J_p)


def accel_pend(R, Om, tau=0.0):
    return J_inv @ (torque_pend(R) + tau - cross(Om, J_p @ Om))


rp_mod, rp_layout, rp_par = library("rigid_pendulum")
rp_par.update({"J": J_p, "\\rho": rho_p, "m": m_p, "g": G})
worst = 0.0
for _ in range(10):
    R, Om, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    worst = max(worst, np.abs(accelerations(rp_mod, rp_layout.pack(R=R, Omega_R=Om), tau, rp_par) - accel_pend(R, Om, tau)).max())
ck.true("3D pendulum: continuous equation used here equals the geomech model", worst < 1e-9, f"err {worst:.1e}")
frb_mod, frb_layout, frb_par = library("free_rigid_body")
sp_mod, sp_layout, sp_par = library("spherical_pendulum")


def rk4_lie(R, Om, h, acc=accel_pend):
    """RK4 with the exponential update: stages on a rotation vector, one exponential at the end."""
    def f(th_, Om_):
        return dexpinv(th_, Om_), acc(R @ expm_so3(th_), Om_)
    z = np.zeros(3)
    k1 = f(z, Om)
    k2 = f(h / 2 * k1[0], Om + h / 2 * k1[1])
    k3 = f(h / 2 * k2[0], Om + h / 2 * k2[1])
    k4 = f(h * k3[0], Om + h * k3[1])
    return R @ expm_so3(h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])), Om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


def rk4_flat(R, Om, h, acc=accel_pend):
    def f(R_, Om_):
        return R_ @ hat(Om_), acc(R_, Om_)
    k1 = f(R, Om)
    k2 = f(R + h / 2 * k1[0], Om + h / 2 * k1[1])
    k3 = f(R + h / 2 * k2[0], Om + h / 2 * k2[1])
    k4 = f(R + h * k3[0], Om + h * k3[1])
    return R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), Om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


def angle_between(Ra, Rb):
    return float(np.linalg.norm(log_so3(Ra.T @ Rb)))


# short-horizon accuracy: 1 s, reference by RK4 at a tiny step
def run_rk4(h, T, R=R0_p, Om=Om0_p, step=rk4_lie):
    R, Om = R.copy(), Om.copy()
    for _ in range(int(round(T / h))):
        R, Om = step(R, Om, h)
    return R, Om


def run_lgvi(h, T, R=R0_p, p=p0_p, **kw):
    R, p = R.copy(), p.copy()
    for _ in range(int(round(T / h))):
        R, p, _ = lgvi_step(R, p, h, **kw)
    return R, p


R_ref, Om_ref = run_rk4(1e-4, 1.0)
err_rk, err_vi = {}, {}
for hh in (0.08, 0.04, 0.02, 0.01, 0.005):
    err_rk[hh] = angle_between(run_rk4(hh, 1.0)[0], R_ref)
    err_vi[hh] = angle_between(run_lgvi(hh, 1.0)[0], R_ref)
order_rk = np.log2(err_rk[0.02] / err_rk[0.01])
order_vi = np.log2(err_vi[0.02] / err_vi[0.01])
ck.true("LGVI is second order (attitude error after 1 s)", 1.8 < order_vi < 2.2, f"order {order_vi:.2f}")
ck.true("RK4 with the exponential update is fourth order", 3.6 < order_rk < 4.4, f"order {order_rk:.2f}")
ck.record("accuracy_1s_attitude_error_deg", {"rk4": {str(k): float(np.degrees(v)) for k, v in err_rk.items()},
                                              "lgvi": {str(k): float(np.degrees(v)) for k, v in err_vi.items()}})
ck.record("order_measured", {"rk4": float(order_rk), "lgvi": float(order_vi)})
ck.true("short horizon, equal step: RK4 is far more accurate than the LGVI", err_rk[0.02] < 0.01 * err_vi[0.02],
        f"{np.degrees(err_rk[0.02]):.4f} deg vs {np.degrees(err_vi[0.02]):.2f} deg at h = 0.02")
ck.true("short horizon, RK4 at twice the step (two torque evaluations per LGVI step) is still more accurate", err_rk[0.04] < 0.05 * err_vi[0.02],
        f"{np.degrees(err_rk[0.04]):.3f} deg vs {np.degrees(err_vi[0.02]):.2f} deg")
ck.true("short horizon, RK4 at four times the step (equal torque evaluations) is less accurate here", err_rk[0.08] > 5 * err_vi[0.02],
        f"{np.degrees(err_rk[0.08]):.1f} deg vs {np.degrees(err_vi[0.02]):.2f} deg")

# the hour
T_HOUR = float(os.environ.get("CH12_T", 3600.0))


def long_rk4(h, T, sample):
    R, Om = R0_p.copy(), Om0_p.copy()
    n = int(round(T / h))
    out = []
    for k in range(n):
        R, Om = rk4_lie(R, Om, h)
        if (k + 1) % sample == 0:
            out.append(energy_pend(R, J_p @ Om) - E0_3d)
    return np.array(out), R, Om


def long_lgvi(h, T):
    R, p = R0_p.copy(), p0_p.copy()
    n = int(round(T / h))
    lo = hi = 0.0
    pi3_err = 0.0
    pi3_0 = E3 @ (R @ p)
    first = None
    for k in range(n):
        R, p, _ = lgvi_step(R, p, h)
        d = energy_pend(R, p) - E0_3d
        lo, hi = min(lo, d), max(hi, d)
        pi3_err = max(pi3_err, abs(E3 @ (R @ p) - pi3_0))
        if k == n // 60:
            first = max(-lo, hi)
    return max(-lo, hi), first, pi3_err, R, p


NEWTON_ITS.clear()
t0 = time.time()
band_02, first_02, pi3_02, R_vi, p_vi = long_lgvi(0.02, T_HOUR)
its_02 = list(NEWTON_ITS)
band_01, _, _, _, _ = long_lgvi(0.01, 600.0)
band_04, _, _, _, _ = long_lgvi(0.04, 600.0)
NEWTON_ITS[:] = its_02
dE_08, R_08, Om_08 = long_rk4(0.08, T_HOUR, 125)        # one sample every 10 s
dE_02, R_02, Om_02 = long_rk4(0.02, T_HOUR, 500)
print(f"  (hour-long runs: {time.time() - t0:.0f} s)")
FULL = T_HOUR >= 3600.0
ck.true("LGVI, one hour at h = 0.02: energy band no wider than in the first minute", band_02 < 1.02 * first_02 or not FULL, f"band {band_02:.4f} J, first minute {first_02:.4f} J")
ck.true("LGVI: energy band scales as h^2", 3.7 < band_04 / band_02 < 4.3 and 3.7 < band_02 / band_01 < 4.3, f"ratios {band_04 / band_02:.2f}, {band_02 / band_01:.2f}")
ck.true("LGVI, one hour: R stays on SO(3) to rounding", np.linalg.norm(R_vi.T @ R_vi - I3) < 1e-11, f"{np.linalg.norm(R_vi.T @ R_vi - I3):.1e}")
ck.true("LGVI, one hour: e3 . R p conserved to rounding", pi3_02 < 1e-12, f"{pi3_02:.1e}")
ck.true("RK4 (exponential update), one hour at h = 0.08: the energy drains away", dE_08[-1] < -5.0 or not FULL, f"{dE_08[-1]:.2f} J of {E0_3d - E_rest:.2f} J")
ck.true("RK4 at h = 0.02 for one hour: drift is smaller than the LGVI band at the same step", abs(dE_02[-1]) < band_02, f"{dE_02[-1]:.4f} J vs band {band_02:.4f} J")
ck.true("RK4 at h = 0.02: the drift is monotone", np.all(np.diff(dE_02) < 0))
tilt_08 = np.degrees(np.arccos(np.clip(-E3 @ (R_08 @ -E3 * -1), -1, 1)))   # angle of R e3 from the vertical
swing_08 = np.degrees(np.arccos(np.clip((R_08 @ rho_p) @ np.array([0, 0, -1.0]) / 0.4, -1, 1)))
ck.record("hour", {"h_lgvi": 0.02, "h_rk4": 0.08, "T": T_HOUR, "E0_above_rest_J": float(E0_3d - E_rest)})
ck.record("hour_lgvi_band_J", float(band_02))
ck.record("hour_lgvi_band_first_minute_J", float(first_02))
ck.record("hour_lgvi_band_h0.04_J", float(band_04))
ck.record("hour_lgvi_band_h0.01_J", float(band_01))
ck.record("hour_lgvi_orth", float(np.linalg.norm(R_vi.T @ R_vi - I3)))
ck.record("hour_lgvi_pi3_err", float(pi3_02))
ck.record("hour_rk4_h0.08_dE_J", float(dE_08[-1]))
ck.record("hour_rk4_h0.08_dE_10min_J", float(dE_08[min(59, len(dE_08) - 1)]))
ck.record("hour_rk4_h0.08_left_J", float(E0_3d - E_rest + dE_08[-1]))
ck.record("hour_rk4_h0.08_swing_end_deg", float(swing_08))
ck.record("hour_rk4_h0.08_pi3_change", float(E3 @ (R_08 @ J_p @ Om_08) - E3 @ (R0_p @ p0_p)))
ck.record("hour_rk4_h0.02_dE_J", float(dE_02[-1]))
ck.record("hour_rk4_h0.02_dE_1min_J", float(long_rk4(0.02, 60.0, 3000)[0][-1]))
ck.true("RK4 at h = 0.08 ends in a steady tilted spin: energy flat over the last 10 minutes", abs(dE_08[-1] - dE_08[-60]) < 1e-3 or not FULL,
        f"center of mass {swing_08:.1f} deg from straight down, change {abs(dE_08[-1] - dE_08[-60]):.1e} J")
ck.record("hour_rk4_h0.08_dE_30min_J", float(dE_08[min(179, len(dE_08) - 1)]))
ck.record("newton_iterations_max", int(max(NEWTON_ITS)))
ck.true("Newton on the implicit equation: at most 8 iterations per step over the hour at h = 0.02", max(NEWTON_ITS) <= 8, f"mean {np.mean(NEWTON_ITS):.1f}, max {max(NEWTON_ITS)}")
ck.record("newton_iterations_mean", float(np.mean(NEWTON_ITS)))

# free rigid body: the book of Chapter 4
J_b = np.diag([0.445e-3, 1.0225e-3, 1.4225e-3])
Om0_b = 2 * np.pi * 2 * np.array([0.01, 1.0, 0.01])
p0_b = J_b @ Om0_b


def book_run(h, T):
    R, p = I3.copy(), p0_b.copy()
    n = int(round(T / h))
    E0 = 0.5 * p @ np.linalg.solve(J_b, p)
    mom = eband = 0.0
    flips = []
    prev = Om0_b[1]
    for k in range(n):
        R, p, Om = lgvi_step(R, p, h, J=J_b, torque=torque_none)
        mom = max(mom, np.abs(R @ p - p0_b).max())
        eband = max(eband, abs(0.5 * p @ np.linalg.solve(J_b, p) - E0))
        w2 = p[1] / J_b[1, 1]
        if prev > 0 >= w2 or prev < 0 <= w2:
            flips.append((k + 1) * h)
        prev = w2
    return mom, eband, flips, E0, R


mom_b, eb_b, flips_b, E0_b, R_b = book_run(0.01, 100.0)
ck.true("free body: world-frame angular momentum R p conserved to rounding (10 000 steps)", mom_b < 1e-15 / 1e-3 * 1e-0, f"{mom_b:.1e} of {np.linalg.norm(p0_b):.1e}")
ck.true("free body: the energy is conserved to rounding as well (Moser-Veselov)", eb_b / E0_b < 1e-10, f"relative {eb_b / E0_b:.1e}")
ck.true("free body: the book still flips, at the times of Chapter 4", abs(flips_b[0] - 0.8805) < 0.02 and abs(flips_b[1] - flips_b[0] - 1.545) < 0.03,
        f"first {flips_b[0]:.2f} s, then every {flips_b[1] - flips_b[0]:.2f} s")
ck.record("book", {"h": 0.01, "T": 100.0, "E0_J": float(E0_b), "momentum_Nms": float(np.linalg.norm(p0_b))})
ck.record("book_momentum_err", float(mom_b))
ck.record("book_energy_band_rel", float(eb_b / E0_b))
ck.record("book_first_flip_s", float(flips_b[0]))
ck.record("book_flip_count_100s", len(flips_b))
# coarse step: 25 steps per revolution, and the momentum still stays on its sphere and on a closed curve
def book_coarse(h, n, method):
    R, p = I3.copy(), p0_b.copy()
    Om = Om0_b.copy()
    out = []
    for k in range(n):
        if method == "lgvi":
            R, p, _ = lgvi_step(R, p, h, J=J_b, torque=torque_none)
        else:                                   # explicit Euler on Euler's equation
            Om = Om + h * np.linalg.solve(J_b, -cross(Om, J_b @ Om))
            p = J_b @ Om
        out.append(p.copy())
    return np.array(out)
pc = book_coarse(0.02, 2000, "lgvi")
E_c = 0.5 * np.einsum("ij,ij->i", pc, pc / np.diag(J_b))
ck.true("free body at h = 0.02 (25 steps per turn): |p| constant to rounding, energy to rounding", np.abs(np.linalg.norm(pc, axis=1) - np.linalg.norm(p0_b)).max() < 1e-14
        and np.abs(E_c / E0_b - 1).max() < 1e-10, f"energy error {np.abs(E_c / E0_b - 1).max():.1e}")
pe = book_coarse(0.02, 150, "euler")
ck.true("explicit Euler on Euler's equation at the same step: |p| grows", np.linalg.norm(pe[-1]) > 1.2 * np.linalg.norm(p0_b), f"x{np.linalg.norm(pe[-1]) / np.linalg.norm(p0_b):.1f} in 3 s")
ck.record("book_coarse_energy_band_rel", float(np.abs(E_c / E0_b - 1).max()))
ck.record("book_euler_momentum_growth_3s", float(np.linalg.norm(pe[-1]) / np.linalg.norm(p0_b)))

# the explicit update of Siravuru (2019), eq. (5.36), on the same 3D pendulum
def thesis_step(R, p, h, torque=torque_pend, J=J_p):
    U = expm_so3(h * np.linalg.solve(J, p))
    R1 = R @ U
    return R1, U.T @ p + h * torque(R1)


def thesis_run(h, T):
    R, p = R0_p.copy(), p0_p.copy()
    pi3_0 = E3 @ (R @ p)
    pi3 = 0.0
    for _ in range(int(round(T / h))):
        R, p = thesis_step(R, p, h)
        pi3 = max(pi3, abs(E3 @ (R @ p) - pi3_0))
    return energy_pend(R, p) - E0_3d, pi3, np.linalg.norm(R.T @ R - I3)


th_dE = {hh: thesis_run(hh, 60.0) for hh in (0.02, 0.01, 0.001)}
ck.true("explicit exponential update: on SO(3) and e3 . R p conserved, to rounding", th_dE[0.02][2] < 1e-12 and th_dE[0.02][1] < 1e-12,
        f"orth {th_dE[0.02][2]:.1e}, momentum {th_dE[0.02][1]:.1e}")
ck.true("explicit exponential update: the energy of the spinning pendulum drifts, in proportion to h", th_dE[0.001][0] > 1.0 and 1.7 < th_dE[0.02][0] / th_dE[0.01][0] < 2.4,
        f"+{th_dE[0.02][0]:.0f} J, +{th_dE[0.01][0]:.0f} J, +{th_dE[0.001][0]:.1f} J in 60 s at h = 0.02, 0.01, 0.001")
ck.record("explicit_update_energy_gain_60s_J", {str(k): float(v[0]) for k, v in th_dE.items()})
Om, dOm = rng.normal(size=3), rng.normal(size=3)
rel = []
for hh in (0.1, 0.01):
    dF = (expm_so3(hh * (Om + 1e-6 * dOm)) - expm_so3(hh * (Om - 1e-6 * dOm))) / 2e-6
    rel.append(np.linalg.norm(vee(dF @ expm_so3(hh * Om).T) - hh * dOm) / np.linalg.norm(hh * dOm))
ck.true("d exp(h hat(Om)) = h hat(dOm) exp(h hat(Om)) holds only to first order in h", 8 < rel[0] / rel[1] < 12 and rel[0] > 1e-3, f"relative error {rel[0]:.3f} at h|Om| = {0.1 * np.linalg.norm(Om):.2f}")
# free book, explicit exponential update: |p| exact, energy climbs to the largest value on the momentum sphere
p_e = 0.0128 * np.array([0.5, 0.0, np.sqrt(0.75)])
E_e = [0.5 * p_e @ (p_e / np.diag(J_b))]
for _ in range(260):
    p_e = expm_so3(0.02 * p_e / np.diag(J_b)).T @ p_e
    E_e.append(0.5 * p_e @ (p_e / np.diag(J_b)))
ck.true("free book, explicit exponential update at h = 0.02: |p| kept, energy rises at every step toward the spin about e1",
        abs(np.linalg.norm(p_e) - 0.0128) < 1e-15 and np.all(np.diff(E_e) > -1e-18) and abs(p_e[0]) / 0.0128 > 0.999,
        f"energy x{E_e[-1] / E_e[0]:.2f}, p along e1 to {abs(p_e[0]) / 0.0128:.5f}")
ck.record("book_explicit_energy_ratio", float(E_e[-1] / E_e[0]))
ck.record("hour_pi3_initial", float(E3 @ (R0_p @ p0_p)))
# the same one-sided discrete Lagrangian with the implicit equation solved exactly: bounded
def onesided_run(h, T):
    R, p = R0_p.copy(), p0_p.copy()
    band = 0.0
    for _ in range(int(round(T / h))):
        U = expm_so3(h * solve_step(p, h, J_p))
        R = R @ U
        p = U.T @ p + h * torque_pend(R)
        band = max(band, abs(energy_pend(R, p) - E0_3d))
    return band
os_band = onesided_run(0.02, 60.0)
ck.true("same update with the step rotation from the implicit equation: energy stays in a band", os_band < 0.5, f"band {os_band:.2f} J")
ck.record("onesided_implicit_band_60s_J", float(os_band))

# =========================================================================
# 4. The spherical pendulum on S^2  (m = 1 kg, l = 1 m)
m_s, l_s = 1.0, 1.0


def Lh_s2(qa, qb, h):
    return 0.5 * m_s * l_s**2 * (qb - qa) @ (qb - qa) / h - 0.5 * h * m_s * G * l_s * (E3 @ qa + E3 @ qb)


def s2_step(q, p, h):
    """Variational step on the sphere. p is the angular momentum about the pivot, world frame, p . q = 0."""
    pb = p - 0.5 * h * m_s * G * l_s * cross(q, E3)          # half kick
    w = pb * h / (m_s * l_s**2)                              # this is q_k x q_{k+1}
    q1 = cross(w, q) + np.sqrt(1 - w @ w) * q
    return q1, pb - 0.5 * h * m_s * G * l_s * cross(q1, E3)


def s2_energy(q, p):
    return 0.5 * p @ p / (m_s * l_s**2) + m_s * G * l_s * E3 @ q


q0_s = np.array([np.sin(1.0), 0.0, -np.cos(1.0)])
om0_s = cross(q0_s, np.array([0.0, 1.6, 0.0]))            # velocity q_dot = (0, 1.6, 0), tangent at q0
p0_s = m_s * l_s**2 * om0_s
h_s = 0.01
qs, pss = [q0_s], [p0_s]
for k in range(6):
    q1, p1 = s2_step(qs[-1], pss[-1], h_s)
    qs.append(q1)
    pss.append(p1)


def fd_s2(fun, q, e=1e-6):
    """Slope of fun(exp(eps hat(xi)) q) in xi."""
    return np.array([(fun(expm_so3(e * I3[i]) @ q) - fun(expm_so3(-e * I3[i]) @ q)) / (2 * e) for i in range(3)])


worst = 0.0
for k in range(1, 6):
    g = fd_s2(lambda y: Lh_s2(qs[k - 1], y, h_s) + Lh_s2(y, qs[k + 1], h_s), qs[k])
    worst = max(worst, np.abs(g - (g @ qs[k]) * qs[k]).max())
ck.true("S2: action sum stationary under q_k -> exp(eps hat(xi)) q_k (finite differences)", worst < 1e-8, f"residual {worst:.1e}")
k = 3
ck.close("S2: p_k is minus the slope of L_h in its first slot", -fd_s2(lambda y: Lh_s2(y, qs[k + 1], h_s), qs[k]), pss[k], tol=1e-7)
ck.close("S2: p_k+1 is the slope of L_h in its second slot", fd_s2(lambda y: Lh_s2(qs[k], y, h_s), qs[k + 1]), pss[k + 1], tol=1e-7)
q, p = q0_s, p0_s
E0s, pz0 = s2_energy(q, p), E3 @ p
nrm = pz = bs = tang = 0.0
for _ in range(100000):
    q, p = s2_step(q, p, h_s)
    nrm = max(nrm, abs(q @ q - 1))
    pz = max(pz, abs(E3 @ p - pz0))
    bs = max(bs, abs(s2_energy(q, p) - E0s))
    tang = max(tang, abs(p @ q))
ck.true("S2, 100 000 steps: |q| = 1, p . q = 0 and e3 . p conserved to rounding", nrm < 1e-12 and pz < 1e-11 and tang < 1e-12, f"{nrm:.1e}, {tang:.1e}, {pz:.1e}")
ck.true("S2, 100 000 steps: energy in a band", bs < 2e-3, f"band {bs:.1e} J of {E0s + m_s * G * l_s:.2f} J above rest")
ck.record("s2", {"h": h_s, "steps": 100000, "norm_err": float(nrm), "pz_err": float(pz), "energy_band_J": float(bs), "E0_above_rest_J": float(E0s + m_s * G * l_s)})
# against the continuous model of Chapter 4 (geomech) over 2 s
sp_par.update(m=m_s, l=l_s, g=G)
def s2_accel(q, om):
    return -(G / l_s) * cross(q, E3)
worst = 0.0
for _ in range(10):
    qq = rng.normal(size=3)
    qq /= np.linalg.norm(qq)
    om = cross(qq, rng.normal(size=3))
    a = accelerations(sp_mod, sp_layout.pack(q=qq, omega_q=om), np.zeros(3), sp_par)
    worst = max(worst, np.abs(a - (a @ qq) * qq - s2_accel(qq, om)).max())
ck.true("spherical pendulum: m l^2 om_dot = -m g l q x e3 equals the geomech model", worst < 1e-9, f"err {worst:.1e}")
def s2_ref(T, h=1e-4):
    q, om = q0_s.copy(), om0_s.copy()
    f = lambda q, om: (cross(om, q), s2_accel(q, om))
    for _ in range(int(round(T / h))):
        k1 = f(q, om); k2 = f(q + h / 2 * k1[0], om + h / 2 * k1[1]); k3 = f(q + h / 2 * k2[0], om + h / 2 * k2[1]); k4 = f(q + h * k3[0], om + h * k3[1])
        q = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); om = om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return q
q_ref = s2_ref(2.0)
errs = []
for hh in (0.02, 0.01):
    q, p = q0_s, p0_s
    for _ in range(int(round(2.0 / hh))):
        q, p = s2_step(q, p, hh)
    errs.append(np.linalg.norm(q - q_ref))
ck.true("S2 variational step is second order", 3.6 < errs[0] / errs[1] < 4.4, f"error ratio {errs[0] / errs[1]:.2f}")

# =========================================================================
# 5. Collocation: orders on a scalar equation, and what entrywise interpolation does
def defect_trap(xa, xb, h, f):
    return xb - xa - 0.5 * h * (f(xa) + f(xb))


def defect_hs(xa, xb, h, f):
    xm = 0.5 * (xa + xb) + h / 8 * (f(xa) - f(xb))
    return xb - xa - h / 6 * (f(xa) + 4 * f(xm) + f(xb))


f_log = lambda x: x * (1 - x)                    # x_dot = x (1 - x), exact solution known
x_log = lambda t: 1 / (1 + 9 * np.exp(-t))       # x(0) = 0.1
d_tr = [abs(defect_trap(x_log(3.0), x_log(3.0 + hh), hh, f_log)) for hh in (0.2, 0.1)]
d_hs = [abs(defect_hs(x_log(3.0), x_log(3.0 + hh), hh, f_log)) for hh in (0.2, 0.1)]
ck.true("trapezoidal defect of the exact solution is O(h^3)", 6.5 < d_tr[0] / d_tr[1] < 9, f"ratio {d_tr[0] / d_tr[1]:.2f}")
ck.true("Hermite-Simpson defect of the exact solution is O(h^5)", 28 < d_hs[0] / d_hs[1] < 36, f"ratio {d_hs[0] / d_hs[1]:.2f}")
ck.record("collocation_defect_h0.1", {"trapezoid": float(d_tr[1]), "hermite_simpson": float(d_hs[1])})
# cubic through the knots with the right slopes: its midpoint is the Hermite-Simpson midpoint
xa, xb, hh = 0.3, 0.9, 0.5
fa, fb = f_log(xa), f_log(xb)
c = np.linalg.solve(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, hh, hh**2, hh**3], [0, 1, 2 * hh, 3 * hh**2]]), [xa, fa, xb, fb])
ck.close("Hermite cubic at the midpoint: (xa + xb)/2 + h (fa - fb)/8", np.polyval(c[::-1], hh / 2), 0.5 * (xa + xb) + hh / 8 * (fa - fb), tol=1e-12)
ck.close("slope of the Hermite cubic at the midpoint: 3 (xb - xa)/(2h) - (fa + fb)/4", np.polyval(np.polyder(c[::-1]), hh / 2), 1.5 * (xb - xa) / hh - 0.25 * (fa + fb), tol=1e-12)
# midpoint of two attitudes, taken entrywise
for deg in (30.0, 90.0):
    Ra = random_rotation(rng)
    Rb = Ra @ expm_so3(np.radians(deg) * np.array([0.6, 0.0, 0.8]))
    Mid = 0.5 * (Ra + Rb)
    sv = np.linalg.svd(Mid, compute_uv=False)
    ck.close(f"entrywise midpoint of attitudes {deg:.0f} deg apart: singular values 1, cos, cos of half the angle", np.sort(sv), np.sort([1, np.cos(np.radians(deg) / 2)] * 1 + [np.cos(np.radians(deg) / 2)]), tol=1e-12)
    ck.record(f"midpoint_det_{deg:.0f}deg", float(np.linalg.det(Mid)))
qa, qb = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
ck.close("midpoint of two directions 90 deg apart has length cos 45 deg", np.linalg.norm(0.5 * (qa + qb)), np.cos(np.pi / 4), tol=1e-14)
ck.record("midpoint_direction_length_90deg", float(np.cos(np.pi / 4)))
# an entrywise trapezoidal step: R_{k+1} (I - h/2 hat(Om1)) = R_k (I + h/2 hat(Om0))
Om_a, Om_b, hh = np.array([0.0, 0.0, 2.0]), np.array([1.5, 0.0, 2.0]), 0.1
R1 = (I3 + 0.5 * hh * hat(Om_a)) @ np.linalg.inv(I3 - 0.5 * hh * hat(Om_b))
R1c = (I3 + 0.5 * hh * hat(Om_a)) @ np.linalg.inv(I3 - 0.5 * hh * hat(Om_a))
ck.true("entrywise trapezoid: a rotation only if the two rates agree (then it is the Cayley map)", np.linalg.norm(R1c.T @ R1c - I3) < 1e-14 and np.linalg.norm(R1.T @ R1 - I3) > 1e-3,
        f"|R'R - I| = {np.linalg.norm(R1.T @ R1 - I3):.1e} when they differ")

# =========================================================================
# 6. A reorientation transcribed three ways and solved with SLSQP
J_o = J_p
Jbar_o = Jbar_p
N_o, tf_o = 20, 2.0
h_o = tf_o / N_o
TAU_MAX = 0.5
axis_o = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
ang_o = np.radians(150.0)
Rf_o = expm_so3(ang_o * axis_o)


def cost_tau(tau):
    """Trapezoidal sum of |tau|^2 / 2."""
    wgt = np.full(N_o + 1, h_o)
    wgt[0] = wgt[-1] = h_o / 2
    return 0.5 * np.sum(wgt[:, None] * tau**2)


def replay(tau, n_sub=200):
    """Fly the torques, linear between knots, with a fine RK4. Returns attitudes and rates at the knots."""
    R, Om = I3.copy(), np.zeros(3)
    Rk, Ok = [R.copy()], [Om.copy()]
    hs = h_o / n_sub
    for k in range(N_o):
        for j in range(n_sub):
            def f(th_, Om_, s):
                tq = tau[k] + (tau[k + 1] - tau[k]) * (j + s) / n_sub
                return dexpinv(th_, Om_), J_inv @ (tq - cross(Om_, J_o @ Om_))
            z = np.zeros(3)
            k1 = f(z, Om, 0.0)
            k2 = f(hs / 2 * k1[0], Om + hs / 2 * k1[1], 0.5)
            k3 = f(hs / 2 * k2[0], Om + hs / 2 * k2[1], 0.5)
            k4 = f(hs * k3[0], Om + hs * k3[1], 1.0)
            R = R @ expm_so3(hs / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
            Om = Om + hs / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        Rk.append(R.copy())
        Ok.append(Om.copy())
    return np.array(Rk), np.array(Ok)


def euler_rhs(Om, tau):
    return tau - cross(Om, J_o @ Om)


# initial guess shared by all: constant-rate turn along the geodesic, zero torque
s_guess = np.linspace(0, 1, N_o + 1)
eta_guess = np.outer(s_guess, ang_o * axis_o)
Om_guess_knot = np.tile(ang_o * axis_o / tf_o, (N_o + 1, 1))
Om_guess_knot[0] = Om_guess_knot[-1] = 0
solutions = {}

# --- (a) trapezoidal collocation in one exponential chart: R_k = exp(hat(eta_k))
def unpack_chart(z):
    z = z.reshape(N_o + 1, 9)
    return z[:, :3], z[:, 3:6], z[:, 6:]


def make_chart(exact):
    def cons(z):
        eta, Om, tau = unpack_chart(z)
        ed = np.array([dexpinv(eta[k], Om[k]) if exact else Om[k] for k in range(N_o + 1)])
        fd = np.array([euler_rhs(Om[k], tau[k]) for k in range(N_o + 1)])
        c = [eta[1:] - eta[:-1] - 0.5 * h_o * (ed[1:] + ed[:-1]), (Om[1:] - Om[:-1]) @ J_o - 0.5 * h_o * (fd[1:] + fd[:-1]),
             eta[:1], Om[:1], Om[-1:], log_so3(Rf_o.T @ expm_so3(eta[-1]))[None]]
        return np.concatenate([a.ravel() for a in c])
    return cons


def solve(name, cons, z0, tau_of, nvar):
    t0 = time.time()
    bounds = [(None, None)] * nvar
    for i in tau_of(np.arange(nvar)).ravel():
        bounds[int(i)] = (-TAU_MAX, TAU_MAX)
    res = minimize(lambda z: cost_tau(tau_of(z)), z0, method="SLSQP", constraints=[{"type": "eq", "fun": cons}], bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10})
    solutions[name] = dict(res=res, time=time.time() - t0, cons=float(np.abs(cons(res.x)).max()), tau=tau_of(res.x).copy())
    return res


z0_chart = np.concatenate([eta_guess, Om_guess_knot, np.zeros((N_o + 1, 3))], axis=1).ravel()
tau_chart = lambda z: z.reshape(N_o + 1, 9)[:, 6:]
res_a = solve("chart", make_chart(True), z0_chart, tau_chart, z0_chart.size)
res_a1 = solve("chart_first_order", make_chart(False), z0_chart, tau_chart, z0_chart.size)

# --- (b) entrywise trapezoid on the nine entries of R
def cons_flat(z):
    z = z.reshape(N_o + 1, 15)
    R, Om, tau = z[:, :9].reshape(-1, 3, 3), z[:, 9:12], z[:, 12:]
    Rd = np.array([R[k] @ hat(Om[k]) for k in range(N_o + 1)])
    fd = np.array([euler_rhs(Om[k], tau[k]) for k in range(N_o + 1)])
    c = [R[1:] - R[:-1] - 0.5 * h_o * (Rd[1:] + Rd[:-1]), (Om[1:] - Om[:-1]) @ J_o - 0.5 * h_o * (fd[1:] + fd[:-1]),
         R[0] - I3, Om[:1], Om[-1:], R[-1] - Rf_o]
    return np.concatenate([a.ravel() for a in c])


R_guess = np.array([expm_so3(e) for e in eta_guess])
z0_flat = np.concatenate([R_guess.reshape(N_o + 1, 9), Om_guess_knot, np.zeros((N_o + 1, 3))], axis=1).ravel()
tau_flat = lambda z: z.reshape(N_o + 1, 15)[:, 12:]
res_b = solve("entrywise", cons_flat, z0_flat, tau_flat, z0_flat.size)

# --- (c) variational: unknowns are the step velocities Omega_k and the torques
def unpack_vi(z):
    return z[:3 * N_o].reshape(N_o, 3), z[3 * N_o:].reshape(N_o + 1, 3)


def vi_rollout(Om):
    R = [I3]
    for k in range(N_o):
        R.append(R[-1] @ expm_so3(h_o * Om[k]))
    return R


def cons_vi(z):
    Om, tau = unpack_vi(z)
    R = vi_rollout(Om)
    gk = np.array([step_momentum(Om[k], h_o, J_o) for k in range(N_o)])
    c = [gk[0] - 0.5 * h_o * tau[0]]                                            # p_0 = 0
    for k in range(1, N_o):
        c.append(gk[k] - (R[k - 1].T @ R[k]).T @ gk[k - 1] - h_o * tau[k])      # discrete Euler-Lagrange at knot k
    c.append((R[N_o - 1].T @ R[N_o]).T @ gk[N_o - 1] + 0.5 * h_o * tau[N_o])    # p_N = 0
    c.append(log_so3(Rf_o.T @ R[N_o]))
    return np.concatenate(c)


z0_vi = np.concatenate([np.tile(ang_o * axis_o / tf_o, (N_o, 1)).ravel(), np.zeros(3 * (N_o + 1))])
tau_vi = lambda z: z[3 * N_o:].reshape(N_o + 1, 3)
res_c = solve("variational", cons_vi, z0_vi, tau_vi, z0_vi.size)

for name in ("chart", "chart_first_order", "entrywise", "variational"):
    s = solutions[name]
    ck.true(f"optimization ({name}): SLSQP converged, constraints satisfied", s["res"].success and s["cons"] < 1e-7,
            f"{s['res'].nit} iterations, {s['time']:.1f} s, residual {s['cons']:.1e}, cost {s['res'].fun:.4f}")
    Rk, Ok = replay(s["tau"])
    s["replay_R"], s["replay_Om"] = Rk, Ok
    s["end_angle_deg"] = float(np.degrees(angle_between(Rk[-1], Rf_o)))
    s["end_rate"] = float(np.linalg.norm(Ok[-1]))
    s["bound_active"] = int(np.sum(np.abs(np.abs(s["tau"]) - TAU_MAX) < 1e-6))

# the entrywise transcription has dependent constraints at its solution
def num_jac(fun, z, e=1e-6):
    f0 = fun(z)
    Jm = np.empty((f0.size, z.size))
    for i in range(z.size):
        d = np.zeros(z.size)
        d[i] = e
        Jm[:, i] = (fun(z + d) - fun(z - d)) / (2 * e)
    return Jm
sv_flat = np.linalg.svd(num_jac(cons_flat, res_b.x), compute_uv=False)
sv_vi = np.linalg.svd(num_jac(cons_vi, res_c.x), compute_uv=False)
n_dep = int(np.sum(sv_flat < 1e-6 * sv_flat[0]))
ck.true("entrywise transcription: some equality constraints are dependent at the solution; variational: none", n_dep > 0 and sv_vi.min() > 1e-6 * sv_vi[0],
        f"{n_dep} of {sv_flat.size} dependent; variational smallest/largest singular value {sv_vi.min() / sv_vi[0]:.1e}")
ck.record("opt_entrywise_dependent_constraints", [n_dep, int(sv_flat.size)])
# knot attitudes of each transcription
eta_a, Om_a_sol, _ = unpack_chart(res_a.x)
solutions["chart"]["R"] = np.array([expm_so3(e) for e in eta_a])
solutions["chart_first_order"]["R"] = np.array([expm_so3(e) for e in unpack_chart(res_a1.x)[0]])
solutions["entrywise"]["R"] = res_b.x.reshape(N_o + 1, 15)[:, :9].reshape(-1, 3, 3)
Om_c_sol, tau_c_sol = unpack_vi(res_c.x)
solutions["variational"]["R"] = np.array(vi_rollout(Om_c_sol))
for name, s in solutions.items():
    s["orth"] = float(max(np.linalg.norm(R.T @ R - I3) for R in s["R"]))
    if name == "entrywise":
        s["knot_mismatch_deg"] = float(max(np.degrees(np.linalg.norm(vee(Ra.T @ Rb - Rb.T @ Ra) / 2)) for Ra, Rb in zip(s["R"], s["replay_R"])))
    else:
        s["knot_mismatch_deg"] = float(max(np.degrees(angle_between(Ra, Rb)) for Ra, Rb in zip(s["R"], s["replay_R"])))

# momentum and energy bookkeeping of the variational solution
p_knots = [np.zeros(3)]
Rv = solutions["variational"]["R"]
for k in range(N_o):
    p_knots.append((Rv[k].T @ Rv[k + 1]).T @ (p_knots[-1] + 0.5 * h_o * tau_c_sol[k]) + 0.5 * h_o * tau_c_sol[k + 1])
p_knots = np.array(p_knots)
ck.true("variational solution: momentum returns to zero at the end", np.abs(p_knots[-1]).max() < 1e-8, f"{np.abs(p_knots[-1]).max():.1e}")
# discrete momentum balance in the world frame: R_N p_N - R_0 p_0 = sum of trapezoid impulses R_k tau_k
imp = sum((h_o if 0 < k < N_o else h_o / 2) * Rv[k] @ tau_c_sol[k] for k in range(N_o + 1))
ck.close("variational solution: world-frame momentum change equals the summed impulses R_k tau_k h, exactly", Rv[-1] @ p_knots[-1] - imp, 0, tol=1e-9)
E_vi = 0.5 * np.einsum("ij,ij->i", p_knots, p_knots / np.diag(J_o))
Rr, Or = solutions["variational"]["replay_R"], solutions["variational"]["replay_Om"]
E_rep = 0.5 * np.einsum("ij,ij->i", Or @ J_o, Or)
solutions["variational"]["energy_gap_J"] = float(np.abs(E_vi - E_rep).max())
E_ch = 0.5 * np.einsum("ij,ij->i", Om_a_sol @ J_o, Om_a_sol)
Or_a = solutions["chart"]["replay_Om"]
solutions["chart"]["energy_gap_J"] = float(np.abs(E_ch - 0.5 * np.einsum("ij,ij->i", Or_a @ J_o, Or_a)).max())
ck.record("opt", {"N": N_o, "h": h_o, "tf": tf_o, "tau_max": TAU_MAX, "angle_deg": 150.0, "peak_energy_J": float(E_rep.max()),
                  "peak_rate_rad_s": float(np.linalg.norm(Or, axis=1).max())})
for name, s in solutions.items():
    ck.record(f"opt_{name}", {"iterations": int(s["res"].nit), "variables": int(s["res"].x.size), "cost": float(s["res"].fun),
                              "constraint_residual": s["cons"], "orth_knots": s["orth"], "end_angle_deg": s["end_angle_deg"], "end_rate_rad_s": s["end_rate"],
                              "knot_mismatch_deg": s["knot_mismatch_deg"], "bound_active_components": s["bound_active"],
                              "energy_gap_J": s.get("energy_gap_J")})
ck.true("the torque bound is active in every solution", all(s["bound_active"] > 0 for s in solutions.values()), str({k: v["bound_active"] for k, v in solutions.items()}))
ck.true("chart and variational transcriptions: knot attitudes are rotations to rounding", solutions["chart"]["orth"] < 1e-12 and solutions["variational"]["orth"] < 1e-12)
ck.true("entrywise transcription: knot matrices are not rotations", solutions["entrywise"]["orth"] > 1e-3, f"max |R'R - I| = {solutions['entrywise']['orth']:.3f}")
ck.true("replayed with a fine integrator, exact-chart and variational plans both miss the target by under two degrees",
        solutions["chart"]["end_angle_deg"] < 2.0 and solutions["variational"]["end_angle_deg"] < 2.0,
        f"chart {solutions['chart']['end_angle_deg']:.3f} deg, variational {solutions['variational']['end_angle_deg']:.3f} deg")
ck.true("the first-order chart (eta_dot = Omega) misses by much more", solutions["chart_first_order"]["end_angle_deg"] > 5 * solutions["chart"]["end_angle_deg"],
        f"{solutions['chart_first_order']['end_angle_deg']:.1f} deg")
# the chart has an edge: dexp is singular at |eta| = 2 pi
def dexp_mat(th):
    return np.linalg.inv(np.column_stack([dexpinv(th, I3[i]) for i in range(3)]))
sv = np.linalg.svd(dexp_mat((2 * np.pi - 1e-3) * axis_o), compute_uv=False)
ck.true("exponential chart: its Jacobian loses rank as |eta| reaches a full turn", sv.min() < 1e-3, f"smallest singular value {sv.min():.1e}")
np.savez(os.path.join(os.path.dirname(__file__), "ch12_opt.npz"), **{f"tau_{k}": v["tau"] for k, v in solutions.items()},
         **{f"R_{k}": v["R"] for k, v in solutions.items()})

ck.record("geomech_check_eom", geomech_errors)
ck.finish()
