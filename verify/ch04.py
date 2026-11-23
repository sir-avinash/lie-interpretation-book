"""Chapter 4: every equation of motion, identity and quoted number, checked.

    python verify/ch04.py

Each equation of motion is derived a second time with geomech (compute_eom, check_eom)
and the hand-written form in the chapter's notation is compared with the accelerations
of the geomech model at random states.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from geomech import (SO3, Cross, Dot, Matrix, Scalar, SystemVariables, Vector, check_eom, compute_eom,
                     getScalars, systems, to_standard_form)
from geomech.codegen import StateLayout, collect_params, load_module, module_source
from geomech.simulation import accelerations

from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee

ck = Checker("ch04")
rng = np.random.default_rng(4)
E3 = np.array([0.0, 0.0, 1.0])
G = 9.81
EOM_TOL = 1e-5


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def tangent(q, v):
    """Part of v perpendicular to q."""
    return v - (v @ q) * q


def plant_of(L, dW, variables, inputs, name):
    """Derive with geomech, run check_eom, return (module, layout, error)."""
    eom = compute_eom(L, dW, variables)
    err = check_eom(L, dW, variables, eom, inputs=inputs)
    sf = to_standard_form(eom, variables, inputs)
    layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
    return load_module(module_source(sf, layout, lagrangian=L), name), layout, err


def library(name, **kw):
    s = systems.get(name, **kw)
    mod, layout, err = plant_of(s.L, s.dW, s.variables, list(s.inputs), name)
    return mod, layout, err, dict(s.params)


geomech_errors = {}


def record_check(name, err):
    geomech_errors[name] = float(err)
    ck.true(f"geomech check_eom: {name}", err < EOM_TOL, f"err {err:.1e}")


# =========================================================================
# 1. The puzzle: the Euler-angle mass matrix of a 3D pendulum
# =========================================================================
m_p, rho_p = 1.0, np.array([0.0, 0.0, -0.4])
J_p = np.diag([0.20, 0.25, 0.10])          # about the pivot, body frame


def R_zyx(psi, th, phi):
    return Rz(psi) @ Ry(th) @ Rx(phi)


def E_mat(th, phi):
    return np.array([[-np.sin(th), 0, 1],
                     [np.cos(th) * np.sin(phi), np.cos(phi), 0],
                     [np.cos(th) * np.cos(phi), -np.sin(phi), 0]])


def D_euler(th, phi):
    Em = E_mat(th, phi)
    return Em.T @ J_p @ Em


# kinetic energy in the two descriptions agrees: (1/2) eta_dot' D eta_dot = (1/2) Omega' J Omega
eta, deta = np.array([0.4, -0.7, 1.1]), np.array([0.3, 0.2, -0.5])
h = 1e-6
Om_fd = vee(R_zyx(*eta).T @ (R_zyx(*(eta + h * deta)) - R_zyx(*(eta - h * deta))) / (2 * h))
ck.close("Euler kinetic energy: eta_dot' E'JE eta_dot = Omega' J Omega",
         deta @ D_euler(eta[1], eta[2]) @ deta, Om_fd @ J_p @ Om_fd, tol=1e-8)
ck.close("det(E'JE) = cos^2(theta) det J", np.linalg.det(D_euler(0.9, 0.3)), np.cos(0.9) ** 2 * np.linalg.det(J_p))

cond_J = np.linalg.cond(J_p)
ck.record("pendulum", {"m": m_p, "rho": rho_p.tolist(), "J_diag": np.diag(J_p).tolist()})
ck.record("cond_J", float(cond_J))
for deg in (0, 60, 85, 89, 89.9):
    c = np.linalg.cond(D_euler(np.radians(deg), np.radians(20.0)))
    ck.record(f"cond_euler_pitch_{deg}", float(c))
ck.true("cond(J) = 2.5", abs(cond_J - 2.5) < 1e-12)
c0, c89 = (np.linalg.cond(D_euler(np.radians(d), np.radians(20.0))) for d in (0, 89))
ck.true("cond(E'JE) is below 10 at level and above 1e4 at 89 deg of pitch (roll 20 deg)",
        c0 < 10 and c89 > 1e4, f"{c0:.2f}, {c89:.3g}")
c1, c2 = (np.linalg.cond(D_euler(np.radians(d), np.radians(20.0))) for d in (89.0, 89.9))
ck.true("cond(E'JE) grows like 1/cos^2(theta)", abs(c2 / c1 - (np.cos(np.radians(89)) / np.cos(np.radians(89.9))) ** 2) / (c2 / c1) < 0.01,
        f"ratio {c2 / c1:.1f}")

# The Euler-angle equations of motion, written out with sympy, and their size.
psi_s, th_s, phi_s = sp.symbols("psi theta phi")
dpsi_s, dth_s, dphi_s = sp.symbols("dpsi dtheta dphi")
ms, gs = sp.symbols("m g", positive=True)
Jsym = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"J{min(i, j) + 1}{max(i, j) + 1}"))
rho_sym = sp.Matrix(sp.symbols("rho1 rho2 rho3"))
Rs = (sp.rot_axis3(-psi_s) * sp.rot_axis2(-th_s) * sp.rot_axis1(-phi_s))      # sympy's rot_axis is the transpose
Es = sp.Matrix([[-sp.sin(th_s), 0, 1],
                [sp.cos(th_s) * sp.sin(phi_s), sp.cos(phi_s), 0],
                [sp.cos(th_s) * sp.cos(phi_s), -sp.sin(phi_s), 0]])
qs_, dqs_ = sp.Matrix([psi_s, th_s, phi_s]), sp.Matrix([dpsi_s, dth_s, dphi_s])
Oms = Es * dqs_
Ls = (sp.Rational(1, 2) * (Oms.T * Jsym * Oms))[0] - ms * gs * (Rs * rho_sym)[2]
Ds = Ls.diff(dqs_).jacobian(dqs_)                       # mass matrix E'JE
hs = Ls.diff(dqs_).jacobian(qs_) * dqs_ - sp.Matrix([Ls.diff(v) for v in qs_])   # D eta_dd + hs = 0


def count_terms(exprs, subs=None):
    return sum(len(sp.Add.make_args(sp.expand(ex.subs(subs or {})))) for ex in exprs)


n_terms = count_terms(list(Ds) + list(hs))
diag_case = {Jsym[0, 1]: 0, Jsym[0, 2]: 0, Jsym[1, 2]: 0, rho_sym[0]: 0, rho_sym[1]: 0}
n_terms_diag = count_terms(list(Ds) + list(hs), diag_case)
ck.record("euler_eom_terms_general", int(n_terms))
ck.record("euler_eom_terms_diagonal_J", int(n_terms_diag))
ck.true("expanded Euler-angle model: more than 100 terms (general J, rho)", n_terms > 100, f"{n_terms} terms; {n_terms_diag} for diagonal J and rho on e3")
ck.close("sympy R(eta) equals Rz Ry Rx", np.array(Rs.subs({psi_s: 0.4, th_s: -0.7, phi_s: 1.1}), float), R_zyx(*eta))

sub = {Jsym[i, j]: J_p[i, j] for i in range(3) for j in range(i, 3)}
sub.update({ms: m_p, gs: G, rho_sym[0]: rho_p[0], rho_sym[1]: rho_p[1], rho_sym[2]: rho_p[2]})
D_fn = sp.lambdify((qs_, dqs_), Ds.subs(sub))
h_fn = sp.lambdify((qs_, dqs_), hs.subs(sub))


def pendulum_accel(R, Om, tau=np.zeros(3)):
    """Chapter eq. (3D pendulum): J Om_dot + Om x J Om = m g (R' e3) x rho + tau."""
    return np.linalg.solve(J_p, -np.cross(Om, J_p @ Om) + m_p * G * np.cross(R.T @ E3, rho_p) + tau)


worst = 0.0
for _ in range(5):
    eta, deta = rng.uniform(-1.2, 1.2, 3), rng.normal(size=3)
    ddeta = np.linalg.solve(D_fn(eta, deta), -h_fn(eta, deta)).ravel()
    Em = E_mat(eta[1], eta[2])
    Edot = (E_mat(eta[1] + h * deta[1], eta[2] + h * deta[2]) - E_mat(eta[1] - h * deta[1], eta[2] - h * deta[2])) / (2 * h)
    Om_dot_euler = Em @ ddeta + Edot @ deta
    worst = max(worst, np.max(np.abs(Om_dot_euler - pendulum_accel(R_zyx(*eta), Em @ deta))))
ck.true("Euler-angle pendulum model and the SO(3) model give the same motion", worst < 1e-6, f"err {worst:.1e}")


# =========================================================================
# 2. Hamilton's principle on R^3: a thrown ball
# =========================================================================
m_b, T_b = 1.0, 1.0
x0_b, v0_b = np.zeros(3), np.array([2.0, 0.0, 0.5 * G * T_b])      # lands back at height 0 after T
tt = np.linspace(0, T_b, 20001)


def action_ball(eps):
    x = x0_b[None] + np.outer(tt, v0_b) - 0.5 * G * np.outer(tt**2, E3) + eps * np.outer(np.sin(np.pi * tt / T_b), E3)
    v = v0_b[None] - G * np.outer(tt, E3) + eps * (np.pi / T_b) * np.outer(np.cos(np.pi * tt / T_b), E3)
    return np.trapezoid(0.5 * m_b * np.sum(v * v, axis=1) - m_b * G * x[:, 2], tt)


S0 = action_ball(0.0)
for eps in (0.1, -0.1, 0.3):
    ck.close(f"ball: S(eps) - S(0) = m eps^2 pi^2 / (4T), eps = {eps}", action_ball(eps) - S0,
             m_b * eps**2 * np.pi**2 / (4 * T_b), tol=1e-6)
ck.record("ball_action_increase_eps_0.1", float(action_ball(0.1) - S0))
ck.true("ball: 10 cm bump raises the action by 0.0247 J s", abs(action_ball(0.1) - S0 - 0.0247) < 5e-5)


# =========================================================================
# 3. S^2: variations, the spherical pendulum
# =========================================================================
def curve_q(t):
    """A smooth test curve on the sphere (not a pendulum trajectory)."""
    return expm_so3(0.9 * np.sin(1.3 * t) * unit([1, 2, 0.5])) @ expm_so3(0.7 * t * E3) @ unit([0.6, 0.2, -0.77])


def field_xi(t, q):
    """A smooth variation field with xi . q = 0."""
    return tangent(q, np.array([np.sin(t), 0.4 * np.cos(2 * t), 0.3 + 0.5 * t]))


def ddt(fn, t, d=1e-5):
    return (fn(t + d) - fn(t - d)) / (2 * d)


def q_var(t, eps):
    q = curve_q(t)
    return expm_so3(eps * field_xi(t, q)) @ q


t0 = 0.37
q = curve_q(t0)
qd = ddt(curve_q, t0)
xi = field_xi(t0, q)
xid = ddt(lambda s: field_xi(s, curve_q(s)), t0)
om = np.cross(q, qd)
e = 1e-5
ck.close("S2: varied curve stays on the sphere", np.linalg.norm(q_var(t0, 0.3)), 1.0)
ck.close("S2: omega = q x q_dot reproduces q_dot = omega x q", np.cross(om, q), qd, tol=1e-9)
dq_fd = (q_var(t0, e) - q_var(t0, -e)) / (2 * e)
ck.close("S2: delta q = xi x q (finite difference)", dq_fd, np.cross(xi, q), tol=1e-8)
dqd_fd = (ddt(lambda s: q_var(s, e), t0) - ddt(lambda s: q_var(s, -e), t0)) / (2 * e)
ck.close("S2: delta q_dot = xi_dot x q + xi x q_dot (finite difference)", dqd_fd, np.cross(xid, q) + np.cross(xi, qd), tol=1e-5)


def omega_of(t, eps):
    return np.cross(q_var(t, eps), ddt(lambda s: q_var(s, eps), t))


dom_fd = (omega_of(t0, e) - omega_of(t0, -e)) / (2 * e)
ck.close("S2: delta omega = xi_dot + 2 (xi . q_dot) q (finite difference)", dom_fd, xid + 2 * (xi @ qd) * q, tol=1e-5)
qdd = (curve_q(t0 + 1e-4) - 2 * q + curve_q(t0 - 1e-4)) / 1e-8
ck.close("S2: q . q_ddot = -|q_dot|^2", q @ qdd, -qd @ qd, tol=1e-5)
omd = ddt(lambda s: np.cross(curve_q(s), ddt(curve_q, s)), t0)
ck.close("S2: q x q_ddot = omega_dot", np.cross(q, qdd), omd, tol=1e-5)
ck.close("S2: q . omega_dot = 0", q @ omd, 0.0, tol=1e-5)

# geomech: spherical pendulum
sp_mod, sp_layout, err, sp_p = library("spherical_pendulum")
record_check("spherical pendulum (S2)", err)
m_s, l_s = 0.7, 1.3
sp_p.update(m=m_s, l=l_s, g=G)


def sph_omega_dot(q, om, M=np.zeros(3)):
    """Chapter eq.: m l^2 omega_dot = -m g l q x e3 + (I - q q') M."""
    return (-m_s * G * l_s * np.cross(q, E3) + tangent(q, M)) / (m_s * l_s**2)


def sph_qddot(q, qd):
    """Chapter eq.: q_dd = -|q_dot|^2 q - (g/l)(I - q q') e3."""
    return -(qd @ qd) * q - (G / l_s) * tangent(q, E3)


worst = worst2 = worst3 = 0.0
for _ in range(20):
    q = unit(rng.normal(size=3))
    om = tangent(q, rng.normal(size=3))
    M = rng.normal(size=3)
    a_g = accelerations(sp_mod, sp_layout.pack(q=q, omega_q=om), M, sp_p)
    worst = max(worst, np.max(np.abs(a_g - sph_omega_dot(q, om, M))))
    qd = np.cross(om, q)
    od = sph_omega_dot(q, om)
    qdd_from_omega = np.cross(od, q) + np.cross(om, qd)
    worst2 = max(worst2, np.max(np.abs(qdd_from_omega - sph_qddot(q, qd))))
    # Newton with tension:  m l q_dd = -m g e3 - T q,  T = m l |q_dot|^2 - m g e3 . q
    Tn = m_s * l_s * (qd @ qd) - m_s * G * (E3 @ q)
    worst3 = max(worst3, np.max(np.abs(m_s * l_s * sph_qddot(q, qd) + m_s * G * E3 + Tn * q)))
ck.true("spherical pendulum: omega_dot form equals geomech", worst < 1e-9, f"err {worst:.1e}")
ck.true("spherical pendulum: q_ddot form equals omega_dot form", worst2 < 1e-9, f"err {worst2:.1e}")
ck.true("spherical pendulum: Newton's law with tension T = m l |q_dot|^2 - m g e3.q", worst3 < 1e-9, f"err {worst3:.1e}")


def sph_rhs(t, s):
    q, om = s[:3], s[3:]
    return np.concatenate([np.cross(om, q), sph_omega_dot(q, om)])


# the action on S^2 is stationary on a true trajectory, and not on another curve
sol = solve_ivp(sph_rhs, [0, 1.0], np.concatenate([unit([0.8, 0, -0.6]), tangent(unit([0.8, 0, -0.6]), [0.3, 1.5, 0.2])]),
                rtol=1e-12, atol=1e-13, dense_output=True)
ts = np.linspace(0, 1.0, 4001)


def action_s2(path, eps):
    """Action of the curve exp(eps * bump(t) * hat(xi)) path(t), with bump(0) = bump(1) = 0."""
    Q = np.array([expm_so3(eps * np.sin(np.pi * t) * field_xi(t, path(t))) @ path(t) for t in ts])
    Qd = np.gradient(Q, ts, axis=0, edge_order=2)
    return np.trapezoid(0.5 * m_s * l_s**2 * np.sum(Qd * Qd, axis=1) - m_s * G * l_s * Q[:, 2], ts)


true_path = lambda t: sol.sol(t)[:3]
dS_true = (action_s2(true_path, 1e-3) - action_s2(true_path, -1e-3)) / 2e-3
dS_other = (action_s2(curve_q, 1e-3) - action_s2(curve_q, -1e-3)) / 2e-3
ck.true("S2: dS/d eps = 0 along a pendulum trajectory", abs(dS_true) < 1e-4, f"{dS_true:.1e}")
# first-variation formula  dS = -int xi . (q x (m l^2 q_dd + m g l e3)) dt  on a curve that is not a solution
integrand = []
for t in ts:
    qq = curve_q(t)
    qdd_ = (curve_q(t + 1e-4) - 2 * qq + curve_q(t - 1e-4)) / 1e-8
    integrand.append(-np.sin(np.pi * t) * field_xi(t, qq) @ np.cross(qq, m_s * l_s**2 * qdd_ + m_s * G * l_s * E3))
ck.close("S2: first variation = -int xi . (q x (m l^2 q_dd + m g l e3)) dt", dS_other, np.trapezoid(integrand, ts), tol=2e-4)
ck.true("S2: the first variation is not zero on a non-solution", abs(dS_other) > 0.1, f"{dS_other:.2f}")

# worked example: 1 kg bob on a 1 m rod released from the horizontal: tension at the bottom is 3 m g
m_s, l_s = 1.0, 1.0
sol = solve_ivp(sph_rhs, [0, 3.0], np.concatenate([[1.0, 0, 0], np.zeros(3)]), rtol=1e-11, atol=1e-12, dense_output=True,
                events=lambda t, s: s[0])
t_bot = sol.t_events[0][0]
qb, ob = sol.sol(t_bot)[:3], sol.sol(t_bot)[3:]
T_bot = m_s * l_s * (ob @ ob) - m_s * G * (E3 @ qb)
ck.close("released from horizontal, tension at the bottom is 3 m g", T_bot, 3 * m_s * G, tol=1e-6)
ck.record("pendulum_bottom_tension_N", float(T_bot))
T_start = -m_s * G * (E3 @ np.array([1.0, 0, 0]))
ck.close("released from horizontal at rest, initial tension is zero", T_start, 0.0)

# conservation on a generic spherical-pendulum run: energy, vertical angular momentum, constraints
q0 = unit([0.8, 0.1, -0.5])
sol = solve_ivp(sph_rhs, [0, 10], np.concatenate([q0, tangent(q0, [0.5, 2.0, 0.3])]), rtol=1e-11, atol=1e-12)
Qs, Ws = sol.y[:3].T, sol.y[3:].T
En = 0.5 * m_s * l_s**2 * np.sum(Ws * Ws, axis=1) + m_s * G * l_s * Qs[:, 2]
ck.true("spherical pendulum: energy conserved", np.ptp(En) < 1e-7, f"spread {np.ptp(En):.1e}")
ck.true("spherical pendulum: e3 . omega (vertical angular momentum) conserved", np.ptp(Ws[:, 2]) < 1e-7, f"spread {np.ptp(Ws[:, 2]):.1e}")
ck.true("spherical pendulum: omega . q stays zero", np.max(np.abs(np.sum(Qs * Ws, axis=1))) < 1e-7)


# =========================================================================
# 4. SO(3): the variation of Omega, Euler's equation, forces and torques
# =========================================================================
def curve_R(t):
    return expm_so3(0.8 * np.sin(t) * unit([1, -1, 0.5])) @ expm_so3(1.1 * t * unit([0.2, 0.3, 1.0]))


def field_eta(t):
    return np.array([np.cos(t), 0.5 * t, np.sin(2 * t) - 0.2])


def R_var(t, eps):
    return curve_R(t) @ expm_so3(eps * field_eta(t))


def Omega_of(t, eps):
    return vee(R_var(t, eps).T @ ddt(lambda s: R_var(s, eps), t))


Om = Omega_of(t0, 0.0)
dOm_fd = (Omega_of(t0, e) - Omega_of(t0, -e)) / (2 * e)
ck.close("SO(3): delta R = R hat(eta)", (R_var(t0, e) - R_var(t0, -e)) / (2 * e), curve_R(t0) @ hat(field_eta(t0)), tol=1e-8)
ck.close("SO(3): delta Omega = eta_dot + Omega x eta (finite difference)", dOm_fd,
         ddt(field_eta, t0) + np.cross(Om, field_eta(t0)), tol=1e-5)
a, b, c = rng.normal(size=(3, 3))
ck.close("J Om . (Om x eta) = -eta . (Om x J Om)", (J_p @ a) @ np.cross(a, b), -b @ np.cross(a, J_p @ a))
R = random_rotation(rng)
ck.close("virtual work of a force at a body point: F . R (eta x rho) = eta . (rho x R'F)",
         a @ (R @ np.cross(b, c)), b @ np.cross(c, R.T @ a))
ck.close("potential of the pendulum: variation of m g e3 . R rho is m g eta . (rho x R'e3)",
         (m_p * G * E3 @ (R @ expm_so3(e * b) @ rho_p) - m_p * G * E3 @ (R @ expm_so3(-e * b) @ rho_p)) / (2 * e),
         m_p * G * b @ np.cross(rho_p, R.T @ E3), tol=1e-8)

# free rigid body with a body torque
frb_mod, frb_layout, err, frb_p = library("free_rigid_body")
record_check("free rigid body (SO(3))", err)
J_g = frb_p["J"]
worst = 0.0
for _ in range(20):
    R, Om, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    a_g = accelerations(frb_mod, frb_layout.pack(R=R, Omega_R=Om), tau, frb_p)
    worst = max(worst, np.max(np.abs(a_g - np.linalg.solve(J_g, tau - np.cross(Om, J_g @ Om)))))
ck.true("Euler's equation J Om_dot + Om x J Om = tau equals geomech", worst < 1e-9, f"err {worst:.1e}")

# 3D pendulum
rp_mod, rp_layout, err, rp_p = library("rigid_pendulum")
record_check("3D pendulum (SO(3))", err)
rp_p.update({"J": J_p, "\\rho": rho_p, "m": m_p, "g": G})
worst = 0.0
for _ in range(20):
    R, Om, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    a_g = accelerations(rp_mod, rp_layout.pack(R=R, Omega_R=Om), tau, rp_p)
    worst = max(worst, np.max(np.abs(a_g - pendulum_accel(R, Om, tau))))
ck.true("3D pendulum J Om_dot + Om x J Om = m g R'e3 x rho + tau equals geomech", worst < 1e-9, f"err {worst:.1e}")
ck.close("3D pendulum: hanging attitude is an equilibrium", pendulum_accel(np.eye(3), np.zeros(3)), np.zeros(3))

# rigid body with a force at a body point (Lagrange-d'Alembert): geomech model built here
half = Scalar("0.5", value=0.5, attr=["Constant"])
m_, g_ = getScalars("m g", attr=["Constant"])
J_ = Matrix("J", attr=["Constant", "SymmetricMatrix"])
e3_ = Vector("e3", attr=["Constant"])
r_ = Vector("r", attr=["Constant"])
x_, R_ = Vector("x"), SO3("R")
Om_, eta_ = R_.get_tangent_vector(), R_.get_variation_vector()
F_, tau_ = Vector("F"), Vector("tau")
L_rb = half * m_ * Dot(x_.t_diff(), x_.t_diff()) + half * Dot(Om_, J_ * Om_) - m_ * g_ * Dot(e3_, x_)
dW_rb = Dot(x_.delta() + R_ * Cross(eta_, r_), F_) + Dot(eta_, tau_)
rb_mod, rb_layout, err = plant_of(L_rb, dW_rb, SystemVariables(vectors=[x_], matrices=[R_]), [F_, tau_], "rb_point_force")
record_check("rigid body, force at a body point (R3 x SO(3))", err)
m_rb, r_rb = 2.0, np.array([0.3, -0.1, 0.2])
J_rb = np.array([[0.2, 0.01, 0.0], [0.01, 0.25, 0.02], [0.0, 0.02, 0.4]])
p_rb = {"m": m_rb, "g": G, "J": J_rb, "e3": E3, "r": r_rb}
worst = 0.0
for _ in range(20):
    R, Om, F, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    a_g = accelerations(rb_mod, rb_layout.pack(x=rng.normal(size=3), R=R, x_dot=rng.normal(size=3), Omega_R=Om),
                        np.concatenate([F, tau]), p_rb)
    vdot = F / m_rb - G * E3
    Omdot = np.linalg.solve(J_rb, tau + np.cross(r_rb, R.T @ F) - np.cross(Om, J_rb @ Om))
    worst = max(worst, np.max(np.abs(a_g - np.concatenate([vdot, Omdot]))))
ck.true("rigid body: m v_dot = F - m g e3, J Om_dot + Om x J Om = tau + r x R'F equals geomech", worst < 1e-9, f"err {worst:.1e}")


# Newton-Euler reading: d/dt (R J Om) = R tau
def frb_rhs(t, s, J, tau_fn):
    R, Om = s[:9].reshape(3, 3), s[9:]
    return np.concatenate([(R @ hat(Om)).ravel(), np.linalg.solve(J, tau_fn(t) - np.cross(Om, J @ Om))])


tau_fn = lambda t: np.array([0.2 * np.sin(t), -0.1, 0.05 * t])
s0 = np.concatenate([random_rotation(rng).ravel(), [0.5, -1.0, 0.8]])
sol = solve_ivp(frb_rhs, [0, 2], s0, args=(J_p, tau_fn), rtol=1e-11, atol=1e-12, dense_output=True)
ang = lambda t: sol.sol(t)[:9].reshape(3, 3) @ J_p @ sol.sol(t)[9:]
tq = 1.1
ck.close("Newton-Euler: d/dt (R J Omega) = R tau", ddt(ang, tq), sol.sol(tq)[:9].reshape(3, 3) @ tau_fn(tq), tol=1e-6)


# =========================================================================
# 5. Conserved quantities and the tennis-racket instability
# =========================================================================
# A paperback: 0.20 m x 0.13 m x 0.03 m, 0.3 kg, body axes along length, width, thickness.
mb, dims = 0.3, np.array([0.20, 0.13, 0.03])
J_b = mb / 12 * np.array([dims[1]**2 + dims[2]**2, dims[0]**2 + dims[2]**2, dims[0]**2 + dims[1]**2])
J_book = np.diag(J_b)
ck.record("book_inertia_1e-3_kgm2", (J_b * 1e3).tolist())
ck.true("book: J1 < J2 < J3, so e2 (across the width, in the cover plane) is the intermediate axis", J_b[0] < J_b[1] < J_b[2],
        f"{np.round(J_b * 1e3, 4)}")
spin = 2 * np.pi * 2.0                    # two revolutions per second
tilt = 0.01                               # 1 % of the spin put on the other two axes


def free_run(Om0, T, n=4001):
    s0 = np.concatenate([np.eye(3).ravel(), Om0])
    sol = solve_ivp(frb_rhs, [0, T], s0, args=(J_book, lambda t: np.zeros(3)), rtol=1e-11, atol=1e-13,
                    t_eval=np.linspace(0, T, n))
    return sol.t, sol.y[:9].T.reshape(-1, 3, 3), sol.y[9:].T


t, Rr, Omr = free_run(spin * np.array([tilt, 1.0, tilt]), 6.0)
Ekin = 0.5 * np.einsum("ti,i,ti->t", Omr, J_b, Omr)
Hs = np.einsum("tij,j,tj->ti", Rr, J_b, Omr)
ck.true("free body: kinetic energy conserved", np.ptp(Ekin) / Ekin[0] < 1e-8, f"rel spread {np.ptp(Ekin) / Ekin[0]:.1e}")
ck.true("free body: spatial angular momentum R J Omega conserved", np.max(np.abs(Hs - Hs[0])) / np.linalg.norm(Hs[0]) < 1e-8,
        f"rel err {np.max(np.abs(Hs - Hs[0])) / np.linalg.norm(Hs[0]):.1e}")
Pi_body = Omr * J_b
ck.true("free body: body momentum J Omega is NOT constant near the intermediate axis",
        np.max(np.abs(Pi_body - Pi_body[0])) / np.linalg.norm(Pi_body[0]) > 1.5)
ck.true("intermediate axis: Omega_2 reverses sign", Omr[:, 1].min() < -0.99 * spin, f"min Omega_2 / spin = {Omr[:, 1].min() / spin:.4f}")
flips = t[1:][np.sign(Omr[1:, 1]) != np.sign(Omr[:-1, 1])]
ck.record("book_first_flip_s", float(flips[0]))
ck.record("book_flip_interval_s", float(flips[1] - flips[0]))
ck.true("book: first reversal of Omega_2 between 0.5 s and 1.5 s", 0.5 < flips[0] < 1.5, f"{flips[0]:.3f} s, then every {flips[1] - flips[0]:.3f} s")
lam = spin * np.sqrt((J_b[1] - J_b[0]) * (J_b[2] - J_b[1]) / (J_b[0] * J_b[2]))
ck.record("book_growth_rate_per_s", float(lam))
ck.record("book_efold_time_s", float(1 / lam))
# measured growth of the wobble while it is still small. In the linearization about e2,
# d/dt (Om1, Om3) = [[0, a], [b, 0]] (Om1, Om3); the combination b Om1 + lam Om3 grows exactly like exp(lam t).
a_lin, b_lin = (J_b[1] - J_b[2]) * spin / J_b[0], (J_b[0] - J_b[1]) * spin / J_b[2]
ck.close("growth rate squared is a b", lam**2, a_lin * b_lin)
early = (t > 0.02) & (t < 0.30)
fit = np.polyfit(t[early], np.log(np.abs(b_lin * Omr[early, 0] + lam * Omr[early, 2])), 1)
ck.true("intermediate axis: wobble grows at rate Omega sqrt((J2-J1)(J3-J2)/(J1 J3))", abs(fit[0] - lam) / lam < 0.03,
        f"measured {fit[0]:.2f} /s, formula {lam:.2f} /s")
# spins about the smallest and largest axes stay put
for ax, name in ((0, "smallest"), (2, "largest")):
    Om0 = spin * tilt * np.ones(3)
    Om0[ax] = spin
    _, _, Oms_ = free_run(Om0, 6.0)
    off = np.max(np.linalg.norm(np.delete(Oms_, ax, axis=1), axis=1)) / spin
    ck.true(f"spin about the {name} axis stays within a few percent", off < 0.05, f"max off-axis rate {off:.3f} of spin")
    ck.record(f"book_offaxis_{name}", float(off))
# linearized rates: purely imaginary about e1, e3
# about axis i the wobble obeys  w_dd = -k_i w,  k_i = Om^2 (J_i - J_j)(J_i - J_k) / (J_j J_k): stable iff k_i > 0
k_axis = [(J_b[i] - J_b[(i + 1) % 3]) * (J_b[i] - J_b[(i + 2) % 3]) for i in range(3)]
ck.true("linearization: restoring for the smallest and largest axes, repelling for the middle one",
        k_axis[0] > 0 and k_axis[2] > 0 and k_axis[1] < 0)


# =========================================================================
# 6. Quadrotor, quadrotor with a suspended load, two quadrotors with a load
# =========================================================================
q_mod, q_layout, err, q_p = library("quadrotor")
record_check("quadrotor (R3 x SO(3))", err)
worst = 0.0
for _ in range(20):
    R, Om, f, tau = random_rotation(rng), rng.normal(size=3), 5 + 20 * rng.random(), rng.normal(size=3)
    a_g = accelerations(q_mod, q_layout.pack(x=rng.normal(size=3), R=R, x_dot=rng.normal(size=3), Omega_R=Om),
                        np.concatenate([[f], tau]), q_p)
    mine = np.concatenate([f * R @ E3 / q_p["m"] - G * E3, np.linalg.solve(q_p["J"], tau - np.cross(Om, q_p["J"] @ Om))])
    worst = max(worst, np.max(np.abs(a_g - mine)))
ck.true("quadrotor: m v_dot = f R e3 - m g e3, J Om_dot + Om x J Om = tau equals geomech", worst < 1e-9, f"err {worst:.1e}")

ql_mod, ql_layout, err, ql_p = library("quadrotor_suspended_load")
record_check("quadrotor with suspended load (R3 x S2 x SO(3))", err)
mQ, mL, ll, JQ = ql_p["m_Q"], ql_p["m_L"], ql_p["l"], ql_p["J"]


def load_accels(q, om, R, Om, f, tau):
    """Chapter eqs. for the quadrotor with a cable-suspended load."""
    u = f * R @ E3
    vL_dot = ((q @ u) - mQ * ll * (om @ om)) * q / (mQ + mL) - G * E3
    om_dot = -np.cross(q, u) / (mQ * ll)
    Om_dot = np.linalg.solve(JQ, tau - np.cross(Om, JQ @ Om))
    return vL_dot, om_dot, Om_dot


worst = worst_T = 0.0
for _ in range(20):
    q = unit(rng.normal(size=3))
    om = tangent(q, rng.normal(size=3))
    R, Om, f, tau = random_rotation(rng), rng.normal(size=3), 5 + 20 * rng.random(), rng.normal(size=3)
    x = ql_layout.pack(x_L=rng.normal(size=3), q=q, R=R, x_L_dot=rng.normal(size=3), omega_q=om, Omega_R=Om)
    a_g = accelerations(ql_mod, x, np.concatenate([[f], tau]), ql_p)
    vLd, omd, Omd = load_accels(q, om, R, Om, f, tau)
    worst = max(worst, np.max(np.abs(a_g - np.concatenate([vLd, omd, Omd]))))
    # free-body check with the tension: load  m_L (vL_dot + g e3) = -T q ; quadrotor m_Q (vQ_dot + g e3) = f R e3 + T q
    T = mL / (mQ + mL) * (mQ * ll * (om @ om) - q @ (f * R @ E3))
    qd = np.cross(om, q)
    qdd = np.cross(omd, q) + np.cross(om, qd)
    vQd = vLd - ll * qdd
    worst_T = max(worst_T, np.max(np.abs(mL * (vLd + G * E3) + T * q)), np.max(np.abs(mQ * (vQd + G * E3) - f * R @ E3 - T * q)))
ck.true("quadrotor + load: the three chapter equations equal geomech", worst < 1e-8, f"err {worst:.1e}")
ck.true("quadrotor + load: Newton's laws for each body with tension T", worst_T < 1e-9, f"err {worst_T:.1e}")
# hover
vLd, omd, _ = load_accels(-E3, np.zeros(3), np.eye(3), np.zeros(3), (mQ + mL) * G, np.zeros(3))
ck.close("quadrotor + load: hover with q = -e3, f = (mQ + mL) g", np.concatenate([vLd, omd]), np.zeros(6))
ck.close("quadrotor + load: hover tension is mL g", mL / (mQ + mL) * (0 - (-E3) @ ((mQ + mL) * G * E3)), mL * G)
ck.record("load_params", {"m_Q": mQ, "m_L": mL, "l": ll})
ck.record("load_hover_thrust_N", float((mQ + mL) * G))
ck.record("load_hover_tension_N", float(mL * G))
# thrust cut: cable direction keeps a constant angular velocity, tension from the spin only
q = unit([0.3, 0.2, -0.9])
om = tangent(q, [1.0, 2.0, 0.0])
vLd, omd, _ = load_accels(q, om, random_rotation(rng), rng.normal(size=3), 0.0, np.zeros(3))
ck.close("quadrotor + load: with f = 0 the cable rate omega is constant", omd, np.zeros(3))
ck.close("quadrotor + load: with f = 0 the mass center is in free fall",
         (mL * vLd + mQ * (vLd - ll * (np.cross(omd, q) + np.cross(om, np.cross(om, q))))) / (mQ + mL), -G * E3)


# energy and constraints along an unforced run, and the work-energy balance with thrust
def load_rhs(t, s, f, tau):
    xL, vL, q, om, R, Om = s[:3], s[3:6], s[6:9], s[9:12], s[12:21].reshape(3, 3), s[21:]
    vLd, omd, Omd = load_accels(q, om, R, Om, f, tau)
    return np.concatenate([vL, vLd, np.cross(om, q), omd, (R @ hat(Om)).ravel(), Omd])


def load_energy(s):
    xL, vL, q, om, Om = s[:3], s[3:6], s[6:9], s[9:12], s[21:]
    vQ = vL - ll * np.cross(om, q)
    return 0.5 * mQ * vQ @ vQ + 0.5 * mL * vL @ vL + 0.5 * Om @ JQ @ Om + mQ * G * (xL - ll * q)[2] + mL * G * xL[2]


q0 = unit([0.3, -0.2, -0.9])
s0 = np.concatenate([np.zeros(3), [0.5, 0.2, 0.1], q0, tangent(q0, [1.0, 2.0, 0.5]), np.eye(3).ravel(), [0.3, -0.2, 0.5]])
sol = solve_ivp(load_rhs, [0, 3], s0, args=(0.0, np.zeros(3)), rtol=1e-11, atol=1e-12)
Es = np.array([load_energy(s) for s in sol.y.T])
ck.true("quadrotor + load: energy conserved without inputs", np.ptp(Es) < 1e-6, f"spread {np.ptp(Es):.1e}")
ck.true("quadrotor + load: |q| = 1 and omega . q = 0 preserved", max(np.max(np.abs(np.linalg.norm(sol.y[6:9], axis=0) - 1)),
                                                                   np.max(np.abs(np.sum(sol.y[6:9] * sol.y[9:12], axis=0)))) < 1e-8)
f_c, tau_c = 20.0, np.array([0.01, -0.02, 0.005])
sol = solve_ivp(load_rhs, [0, 1.5], s0, args=(f_c, tau_c), rtol=1e-11, atol=1e-12, t_eval=np.linspace(0, 1.5, 3001))
power = []
for s in sol.y.T:
    vL, q, om, R, Om = s[3:6], s[6:9], s[9:12], s[12:21].reshape(3, 3), s[21:]
    power.append(f_c * (R @ E3) @ (vL - ll * np.cross(om, q)) + tau_c @ Om)
ck.close("quadrotor + load: energy change equals the work of thrust and torque",
         load_energy(sol.y[:, -1]) - load_energy(sol.y[:, 0]), np.trapezoid(power, sol.t), tol=1e-5)

# two quadrotors, one point-mass load
mq_mod, mq_layout, err, mq_p = library("multi_quadrotor_point_mass", n_quads=2)
record_check("two quadrotors with a point-mass load (R3 x (S2 x SO(3))^2)", err)
worst = 0.0
for _ in range(10):
    qs = [unit(rng.normal(size=3)) for _ in range(2)]
    oms = [tangent(qi, rng.normal(size=3)) for qi in qs]
    Rs_ = [random_rotation(rng) for _ in range(2)]
    Oms_ = [rng.normal(size=3) for _ in range(2)]
    fs = [5 + 20 * rng.random() for _ in range(2)]
    taus = [rng.normal(size=3) for _ in range(2)]
    ms_ = [mq_p["m1"], mq_p["m2"]]
    ls_ = [mq_p["l1"], mq_p["l2"]]
    Js_ = [mq_p["J1"], mq_p["J2"]]
    us = [fs[i] * Rs_[i] @ E3 for i in range(2)]
    # chapter eqs.: (m_L I + sum m_i q_i q_i') (vL_dot + g e3) = sum (q_i . u_i - m_i l_i |omega_i|^2) q_i
    A = mq_p["m_L"] * np.eye(3) + sum(ms_[i] * np.outer(qs[i], qs[i]) for i in range(2))
    rhs = sum(((qs[i] @ us[i]) - ms_[i] * ls_[i] * (oms[i] @ oms[i])) * qs[i] for i in range(2))
    acc = np.linalg.solve(A, rhs)
    #               m_i l_i omega_i_dot = q_i x (m_i (vL_dot + g e3) - u_i)
    omds = [np.cross(qs[i], ms_[i] * acc - us[i]) / (ms_[i] * ls_[i]) for i in range(2)]
    Omds = [np.linalg.solve(Js_[i], taus[i] - np.cross(Oms_[i], Js_[i] @ Oms_[i])) for i in range(2)]
    x = mq_layout.pack(x_L=rng.normal(size=3), q1=qs[0], q2=qs[1], R1=Rs_[0], R2=Rs_[1], x_L_dot=rng.normal(size=3),
                       omega_q1=oms[0], omega_q2=oms[1], Omega_R1=Oms_[0], Omega_R2=Oms_[1])
    a_g = accelerations(mq_mod, x, np.concatenate([[fs[0]], taus[0], [fs[1]], taus[1]]), mq_p)
    worst = max(worst, np.max(np.abs(a_g - np.concatenate([acc - G * E3, omds[0], omds[1], Omds[0], Omds[1]]))))
ck.true("two quadrotors + load: chapter equations equal geomech", worst < 1e-8, f"err {worst:.1e}")

# telescoping stance leg (Reaction Mass Biped building block) on R x SO(3)
rho_s = Scalar("rho")
f_s = Scalar("f")
b_s = (R_ * e3_) * rho_s
L_leg = half * m_ * Dot(b_s.t_diff(), b_s.t_diff()) + half * Dot(Om_, J_ * Om_) - m_ * g_ * Dot(e3_, b_s)
dW_leg = rho_s.delta() * f_s + Dot(eta_, tau_)
leg_mod, leg_layout, err = plant_of(L_leg, dW_leg, SystemVariables(scalars=[rho_s], matrices=[R_]), [f_s, tau_], "leg")
record_check("telescoping leg (R x SO(3))", err)
m_leg, J_leg = 30.0, np.diag([0.9, 1.1, 0.4])
p_leg = {"m": m_leg, "g": G, "J": J_leg, "e3": E3}
worst = 0.0
K = -hat(E3) @ hat(E3)                       # = I - e3 e3'
for _ in range(20):
    R, Om, rho, rhod, f, tau = random_rotation(rng), rng.normal(size=3), 0.5 + rng.random(), rng.normal(), 50 * rng.normal(), rng.normal(size=3)
    a_g = accelerations(leg_mod, leg_layout.pack(rho=rho, R=R, rho_dot=rhod, Omega_R=Om), np.concatenate([[f], tau]), p_leg)
    # chapter eqs.:  m rho_dd = m rho Om'K Om - m g e3 . R e3 + f
    #                J(rho) Om_dot + Om x J(rho) Om + 2 m rho rho_dot K Om = m g rho (R'e3) x e3 + tau,  J(rho) = J0 + m rho^2 K
    Jr = J_leg + m_leg * rho**2 * K
    rhodd = rho * Om @ K @ Om - G * E3 @ (R @ E3) + f / m_leg
    Omd = np.linalg.solve(Jr, tau + m_leg * G * rho * np.cross(R.T @ E3, E3) - np.cross(Om, Jr @ Om) - 2 * m_leg * rho * rhod * K @ Om)
    worst = max(worst, np.max(np.abs(a_g - np.concatenate([[rhodd], Omd]))))
ck.true("telescoping leg: chapter equations equal geomech", worst < 1e-8, f"err {worst:.1e}")


# =========================================================================
# 7. Integrating on the manifold
# =========================================================================
Om = rng.normal(size=3)
hh = 0.01
naive = np.eye(3) + hh * hat(Om)
ck.close("explicit Euler step: (I + h hat(Om))'(I + h hat(Om)) = I - h^2 hat(Om)^2", naive.T @ naive, np.eye(3) - hh**2 * hat(Om) @ hat(Om))
ck.close("exponential step is a rotation", expm_so3(hh * Om).T @ expm_so3(hh * Om), np.eye(3), tol=1e-14)
qv = unit(rng.normal(size=3))
ck.close("exp(h hat(omega)) q stays on the sphere", np.linalg.norm(expm_so3(hh * Om) @ qv), 1.0, tol=1e-14)


# N naive steps at constant Omega: |R'R - I| (spectral norm) = (1 + h^2 |Om|^2)^N - 1, close to N h^2 |Om|^2
N_st = 200
Rn = np.linalg.matrix_power(naive, N_st)
ck.close("N naive steps: spectral norm of R'R - I is (1 + h^2 |Om|^2)^N - 1", np.linalg.norm(Rn.T @ Rn - np.eye(3), 2),
         (1 + hh**2 * Om @ Om) ** N_st - 1, tol=1e-12)
ck.true("... which is close to N h^2 |Om|^2", abs(np.linalg.norm(Rn.T @ Rn - np.eye(3), 2) / (N_st * hh**2 * Om @ Om) - 1) < 0.05)

# 3D pendulum without torque: energy and the vertical component of R J Omega are conserved, the others are not
solp = solve_ivp(lambda t, s_: np.concatenate([(s_[:9].reshape(3, 3) @ hat(s_[9:])).ravel(), pendulum_accel(s_[:9].reshape(3, 3), s_[9:])]),
                 [0, 5], np.concatenate([expm_so3(np.array([1.0, 0.4, 0.0])).ravel(), [0.3, -0.5, 2.0]]), rtol=1e-11, atol=1e-12)
Rp_ = solp.y[:9].T.reshape(-1, 3, 3)
Hp = np.einsum("tij,jk,tk->ti", Rp_, J_p, solp.y[9:].T)
Ep = 0.5 * np.einsum("ti,ij,tj->t", solp.y[9:].T, J_p, solp.y[9:].T) + m_p * G * np.einsum("i,tij,j->t", E3, Rp_, rho_p)
ck.true("3D pendulum: energy conserved", np.ptp(Ep) < 1e-7, f"spread {np.ptp(Ep):.1e}")
ck.true("3D pendulum: e3 . R J Omega conserved, horizontal components not", np.ptp(Hp[:, 2]) < 1e-7 and np.ptp(Hp[:, 0]) > 0.1,
        f"spread {np.ptp(Hp[:, 2]):.1e} vs {np.ptp(Hp[:, 0]):.2f}")


def dexpinv(th, Om):
    """Rate of the rotation vector theta in R = R_k exp(hat(theta)) when the body rate is Omega (to second order)."""
    return Om + 0.5 * np.cross(th, Om) + np.cross(th, np.cross(th, Om)) / 12


th = 0.05 * rng.normal(size=3)
thd = rng.normal(size=3)
Om_true = vee(expm_so3(th).T @ (expm_so3(th + e * thd) - expm_so3(th - e * thd)) / (2 * e))
ck.close("theta_dot = Om + (theta x Om)/2 + theta x (theta x Om)/12 for small theta", dexpinv(th, Om_true), thd, tol=1e-5)


def pend_energy(R, Om):
    return 0.5 * Om @ J_p @ Om + m_p * G * E3 @ (R @ rho_p)


def rk4_flat(R, Om, h):
    """Classical RK4 on the twelve numbers (R, Omega)."""
    def f(R_, Om_):
        return R_ @ hat(Om_), pendulum_accel(R_, Om_)
    k1 = f(R, Om)
    k2 = f(R + h / 2 * k1[0], Om + h / 2 * k1[1])
    k3 = f(R + h / 2 * k2[0], Om + h / 2 * k2[1])
    k4 = f(R + h * k3[0], Om + h * k3[1])
    return R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), Om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


def rk4_lie(R, Om, h):
    """RK4 on (theta, Omega) with R = R_k exp(hat(theta)): the update is a rotation by construction."""
    def f(th_, Om_):
        return dexpinv(th_, Om_), pendulum_accel(R @ expm_so3(th_), Om_)
    z = np.zeros(3)
    k1 = f(z, Om)
    k2 = f(h / 2 * k1[0], Om + h / 2 * k1[1])
    k3 = f(h / 2 * k2[0], Om + h / 2 * k2[1])
    k4 = f(h * k3[0], Om + h * k3[1])
    th_ = h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    return R @ expm_so3(th_), Om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


def integrate(step, h, T, R0, Om0):
    R, Om = R0.copy(), Om0.copy()
    E0 = pend_energy(R, Om)
    out = []
    for k in range(int(round(T / h))):
        R, Om = step(R, Om, h)
        out.append((np.linalg.norm(R.T @ R - np.eye(3)), abs(pend_energy(R, Om) - E0)))
    return np.array(out), R, Om


R0_i, Om0_i = expm_so3(np.array([2.0, 0.0, 0.0])), np.array([0.0, 0.5, 6.0])
h_i, T_i = 0.02, 60.0
ref = solve_ivp(lambda t, s: np.concatenate([(s[:9].reshape(3, 3) @ hat(s[9:])).ravel(), pendulum_accel(s[:9].reshape(3, 3), s[9:])]),
                [0, 1.0], np.concatenate([R0_i.ravel(), Om0_i]), rtol=1e-12, atol=1e-13)
d_flat, _, _ = integrate(rk4_flat, h_i, T_i, R0_i, Om0_i)
d_lie, _, _ = integrate(rk4_lie, h_i, T_i, R0_i, Om0_i)
E_scale = abs(pend_energy(R0_i, Om0_i) - m_p * G * rho_p[2])       # energy above the hanging rest state
ck.record("integrator", {"h": h_i, "T": T_i, "energy_above_rest_J": float(E_scale)})
ck.record("int_orth_naive_end", float(d_flat[-1, 0]))
ck.record("int_orth_lie_max", float(d_lie[:, 0].max()))
ck.record("int_energy_naive_end_J", float(d_flat[-1, 1]))
ck.record("int_energy_lie_end_J", float(d_lie[-1, 1]))
ck.record("int_energy_naive_end_rel", float(d_flat[-1, 1] / E_scale))
ck.record("int_energy_lie_end_rel", float(d_lie[-1, 1] / E_scale))
ck.true("Lie-group RK4 keeps R'R = I to round-off", d_lie[:, 0].max() < 1e-12, f"{d_lie[:, 0].max():.1e}")
ck.true("RK4 on the entries of R drifts off SO(3)", d_flat[-1, 0] > 1e-5, f"{d_flat[-1, 0]:.1e}")
ck.true("neither fourth-order scheme conserves energy exactly", d_lie[-1, 1] > 1e-9 and d_flat[-1, 1] > 1e-9,
        f"naive {d_flat[-1, 1]:.1e} J, Lie {d_lie[-1, 1]:.1e} J")
# both are fourth order: halving h divides the one-second error by about 16
errs = []
for hh in (0.02, 0.01):
    _, Rl, Oml = integrate(rk4_lie, hh, 1.0, R0_i, Om0_i)
    errs.append(np.linalg.norm(Rl - ref.y[:9, -1].reshape(3, 3)))
ck.true("Lie-group RK4 is fourth order", 10 < errs[0] / errs[1] < 24, f"error ratio {errs[0] / errs[1]:.1f}")
ck.record("geomech_check_eom", geomech_errors)

ck.finish()
