"""Chapter 3: every identity and every quoted number, checked numerically.

    python verify/ch03.py

Variation and derivative formulas are checked by finite differences along
actual varied curves, never by re-deriving them symbolically.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, logm

from common import Checker, Rx, Rz, expm_so3, hat, random_rotation, vee

ck = Checker("ch03")
rng = np.random.default_rng(3)
I3 = np.eye(3)
E1, E2, E3 = I3


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def log_so3(R):
    """Rotation vector theta*u of R, for angles below pi."""
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-10:
        return vee(R - R.T) / 2
    return th / (2 * np.sin(th)) * vee(R - R.T)


def fd(f, x, h=1e-5):
    """Central difference of a function of one real variable."""
    return (f(x + h) - f(x - h)) / (2 * h)


# =====================================================================
# 0. The opening puzzle: same motion, different body frames
# =====================================================================
def Rd_spin(t):
    return Rz(1.0 * t)                       # reference spins about world e3 at 1 rad/s


Roff = Rx(np.pi / 2)                         # body frame bolted on a quarter turn away


def R_spin(t):
    return Rd_spin(t) @ Roff


t0 = 0.7
Om_d = vee(Rd_spin(t0).T @ fd(Rd_spin, t0))
Om = vee(R_spin(t0).T @ fd(R_spin, t0))
ck.close("puzzle: reference gyro reads (0,0,1)", Om_d, [0, 0, 1], tol=1e-8)
ck.close("puzzle: body gyro reads (0,1,0)", Om, [0, 1, 0], tol=1e-8)
ck.close("puzzle: naive difference has norm sqrt(2)", np.linalg.norm(Om - Om_d), np.sqrt(2), tol=1e-8)
ck.close("puzzle: relative attitude Rd^T R is constant", R_spin(0.1).T @ Rd_spin(0.1), R_spin(2.3).T @ Rd_spin(2.3))
ck.close("puzzle: transported difference is zero", Om - R_spin(t0).T @ Rd_spin(t0) @ Om_d, np.zeros(3), tol=1e-8)
ck.record("puzzle_naive_velocity_error", float(np.linalg.norm(Om - Om_d)))


# =====================================================================
# 1. Variations on SO(3)
# =====================================================================
def Omega_fun(t):
    return np.array([np.sin(t), 0.6 * np.cos(2 * t), 0.4 + 0.3 * t])


def eta_fun(t):
    return np.array([0.5 * np.cos(t), 0.3 * t, np.sin(1.3 * t) - 0.2])


# A reference curve R(t) with body velocity Omega_fun, obtained by integration.
sol_R = solve_ivp(lambda t, y: (y.reshape(3, 3) @ hat(Omega_fun(t))).ravel(), [0, 2], random_rotation(rng).ravel(),
                  rtol=1e-12, atol=1e-13, dense_output=True)


def R_ref(t):
    return sol_R.sol(t).reshape(3, 3)


def R_eps(t, eps):
    return R_ref(t) @ expm_so3(eps * eta_fun(t))


def Omega_eps(t, eps):
    return vee(R_eps(t, eps).T @ fd(lambda s: R_eps(s, eps), t))


t0 = 0.9
R0 = R_ref(t0)
dR = fd(lambda e: R_eps(t0, e), 0.0)
ck.close("delta R = R hat(eta)", dR, R0 @ hat(eta_fun(t0)), tol=1e-8)
S = R0.T @ dR
ck.close("R^T delta R is skew-symmetric", S + S.T, np.zeros((3, 3)), tol=1e-8)
bad = R0 + 1e-3 * rng.normal(size=(3, 3))       # an arbitrary matrix perturbation leaves the group
ck.true("an arbitrary matrix perturbation breaks R^T R = I at first order",
        np.linalg.norm(bad.T @ bad - I3) > 1e-4)
good = R0 @ expm_so3(1e-3 * rng.normal(size=3))
ck.close("R exp(eps hat(eta)) stays on SO(3)", good.T @ good, I3, tol=1e-12)

dOm = fd(lambda e: Omega_eps(t0, e), 0.0, h=1e-4)
ck.close("delta Omega = eta_dot + Omega x eta  (finite differences)",
         dOm, fd(eta_fun, t0) + np.cross(Omega_fun(t0), eta_fun(t0)), tol=1e-6)
ck.true("the cross term matters: delta Omega != eta_dot",
        np.linalg.norm(dOm - fd(eta_fun, t0)) > 0.1)
# mixed partials commute: d/dt (delta R) = delta (R_dot)
lhs = fd(lambda s: R_ref(s) @ hat(eta_fun(s)), t0)
rhs = fd(lambda e: fd(lambda s: R_eps(s, e), t0), 0.0, h=1e-4)
ck.close("d/dt(delta R) = delta(dR/dt)", lhs, rhs, tol=1e-6)

# world-frame version: R_dot = hat(w) R, delta R = hat(zeta) R  gives  delta w = zeta_dot - w x zeta
def R_eps_world(t, eps):
    return expm_so3(eps * eta_fun(t)) @ R_ref(t)


def w_eps(t, eps):
    return vee(fd(lambda s: R_eps_world(s, eps), t) @ R_eps_world(t, eps).T)


w0 = R0 @ Omega_fun(t0)
ck.close("world-frame variation: delta w = zeta_dot - w x zeta (sign flips)", fd(lambda e: w_eps(t0, e), 0.0, h=1e-4),
         fd(eta_fun, t0) - np.cross(w0, eta_fun(t0)), tol=1e-6)

# a slightly remounted gyro
Om_ex, eta_ex0 = np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.0, 0.0])
ck.close("remount example: Omega x eta = (0, 0.1, 0)", np.cross(Om_ex, eta_ex0), [0, 0.1, 0])
ck.close("remount example: exact reading (0, 0.0998, 0.9950)", expm_so3(-eta_ex0) @ Om_ex, [0, 0.0998, 0.9950], tol=5e-5)
ck.close("remount by a quarter turn gives (0, 1, 0)", expm_so3(-np.pi / 2 * E1) @ Om_ex, [0, 1, 0], tol=1e-12)
eta_c = np.array([0.4, -0.2, 0.3])
Rc = lambda s: R_ref(s) @ expm_so3(eta_c)
ck.close("constant eta: the remounted gyro reads exp(-hat(eta)) Omega", vee(Rc(t0).T @ fd(Rc, t0)), expm_so3(-eta_c) @ Omega_fun(t0), tol=1e-7)

# variation of the rotational kinetic energy: where Omega x J Omega comes from
J = np.diag([0.05, 0.07, 0.10])
dT = fd(lambda e: 0.5 * Omega_eps(t0, e) @ J @ Omega_eps(t0, e), 0.0, h=1e-4)
Om0, eta0, deta0 = Omega_fun(t0), eta_fun(t0), fd(eta_fun, t0)
ck.close("delta(1/2 Om.J Om) = J Om . eta_dot - (Om x J Om) . eta",
         dT, (J @ Om0) @ deta0 - np.cross(Om0, J @ Om0) @ eta0, tol=1e-6)


# =====================================================================
# 2. Variations on S^2
# =====================================================================
def q_ref(t):
    return unit([np.cos(t) * np.cos(0.4 * t), np.sin(t) * np.cos(0.4 * t) + 0.3, 0.5 + np.sin(0.4 * t)])


def a_fun(t):
    return np.array([np.sin(0.7 * t) + 0.2, np.cos(1.1 * t), 0.5 * t])


def xi_fun(t):
    """A smooth variation field with xi . q = 0 at every time."""
    q = q_ref(t)
    return a_fun(t) - (a_fun(t) @ q) * q


def q_eps(t, eps, xi=xi_fun):
    return expm_so3(eps * xi(t)) @ q_ref(t)


def omega_eps(t, eps, xi=xi_fun):
    return np.cross(q_eps(t, eps, xi), fd(lambda s: q_eps(s, eps, xi), t))


t0 = 0.8
q0, xi0 = q_ref(t0), xi_fun(t0)
qdot0, xidot0 = fd(q_ref, t0), fd(xi_fun, t0)
om0 = np.cross(q0, qdot0)
ck.close("omega = q x q_dot is perpendicular to q", om0 @ q0, 0.0, tol=1e-9)
ck.close("q_dot = omega x q", qdot0, np.cross(om0, q0), tol=1e-8)
ck.close("xi . q = 0", xi0 @ q0, 0.0, tol=1e-12)
dq = fd(lambda e: q_eps(t0, e), 0.0)
ck.close("delta q = xi x q", dq, np.cross(xi0, q0), tol=1e-8)
ck.close("q . delta q = 0", q0 @ dq, 0.0, tol=1e-8)
dqdot = fd(lambda e: fd(lambda s: q_eps(s, e), t0), 0.0, h=1e-4)
ck.close("delta(q_dot) = xi_dot x q + xi x q_dot", dqdot, np.cross(xidot0, q0) + np.cross(xi0, qdot0), tol=1e-6)
dom = fd(lambda e: omega_eps(t0, e), 0.0, h=1e-4)
ck.close("delta omega = xi_dot - 2 omega x xi  (finite differences)", dom, xidot0 - 2 * np.cross(om0, xi0), tol=1e-6)
ck.true("the 'SO(3)-style' guess xi_dot + omega x xi is wrong on the sphere",
        np.linalg.norm(dom - (xidot0 + np.cross(om0, xi0))) > 0.1)
ck.close("xi_dot - omega x xi also generates delta(q_dot), without staying perpendicular to q",
         np.cross(xidot0 - np.cross(om0, xi0), q0) + np.cross(om0, np.cross(xi0, q0)), dqdot, tol=1e-6)
P = I3 - np.outer(q0, q0)
ck.close("Lee et al. (5.13): delta omega = -omega x xi + (I - q q^T) xi_dot", dom, -np.cross(om0, xi0) + P @ xidot0, tol=1e-6)
ck.close("tangential part of delta omega = xi_dot - omega x xi", P @ dom, xidot0 - np.cross(om0, xi0), tol=1e-6)
ck.close("omega x xi is parallel to q", np.cross(np.cross(om0, xi0), q0), np.zeros(3), tol=1e-8)
ck.close("q . xi_dot = q . (omega x xi)", q0 @ xidot0, q0 @ np.cross(om0, xi0), tol=1e-8)
ck.close("normal part: q . delta omega = -q . (omega x xi)", q0 @ dom, -q0 @ np.cross(om0, xi0), tol=1e-6)
ck.close("linearized constraint: (xi x q) . omega + q . delta omega = 0", np.cross(xi0, q0) @ om0 + q0 @ dom, 0.0, tol=1e-6)
ck.true("delta omega is not tangent (normal part is not small)", abs(q0 @ dom) > 0.05, f"{q0 @ dom:.3f}")


# xi not constrained to be perpendicular to q: the general formula
def xi_free(t):
    return a_fun(t)


xf, xfdot = xi_free(t0), fd(xi_free, t0)
dom_free = fd(lambda e: omega_eps(t0, e, xi_free), 0.0, h=1e-4)
general = P @ xfdot - q0 * (q0 @ np.cross(om0, xf)) + (xf @ q0) * np.cross(q0, om0)
ck.close("unconstrained xi: delta omega = P xi_dot - q q.(omega x xi) + (xi.q) q x omega", dom_free, general, tol=1e-6)
ck.true("unconstrained xi: xi_dot - 2 omega x xi no longer holds",
        np.linalg.norm(dom_free - (xfdot - 2 * np.cross(om0, xf))) > 0.05)

# the symbolic library applies the same two rules
try:
    from geomech import S2, SO3, SystemVariables
    from geomech.core.transformations.manifold_rules import apply_manifold_rules
    qs, Rs = S2("q"), SO3("R")
    sv = SystemVariables(scalars=[], vectors=[qs], matrices=[Rs])
    s_om = str(apply_manifold_rules(qs.get_tangent_vector().delta(), sv))
    s_Om = str(apply_manifold_rules(Rs.get_tangent_vector().delta(), sv))
    ck.true("geomech: delta omega -> xi_dot + (omega x xi)(-2)", s_om == r"(\frac{d}{dt}(\xi_{q})+Cross(\omega_{q},\xi_{q})*(-2))", s_om)
    ck.true("geomech: delta Omega -> eta_dot + Omega x eta", s_Om == r"(\frac{d}{dt}(\eta_{R})+Cross(\Omega_{R},\eta_{R}))", s_Om)
except ImportError:                                           # pragma: no cover
    ck.true("geomech importable", False, "pip install geomech")


# =====================================================================
# 3. Error functions on SO(3): heights and slopes
# =====================================================================
def Psi_R(R, Rd):
    return 0.5 * np.trace(I3 - Rd.T @ R)


def e_R(R, Rd):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def Psi_D(R, Rd, D):
    return np.trace(D @ (I3 - Rd.T @ R))


def e_D(R, Rd, D):
    return vee(D @ Rd.T @ R - R.T @ Rd @ D)


R, Rd = random_rotation(rng), random_rotation(rng)
eta = rng.normal(size=3)
D = np.diag([0.9, 0.6, 0.3])
ck.close("delta Psi_R = e_R . eta", fd(lambda e: Psi_R(R @ expm_so3(e * eta), Rd), 0.0), e_R(R, Rd) @ eta, tol=1e-8)
ck.close("delta Psi_D = e_D . eta, e_D = (D Rd^T R - R^T Rd D)^vee",
         fd(lambda e: Psi_D(R @ expm_so3(e * eta), Rd, D), 0.0), e_D(R, Rd, D) @ eta, tol=1e-8)
ck.close("D = I/2 recovers Psi_R", Psi_D(R, Rd, I3 / 2), Psi_R(R, Rd))
ck.close("D = I/2 recovers e_R", e_D(R, Rd, I3 / 2), e_R(R, Rd))
ck.close("tr(A hat(eta)) = -(A - A^T)^vee . eta", np.trace(R @ hat(eta)), -vee(R - R.T) @ eta)

# small-error stiffness of the weighted error: e_D ~ (tr(D) I - D) eta
small = 1e-6 * eta
ck.close("near the target e_D = (tr(D) I - D) eta", e_D(Rd @ expm_so3(small), Rd, D) / 1e-6,
         (np.trace(D) * I3 - D) @ eta, tol=1e-5)

# critical points of Psi_D with distinct weights g1 > g2 > g3 > 0
g1, g2, g3 = np.diag(D)
flips = {"identity": np.diag([1.0, 1, 1]), "flip about e1": np.diag([1.0, -1, -1]),
         "flip about e2": np.diag([-1.0, 1, -1]), "flip about e3": np.diag([-1.0, -1, 1])}
want_val = {"identity": 0.0, "flip about e1": 2 * (g2 + g3), "flip about e2": 2 * (g1 + g3), "flip about e3": 2 * (g1 + g2)}
want_index = {"identity": 0, "flip about e1": 1, "flip about e2": 2, "flip about e3": 3}


def hessian(E, G_):
    f = lambda x: np.trace(G_ @ (I3 - E @ expm_so3(x)))
    H, h = np.zeros((3, 3)), 1e-4
    for i in range(3):
        for j in range(3):
            H[i, j] = (f(h * (I3[i] + I3[j])) - f(h * (I3[i] - I3[j])) - f(h * (I3[j] - I3[i])) + f(-h * (I3[i] + I3[j]))) / (4 * h * h)
    return H


for name, E in flips.items():
    ck.close(f"weighted error: e_D = 0 at {name}", e_D(E, I3, D), np.zeros(3))
    ck.close(f"weighted error: Psi_D at {name}", Psi_D(E, I3, D), want_val[name])
    eig = np.linalg.eigvalsh(hessian(E, D))
    d = np.diag(D @ E)
    ck.close(f"Hessian eigenvalues at {name} are sums of the other two d_i", np.sort(eig), np.sort(d.sum() - d), tol=1e-5)
    ck.true(f"Morse index at {name} is {want_index[name]}", int(np.sum(eig < -1e-6)) == want_index[name])
ck.record("weighted_values", {k: float(v) for k, v in want_val.items()})
# no other critical points: a random search for zeros of e_D finds only these four
found = set()
for _ in range(300):
    x = log_so3(random_rotation(rng))
    for _ in range(60):                                       # Newton on e_D(exp(x)) = 0
        E = expm_so3(x)
        F = e_D(E, I3, D)
        Jn = np.column_stack([(e_D(expm_so3(x + 1e-6 * I3[i]), I3, D) - F) / 1e-6 for i in range(3)])
        try:
            x = x - np.linalg.solve(Jn, F)
        except np.linalg.LinAlgError:
            break
    E = expm_so3(x)
    if np.linalg.norm(e_D(E, I3, D)) < 1e-9:
        found.add(tuple(np.round(np.diag(E)).astype(int)) if np.allclose(E, np.diag(np.diag(E)), atol=1e-6) else "other")
ck.true("distinct weights: exactly four critical points", found == {(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)}, str(found))
# equal weights: every half turn is critical
u = unit(rng.normal(size=3))
ck.close("equal weights: e_R = 0 at every half turn", e_R(expm_so3(np.pi * u), I3), np.zeros(3), tol=1e-12)
ck.close("equal weights: Psi_R = 2 at every half turn", Psi_R(expm_so3(np.pi * u), I3), 2.0)

# the alternatives, as functions of the error angle about a fixed axis
th, u = 1.9, unit(rng.normal(size=3))
E = expm_so3(th * u)
ck.close("trace error: Psi = 1 - cos(theta), e = sin(theta) u", np.r_[Psi_R(E, I3), e_R(E, I3)], np.r_[1 - np.cos(th), np.sin(th) * u])
ck.close("log error: log(E) = theta u", log_so3(E), th * u)
ck.close("d theta = u . eta  (so 1/2 theta^2 has slope theta u)",
         fd(lambda e: 0.5 * np.linalg.norm(log_so3(E @ expm_so3(e * eta))) ** 2, 0.0), th * u @ eta, tol=1e-7)


def quat_of(Rm):
    w = log_so3(Rm)
    a = np.linalg.norm(w)
    return np.r_[np.cos(a / 2), np.sin(a / 2) * w / a]


ck.close("quaternion error: q_v = sin(theta/2) u", quat_of(E)[1:], np.sin(th / 2) * u)
ck.close("Psi = 4(1 - q0) has slope 2 q_v = 2 sin(theta/2) u",
         fd(lambda e: 4 * (1 - quat_of(E @ expm_so3(e * eta))[0]), 0.0), 2 * np.sin(th / 2) * u @ eta, tol=1e-7)
ck.close("4(1 - q0) = 4(1 - cos(theta/2))", 4 * (1 - quat_of(E)[0]), 4 * (1 - np.cos(th / 2)))
for f_, name_ in [(np.sin, "trace"), (lambda a: a, "log"), (lambda a: 2 * np.sin(a / 2), "quaternion")]:
    ck.close(f"{name_} slope has unit stiffness at zero", fd(f_, 0.0), 1.0, tol=1e-9)
ck.close("sin peaks at 90 degrees", np.sin(np.pi / 2), 1.0)
ck.record("restoring_at_170deg", {"trace_sin": float(np.sin(np.radians(170))), "log_theta": float(np.radians(170)),
                                  "quat_2sin_half": float(2 * np.sin(np.radians(85)))})

# tilt and yaw split (Mueller 2018, eq. 41): E = E_tilt E_yaw
E = random_rotation(rng)
rho_e = np.linalg.norm(log_so3(E))
rho_r = np.arccos(E[2, 2])                                     # angle between e3 and E e3
n_r = unit(np.cross(E3, E @ E3))
E_tilt = expm_so3(rho_r * n_r)
E_yaw = E_tilt.T @ E
ck.close("tilt rotation carries e3 to E e3", E_tilt @ E3, E @ E3)
ck.close("the remainder is a pure yaw (fixes e3)", E_yaw @ E3, E3)
rho_y = np.linalg.norm(log_so3(E_yaw))
ck.close("cos(rho_e/2) = cos(rho_r/2) cos(rho_y/2)", np.cos(rho_e / 2), np.cos(rho_r / 2) * np.cos(rho_y / 2))
ck.true("tilt angle <= total angle", rho_r <= rho_e + 1e-12)
Rd_, R_ = random_rotation(rng), random_rotation(rng)
E_ = Rd_.T @ R_
ck.close("tilt height is the S^2 error of the thrust axis: 1 - cos(rho_r) = 1 - (R e3).(Rd e3)",
         1 - E_[2, 2], 1 - (R_ @ E3) @ (Rd_ @ E3))


# lingering: one axis, theta_ddot = -kp f(theta) - kd theta_dot, started at rest
kp, kd = 4.0, 2 * np.sqrt(2)
laws = {"trace": np.sin, "log": lambda a: a, "quat": lambda a: 2 * np.sin(a / 2)}


def settle_time(law, th0_deg, band_deg=5.0, T=40.0):
    sol = solve_ivp(lambda t, y: [y[1], -kp * laws[law](y[0]) - kd * y[1]], [0, T], [np.radians(th0_deg), 0.0],
                    rtol=1e-10, atol=1e-12, dense_output=True)
    tt = np.linspace(0, T, 40001)
    a = np.abs(np.degrees(sol.sol(tt)[0]))
    outside = np.nonzero(a > band_deg)[0]
    return float(tt[outside[-1] + 1])


times = {law: {s: settle_time(law, s) for s in (179.0, 179.9)} for law in laws}
ck.record("linger_gains", {"kp": kp, "kd": float(kd)})
ck.record("linger_settle_times_s", times)
lam = (-kd + np.sqrt(kd**2 + 4 * kp)) / 2
ck.record("linger_escape_rate", float(lam))
ck.record("linger_seconds_per_decade", float(np.log(10) / lam))
ck.close("trace law: each factor of ten closer to 180 deg costs ln(10)/lambda seconds",
         times["trace"][179.9] - times["trace"][179.0], np.log(10) / lam, tol=0.02)
ck.true("log and quaternion laws do not care about the last degree",
        abs(times["log"][179.9] - times["log"][179.0]) < 0.02 and abs(times["quat"][179.9] - times["quat"][179.0]) < 0.02)
ck.true("trace law is slowest from 179 deg", times["trace"][179.0] > max(times["log"][179.0], times["quat"][179.0]))
ck.true("quoted settle times (trace 6.1, log 2.8, quat 2.9 s from 179 deg)",
        abs(times["trace"][179.0] - 6.1) < 0.06 and abs(times["log"][179.0] - 2.8) < 0.06 and abs(times["quat"][179.0] - 2.9) < 0.06,
        str({k: round(v[179.0], 2) for k, v in times.items()}))
ck.true("quoted: trace law from 179.9 deg takes 8.4 s", abs(times["trace"][179.9] - 8.4) < 0.06, f"{times['trace'][179.9]:.2f}")
ck.true("quoted: 2.2 s per decade", abs(np.log(10) / lam - 2.2) < 0.05, f"{np.log(10) / lam:.3f}")


# =====================================================================
# 4. Transport and velocity error on SO(3)
# =====================================================================
def Omd_fun(t):
    return np.array([0.3 * np.cos(t), -0.5, 0.4 * np.sin(2 * t)])


sol_Rd = solve_ivp(lambda t, y: (y.reshape(3, 3) @ hat(Omd_fun(t))).ravel(), [0, 2], random_rotation(rng).ravel(),
                   rtol=1e-12, atol=1e-13, dense_output=True)


def Rd_ref(t):
    return sol_Rd.sol(t).reshape(3, 3)


t0 = 1.1
R, Rd, Om, Omd = R_ref(t0), Rd_ref(t0), Omega_fun(t0), Omd_fun(t0)
E = Rd.T @ R
e_Om = Om - R.T @ Rd @ Omd
ck.close("d/dt (Rd^T R) = (Rd^T R) hat(e_Omega)", fd(lambda s: Rd_ref(s).T @ R_ref(s), t0), E @ hat(e_Om), tol=1e-7)
ck.close("transport is the adjoint: R^T Rd Omd = Ad_{E^-1} Omd", hat(R.T @ Rd @ Omd), E.T @ hat(Omd) @ E)
ck.close("transport is right translation: R hat(R^T Rd Omd) = Rd_dot (Rd^T R)", R @ hat(R.T @ Rd @ Omd), Rd @ hat(Omd) @ Rd.T @ R)
ck.close("transported velocity is the same world-frame angular velocity", R @ (R.T @ Rd @ Omd), Rd @ Omd)
dPsi = fd(lambda s: Psi_R(R_ref(s), Rd_ref(s)), t0)
ck.close("d/dt Psi_R = e_R . e_Omega", dPsi, e_R(R, Rd) @ e_Om, tol=1e-7)
ck.close("with equal weights, d/dt Psi_R = e_R . (Omega - Omega_d) as well", dPsi, e_R(R, Rd) @ (Om - Omd), tol=1e-7)
ck.close("reason: e_R is the rotation axis, so (Rd^T R) e_R = e_R", E @ e_R(R, Rd), e_R(R, Rd))
dPsiG = fd(lambda s: Psi_D(R_ref(s), Rd_ref(s), D), t0)
ck.close("weighted: d/dt Psi_D = e_D . e_Omega", dPsiG, e_D(R, Rd, D) @ e_Om, tol=1e-7)
ck.true("weighted: e_D . (Omega - Omega_d) is NOT d/dt Psi_D", abs(dPsiG - e_D(R, Rd, D) @ (Om - Omd)) > 1e-2,
        f"gap {abs(dPsiG - e_D(R, Rd, D) @ (Om - Omd)):.3f}")
C = 0.5 * (np.trace(R.T @ Rd) * I3 - R.T @ Rd)
ck.close("d/dt e_R = 1/2 (tr(R^T Rd) I - R^T Rd) e_Omega", fd(lambda s: e_R(R_ref(s), Rd_ref(s)), t0), C @ e_Om, tol=1e-7)
ck.close("C = I at zero error", 0.5 * (np.trace(I3) * I3 - I3), I3)
worst = max(np.linalg.norm(0.5 * (np.trace(Q_) * I3 - Q_), 2) for Q_ in (random_rotation(rng) for _ in range(2000)))
ck.true("operator norm of C is at most 1 (2000 random rotations)", worst <= 1 + 1e-12, f"max {worst:.4f}")
A_, x_ = rng.normal(size=(3, 3)), rng.normal(size=3)
ck.close("(A hat(x) + hat(x) A^T)^vee = (tr(A) I - A^T) x", vee(A_ @ hat(x_) + hat(x_) @ A_.T), (np.trace(A_) * I3 - A_.T) @ x_)


# =====================================================================
# 5. Transport and velocity error on S^2
# =====================================================================
def qd_ref(t):
    return unit([0.4 + np.sin(0.9 * t), np.cos(0.6 * t), 0.8 + 0.2 * t])


t0 = 0.6
q, qd = q_ref(t0), qd_ref(t0)
qdot, qddot = fd(q_ref, t0), fd(qd_ref, t0)
om, omd = np.cross(q, qdot), np.cross(qd, qddot)
Pq = I3 - np.outer(q, q)
e_q = np.cross(qd, q)
e_om = om - Pq @ omd
dPsi_q = fd(lambda s: 1 - q_ref(s) @ qd_ref(s), t0)
ck.close("delta Psi_q = e_q . xi", fd(lambda e: 1 - (expm_so3(e * xi_fun(0.8)) @ q_ref(0.8)) @ qd, 0.0),
         np.cross(qd, q_ref(0.8)) @ xi_fun(0.8), tol=1e-8)
ck.close("d/dt Psi_q = e_q . e_omega, e_omega = omega - (I - q q^T) omega_d", dPsi_q, e_q @ e_om, tol=1e-7)
ck.close("e_omega is tangent at q", e_om @ q, 0.0, tol=1e-8)
ck.close("-hat(q)^2 = I - q q^T", -hat(q) @ hat(q), Pq)
ck.close("e_q is perpendicular to both q and q_d", [e_q @ q, e_q @ qd], [0, 0], tol=1e-12)
ck.close("d/dt Psi_q = e_q . (omega - omega_d) too", dPsi_q, e_q @ (om - omd), tol=1e-7)
e_qdot = qdot - np.cross(np.cross(qd, qddot), q)
ck.close("velocity form: q_dot - (q_d x q_d_dot) x q = e_omega x q", e_qdot, np.cross(e_om, q), tol=1e-8)
grad = -(Pq @ qd)
ck.close("gradient of Psi_q at q is -(I - q q^T) q_d, and e_q = q x grad", np.cross(q, grad), e_q)
ck.close("d/dt Psi_q = grad . e_qdot", dPsi_q, grad @ e_qdot, tol=1e-7)
deq = fd(lambda s: np.cross(qd_ref(s), q_ref(s)), t0)
ck.close("d/dt e_q = (q.q_d) e_omega + (q.omega_d) q x e_q - (q_d.omega) q",
         deq, (q @ qd) * e_om + (q @ omd) * np.cross(q, e_q) - (qd @ om) * q, tol=1e-7)

# the transport map against parallel transport along the connecting great circle
thq = 1.1
qd_, q_ = E3, np.array([np.sin(thq), 0, np.cos(thq)])
along, across = E1, E2                                         # reference velocities at q_d
T = lambda v: np.cross(np.cross(qd_, v), q_)                   # q_d_dot -> (q_d x q_d_dot) x q
Qpar = expm_so3(thq * unit(np.cross(qd_, q_)))                 # parallel transport along the geodesic
ck.close("along the great circle the transport map is parallel transport", T(along), Qpar @ along)
ck.close("across it, the transport map shrinks the vector by cos(theta)", T(across), np.cos(thq) * (Qpar @ across))
q90 = E1
ck.close("at 90 degrees the sideways velocity is transported to zero", np.cross(np.cross(E3, E2), q90), np.zeros(3))
ck.close("example: (q_d, q, q_d_dot) = (e3, e1, e1) transports to -e3", np.cross(np.cross(E3, E1), E1), -E3)
omd_ = np.cross(qd_, 0.7 * along - 0.4 * across)
eq_ = np.cross(qd_, q_)
ck.close("parallel transport is compatible too: e_q . (Q omega_d) = e_q . omega_d", eq_ @ (Qpar @ omd_), eq_ @ omd_)

# holonomy: carry an arrow around one octant of the sphere
def carry(v, a, b):
    """Parallel transport of tangent vector v from a to b along the short great circle."""
    return expm_so3(np.arccos(np.clip(a @ b, -1, 1)) * unit(np.cross(a, b))) @ v


v0 = E1                                                        # at the pole e3, pointing down the meridian to e1
v = carry(carry(carry(v0, E3, E1), E1, E2), E2, E3)
ck.close("around the octant the arrow comes back as e2", v, E2)
ck.close("holonomy angle = enclosed area = pi/2", np.arccos(v0 @ v), 4 * np.pi / 8)
ck.record("holonomy_octant_deg", float(np.degrees(np.arccos(v0 @ v))))
# north at two places are different vectors of R^3
def north(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])


ck.close("north at the equator and at 60N differ by 60 degrees as vectors",
         np.degrees(np.arccos(north(0, 0) @ north(60, 0))), 60.0)


# =====================================================================
# 6. Left- and right-invariant errors
# =====================================================================
L = random_rotation(rng)
R, Rd = random_rotation(rng), random_rotation(rng)
ck.close("left error Rd^T R is unchanged by a new world frame (R, Rd) -> (L R, L Rd)", (L @ Rd).T @ (L @ R), Rd.T @ R)
ck.close("right error R Rd^T is unchanged by a new body frame (R, Rd) -> (R L, Rd L)", (R @ L) @ (Rd @ L).T, R @ Rd.T)
ck.true("left error changes with the body frame", np.linalg.norm((Rd @ L).T @ (R @ L) - Rd.T @ R) > 0.1)
ck.true("right error changes with the world frame", np.linalg.norm((L @ R) @ (L @ Rd).T - R @ Rd.T) > 0.1)
ck.close("both errors have the same angle", np.trace(Rd.T @ R), np.trace(R @ Rd.T))

t0 = 1.1
R, Rd, Om, Omd = R_ref(t0), Rd_ref(t0), Omega_fun(t0), Omd_fun(t0)
ck.close("right error: d/dt (R Rd^T) = R hat(Omega - Omega_d) Rd^T", fd(lambda s: R_ref(s) @ Rd_ref(s).T, t0),
         R @ hat(Om - Omd) @ Rd.T, tol=1e-7)
w, wd, Er = R @ Om, Rd @ Omd, R @ Rd.T
ck.close("right error with world-frame rates: d/dt Er = hat(w) Er - Er hat(wd)", fd(lambda s: R_ref(s) @ Rd_ref(s).T, t0),
         hat(w) @ Er - Er @ hat(wd), tol=1e-7)

# dead reckoning: truth and estimate integrate the same gyro signal
def gyro(t):
    return np.array([1.2 * np.sin(1.3 * t), 0.9 * np.cos(0.7 * t), 0.8 + 0.5 * np.sin(2.1 * t)])


Rt0 = Rz(0.3) @ Rx(-0.2)
Rh0 = Rt0 @ expm_so3(np.radians(5.0) * unit([1.0, 2.0, -1.0]))


def two_frames(t, y):
    A, B = y[:9].reshape(3, 3), y[9:].reshape(3, 3)
    return np.r_[(A @ hat(gyro(t))).ravel(), (B @ hat(gyro(t))).ravel()]


sol2 = solve_ivp(two_frames, [0, 10], np.r_[Rt0.ravel(), Rh0.ravel()], rtol=1e-12, atol=1e-13, dense_output=True)


def euler_zyx(Rm):
    return np.array([np.arctan2(Rm[1, 0], Rm[0, 0]), -np.arcsin(np.clip(Rm[2, 0], -1, 1)), np.arctan2(Rm[2, 1], Rm[2, 2])])


tt = np.linspace(0, 10, 2001)
right_err, left_ang, eul = [], [], []
for t in tt:
    A, B = sol2.sol(t)[:9].reshape(3, 3), sol2.sol(t)[9:].reshape(3, 3)
    right_err.append(log_so3(B @ A.T))
    left_ang.append(np.linalg.norm(log_so3(A.T @ B)))
    d = euler_zyx(B) - euler_zyx(A)
    eul.append(np.linalg.norm((d + np.pi) % (2 * np.pi) - np.pi))
right_err, left_ang, eul = np.array(right_err), np.array(left_ang), np.degrees(np.array(eul))
ck.close("dead reckoning: right-invariant error R_hat R^T does not move at all", right_err, np.tile(right_err[0], (len(tt), 1)), tol=1e-9)
ck.close("dead reckoning: the left-invariant error keeps its angle (5 deg)", np.degrees(left_ang), 5.0, tol=1e-7)
ck.true("dead reckoning: the Euler-angle difference wanders", eul.max() - eul.min() > 1.0, f"{eul.min():.2f} to {eul.max():.2f} deg")
ck.record("dead_reckoning_euler_diff_deg", {"min": float(eul.min()), "max": float(eul.max())})
ck.true("quoted Euler wander 4.6 to 64 deg", abs(eul.min() - 4.6) < 0.06 and abs(eul.max() - 64) < 0.5, f"{eul.min():.2f}, {eul.max():.2f}")
# left-invariant estimation error: dE/dt = E hat(Om) - hat(Om) E, no R anywhere
t0 = 3.3
A, B = sol2.sol(t0)[:9].reshape(3, 3), sol2.sol(t0)[9:].reshape(3, 3)
dEl = fd(lambda s: sol2.sol(s)[:9].reshape(3, 3).T @ sol2.sol(s)[9:].reshape(3, 3), t0)
El = A.T @ B
ck.close("left estimation error obeys dE/dt = E hat(Om) - hat(Om) E", dEl, El @ hat(gyro(t0)) - hat(gyro(t0)) @ El, tol=1e-6)

# a general matrix group: SE(3), tracking error E = Xd^-1 X
def se3_hat(xi):
    M = np.zeros((4, 4))
    M[:3, :3] = hat(xi[:3])
    M[:3, 3] = xi[3:]
    return M


def se3_vee(M):
    return np.r_[vee(M[:3, :3]), M[:3, 3]]


def xi_a(t):
    return np.array([np.sin(t), 0.4, -0.3 * t, 1.0, 0.2 * np.cos(t), 0.5])


def xi_b(t):
    return np.array([0.2, -0.6 * np.cos(t), 0.5, 0.3 * t, 1.0, -0.4])


def flow(xi, X0):
    return solve_ivp(lambda t, y: (y.reshape(4, 4) @ se3_hat(xi(t))).ravel(), [0, 2], X0.ravel(), rtol=1e-12, atol=1e-13, dense_output=True)


X0 = expm(se3_hat(rng.normal(size=6)))
Xd0 = expm(se3_hat(rng.normal(size=6)))
sX, sXd = flow(xi_a, X0), flow(xi_b, Xd0)
Xf = lambda t: sX.sol(t).reshape(4, 4)
Xdf = lambda t: sXd.sol(t).reshape(4, 4)
t0 = 0.9
E = np.linalg.inv(Xdf(t0)) @ Xf(t0)
Ad_inv_xid = se3_vee(np.linalg.inv(E) @ se3_hat(xi_b(t0)) @ E)
ck.close("SE(3): d/dt (Xd^-1 X) = E hat(xi - Ad_{E^-1} xi_d)", fd(lambda s: np.linalg.inv(Xdf(s)) @ Xf(s), t0),
         E @ se3_hat(xi_a(t0) - Ad_inv_xid), tol=1e-6)
# same error, same velocities, different absolute state: identical error rate
Lg = expm(se3_hat(rng.normal(size=6)))
sX2, sXd2 = flow(xi_a, Lg @ X0), flow(xi_b, Lg @ Xd0)
E2dot = fd(lambda s: np.linalg.inv(sXd2.sol(s).reshape(4, 4)) @ sX2.sol(s).reshape(4, 4), t0)
ck.close("SE(3): the error rate is the same wherever the pair sits", E2dot, fd(lambda s: np.linalg.inv(Xdf(s)) @ Xf(s), t0), tol=1e-6)

# Lie-algebra error psi = log(Rd^T R): first-order dynamics  psi_dot = -Omega_d x psi + (Omega - Omega_d)
eps = 1e-3
sol_small = solve_ivp(lambda t, y: (y.reshape(3, 3) @ hat(Omd_fun(t) + eps * np.array([np.sin(t), 1.0, -0.5]))).ravel(),
                      [0, 2], (Rd_ref(0) @ expm_so3(eps * np.array([0.3, -1.0, 0.8]))).ravel(), rtol=1e-12, atol=1e-13, dense_output=True)
psi = lambda s: log_so3(Rd_ref(s).T @ sol_small.sol(s).reshape(3, 3))
t0 = 1.2
dOmega = eps * np.array([np.sin(t0), 1.0, -0.5])
ck.close("psi_dot = -Omega_d x psi + (Omega - Omega_d) to first order", fd(psi, t0) / eps,
         (-np.cross(Omd_fun(t0), psi(t0)) + dOmega) / eps, tol=5e-3)
ck.close("ad on so(3) is the cross product: [hat(a), hat(b)] = hat(a x b)",
         hat(Om) @ hat(Omd) - hat(Omd) @ hat(Om), hat(np.cross(Om, Omd)))


# =====================================================================
# 7. First-order equivalence between errors and variations
# =====================================================================
t0 = 1.0
Rd, Omd = Rd_ref(t0), Omd_fun(t0)
eta0, deta0 = eta_fun(t0), fd(eta_fun, t0)
ck.record("first_order_eta_norm", float(np.linalg.norm(eta0)))
ck.true("quoted: |eta| = 0.86 in the first-order table", abs(np.linalg.norm(eta0) - 0.86) < 0.005)
table = []
for eps in (0.4, 0.2, 0.1, 0.05):
    Rp = lambda s, e=eps: Rd_ref(s) @ expm_so3(e * eta_fun(s))
    Rn = Rp(t0)
    Omn = vee(Rn.T @ fd(Rp, t0))
    eR = e_R(Rn, Rd)
    eOm = Omn - Rn.T @ Rd @ Omd
    table.append((eps, np.linalg.norm(eR - eps * eta0), np.linalg.norm(eOm - eps * deta0),
                  np.linalg.norm(eOm - eps * (deta0 + np.cross(Omd, eta0)))))
table = np.array(table)
ck.close("e_R = sin(eps |eta|) eta/|eta| exactly", e_R(Rd @ expm_so3(0.3 * eta0), Rd),
         np.sin(0.3 * np.linalg.norm(eta0)) * unit(eta0))
ck.true("e_R - eps eta shrinks like eps^3", np.all(np.abs(np.log2(table[:-1, 1] / table[1:, 1]) - 3) < 0.1),
        str(np.round(np.log2(table[:-1, 1] / table[1:, 1]), 2)))
ck.true("e_Omega - eps eta_dot shrinks at least like eps^2", np.all(np.log2(table[:-1, 2] / table[1:, 2]) > 1.9),
        str(np.round(np.log2(table[:-1, 2] / table[1:, 2]), 2)))
ck.true("e_Omega - eps delta Omega shrinks only like eps (they differ at first order)",
        np.all(np.abs(np.log2(table[:-1, 3] / table[1:, 3]) - 1) < 0.2), str(np.round(np.log2(table[:-1, 3] / table[1:, 3]), 2)))
ck.close("first order: delta Omega = e_Omega + Omega_d x e_R", (table[-1, 0] * (deta0 + np.cross(Omd, eta0))),
         (lambda Rn, Omn: (Omn - Rn.T @ Rd @ Omd) + np.cross(Omd, e_R(Rn, Rd)))(
             Rd @ expm_so3(0.05 * eta0), vee((Rd @ expm_so3(0.05 * eta0)).T @ fd(lambda s: Rd_ref(s) @ expm_so3(0.05 * eta_fun(s)), t0))),
         tol=2e-3)
ck.record("first_order_table_SO3", [{"eps": float(r[0]), "eR_minus_eps_eta": float(r[1]), "eOm_minus_eps_etadot": float(r[2])} for r in table])
ck.true("quoted table: e_R gap 6.8e-3, 8.6e-4, 1.1e-4, 1.3e-5", np.allclose(table[:, 1], [6.8e-3, 8.6e-4, 1.1e-4, 1.3e-5], rtol=0.04))
ck.true("quoted table: e_Omega gap 3.8e-2, 9.6e-3, 2.4e-3, 6.0e-4", np.allclose(table[:, 2], [3.8e-2, 9.6e-3, 2.4e-3, 6.0e-4], rtol=0.02))
ck.true("quoted table: e_Omega vs eps delta Omega gap 2.3e-1, 1.1e-1, 5.4e-2, 2.7e-2", np.allclose(table[:, 3], [2.3e-1, 1.1e-1, 5.4e-2, 2.7e-2], rtol=0.03))
shortfall = lambda deg: 1 - np.sin(np.radians(deg)) / np.radians(deg)
ck.record("eR_shortfall_percent", {"10deg": float(100 * shortfall(10)), "30deg": float(100 * shortfall(30))})
ck.true("quoted: e_R is 0.5% short at 10 deg and 4.5% short at 30 deg",
        abs(100 * shortfall(10) - 0.5) < 0.05 and abs(100 * shortfall(30) - 4.5) < 0.05,
        f"{100 * shortfall(10):.2f}%, {100 * shortfall(30):.2f}%")

# the worked example: eta = (0.1, 0, 0.2) about a fixed target
eta_ex = np.array([0.1, 0.0, 0.2])
eR_ex = e_R(expm_so3(eta_ex), I3)
ck.record("example_eR", [float(x) for x in eR_ex])
ck.close("worked example: e_R = (0.0992, 0, 0.1983)", eR_ex, [0.0992, 0.0, 0.1983], tol=6e-5)
ck.record("example_angle_deg", float(np.degrees(np.linalg.norm(eta_ex))))
ck.true("worked example: the angle is 12.8 deg", abs(np.degrees(np.linalg.norm(eta_ex)) - 12.8) < 0.05)

# S^2
t0 = 1.5
qd, omd = qd_ref(t0), np.cross(qd_ref(t0), fd(qd_ref, t0))


def xi_d(t):
    """Variation field perpendicular to the reference direction."""
    q = qd_ref(t)
    return a_fun(t) - (a_fun(t) @ q) * q


xi0, xidot0 = xi_d(t0), fd(xi_d, t0)
ck.true("test point: omega_d x xi is not small", np.linalg.norm(np.cross(omd, xi0)) > 0.1, f"{np.linalg.norm(np.cross(omd, xi0)):.3f}")
rows = []
for eps in (0.4, 0.2, 0.1, 0.05):
    qp = lambda s, e=eps: expm_so3(e * xi_d(s)) @ qd_ref(s)
    qn = qp(t0)
    omn = np.cross(qn, fd(qp, t0))
    eq = np.cross(qd, qn)
    eom = omn - (I3 - np.outer(qn, qn)) @ omd
    rows.append((eps, np.linalg.norm(eq - eps * xi0), np.linalg.norm(eom - eps * (xidot0 - np.cross(omd, xi0))),
                 np.linalg.norm(eom - eps * (xidot0 - 2 * np.cross(omd, xi0)))))
rows = np.array(rows)
ck.close("e_q = sin(eps |xi|) xi/|xi| exactly", np.cross(qd, expm_so3(0.3 * xi0) @ qd), np.sin(0.3 * np.linalg.norm(xi0)) * unit(xi0))
ck.true("e_q - eps xi shrinks like eps^3", np.all(np.abs(np.log2(rows[:-1, 1] / rows[1:, 1]) - 3) < 0.1),
        str(np.round(np.log2(rows[:-1, 1] / rows[1:, 1]), 2)))
ck.true("e_omega - eps (xi_dot - omega_d x xi) shrinks at least like eps^2", np.all(np.log2(rows[:-1, 2] / rows[1:, 2]) > 1.9),
        str(np.round(np.log2(rows[:-1, 2] / rows[1:, 2]), 2)))
ck.true("e_omega - eps delta omega shrinks only like eps", np.all(np.abs(np.log2(rows[:-1, 3] / rows[1:, 3]) - 1) < 0.3),
        str(np.round(np.log2(rows[:-1, 3] / rows[1:, 3]), 2)))

ck.finish()
