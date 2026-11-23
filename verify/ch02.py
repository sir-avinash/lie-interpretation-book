"""Chapter 2: every identity and every quoted number, checked numerically.

    python verify/ch02.py
"""
import numpy as np
from scipy.linalg import expm, logm

from common import Checker, Rx, Rz, expm_so3, hat, random_rotation, vee

ck = Checker("ch02")
rng = np.random.default_rng(2)
E1, E2, E3 = np.eye(3)
I3 = np.eye(3)


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def log_so3(R):
    """Rotation vector of R, valid for angles below pi."""
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-10:
        return vee(R - R.T) / 2
    return th / (2 * np.sin(th)) * vee(R - R.T)


def angle_of(R):
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))


# ---- opening puzzle: Euler steps stretch a unit vector ---------------------
h, w, T = 0.01, 2.0, 10.0
N = round(T / h)
q = E1.copy()
om = w * E3                                  # perpendicular to q, and stays so
for _ in range(N):
    q = q + h * np.cross(om, q)
ck.close("Euler on S2: |q_N| = (1 + h^2 w^2)^(N/2)", np.linalg.norm(q), (1 + h**2 * w**2) ** (N / 2), tol=1e-12)
ck.true("cable grows to about 1.22 after 10 s", abs(np.linalg.norm(q) - 1.22) < 0.005, f"{np.linalg.norm(q):.4f}")
ck.close("growth is close to exp(h w^2 T / 2)", np.linalg.norm(q), np.exp(h * w**2 * T / 2), tol=1e-4)
ck.record("puzzle_cable_length", float(np.linalg.norm(q)))
ck.record("puzzle_steps", N)
qr, wr = unit(rng.normal(size=3)), rng.normal(size=3)
ck.close("one Euler step: |q + h w x q|^2 = 1 + h^2 |w x q|^2",
         np.linalg.norm(qr + h * np.cross(wr, qr)) ** 2, 1 + h**2 * np.linalg.norm(np.cross(wr, qr)) ** 2, tol=1e-14)


# ---- circle and sphere: tangent vectors -------------------------------------
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
th0, thd = 0.7, 1.3
qc = np.array([np.cos(th0), np.sin(th0)])
qc_dot = thd * np.array([-np.sin(th0), np.cos(th0)])
ck.close("S1: q_dot = omega J q", qc_dot, thd * J2 @ qc)
ck.close("S1: q . q_dot = 0", qc @ qc_dot, 0.0)


def curve(t):                                # a curve on the sphere
    return unit([np.cos(t), np.sin(2 * t), 0.5 + 0.3 * t])


t0, d = 0.4, 1e-6
qs = curve(t0)
qs_dot = (curve(t0 + d) - curve(t0 - d)) / (2 * d)
ck.close("S2: velocity of a curve is perpendicular to q", qs @ qs_dot, 0.0, tol=1e-8)
om_s = np.cross(qs, qs_dot)
ck.close("S2: omega = q x q_dot reproduces q_dot = omega x q", np.cross(om_s, qs), qs_dot, tol=1e-8)
ck.close("S2: that omega is perpendicular to q", om_s @ qs, 0.0, tol=1e-12)
lam = 0.83
ck.close("S2: adding a multiple of q to omega changes nothing", np.cross(om_s + lam * qs, qs), np.cross(om_s, qs), tol=1e-12)
ck.close("S2: |q_dot| = |omega| when omega . q = 0", np.linalg.norm(qs_dot), np.linalg.norm(om_s), tol=1e-8)

# Example in the text: q = e3 (north pole), velocity due east and north.
qn = E3
v_n = np.array([0.3, 0.4, 0.0])
ck.close("example: omega = q x q_dot at the pole", np.cross(qn, v_n), [-0.4, 0.3, 0.0])
ck.record("example_tangent_speed", float(np.linalg.norm(v_n)))


# ---- SO(3): tangent space, body and spatial velocity, adjoint ---------------
R = random_rotation(rng)
Om = rng.normal(size=3)
x, y = rng.normal(size=3), rng.normal(size=3)


def Rcurve(t):
    return R @ expm_so3(t * Om + 0.5 * t**2 * x)


Rdot = (Rcurve(d) - Rcurve(-d)) / (2 * d)
S = R.T @ Rdot
ck.close("R^T R_dot is skew-symmetric", S + S.T, np.zeros((3, 3)), tol=1e-8)
ck.close("R_dot = R hat(Omega)", Rdot, R @ hat(Om), tol=1e-8)
ck.close("R hat(x) R^T = hat(R x)", R @ hat(x) @ R.T, hat(R @ x))
om_sp = R @ Om
ck.close("R_dot = hat(omega) R with omega = R Omega", Rdot, hat(om_sp) @ R, tol=1e-8)
b = unit(rng.normal(size=3))
ck.close("a body-fixed direction q = R b moves as q_dot = omega x q", Rdot @ b, np.cross(om_sp, R @ b), tol=1e-8)
# Example in the text: phone flat on a turntable, then stood upright.
Om_flat = I3.T @ (0.5 * E3)
R_side = Rx(np.pi / 2)
Om_side = R_side.T @ (0.5 * E3)
ck.close("turntable, phone flat: gyro reads (0, 0, 0.5)", Om_flat, [0, 0, 0.5])
ck.close("turntable, phone stood upright (long axis vertical): gyro reads (0, 0.5, 0)", Om_side, [0, 0.5, 0], tol=1e-15)


# ---- exp and log on SO(3) ----------------------------------------------------
phi = rng.normal(size=3)
ck.close("Rodrigues equals the matrix exponential", expm_so3(phi), expm(hat(phi)), tol=1e-12)
phi = 2.5 * unit(rng.normal(size=3))
ck.close("log(exp(phi)) = phi for |phi| < pi", log_so3(expm_so3(phi)), phi, tol=1e-9)
ck.close("log agrees with scipy logm", hat(log_so3(expm_so3(phi))), np.real(logm(expm_so3(phi))), tol=1e-9)
u = unit(rng.normal(size=3))
K = hat(u)
ck.close("hat(u)^3 = -hat(u) for a unit vector", K @ K @ K, -K, tol=1e-14)
ck.close("exp(pi u) = exp(-pi u)", expm_so3(np.pi * u), expm_so3(-np.pi * u), tol=1e-12)
ck.close("exp(t Omega) is the solution of R_dot = R hat(Omega), R(0) = R0",
         (R @ expm_so3((1 + d) * Om) - R @ expm_so3((1 - d) * Om)) / (2 * d), R @ expm_so3(Om) @ hat(Om), tol=1e-7)
ck.close("tr(exp(theta u)) = 1 + 2 cos(theta)", np.trace(expm_so3(1.1 * u)), 1 + 2 * np.cos(1.1))

# Worked example: quarter turn about the diagonal-free axis (1, 1, 0)/sqrt 2.
phi_ex = (np.pi / 2) * unit([1, 1, 0])
R_ex = expm_so3(phi_ex)
s2 = 1 / np.sqrt(2)
ck.close("example exp: closed form", R_ex, [[0.5, 0.5, s2], [0.5, 0.5, -s2], [-s2, s2, 0.0]], tol=1e-14)
ck.close("example exp: body e3 ends up at (1, -1, 0)/sqrt 2", R_ex @ E3, [s2, -s2, 0], tol=1e-14)
ck.close("example log: trace gives the angle back", angle_of(R_ex), np.pi / 2, tol=1e-12)
ck.close("example log: recovers phi", log_so3(R_ex), phi_ex, tol=1e-12)
ck.record("example_exp_matrix", np.round(R_ex, 4).tolist())

# exp on the sphere and the circle
v = rng.normal(size=3)
v -= (v @ qs) * qs                           # tangent at qs
nv = np.linalg.norm(v)
exp_q = np.cos(nv) * qs + np.sin(nv) * v / nv
ck.close("S2 exp: cos|v| q + sin|v| v/|v| is a unit vector", np.linalg.norm(exp_q), 1.0, tol=1e-14)
ck.close("S2 exp: equals rotating q by the vector q x v", exp_q, expm_so3(np.cross(qs, v)) @ qs, tol=1e-13)
ck.close("S2 exp: arc length travelled is |v|", np.arccos(np.clip(exp_q @ qs, -1, 1)), nv if nv <= np.pi else 2 * np.pi - nv, tol=1e-9)
ck.close("SO(2) exp: exp(theta J) is the rotation by theta", expm(th0 * J2), [[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]], tol=1e-13)


# ---- the bracket ---------------------------------------------------------------
ck.close("[hat x, hat y] = hat(x cross y)", hat(x) @ hat(y) - hat(y) @ hat(x), hat(np.cross(x, y)), tol=1e-13)
A, B = Rx(np.pi / 2), Rz(np.pi / 2)
ck.true("quarter turns about x and z do not commute", np.linalg.norm(A @ B - B @ A) > 1.0)
rel = (A @ B).T @ (B @ A)
ck.close("the two orders differ by a 120 degree rotation", np.degrees(angle_of(rel)), 120.0, tol=1e-9)
ck.record("book_orders_differ_deg", float(np.degrees(angle_of(rel))))
# What each order does to the book. Body-frame convention: R = (first)(second).
ck.close("x then z (space-fixed axes): cover normal e3 goes to e1", B @ A @ E3, [1, 0, 0], tol=1e-15)
ck.close("z then x (space-fixed axes): cover normal e3 goes to -e2", A @ B @ E3, [0, -1, 0], tol=1e-15)
ck.close("x then z: spine direction e2 goes to e3", B @ A @ E2, [0, 0, 1], tol=1e-15)
ck.close("z then x: spine direction e2 goes to -e1", A @ B @ E2, [-1, 0, 0], tol=1e-15)

a_, b_ = E1, E2
for eps, key in [(0.1, "commutator_angle_eps_0p1"), (0.01, "commutator_angle_eps_0p01")]:
    C = expm_so3(eps * a_) @ expm_so3(eps * b_) @ expm_so3(-eps * a_) @ expm_so3(-eps * b_)
    c = log_so3(C)
    ck.record(key, float(np.linalg.norm(c)))
    ck.true(f"small-rotation commutator, eps={eps}: log = eps^2 (a x b) + O(eps^3)",
            np.linalg.norm(c - eps**2 * np.cross(a_, b_)) < 1.5 * eps**3, f"{np.linalg.norm(c - eps**2 * np.cross(a_, b_)):.1e}")
C = expm_so3(0.1 * a_) @ expm_so3(0.1 * b_) @ expm_so3(-0.1 * a_) @ expm_so3(-0.1 * b_)
ck.true("eps = 0.1 rad: leftover rotation is about 0.01 rad (0.57 deg), mostly about e3",
        abs(np.linalg.norm(log_so3(C)) - 0.01) < 2e-4 and unit(log_so3(C)) @ E3 > 0.99,
        f"{np.linalg.norm(log_so3(C)):.5f} rad, {np.degrees(np.linalg.norm(log_so3(C))):.3f} deg")
ck.record("commutator_angle_deg_eps_0p1", float(np.degrees(np.linalg.norm(log_so3(C)))))
ck.close("SO(2) is commutative", expm(0.3 * J2) @ expm(1.9 * J2), expm(1.9 * J2) @ expm(0.3 * J2), tol=1e-14)
# Ad is the derivative of conjugation; ad is the bracket.
ck.close("d/dt Ad_exp(t x) y at 0 = x cross y", (expm_so3(d * x) @ y - expm_so3(-d * x) @ y) / (2 * d), np.cross(x, y), tol=1e-8)


# ---- S2 as a homogeneous space ---------------------------------------------------
qa, qb = unit(rng.normal(size=3)), unit(rng.normal(size=3))
ax = np.cross(qa, qb)
Rab = expm_so3(np.arcsin(np.linalg.norm(ax)) * unit(ax)) if qa @ qb > 0 else expm_so3((np.pi - np.arcsin(np.linalg.norm(ax))) * unit(ax))
ck.close("SO(3) acts transitively: some R takes q_a to q_b", Rab @ qa, qb, tol=1e-12)
ck.close("rotations about e3 leave e3 alone (the stabiliser is a circle)", Rz(1.234) @ E3, E3)
ck.close("so R and R Rz(a) give the same direction R e3", R @ Rz(0.77) @ E3, R @ E3, tol=1e-14)
ck.true("dimension count: 3 - 1 = 2", 3 - 1 == 2)


# ---- SE(3) --------------------------------------------------------------------------
def se3(Rm, p):
    g = np.eye(4)
    g[:3, :3], g[:3, 3] = Rm, p
    return g


def hat6(xi):                                # xi = (V, Omega): linear part first
    X = np.zeros((4, 4))
    X[:3, :3], X[:3, 3] = hat(xi[3:]), xi[:3]
    return X


def vee6(X):
    return np.concatenate([X[:3, 3], vee(X[:3, :3])])


def Ad(g):
    Rm, p = g[:3, :3], g[:3, 3]
    return np.block([[Rm, hat(p) @ Rm], [np.zeros((3, 3)), Rm]])


p = rng.normal(size=3)
g = se3(R, p)
xi = rng.normal(size=6)
ck.close("SE(3) inverse is (R^T, -R^T x)", np.linalg.inv(g), se3(R.T, -R.T @ p), tol=1e-13)
ck.close("g maps a body point to the world: R y + x", (g @ np.append(y, 1))[:3], R @ y + p, tol=1e-14)
ck.close("g hat(xi) g^-1 = hat(Ad_g xi)", g @ hat6(xi) @ np.linalg.inv(g), hat6(Ad(g) @ xi), tol=1e-12)
g2 = se3(random_rotation(rng), rng.normal(size=3))
ck.close("Ad_{g1 g2} = Ad_g1 Ad_g2", Ad(g @ g2), Ad(g) @ Ad(g2), tol=1e-12)


def gcurve(t):
    return g @ expm(hat6(t * xi))


gdot = (gcurve(d) - gcurve(-d)) / (2 * d)
ck.close("g_dot = g hat(xi) gives R_dot = R hat(Omega)", gdot[:3, :3], R @ hat(xi[3:]), tol=1e-7)
ck.close("g_dot = g hat(xi) gives x_dot = R V, so V = R^T v", gdot[:3, 3], R @ xi[:3], tol=1e-7)
ck.close("spatial twist = Ad_g (body twist)", vee6(gdot @ np.linalg.inv(g)), Ad(g) @ xi, tol=1e-6)
# The product group R^3 x SO(3) composes differently from SE(3).
(R1, p1), (R2, p2) = (R, p), (g2[:3, :3], g2[:3, 3])
ck.close("SE(3) product: (R1 R2, x1 + R1 x2)", (g @ g2)[:3, 3], p1 + R1 @ p2, tol=1e-13)
ck.true("which differs from the direct product x1 + x2", np.linalg.norm((g @ g2)[:3, 3] - (p1 + p2)) > 0.1)

# SE_2(3): 5x5 matrices (R, v, x)
def se23(Rm, vv, pp):
    X = np.eye(5)
    X[:3, :3], X[:3, 3], X[:3, 4] = Rm, vv, pp
    return X


Xa = se23(R, rng.normal(size=3), rng.normal(size=3))
Xb = se23(g2[:3, :3], rng.normal(size=3), rng.normal(size=3))
Xc = Xa @ Xb
ck.close("SE_2(3) is closed under multiplication: rotation block", Xc[:3, :3].T @ Xc[:3, :3], I3, tol=1e-13)
ck.close("SE_2(3) is closed under multiplication: bottom rows", Xc[3:, :], np.eye(5)[3:, :], tol=1e-15)
ck.close("SE_2(3) inverse is (R^T, -R^T v, -R^T x)", np.linalg.inv(Xa), se23(R.T, -R.T @ Xa[:3, 3], -R.T @ Xa[:3, 4]), tol=1e-13)
ck.true("SE_2(3) has dimension 3 + 3 + 3 = 9", 3 + 3 + 3 == 9)


# ---- product manifolds: the table ---------------------------------------------------
dim = {"R3": 3, "S1": 1, "S2": 2, "SO3": 3, "SE3": 6, "R": 1}
stored = {"R3": 3, "S1": 2, "S2": 3, "SO3": 9, "SE3": 12, "R": 1}


def count(parts):
    return sum(dim[k] for k in parts), sum(stored[k] for k in parts)


rows = {
    "spherical pendulum": (["S2"], 2, 3),
    "3D pendulum": (["SO3"], 3, 9),
    "quadrotor": (["SE3"], 6, 12),
    "quadrotor + cable load": (["R3", "S2", "SO3"], 8, 15),
    "3 quadrotors + rigid payload": (["SE3"] + ["S2", "SO3"] * 3, 21, 48),
    "reaction mass biped (stance)": (["R", "SO3", "SO3", "SO3", "R", "R", "R", "R"], 14, 32),
    "Cassie model": (["SE3", "SO3", "SO3", "R", "R"], 14, 32),
}
for name, (parts, want_dim, want_stored) in rows.items():
    got = count(parts)
    ck.true(f"table: {name} has dimension {want_dim}, stores {want_stored} numbers", got == (want_dim, want_stored), str(got))
ck.true("n quadrotors + rigid payload: 6 + 5n", all(count(["SE3"] + ["S2", "SO3"] * n)[0] == 6 + 5 * n for n in range(1, 6)))
ck.true("quadrotor + load: 15 numbers - 7 constraints = 8", 15 - (1 + 6) == 8)
ck.true("quadrotor + load: 8 dimensions, 4 inputs, 4 unactuated", 8 - 4 == 4)

# Example: quadrotor position and velocity from the load state.
L = 1.0
xL, vL = np.array([2.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
ql = -E3                                      # cable hangs straight down: q points from quadrotor to load
wl = np.array([0.5, 0.0, 0.0])
xQ = xL - L * ql
vQ = vL - L * np.cross(wl, ql)
ck.close("example: quadrotor sits one cable length above the load", xQ, [2.0, 0.0, 2.0])
ck.close("example: quadrotor velocity x_dot_L - l (omega x q)", vQ, [1.0, -0.5, 0.0])


def xQ_of(t):                                # finite-difference check of the velocity relation
    qt = expm_so3(t * wl) @ ql
    return (xL + t * vL) - L * qt


ck.close("example: velocity relation matches finite differences", (xQ_of(d) - xQ_of(-d)) / (2 * d), vQ, tol=1e-8)
ck.record("example_quad_velocity", vQ.tolist())


# ---- integration ----------------------------------------------------------------------
Om1 = rng.normal(size=3)
Rn = R + h * R @ hat(Om1)
ck.close("Euler step: R1^T R1 = I - h^2 hat(Omega)^2", Rn.T @ Rn, I3 - h**2 * hat(Om1) @ hat(Om1), tol=1e-14)
ck.close("Euler step: det R1 = 1 + h^2 |Omega|^2", np.linalg.det(Rn), 1 + h**2 * Om1 @ Om1, tol=1e-13)
Re = R @ expm_so3(h * Om1)
ck.close("exponential step stays orthogonal", Re.T @ Re, I3, tol=1e-14)
ck.close("exponential step keeps det = 1", np.linalg.det(Re), 1.0, tol=1e-14)
qe = expm_so3(h * wr) @ qr
ck.close("exponential step on S2 keeps unit length", np.linalg.norm(qe), 1.0, tol=1e-14)
wperp = wr - (wr @ qr) * qr
ck.close("S2 step in closed form: cos(h|w|) q + sin(h|w|) (w x q)/|w|, for w . q = 0",
         expm_so3(h * wperp) @ qr,
         np.cos(h * np.linalg.norm(wperp)) * qr + np.sin(h * np.linalg.norm(wperp)) * np.cross(wperp, qr) / np.linalg.norm(wperp), tol=1e-14)


def Omega_of(t):                              # a tumbling body rate used in the example and the figure
    return np.array([1.0 * np.cos(t), 1.0 * np.sin(0.7 * t), 0.5])


def integrate(step, hh, TT, midpoint=False):
    Rk = np.eye(3)
    n = round(TT / hh)
    hist = np.zeros(n + 1)
    for k in range(n):
        Ok = Omega_of((k + 0.5) * hh) if midpoint else Omega_of(k * hh)
        Rk = step(Rk, Ok, hh)
        hist[k + 1] = np.linalg.norm(Rk.T @ Rk - I3)
    return Rk, hist


def euler_step(Rk, Ok, hh):
    return Rk + hh * Rk @ hat(Ok)


def exp_step(Rk, Ok, hh):
    return Rk @ expm_so3(hh * Ok)


R_ref, _ = integrate(exp_step, 1e-5, 10.0, midpoint=True)
R_eu, hist_eu = integrate(euler_step, 0.01, 10.0)
R_ex1, hist_ex = integrate(exp_step, 0.01, 10.0)
ck.record("drift_euler_orthogonality_10s", float(hist_eu[-1]))
ck.record("drift_euler_det_10s", float(np.linalg.det(R_eu)))
ck.record("drift_exp_orthogonality_10s", float(hist_ex.max()))
ck.true("Euler, h = 0.01, 10 s: |R^T R - I| is about 0.16", abs(hist_eu[-1] - 0.16) < 0.005, f"{hist_eu[-1]:.4f}")
ck.true("Euler: det R has grown to about 1.13", abs(np.linalg.det(R_eu) - 1.13) < 0.005, f"{np.linalg.det(R_eu):.4f}")
ck.true("exponential step: |R^T R - I| stays below 1e-13", hist_ex.max() < 1e-13, f"{hist_ex.max():.1e}")
err_ex1 = np.degrees(angle_of(R_ref.T @ R_ex1))
ck.record("exp_step_attitude_error_deg_h0p01", float(err_ex1))
ck.true("exponential step with left-endpoint Omega is still off by about 0.29 deg", abs(err_ex1 - 0.29) < 0.005, f"{err_ex1:.3f} deg")

# Order of accuracy: left endpoint is first order, midpoint is second order.
errs_left, errs_mid = [], []
for hh in (0.02, 0.01, 0.005):
    errs_left.append(angle_of(R_ref.T @ integrate(exp_step, hh, 10.0)[0]))
    errs_mid.append(angle_of(R_ref.T @ integrate(exp_step, hh, 10.0, midpoint=True)[0]))
ck.true("left-endpoint exponential step: error halves with h (first order)",
        1.8 < errs_left[0] / errs_left[1] < 2.2 and 1.8 < errs_left[1] / errs_left[2] < 2.2,
        f"ratios {errs_left[0] / errs_left[1]:.2f}, {errs_left[1] / errs_left[2]:.2f}")
ck.true("midpoint exponential step: error quarters with h (second order)",
        3.6 < errs_mid[0] / errs_mid[1] < 4.4 and 3.6 < errs_mid[1] / errs_mid[2] < 4.4,
        f"ratios {errs_mid[0] / errs_mid[1]:.2f}, {errs_mid[1] / errs_mid[2]:.2f}")
err_mid = np.degrees(errs_mid[1])
ck.record("exp_midpoint_attitude_error_deg_h0p01", float(err_mid))
ck.true("midpoint version at h = 0.01 is below 0.002 deg", err_mid < 0.002, f"{err_mid:.5f} deg")
ck.close("constant Omega: one big exponential step is exact", R @ expm_so3(3.0 * Om1),
         np.linalg.multi_dot([R] + [expm_so3(0.01 * Om1)] * 300), tol=1e-12)

# Projection back to SO(3): the polar factor is the nearest rotation.
U, _, Vt = np.linalg.svd(R_eu)
R_proj = U @ Vt
ck.close("SVD projection of the drifted matrix is a rotation", R_proj.T @ R_proj, I3, tol=1e-13)
cands = [R_proj @ expm_so3(1e-3 * rng.normal(size=3)) for _ in range(200)]
ck.true("no nearby rotation is closer to the drifted matrix than its polar factor",
        all(np.linalg.norm(c - R_eu) >= np.linalg.norm(R_proj - R_eu) for c in cands))

ck.finish()
