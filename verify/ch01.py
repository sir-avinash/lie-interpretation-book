"""Chapter 1: every identity and every quoted number, checked numerically.

    python verify/ch01.py
"""
import numpy as np
from scipy.integrate import solve_ivp

from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee

ck = Checker("ch01")
rng = np.random.default_rng(1)


# ---- hat map and kinematics --------------------------------------------
x, y = rng.normal(size=3), rng.normal(size=3)
ck.close("hat(x) y = x cross y", hat(x) @ y, np.cross(x, y))
R = random_rotation(rng)
ck.close("R hat(x) R^T = hat(R x)", R @ hat(x) @ R.T, hat(R @ x))


# ---- Euler angles (ZYX) -------------------------------------------------
def R_zyx(psi, th, phi):
    return Rz(psi) @ Ry(th) @ Rx(phi)


def E(th, phi):
    return np.array([[-np.sin(th), 0, 1],
                     [np.cos(th) * np.sin(phi), np.cos(phi), 0],
                     [np.cos(th) * np.cos(phi), -np.sin(phi), 0]])


eta = np.array([0.4, -0.7, 1.1])
deta = np.array([0.3, 0.2, -0.5])
h = 1e-6
Rp, Rm = R_zyx(*(eta + h * deta)), R_zyx(*(eta - h * deta))
Omega_fd = vee(R_zyx(*eta).T @ (Rp - Rm) / (2 * h))
ck.close("Omega = E(theta, phi) eta_dot", Omega_fd, E(eta[1], eta[2]) @ deta, tol=1e-8)
ck.close("det E = -cos(theta)", np.linalg.det(E(eta[1], eta[2])), -np.cos(eta[1]))

# Worked example: calm body rate, pitch one degree short of vertical.
Om = np.array([0.0, 0.5, 0.5])
th, phi = np.radians(89.0), np.radians(45.0)
rates = np.linalg.solve(E(th, phi), Om)
ck.record("euler_body_speed", float(np.linalg.norm(Om)))
ck.record("euler_yaw_rate_89deg", float(rates[0]))
ck.record("euler_roll_rate_89deg", float(rates[2]))
ck.record("euler_yaw_rate_degps", float(np.degrees(rates[0])))
ck.close("yaw rate = (O2 sin(phi) + O3 cos(phi)) / cos(theta)", rates[0],
         (Om[1] * np.sin(phi) + Om[2] * np.cos(phi)) / np.cos(th))
ck.true("yaw rate at 89 deg is about 40.5 rad/s", abs(rates[0] - 40.5) < 0.1, f"{rates[0]:.2f}")

# Euler-angle difference is not invariant under a change of world frame.
def euler_of(Rm_):
    return np.array([np.arctan2(Rm_[1, 0], Rm_[0, 0]), -np.arcsin(Rm_[2, 0]), np.arctan2(Rm_[2, 1], Rm_[2, 2])])


Ra, Rb = R_zyx(0.2, 0.3, -0.4), R_zyx(-0.5, 0.6, 0.1)
ck.close("euler_of inverts R_zyx", euler_of(Ra), [0.2, 0.3, -0.4])
Q = Rx(0.9)
diff_before = euler_of(Ra) - euler_of(Rb)
diff_after = euler_of(Q @ Ra) - euler_of(Q @ Rb)
ck.true("Euler difference changes with the world frame", np.linalg.norm(diff_before - diff_after) > 0.1)
ck.close("Rd^T R does not change with the world frame", (Q @ Rb).T @ (Q @ Ra), Rb.T @ Ra)


# ---- quaternions ---------------------------------------------------------
def quat_to_R(q):
    q0, qv = q[0], q[1:]
    return np.eye(3) + 2 * q0 * hat(qv) + 2 * hat(qv) @ hat(qv)


def quat_mul(a, b):
    return np.concatenate([[a[0] * b[0] - a[1:] @ b[1:]], a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])])


q = rng.normal(size=4)
q /= np.linalg.norm(q)
p = rng.normal(size=4)
p /= np.linalg.norm(p)
ck.close("R(q) is orthogonal", quat_to_R(q).T @ quat_to_R(q), np.eye(3))
ck.close("R(q) = R(-q)", quat_to_R(q), quat_to_R(-q))
ck.close("R(q p) = R(q) R(p)", quat_to_R(quat_mul(q, p)), quat_to_R(q) @ quat_to_R(p))
u, ang = np.array([0.0, 0.0, 1.0]), 0.7
ck.close("q = (cos(a/2), sin(a/2) u) gives exp(a hat(u))",
         quat_to_R(np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * u])), expm_so3(ang * u))

# Unwinding: same physical start, two quaternion signs, PD law tau = -kp qv - kd Omega.
J = np.diag([0.05, 0.05, 0.10])
kp, kd = 2.0, 0.6


def closed_loop(t, s):
    qq, Om_ = s[:4], s[4:]
    tau = -kp * qq[1:] - kd * Om_
    dq = 0.5 * quat_mul(qq, np.concatenate([[0.0], Om_]))
    dOm = np.linalg.solve(J, tau - np.cross(Om_, J @ Om_))
    return np.concatenate([dq, dOm])


def run(q_init):
    sol = solve_ivp(closed_loop, [0, 12], np.concatenate([q_init, np.zeros(3)]),
                    rtol=1e-10, atol=1e-12, dense_output=True)
    t = np.linspace(0, 12, 4001)
    s = sol.sol(t)
    tau = -kp * s[1:4] - kd * s[4:]
    effort = np.trapezoid(np.sum(tau**2, axis=0), t)
    travelled = np.trapezoid(np.abs(s[6]), t)  # rotation is about e3 only
    return t, s, effort, travelled


a0 = np.radians(-10.0)
q_plus = np.array([np.cos(a0 / 2), 0, 0, np.sin(a0 / 2)])
t, s_plus, effort_plus, turn_plus = run(q_plus)
_, s_minus, effort_minus, turn_minus = run(-q_plus)
ck.close("both starts are the same rotation", quat_to_R(q_plus), quat_to_R(-q_plus))
ck.true("short start turns about 10 deg", abs(np.degrees(turn_plus) - 10) < 1.5, f"{np.degrees(turn_plus):.1f} deg")
ck.true("flipped start turns about 350 deg", abs(np.degrees(turn_minus) - 350) < 25, f"{np.degrees(turn_minus):.1f} deg")
ck.close("both runs end at the identity rotation", quat_to_R(s_minus[:4, -1]), np.eye(3), tol=1e-3)
ck.record("unwind_turn_short_deg", float(np.degrees(turn_plus)))
ck.record("unwind_turn_long_deg", float(np.degrees(turn_minus)))
ck.record("unwind_effort_ratio", float(effort_minus / effort_plus))
ck.record("unwind_gains", {"kp": kp, "kd": kd, "J": [0.05, 0.05, 0.10]})


# ---- error functions on SO(3) --------------------------------------------
def Psi_R(R_, Rd_):
    return 0.5 * np.trace(np.eye(3) - Rd_.T @ R_)


def e_R(R_, Rd_):
    return 0.5 * vee(Rd_.T @ R_ - R_.T @ Rd_)


R, Rd, Q = random_rotation(rng), random_rotation(rng), random_rotation(rng)
Om, Omd = rng.normal(size=3), rng.normal(size=3)
ck.close("Psi_R is unchanged by a common left rotation", Psi_R(Q @ R, Q @ Rd), Psi_R(R, Rd))
ck.close("e_R is unchanged by a common left rotation", e_R(Q @ R, Q @ Rd), e_R(R, Rd))
angle = np.arccos((np.trace(Rd.T @ R) - 1) / 2)
ck.close("Psi_R = 1 - cos(angle)", Psi_R(R, Rd), 1 - np.cos(angle))
ck.close("|e_R| = sin(angle)", np.linalg.norm(e_R(R, Rd)), np.sin(angle))
ck.close("|e_R|^2 = Psi (2 - Psi)", e_R(R, Rd) @ e_R(R, Rd), Psi_R(R, Rd) * (2 - Psi_R(R, Rd)))
h = 1e-6
dPsi = (Psi_R(R @ expm_so3(h * Om), Rd @ expm_so3(h * Omd)) - Psi_R(R @ expm_so3(-h * Om), Rd @ expm_so3(-h * Omd))) / (2 * h)
e_Om = Om - R.T @ Rd @ Omd
ck.close("d/dt Psi_R = e_R . e_Omega", dPsi, e_R(R, Rd) @ e_Om, tol=1e-8)
eta_ = rng.normal(size=3)
dPsi_var = (Psi_R(R @ expm_so3(h * eta_), Rd) - Psi_R(R @ expm_so3(-h * eta_), Rd)) / (2 * h)
ck.close("variation of Psi_R along R hat(eta) is e_R . eta", dPsi_var, e_R(R, Rd) @ eta_, tol=1e-8)

Rlock = Ry(np.pi / 2)
ck.close("at pitch 90 deg: Psi_R = 1", Psi_R(Rlock, np.eye(3)), 1.0)
ck.close("at pitch 90 deg: e_R = e2", e_R(Rlock, np.eye(3)), [0, 1, 0])
Rflip = np.diag([1.0, -1.0, -1.0])
ck.close("at a half turn: Psi_R = 2", Psi_R(Rflip, np.eye(3)), 2.0)
ck.close("at a half turn: e_R = 0", e_R(Rflip, np.eye(3)), np.zeros(3))


# ---- error functions on S^2 ----------------------------------------------
q = rng.normal(size=3)
q /= np.linalg.norm(q)
qd = rng.normal(size=3)
qd /= np.linalg.norm(qd)
om = np.cross(q, rng.normal(size=3))  # any omega with omega . q = 0
e_q = np.cross(qd, q)
ck.close("e_q is tangent at q", e_q @ q, 0.0)
qp, qm = expm_so3(h * om) @ q, expm_so3(-h * om) @ q
dPsi_q = ((1 - qp @ qd) - (1 - qm @ qd)) / (2 * h)
ck.close("d/dt Psi_q = e_q . omega (q_d fixed)", dPsi_q, e_q @ om, tol=1e-8)
ang_q = np.arccos(q @ qd)
ck.close("|e_q| = sin(angle)", np.linalg.norm(e_q), np.sin(ang_q))

# Kinematic law omega = -k e_q on S^2: the antipode is an unstable equilibrium.
k = 1.0
qd = np.array([0.0, 0.0, 1.0])
q0 = np.array([np.sin(3.1), 0.0, np.cos(3.1)])  # 177.6 degrees from q_d
sol = solve_ivp(lambda t, q_: np.cross(-k * np.cross(qd, q_), q_), [0, 12], q0, rtol=1e-10, atol=1e-12)
ck.true("from near the antipode the flow still reaches q_d", sol.y[:, -1] @ qd > 0.999)
# In terms of the angle a from q_d the law reads a_dot = -k sin(a); slope at pi is +k.
a = 3.1
adot = (np.arccos(np.clip((q0 + 1e-6 * np.cross(-k * np.cross(qd, q0), q0)) @ qd, -1, 1)) - a) / 1e-6
ck.close("angle obeys a_dot = -k sin(a)", adot, -k * np.sin(a), tol=1e-4)

ck.finish()
