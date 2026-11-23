"""Chapter 13: every identity, linearization, QP solution and quoted number, checked.

    python verify/ch13.py

The functions above ``main`` are the chapter's solvers and controllers. The figure
script imports them, so the figures show exactly what is verified here.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # small dense problems: threads only get in the way

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee  # noqa: E402

_spec = importlib.util.spec_from_file_location("ch07_verify", Path(__file__).resolve().parent / "ch07.py")
V7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V7)

E1, E2, E3 = np.eye(3)
I3 = np.eye(3)
ZERO3 = np.zeros(3)
J_B = V7.J_B                                   # diag(0.03, 0.05, 0.07) kg m^2, as in Chapter 7
J_INV = np.linalg.inv(J_B)
log_so3 = V7.log_so3


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# =====================================================================
# The derivative of the exponential on SO(3)
#     exp((x + d)^) = exp(x^) exp((dexp(x) d)^) + O(|d|^2)
# =====================================================================
def dexp(x):
    th = np.linalg.norm(x)
    X = hat(x)
    if th < 1e-6:
        return I3 - 0.5 * X + X @ X / 6
    return I3 - (1 - np.cos(th)) / th**2 * X + (th - np.sin(th)) / th**3 * X @ X


def dexp_inv(x):
    th = np.linalg.norm(x)
    X = hat(x)
    if th < 1e-6:
        return I3 + 0.5 * X + X @ X / 12
    return I3 + 0.5 * X + (1 / th**2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * X @ X


# =====================================================================
# Section 2. Newton's method on the sphere and on SO(3): the mean
# =====================================================================
def s2_log(q, p):
    """Tangent vector at q that points to p, with length the angle between them."""
    c = np.clip(q @ p, -1, 1)
    th = np.arccos(c)
    v = p - c * q
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-14 else th * v / n


def s2_mean_cost(q, pts):
    return 0.5 * sum(np.arccos(np.clip(q @ p, -1, 1)) ** 2 for p in pts)


def s2_mean_grad_hess(q, pts):
    """Gradient and Hessian of the cost along q -> exp(xi^) q, with xi in the tangent plane."""
    g, H = np.zeros(3), np.zeros((3, 3))
    for p in pts:
        v = s2_log(q, p)
        th = np.linalg.norm(v)
        g -= np.cross(q, v)
        if th < 1e-9:
            H += I3 - np.outer(q, q)
            continue
        t = v / th                                 # along the great circle to p
        nrm = np.cross(q, t)                       # turning about nrm moves q along it
        H += np.outer(nrm, nrm) + th / np.tan(th) * np.outer(t, t)
    return g, H


def s2_mean_newton(q0, pts, iters=6):
    """Newton iterates q <- exp(xi^) q. Returns the iterates and the steps."""
    q, qs, steps = np.array(q0, float), [np.array(q0, float)], []
    for _ in range(iters):
        g, H = s2_mean_grad_hess(q, pts)
        xi = -np.linalg.solve(H + np.outer(q, q), g)   # the q q^T block fixes the unused direction
        xi -= (xi @ q) * q
        q = expm_so3(xi) @ q
        qs.append(q)
        steps.append(xi)
    return np.array(qs), np.array(steps)


def so3_mean_cost(R, Rs):
    return 0.5 * sum(np.linalg.norm(log_so3(Ri.T @ R)) ** 2 for Ri in Rs)


def so3_mean_grad(R, Rs):
    """Gradient along R -> R exp(eta^): the sum of the logarithm errors."""
    return sum(log_so3(Ri.T @ R) for Ri in Rs)


def so3_mean_hess(R, Rs):
    H = np.zeros((3, 3))
    for Ri in Rs:
        chi = log_so3(Ri.T @ R)
        th = np.linalg.norm(chi)
        if th < 1e-9:
            H += I3
            continue
        a, k = chi / th, 0.5 * th / np.tan(0.5 * th)
        H += k * I3 + (1 - k) * np.outer(a, a)
    return H


def project_so3(M):
    """Nearest rotation in the Frobenius norm."""
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def so3_mean_iterate(R0, Rs, method="newton", iters=8):
    """Iterates of four ways to use the same tangent-space information.

    newton    : eta = -H^-1 g,  R <- R exp(eta^)          (exponential retraction)
    project   : eta = -H^-1 g,  R <- project(R + R eta^)  (projection retraction)
    gradient  : eta = -g / n,   R <- R exp(eta^)
    additive  : eta = -H^-1 g,  R <- R + R eta^           (leaves the group)
    """
    R, out = np.array(R0, float), [np.array(R0, float)]
    for _ in range(iters):
        Rp = project_so3(R) if method == "additive" else R   # the cost needs a rotation
        g = so3_mean_grad(Rp, Rs)
        eta = -g / len(Rs) if method == "gradient" else -np.linalg.solve(so3_mean_hess(Rp, Rs), g)
        if method in ("newton", "gradient"):
            R = R @ expm_so3(eta)
        elif method == "project":
            R = project_so3(R + R @ hat(eta))
        else:
            R = R + R @ hat(eta)
        out.append(R)
    return out


def mean_data():
    """Five attitude readings scattered about one attitude, and the same for four directions."""
    rng = np.random.default_rng(13)
    Rc = Rz(0.4) @ Ry(1.1) @ Rx(-0.3)
    Rs = [Rc @ expm_so3(np.radians(40) * unit(rng.normal(size=3)) * rng.uniform(0.5, 1.0)) for _ in range(5)]
    pts = [unit(v) for v in ([0.95, 0.05, 0.31], [0.25, 0.80, 0.55], [0.35, -0.45, 0.82], [0.70, 0.45, 0.55])]
    return Rs, pts


# =====================================================================
# Section 3. Error-state model of the rigid body:  s = (chi, dOmega)
#     chi' = -Omega_d x chi + dOmega
#     J dOmega' = ((J Omega_d)^ - Omega_d^ J) dOmega + dtau
# =====================================================================
def err_A(Od, J=J_B):
    A = np.zeros((6, 6))
    A[:3, :3] = -hat(Od)
    A[:3, 3:] = I3
    A[3:, 3:] = np.linalg.solve(J, hat(J @ Od) - hat(Od) @ J)
    return A


ERR_B = np.vstack([np.zeros((3, 3)), J_INV])


def zoh(A, B, h):
    """Exact sampling of s' = A s + B u with u held over a step of length h."""
    n, m = B.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = A, B
    E = expm(blk * h)
    return E[:n, :n], E[:n, n:]


def chi_dot_exact(chi, Om, Od):
    """Exact rate of the logarithm of the left-invariant error on SO(3)."""
    return dexp_inv(chi) @ (Om - expm_so3(-chi) @ Od)


# ---- SE(3), twists ordered (V, Omega) ---------------------------------
def ad_se3(zeta):
    V, Om = zeta[:3], zeta[3:]
    out = np.zeros((6, 6))
    out[:3, :3], out[:3, 3:], out[3:, 3:] = hat(Om), hat(V), hat(Om)
    return out


def hat_se3(zeta):
    out = np.zeros((4, 4))
    out[:3, :3], out[:3, 3] = hat(zeta[3:]), zeta[:3]
    return out


def vee_se3(M):
    return np.r_[M[:3, 3], vee(M[:3, :3])]


def exp_se3(zeta):
    return expm(hat_se3(zeta))


def log_se3(X):
    """Logarithm of a rigid motion, as a twist (V, Omega)."""
    phi = log_so3(X[:3, :3])
    th = np.linalg.norm(phi)
    P = hat(phi)
    Vinv = I3 - 0.5 * P + (P @ P / 12 if th < 1e-6
                           else (1 / th**2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * P @ P)
    return np.r_[Vinv @ X[:3, 3], phi]


M_SE3 = 1.2                                              # kg


def se3_mass(m=M_SE3, J=J_B):
    M = np.zeros((6, 6))
    M[:3, :3], M[3:, 3:] = m * I3, J
    return M


def se3_acc(zeta, u, m=M_SE3, J=J_B):
    """Euler-Poincare equations  M zeta' = ad_zeta^T M zeta + u,  with u = (force, torque) in the body frame."""
    M = se3_mass(m, J)
    return np.linalg.solve(M, ad_se3(zeta).T @ M @ zeta + u)


def se3_err_A(zd, m=M_SE3, J=J_B):
    """A(t) for s = (chi, dzeta) on SE(3). It contains the reference twist and nothing else."""
    Vd, Od = zd[:3], zd[3:]
    Hd = np.zeros((6, 6))
    Hd[:3, :3], Hd[:3, 3:] = -m * hat(Od), m * hat(Vd)
    Hd[3:, 3:] = hat(J @ Od) - hat(Od) @ J
    A = np.zeros((12, 12))
    A[:6, :6], A[:6, 6:] = -ad_se3(zd), np.eye(6)
    A[6:, 6:] = np.linalg.solve(se3_mass(m, J), Hd)
    return A


def se3_err_B(m=M_SE3, J=J_B):
    return np.vstack([np.zeros((6, 6)), np.linalg.inv(se3_mass(m, J))])


def se3_sim(X0, z0, ufun, T, rtol=1e-11):
    def rhs(t, y):
        X, z = y[:16].reshape(4, 4), y[16:]
        return np.r_[(X @ hat_se3(z)).ravel(), se3_acc(z, ufun(t))]

    return solve_ivp(rhs, [0, T], np.r_[X0.ravel(), z0], rtol=rtol, atol=rtol * 1e-2, method="DOP853",
                     dense_output=True)


# =====================================================================
# Section 4. The finite-horizon problem as a QP with box constraints
# =====================================================================
H_MPC = 0.02                                             # s, sampling period
N_MPC = 25                                               # horizon length
Q_MPC = np.diag([100.0] * 3 + [1.0] * 3)                 # the weights of Chapter 7
RU_MPC = 20.0 * np.eye(3)


def terminal_weight(h=H_MPC, Q=Q_MPC, Ru=RU_MPC):
    """Riccati solution and gain of the sampled regulator for a reference at rest."""
    A, B = zoh(err_A(ZERO3), ERR_B, h)
    P = solve_discrete_are(A, B, h * Q, h * Ru)
    K = np.linalg.solve(h * Ru + B.T @ P @ B, B.T @ P @ A)
    return P, K


def condense(As, Bs, Q, Ru, Pf):
    """Stack s_1..s_N = Sx s_0 + Su U and return the QP data  1/2 U^T H U + (F s_0) . U.

    Stage cost 1/2 s_k^T Q s_k for k = 1..N-1, terminal 1/2 s_N^T Pf s_N, input 1/2 du_k^T Ru du_k.
    """
    N, (n, m) = len(As), Bs[0].shape
    Sx, Su = np.zeros((N * n, n)), np.zeros((N * n, N * m))
    Phi = np.eye(n)
    for k in range(N):
        if k > 0:
            Su[k * n:(k + 1) * n] = As[k] @ Su[(k - 1) * n:k * n]
        Su[k * n:(k + 1) * n, k * m:(k + 1) * m] = Bs[k]
        Phi = As[k] @ Phi
        Sx[k * n:(k + 1) * n] = Phi
    Qbar = np.kron(np.eye(N), Q)
    Qbar[-n:, -n:] = Pf
    H = Su.T @ Qbar @ Su + np.kron(np.eye(N), Ru)
    F = Su.T @ Qbar @ Sx
    return 0.5 * (H + H.T), F, Sx, Su


def box_qp(H, p, lo, hi, U0=None, tol=1e-10, max_iter=200):
    """Minimize 1/2 U^T H U + p . U subject to lo <= U <= hi by projected Newton.

    Inputs that sit on a bound and want to push through it are held. A Newton step is
    taken in the others, then clipped to the bounds, with a backtracking line search.
    Returns U and the number of iterations.
    """
    U = np.clip(np.zeros_like(p) if U0 is None else U0, lo, hi)
    for it in range(max_iter):
        g = H @ U + p
        act = ((U <= lo + 1e-12) & (g > 0)) | ((U >= hi - 1e-12) & (g < 0))
        res = np.where(act, 0.0, g)
        inner = ~act & (U > lo + 1e-12) & (U < hi - 1e-12)
        if np.max(np.abs(np.where(inner, g, 0.0)), initial=0.0) <= tol and \
                np.max(np.abs(np.clip(U - res, lo, hi) - U)) <= tol:
            return U, it
        d = np.zeros_like(U)
        free = ~act
        d[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
        f0, a = 0.5 * U @ H @ U + p @ U, 1.0
        while True:
            Un = np.clip(U + a * d, lo, hi)
            if 0.5 * Un @ H @ Un + p @ Un <= f0 + 1e-4 * g @ (Un - U) or a < 1e-10:
                break
            a *= 0.5
        U = Un
    return U, max_iter


def box_qp_kkt(H, p, lo, hi, U):
    """Largest violation of the optimality conditions of the box-constrained QP."""
    g = H @ U + p
    at_lo, at_hi = U <= lo + 1e-9, U >= hi - 1e-9
    stat = np.where(at_lo, np.minimum(g, 0), np.where(at_hi, np.maximum(g, 0), g))
    feas = max(np.max(lo - U), np.max(U - hi), 0.0)
    return float(max(np.max(np.abs(stat)), feas))


def box_qp_reference(H, p, lo, hi):
    """The same QP solved by SciPy's bounded-variable least squares (an active-set method)."""
    L = np.linalg.cholesky(H)
    sol = lsq_linear(L.T, -np.linalg.solve(L, p), bounds=(lo, hi), method="bvls", tol=1e-14, max_iter=2000)
    return sol.x


# =====================================================================
# Section 4. Closed loop: error-state MPC, saturated LQR, Euler-angle MPC
# =====================================================================
TAU_MAX = 1.0                                            # N m, per axis


class ErrorStateMPC:
    """Error-state MPC along a reference. The QP matrices are tabulated before the flight."""

    def __init__(self, ref, T_end, h=H_MPC, N=N_MPC, Q=Q_MPC, Ru=RU_MPC, tau_max=TAU_MAX, bounded=True):
        self.ref, self.h, self.N, self.tau_max, self.bounded = ref, h, N, tau_max, bounded
        self.Pf, self.K_rest = terminal_weight(h, Q, Ru)
        self.Qh, self.Rh = h * Q, h * Ru
        n_steps = int(round(T_end / h))
        self.AB = [zoh(err_A(ref(k * h + 0.5 * h)[1]), ERR_B, h) for k in range(n_steps + N)]
        self.tau_d = np.array([ref(k * h + 0.5 * h)[2] for k in range(n_steps + N)])
        self.qp = {}
        self.U_prev = None
        self.iters, self.times = [], []

    def data(self, k):
        if k not in self.qp:
            As = [self.AB[k + i][0] for i in range(self.N)]
            Bs = [self.AB[k + i][1] for i in range(self.N)]
            self.qp[k] = condense(As, Bs, self.Qh, self.Rh, self.Pf)[:2]
        return self.qp[k]

    def error_state(self, k, R, Om):
        Rd, Od, _ = self.ref(k * self.h)
        return np.r_[log_so3(Rd.T @ R), Om - Od]

    def __call__(self, k, R, Om):
        H, F = self.data(k)
        s0 = self.error_state(k, R, Om)
        ff = self.tau_d[k:k + self.N].ravel()
        t0 = time.perf_counter()
        if self.bounded:
            warm = None if self.U_prev is None else np.r_[self.U_prev[3:], np.zeros(3)]
            U, it = box_qp(H, F @ s0, -self.tau_max - ff, self.tau_max - ff, warm)
        else:
            U, it = -np.linalg.solve(H, F @ s0), 1
        self.times.append(time.perf_counter() - t0)
        self.iters.append(it)
        self.U_prev = U
        return ff[:3] + U[:3]


class SaturatedLQR:
    """The regulator of Chapter 7 in sampled form, with the torque clipped afterwards."""

    def __init__(self, ref, T_end, h=H_MPC, Q=Q_MPC, Ru=RU_MPC, tau_max=TAU_MAX):
        self.ref, self.h, self.tau_max = ref, h, tau_max
        P, _ = terminal_weight(h, Q, Ru)
        n_steps = int(round(T_end / h)) + N_MPC
        self.K = [None] * n_steps
        for k in reversed(range(n_steps)):
            A, B = zoh(err_A(ref(k * h + 0.5 * h)[1]), ERR_B, h)
            self.K[k] = np.linalg.solve(h * Ru + B.T @ P @ B, B.T @ P @ A)
            P = h * Q + A.T @ P @ (A - B @ self.K[k])
        self.P0 = P

    def __call__(self, k, R, Om):
        Rd, Od, _ = self.ref(k * self.h)
        tau_d = self.ref(k * self.h + 0.5 * self.h)[2]
        tau = tau_d - self.K[k] @ np.r_[log_so3(Rd.T @ R), Om - Od]
        return np.clip(tau, -self.tau_max, self.tau_max)


def euler_ref(ref, ts):
    eul = np.array([V7.euler_of(ref(t)[0]) for t in ts])
    eul[:, 0], eul[:, 2] = np.unwrap(eul[:, 0]), np.unwrap(eul[:, 2])
    return eul


class EulerMPC:
    """The same QP, written in yaw, pitch and roll. Its matrices depend on the reference attitude."""

    def __init__(self, ref, T_end, h=H_MPC, N=N_MPC, Q=Q_MPC, Ru=RU_MPC, tau_max=TAU_MAX):
        self.ref, self.h, self.N, self.tau_max = ref, h, N, tau_max
        self.Pf, _ = terminal_weight(h, Q, Ru)
        self.Qh, self.Rh = h * Q, h * Ru
        n_steps = int(round(T_end / h))
        tm = (np.arange(n_steps + N) + 0.5) * h
        eul_m = euler_ref(ref, np.r_[0.0, tm])[1:]
        self.eul_k = euler_ref(ref, np.arange(n_steps + N + 1) * h)
        self.A_cont = [V7.euler_A(np.r_[e, ref(t)[1]], ref(t)[2]) for e, t in zip(eul_m, tm)]
        self.AB = [zoh(A, ERR_B, h) for A in self.A_cont]
        self.tau_d = np.array([ref(t)[2] for t in tm])
        self.U_prev = None
        self.iters, self.times = [], []

    def __call__(self, k, R, Om):
        As = [self.AB[k + i][0] for i in range(self.N)]
        Bs = [self.AB[k + i][1] for i in range(self.N)]
        H, F = condense(As, Bs, self.Qh, self.Rh, self.Pf)[:2]
        s0 = np.r_[V7.wrap(V7.euler_of(R) - self.eul_k[k]), Om - self.ref(k * self.h)[1]]
        ff = self.tau_d[k:k + self.N].ravel()
        t0 = time.perf_counter()
        warm = None if self.U_prev is None else np.r_[self.U_prev[3:], np.zeros(3)]
        U, it = box_qp(H, F @ s0, -self.tau_max - ff, self.tau_max - ff, warm)
        self.times.append(time.perf_counter() - t0)
        self.iters.append(it)
        self.U_prev = U
        return ff[:3] + U[:3]


class Turn:
    """Rest-to-rest turn by `angle` about a fixed body axis in time T, then hold.

    Turn(2 pi, axis tilted 5 deg from e2, 1.5) is the flip of Chapter 7.
    """

    def __init__(self, angle, axis, T):
        self.total, self.n, self.T = angle, unit(axis), T

    def __call__(self, t):
        s = np.clip(t / self.T, 0.0, 1.0)
        a = self.total * (10 * s**3 - 15 * s**4 + 6 * s**5)
        da = self.total * (30 * s**2 - 60 * s**3 + 30 * s**4) / self.T
        dda = self.total * (60 * s - 180 * s**2 + 120 * s**3) / self.T**2
        Od = da * self.n
        return expm_so3(a * self.n), Od, J_B @ (dda * self.n) + np.cross(Od, J_B @ Od)


def flip_ref():
    d = np.radians(5.0)
    return Turn(2 * np.pi, [np.sin(d), np.cos(d), 0.0], 1.5)


def pitch_up_ref(pitch_deg=88.0):
    return Turn(np.radians(pitch_deg), E2, 1.0)


def plant_step(R, Om, tau, h, J=J_B):
    """The nonlinear rigid body over one sampling period with the torque held."""
    def rhs(t, y):
        Rm, w = y[:9].reshape(3, 3), y[9:]
        return np.r_[(Rm @ hat(w)).ravel(), np.linalg.solve(J, np.cross(J @ w, w) + tau)]

    y = solve_ivp(rhs, [0, h], np.r_[R.ravel(), Om], rtol=1e-10, atol=1e-12, method="DOP853").y[:, -1]
    return project_so3(y[:9].reshape(3, 3)), y[9:]


def closed_loop(ctrl, ref, R0, Om0, T_end, h=H_MPC):
    """Run a sampled controller on the nonlinear plant. Returns times, attitudes, torques, error angles."""
    n = int(round(T_end / h))
    R, Om = np.array(R0, float), np.array(Om0, float)
    Rs, Oms, taus, ang = [R], [Om], [], [np.degrees(np.linalg.norm(log_so3(ref(0.0)[0].T @ R)))]
    for k in range(n):
        tau = ctrl(k, R, Om)
        R, Om = plant_step(R, Om, tau, h)
        Rs.append(R)
        Oms.append(Om)
        taus.append(tau)
        ang.append(np.degrees(np.linalg.norm(log_so3(ref((k + 1) * h)[0].T @ R))))
    return dict(t=np.arange(n + 1) * h, R=np.array(Rs), Om=np.array(Oms), tau=np.array(taus), angle=np.array(ang))


# =====================================================================
# Section 6. Direct transcription on SO(3) and an on-manifold interior point
#
#   variables   R_1..R_{N-1} in SO(3),  Omega_1..Omega_{N-1},  tau_0..tau_{N-1}
#   cost        Phi = (h/2) sum |tau_k|^2
#   dynamics    r^R_k = log(R_{k+1}^T R_k exp(h Omega_k^))^vee = 0
#               r^W_k = J (Omega_{k+1} - Omega_k) - h (J Omega_k x Omega_k + tau_k) = 0
#   bounds      |tau_k,i| <= tau_max
#   keep-out    q_c . R_k e_1 - cos(theta_c) <= 0
# =====================================================================
class Reorient:
    """Minimum-effort rest-to-rest reorientation with torque bounds and a keep-out cone."""

    def __init__(self, N=20, h=0.1, R0=I3, RN=None, q_c=None, cone_deg=25.0, tau_max=0.12, J=J_B):
        self.N, self.h, self.J, self.tau_max = N, h, J, tau_max
        self.R0 = np.array(R0, float)
        self.RN = Rz(np.radians(120.0)) if RN is None else np.array(RN, float)
        self.q_c = unit([np.cos(np.radians(60.0)), np.sin(np.radians(60.0)), 0.10] if q_c is None else q_c)
        self.cos_c = np.cos(np.radians(cone_deg))
        self.cone_deg = cone_deg
        self.n = 9 * N - 6                               # tangent dimension
        self.n_eq = 6 * N
        self.n_in = 6 * N + (N - 1)

    # ---- index maps into the tangent vector -------------------------------
    def i_tau(self, k):
        return slice(9 * k, 9 * k + 3)

    def i_eta(self, k):                                  # k = 1..N-1
        return slice(9 * (k - 1) + 3, 9 * (k - 1) + 6)

    def i_dOm(self, k):                                  # k = 1..N-1
        return slice(9 * (k - 1) + 6, 9 * (k - 1) + 9)

    # ---- points -----------------------------------------------------------
    def initial_guess(self, bulge=0.0):
        """Attitudes on the geodesic from R0 to RN, optionally pushed sideways, with zero torque."""
        N, h = self.N, self.h
        phi = log_so3(self.R0.T @ self.RN)
        R = np.zeros((N + 1, 3, 3))
        for k in range(N + 1):
            sk = k / N
            side = bulge * np.sin(np.pi * sk) * unit(np.cross(phi, E1))
            R[k] = self.R0 @ expm_so3(sk * phi) @ expm_so3(side)
        R[N] = self.RN
        Om = np.zeros((N + 1, 3))
        for k in range(1, N):
            Om[k] = log_so3(R[k].T @ R[k + 1]) / h
        return dict(R=R, Om=Om, tau=np.zeros((N, 3)))

    def retract(self, x, d, alpha=1.0):
        """Move along the tangent vector d: attitudes by the exponential, the rest by addition."""
        R, Om, tau = x["R"].copy(), x["Om"].copy(), x["tau"].copy()
        for k in range(self.N):
            tau[k] += alpha * d[self.i_tau(k)]
        for k in range(1, self.N):
            R[k] = R[k] @ expm_so3(alpha * d[self.i_eta(k)])
            Om[k] += alpha * d[self.i_dOm(k)]
        return dict(R=R, Om=Om, tau=tau)

    # ---- functions --------------------------------------------------------
    def cost(self, x):
        return 0.5 * self.h * float(np.sum(x["tau"] ** 2))

    def cost_grad(self, x):
        g = np.zeros(self.n)
        for k in range(self.N):
            g[self.i_tau(k)] = self.h * x["tau"][k]
        return g

    def eq(self, x):
        N, h, J = self.N, self.h, self.J
        R, Om, tau = x["R"], x["Om"], x["tau"]
        r = np.zeros(6 * N)
        for k in range(N):
            r[6 * k:6 * k + 3] = log_so3(R[k + 1].T @ R[k] @ expm_so3(h * Om[k]))
            r[6 * k + 3:6 * k + 6] = J @ (Om[k + 1] - Om[k]) - h * (np.cross(J @ Om[k], Om[k]) + tau[k])
        return r

    def eq_jac(self, x):
        """Jacobian of the residuals with respect to the tangent vector."""
        N, h, J = self.N, self.h, self.J
        R, Om = x["R"], x["Om"]
        A = np.zeros((6 * N, self.n))
        for k in range(N):
            F = expm_so3(h * Om[k])
            rho = log_so3(R[k + 1].T @ R[k] @ F)
            Di = dexp_inv(rho)
            rows, rows_w = slice(6 * k, 6 * k + 3), slice(6 * k + 3, 6 * k + 6)
            if k >= 1:
                A[rows, self.i_eta(k)] = Di @ F.T
                A[rows, self.i_dOm(k)] = h * Di @ dexp(h * Om[k])
                A[rows_w, self.i_dOm(k)] = -J - h * (hat(J @ Om[k]) - hat(Om[k]) @ J)
            if k + 1 <= N - 1:
                A[rows, self.i_eta(k + 1)] = -dexp_inv(-rho)
                A[rows_w, self.i_dOm(k + 1)] = J
            A[rows_w, self.i_tau(k)] = -h * I3
        return A

    def ineq(self, x):
        N = self.N
        c = np.zeros(self.n_in)
        c[:3 * N] = x["tau"].ravel() - self.tau_max
        c[3 * N:6 * N] = -x["tau"].ravel() - self.tau_max
        for k in range(1, N):
            c[6 * N + k - 1] = self.q_c @ x["R"][k] @ E1 - self.cos_c
        return c

    def ineq_jac(self, x):
        N = self.N
        A = np.zeros((self.n_in, self.n))
        for k in range(N):
            for i in range(3):
                A[3 * k + i, 9 * k + i] = 1.0
                A[3 * N + 3 * k + i, 9 * k + i] = -1.0
        for k in range(1, N):
            A[6 * N + k - 1, self.i_eta(k)] = np.cross(E1, x["R"][k].T @ self.q_c)
        return A

    def lag_grad(self, x, y, z):
        return self.cost_grad(x) + self.eq_jac(x).T @ y + self.ineq_jac(x).T @ z

    def lag_hess(self, x, y, z, eps=1e-5):
        """Hessian of the Lagrangian along the retraction: the symmetrized difference of its gradient.

        The Lagrangian couples only neighboring stages, so three colors of stages suffice.
        """
        n, H = self.n, np.zeros((self.n, self.n))
        nb = self.N                                      # blocks of 9 (the last has 3)
        for color in range(3):
            for comp in range(9):
                cols = [9 * j + comp for j in range(color, nb, 3) if 9 * j + comp < n]
                if not cols:
                    continue
                d = np.zeros(n)
                d[cols] = eps
                gp = self.lag_grad(self.retract(x, d), y, z)
                gm = self.lag_grad(self.retract(x, -d), y, z)
                dg = (gp - gm) / (2 * eps)
                for col in cols:
                    j = col // 9
                    lo, hi = max(9 * (j - 1), 0), min(9 * (j + 2), n)
                    H[lo:hi, col] = dg[lo:hi]
        return 0.5 * (H + H.T)

    def boresight(self, x):
        return x["R"][:, :, 0]


def kkt_error(prob, x, y, z, w, mu):
    """The four blocks of the primal-dual residual and its largest entry."""
    gL = prob.lag_grad(x, y, z)
    r, c = prob.eq(x), prob.ineq(x)
    blocks = (gL, r, c + w, w * z - mu)
    return blocks, max(float(np.max(np.abs(b))) for b in blocks)


def interior_point(prob, x0, mu0=0.1, tol=1e-8, max_iter=300, fixed_mu=None, state=None, log=None, split=True):
    """Primal-dual interior point with on-manifold primal steps.

    Unknowns: x (attitudes, velocities, torques), multipliers y of the dynamics, multipliers z >= 0
    and slacks w > 0 of the inequalities c(x) + w = 0. Each iteration solves the Newton system of
        grad Phi + A_r^T y + A_c^T z = 0,  r = 0,  c + w = 0,  w z = mu,
    shortens the step so that w and z stay positive, backtracks on the norm of that residual,
    retracts x and adds the rest. The barrier parameter mu is reduced when the residual is below
    10 mu. With ``fixed_mu`` the subproblem for that mu alone is solved (a point of the central path).
    """
    x = x0
    mu = mu0 if fixed_mu is None else fixed_mu
    if state is None:
        y = np.zeros(prob.n_eq)
        w = np.maximum(-prob.ineq(x), 0.05)
        z = mu / w
    else:
        y, z, w = (np.array(v, float) for v in state)
    hist = []
    for it in range(max_iter):
        blocks, E_mu = kkt_error(prob, x, y, z, w, mu)
        E_0 = max(float(np.max(np.abs(b))) for b in (blocks[0], blocks[1], blocks[2], w * z))
        hist.append(dict(E0=E_0, Emu=E_mu, mu=mu, cost=prob.cost(x), alpha=np.nan))
        if log is not None:
            log(it, x, mu)
        if fixed_mu is None:
            if E_0 <= tol:
                break
            if E_mu <= 10 * mu and mu > tol / 10:
                mu = max(tol / 10, min(0.2 * mu, mu**1.5))
                blocks, E_mu = kkt_error(prob, x, y, z, w, mu)
        elif E_mu <= tol:
            break
        gL, r, cw, comp = blocks
        Ar, Ac = prob.eq_jac(x), prob.ineq_jac(x)
        H = prob.lag_hess(x, y, z)
        Sig = z / w
        K = np.zeros((prob.n + prob.n_eq, prob.n + prob.n_eq))
        Hbar = H + Ac.T @ (Sig[:, None] * Ac)
        K[:prob.n, prob.n:], K[prob.n:, :prob.n] = Ar.T, Ar
        rhs = -np.r_[gL + Ac.T @ (Sig * cw - comp / w), r]
        delta = 0.0
        while True:                                      # inertia correction: n positive, n_eq negative
            K[:prob.n, :prob.n] = Hbar + delta * np.eye(prob.n)
            ev = np.linalg.eigvalsh(K)
            if np.sum(ev > 0) == prob.n and np.sum(ev < 0) == prob.n_eq:
                break
            delta = 1e-4 if delta == 0.0 else 10 * delta
        sol = np.linalg.solve(K, rhs)
        dx, dy = sol[:prob.n], sol[prob.n:]
        dw = -cw - Ac @ dx
        dz = -comp / w - Sig * dw
        frac = max(0.99, 1 - mu)
        a, az = 1.0, 1.0                                 # primal step for (x, w, y), dual step for z
        if np.any(dw < 0):
            a = min(1.0, float(np.min(-frac * w[dw < 0] / dw[dw < 0])))
        if np.any(dz < 0):
            az = min(1.0, float(np.min(-frac * z[dz < 0] / dz[dz < 0])))
        if not split:
            a = az = min(a, az)
        psi0 = np.sqrt(sum(float(b @ b) for b in blocks))
        for _ in range(30):
            xn, yn, zn, wn = prob.retract(x, dx, a), y + a * dy, z + az * dz, w + a * dw
            bn, _ = kkt_error(prob, xn, yn, zn, wn, mu)
            if np.sqrt(sum(float(b @ b) for b in bn)) <= (1 - 1e-4 * a) * psi0 or delta > 0:
                break
            a *= 0.5
            az = a if not split else az
        x, y, z, w = xn, yn, zn, wn
        hist[-1]["alpha"], hist[-1]["delta"] = a, delta
    return dict(x=x, y=y, z=z, w=w, mu=mu, hist=hist, iters=len(hist) - 1,
                converged=(hist[-1]["E0"] <= tol) if fixed_mu is None else (hist[-1]["Emu"] <= tol))


def central_path(prob, x0, mus):
    """Solve the barrier problem for each mu in turn, each started from the previous one."""
    out, state, x = [], None, x0
    for mu in mus:
        if state is not None:                            # keep w z near the new mu
            state = (state[0], mu / state[2], state[2])
        res = interior_point(prob, x, fixed_mu=mu, tol=1e-9, state=state)
        x, state = res["x"], (res["y"], res["z"], res["w"])
        out.append(res)
    return out


def rollout(prob, taus):
    """Shooting: the discrete dynamics applied to a torque sequence."""
    R, Om = [prob.R0], [np.zeros(3)]
    for k in range(prob.N):
        R.append(R[-1] @ expm_so3(prob.h * Om[-1]))
        Om.append(Om[-1] + prob.h * np.linalg.solve(prob.J, np.cross(prob.J @ Om[-1], Om[-1]) + taus[k]))
    return np.array(R), np.array(Om)


def shooting_check(prob, tau0):
    """The same problem in the torques alone, solved by SciPy's SLSQP from tau0."""
    N = prob.N

    def term(u):
        R, Om = rollout(prob, u.reshape(N, 3))
        return np.r_[log_so3(prob.RN.T @ R[-1]), Om[-1]]

    def cone(u):
        R, _ = rollout(prob, u.reshape(N, 3))
        return prob.cos_c - R[1:N, :, 0] @ prob.q_c

    res = minimize(lambda u: 0.5 * prob.h * u @ u, tau0.ravel(), jac=lambda u: prob.h * u, method="SLSQP",
                   bounds=[(-prob.tau_max, prob.tau_max)] * (3 * N),
                   constraints=[dict(type="eq", fun=term), dict(type="ineq", fun=cone)],
                   options=dict(maxiter=500, ftol=1e-13))
    return res


def order(errs):
    """Observed convergence orders log(e2/e1)/log(e1/e0) of a sequence of errors."""
    e = np.asarray(errs, float)
    return [float(np.log(e[i + 2] / e[i + 1]) / np.log(e[i + 1] / e[i])) for i in range(len(e) - 2)]


def scenario_flip():
    """The flip of Chapter 7 with a torque bound below what the reference asks for."""
    ref = flip_ref()
    R0 = ref(0.0)[0] @ expm_so3(np.radians(20.0) * unit([1.0, 1.0, 1.0]))
    return dict(ref=ref, R0=R0, T_end=2.2, tau_max=0.6)


def scenario_hold():
    """Hold a nose-up attitude at 85 degrees of pitch, as in Example 7.1, with the same bound."""
    Rd = Ry(np.radians(85.0))
    ref = lambda t: (Rd, ZERO3, ZERO3)                   # noqa: E731
    R0 = Rd @ expm_so3(np.radians(20.0) * unit([1.0, 1.0, 1.0]))
    return dict(ref=ref, R0=R0, T_end=3.0, tau_max=0.6)


def run_scenario(sc, which):
    kw = dict(tau_max=sc["tau_max"])
    ctrl = {"mpc": lambda: ErrorStateMPC(sc["ref"], sc["T_end"], **kw),
            "lqr": lambda: SaturatedLQR(sc["ref"], sc["T_end"], **kw),
            "euler": lambda: EulerMPC(sc["ref"], sc["T_end"], **kw)}[which]()
    out = closed_loop(ctrl, sc["ref"], sc["R0"], ZERO3, sc["T_end"])
    out["ctrl"] = ctrl
    out["rms"] = float(np.sqrt(np.mean(out["angle"] ** 2)))
    return out


def prediction_error(theta_deg, t0=0.45, N=N_MPC, h=H_MPC):
    """How well the linear model predicts the error N steps ahead, with the feedforward torque alone."""
    ref = flip_ref()
    k0 = int(round(t0 / h))
    Rd, Od, _ = ref(k0 * h)
    chi0 = np.radians(theta_deg) * unit([1.0, -0.6, 0.8])
    R, Om, s = Rd @ expm_so3(chi0), Od.copy(), np.r_[chi0, ZERO3]
    for i in range(N):
        tm = (k0 + i + 0.5) * h
        R, Om = plant_step(R, Om, ref(tm)[2], h)
        A, _ = zoh(err_A(ref(tm)[1]), ERR_B, h)
        s = A @ s
    Rd, Od, _ = ref((k0 + N) * h)
    true = np.r_[log_so3(Rd.T @ R), Om - Od]
    return float(np.linalg.norm(s - true) / np.linalg.norm(true))


def main():
    ck = Checker("ch13")
    rng = np.random.default_rng(13)

    # ---- the opening puzzle -------------------------------------------------
    Ra, Rb = Rz(0.9), Rx(1.2)
    ck.record("midpoint_det", float(np.linalg.det(0.5 * (Ra + Rb))))
    ck.true("the midpoint of two rotation matrices is not a rotation", abs(np.linalg.det(0.5 * (Ra + Rb)) - 1) > 0.1)
    ck.record("puzzle_vars", 9 * N_MPC)
    ck.record("puzzle_constraints", 6 * N_MPC)

    # ---- derivative of the exponential --------------------------------------
    worst = worst_inv = 0.0
    for _ in range(20):
        x, d, e = rng.normal(size=3) * 1.2, rng.normal(size=3), 1e-6
        fd = vee(expm_so3(x).T @ (expm_so3(x + e * d) - expm_so3(x - e * d))) / (2 * e)
        worst = max(worst, np.abs(fd - dexp(x) @ d).max())
        worst_inv = max(worst_inv, np.abs(dexp(x) @ dexp_inv(x) - I3).max())
    ck.close("dexp: exp((x+d)^) = exp(x^) exp((dexp_x d)^) to first order", worst, 0, tol=1e-8)
    ck.close("dexp_inv is the inverse of dexp", worst_inv, 0, tol=1e-12)
    x = rng.normal(size=3)
    ck.close("x^T dexp_inv(x) = x^T", x @ dexp_inv(x), x, tol=1e-12)

    # ---- Newton on the sphere ------------------------------------------------
    Rs, pts = mean_data()
    q = unit(pts[0] + 0.3 * rng.normal(size=3))
    g, H = s2_mean_grad_hess(q, pts)
    f = lambda xi: s2_mean_cost(expm_so3(xi) @ q, pts)                      # noqa: E731
    e = 1e-4
    gfd = np.array([(f(e * d) - f(-e * d)) / (2 * e) for d in I3])
    Hfd = np.array([[(f(e * a + e * b) - f(e * a - e * b) - f(-e * a + e * b) + f(-e * a - e * b)) / (4 * e * e)
                     for b in I3] for a in I3])
    Pq = I3 - np.outer(q, q)
    ck.close("S2 mean: gradient along q -> exp(xi^) q by finite differences", Pq @ g, Pq @ gfd, tol=1e-7)
    ck.close("S2 mean: Hessian on the tangent plane by finite differences", Pq @ H @ Pq, Pq @ Hfd @ Pq, tol=1e-5)
    qs, steps = s2_mean_newton(pts[0], pts, 6)
    ck.close("S2 Newton: every iterate is a unit vector", np.linalg.norm(qs, axis=1), 1, tol=1e-14)
    s2_err = [float(np.arccos(np.clip(qq @ qs[-1], -1, 1))) for qq in qs[:4]]
    ck.true("S2 Newton: the error is at least squared at each step", s2_err[1] < s2_err[0] ** 2 and s2_err[2] < 5 * s2_err[1] ** 2,
            " ".join(f"{np.degrees(v):.2e}" for v in s2_err[:3]) + " deg")
    avg = sum(pts) / len(pts)
    ck.record("s2_average_norm", float(np.linalg.norm(avg)))
    ck.record("s2_err_deg", [float(np.degrees(v)) for v in s2_err[:3]])
    ck.record("s2_average_vs_mean_deg", float(np.degrees(np.arccos(unit(avg) @ qs[-1]))))
    ck.record("s2_first_step_deg", float(np.degrees(np.linalg.norm(steps[0]))))

    # ---- Newton on SO(3) -----------------------------------------------------
    R = Rs[0] @ expm_so3(0.4 * rng.normal(size=3))
    g = so3_mean_grad(R, Rs)
    f = lambda eta: so3_mean_cost(R @ expm_so3(eta), Rs)                    # noqa: E731
    gfd = np.array([(f(e * d) - f(-e * d)) / (2 * e) for d in I3])
    Hfd = np.array([[(f(e * a + e * b) - f(e * a - e * b) - f(-e * a + e * b) + f(-e * a - e * b)) / (4 * e * e)
                     for b in I3] for a in I3])
    ck.close("SO3 mean: the gradient along R exp(eta^) is the sum of the logarithm errors", g, gfd, tol=1e-7)
    ck.close("SO3 mean: Hessian formula by finite differences of the cost", so3_mean_hess(R, Rs), Hfd, tol=1e-5)
    Jg = np.column_stack([(so3_mean_grad(R @ expm_so3(1e-6 * d), Rs) - so3_mean_grad(R @ expm_so3(-1e-6 * d), Rs)) / 2e-6
                          for d in I3])
    ck.close("SO3 mean: the Hessian is the symmetric part of the differenced gradient", 0.5 * (Jg + Jg.T),
             so3_mean_hess(R, Rs), tol=1e-7)
    ck.true("SO3 mean: the differenced gradient itself is not symmetric", np.abs(Jg - Jg.T).max() > 0.1)
    chi = log_so3(Rs[1].T @ R)
    ck.close("SO3: sym(dexp_inv(chi)) = (th/2) cot(th/2) I + (1 - (th/2) cot(th/2)) a a^T",
             0.5 * (dexp_inv(chi) + dexp_inv(chi).T), so3_mean_hess(R, [Rs[1]]), tol=1e-12)

    R_start = Rs[0] @ expm_so3(np.radians(100.0) * unit([0.2, 1.0, -0.4]))
    R_star = so3_mean_iterate(R_start, Rs, "newton", 12)[-1]
    ck.close("SO3 mean: the gradient vanishes at the Newton limit", so3_mean_grad(R_star, Rs), 0, tol=1e-13)
    dist = lambda Rk: float(np.linalg.norm(log_so3(R_star.T @ project_so3(Rk))))   # noqa: E731
    errs = {m: [dist(Rk) for Rk in so3_mean_iterate(R_start, Rs, m, 10)] for m in
            ("newton", "project", "gradient", "additive")}
    on_group = max(np.abs(Rk.T @ Rk - I3).max() for Rk in so3_mean_iterate(R_start, Rs, "newton", 10))
    ck.close("SO3 Newton: every iterate is a rotation", on_group, 0, tol=1e-14)
    en = [v for v in errs["newton"] if v > 1e-13]
    od = order(en)
    ck.true("SO3 Newton: observed order of convergence is 2", 1.7 < od[-1] < 2.4, f"errors {' '.join(f'{v:.1e}' for v in en)}; orders {od}")
    ep = [v for v in errs["project"] if v > 1e-13]
    ck.true("SO3 Newton with the projection retraction: also order 2", order(ep)[-1] > 1.7, f"{' '.join(f'{v:.1e}' for v in ep)}")
    eg = errs["gradient"]
    ratios = [eg[i + 1] / eg[i] for i in range(3, 7)]
    ck.true("SO3 gradient steps: linear convergence (constant ratio)", max(ratios) / min(ratios) < 1.6 and max(ratios) < 0.2,
            f"ratios {[round(v, 4) for v in ratios]}")
    add = so3_mean_iterate(R_start, Rs, "additive", 10)
    off = [float(np.linalg.norm(Rk.T @ Rk - I3)) for Rk in add]
    ck.true("additive update R + R eta^: the iterate leaves the group and never returns", off[1] > 0.1 and off[-1] > 0.1,
            f"|R^T R - I| = {off[1]:.2f} after one step")
    M_avg = sum(Rs) / len(Rs)
    ck.record("so3_average_det", float(np.linalg.det(M_avg)))
    ck.record("so3_average_vs_mean_deg", float(np.degrees(dist(project_so3(M_avg)))))
    ck.record("so3_spread_deg", float(max(np.degrees(np.linalg.norm(log_so3(R_star.T @ Ri))) for Ri in Rs)))
    ck.record("so3_newton_err_deg", [float(np.degrees(v)) for v in errs["newton"][:6]])
    ck.record("so3_newton_order", od[-1])
    ck.record("so3_gradient_err_deg", [float(np.degrees(v)) for v in eg[:6]])
    ck.record("so3_gradient_ratio", float(ratios[-1]))
    ck.record("so3_additive_offgroup", off[1])
    ck.record("so3_start_err_deg", float(np.degrees(errs["newton"][0])))

    # gradient of the weighted quadratic cost in the Lie algebra (Teng et al. 2022, Theorem 1)
    chi = np.radians(60.0) * unit([0.3, -0.8, 0.5])
    slope = lambda P, eta: (0.5 * (lambda c: c @ P @ c)(log_so3(expm_so3(chi) @ expm_so3(1e-6 * eta)))      # noqa: E731
                            - 0.5 * (lambda c: c @ P @ c)(log_so3(expm_so3(chi) @ expm_so3(-1e-6 * eta)))) / 2e-6
    P_iso, P_w = I3, np.diag([1.0, 2.0, 3.0])
    g_iso = np.array([slope(P_iso, d) for d in I3])
    g_w = np.array([slope(P_w, d) for d in I3])
    ck.close("SO3: the gradient of 1/2 |chi|^2 along R exp(eta^) is chi exactly", g_iso, chi, tol=1e-8)
    ck.close("SO3: the gradient of 1/2 chi^T P chi is dexp_inv(chi)^T P chi", g_w, dexp_inv(chi).T @ P_w @ chi, tol=1e-8)
    gap = float(np.linalg.norm(g_w - P_w @ chi) / np.linalg.norm(P_w @ chi))
    ck.true("... which is P chi only to first order when P is not a multiple of I", gap > 0.05, f"relative gap {gap:.2f} at 60 deg")
    ck.record("weighted_gradient_gap_60deg", gap)

    # ---- exact and linearized error dynamics on SO(3) -------------------------
    ref_t = lambda t: np.array([0.4 * np.sin(3 * t), 0.5 * np.cos(2 * t), 0.3 * np.sin(4 * t + 1.0)])   # noqa: E731
    dtau = lambda t: np.array([np.cos(5 * t), -0.5, np.sin(3 * t)])                                    # noqa: E731
    Rd0, Od0 = random_rotation(rng), np.array([2.0, -3.0, 1.5])

    def fly(R0, O0, tfun, T=1.0):
        return V7.so3_sim(R0, O0, lambda t, R, Om: tfun(t), T, rtol=1e-12)

    base = fly(Rd0, Od0, ref_t)
    t1, e = 0.37, 1e-5
    big = fly(Rd0 @ expm_so3(np.array([0.5, -0.7, 0.4])), Od0 + np.array([0.3, 0.2, -0.5]), ref_t)
    chi_at = lambda t: log_so3(base.sol(t)[:9].reshape(3, 3).T @ big.sol(t)[:9].reshape(3, 3))         # noqa: E731
    fd = (chi_at(t1 + e) - chi_at(t1 - e)) / (2 * e)
    ck.close("exact error dynamics: chi' = dexp_inv(chi) (Omega - exp(-chi^) Omega_d), along a 55 deg error",
             fd, chi_dot_exact(chi_at(t1), big.sol(t1)[9:], base.sol(t1)[9:]), tol=1e-6)
    ck.record("exact_check_angle_deg", float(np.degrees(np.linalg.norm(chi_at(t1)))))

    worst = 0.0
    for _ in range(10):
        Rr, Orr = random_rotation(rng), rng.normal(size=3) * 3
        worst = max(worst, np.abs(err_A(Orr) - V7.so3_A(Rr, Orr)).max())
    ck.close("error-state A equals the verified matrix of Chapter 7, at any reference attitude", worst, 0, tol=1e-14)
    err_eom, modR, layR = V7.geomech_so3()
    ck.true("geomech check_eom below 1e-5", err_eom < 1e-5, f"{err_eom:.1e}")
    worst = 0.0
    for _ in range(10):
        Rr, Orr = random_rotation(rng), rng.normal(size=3) * 3
        pp = {"J": J_B, "\\rho": ZERO3, "e3": E3, "g": 9.81, "m": 1.0}
        Ag, Bg = V7.geomech_AB(modR, layR.pack(R=Rr, Omega_R=Orr), rng.normal(size=3), pp)
        worst = max(worst, np.abs(Ag - err_A(Orr)).max(), np.abs(Bg - ERR_B).max())
    ck.close("error-state A and B equal geomech's linearization entry by entry", worst, 0, tol=1e-9)

    def lin_gap(eps, chi0, dO0):
        up = fly(Rd0 @ expm_so3(eps * chi0), Od0 + eps * dO0, lambda t: ref_t(t) + eps * dtau(t))
        dn = fly(Rd0 @ expm_so3(-eps * chi0), Od0 - eps * dO0, lambda t: ref_t(t) - eps * dtau(t))
        lin = solve_ivp(lambda t, s: err_A(base.sol(t)[9:]) @ s + ERR_B @ dtau(t), [0, 1.0], np.r_[chi0, dO0],
                        rtol=1e-12, atol=1e-14, dense_output=True)
        gap = 0.0
        for t in np.linspace(0, 1.0, 21):
            Rb = base.sol(t)[:9].reshape(3, 3)
            cp = log_so3(Rb.T @ up.sol(t)[:9].reshape(3, 3))
            cm = log_so3(Rb.T @ dn.sol(t)[:9].reshape(3, 3))
            sfd = np.r_[(cp - cm), up.sol(t)[9:] - dn.sol(t)[9:]] / (2 * eps)
            gap = max(gap, np.abs(sfd - lin.sol(t)).max())
        return gap

    chi0, dO0 = rng.normal(size=3), rng.normal(size=3)
    g3, g2 = lin_gap(1e-3, chi0, dO0), lin_gap(1e-2, chi0, dO0)
    ck.true("linearized error dynamics match perturbed trajectories of the tumbling body", g3 < 1e-5, f"gap {g3:.1e} at eps 1e-3")
    ck.true("... and the mismatch is second order in the perturbation", 50 < g2 / g3 < 200, f"ratio {g2 / g3:.0f}")
    ck.record("so3_lin_gap_1e-3", g3)
    ck.record("so3_lin_gap_ratio", g2 / g3)

    # ---- SE(3), twists ordered (V, Omega) --------------------------------------
    za, zb = rng.normal(size=6), rng.normal(size=6)
    ck.close("ad on se(3): (ad_a b)^ = [a^, b^] with twists (V, Omega)",
             hat_se3(ad_se3(za) @ zb), hat_se3(za) @ hat_se3(zb) - hat_se3(zb) @ hat_se3(za), tol=1e-12)
    X = exp_se3(rng.normal(size=6))
    AdX = np.block([[X[:3, :3], hat(X[:3, 3]) @ X[:3, :3]], [np.zeros((3, 3)), X[:3, :3]]])
    ck.close("Ad_X of (2.x) acts as X zeta^ X^-1", hat_se3(AdX @ za), X @ hat_se3(za) @ np.linalg.inv(X), tol=1e-12)
    perm = np.block([[np.zeros((3, 3)), I3], [I3, np.zeros((3, 3))]])       # (V, Omega) <-> (omega, v)
    ad_jang = np.block([[hat(za[3:]), np.zeros((3, 3))], [hat(za[:3]), hat(za[3:])]])
    ck.close("ad of Jang et al. (twist ordered (omega, v)) is ours with the blocks swapped",
             perm @ ad_jang @ perm, ad_se3(za), tol=1e-14)
    zl = rng.normal(size=6) * 0.8
    ck.close("log_se3 inverts exp_se3", log_se3(exp_se3(zl)), zl, tol=1e-10)
    V_, Om_, u_ = za[:3], za[3:], zb
    newton_euler = np.r_[u_[:3] / M_SE3 - np.cross(Om_, V_), J_INV @ (u_[3:] - np.cross(Om_, J_B @ Om_))]
    ck.close("Euler-Poincare on SE(3) is Newton-Euler in the body frame", se3_acc(za, u_), newton_euler, tol=1e-12)

    u_ref = lambda t: np.array([0.5 * np.sin(2 * t), 0.3, -0.4 * np.cos(3 * t), 0.02 * np.sin(3 * t), 0.03, 0.01])   # noqa: E731
    du = lambda t: np.array([1.0, np.sin(4 * t), -0.5, 0.05 * np.cos(2 * t), 0.02, -0.03])                           # noqa: E731
    X0, z0 = exp_se3(rng.normal(size=6)), np.array([0.8, -0.5, 0.3, 1.0, -1.5, 0.7])
    base6 = se3_sim(X0, z0, u_ref, 1.0)
    B6 = se3_err_B()

    def lin_gap6(eps, c0, dz0):
        up = se3_sim(X0 @ exp_se3(eps * c0), z0 + eps * dz0, lambda t: u_ref(t) + eps * du(t), 1.0)
        dn = se3_sim(X0 @ exp_se3(-eps * c0), z0 - eps * dz0, lambda t: u_ref(t) - eps * du(t), 1.0)
        lin = solve_ivp(lambda t, s: se3_err_A(base6.sol(t)[16:]) @ s + B6 @ du(t), [0, 1.0], np.r_[c0, dz0],
                        rtol=1e-12, atol=1e-14, dense_output=True)
        gap = 0.0
        for t in np.linspace(0, 1.0, 21):
            Xi = np.linalg.inv(base6.sol(t)[:16].reshape(4, 4))
            cp = log_se3(Xi @ up.sol(t)[:16].reshape(4, 4))
            cm = log_se3(Xi @ dn.sol(t)[:16].reshape(4, 4))
            sfd = np.r_[cp - cm, up.sol(t)[16:] - dn.sol(t)[16:]] / (2 * eps)
            gap = max(gap, np.abs(sfd - lin.sol(t)).max())
        return gap

    c0, dz0 = rng.normal(size=6), rng.normal(size=6)
    g3, g2 = lin_gap6(1e-3, c0, dz0), lin_gap6(1e-2, c0, dz0)
    ck.true("SE(3): linearized error dynamics match perturbed trajectories", g3 < 1e-4, f"gap {g3:.1e} at eps 1e-3")
    ck.true("SE(3): the mismatch is second order", 50 < g2 / g3 < 200, f"ratio {g2 / g3:.0f}")
    ck.record("se3_lin_gap_1e-3", g3)
    ck.record("se3_lin_gap_ratio", g2 / g3)
    Y = exp_se3(rng.normal(size=6))
    Xd, Xa = exp_se3(rng.normal(size=6) * 0.5), exp_se3(rng.normal(size=6) * 0.5)
    ck.close("left invariance: chi is unchanged when X and X_d are both multiplied on the left",
             log_se3(np.linalg.inv(Y @ Xd) @ (Y @ Xa)), log_se3(np.linalg.inv(Xd) @ Xa), tol=1e-10)
    # on SE(3) the slope of 1/2 |chi|^2 is chi only to first order
    cchi = np.r_[0.4, -0.3, 0.5, 0.6, -0.2, 0.3]
    sl = np.array([(0.5 * np.sum(log_se3(exp_se3(cchi) @ exp_se3(1e-6 * d)) ** 2)
                    - 0.5 * np.sum(log_se3(exp_se3(cchi) @ exp_se3(-1e-6 * d)) ** 2)) / 2e-6 for d in np.eye(6)])
    ck.true("SE(3): the slope of 1/2 |chi|^2 along X exp(eta^) is chi only to first order",
            np.linalg.norm(sl - cchi) / np.linalg.norm(cchi) > 0.05, f"relative gap {np.linalg.norm(sl - cchi) / np.linalg.norm(cchi):.2f}")

    # ---- sampling and the QP ---------------------------------------------------
    Od = np.array([1.0, -4.0, 2.0])
    A, B = zoh(err_A(Od), ERR_B, H_MPC)
    s0, u0 = rng.normal(size=6), rng.normal(size=3)
    ex = solve_ivp(lambda t, s: err_A(Od) @ s + ERR_B @ u0, [0, H_MPC], s0, rtol=1e-12, atol=1e-14).y[:, -1]
    ck.close("zoh: exact sampling of the linear model", A @ s0 + B @ u0, ex, tol=1e-10)

    ref = flip_ref()
    f7 = V7.Flip()
    ck.close("the flip reference is the one of Chapter 7", np.r_[ref(0.7)[0].ravel(), ref(0.7)[1], ref(0.7)[2]],
             np.r_[f7(0.7)[0].ravel(), f7(0.7)[1], f7(0.7)[2]], tol=1e-14)
    tt = np.linspace(0, 1.5, 3001)
    ff = np.array([ref(t)[2] for t in tt])
    ck.record("flip_ff_peak", float(np.abs(ff).max()))
    Pf, K_rest = terminal_weight()
    mpc = ErrorStateMPC(ref, 2.2, tau_max=0.6)
    k0 = 20
    Hq, Fq = mpc.data(k0)
    As = [mpc.AB[k0 + i][0] for i in range(N_MPC)]
    Bs = [mpc.AB[k0 + i][1] for i in range(N_MPC)]
    _, _, Sx, Su = condense(As, Bs, mpc.Qh, mpc.Rh, Pf)
    s0, U = rng.normal(size=6) * 0.3, rng.normal(size=3 * N_MPC) * 0.2
    s, cost = s0.copy(), 0.0
    for i in range(N_MPC):
        ui = U[3 * i:3 * i + 3]
        cost += 0.5 * ui @ mpc.Rh @ ui + (0.5 * s @ mpc.Qh @ s if i > 0 else 0.0)
        s = As[i] @ s + Bs[i] @ ui
    cost += 0.5 * s @ Pf @ s
    sN = (Sx @ s0 + Su @ U)[-6:]
    ck.close("condensing: the stacked prediction equals the rollout", sN, s, tol=1e-12)
    cst = 0.5 * (Sx @ s0) @ np.kron(np.eye(N_MPC), mpc.Qh) @ (Sx @ s0)
    Qb = np.kron(np.eye(N_MPC), mpc.Qh)
    Qb[-6:, -6:] = Pf
    cst = 0.5 * (Sx @ s0) @ Qb @ (Sx @ s0)
    ck.close("condensing: 1/2 U^T H U + (F s0).U + const equals the rollout cost", 0.5 * U @ Hq @ U + (Fq @ s0) @ U + cst, cost, tol=1e-11)
    ev = np.linalg.eigvalsh(Hq)
    ck.true("the QP is strictly convex: H is positive definite", ev.min() > 0, f"eigenvalues in [{ev.min():.3g}, {ev.max():.3g}]")
    ck.record("qp_cond", float(ev.max() / ev.min()))
    ck.record("qp_vars", 3 * N_MPC)

    # invariance: the same QP in another world frame, and at another reference attitude
    Yw = random_rotation(rng)
    Rd, Od, _ = ref(k0 * H_MPC)
    Rk = Rd @ expm_so3(np.array([0.3, -0.2, 0.25]))
    ck.close("left invariance: the error state does not change when R and R_d are both turned by Y",
             log_so3((Yw @ Rd).T @ (Yw @ Rk)), log_so3(Rd.T @ Rk), tol=1e-12)

    class Turned:                                        # the same body-frame motion started from another attitude
        def __call__(self, t):
            Rd_, Od_, td_ = ref(t)
            return Yw @ Rd_, Od_, td_

    Hq2, Fq2 = ErrorStateMPC(Turned(), 2.2, tau_max=0.6).data(k0)
    ck.close("left invariance: H and F of the QP are the same numbers at any reference attitude",
             np.r_[Hq2.ravel(), Fq2.ravel()], np.r_[Hq.ravel(), Fq.ravel()], tol=1e-14)

    # solver against SciPy and the optimality conditions
    worst_x = worst_kkt = 0.0
    n_active, it_max = [], 0
    for _ in range(40):
        s0 = np.r_[np.radians(rng.uniform(5, 90)) * unit(rng.normal(size=3)), rng.normal(size=3) * 2]
        ffv = mpc.tau_d[k0:k0 + N_MPC].ravel()
        lo, hi = -0.6 - ffv, 0.6 - ffv
        U, it = box_qp(Hq, Fq @ s0, lo, hi)
        Uref = box_qp_reference(Hq, Fq @ s0, lo, hi)
        worst_x = max(worst_x, np.abs(U - Uref).max())
        worst_kkt = max(worst_kkt, box_qp_kkt(Hq, Fq @ s0, lo, hi, U))
        n_active.append(int(np.sum((U <= lo + 1e-9) | (U >= hi - 1e-9))))
        it_max = max(it_max, it)
    ck.close("box QP: projected Newton agrees with SciPy's bounded least squares on 40 problems", worst_x, 0, tol=1e-7)
    ck.close("box QP: optimality conditions hold at the solution", worst_kkt, 0, tol=1e-8)
    ck.true("box QP: the test problems have active bounds", np.mean(n_active) > 5, f"mean active {np.mean(n_active):.1f} of 75, at most {it_max} iterations")
    ck.record("qp_test_iters_max", it_max)

    # MPC without bounds is LQR
    Arest, Brest = zoh(err_A(ZERO3), ERR_B, H_MPC)
    gaps = []
    for N in (1, 5, 25, 60):
        Hn, Fn, _, _ = condense([Arest] * N, [Brest] * N, mpc.Qh, mpc.Rh, Pf)
        Kmpc = np.linalg.solve(Hn, Fn)[:3]
        gaps.append(np.abs(Kmpc - K_rest).max())
    ck.close("MPC = LQR: with no bounds and the Riccati terminal weight, the first move is -K s0 for any N", max(gaps), 0, tol=1e-9)
    no_riccati = {}
    for N in (5, 25, 100):
        Hn, Fn, _, _ = condense([Arest] * N, [Brest] * N, mpc.Qh, mpc.Rh, mpc.Qh)
        no_riccati[N] = float(np.linalg.norm(np.linalg.solve(Hn, Fn)[:3] - K_rest) / np.linalg.norm(K_rest))
    ck.true("MPC -> LQR as the horizon grows, with terminal weight Q", no_riccati[100] < 1e-3 < no_riccati[5],
            f"gain mismatch {no_riccati}")
    ck.record("mpc_lqr_gap_no_riccati", no_riccati)
    ck.record("K_rest_kR", float(K_rest[0, 0]))
    ck.record("K_rest_kOmega", [float(K_rest[i, 3 + i]) for i in range(3)])
    # along the flip: the unconstrained MPC move equals the Riccati recursion when the terminal weights agree
    lqr = SaturatedLQR(ref, 2.2, tau_max=np.inf)
    P = terminal_weight()[0]
    Ps = {}
    for k in reversed(range(len(lqr.K))):
        Ps[k + 1] = P
        Ak, Bk = mpc.AB[k] if k < len(mpc.AB) else zoh(err_A(ref(k * H_MPC + 0.5 * H_MPC)[1]), ERR_B, H_MPC)
        P = mpc.Qh + Ak.T @ P @ (Ak - Bk @ lqr.K[k])
    Hn, Fn, _, _ = condense(As, Bs, mpc.Qh, mpc.Rh, Ps[k0 + N_MPC])
    ck.close("MPC = LQR along the flip: same first move as the Riccati recursion", np.linalg.solve(Hn, Fn)[:3], lqr.K[k0], tol=1e-9)
    ck.record("mpc_vs_lqr_flip_gain_gap", float(np.linalg.norm(np.linalg.solve(Hq, Fq)[:3] - lqr.K[k0]) / np.linalg.norm(lqr.K[k0])))

    # ---- closed loop -------------------------------------------------------------
    sc = scenario_flip()
    runs = {w: run_scenario(sc, w) for w in ("mpc", "lqr", "euler")}
    for w, r in runs.items():
        ck.true(f"flip, {w}: the applied torque respects the bound at every step", np.abs(r["tau"]).max() <= 0.6 + 1e-12,
                f"max {np.abs(r['tau']).max():.6f}")
        ck.close(f"flip, {w}: the attitude stays a rotation", np.abs(np.einsum("kij,kil->kjl", r["R"], r["R"]) - I3).max(), 0, tol=1e-12)
    late = lambda r: float(r["angle"][int(0.5 / H_MPC):].max())               # noqa: E731
    ck.true("flip: error-state MPC tracks through the infeasible stretch far better than the saturated LQR",
            late(runs["mpc"]) < 0.25 * late(runs["lqr"]),
            f"worst error after 0.5 s: mpc {late(runs['mpc']):.1f}, lqr {late(runs['lqr']):.1f}, euler {late(runs['euler']):.1f} deg")
    ck.true("flip: the Euler-angle MPC is comparable here (the reference passes the pole quickly)",
            late(runs["euler"]) < 2.0 * late(runs["mpc"]))
    for w, r in runs.items():
        ck.record(f"flip_{w}_late_peak_deg", late(r))
        ck.record(f"flip_{w}_final_deg", float(r["angle"][-1]))
        ck.record(f"flip_{w}_rms_deg", r["rms"])
        ck.record(f"flip_{w}_sat_fraction", float(np.mean(np.abs(r["tau"]).max(axis=1) >= 0.6 - 1e-9)))
    for w in ("mpc", "euler"):
        c = runs[w]["ctrl"]
        ck.record(f"flip_{w}_qp_iters_mean", float(np.mean(c.iters)))
        ck.record(f"flip_{w}_qp_iters_max", int(max(c.iters)))
        ck.record(f"flip_{w}_solve_ms_median", float(1e3 * np.median(c.times)))
        ck.record(f"flip_{w}_solve_ms_max", float(1e3 * max(c.times)))
    ck.true("flip: the QP is solved well inside the 20 ms sampling period", 1e3 * max(runs["mpc"]["ctrl"].times) < 20.0,
            f"median {1e3 * np.median(runs['mpc']['ctrl'].times):.2f} ms, max {1e3 * max(runs['mpc']['ctrl'].times):.2f} ms")
    # closed-loop QPs once more against SciPy
    c = ErrorStateMPC(ref, 2.2, tau_max=0.6)
    worst = 0.0
    for k in range(0, 110, 5):
        Hk, Fk = c.data(k)
        s0 = c.error_state(k, runs["mpc"]["R"][k], runs["mpc"]["Om"][k])
        ffv = c.tau_d[k:k + N_MPC].ravel()
        U, _ = box_qp(Hk, Fk @ s0, -0.6 - ffv, 0.6 - ffv)
        worst = max(worst, np.abs(U - box_qp_reference(Hk, Fk @ s0, -0.6 - ffv, 0.6 - ffv)).max())
    ck.close("flip: the QPs met in closed loop agree with SciPy", worst, 0, tol=1e-7)
    # the pole: Euler matrices against ours
    eA = max(np.abs(a).max() for a in runs["euler"]["ctrl"].A_cont)
    ck.record("euler_A_max", float(eA))
    ck.record("err_A_max", float(max(np.abs(err_A(ref(t)[1])).max() for t in tt)))

    sh = scenario_hold()
    hold = {w: run_scenario(sh, w) for w in ("mpc", "euler")}
    ck.true("hold at 85 deg of pitch: error-state MPC never exceeds the initial error",
            hold["mpc"]["angle"].max() <= 20.0 + 1e-9 and hold["mpc"]["angle"][-1] < 0.01)
    ck.true("hold at 85 deg of pitch: the Euler-angle MPC tumbles", hold["euler"]["angle"].max() > 150,
            f"peak {hold['euler']['angle'].max():.1f} deg, {hold['euler']['angle'][-1]:.1f} deg after 3 s")
    ck.true("hold: both respect the bound", max(np.abs(hold[w]["tau"]).max() for w in hold) <= 0.6 + 1e-12)
    ck.record("hold_mpc_peak_torque", float(np.abs(hold["mpc"]["tau"]).max()))
    ck.record("hold_mpc_settle_1deg_s", float(hold["mpc"]["t"][np.nonzero(hold["mpc"]["angle"] > 1.0)[0][-1] + 1]))
    ck.record("hold_euler_peak_deg", float(hold["euler"]["angle"].max()))
    ck.record("hold_euler_final_deg", float(hold["euler"]["angle"][-1]))
    ck.record("hold_euler_sat_fraction", float(np.mean(np.abs(hold["euler"]["tau"]).max(axis=1) >= 0.6 - 1e-9)))

    # ---- how far the linearization reaches -----------------------------------------
    pe = {th: prediction_error(th) for th in (2, 10, 30, 60, 120)}
    ck.true("prediction error of the linear model grows in proportion to the error angle", 3 < pe[10] / pe[2] < 7 and pe[120] > 0.3,
            " ".join(f"{k}deg:{100 * v:.1f}%" for k, v in pe.items()))
    ck.record("prediction_error", {str(k): v for k, v in pe.items()})
    Rd, Od, _ = ref(0.0)
    a = unit([0.3, 1.0, 0.2])
    t179 = ErrorStateMPC(ref, 2.2, tau_max=0.6)(0, Rd @ expm_so3(np.radians(179.0) * a), Od)
    t181 = ErrorStateMPC(ref, 2.2, tau_max=0.6)(0, Rd @ expm_so3(np.radians(181.0) * a), Od)
    cosang = float(t179 @ t181 / np.linalg.norm(t179) / np.linalg.norm(t181))
    ck.true("cut locus: errors of 179 and 181 deg about one axis get opposite torques", cosang < -0.99, f"cosine {cosang:.4f}")
    ck.record("cut_locus_cosine", cosang)

    # ---- the barrier in one variable --------------------------------------------------
    for mu in (1.0, 1e-2, 1e-6):
        xb = minimize(lambda v: v[0] - mu * np.log(v[0]), [0.5], bounds=[(1e-12, None)], tol=1e-14).x[0]
        ck.close(f"barrier: min x - mu ln x is at x = mu (mu = {mu:g})", xb, mu, tol=1e-6 * max(mu, 1e-3) + 1e-9)
    xk, zk, mu = 2.0, 0.3, 0.1                           # Newton on (1 - z, x z - mu) = 0
    for _ in range(3):
        dz = 1 - zk
        dx = (-(xk * zk - mu) - xk * dz) / zk
        xk, zk = xk + dx, zk + dz
    ck.close("primal-dual Newton on the one-variable problem lands on (mu, 1)", [xk, zk], [mu, 1.0], tol=1e-12)

    # ---- direct transcription: derivatives ----------------------------------------------
    prob = Reorient()
    x = prob.retract(prob.initial_guess(0.3), 0.1 * rng.normal(size=prob.n))
    d, e = rng.normal(size=prob.n), 1e-6
    fd = (prob.eq(prob.retract(x, d, e)) - prob.eq(prob.retract(x, d, -e))) / (2 * e)
    ck.close("transcription: Jacobian of the dynamics residuals with respect to (eta, dOmega, dtau)", fd, prob.eq_jac(x) @ d, tol=1e-8)
    fd = (prob.ineq(prob.retract(x, d, e)) - prob.ineq(prob.retract(x, d, -e))) / (2 * e)
    ck.close("transcription: Jacobian of the bounds and the keep-out cone", fd, prob.ineq_jac(x) @ d, tol=1e-8)
    y, z = rng.normal(size=prob.n_eq), rng.uniform(size=prob.n_in)
    Lf = lambda xx: prob.cost(xx) + y @ prob.eq(xx) + z @ prob.ineq(xx)     # noqa: E731
    Hl = prob.lag_hess(x, y, z)
    e2 = 1e-4
    quad = (Lf(prob.retract(x, d, e2)) - 2 * Lf(x) + Lf(prob.retract(x, d, -e2))) / e2**2
    ck.close("transcription: Hessian of the Lagrangian along the retraction (second difference of L)", d @ Hl @ d / quad, 1, tol=1e-5)
    bw = max(abs(i - j) for i, j in zip(*np.nonzero(np.abs(Hl) > 1e-9)))
    ck.true("transcription: the Hessian is block tridiagonal along the horizon", bw < 18, f"bandwidth {bw} of {prob.n}")
    Rk, eta = x["R"][5], rng.normal(size=3)
    v = Rk.T @ prob.q_c
    cone = lambda t_: prob.q_c @ Rk @ expm_so3(t_ * eta) @ E1               # noqa: E731
    Hc = 0.5 * (np.outer(E1, v) + np.outer(v, E1)) - (v @ E1) * I3
    ck.close("cone: gradient e1 x R^T q_c", (cone(e) - cone(-e)) / (2 * e), np.cross(E1, v) @ eta, tol=1e-9)
    ck.close("cone: Hessian sym(e1 v^T) - (v.e1) I with v = R^T q_c", (cone(e2) - 2 * cone(0) + cone(-e2)) / e2**2, eta @ Hc @ eta, tol=1e-6)
    # the second-order model of the kinematics in LieIPM (Table 1, first row), by brute force
    Fk, Yk = expm_so3(rng.normal(size=3)), expm_so3(0.01 * rng.normal(size=3))
    a1, a2, a3 = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)

    def bch_gap(t_):
        lhs = log_so3(expm_so3(-t_ * Yk.T @ a1) @ expm_so3(t_ * Fk.T @ a2) @ expm_so3(t_ * a3))
        b1, b2, b3 = -Yk.T @ a1, Fk.T @ a2, a3
        model = t_ * (b1 + b2 + b3) + 0.5 * t_**2 * (np.cross(b1, b2) + np.cross(b1, b3) + np.cross(b2, b3))
        return np.linalg.norm(lhs - model)

    ck.true("LieIPM Table 1: second-order model of the SO(3) kinematics is exact to third order",
            6 < bch_gap(2e-2) / bch_gap(1e-2) < 10, f"ratio {bch_gap(2e-2) / bch_gap(1e-2):.2f} (8 expected)")
    Jd = 0.5 * np.trace(J_B) * I3 - J_B
    ck.close("variational integrator: Omega^ J_d + J_d Omega^ = (J Omega)^ with J_d = tr(J)/2 I - J",
             hat(a1) @ Jd + Jd @ hat(a1), hat(J_B @ a1), tol=1e-14)

    # ---- direct transcription: the interior point -------------------------------------------
    worst_group = [0.0]

    def watch(it, xx, mu_):
        worst_group[0] = max(worst_group[0], float(np.abs(np.einsum("kij,kil->kjl", xx["R"], xx["R"]) - I3).max()))

    t0 = time.perf_counter()
    sol = interior_point(prob, prob.initial_guess(0.0), log=watch)
    t_ipm = time.perf_counter() - t0
    xs = sol["x"]
    ck.true("interior point: converged to a KKT residual below 1e-8", sol["converged"], f"{sol['iters']} iterations, {t_ipm:.1f} s")
    ck.close("interior point: every iterate keeps every attitude on SO(3)", worst_group[0], 0, tol=1e-12)
    ck.close("interior point: the discrete dynamics hold", prob.eq(xs), 0, tol=1e-8)
    Rr, Orr = rollout(prob, xs["tau"])
    ck.close("interior point: rolling out the torques reproduces the attitudes and reaches the target at rest",
             np.r_[(Rr - xs["R"]).ravel(), log_so3(prob.RN.T @ Rr[-1]), Orr[-1]], 0, tol=1e-7)
    ck.true("interior point: torque bounds and keep-out cone hold", prob.ineq(xs).max() <= 1e-9, f"max c = {prob.ineq(xs).max():.1e}")
    ck.true("interior point: multipliers are nonnegative and complementary", sol["z"].min() >= 0 and np.abs(sol["z"] * prob.ineq(xs)).max() < 1e-7)
    act = np.nonzero(-prob.ineq(xs) < 1e-6)[0]
    n_tau_act, n_cone_act = int(np.sum(act < 6 * prob.N)), int(np.sum(act >= 6 * prob.N))
    Aact = np.vstack([prob.eq_jac(xs), prob.ineq_jac(xs)[act]])
    _, sv, Vt = np.linalg.svd(Aact)
    Zn = Vt[np.sum(sv > 1e-9):].T
    red = np.linalg.eigvalsh(Zn.T @ prob.lag_hess(xs, sol["y"], sol["z"]) @ Zn)
    ck.true("interior point: second-order condition, reduced Hessian positive definite (a local minimum)", red.min() > 0,
            f"{n_tau_act} torque and {n_cone_act} cone constraints active, min eigenvalue {red.min():.2e}")
    chk = shooting_check(prob, xs["tau"] + 0.01 * rng.normal(size=xs["tau"].shape))
    ck.close("interior point: SciPy SLSQP on the shooting form finds the same cost", chk.fun, prob.cost(xs), tol=1e-7)
    ck.close("interior point: ... and the same torques", chk.x.reshape(-1, 3), xs["tau"], tol=2e-4)
    geo = prob.initial_guess(0.0)
    ck.true("the geodesic initial guess passes through the cone", prob.ineq(geo)[6 * prob.N:].max() > 0,
            f"deepest {np.degrees(np.arccos(prob.ineq(geo)[6 * prob.N:].max() + prob.cos_c)):.1f} deg from the cone axis")
    ck.record("ipm_iters", sol["iters"])
    ck.record("ipm_cost", prob.cost(xs))
    ck.record("ipm_active_torque", n_tau_act)
    ck.record("ipm_active_cone", n_cone_act)
    ck.record("ipm_n", prob.n)
    ck.record("ipm_n_eq", prob.n_eq)
    ck.record("ipm_n_in", prob.n_in)
    ck.record("ipm_guess_min_angle_deg", float(np.degrees(np.arccos(prob.ineq(geo)[6 * prob.N:].max() + prob.cos_c))))
    ck.record("ipm_peak_rate", float(np.linalg.norm(xs["Om"], axis=1).max()))
    ck.record("ipm_inertia_corrections", int(sum(hh.get("delta", 0) > 0 for hh in sol["hist"])))

    # no cone: how much the detour costs
    free = Reorient(cone_deg=0.01, q_c=[0, 0, 1.0])
    sol_free = interior_point(free, free.initial_guess(0.0))
    ck.true("without the cone the problem converges too", sol_free["converged"])
    ck.record("ipm_cost_no_cone", free.cost(sol_free["x"]))

    # local minima: other initial guesses, other answers
    costs = {}
    for bulge in (0.6, -0.3):
        sb = interior_point(prob, prob.initial_guess(bulge))
        ck.true(f"interior point from a guess bulged by {bulge:+.1f} rad converges", sb["converged"], f"{sb['iters']} iterations, cost {prob.cost(sb['x']):.6f}")
        costs[bulge] = prob.cost(sb["x"])
        side = float(np.sign((sb["x"]["R"][prob.N // 2] @ E1) @ E3))
        ck.record(f"ipm_side_{bulge:+.1f}", side)
    costs[0.0] = prob.cost(xs)
    vals = sorted(costs.values())
    ck.true("the problem has several local minima: different guesses, different costs", vals[-1] - vals[0] > 1e-5, f"{[round(v, 6) for v in vals]}")
    ck.record("ipm_local_costs", {f"{k:+.1f}": v for k, v in costs.items()})

    # central path
    mus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    path = central_path(prob, prob.initial_guess(0.0), mus)
    ck.true("central path: every barrier subproblem converges", all(p_["converged"] for p_ in path))
    gapc = [float(-prob.ineq(p_["x"])[6 * prob.N:].max()) for p_ in path]
    pc = [prob.cost(p_["x"]) for p_ in path]
    ck.true("central path: the cost falls and the margin to the cone closes as mu shrinks",
            all(np.diff(pc) < 0) and all(np.diff(gapc) < 0), f"cost {[round(v, 5) for v in pc]}, margin {[f'{v:.1e}' for v in gapc]}")
    comp = max(np.abs(p_["w"] * p_["z"] - mu_).max() for p_, mu_ in zip(path, mus))
    ck.close("central path: w z = mu for every inequality", comp, 0, tol=1e-9)
    ck.close("central path: the last point is the solution found before", prob.cost(path[-1]["x"]), prob.cost(xs), tol=1e-4)
    ck.record("central_path_cost", pc)
    ck.record("central_path_margin_deg", [float(np.degrees(np.arccos(prob.cos_c - g_) - np.radians(prob.cone_deg))) for g_ in gapc])
    em = [hh["Emu"] for hh in path[2]["hist"]]
    tail = [v for v in em if v < 1e-1][-4:]
    ck.true("Newton at fixed mu: the residual falls faster than linearly at the end", len(tail) >= 3 and tail[-1] < tail[-2] ** 1.5,
            f"last residuals {' '.join(f'{v:.1e}' for v in tail)}")
    ck.record("fixed_mu_residuals", tail)

    # sparsity of the reduced KKT matrix
    Hs = prob.lag_hess(xs, sol["y"], sol["z"])
    Ar = prob.eq_jac(xs)
    Kmat = np.block([[Hs + prob.ineq_jac(xs).T @ prob.ineq_jac(xs), Ar.T], [Ar, np.zeros((prob.n_eq, prob.n_eq))]])
    ck.record("kkt_size", int(Kmat.shape[0]))
    ck.record("kkt_fill", float(np.mean(np.abs(Kmat) > 1e-12)))
    ck.true("the KKT matrix is sparse", np.mean(np.abs(Kmat) > 1e-12) < 0.05, f"{100 * np.mean(np.abs(Kmat) > 1e-12):.1f}% nonzero")

    ck.finish()


if __name__ == "__main__":
    main()
