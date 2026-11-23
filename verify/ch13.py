"""Chapter 13: every identity, linearization, QP solution and quoted number, checked.

    python verify/ch13.py

The functions above ``main`` are the chapter's solvers and controllers. The figure
script imports them, so the figures show exactly what is verified here.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
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


if __name__ == "__main__":
    pass
