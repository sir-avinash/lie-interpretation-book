"""Numerical checks for Chapter 14 (filtering on Lie groups).  Run: python verify/ch14.py

The filters, simulations and Monte Carlo studies live here, and the figure
script and the notebook reproduce them, so the figures show what is verified.

Conventions (the chapter's):
  * a hat marks an estimate; the errors are estimate relative to truth,
        E_r = Xhat X^{-1} = exp(chi_r^),     E_l = X^{-1} Xhat = exp(chi^);
  * the vector of an element of the Lie algebra of SE_K(3) lists the linear
    blocks in the order of the columns of the matrix, then the rotation:
        SE_2(3):  X = [R v x],   chi = (chi_v, chi_x, chi_R);
  * a correction subtracts: Xhat+ = exp(-(Ktilde ytilde)^) Xhat.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.linalg import expm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, expm_so3, hat, random_rotation, vee  # noqa: E402

G = 9.81
E1, E2, E3 = np.eye(3)


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def wrap(a):
    """Wrap an angle into (-pi, pi]: the logarithm on the circle."""
    return np.pi - np.mod(np.pi - a, 2 * np.pi)


# ------------------------------------------------------------ SO(3), batched
def hat_b(w):
    """Hat map on an array (..., 3) -> (..., 3, 3)."""
    w = np.asarray(w, float)
    M = np.zeros(w.shape[:-1] + (3, 3))
    M[..., 0, 1], M[..., 0, 2] = -w[..., 2], w[..., 1]
    M[..., 1, 0], M[..., 1, 2] = w[..., 2], -w[..., 0]
    M[..., 2, 0], M[..., 2, 1] = -w[..., 1], w[..., 0]
    return M


def _coef(w):
    th2 = np.sum(w * w, axis=-1)[..., None, None]
    th = np.sqrt(th2)
    small = th < 1e-5
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1 - th2 / 6, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th2 / 24, (1 - np.cos(ths)) / ths**2)
    c = np.where(small, 1 / 6 - th2 / 120, (ths - np.sin(ths)) / ths**3)
    return a, b, c


def exp_so3_b(w):
    """exp(w^) for an array of rotation vectors (..., 3)."""
    a, b, _ = _coef(w)
    W = hat_b(w)
    return np.eye(3) + a * W + b * W @ W


def jl_b(w):
    """Left Jacobian J(w): exp((w)^ on SE_K(3)) has linear columns J(w_R) w_lin."""
    _, b, c = _coef(w)
    W = hat_b(w)
    return np.eye(3) + b * W + c * W @ W


def log_so3(R):
    """Rotation vector of R (angle in [0, pi])."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-7:
        return vee(R - R.T) / 2
    if np.pi - th < 1e-5:
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        a = A[:, i] / np.sqrt(A[i, i])
        s = vee(R - R.T)
        if s @ a < 0:
            a = -a
        return th * a
    return th / (2 * np.sin(th)) * vee(R - R.T)


def log_so3_b(R):
    """Rotation vectors of a batch (..., 3, 3). Near half a turn it falls back to log_so3."""
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    th = angle_b(R)
    s = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    small = th < 1e-7
    f = np.where(small, 0.5, th / (2 * np.sin(np.where(small, 1.0, th))))
    w = f[:, None] * s
    for i in np.nonzero(np.pi - th < 1e-4)[0]:
        w[i] = log_so3(R[i])
    return w.reshape(shape + (3,))


def angle_b(R):
    """Rotation angle of each matrix in a batch, in radians."""
    return np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


# --------------------------------------------- SE_K(3) as (K+3)x(K+3) matrices
def sek_mat(R, cols):
    """X = [R c_1 ... c_K; 0 I]."""
    K = len(cols)
    X = np.eye(3 + K)
    X[:3, :3] = R
    for i, c in enumerate(cols):
        X[:3, 3 + i] = c
    return X


def sek_hat(chi):
    """chi = (lin_1, ..., lin_K, rot) -> matrix of the Lie algebra."""
    K = len(chi) // 3 - 1
    M = np.zeros((3 + K, 3 + K))
    M[:3, :3] = hat(chi[3 * K:])
    for i in range(K):
        M[:3, 3 + i] = chi[3 * i:3 * i + 3]
    return M


def sek_vee(M):
    K = M.shape[0] - 3
    return np.concatenate([M[:3, 3 + i] for i in range(K)] + [vee(M[:3, :3])])


def sek_exp(chi):
    K = len(chi) // 3 - 1
    w = chi[3 * K:]
    J = jl_b(w)
    return sek_mat(exp_so3_b(w), [J @ chi[3 * i:3 * i + 3] for i in range(K)])


def sek_log(X):
    K = X.shape[0] - 3
    w = log_so3(X[:3, :3])
    Ji = np.linalg.inv(jl_b(w))
    return np.concatenate([Ji @ X[:3, 3 + i] for i in range(K)] + [w])


def sek_inv(X):
    K = X.shape[0] - 3
    R = X[:3, :3]
    return sek_mat(R.T, [-R.T @ X[:3, 3 + i] for i in range(K)])


def sek_Ad(X):
    """Adjoint matrix: X chi^ X^{-1} = (Ad_X chi)^, in the ordering (lin..., rot)."""
    K = X.shape[0] - 3
    R = X[:3, :3]
    A = np.zeros((3 * K + 3, 3 * K + 3))
    for i in range(K + 1):
        A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    for i in range(K):
        A[3 * i:3 * i + 3, 3 * K:] = hat(X[:3, 3 + i]) @ R
    return A


def Ad_b(R, cols):
    """Batched adjoint of SE_K(3) from R (N,3,3) and a list of K columns (N,3)."""
    K = len(cols)
    N = R.shape[0]
    A = np.zeros((N, 3 * K + 3, 3 * K + 3))
    for i in range(K + 1):
        A[:, 3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    for i, c in enumerate(cols):
        A[:, 3 * i:3 * i + 3, 3 * K:] = hat_b(c) @ R
    return A


# ------------------------------------------------- IMU kinematics on SE_2(3)
def imu_f(X, Om, a_s):
    """Continuous dynamics f_u(X) of [R v x]: R' = R Om^, v' = R a_s - g e3, x' = v."""
    R, v = X[:3, :3], X[:3, 3]
    F = np.zeros((5, 5))
    F[:3, :3] = R @ hat(Om)
    F[:3, 3] = R @ a_s - G * E3
    F[:3, 4] = v
    return F


def imu_step(R, v, x, Om, a_s, h):
    """One step with the inputs held: batched over the leading axis."""
    acc = np.einsum("...ij,...j->...i", R, a_s) - G * E3
    return R @ exp_so3_b(Om * h), v + acc * h, x + v * h + 0.5 * acc * h * h


def A_right(K=2):
    """Matrix of the log of the right-invariant error of IMU kinematics, order (v, x, [c], R)."""
    n = 3 * K + 3
    A = np.zeros((n, n))
    A[0:3, 3 * K:] = -G * hat(E3)      # chi_v' = -g e3 x chi_R
    A[3:6, 0:3] = np.eye(3)            # chi_x' = chi_v
    return A


def A_left(Om, a_s):
    """Matrix of the log of the left-invariant error: depends on the inputs, not on the state."""
    A = np.zeros((9, 9))
    for i in range(3):
        A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = -hat(Om)
    A[0:3, 6:9] = -hat(a_s)
    A[3:6, 0:3] = np.eye(3)
    return A


def Phi_right(h, K=2):
    A = A_right(K)
    return np.eye(A.shape[0]) + A * h + A @ A * (h * h / 2)     # A^3 = 0


# -------------------------------------------- navigation scenario (worked case)
class Scenario:
    """A vehicle flying a banked figure with an IMU, seeing known landmarks in its body frame."""

    def __init__(self, T=20.0, h=0.01, every=10,
                 sig_g=0.01, sig_a=0.1, sig_y=0.1):
        self.h, self.T, self.every = h, T, every
        self.n = int(round(T / h))
        self.sig_g, self.sig_a, self.sig_y = sig_g, sig_a, sig_y
        self.landmarks = np.array([[3.5, 0.5, 0.0], [-2.5, 3.0, 0.0], [-1.5, -3.5, 0.8]])
        t = np.arange(self.n + 1) * h
        w1 = 2 * np.pi / 10.0
        pos = np.stack([2.5 * np.cos(w1 * t), 2.0 * np.sin(w1 * t), 1.5 + 0.5 * np.sin(2 * w1 * t)], -1)
        vel = np.stack([-2.5 * w1 * np.sin(w1 * t), 2.0 * w1 * np.cos(w1 * t), w1 * np.cos(2 * w1 * t)], -1)
        acc = np.stack([-2.5 * w1**2 * np.cos(w1 * t), -2.0 * w1**2 * np.sin(w1 * t),
                        -2 * w1**2 * np.sin(2 * w1 * t)], -1)
        Rs = np.zeros((self.n + 1, 3, 3))
        for k in range(self.n + 1):
            b3 = unit(acc[k] + G * E3)                 # thrust axis of a multicopter
            hd = np.array([vel[k, 0], vel[k, 1], 0.0])
            b2 = unit(np.cross(b3, hd))
            Rs[k] = np.column_stack([np.cross(b2, b3), b2, b3])
        self.Om = np.array([log_so3(Rs[k].T @ Rs[k + 1]) / h for k in range(self.n)])
        self.a_s = np.einsum("kji,kj->ki", Rs[:-1], acc[:-1] + G * E3)
        # the truth is the discrete model driven by these inputs
        R, v, x = Rs[0], vel[0], pos[0]
        self.R, self.v, self.x = [R], [v], [x]
        for k in range(self.n):
            R, v, x = imu_step(R, v, x, self.Om[k], self.a_s[k], h)
            self.R.append(R), self.v.append(v), self.x.append(x)
        self.R, self.v, self.x = np.array(self.R), np.array(self.v), np.array(self.x)
        self.t = t

    def meas(self, k):
        """Noise-free landmark positions in the body frame at step k, shape (L, 3)."""
        return (self.landmarks - self.x[k]) @ self.R[k]


def initial_estimates(sc, rng, N, ang_deg, sig_v, sig_x, fixed_angle=False):
    """Estimates at k = 0: Rhat = R exp(dtheta^), vhat = v + dv, xhat = x + dx."""
    if fixed_angle:
        dth = np.radians(ang_deg) * np.array([unit(a) for a in rng.normal(size=(N, 3))])
    else:
        dth = rng.normal(size=(N, 3)) * np.radians(ang_deg)
    R = sc.R[0] @ exp_so3_b(dth)
    v = sc.v[0] + rng.normal(size=(N, 3)) * sig_v
    x = sc.x[0] + rng.normal(size=(N, 3)) * sig_x
    return R, v, x


def sigma0_std(N, ang_deg, sig_v, sig_x):
    d = np.concatenate([np.full(3, sig_v**2), np.full(3, sig_x**2), np.full(3, np.radians(ang_deg)**2)])
    return np.tile(np.diag(d), (N, 1, 1))


def run_filters(sc, rng, N, ang_deg, sig_v, sig_x, which=("inv", "std"), fixed_angle=False,
                cov_deg=None, keep=False, side="left"):
    """Monte Carlo of the right-invariant EKF and the conventional EKF on one scenario.

    Both filters get the same initial estimate, the same initial covariance (written in
    each filter's own error coordinates) and the same noisy readings.
    Returns, per filter, arrays over (step, run): attitude error [rad], position error,
    velocity error, and the normalized estimation error squared (NEES).
    """
    h = sc.h
    R0, v0, x0 = initial_estimates(sc, rng, N, ang_deg, sig_v, sig_x, fixed_angle)
    S0 = sigma0_std(N, cov_deg or ang_deg, sig_v, sig_x)
    Qd = np.zeros((9, 9))
    Qd[0:3, 0:3] = np.eye(3) * (sc.sig_a * h) ** 2
    Qd[6:9, 6:9] = np.eye(3) * (sc.sig_g * h) ** 2
    Ny = np.eye(3 * len(sc.landmarks)) * sc.sig_y**2
    Phi = Phi_right(h)
    Cy = np.concatenate([np.hstack([np.zeros((3, 3)), np.eye(3), -hat(p)]) for p in sc.landmarks])
    I9 = np.eye(9)
    st = {}
    if "inv" in which:
        T0 = Ad_b(R0, [v0, x0])                      # chi = T delta at the estimate
        T0[:, 0:3, 0:3] = np.eye(3)
        T0[:, 3:6, 3:6] = np.eye(3)
        st["inv"] = [R0.copy(), v0.copy(), x0.copy(), T0 @ S0 @ np.swapaxes(T0, 1, 2)]
    if "std" in which:
        st["std"] = [R0.copy(), v0.copy(), x0.copy(), S0.copy()]
    out = {f: dict(att=[], pos=[], vel=[], nees=[]) for f in st}
    hist = {f: [] for f in st} if keep else None
    gn = rng.normal(size=(sc.n, N, 3)) * sc.sig_g
    an = rng.normal(size=(sc.n, N, 3)) * sc.sig_a
    L = len(sc.landmarks)

    def log_state(k):
        Rt, vt, xt = sc.R[k], sc.v[k], sc.x[k]
        for f, (R, v, x, S) in st.items():
            ER = R @ Rt.T
            out[f]["att"].append(angle_b(ER))
            out[f]["pos"].append(np.linalg.norm(x - xt, axis=1))
            out[f]["vel"].append(np.linalg.norm(v - vt, axis=1))
            if f == "inv":
                w = log_so3_b(ER)
                Ji = np.linalg.inv(jl_b(w))
                e = np.concatenate([np.einsum("nij,nj->ni", Ji, v - ER @ vt),
                                    np.einsum("nij,nj->ni", Ji, x - ER @ xt), w], axis=1)
            else:
                e = np.concatenate([v - vt, x - xt, log_so3_b(Rt.T @ R)], axis=1)
            out[f]["nees"].append(np.einsum("ni,ni->n", e, np.linalg.solve(S, e[..., None])[..., 0]) / 9)
            if keep:
                hist[f].append((R.copy(), v.copy(), x.copy(), S.copy()))

    log_state(0)
    for k in range(sc.n):
        Om = sc.Om[k] + gn[k]
        a_s = sc.a_s[k] + an[k]
        if "inv" in st:
            R, v, x, S = st["inv"]
            Ad = Ad_b(R, [v, x])
            S = Phi @ (S + Ad @ Qd @ np.swapaxes(Ad, 1, 2)) @ Phi.T
            R, v, x = imu_step(R, v, x, Om, a_s, h)
            st["inv"] = [R, v, x, S]
        if "std" in st:
            R, v, x, S = st["std"]
            F = np.tile(I9, (N, 1, 1))
            Ra = -R @ hat_b(a_s)
            F[:, 0:3, 6:9] = Ra * h
            F[:, 3:6, 0:3] = np.eye(3) * h
            F[:, 3:6, 6:9] = Ra * (h * h / 2)
            F[:, 6:9, 6:9] = exp_so3_b(-Om * h)
            Gn = np.zeros((N, 9, 9))
            Gn[:, 0:3, 0:3] = R
            Gn[:, 6:9, 6:9] = np.eye(3)
            S = F @ (S + Gn @ Qd @ np.swapaxes(Gn, 1, 2)) @ np.swapaxes(F, 1, 2)
            R, v, x = imu_step(R, v, x, Om, a_s, h)
            st["std"] = [R, v, x, S]
        if (k + 1) % sc.every == 0:
            y = sc.meas(k + 1)[None] + rng.normal(size=(N, L, 3)) * sc.sig_y        # (N, L, 3)
            if "inv" in st:
                R, v, x, S = st["inv"]
                yt = (np.einsum("nij,nlj->nli", R, y) + x[:, None, :] - sc.landmarks).reshape(N, -1)
                Nr = Ny                                   # isotropic noise: R Ny R^T = Ny
                Sy = Cy @ S @ Cy.T + Nr
                Kt = np.swapaxes(np.linalg.solve(Sy, Cy @ S), 1, 2)
                d = -np.einsum("nij,nj->ni", Kt, yt)
                E = exp_so3_b(d[:, 6:9])
                J = jl_b(d[:, 6:9])
                if side == "left":                        # exp(d^) Xhat: the side of the error
                    v = np.einsum("nij,nj->ni", E, v) + np.einsum("nij,nj->ni", J, d[:, 0:3])
                    x = np.einsum("nij,nj->ni", E, x) + np.einsum("nij,nj->ni", J, d[:, 3:6])
                    R = E @ R
                else:                                     # Xhat exp(d^): the wrong side (Pitfall)
                    v = v + np.einsum("nij,njk,nk->ni", R, J, d[:, 0:3])
                    x = x + np.einsum("nij,njk,nk->ni", R, J, d[:, 3:6])
                    R = R @ E
                IK = I9 - Kt @ Cy
                S = IK @ S @ np.swapaxes(IK, 1, 2) + Kt @ Nr @ np.swapaxes(Kt, 1, 2)
                st["inv"] = [R, v, x, S]
            if "std" in st:
                R, v, x, S = st["std"]
                yh = np.einsum("nji,nlj->nli", R, sc.landmarks[None] - x[:, None, :])   # R^T (p - x)
                yt = (yh - y).reshape(N, -1)
                H = np.zeros((N, 3 * L, 9))
                for i in range(L):
                    H[:, 3 * i:3 * i + 3, 3:6] = -np.swapaxes(R, 1, 2)
                    H[:, 3 * i:3 * i + 3, 6:9] = hat_b(yh[:, i])
                Sy = H @ S @ np.swapaxes(H, 1, 2) + Ny
                Kt = np.swapaxes(np.linalg.solve(Sy, H @ S), 1, 2)
                d = -np.einsum("nij,nj->ni", Kt, yt)
                v = v + d[:, 0:3]
                x = x + d[:, 3:6]
                R = R @ exp_so3_b(d[:, 6:9])
                IK = I9 - Kt @ H
                S = IK @ S @ np.swapaxes(IK, 1, 2) + Kt @ Ny @ np.swapaxes(Kt, 1, 2)
                st["std"] = [R, v, x, S]
        log_state(k + 1)
    res = {f: {key: np.array(val) for key, val in o.items()} for f, o in out.items()}
    if keep:
        res["hist"] = hist
    return res


# ------------------------------------------------------- heading on the circle
def compass_demo(seed=3, T=12.0, h=0.02, every=10, sig_g=np.radians(0.5), sig_y=np.radians(3.0)):
    """A gyroscope and a compass. Truth crosses the wrap at 180 degrees.

    Returns time, true heading, and the estimates of the filter that subtracts
    and of the filter that takes the logarithm on the circle (wraps).
    """
    rng = np.random.default_rng(seed)
    n = int(round(T / h))
    rate = np.radians(4.0)
    t = np.arange(n + 1) * h
    truth = np.radians(170.0) + rate * t
    est = {"subtract": truth[0], "wrap": truth[0]}
    var = {"subtract": sig_y**2, "wrap": sig_y**2}
    hist = {f: [est[f]] for f in est}
    for k in range(n):
        om = rate + rng.normal() * sig_g
        y = wrap(truth[k + 1] + rng.normal() * sig_y) if (k + 1) % every == 0 else None
        for f in est:
            est[f] += om * h
            var[f] += (sig_g * h) ** 2
            if y is not None:
                innov = est[f] - y if f == "subtract" else wrap(est[f] - y)
                gain = var[f] / (var[f] + sig_y**2)
                est[f] -= gain * innov
                var[f] *= 1 - gain
            hist[f].append(est[f])
    return t, truth, {f: np.array(v) for f, v in hist.items()}


# ---------------------------------------------------- dead reckoning on SE(2)
def se2_exp(z):
    """exp of z = (V1, V2, Om) (N, 3) as (theta, x, y) increments in the body frame."""
    th = z[:, 2]
    small = np.abs(th) < 1e-9
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1 - th**2 / 6, np.sin(ths) / ths)
    b = np.where(small, th / 2, (1 - np.cos(ths)) / ths)
    return th, a * z[:, 0] - b * z[:, 1], b * z[:, 0] + a * z[:, 1]


def se2_log(th, x, y):
    th = wrap(th)
    small = np.abs(th) < 1e-9
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1 - th**2 / 6, np.sin(ths) / ths)
    b = np.where(small, th / 2, (1 - np.cos(ths)) / ths)
    det = a * a + b * b
    return np.stack([(a * x + b * y) / det, (-b * x + a * y) / det, th], -1)


def se2_mul(p1, p2):
    th1, x1, y1 = p1
    th2, x2, y2 = p2
    c, s = np.cos(th1), np.sin(th1)
    return th1 + th2, x1 + c * x2 - s * y2, y1 + s * x2 + c * y2


def se2_inv(p):
    th, x, y = p
    c, s = np.cos(th), np.sin(th)
    return -th, -(c * x + s * y), -(-s * x + c * y)


def se2_Ad(th, x, y):
    """Adjoint on (V1, V2, Om) of the pose (th, x, y)."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, y], [s, c, -x], [0, 0, 1.0]])


def banana(N=4000, seed=5, T=10.0, h=0.1, speed=1.0, turn=0.0,
           sig_V=0.05, sig_Om=0.12, sig0=(0.0, 0.0, 0.0), noisy=True):
    """Dead reckoning of a planar robot: X_{k+1} = X_k exp((zeta + n_k) h).

    Returns the sample poses at every step, the noise-free pose, the logarithm
    chi = log(Xbar^{-1} X) of every sample and the covariance propagated by the
    linear equation of the left-invariant error.
    """
    rng = np.random.default_rng(seed)
    n = int(round(T / h))
    zeta = np.array([speed, 0.0, turn])
    chi0 = rng.normal(size=(N, 3)) * np.asarray(sig0)
    X = se2_exp(chi0)
    Xbar = (np.zeros(1), np.zeros(1), np.zeros(1))
    S = np.diag(np.asarray(sig0, float) ** 2)
    Qd = np.diag([sig_V**2, (0.2 * sig_V) ** 2, sig_Om**2]) * h if noisy else np.zeros((3, 3))
    step_bar = se2_exp(zeta[None] * h)
    Phi = se2_Ad(*[c[0] for c in se2_inv(step_bar)])        # Ad of exp(-zeta h)
    poses, chis, covs, bars = [], [], [], []
    for k in range(n + 1):
        poses.append(np.stack(X, -1))
        bars.append(np.array([c[0] for c in Xbar]))
        chis.append(se2_log(*se2_mul(se2_inv(Xbar), X)))
        covs.append(S.copy())
        if k == n:
            break
        noise = rng.multivariate_normal(np.zeros(3), Qd / h**2, size=N) if noisy else np.zeros((N, 3))
        X = se2_mul(X, se2_exp((zeta[None] + noise) * h))
        Xbar = se2_mul(Xbar, step_bar)
        S = Phi @ S @ Phi.T + Qd
    return dict(poses=np.array(poses), bar=np.array(bars), chi=np.array(chis), cov=np.array(covs),
                h=h, Phi=Phi)


def skewness(x):
    x = x - x.mean()
    return float(np.mean(x**3) / np.mean(x**2) ** 1.5)


# ------------------------------------------- contact-aided filter (simplified)
def contact_demo(N=100, seed=11, T=6.0, h=0.005, sig_g=0.002, sig_a=0.04, sig_c=0.05, sig_y=0.01,
                 tilt_deg=20.0, yaw_deg=30.0, sig_v=1.0, sig_x=0.1):
    """A body with an IMU standing on one fixed contact point: a simplified demonstration.

    State [R v x x_c] in SE_3(3), algebra order (v, x, c, R). The measurement is the
    contact point seen from the body, y = R^T (x_c - x) + noise, as leg kinematics gives it.
    Returns time and, per run, tilt error, yaw error, velocity error and position error.
    """
    rng = np.random.default_rng(seed)
    n = int(round(T / h))
    t = np.arange(n + 1) * h
    pos = np.stack([0.10 * np.sin(2.0 * t), 0.06 * np.sin(3.1 * t), 0.90 + 0.04 * np.sin(4.2 * t)], -1)
    vel = np.stack([0.20 * np.cos(2.0 * t), 0.186 * np.cos(3.1 * t), 0.168 * np.cos(4.2 * t)], -1)
    acc = np.stack([-0.40 * np.sin(2.0 * t), -0.5766 * np.sin(3.1 * t), -0.7056 * np.sin(4.2 * t)], -1)
    Rs = exp_so3_b(np.stack([0.15 * np.sin(1.7 * t), 0.20 * np.sin(2.3 * t), 0.10 * np.sin(1.1 * t)], -1))
    Om = np.array([log_so3(Rs[k].T @ Rs[k + 1]) / h for k in range(n)])
    a_s = np.einsum("kji,kj->ki", Rs[:-1], acc[:-1] + G * E3)
    foot = np.array([0.05, -0.12, 0.0])
    Rt, vt, xt = Rs[0], vel[0], pos[0]
    # initial errors, drawn in right-invariant coordinates: tilt, yaw, velocity, positions
    chi0 = np.zeros((N, 12))
    chi0[:, 0:3] = rng.normal(size=(N, 3)) * sig_v
    chi0[:, 3:6] = rng.normal(size=(N, 3)) * sig_x
    chi0[:, 6:9] = rng.normal(size=(N, 3)) * sig_x
    ax = rng.normal(size=(N, 2))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    chi0[:, 9:11] = ax * np.radians(tilt_deg)
    chi0[:, 11] = np.radians(yaw_deg) * rng.choice([-1.0, 1.0], size=N)
    E = exp_so3_b(chi0[:, 9:12])
    J = jl_b(chi0[:, 9:12])
    R = E @ Rt
    v = E @ vt + np.einsum("nij,nj->ni", J, chi0[:, 0:3])
    x = E @ xt + np.einsum("nij,nj->ni", J, chi0[:, 3:6])
    c = E @ foot + np.einsum("nij,nj->ni", J, chi0[:, 6:9])
    S = np.tile(np.diag(np.concatenate([np.full(3, sig_v**2), np.full(6, sig_x**2),
                                        np.full(3, np.radians(max(tilt_deg, yaw_deg))**2)])), (N, 1, 1))
    Qd = np.zeros((12, 12))
    Qd[0:3, 0:3] = np.eye(3) * (sig_a * h) ** 2
    Qd[6:9, 6:9] = np.eye(3) * (sig_c * h) ** 2
    Qd[9:12, 9:12] = np.eye(3) * (sig_g * h) ** 2
    Phi = Phi_right(h, K=3)
    Cy = np.hstack([np.zeros((3, 3)), np.eye(3), -np.eye(3), np.zeros((3, 3))])
    Ny = np.eye(3) * sig_y**2
    I12 = np.eye(12)
    out = dict(tilt=[], yaw=[], vel=[], pos=[], rel=[])

    def record():
        ER = R @ Rt.T
        out["tilt"].append(np.arccos(np.clip(ER[:, 2, 2], -1, 1)))          # angle between E_R e3 and e3
        w = log_so3_b(ER)
        out["yaw"].append(np.abs(w[:, 2]))
        out["vel"].append(np.linalg.norm(np.einsum("nji,nj->ni", R, v) - Rt.T @ vt, axis=1))   # body-frame velocity
        out["pos"].append(np.linalg.norm(x - xt, axis=1))
        out["rel"].append(np.linalg.norm((c - x) - (foot - xt), axis=1))

    record()
    for k in range(n):
        Ad = Ad_b(R, [v, x, c])
        S = Phi @ (S + Ad @ Qd @ np.swapaxes(Ad, 1, 2)) @ Phi.T
        R, v, x = imu_step(R, v, x, Om[k] + rng.normal(size=(N, 3)) * sig_g,
                           a_s[k] + rng.normal(size=(N, 3)) * sig_a, h)
        Rt, vt, xt = imu_step(Rt, vt, xt, Om[k], a_s[k], h)
        y = (foot - xt) @ Rt + rng.normal(size=(N, 3)) * sig_y
        yt = np.einsum("nij,nj->ni", R, y) - (c - x)
        Kt = np.swapaxes(np.linalg.solve(Cy @ S @ Cy.T + Ny, Cy @ S), 1, 2)
        d = -np.einsum("nij,nj->ni", Kt, yt)
        E = exp_so3_b(d[:, 9:12])
        J = jl_b(d[:, 9:12])
        v = np.einsum("nij,nj->ni", E, v) + np.einsum("nij,nj->ni", J, d[:, 0:3])
        x = np.einsum("nij,nj->ni", E, x) + np.einsum("nij,nj->ni", J, d[:, 3:6])
        c = np.einsum("nij,nj->ni", E, c) + np.einsum("nij,nj->ni", J, d[:, 6:9])
        R = E @ R
        IK = I12 - Kt @ Cy
        S = IK @ S @ np.swapaxes(IK, 1, 2) + Kt @ Ny @ np.swapaxes(Kt, 1, 2)
        record()
    return t, {key: np.array(val) for key, val in out.items()}, (Phi, Cy)


# ------------------------------------------------------- the sphere: S^2 filter
L_PEND = 1.0


def pend_f(q, om, l=L_PEND):
    return np.cross(om, q), -(G / l) * np.cross(q, E3)


def pend_step(q, om, h, l=L_PEND):
    """RK4 step of the spherical pendulum, put back on the sphere and its tangent plane."""
    def f(z):
        a, b = pend_f(z[:3], z[3:], l)
        return np.concatenate([a, b])
    z = np.concatenate([q, om])
    k1 = f(z)
    k2 = f(z + h / 2 * k1)
    k3 = f(z + h / 2 * k2)
    k4 = f(z + h * k3)
    z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    q = unit(z[:3])
    return q, z[3:] - q * (q @ z[3:])


def expm_series(M):
    """Matrix exponential to third order, for a small step."""
    M2 = M @ M
    return np.eye(M.shape[0]) + M + M2 / 2 + M2 @ M / 6


def s2_A(q, om, l=L_PEND):
    """Variation-based linearization of the free pendulum: the matrix A of the chapter on linearization."""
    A = np.zeros((6, 6))
    A[:3, :3] = hat(om)
    A[:3, 3:] = np.eye(3) - np.outer(q, q)
    A[3:, :3] = -(G / l) * hat(E3) @ hat(q)
    return A


def s2_C(q, om):
    """Constraint matrix: the linear state s = (xi, delta omega) satisfies C s = 0."""
    C = np.zeros((2, 6))
    C[0, :3] = q
    C[1, :3] = np.cross(q, om)
    C[1, 3:] = q
    return C


def s2_E(q, om):
    """Projector onto ker C that keeps xi's tangent part and delta omega's tangent part."""
    Pq = np.eye(3) - np.outer(q, q)
    E = np.zeros((6, 6))
    E[:3, :3] = Pq
    E[3:, 3:] = Pq
    E[3:, :3] = -np.outer(q, np.cross(q, om))
    return E


def s2_oplus(q, om, s):
    """(q, om) (+) s: turn by exp(xi^) and carry the velocity along."""
    Pq = np.eye(3) - np.outer(q, q)
    Rs = expm_so3(Pq @ s[:3])
    return Rs @ q, Rs @ (om + Pq @ s[3:])


def s2_ominus(q2, om2, q1, om1):
    """The variation s with (q1, om1) (+) s = (q2, om2)."""
    ax = np.cross(q1, q2)
    na = np.linalg.norm(ax)
    xi = np.zeros(3) if na < 1e-14 else np.arctan2(na, q1 @ q2) * ax / na
    Rs = expm_so3(xi)
    dom_t = Rs.T @ om2 - om1
    return np.concatenate([xi, dom_t - (om1 @ np.cross(xi, q1)) * q1])


def s2_truth(T=10.0, h=0.01, seed=21, sig_proc=np.sqrt(1e-5)):
    rng = np.random.default_rng(seed)
    n = int(round(T / h))
    q, om = E1.copy(), np.array([0.0, 0.0, 1.2])
    qs, oms = [q], [om]
    for _ in range(n):
        q, om = pend_step(q, om, h)
        Pq = np.eye(3) - np.outer(q, q)
        q, om = s2_oplus(q, om, np.concatenate([Pq @ rng.normal(size=3), Pq @ rng.normal(size=3)]) * sig_proc)
        qs.append(q), oms.append(om)
    return np.arange(n + 1) * h, np.array(qs), np.array(oms)


def s2_filters(truth, seed, sig0=1.0, sig_y=np.sqrt(1e-3), q_proc=1e-5, l=L_PEND, which=("var", "naive"),
               keep_cov=False):
    """One run of the variation-based EKF and of the EKF on the six numbers (q, omega)."""
    t, qs, oms = truth
    h = t[1] - t[0]
    rng = np.random.default_rng(seed)
    Pq = np.eye(3) - np.outer(qs[0], qs[0])
    s0 = np.concatenate([Pq @ rng.normal(size=3), Pq @ rng.normal(size=3)]) * sig0
    q0, om0 = s2_oplus(qs[0], oms[0], s0)
    Sy = np.eye(3) * sig_y**2
    Qn = np.eye(6) * q_proc
    out = {}
    ys = l * qs + rng.normal(size=qs.shape) * sig_y
    if "var" in which:
        q, om = q0.copy(), om0.copy()
        E = s2_E(q, om)
        S = E @ (np.eye(6) * sig0**2) @ E.T
        rec = dict(q=[q], om=[om], nees=[], cmax=0.0, cov=[S])
        for k in range(len(t) - 1):
            Ak = expm_series(s2_A(q, om, l) * h)
            q, om = pend_step(q, om, h, l)
            E = s2_E(q, om)
            S = E @ (Ak @ S @ Ak.T + Qn) @ E.T
            Cy = np.hstack([-l * hat(q), np.zeros((3, 3))])
            yt = l * q - ys[k + 1]
            Kt = S @ Cy.T @ np.linalg.inv(Cy @ S @ Cy.T + Sy)
            s = -Kt @ yt
            rec["cmax"] = max(rec["cmax"], float(np.abs(s2_C(q, om) @ s).max()))
            IK = np.eye(6) - Kt @ Cy
            S = IK @ S @ IK.T + Kt @ Sy @ Kt.T
            Rs = expm_so3(s[:3])
            q, om = s2_oplus(q, om, s)
            Tm = np.zeros((6, 6))
            Tm[:3, :3] = Rs
            Tm[3:, 3:] = Rs
            E = s2_E(q, om)
            S = E @ Tm @ S @ Tm.T @ E.T
            rec["cmax"] = max(rec["cmax"], float(np.abs(s2_C(q, om) @ S).max()))
            err = s2_ominus(q, om, qs[k + 1], oms[k + 1])
            rec["nees"].append(float(err @ np.linalg.pinv(S, rcond=1e-10) @ err) / 4)
            rec["q"].append(q), rec["om"].append(om)
            if keep_cov:
                rec["cov"].append(S)
        out["var"] = {key: (np.array(val) if key != "cmax" else val) for key, val in rec.items()}
    if "naive" in which:
        q, om = q0.copy(), om0.copy()
        S = np.eye(6) * sig0**2
        H = np.hstack([l * np.eye(3), np.zeros((3, 3))])
        rec = dict(q=[q], om=[om])
        for k in range(len(t) - 1):
            F = np.zeros((6, 6))
            F[:3, :3] = hat(om)
            F[:3, 3:] = -hat(q)
            F[3:, :3] = (G / l) * hat(E3)
            Fk = expm_series(F * h)
            z = np.concatenate([q, om])

            def f(z):
                a, b = pend_f(z[:3], z[3:], l)
                return np.concatenate([a, b])
            k1 = f(z)
            k2 = f(z + h / 2 * k1)
            k3 = f(z + h / 2 * k2)
            k4 = f(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            S = Fk @ S @ Fk.T + Qn
            Kt = S @ H.T @ np.linalg.inv(H @ S @ H.T + Sy)
            z = z - Kt @ (l * z[:3] - ys[k + 1])
            IK = np.eye(6) - Kt @ H
            S = IK @ S @ IK.T + Kt @ Sy @ Kt.T
            q, om = z[:3], z[3:]
            rec["q"].append(q), rec["om"].append(om)
        out["naive"] = {key: np.array(val) for key, val in rec.items()}
    return out


# ------------------------------------------------ examples used by the chapter
def two_trajectories(T=3.0, h=0.01):
    """The same initial error on two different flights: an aggressive figure and a hover."""
    sc = Scenario(T=T, h=h)
    n = sc.n
    chi0 = np.concatenate([[0.8, -0.5, 0.3], [1.0, 0.5, -0.5], 2.0 * unit([1.0, 2.0, -1.0])])
    E0 = sek_exp(chi0)
    flights = {
        "figure": (sc.R[0], sc.v[0], sc.x[0], sc.Om, sc.a_s),
        "hover": (expm_so3(1.0 * E3), np.zeros(3), np.array([5.0, -2.0, 3.0]),
                  np.zeros((n, 3)), np.tile(G * E3, (n, 1))),
    }
    Phi = Phi_right(h)
    out = {"t": np.arange(n + 1) * h, "chi0": chi0}
    for name, (R, v, x, Om, a_s) in flights.items():
        Xh = E0 @ sek_mat(R, [v, x])
        Rh, vh, xh = Xh[:3, :3], Xh[:3, 3], Xh[:3, 4]
        # the conventional error (dv, dx, dtheta) and its linearized prediction
        d = np.concatenate([vh - v, xh - x, log_so3(R.T @ Rh)])
        Er, add_v, lin_v, chi_lin = [], [], [], []
        chi = chi0.copy()
        for k in range(n + 1):
            Er.append(sek_mat(Rh, [vh, xh]) @ sek_inv(sek_mat(R, [v, x])))
            add_v.append(vh - v)
            lin_v.append(d[0:3].copy())
            chi_lin.append(chi.copy())
            if k == n:
                break
            F = np.eye(9)
            Ra = -Rh @ hat(a_s[k])
            F[0:3, 6:9] = Ra * h
            F[3:6, 0:3] = np.eye(3) * h
            F[3:6, 6:9] = Ra * (h * h / 2)
            F[6:9, 6:9] = expm_so3(-Om[k] * h)
            d = F @ d
            chi = Phi @ chi
            R, v, x = imu_step(R, v, x, Om[k], a_s[k], h)
            Rh, vh, xh = imu_step(Rh, vh, xh, Om[k], a_s[k], h)
        out[name] = dict(Er=np.array(Er), add_v=np.array(add_v), lin_v=np.array(lin_v), chi=np.array(chi_lin))
    return out


def dead_reckoning_cov(N=4000, T=4.0, seed=7):
    """Sample covariance of the right-invariant error of noisy dead reckoning, against the propagated one."""
    sc = Scenario(T=T)
    rng = np.random.default_rng(seed)
    h = sc.h
    R = np.tile(sc.R[0], (N, 1, 1))
    v = np.tile(sc.v[0], (N, 1))
    x = np.tile(sc.x[0], (N, 1))
    Qd = np.zeros((9, 9))
    Qd[0:3, 0:3] = np.eye(3) * (sc.sig_a * h) ** 2
    Qd[6:9, 6:9] = np.eye(3) * (sc.sig_g * h) ** 2
    Phi = Phi_right(h)
    S = np.zeros((9, 9))
    for k in range(sc.n):
        Ad = sek_Ad(sek_mat(sc.R[k], [sc.v[k], sc.x[k]]))
        S = Phi @ (S + Ad @ Qd @ Ad.T) @ Phi.T
        R, v, x = imu_step(R, v, x, sc.Om[k] + rng.normal(size=(N, 3)) * sc.sig_g,
                           sc.a_s[k] + rng.normal(size=(N, 3)) * sc.sig_a, h)
    ER = R @ sc.R[-1].T
    w = log_so3_b(ER)
    Ji = np.linalg.inv(jl_b(w))
    chi = np.concatenate([np.einsum("nij,nj->ni", Ji, v - ER @ sc.v[-1]),
                          np.einsum("nij,nj->ni", Ji, x - ER @ sc.x[-1]), w], axis=1)
    return np.cov(chi.T), S, chi


def bias_rate(Xh, chi, zg, za, Om, a_s, dt=1e-6):
    """d/dt of the log of the right-invariant error when the estimate uses inputs off by a bias error."""
    X = sek_inv(sek_exp(chi)) @ Xh
    R, v, x = imu_step(X[:3, :3], X[:3, 3], X[:3, 4], Om, a_s, dt)
    Rh, vh, xh = imu_step(Xh[:3, :3], Xh[:3, 3], Xh[:3, 4], Om - zg, a_s - za, dt)
    chi1 = sek_log(sek_mat(Rh, [vh, xh]) @ sek_inv(sek_mat(R, [v, x])))
    return (chi1 - chi) / dt


def mc_table(N=200, seed=1):
    """The Monte Carlo study of the worked case: four sizes of initial attitude error."""
    sc = Scenario()
    rows = {}
    for ang, sv, sx in [(5, 0.2, 0.2), (60, 1.0, 1.0), (120, 1.0, 1.0), (170, 1.0, 1.0)]:
        r = run_filters(sc, np.random.default_rng(seed), N, ang, sv, sx, fixed_angle=True)
        row = {}
        for f in ("inv", "std"):
            a = np.degrees(r[f]["att"])
            bad = a >= 2.0
            last = np.where(bad.any(axis=0), a.shape[0] - 1 - np.argmax(bad[::-1], axis=0), 0)
            ts = (last + 1) * sc.h * bad.any(axis=0)
            row[f] = dict(att2=float(np.median(a[int(2 / sc.h)])), att5=float(np.median(a[int(5 / sc.h)])),
                          settle50=float(np.median(ts)), settle90=float(np.percentile(ts, 90)),
                          nees=float(np.mean(r[f]["nees"][-500:])),
                          unsettled=float(np.mean(a[-1] >= 2.0)),
                          att0=float(np.median(a[0])),
                          rms_pos=float(np.sqrt(np.mean(r[f]["pos"][-500:] ** 2))),
                          rms_att=float(np.degrees(np.sqrt(np.mean(r[f]["att"][-500:] ** 2)))),
                          curve50=np.median(a, axis=1), curve90=np.percentile(a, 90, axis=1))
        rows[ang] = row
    return sc, rows


def s2_mc(runs=100, T=5.0):
    """Many runs of the two filters on the pendulum from random initial states."""
    truth = s2_truth(T=T)
    t, qs, oms = truth
    stats = dict(var_norm=0.0, naive_norm_lo=1.0, naive_norm_hi=1.0, naive_norm_lo1=1.0, naive_norm_hi1=1.0,
                 cmax=0.0, var_tan=0.0, naive_tan=0.0)
    ang = {"var": [], "naive": []}
    nees, ang0 = [], []
    i1 = int(1.0 / (t[1] - t[0]))
    for seed in range(runs):
        r = s2_filters(truth, 100 + seed)
        for f in ("var", "naive"):
            nrm = np.linalg.norm(r[f]["q"], axis=1)
            ang[f].append(np.degrees(np.arccos(np.clip(np.sum(r[f]["q"] * qs, axis=1) / nrm, -1, 1))))
            tan = np.abs(np.sum(r[f]["q"] * r[f]["om"], axis=1)).max()
            if f == "var":
                stats["var_norm"] = max(stats["var_norm"], float(np.abs(nrm - 1).max()))
                stats["var_tan"] = max(stats["var_tan"], float(tan))
            else:
                stats["naive_norm_lo"] = min(stats["naive_norm_lo"], float(nrm.min()))
                stats["naive_norm_hi"] = max(stats["naive_norm_hi"], float(nrm.max()))
                stats["naive_norm_lo1"] = min(stats["naive_norm_lo1"], float(nrm[i1:].min()))
                stats["naive_norm_hi1"] = max(stats["naive_norm_hi1"], float(nrm[i1:].max()))
                stats["naive_tan"] = max(stats["naive_tan"], float(tan))
        stats["cmax"] = max(stats["cmax"], r["var"]["cmax"])
        nees.append(np.mean(r["var"]["nees"][i1:]))
        ang0.append(ang["var"][-1][0])
    stats["nees"] = float(np.mean(nees))
    stats["ang0_max"] = float(np.max(ang0))
    stats["ang0_med"] = float(np.median(ang0))
    return t, {f: np.array(v) for f, v in ang.items()}, stats, i1


# ----------------------------------------------------------------- the checks
def main():
    ck = Checker("ch14")
    rng = np.random.default_rng(14)

    # --- the group SE_2(3): exponential, logarithm, inverse, adjoint
    chi = rng.normal(size=9) * 0.8
    X = sek_exp(rng.normal(size=9))
    Y2 = sek_exp(rng.normal(size=9))
    ck.close("exp on SE_2(3) is the matrix exponential", sek_exp(chi), expm(sek_hat(chi)), 1e-12)
    ck.close("log inverts exp on SE_2(3)", sek_log(sek_exp(chi)), chi, 1e-10)
    ck.close("inverse on SE_2(3)", sek_inv(X) @ X, np.eye(5), 1e-12)
    ck.close("adjoint: X chi^ X^-1 = (Ad_X chi)^", X @ sek_hat(chi) @ sek_inv(X), sek_hat(sek_Ad(X) @ chi), 1e-12)
    ck.close("Ad_{XY} = Ad_X Ad_Y", sek_Ad(X @ Y2), sek_Ad(X) @ sek_Ad(Y2), 1e-11)
    ck.close("exp(chi^) X = X exp((Ad_{X^-1} chi)^)", sek_exp(chi) @ X, X @ sek_exp(sek_Ad(sek_inv(X)) @ chi), 1e-11)
    chi12 = rng.normal(size=12) * 0.7
    X12 = sek_exp(rng.normal(size=12))
    ck.close("exp and adjoint on SE_3(3)", X12 @ expm(sek_hat(chi12)) @ sek_inv(X12),
             sek_exp(sek_Ad(X12) @ chi12), 1e-11)
    wb = rng.normal(size=(5, 3))
    ck.close("batched exp, left Jacobian and log on SO(3)",
             [np.abs(exp_so3_b(wb)[i] - expm_so3(wb[i])).max() + np.abs(log_so3_b(exp_so3_b(wb))[i] - wb[i]).max()
              for i in range(5)], 0, 1e-10)

    # --- left and right errors and their invariance (def 3.x, thought experiment of the remount)
    Xh = sek_exp(rng.normal(size=9))
    Er = Xh @ sek_inv(X)
    El = sek_inv(X) @ Xh
    ck.close("right-invariant error ignores a right multiplication", (Xh @ Y2) @ sek_inv(X @ Y2), Er, 1e-11)
    ck.close("left-invariant error ignores a left multiplication", sek_inv(Y2 @ X) @ (Y2 @ Xh), El, 1e-11)
    ck.close("chi_r = Ad_X chi_l", sek_log(Er), sek_Ad(X) @ sek_log(El), 1e-9)

    # --- group-affine property of IMU kinematics, and a system that fails it
    Om, a_s = rng.normal(size=3), rng.normal(size=3) * 3
    f = lambda Z: imu_f(Z, Om, a_s)  # noqa: E731
    ck.close("IMU kinematics are group-affine", f(X @ Y2), f(X) @ Y2 + X @ f(Y2) - X @ f(np.eye(5)) @ Y2, 1e-10)

    def f_drag(Z):          # drag along the body's first axis: not group-affine
        F = imu_f(Z, Om, a_s)
        F[:3, 3] -= 0.5 * Z[:3, 0] * (Z[:3, 0] @ Z[:3, 3])
        return F
    gap = np.abs(f_drag(X @ Y2) - (f_drag(X) @ Y2 + X @ f_drag(Y2) - X @ f_drag(np.eye(5)) @ Y2)).max()
    ck.true("body-axis drag breaks the group-affine property", gap > 1e-2, f"gap {gap:.2f}")

    def f_lin(Z):           # linear drag: still group-affine
        F = imu_f(Z, Om, a_s)
        F[:3, 3] -= 0.5 * Z[:3, 3]
        return F
    ck.close("linear drag keeps the group-affine property",
             f_lin(X @ Y2), f_lin(X) @ Y2 + X @ f_lin(Y2) - X @ f_lin(np.eye(5)) @ Y2, 1e-10)
    # the error equations of the theorem, and the matrices A as their first-order parts
    Xt_ = sek_exp(rng.normal(size=9))
    Xh_ = sek_exp(rng.normal(size=9))
    Er_, El_ = Xh_ @ sek_inv(Xt_), sek_inv(Xt_) @ Xh_
    Xi_ = sek_inv(Xt_)
    ck.close("right-invariant error: rate is f(E) - E f(I)",
             f(Xh_) @ Xi_ - Xh_ @ Xi_ @ f(Xt_) @ Xi_, f(Er_) - Er_ @ f(np.eye(5)), 1e-9)
    ck.close("left-invariant error: rate is f(E) - f(I) E",
             -Xi_ @ f(Xt_) @ Xi_ @ Xh_ + Xi_ @ f(Xh_), f(El_) - f(np.eye(5)) @ El_, 1e-9)
    c1 = rng.normal(size=9)
    Ee = sek_exp(1e-6 * c1)
    ck.close("A of the right-invariant error is the first-order part of f(E) - E f(I)",
             sek_vee(f(Ee) - Ee @ f(np.eye(5))) / 1e-6, A_right() @ c1, 1e-4)
    ck.close("A of the left-invariant error is the first-order part of f(E) - f(I) E",
             sek_vee(f(Ee) - f(np.eye(5)) @ Ee) / 1e-6, A_left(Om, a_s) @ c1, 1e-4)

    # --- autonomous error and log-linear property (right-invariant, discrete model, large error)
    two = two_trajectories()
    dEr = np.abs(two["figure"]["Er"] - two["hover"]["Er"]).max()
    ck.close("same right-invariant error on two different flights", dEr, 0, 1e-10)
    ll = max(np.abs(two[f]["Er"][k] - sek_exp(two[f]["chi"][k])).max()
             for f in ("figure", "hover") for k in range(0, len(two["t"]), 10))
    ck.close("log-linear: E_r(t) = exp((Phi^k chi_0)^) at 2 rad of error", ll, 0, 1e-10)
    k2 = int(round(2.0 / 0.01))
    ER0 = two["figure"]["Er"][0][:3, :3]
    eta_v0 = two["figure"]["Er"][0][:3, 3]
    ck.close("closed form: eta_v(t) = eta_v(0) - g t (I - E_R) e3",
             two["figure"]["Er"][k2][:3, 3], eta_v0 - G * 2.0 * (np.eye(3) - ER0) @ E3, 1e-10)
    ck.close("A_right is nilpotent: A^3 = 0", np.linalg.matrix_power(A_right(), 3), 0, 1e-15)
    ck.close("Phi = exp(A h)", Phi_right(0.3), expm(A_right() * 0.3), 1e-12)
    ck.record("loglinear_error_deg", float(np.degrees(np.linalg.norm(two["chi0"][6:]))))
    ck.record("two_flights_Er_gap", float(dEr))
    ck.record("loglinear_gap", float(ll))
    for f in ("figure", "hover"):
        ck.record(f"additive_v_error_2s_{f}", float(np.linalg.norm(two[f]["add_v"][k2])))
    ck.record("additive_v_error_gap_2s", float(np.linalg.norm(two["figure"]["add_v"][k2] - two["hover"]["add_v"][k2])))
    lin_rel = np.linalg.norm(two["figure"]["lin_v"][k2] - two["figure"]["add_v"][k2]) / np.linalg.norm(two["figure"]["add_v"][k2])
    ck.record("std_linear_prediction_rel_error_2s", float(lin_rel))
    ck.true("the conventional linearized prediction is off at a large error", lin_rel > 0.05, f"{lin_rel:.2f}")
    tilt = np.radians(10.0)
    leak = G * np.linalg.norm((np.eye(3) - expm_so3(tilt * E1)) @ E3)
    ck.close("gravity leak at 10 degrees of tilt is 2 g sin(5 deg)", leak, 2 * G * np.sin(tilt / 2), 1e-12)
    ck.record("gravity_leak_10deg", float(leak))
    t_leak = np.sqrt(2 * 1.0 / (2 * G * np.sin(np.radians(0.5))))
    chi_t = expm(A_right() * t_leak) @ np.concatenate([np.zeros(6), np.radians(1.0) * E1])
    ck.close("a 1 degree tilt gives 1 m of position error after t_leak (to first order in the tilt)",
             np.linalg.norm(chi_t[3:6]), 1.0, 1e-3)
    ck.close("a yaw error leaks nothing", A_right() @ np.concatenate([np.zeros(6), E3]), 0, 1e-15)
    ck.record("leak_time_1deg_1m", float(t_leak))
    ck.record("loop_max_specific_force_g", float(np.linalg.norm(Scenario().a_s, axis=1).max() / G))

    # --- left-invariant error: continuous time, random smooth inputs
    cf = rng.normal(size=(2, 3, 3))

    def inputs(t):
        return (cf[0] @ np.array([np.sin(1.3 * t), np.cos(0.7 * t), 1.0]),
                G * E3 + 2 * cf[1] @ np.array([np.sin(0.9 * t), np.cos(1.9 * t), 0.5]))

    def rhs(t, z):
        Xa, Xb, c = z[:25].reshape(5, 5), z[25:50].reshape(5, 5), z[50:59]
        Om_t, a_t = inputs(t)
        return np.concatenate([imu_f(Xa, Om_t, a_t).ravel(), imu_f(Xb, Om_t, a_t).ravel(),
                               A_left(Om_t, a_t) @ c, A_right() @ z[59:]])
    X0 = sek_exp(rng.normal(size=9))
    c0 = np.concatenate([rng.normal(size=6), 1.8 * unit(rng.normal(size=3))])
    Xh0 = X0 @ sek_exp(c0)
    cr0 = sek_log(Xh0 @ sek_inv(X0))
    sol = solve_ivp(rhs, (0, 2.0), np.concatenate([X0.ravel(), Xh0.ravel(), c0, cr0]), rtol=1e-11, atol=1e-12)
    Xa, Xb = sol.y[:25, -1].reshape(5, 5), sol.y[25:50, -1].reshape(5, 5)
    ck.close("log-linear, left-invariant error, continuous time", sek_inv(Xa) @ Xb, sek_exp(sol.y[50:59, -1]), 1e-6)
    ck.close("log-linear, right-invariant error, continuous time", Xb @ sek_inv(Xa), sek_exp(sol.y[59:, -1]), 1e-6)

    # --- the noise and the bias bring the estimate back into the error equation
    Xh = sek_mat(random_rotation(rng), [rng.normal(size=3), rng.normal(size=3) * 3])
    c_small, zg, za = rng.normal(size=9) * 1e-4, rng.normal(size=3) * 1e-4, rng.normal(size=3) * 1e-4
    rate = bias_rate(Xh, c_small, zg, za, Om, a_s)
    pred = A_right() @ c_small - sek_Ad(Xh) @ np.concatenate([za, np.zeros(3), zg])
    ck.close("bias error enters as -Ad_Xhat (zeta_a, 0, zeta_g), to first order", rate, pred, 2e-7)
    Xh2 = sek_mat(random_rotation(rng), [rng.normal(size=3), rng.normal(size=3) * 3])
    ck.true("with a bias error the rate of the error depends on the estimate",
            np.linalg.norm(bias_rate(Xh2, c_small, zg, za, Om, a_s) - rate) > 0.3 * np.linalg.norm(rate))

    # --- measurement models: Jacobians by finite differences
    sc = Scenario(T=2.0)
    p = sc.landmarks[0]
    Xt = sek_mat(sc.R[50], [sc.v[50], sc.x[50]])
    y = sc.R[50].T @ (p - sc.x[50])
    dvec = rng.normal(size=9)
    eps = 1e-6
    Xe = sek_exp(eps * dvec) @ Xt
    yt = Xe[:3, :3] @ y + Xe[:3, 4] - p
    Cy = np.hstack([np.zeros((3, 3)), np.eye(3), -hat(p)])
    ck.close("landmark innovation: ytilde = C_y chi_r, C_y = [0 I -p^]", yt / eps, Cy @ dvec, 1e-5)
    d5 = np.concatenate([p, [0.0, 1.0]])
    ck.close("landmark reading is X^-1 d with d = (p, 0, 1)", (sek_inv(Xt) @ d5)[:3], y, 1e-12)
    ck.close("innovation is the top of Xhat Y - d", (Xe @ np.concatenate([y, [0.0, 1.0]]) - d5)[:3], yt, 1e-12)
    Rt, xt = sc.R[50], sc.x[50]
    Rh, xh = Rt @ expm_so3(eps * dvec[6:]), xt + eps * dvec[3:6]
    yh = Rh.T @ (p - xh)
    Hs = np.hstack([np.zeros((3, 3)), -Rh.T, hat(yh)])
    ck.close("conventional EKF: measurement Jacobian", (yh - y) / eps, Hs @ dvec, 1e-5)
    # conventional error propagation over one step
    vh = sc.v[50] + eps * dvec[0:3]
    R1, v1, x1 = imu_step(Rt, sc.v[50], xt, sc.Om[50], sc.a_s[50], sc.h)
    R1h, v1h, x1h = imu_step(Rh, vh, xh, sc.Om[50], sc.a_s[50], sc.h)
    F = np.eye(9)
    Ra = -Rh @ hat(sc.a_s[50])
    F[0:3, 6:9], F[3:6, 0:3], F[3:6, 6:9] = Ra * sc.h, np.eye(3) * sc.h, Ra * sc.h**2 / 2
    F[6:9, 6:9] = expm_so3(-sc.Om[50] * sc.h)
    ck.close("conventional EKF: error transition matrix",
             np.concatenate([v1h - v1, x1h - x1, log_so3(R1.T @ R1h)]) / eps, F @ dvec, 1e-5)
    # contact: forward kinematics as a right-invariant observation
    X4 = sek_exp(rng.normal(size=12))
    y4 = X4[:3, :3].T @ (X4[:3, 5] - X4[:3, 4])
    d6 = np.array([0, 0, 0, 0, 1.0, -1.0])
    ck.close("contact reading R^T (x_c - x) is X^-1 d, d = (0, 0, 1, -1)", (sek_inv(X4) @ d6)[:3], y4, 1e-12)
    d12 = rng.normal(size=12)
    X4e = sek_exp(eps * d12) @ X4
    Cc = np.hstack([np.zeros((3, 3)), np.eye(3), -np.eye(3), np.zeros((3, 3))])
    ck.close("contact innovation: C_y = [0 I -I 0]", (X4e[:3, :3] @ y4 - (X4e[:3, 5] - X4e[:3, 4])) / eps, Cc @ d12, 1e-5)

    # --- one update in components equals the update on the group
    S = np.diag(rng.uniform(0.1, 1.0, size=9))
    Xh = sek_exp(0.3 * rng.normal(size=9)) @ Xt
    yt = Xh[:3, :3] @ y + Xh[:3, 4] - p
    Kt = S @ Cy.T @ np.linalg.inv(Cy @ S @ Cy.T + 0.01 * np.eye(3))
    Xp = sek_exp(-Kt @ yt) @ Xh
    dlt = -Kt @ yt
    Ee, Je = expm_so3(dlt[6:]), jl_b(dlt[6:])
    ck.close("update by exp(-(K ytilde)^) Xhat, in components",
             sek_mat(Ee @ Xh[:3, :3], [Ee @ Xh[:3, 3] + Je @ dlt[0:3], Ee @ Xh[:3, 4] + Je @ dlt[3:6]]), Xp, 1e-12)
    ck.true("the update reduces the error", np.linalg.norm(sek_log(Xp @ sek_inv(Xt))) < np.linalg.norm(sek_log(Xh @ sek_inv(Xt))))

    # --- covariance propagation against Monte Carlo dead reckoning
    Sm, Sp, chis = dead_reckoning_cov()
    rel = np.linalg.norm(Sm - Sp) / np.linalg.norm(Sp)
    ratio = np.sqrt(np.diag(Sm) / np.diag(Sp))
    ck.true("propagated covariance matches the sample covariance of chi_r", rel < 0.08 and np.abs(ratio - 1).max() < 0.05,
            f"rel {rel:.3f}, std ratios {ratio.min():.3f}..{ratio.max():.3f}")
    ck.record("cov_mc_rel_error", float(rel))
    ck.record("cov_mc_std_ratio_min", float(ratio.min()))
    ck.record("cov_mc_std_ratio_max", float(ratio.max()))

    # --- Monte Carlo of the worked case
    sc, rows = mc_table()
    ck.record("mc_runs", 200)
    for ang, row in rows.items():
        for f in ("inv", "std"):
            for key in ("att2", "att5", "settle50", "settle90", "nees", "rms_pos", "rms_att", "unsettled"):
                ck.record(f"mc_{ang}_{f}_{key}", row[f][key])
    g = rows[5]
    ck.true("good start: the two filters perform alike",
            abs(g["inv"]["rms_att"] / g["std"]["rms_att"] - 1) < 0.02 and abs(g["inv"]["rms_pos"] / g["std"]["rms_pos"] - 1) < 0.02,
            f"rms att {g['inv']['rms_att']:.3f} vs {g['std']['rms_att']:.3f} deg")
    ck.true("good start: both filters are consistent (NEES near 1)",
            abs(g["inv"]["nees"] - 1) < 0.1 and abs(g["std"]["nees"] - 1) < 0.1, f"{g['inv']['nees']:.2f}, {g['std']['nees']:.2f}")
    for ang in (60, 120, 170):
        ck.true(f"poor start ({ang} deg): the invariant filter settles sooner",
                rows[ang]["inv"]["settle50"] < rows[ang]["std"]["settle50"] and rows[ang]["inv"]["settle90"] < rows[ang]["std"]["settle90"],
                f"{rows[ang]['inv']['settle50']:.2f} s vs {rows[ang]['std']['settle50']:.2f} s")
    ck.true("170 deg: the invariant filter is consistent at the end, the conventional one is not",
            rows[170]["inv"]["nees"] < 2 and rows[170]["std"]["nees"] > 10)
    # the Pitfall: a correction applied on the wrong side
    diverged = False
    try:
        with np.errstate(all="ignore"):
            r = run_filters(Scenario(T=5.0), np.random.default_rng(2), 50, 2, 0.2, 0.2, which=("inv",),
                            fixed_angle=True, side="right")
        diverged = not np.all(np.isfinite(r["inv"]["pos"][-1])) or np.median(r["inv"]["pos"][-1]) > 10
    except np.linalg.LinAlgError:
        diverged = True
    ck.true("a correction on the wrong side makes the filter diverge from a 2 degree start", diverged)

    # --- heading on the circle
    t, truth, hist = compass_demo()
    es = np.degrees(np.abs(wrap(hist["subtract"] - truth)))
    ew = np.degrees(np.abs(wrap(hist["wrap"] - truth)))
    ck.true("compass: subtracting fails at the wrap, the logarithm does not", es.max() > 150 and ew.max() < 5,
            f"{es.max():.0f} vs {ew.max():.1f} deg")
    ck.record("compass_subtract_max_deg", float(es.max()))
    ck.record("compass_subtract_final_deg", float(es[-1]))
    ck.record("compass_wrap_max_deg", float(ew.max()))
    ck.record("compass_cross_time", 2.5)
    ck.close("mean of 179 and -179 degrees on the circle", np.degrees(wrap(np.radians(179) + 0.5 * wrap(np.radians(-179 - 179)))),
             180.0, 1e-9)

    # --- the banana
    b = banana(noisy=False, sig0=(0.25, 0.25, 0.4))
    n = b["chi"].shape[0] - 1
    exact = np.abs(b["chi"][-1] - b["chi"][0] @ np.linalg.matrix_power(b["Phi"], n).T).max()
    ck.close("banana: chi(t) = Phi^k chi(0) exactly for every sample", exact, 0, 1e-10)
    sk_x = skewness(b["poses"][-1][:, 1])
    sk_chi = max(abs(skewness(b["chi"][-1][:, i])) for i in range(3))
    ck.true("banana: skewed in position, not in exponential coordinates", sk_x < -1.5 and sk_chi < 0.1,
            f"{sk_x:.2f} vs {sk_chi:.3f}")
    ck.record("banana_skew_forward", sk_x)
    ck.record("banana_skew_chi_max", sk_chi)
    ck.record("banana_cov_rel", float(np.linalg.norm(np.cov(b["chi"][-1].T) - b["cov"][-1]) / np.linalg.norm(b["cov"][-1])))
    ck.record("banana_heading_std_deg", float(np.degrees(0.4)))
    bn = banana()
    Sn, Spn = np.cov(bn["chi"][-1].T), bn["cov"][-1]
    ck.record("banana_noisy_cov_rel", float(np.linalg.norm(Sn - Spn) / np.linalg.norm(Spn)))
    ck.record("banana_noisy_forward_var_ratio", float(Sn[0, 0] / Spn[0, 0]))
    ck.record("banana_noisy_skew_forward_chi", skewness(bn["chi"][-1][:, 0]))
    ck.record("banana_noisy_skew_lateral_chi", skewness(bn["chi"][-1][:, 1]))
    ck.record("banana_noisy_skew_x", skewness(bn["poses"][-1][:, 1]))
    ck.true("noisy odometry: Gaussian in exponential coordinates only approximately",
            Sn[0, 0] / Spn[0, 0] > 1.5 and abs(Sn[1, 1] / Spn[1, 1] - 1) < 0.1)
    ck.close("SE(2) adjoint", se2_log(*se2_mul(se2_mul((np.array([0.7]), np.array([1.0]), np.array([-2.0])),
                                                       se2_exp(np.array([[0.3, -0.2, 0.5]]))),
                                               se2_inv((np.array([0.7]), np.array([1.0]), np.array([-2.0])))))[0],
             se2_Ad(0.7, 1.0, -2.0) @ np.array([0.3, -0.2, 0.5]), 1e-12)

    # --- contact-aided filter, simplified
    t, o, (Phi3, Cy3) = contact_demo()
    Ob = np.vstack([Cy3 @ np.linalg.matrix_power(Phi3, k) for k in range(12)])
    rank = int(np.linalg.matrix_rank(Ob))
    ck.true("contact: observability matrix has rank 8 of 12", rank == 8)
    null = np.zeros((4, 12))
    null[0, 11] = 1.0                                   # yaw about e3
    null[1:, 3:6] = np.eye(3)
    null[1:, 6:9] = np.eye(3)                           # body and contact moved together
    ck.close("contact: yaw and absolute position are unobservable", Ob @ null.T, 0, 1e-12)
    i1 = int(1.0 / (t[1] - t[0]))
    ck.true("contact: tilt and velocity converge, yaw and position do not",
            np.degrees(np.median(o["tilt"][i1])) < 1 and np.median(o["vel"][i1]) < 0.05
            and np.degrees(np.median(o["yaw"][-1])) > 25 and np.median(o["pos"][-1]) > 0.05)
    ck.record("contact_rank", rank)
    ck.record("contact_tilt0_deg", float(np.degrees(np.median(o["tilt"][0]))))
    ck.record("contact_tilt_1s_deg", float(np.degrees(np.median(o["tilt"][i1]))))
    ck.record("contact_vel0", float(np.median(o["vel"][0])))
    ck.record("contact_vel_1s", float(np.median(o["vel"][i1])))
    ck.record("contact_yaw_end_deg", float(np.degrees(np.median(o["yaw"][-1]))))
    ck.record("contact_pos0", float(np.median(o["pos"][0])))
    ck.record("contact_pos_end", float(np.median(o["pos"][-1])))

    # --- the sphere: operators, linearization, filter
    q = unit(rng.normal(size=3))
    om = np.cross(q, rng.normal(size=3))
    C, E = s2_C(q, om), s2_E(q, om)
    Z = np.linalg.svd(C)[2][2:].T                        # orthonormal basis of ker C
    ck.close("E projects onto ker C: C E = 0, E E = E, E Z = Z",
             [np.abs(C @ E).max(), np.abs(E @ E - E).max(), np.abs(E @ Z - Z).max()], 0, 1e-12)
    s = Z @ rng.normal(size=4) * 0.7
    q2, om2 = s2_oplus(q, om, s)
    ck.close("oplus lands on the sphere and its tangent plane", [np.linalg.norm(q2) - 1, q2 @ om2], 0, 1e-12)
    ck.close("ominus inverts oplus", s2_ominus(q2, om2, q, om), s, 1e-10)
    Rs = expm_so3(s[:3])
    dq1, dq2 = np.cross(om, q), np.cross(om2, q2)
    dom_K = -hat(Rs.T @ dq2 - dq1) @ q - (om @ np.cross(s[:3], q)) * q
    ck.close("delta omega of the mapping operator (Kotaru, eq. B.12)", dom_K, s[3:], 1e-10)
    dqk = Rs @ (dq1 - hat(q) @ s[3:])                    # Kotaru, eqs. (B.15)-(B.17)
    ck.close("oplus agrees with Kotaru, eqs. (B.14)-(B.17)", np.cross(q2, dqk), om2, 1e-10)
    ck.close("the turn exp(xi^) carries the velocity by parallel transport along the arc",
             Rs @ om, om - (om @ q2) / (1 + q @ q2) * (q + q2), 1e-10)
    # variation-based linearization against the flow
    def flow(q, om, T=0.3, n=300):
        for _ in range(n):
            q, om = pend_step(q, om, T / n)
        return q, om
    qf, omf = flow(q, om)
    Phi = np.eye(6)
    qq, oo = q, om
    for _ in range(300):
        Phi = expm(s2_A(*pend_step(qq, oo, 5e-4)) * 1e-3) @ Phi      # A at the middle of the step
        qq, oo = pend_step(qq, oo, 1e-3)
    s_dir = Z @ rng.normal(size=4)
    qp, omp = flow(*s2_oplus(q, om, 1e-5 * s_dir))
    fd = s2_ominus(qp, omp, qf, omf) / 1e-5
    ck.close("variation-based transition matrix against the flow", fd, Phi @ s_dir, 2e-5)
    ck.close("the flow keeps the constraint: C(t) Phi Z = 0", s2_C(qf, omf) @ Phi @ Z, 0, 2e-5)
    # moving the covariance: Jacobian of the re-centering against E' T
    sp = Z @ rng.normal(size=4) * 0.02
    qn, omn = s2_oplus(q, om, sp)
    Tm = np.zeros((6, 6))
    Tm[:3, :3] = Tm[3:, 3:] = expm_so3(sp[:3])
    Gj = np.array([(s2_ominus(*s2_oplus(q, om, sp + 1e-6 * Z[:, i]), qn, omn)) / 1e-6 for i in range(4)]).T
    gap = np.abs(Gj - s2_E(qn, omn) @ Tm @ Z).max()
    ck.true("re-centering Jacobian equals E' T up to the size of the step", gap < 0.05, f"gap {gap:.3f} at |s| = {np.linalg.norm(sp):.3f}")
    t, ang, st, i1 = s2_mc()
    ck.true("variation-based EKF: estimate on the sphere to rounding", st["var_norm"] < 1e-12 and st["var_tan"] < 1e-12)
    ck.true("variation-based EKF: covariance stays in the constraint subspace", st["cmax"] < 1e-9, f"{st['cmax']:.1e}")
    ck.true("variation-based EKF: consistent", abs(st["nees"] - 1) < 0.15, f"NEES {st['nees']:.2f}")
    ck.true("variation-based EKF: converges in every run", ang["var"][:, i1:].max() < 5.0, f"max {ang['var'][:, i1:].max():.2f} deg")
    ck.true("EKF on six numbers leaves the sphere", st["naive_norm_hi"] - st["naive_norm_lo"] > 0.02)
    ck.record("s2_runs", 100)
    ck.record("s2_ang0_med_deg", st["ang0_med"])
    ck.record("s2_ang0_max_deg", st["ang0_max"])
    ck.record("s2_var_norm_dev", st["var_norm"])
    ck.record("s2_naive_norm_lo", st["naive_norm_lo"])
    ck.record("s2_naive_norm_hi", st["naive_norm_hi"])
    ck.record("s2_naive_norm_lo_after1s", st["naive_norm_lo1"])
    ck.record("s2_naive_norm_hi_after1s", st["naive_norm_hi1"])
    ck.record("s2_naive_q_dot_omega_max", st["naive_tan"])
    ck.record("s2_cov_constraint_max", st["cmax"])
    ck.record("s2_nees", st["nees"])
    for f in ("var", "naive"):
        ck.record(f"s2_{f}_rms_deg_after1s", float(np.sqrt(np.mean(ang[f][:, i1:] ** 2))))
        ck.record(f"s2_{f}_max_deg_after1s", float(ang[f][:, i1:].max()))
        ck.record(f"s2_{f}_med_deg_at_0p2s", float(np.median(ang[f][:, int(0.2 / (t[1] - t[0]))])))

    ck.finish()


if __name__ == "__main__":
    main()
