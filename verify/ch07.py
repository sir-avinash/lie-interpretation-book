"""Chapter 7: every linearization, rank, Riccati solution and quoted number, checked.

    python verify/ch07.py

The functions above ``main`` are the chapter's models and controllers. The figure
script imports them, so the figures show exactly what is verified here.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee  # noqa: E402

E1, E2, E3 = np.eye(3)
I3 = np.eye(3)
GRAV = 9.81


def log_so3(R):
    """Rotation vector of R (angle below pi)."""
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-8:
        return vee(R - R.T) / 2
    return th / (2 * np.sin(th)) * vee(R - R.T)


def ctrb(A, B):
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(A.shape[0])])


# =====================================================================
# Spherical pendulum on S^2: mass m at l q, force F on the bob.
#     q' = w x q,      m l w' = q x (F - m g e3)
# =====================================================================
M_P, L_P = 0.5, 0.8


def s2_f(q, w, F):
    return np.cross(w, q), np.cross(q, F - M_P * GRAV * E3) / (M_P * L_P)


def s2_A(qd, wd, Fd):
    """Hand-derived A(t) for s = (xi, delta omega), equation (eq7-s2-lin)."""
    A = np.zeros((6, 6))
    A[:3, :3] = hat(wd)
    A[:3, 3:] = I3 - np.outer(qd, qd)
    A[3:, :3] = hat(Fd - M_P * GRAV * E3) @ hat(qd) / (M_P * L_P)
    return A


def s2_B(qd):
    return np.vstack([np.zeros((3, 3)), hat(qd) / (M_P * L_P)])


def s2_C(qd, wd):
    """Constraint matrix: C s = 0, equation (eq7-s2-constraint)."""
    return np.block([[qd, np.zeros(3)], [np.cross(qd, wd), qd]])


def s2_null(qd, wd):
    """Orthonormal basis (6 x 4) of the constraint subspace ker C."""
    _, _, Vt = np.linalg.svd(s2_C(qd, wd))
    return Vt[2:].T


def s2_sim(q0, w0, Ffun, T, t_eval=None, events=None, rtol=1e-10):
    def rhs(t, x):
        dq, dw = s2_f(x[:3], x[3:], Ffun(t, x[:3], x[3:]))
        return np.r_[dq, dw]

    return solve_ivp(rhs, [0, T], np.r_[q0, w0], t_eval=t_eval, events=events, rtol=rtol,
                     atol=rtol * 1e-2, method="DOP853", dense_output=t_eval is None)


def s2_swing():
    """Reference used for the variation figures: a free swing released high up."""
    th0 = np.radians(140.0)                       # angle from the hanging direction
    q0 = np.array([np.sin(th0), 0.0, -np.cos(th0)])
    w0 = np.cross(q0, [0.0, 1.1, 0.0]) * 0 + 1.3 * np.cross(q0, E2) + 0.9 * E2
    w0 = w0 - (w0 @ q0) * q0
    return q0, w0


def s2_linear_flow(ref, s0, T, dF=None, Fd=lambda t: np.zeros(3)):
    """Integrate s' = A(t) s + B(t) dF(t) along a reference dense solution."""
    def rhs(t, s):
        x = ref.sol(t)
        out = s2_A(x[:3], x[3:], Fd(t)) @ s
        return out if dF is None else out + s2_B(x[:3]) @ dF(t)

    return solve_ivp(rhs, [0, T], s0, rtol=1e-10, atol=1e-12, method="DOP853", dense_output=True)


def s2_inverted_lqr(Q=np.diag([10.0, 10.0, 1.0, 1.0]), Rw=0.1 * np.eye(3)):
    """LQR for the upright equilibrium, designed on the 4-dimensional constraint subspace."""
    qd = E3
    A, B = s2_A(qd, np.zeros(3), np.zeros(3)), s2_B(qd)
    N = np.zeros((6, 4))
    N[0, 0] = N[1, 1] = N[3, 2] = N[4, 3] = 1.0          # tangent plane at e3 is span(e1, e2)
    Ar, Br = N.T @ A @ N, N.T @ B
    P = solve_continuous_are(Ar, Br, Q, Rw)
    Kr = np.linalg.solve(Rw, Br.T @ P)
    return dict(A=A, B=B, N=N, Ar=Ar, Br=Br, P=P, Kr=Kr, K=Kr @ N.T, Q=Q, Rw=Rw)


def s2_error(qd, q, mode="sin"):
    e = np.cross(qd, q)
    if mode == "log":
        sn = np.linalg.norm(e)
        e = e * (np.arctan2(sn, qd @ q) / sn if sn > 1e-9 else 1.0)
    return e


def s2_settle(K, th0, mode="sin", Tmax=20.0, tol_deg=1.0):
    """Time after which the bob stays within tol_deg of upright; inf if it falls."""
    q0 = np.array([np.sin(th0), 0.0, np.cos(th0)])
    hang = lambda t, x: x[2] + 0.99
    hang.terminal = True
    grid = np.linspace(0, Tmax, 4001)
    s = s2_sim(q0, np.zeros(3), lambda t, q, w: -K @ np.r_[s2_error(E3, q, mode), w], Tmax,
               t_eval=grid, events=hang, rtol=1e-9)
    ang = np.degrees(np.arccos(np.clip(s.y[2], -1, 1)))
    if len(s.t) < len(grid) or ang[-1] > tol_deg:
        return np.inf
    bad = np.nonzero(ang > tol_deg)[0]
    return float(s.t[bad[-1] + 1]) if len(bad) else 0.0


# =====================================================================
# Rigid body on SO(3) with an offset centre of mass (3D pendulum):
#     R' = R hat(W),   J W' = J W x W + m g (R^T e3) x rho + tau
# rho = 0 is the attitude dynamics of a multirotor.
# =====================================================================
J_B = np.diag([0.03, 0.05, 0.07])
J_INV = np.linalg.inv(J_B)
ZERO3 = np.zeros(3)


def so3_acc(R, Om, tau, mgr=ZERO3, J=J_B):
    return np.linalg.solve(J, np.cross(J @ Om, Om) + np.cross(R.T @ E3, mgr) + tau)


def so3_A(Rd, Od, mgr=ZERO3, J=J_B):
    """Hand-derived A(t) for s = (eta, delta Omega), equation (eq7-so3-lin)."""
    Ji = np.linalg.inv(J)
    A = np.zeros((6, 6))
    A[:3, :3] = -hat(Od)
    A[:3, 3:] = I3
    A[3:, :3] = -Ji @ hat(mgr) @ hat(Rd.T @ E3)
    A[3:, 3:] = Ji @ (hat(J @ Od) - hat(Od) @ J)
    return A


SO3_B = np.vstack([np.zeros((3, 3)), J_INV])


def e_R(Rd, R):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def e_Om(Rd, R, Od, Om):
    return Om - R.T @ Rd @ Od


def so3_sim(R0, O0, ctrl, T, t_eval=None, t0=0.0, mgr=ZERO3, rtol=1e-9, max_step=np.inf):
    def rhs(t, x):
        R, Om = x[:9].reshape(3, 3), x[9:]
        return np.r_[(R @ hat(Om)).ravel(), so3_acc(R, Om, ctrl(t, R, Om), mgr)]

    return solve_ivp(rhs, [t0, T], np.r_[R0.ravel(), O0], t_eval=t_eval, rtol=rtol, atol=rtol * 1e-2,
                     method="DOP853", dense_output=t_eval is None, max_step=max_step)


class Flip:
    """One full turn about an axis tilted `delta` away from the pitch axis."""

    def __init__(self, T=1.5, delta_deg=5.0):
        d = np.radians(delta_deg)
        self.T, self.n = T, np.array([np.sin(d), np.cos(d), 0.0])

    def angle(self, t):
        s = np.clip(t / self.T, 0.0, 1.0)
        two_pi = 2 * np.pi
        return (two_pi * (10 * s**3 - 15 * s**4 + 6 * s**5),
                two_pi * (30 * s**2 - 60 * s**3 + 30 * s**4) / self.T,
                two_pi * (60 * s - 180 * s**2 + 120 * s**3) / self.T**2)

    def __call__(self, t):
        a, da, dda = self.angle(t)
        Rd, Od = expm_so3(a * self.n), da * self.n
        return Rd, Od, J_B @ (dda * self.n) + np.cross(Od, J_B @ Od)


def riccati_ltv(Afun, Bfun, Q, Rw, Pf, T, n_grid=1501):
    """Integrate -P' = A'P + PA - P B R^-1 B' P + Q backwards from P(T) = Pf."""
    n, Rinv = Q.shape[0], np.linalg.inv(Rw)

    def rhs(t, p):
        P, A, B = p.reshape(n, n), Afun(t), Bfun(t)
        return -(A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q).ravel()

    ts = np.linspace(T, 0, n_grid)
    sol = solve_ivp(rhs, [T, 0], Pf.ravel(), t_eval=ts, rtol=1e-9, atol=1e-11, method="LSODA")
    Ps = sol.y.T.reshape(-1, n, n)[::-1]
    ts = ts[::-1]
    Ks = np.array([Rinv @ Bfun(t).T @ P for t, P in zip(ts, Ps)])
    return ts, Ps, Ks


Q_ATT = np.diag([100.0] * 3 + [1.0] * 3)
R_ATT = 20.0 * np.eye(3)


def hover_lqr():
    A0 = so3_A(I3, ZERO3)
    P = solve_continuous_are(A0, SO3_B, Q_ATT, R_ATT)
    return A0, P, np.linalg.solve(R_ATT, SO3_B.T @ P)


def flip_lqr(ref, n_grid=1501):
    """Finite-horizon LQR along the flip. Returns the gain as a function of time."""
    _, P_inf, _ = hover_lqr()
    tg = np.linspace(0, ref.T, n_grid)
    A_tab = interp1d(tg, np.array([so3_A(*ref(t)[:2]) for t in tg]), axis=0)
    ts, Ps, Ks = riccati_ltv(lambda t: A_tab(t), lambda t: SO3_B, Q_ATT, R_ATT, P_inf, ref.T, n_grid)
    return dict(ts=ts, Ps=Ps, Ks=Ks, K=interp1d(ts, Ks, axis=0), P_inf=P_inf)


def flip_controller(ref, K, velocity="transported"):
    """tau = tau_d - K(t) s with s built from the geometric errors.

    velocity="transported": s = (e_R, e_Omega + Omega_d x e_R), the first-order image of (eta, dOmega).
    velocity="naive":       s = (e_R, e_Omega), which is (eta, eta_dot) to first order.
    """
    def ctrl(t, R, Om):
        t = min(max(t, 0.0), ref.T)
        Rd, Od, tau_d = ref(t)
        eR, eO = e_R(Rd, R), e_Om(Rd, R, Od, Om)
        s = np.r_[eR, eO + np.cross(Od, eR)] if velocity == "transported" else np.r_[eR, eO]
        return tau_d - K(t) @ s

    return ctrl


def flip_cost(ref, lq, ctrl, eta0, t0):
    """Closed-loop LQR cost from an attitude error eta0 injected at time t0."""
    Rd0, Od0, _ = ref(t0)

    def rhs(t, x):
        R, Om = x[:9].reshape(3, 3), x[9:12]
        Rd, Od, tau_d = ref(t)
        tau = ctrl(t, R, Om)
        s, du = np.r_[log_so3(Rd.T @ R), Om - Od], tau - tau_d
        return np.r_[(R @ hat(Om)).ravel(), so3_acc(R, Om, tau), s @ Q_ATT @ s + du @ R_ATT @ du]

    sol = solve_ivp(rhs, [t0, ref.T], np.r_[(Rd0 @ expm_so3(eta0)).ravel(), Od0, 0.0],
                    rtol=1e-9, atol=1e-13, method="DOP853")
    x = sol.y[:, -1]
    RdT, OdT, _ = ref(ref.T)
    sT = np.r_[log_so3(RdT.T @ x[:9].reshape(3, 3)), x[9:12] - OdT]
    return x[12] + sT @ lq["P_inf"] @ sT, np.degrees(np.linalg.norm(sT[:3]))


# ---- the Euler-angle competitor (ZYX, as in Chapter 1) -------------------
def euler_of(R):
    return np.array([np.arctan2(R[1, 0], R[0, 0]), -np.arcsin(np.clip(R[2, 0], -1, 1)),
                     np.arctan2(R[2, 1], R[2, 2])])


def euler_E(th, ph):
    return np.array([[-np.sin(th), 0, 1],
                     [np.cos(th) * np.sin(ph), np.cos(ph), 0],
                     [np.cos(th) * np.cos(ph), -np.sin(ph), 0]])


def euler_f(x, tau):
    eul, Om = x[:3], x[3:]
    return np.r_[np.linalg.solve(euler_E(eul[1], eul[2]), Om), J_INV @ (np.cross(J_B @ Om, Om) + tau)]


def euler_A(x, tau, h=1e-6):
    """Jacobian of the Euler-chart dynamics with respect to (angles, Omega)."""
    A = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = h
        A[:, i] = (euler_f(x + d, tau) - euler_f(x - d, tau)) / (2 * h)
    return A


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def hold_compare(pitch_deg, err_deg, axis, T=4.0, n=801):
    """Hold R_d = R_y(pitch) from an initial error; geometric LQR against Euler-angle LQR."""
    Rd = Ry(np.radians(pitch_deg))
    eul_d = euler_of(Rd)
    _, _, K = hover_lqr()
    AE = euler_A(np.r_[eul_d, ZERO3], ZERO3)
    KE = np.linalg.solve(R_ATT, SO3_B.T @ solve_continuous_are(AE, SO3_B, Q_ATT, R_ATT))
    R0 = Rd @ expm_so3(np.radians(err_deg) * np.asarray(axis, float) / np.linalg.norm(axis))
    laws = {"geometric": lambda t, R, Om: -K @ np.r_[e_R(Rd, R), Om],
            "euler": lambda t, R, Om: -KE @ np.r_[wrap(euler_of(R) - eul_d), Om]}
    out = {"Rd": Rd, "KE": KE, "K": K}
    te = np.linspace(0, T, n)
    for name, law in laws.items():
        s = so3_sim(R0, ZERO3, law, T, t_eval=te, rtol=1e-8, max_step=2e-3)
        Rs = s.y[:9].T.reshape(-1, 3, 3)
        ang = np.array([np.degrees(np.linalg.norm(log_so3(Rd.T @ R))) for R in Rs])
        tq = np.array([np.linalg.norm(law(t, R, x[9:])) for t, R, x in zip(s.t, Rs, s.y.T)])
        out[name] = dict(t=s.t, R=Rs, angle=ang, torque=tq, effort=float(np.trapezoid(tq**2, s.t)))
    return out


def so3_settle(K, th0, mode="sin", Tmax=40.0, tol_deg=1.0, axis=(1.0, 0.5, -0.7)):
    """Rest-to-rest recovery time of the hover LQR from an error of angle th0."""
    ax = np.asarray(axis) / np.linalg.norm(axis)
    err = (lambda R: e_R(I3, R)) if mode == "sin" else log_so3
    grid = np.linspace(0, Tmax, 4001)
    s = so3_sim(expm_so3(th0 * ax), ZERO3, lambda t, R, Om: -K @ np.r_[err(R), Om], Tmax, t_eval=grid, rtol=1e-8)
    ang = np.array([np.degrees(np.linalg.norm(log_so3(R))) for R in s.y[:9].T.reshape(-1, 3, 3)])
    if ang[-1] > tol_deg:
        return np.inf
    bad = np.nonzero(ang > tol_deg)[0]
    return float(s.t[bad[-1] + 1]) if len(bad) else 0.0


# =====================================================================
# geomech: the same linearizations, derived symbolically
# =====================================================================
def geomech_s2():
    from geomech import S2, Cross, Dot, Scalar, SystemVariables, Vector, check_eom, compute_eom, getScalars, linearize
    from geomech.codegen import StateLayout, collect_params, load_module, module_source

    half = Scalar("0.5", value=0.5, attr=["Constant"])
    m, g, l = getScalars("m g l", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    q, F = S2("q"), Vector("F")
    om, xi = q.get_tangent_vector(), q.get_variation_vector()
    L = half * m * l * l * Dot(om, om) - m * g * l * Dot(q, e3)
    dW = Dot(Cross(xi, q) * l, F)                       # a force F applied at the bob, l q
    V = SystemVariables(vectors=[q])
    eom = compute_eom(L, dW, V)
    err = check_eom(L, dW, V, eom, inputs=[F])
    lin = linearize(eom, V, [F])
    layout = StateLayout(V, [F], collect_params([L, dW], V, [F]))
    mod = load_module(module_source(lin, layout), "ch07_s2_lin")
    p = {"e3": E3, "g": GRAV, "l": L_P, "m": M_P}
    return err, mod, layout, p


def geomech_so3():
    from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, check_eom, compute_eom, getScalars, linearize
    from geomech.codegen import StateLayout, collect_params, load_module, module_source

    half = Scalar("0.5", value=0.5, attr=["Constant"])
    Jm = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    rho = Vector("\\rho", attr=["Constant"])
    m, g = getScalars("m g", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    tau, R = Vector("M"), SO3("R")
    Om, eta = R.get_tangent_vector(), R.get_variation_vector()
    L = half * Dot(Om, Jm * Om) - m * g * Dot(R * rho, e3)
    dW = Dot(eta, tau)
    V = SystemVariables(matrices=[R])
    eom = compute_eom(L, dW, V)
    err = check_eom(L, dW, V, eom, inputs=[tau])
    lin = linearize(eom, V, [tau])
    layout = StateLayout(V, [tau], collect_params([L, dW], V, [tau]))
    mod = load_module(module_source(lin, layout), "ch07_so3_lin")
    return err, mod, layout


def geomech_AB(mod, state, u, p):
    a = np.zeros(mod.LAYOUT["n_accel"])
    M = mod.M_lin(state, a, u, p)
    return np.linalg.solve(M, mod.A_lin(state, a, u, p)), np.linalg.solve(M, mod.B_lin(state, a, u, p))


# =====================================================================
def main():
    ck = Checker("ch07")
    rng = np.random.default_rng(7)

    # ---- the opening puzzle: two linearizations that do not work ----------
    def embedded_f(x, tau):
        R, Om = x[:9].reshape(3, 3), x[9:]
        return np.r_[(R @ hat(Om)).ravel(), so3_acc(R, Om, tau)]

    x_hover, h = np.r_[I3.ravel(), ZERO3], 1e-6
    A12 = np.column_stack([(embedded_f(x_hover + h * d, ZERO3) - embedded_f(x_hover - h * d, ZERO3)) / (2 * h)
                           for d in np.eye(12)])
    B12 = np.column_stack([(embedded_f(x_hover, h * d) - embedded_f(x_hover, -h * d)) / (2 * h) for d in np.eye(3)])
    rank12 = int(np.linalg.matrix_rank(ctrb(A12, B12), tol=1e-6))
    ck.true("nine-entry linearization at hover: controllability rank 6 of 12", rank12 == 6, f"rank {rank12}")
    ck.record("embedded_rank", rank12)
    sym = np.zeros((3, 3))
    sym[0, 1] = sym[1, 0] = 1.0                           # a symmetric (off-manifold) direction of dR
    w = np.r_[sym.ravel(), ZERO3]
    ck.close("off-manifold directions of dR are unreachable (left null vector of the controllability matrix)",
             w @ ctrb(A12, B12), 0, tol=1e-6)
    ck.record("euler_inv_cos_85", float(1 / np.cos(np.radians(85))))
    ck.record("euler_inv_cos_89p5", float(1 / np.cos(np.radians(89.5))))
    ck.close("det E = -cos(theta)", np.linalg.det(euler_E(0.7, -0.4)), -np.cos(0.7))

    # ---- variation formulas on S^2 (Chapter 3), by finite differences ------
    q = rng.normal(size=3)
    q /= np.linalg.norm(q)
    xi = np.cross(q, rng.normal(size=3))
    eps = 1e-6
    dq = (expm_so3(eps * xi) @ q - expm_so3(-eps * xi) @ q) / (2 * eps)
    ck.close("S2: d/de exp(e xi^) q = xi x q", dq, np.cross(xi, q), tol=1e-9)
    ck.close("S2: q x (xi x q) = xi when xi.q = 0", np.cross(q, np.cross(xi, q)), xi)
    om = np.cross(q, rng.normal(size=3))
    ck.close("S2: omega x xi is along q when both are tangent", np.cross(om, xi), np.outer(q, q) @ hat(om) @ xi)

    # ---- S^2 linearization against perturbed nonlinear trajectories --------
    def Fd(t):
        return np.array([0.6 * np.sin(2 * t), 0.4 * np.cos(3 * t), 0.3])

    def dF(t):
        return np.array([np.cos(t), -0.5, np.sin(2 * t)])

    q0, w0 = s2_swing()
    T = 1.5
    ref = s2_sim(q0, w0, lambda t, q_, w_: Fd(t), T)
    xi0, dw0 = np.cross(q0, rng.normal(size=3)), rng.normal(size=3)

    def perturbed(e):
        qe = expm_so3(e * xi0) @ q0
        we = w0 + e * dw0
        return s2_sim(qe, we - (we @ qe) * qe, lambda t, q_, w_: Fd(t) + e * dF(t), T)

    errs = {}
    for e in (1e-3, 1e-4):
        sp, sm = perturbed(e), perturbed(-e)

        def s_fd(t):
            xr, d = ref.sol(t), (sp.sol(t) - sm.sol(t)) / (2 * e)
            return np.r_[np.cross(xr[:3], d[:3]), d[3:]]

        lin = s2_linear_flow(ref, s_fd(0.0), T, dF=dF, Fd=Fd)
        errs[e] = max(np.abs(s_fd(t) - lin.sol(t)).max() for t in np.linspace(0, T, 31))
        if e == 1e-4:
            cs = max(np.abs(s2_C(ref.sol(t)[:3], ref.sol(t)[3:]) @ s_fd(t)).max() for t in np.linspace(0, T, 31))
            ck.close("S2: finite-difference variation satisfies C(t) s = 0 along the trajectory", cs, 0, tol=1e-6)
            ck.true("S2: the perturbation is not small (|s| > 1 at the end)", np.linalg.norm(s_fd(T)) > 1)
    ck.close("S2: hand A(t), B(t) match finite differences of perturbed trajectories", errs[1e-4], 0, tol=1e-7)
    ck.true("S2: mismatch shrinks like eps^2", 50 < errs[1e-3] / errs[1e-4] < 200, f"ratio {errs[1e-3] / errs[1e-4]:.0f}")
    ck.record("s2_fd_mismatch", float(errs[1e-4]))

    # ---- S^2 linearization against geomech --------------------------------
    err_eom, mod, layout, p = geomech_s2()
    ck.true("geomech check_eom, spherical pendulum, below 1e-5", err_eom < 1e-5, f"{err_eom:.1e}")
    worst_dyn = worst_kin = worst_B = off = 0.0
    for _ in range(20):
        qd = rng.normal(size=3)
        qd /= np.linalg.norm(qd)
        wd = np.cross(qd, rng.normal(size=3))
        Fr = rng.normal(size=3) * 3
        Ag, Bg = geomech_AB(mod, layout.pack(q=qd, omega_q=wd), Fr, p)
        Ah, Bh, N = s2_A(qd, wd, Fr), s2_B(qd), s2_null(qd, wd)
        worst_dyn = max(worst_dyn, np.abs(Ag[3:] - Ah[3:]).max())
        worst_B = max(worst_B, np.abs(Bg - Bh).max())
        worst_kin = max(worst_kin, np.abs((Ag - Ah) @ N).max())
        off = max(off, np.abs(Ag[:3] - Ah[:3]).max())
    ck.close("geomech: dynamic row of A equals the hand formula entry by entry", worst_dyn, 0, tol=1e-10)
    ck.close("geomech: B equals the hand formula entry by entry", worst_B, 0, tol=1e-10)
    ck.close("geomech: kinematic row agrees on the constraint subspace", worst_kin, 0, tol=1e-10)
    ck.true("geomech: kinematic row differs off the subspace (q q^T hat(w) against hat(w))", off > 0.1, f"{off:.2f}")

    # ---- the constraint subspace is invariant ------------------------------
    worst = 0.0
    for t in np.linspace(0.1, T - 0.1, 9):
        x = ref.sol(t)
        qdot, wdot = s2_f(x[:3], x[3:], Fd(t))
        Cdot = np.block([[qdot, np.zeros(3)], [np.cross(qdot, x[3:]) + np.cross(x[:3], wdot), qdot]])
        C, A, B = s2_C(x[:3], x[3:]), s2_A(x[:3], x[3:], Fd(t)), s2_B(x[:3])
        worst = max(worst, np.abs(C @ B).max(), np.abs(C @ A + Cdot).max())
    ck.close("S2: C B = 0 and C A + dC/dt = 0 along the reference", worst, 0, tol=1e-9)

    # ---- controllability on the constraint subspace ------------------------
    inv = s2_inverted_lqr()
    A, B, N = inv["A"], inv["B"], inv["N"]
    r_full = int(np.linalg.matrix_rank(ctrb(A, B)))
    r_red = int(np.linalg.matrix_rank(ctrb(inv["Ar"], inv["Br"])))
    ck.true("upright pendulum: naive rank test gives 4 of 6", r_full == 4, f"rank {r_full}")
    ck.true("upright pendulum: reduced pair is controllable, rank 4 of 4", r_red == 4, f"rank {r_red}")
    ck.close("upright pendulum: range of the controllability matrix is exactly ker C",
             s2_C(E3, ZERO3) @ ctrb(A, B), 0)
    ck.record("s2_rank_naive", r_full)
    ck.record("s2_rank_reduced", r_red)
    ck.close("upright pendulum: reduced A", inv["Ar"], np.block([[np.zeros((2, 2)), np.eye(2)],
                                                                  [GRAV / L_P * np.eye(2), np.zeros((2, 2))]]))
    ck.record("s2_g_over_l", GRAV / L_P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            solve_continuous_are(A, B, np.diag([10.0] * 3 + [1.0] * 3), inv["Rw"])
            failed = False
        except Exception:
            failed = True
    ck.true("Riccati equation on the unreduced 6-state pendulum has no stabilising solution (scipy raises)", failed)

    # time-varying test: M0 = B, M1 = dB/dt - A B, along the swing
    def M01(t, sign=-1.0):
        x, hh = ref.sol(t), 1e-5
        Bdot = (s2_B(ref.sol(t + hh)[:3]) - s2_B(ref.sol(t - hh)[:3])) / (2 * hh)
        return np.hstack([s2_B(x[:3]), Bdot + sign * s2_A(x[:3], x[3:], Fd(t)) @ s2_B(x[:3])])

    ranks, leak = [], 0.0
    for t in np.linspace(0.1, T - 0.1, 9):
        Mt, x = M01(t), ref.sol(t)
        ranks.append(np.linalg.matrix_rank(Mt, tol=1e-6))
        leak = max(leak, np.abs(s2_C(x[:3], x[3:]) @ Mt).max())
    ck.true("moving reference: rank [B, dB/dt - A B] = 4 = dim ker C at every sampled time", set(ranks) == {4})
    ck.close("moving reference: those columns lie in ker C(t)", leak, 0, tol=1e-6)

    def gram_rhs(t, y):                                     # Phi(0,t) and the Gramian on [0, t]
        Phi = y[:36].reshape(6, 6)
        x = ref.sol(t)
        PB = Phi @ s2_B(x[:3])
        return np.r_[(-Phi @ s2_A(x[:3], x[3:], Fd(t))).ravel(), (PB @ PB.T).ravel()]

    y = solve_ivp(gram_rhs, [0, 1.0], np.r_[np.eye(6).ravel(), np.zeros(36)], rtol=1e-10, atol=1e-12).y[:, -1]
    W = y[36:].reshape(6, 6)
    N0 = s2_null(q0, w0)
    sv = np.linalg.svd(W, compute_uv=False)
    ck.true("moving reference: controllability Gramian has rank 4 in R^6", sv[3] > 1e-4 and sv[4] < 1e-9 * sv[0],
            f"singular values {sv[3]:.1e}, {sv[4]:.1e}")
    ck.true("moving reference: Gramian is positive definite on ker C(0)", np.linalg.eigvalsh(N0.T @ W @ N0).min() > 1e-4)

    # the sign in the time-varying test matters: x' = A x + exp(A t) b u is not controllable
    A2, b2 = np.array([[0.0, 1.0], [-2.0, 0.3]]), np.array([[0.0], [1.0]])
    from scipy.linalg import expm
    Bt = lambda t: expm(A2 * t) @ b2
    Bd = (Bt(0.5 + 1e-6) - Bt(0.5 - 1e-6)) / 2e-6
    r_right = np.linalg.matrix_rank(np.hstack([Bt(0.5), Bd - A2 @ Bt(0.5)]), tol=1e-6)
    r_wrong = np.linalg.matrix_rank(np.hstack([Bt(0.5), Bd + A2 @ Bt(0.5)]), tol=1e-6)
    ck.true("LTV rank test needs dB/dt - A B: the + sign calls an uncontrollable system controllable",
            r_right == 1 and r_wrong == 2, f"ranks {r_right} and {r_wrong}")

    # ---- LQR for the upright pendulum ---------------------------------------
    K, Kr = inv["K"], inv["Kr"]
    kp, kd = K[0, 1], K[0, 4]
    ck.close("upright LQR: F = -K s = kp e3 x e_q + kd e3 x omega", -K, np.hstack([kp * hat(E3), kd * hat(E3)]), tol=1e-9)
    # closed form for the scalar problem  x'' = a x + b u,  cost  qp x^2 + qd x'^2 + r u^2
    a_, b_, qp_, qd_, r_ = GRAV / L_P, 1 / (M_P * L_P), 10.0, 1.0, 0.1
    kp_cf = (a_ + np.sqrt(a_**2 + b_**2 * qp_ / r_)) / b_
    kd_cf = np.sqrt(2 * kp_cf / b_ + qd_ / r_)
    ck.close("upright LQR gains equal the closed form of the scalar problem", [kp, kd], [kp_cf, kd_cf], tol=1e-9)
    eig = np.sort(np.linalg.eigvals(inv["Ar"] - inv["Br"] @ Kr).real)
    ck.record("s2_kp", float(kp))
    ck.record("s2_kd", float(kd))
    ck.record("s2_poles", [float(eig[0]), float(eig[-1])])
    ts, Ps, _ = riccati_ltv(lambda t: inv["Ar"], lambda t: inv["Br"], inv["Q"], inv["Rw"], np.zeros((4, 4)), 8.0, 201)
    ck.close("backward Riccati integration converges to scipy's algebraic solution", Ps[0], inv["P"], tol=1e-6)
    # geomech's own LQR about the same equilibrium
    from geomech import S2, Cross, Dot, Scalar, SystemVariables, Vector, compute_eom, getScalars, to_standard_form
    from geomech.codegen import StateLayout, collect_params, load_module, module_source
    from geomech.simulation import lqr as geomech_lqr
    half = Scalar("0.5", value=0.5, attr=["Constant"])
    m_, g_, l_ = getScalars("m g l", attr=["Constant"])
    e3_ = Vector("e3", attr=["Constant"])
    q_, F_ = S2("q"), Vector("F")
    L_ = half * m_ * l_ * l_ * Dot(q_.get_tangent_vector(), q_.get_tangent_vector()) - m_ * g_ * l_ * Dot(q_, e3_)
    dW_ = Dot(Cross(q_.get_variation_vector(), q_) * l_, F_)
    V_ = SystemVariables(vectors=[q_])
    lay_ = StateLayout(V_, [F_], collect_params([L_, dW_], V_, [F_]))
    plant = load_module(module_source(to_standard_form(compute_eom(L_, dW_, V_), V_, [F_]), lay_, lagrangian=L_), "ch07_s2")
    gc = geomech_lqr(mod, lay_.pack(q=E3), np.zeros(3), p, Q=np.array([10.0, 10.0, 1.0, 1.0]), R=0.1, module=plant)
    ck.close("geomech.simulation.lqr gives the same feedback K N^T", gc.K @ gc.T.T, K, tol=1e-8)

    # region of attraction from rest, and two errors that agree to first order
    th_sin = np.degrees(np.arccos(M_P * GRAV / kp))
    from scipy.optimize import brentq
    th_log = np.degrees(brentq(lambda th: kp * th * np.cos(th) - M_P * GRAV * np.sin(th), 0.5, 1.55))
    ck.record("s2_roa_sin_deg", float(th_sin))
    ck.record("s2_roa_log_deg", float(th_log))
    below, above = s2_settle(K, np.radians(th_sin - 0.5)), s2_settle(K, np.radians(th_sin + 0.5))
    ck.true("pendulum with e_q feedback recovers below arccos(mg/kp) and falls above", np.isfinite(below) and np.isinf(above),
            f"{th_sin:.1f} deg")
    below, above = s2_settle(K, np.radians(th_log - 0.5), "log"), s2_settle(K, np.radians(th_log + 0.5), "log")
    ck.true("pendulum with log-error feedback recovers below its own boundary and falls above",
            np.isfinite(below) and np.isinf(above), f"{th_log:.1f} deg")
    t30 = s2_settle(K, np.radians(30.0))
    ck.record("s2_settle_30deg", t30)
    ck.true("pendulum recovers from 30 deg", np.isfinite(t30), f"{t30:.2f} s")

    # ---- variation formulas on SO(3) and the hand linearization -----------
    mgr = np.array([0.1, -0.2, 0.9])                       # m g rho of a generic 3D pendulum
    Jg = np.array([[0.05, 0.004, -0.002], [0.004, 0.06, 0.003], [-0.002, 0.003, 0.03]])

    def tau_d(t):
        return np.array([0.3 * np.sin(3 * t), -0.2, 0.25 * np.cos(2 * t)])

    def dtau(t):
        return np.array([np.cos(2 * t), np.sin(t), -0.7])

    def sim_gen(R0, O0, e, T_):
        def rhs(t, x):
            R, Om = x[:9].reshape(3, 3), x[9:]
            return np.r_[(R @ hat(Om)).ravel(), so3_acc(R, Om, tau_d(t) + e * dtau(t), mgr, Jg)]
        return solve_ivp(rhs, [0, T_], np.r_[R0.ravel(), O0], rtol=1e-11, atol=1e-13, method="DOP853", dense_output=True)

    R0, O0 = random_rotation(rng), np.array([2.0, -1.5, 3.0])
    eta0, dO0, T = rng.normal(size=3), rng.normal(size=3), 1.2
    refR = sim_gen(R0, O0, 0.0, T)
    errs = {}
    for e in (1e-3, 1e-4):
        sp, sm = sim_gen(R0 @ expm_so3(e * eta0), O0 + e * dO0, e, T), sim_gen(R0 @ expm_so3(-e * eta0), O0 - e * dO0, -e, T)

        def s_fd(t):
            xr, d = refR.sol(t), (sp.sol(t) - sm.sol(t)) / (2 * e)
            return np.r_[vee(xr[:9].reshape(3, 3).T @ d[:9].reshape(3, 3)), d[9:]]

        def lin_rhs(t, s):
            xr = refR.sol(t)
            return so3_A(xr[:9].reshape(3, 3), xr[9:], mgr, Jg) @ s + np.r_[ZERO3, np.linalg.solve(Jg, dtau(t))]

        lin = solve_ivp(lin_rhs, [0, T], s_fd(0.0), rtol=1e-11, atol=1e-13, method="DOP853", dense_output=True)
        errs[e] = max(np.abs(s_fd(t) - lin.sol(t)).max() for t in np.linspace(0, T, 25))
        if e == 1e-4:
            t, hh = 0.6, 1e-4
            eta_dot = (s_fd(t + hh)[:3] - s_fd(t - hh)[:3]) / (2 * hh)
            ck.close("SO3: delta Omega = eta_dot + Omega_d x eta along perturbed trajectories",
                     s_fd(t)[3:], eta_dot + np.cross(refR.sol(t)[9:], s_fd(t)[:3]), tol=1e-5)
            skew = refR.sol(t)[:9].reshape(3, 3).T @ ((sp.sol(t) - sm.sol(t)) / (2 * e))[:9].reshape(3, 3)
            ck.close("SO3: R_d^T delta R is skew, so delta R = R_d hat(eta)", skew + skew.T, 0, tol=1e-6)
    ck.close("SO3: hand A(t), B match finite differences of perturbed trajectories", errs[1e-4], 0, tol=5e-5)
    ck.true("SO3: mismatch shrinks like eps^2", 50 < errs[1e-3] / errs[1e-4] < 200, f"ratio {errs[1e-3] / errs[1e-4]:.0f}")
    ck.record("so3_fd_mismatch", float(errs[1e-4]))

    err_eom, modR, layR = geomech_so3()
    ck.true("geomech check_eom, 3D pendulum, below 1e-5", err_eom < 1e-5, f"{err_eom:.1e}")
    worst = 0.0
    for _ in range(20):
        Rr, Orr, rho = random_rotation(rng), rng.normal(size=3) * 3, rng.normal(size=3)
        pp = {"J": Jg, "\\rho": rho, "e3": E3, "g": GRAV, "m": 1.3}
        Ag, Bg = geomech_AB(modR, layR.pack(R=Rr, Omega_R=Orr), rng.normal(size=3), pp)
        Ah = so3_A(Rr, Orr, 1.3 * GRAV * rho, Jg)
        worst = max(worst, np.abs(Ag - Ah).max(), np.abs(Bg - np.vstack([np.zeros((3, 3)), np.linalg.inv(Jg)])).max())
    ck.close("geomech: A and B of the 3D pendulum equal the hand formulas entry by entry", worst, 0, tol=1e-9)

    # controllability along any reference: [B, dB/dt - A B] has rank 6
    Rr, Orr = random_rotation(rng), rng.normal(size=3) * 4
    r6 = np.linalg.matrix_rank(np.hstack([SO3_B, -so3_A(Rr, Orr, mgr) @ SO3_B]))
    ck.true("SO3: rank [B, -A B] = 6 at an arbitrary reference state", r6 == 6)

    # ---- what the linear state means ---------------------------------------
    Rd, Od = random_rotation(rng), np.array([3.0, -2.0, 4.0])
    eta, dO = rng.normal(size=3), rng.normal(size=3)
    res = {}
    for e in (1e-2, 1e-3):
        R, Om = Rd @ expm_so3(e * eta), Od + e * dO
        res[e] = (np.abs(e_R(Rd, R) - e * eta).max(),
                  np.abs(e_Om(Rd, R, Od, Om) - e * (dO - np.cross(Od, eta))).max(),
                  np.abs(e_Om(Rd, R, Od, Om) - e * dO).max())
    ck.true("e_R = eta to first order (remainder is third order)", 500 < res[1e-2][0] / res[1e-3][0] < 2000)
    ck.true("e_Omega = dOmega - Omega_d x eta to first order (remainder is second order)",
            50 < res[1e-2][1] / res[1e-3][1] < 200)
    ck.true("e_Omega is NOT dOmega to first order when Omega_d is not zero", 8 < res[1e-2][2] / res[1e-3][2] < 12)
    qd = rng.normal(size=3)
    qd /= np.linalg.norm(qd)
    xi = np.cross(qd, rng.normal(size=3))
    r1 = [np.abs(np.cross(qd, expm_so3(e * xi) @ qd) - e * xi).max() for e in (1e-2, 1e-3)]
    ck.true("e_q = xi to first order (remainder is third order)", 500 < r1[0] / r1[1] < 2000)

    # nonlinear error against the linear prediction, as the perturbation grows (figure 7.2)
    q0, w0 = s2_swing()
    Tsw = 1.2
    ref = s2_sim(q0, w0, lambda t, q_, w_: np.zeros(3), Tsw)
    axis = np.cross(q0, [0.3, 1.0, 0.2])
    axis /= np.linalg.norm(axis)
    lin = s2_linear_flow(ref, np.r_[axis, ZERO3], Tsw)
    rel = {}
    for deg in (1.0, 10.0, 30.0):
        e = np.radians(deg)
        pert = s2_sim(expm_so3(e * axis) @ q0, w0 - (w0 @ (expm_so3(e * axis) @ q0)) * (expm_so3(e * axis) @ q0),
                      lambda t, q_, w_: np.zeros(3), Tsw)
        eq = np.cross(ref.sol(Tsw)[:3], pert.sol(Tsw)[:3])
        rel[deg] = float(np.linalg.norm(eq - e * lin.sol(Tsw)[:3]) / np.linalg.norm(e * lin.sol(Tsw)[:3]))
    ck.true("linear prediction of e_q: relative error grows with the perturbation",
            rel[1.0] < 0.05 and rel[1.0] < rel[10.0] < rel[30.0], f"{rel}")
    ck.record("s2_linear_rel_error", rel)
    ck.record("s2_swing_growth", float(np.linalg.norm(lin.sol(Tsw)[:3])))

    # ---- LQR on SO(3): one gain for every attitude ------------------------
    A0, P0, K0 = hover_lqr()
    kR = np.sqrt(Q_ATT[0, 0] / R_ATT[0, 0])
    kO = np.sqrt(Q_ATT[3, 3] / R_ATT[0, 0] + 2 * np.diag(J_B) * kR)
    ck.close("attitude-hold LQR gain is diag(kR) and diag(kOmega) in closed form", K0, np.hstack([kR * I3, np.diag(kO)]), tol=1e-9)
    ck.record("so3_kR", float(kR))
    ck.record("so3_kOmega", [float(v) for v in kO])
    ck.close("A, B at rest do not depend on the hold attitude (free body)", so3_A(random_rotation(rng), ZERO3), A0)
    ts, Ps, _ = riccati_ltv(lambda t: A0, lambda t: SO3_B, Q_ATT, R_ATT, np.zeros((6, 6)), 6.0, 201)
    ck.close("backward Riccati integration (attitude) converges to scipy's algebraic solution", Ps[0], P0, tol=1e-6)

    # ---- flip: finite-horizon LQR along a time-varying reference ----------
    ref = Flip()
    lq = flip_lqr(ref)
    i = 700
    Pdot = (lq["Ps"][i + 1] - lq["Ps"][i - 1]) / (lq["ts"][i + 1] - lq["ts"][i - 1])
    Ai, Pi = so3_A(*ref(lq["ts"][i])[:2]), lq["Ps"][i]
    resid = Pdot + Ai.T @ Pi + Pi @ Ai - Pi @ SO3_B @ np.linalg.solve(R_ATT, SO3_B.T @ Pi) + Q_ATT
    ck.close("flip: P(t) satisfies the differential Riccati equation", resid / np.abs(Pi).max(), 0, tol=1e-4)
    ck.close("flip: P(T) is the terminal weight", lq["Ps"][-1], lq["P_inf"], tol=1e-9)
    Rd, Od, td = ref(0.6)
    hh = 1e-6
    Rdot = (ref(0.6 + hh)[0] - ref(0.6 - hh)[0]) / (2 * hh)
    ck.close("flip reference: R_d' = R_d hat(Omega_d)", Rdot, Rd @ hat(Od), tol=1e-6)
    Odot = (ref(0.6 + hh)[1] - ref(0.6 - hh)[1]) / (2 * hh)
    ck.close("flip reference: tau_d satisfies Euler's equation", J_B @ Odot, np.cross(J_B @ Od, Od) + td, tol=1e-6)
    pitch = np.degrees([euler_of(ref(t)[0])[1] for t in np.linspace(0, ref.T, 3001)])
    ck.close("flip reference: pitch peaks 5 deg short of the singularity", np.abs(pitch).max(), 85.0, tol=1e-3)
    ck.record("flip_peak_rate", float(ref.angle(ref.T / 2)[1]))
    ck.record("flip_peak_torque", float(max(np.linalg.norm(ref(t)[2]) for t in np.linspace(0, ref.T, 601))))
    knorm = np.linalg.norm(lq["Ks"], axis=(1, 2))
    ck.record("flip_gain_norm_range", [float(knorm.min()), float(knorm.max())])
    ck.record("flip_gain_coupling_max", float(np.abs(lq["Ks"][:, 0, 2]).max()))

    ax = np.array([1.0, 0.5, -0.7])
    ax /= np.linalg.norm(ax)
    good, naive = flip_controller(ref, lq["K"]), flip_controller(ref, lq["K"], "naive")
    te = np.linspace(0, ref.T, 601)
    s = so3_sim(expm_so3(np.radians(15) * ax), ZERO3, good, ref.T, t_eval=te)
    final = np.degrees(np.linalg.norm(log_so3(ref(ref.T)[0].T @ s.y[:9, -1].reshape(3, 3))))
    peak = max(np.linalg.norm(good(t, x[:9].reshape(3, 3), x[9:])) for t, x in zip(s.t, s.y.T))
    ck.true("flip: 15 deg initial error is tracked out", final < 0.05, f"final {final:.4f} deg")
    ck.record("flip_final_error_deg", float(final))
    ck.record("flip_peak_torque_closed_loop", float(peak))
    ck.close("flip: the simulated attitude stays on SO(3)", s.y[:9, -1].reshape(3, 3).T @ s.y[:9, -1].reshape(3, 3), I3, tol=1e-7)

    # LQR optimality is a sharp test of the state dictionary: actual cost / s0' P(t0) s0
    t0 = 0.45
    P_t0 = lq["Ps"][np.argmin(np.abs(lq["ts"] - t0))]
    ratios = {}
    for deg in (1.0, 15.0, 45.0):
        eta0 = np.radians(deg) * ax
        pred = eta0 @ P_t0[:3, :3] @ eta0
        cg, fg = flip_cost(ref, lq, good, eta0, t0)
        cn, fn = flip_cost(ref, lq, naive, eta0, t0)
        ratios[deg] = (float(cg / pred), float(cn / pred), float(fg), float(fn))
    ck.true("flip: with s = (e_R, e_Omega + Omega_d x e_R) the cost equals s0' P s0 for small errors",
            abs(ratios[1.0][0] - 1) < 1e-3, f"ratio {ratios[1.0][0]:.4f}")
    ck.true("flip: feeding (e_R, e_Omega) to the same gain doubles the cost", 1.9 < ratios[1.0][1] < 2.1,
            f"ratio {ratios[1.0][1]:.3f}")
    ck.record("flip_cost_ratio", {str(k): v for k, v in ratios.items()})
    ck.record("flip_rate_at_injection", float(ref.angle(t0)[1]))

    # ---- the Euler-angle LQR ------------------------------------------------
    tg = np.linspace(0, ref.T, 1501)
    eul = np.array([euler_of(ref(t)[0]) for t in tg])
    eul[:, 0], eul[:, 2] = np.unwrap(eul[:, 0]), np.unwrap(eul[:, 2])
    AE = np.array([euler_A(np.r_[e, ref(t)[1]], ref(t)[2]) for e, t in zip(eul, tg)])
    ck.record("flip_euler_A_max", float(np.abs(AE).max()))
    AEf = interp1d(tg, AE, axis=0)
    _, _, KsE = riccati_ltv(lambda t: AEf(t), lambda t: SO3_B, Q_ATT, R_ATT, lq["P_inf"], ref.T, 1501)
    kE = np.linalg.norm(KsE, axis=(1, 2))
    ck.record("flip_euler_gain_norm_max", float(kE.max()))
    ck.true("flip: Euler-chart gain peaks several times above the geometric gain", kE.max() > 4 * knorm.max(),
            f"{kE.max():.1f} against {knorm.max():.1f}")

    h0 = hold_compare(0.0, 20.0, (1, 1, 1))
    h85 = hold_compare(85.0, 20.0, (1, 1, 1))
    ck.close("hold: the geometric closed loop is the same at pitch 0 and pitch 85", h0["geometric"]["angle"],
             h85["geometric"]["angle"], tol=1e-5)
    ck.true("hold at pitch 0: the two LQRs behave alike",
            abs(h0["euler"]["torque"].max() / h0["geometric"]["torque"].max() - 1) < 0.1,
            f"peak torques {h0['euler']['torque'].max():.2f}, {h0['geometric']['torque'].max():.2f}")
    ck.true("hold at pitch 85: the Euler LQR tumbles through more than 170 deg", h85["euler"]["angle"].max() > 170,
            f"{h85['euler']['angle'].max():.1f} deg")
    ck.true("hold at pitch 85: both end at the target", max(h85["euler"]["angle"][-1], h85["geometric"]["angle"][-1]) < 0.1)
    for key, hc in (("hold0", h0), ("hold85", h85)):
        ck.record(key, {n: {"max_angle_deg": float(hc[n]["angle"].max()), "peak_torque": float(hc[n]["torque"].max()),
                            "effort": hc[n]["effort"]} for n in ("geometric", "euler")})
    ck.record("hold85_torque_ratio", float(h85["euler"]["torque"].max() / h85["geometric"]["torque"].max()))
    ck.record("hold85_effort_ratio", float(h85["euler"]["effort"] / h85["geometric"]["effort"]))
    h88 = hold_compare(88.0, 5.0, (1, 1, 1))
    ck.true("hold at pitch 88: a 5 deg error is enough to tumble the Euler LQR", h88["euler"]["angle"].max() > 170,
            f"{h88['euler']['angle'].max():.1f} deg")
    ck.record("hold88_max_angle", float(h88["euler"]["angle"].max()))

    # ---- large errors: e_R against the logarithm ---------------------------
    settle = {deg: (so3_settle(K0, np.radians(deg)), so3_settle(K0, np.radians(deg), "log")) for deg in (30.0, 150.0, 179.0)}
    ck.true("attitude hold: e_R and log feedback agree for small errors",
            abs(settle[30.0][0] - settle[30.0][1]) < 0.1 * settle[30.0][0], f"{settle[30.0]}")
    ck.true("attitude hold: e_R feedback lingers near 180 deg, log feedback does not",
            settle[179.0][0] > 2 * settle[179.0][1], f"{settle[179.0]}")
    ck.record("so3_settle", {str(k): v for k, v in settle.items()})
    half_turn = expm_so3(np.pi * ax)
    ck.close("at exactly 180 deg, e_R = 0: the LQR applies no torque", e_R(I3, half_turn), 0, tol=1e-12)

    # ---- the Lie-algebraic route -------------------------------------------
    Rd, Od = random_rotation(rng), np.array([3.0, -2.0, 4.0])
    psi, dO = rng.normal(size=3), rng.normal(size=3)
    res = []
    for e in (1e-2, 1e-3):
        R, Om, hh = Rd @ expm_so3(e * psi), Od + e * dO, 1e-6
        f = lambda sgn: log_so3((Rd @ expm_so3(sgn * hh * Od)).T @ (R @ expm_so3(sgn * hh * Om)))
        psi_dot = (f(1) - f(-1)) / (2 * hh)
        res.append(np.abs(psi_dot - (-np.cross(Od, e * psi) + Om - Od)).max())
    ck.true("psi = log(R_d^T R): psi_dot = -ad(Omega_d) psi + Omega - Omega_d up to second order",
            50 < res[0] / res[1] < 200, f"ratio {res[0] / res[1]:.0f}")
    ck.close("on so(3), ad(Omega) is hat(Omega): the kinematic row of A", so3_A(Rd, Od)[:3, :3], -hat(Od))

    ck.finish()


if __name__ == "__main__":
    main()
