"""Numerical checks for Chapter 18 (case study: a quadrotor with a cable-suspended load).

Run: python verify/ch18.py

The mission, the controllers, the safety filter, the adaptation law and the
estimator of the chapter live here. The figure script imports this file, so the
figures show what is verified. Blocks that an earlier chapter verified are
imported from that chapter's script (verify/ch07.py, ch08.py, ch10.py, ch14.py)
or from the robot model library (case_studies/robots), and the pieces written
here are compared with them.

Conventions (Appendix A): gravity along -e3; the cable direction q points from
the quadrotor to the load, x_Q = x_L - l q; -T q = m_L (x_L_ddot + g e3);
variations xi on S^2 with xi . q = 0 and eta on SO(3); an estimate carries a
hat (qhat, omegahat); the innovation is predicted minus measured.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "verify"))
sys.path.insert(0, str(ROOT / "case_studies"))
import robots  # noqa: E402
from common import Checker, random_rotation  # noqa: E402
from robots.core import (E1, E2, E3, I3, attitude_tracking_torque, e_Omega, e_omega, e_q, e_R,  # noqa: E402
                         expm_so3, hat, tangent, unit)
from robots.quadrotor import desired_attitude  # noqa: E402


def cross(a, b):
    """a x b for two vectors of R^3 (np.cross is slow on single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def chapter(name):
    """verify/<name>.py as a module. These four scripts only define functions at import."""
    spec = importlib.util.spec_from_file_location(f"verify_{name}", ROOT / "verify" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V7, V8, V10, V14 = (chapter(n) for n in ("ch07", "ch08", "ch10", "ch14"))

# =====================================================================
# The robot and the mission
# =====================================================================
RB = robots.get("quadrotor_load")
P0 = RB.params                              # what the controller believes: m_L = 0.3 kg
M_L_TRUE = 0.45                             # what was picked up
P_TRUE = replace(P0, m_L=M_L_TRUE)
PEND = robots.get("spherical_pendulum")
BODY = robots.get("rigid_body")
H = 2e-3                                    # time step and sampling period, s
F_MAX = 2.0 * (P0.m_Q + P0.m_L) * P0.g      # thrust limit: twice the believed hover thrust
TAU_MAX = 1.0                               # torque limit per axis, N·m
PSI0 = np.zeros(3)                          # heading psi_d = 0 and its derivatives

# the course of the load: lift, over a wall, set down on a platform
WAYPOINTS = np.array([[0.0, 0.0, 0.3], [1.3, -0.45, 1.1], [2.5, 0.9, 1.3], [3.7, 2.25, 1.05],
                      [5.0, 1.8, 0.5]])
TIME_FRACTIONS = np.array([0.0, 0.314, 0.5, 0.686, 1.0])   # when the load passes each waypoint
T_START, T_TRAVEL, T_END = 0.5, 6.0, 9.0    # wait, travel, end of the run
GUST = (2.5, 5.0)                           # the crosswind blows between these times
WALL = dict(x=2.5, height=1.0, y=(-0.9, 2.7), thick=0.08)
CONE_DEG = 25.0                             # half-angle of the cable cone about -e3
Q_C = -E3
T_MIN = 0.5                                 # the tension the filter keeps, N
ALPHA = (12.0, 12.0)                        # rates alpha_1, alpha_2 of the cone condition, 1/s
T_HURRY = 5.0                               # the same course, asked for in less time
DROP = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]])   # a descent that all but slackens the cable
T_DROP = 1.1


@dataclass
class Gains:
    """Gains of the load cascade for the library's default vehicle (m_Q = 1.5 kg).
    Position loop at 2 rad/s and cable loop at 5 rad/s, as in sec6-cascade. The
    attitude loop is at 20 rad/s about the two light axes."""
    k_x: float = 4.0 * (P0.m_Q + P0.m_L)
    k_v: float = 2.8 * (P0.m_Q + P0.m_L)
    k_q: float = 25.0
    k_omega: float = 7.0
    k_R: float = 8.0
    k_Omega: float = 1.2


class Plan:
    """A curve for the load, at rest before T_START and after T_START + T_TRAVEL,
    with the flat map thm10-load evaluated along it for the believed parameters."""

    def __init__(self, waypoints=WAYPOINTS, travel=T_TRAVEL, start=T_START, params=P0, times=None):
        self.waypoints = np.asarray(waypoints, float)
        self.start, self.travel, self.params = start, travel, params
        if times is None:
            times = (TIME_FRACTIONS * travel if len(self.waypoints) == len(TIME_FRACTIONS)
                     else V10.chord_times(self.waypoints, travel))
        times = np.asarray(times, float)
        self.times = times
        # minimum sixth derivative: pieces of degree eleven, at rest up to the fifth derivative
        self.spline = V10.min_derivative_spline(self.waypoints, times, r=6, method="spline")
        self._cache = {}

    def jet(self, t, order=6):
        s = float(np.clip(t - self.start, 0.0, self.travel))
        jet = self.spline.jet(s, order)
        if not 0.0 <= t - self.start < self.travel:      # at rest outside the travel time
            jet[1:] = 0.0
        return jet

    def flat(self, t):
        """State and inputs of the reference at time t (eq10-load-map, eq10-load-force)."""
        key = round(t / (H / 2))
        on_grid = abs(t - key * (H / 2)) < 1e-9          # sample and stage times are cached
        if on_grid and key in self._cache:
            return self._cache[key]
        o = V10.load_flat(self.jet(t), PSI0, self.params)
        out = dict(x_L=o["x_L"], v_L=o["v_L"], a_L=o["a_L"], q=o["q"], omega=o["omega"],
                   omega_dot=o["omega_dot"], R=o["R"], Omega=o["Omega"], Omega_dot=o["Omega_dot"],
                   f=o["f"], tau=o["tau"], T=o["T"], x_Q=o["x_Q"], F=o["F"])
        if on_grid:
            self._cache[key] = out
        return out

    def state(self, t):
        o = self.flat(t)
        return RB.State(x_L=o["x_L"], v_L=o["v_L"], q=o["q"], omega=o["omega"], R=o["R"], Omega=o["Omega"])


def plan_report(plan, t_end=None, n=1401):
    """Feasibility of a plan, read off the flat map."""
    ts = np.linspace(0.0, t_end or (plan.start + plan.travel), n)
    O = [plan.flat(t) for t in ts]
    swing = np.degrees([np.arccos(np.clip(o["q"] @ Q_C, -1, 1)) for o in O])
    tilt = np.degrees([np.arccos(np.clip(o["R"][2, 2], -1, 1)) for o in O])
    return dict(t=ts, swing=swing, tilt=tilt, T=np.array([o["T"] for o in O]),
                f=np.array([o["f"] for o in O]), tau=np.array([np.abs(o["tau"]).max() for o in O]),
                speed=np.array([np.linalg.norm(o["v_L"]) for o in O]),
                Omega=np.array([np.linalg.norm(o["Omega"]) for o in O]),
                lead=np.array([np.linalg.norm((o["x_Q"] - o["x_L"])[:2]) for o in O]))


# =====================================================================
# The plant: the library model, a different load mass, and a gust
# =====================================================================
class LoadInGust(type(RB)):
    """eq4-load with a force d on the quadrotor, in the world frame. The thrust
    acts at the same point, so d adds to the thrust vector f R e3. The input is
    u = (f, tau, d). With d = 0 this is the library model."""

    def dynamics(self, state, u, params=None):
        p = params or self.params
        f, tau = u[0], u[1]
        d = u[2] if len(u) > 2 else np.zeros(3)
        q, omega = state.q, state.omega
        thrust = f * state.R @ E3 + d
        q_dot = cross(omega, q)
        v_L_dot = (q @ thrust - p.m_Q * p.l * q_dot @ q_dot) * q / (p.m_Q + p.m_L) - p.g * E3
        omega_dot = -cross(q, thrust) / (p.m_Q * p.l)
        Omega_dot = np.linalg.solve(p.J, tau - cross(state.Omega, p.J @ state.Omega))
        return v_L_dot, omega_dot, Omega_dot

    def _rates(self, increments, velocities):
        """The library's `_rates` with a faster cross product (checked against it)."""
        d, v = increments, velocities
        return [v[0], v[1] - 0.5 * cross(d[1], v[1]) + cross(d[1], cross(d[1], v[1])) / 12.0,
                v[2] + 0.5 * cross(d[2], v[2]) + cross(d[2], cross(d[2], v[2])) / 12.0]

    def tension(self, state, u, params=None):
        p = params or self.params
        d = u[2] if len(u) > 2 else np.zeros(3)
        q_dot = cross(state.omega, state.q)
        return p.m_L / (p.m_Q + p.m_L) * (p.m_Q * p.l * q_dot @ q_dot
                                          - state.q @ (u[0] * state.R @ E3 + d))

    def lagrangian(self):
        """The library's Lagrangian, with the virtual work of d at x_Q = x_L - l q added."""
        from geomech import Cross, Dot, Scalar, Vector
        L, dW, variables, inputs = super().lagrangian()
        x_L, q = variables.vectors
        l = Scalar("l", attr=["Constant"])
        d = Vector("d")
        dW = dW + Dot(x_L.delta() - Cross(q.get_variation_vector(), q) * l, d)
        return L, dW, variables, inputs + [d]


PLANT = LoadInGust(P_TRUE)


def gust(t, peak=2.0, t_on=GUST[0], t_off=GUST[1], direction=(0.25, -1.0, 0.0)):
    """The external force F_e on the quadrotor: a crosswind, a smooth pulse of `peak` newtons."""
    if not t_on < t < t_off:
        return np.zeros(3)
    return peak * np.sin(np.pi * (t - t_on) / (t_off - t_on)) ** 2 * unit(direction)


def no_gust(t):
    return np.zeros(3)


# =====================================================================
# Linearization on R^3 x S^2 x SO(3), assembled from the blocks of Chapter 7
# =====================================================================
def thrust_vector(state, f):
    return f * state.R @ E3


def linearize(state, u, p=P0):
    """Variation-based linearization of eq4-load along a solution.

    Linear state s = (delta x_L, delta v_L, xi, delta omega, eta, delta Omega),
    input (delta f, delta tau). Returns A (18 x 18), B (18 x 4) and the
    constraint matrix C (2 x 18) with C s = 0.

    The cable block is thm7-s2 for a pendulum of mass m_Q pushed by the force
    m_Q g e3 - f R e3 (the cable equation m_Q l omega_dot = -q x f R e3 is
    eq7-pendulum with that force). The attitude block is thm7-so3 with rho = 0.
    Both are taken from the robot model library. The coupling is the variation
    of the thrust vector, delta(f R e3) = R e3 delta f - f R hat(e3) eta.
    """
    f, q, omega, R, Omega = u[0], state.q, state.omega, state.R, state.Omega
    m, b3 = p.m_Q + p.m_L, R @ E3
    F = f * b3
    A_c, B_c, C_c = PEND.linearize(q, omega, p.m_Q * p.g * E3 - F, PEND.Params(m=p.m_Q, l=p.l, g=p.g))
    A_a, B_a = BODY.linearize(R, Omega, BODY.Params(J=p.J, m=0.0, rho=np.zeros(3), g=p.g))
    dF_eta, dF_f = -f * R @ hat(E3), b3                    # delta F = dF_eta eta + dF_f delta f
    c = q @ F - p.m_Q * p.l * omega @ omega
    A, B = np.zeros((18, 18)), np.zeros((18, 4))
    A[0:3, 3:6] = I3
    # load: (m_Q + m_L)(v_L_dot + g e3) = (q . F - m_Q l |omega|^2) q
    A[3:6, 6:9] = (np.outer(q, cross(q, F)) - c * hat(q)) / m
    A[3:6, 9:12] = -2 * p.m_Q * p.l * np.outer(q, omega) / m
    A[3:6, 12:15] = np.outer(q, q) @ dF_eta / m
    B[3:6, 0] = np.outer(q, q) @ dF_f / m
    # cable: the pendulum block, driven by -delta F
    A[6:12, 6:12] = A_c
    A[6:12, 12:15] = -B_c @ dF_eta
    B[6:12, 0] = -B_c @ dF_f
    # attitude
    A[12:18, 12:18] = A_a
    B[12:18, 1:4] = B_a
    C = np.zeros((2, 18))
    C[:, 6:12] = C_c
    return A, B, C


def reachable_dimension(A, B, tol=1e-9):
    """Dimension of span [B, A B, A^2 B, ...], built one orthonormal direction at a time
    (the powers of A themselves are too badly scaled for a rank computation)."""
    basis = np.zeros((A.shape[0], 0))
    block = B
    for _ in range(A.shape[0]):
        new = []
        for col in block.T:
            col = col / max(np.linalg.norm(col), 1e-300)
            for _ in range(2):
                col = col - basis @ (basis.T @ col)
            if np.linalg.norm(col) > tol:
                col = col / np.linalg.norm(col)
                basis = np.column_stack([basis, col])
                new.append(col)
        if not new:
            break
        block = A @ np.column_stack(new)
    return basis.shape[1]


def null_basis(C):
    """Orthonormal basis Z of the constraint subspace ker C."""
    _, _, Vt = np.linalg.svd(C)
    return Vt[C.shape[0]:].T


def linear_state(state, ref):
    """The linear state computed from a state and a reference state (tbl7-dictionary):
    (e_x, e_v, e_q, omega - omegad, e_R, e_Omega + Omegad x e_R)."""
    err_R = e_R(state.R, ref.R)
    return np.concatenate([state.x_L - ref.x_L, state.v_L - ref.v_L, e_q(state.q, ref.q),
                           state.omega - ref.omega, err_R,
                           e_Omega(state.R, state.Omega, ref.R, ref.Omega) + cross(ref.Omega, err_R)])


def perturb(state, s, eps=1.0):
    """The state reached from `state` by the finite variation eps s, on the manifold.
    The cable part is the map (+) of eq14-oplus, taken from verify/ch14.py."""
    q, omega = V14.s2_oplus(state.q, state.omega, eps * s[6:12])
    return RB.State(x_L=state.x_L + eps * s[0:3], v_L=state.v_L + eps * s[3:6], q=q, omega=omega,
                    R=state.R @ expm_so3(eps * s[12:15]), Omega=state.Omega + eps * s[15:18])


def variation_between(state, base):
    """The linear state s with base (+) s = state, exactly (inverse of `perturb`)."""
    return np.concatenate([state.x_L - base.x_L, state.v_L - base.v_L,
                           V14.s2_ominus(state.q, state.omega, base.q, base.omega),
                           V14.log_so3(base.R.T @ state.R), state.Omega - base.Omega])


def tangent_start(state, rng, scale=1.0):
    """A random linear state in the constraint subspace at `state`."""
    s = scale * rng.normal(size=18)
    s[6:12] = V14.s2_Pi(state.q, state.omega) @ s[6:12]
    return s


def riccati(plan, Q, R_u, P_f, t0, t1, n_grid=None, p=P0):
    """Integrate eq7-riccati backwards along the plan with the classical fourth-order
    Runge-Kutta method on the sampling grid. Returns the times and the gains K(t)."""
    n = int(round((t1 - t0) / H)) if n_grid is None else n_grid
    hh = (t1 - t0) / n
    Ri = np.linalg.inv(R_u)

    def AB(t):
        o = plan.flat(t)
        A, B, _ = linearize(plan.state(t), (o["f"], o["tau"]), p)
        return A, B

    def rhs(P, A, B):
        return A.T @ P + P @ A - P @ B @ Ri @ B.T @ P + Q

    ts = t0 + hh * np.arange(n + 1)
    P = P_f.copy()
    Ks = np.zeros((n + 1, 4, 18))
    A1, B1 = AB(ts[n])
    Ks[n] = Ri @ B1.T @ P
    for k in range(n, 0, -1):
        Am, Bm = AB(ts[k] - hh / 2)
        A0, B0 = AB(ts[k - 1])
        k1 = rhs(P, A1, B1)
        k2 = rhs(P + hh / 2 * k1, Am, Bm)
        k3 = rhs(P + hh / 2 * k2, Am, Bm)
        k4 = rhs(P + hh * k3, A0, B0)
        P = P + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        P = 0.5 * (P + P.T)
        A1, B1 = A0, B0
        Ks[k - 1] = Ri @ B1.T @ P
    return ts, Ks, P


Q_LQR = np.diag([40.0] * 3 + [8.0] * 3 + [20.0] * 3 + [2.0] * 3 + [10.0] * 3 + [0.2] * 3)
R_LQR = np.diag([0.5, 30.0, 30.0, 30.0])


# =====================================================================
# Control: the cascade eq6-load for a moving reference
# =====================================================================
def cascade_thrust(state, ref, k, p=P0, feedforward=True):
    """The thrust vector the load cascade asks for (eq6-load), for a moving
    reference `ref` (a dict from Plan.flat). With feedforward=True the cable
    law eq6-law-s2 uses the reference's omegad and its rate from the flat map.
    Returns F and the cable direction qd that the position loop asks for."""
    q, omega = state.q, state.omega
    m = p.m_Q + p.m_L
    F_L = -k.k_x * (state.x_L - ref["x_L"]) - k.k_v * (state.v_L - ref["v_L"]) + m * (ref["a_L"] + p.g * E3)
    A = F_L + p.m_Q * p.l * (omega @ omega) * q
    qd = -unit(A)
    omegad, omegad_dot = (ref["omega"], ref["omega_dot"]) if feedforward else (np.zeros(3), np.zeros(3))
    omega_dot = (tangent(q, omegad_dot) - (q @ omegad) * cross(omega, q)
                 - k.k_q * e_q(q, qd) - k.k_omega * e_omega(q, omega, omegad))      # eq6-law-s2
    return (A @ q) * q + p.m_Q * p.l * cross(q, omega_dot), qd


def attitude_stage(state, F, ref, k, p=P0, feedforward=True):
    """eq6-Rd and eq6-law: thrust along the body axis and the torque that turns it."""
    Rd = desired_attitude(F, E1)
    Omegad, Omegad_dot = (ref["Omega"], ref["Omega_dot"]) if feedforward else (np.zeros(3), np.zeros(3))
    tau = attitude_tracking_torque(state.R, state.Omega, Rd, Omegad, Omegad_dot, p.J, k.k_R, k.k_Omega)
    return F @ (state.R @ E3), tau


def bare_quadrotor_thrust(state, ref, k, p=P0):
    """A controller that does not know about the cable: eq6-Fd for the quadrotor,
    sent one cable length above the load's reference, carrying the believed
    total weight. The swing of the load is a disturbance to it."""
    m = p.m_Q + p.m_L
    x_Q = state.x_L - p.l * state.q
    v_Q = state.v_L - p.l * cross(state.omega, state.q)
    return -k.k_x * (x_Q - (ref["x_L"] + p.l * E3)) - k.k_v * (v_Q - ref["v_L"]) + m * (ref["a_L"] + p.g * E3)


# =====================================================================
# Safety: the cable cone (eq8-cone, eq8-rd2) and the tension (eq4-load-tension)
# =====================================================================
def cone_terms(q, omega, p=P0, q_c=Q_C, theta_max=np.radians(CONE_DEG)):
    """H, H_dot and (a, b) with H_ddot = a + b . F for the thrust vector F.
    This is eq8-cone-ddot for the cable: mass m_Q, and F - m g e3 replaced by -F."""
    H = q @ q_c - np.cos(theta_max)
    H_dot = omega @ cross(q, q_c)
    tang = q_c - (q @ q_c) * q
    return H, H_dot, -(omega @ omega) * (q @ q_c), -tang / (p.m_Q * p.l)


def safety_filter(state, F, p=P0, alpha=ALPHA, theta_max=np.radians(CONE_DEG), T_min=T_MIN,
                  Delta=None, cone=True, tension=True):
    """The thrust vector closest to F that satisfies the cone condition eq8-rd2 and
    keeps the tension of eq4-load-tension above T_min. Both are linear in F.
    Delta is an estimate of the lumped uncertainty, added to F in the model."""
    q, omega = state.q, state.omega
    Delta = np.zeros(3) if Delta is None else Delta
    rows, rhs = [], []
    if cone:
        Hc, Hd, a, b = cone_terms(q, omega, p, theta_max=theta_max)
        a1, a2 = alpha
        rows.append(-b)                                    # a + b.(F + Delta) + (a1+a2) Hd + a1 a2 H >= 0
        rhs.append(a + b @ Delta + (a1 + a2) * Hd + a1 * a2 * Hc)
    if tension:
        rows.append(q)                                     # q.(F + Delta) <= m_Q l |omega|^2 - (m/m_L) T_min
        rhs.append(p.m_Q * p.l * omega @ omega - (p.m_Q + p.m_L) / p.m_L * T_min - q @ Delta)
    if not rows:
        return F, False
    A, bb = np.array(rows), np.array(rhs)
    if np.all(A @ F <= bb):
        return F, False
    z = V8.solve_qp(np.eye(3), -F, A, bb)
    return (F if z is None else z), True


def tension_floor(state, f, p=P0, T_min=T_MIN):
    """The smallest thrust for which eq4-load-tension gives at least T_min, at the
    attitude the vehicle has now. The thrust enters the tension directly."""
    c = -state.q @ (state.R @ E3)
    if c <= 1e-3:
        return f
    need = ((p.m_Q + p.m_L) / p.m_L * T_min - p.m_Q * p.l * state.omega @ state.omega) / c
    return max(f, need)


# =====================================================================
# Uncertainty: a lumped force and its estimate
# =====================================================================
@dataclass
class AdaptGains:
    k_Delta: float = 2000.0       # adaptation gain
    k_f: float = 6.0              # cutoff, rad/s
    k_pred: float = 10.0          # gain on the prediction errors, 1/s
    Delta_max: float = 6.0        # radius of the projection ball, N


class Adaptation:
    """A state predictor for the load velocity and the cable angular velocity,
    an adaptation law with the projection of eq8-proj, and the low-pass filter
    eq8-filter. The lumped uncertainty Delta is a force in the world frame that
    adds to the thrust vector."""

    def __init__(self, state, gains=None, p=P0):
        self.k, self.p = gains or AdaptGains(), p
        self.v_p, self.omega_p = state.v_L.copy(), state.omega.copy()
        self.Deltahat, self.Delta_f = np.zeros(3), np.zeros(3)

    def errors(self, state):
        return self.v_p - state.v_L, self.omega_p - state.omega

    def direction(self, state):
        """y of the adaptation law: -(q q^T) v_tilde + l omega_tilde x q."""
        v_t, om_t = self.errors(state)
        return -(state.q @ v_t) * state.q + self.p.l * cross(om_t, state.q)

    def rates(self, state, F_applied):
        p, k, q = self.p, self.k, state.q
        m = p.m_Q + p.m_L
        v_t, om_t = self.errors(state)
        Fe = F_applied + self.Deltahat
        v_p_dot = (q @ Fe - p.m_Q * p.l * state.omega @ state.omega) * q / m - p.g * E3 - k.k_pred * v_t
        omega_p_dot = -cross(q, Fe) / (p.m_Q * p.l) - k.k_pred * om_t
        return v_p_dot, omega_p_dot

    def step(self, state, F_applied, h=H):
        k = self.k
        y = self.direction(state)
        v_p_dot, omega_p_dot = self.rates(state, F_applied)
        self.v_p = self.v_p + h * v_p_dot
        self.omega_p = self.omega_p + h * omega_p_dot
        self.Deltahat = self.Deltahat + h * k.k_Delta * V8.proj(self.Deltahat, y, k.Delta_max)
        norm = np.linalg.norm(self.Deltahat)
        if norm > k.Delta_max:
            self.Deltahat *= k.Delta_max / norm
        self.Delta_f = self.Delta_f + (1.0 - np.exp(-k.k_f * h)) * (self.Deltahat - self.Delta_f)


def lumped_uncertainty(v_L_dot, d, p_true=P_TRUE, p=P0):
    """Delta = d - (m_L - m_L0)(v_L_dot + g e3): what must be added to the thrust
    vector in the believed model to reproduce the true motion."""
    return d - (p_true.m_L - p.m_L) * (v_L_dot + p.g * E3)


# =====================================================================
# Estimation: the variation-based EKF of def14-vekf for the cable
# =====================================================================
def cable_rhs(q, omega, F, p=P0):
    """q_dot and omega_dot of the cable under the thrust vector F (second line of eq4-load)."""
    return cross(omega, q), -cross(q, F) / (p.m_Q * p.l)


def cable_step(q, omega, F, h, p=P0):
    """One fourth-order step of the cable with F held, then back onto the sphere."""
    def f(z):
        a, b = cable_rhs(z[:3], z[3:], F, p)
        return np.concatenate([a, b])
    z = np.concatenate([q, omega])
    k1 = f(z)
    k2 = f(z + h / 2 * k1)
    k3 = f(z + h / 2 * k2)
    k4 = f(z + h * k3)
    z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    q = unit(z[:3])
    return q, tangent(q, z[3:])


def cable_A(q, omega, F, p=P0):
    """A of eq7-s2-lin for the cable: the library's pendulum block."""
    return PEND.linearize(q, omega, p.m_Q * p.g * E3 - F, PEND.Params(m=p.m_Q, l=p.l, g=p.g))[0]


@dataclass
class FilterNoise:
    sig_y: float = np.sqrt(1e-3)             # reading of l q, per coordinate, m (ex14-pendulum)
    q_xi: float = 1e-9                       # process noise per step, direction
    q_omega: float = 2e-5                    # process noise per step, angular velocity
    sig0_q: float = 0.3                      # initial standard deviation, rad
    sig0_omega: float = 0.5                  # rad/s
    every: int = 5                           # a reading every `every` steps: 100 Hz


class CableFilter:
    """def14-vekf with the cable equation in place of the free pendulum.
    The quadrotor's attitude and thrust are taken as known."""

    def __init__(self, qhat, omegahat, noise=None, p=P0):
        self.n, self.p = noise or FilterNoise(), p
        self.qhat, self.omegahat = unit(qhat), tangent(unit(qhat), omegahat)
        Pi = V14.s2_Pi(self.qhat, self.omegahat)
        S0 = np.diag([self.n.sig0_q**2] * 3 + [self.n.sig0_omega**2] * 3)
        self.Sigma = Pi @ S0 @ Pi.T
        self.Sigma_n = np.diag([self.n.q_xi] * 3 + [self.n.q_omega] * 3)
        self.Sigma_y = np.eye(3) * self.n.sig_y**2

    def propagate(self, F, h=H):
        A_k = V14.expm_series(cable_A(self.qhat, self.omegahat, F, self.p) * h)
        self.qhat, self.omegahat = cable_step(self.qhat, self.omegahat, F, h, self.p)
        Pi = V14.s2_Pi(self.qhat, self.omegahat)
        self.Sigma = Pi @ (A_k @ self.Sigma @ A_k.T + self.Sigma_n) @ Pi.T

    def correct(self, y):
        l = self.p.l
        C_y = np.hstack([-l * hat(self.qhat), np.zeros((3, 3))])
        ytilde = l * self.qhat - y
        S = self.Sigma
        Ktilde = S @ C_y.T @ np.linalg.inv(C_y @ S @ C_y.T + self.Sigma_y)
        s = -Ktilde @ ytilde
        IK = np.eye(6) - Ktilde @ C_y
        S = IK @ S @ IK.T + Ktilde @ self.Sigma_y @ Ktilde.T
        turn = expm_so3(s[:3])
        self.qhat, self.omegahat = V14.s2_oplus(self.qhat, self.omegahat, s)
        Tm = np.zeros((6, 6))
        Tm[:3, :3] = turn
        Tm[3:, 3:] = turn
        Pi = V14.s2_Pi(self.qhat, self.omegahat)
        self.Sigma = Pi @ Tm @ S @ Tm.T @ Pi.T
        return s

    def nees(self, q, omega):
        err = V14.s2_ominus(self.qhat, self.omegahat, q, omega)
        return float(err @ np.linalg.pinv(self.Sigma, rcond=1e-10) @ err) / 4


# =====================================================================
# The closed loop
# =====================================================================
@dataclass
class Options:
    controller: str = "cascade"      # "cascade", "lqr" or "bare"
    feedforward: bool = True         # the flat feedforward: omegad, Omegad and their rates
    feedforward_attitude: bool | None = None   # None: the same as `feedforward`
    adapt: bool = True
    safe: bool = True                # cone and tension filter
    estimate: bool = True            # run the filter on the readings of the cable
    use_estimate: bool = True        # and let the controller see its estimate, not the truth
    gust: bool = True
    m_L: float = M_L_TRUE
    cone_deg: float = CONE_DEG
    t_end: float = T_END
    seed: int = 18
    saturate: bool = True
    qhat0: np.ndarray | None = None  # initial estimate of the cable direction
    state0: object = None            # initial state (default: the plan's)
    gust_fn: object = None
    adapt_gains: AdaptGains | None = None
    noise: FilterNoise | None = None
    barrier_uses_estimate: bool = True
    filter_uses_estimate: bool = True


def run(plan, opt=None, gains=None, lqr=None):
    """Fly the plan. Returns a dict of logged arrays, one row per step."""
    opt, k = opt or Options(), gains or Gains()
    rng = np.random.default_rng(opt.seed)
    plant = LoadInGust(replace(P0, m_L=opt.m_L))
    pt = plant.params
    n = int(round(opt.t_end / H))
    state = opt.state0 if opt.state0 is not None else plan.state(0.0)
    theta_max = np.radians(opt.cone_deg)
    wind = (opt.gust_fn or gust) if opt.gust else no_gust
    noise = opt.noise or FilterNoise()
    filt = CableFilter(opt.qhat0 if opt.qhat0 is not None else state.q, np.zeros(3), noise)
    adapt = Adaptation(state, opt.adapt_gains)
    log = {key: [] for key in ("t", "x_L", "x_d", "q", "qd", "qhat", "omega", "omegahat", "R", "x_Q", "f",
                               "tau", "T", "H", "H1", "Deltahat", "Delta_f", "Delta", "d", "active", "manifold",
                               "nees", "v_tilde", "omega_tilde", "e_x", "swing", "F_cmd", "F_safe", "sat",
                               "T_ref", "qd_plan", "e_qhat", "e_omegahat", "Sigma_q", "v_L", "Omega", "filter_res", "y_err")}
    for i in range(n + 1):
        t = i * H
        ref = plan.flat(t)
        # --- the reading of the cable, and the state the controller sees
        y_err = np.nan
        if opt.estimate and i % noise.every == 0:
            y = P0.l * state.q + noise.sig_y * rng.normal(size=3)
            y_err = np.degrees(np.arccos(np.clip(unit(y) @ state.q, -1, 1)))
            qhat_before = filt.qhat
            filt.correct(y)
            if i == 0:
                first_error = np.degrees(np.arccos(np.clip(qhat_before @ state.q, -1, 1)))
        if opt.estimate and opt.use_estimate:
            x_Q = state.x_L - pt.l * state.q                      # the quadrotor's own state: known
            v_Q = state.v_L - pt.l * cross(state.omega, state.q)
            seen = RB.State(x_L=x_Q + P0.l * filt.qhat, v_L=v_Q + P0.l * cross(filt.omegahat, filt.qhat),
                            q=filt.qhat, omega=filt.omegahat, R=state.R, Omega=state.Omega)
        else:
            seen = state
        # --- the controller
        active = False
        if opt.controller == "lqr":
            s = linear_state(seen, plan.state(t))
            du = -lqr[min(i, len(lqr) - 1)] @ s
            f, tau = ref["f"] + du[0], ref["tau"] + du[1:]
            F = f * seen.R @ E3
            qd = ref["q"]
        else:
            if opt.controller == "bare":
                F, qd = bare_quadrotor_thrust(seen, ref, k), ref["q"]
            else:
                F, qd = cascade_thrust(seen, ref, k, feedforward=opt.feedforward)
            if opt.adapt:
                F = F - adapt.Delta_f
            F_cmd = F
            if opt.safe:
                F, active = safety_filter(seen, F, theta_max=theta_max,
                                          Delta=adapt.Delta_f if (opt.adapt and opt.barrier_uses_estimate)
                                          else None)
            ff_att = opt.feedforward if opt.feedforward_attitude is None else opt.feedforward_attitude
            f, tau = attitude_stage(seen, F, ref, k, feedforward=ff_att)
            if opt.safe:
                f = tension_floor(seen, f)
        if opt.controller == "lqr":
            F_cmd = F
        sat = False
        if opt.saturate:
            f2, tau2 = float(np.clip(f, 0.0, F_MAX)), np.clip(tau, -TAU_MAX, TAU_MAX)
            sat = (f2 != f) or bool(np.any(tau2 != tau))
            f, tau = f2, tau2
        d = wind(t)
        u = (f, tau, d)
        # --- log
        acc = plant.dynamics(state, u)
        Hc, Hd, _, _ = cone_terms(state.q, state.omega, theta_max=theta_max)
        v_t, om_t = adapt.errors(seen)
        row = dict(t=t, x_L=state.x_L, x_d=ref["x_L"], q=state.q, qd=qd, qhat=filt.qhat, omega=state.omega,
                   omegahat=filt.omegahat, R=state.R, x_Q=state.x_L - pt.l * state.q, f=f, tau=tau,
                   T=plant.tension(state, u), H=Hc, H1=Hd + ALPHA[0] * Hc, Deltahat=adapt.Deltahat.copy(),
                   Delta_f=adapt.Delta_f.copy(), Delta=lumped_uncertainty(acc[0], d, pt), d=d, active=active,
                   manifold=plant.on_manifold(state), nees=(filt.nees(state.q, state.omega) if (opt.estimate and i % noise.every == 0)
                                                           else np.nan),
                   v_tilde=np.linalg.norm(v_t), omega_tilde=np.linalg.norm(om_t),
                   e_x=np.linalg.norm(state.x_L - ref["x_L"]),
                   swing=np.degrees(np.arccos(np.clip(state.q @ Q_C, -1, 1))), F_cmd=F_cmd, F_safe=F, sat=sat,
                   T_ref=ref["T"], qd_plan=ref["q"],
                   e_qhat=np.degrees(np.arccos(np.clip(filt.qhat @ state.q, -1, 1))),
                   e_omegahat=np.linalg.norm(filt.omegahat - state.omega),
                   Sigma_q=np.sqrt(max(np.trace(filt.Sigma[:3, :3]), 0.0)), v_L=state.v_L, Omega=state.Omega,
                   y_err=y_err, filter_res=max(abs(np.linalg.norm(filt.qhat) - 1.0), abs(filt.qhat @ filt.omegahat),
                                  np.abs(V14.s2_C(filt.qhat, filt.omegahat) @ filt.Sigma).max()))
        for key, val in row.items():
            log[key].append(val)
        if i == n or not np.all(np.isfinite(state.x_L)) or np.linalg.norm(state.x_L - ref["x_L"]) > 20.0:
            break
        # --- adaptation and filter see the thrust that was applied
        F_applied = f * state.R @ E3
        if opt.adapt:
            adapt.step(seen, F_applied)
        if opt.estimate:
            filt.propagate(F_applied + (adapt.Delta_f if (opt.adapt and opt.filter_uses_estimate) else 0.0))
        state = plant.step(state, u, H)
    out = {key: np.array(val) for key, val in log.items()}
    out["final"] = state
    out["first_error"] = first_error if opt.estimate else np.nan
    out["completed"] = len(out["t"]) == n + 1
    return out


def wall_clearance(o):
    """Smallest height of the load above the top of the wall while it is over it, m."""
    x = o["x_L"]
    near = np.abs(x[:, 0] - WALL["x"]) < 0.15
    return float((x[near, 2] - WALL["height"]).min()) if near.any() else float("nan")


def settle_time(o, tol=0.05):
    """Time after which the load stays within `tol` of its reference."""
    late = o["e_x"] > tol
    if not late.any():
        return 0.0
    return float(o["t"][min(len(late) - 1, len(late) - int(np.argmax(late[::-1])))])


def summarize(o, plan):
    """The numbers of the mission report and of the ablation table."""
    t = o["t"]
    t_set = plan.start + plan.travel
    travel = (t >= plan.start) & (t <= t_set)
    after = t >= t_set
    i_set = int(np.argmin(np.abs(t - t_set)))
    seen = t >= 1.0
    return dict(
        completed=bool(o["completed"]),
        err_rms_cm=float(100 * np.sqrt(np.mean(o["e_x"][travel] ** 2))) if travel.any() else float("nan"),
        err_max_cm=float(100 * o["e_x"].max()),
        err_setdown_cm=float(100 * o["e_x"][min(i_set, len(t) - 1)]),
        err_final_cm=float(100 * o["e_x"][-1]),
        swing_setdown_deg=float(o["swing"][min(i_set, len(t) - 1)]),
        swing_after_deg=float(o["swing"][after].max()) if after.any() else float("nan"),
        speed_setdown=float(np.linalg.norm(o["v_L"][min(i_set, len(t) - 1)])),
        swing_max_deg=float(o["swing"].max()),
        clearance_cm=100 * wall_clearance(o),
        T_min=float(o["T"].min()), H_min=float(o["H"].min()),
        f_max=float(o["f"].max()), f_min=float(o["f"].min()), tau_max=float(np.abs(o["tau"]).max()),
        est_rms_deg=float(np.sqrt(np.mean(o["e_qhat"][seen] ** 2))),
        est_omega_rms=float(np.sqrt(np.mean(o["e_omegahat"][seen] ** 2))),
        nees=float(np.nanmean(o["nees"][seen])) if np.isfinite(o["nees"][seen]).any() else float("nan"),
        active_frac=float(np.mean(o["active"])), saturated_frac=float(np.mean(o["sat"])),
        manifold=float(o["manifold"].max()), filter_res=float(o["filter_res"].max()))


# =====================================================================
# Every simulation the chapter reports
# =====================================================================
ABLATION = [("full", "Everything on"), ("no_ff", "No flat feedforward"), ("no_adapt", "No adaptation"),
            ("no_filter", "No cone and tension filter"), ("truth", "Truth in place of the estimate"),
            ("bare", "Load treated as a disturbance")]


def swing_start(state, deg, axis=E1):
    """The same state with the cable turned by `deg` about `axis`, and the load moved."""
    return RB.State(x_L=state.x_L + np.array([0.3, -0.2, 0.1]), v_L=state.v_L,
                    q=expm_so3(np.radians(deg) * axis) @ state.q, omega=state.omega, R=state.R,
                    Omega=state.Omega)


def all_runs(verbose=False):
    """Fly everything. Returns the plans, the LQR gains and a dict of logs."""
    plan, hurry = Plan(), Plan(travel=T_HURRY)
    drop = Plan(waypoints=DROP, travel=T_DROP)
    t_lqr, K_lqr, _ = riccati(plan, Q_LQR, R_LQR, Q_LQR, 0.0, T_END)
    ideal = dict(adapt=False, safe=False, estimate=False, gust=False, m_L=P0.m_L)
    known = dict(use_estimate=False)
    wrong = expm_so3(np.radians(40.0) * unit(np.array([1.0, 1.0, 0.0]))) @ -E3
    todo = {
        # the mission and its ablations
        "full": (plan, Options()),
        "no_ff": (plan, Options(feedforward=False)),
        "no_adapt": (plan, Options(adapt=False)),
        "no_filter": (plan, Options(safe=False)),
        "truth": (plan, Options(**known)),
        "bare": (plan, Options(controller="bare", adapt=False, safe=False)),
        # the model as believed: no gust, the right mass, the true state
        "nominal": (plan, Options(**ideal)),
        "nominal_no_ff": (plan, Options(feedforward=False, **ideal)),
        "nominal_cable_ff": (plan, Options(feedforward=True, feedforward_attitude=False, **ideal)),
        "lqr_nominal": (plan, Options(controller="lqr", **ideal)),
        # uncertainty
        "plain": (plan, Options(adapt=False, safe=False, **known)),
        "kD_500": (plan, Options(safe=False, t_end=6.0, adapt_gains=AdaptGains(k_Delta=500.0), **known)),
        "kD_125": (plan, Options(safe=False, t_end=6.0, adapt_gains=AdaptGains(k_Delta=125.0), **known)),
        "kD_8000": (plan, Options(safe=False, t_end=6.0, adapt_gains=AdaptGains(k_Delta=8000.0), **known)),
        # safety: a hurried course and a steep descent
        "hurry_free": (hurry, Options(safe=False, gust=False, t_end=9.5, **known)),
        "hurry_safe": (hurry, Options(gust=False, t_end=9.5, **known)),
        "hurry_free_est": (hurry, Options(safe=False, gust=False, t_end=9.5)),
        "hurry_safe_est": (hurry, Options(gust=False, t_end=9.5)),
        "hurry_safe_no_ff": (hurry, Options(gust=False, t_end=9.5, feedforward=False, **known)),
        "drop_free": (drop, Options(safe=False, gust=False, t_end=4.0, **known)),
        "drop_safe": (drop, Options(gust=False, t_end=4.0, **known)),
        "drop_free_est": (drop, Options(safe=False, gust=False, t_end=4.0)),
        "drop_safe_est": (drop, Options(gust=False, t_end=4.0)),
        # estimation: a wrong first guess
        "est_wrong": (plan, Options(t_end=3.0, qhat0=wrong, **known)),
    }
    for deg in (10, 30, 50):
        start = swing_start(plan.state(0.0), deg)
        todo[f"lqr_off{deg}"] = (plan, Options(controller="lqr", state0=start, t_end=7.0, **ideal))
        todo[f"cas_off{deg}"] = (plan, Options(state0=start, t_end=7.0, **ideal))
    runs = {}
    for name, (pl, opt) in todo.items():
        t0 = time.time()
        runs[name] = run(pl, opt, lqr=K_lqr)
        if verbose:
            print(f"    {name:18s} {time.time() - t0:5.1f} s")
    return dict(plan=plan, hurry=hurry, drop=drop, runs=runs,
                summary={name: summarize(o, todo[name][0]) for name, o in runs.items()})


CACHE = ROOT / "verify" / "ch18_cache.npz"
KEEP = ("t", "x_L", "x_d", "q", "qd_plan", "qhat", "R", "x_Q", "f", "T", "H", "Delta_f", "Delta", "d", "e_x",
        "swing", "e_qhat", "e_omegahat", "y_err", "active")


def compact(data, every=10):
    """What the figures draw: a few logged arrays of each run, every `every` steps
    (50 samples a second), in single precision. A reading error that falls
    between two kept samples is moved to the kept one."""
    out = {}
    for name, o in data["runs"].items():
        n = len(o["t"])
        idx = np.arange(0, n, every)
        for key in KEEP:
            a = np.asarray(o[key], float)
            if key == "y_err":
                a = np.array([np.nanmean(a[i:i + every]) if np.isfinite(a[i:i + every]).any() else np.nan
                              for i in idx])
            elif key == "active":
                a = np.array([a[i:i + every].max() for i in idx])
            else:
                a = a[idx]
            out[f"{name}/{key}"] = a.astype(np.float32)
    return out


def _source_hash():
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def save_runs(data):
    np.savez_compressed(CACHE, source=np.array(_source_hash()), **compact(data))


def load_runs():
    """The flights for the figures: {run: {key: array}}, sampled 50 times a second.
    Read from the cache that the last run of this file left, or flown again when
    there is no cache or this file has changed since."""
    if CACHE.exists():
        with np.load(CACHE) as z:
            if str(z["source"]) == _source_hash():
                flat = {k: z[k].astype(float) for k in z.files if k != "source"}
            else:
                flat = None
    else:
        flat = None
    if flat is None:
        data = all_runs(verbose=True)
        save_runs(data)
        flat = {k: v.astype(float) for k, v in compact(data).items()}
    runs = {}
    for key, val in flat.items():
        name, field_ = key.split("/")
        runs.setdefault(name, {})[field_] = val
    return runs


def plans():
    return Plan(), Plan(travel=T_HURRY), Plan(waypoints=DROP, travel=T_DROP)


# =====================================================================
# Checks
# =====================================================================
def geomech_models():
    """The library model derived again, with and without the gust, and its linearization."""
    from geomech import check_eom, compute_eom, linearize as gm_linearize, to_standard_form
    from geomech.codegen import StateLayout, collect_params, load_module, module_source
    out = {}
    L, dW, variables, inputs = RB.lagrangian()
    eom = compute_eom(L, dW, variables)
    out["check_eom"] = float(check_eom(L, dW, variables, eom, inputs=inputs))
    layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
    out["lin"] = load_module(module_source(gm_linearize(eom, variables, inputs), layout), "ch18_lin")
    out["layout"] = layout
    out["sf"] = load_module(module_source(to_standard_form(eom, variables, inputs), layout, lagrangian=L), "ch18_sf")
    Lg, dWg, vg, ig = PLANT.lagrangian()
    eom_g = compute_eom(Lg, dWg, vg)
    out["check_eom_gust"] = float(check_eom(Lg, dWg, vg, eom_g, inputs=ig))
    lay_g = StateLayout(vg, ig, collect_params([Lg, dWg], vg, ig))
    out["gust"] = load_module(module_source(to_standard_form(eom_g, vg, ig), lay_g, lagrangian=Lg), "ch18_gust")
    out["layout_gust"] = lay_g
    return out


def gm_params(p):
    return {"J": p.J, "e3": E3, "g": p.g, "l": p.l, "m_L": p.m_L, "m_Q": p.m_Q}


# geomech orders the linear state (dx_L, xi, eta, dv_L, domega, dOmega); this file
# orders it (dx_L, dv_L, xi, domega, eta, dOmega)
GM_ORDER = np.r_[0:3, 6:9, 12:15, 3:6, 9:12, 15:18]


def linear_flow(plan, s0, t0, t1, du=None, p=P0):
    """Integrate s_dot = A s + B du along the plan with the fourth-order Runge-Kutta method."""
    n = int(round((t1 - t0) / H))
    du = du or (lambda t: np.zeros(4))

    def rhs(t, s):
        o = plan.flat(t)
        A, B, _ = linearize(plan.state(t), (o["f"], o["tau"]), p)
        return A @ s + B @ du(t)
    s, out = s0.copy(), [s0.copy()]
    for k in range(n):
        t = t0 + k * H
        k1 = rhs(t, s)
        k2 = rhs(t + H / 2, s + H / 2 * k1)
        k3 = rhs(t + H / 2, s + H / 2 * k2)
        k4 = rhs(t + H, s + H * k3)
        s = s + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(s.copy())
    return np.array(out)


def open_loop(plan, state, t0, t1, du=None, eps=0.0):
    """Fly the plan's inputs open loop on the library model from `state`, the input
    evaluated at every stage (hold=False). du(t) is a perturbation (delta f, delta tau)."""
    def u(t, s):
        o = plan.flat(t)
        if du is None:
            return o["f"], o["tau"]
        d = eps * du(t)
        return o["f"] + d[0], o["tau"] + d[1:]
    return RB.rollout(state, u, H, int(round((t1 - t0) / H)), hold=False, t0=t0)


def finite_difference_check(plan, rng, t0=3.0, t1=4.0):
    """Perturb the state and the input by +-eps on the manifold, fly both neighbors
    open loop, and compare their centered difference with the linear flow."""
    base = plan.state(t0)
    s0 = tangent_start(base, rng)
    w = rng.normal(size=4) * np.array([2.0, 0.05, 0.05, 0.05])

    def du(t):
        return w * np.sin(3.0 * (t - t0) + 0.4)
    lin = linear_flow(plan, s0, t0, t1, du)
    ref = open_loop(plan, base, t0, t1)
    out = {}
    for eps in (1e-3, 1e-4):
        plus = open_loop(plan, perturb(base, s0, eps), t0, t1, du, eps)
        minus = open_loop(plan, perturb(base, s0, -eps), t0, t1, du, -eps)
        fd = np.array([(variation_between(a, r) - variation_between(b, r)) / (2 * eps)
                       for a, b, r in zip(plus, minus, ref)])
        out[eps] = float(np.abs(fd - lin).max() / np.abs(lin).max())
    out["constraint"] = float(max(np.abs(linearize(plan.state(t0 + k * H), (0.0, np.zeros(3)))[2] @ s).max()
                                  for k, s in enumerate(lin)) / np.abs(lin).max())
    return out


def gramian_rank(plan, t0, t1, n=60):
    """Singular values of the reachability Gramian of the linearization over [t0, t1],
    restricted to the constraint subspace at t1."""
    hh = (t1 - t0) / n
    Phi = np.eye(18)                                   # transition matrix from t to t1, built backwards
    W = np.zeros((18, 18))
    for k in range(n, 0, -1):
        t = t0 + (k - 0.5) * hh
        o = plan.flat(round(t / H) * H)
        A, B, _ = linearize(plan.state(round(t / H) * H), (o["f"], o["tau"]))
        G = Phi @ B
        W += hh * G @ G.T
        Phi = Phi @ V14.expm(A * hh)
    Z = null_basis(linearize(plan.state(t1), (0.0, np.zeros(3)))[2])
    return np.linalg.svd(Z.T @ W @ Z, compute_uv=False), np.linalg.svd(W, compute_uv=False)


def main():
    from scipy.linalg import solve_continuous_are
    ck = Checker("ch18")
    rng = np.random.default_rng(18)
    t_all = time.time()
    plan = Plan()
    p = P0
    m0 = p.m_Q + p.m_L
    k = Gains()
    ck.record("params", dict(m_Q=p.m_Q, m_L=p.m_L, l=p.l, m_L_true=M_L_TRUE, h_ms=1e3 * H,
                             f_hover=m0 * p.g, f_hover_true=(p.m_Q + M_L_TRUE) * p.g, f_max=F_MAX,
                             tau_max=TAU_MAX, cone_deg=CONE_DEG, T_min=T_MIN, alpha=list(ALPHA),
                             travel=T_TRAVEL, start=T_START, end=T_END, hurry=T_HURRY, drop=T_DROP,
                             wall_height=WALL["height"], gust_peak=2.0, gust=list(GUST),
                             weight_true=M_L_TRUE * p.g, weight_believed=p.m_L * p.g))
    ck.record("gains", dict(k_x=k.k_x, k_v=k.k_v, k_q=k.k_q, k_omega=k.k_omega, k_R=k.k_R, k_Omega=k.k_Omega))

    # ------------------------------------------------ 1. model and symbolic check
    gm = geomech_models()
    ck.true("geomech check_eom, library model, below 1e-5", gm["check_eom"] < 1e-5, f"{gm['check_eom']:.1e}")
    ck.true("geomech check_eom, model with a gust on the quadrotor, below 1e-5", gm["check_eom_gust"] < 1e-5,
            f"{gm['check_eom_gust']:.1e}")
    ck.record("check_eom", gm["check_eom"])
    from geomech.simulation import accelerations
    w_lib = w_step = w_gm = 0.0
    for _ in range(20):
        s = RB.random_state(rng)
        u = (5 + 20 * rng.random(), rng.normal(size=3) * 0.3)
        d = rng.normal(size=3) * 2.0
        a_mine = np.concatenate(PLANT.dynamics(s, u, P0))
        w_lib = max(w_lib, np.abs(a_mine - np.concatenate(RB.dynamics(s, u, P0))).max())
        s1, s2 = PLANT.step(s, u, H, P0), RB.step(s, u, H, P0)
        w_step = max(w_step, max(np.abs(getattr(s1, f) - getattr(s2, f)).max()
                                 for f in ("x_L", "v_L", "q", "omega", "R", "Omega")))
        xs = gm["layout_gust"].pack(x_L=s.x_L, q=s.q, R=s.R, x_L_dot=s.v_L, omega_q=s.omega, Omega_R=s.Omega)
        a_g = accelerations(gm["gust"], xs, np.concatenate([[u[0]], u[1], d]), gm_params(P_TRUE))
        w_gm = max(w_gm, np.abs(a_g - np.concatenate(PLANT.dynamics(s, (u[0], u[1], d), P_TRUE))).max())
    ck.close("plant without gust equals the library dynamics", w_lib, 0, 1e-12)
    ck.close("plant step equals the library step (faster cross product)", w_step, 0, 1e-12)
    ck.close("plant with gust and the true load mass equals the geomech model", w_gm, 0, 1e-9)
    hover = RB.State()
    ck.close("hover tension is the load's weight (eq4-load-tension)", RB.tension(hover, RB.hover_input()),
             p.m_L * p.g, 1e-12)

    # ------------------------------------------------ 2. the plan and the flat map
    rep = plan_report(plan)
    hurry, drop = Plan(travel=T_HURRY), Plan(waypoints=DROP, travel=T_DROP)
    rep_h, rep_d = plan_report(hurry), plan_report(drop)
    w = 0.0
    for t in np.linspace(0.6, T_START + T_TRAVEL - 0.1, 30):
        o = plan.flat(t)
        w = max(w, abs(RB.tension(plan.state(t), (o["f"], o["tau"])) - o["T"]))
    ck.close("tension of eq10-load-map equals eq4-load-tension along the plan", w, 0, 1e-10)
    for jet_order in (5,):
        a = plan.jet(T_START + 1e-9)[1:6]
        b = plan.jet(T_START + T_TRAVEL - 1e-9)[1:6]
        ck.close("the curve is at rest up to the fifth derivative at both ends", np.r_[a.ravel(), b.ravel()], 0, 1e-6)
    interior = plan.times[1:-1]
    jump = max(np.abs(plan.spline.jet_side(i, 1.0, 10) - plan.spline.jet_side(i + 1, 0.0, 10))[:11].max()
               for i in range(len(interior)))
    ck.close("the curve is continuous up to the tenth derivative at the waypoints", jump, 0, 1e-6)
    flight = open_loop(plan, plan.state(T_START), T_START, T_START + T_TRAVEL)
    ol = float(np.linalg.norm(flight[-1].x_L - plan.flat(T_START + T_TRAVEL)["x_L"]))
    ck.true("flown open loop on the library model, the load arrives within 1e-6 m", ol < 1e-6, f"{ol:.1e} m")
    course = np.array([plan.flat(t)["x_L"] for t in rep["t"]])
    ck.true("the course moves forward all the way (no doubling back over the wall)", np.all(np.diff(course[:, 0]) > -1e-9))
    over = np.abs(course[:, 0] - WALL["x"]) < 0.15
    ck.record("plan", dict(
        swing_max=float(rep["swing"].max()), t_swing=float(rep["t"][np.argmax(rep["swing"])]),
        T_min=float(rep["T"].min()), T_max=float(rep["T"].max()), f_min=float(rep["f"].min()),
        f_max=float(rep["f"].max()), tilt_max=float(rep["tilt"].max()), tau_max=float(rep["tau"].max()),
        speed_max=float(rep["speed"].max()), lead_max_cm=float(100 * rep["lead"].max()),
        omega_max=float(max(np.linalg.norm(plan.flat(t)["omega"]) for t in rep["t"])),
        Omega_max=float(rep["Omega"].max()), open_loop_error=ol,
        clearance_cm=float(100 * (course[over, 2].min() - WALL["height"])),
        torque_jump=float(np.abs(plan.flat(T_START)["tau"]).max()),
        length=float(np.sum(np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1)))))
    ck.record("hurry_plan", dict(swing_max=float(rep_h["swing"].max()), T_min=float(rep_h["T"].min()),
                                 f_max=float(rep_h["f"].max()), tau_max=float(rep_h["tau"].max())))
    ck.record("drop_plan", dict(T_min=float(rep_d["T"].min()), f_min=float(rep_d["f"].min()),
                                f_max=float(rep_d["f"].max()),
                                T_min_percent=float(100 * rep_d["T"].min() / (p.m_L * p.g)),
                                a_down_max_g=float(max(-plan_a[2] for plan_a in
                                                       [drop.flat(t)["a_L"] for t in rep_d["t"]]) / p.g)))
    ck.true("the plan stays inside the cone, the hurried one does not",
            rep["swing"].max() < CONE_DEG < rep_h["swing"].max())
    ck.true("the plan respects the thrust and torque limits",
            rep["f"].max() < F_MAX and rep["tau"].max() < TAU_MAX and rep["T"].min() > T_MIN)

    # ------------------------------------------------ 3. the cascade for a moving reference
    w_lib = w_ref = 0.0
    for _ in range(30):
        s = RB.random_state(rng)
        xd = rng.normal(size=3)
        ref = dict(x_L=xd, v_L=np.zeros(3), a_L=np.zeros(3), omega=np.zeros(3), omega_dot=np.zeros(3),
                   Omega=np.zeros(3), Omega_dot=np.zeros(3))
        F, _ = cascade_thrust(s, ref, k, feedforward=False)
        f, tau = attitude_stage(s, F, ref, k, feedforward=False)
        f_l, tau_l = RB.tracking_controller(s, xd, gains=k)
        w_lib = max(w_lib, abs(max(f, 0.0) - f_l), np.abs(tau - tau_l).max())
    ck.close("for a fixed target and no feedforward the cascade is the library's eq6-load", w_lib, 0, 1e-10)
    for t in np.linspace(0.6, T_START + T_TRAVEL - 0.1, 40):
        o, s = plan.flat(t), plan.state(t)
        F, qd = cascade_thrust(s, o, k)
        f, tau = attitude_stage(s, F, o, k)
        w_ref = max(w_ref, abs(f - o["f"]), np.abs(tau - o["tau"]).max(), np.abs(qd - o["q"]).max())
    ck.close("on the reference the cascade returns the inputs of the flat map", w_ref, 0, 1e-9)
    poles = np.linalg.eigvals(np.block([[np.zeros((3, 3)), I3], [-k.k_R * np.linalg.inv(p.J),
                                                               -k.k_Omega * np.linalg.inv(p.J)]]))
    ck.record("loops", dict(position=float(np.sqrt(k.k_x / m0)), cable=float(np.sqrt(k.k_q)),
                            attitude_slowest=float(np.min(np.abs(poles.real))),
                            attitude_fastest=float(np.max(np.abs(poles)))))

    # ------------------------------------------------ 4. linearization, three ways
    w_gm = w_raw = 0.0
    for _ in range(20):
        s = RB.random_state(rng)
        u = (5 + 20 * rng.random(), rng.normal(size=3) * 0.3)
        A, B, C = linearize(s, u)
        sv, du = tangent_start(s, rng), rng.normal(size=4)
        s_dot = A @ sv + B @ du
        xs = gm["layout"].pack(x_L=s.x_L, q=s.q, R=s.R, x_L_dot=s.v_L, omega_q=s.omega, Omega_R=s.Omega)
        acc = np.concatenate(RB.dynamics(s, u))
        uu = np.concatenate([[u[0]], u[1]])
        Mg, Ag, Bg = (fn(xs, acc, uu, gm_params(P0)) for fn in (gm["lin"].M_lin, gm["lin"].A_lin, gm["lin"].B_lin))
        res = Mg @ s_dot[GM_ORDER] - Ag @ sv[GM_ORDER] - Bg @ du
        # An equation on the sphere holds after projection by P = I - q q^T. geomech varies
        # the unprojected coefficient E_xi, so the variation of P is added here:
        # delta(P E_xi) = P delta(E_xi) - (xi x q)(q . E_xi) on a solution, where P E_xi = 0.
        sf = gm["sf"]
        E_xi = (sf.mass(xs, gm_params(P0)) @ acc + sf.bias(xs, gm_params(P0))
                + sf.input_matrix(xs, gm_params(P0)) @ uu)[3:6]
        Pq = I3 - np.outer(s.q, s.q)
        raw = np.abs(Pq @ res[12:15]).max()
        res[3:6] = Pq @ res[3:6]
        res[12:15] = Pq @ res[12:15] + cross(sv[6:9], s.q) * (s.q @ E_xi)
        w_gm = max(w_gm, np.abs(res).max() / max(1.0, np.abs(Ag @ sv[GM_ORDER]).max()))
        w_raw = max(w_raw, raw)
    ck.close("linearization: satisfies geomech's linearized equations on the constraint subspace", w_gm, 0, 1e-9)
    ck.true("linearization: without the variation of the projection the cable rows differ", w_raw > 0.1,
            f"{w_raw:.2f}")
    w_k = 0.0
    for _ in range(20):
        s = RB.random_state(rng)
        u = (5 + 20 * rng.random(), np.zeros(3))
        A, B, C = linearize(s, u)
        q, om, F, m_, ml = s.q, s.omega, u[0] * s.R @ E3, p.m_Q + p.m_L, p.m_Q * p.l
        q_dot = cross(om, q)
        a23 = -((q @ F - ml * q_dot @ q_dot) * I3 + np.outer(q, F)) @ hat(q) / m_
        a24 = 2 * ml / m_ * np.outer(q, q_dot) @ hat(q)
        a43 = -hat(F) @ hat(q) / ml
        sv, dF = tangent_start(s, rng), rng.normal(size=3)
        mine_v = A[3:6, 6:12] @ sv[6:12] + np.outer(q, q) @ dF / m_
        mine_w = A[9:12, 6:9] @ sv[6:9] - hat(q) @ dF / ml
        w_k = max(w_k, np.abs(mine_v - (a23 @ sv[6:9] + a24 @ sv[9:12] + np.outer(q, q) @ dF / m_)).max(),
                  np.abs(mine_w - (a43 @ sv[6:9] - hat(q) @ dF / ml)).max(),
                  np.abs((A[6:9, 6:9] - np.outer(q, q) @ hat(om)) @ sv[6:9]).max())
    ck.close("linearization: load and cable blocks equal Kotaru (2022), eq. (6.12), on the constraint subspace",
             w_k, 0, 1e-10)
    fd = finite_difference_check(plan, rng)
    ck.true("linearization: finite differences on the manifold agree, error below 1e-5 at eps = 1e-4",
            fd[1e-4] < 1e-5, f"{fd[1e-4]:.1e}")
    ck.true("linearization: the finite-difference error is second order in eps", 50 < fd[1e-3] / fd[1e-4] < 200,
            f"ratio {fd[1e-3] / fd[1e-4]:.0f}")
    ck.close("linearization: C s = 0 is kept by the linear flow", fd["constraint"], 0, 1e-9)
    s = RB.random_state(rng)
    A, B, C = linearize(s, (17.0, rng.normal(size=3) * 0.1))
    ck.close("linearization: C B = 0", C @ B, 0, 1e-12)
    A0, B0, C0 = linearize(RB.State(), RB.hover_input())
    Z0 = null_basis(C0)
    Ar, Br = Z0.T @ A0 @ Z0, Z0.T @ B0
    rank_hover, rank_full = reachable_dimension(Ar, Br), reachable_dimension(A0, B0)
    pbh = min(np.linalg.svd(np.hstack([Ar - lam * np.eye(16), Br]), compute_uv=False)[-1]
              for lam in np.linalg.eigvals(Ar))
    ck.true("hover: the Hautus test agrees, [A - lambda I, B] has rank 16 at every eigenvalue", pbh > 1e-6,
            f"smallest singular value {pbh:.1e}")
    ck.true("hover: controllable on the constraint subspace, rank 16", rank_hover == 16, f"rank {rank_hover}")
    ck.true("hover: the unreduced rank test reports 16 of 18", rank_full == 16, f"rank {rank_full}")
    sv_c, sv_all = gramian_rank(plan, 3.0, 5.0)
    ck.true("along the plan: the reachability Gramian has rank 16 on the constraint subspace",
            sv_c[-1] > 1e-9 * sv_c[0] and sv_all[16] < 1e-9 * sv_all[0],
            f"sixteenth/first {sv_c[-1] / sv_c[0]:.1e}, seventeenth/first {sv_all[16] / sv_all[0]:.1e}")
    ck.record("linearization", dict(geomech=float(w_gm), geomech_unprojected=float(w_raw), fd_1e3=fd[1e-3], fd_1e4=fd[1e-4],
                                    fd_ratio=fd[1e-3] / fd[1e-4], rank_hover=rank_hover, rank_full=rank_full,
                                    gramian_ratio=float(sv_c[-1] / sv_c[0]),
                                    gramian_17=float(sv_all[16] / sv_all[0])))
    # LQR at hover: the Riccati equation on the 18 numbers, run long, against the design on ker C
    Pr = solve_continuous_are(Ar, Br, Z0.T @ Q_LQR @ Z0, R_LQR)
    Kr = np.linalg.solve(R_LQR, Br.T @ Pr)
    hover_plan = Plan(waypoints=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + 1e-9]]), travel=1.0)
    _, K_h, _ = riccati(hover_plan, Q_LQR, R_LQR, Q_LQR, 0.0, 12.0)
    ck.close("hover LQR: the gain from the differential Riccati equation, on ker C, equals the reduced design",
             K_h[0] @ Z0, Kr, 1e-5)

    # ------------------------------------------------ 5. uncertainty and adaptation
    w_d = w_pred = w_V = 0.0
    for _ in range(30):
        s = RB.random_state(rng)
        u = (5 + 20 * rng.random(), rng.normal(size=3) * 0.3, rng.normal(size=3) * 2)
        acc = PLANT.dynamics(s, u, P_TRUE)
        Delta = lumped_uncertainty(acc[0], u[2])
        F = u[0] * s.R @ E3
        believed = PLANT.dynamics(s, (0.0, u[1], F + Delta), P0)        # thrust vector F + Delta
        w_d = max(w_d, np.abs(believed[0] - acc[0]).max(), np.abs(believed[1] - acc[1]).max())
        ad = Adaptation(s)
        ad.v_p, ad.omega_p = s.v_L + rng.normal(size=3), s.omega + tangent(s.q, rng.normal(size=3))
        ad.Deltahat = rng.normal(size=3)
        v_t, om_t = ad.errors(s)
        vp_dot, op_dot = ad.rates(s, F)
        D_t = ad.Deltahat - Delta
        kp = ad.k.k_pred
        w_pred = max(w_pred, np.abs(m0 * (vp_dot - acc[0]) - ((s.q @ D_t) * s.q - kp * m0 * v_t)).max(),
                     np.abs(p.m_Q * p.l * (op_dot - acc[1]) - (-cross(s.q, D_t) - kp * p.m_Q * p.l * om_t)).max())
        rate = m0 * v_t @ (vp_dot - acc[0]) + p.m_Q * p.l**2 * om_t @ (op_dot - acc[1]) + D_t @ ad.direction(s)
        w_V = max(w_V, abs(rate + kp * (m0 * v_t @ v_t + p.m_Q * p.l**2 * om_t @ om_t)))
    ck.close("lumped uncertainty: the believed model with thrust vector F + Delta is the true motion", w_d, 0, 1e-10)
    ck.close("prediction errors obey the stated linear equations driven by the estimation error", w_pred, 0, 1e-10)
    ck.close("adaptation direction y cancels the estimation error in the Lyapunov rate", w_V, 0, 1e-9)
    y, th = rng.normal(size=3), rng.normal(size=3)
    ck.close("projection operator is the one of verify/ch08.py (eq8-proj), inactive inside the ball",
             V8.proj(0.1 * th, y, 6.0), y, 1e-15)

    # ------------------------------------------------ 6. safety
    w_c = w_fd = w_T = w_floor = 0.0
    keep = V8.M_P, V8.L_P
    V8.M_P, V8.L_P = p.m_Q, p.l
    for _ in range(30):
        s = RB.random_state(rng)
        u = (5 + 20 * rng.random(), rng.normal(size=3) * 0.3)
        F = u[0] * s.R @ E3
        Hc, Hd, a, b = cone_terms(s.q, s.omega)
        H8, Hd8, a8, b8 = V8.cone_H(s.q, s.omega, Q_C, np.zeros(3), np.zeros(3), angle=np.radians(CONE_DEG))
        Fp = p.m_Q * p.g * E3 - F                              # the force of eq7-pendulum on the bob
        w_c = max(w_c, abs(Hc - H8), abs(Hd - Hd8), abs(a + b @ F - (a8 + b8 @ Fp)))
        e = 1e-4
        sp, sm = RB.step(s, u, e), RB.step(s, u, -e)
        Hp, Hm = (cone_terms(x.q, x.omega)[0] for x in (sp, sm))
        w_fd = max(w_fd, abs((Hp - 2 * Hc + Hm) / e**2 - (a + b @ F)), abs((Hp - Hm) / (2 * e) - Hd))
        T = RB.tension(s, u)
        w_T = max(w_T, abs(T - p.m_L / m0 * (p.m_Q * p.l * s.omega @ s.omega - s.q @ F)))
        if -s.q @ (s.R @ E3) > 0.05:
            f2 = tension_floor(s, 0.0)
            w_floor = max(w_floor, max(0.0, T_MIN - RB.tension(s, (f2, u[1]))))
        Fs, act = safety_filter(s, F)
        if act:
            Hc, Hd, a, b = cone_terms(s.q, s.omega)
            lhs = a + b @ Fs + (ALPHA[0] + ALPHA[1]) * Hd + ALPHA[0] * ALPHA[1] * Hc
            Ts = p.m_L / m0 * (p.m_Q * p.l * s.omega @ s.omega - s.q @ Fs)
            w_floor = max(w_floor, max(0.0, -lhs) / (1 + abs(lhs)), max(0.0, T_MIN - Ts))
    V8.M_P, V8.L_P = keep
    ck.close("cone barrier: H, H_dot, H_ddot equal verify/ch08.py's cone_H for the cable", w_c, 0, 1e-10)
    ck.close("cone barrier: H_dot and H_ddot = a + b . F against finite differences of the model", w_fd, 0, 2e-5)
    ck.close("tension is linear in the thrust vector, as the filter assumes", w_T, 0, 1e-12)
    ck.close("the filtered thrust vector satisfies eq8-rd2 and the tension floor", w_floor, 0, 1e-8)

    # ------------------------------------------------ 7. the estimator
    truth = V14.s2_truth(T=2.0)
    ref14 = V14.s2_filters(truth, seed=5, which=("var",))["var"]
    r14 = np.random.default_rng(5)
    Pq = I3 - np.outer(truth[1][0], truth[1][0])
    s0 = np.concatenate([Pq @ r14.normal(size=3), Pq @ r14.normal(size=3)]) * 1.0
    q0, om0 = V14.s2_oplus(truth[1][0], truth[2][0], s0)
    ys = truth[1] + r14.normal(size=truth[1].shape) * np.sqrt(1e-3)
    one_m = replace(P0, l=1.0)
    cf = CableFilter(q0, om0, FilterNoise(q_xi=1e-5, q_omega=1e-5, sig0_q=1.0, sig0_omega=1.0), one_m)
    cf.qhat, cf.omegahat = q0, om0
    Pi = V14.s2_Pi(q0, om0)
    cf.Sigma = Pi @ Pi.T
    w = 0.0
    for i in range(len(truth[0]) - 1):
        cf.propagate(p.m_Q * p.g * E3, h=truth[0][1] - truth[0][0])   # this thrust vector makes the cable a pendulum
        cf.correct(ys[i + 1])
        w = max(w, np.abs(cf.qhat - ref14["q"][i + 1]).max(), np.abs(cf.omegahat - ref14["om"][i + 1]).max())
    ck.close("the cable filter reproduces verify/ch14.py's filter on the pendulum of ex14-pendulum", w, 0, 1e-9)
    w = 0.0
    keep7 = V7.M_P, V7.L_P
    V7.M_P, V7.L_P = p.m_Q, p.l
    for _ in range(10):
        s = RB.random_state(rng)
        F = rng.normal(size=3) * 10
        w = max(w, np.abs(cable_A(s.q, s.omega, F) - V7.s2_A(s.q, s.omega, p.m_Q * p.g * E3 - F)).max())
    V7.M_P, V7.L_P = keep7
    ck.close("cable block of the linearization equals verify/ch07.py's s2_A", w, 0, 1e-12)

    # ------------------------------------------------ 8. the flights
    print(f"  checks before the flights took {time.time() - t_all:.0f} s; flying ...")
    data = all_runs()
    save_runs(data)
    R, S = data["runs"], data["summary"]
    ck.record("runs", S)
    # nominal tracking and feedforward
    ck.true("nominal: the cascade with flat feedforward tracks to better than 1 mm",
            S["nominal"]["err_max_cm"] < 0.1, f"{S['nominal']['err_max_cm']:.3f} cm")
    ck.true("nominal: without the flat feedforward the error is hundreds of times larger",
            S["nominal_no_ff"]["err_rms_cm"] > 100 * S["nominal"]["err_rms_cm"])
    ck.true("nominal: the cable's feedforward removes most of it",
            S["nominal_cable_ff"]["err_rms_cm"] < 0.2 * S["nominal_no_ff"]["err_rms_cm"])
    om_max = max(np.linalg.norm(plan.flat(t)["omega"]) for t in rep["t"])
    lag = k.k_omega / k.k_q * om_max
    ck.record("no_ff_estimate", dict(lag_deg=float(np.degrees(lag)), force=float(m0 * p.g * np.sin(lag)),
                                     error_cm=float(100 * m0 * p.g * np.sin(lag) / k.k_x)))
    # LQR against the cascade
    st = {n: settle_time(R[n]) for n in R if n.startswith(("lqr_off", "cas_off"))}
    ck.record("settle", st)
    ck.true("LQR settles faster than the cascade from 10 and 30 degrees",
            st["lqr_off10"] < st["cas_off10"] and st["lqr_off30"] < st["cas_off30"])
    ck.true("from 50 degrees the LQR fails and the cascade recovers",
            (not R["lqr_off50"]["completed"]) and R["cas_off50"]["completed"] and st["cas_off50"] < 7.0)
    ck.true("LQR on the nominal plan tracks to better than 1 mm", S["lqr_nominal"]["err_max_cm"] < 0.1)
    # uncertainty
    sag = (M_L_TRUE - p.m_L) * p.g / k.k_x
    ck.record("sag", dict(predicted_cm=100 * sag, measured_cm=S["plain"]["err_final_cm"]))
    ck.true("without adaptation the load hangs low by (m_L - m_L0) g / k_x", abs(S["plain"]["err_final_cm"] / 100 - sag) < 0.01,
            f"{S['plain']['err_final_cm']:.1f} cm against {100 * sag:.1f} cm")
    o = R["truth"]
    late = o["t"] > T_END - 1.0
    ck.record("adaptation", dict(
        Delta_true_z=float(np.mean(o["Delta"][late, 2])), Delta_f_z=float(np.mean(o["Delta_f"][late, 2])),
        Delta_err_final=float(np.linalg.norm(o["Delta_f"][-1] - o["Delta"][-1])),
        Delta_err_gust=float(np.max(np.linalg.norm(o["Delta_f"] - o["Delta"], axis=1)[(o["t"] > GUST[0]) & (o["t"] < GUST[1])])),
        v_tilde={kd: float(R[n]["v_tilde"][(R[n]["t"] > 1.0) & (R[n]["t"] < 6.0)].max()) for kd, n in
                 (("125", "kD_125"), ("500", "kD_500"), ("2000", "truth"), ("8000", "kD_8000"))},
        euler_factor_8000=float(abs(1 + H * (-AdaptGains().k_pred / 2
                                             + 1j * np.sqrt(8000.0 / m0 - AdaptGains().k_pred**2 / 4)))),
        euler_factor_2000=float(abs(1 + H * (-AdaptGains().k_pred / 2
                                             + 1j * np.sqrt(2000.0 / m0 - AdaptGains().k_pred**2 / 4))))))
    ck.true("adaptation: the filtered estimate ends at the true lumped force", np.linalg.norm(o["Delta_f"][-1] - o["Delta"][-1]) < 0.05,
            f"{np.linalg.norm(o['Delta_f'][-1] - o['Delta'][-1]):.3f} N")
    ck.true("adaptation: tracking error falls by more than five", S["truth"]["err_rms_cm"] < S["plain"]["err_rms_cm"] / 5)
    vt = [float(R[n]["v_tilde"][(R[n]["t"] > 1.0) & (R[n]["t"] < 6.0)].max())
          for n in ("kD_125", "kD_500", "truth", "kD_8000")]
    ck.true("adaptation: the prediction error shrinks as k_Delta grows from 125 to 2000", vt[0] > vt[1] > vt[2],
            f"{vt[0]:.1e}, {vt[1]:.1e}, {vt[2]:.1e} m/s")
    ck.true("adaptation: at k_Delta = 8000 the sampled law is worse again", vt[3] > vt[2], f"{vt[3]:.1e} m/s")
    # safety
    ck.true("hurried plan, no filter: the cable leaves the cone", S["hurry_free"]["H_min"] < -0.03)
    ck.true("hurried plan, filter on the true state: the cable is held within one degree of the cone",
            S["hurry_safe"]["swing_max_deg"] < CONE_DEG + 1.0, f"{S['hurry_safe']['swing_max_deg']:.2f} deg")
    ck.true("steep descent, no filter: the tension reaches zero", S["drop_free"]["T_min"] < 1e-6)
    ck.true("steep descent, filter: the tension stays above T_min", S["drop_safe"]["T_min"] >= T_MIN - 1e-9,
            f"{S['drop_safe']['T_min']:.2f} N")
    ck.true("steep descent on the estimate: the filter keeps tension and cone",
            S["drop_safe_est"]["T_min"] >= T_MIN and S["drop_safe_est"]["H_min"] > 0)
    # estimation
    o = R["est_wrong"]
    conv = float(o["t"][np.argmax(o["e_qhat"] < 2.0)])
    ck.record("estimator", dict(wrong_by_deg=float(o["first_error"]), start_error_deg=float(o["e_qhat"][0]),
                                reading_error_deg=float(np.degrees(FilterNoise().sig_y / P0.l)), below_2deg_after=conv,
                                rms_deg=S["truth"]["est_rms_deg"], omega_rms=S["truth"]["est_omega_rms"],
                                nees=S["truth"]["nees"], rms_deg_loop=S["full"]["est_rms_deg"],
                                omega_rms_loop=S["full"]["est_omega_rms"], nees_loop=S["full"]["nees"],
                                residual=max(S[n]["filter_res"] for n in ("full", "truth", "est_wrong"))))
    ck.true("estimator: from 40 degrees wrong, within 2 degrees in under 0.2 s", conv < 0.2, f"{conv:.3f} s")
    ck.true("estimator: |qhat| = 1, qhat . omegahat = 0 and C Sigma = 0 to 1e-10",
            max(S[n]["filter_res"] for n in ("full", "truth", "est_wrong")) < 1e-10)
    ck.true("estimator: NEES in the four constrained dimensions between 0.5 and 2",
            0.5 < S["truth"]["nees"] < 2.0 and 0.5 < S["full"]["nees"] < 2.0,
            f"{S['truth']['nees']:.2f} with the truth in the loop, {S['full']['nees']:.2f} with the estimate")
    # the mission
    F_ = S["full"]
    ck.true("mission: completed, cable inside the cone, cable taut", F_["completed"] and F_["H_min"] > 0 and F_["T_min"] > T_MIN,
            f"H >= {F_['H_min']:.3f}, T >= {F_['T_min']:.2f} N")
    ck.true("mission: inputs inside their limits", F_["f_max"] < F_MAX and F_["tau_max"] < TAU_MAX and F_["saturated_frac"] == 0.0,
            f"f <= {F_['f_max']:.1f} N, |tau_i| <= {F_['tau_max']:.2f} N·m")
    ck.true("mission: the state stays on the manifold to 1e-12", max(S[n]["manifold"] for n in S if S[n]["completed"]) < 1e-12,
            f"{max(S[n]['manifold'] for n in S if S[n]['completed']):.1e}")
    o = R["full"]
    ck.record("mission", dict(
        R_residual=float(max(np.abs(Rm.T @ Rm - I3).max() for Rm in o["R"][::50])),
        q_residual=float(np.abs(np.linalg.norm(o["q"], axis=1) - 1).max()),
        T_max=float(o["T"].max()), tilt_max=float(np.degrees(np.arccos(o["R"][:, 2, 2].min()))),
        active_percent=100 * F_["active_frac"]))
    ck.true("ablation: each of feedforward, adaptation and the cascade itself matters",
            min(S["no_ff"]["err_rms_cm"], S["no_adapt"]["err_rms_cm"], S["bare"]["err_rms_cm"]) > 3 * F_["err_rms_cm"])
    ck.true("ablation: the filter changes the mission's tracking error by less than 1 mm",
            abs(S["no_filter"]["err_rms_cm"] - F_["err_rms_cm"]) < 0.1)
    print(f"  total {time.time() - t_all:.0f} s")
    ck.finish()


if __name__ == "__main__":
    main()
