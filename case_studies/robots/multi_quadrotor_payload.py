"""Several quadrotors carrying one body on cables.

Two models, both with cable direction q_i pointing from quadrotor i to the
body it carries, and with every quadrotor attached at its center of mass.

MultiQuadrotorLoad: n quadrotors and a point-mass load, on
R^3 x (S^2 x SO(3))^n, quadrotor i at x_L - l_i q_i (eq4-multi):

    (m_L I + sum_i m_i q_i q_i')(x_L_ddot + g e3)
        = sum_i (q_i . f_i R_i e3 - m_i l_i |q_i_dot|^2) q_i,
    m_i l_i omega_i_dot = q_i x (m_i (x_L_ddot + g e3) - f_i R_i e3),
    J_i Omega_i_dot + Omega_i x J_i Omega_i = tau_i.

MultiQuadrotorPayload: n quadrotors and a rigid payload, on
SE(3) x (S^2 x SO(3))^n (Wu, 2017, ch. 3). The payload has mass m_L, inertia
J_L, position x_L and attitude R_L. Cable i is attached at the body-frame
point r_i, so quadrotor i is at x_L + R_L r_i - l_i q_i. The book states this
model in words only (sec4-products, sec5-nbodies). Here it is written as the
Lagrange-d'Alembert principle gives it, one equation per variation:

    delta x_L:  m_L (a_L + g e3) + sum_i (m_i (a_i + g e3) - f_i R_i e3) = 0,
    eta_L:      J_L Omega_L_dot + Omega_L x J_L Omega_L
                    + sum_i r_i x R_L' (m_i (a_i + g e3) - f_i R_i e3) = 0,
    xi_i:       q_i x (m_i (a_i + g e3) - f_i R_i e3) = 0,

where a_i is the acceleration of quadrotor i,

    a_i = a_L + R_L (Omega_L_dot x r_i + Omega_L x (Omega_L x r_i))
              - l_i (omega_i_dot x q_i - |omega_i|^2 q_i).

These are linear in the accelerations and are solved as one linear system.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E3, GRAVITY, I3, Factor, Robot, hat

J_QUAD = np.array([[0.02, 0.0005, 0.0], [0.0005, 0.025, 0.001], [0.0, 0.001, 0.04]])


def _euler(J, Omega, tau):
    """Omega_dot from J Omega_dot + Omega x J Omega = tau, for stacked quadrotors."""
    return np.array([np.linalg.solve(Ji, ti - np.cross(Oi, Ji @ Oi))
                     for Ji, Oi, ti in zip(J, Omega, tau)])


# =====================================================================
# point-mass load
# =====================================================================
@dataclass
class LoadParams:
    n: int = 2
    m_L: float = 0.5                          # load mass, kg
    m: np.ndarray = None                      # quadrotor masses, kg       (n,)
    l: np.ndarray = None                      # cable lengths, m           (n,)
    J: np.ndarray = None                      # quadrotor inertias, kg m^2 (n, 3, 3)
    g: float = GRAVITY

    def __post_init__(self):
        self.m = np.full(self.n, 1.5) if self.m is None else np.asarray(self.m, float)
        self.l = np.full(self.n, 1.0) if self.l is None else np.asarray(self.l, float)
        self.J = np.array([J_QUAD] * self.n) if self.J is None else np.asarray(self.J, float)


@dataclass
class LoadState:
    x_L: np.ndarray       # load position                      (3,)
    v_L: np.ndarray       # load velocity                      (3,)
    q: np.ndarray         # cable directions                   (n, 3)
    omega: np.ndarray     # their angular velocities           (n, 3)
    R: np.ndarray         # quadrotor attitudes                (n, 3, 3)
    Omega: np.ndarray     # body angular velocities            (n, 3)


class MultiQuadrotorLoad(Robot):
    name = "multi_quadrotor_load"
    Params, State = LoadParams, LoadState
    labels = {"equations": "eq4-multi", "manifold": "tbl2-robots", "section": "sec4-products"}

    def __init__(self, n=2, params=None):
        super().__init__(params if params is not None else LoadParams(n=n))
        n = self.params.n
        self.n = n
        self.manifold = f"R^3 x (S^2 x SO(3))^{n}"
        self.factors = [Factor("R", "x_L", "v_L"), Factor("S2", "q", "omega", count=n, stacked=True),
                        Factor("SO3", "R", "Omega", count=n, stacked=True)]

    def zero_input(self):
        return np.zeros(self.n), np.zeros((self.n, 3))

    def dynamics(self, state, u, params=None):
        """(v_L_dot, omega_dot, Omega_dot) for u = (f, tau), f of shape (n,), tau (n, 3)."""
        p = params or self.params
        f, tau = u
        q, omega = state.q, state.omega
        thrust = np.array([fi * Ri @ E3 for fi, Ri in zip(f, state.R)])
        mass = p.m_L * I3 + sum(mi * np.outer(qi, qi) for mi, qi in zip(p.m, q))
        pull = sum((qi @ ui - mi * li * wi @ wi) * qi
                   for qi, ui, mi, li, wi in zip(q, thrust, p.m, p.l, omega))
        a = np.linalg.solve(mass, pull)                       # x_L_ddot + g e3
        omega_dot = np.array([np.cross(qi, mi * a - ui) / (mi * li)
                              for qi, ui, mi, li in zip(q, thrust, p.m, p.l)])
        return a - p.g * E3, omega_dot, _euler(p.J, state.Omega, tau)

    def quadrotor_positions(self, state, params=None):
        p = params or self.params
        return state.x_L - p.l[:, None] * state.q

    def energy(self, state, params=None):
        p = params or self.params
        v_Q = state.v_L - p.l[:, None] * np.cross(state.omega, state.q)
        x_Q = self.quadrotor_positions(state, p)
        return (0.5 * p.m_L * state.v_L @ state.v_L + p.m_L * p.g * state.x_L[2]
                + sum(0.5 * mi * vi @ vi + mi * p.g * xi[2] + 0.5 * Oi @ Ji @ Oi
                      for mi, vi, xi, Oi, Ji in zip(p.m, v_Q, x_Q, state.Omega, p.J)))

    def tensions(self, state, u, params=None):
        """Tension of each cable, positive when it pulls: from Newton's law for
        quadrotor i, m_i (a_i + g e3) = f_i R_i e3 + T_i q_i."""
        p = params or self.params
        v_L_dot, omega_dot, _ = self.dynamics(state, u, p)
        out = []
        for i in range(self.n):
            qi, wi = state.q[i], state.omega[i]
            a_i = v_L_dot - p.l[i] * (np.cross(omega_dot[i], qi) - (wi @ wi) * qi)
            out.append(qi @ (p.m[i] * (a_i + p.g * E3) - u[0][i] * state.R[i] @ E3))
        return np.array(out)

    def lagrangian(self):
        from geomech import (S2, SO3, Cross, Dot, Matrix, Scalar, SystemVariables, Vector,
                             getScalars)
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m_L, g = getScalars("m_L g", attr=["Constant"])
        e3 = Vector("e3", attr=["Constant"])
        x_L = Vector("x_L")
        v_L = x_L.t_diff()
        L = half * m_L * Dot(v_L, v_L) - m_L * g * Dot(e3, x_L)
        dW, qs, Rs, inputs = None, [], [], []
        for i in range(1, self.n + 1):
            m, l = getScalars(f"m{i} l{i}", attr=["Constant"])
            J = Matrix(f"J{i}", attr=["Constant", "SymmetricMatrix"])
            q, R = S2(f"q{i}"), SO3(f"R{i}")
            Omega, eta, xi = R.get_tangent_vector(), R.get_variation_vector(), q.get_variation_vector()
            f, tau = Scalar(f"f{i}"), Vector(f"tau{i}")
            x_Q = x_L - q * l
            v_Q = x_Q.t_diff()
            L = L + half * m * Dot(v_Q, v_Q) + half * Dot(Omega, J * Omega) - m * g * Dot(e3, x_Q)
            work = Dot(x_L.delta() - Cross(xi, q) * l, (R * e3) * f) + Dot(eta, tau)
            dW = work if dW is None else dW + work
            qs.append(q), Rs.append(R), inputs.extend([f, tau])
        return L, dW, SystemVariables(vectors=[x_L, *qs], matrices=Rs), inputs


# =====================================================================
# rigid payload
# =====================================================================
@dataclass
class PayloadParams:
    n: int = 3
    m_L: float = 0.8                                                             # kg
    J_L: np.ndarray = field(default_factory=lambda: np.diag([0.03, 0.05, 0.08]))  # kg m^2
    r: np.ndarray = None                      # attachment points, payload frame, m (n, 3)
    m: np.ndarray = None                      # quadrotor masses, kg
    l: np.ndarray = None                      # cable lengths, m
    J: np.ndarray = None                      # quadrotor inertias, kg m^2
    g: float = GRAVITY

    def __post_init__(self):
        n = self.n
        if self.r is None:                    # evenly spaced on a circle of radius 0.4 m
            a = 2 * np.pi * np.arange(n) / n
            self.r = 0.4 * np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])
        self.r = np.asarray(self.r, float)
        self.m = np.full(n, 1.5) if self.m is None else np.asarray(self.m, float)
        self.l = np.full(n, 1.0) if self.l is None else np.asarray(self.l, float)
        self.J = np.array([J_QUAD] * n) if self.J is None else np.asarray(self.J, float)


@dataclass
class PayloadState:
    x_L: np.ndarray       # payload position                   (3,)
    v_L: np.ndarray       # payload velocity                   (3,)
    R_L: np.ndarray       # payload attitude                   (3, 3)
    Omega_L: np.ndarray   # payload body angular velocity      (3,)
    q: np.ndarray         # cable directions                   (n, 3)
    omega: np.ndarray     # their angular velocities           (n, 3)
    R: np.ndarray         # quadrotor attitudes                (n, 3, 3)
    Omega: np.ndarray     # body angular velocities            (n, 3)


class MultiQuadrotorPayload(Robot):
    name = "multi_quadrotor_payload"
    Params, State = PayloadParams, PayloadState
    labels = {"manifold": "tbl2-robots", "section": "sec4-products", "symbolic": "sec5-nbodies"}

    def __init__(self, n=3, params=None):
        super().__init__(params if params is not None else PayloadParams(n=n))
        n = self.params.n
        self.n = n
        self.manifold = f"R^3 x SO(3) x (S^2 x SO(3))^{n}"
        self.factors = [Factor("R", "x_L", "v_L"), Factor("SO3", "R_L", "Omega_L"),
                        Factor("S2", "q", "omega", count=n, stacked=True),
                        Factor("SO3", "R", "Omega", count=n, stacked=True)]

    def zero_input(self):
        return np.zeros(self.n), np.zeros((self.n, 3))

    def dynamics(self, state, u, params=None):
        """(v_L_dot, Omega_L_dot, omega_dot, Omega_dot) for u = (f, tau)."""
        p = params or self.params
        f, tau = u
        n, R_L, Om_L = self.n, state.R_L, state.Omega_L
        # Unknowns z = (a_L, Omega_L_dot, omega_1_dot, ..., omega_n_dot):  A z = b.
        A, b = np.zeros((6 + 3 * n, 6 + 3 * n)), np.zeros(6 + 3 * n)
        A[:3, :3] = p.m_L * I3
        b[:3] = -p.m_L * p.g * E3
        A[3:6, 3:6] = p.J_L
        b[3:6] = -np.cross(Om_L, p.J_L @ Om_L)
        for i in range(n):
            qi, wi, ri, mi, li = state.q[i], state.omega[i], p.r[i], p.m[i], p.l[i]
            # a_i = Ja z + bias, with the three blocks of Ja:
            Ja = np.zeros((3, 6 + 3 * n))
            Ja[:, :3] = I3
            Ja[:, 3:6] = -R_L @ hat(ri)
            Ja[:, 6 + 3 * i:9 + 3 * i] = li * hat(qi)
            bias = R_L @ np.cross(Om_L, np.cross(Om_L, ri)) + li * (wi @ wi) * qi
            force = f[i] * state.R[i] @ E3 - mi * p.g * E3      # thrust and weight
            # m_i a_i - force, weighted by each variation
            A[:3] += mi * Ja
            b[:3] += force - mi * bias
            A[3:6] += hat(ri) @ R_L.T @ (mi * Ja)
            b[3:6] += hat(ri) @ R_L.T @ (force - mi * bias)
            rows = slice(6 + 3 * i, 9 + 3 * i)
            A[rows] = hat(qi) @ (mi * Ja)
            b[rows] = hat(qi) @ (force - mi * bias)
            A[rows, rows] += mi * li * np.outer(qi, qi)          # q_i . omega_i_dot = 0
        z = np.linalg.solve(A, b)
        return z[:3], z[3:6], z[6:].reshape(n, 3), _euler(p.J, state.Omega, tau)

    def quadrotor_positions(self, state, params=None):
        p = params or self.params
        return state.x_L + p.r @ state.R_L.T - p.l[:, None] * state.q

    def energy(self, state, params=None):
        p = params or self.params
        v_Q = (state.v_L + np.cross(state.Omega_L, p.r) @ state.R_L.T
               - p.l[:, None] * np.cross(state.omega, state.q))
        x_Q = self.quadrotor_positions(state, p)
        return (0.5 * p.m_L * state.v_L @ state.v_L + p.m_L * p.g * state.x_L[2]
                + 0.5 * state.Omega_L @ p.J_L @ state.Omega_L
                + sum(0.5 * mi * vi @ vi + mi * p.g * xi[2] + 0.5 * Oi @ Ji @ Oi
                      for mi, vi, xi, Oi, Ji in zip(p.m, v_Q, x_Q, state.Omega, p.J)))

    def lagrangian(self):
        from geomech import (S2, SO3, Cross, Dot, Matrix, Scalar, SystemVariables, Vector,
                             getScalars)
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m_L, g = getScalars("m_L g", attr=["Constant"])
        J_L = Matrix("J_L", attr=["Constant", "SymmetricMatrix"])
        e3 = Vector("e3", attr=["Constant"])
        x_L, R_L = Vector("x_L"), SO3("R_L")
        Om_L, eta_L = R_L.get_tangent_vector(), R_L.get_variation_vector()
        v_L = x_L.t_diff()
        L = half * m_L * Dot(v_L, v_L) + half * Dot(Om_L, J_L * Om_L) - m_L * g * Dot(e3, x_L)
        dW, qs, Rs, inputs = None, [], [], []
        for i in range(1, self.n + 1):
            m, l = getScalars(f"m{i} l{i}", attr=["Constant"])
            J = Matrix(f"J{i}", attr=["Constant", "SymmetricMatrix"])
            r = Vector(f"r{i}", attr=["Constant"])
            q, R = S2(f"q{i}"), SO3(f"R{i}")
            Omega, eta, xi = R.get_tangent_vector(), R.get_variation_vector(), q.get_variation_vector()
            f, tau = Scalar(f"f{i}"), Vector(f"tau{i}")
            x_Q = x_L + R_L * r - q * l
            v_Q = x_Q.t_diff()
            L = L + half * m * Dot(v_Q, v_Q) + half * Dot(Omega, J * Omega) - m * g * Dot(e3, x_Q)
            work = (Dot(x_L.delta() + R_L * Cross(eta_L, r) - Cross(xi, q) * l, (R * e3) * f)
                    + Dot(eta, tau))
            dW = work if dW is None else dW + work
            qs.append(q), Rs.append(R), inputs.extend([f, tau])
        return L, dW, SystemVariables(vectors=[x_L, *qs], matrices=[R_L, *Rs]), inputs
