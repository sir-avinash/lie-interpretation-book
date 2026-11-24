"""A flexible cable as a chain of n links on (S^2)^n.

The cable is n massless links joined by spherical joints, with a point mass at
every joint (Kotaru, 2022, ch. 7). Joint 0 is at x_0 and

    x_k = x_0 + sum_{j <= k} l_j q_j,        k = 1, ..., n,

so q_j points from joint j - 1 to joint j. Joint k carries the mass m_k.

HangingChain: joint 0 is a fixed pivot. The configuration manifold is (S^2)^n
and the equation of link i is eq5-chain, on the tangent plane of q_i:

    sum_j M_ij [ -q_i x (omega_j_dot x q_j) + |omega_j|^2 q_i x q_j ]
        - (sum_{k >= i} m_k) g l_i q_i x e3 = 0,
    M_ij = (sum_{k >= max(i, j)} m_k) l_i l_j.

QuadrotorCable: joint 0 moves, and quadrotors are attached at their centers
of mass to some of the joints. The configuration manifold is
R^3 x (S^2)^n x SO(3)^{n_Q}. A quadrotor at joint k adds its mass to m_k and
pushes that joint with f R e3. With M_00 = sum_k m_k and
M_0i = l_i sum_{k >= i} m_k (Kotaru, 2022, eq. 7.12),

    M_00 (x_0_ddot + g e3) + sum_j M_0j c_j = sum_{k attached} f_k R_k e3,
    q_i x [ M_0i (x_0_ddot + g e3) + sum_j M_ij c_j
            - l_i sum_{k attached, k >= i} f_k R_k e3 ] = 0,
    c_j = omega_j_dot x q_j - |omega_j|^2 q_j        (so that q_j_ddot = c_j).

One quadrotor at joint 0 is a quadrotor carrying a load on a flexible cable.
With n = 1 it is the quadrotor with a cable-suspended load, eq4-load, written
with x_Q in place of x_L. Quadrotors at joints 0 and n carry a cable or a hose
between them.
"""
from dataclasses import dataclass

import numpy as np

from .core import E3, GRAVITY, I3, Factor, Robot, hat
from .multi_quadrotor_payload import J_QUAD, _euler


def chain_mass_matrix(m, l):
    """M_ij = (sum of the masses below both link i and link j) l_i l_j, eq5-chain."""
    n = len(l)
    below = np.cumsum(m[::-1])[::-1]                    # below[i] = m_i + ... + m_n
    return np.array([[below[max(i, j)] * l[i] * l[j] for j in range(n)] for i in range(n)])


# =====================================================================
# chain hanging from a fixed pivot
# =====================================================================
@dataclass
class ChainParams:
    """Defaults: the chain of equal links simulated in sec5-scaling (n = 4)."""
    n: int = 4
    m: np.ndarray = None      # point masses at joints 1..n, kg  (n,)
    l: np.ndarray = None      # link lengths, m                  (n,)
    g: float = GRAVITY

    def __post_init__(self):
        self.m = np.full(self.n, 0.25) if self.m is None else np.asarray(self.m, float)
        self.l = np.full(self.n, 0.25) if self.l is None else np.asarray(self.l, float)


@dataclass
class ChainState:
    q: np.ndarray             # link directions                  (n, 3)
    omega: np.ndarray         # their angular velocities         (n, 3)


class HangingChain(Robot):
    name = "flexible_cable"
    Params, State = ChainParams, ChainState
    labels = {"equations": "eq5-chain", "manifold": "tbl2-robots", "section": "sec5-scaling",
              "figure": "fig5-chain"}

    def __init__(self, n=4, params=None):
        super().__init__(params if params is not None else ChainParams(n=n))
        self.n = self.params.n
        self.manifold = f"(S^2)^{self.n}"
        self.factors = [Factor("S2", "q", "omega", count=self.n, stacked=True)]

    def zero_input(self):
        return None

    def dynamics(self, state, u=None, params=None):
        """(omega_dot,) of shape (n, 3). There is no input."""
        p = params or self.params
        n, q, omega = self.n, state.q, state.omega
        M = chain_mass_matrix(p.m, p.l)
        below = np.cumsum(p.m[::-1])[::-1]
        A, b = np.zeros((3 * n, 3 * n)), np.zeros(3 * n)
        for i in range(n):
            rows = slice(3 * i, 3 * i + 3)
            for j in range(n):
                # -q_i x (omega_j_dot x q_j) = hat(q_i) hat(q_j) omega_j_dot
                A[rows, 3 * j:3 * j + 3] = M[i, j] * hat(q[i]) @ hat(q[j])
                b[rows] -= M[i, j] * (omega[j] @ omega[j]) * np.cross(q[i], q[j])
            b[rows] += below[i] * p.g * p.l[i] * np.cross(q[i], E3)
            A[rows, rows] += M[i, i] * np.outer(q[i], q[i])      # q_i . omega_i_dot = 0
        return (np.linalg.solve(A, b).reshape(n, 3),)

    def positions(self, state, params=None):
        """Joint positions x_1..x_n, shape (n, 3)."""
        p = params or self.params
        return np.cumsum(p.l[:, None] * state.q, axis=0)

    def energy(self, state, params=None):
        p = params or self.params
        v = np.cumsum(p.l[:, None] * np.cross(state.omega, state.q), axis=0)
        x = self.positions(state, p)
        return float(np.sum(0.5 * p.m * np.sum(v * v, axis=1) + p.m * p.g * x[:, 2]))

    def momentum(self, state, params=None):
        """Angular momentum about the pivot. Its component along e3 is conserved."""
        p = params or self.params
        v = np.cumsum(p.l[:, None] * np.cross(state.omega, state.q), axis=0)
        return np.sum(p.m[:, None] * np.cross(self.positions(state, p), v), axis=0)

    def lagrangian(self):
        from geomech import S2, Dot, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        (g,) = getScalars("g", attr=["Constant"])
        e3 = Vector("e3", attr=["Constant"])
        qs, L, x = [], None, None
        for i in range(1, self.n + 1):
            m, l = getScalars(f"m{i} l{i}", attr=["Constant"])
            q = S2(f"q{i}")
            x = q * l if x is None else x + q * l
            v = x.t_diff()
            term = half * m * Dot(v, v) - m * g * Dot(e3, x)
            L = term if L is None else L + term
            qs.append(q)
        return L, None, SystemVariables(vectors=qs), []


# =====================================================================
# cable carried by quadrotors
# =====================================================================
@dataclass
class CableParams:
    n: int = 5
    attach: tuple = (0,)      # joints that carry a quadrotor
    m: np.ndarray = None      # point masses at joints 0..n, kg   (n + 1,)
    l: np.ndarray = None      # link lengths, m                   (n,)
    m_Q: np.ndarray = None    # quadrotor masses, kg              (n_Q,)
    J: np.ndarray = None      # quadrotor inertias, kg m^2        (n_Q, 3, 3)
    g: float = GRAVITY

    def __post_init__(self):
        n, n_Q = self.n, len(self.attach)
        if self.m is None:                       # light cable; a 0.3 kg load at a free end
            self.m = np.full(n + 1, 0.05)
            self.m[0] = 0.0
            if n not in self.attach:
                self.m[n] = 0.3
        self.m = np.asarray(self.m, float)
        self.l = np.full(n, 1.0 / n) if self.l is None else np.asarray(self.l, float)
        self.m_Q = np.full(n_Q, 1.5) if self.m_Q is None else np.asarray(self.m_Q, float)
        self.J = np.array([J_QUAD] * n_Q) if self.J is None else np.asarray(self.J, float)


@dataclass
class CableState:
    x_0: np.ndarray           # position of joint 0              (3,)
    v_0: np.ndarray           # its velocity                     (3,)
    q: np.ndarray             # link directions                  (n, 3)
    omega: np.ndarray         # their angular velocities         (n, 3)
    R: np.ndarray             # quadrotor attitudes              (n_Q, 3, 3)
    Omega: np.ndarray         # body angular velocities          (n_Q, 3)


class QuadrotorCable(Robot):
    name = "quadrotor_cable"
    Params, State = CableParams, CableState
    labels = {"manifold": "tbl2-robots", "section": "sec5-scaling", "chain": "eq5-chain",
              "one_link": "eq4-load"}

    def __init__(self, n=5, attach=(0,), params=None):
        super().__init__(params if params is not None else CableParams(n=n, attach=tuple(attach)))
        p = self.params
        self.n, self.n_Q = p.n, len(p.attach)
        self.manifold = f"R^3 x (S^2)^{p.n} x SO(3)^{self.n_Q}"
        self.factors = [Factor("R", "x_0", "v_0"), Factor("S2", "q", "omega", count=p.n, stacked=True),
                        Factor("SO3", "R", "Omega", count=self.n_Q, stacked=True)]

    def zero_input(self):
        return np.zeros(self.n_Q), np.zeros((self.n_Q, 3))

    def joint_masses(self, params=None):
        """m_k plus the mass of a quadrotor attached at joint k, k = 0..n."""
        p = params or self.params
        total = p.m.copy()
        for k, m_Q in zip(p.attach, p.m_Q):
            total[k] += m_Q
        return total

    def dynamics(self, state, u, params=None):
        """(v_0_dot, omega_dot, Omega_dot) for u = (f, tau), one row per quadrotor."""
        p = params or self.params
        f, tau = u
        n, q, omega = self.n, state.q, state.omega
        mass = self.joint_masses(p)
        below = np.cumsum(mass[::-1])[::-1]                # below[k] = mass_k + ... + mass_n
        M = chain_mass_matrix(mass[1:], p.l)
        M0 = p.l * below[1:]                               # M_0i
        thrust = [fk * Rk @ E3 for fk, Rk in zip(f, state.R)]
        # push[i] = sum of the thrusts applied at joints k >= i  (push[0]: all of them)
        push = np.zeros((n + 1, 3))
        for k, Fk in zip(p.attach, thrust):
            push[:k + 1] += Fk
        # Unknowns z = (x_0_ddot + g e3, omega_1_dot, ..., omega_n_dot):  A z = b,
        # using c_j = -hat(q_j) omega_j_dot - |omega_j|^2 q_j.
        A, b = np.zeros((3 + 3 * n, 3 + 3 * n)), np.zeros(3 + 3 * n)
        A[:3, :3] = below[0] * I3
        b[:3] = push[0]
        for j in range(n):
            cols = slice(3 + 3 * j, 6 + 3 * j)
            A[:3, cols] = -M0[j] * hat(q[j])
            b[:3] += M0[j] * (omega[j] @ omega[j]) * q[j]
        for i in range(n):
            rows = slice(3 + 3 * i, 6 + 3 * i)
            A[rows, :3] = M0[i] * hat(q[i])
            b[rows] = p.l[i] * np.cross(q[i], push[i + 1])
            for j in range(n):
                cols = slice(3 + 3 * j, 6 + 3 * j)
                A[rows, cols] = -M[i, j] * hat(q[i]) @ hat(q[j])
                b[rows] += M[i, j] * (omega[j] @ omega[j]) * np.cross(q[i], q[j])
            A[rows, rows] += M[i, i] * np.outer(q[i], q[i])      # q_i . omega_i_dot = 0
        z = np.linalg.solve(A, b)
        return z[:3] - p.g * E3, z[3:].reshape(n, 3), _euler(p.J, state.Omega, tau)

    def positions(self, state, params=None):
        """Joint positions x_0..x_n, shape (n + 1, 3)."""
        p = params or self.params
        return state.x_0 + np.vstack([np.zeros(3), np.cumsum(p.l[:, None] * state.q, axis=0)])

    def velocities(self, state, params=None):
        p = params or self.params
        rel = np.cumsum(p.l[:, None] * np.cross(state.omega, state.q), axis=0)
        return state.v_0 + np.vstack([np.zeros(3), rel])

    def energy(self, state, params=None):
        p = params or self.params
        mass, x, v = self.joint_masses(p), self.positions(state, p), self.velocities(state, p)
        return float(np.sum(0.5 * mass * np.sum(v * v, axis=1) + mass * p.g * x[:, 2])
                     + sum(0.5 * Oi @ Ji @ Oi for Oi, Ji in zip(state.Omega, p.J)))

    def lagrangian(self):
        from geomech import (S2, SO3, Cross, Dot, Matrix, Scalar, SystemVariables, Vector,
                             getScalars)
        p = self.params
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        (g,) = getScalars("g", attr=["Constant"])
        e3 = Vector("e3", attr=["Constant"])
        x_0 = Vector("x_0")
        qs = [S2(f"q{i}") for i in range(1, p.n + 1)]
        joints, deltas = [x_0], [x_0.delta()]
        for i, q in enumerate(qs, start=1):
            (l,) = getScalars(f"l{i}", attr=["Constant"])
            joints.append(joints[-1] + q * l)
            deltas.append(deltas[-1] + Cross(q.get_variation_vector(), q) * l)
        L = None
        for k, x in enumerate(joints):
            (m,) = getScalars(f"m{k}", attr=["Constant"])   # includes an attached quadrotor
            v = x.t_diff()
            term = half * m * Dot(v, v) - m * g * Dot(e3, x)
            L = term if L is None else L + term
        dW, Rs, inputs = None, [], []
        for k in p.attach:
            J = Matrix(f"J{k}", attr=["Constant", "SymmetricMatrix"])
            R = SO3(f"R{k}")
            Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
            f, tau = Scalar(f"f{k}"), Vector(f"tau{k}")
            L = L + half * Dot(Omega, J * Omega)
            work = Dot(deltas[k], (R * e3) * f) + Dot(eta, tau)
            dW = work if dW is None else dW + work
            Rs.append(R), inputs.extend([f, tau])
        return L, dW, SystemVariables(vectors=[x_0, *qs], matrices=Rs), inputs
