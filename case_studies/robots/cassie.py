"""Cassie, reduced geometric model (Siravuru, 2019, ch. 6): floating base only.

The model has a floating pelvis with position x and attitude R, one rotation
matrix per leg (R_r, R_l) and two leg lengths (l_r, l_l), on
SE(3) x SO(3)^2 x R_+^2: fourteen degrees of freedom, eight of them actuated
(a three-axis hip torque and a leg force per leg). The knees are prismatic and
the feet are massless. As we read the thesis,

    hips at   x -+ d R e2            (d: half the hip width; right is -e2),
    knee masses at   hip + (l/2) R_leg e3,
    feet at   hip + l R_leg e3,

with a point mass at the pelvis, at each hip and at each knee, and an inertia
for the pelvis and for each leg.

What is included: the floating-base equations of motion for this Lagrangian.
The book does not state them (it lists the manifold in tbl2-robots), and the
matrices D, C, G, B of the thesis (eq. 6.3) could not be read reliably from
the source, so `dynamics` assembles the Lagrange-d'Alembert equations
directly: every point mass m_p with acceleration a_p = Ja z + bias, linear in
the accelerations z, contributes Ja^T m_p (a_p + g e3) to the left side.
`verify/robots.py` checks the result against geomech.

What is NOT included: the ground-contact constraints (thesis eqs. 6.8 to
6.14) that turn this model into the one the thesis controls, and the
Hovershoes of thesis sec. 6.1.3. Without them the robot is in free fall, so
this model is a building block and not yet a walking robot. The parameter
values are placeholders: the thesis does not tabulate its own.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E2, E3, GRAVITY, I3, Factor, Robot, hat


@dataclass
class Params:
    """Placeholders of a plausible size. The thesis does not tabulate its values."""
    m_b: float = 10.0         # pelvis mass, kg
    m_h: float = 5.0          # mass at each hip, kg
    m_k: float = 3.0          # mass at each knee, kg
    d: float = 0.135          # half the hip width, m
    J_b: np.ndarray = field(default_factory=lambda: np.diag([0.09, 0.05, 0.07]))
    J_leg: np.ndarray = field(default_factory=lambda: np.diag([0.06, 0.06, 0.01]))
    g: float = GRAVITY


@dataclass
class State:
    x: np.ndarray             # pelvis position
    v: np.ndarray
    R: np.ndarray             # pelvis attitude
    Omega: np.ndarray
    R_r: np.ndarray           # right leg attitude
    Omega_r: np.ndarray
    R_l: np.ndarray           # left leg attitude
    Omega_l: np.ndarray
    l_r: float                # right leg length, m
    l_r_dot: float
    l_l: float                # left leg length, m
    l_l_dot: float


class Cassie(Robot):
    name = "cassie"
    manifold = "R^3 x SO(3) x SO(3)^2 x R^2"
    factors = [Factor("R", "x", "v"), Factor("SO3", "R", "Omega"),
               Factor("SO3", "R_r", "Omega_r"), Factor("SO3", "R_l", "Omega_l"),
               Factor("R", "l_r", "l_r_dot", size=1), Factor("R", "l_l", "l_l_dot", size=1)]
    Params, State = Params, State
    labels = {"manifold": "tbl2-robots", "section": "sec2-products"}

    def zero_input(self):
        return np.zeros(3), 0.0, np.zeros(3), 0.0

    def _legs(self, state):
        """(side, attitude, angular velocity, length, length rate) of the right and left leg."""
        return ((-1.0, state.R_r, state.Omega_r, state.l_r, state.l_r_dot),
                (+1.0, state.R_l, state.Omega_l, state.l_l, state.l_l_dot))

    def dynamics(self, state, u, params=None):
        """(v_dot, Omega_dot, Omega_r_dot, Omega_l_dot, l_r_ddot, l_l_ddot)
        for u = (tau_r, f_r, tau_l, f_l): hip torques in the leg frames, leg forces.
        Floating base: no ground contact."""
        p = params or self.params
        R, Om = state.R, state.Omega
        # Unknowns z = (v_dot, Omega_dot, Omega_r_dot, Omega_l_dot, l_r_ddot, l_l_ddot).
        A, b = np.zeros((14, 14)), np.zeros(14)
        A[:3, :3], b[:3] = p.m_b * I3, -p.m_b * p.g * E3
        A[3:6, 3:6], b[3:6] = p.J_b, -np.cross(Om, p.J_b @ Om)
        for k, (side, R_leg, Om_leg, l, l_dot) in enumerate(self._legs(state)):
            rot, ext = slice(6 + 3 * k, 9 + 3 * k), 12 + k
            tau, f = u[2 * k], u[2 * k + 1]
            # hip: acceleration Ja z + bias
            Ja = np.zeros((3, 14))
            Ja[:, :3], Ja[:, 3:6] = I3, -side * p.d * R @ hat(E2)
            bias = side * p.d * R @ np.cross(Om, np.cross(Om, E2))
            A += p.m_h * Ja.T @ Ja
            b -= p.m_h * Ja.T @ (bias + p.g * E3)
            # knee: the hip, plus half the leg
            Ja[:, rot], Ja[:, ext] = -0.5 * l * R_leg @ hat(E3), 0.5 * R_leg @ E3
            bias = bias + R_leg @ (l_dot * np.cross(Om_leg, E3)
                                   + 0.5 * l * np.cross(Om_leg, np.cross(Om_leg, E3)))
            A += p.m_k * Ja.T @ Ja
            b -= p.m_k * Ja.T @ (bias + p.g * E3)
            # rotation of the leg, hip torque (reaction on the pelvis) and leg force
            A[rot, rot] += p.J_leg
            b[rot] += tau - np.cross(Om_leg, p.J_leg @ Om_leg)
            b[3:6] -= R.T @ R_leg @ tau
            b[ext] += f
        z = np.linalg.solve(A, b)
        return z[:3], z[3:6], z[6:9], z[9:12], z[12], z[13]

    def feet(self, state, params=None):
        """Positions of the right and left foot."""
        p = params or self.params
        return [state.x + side * p.d * state.R @ E2 + l * R_leg @ E3
                for side, R_leg, _, l, _ in self._legs(state)]

    def energy(self, state, params=None):
        """Kinetic plus potential energy of the floating-base model."""
        p = params or self.params
        total = (0.5 * p.m_b * state.v @ state.v + p.m_b * p.g * state.x[2]
                 + 0.5 * state.Omega @ p.J_b @ state.Omega)
        for side, R_leg, Om, l, l_dot in self._legs(state):
            hip = state.x + side * p.d * state.R @ E2
            v_hip = state.v + side * p.d * state.R @ np.cross(state.Omega, E2)
            knee = hip + 0.5 * l * R_leg @ E3
            v_knee = v_hip + 0.5 * l_dot * R_leg @ E3 + 0.5 * l * R_leg @ np.cross(Om, E3)
            total += (0.5 * p.m_h * v_hip @ v_hip + p.m_h * p.g * hip[2]
                      + 0.5 * p.m_k * v_knee @ v_knee + p.m_k * p.g * knee[2]
                      + 0.5 * Om @ p.J_leg @ Om)
        return total

    def lagrangian(self):
        from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m_b, m_h, m_k, d, g = getScalars("m_b m_h m_k d g", attr=["Constant"])
        J_b = Matrix("J_b", attr=["Constant", "SymmetricMatrix"])
        J_leg = Matrix("J_leg", attr=["Constant", "SymmetricMatrix"])
        e2, e3 = Vector("e2", attr=["Constant"]), Vector("e3", attr=["Constant"])
        x, R = Vector("x"), SO3("R")
        Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
        L = (half * m_b * Dot(x.t_diff(), x.t_diff()) + half * Dot(Omega, J_b * Omega)
             - m_b * g * Dot(e3, x))
        dW, legs, lengths, inputs = None, [], [], []
        for name, sign in (("r", -1.0), ("l", 1.0)):
            side = Scalar(str(sign), value=sign, attr=["Constant"])
            R_leg, l = SO3(f"R_{name}"), Scalar(f"l_{name}")
            Om, eta_leg = R_leg.get_tangent_vector(), R_leg.get_variation_vector()
            tau, f = Vector(f"tau_{name}"), Scalar(f"f_{name}")
            hip = x + (R * e2) * d * side
            knee = hip + (R_leg * e3) * l * half
            L = (L + half * m_h * Dot(hip.t_diff(), hip.t_diff()) - m_h * g * Dot(e3, hip)
                 + half * m_k * Dot(knee.t_diff(), knee.t_diff()) - m_k * g * Dot(e3, knee)
                 + half * Dot(Om, J_leg * Om))
            # hip torque between pelvis and leg (leg frame), force along the leg
            work = Dot(R_leg * eta_leg - R * eta, R_leg * tau) + l.delta() * f
            dW = work if dW is None else dW + work
            legs.append(R_leg), lengths.append(l), inputs.extend([tau, f])
        return L, dW, SystemVariables(scalars=lengths, vectors=[x], matrices=[R, *legs]), inputs
