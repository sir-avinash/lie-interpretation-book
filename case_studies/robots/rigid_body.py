"""Rigid body rotating on SO(3): the free rigid body and the 3D pendulum.

    J Omega_dot + Omega x J Omega = m g (R^T e3) x rho + tau,     R_dot = R hat(Omega)

is eq4-pendulum: a body hinged at a pivot, with inertia J about the pivot and
its center of mass at R rho. With rho = 0 the gravity term vanishes and the
equation is Euler's equation eq4-euler-free with a body torque (the attitude
dynamics of any vehicle).
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E3, GRAVITY, I3, Factor, Robot, attitude_tracking_torque, hat


@dataclass
class Params:
    """The 3D pendulum of Chapter 4's opening."""
    J: np.ndarray = field(default_factory=lambda: np.diag([0.20, 0.25, 0.10]))  # kg·m²
    m: float = 1.0                                                               # kg
    rho: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.4]))  # m, body frame
    g: float = GRAVITY


def book_params():
    """The paperback of sec4-conservation: 0.20 m x 0.13 m x 0.03 m, 0.3 kg, free."""
    m, a, b, c = 0.3, 0.20, 0.13, 0.03
    J = m / 12 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
    return Params(J=J, m=m, rho=np.zeros(3))


@dataclass
class State:
    R: np.ndarray = field(default_factory=lambda: np.eye(3))        # attitude, body to world
    Omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body angular velocity


class Pendulum3D(Robot):
    name = "pendulum_3d"
    manifold = "SO(3)"
    factors = [Factor("SO3", "R", "Omega")]
    Params, State = Params, State
    labels = {"equations": "eq4-pendulum", "manifold": "tbl2-robots", "section": "sec4-forces",
              "tracking": "eq6-law", "linearization": "eq7-so3-lin"}

    def zero_input(self):
        return np.zeros(3)

    def gravity_torque(self, R, params=None):
        p = params or self.params
        return p.m * p.g * np.cross(R.T @ E3, p.rho)

    def dynamics(self, state, u, params=None):
        """(Omega_dot,) for a body torque u = tau."""
        p = params or self.params
        R, Omega = state.R, state.Omega
        return (np.linalg.solve(p.J, self.gravity_torque(R, p) + u
                                - np.cross(Omega, p.J @ Omega)),)

    def energy(self, state, params=None):
        p = params or self.params
        return 0.5 * state.Omega @ p.J @ state.Omega + p.m * p.g * E3 @ (state.R @ p.rho)

    def momentum(self, state, params=None):
        """World-frame angular momentum R J Omega. With tau = 0 its component along
        e3 is conserved, and all of it for the free rigid body."""
        p = params or self.params
        return state.R @ p.J @ state.Omega

    def tracking_controller(self, state, Rd, Omegad, Omegad_dot, k_R=8.81, k_Omega=2.54,
                            params=None):
        """The attitude tracking law eq6-law. For the pendulum the gravity torque of
        eq4-pendulum is canceled as well (it is zero when rho = 0)."""
        p = params or self.params
        return (attitude_tracking_torque(state.R, state.Omega, Rd, Omegad, Omegad_dot,
                                         p.J, k_R, k_Omega) - self.gravity_torque(state.R, p))

    def linearize(self, Rd, Omegad, params=None):
        """Variation-based linearization along a solution (Rd, Omegad), thm7-so3.
        Returns A, B of eq7-so3-lin for the linear state s = (eta, delta Omega)."""
        p = params or self.params
        Ji = np.linalg.inv(p.J)
        A = np.zeros((6, 6))
        A[:3, :3] = -hat(Omegad)
        A[:3, 3:] = I3
        A[3:, :3] = -p.m * p.g * Ji @ hat(p.rho) @ hat(Rd.T @ E3)
        A[3:, 3:] = Ji @ (hat(p.J @ Omegad) - hat(Omegad) @ p.J)
        return A, np.vstack([np.zeros((3, 3)), Ji])

    def lagrangian(self):
        from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m, g = getScalars("m g", attr=["Constant"])
        J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
        e3, rho = Vector("e3", attr=["Constant"]), Vector("rho", attr=["Constant"])
        R, tau = SO3("R"), Vector("tau")
        Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
        L = half * Dot(Omega, J * Omega) - m * g * Dot(e3, R * rho)
        return L, Dot(eta, tau), SystemVariables(matrices=[R]), [tau]


class FreeRigidBody(Pendulum3D):
    """Euler's equation J Omega_dot + Omega x J Omega = tau (eq4-euler-free with a torque)."""
    name = "rigid_body"
    labels = {"equations": "eq4-euler-free", "manifold": "tbl2-robots", "section": "sec4-so3",
              "tracking": "eq6-law", "linearization": "eq7-so3-lin"}

    def __init__(self, params=None):
        super().__init__(params if params is not None else book_params())
