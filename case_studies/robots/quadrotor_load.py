"""Quadrotor with a cable-suspended load on R^3 x S^2 x SO(3) (eq4-load).

A point mass m_L hangs from the center of mass of the quadrotor (mass m_Q,
inertia J) on a massless cable of length l. The cable direction q points from
the quadrotor to the load, so x_Q = x_L - l q. While the cable is taut,

    (m_Q + m_L)(x_L_ddot + g e3) = (q . f R e3 - m_Q l |q_dot|^2) q,
    m_Q l omega_dot = -q x f R e3,                  q_dot = omega x q,
    J Omega_dot + Omega x J Omega = tau,            R_dot = R hat(Omega).

The tension is eq4-load-tension. The equations stay smooth when it goes
negative (Pitfall warn4-slack): check `tension` when you simulate.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E1, E3, GRAVITY, Factor, Robot, attitude_tracking_torque, e_q, unit
from .quadrotor import desired_attitude


@dataclass
class Params:
    """The worked example of sec4-products."""
    m_Q: float = 1.5        # quadrotor mass, kg
    m_L: float = 0.3        # load mass, kg
    l: float = 1.0          # cable length, m
    J: np.ndarray = field(default_factory=lambda: np.array(
        [[0.02, 0.0005, 0.0], [0.0005, 0.025, 0.001], [0.0, 0.001, 0.04]]))   # kg m^2
    g: float = GRAVITY


def chapter6_params():
    """The vehicle of Chapter 6 carrying 0.5 kg on a 1 m cable."""
    return Params(m_Q=4.34, m_L=0.5, l=1.0, J=np.diag([0.0820, 0.0845, 0.1377]))


@dataclass
class Gains:
    """Gains of the load cascade in sec6-cascade, for chapter6_params(). The attitude
    loop is the one of the bare vehicle made twice as fast, (k_R, k_Omega) scaled by
    (4, 2): with the bare gains the cascade diverges from a 40 degree swing."""
    k_x: float = 4.0 * (4.34 + 0.5)
    k_v: float = 2.8 * (4.34 + 0.5)
    k_q: float = 25.0
    k_omega: float = 7.0
    k_R: float = 4 * 8.81
    k_Omega: float = 2 * 2.54


@dataclass
class State:
    x_L: np.ndarray = field(default_factory=lambda: np.zeros(3))    # load position
    v_L: np.ndarray = field(default_factory=lambda: np.zeros(3))    # load velocity
    q: np.ndarray = field(default_factory=lambda: -E3.copy())       # cable direction
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # of the cable, world frame
    R: np.ndarray = field(default_factory=lambda: np.eye(3))        # quadrotor attitude
    Omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body angular velocity


class QuadrotorLoad(Robot):
    name = "quadrotor_load"
    manifold = "R^3 x S^2 x SO(3)"
    factors = [Factor("R", "x_L", "v_L"), Factor("S2", "q", "omega"),
               Factor("SO3", "R", "Omega")]
    Params, State, Gains = Params, State, Gains
    labels = {"equations": "eq4-load", "manifold": "tbl2-robots", "figure": "fig2-quadload",
              "section": "sec4-products", "tension": "eq4-load-tension",
              "symbolic": "eq5-wu", "tracking": "eq6-load", "pitfall": "warn4-slack"}

    def zero_input(self):
        return 0.0, np.zeros(3)

    def hover_input(self, params=None):
        p = params or self.params
        return (p.m_Q + p.m_L) * p.g, np.zeros(3)

    def dynamics(self, state, u, params=None):
        """(v_L_dot, omega_dot, Omega_dot) for u = (f, tau)."""
        p = params or self.params
        f, tau = u
        q, omega = state.q, state.omega
        thrust = f * state.R @ E3
        q_dot = np.cross(omega, q)
        v_L_dot = (q @ thrust - p.m_Q * p.l * q_dot @ q_dot) * q / (p.m_Q + p.m_L) - p.g * E3
        omega_dot = -np.cross(q, thrust) / (p.m_Q * p.l)
        Omega_dot = np.linalg.solve(p.J, tau - np.cross(state.Omega, p.J @ state.Omega))
        return v_L_dot, omega_dot, Omega_dot

    def quadrotor_position(self, state, params=None):
        p = params or self.params
        return state.x_L - p.l * state.q

    def energy(self, state, params=None):
        p = params or self.params
        v_Q = state.v_L - p.l * np.cross(state.omega, state.q)
        return (0.5 * p.m_Q * v_Q @ v_Q + 0.5 * p.m_L * state.v_L @ state.v_L
                + 0.5 * state.Omega @ p.J @ state.Omega
                + p.m_Q * p.g * E3 @ self.quadrotor_position(state, p)
                + p.m_L * p.g * E3 @ state.x_L)

    def tension(self, state, u, params=None):
        """eq4-load-tension. Positive when the cable pulls."""
        p = params or self.params
        q_dot = np.cross(state.omega, state.q)
        return p.m_L / (p.m_Q + p.m_L) * (p.m_Q * p.l * q_dot @ q_dot
                                          - state.q @ (u[0] * state.R @ E3))

    def tracking_controller(self, state, xd, gains=None, params=None, heading=E1):
        """The load cascade eq6-load for a fixed target xd, then eq6-Rd and eq6-law.

        F_L = -k_x e_x - k_v e_v + (m_Q + m_L) g e3 is the force wanted on the pair.
        The cable is steered to q_d = -(F_L + m_Q l |q_dot|^2 q) / |...| with the
        direction law (omega_d = 0), and the attitude follows the thrust vector.
        Returns u = (f, tau).
        """
        p, k = params or self.params, gains or self.Gains()
        q, omega, R, Omega = state.q, state.omega, state.R, state.Omega
        F_L = -k.k_x * (state.x_L - xd) - k.k_v * state.v_L + (p.m_Q + p.m_L) * p.g * E3
        A = F_L + p.m_Q * p.l * (omega @ omega) * q
        qd = -unit(A)
        omega_dot = -k.k_q * e_q(q, qd) - k.k_omega * omega
        thrust = (A @ q) * q + p.m_Q * p.l * np.cross(q, omega_dot)          # eq6-load
        f = max(thrust @ (R @ E3), 0.0)
        Rd = desired_attitude(thrust, heading)                                # eq6-Rd
        tau = attitude_tracking_torque(R, Omega, Rd, np.zeros(3), np.zeros(3), p.J,
                                       k.k_R, k.k_Omega)                      # eq6-law
        return f, tau

    def lagrangian(self):
        from geomech import (S2, SO3, Cross, Dot, Matrix, Scalar, SystemVariables, Vector,
                             getScalars)
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m_Q, m_L, l, g = getScalars("m_Q m_L l g", attr=["Constant"])
        J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
        e3 = Vector("e3", attr=["Constant"])
        x_L, q, R = Vector("x_L"), S2("q"), SO3("R")
        Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
        xi = q.get_variation_vector()
        f, tau = Scalar("f"), Vector("tau")
        x_Q = x_L - q * l
        v_Q, v_L = x_Q.t_diff(), x_L.t_diff()
        L = (half * m_Q * Dot(v_Q, v_Q) + half * m_L * Dot(v_L, v_L) + half * Dot(Omega, J * Omega)
             - m_Q * g * Dot(e3, x_Q) - m_L * g * Dot(e3, x_L))
        dW = Dot(x_L.delta() - Cross(xi, q) * l, (R * e3) * f) + Dot(eta, tau)
        return L, dW, SystemVariables(vectors=[x_L, q], matrices=[R]), [f, tau]
