"""Quadrotor on SE(3), stored as R^3 x SO(3) (eq4-quadrotor).

    m x_ddot = f R e3 - m g e3,      J Omega_dot + Omega x J Omega = tau,
    x_dot = v,                       R_dot = R hat(Omega).

The input is u = (f, tau): the thrust f along the body axis R e3 and the body
torque tau. Six degrees of freedom, four inputs.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E3, GRAVITY, Factor, Robot, attitude_tracking_torque, hat, unit, vee


@dataclass
class Params:
    """The vehicle of Chapter 6 (Lee, Leok and McClamroch, 2010)."""
    m: float = 4.34                                                                    # kg
    J: np.ndarray = field(default_factory=lambda: np.diag([0.0820, 0.0845, 0.1377]))  # kg·m²
    g: float = GRAVITY


@dataclass
class Gains:
    """Gains of the cascade in Chapter 6, for the default vehicle."""
    k_x: float = 16 * 4.34
    k_v: float = 5.6 * 4.34
    k_R: float = 8.81
    k_Omega: float = 2.54


@dataclass
class State:
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))      # position, world frame
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))      # velocity, world frame
    R: np.ndarray = field(default_factory=lambda: np.eye(3))        # attitude, body to world
    Omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body angular velocity


def desired_attitude(F, heading):
    """eq6-Rd: third axis along F, first axis as close to the heading direction as
    that allows. heading is the unit vector R_z(psi_d) e1, with psi_d the desired
    heading (not the Euler yaw, unless the roll or the pitch is zero)."""
    b3 = unit(F)
    b2 = unit(np.cross(b3, heading))
    return np.column_stack([np.cross(b2, b3), b2, b3])


def desired_attitude_rates(F, F1, F2, hd, hd1, hd2, d=1e-4):
    """Rd, Omegad and Omegad_dot from F_d, the heading direction and their first two time
    derivatives, by central differences of eq6-Rd along the second-order curve."""
    def along(s):
        return desired_attitude(F + s * F1 + 0.5 * s * s * F2, hd + s * hd1 + 0.5 * s * s * hd2)

    Rd, Rp, Rm = along(0.0), along(d), along(-d)
    W = Rd.T @ (Rp - Rm) / (2 * d)
    Omegad = vee(0.5 * (W - W.T))
    W2 = Rd.T @ (Rp - 2 * Rd + Rm) / d**2 - hat(Omegad) @ hat(Omegad)
    return Rd, Omegad, vee(0.5 * (W2 - W2.T))


def circle_reference(t, rate=1.0, radius=1.5):
    """The reference of sec6-cascade: a circle of radius 1.5 m with a vertical wobble,
    heading along the tangent. Returns (xd, vd, ad, jd, sd, hd, hd1, hd2)."""
    c, s = np.cos(rate * t), np.sin(rate * t)
    c2, s2 = np.cos(2 * rate * t), np.sin(2 * rate * t)
    xd = np.array([radius * c, radius * s, 0.3 * s2])
    vd = np.array([-radius * rate * s, radius * rate * c, 0.6 * rate * c2])
    ad = np.array([-radius * rate**2 * c, -radius * rate**2 * s, -1.2 * rate**2 * s2])
    jd = np.array([radius * rate**3 * s, -radius * rate**3 * c, -2.4 * rate**3 * c2])
    sd = np.array([radius * rate**4 * c, radius * rate**4 * s, 4.8 * rate**4 * s2])
    psi = rate * t + np.pi / 2
    hd = np.array([np.cos(psi), np.sin(psi), 0.0])
    return xd, vd, ad, jd, sd, hd, rate * np.array([-np.sin(psi), np.cos(psi), 0.0]), -rate**2 * hd


class Quadrotor(Robot):
    name = "quadrotor"
    manifold = "R^3 x SO(3)"
    factors = [Factor("R", "x", "v"), Factor("SO3", "R", "Omega")]
    Params, State, Gains = Params, State, Gains
    labels = {"equations": "eq4-quadrotor", "manifold": "tbl2-robots", "section": "sec4-products",
              "tracking": ["eq6-Fd", "eq6-Rd", "eq6-law"], "linearization": "eq7-so3-lin"}

    def zero_input(self):
        return 0.0, np.zeros(3)

    def hover_input(self, params=None):
        p = params or self.params
        return p.m * p.g, np.zeros(3)

    def dynamics(self, state, u, params=None):
        """(v_dot, Omega_dot) for u = (f, tau)."""
        p = params or self.params
        f, tau = u
        v_dot = f * state.R @ E3 / p.m - p.g * E3
        Omega_dot = np.linalg.solve(p.J, tau - np.cross(state.Omega, p.J @ state.Omega))
        return v_dot, Omega_dot

    def energy(self, state, params=None):
        p = params or self.params
        return (0.5 * p.m * state.v @ state.v + 0.5 * state.Omega @ p.J @ state.Omega
                + p.m * p.g * E3 @ state.x)

    def tracking_controller(self, state, reference, gains=None, params=None):
        """The cascade of sec6-cascade: eq6-Fd, eq6-Rd, then eq6-law.

        reference = (xd, vd, ad, jd, sd, hd, hd1, hd2): the desired position and its
        first four time derivatives, then the heading direction R_z(psi_d) e1 and its
        first two derivatives. Returns u = (f, tau).

        Omegad and its rate are obtained by differentiating F_d along the model,
        as in Chapter 6: e_v_dot = f R e3 / m - g e3 - ad.
        """
        p, k = params or self.params, gains or self.Gains()
        xd, vd, ad, jd, sd, hd, hd1, hd2 = reference
        x, v, R, Omega = state.x, state.v, state.R, state.Omega
        e_x, e_v = x - xd, v - vd
        Fd = -k.k_x * e_x - k.k_v * e_v + p.m * p.g * E3 + p.m * ad          # eq6-Fd
        b3 = R @ E3
        f = max(Fd @ b3, 0.0)                                                 # eq6-Rd
        on = f > 0.0
        e_v1 = f * b3 / p.m - p.g * E3 - ad
        F1 = -k.k_x * e_v - k.k_v * e_v1 + p.m * jd
        b3_dot = R @ np.cross(Omega, E3)
        f1 = (F1 @ b3 + Fd @ b3_dot) if on else 0.0
        e_v2 = (f1 * b3 + f * b3_dot) / p.m - jd
        F2 = -k.k_x * e_v1 - k.k_v * e_v2 + p.m * sd
        Rd, Omegad, Omegad_dot = desired_attitude_rates(Fd, F1, F2, hd, hd1, hd2)
        tau = attitude_tracking_torque(R, Omega, Rd, Omegad, Omegad_dot, p.J,
                                       k.k_R, k.k_Omega)                      # eq6-law
        return f, tau

    def lagrangian(self):
        from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m, g = getScalars("m g", attr=["Constant"])
        J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
        e3 = Vector("e3", attr=["Constant"])
        x, R = Vector("x"), SO3("R")
        Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
        f, tau = Scalar("f"), Vector("tau")
        L = (half * m * Dot(x.t_diff(), x.t_diff()) + half * Dot(Omega, J * Omega)
             - m * g * Dot(e3, x))
        dW = Dot(x.delta(), (R * e3) * f) + Dot(eta, tau)
        return L, dW, SystemVariables(vectors=[x], matrices=[R]), [f, tau]
