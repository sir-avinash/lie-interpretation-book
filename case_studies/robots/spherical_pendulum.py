"""Spherical pendulum on S^2 (Section sec4-sphere, eq4-sph).

A point mass m on a massless rod of length l, hinged at the origin. The
direction q points from the pivot to the bob, so the bob is at l q. The input
is a force F on the bob, in the world frame (the convention of Chapters 6
and 7):

    m l^2 omega_dot = -m g l q x e3 + l q x F,        q_dot = omega x q.

With F = 0 this is eq4-sph. The rod tension is eq4-tension.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E3, GRAVITY, I3, Factor, Robot, e_omega, e_q, hat, tangent


@dataclass
class Params:
    m: float = 1.0          # mass of the bob, kg
    l: float = 1.0          # length of the rod, m
    g: float = GRAVITY


@dataclass
class State:
    q: np.ndarray = field(default_factory=lambda: -E3.copy())       # direction, in S^2
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world frame, omega . q = 0


class SphericalPendulum(Robot):
    name = "spherical_pendulum"
    manifold = "S^2"
    factors = [Factor("S2", "q", "omega")]
    Params, State = Params, State
    labels = {"equations": "eq4-sph", "manifold": "tbl2-robots", "section": "sec4-sphere",
              "tension": "eq4-tension", "tracking": "eq6-law-s2",
              "linearization": "eq7-s2-lin", "constraint": "eq7-s2-constraint"}

    def zero_input(self):
        return np.zeros(3)

    def dynamics(self, state, u, params=None):
        """(omega_dot,) for a force u = F on the bob."""
        p = params or self.params
        q = state.q
        return ((-p.m * p.g * p.l * np.cross(q, E3) + p.l * np.cross(q, u)) / (p.m * p.l**2),)

    def energy(self, state, params=None):
        p = params or self.params
        q_dot = np.cross(state.omega, state.q)
        return 0.5 * p.m * p.l**2 * q_dot @ q_dot + p.m * p.g * p.l * E3 @ state.q

    def tension(self, state, params=None):
        """eq4-tension: T = m l |q_dot|^2 - m g e3 . q (no force on the bob)."""
        p = params or self.params
        q_dot = np.cross(state.omega, state.q)
        return p.m * p.l * q_dot @ q_dot - p.m * p.g * E3 @ state.q

    def tracking_controller(self, state, qd, omegad, omegad_dot, k_q=16.0, k_omega=5.6,
                            params=None):
        """The direction tracking law, eq6-law-s2. Returns the force F on the bob.

        omega_dot = (I - q q^T) omegad_dot - (q . omegad) q_dot - k_q e_q - k_omega e_omega,
        F = m g e3 + m l omega_dot x q.
        """
        p = params or self.params
        q, omega = state.q, state.omega
        omega_dot = (tangent(q, omegad_dot) - (q @ omegad) * np.cross(omega, q)
                     - k_q * e_q(q, qd) - k_omega * e_omega(q, omega, omegad))
        return p.m * p.g * E3 + p.m * p.l * np.cross(omega_dot, q)

    def linearize(self, qd, omegad, Fd, params=None):
        """Variation-based linearization along a solution (qd, omegad, Fd), thm7-s2.

        Returns A, B of eq7-s2-lin for the linear state s = (xi, delta omega), and
        the constraint matrix C of eq7-s2-constraint, with C s = 0.
        """
        p = params or self.params
        A = np.zeros((6, 6))
        A[:3, :3] = hat(omegad)
        A[:3, 3:] = I3 - np.outer(qd, qd)
        A[3:, :3] = hat(Fd - p.m * p.g * E3) @ hat(qd) / (p.m * p.l)
        B = np.vstack([np.zeros((3, 3)), hat(qd) / (p.m * p.l)])
        C = np.block([[qd, np.zeros(3)], [np.cross(qd, omegad), qd]])
        return A, B, C

    def lagrangian(self):
        from geomech import S2, Cross, Dot, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m, l, g = getScalars("m l g", attr=["Constant"])
        e3 = Vector("e3", attr=["Constant"])
        q, F = S2("q"), Vector("F")
        omega, xi = q.get_tangent_vector(), q.get_variation_vector()
        q_dot = Cross(omega, q)
        L = half * m * l * l * Dot(q_dot, q_dot) - m * g * l * Dot(e3, q)
        dW = Dot(Cross(xi, q) * l, F)
        return L, dW, SystemVariables(vectors=[q]), [F]
