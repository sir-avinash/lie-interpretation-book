"""Reaction Mass Biped, stance phase (Siravuru, 2019, ch. 5), and its telescoping leg.

The biped stands on one foot, pinned at the origin. It has

* a telescoping stance leg with attitude R1 and length l: the hip is at
  l R1 e3, and every mass of the robot is balanced about the hip, so the
  center of mass is the hip;
* a torso with attitude RP that carries three pairs of reaction masses. Pair i
  slides along the torso axis RP e_i, one mass on each side of the hip at
  distance slide[i];
* a swing leg with attitude R2, whose center of mass is at the hip.

Write Pi = I - e3 e3' and Pi_i = I - e_i e_i'. With m the total mass, m_r one
reaction mass and J_0 the inertia of a leg, the equations of motion are

    m l_ddot = m l Omega1 . Pi Omega1 - m g e3 . R1 e3 + f,
    J1(l) Omega1_dot + Omega1 x J1(l) Omega1 + 2 m l l_dot Pi Omega1
        = m g l (R1' e3) x e3 + tau_1 - R1' RP tau_P,
    JP(s) OmegaP_dot + OmegaP x JP(s) OmegaP + JP_dot OmegaP = tau_P - RP' R2 tau_2,
    2 m_r slide_i_ddot = 2 m_r slide_i OmegaP . Pi_i OmegaP + f_s_i,
    J_0 Omega2_dot + Omega2 x J_0 Omega2 = tau_2,

    J1(l) = J_0 + m l^2 Pi,   JP(s) = J_P + 2 m_r sum_i slide_i^2 Pi_i,
    JP_dot = 4 m_r sum_i slide_i slide_i_dot Pi_i.

The first two lines, with the hip torque tau_P left out, are the telescoping
leg eq4-leg. The inputs are the leg force f, the ankle torque tau_1, the hip
torques tau_P (stance leg on torso) and tau_2 (torso on swing leg), all in the
body frame of the body they act on, and the forces f_s on the three pairs.

What is included: the continuous stance dynamics, thirteen degrees of freedom,
all actuated. What is not: the length of the swing leg (it does not enter the
stance dynamics, because the swing leg's center of mass is at the hip), the
floating-base model, the impact map and the walking controller of the thesis.

A sign differs from the thesis as printed: its eq. (5.14) has -L(s, Omega_P)
where the Lagrangian gives +, a centrifugal force that pushes the reaction
masses outward. The form above is the one geomech derives from the Lagrangian.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import E3, GRAVITY, I3, Factor, Robot

PI = I3 - np.outer(E3, E3)                                     # projection off the leg axis
PI_I = [I3 - np.outer(e, e) for e in np.eye(3)]                # projections off e_1, e_2, e_3


# =====================================================================
# the telescoping leg alone (Example ex4-leg)
# =====================================================================
@dataclass
class LegParams:
    m: float = 30.0                                                           # kg
    J_0: np.ndarray = field(default_factory=lambda: np.diag([0.9, 1.1, 0.4]))  # kg m^2
    g: float = GRAVITY


@dataclass
class LegState:
    l: float = 1.0                                                  # leg length, m
    l_dot: float = 0.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))        # leg attitude
    Omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body angular velocity


def leg_accelerations(l, l_dot, R, Omega, f, tau, m, J_0, g):
    """eq4-leg: returns (l_ddot, Omega_dot)."""
    J = J_0 + m * l**2 * PI
    l_ddot = l * Omega @ PI @ Omega - g * E3 @ (R @ E3) + f / m
    Omega_dot = np.linalg.solve(J, m * g * l * np.cross(R.T @ E3, E3) + tau
                                - np.cross(Omega, J @ Omega) - 2 * m * l * l_dot * PI @ Omega)
    return l_ddot, Omega_dot


class TelescopingLeg(Robot):
    name = "telescoping_leg"
    manifold = "R x SO(3)"
    factors = [Factor("R", "l", "l_dot", size=1), Factor("SO3", "R", "Omega")]
    Params, State = LegParams, LegState
    labels = {"equations": "eq4-leg", "example": "ex4-leg", "section": "sec4-products"}

    def zero_input(self):
        return 0.0, np.zeros(3)

    def dynamics(self, state, u, params=None):
        """(l_ddot, Omega_dot) for u = (f, tau)."""
        p = params or self.params
        return leg_accelerations(state.l, state.l_dot, state.R, state.Omega, u[0], u[1],
                                 p.m, p.J_0, p.g)

    def energy(self, state, params=None):
        p = params or self.params
        J = p.J_0 + p.m * state.l**2 * PI
        return (0.5 * p.m * state.l_dot**2 + 0.5 * state.Omega @ J @ state.Omega
                + p.m * p.g * state.l * E3 @ (state.R @ E3))

    def lagrangian(self):
        from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m, g = getScalars("m g", attr=["Constant"])
        J_0 = Matrix("J_0", attr=["Constant", "SymmetricMatrix"])
        e3 = Vector("e3", attr=["Constant"])
        l, R = Scalar("l"), SO3("R")
        Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
        f, tau = Scalar("f"), Vector("tau")
        hip = (R * e3) * l
        L = (half * m * Dot(hip.t_diff(), hip.t_diff()) + half * Dot(Omega, J_0 * Omega)
             - m * g * Dot(e3, hip))
        return L, l.delta() * f + Dot(eta, tau), SystemVariables(scalars=[l], matrices=[R]), [f, tau]


# =====================================================================
# the biped in stance
# =====================================================================
@dataclass
class Params:
    """Mass properties chosen by Siravuru (2019, sec. 5.2) to resemble a NAO robot."""
    m_leg: float = 0.882                      # mass of one leg, at the hip, kg
    m_r: float = 0.32                         # one reaction mass, kg
    J_0: np.ndarray = field(default_factory=lambda: 0.5 * np.diag([0.98, 0.91, 0.63]))
    J_P: np.ndarray = field(default_factory=lambda: np.array(
        [[0.2126, 0.0004, -0.0002], [0.0004, 0.2042, 0.0010], [-0.0002, 0.0010, 0.2246]]))
    g: float = GRAVITY

    @property
    def m(self):
        """Total mass: two legs and six reaction masses."""
        return 2 * self.m_leg + 6 * self.m_r


@dataclass
class State:
    l: float                  # stance leg length, m
    l_dot: float
    R1: np.ndarray            # stance leg attitude
    Omega1: np.ndarray
    RP: np.ndarray            # torso attitude
    OmegaP: np.ndarray
    R2: np.ndarray            # swing leg attitude
    Omega2: np.ndarray
    slide: np.ndarray         # distance of each pair of reaction masses from the hip, m (3,)
    slide_dot: np.ndarray


class ReactionMassBiped(Robot):
    name = "reaction_mass_biped"
    manifold = "R x SO(3)^3 x R^3"
    factors = [Factor("R", "l", "l_dot", size=1), Factor("SO3", "R1", "Omega1"),
               Factor("SO3", "RP", "OmegaP"), Factor("SO3", "R2", "Omega2"),
               Factor("R", "slide", "slide_dot")]
    Params, State = Params, State
    labels = {"equations": "eq4-leg", "example": "ex4-leg", "manifold": "tbl2-robots",
              "section": "sec4-products"}

    def zero_input(self):
        return 0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    def torso_inertia(self, slide, params=None):
        p = params or self.params
        return p.J_P + 2 * p.m_r * sum(s**2 * P for s, P in zip(slide, PI_I))

    def dynamics(self, state, u, params=None):
        """(l_ddot, Omega1_dot, OmegaP_dot, Omega2_dot, slide_ddot)
        for u = (f, tau_1, tau_P, tau_2, f_s)."""
        p = params or self.params
        f, tau_1, tau_P, tau_2, f_s = u
        R1, RP, R2, OmP, Om2 = state.R1, state.RP, state.R2, state.OmegaP, state.Omega2
        l_ddot, Omega1_dot = leg_accelerations(state.l, state.l_dot, R1, state.Omega1, f,
                                               tau_1 - R1.T @ RP @ tau_P, p.m, p.J_0, p.g)
        JP = self.torso_inertia(state.slide, p)
        JP_dot = 4 * p.m_r * sum(s * sd * P for s, sd, P in zip(state.slide, state.slide_dot, PI_I))
        OmegaP_dot = np.linalg.solve(JP, tau_P - RP.T @ R2 @ tau_2
                                     - np.cross(OmP, JP @ OmP) - JP_dot @ OmP)
        Omega2_dot = np.linalg.solve(p.J_0, tau_2 - np.cross(Om2, p.J_0 @ Om2))
        slide_ddot = np.array([s * OmP @ P @ OmP for s, P in zip(state.slide, PI_I)]) \
            + np.asarray(f_s) / (2 * p.m_r)
        return l_ddot, Omega1_dot, OmegaP_dot, Omega2_dot, slide_ddot

    def energy(self, state, params=None):
        p = params or self.params
        J1 = p.J_0 + p.m * state.l**2 * PI
        JP = self.torso_inertia(state.slide, p)
        return (0.5 * p.m * state.l_dot**2 + p.m_r * state.slide_dot @ state.slide_dot
                + 0.5 * state.Omega1 @ J1 @ state.Omega1 + 0.5 * state.OmegaP @ JP @ state.OmegaP
                + 0.5 * state.Omega2 @ p.J_0 @ state.Omega2
                + p.m * p.g * state.l * E3 @ (state.R1 @ E3))

    def lagrangian(self):
        """Built from the positions of the masses, not from the reduced kinetic energy:
        hip b = l R1 e3, reaction masses at b +- slide_i RP e_i."""
        from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, getScalars
        half = Scalar("0.5", value=0.5, attr=["Constant"])
        m_leg, m_r, g = getScalars("m_leg m_r g", attr=["Constant"])
        J_0 = Matrix("J_0", attr=["Constant", "SymmetricMatrix"])
        J_P = Matrix("J_P", attr=["Constant", "SymmetricMatrix"])
        e = [Vector(f"e{i}", attr=["Constant"]) for i in (1, 2, 3)]
        l, R1, RP, R2 = Scalar("l"), SO3("R1"), SO3("RP"), SO3("R2")
        s = [Scalar(f"s{i}") for i in (1, 2, 3)]
        Om1, OmP, Om2 = (R.get_tangent_vector() for R in (R1, RP, R2))
        eta1, etaP, eta2 = (R.get_variation_vector() for R in (R1, RP, R2))
        f, tau_1, tau_P, tau_2 = Scalar("f"), Vector("tau_1"), Vector("tau_P"), Vector("tau_2")
        f_s = [Scalar(f"f_s{i}") for i in (1, 2, 3)]
        hip = (R1 * e[2]) * l
        v = hip.t_diff()
        L = (m_leg * Dot(v, v) - (m_leg + m_leg) * g * Dot(e[2], hip)       # two legs at the hip
             + half * Dot(Om1, J_0 * Om1) + half * Dot(OmP, J_P * OmP) + half * Dot(Om2, J_0 * Om2))
        for si, ei in zip(s, e):
            for sign in (1.0, -1.0):
                side = Scalar(str(sign), value=sign, attr=["Constant"])
                p = hip + (RP * ei) * si * side
                L = L + half * m_r * Dot(p.t_diff(), p.t_diff()) - m_r * g * Dot(e[2], p)
        # eta_P1 = eta_P - RP' R1 eta_1 and eta_P2 = eta_2 - R2' RP eta_P, written in the world frame
        dW = (l.delta() * f + Dot(eta1, tau_1) + Dot(RP * etaP - R1 * eta1, RP * tau_P)
              + Dot(R2 * eta2 - RP * etaP, R2 * tau_2))
        for si, fi in zip(s, f_s):
            dW = dW + si.delta() * fi
        variables = SystemVariables(scalars=[l, *s], matrices=[R1, RP, R2])
        return L, dW, variables, [f, tau_1, tau_P, tau_2, *f_s]
