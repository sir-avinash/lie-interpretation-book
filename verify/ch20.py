"""Chapter 20 verification: the legged case study.

What is assembled here, and from where:

* the 3D pendulum and the two feedback-linearizing laws: verify/ch06.py (functions
  taken by AST, because that script runs its checks at import);
* the stance dynamics of the Reaction Mass Biped: the robot model library
  (case_studies/robots/reaction_mass_biped.py), derived again with geomech;
* the floating-base model, the impact map, the motion primitive, the two tracking
  laws, the hybrid simulation, the linearization and the LQR: written here, each
  checked below.

Run:  python verify/ch20.py
"""
from __future__ import annotations

import ast
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "case_studies"))

import robots  # noqa: E402
from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee  # noqa: E402
from robots.core import log_so3  # noqa: E402

E1, E2, E3 = np.eye(3)
I3 = np.eye(3)
G = 9.81
PI = I3 - np.outer(E3, E3)                           # projection off the leg axis
PI_I = [I3 - np.outer(e, e) for e in np.eye(3)]      # projections off e_1, e_2, e_3

biped = robots.get("reaction_mass_biped")
PRM = biped.params
M_TOT = PRM.m                                        # 2 m_leg + 6 m_r


# =====================================================================
# 1. The floating-base model and the impact map
# =====================================================================
# Floating-base state: hip position x_P and velocity v_P, the three attitudes and
# body rates, the slides. Both leg axes R1 e3 and R2 e3 point from foot to hip, so
#     stance foot = x_P - l R1 e3,     swing foot = x_P - l_0 R2 e3.
def hip_velocity(l, l_dot, R1, Omega1):
    """Velocity of the hip x_P = l R1 e3 in stance."""
    return l_dot * R1 @ E3 + l * R1 @ np.cross(Omega1, E3)


def swing_foot(state, l_0):
    """Position of the swing foot relative to the stance foot."""
    return state.l * state.R1 @ E3 - l_0 * state.R2 @ E3


def swing_foot_velocity(state, l_0):
    return hip_velocity(state.l, state.l_dot, state.R1, state.Omega1) \
        - l_0 * state.R2 @ np.cross(state.Omega2, E3)


def impact(state, l_0, p=PRM):
    """The swing foot lands and sticks: eq20-impact.

    Returns (new stance state, impulse on the foot, step vector). Only the swing
    leg and the hip velocity change:
        J(l_0) Omega2+ = J_0 Omega2- + m l_0 e3 x R2^T v_P-,   J(l) = J_0 + m l^2 Pi,
        v_P+ = l_0 R2 (Omega2+ x e3),      impulse = m (v_P+ - v_P-).
    Then the legs trade names: the old swing leg is the new stance leg.
    """
    m = p.m
    v_minus = hip_velocity(state.l, state.l_dot, state.R1, state.Omega1)
    J_c = p.J_0 + m * l_0**2 * PI
    Omega2_plus = np.linalg.solve(J_c, p.J_0 @ state.Omega2 + m * l_0 * np.cross(E3, state.R2.T @ v_minus))
    v_plus = l_0 * state.R2 @ np.cross(Omega2_plus, E3)
    new = replace(state, l=l_0, l_dot=0.0, R1=state.R2, Omega1=Omega2_plus,
                  R2=state.R1, Omega2=state.Omega1)
    return new, m * (v_plus - v_minus), swing_foot(state, l_0)


def impact_matrices(R2, slide, l_0, p=PRM):
    """D and A of the floating-base model in the velocity order
    (Omega1, OmegaP, slide_dot, Omega2, v_P): D v+ - D v- = A^T impulse, A v+ = 0."""
    D = np.zeros((15, 15))
    D[0:3, 0:3] = p.J_0
    D[3:6, 3:6] = biped.torso_inertia(slide, p)
    D[6:9, 6:9] = 2 * p.m_r * I3
    D[9:12, 9:12] = p.J_0
    D[12:15, 12:15] = p.m * I3
    A = np.zeros((3, 15))
    A[:, 9:12] = l_0 * R2 @ hat(E3)          # d/dt(-l_0 R2 e3) = l_0 R2 hat(e3) Omega2
    A[:, 12:15] = I3
    return D, A


def impact_kkt(state, l_0, p=PRM, rhs_sign=+1.0):
    """The impact solved as the linear system of the thesis, eq. (5.20), with the right
    side +D v- (rhs_sign=-1 gives the system as printed)."""
    D, A = impact_matrices(state.R2, state.slide, l_0, p)
    v = np.concatenate([state.Omega1, state.OmegaP, state.slide_dot, state.Omega2,
                        hip_velocity(state.l, state.l_dot, state.R1, state.Omega1)])
    K = np.block([[D, -A.T], [A, np.zeros((3, 3))]])
    sol = np.linalg.solve(K, np.concatenate([rhs_sign * D @ v, np.zeros(3)]))
    return v, sol[:15], sol[15:], D, A


def relabel(v):
    """Trade the names of the legs in the floating-base velocity."""
    return np.concatenate([v[9:12], v[3:9], v[0:3], v[12:15]])


def particles(state, v_P, p=PRM):
    """The point masses of the floating-base model: (mass, position from the hip, velocity)."""
    out = [(2 * p.m_leg, np.zeros(3), v_P)]
    for i in range(3):
        a = state.RP[:, i]
        a_dot = state.RP @ np.cross(state.OmegaP, np.eye(3)[i])
        for sign in (1.0, -1.0):
            out.append((p.m_r, sign * state.slide[i] * a,
                        v_P + sign * (state.slide_dot[i] * a + state.slide[i] * a_dot)))
    return out


def angular_momentum_about(point_from_hip, state, v_P, Omega1, OmegaP, Omega2, p=PRM):
    """World-frame angular momentum about a point, built mass by mass."""
    H = state.R1 @ p.J_0 @ Omega1 + state.RP @ p.J_P @ OmegaP + state.R2 @ p.J_0 @ Omega2
    for mass, r, v in particles(replace(state, OmegaP=OmegaP), v_P, p):
        H = H + mass * np.cross(r - point_from_hip, v)
    return H


def kinetic_floating(state, v_P, Omega1, OmegaP, Omega2, slide_dot, p=PRM):
    """Kinetic energy of the floating-base model, built mass by mass."""
    T = 0.5 * (Omega1 @ p.J_0 @ Omega1 + OmegaP @ p.J_P @ OmegaP + Omega2 @ p.J_0 @ Omega2)
    for mass, _, v in particles(replace(state, OmegaP=OmegaP, slide_dot=slide_dot), v_P, p):
        T += 0.5 * mass * v @ v
    return T


# =====================================================================
# 2. The motion primitive (thesis eqs. 5.41 and 5.42, in the book's convention)
# =====================================================================
@dataclass
class Gait:
    l_0: float = 0.9                    # nominal leg length, m        (thesis rho_0)
    l_bar: float = 0.1                  # leg extension in mid-stance  (thesis rho_bar)
    r_0: float = 0.125                  # nominal slide, m             (thesis s)
    r_bar: float = 0.025                # slide excursion, m           (thesis s_bar)
    t_s: float = 1.0                    # duration of the primitive, s (thesis T)
    theta_s: float = np.radians(15.0)   # legs swing from -15 to 15 degrees
    lean: float = np.radians(10.0)      # torso pitch, forward (our choice)
    roll: float = 0.0                   # torso roll into a turn


def primitive(t, psi, gait):
    """Desired stance state and accelerations at time t of a step with heading psi."""
    w = np.pi / gait.t_s
    th_s = gait.theta_s
    th = -th_s + 2 * th_s * np.sin(w * t / 2)
    th1 = th_s * w * np.cos(w * t / 2)
    th2 = -0.5 * th_s * w**2 * np.sin(w * t / 2)
    Rpsi = Rz(psi)
    one = np.ones(3)
    return dict(
        l=gait.l_0 + gait.l_bar * np.sin(w * t), l_dot=gait.l_bar * w * np.cos(w * t),
        l_ddot=-gait.l_bar * w**2 * np.sin(w * t),
        R1=Rpsi @ Ry(th), Omega1=th1 * E2, Omega1_dot=th2 * E2,
        R2=Rpsi @ Ry(-th), Omega2=-th1 * E2, Omega2_dot=-th2 * E2,
        RP=Rpsi @ Rx(gait.roll) @ Ry(gait.lean), OmegaP=np.zeros(3), OmegaP_dot=np.zeros(3),
        slide=(gait.r_0 + gait.r_bar * np.sin(w * t)) * one,
        slide_dot=gait.r_bar * w * np.cos(w * t) * one,
        slide_ddot=-gait.r_bar * w**2 * np.sin(w * t) * one)


def primitive_state(t, psi, gait):
    d = primitive(t, psi, gait)
    return biped.State(l=d["l"], l_dot=d["l_dot"], R1=d["R1"], Omega1=d["Omega1"], RP=d["RP"],
                       OmegaP=d["OmegaP"], R2=d["R2"], Omega2=d["Omega2"], slide=d["slide"],
                       slide_dot=d["slide_dot"])


# =====================================================================
# 3. Two tracking laws
# =====================================================================
@dataclass
class ThesisGains:
    """Table 5.3 of the thesis with eps = 0.25. The table lists L and no D_P: we use
    L as the torso damping."""
    eps: float = 0.25

    @property
    def k_l(self):
        return 16 / self.eps**2

    @property
    def k_ldot(self):
        return 8 / self.eps

    @property
    def D(self):
        return np.array([1.2, 1.5, 1.8]) / self.eps**2

    @property
    def k_Omega(self):
        """Damping of the stance leg, the torso and the swing leg."""
        return 2 / self.eps, 1.5 * 1.2, 0.5 / self.eps

    @property
    def k_r(self):
        return 1.2 / self.eps**2

    @property
    def k_rdot(self):
        return 1.5 / self.eps


def Psi_right(R, Rd, D):
    """tr(D (I - R Rd^T)): the weighted error function of the thesis."""
    return float(np.sum(D * (1.0 - np.diag(R @ Rd.T))))


def e_right(R, Rd, D):
    """Slope of Psi_right under delta R = R hat(eta): sum_i d_i (R^T e_i) x (Rd^T e_i)."""
    return sum(D[i] * np.cross(R[i, :], Rd[i, :]) for i in range(3))


def torso_inertia_rate(slide, slide_dot, p=PRM):
    return 4 * p.m_r * sum(s * sd * P for s, sd, P in zip(slide, slide_dot, PI_I))


def _stack(n_1, n_P, n_2, state):
    """Joint torques (tau_1, tau_P, tau_2) that put the net torques n on the bodies."""
    tau_2 = n_2
    tau_P = n_P + state.RP.T @ state.R2 @ tau_2
    tau_1 = n_1 + state.R1.T @ state.RP @ tau_P
    return tau_1, tau_P, tau_2


def thesis_law(state, d, gains=ThesisGains(), p=PRM):
    """Thesis eqs. (5.61) to (5.65) with Phi the identity, gravity along -e3."""
    m, D = p.m, gains.D
    k1, kP, k2 = gains.k_Omega
    l, l_dot, R1, Om1 = state.l, state.l_dot, state.R1, state.Omega1
    f = (m * d["l_ddot"] - m * l * Om1 @ PI @ Om1 + m * G * E3 @ (R1 @ E3)
         - gains.k_l * (l - d["l"]) - gains.k_ldot * (l_dot - d["l_dot"]))
    J1 = p.J_0 + m * l**2 * PI
    n_1 = (J1 @ d["Omega1_dot"] + np.cross(d["Omega1"], J1 @ Om1)
           - m * G * l * np.cross(R1.T @ E3, E3) + m * l * l_dot * PI @ (Om1 + d["Omega1"])
           - e_right(R1, d["R1"], D) - k1 * (Om1 - d["Omega1"]))
    JP = biped.torso_inertia(state.slide, p)
    JP_dot = torso_inertia_rate(state.slide, state.slide_dot, p)
    OmP = state.OmegaP
    n_P = (np.cross(d["OmegaP"], JP @ OmP) + JP @ d["OmegaP_dot"] + 0.5 * JP_dot @ (OmP + d["OmegaP"])
           - e_right(state.RP, d["RP"], D) - kP * (OmP - d["OmegaP"]))
    Om2 = state.Omega2
    n_2 = (np.cross(d["Omega2"], p.J_0 @ Om2) + p.J_0 @ d["Omega2_dot"]
           - e_right(state.R2, d["R2"], D) - k2 * (Om2 - d["Omega2"]))
    f_s = (2 * p.m_r * d["slide_ddot"]
           - 2 * p.m_r * np.array([s * OmP @ P @ OmP for s, P in zip(state.slide, PI_I)])
           - gains.k_r * (state.slide - d["slide"]) - gains.k_rdot * (state.slide_dot - d["slide_dot"]))
    tau_1, tau_P, tau_2 = _stack(n_1, n_P, n_2, state)
    return f, tau_1, tau_P, tau_2, f_s


def thesis_V(state, d, gains=ThesisGains(), p=PRM):
    """The Lyapunov function of the thesis, eq. (5.66), with Phi the identity."""
    m, D = p.m, gains.D
    J1 = p.J_0 + m * state.l**2 * PI
    JP = biped.torso_inertia(state.slide, p)
    w1, wP, w2 = state.Omega1 - d["Omega1"], state.OmegaP - d["OmegaP"], state.Omega2 - d["Omega2"]
    rt, rt_dot = state.slide - d["slide"], state.slide_dot - d["slide_dot"]
    return (0.5 * m * (state.l_dot - d["l_dot"])**2 + 0.5 * gains.k_l * (state.l - d["l"])**2
            + 0.5 * w1 @ J1 @ w1 + Psi_right(state.R1, d["R1"], D)
            + 0.5 * wP @ JP @ wP + p.m_r * rt_dot @ rt_dot + 0.5 * gains.k_r * rt @ rt
            + Psi_right(state.RP, d["RP"], D)
            + 0.5 * w2 @ p.J_0 @ w2 + Psi_right(state.R2, d["R2"], D))


def thesis_V_rate(state, d, gains=ThesisGains()):
    """Thesis eq. (5.70)."""
    k1, kP, k2 = gains.k_Omega
    w1, wP, w2 = state.Omega1 - d["Omega1"], state.OmegaP - d["OmegaP"], state.Omega2 - d["Omega2"]
    rt_dot = state.slide_dot - d["slide_dot"]
    return -(gains.k_ldot * (state.l_dot - d["l_dot"])**2 + k1 * w1 @ w1 + kP * wP @ wP
             + k2 * w2 @ w2 + gains.k_rdot * rt_dot @ rt_dot)


@dataclass
class BookGains:
    """Scalar gains for eq6-law on each factor, matched to the thesis gains near zero error."""
    k_l: float = 256.0
    k_ldot: float = 32.0
    k_R: tuple = (48.0, 48.0, 48.0)          # stance leg, torso, swing leg
    k_Omega: tuple = (8.0, 1.8, 2.0)
    k_r: float = 19.2
    k_rdot: float = 6.0
    half_rate: bool = True                   # the term -(1/2) J_dot e_Omega
    feedforward: bool = True


def book_e_R(R, Rd):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def book_law(state, d, gains=BookGains(), p=PRM):
    """eq6-law on each of the three factors, with the gravity torque canceled and the
    two terms that a configuration-dependent inertia adds: + J_dot Omega - (1/2) J_dot e_Omega."""
    m = p.m
    ff = 1.0 if gains.feedforward else 0.0
    l, l_dot, R1 = state.l, state.l_dot, state.R1
    f = (ff * m * d["l_ddot"] - m * l * state.Omega1 @ PI @ state.Omega1 + m * G * E3 @ (R1 @ E3)
         - gains.k_l * (l - d["l"]) - gains.k_ldot * (l_dot - ff * d["l_dot"]))
    J1 = p.J_0 + m * l**2 * PI
    J1_dot = 2 * m * l * l_dot * PI
    JP = biped.torso_inertia(state.slide, p)
    JP_dot = torso_inertia_rate(state.slide, state.slide_dot, p)
    n = []
    for i, (R, Om, J, J_dot, key) in enumerate(((R1, state.Omega1, J1, J1_dot, "1"),
                                                 (state.RP, state.OmegaP, JP, JP_dot, "P"),
                                                 (state.R2, state.Omega2, p.J_0, 0 * I3, "2"))):
        Rd, Omd, Omd_dot = d["R" + key], ff * d["Omega" + key], ff * d["Omega" + key + "_dot"]
        e_Omega = Om - R.T @ Rd @ Omd
        tau = (-gains.k_R[i] * book_e_R(R, Rd) - gains.k_Omega[i] * e_Omega
               + np.cross(Om, J @ Om) - J @ (np.cross(Om, R.T @ Rd @ Omd) - R.T @ Rd @ Omd_dot)
               + J_dot @ Om - (0.5 * J_dot @ e_Omega if gains.half_rate else 0.0))
        n.append(tau)
    n[0] = n[0] - m * G * l * np.cross(R1.T @ E3, E3)
    OmP = state.OmegaP
    f_s = (ff * 2 * p.m_r * d["slide_ddot"]
           - 2 * p.m_r * np.array([s * OmP @ P @ OmP for s, P in zip(state.slide, PI_I)])
           - gains.k_r * (state.slide - d["slide"]) - gains.k_rdot * (state.slide_dot - ff * d["slide_dot"]))
    tau_1, tau_P, tau_2 = _stack(n[0], n[1], n[2], state)
    return f, tau_1, tau_P, tau_2, f_s


def book_V0(state, d, gains=BookGains(), p=PRM):
    """Sum over the factors of eq6-V0, plus the two PD energies of the lengths."""
    m = p.m
    J1 = p.J_0 + m * state.l**2 * PI
    JP = biped.torso_inertia(state.slide, p)
    V = 0.5 * m * (state.l_dot - d["l_dot"])**2 + 0.5 * gains.k_l * (state.l - d["l"])**2
    rt, rt_dot = state.slide - d["slide"], state.slide_dot - d["slide_dot"]
    V += p.m_r * rt_dot @ rt_dot + 0.5 * gains.k_r * rt @ rt
    rate = -gains.k_ldot * (state.l_dot - d["l_dot"])**2 - gains.k_rdot * rt_dot @ rt_dot
    for i, (R, Om, J, key) in enumerate(((state.R1, state.Omega1, J1, "1"), (state.RP, state.OmegaP, JP, "P"),
                                         (state.R2, state.Omega2, p.J_0, "2"))):
        Rd = d["R" + key]
        e_Omega = Om - R.T @ Rd @ d["Omega" + key]
        V += 0.5 * e_Omega @ J @ e_Omega + gains.k_R[i] * 0.5 * np.trace(I3 - Rd.T @ R)
        rate += -gains.k_Omega[i] * e_Omega @ e_Omega
    return V, rate


# =====================================================================
# 4. The hybrid simulation
# =====================================================================
def _dexpinv(x, y):
    return y + 0.5 * np.cross(x, y) + np.cross(x, np.cross(x, y)) / 12.0


def fast_step(state, u, h, p=PRM):
    """The step of the library (robots/core.py, Robot.step) written out for the biped:
    classical Runge-Kutta stages on one increment per factor, exponential update.
    Checked against the step of the library in verify/ch20.py. It is faster, nothing else."""
    S = biped.State
    R1, RP, R2 = state.R1, state.RP, state.R2
    z = np.zeros(3)
    vel0 = (state.l_dot, state.Omega1, state.OmegaP, state.Omega2, state.slide_dot)

    def slope(inc, vel):
        at = S(l=state.l + inc[0], l_dot=vel[0], R1=R1 @ expm_so3(inc[1]), Omega1=vel[1],
               RP=RP @ expm_so3(inc[2]), OmegaP=vel[2], R2=R2 @ expm_so3(inc[3]), Omega2=vel[3],
               slide=state.slide + inc[4], slide_dot=vel[4])
        rates = (vel[0], _dexpinv(inc[1], vel[1]), _dexpinv(inc[2], vel[2]), _dexpinv(inc[3], vel[3]), vel[4])
        return rates, biped.dynamics(at, u, p)

    def add(a, b, s):
        return tuple(x + s * y for x, y in zip(a, b))

    zero = (0.0, z, z, z, z)
    k1 = slope(zero, vel0)
    k2 = slope(add(zero, k1[0], h / 2), add(vel0, k1[1], h / 2))
    k3 = slope(add(zero, k2[0], h / 2), add(vel0, k2[1], h / 2))
    k4 = slope(add(zero, k3[0], h), add(vel0, k3[1], h))
    inc = tuple(h / 6 * (a + 2 * b + 2 * c + d) for a, b, c, d in zip(k1[0], k2[0], k3[0], k4[0]))
    vel = tuple(v + h / 6 * (a + 2 * b + 2 * c + d) for v, a, b, c, d in zip(vel0, k1[1], k2[1], k3[1], k4[1]))
    return S(l=state.l + inc[0], l_dot=vel[0], R1=R1 @ expm_so3(inc[1]), Omega1=vel[1],
             RP=RP @ expm_so3(inc[2]), OmegaP=vel[2], R2=R2 @ expm_so3(inc[3]), Omega2=vel[3],
             slide=state.slide + inc[4], slide_dot=vel[4])


def hip_acceleration(state, acc):
    """Second derivative of x_P = l R1 e3 from the stance accelerations."""
    l, ld, R, Om = state.l, state.l_dot, state.R1, state.Omega1
    l_dd, Om_d = acc[0], acc[1]
    return R @ (l_dd * E3 + 2 * ld * np.cross(Om, E3)
                + l * (np.cross(Om_d, E3) + np.cross(Om, np.cross(Om, E3))))


def attitude_angle(R, Rd):
    return float(np.linalg.norm(log_so3(Rd.T @ R)))


def walk(headings, gait=Gait(), law=thesis_law, gains=None, h=1e-3, state0=None, rolls=None,
         max_step_time=3.0, p=PRM, record=5):
    """Walk len(headings) steps. Step i tracks the primitive with heading headings[i]; it
    ends when the swing foot comes down to the ground, where the impact map is applied.

    The controller is sampled every h and held (the library step). Returns a dict of
    arrays sampled every h and a list of per-step summaries."""
    gains = gains if gains is not None else (ThesisGains() if law is thesis_law else BookGains())
    state = state0 if state0 is not None else primitive_state(0.0, headings[0], gait)
    foot = np.zeros(3)
    log = {k: [] for k in ("t", "step", "hip", "foot", "swing", "R1", "RP", "R2", "l", "slide", "e_l", "e_r",
                           "a1", "aP", "a2", "V", "F", "tau_1", "tau_P", "tau_2", "f", "E", "H_foot", "ts")}
    steps, t0 = [], 0.0
    peak = dict(tau_1=np.zeros(3), tau_P=np.zeros(3), tau_2=np.zeros(3))
    for i, psi in enumerate(headings):
        g_i = gait if rolls is None else replace(gait, roll=rolls[i])
        ts, done, count = 0.0, False, 0
        clearance, e_start = np.inf, None
        while not done:
            d = primitive(ts, psi, g_i)
            u = law(state, d, gains, p)
            for key, val in zip(("tau_1", "tau_P", "tau_2"), u[1:4]):
                peak[key] = np.maximum(peak[key], np.abs(val))
            count += 1
            if record and (count - 1) % record == 0:
                acc = biped.dynamics(state, u, p)
                F = M_TOT * (hip_acceleration(state, acc) + G * E3)
                hip = foot + state.l * state.R1 @ E3
                vP = hip_velocity(state.l, state.l_dot, state.R1, state.Omega1)
                for k, val in (("t", t0 + ts), ("step", i), ("hip", hip), ("foot", foot.copy()),
                               ("swing", foot + swing_foot(state, gait.l_0)), ("R1", state.R1),
                               ("RP", state.RP), ("R2", state.R2), ("l", state.l), ("slide", state.slide.copy()),
                               ("e_l", state.l - d["l"]), ("e_r", np.linalg.norm(state.slide - d["slide"])),
                               ("a1", attitude_angle(state.R1, d["R1"])), ("aP", attitude_angle(state.RP, d["RP"])),
                               ("a2", attitude_angle(state.R2, d["R2"])),
                               ("V", thesis_V(state, d, gains, p) if law is thesis_law else book_V0(state, d, gains, p)[0]),
                               ("F", F), ("tau_1", u[1]), ("tau_P", u[2]), ("tau_2", u[3]), ("f", u[0]),
                               ("E", biped.energy(state, p)),
                               ("H_foot", angular_momentum_about(state.l * state.R1 @ E3, state, vP, state.Omega1,
                                                                 state.OmegaP, state.Omega2, p)),
                               ("ts", ts)):
                    log[k].append(val)
            if e_start is None:
                e_start = (attitude_angle(state.R1, d["R1"]), attitude_angle(state.R2, d["R2"]))
            new = fast_step(state, u, h, p)
            z0, z1 = swing_foot(state, gait.l_0)[2], swing_foot(new, gait.l_0)[2]
            if ts > 0.5 * gait.t_s and z1 <= 0.0 < z0:
                lo, hi = 0.0, h                               # bisect for the touchdown time
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if swing_foot(fast_step(state, u, mid, p), gait.l_0)[2] > 0.0:
                        lo = mid
                    else:
                        hi = mid
                state, ts, done = fast_step(state, u, hi, p), ts + hi, True
            else:
                if 0.1 * gait.t_s < ts < 0.9 * gait.t_s:
                    clearance = min(clearance, z1)
                state, ts = new, ts + h
                if ts > max_step_time:
                    raise RuntimeError(f"step {i}: no touchdown within {max_step_time} s")
        d = primitive(ts, psi, g_i)
        pre = state
        E_minus = biped.energy(pre, p)
        state, impulse, stride = impact(pre, gait.l_0, p)
        steps.append(dict(step_time=ts, step_vector=stride, step_length=float(np.linalg.norm(stride[:2])),
                          foot_height_error=float(stride[2]), impulse=impulse, clearance=clearance,
                          e_start=e_start, e_end=(attitude_angle(pre.R1, d["R1"]), attitude_angle(pre.R2, d["R2"])),
                          energy_lost=E_minus - biped.energy(state, p), pre=pre, post=state, foot=foot.copy()))
        foot = foot + np.array([stride[0], stride[1], 0.0])
        t0 += ts
    out = {k: np.array(v) for k, v in log.items()}
    out["final_state"], out["final_foot"], out["peak"] = state, foot, peak
    return out, steps


# =====================================================================
# 5. The explicit update of the thesis (eqs. 5.36 to 5.40), for comparison
# =====================================================================
def explicit_update(state, u, h, p=PRM):
    """One step of the thesis's discrete equations, in the book's convention and with the
    sign of the centrifugal term on the slides corrected. Inputs are held."""
    f, tau_1, tau_P, tau_2, f_s = u
    m = p.m
    J1 = p.J_0 + m * state.l**2 * PI
    JP = biped.torso_inertia(state.slide, p)
    U1, UP, U2 = (expm_so3(h * w) for w in (state.Omega1, state.OmegaP, state.Omega2))
    R1, RP, R2 = state.R1 @ U1, state.RP @ UP, state.R2 @ U2
    l, slide = state.l + h * state.l_dot, state.slide + h * state.slide_dot
    J1n, JPn = p.J_0 + m * l**2 * PI, biped.torso_inertia(slide, p)
    Om1 = np.linalg.solve(J1n, U1.T @ J1 @ state.Omega1
                          + h * (m * G * l * np.cross(R1.T @ E3, E3) + tau_1 - R1.T @ RP @ tau_P))
    OmP = np.linalg.solve(JPn, UP.T @ JP @ state.OmegaP + h * (tau_P - RP.T @ R2 @ tau_2))
    Om2 = np.linalg.solve(p.J_0, U2.T @ p.J_0 @ state.Omega2 + h * tau_2)
    l_dot = state.l_dot + h * (l * Om1 @ PI @ Om1 - G * E3 @ (R1 @ E3) + f / m)
    slide_dot = state.slide_dot + h * (np.array([s * OmP @ P @ OmP for s, P in zip(slide, PI_I)])
                                       + np.asarray(f_s) / (2 * p.m_r))
    return biped.State(l=l, l_dot=l_dot, R1=R1, Omega1=Om1, RP=RP, OmegaP=OmP, R2=R2, Omega2=Om2,
                       slide=slide, slide_dot=slide_dot)


# =====================================================================
# 6. Linearization about the primitive and a time-varying LQR
# =====================================================================
# Linear state s = (dl, dr, eta_1, eta_P, eta_2, dl_dot, dr_dot, dOmega_1, dOmega_P, dOmega_2),
# 26 numbers: the variations of sec7-variation, one block per factor.
def perturbed(state, s):
    return biped.State(l=state.l + s[0], slide=state.slide + s[1:4], R1=state.R1 @ expm_so3(s[4:7]),
                       RP=state.RP @ expm_so3(s[7:10]), R2=state.R2 @ expm_so3(s[10:13]),
                       l_dot=state.l_dot + s[13], slide_dot=state.slide_dot + s[14:17],
                       Omega1=state.Omega1 + s[17:20], OmegaP=state.OmegaP + s[20:23],
                       Omega2=state.Omega2 + s[23:26])


def pack_u(u):
    return np.concatenate([[u[0]], u[1], u[2], u[3], u[4]])


def unpack_u(v):
    return float(v[0]), v[1:4], v[4:7], v[7:10], v[10:13]


def accel_vector(state, u, p=PRM):
    a = biped.dynamics(state, u, p)
    return np.concatenate([[a[0]], a[4], a[1], a[2], a[3]])       # l, slide, Omega1, OmegaP, Omega2


def linearize_stance(state, u, p=PRM, d=1e-6):
    """A (26 x 26) and B (26 x 13) of s_dot = A s + B du about a state and input.

    The kinematic rows are exact: d(dl)/dt = dl_dot, d(eta)/dt = dOmega - Omega x eta
    (eq3-delta-Omega). The dynamic rows are central differences of the library dynamics
    along the variations."""
    A, B = np.zeros((26, 26)), np.zeros((26, 13))
    A[0:4, 13:17] = np.eye(4)
    for k, Om in enumerate((state.Omega1, state.OmegaP, state.Omega2)):
        A[4 + 3 * k:7 + 3 * k, 4 + 3 * k:7 + 3 * k] = -hat(Om)
        A[4 + 3 * k:7 + 3 * k, 17 + 3 * k:20 + 3 * k] = I3
    for j in range(26):
        e = np.zeros(26)
        e[j] = d
        A[13:, j] = (accel_vector(perturbed(state, e), u, p) - accel_vector(perturbed(state, -e), u, p)) / (2 * d)
    uv = pack_u(u)
    for j in range(13):
        e = np.zeros(13)
        e[j] = 1.0
        B[13:, j] = (accel_vector(state, unpack_u(uv + e), p) - accel_vector(state, unpack_u(uv - e), p)) / 2
    return A, B


def error_state(state, d):
    """The error state of sec13-error on the product: logarithm errors and velocity deviations."""
    return np.concatenate([[state.l - d["l"]], state.slide - d["slide"],
                           log_so3(d["R1"].T @ state.R1), log_so3(d["RP"].T @ state.RP),
                           log_so3(d["R2"].T @ state.R2),
                           [state.l_dot - d["l_dot"]], state.slide_dot - d["slide_dot"],
                           state.Omega1 - d["Omega1"], state.OmegaP - d["OmegaP"], state.Omega2 - d["Omega2"]])


def nominal_input(t, psi, gait, p=PRM):
    """The input that keeps the biped on the primitive: the law evaluated with zero error."""
    return thesis_law(primitive_state(t, psi, gait), primitive(t, psi, gait), ThesisGains(), p)


def lqr_gains(psi, gait, Q, R_u, P_f, n=200, p=PRM):
    """Gains K(t_k) of the time-varying LQR along one primitive, by backward integration of
    the Riccati equation eq7-riccati with classical Runge-Kutta steps."""
    ts = np.linspace(0.0, gait.t_s, n + 1)
    R_inv = np.linalg.inv(R_u)

    def AB(t):
        return linearize_stance(primitive_state(t, psi, gait), nominal_input(t, psi, gait, p), p)

    mats = {}

    def rhs(t, P):
        key = round(t, 12)
        if key not in mats:
            mats[key] = AB(t)
        A, B = mats[key]
        return -(A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q)

    P, Ks = P_f.copy(), [None] * (n + 1)
    Ks[n] = R_inv @ AB(ts[n])[1].T @ P
    for k in range(n, 0, -1):
        h = ts[k - 1] - ts[k]
        k1 = rhs(ts[k], P)
        k2 = rhs(ts[k] + h / 2, P + h / 2 * k1)
        k3 = rhs(ts[k] + h / 2, P + h / 2 * k2)
        k4 = rhs(ts[k - 1], P + h * k3)
        P = P + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        P = 0.5 * (P + P.T)
        Ks[k - 1] = R_inv @ mats[round(ts[k - 1], 12)][1].T @ P
    return ts, np.array(Ks), P


def stance_phase(state0, psi, gait, controller, h=1e-3, p=PRM):
    """One stance phase of duration t_s from state0 (no impact). Returns errors and inputs."""
    state, out = state0, dict(t=[], a1=[], aP=[], a2=[], e_l=[], tau=[], u2=[])
    n = int(round(gait.t_s / h))
    for k in range(n + 1):
        t = k * h
        d = primitive(t, psi, gait)
        u = controller(t, state, d)
        out["t"].append(t)
        out["a1"].append(attitude_angle(state.R1, d["R1"]))
        out["aP"].append(attitude_angle(state.RP, d["RP"]))
        out["a2"].append(attitude_angle(state.R2, d["R2"]))
        out["e_l"].append(state.l - d["l"])
        out["tau"].append(np.concatenate([u[1], u[2], u[3]]))
        du = pack_u(u) - pack_u(nominal_input(t, psi, gait, p))
        out["u2"].append(du @ du)
        if k < n:
            state = fast_step(state, u, h, p)
    return {k: np.array(v) for k, v in out.items()}, state


# =====================================================================
# 7. The 3D pendulum of Chapter 6, taken from verify/ch06.py
# =====================================================================
def chapter6_pendulum():
    """tau_geo, tau_euler and fl_run of verify/ch06.py, compiled from its source with the
    constants that the chapter uses. Importing that script would run all of its checks."""
    from scipy.integrate import solve_ivp
    path = HERE / "ch06.py"
    J_3 = np.diag([0.1625, 0.1625, 0.01])
    ns = {"np": np, "hat": hat, "I3": I3, "E3": E3, "G": G, "solve_ivp": solve_ivp,
          "e_R": lambda R, Rd: 0.5 * vee(Rd.T @ R - R.T @ Rd),
          "m_3": 1.0, "rho_3": np.array([0.0, 0.0, -0.5]), "J_3": J_3, "J3inv": np.linalg.inv(J_3),
          "k_p": 100.0, "k_d": 20.0}
    names = ("grav_3", "euler_zyx", "E_mat", "E_dot", "tau_geo", "tau_euler", "fl_run")
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]).strip("()") in ("J_3", "k_p, k_d", "m_3, rho_3"):
            ns.setdefault("_source", {})[ast.unparse(node.targets[0]).strip("()")] = ast.unparse(node.value)
    return ns


def pendulum_deadline(ns, angle_deg, n, deadline, rng):
    """n random starts at rest, each a rotation by angle_deg about a random axis. Both laws
    are run for the deadline. Returns the angle left, the peak torque and the effort."""
    from scipy.integrate import solve_ivp
    J_3, J3inv, grav = ns["J_3"], ns["J3inv"], ns["grav_3"]
    rows = []
    for _ in range(n):
        axis = rng.normal(size=3)
        R0 = expm_so3(np.radians(angle_deg) * axis / np.linalg.norm(axis))
        row = []
        for law in (ns["tau_geo"], ns["tau_euler"]):
            def rhs(t, s, law=law):
                R, Om = s[:9].reshape(3, 3), s[9:12]
                tau = law(R, Om)
                return np.concatenate([(R @ hat(Om)).ravel(), J3inv @ (tau + grav(R) - np.cross(Om, J_3 @ Om)),
                                       [tau @ tau]])
            ts = np.linspace(0.0, deadline, 141)
            sol = solve_ivp(rhs, [0, deadline], np.concatenate([R0.ravel(), np.zeros(3), [0.0]]), t_eval=ts,
                            rtol=1e-8, atol=1e-10)
            Rs = sol.y[:9].T.reshape(-1, 3, 3)
            peak = max(np.linalg.norm(law(R, Om)) for R, Om in zip(Rs, sol.y[9:12].T))
            left = np.degrees(np.arccos(np.clip((np.trace(Rs[-1]) - 1) / 2, -1, 1)))
            row += [left, peak, sol.y[12, -1]]
        rows.append(row)
    return np.array(rows)


# =====================================================================
# 8. The floating-base model derived by geomech
# =====================================================================
def floating_lagrangian():
    """The Lagrangian of the floating-base model, built mass by mass, with no inputs."""
    from geomech import SO3, Dot, Matrix, Scalar, SystemVariables, Vector, getScalars
    half = Scalar("0.5", value=0.5, attr=["Constant"])
    m_leg, m_r, g = getScalars("m_leg m_r g", attr=["Constant"])
    J_0 = Matrix("J_0", attr=["Constant", "SymmetricMatrix"])
    J_P = Matrix("J_P", attr=["Constant", "SymmetricMatrix"])
    e = [Vector(f"e{i}", attr=["Constant"]) for i in (1, 2, 3)]
    x, R1, RP, R2 = Vector("x"), SO3("R1"), SO3("RP"), SO3("R2")
    s = [Scalar(f"s{i}") for i in (1, 2, 3)]
    Om1, OmP, Om2 = (R.get_tangent_vector() for R in (R1, RP, R2))
    v = x.t_diff()
    L = (m_leg * Dot(v, v) - (m_leg + m_leg) * g * Dot(e[2], x)
         + half * Dot(Om1, J_0 * Om1) + half * Dot(OmP, J_P * OmP) + half * Dot(Om2, J_0 * Om2))
    for si, ei in zip(s, e):
        for sign in (1.0, -1.0):
            side = Scalar(str(sign), value=sign, attr=["Constant"])
            pt = x + (RP * ei) * si * side
            L = L + half * m_r * Dot(pt.t_diff(), pt.t_diff()) - m_r * g * Dot(e[2], pt)
    return L, SystemVariables(scalars=s, vectors=[x], matrices=[R1, RP, R2])


def run_table_row(law, gains, gait, n_steps=8, psi=0.0, done=None):
    """One row of tbl20-compare: the last of n_steps steps of a straight walk."""
    o, steps = done if done is not None else walk([psi] * n_steps, gait=gait, law=law, gains=gains)
    steps = steps[:n_steps]
    last = steps[-1]
    sel = o["step"] == n_steps - 1
    F = o["F"][sel]
    return dict(step_length=last["step_length"], step_time=last["step_time"],
                leg_error_deg=float(np.degrees(max(o["a1"][sel].max(), o["a2"][sel].max()))),
                leg_error_end_deg=float(np.degrees(max(last["e_end"]))),
                length_error_mm=float(1e3 * np.abs(o["e_l"][sel]).max()),
                peak_ankle=float(np.abs(o["tau_1"][sel]).max()),
                friction=float((np.linalg.norm(F[:, :2], axis=1) / F[:, 2]).max()),
                clearance_mm=float(1e3 * last["clearance"]),
                drift=float(abs(steps[-1]["step_length"] - steps[-2]["step_length"])))


def main():
    t_start = time.time()
    ck = Checker("ch20")
    rng = np.random.default_rng(20)
    p = PRM
    gait = Gait()

    # ------------------------------------------------------------------
    # Stage 1: the pendulum
    # ------------------------------------------------------------------
    ns = chapter6_pendulum()
    ck.true("the constants used with the functions of verify/ch06.py are the ones in its source",
            ns["_source"].get("J_3") == "np.diag([0.1625, 0.1625, 0.01])" and ns["_source"].get("k_p, k_d") == "(100.0, 20.0)"
            and ns["_source"].get("m_3, rho_3") == "(1.0, np.array([0.0, 0.0, -0.5]))",
            str(ns["_source"]))
    w = 0.0
    for _ in range(20):
        R, Om = random_rotation(rng), rng.normal(size=3)
        Om_dot = ns["J3inv"] @ (ns["tau_geo"](R, Om) + ns["grav_3"](R) - np.cross(Om, ns["J_3"] @ Om))
        w = max(w, np.abs(Om_dot + 100 * book_e_R(R, I3) + 20 * Om).max())
    ck.true("pendulum: the geometric law of verify/ch06.py imposes Omega_dot = -k_p e_R - k_d Omega", w < 1e-10, f"err {w:.1e}")
    # a leg of fixed length is the 3D pendulum of eq4-pendulum, standing on its pivot
    leg, pend = robots.get("telescoping_leg"), robots.get("pendulum_3d")
    w = 0.0
    for _ in range(10):
        st = leg.random_state(rng)
        st.l_dot = 0.0
        tau = rng.normal(size=3)
        lp = leg.params
        Jl = lp.J_0 + lp.m * st.l**2 * PI
        pp = pend.Params(m=lp.m, J=Jl, rho=st.l * E3, g=lp.g)
        a = pend.dynamics(pend.State(R=st.R, Omega=st.Omega), tau, pp)
        b = np.linalg.solve(Jl, lp.m * lp.g * st.l * np.cross(st.R.T @ E3, E3) + tau - np.cross(st.Omega, Jl @ st.Omega))
        w = max(w, np.abs(np.asarray(a).ravel() - b).max())
    ck.true("a leg of fixed length is pendulum_3d with rho = l e3 and J = J_0 + m l^2 Pi", w < 1e-12, f"err {w:.1e}")
    deadline = 0.35
    table = {}
    for ang in (10, 30, 60, 90, 120, 150):
        r = pendulum_deadline(ns, ang, 30, deadline, rng)
        ok = np.all(np.isfinite(r), axis=1)
        table[str(ang)] = dict(geo_left_deg=float(np.median(r[ok, 0])), euler_left_deg=float(np.median(r[ok, 3])),
                               geo_left_worst=float(r[ok, 0].max()), euler_left_worst=float(r[ok, 3].max()),
                               geo_peak=float(np.median(r[ok, 1])), euler_peak=float(np.median(r[ok, 4])),
                               euler_peak_worst=float(r[ok, 4].max()), geo_peak_worst=float(r[ok, 1].max()),
                               effort_ratio=float(np.median(r[ok, 5] / r[ok, 2])), n=int(ok.sum()))
    ck.record("pendulum_deadline", {"deadline_s": deadline, "starts_per_angle": 30, **table})
    ck.true("pendulum, 10 degree starts: the two laws leave the same error at the deadline and use the same peak torque within 10%",
            abs(table["10"]["geo_left_deg"] - table["10"]["euler_left_deg"]) < 0.1
            and abs(table["10"]["euler_peak"] / table["10"]["geo_peak"] - 1) < 0.10,
            f"left {table['10']['geo_left_deg']:.2f} / {table['10']['euler_left_deg']:.2f} deg, peak "
            f"{table['10']['geo_peak']:.2f} / {table['10']['euler_peak']:.2f} N m")
    ck.true("pendulum, 30 degree starts: still within 15% in peak torque", abs(table["30"]["euler_peak"] / table["30"]["geo_peak"] - 1) < 0.15,
            f"{table['30']['geo_peak']:.2f} / {table['30']['euler_peak']:.2f} N m")
    ck.true("pendulum, 120 degree starts: the Euler law needs more than twice the peak torque, and the geometric law leaves more error",
            table["120"]["euler_peak"] > 2 * table["120"]["geo_peak"] and table["120"]["geo_left_deg"] > table["120"]["euler_left_deg"],
            f"peak {table['120']['geo_peak']:.1f} / {table['120']['euler_peak']:.1f} N m, left "
            f"{table['120']['geo_left_deg']:.1f} / {table['120']['euler_left_deg']:.1f} deg")

    # ------------------------------------------------------------------
    # Stage 2: the stance model
    # ------------------------------------------------------------------
    from geomech import check_eom, compute_eom, to_standard_form, linearize
    from geomech.codegen import StateLayout, collect_params, load_module, module_source
    from geomech.simulation import accelerations
    L, dW, variables, inputs = biped.lagrangian()
    eom = compute_eom(L, dW, variables)
    err = float(check_eom(L, dW, variables, eom, inputs=inputs))
    ck.true("stance model: geomech check_eom < 1e-5", err < 1e-5, f"err {err:.1e}")
    ck.record("stance_check_eom", err)
    sf = to_standard_form(eom, variables, inputs)
    layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
    gm = load_module(module_source(sf, layout, lagrangian=L), "ch20_stance")
    gm_p = {"m_leg": p.m_leg, "m_r": p.m_r, "J_0": p.J_0, "J_P": p.J_P, "g": p.g, "e1": E1, "e2": E2, "e3": E3}

    def gm_pack(st):
        return layout.pack(l=st.l, s1=st.slide[0], s2=st.slide[1], s3=st.slide[2], R1=st.R1, RP=st.RP, R2=st.R2,
                           l_dot=st.l_dot, s1_dot=st.slide_dot[0], s2_dot=st.slide_dot[1], s3_dot=st.slide_dot[2],
                           Omega_R1=st.Omega1, Omega_RP=st.OmegaP, Omega_R2=st.Omega2)

    w = 0.0
    for _ in range(20):
        st = biped.random_state(rng)
        u = (3 * rng.normal(), *(3 * rng.normal(size=3) for _ in range(4)))
        w = max(w, np.abs(accelerations(gm, gm_pack(st), pack_u(u), gm_p) - accel_vector(st, u)).max())
    ck.true("stance model: library dynamics equal the geomech model at 20 random states", w < 1e-9, f"err {w:.1e}")
    ck.record("stance_vs_geomech", float(w))
    ck.close("total mass 2 m_leg + 6 m_r", p.m, 3.684)
    ck.true("the biped has 13 velocity degrees of freedom and stores 31 numbers", biped.dim == 13 and biped.stored == 31)
    # with no hip torque the stance leg is the telescoping leg eq4-leg
    w = 0.0
    leg_p = leg.Params(m=p.m, J_0=p.J_0)
    for _ in range(10):
        st = biped.random_state(rng)
        u = (rng.normal(), rng.normal(size=3), np.zeros(3), rng.normal(size=3), rng.normal(size=3))
        a = biped.dynamics(st, u)
        b = leg.dynamics(leg.State(l=st.l, l_dot=st.l_dot, R=st.R1, Omega=st.Omega1), (u[0], u[1]), leg_p)
        w = max(w, abs(a[0] - b[0]), np.abs(a[1] - b[1]).max())
    ck.true("with tau_P = 0 the stance leg obeys eq4-leg", w < 1e-12, f"err {w:.1e}")
    # the sign of thesis eq. (5.14): the Lagrangian pushes the masses outward
    st = biped.random_state(rng)
    st.slide, st.slide_dot = np.array([0.1, 0.2, 0.3]), np.zeros(3)
    ck.true("a spinning torso pushes its reaction masses outward (the sign of thesis eq. 5.14)",
            np.all(biped.dynamics(st, biped.zero_input())[4] > 0))
    # fast_step is the library step
    w = 0.0
    for _ in range(5):
        st = biped.random_state(rng)
        u = (rng.normal(), *(rng.normal(size=3) for _ in range(4)))
        a, b = biped.step(st, u, 5e-3), fast_step(st, u, 5e-3)
        w = max(w, max(np.abs(np.asarray(getattr(a, k)) - np.asarray(getattr(b, k))).max() for k in a.__dataclass_fields__))
    ck.true("fast_step equals the step of the library", w < 1e-13, f"err {w:.1e}")
    # the inertias
    J1 = p.J_0 + p.m * gait.l_0**2 * PI
    Jin = biped.torso_inertia(np.full(3, gait.r_0 - gait.r_bar))
    Jout = biped.torso_inertia(np.full(3, gait.r_0 + gait.r_bar))
    ck.record("inertia", {"stance_leg_at_l0": np.diag(J1).tolist(), "leg_alone": np.diag(p.J_0).tolist(),
                          "torso_yaw_in": float(Jin[2, 2]), "torso_yaw_out": float(Jout[2, 2]),
                          "torso_yaw_change_percent": float(100 * (Jout[2, 2] / Jin[2, 2] - 1)),
                          "torso_yaw_r005": float(biped.torso_inertia(np.full(3, 0.05))[2, 2]),
                          "torso_yaw_r025": float(biped.torso_inertia(np.full(3, 0.25))[2, 2])})
    ck.close("torso yaw inertia: J_P33 + 4 m_r r^2 for equal slides", Jout[2, 2], p.J_P[2, 2] + 4 * p.m_r * 0.15**2, tol=1e-12)
    # the skater, commanded: masses pulled in from 0.25 m to 0.05 m while the torso spins freely
    sk = skater()
    ck.record("skater", {k: float(v) for k, v in sk["summary"].items()})
    ck.true("skater: pulling the masses in from 0.25 m to 0.05 m raises the spin by the ratio of the inertias",
            abs(sk["summary"]["spin_end"] / sk["summary"]["spin_start"] - sk["summary"]["inertia_ratio"]) < 2e-3,
            f"{sk['summary']['spin_start']:.2f} -> {sk['summary']['spin_end']:.3f} rad/s, ratio of inertias {sk['summary']['inertia_ratio']:.3f}")
    ck.true("skater: the torso's angular momentum is unchanged", sk["summary"]["momentum_error"] < 1e-9, f"err {sk['summary']['momentum_error']:.1e}")
    ck.true("skater: the kinetic energy rises by H^2 (1/J_in - 1/J_out) / 2, paid by the forces on the slides",
            abs(sk["summary"]["energy_gain"] - sk["summary"]["energy_gain_formula"]) < 1e-4, f"{sk['summary']['energy_gain']:.4f} J")

    # ------------------------------------------------------------------
    # Stage 3: the impact map
    # ------------------------------------------------------------------
    # (a) the floating-base model by geomech, and its mass matrix
    Lf, Vf = floating_lagrangian()
    eom_f = compute_eom(Lf, None, Vf)
    err_f = float(check_eom(Lf, None, Vf, eom_f))
    ck.true("floating-base model: geomech check_eom < 1e-5", err_f < 1e-5, f"err {err_f:.1e}")
    ck.record("floating_check_eom", err_f)
    sf_f = to_standard_form(eom_f, Vf, [])
    lay_f = StateLayout(Vf, [], collect_params([Lf], Vf, []))
    gm_f = load_module(module_source(sf_f, lay_f, lagrangian=Lf), "ch20_floating")
    w_mass, w_T = 0.0, 0.0
    for _ in range(10):
        st = biped.random_state(rng)
        vP = rng.normal(size=3)
        xf = lay_f.pack(s1=st.slide[0], s2=st.slide[1], s3=st.slide[2], x=rng.normal(size=3), R1=st.R1, RP=st.RP, R2=st.R2,
                        s1_dot=st.slide_dot[0], s2_dot=st.slide_dot[1], s3_dot=st.slide_dot[2], x_dot=vP,
                        Omega_R1=st.Omega1, Omega_RP=st.OmegaP, Omega_R2=st.Omega2)
        Mg = np.asarray(gm_f.mass(xf, gm_p))
        D, A = impact_matrices(st.R2, st.slide, gait.l_0)
        order = []                                    # geomech's acceleration order -> ours
        for key in ("Omega_R1_dot", "Omega_RP_dot", "s1_ddot", "s2_ddot", "s3_ddot", "Omega_R2_dot", "x_ddot"):
            sl = next(a["slice"] for a in gm_f.LAYOUT["acceleration"] if a["name"] == key)
            order += list(range(sl[0], sl[1]))
        Mg = Mg[np.ix_(order, order)]
        w_mass = max(w_mass, min(np.abs(Mg - D).max(), np.abs(Mg + D).max()))
        v = np.concatenate([st.Omega1, st.OmegaP, st.slide_dot, st.Omega2, vP])
        w_T = max(w_T, abs(0.5 * v @ D @ v - kinetic_floating(st, vP, st.Omega1, st.OmegaP, st.Omega2, st.slide_dot)))
    ck.true("floating-base model: the matrix D of the impact equals geomech's mass matrix (up to its overall sign)", w_mass < 1e-10, f"err {w_mass:.1e}")
    ck.true("floating-base model: v.D v/2 equals the kinetic energy summed mass by mass", w_T < 1e-12, f"err {w_T:.1e}")

    # (b) the closed form against the linear system, momentum, energy, involution
    worst = dict(kkt=0.0, mom=0.0, others=0.0, carnot=0.0, stick=0.0, idem=0.0, kin=0.0, printed=0.0, loss=np.inf)
    for _ in range(50):
        st = biped.random_state(rng)
        l_0 = 0.6 + rng.random()
        new, imp, _ = impact(st, l_0)
        v, v_plus, imp_kkt, D, A = impact_kkt(st, l_0)
        vP_plus = v_plus[12:15]
        worst["kkt"] = max(worst["kkt"], np.abs(v_plus[9:12] - new.Omega1).max(), np.abs(imp - imp_kkt).max(),
                           np.abs(v_plus[0:9] - v[0:9]).max())
        worst["stick"] = max(worst["stick"], np.abs(A @ v_plus).max())
        # angular momentum about the landing foot, mass by mass (the foot is at -l_0 R2 e3 from the hip)
        c = -l_0 * st.R2 @ E3
        H_minus = angular_momentum_about(c, st, v[12:15], st.Omega1, st.OmegaP, st.Omega2)
        H_plus = angular_momentum_about(c, st, vP_plus, v_plus[0:3], v_plus[3:6], v_plus[9:12])
        worst["mom"] = max(worst["mom"], np.abs(H_plus - H_minus).max())
        T_minus, T_plus = 0.5 * v @ D @ v, 0.5 * v_plus @ D @ v_plus
        dv = v_plus - v
        worst["carnot"] = max(worst["carnot"], abs((T_minus - T_plus) - 0.5 * dv @ D @ dv))
        worst["loss"] = min(worst["loss"], T_minus - T_plus)
        # the new stance state reproduces the hip velocity after the impact
        worst["kin"] = max(worst["kin"], np.abs(hip_velocity(new.l, new.l_dot, new.R1, new.Omega1) - vP_plus).max())
        # landing again on the same foot changes nothing
        again, imp2, _ = impact(replace(new, R1=new.R2, Omega1=new.Omega2, R2=new.R1, Omega2=new.Omega1,
                                        l=st.l, l_dot=0.0), l_0)
        st_b = replace(st, Omega2=new.Omega1)
        v_b = np.concatenate([v_plus[0:9], new.Omega1, vP_plus])
        K = np.block([[D, -A.T], [A, np.zeros((3, 3))]])
        sol = np.linalg.solve(K, np.concatenate([D @ v_b, np.zeros(3)]))
        worst["idem"] = max(worst["idem"], np.abs(sol[:15] - v_b).max(), np.abs(sol[15:]).max())
        # eq. (5.20) as printed, with -D v on the right: on a velocity that already satisfies A v = 0 it returns -v
        sol_p = np.linalg.solve(K, np.concatenate([-D @ v_b, np.zeros(3)]))
        worst["printed"] = max(worst["printed"], np.abs(sol_p[:15] + v_b).max())
    ck.true("impact: the closed form equals the solution of the 18 x 18 linear system", worst["kkt"] < 1e-10, f"err {worst['kkt']:.1e}")
    ck.true("impact: the landing foot is at rest afterwards", worst["stick"] < 1e-12, f"err {worst['stick']:.1e}")
    ck.true("impact: angular momentum about the landing foot, summed mass by mass, is conserved", worst["mom"] < 1e-11, f"err {worst['mom']:.1e}")
    ck.true("impact: the energy lost is the kinetic energy of the velocity jump, and is never negative",
            worst["carnot"] < 1e-11 and worst["loss"] >= 0, f"err {worst['carnot']:.1e}, smallest loss {worst['loss']:.2e} J")
    ck.true("impact: the new stance state has the hip velocity of the floating-base model (l_dot = 0)", worst["kin"] < 1e-12, f"err {worst['kin']:.1e}")
    ck.true("impact: a second landing on the same foot changes nothing", worst["idem"] < 1e-10, f"err {worst['idem']:.1e}")
    ck.true("thesis eq. (5.20) with -D v on the right returns -v for a velocity with A v = 0", worst["printed"] < 1e-9, f"err {worst['printed']:.1e}")
    vv = rng.normal(size=15)
    ck.close("relabeling the legs twice is the identity", relabel(relabel(vv)), vv, tol=0)
    # thesis p. 46 writes the swing foot at x_P + l_0 R2 e3; eq. (5.43) and the switching surface write x_P - l_0 R2 e3.
    # With the first, the hip velocity after the impact is the opposite of what the relabeled stance leg implies.
    st = biped.random_state(rng)
    D, A = impact_matrices(st.R2, st.slide, 0.9)
    A_alt = A.copy()
    A_alt[:, 9:12] *= -1
    v = np.concatenate([st.Omega1, st.OmegaP, st.slide_dot, st.Omega2, hip_velocity(st.l, st.l_dot, st.R1, st.Omega1)])
    sol = np.linalg.solve(np.block([[D, -A_alt.T], [A_alt, np.zeros((3, 3))]]), np.concatenate([D @ v, np.zeros(3)]))
    implied = hip_velocity(0.9, 0.0, st.R2, sol[9:12])
    ck.true("with the swing foot at x_P + l_0 R2 e3 the relabeled stance leg implies minus the hip velocity",
            np.abs(implied + sol[12:15]).max() < 1e-12 and np.linalg.norm(sol[12:15]) > 1e-3, f"|v_P| {np.linalg.norm(sol[12:15]):.3f} m/s")

    # ------------------------------------------------------------------
    # Stage 4a: the two tracking laws do what they claim
    # ------------------------------------------------------------------
    # the right-invariant error function of the thesis and its slope
    w1 = w2 = 0.0
    Dw = ThesisGains().D
    for _ in range(20):
        R, Rd, Om, Omd = random_rotation(rng), random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
        d = 1e-6
        rate = (Psi_right(R @ expm_so3(d * Om), Rd @ expm_so3(d * Omd), Dw)
                - Psi_right(R @ expm_so3(-d * Om), Rd @ expm_so3(-d * Omd), Dw)) / (2 * d)
        w1 = max(w1, abs(rate - e_right(R, Rd, Dw) @ (Om - Omd)))
        e_thesis = Rd.T @ sum(Dw[i] * np.cross((R @ Rd.T).T @ np.eye(3)[i], np.eye(3)[i]) for i in range(3))
        w2 = max(w2, np.abs(e_thesis - e_right(R, Rd, Dw)).max())
    ck.true("d/dt tr(D(I - R Rd^T)) = e . (Omega - Omega_d): no transport map is needed for this error", w1 < 1e-7, f"err {w1:.1e}")
    ck.true("its slope equals Rd^T S(Q) of thesis eq. (5.56)", w2 < 1e-12, f"err {w2:.1e}")
    stiff = [(Psi_right(expm_so3(1e-3 * e) @ I3, I3, Dw)) / (0.5e-6) for e in np.eye(3)]
    ck.close("stiffness about the world axes: d2 + d3, d1 + d3, d1 + d2", stiff, [Dw[1] + Dw[2], Dw[0] + Dw[2], Dw[0] + Dw[1]], tol=1e-4)
    ck.record("thesis_gains", {"eps": 0.25, "k_l": ThesisGains().k_l, "k_ldot": ThesisGains().k_ldot, "D": Dw.tolist(),
                               "k_Omega": list(ThesisGains().k_Omega), "k_r": ThesisGains().k_r, "k_rdot": ThesisGains().k_rdot,
                               "stiffness_world_axes": [float(x) for x in stiff]})
    # closed-loop error dynamics (5.72) to (5.76), and V_dot = (5.70)
    wa = wb = wc = wd = 0.0
    tg, bg = ThesisGains(), BookGains()
    for _ in range(20):
        st = biped.random_state(rng, speed=0.5)
        st.slide = 0.1 + 0.1 * rng.random(3)
        t, psi = rng.random(), rng.normal()
        d = primitive(t, psi, gait)
        u = thesis_law(st, d, tg)
        acc = biped.dynamics(st, u)
        k1, kP, k2 = tg.k_Omega
        J1s = p.J_0 + p.m * st.l**2 * PI
        w1v = st.Omega1 - d["Omega1"]
        lhs = J1s @ (acc[1] - d["Omega1_dot"])
        rhs = (-np.cross(w1v, J1s @ st.Omega1) - p.m * st.l * st.l_dot * PI @ w1v - k1 * w1v - e_right(st.R1, d["R1"], tg.D))
        wa = max(wa, np.abs(lhs - rhs).max(), abs(p.m * (acc[0] - d["l_ddot"]) + tg.k_ldot * (st.l_dot - d["l_dot"]) + tg.k_l * (st.l - d["l"])),
                 np.abs(2 * p.m_r * (acc[4] - d["slide_ddot"]) + tg.k_rdot * (st.slide_dot - d["slide_dot"]) + tg.k_r * (st.slide - d["slide"])).max())
        hh = 1e-5
        Vp = thesis_V(fast_step(st, u, hh), primitive(t + hh, psi, gait), tg)
        Vm = thesis_V(fast_step(st, u, -hh), primitive(t - hh, psi, gait), tg)
        wb = max(wb, abs((Vp - Vm) / (2 * hh) - thesis_V_rate(st, d, tg)) / (1 + abs(thesis_V_rate(st, d, tg))))
        ub = book_law(st, d, bg)
        V0, rate = book_V0(st, d, bg)
        Vp = book_V0(fast_step(st, ub, hh), primitive(t + hh, psi, gait), bg)[0]
        Vm = book_V0(fast_step(st, ub, -hh), primitive(t - hh, psi, gait), bg)[0]
        wc = max(wc, abs((Vp - Vm) / (2 * hh) - rate) / (1 + abs(rate)))
        # without the half-rate term the rate has an extra (1/2) e_Omega . J_dot e_Omega on stance leg and torso
        ub2 = book_law(st, d, replace(bg, half_rate=False))
        Vp = book_V0(fast_step(st, ub2, hh), primitive(t + hh, psi, gait), bg)[0]
        Vm = book_V0(fast_step(st, ub2, -hh), primitive(t - hh, psi, gait), bg)[0]
        e1 = st.Omega1 - st.R1.T @ d["R1"] @ d["Omega1"]
        eP = st.OmegaP - st.RP.T @ d["RP"] @ d["OmegaP"]
        extra = 0.5 * e1 @ (2 * p.m * st.l * st.l_dot * PI) @ e1 + 0.5 * eP @ torso_inertia_rate(st.slide, st.slide_dot) @ eP
        wd = max(wd, abs((Vp - Vm) / (2 * hh) - rate - extra) / (1 + abs(rate)))
    ck.true("thesis law: the closed loop is thesis eqs. (5.72), (5.73) and (5.75)", wa < 1e-9, f"err {wa:.1e}")
    ck.true("thesis law: V_dot along the closed loop equals thesis eq. (5.70)", wb < 1e-6, f"rel err {wb:.1e}")
    ck.true("book law (eq6-law per factor, with -J_dot e_Omega/2): V0_dot = -sum k_Omega |e_Omega|^2 - PD terms", wc < 1e-6, f"rel err {wc:.1e}")
    ck.true("book law without that term: V0_dot gains e_Omega . J_dot e_Omega / 2 on the stance leg and the torso", wd < 1e-6, f"rel err {wd:.1e}")
    # on the primitive the two laws give the same input
    st, d = primitive_state(0.37, 0.4, gait), primitive(0.37, 0.4, gait)
    ck.close("on the primitive, with zero error, the two laws ask for the same input", pack_u(thesis_law(st, d)), pack_u(book_law(st, d)), tol=1e-10)

    # ------------------------------------------------------------------
    # Stage 4b: walking
    # ------------------------------------------------------------------
    n_straight = 10
    o, steps = walk([0.0] * n_straight)
    last, prev = steps[-1], steps[-2]
    sel = o["step"] == n_straight - 1
    F = o["F"]
    fr = np.linalg.norm(F[:, :2], axis=1) / F[:, 2]
    pre, post = last["pre"], last["post"]
    v_minus = hip_velocity(pre.l, pre.l_dot, pre.R1, pre.Omega1)
    periodic = max(abs(last["step_length"] - prev["step_length"]), abs(last["step_time"] - prev["step_time"]),
                   np.abs(last["post"].Omega1 - prev["post"].Omega1).max(), np.abs(last["post"].Omega2 - prev["post"].Omega2).max(),
                   np.abs(last["post"].R2 - prev["post"].R2).max(), np.abs(last["post"].RP - prev["post"].RP).max())
    V_sel = o["V"][sel]
    walk_rec = dict(steps=n_straight, step_time=last["step_time"], step_length=last["step_length"],
                    nominal_step_length=float(2 * gait.l_0 * np.sin(gait.theta_s)),
                    speed=last["step_length"] / last["step_time"],
                    first_step_length=steps[0]["step_length"], first_step_time=steps[0]["step_time"],
                    step_lengths=[s["step_length"] for s in steps], step_times=[s["step_time"] for s in steps],
                    periodic_residual=float(periodic),
                    change_step_3_to_4=float(abs(steps[3]["step_length"] - steps[2]["step_length"])),
                    leg_error_max_deg=float(np.degrees(max(o["a1"][sel].max(), o["a2"][sel].max()))),
                    leg_error_touchdown_deg=[float(np.degrees(x)) for x in last["e_end"]],
                    torso_error_max_deg=float(np.degrees(o["aP"][sel].max())),
                    length_error_max_mm=float(1e3 * np.abs(o["e_l"][sel]).max()),
                    slide_error_max_mm=float(1e3 * o["e_r"][sel].max()),
                    clearance_mm=float(1e3 * last["clearance"]), foot_height_at_touchdown=float(abs(last["foot_height_error"])),
                    F_z_min=float(F[:, 2].min()), F_z_max=float(F[:, 2].max()), weight=float(p.m * G),
                    friction_ratio_max=float(fr.max()),
                    impulse=last["impulse"].tolist(), impulse_ratio=float(abs(last["impulse"][0]) / last["impulse"][2]),
                    energy_lost_J=float(last["energy_lost"]), v_hip_before=v_minus.tolist(),
                    v_hip_after=hip_velocity(post.l, post.l_dot, post.R1, post.Omega1).tolist(),
                    Omega1_after=post.Omega1.tolist(), Omega_d_start=float(gait.theta_s * np.pi / gait.t_s),
                    peak_ankle_torque=float(np.abs(o["tau_1"][sel]).max()), peak_hip_torque=float(max(np.abs(o["tau_P"][sel]).max(), np.abs(o["tau_2"][sel]).max())),
                    V_after_impact=float(V_sel[0]), V_before_impact=float(V_sel[-1]),
                    manifold=float(biped.on_manifold(o["final_state"])),
                    distance=float(o["final_foot"][0]), leg_force_min=float(o["f"].min()), leg_force_max=float(o["f"].max()))
    ck.record("walk_straight", walk_rec)
    ck.true("walking straight: every step ends in a touchdown and the gait settles to a periodic one",
            periodic < 1e-4, f"step-to-step change {periodic:.1e} after {n_straight} steps")
    ck.true("walking straight: step length within 6% of 2 l_0 sin(theta_s), step time within 2% of t_s",
            abs(last["step_length"] / walk_rec["nominal_step_length"] - 1) < 0.06 and abs(last["step_time"] - 1) < 0.02,
            f"{last['step_length']:.4f} m, {last['step_time']:.4f} s")
    ck.true("walking straight: the ground pushes up throughout and the friction ratio stays below 0.6",
            walk_rec["F_z_min"] > 0 and walk_rec["friction_ratio_max"] < 0.6 and walk_rec["impulse"][2] > 0 and walk_rec["impulse_ratio"] < 0.6,
            f"F_z in [{walk_rec['F_z_min']:.1f}, {walk_rec['F_z_max']:.1f}] N, ratio {walk_rec['friction_ratio_max']:.2f}, impulse ratio {walk_rec['impulse_ratio']:.2f}")
    ck.true("walking straight: the swing foot clears the ground in mid-step", walk_rec["clearance_mm"] > 5, f"{walk_rec['clearance_mm']:.1f} mm at worst between 10% and 90% of the step")
    ck.true("walking straight: the Lyapunov function jumps up at each impact and decays within the step",
            V_sel[0] > 10 * V_sel[-1] and np.diff(V_sel).max() < 1e-4, f"{V_sel[0]:.3f} -> {V_sel[-1]:.4f} J, largest rise between samples {np.diff(V_sel).max():.1e}")
    ck.true("walking straight: the attitudes stay on SO(3) to rounding error over the whole walk", walk_rec["manifold"] < 1e-12, f"{walk_rec['manifold']:.1e}")
    # the same gait at a heading of 90 degrees: the world-axis weights make it a different gait
    o90, steps90 = walk([np.pi / 2] * 8)
    ob, stepsb = walk([0.0] * 8, law=book_law)
    ob90, stepsb90 = walk([np.pi / 2] * 8, law=book_law, record=0)
    ck.record("heading", {"thesis_0": steps[7]["step_length"], "thesis_90": steps90[-1]["step_length"],
                          "book_0": stepsb[-1]["step_length"], "book_90": stepsb90[-1]["step_length"],
                          "thesis_time_0": steps[7]["step_time"], "thesis_time_90": steps90[-1]["step_time"]})
    ck.true("thesis law: the periodic step is longer at a heading of 90 degrees than at 0 (weights on world axes)",
            steps90[-1]["step_length"] - steps[7]["step_length"] > 0.005, f"{steps[7]['step_length']:.4f} m against {steps90[-1]['step_length']:.4f} m")
    ck.true("book law with scalar gains: the gait does not depend on the heading",
            abs(stepsb[-1]["step_length"] - stepsb90[-1]["step_length"]) < 1e-9, f"difference {abs(stepsb[-1]['step_length'] - stepsb90[-1]['step_length']):.1e} m")

    # a turn: four steps straight, six steps of 15 degrees each, four steps straight
    hd, rolls = turn_plan()
    ot, stepst = walk(hd, rolls=rolls)
    Ft = ot["F"]
    frt = np.linalg.norm(Ft[:, :2], axis=1) / Ft[:, 2]
    imp = np.array([s["impulse"] for s in stepst])
    imp_ratio = np.linalg.norm(imp[:, :2], axis=1) / imp[:, 2]
    heading_end = np.arctan2(stepst[-1]["step_vector"][1], stepst[-1]["step_vector"][0])
    turning = slice(4, 10)
    ck.record("turn", dict(steps=len(hd), heading_step_deg=15.0, roll_deg=float(np.degrees(-rolls[5])),
                           final_heading_deg=float(np.degrees(heading_end)), final_foot=ot["final_foot"].tolist(),
                           leg_error_start_deg=float(np.degrees(max(max(s["e_start"]) for s in stepst[turning]))),
                           leg_error_touchdown_deg=float(np.degrees(max(max(s["e_end"]) for s in stepst[turning]))),
                           torso_error_max_deg=float(np.degrees(ot["aP"].max())),
                           friction_ratio_max=float(frt.max()), F_z_min=float(Ft[:, 2].min()),
                           impulse_z_min=float(imp[:, 2].min()), impulse_ratio_max=float(imp_ratio.max()),
                           peak_ankle_yaw=float(ot["peak"]["tau_1"][2]), peak_ankle_pitch=float(ot["peak"]["tau_1"][1]),
                           step_time_range=[float(min(s["step_time"] for s in stepst)), float(max(s["step_time"] for s in stepst))],
                           step_length_range=[float(min(s["step_length"] for s in stepst)), float(max(s["step_length"] for s in stepst))],
                           clearance_min_mm=float(1e3 * min(s["clearance"] for s in stepst)),
                           manifold=float(biped.on_manifold(ot["final_state"]))))
    ck.true("turn: after six heading changes of 15 degrees the biped walks along +e2", abs(np.degrees(heading_end) - 90) < 0.5, f"{np.degrees(heading_end):.2f} deg")
    ck.true("turn: each 15 degree error in the legs is brought below 3 degrees before the next touchdown",
            np.degrees(max(max(s["e_end"]) for s in stepst[turning])) < 3.0)
    ck.true("turn: unilateral contact and a friction ratio below 0.6 hold, for the force and for the impulses",
            Ft[:, 2].min() > 0 and frt.max() < 0.6 and imp[:, 2].min() > 0 and imp_ratio.max() < 0.6,
            f"ratio {frt.max():.2f}, impulse ratio {imp_ratio.max():.2f}")

    # ------------------------------------------------------------------
    # The comparison table
    # ------------------------------------------------------------------
    rows = {
        "thesis": run_table_row(thesis_law, ThesisGains(), gait, done=(o, steps)),
        "thesis_90": run_table_row(thesis_law, ThesisGains(), gait, done=(o90, steps90)),
        "thesis_stiff": run_table_row(thesis_law, ThesisGains(eps=0.125), gait, n_steps=7),
        "book": run_table_row(book_law, BookGains(), gait, done=(ob, stepsb)),
        "book_no_half": run_table_row(book_law, BookGains(half_rate=False), gait, n_steps=7),
        "book_no_ff": run_table_row(book_law, BookGains(feedforward=False), gait, n_steps=7),
        "masses_fixed": run_table_row(thesis_law, ThesisGains(), replace(gait, r_bar=0.0), n_steps=7),
    }
    ck.record("table", rows)
    ck.true("table: every variant settles to a periodic gait within 7 or 8 steps (change of step length below 0.1 mm)",
            all(r["drift"] < 1e-4 for r in rows.values()), str({k: f"{r['drift']:.1e}" for k, r in rows.items()}))
    ck.true("table: holding the reaction masses still changes the step by less than a micrometer (the torso is decoupled)",
            abs(rows["masses_fixed"]["step_length"] - steps[6]["step_length"]) < 1e-6,
            f"{abs(rows['masses_fixed']['step_length'] - steps[6]['step_length']):.1e} m")
    ck.true("table: at heading 0 the thesis law and eq6-law with k_R = d_1 + d_3 are the same law (planar motion about e2)",
            abs(rows["book"]["step_length"] - rows["thesis"]["step_length"]) < 1e-9 and abs(rows["book"]["peak_ankle"] - rows["thesis"]["peak_ankle"]) < 1e-6)
    ck.true("table: halving eps reduces the largest leg error and raises the peak ankle torque",
            rows["thesis_stiff"]["leg_error_deg"] < rows["thesis"]["leg_error_deg"] and rows["thesis_stiff"]["peak_ankle"] > rows["thesis"]["peak_ankle"])
    ck.true("table: without feedforward the leg error at touchdown is several times larger",
            rows["book_no_ff"]["leg_error_end_deg"] > 3 * rows["book"]["leg_error_end_deg"],
            f"{rows['book_no_ff']['leg_error_end_deg']:.2f} against {rows['book']['leg_error_end_deg']:.2f} deg")

    # ------------------------------------------------------------------
    # Stage 5: linearization, LQR, integration
    # ------------------------------------------------------------------
    lin = linearize(eom, variables, inputs)
    gl = load_module(module_source(lin, layout), "ch20_stance_lin")
    wA = wB = 0.0
    for t in (0.1, 0.5, 0.9):
        st = primitive_state(t, 0.3, gait)
        st = perturbed(st, 0.05 * rng.normal(size=26))
        u = nominal_input(t, 0.3, gait)
        A, B = linearize_stance(st, u)
        a = accelerations(gm, gm_pack(st), pack_u(u), gm_p)
        Mg = np.asarray(gl.M_lin(gm_pack(st), a, pack_u(u), gm_p))
        Ag = np.linalg.solve(Mg, np.asarray(gl.A_lin(gm_pack(st), a, pack_u(u), gm_p)))
        Bg = np.linalg.solve(Mg, np.asarray(gl.B_lin(gm_pack(st), a, pack_u(u), gm_p)))
        wA = max(wA, np.abs(Ag - A).max())
        wB = max(wB, np.abs(Bg - B).max())
    ck.true("linearization of the stance dynamics: finite differences along the variations equal geomech linearize, all 26 rows", wA < 1e-7 and wB < 1e-9, f"A {wA:.1e}, B {wB:.1e}")
    # the linear model predicts a neighboring solution
    t0 = 0.2
    st0 = primitive_state(t0, 0.0, gait)
    s0 = 1e-3 * rng.normal(size=26)
    sa, sb, s = st0, perturbed(st0, s0), s0.copy()
    hh = 2e-3
    for k in range(100):
        t = t0 + k * hh
        u = nominal_input(t, 0.0, gait)
        A, _ = linearize_stance(primitive_state(t, 0.0, gait), u)
        A2, _ = linearize_stance(primitive_state(t + hh / 2, 0.0, gait), nominal_input(t + hh / 2, 0.0, gait))
        A3, _ = linearize_stance(primitive_state(t + hh, 0.0, gait), nominal_input(t + hh, 0.0, gait))
        k1 = A @ s
        k2 = A2 @ (s + hh / 2 * k1)
        k3 = A2 @ (s + hh / 2 * k2)
        k4 = A3 @ (s + hh * k3)
        s = s + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        um = lambda tt, _s: nominal_input(tt, 0.0, gait)
        sb = biped.step(sb, um, hh, t=t)
    actual = error_state(sb, primitive(t0 + 100 * hh, 0.0, gait))
    rel = np.linalg.norm(actual - s) / np.linalg.norm(actual)
    ck.true("the linear model predicts a neighbor of the primitive over 0.2 s of open-loop motion to within 1%", rel < 0.01, f"rel err {rel:.1e}, growth x{np.linalg.norm(actual) / np.linalg.norm(s0):.1f}")
    ck.record("linear_prediction", {"rel_err": float(rel), "growth": float(np.linalg.norm(actual) / np.linalg.norm(s0))})
    # open loop, the stance phase is unstable: the leg is an inverted pendulum
    A_mid, _ = linearize_stance(primitive_state(0.5, 0.0, gait), nominal_input(0.5, 0.0, gait))
    lam = float(np.linalg.eigvals(A_mid).real.max())
    ck.true("the linearization at mid-stance has an unstable eigenvalue near sqrt(m g l / J_2(l))",
            abs(lam - np.sqrt(p.m * G * 1.0 / (p.J_0[1, 1] + p.m * 1.0))) < 0.15, f"{lam:.2f} 1/s")
    ck.record("unstable_rate", lam)
    # LQR for one stance phase from the state just after a periodic impact
    Qw = np.diag(np.r_[400.0, 400 * np.ones(3), 400 * np.ones(9), 1.0, np.ones(3), np.ones(9)])
    Ru = np.eye(13)
    tk, Ks, P0 = lqr_gains(0.0, gait, Qw, Ru, Qw, n=100)
    ck.true("LQR: the Riccati solution at the start of the step is symmetric positive definite", np.linalg.eigvalsh(P0).min() > 0, f"lambda_min {np.linalg.eigvalsh(P0).min():.2f}")

    def lqr_ctrl(t, st, d):
        k = min(int(round(t / (tk[1] - tk[0]))), len(tk) - 1)
        return unpack_u(pack_u(nominal_input(t, 0.0, gait)) - Ks[k] @ error_state(st, d))

    post = steps[-1]["post"]
    run_l, _ = stance_phase(post, 0.0, gait, lqr_ctrl)
    run_t, _ = stance_phase(post, 0.0, gait, lambda t, st, d: thesis_law(st, d))
    run_b, _ = stance_phase(post, 0.0, gait, lambda t, st, d: book_law(st, d))
    s_post = error_state(post, primitive(0.0, 0.0, gait))

    def cost(run):
        return float(np.trapezoid(run["u2"], run["t"]))

    def summary(run):
        return dict(leg_error_max_deg=float(np.degrees(max(run["a1"].max(), run["a2"].max()))),
                    leg_error_end_deg=float(np.degrees(max(run["a1"][-1], run["a2"][-1]))),
                    peak_torque=float(np.abs(run["tau"]).max()), input_cost=cost(run))

    ck.record("lqr", {"Q_position": 400.0, "Q_velocity": 1.0, "R_u": 1.0, "predicted_cost": float(s_post @ P0 @ s_post),
                      "lqr": summary(run_l), "thesis": summary(run_t), "book": summary(run_b)})
    ck.true("LQR from the post-impact state: smaller leg error at the end of the stance phase than the thesis law",
            summary(run_l)["leg_error_end_deg"] < summary(run_t)["leg_error_end_deg"],
            f"{summary(run_l)['leg_error_end_deg']:.3f} against {summary(run_t)['leg_error_end_deg']:.3f} deg")
    # integration: uncontrolled stance with a constant leg force, where energy minus f l is conserved
    st0 = primitive_state(0.2, 0.0, gait)
    st0 = replace(st0, OmegaP=np.array([0.3, -0.2, 1.0]), Omega2=np.array([0.2, 0.5, -0.3]))
    u_c = (0.9 * p.m * G, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    inv = lambda st: biped.energy(st) - u_c[0] * st.l
    drift = {}
    for name, stepper in (("library", fast_step), ("explicit", explicit_update)):
        for hh in (1e-3, 5e-4):
            st, worst_e, worst_m = st0, 0.0, 0.0
            for _ in range(int(round(0.6 / hh))):
                st = stepper(st, u_c, hh)
                worst_e = max(worst_e, abs(inv(st) - inv(st0)))
            drift[f"{name}_{hh:g}"] = dict(energy=float(worst_e), manifold=float(biped.on_manifold(st)))
    ck.record("integrators", drift)
    ck.true("uncontrolled stance, 0.6 s: the library step keeps the energy to 1e-9 J, the explicit update of the thesis does not",
            drift["library_0.001"]["energy"] < 1e-9 and drift["explicit_0.001"]["energy"] > 1e-4,
            f"{drift['library_0.001']['energy']:.1e} J against {drift['explicit_0.001']['energy']:.1e} J")
    ck.true("explicit update: the energy error is proportional to h",
            1.7 < drift["explicit_0.001"]["energy"] / drift["explicit_0.0005"]["energy"] < 2.3,
            f"ratio {drift['explicit_0.001']['energy'] / drift['explicit_0.0005']['energy']:.2f}")
    ck.true("both stay on SO(3) to rounding error", max(d["manifold"] for d in drift.values()) < 1e-13)
    # the walk does not depend on the time step: 12 steps at h = 1 ms and at 0.5 ms
    o2, steps2 = walk([0.0] * 4, h=5e-4, record=0)
    ck.record("walk_h", {"h_1ms": steps[3]["step_length"], "h_05ms": steps2[3]["step_length"]})
    ck.true("walking: halving the time step and the sampling period changes the fourth step by less than 1 mm",
            abs(steps[3]["step_length"] - steps2[3]["step_length"]) < 1e-3, f"{abs(steps[3]['step_length'] - steps2[3]['step_length']):.1e} m")

    # ------------------------------------------------------------------
    # Stage 6: Cassie. Nothing is simulated: only what the library states is recorded.
    # ------------------------------------------------------------------
    cassie = robots.get("cassie")
    ck.true("cassie in the library: 14 degrees of freedom, floating base, 8 inputs", cassie.dim == 14 and
            sum(np.size(x) for x in cassie.zero_input()) == 8)
    ck.record("runtime_s", round(time.time() - t_start, 1))
    ck.finish()


def turn_plan():
    """Headings and torso rolls of the turn: four steps straight, six steps that add
    15 degrees each with the torso rolled 20 degrees into the turn, four steps straight."""
    hd = [0.0] * 4 + list(np.radians(15.0) * np.arange(1, 7)) + [np.radians(90.0)] * 4
    rolls = [0.0] * 4 + [np.radians(-20.0)] * 6 + [0.0] * 4
    return hd, rolls


def skater(h=1e-3, t_f=1.5, spin=3.0, r_out=0.25, r_in=0.05):
    """The torso spins freely about the vertical while the slide forces pull the reaction
    masses in from r_out to r_in. No torque acts on the torso."""
    p = PRM
    st = biped.State(l=0.9, l_dot=0.0, R1=I3, Omega1=np.zeros(3), RP=I3, OmegaP=np.array([0.0, 0.0, spin]),
                     R2=I3, Omega2=np.zeros(3), slide=np.full(3, r_out), slide_dot=np.zeros(3))
    mom = lambda s: s.RP @ biped.torso_inertia(s.slide) @ s.OmegaP
    kin = lambda s: 0.5 * s.OmegaP @ biped.torso_inertia(s.slide) @ s.OmegaP + p.m_r * s.slide_dot @ s.slide_dot
    mom0, T0 = mom(st), kin(st)
    log = dict(t=[], slide=[], spin=[], RP=[], mom=[], J=[])
    n = int(round(t_f / h))
    for k in range(n + 1):
        t = k * h
        x = min(max((t - 0.3) / 0.6, 0.0), 1.0)
        blend = 3 * x**2 - 2 * x**3
        r_d = r_out + (r_in - r_out) * blend
        r_d_dot = (r_in - r_out) * (6 * x - 6 * x**2) / 0.6 if 0.0 < x < 1.0 else 0.0
        OmP = st.OmegaP
        f_s = (-2 * p.m_r * np.array([s * OmP @ P @ OmP for s, P in zip(st.slide, PI_I)])
               - 400.0 * (st.slide - r_d) - 30.0 * (st.slide_dot - r_d_dot))
        log["t"].append(t); log["slide"].append(st.slide.copy()); log["spin"].append(OmP[2]); log["RP"].append(st.RP)
        log["mom"].append(mom(st)[2]); log["J"].append(biped.torso_inertia(st.slide)[2, 2])
        if k < n:
            u = (p.m * G, np.zeros(3), np.zeros(3), np.zeros(3), f_s)
            st = fast_step(st, u, h)
    out = {k: np.array(v) for k, v in log.items()}
    out["summary"] = dict(spin_start=out["spin"][0], spin_end=out["spin"][-1],
                          inertia_ratio=out["J"][0] / out["J"][-1], momentum_error=float(np.abs(out["mom"] - mom0[2]).max()),
                          energy_gain=float(kin(st) - T0),
                          energy_gain_formula=float(0.5 * mom0[2]**2 * (1 / out["J"][-1] - 1 / out["J"][0])),
                          J_start=out["J"][0], J_end=out["J"][-1], slide_end=float(st.slide[0]))
    return out


if __name__ == "__main__":
    main()
