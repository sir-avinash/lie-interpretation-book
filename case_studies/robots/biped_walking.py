"""Walking for the Reaction Mass Biped: impact map, motion primitive, tracking laws, hybrid loop.

The stance dynamics live in `reaction_mass_biped.py` (eq20-stance). This module adds
what Chapter 20 needs to make the biped walk, each piece with the label it implements:

* the impact map of a swing foot that lands and sticks (sec20-impact): `impact`,
  closed form eq20-impact, the linear system eq20-kkt it solves (`impact_matrices`,
  `impact_kkt`) and the exchange of the legs eq20-relabel (`relabel`);
* the floating-base model behind the impact, built mass by mass: `particles`,
  `angular_momentum_about`, `kinetic_floating`, and its geomech Lagrangian
  `floating_lagrangian`;
* the motion primitive eq20-primitive (`Gait`, `primitive`, `primitive_state`);
* two tracking laws: the thesis's, with the weighted error eq20-right and the
  torque eq20-law (`thesis_law`, `ThesisGains`, its Lyapunov function `thesis_V`),
  and the attitude law eq6-law on each factor with the -(1/2) J_dot e_Omega term
  (`book_law`, `BookGains`, `book_V0`). Both are reached through
  `ReactionMassBiped.tracking_controller`;
* the linearization of the stance dynamics along the variations (sec7-variation),
  reached through `ReactionMassBiped.linearize`;
* the hybrid walking loop of sec20-walk (`walk`);
* the explicit update of the thesis (its eqs. 5.36 to 5.40), kept only for
  comparison: Chapter 20 shows that its energy drifts in proportion to h
  (`explicit_update`). Simulate with `ReactionMassBiped.step` instead.

Conventions. A desired state is a dict `d` with the keys of the stance state and
the accelerations `l_ddot`, `Omega1_dot`, `OmegaP_dot`, `Omega2_dot`, `slide_ddot`.
The hip is at x_P = l R1 e3 from the stance foot; both leg axes R1 e3 and R2 e3
point from foot to hip, so the swing foot is at x_P - l_0 R2 e3. `params` may be
omitted everywhere: the default `ReactionMassBiped().params` is used.

What is not here: the floating base in flight (a foot always sticks), friction
cones (check the impulse and the ground force yourself, warn20-impact), and an
implicit integrator for the variable inertias (sec12-biped).

Only NumPy is needed. geomech is imported inside `floating_lagrangian()`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .core import E2, E3, I3, expm_so3, hat, log_so3, vee
from .reaction_mass_biped import PI, PI_I, ReactionMassBiped

_ROBOT = ReactionMassBiped()


def _params(p):
    return p if p is not None else _ROBOT.params


def Rx(a):
    """Rotation by a about e1."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    """Rotation by a about e2."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    """Rotation by a about e3."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# =====================================================================
# 1. The impact map and the floating-base model (sec20-impact)
# =====================================================================
def hip_velocity(l, l_dot, R1, Omega1):
    """Velocity v_P of the hip x_P = l R1 e3 in stance."""
    return l_dot * R1 @ E3 + l * R1 @ np.cross(Omega1, E3)


def swing_foot(state, l_0):
    """Position of the swing foot relative to the stance foot: l R1 e3 - l_0 R2 e3."""
    return state.l * state.R1 @ E3 - l_0 * state.R2 @ E3


def swing_foot_velocity(state, l_0):
    """Velocity of the swing foot."""
    return hip_velocity(state.l, state.l_dot, state.R1, state.Omega1) \
        - l_0 * state.R2 @ np.cross(state.Omega2, E3)


def impact(state, l_0, p=None):
    """The swing foot lands and sticks: eq20-impact, then eq20-relabel.

    Returns (new stance state, impulse p_c on the foot, step vector). Only the swing
    leg and the hip velocity change:
        J(l_0) Omega2+ = J_0 Omega2- + m l_0 e3 x R2^T v_P-,   J(l) = J_0 + m l^2 Pi,
        v_P+ = l_0 R2 (Omega2+ x e3),      p_c = m (v_P+ - v_P-).
    Then the legs trade names: the old swing leg is the new stance leg.
    """
    p = _params(p)
    m = p.m
    v_minus = hip_velocity(state.l, state.l_dot, state.R1, state.Omega1)
    J_c = p.J_0 + m * l_0**2 * PI
    Omega2_plus = np.linalg.solve(J_c, p.J_0 @ state.Omega2 + m * l_0 * np.cross(E3, state.R2.T @ v_minus))
    v_plus = l_0 * state.R2 @ np.cross(Omega2_plus, E3)
    new = replace(state, l=l_0, l_dot=0.0, R1=state.R2, Omega1=Omega2_plus,
                  R2=state.R1, Omega2=state.Omega1)
    return new, m * (v_plus - v_minus), swing_foot(state, l_0)


def impact_matrices(R2, slide, l_0, p=None):
    """D and A of eq20-kkt for the floating-base model, in the velocity order
    (Omega1, OmegaP, slide_dot, Omega2, v_P): D v+ - D v- = A^T p_c, A v+ = 0."""
    p = _params(p)
    D = np.zeros((15, 15))
    D[0:3, 0:3] = p.J_0
    D[3:6, 3:6] = _ROBOT.torso_inertia(slide, p)
    D[6:9, 6:9] = 2 * p.m_r * I3
    D[9:12, 9:12] = p.J_0
    D[12:15, 12:15] = p.m * I3
    A = np.zeros((3, 15))
    A[:, 9:12] = l_0 * R2 @ hat(E3)          # d/dt(-l_0 R2 e3) = l_0 R2 hat(e3) Omega2
    A[:, 12:15] = I3
    return D, A


def impact_kkt(state, l_0, p=None):
    """The impact solved as the 18 x 18 linear system eq20-kkt (thesis eq. 5.20 with the
    right side +D v-, as the relation above it requires).

    Returns (v-, v+, p_c, D, A), velocities in the order of `impact_matrices`."""
    p = _params(p)
    D, A = impact_matrices(state.R2, state.slide, l_0, p)
    v = np.concatenate([state.Omega1, state.OmegaP, state.slide_dot, state.Omega2,
                        hip_velocity(state.l, state.l_dot, state.R1, state.Omega1)])
    K = np.block([[D, -A.T], [A, np.zeros((3, 3))]])
    sol = np.linalg.solve(K, np.concatenate([D @ v, np.zeros(3)]))
    return v, sol[:15], sol[15:], D, A


def relabel(v):
    """eq20-relabel on a floating-base velocity: trade the names of the legs."""
    return np.concatenate([v[9:12], v[3:9], v[0:3], v[12:15]])


def particles(state, v_P, p=None):
    """The point masses of the floating-base model: (mass, position from the hip, velocity)."""
    p = _params(p)
    out = [(2 * p.m_leg, np.zeros(3), v_P)]
    for i in range(3):
        a = state.RP[:, i]
        a_dot = state.RP @ np.cross(state.OmegaP, np.eye(3)[i])
        for sign in (1.0, -1.0):
            out.append((p.m_r, sign * state.slide[i] * a,
                        v_P + sign * (state.slide_dot[i] * a + state.slide[i] * a_dot)))
    return out


def angular_momentum_about(point_from_hip, state, v_P, Omega1, OmegaP, Omega2, p=None):
    """World-frame angular momentum about a point, built mass by mass."""
    p = _params(p)
    H = state.R1 @ p.J_0 @ Omega1 + state.RP @ p.J_P @ OmegaP + state.R2 @ p.J_0 @ Omega2
    for mass, r, v in particles(replace(state, OmegaP=OmegaP), v_P, p):
        H = H + mass * np.cross(r - point_from_hip, v)
    return H


def kinetic_floating(state, v_P, Omega1, OmegaP, Omega2, slide_dot, p=None):
    """Kinetic energy of the floating-base model, built mass by mass."""
    p = _params(p)
    T = 0.5 * (Omega1 @ p.J_0 @ Omega1 + OmegaP @ p.J_P @ OmegaP + Omega2 @ p.J_0 @ Omega2)
    for mass, _, v in particles(replace(state, OmegaP=OmegaP, slide_dot=slide_dot), v_P, p):
        T += 0.5 * mass * v @ v
    return T


def floating_lagrangian():
    """The geomech Lagrangian of the floating-base model, built mass by mass, with no
    inputs: hip x, the three attitudes, the slides. Returns (L, variables)."""
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


# =====================================================================
# 2. The motion primitive eq20-primitive (thesis eqs. 5.41 and 5.42)
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
    """eq20-primitive: desired stance state and accelerations, a dict, at time t of a
    step with heading psi."""
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
    """The stance state of the primitive at time t."""
    d = primitive(t, psi, gait)
    return _ROBOT.State(l=d["l"], l_dot=d["l_dot"], R1=d["R1"], Omega1=d["Omega1"], RP=d["RP"],
                        OmegaP=d["OmegaP"], R2=d["R2"], Omega2=d["Omega2"], slide=d["slide"],
                        slide_dot=d["slide_dot"])


# =====================================================================
# 3. Two tracking laws (eq20-right, eq20-law; eq6-law per factor)
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
        """The weights d_i of eq20-right."""
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
    """tr(D (I - R Rd^T)): the weighted error function of eq20-right."""
    return float(np.sum(D * (1.0 - np.diag(R @ Rd.T))))


def e_right(R, Rd, D):
    """The error e of eq20-right, the slope of Psi_right under delta R = R hat(eta):
    sum_i d_i (R^T e_i) x (Rd^T e_i)."""
    return sum(D[i] * np.cross(R[i, :], Rd[i, :]) for i in range(3))


def torso_inertia_rate(slide, slide_dot, p=None):
    """JP_dot = 4 m_r sum_i slide_i slide_dot_i Pi_i."""
    p = _params(p)
    return 4 * p.m_r * sum(s * sd * P for s, sd, P in zip(slide, slide_dot, PI_I))


def _stack(n_1, n_P, n_2, state):
    """Joint torques (tau_1, tau_P, tau_2) that put the net torques n on the bodies."""
    tau_2 = n_2
    tau_P = n_P + state.RP.T @ state.R2 @ tau_2
    tau_1 = n_1 + state.R1.T @ state.RP @ tau_P
    return tau_1, tau_P, tau_2


def thesis_law(state, d, gains=ThesisGains(), p=None):
    """The thesis's law, eq20-law with the error eq20-right on each body: thesis
    eqs. (5.61) to (5.65) with Phi the identity, gravity along -e3.
    Returns u = (f, tau_1, tau_P, tau_2, f_s)."""
    p = _params(p)
    m, D, G = p.m, gains.D, p.g
    k1, kP, k2 = gains.k_Omega
    l, l_dot, R1, Om1 = state.l, state.l_dot, state.R1, state.Omega1
    f = (m * d["l_ddot"] - m * l * Om1 @ PI @ Om1 + m * G * E3 @ (R1 @ E3)
         - gains.k_l * (l - d["l"]) - gains.k_ldot * (l_dot - d["l_dot"]))
    J1 = p.J_0 + m * l**2 * PI
    n_1 = (J1 @ d["Omega1_dot"] + np.cross(d["Omega1"], J1 @ Om1)
           - m * G * l * np.cross(R1.T @ E3, E3) + m * l * l_dot * PI @ (Om1 + d["Omega1"])
           - e_right(R1, d["R1"], D) - k1 * (Om1 - d["Omega1"]))
    JP = _ROBOT.torso_inertia(state.slide, p)
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


def thesis_V(state, d, gains=ThesisGains(), p=None):
    """The Lyapunov function V of the thesis's law, its eq. (5.66), with Phi the identity."""
    p = _params(p)
    m, D = p.m, gains.D
    J1 = p.J_0 + m * state.l**2 * PI
    JP = _ROBOT.torso_inertia(state.slide, p)
    w1, wP, w2 = state.Omega1 - d["Omega1"], state.OmegaP - d["OmegaP"], state.Omega2 - d["Omega2"]
    rt, rt_dot = state.slide - d["slide"], state.slide_dot - d["slide_dot"]
    return (0.5 * m * (state.l_dot - d["l_dot"])**2 + 0.5 * gains.k_l * (state.l - d["l"])**2
            + 0.5 * w1 @ J1 @ w1 + Psi_right(state.R1, d["R1"], D)
            + 0.5 * wP @ JP @ wP + p.m_r * rt_dot @ rt_dot + 0.5 * gains.k_r * rt @ rt
            + Psi_right(state.RP, d["RP"], D)
            + 0.5 * w2 @ p.J_0 @ w2 + Psi_right(state.R2, d["R2"], D))


def thesis_V_rate(state, d, gains=ThesisGains()):
    """The rate of thesis_V along the closed loop, thesis eq. (5.70)."""
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
    """e_R of Chapter 3: (Rd^T R - R^T Rd)^vee / 2."""
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def book_law(state, d, gains=BookGains(), p=None):
    """eq6-law on each of the three factors, with the gravity torque canceled and the
    two terms that a configuration-dependent inertia adds: + J_dot Omega - (1/2) J_dot e_Omega.
    Returns u = (f, tau_1, tau_P, tau_2, f_s)."""
    p = _params(p)
    m, G = p.m, p.g
    ff = 1.0 if gains.feedforward else 0.0
    l, l_dot, R1 = state.l, state.l_dot, state.R1
    f = (ff * m * d["l_ddot"] - m * l * state.Omega1 @ PI @ state.Omega1 + m * G * E3 @ (R1 @ E3)
         - gains.k_l * (l - d["l"]) - gains.k_ldot * (l_dot - ff * d["l_dot"]))
    J1 = p.J_0 + m * l**2 * PI
    J1_dot = 2 * m * l * l_dot * PI
    JP = _ROBOT.torso_inertia(state.slide, p)
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


def book_V0(state, d, gains=BookGains(), p=None):
    """Sum over the factors of eq6-V0, plus the two PD energies of the lengths.
    Returns (V0, its rate eq6-V0-dot along the closed loop of book_law)."""
    p = _params(p)
    m = p.m
    J1 = p.J_0 + m * state.l**2 * PI
    JP = _ROBOT.torso_inertia(state.slide, p)
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
# 4. Linearization of the stance dynamics (sec7-variation, eq3-delta-Omega)
# =====================================================================
# Linear state s = (dl, dr, eta_1, eta_P, eta_2, dl_dot, dr_dot, dOmega_1, dOmega_P, dOmega_2),
# 26 numbers: the variations of sec7-variation, one block per factor.
def perturbed(state, s):
    """The stance state moved by a linear state s (26 numbers)."""
    return _ROBOT.State(l=state.l + s[0], slide=state.slide + s[1:4], R1=state.R1 @ expm_so3(s[4:7]),
                        RP=state.RP @ expm_so3(s[7:10]), R2=state.R2 @ expm_so3(s[10:13]),
                        l_dot=state.l_dot + s[13], slide_dot=state.slide_dot + s[14:17],
                        Omega1=state.Omega1 + s[17:20], OmegaP=state.OmegaP + s[20:23],
                        Omega2=state.Omega2 + s[23:26])


def pack_u(u):
    """(f, tau_1, tau_P, tau_2, f_s) -> 13 numbers."""
    return np.concatenate([[u[0]], u[1], u[2], u[3], u[4]])


def unpack_u(v):
    """13 numbers -> (f, tau_1, tau_P, tau_2, f_s)."""
    return float(v[0]), v[1:4], v[4:7], v[7:10], v[10:13]


def accel_vector(state, u, p=None):
    """The accelerations in the order of the linear state: l, slide, Omega1, OmegaP, Omega2."""
    a = _ROBOT.dynamics(state, u, _params(p))
    return np.concatenate([[a[0]], a[4], a[1], a[2], a[3]])


def linearize_stance(state, u, p=None, d=1e-6):
    """A (26 x 26) and B (26 x 13) of s_dot = A s + B du about a state and input.

    The kinematic rows are exact: d(dl)/dt = dl_dot, d(eta)/dt = dOmega - Omega x eta
    (eq3-delta-Omega). The dynamic rows are central differences of the library dynamics
    along the variations."""
    p = _params(p)
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
    """The error state of sec13-error on the product: logarithm errors and velocity deviations,
    in the order of the linear state."""
    return np.concatenate([[state.l - d["l"]], state.slide - d["slide"],
                           log_so3(d["R1"].T @ state.R1), log_so3(d["RP"].T @ state.RP),
                           log_so3(d["R2"].T @ state.R2),
                           [state.l_dot - d["l_dot"]], state.slide_dot - d["slide_dot"],
                           state.Omega1 - d["Omega1"], state.OmegaP - d["OmegaP"], state.Omega2 - d["Omega2"]])


# =====================================================================
# 5. The hybrid walking loop (sec20-walk)
# =====================================================================
def hip_acceleration(state, acc):
    """Second derivative of x_P = l R1 e3 from the stance accelerations."""
    l, ld, R, Om = state.l, state.l_dot, state.R1, state.Omega1
    l_dd, Om_d = acc[0], acc[1]
    return R @ (l_dd * E3 + 2 * ld * np.cross(Om, E3)
                + l * (np.cross(Om_d, E3) + np.cross(Om, np.cross(Om, E3))))


def attitude_angle(R, Rd):
    """Angle of the rotation Rd^T R."""
    return float(np.linalg.norm(log_so3(Rd.T @ R)))


def walk(headings, gait=Gait(), law=thesis_law, gains=None, h=1e-3, state0=None, rolls=None,
         max_step_time=3.0, p=None, record=5):
    """Walk len(headings) steps, sec20-walk. Step i tracks the primitive with heading
    headings[i]; it ends when the swing foot comes down to the ground, where the impact
    map eq20-impact is applied.

    law is thesis_law or book_law, gains defaults to the matching gains. The controller
    is sampled every h and held, and the stance dynamics are advanced with
    ReactionMassBiped.step. The touchdown time is found by bisection inside the step.
    rolls, if given, sets the torso roll of each step. Every `record` samples the
    state is logged (record=0 logs nothing).

    Returns (log, steps): a dict of arrays, with also "final_state", "final_foot" and
    "peak" (the largest |tau_1|, |tau_P|, |tau_2| componentwise), and a list of per-step
    summaries (step_time, step_vector, step_length, impulse, clearance, energy_lost,
    pre and post impact states, ...). Unilateral contact and friction are not enforced:
    read log["F"] and the impulses."""
    p = _params(p)
    robot = _ROBOT
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
                acc = robot.dynamics(state, u, p)
                F = p.m * (hip_acceleration(state, acc) + p.g * E3)
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
                               ("E", robot.energy(state, p)),
                               ("H_foot", angular_momentum_about(state.l * state.R1 @ E3, state, vP, state.Omega1,
                                                                 state.OmegaP, state.Omega2, p)),
                               ("ts", ts)):
                    log[k].append(val)
            if e_start is None:
                e_start = (attitude_angle(state.R1, d["R1"]), attitude_angle(state.R2, d["R2"]))
            new = robot.step(state, u, h, p)
            z0, z1 = swing_foot(state, gait.l_0)[2], swing_foot(new, gait.l_0)[2]
            if ts > 0.5 * gait.t_s and z1 <= 0.0 < z0:
                lo, hi = 0.0, h                               # bisect for the touchdown time
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if swing_foot(robot.step(state, u, mid, p), gait.l_0)[2] > 0.0:
                        lo = mid
                    else:
                        hi = mid
                state, ts, done = robot.step(state, u, hi, p), ts + hi, True
            else:
                if 0.1 * gait.t_s < ts < 0.9 * gait.t_s:
                    clearance = min(clearance, z1)
                state, ts = new, ts + h
                if ts > max_step_time:
                    raise RuntimeError(f"step {i}: no touchdown within {max_step_time} s")
        d = primitive(ts, psi, g_i)
        pre = state
        E_minus = robot.energy(pre, p)
        state, impulse, stride = impact(pre, gait.l_0, p)
        steps.append(dict(step_time=ts, step_vector=stride, step_length=float(np.linalg.norm(stride[:2])),
                          foot_height_error=float(stride[2]), impulse=impulse, clearance=clearance,
                          e_start=e_start, e_end=(attitude_angle(pre.R1, d["R1"]), attitude_angle(pre.R2, d["R2"])),
                          energy_lost=E_minus - robot.energy(state, p), pre=pre, post=state, foot=foot.copy()))
        foot = foot + np.array([stride[0], stride[1], 0.0])
        t0 += ts
    out = {k: np.array(v) for k, v in log.items()}
    out["final_state"], out["final_foot"], out["peak"] = state, foot, peak
    return out, steps


# =====================================================================
# 6. The explicit update of the thesis, for comparison only
# =====================================================================
def explicit_update(state, u, h, p=None):
    """THE THESIS'S SCHEME, NOT THE LIBRARY'S: one step of its discrete equations
    (eqs. 5.36 to 5.40), in the book's convention and with the sign of the centrifugal
    term on the slides corrected. Inputs are held.

    Chapter 20 (sec20-recipe) shows that its energy error grows in proportion to h,
    where ReactionMassBiped.step keeps it at rounding level. Kept to reproduce that
    comparison; simulate with ReactionMassBiped.step."""
    p = _params(p)
    f, tau_1, tau_P, tau_2, f_s = u
    m, G = p.m, p.g
    J1 = p.J_0 + m * state.l**2 * PI
    JP = _ROBOT.torso_inertia(state.slide, p)
    U1, UP, U2 = (expm_so3(h * w) for w in (state.Omega1, state.OmegaP, state.Omega2))
    R1, RP, R2 = state.R1 @ U1, state.RP @ UP, state.R2 @ U2
    l, slide = state.l + h * state.l_dot, state.slide + h * state.slide_dot
    J1n, JPn = p.J_0 + m * l**2 * PI, _ROBOT.torso_inertia(slide, p)
    Om1 = np.linalg.solve(J1n, U1.T @ J1 @ state.Omega1
                          + h * (m * G * l * np.cross(R1.T @ E3, E3) + tau_1 - R1.T @ RP @ tau_P))
    OmP = np.linalg.solve(JPn, UP.T @ JP @ state.OmegaP + h * (tau_P - RP.T @ R2 @ tau_2))
    Om2 = np.linalg.solve(p.J_0, U2.T @ p.J_0 @ state.Omega2 + h * tau_2)
    l_dot = state.l_dot + h * (l * Om1 @ PI @ Om1 - G * E3 @ (R1 @ E3) + f / m)
    slide_dot = state.slide_dot + h * (np.array([s * OmP @ P @ OmP for s, P in zip(slide, PI_I)])
                                       + np.asarray(f_s) / (2 * p.m_r))
    return _ROBOT.State(l=l, l_dot=l_dot, R1=R1, Omega1=Om1, RP=RP, OmegaP=OmP, R2=R2, Omega2=Om2,
                        slide=slide, slide_dot=slide_dot)
