"""Chapter 19 verification: three quadrotors carry a payload on cables, end to end.

The mission is assembled from blocks that the earlier chapters verified:

* the team, the cable and their step: the robot model library (case_studies/robots);
* jets, the attitude part of the flat map and the chain recursion: verify/ch10.py (imported);
* the small QP solver: verify/ch08.py (imported);
* the attitude law eq6-law: robots.core.attitude_tracking_torque;
* the sphere law eq6-law-s2: written here and checked against verify/ch06.py;
* the variation-based linearization: written here for any robot of the library,
  checked against geomech's and against perturbed simulations.

Run:  python verify/ch19.py          (a few minutes; `--fresh` ignores the cache)
"""
from __future__ import annotations

import ast
import math
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "case_studies"))

import robots  # noqa: E402
from common import Checker  # noqa: E402
from robots.core import (E1, E2, E3, I3, attitude_tracking_torque, e_omega, e_q, e_R,  # noqa: E402
                         expm_so3, hat, log_so3, tangent, unit, vee)
from robots.multi_quadrotor_payload import PayloadParams  # noqa: E402
from robots.flexible_cable import CableParams  # noqa: E402
import ch08  # noqa: E402
import ch10  # noqa: E402
from ch10 import (attitude_from_force, chain_flat, derivs, jcossin, jcross, jder, jdiv,  # noqa: E402
                  jmul, jnorm)

G = 9.81
CACHE = HERE / "ch19_cache.pkl"

# =====================================================================
# 0. The mission, in numbers
# =====================================================================
N_TEAM = 3
F_MAX = 22.0                 # N, largest thrust of one vehicle (our assumption)
R_MIN = 0.8                  # m, smallest distance allowed between two vehicles
T_MIN = 0.5                  # N, smallest tension allowed in a cable
D_PLAN = 0.83                # m, distance that the tension distribution keeps between vehicles
ALPHA1, ALPHA2 = 4.0, 4.0    # s^-1, rates of the barrier condition eq8-rd2
H_STEP = 5e-3                # s, time step and sampling period
T_END = 12.0                 # s
GAP_X, GAP_W = 3.0, 1.45     # m, where the gap is and how wide
ARM = 0.25                   # m, reach of a rotor from the center of a vehicle
S_WIDE, S_GAP = 0.98, 0.10   # N, planned spreading force per cable: cruise, gap
PUSH = dict(vehicle=0, t0=4.7, t1=5.2, force=np.array([0.0, -8.0, 0.0]))
GAINS = dict(k_x=4.0, k_v=3.6,          # payload position, per unit mass (2 rad/s)
             k_RL=4.0, k_OmegaL=3.6,    # payload attitude, per unit inertia (2 rad/s)
             k_q=49.0, k_omega=11.2,    # cable directions (7 rad/s)
             k_R=10.0, k_Omega=1.0)     # vehicle attitude, N·m and N·m·s (about 20 rad/s)


def team(n=N_TEAM, **changes):
    """The library's team with n vehicles and default parameters."""
    return robots.get("multi_quadrotor_payload", params=PayloadParams(n=n, **changes))


# =====================================================================
# 1. Model helpers that the library does not have
# =====================================================================
def vehicle_kinematics(robot, state, acc=None, params=None):
    """Positions and velocities of the vehicles, and accelerations if acc is given."""
    p = params or robot.params
    x = robot.quadrotor_positions(state, p)
    v = (state.v_L + np.cross(state.Omega_L, p.r) @ state.R_L.T
         - p.l[:, None] * np.cross(state.omega, state.q))
    if acc is None:
        return x, v
    a_L, Om_L_dot, om_dot = acc[0], acc[1], acc[2]
    a = np.array([a_L + state.R_L @ (np.cross(Om_L_dot, ri)
                                     + np.cross(state.Omega_L, np.cross(state.Omega_L, ri)))
                  - li * (np.cross(wd, qi) - (wi @ wi) * qi)
                  for ri, li, wd, qi, wi in zip(p.r, p.l, om_dot, state.q, state.omega)])
    return x, v, a


def with_thrust_vectors(state, F):
    """A state whose `R[i] @ e3` equals F[i]: the library's dynamics then sees the
    thrust vectors F with f = 1. Only for evaluating accelerations."""
    Rf = np.zeros_like(state.R)
    Rf[:, :, 2] = F
    return replace(state, R=Rf)


def accel_of_thrust_vectors(robot, state, F, params=None):
    """Accelerations (a_L, Omega_L_dot, omega_dot) for thrust vectors F (n, 3)."""
    n = robot.n
    return robot.dynamics(with_thrust_vectors(state, F), (np.ones(n), np.zeros((n, 3))), params)[:3]


def tensions(robot, state, u, params=None, F=None):
    """Tension of each cable: T_i = q_i . (m_i (a_i + g e3) - f_i R_i e3)."""
    p = params or robot.params
    F = np.array([fi * Ri @ E3 for fi, Ri in zip(u[0], state.R)]) if F is None else F
    acc = accel_of_thrust_vectors(robot, state, F, p)
    _, _, a = vehicle_kinematics(robot, state, acc, p)
    return np.array([qi @ (mi * (ai + p.g * E3) - Fi)
                     for qi, mi, ai, Fi in zip(state.q, p.m, a, F)])


def team_matrices(p, state):
    """The linear system of the library's team, A z = b0 + Bf F, for the unknowns
    z = (a_L, Omega_L_dot, omega_1_dot, ..., omega_n_dot) and the stacked thrust
    vectors F. A copy of MultiQuadrotorPayload.dynamics that keeps F symbolic.
    Also returns Ja, bias with a_i = Ja[i] z + bias[i] (vehicle accelerations)."""
    n, R_L, Om_L = p.n, state.R_L, state.Omega_L
    A, b0, Bf = np.zeros((6 + 3 * n, 6 + 3 * n)), np.zeros(6 + 3 * n), np.zeros((6 + 3 * n, 3 * n))
    A[:3, :3] = p.m_L * I3
    b0[:3] = -p.m_L * p.g * E3
    A[3:6, 3:6] = p.J_L
    b0[3:6] = -hat(Om_L) @ p.J_L @ Om_L
    Jas, biases = [], []
    for i in range(n):
        qi, wi, ri, mi, li = state.q[i], state.omega[i], p.r[i], p.m[i], p.l[i]
        Ja = np.zeros((3, 6 + 3 * n))
        Ja[:, :3] = I3
        Ja[:, 3:6] = -R_L @ hat(ri)
        Ja[:, 6 + 3 * i:9 + 3 * i] = li * hat(qi)
        bias = R_L @ (hat(Om_L) @ hat(Om_L) @ ri) + li * (wi @ wi) * qi
        known = -mi * p.g * E3 - mi * bias
        cols = slice(3 * i, 3 * i + 3)
        A[:3] += mi * Ja
        b0[:3] += known
        Bf[:3, cols] = I3
        lever = hat(ri) @ R_L.T
        A[3:6] += lever @ (mi * Ja)
        b0[3:6] += lever @ known
        Bf[3:6, cols] = lever
        rows = slice(6 + 3 * i, 9 + 3 * i)
        A[rows] = hat(qi) @ (mi * Ja)
        b0[rows] = hat(qi) @ known
        Bf[rows, cols] = hat(qi)
        A[rows, rows] += mi * li * np.outer(qi, qi)
        Jas.append(Ja), biases.append(bias)
    return A, b0, Bf, Jas, biases


def affine_maps(robot, state, params=None):
    """Vehicle accelerations and tensions are affine in the stacked thrust vectors:
    a = a0 + Da F,  T = T0 + DT F,  F in R^{3n}."""
    p = params or robot.params
    n = p.n
    A, b0, Bf, Jas, biases = team_matrices(p, state)
    sol = np.linalg.solve(A, np.column_stack([b0, Bf]))
    z0, Dz = sol[:, 0], sol[:, 1:]
    a0 = np.concatenate([Ja @ z0 + bias for Ja, bias in zip(Jas, biases)])
    Da = np.vstack([Ja @ Dz for Ja in Jas])
    T0, DT = np.zeros(n), np.zeros((n, 3 * n))
    for i in range(n):
        rows = slice(3 * i, 3 * i + 3)
        T0[i] = state.q[i] @ (p.m[i] * (a0[rows] + p.g * E3))
        DT[i] = p.m[i] * state.q[i] @ Da[rows]
        DT[i, rows] -= state.q[i]
    return a0, Da, T0, DT


def affine_maps_probed(robot, state, params=None):
    """The same maps, found by probing the library's model. Used as a check."""
    p = params or robot.params
    n = robot.n

    def both(Fv):
        F = Fv.reshape(n, 3)
        acc = accel_of_thrust_vectors(robot, state, F, p)
        _, _, a = vehicle_kinematics(robot, state, acc, p)
        T = np.array([qi @ (mi * (ai + p.g * E3) - Fi)
                      for qi, mi, ai, Fi in zip(state.q, p.m, a, F)])
        return a.ravel(), T

    a0, T0 = both(np.zeros(3 * n))
    Da, DT = np.zeros((3 * n, 3 * n)), np.zeros((n, 3 * n))
    for k in range(3 * n):
        e = np.zeros(3 * n)
        e[k] = 1.0
        a, T = both(e)
        Da[:, k], DT[:, k] = a - a0, T - T0
    return a0, Da, T0, DT


class Plant(robots.multi_quadrotor_payload.MultiQuadrotorPayload):
    """The library's team with two things it has no input for: an external force on
    each vehicle, applied at its center of mass, and a loss of thrust."""

    def __init__(self, n, params):
        super().__init__(n=n, params=params)
        self.ext = np.zeros((n, 3))          # N, world frame
        self.eff = np.ones(n)                # delivered thrust / commanded thrust

    def dynamics(self, state, u, params=None):
        f, tau = u
        F = np.array([e * fi * Ri @ E3 for e, fi, Ri in zip(self.eff, f, state.R)]) + self.ext
        base = super().dynamics
        a_L, Om_L_dot, om_dot, _ = base(with_thrust_vectors(state, F),
                                        (np.ones(self.n), np.zeros((self.n, 3))), params)
        return a_L, Om_L_dot, om_dot, base(state, u, params)[3]


# =====================================================================
# 2. The redundancy: grasp map and tension distribution
# =====================================================================
def grasp_map(r):
    """G_L = [[I ... I], [hat(r_1) ... hat(r_n)]], 6 x 3n. It maps the forces that
    the cables apply to the payload, in the payload frame, to the wrench."""
    return np.block([[I3 for _ in r], [hat(ri) for ri in r]])


def null_basis(Gm, tol=1e-10):
    """Orthonormal basis of ker G_L: the internal forces."""
    _, s, Vt = np.linalg.svd(Gm)
    rank = int(np.sum(s > tol * s[0]))
    return Vt[rank:].T, rank


def spread_direction(r):
    """Stacked unit vectors r_i / |r_i|: each cable pulls its attachment outward."""
    return np.concatenate([ri / np.linalg.norm(ri) for ri in r])


def distribute(r, wrench, base=None, l=None, d_min=None, t_min=None, solver=None):
    """Forces lam (n, 3) of the cables on the payload, payload frame, with
    G_L lam = wrench, closest to `base` (minimum norm when base is None), subject to
        e3 . lam_i >= t_min                                    (every cable pulls up),
        |r_i - r_j| + e_ij . (l_i lam_i / T_i - l_j lam_j / T_j) >= d_min   (vehicles apart),
    where e_ij is the unit vector from attachment j to attachment i and T_i is the
    size of the unconstrained solution, so that the second line is linear. Its left
    side is the distance of vehicles i and j measured along e_ij, which is never more
    than their distance. Returns (lam, number of active constraints), or (None, -1)."""
    n = len(r)
    Gm = grasp_map(r)
    N, _ = null_basis(Gm)
    base = np.zeros(3 * n) if base is None else np.ravel(base)
    lam0 = base + np.linalg.pinv(Gm) @ (wrench - Gm @ base)
    size = np.linalg.norm(lam0.reshape(n, 3), axis=1)
    rows, rhs = [], []

    def sel(i):
        S = np.zeros((3, 3 * n))
        S[:, 3 * i:3 * i + 3] = I3
        return S

    if t_min is not None:
        for i in range(n):
            rows.append(-E3 @ sel(i)), rhs.append(-t_min)
    if d_min is not None:
        for i in range(n):
            for j in range(i + 1, n):
                gap = np.linalg.norm(r[i] - r[j])
                e = (r[i] - r[j]) / gap
                rows.append(-e @ (l[i] / size[i] * sel(i) - l[j] / size[j] * sel(j)))
                rhs.append(gap - d_min)
    if not rows:
        return lam0.reshape(n, 3), 0
    A, b = np.array(rows), np.array(rhs)
    if np.all(A @ lam0 <= b + 1e-12) or N.shape[1] == 0:
        return lam0.reshape(n, 3), 0
    AN, bN = A @ N, b - A @ lam0
    c = (solver or ch08.solve_qp)(np.eye(N.shape[1]), np.zeros(N.shape[1]), AN, bN)
    if c is None:
        return None, -1
    return (lam0 + N @ c).reshape(n, 3), int(np.sum(np.abs(AN @ c - bN) < 1e-7))


# =====================================================================
# 3. Smooth schedules and the flat map of the team
# =====================================================================
def _smoothstep_poly(order=6):
    """Polynomial that rises from 0 to 1 on [0, 1] with `order` zero derivatives at both ends."""
    P = np.polynomial.Polynomial
    p = (P([0, 1]) * P([1, -1])) ** order
    p = p.integ()
    return p / p(1.0)


_STEP = _smoothstep_poly()
_STEP_D = [_STEP.deriv(k) if k else _STEP for k in range(9)]


def step_jet(t, t0, dur, K=7):
    """Taylor coefficients at t of the smooth step that starts at t0 and takes dur."""
    s = (t - t0) / dur
    out = np.zeros(K)
    if s <= 0:
        return out
    if s >= 1:
        out[0] = 1.0
        return out
    for k in range(K):
        out[k] = _STEP_D[k](s) / math.factorial(k) / dur ** k
    return out


class Plan:
    """The reference of the mission as functions of time, each returning a jet.

    x_L: payload position.  psi_L: payload heading (the payload stays level).
    spread: the internal force, N per cable, along rhat_i.
    """

    def __init__(self, start=(0.0, 0.0, 1.6), length=6.0, drop=1.0, turn=np.radians(60.0),
                 s_wide=S_WIDE, s_gap=S_GAP, t_end=T_END, rate=1.0):
        self.start, self.length, self.drop, self.turn = np.array(start, float), length, drop, turn
        self.s_wide, self.s_gap, self.t_end, self.rate = s_wide, s_gap, t_end, rate

    def x_L(self, t, K=7):
        t = t * self.rate
        j = np.zeros((K, 3))
        j[0] = self.start
        j[:, 0] += self.length * step_jet(t, 1.0, 8.0, K)
        j[:, 2] -= self.drop * step_jet(t, 7.5, 4.0, K)
        return j * self.rate ** np.arange(K)[:, None]

    def psi_L(self, t, K=7):
        t = t * self.rate
        return self.turn * step_jet(t, 1.5, 3.0, K) * self.rate ** np.arange(K)

    def spread(self, t, K=7):
        t = t * self.rate
        j = np.zeros(K)
        j[0] = self.s_wide
        j += (self.s_gap - self.s_wide) * (step_jet(t, 2.5, 1.7, K) - step_jet(t, 5.8, 1.7, K))
        return j * self.rate ** np.arange(K)


def rotz_jet(psi):
    """Jet of R_z(psi) for a scalar jet psi."""
    c, s = jcossin(psi)
    R = np.zeros((len(psi), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, -s, s, c
    R[0, 2, 2] = 1.0
    return R


def team_flat(p, x_L, R_L, internal, psi=None, K=7):
    """Flat map of the team. Inputs are jets (K coefficients): the payload position
    x_L, its attitude R_L, and `internal`, a jet of vectors in R^{3n} that lies in
    ker G_L (the internal force). Optional: heading jets psi of the vehicles.

    Per cable this is eq10-load-map and eq10-load-force of thm10-load:
        T_i q_i = -R_L lam_i,   x_i = x_L + R_L r_i - l_i q_i,
        f_i R_i e3 = m_i (x_i'' + g e3) + R_L lam_i,
    with lam = pinv(G_L) wrench + internal, and the wrench that the payload's own
    equations of motion ask for. Returns the state and input at the expansion point,
    and the pieces the tracking controller needs."""
    n = p.n
    Gm = grasp_map(p.r)
    Rt = np.transpose(R_L, (0, 2, 1))
    W = jmul(Rt[:K - 1], jder(R_L), np.matmul)                 # hat(Omega_L)
    Om = np.array([vee(0.5 * (w - w.T)) for w in W])           # K - 1
    Om_dot = jder(Om)                                           # K - 2
    acc = jder(x_L, 2)                                          # K - 2
    acc[0] = acc[0] + p.g * E3
    K2 = K - 2
    force_b = p.m_L * jmul(Rt[:K2], acc, np.matmul)             # R_L^T m_L (x_L'' + g e3)
    JOm = Om[:K2] @ p.J_L.T
    torque = Om_dot @ p.J_L.T + jcross(Om[:K2], JOm)
    wrench = np.concatenate([force_b, torque], axis=1)          # (K2, 6)
    lam = wrench @ np.linalg.pinv(Gm).T + internal[:K2]         # (K2, 3n)
    psi = np.zeros((n, K)) if psi is None else psi
    out = dict(x_L=x_L[0], v_L=derivs(x_L)[1], a_L=derivs(x_L)[2], R_L=R_L[0], Omega_L=Om[0],
               Omega_L_dot=Om_dot[0], wrench=wrench[0], lam=lam[0].reshape(n, 3))
    q, om, om_dot, T, f, R, Omega, Omega_dot, tau, xq, Fv = ([] for _ in range(11))
    for i in range(n):
        pull = jmul(R_L[:K2], lam[:, 3 * i:3 * i + 3], np.matmul)   # R_L lam_i = -T_i q_i
        Ti = jnorm(pull)
        qi = -jdiv(pull, Ti)
        ri = np.zeros((K2, 3))
        ri[0] = p.r[i]
        xi = x_L[:K2] + jmul(R_L[:K2], ri, np.matmul) - p.l[i] * qi
        ai = jder(xi, 2)                                         # K - 4
        ai[0] = ai[0] + p.g * E3
        F = p.m[i] * ai + pull[:K - 4]
        att = attitude_from_force(F, psi[i][:K - 4], p.J[i])
        qd = derivs(qi)
        q.append(qd[0]), om.append(np.cross(qd[0], qd[1])), om_dot.append(np.cross(qd[0], qd[2]))
        T.append(Ti[0]), xq.append(xi[0]), Fv.append(F[0])
        for store, key in ((f, "f"), (R, "R"), (Omega, "Omega"), (Omega_dot, "Omega_dot"),
                           (tau, "tau")):
            store.append(att[key])
    out.update(q=np.array(q), omega=np.array(om), omega_dot=np.array(om_dot), T=np.array(T),
               f=np.array(f), R=np.array(R), Omega=np.array(Omega),
               Omega_dot=np.array(Omega_dot), tau=np.array(tau), x_Q=np.array(xq), F=np.array(Fv))
    return out


def plan_point(p, plan, t, spread=True):
    """The reference at time t: flat map of the team along the plan."""
    K = 7
    internal = np.outer(plan.spread(t, K) if spread else np.zeros(K), spread_direction(p.r))
    return team_flat(p, plan.x_L(t, K), rotz_jet(plan.psi_L(t, K)), internal, K=K)


def state_of(robot, ref):
    return robot.State(x_L=ref["x_L"], v_L=ref["v_L"], R_L=ref["R_L"], Omega_L=ref["Omega_L"],
                       q=ref["q"], omega=ref["omega"], R=ref["R"], Omega=ref["Omega"])


# =====================================================================
# 4. Tracking: the cascade
# =====================================================================
def omega_dot_des(q, om, qd, omd, omd_dot, k_q, k_omega):
    """The angular acceleration of eq6-law-s2."""
    return (tangent(q, omd_dot) - (q @ omd) * np.cross(om, q)
            - k_q * e_q(q, qd) - k_omega * e_omega(q, om, omd))


def desired_attitude(F, psi=0.0):
    """eq6-Rd: the attitude whose third axis is along F, with heading psi."""
    b3 = unit(F)
    b2 = unit(np.cross(b3, np.array([np.cos(psi), np.sin(psi), 0.0])))
    return np.column_stack([np.cross(b2, b3), b2, b3])


def safety_filter(robot, state, Fd, params, r_min=R_MIN, t_min=T_MIN, a1=ALPHA1, a2=ALPHA2,
                  separation=True, tension=True, F_ext=None):
    """Smallest change of the thrust vectors (n, 3) that satisfies eq8-rd2 for every
    barrier H_ij = |x_i - x_j|^2 - r_min^2 and keeps every tension above t_min,
    with the model's accelerations and tensions, which are affine in the thrust vectors.
    Returns (F, info)."""
    n = robot.n
    a0, Da, T0, DT = affine_maps(robot, state, params)
    if F_ext is not None:                                  # an external force the filter is told about
        a0, T0 = a0 + Da @ F_ext.ravel(), T0 + DT @ F_ext.ravel()
    x, v = vehicle_kinematics(robot, state, None, params)
    rows, rhs, H_all = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d, dv = x[i] - x[j], v[i] - v[j]
            H, H_dot = d @ d - r_min ** 2, 2 * d @ dv
            H_all.append(H)
            if separation:
                g = 2 * d @ (Da[3 * i:3 * i + 3] - Da[3 * j:3 * j + 3])
                c0 = 2 * dv @ dv + 2 * d @ (a0[3 * i:3 * i + 3] - a0[3 * j:3 * j + 3])
                rows.append(-g), rhs.append(c0 + (a1 + a2) * H_dot + a1 * a2 * H)
    if tension:
        for i in range(n):
            rows.append(-DT[i]), rhs.append(T0[i] - t_min)
    info = dict(H=np.array(H_all), active=False, feasible=True)
    if not rows:
        return Fd, info
    A, b = np.array(rows), np.array(rhs)
    z0 = Fd.ravel()
    if np.all(A @ z0 <= b):
        return Fd, info
    z = ch08.solve_qp(np.eye(3 * n), -z0, A, b)
    info["active"] = True
    if z is None:
        info["feasible"] = False
        return Fd, info
    return z.reshape(n, 3), info


_REF_CACHE = {}


class Cascade:
    """Payload wrench -> cable forces -> cable directions -> thrust vectors -> attitudes."""

    def __init__(self, robot, plan, gains=GAINS, params=None, feedforward=True, optimize=True,
                 separation=True, tension=True, spread=True, d_plan=D_PLAN,
                 f_max=F_MAX, kappa=1.0, ff_attitude=True, rates=(ALPHA1, ALPHA2), told=None):
        self.robot, self.plan, self.g = robot, plan, dict(gains)
        self.p = params or robot.params
        self.feedforward, self.optimize, self.spread = feedforward, optimize, spread
        self.separation, self.tension, self.d_plan = separation, tension, d_plan
        self.f_max, self.kappa, self.ff_attitude = f_max, kappa, ff_attitude
        self.rates, self.told = rates, told                 # told(t): external forces the filter knows
        self.h, self._qd_prev = H_STEP, None
        self._ref = _REF_CACHE.setdefault((id(plan), id(self.p), spread), {})
        self.log = {}

    def reference(self, t):
        key = int(round(t / 1e-4))
        if key not in self._ref:
            self._ref[key] = plan_point(self.p, self.plan, t, spread=self.spread)
        return self._ref[key]

    def thrust_vectors(self, t, state):
        """Steps 1 to 4 of the cascade: the thrust vectors before the safety filter."""
        p, g, ref = self.p, self.g, self.reference(t)
        n = self.robot.n
        ff = 1.0 if self.feedforward else 0.0
        # 1. the wrench the payload should feel: eq6-Fd and eq6-law, for the payload
        e_x, e_v = state.x_L - ref["x_L"], state.v_L - ref["v_L"]
        F_L = p.m_L * (-g["k_x"] * e_x - g["k_v"] * e_v + p.g * E3 + ff * ref["a_L"])
        k_R, k_Om = g["k_RL"] * p.J_L, g["k_OmegaL"] * p.J_L
        Rd, Omd, Omd_dot = ref["R_L"], ff * ref["Omega_L"], ff * ref["Omega_L_dot"]
        R_L, Om_L = state.R_L, state.Omega_L
        tau_L = (-k_R @ e_R(R_L, Rd) - k_Om @ (Om_L - R_L.T @ Rd @ Omd)
                 + np.cross(Om_L, p.J_L @ Om_L)
                 - p.J_L @ (np.cross(Om_L, R_L.T @ Rd @ Omd) - R_L.T @ Rd @ Omd_dot))
        wrench = np.concatenate([R_L.T @ F_L, tau_L])
        # 2. the forces of the cables on the payload
        if self.optimize:
            lam, act = distribute(p.r, wrench, base=ref["lam"], l=p.l, d_min=self.d_plan,
                                  t_min=T_MIN)
            if lam is None:
                lam, act = distribute(p.r, wrench, base=ref["lam"])[0], -1
        else:
            lam, act = distribute(p.r, wrench)
        # 3. cable directions and the angular acceleration asked of each: eq6-law-s2
        pull = lam @ R_L.T                                    # R_L lam_i, world frame
        a_c = F_L / p.m_L - p.g * E3
        Om_dot_c = np.linalg.solve(p.J_L, tau_L - np.cross(Om_L, p.J_L @ Om_L))
        F = np.zeros((n, 3))
        qd_all = np.zeros((n, 3))
        for i in range(n):
            q, om = state.q[i], state.omega[i]
            qd = -unit(pull[i])
            qd_all[i] = qd
            omd, omd_dot = ff * ref["omega"][i], ff * ref["omega_dot"][i]
            if act != 0 and self.feedforward:
                # a constrained q_d does not move as the plan says: difference it instead
                prev = self._qd_prev[i] if self._qd_prev is not None else qd
                omd, omd_dot = np.cross(prev, qd - prev) / self.h, np.zeros(3)
            wd = omega_dot_des(q, om, qd, tangent(qd, omd), omd_dot, g["k_q"], g["k_omega"])
            a_att = a_c + R_L @ (np.cross(Om_dot_c, p.r[i])
                                 + np.cross(Om_L, np.cross(Om_L, p.r[i])))
            # 4. thrust vector: along the cable it delivers the force, across it turns the cable
            F[i] = (p.m[i] * (a_att + p.g * E3 + p.l[i] * (om @ om) * q
                              - p.l[i] * np.cross(wd, q)) + (q @ pull[i]) * q)
        self._qd_prev = qd_all
        self.log = dict(wrench=wrench, lam=lam, active=act, qd=qd_all, F_d=F.copy())
        return F, ref

    def __call__(self, t, state):
        n = self.robot.n
        F, ref = self.thrust_vectors(t, state)
        # 5. safety filter on the thrust vectors
        F, info = safety_filter(self.robot, state, F, self.p, separation=self.separation,
                                tension=self.tension, a1=self.rates[0], a2=self.rates[1],
                                F_ext=None if self.told is None else self.told(t))
        self.log.update(H=info["H"], filtered=info["active"], feasible=info["feasible"], F=F)
        # 6. each vehicle: eq6-Rd and eq6-law
        f, tau = np.zeros(n), np.zeros((n, 3))
        ffa = 1.0 if (self.feedforward and self.ff_attitude) else 0.0
        for i in range(n):
            R, Om = state.R[i], state.Omega[i]
            f[i] = np.clip(F[i] @ R @ E3, 0.0, self.f_max)
            Rd = desired_attitude(F[i])
            tau[i] = attitude_tracking_torque(R, Om, Rd, ffa * ref["Omega"][i],
                                              ffa * ref["Omega_dot"][i], self.p.J[i],
                                              self.kappa ** 2 * self.g["k_R"],
                                              self.kappa * self.g["k_Omega"])
        return f, tau


def fly(controller, plan, n=N_TEAM, t_end=T_END, h=H_STEP, true_params=None, push=None,
        eff=None, state0=None, model_params=None):
    """One mission. The plant is the library's team with the true parameters, a push
    and a loss of thrust. Returns a dict of time series."""
    p_model = model_params or controller.p
    p_true = true_params or p_model
    plant = Plant(n, p_true)
    if eff is not None:
        plant.eff = np.asarray(eff, float)
    state = state0 if state0 is not None else state_of(plant, plan_point(p_model, plan, 0.0,
                                                                         controller.spread))
    steps = int(round(t_end / h))
    keys = ("t", "x_L", "x_d", "R_L", "R_d", "x_Q", "x_Qd", "q", "T", "f", "tau", "H", "filtered",
            "feasible", "dist", "active", "resid", "tilt", "psi_err", "R", "dF")
    out = {k: [] for k in keys}
    for k in range(steps + 1):
        t = k * h
        plant.ext[:] = 0.0
        if push is not None and push["t0"] <= t < push["t1"]:
            plant.ext[push["vehicle"]] = push["force"]
        u = controller(t, state)
        ref = controller.reference(t)
        x = plant.quadrotor_positions(state)
        F_true = np.array([e * fi * Ri @ E3
                           for e, fi, Ri in zip(plant.eff, u[0], state.R)]) + plant.ext
        _, _, T0, DT = affine_maps(plant, state, p_true)
        Tn = T0 + DT @ F_true.ravel()
        d = [np.linalg.norm(x[i] - x[j]) for i in range(n) for j in range(i + 1, n)]
        log = controller.log
        vals = dict(t=t, x_L=state.x_L, x_d=ref["x_L"], R_L=state.R_L, R_d=ref["R_L"], x_Q=x,
                    x_Qd=ref["x_Q"], q=state.q, T=Tn, f=u[0], tau=u[1],
                    H=np.array(d) ** 2 - R_MIN ** 2, filtered=log.get("filtered", False),
                    feasible=log.get("feasible", True), dist=np.array(d),
                    active=log.get("active", 0), resid=plant.on_manifold(state),
                    tilt=np.degrees(np.arccos(np.clip(state.R_L[2, 2], -1, 1))),
                    psi_err=np.degrees(np.linalg.norm(log_so3(ref["R_L"].T @ state.R_L))),
                    R=state.R, dF=np.linalg.norm(log.get("F", 0) - log.get("F_d", 0))
                    if "F" in log else 0.0)
        for key in keys:
            out[key].append(vals[key])
        if not np.all(np.isfinite(state.x_L)) or np.linalg.norm(state.x_L - ref["x_L"]) > 20:
            break
        if k < steps:
            state = plant.step(state, u, h)
    out = {k: np.array(v) for k, v in out.items()}
    out["done"] = len(out["t"]) == steps + 1
    return out


def summarize(o, t_from=1.0):
    """The numbers of one mission that the chapter quotes."""
    err = np.linalg.norm(o["x_L"] - o["x_d"], axis=1)
    m = o["t"] >= t_from
    walls = GAP_W / 2 - ARM
    near = np.abs(o["x_Q"][:, :, 0] - GAP_X) < ARM
    side = np.where(near, np.abs(o["x_Q"][:, :, 1]), 0.0)
    return dict(done=bool(o["done"]), rms=float(np.sqrt(np.mean(err[m] ** 2))),
                peak=float(err.max()), final=float(err[-1]), tilt_end=float(o["tilt"][-1]),
                tilt_max=float(o["tilt"].max()), att_max=float(o["psi_err"].max()),
                T_min=float(o["T"].min()), T_max=float(o["T"].max()),
                d_min=float(o["dist"].min()), H_min=float(o["H"].min()),
                f_max=float(o["f"].max()), f_min=float(o["f"].min()),
                tau_max=float(np.abs(o["tau"]).max()),
                filtered=float(np.mean(o["filtered"]) * (o["t"][-1] + H_STEP)),
                infeasible=int(np.sum(~o["feasible"].astype(bool))),
                wall=float(walls - side.max()), resid=float(o["resid"].max()))


# =====================================================================
# 5. Variation-based linearization of any robot of the library
# =====================================================================
def slots(robot):
    """The factors of a robot, one entry per copy, ordered as geomech orders its
    variations: flat factors, then directions, then attitudes."""
    out = []
    for kind in ("R", "S2", "SO3"):
        for f in robot.factors:
            if f.kind == kind:
                out += [(kind, f.config, f.velocity, i if f.stacked else None)
                        for i in range(f.count)]
    return out


def _get(state, name, i):
    v = getattr(state, name)
    return v if i is None else v[i]


def _set(fields, state, name, i, value):
    if i is None:
        fields[name] = value
    else:
        fields.setdefault(name, np.array(getattr(state, name), float).copy())[i] = value


def perturb(robot, state, s):
    """The state reached from `state` by the linear state s = (variations, velocity
    variations): x + dx, exp(hat(xi)) q, R exp(hat(eta)), and v + dv."""
    sl = slots(robot)
    m = len(sl)
    fields = {}
    for k, (kind, c, v, i) in enumerate(sl):
        d, dv = s[3 * k:3 * k + 3], s[3 * (m + k):3 * (m + k) + 3]
        if kind == "R":
            _set(fields, state, c, i, _get(state, c, i) + d)
            _set(fields, state, v, i, _get(state, v, i) + dv)
        elif kind == "S2":
            q = expm_so3(d) @ _get(state, c, i)
            _set(fields, state, c, i, q)
            _set(fields, state, v, i, tangent(q, _get(state, v, i) + dv))
        else:
            _set(fields, state, c, i, _get(state, c, i) @ expm_so3(d))
            _set(fields, state, v, i, _get(state, v, i) + dv)
    return replace(state, **fields)


def error_state(robot, state, ref):
    """The linear state that the gain is fed: e_x, e_q = q_d x q, e_R, and the plain
    differences of the velocities (tbl7-dictionary)."""
    sl = slots(robot)
    m = len(sl)
    s = np.zeros(6 * m)
    for k, (kind, c, v, i) in enumerate(sl):
        a, d = _get(state, c, i), _get(ref, c, i)
        s[3 * k:3 * k + 3] = a - d if kind == "R" else (e_q(a, d) if kind == "S2" else e_R(a, d))
        s[3 * (m + k):3 * (m + k) + 3] = _get(state, v, i) - _get(ref, v, i)
    return s


def acc_vector(robot, state, u, params=None):
    """Accelerations in slot order, three numbers per slot."""
    acc = dict(zip([f.config for f in robot.factors], robot.dynamics(state, u, params)))
    return np.concatenate([np.ravel(acc[c] if i is None else acc[c][i])
                           for _, c, _, i in slots(robot)])


def flat_u(u):
    return np.concatenate([np.ravel(u[0]), np.ravel(u[1])])


def unflat_u(z, n):
    return z[:n], z[n:].reshape(n, 3)


def constraint_matrix(robot, state):
    """C of eq7-s2-constraint, two rows for each direction."""
    sl = slots(robot)
    m = len(sl)
    rows = []
    for k, (kind, c, v, i) in enumerate(sl):
        if kind == "S2":
            q, w = _get(state, c, i), _get(state, v, i)
            r1, r2 = np.zeros(6 * m), np.zeros(6 * m)
            r1[3 * k:3 * k + 3] = q
            r2[3 * k:3 * k + 3] = np.cross(q, w)
            r2[3 * (m + k):3 * (m + k) + 3] = q
            rows += [r1, r2]
    return np.array(rows) if rows else np.zeros((0, 6 * m))


def constraint_basis(robot, state):
    C = constraint_matrix(robot, state)
    if not len(C):
        return np.eye(C.shape[1])
    _, _, Vt = np.linalg.svd(C)
    return Vt[len(C):].T


def linearize_fd(robot, state, u, eps=1e-6, params=None):
    """Variation-based linearization about (state, u), by differences along variations.

    Returns (A, B, Z): s_dot = A s + B du for s in the constraint subspace ker C,
    whose orthonormal basis is Z. The kinematic rows are the closed forms
    eq7-xi-dot (directions) and eq7-var-so3 (attitudes). The dynamic rows are centered
    differences of the library's accelerations along perturb(state, +-eps z).
    A is defined on ker C and set to zero on its complement: A = (A Z) Z^T."""
    sl = slots(robot)
    m = len(sl)
    N = 6 * m
    Z = constraint_basis(robot, state)
    AZ = np.zeros((N, Z.shape[1]))
    for j, z in enumerate(Z.T):
        for k, (kind, c, v, i) in enumerate(sl):
            d, dv = z[3 * k:3 * k + 3], z[3 * (m + k):3 * (m + k) + 3]
            if kind == "R":
                AZ[3 * k:3 * k + 3, j] = dv
            elif kind == "S2":
                q, w = _get(state, c, i), _get(state, v, i)
                AZ[3 * k:3 * k + 3, j] = np.cross(w, d) + tangent(q, dv)
            else:
                AZ[3 * k:3 * k + 3, j] = dv - np.cross(_get(state, v, i), d)
        up = acc_vector(robot, perturb(robot, state, eps * z), u, params)
        dn = acc_vector(robot, perturb(robot, state, -eps * z), u, params)
        AZ[3 * m:, j] = (up - dn) / (2 * eps)
    uz = flat_u(u)
    n = len(u[0])
    B = np.zeros((N, len(uz)))
    for j in range(len(uz)):
        e = np.zeros(len(uz))
        e[j] = 1.0
        B[3 * m:, j] = (acc_vector(robot, state, unflat_u(uz + e, n), params)
                        - acc_vector(robot, state, unflat_u(uz - e, n), params)) / 2.0
    return AZ @ Z.T, B, Z


def ctrb_rank(A, B, tol=1e-8):
    """Rank of the controllability matrix, built on an orthonormalized Krylov basis."""
    Qb, _ = np.linalg.qr(B)
    basis = Qb
    block = Qb
    for _ in range(A.shape[0]):
        block = A @ block
        block = block - basis @ (basis.T @ block)
        block = block - basis @ (basis.T @ block)
        U, sv, _ = np.linalg.svd(block, full_matrices=False)
        keep = sv > tol * max(1.0, sv[0])
        if not np.any(keep):
            break
        block = U[:, keep]
        basis = np.hstack([basis, block])
    return basis.shape[1]


def lqr_equilibrium(robot, state, u, Qw, Rw, params=None):
    """Infinite-horizon LQR at an equilibrium, designed on ker C (def7-constrained).
    Returns (K for the full linear state, reduced A, reduced B, Z)."""
    A, B, Z = linearize_fd(robot, state, u, params=params)
    Ar, Br = Z.T @ A @ Z, Z.T @ B
    P = solve_continuous_are(Ar, Br, Z.T @ Qw @ Z, Rw)
    K = np.linalg.solve(Rw, Br.T @ P) @ Z.T
    return K, Ar, Br, Z


def lqr_finite(robot, refs, h, Qw, Rw, Pf=None, params=None):
    """Finite-horizon LQR about a reference given at knots h apart (eq7-riccati,
    integrated backward with the sampled system of each knot). refs is a list of
    (state, u). Returns the list of gains K_k for the full linear state."""
    P = Qw.copy() if Pf is None else Pf.copy()
    gains = [None] * len(refs)
    for k in range(len(refs) - 1, -1, -1):
        A, B, Z = linearize_fd(robot, refs[k][0], refs[k][1], params=params)
        Ad = np.eye(len(A)) + h * A + 0.5 * h * h * A @ A
        Bd = h * B + 0.5 * h * h * A @ B
        S = Rw * h + Bd.T @ P @ Bd
        K = np.linalg.solve(S, Bd.T @ P @ Ad)
        P = Qw * h + Ad.T @ P @ (Ad - Bd @ K)
        P = 0.5 * (P + P.T)
        gains[k] = K @ Z @ Z.T
    return gains


def weights(robot, w_config, w_velocity):
    """Diagonal Q for the linear state: one weight per kind of factor."""
    sl = slots(robot)
    return np.diag(np.concatenate([np.repeat([w_config[k] for k, *_ in sl], 3),
                                   np.repeat([w_velocity[k] for k, *_ in sl], 3)]))


def geomech_linear(robot, tag):
    """geomech's own variation-based linearization of the robot's Lagrangian."""
    from geomech import check_eom, compute_eom, linearize, to_standard_form
    from geomech.codegen import StateLayout, collect_params, load_module, module_source
    L, dW, V, inputs = robot.lagrangian()
    t0 = time.time()
    eom = compute_eom(L, dW, V)
    err = float(check_eom(L, dW, V, eom, inputs=inputs))
    took = time.time() - t0
    lin = linearize(eom, V, inputs)
    layout = StateLayout(V, inputs, collect_params([L, dW], V, inputs))
    mod = load_module(module_source(lin, layout), tag)
    # the coefficients themselves, E = M v_dot + b + G u (eq5-standard), for the term of eq18-projection
    mod.sf = load_module(module_source(to_standard_form(eom, V, inputs), layout, lagrangian=L), tag + "_sf")
    return mod, layout, err, took, len(str(eom))


def geomech_AB(robot, mod, layout, state, u, gp, names):
    """M, A and B of geomech's linearization M s_dot = A s + B du at a state of the
    library, in the slot order of `slots`. M is singular along each q (the equation
    of a direction is a projected one), so compare M A_ours with A, not inverses.
    Also returns, for each slot, the coefficient E = M v_dot + b + G u of eq5-standard
    at this state, which is zero up to its part along q."""
    pack, acc, rows = {}, np.zeros(mod.LAYOUT["n_accel"]), []
    a = acc_vector(robot, state, u)
    accel = {e["name"]: e["slice"] for e in mod.LAYOUT["acceleration"]}
    for k, (kind, c, v, i) in enumerate(slots(robot)):
        name = names(c, i)
        vel = {"R": f"{name}_dot", "S2": f"omega_{name}", "SO3": f"Omega_{name}"}[kind]
        pack[name], pack[vel] = _get(state, c, i), _get(state, v, i)
        lo, hi = accel[{"R": f"{name}_ddot", "S2": f"omega_{name}_dot",
                        "SO3": f"Omega_{name}_dot"}[kind]]
        acc[lo:hi] = a[3 * k:3 * k + 3]
        rows.append((lo, hi))
    x = layout.pack(**pack)
    n = len(u[0])
    ug = np.concatenate([np.r_[u[0][i], u[1][i]] for i in range(n)])
    perm = [4 * i for i in range(n)] + [4 * i + 1 + j for i in range(n) for j in range(3)]
    E = mod.sf.mass(x, gp) @ acc + mod.sf.bias(x, gp) + mod.sf.input_matrix(x, gp) @ ug
    return (mod.M_lin(x, acc, ug, gp), mod.A_lin(x, acc, ug, gp), mod.B_lin(x, acc, ug, gp)[:, perm],
            [E[lo:hi] for lo, hi in rows])


# =====================================================================
# 6. Feedback from the linearization
# =====================================================================
class LinearFeedback:
    """u = u_d - K s (def7-feedback, for every factor at once)."""

    def __init__(self, robot, ref_fn, gain_fn, f_max=F_MAX):
        self.robot, self.ref_fn, self.gain_fn, self.f_max = robot, ref_fn, gain_fn, f_max

    def __call__(self, t, state):
        ref_state, u_d = self.ref_fn(t)
        n = len(u_d[0])
        z = flat_u(u_d) - self.gain_fn(t) @ error_state(self.robot, state, ref_state)
        f, tau = unflat_u(z, n)
        return np.clip(f, 0.0, self.f_max), tau


def displaced(robot, state, dx=(0.0, 0.0, 0.0), yaw=0.0, swing=0.0):
    """A hover state moved by dx, with the payload turned by yaw and cable 0 swung."""
    q = state.q.copy()
    q[0] = expm_so3(swing * E2) @ q[0]
    return replace(state, x_L=state.x_L + np.array(dx), R_L=state.R_L @ expm_so3(yaw * E3), q=q)


def recover(robot, controller, state0, target, t_end=6.0, h=H_STEP, tol=0.02):
    """Fly from state0 to a hover. Returns settle time of the payload position, peak
    thrust, smallest tension, largest payload attitude error."""
    state, t_settle = state0, 0.0
    f_peak, T_low, att = 0.0, np.inf, 0.0
    for k in range(int(round(t_end / h))):
        t = k * h
        u = controller(t, state)
        e = np.linalg.norm(state.x_L - target.x_L)
        if not np.isfinite(e) or e > 50:
            return dict(settle=np.inf, f_peak=f_peak, T_low=T_low, att=att, final=np.inf)
        if e > tol:
            t_settle = t + h
        f_peak, T_low = max(f_peak, float(np.max(u[0]))), min(T_low, float(np.min(tensions(robot, state, u))))
        att = max(att, float(np.degrees(np.linalg.norm(log_so3(target.R_L.T @ state.R_L)))))
        state = robot.step(state, u, h)
    return dict(settle=t_settle, f_peak=f_peak, T_low=T_low, att=att,
                final=float(np.linalg.norm(state.x_L - target.x_L)))


def team_lqr_study():
    """Hover LQR of the team against the cascade."""
    rb = team()
    p = rb.params
    hover = Plan(length=0.0, drop=0.0, turn=0.0, s_gap=S_WIDE)
    ref = plan_point(p, hover, 0.0)
    s0, u0 = state_of(rb, ref), (ref["f"], ref["tau"])
    Qw = weights(rb, {"R": 60.0, "S2": 20.0, "SO3": 10.0}, {"R": 15.0, "S2": 2.0, "SO3": 1.0})
    Rw = np.diag(np.r_[np.full(3, 0.05), np.full(9, 20.0)])
    K, Ar, Br, Z = lqr_equilibrium(rb, s0, u0, Qw, Rw)
    poles = np.linalg.eigvals(Ar - Br @ K @ Z)
    out = dict(dim=Ar.shape[0], rank=ctrb_rank(Ar, Br), open_max=float(np.linalg.eigvals(Ar).real.max()),
               slowest=float(-poles.real.max()), fastest=float(-poles.real.min()),
               K_shape=K.shape, K_norm=float(np.abs(K).max()),
               K_f_x=float(np.abs(K[:3, :3]).max()), cases={})
    lqr = LinearFeedback(rb, lambda t: (s0, u0), lambda t: K)
    for name, kw in (("offset 0.3 m", dict(dx=(0.0, 0.3, 0.0))), ("offset 1 m", dict(dx=(0.0, 1.0, 0.0))),
                     ("offset 3 m", dict(dx=(0.0, 3.0, 0.0))),
                     ("yaw 30 deg", dict(yaw=np.radians(30.0))), ("yaw 90 deg", dict(yaw=np.radians(90.0))),
                     ("cable swung 25 deg", dict(swing=np.radians(25.0)))):
        start = displaced(rb, s0, **kw)
        casc = Cascade(rb, hover, separation=False, tension=False, d_plan=None)
        out["cases"][name] = dict(lqr=recover(rb, lqr, start, s0), cascade=recover(rb, casc, start, s0))
    return out


# =====================================================================
# 7. The flexible cable between two quadrotors
# =====================================================================
CHAIN_N = 5


def chain_robot():
    return robots.get("quadrotor_cable", n=CHAIN_N, attach=(0, CHAIN_N))


_CHAIN_REF = {}


def chain_reference(rb, t, moving=True, K=None):
    key = (id(rb), round(t * 1e5), moving, K)
    if key not in _CHAIN_REF:
        _CHAIN_REF[key] = _chain_reference(rb, t, moving, K)
    return _CHAIN_REF[key]


def _chain_reference(rb, t, moving=True, K=None):
    """Flat map of thm10-chain for a quadrotor at each end: the flat output is the
    position of joint n, the two headings and the tension vector of link n."""
    p = rb.params
    n = p.n
    K = K or 2 * n + 5
    mass = rb.joint_masses(p)
    xn = np.zeros((K, 3))
    xn[0] = (0.45, 0.0, 1.5)
    if moving:                                  # a slow figure-eight for joint n
        for (amp, rate, axis, phase) in ((0.5, 0.7, 1, 0.0), (0.25, 1.4, 0, 0.0), (0.1, 0.7, 2, 0.5)):
            for k in range(K):
                xn[k, axis] += amp * rate ** k * math.sin(rate * t + phase + k * math.pi / 2) / math.factorial(k)
    weight = np.sum(p.m[1:n]) * p.g
    pull = np.zeros((K, 3))
    pull[0] = (0.6, 0.0, weight / 2)            # T_n q_n: half the cable's weight, and 0.6 N outward
    psi = np.zeros(K)
    o = chain_flat(xn, [psi, psi], mass, p.l, [p.J[0], p.J[1]], g=p.g, attach=(0, n), pulls={n: pull})
    # chain_flat lists the quadrotors in the order of `attach`
    state = rb.State(x_0=o["x_0"], v_0=o["v_0"], q=o["q"], omega=o["omega"], R=o["R"], Omega=o["Omega"])
    return state, (o["f"], o["tau"]), o


def chain_perturbed(rb, state, rng, angle=np.radians(12.0), dx=0.08):
    q = np.array([expm_so3(angle * unit(rng.normal(size=3))) @ qi for qi in state.q])
    om = np.array([tangent(qi, wi) for qi, wi in zip(q, state.omega)])
    return replace(state, x_0=state.x_0 + dx * unit(rng.normal(size=3)), q=q, omega=om)


def chain_fly(rb, controller, state0, ref_fn, t_end, h=H_STEP, thin=4):
    state, rows = state0, []
    for k in range(int(round(t_end / h)) + 1):
        t = k * h
        ref = ref_fn(t)[0]
        x, xr = rb.positions(state), rb.positions(ref)
        psi = float(np.max(1.0 - np.sum(state.q * ref.q, axis=1)))
        u = controller(t, state)
        if not np.all(np.isfinite(x)) or np.abs(x - xr).max() > 30:
            break
        if k % thin == 0:
            rows.append((t, x, xr, psi, np.linalg.norm(x - xr, axis=1).max(), u[0].copy(), state.R.copy(),
                         rb.on_manifold(state)))
        state = rb.step(state, u, h)
    keys = ("t", "x", "x_d", "Psi_q", "err", "f", "R", "resid")
    out = {k: np.array([r[i] for r in rows]) for i, k in enumerate(keys)}
    out["done"] = abs(out["t"][-1] - t_end) < 1e-9
    return out


def chain_study(t_end=8.0, knot=0.02):
    rb = chain_robot()
    p = rb.params
    rng = np.random.default_rng(19)
    out = dict(dim=rb.dim, n=p.n, inputs=4 * rb.n_Q)
    # the flat map satisfies the model
    worst, T_low = 0.0, np.inf
    for t in np.linspace(0.0, t_end, 9):
        s, u, o = chain_reference(rb, t)
        acc = rb.dynamics(s, u)
        worst = max(worst, np.abs(acc[0] - o["a_0"]).max(), np.abs(acc[1] - o["omega_dot"]).max(),
                    np.abs(acc[2] - o["Omega_dot"]).max())
        T_low = min(T_low, o["T"].min())
    out.update(flat_residual=float(worst), T_low=float(T_low))
    # hanging equilibrium
    s_eq, u_eq, o_eq = chain_reference(rb, 0.0, moving=False)
    out["eq_residual"] = float(max(np.abs(a).max() for a in rb.dynamics(s_eq, u_eq)))
    x_eq = rb.positions(s_eq)
    out.update(span=float(np.linalg.norm(x_eq[-1] - x_eq[0])), sag=float(x_eq[[0, -1], 2].mean() - x_eq[:, 2].min()),
               f_eq=[float(v) for v in u_eq[0]], T_eq=[float(v) for v in o_eq["T"]])
    Qw = weights(rb, {"R": 50.0, "S2": 20.0, "SO3": 10.0}, {"R": 10.0, "S2": 2.0, "SO3": 1.0})
    Rw = np.diag(np.r_[np.full(2, 0.05), np.full(6, 20.0)])
    K, Ar, Br, Z = lqr_equilibrium(rb, s_eq, u_eq, Qw, Rw)
    poles = np.linalg.eigvals(Ar - Br @ K @ Z)
    open_poles = np.linalg.eigvals(Ar)
    out.update(reduced=Ar.shape[0], full=K.shape[1], rank=ctrb_rank(Ar, Br), K_shape=K.shape,
               K_max=float(np.abs(K).max()), slowest=float(-poles.real.max()),
               fastest=float(-poles.real.min()), open_max=float(open_poles.real.max()),
               swing_modes=sorted(set(np.round(np.abs(open_poles.imag[np.abs(open_poles.imag) > 0.1]), 2)))[:3])
    start = chain_perturbed(rb, s_eq, rng)
    hold = chain_fly(rb, LinearFeedback(rb, lambda t: (s_eq, u_eq), lambda t: K), start, lambda t: (s_eq, u_eq), 6.0)
    free = chain_fly(rb, lambda t, s: u_eq, start, lambda t: (s_eq, u_eq), 6.0)
    out.update(hold_err0=float(hold["err"][0]), hold_err_end=float(hold["err"][-1]),
               hold_settle=float(hold["t"][np.nonzero(hold["err"] > 0.01)[0][-1]] if np.any(hold["err"] > 0.01) else 0.0),
               free_err_end=float(free["err"][-1]), free_done=bool(free["done"]),
               free_err_max=float(free["err"].max()))
    # moving reference: finite-horizon LQR
    knots = int(round(t_end / knot)) + 1
    refs = [chain_reference(rb, k * knot)[:2] for k in range(knots)]
    gains = lqr_finite(rb, refs, knot, Qw, Rw)

    def ref_fn(t):
        return chain_reference(rb, t)[:2]

    start = chain_perturbed(rb, refs[0][0], rng)
    track = chain_fly(rb, LinearFeedback(rb, ref_fn, lambda t: gains[min(int(round(t / knot)), knots - 1)]),
                      start, ref_fn, t_end)
    open_loop = chain_fly(rb, lambda t, s: ref_fn(t)[1], start, ref_fn, t_end)
    late = track["t"] >= 3.0
    out.update(track_err0=float(track["err"][0]), track_rms_late=float(np.sqrt(np.mean(track["err"][late] ** 2))),
               track_err_end=float(track["err"][-1]), track_Psi_end=float(track["Psi_q"][-1]),
               track_f=[float(track["f"].min()), float(track["f"].max())],
               open_err_max=float(open_loop["err"].max()), open_done=bool(open_loop["done"]),
               open_t_end=float(open_loop["t"][-1]), resid=float(track["resid"].max()),
               K_max_moving=float(max(np.abs(Kk).max() for Kk in gains)))
    # the linearization along the moving reference predicts a perturbed simulation
    s0, u0 = refs[0]
    Z0 = constraint_basis(rb, s0)
    z = Z0 @ rng.normal(size=Z0.shape[1])
    preds = {}
    for eps in (1e-3, 1e-4):
        sp, sm, lin = perturb(rb, s0, eps * z), perturb(rb, s0, -eps * z), z.copy()
        hh, steps = 2e-3, 100
        for k in range(steps):
            t = k * hh
            ref_s, ref_u = ref_fn(t)
            A, B, _ = linearize_fd(rb, ref_s, ref_u)
            A2, _, _ = linearize_fd(rb, *ref_fn(t + hh / 2))
            A3, _, _ = linearize_fd(rb, *ref_fn(t + hh))
            k1 = A @ lin
            k2 = A2 @ (lin + hh / 2 * k1)
            k3 = A2 @ (lin + hh / 2 * k2)
            k4 = A3 @ (lin + hh * k3)
            lin = lin + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            sp = rb.step(sp, lambda tt, s: ref_fn(tt)[1], hh, t=t)
            sm = rb.step(sm, lambda tt, s: ref_fn(tt)[1], hh, t=t)
        ref_end = ref_fn(steps * hh)[0]
        fd = (error_state(rb, sp, ref_end) - error_state(rb, sm, ref_end)) / (2 * eps)
        preds[eps] = float(np.abs(fd - lin).max() / np.abs(lin).max())
    out["lin_vs_sim"] = preds
    out["series"] = dict(track={k: track[k].astype(np.float32) for k in ("t", "x", "x_d", "err", "f", "R")},
                         open={k: open_loop[k].astype(np.float32) for k in ("t", "x", "err")},
                         hold={k: hold[k].astype(np.float32) for k in ("t", "err")},
                         free={k: free[k].astype(np.float32) for k in ("t", "err")})
    return out


# =====================================================================
# 8. The missions
# =====================================================================
SAFE_PLAN = dict(s_gap=0.32)       # a plan that keeps 0.9 m between vehicles in the gap
MISSIONS = {
    # ---- ablation: the mission of the chapter with one thing changed ----
    "all": dict(),
    "min_norm": dict(cascade=dict(optimize=False)),
    "barrier_only": dict(cascade=dict(d_plan=None)),
    "pairs_only": dict(cascade=dict(separation=False)),
    "unprotected": dict(cascade=dict(d_plan=None, separation=False, tension=False)),
    "no_tension_floor": dict(cascade=dict(tension=False)),
    "no_ff": dict(cascade=dict(feedforward=False)),
    "weak": dict(eff=[1.0, 0.85, 1.0]),
    "long_cable": dict(true=dict(l=[1.1, 1.0, 1.0])),
    "push": dict(push=True),
    "push_told": dict(push=True, told=True),
    "slow_barrier": dict(cascade=dict(rates=(2.0, 3.0))),
    # ---- the cascade alone on a safe plan: loop speeds and feedforward ----
    "order_base": dict(plan=SAFE_PLAN, cascade=dict(separation=False, tension=False, d_plan=None)),
    "order_payload_fast": dict(plan=SAFE_PLAN, gains=dict(k_RL=16.0, k_OmegaL=7.2),
                               cascade=dict(separation=False, tension=False, d_plan=None)),
    "order_payload_fast_kappa": dict(plan=SAFE_PLAN, gains=dict(k_RL=16.0, k_OmegaL=7.2),
                                     cascade=dict(separation=False, tension=False, d_plan=None, kappa=1.5)),
    "order_payload_faster": dict(plan=SAFE_PLAN, gains=dict(k_RL=36.0, k_OmegaL=10.8),
                                 cascade=dict(separation=False, tension=False, d_plan=None)),
    "order_cable_fast": dict(plan=SAFE_PLAN, gains=dict(k_q=100.0, k_omega=16.0),
                             cascade=dict(separation=False, tension=False, d_plan=None)),
    "order_cable_faster": dict(plan=SAFE_PLAN, gains=dict(k_q=196.0, k_omega=22.4),
                               cascade=dict(separation=False, tension=False, d_plan=None)),
    "order_cable_faster_kappa": dict(plan=SAFE_PLAN, gains=dict(k_q=196.0, k_omega=22.4),
                                     cascade=dict(separation=False, tension=False, d_plan=None, kappa=2.0)),
    "order_no_ff": dict(plan=SAFE_PLAN, cascade=dict(separation=False, tension=False, d_plan=None,
                                                     feedforward=False)),
    "order_no_ff_kappa": dict(plan=SAFE_PLAN, cascade=dict(separation=False, tension=False, d_plan=None,
                                                           feedforward=False, kappa=2.0)),
    "order_no_att_ff": dict(plan=SAFE_PLAN, cascade=dict(separation=False, tension=False, d_plan=None,
                                                         ff_attitude=False)),
    # ---- scaling: the same mission with two and four vehicles ----
    "n2": dict(n=2),
    "n3_safe": dict(plan=SAFE_PLAN),
    "n4_safe": dict(n=4, plan=dict(s_gap=0.6)),
    "n4": dict(n=4),
}
SERIES = ("t", "x_L", "x_d", "R_L", "x_Q", "x_Qd", "T", "f", "H", "dist", "filtered", "tilt", "R", "active")


def run_mission(name):
    spec = MISSIONS[name]
    n = spec.get("n", N_TEAM)
    rb = team(n)
    p = rb.params
    plan = Plan(**spec.get("plan", {}))
    gains = dict(GAINS, **spec.get("gains", {}))
    kw = dict(spec.get("cascade", {}))
    push = dict(PUSH) if spec.get("push") else None
    if spec.get("told"):
        def told(t):
            F = np.zeros((n, 3))
            if push["t0"] <= t < push["t1"]:
                F[push["vehicle"]] = push["force"]
            return F
        kw["told"] = told
    true = None
    if "true" in spec:
        true = PayloadParams(n=n, **{k: np.array(v) for k, v in spec["true"].items()})
    t0 = time.time()
    ctrl = Cascade(rb, plan, gains=gains, **kw)
    o = fly(ctrl, plan, n=n, push=push, eff=spec.get("eff"), true_params=true)
    s = summarize(o)
    s["seconds"] = time.time() - t0
    s["steps"] = len(o["t"])
    s["plan_d_min"] = float(min(np.linalg.norm(a - b) for xq in o["x_Qd"] for i, a in enumerate(xq)
                                for b in xq[i + 1:]))
    s["plan_f_max"] = float(max(ctrl.reference(t)["f"].max() for t in o["t"][::10]))
    s["plan_T_min"] = float(min(ctrl.reference(t)["T"].min() for t in o["t"][::10]))
    s["qp_active_steps"] = int(np.sum(o["active"] > 0))
    gap = np.abs(o["x_d"][:, 0] - GAP_X) < 1.0
    s["rms_gap"] = float(np.sqrt(np.mean(np.sum((o["x_L"] - o["x_d"])[gap] ** 2, axis=1))))
    series = {k: o[k][::4].astype(np.float32) for k in SERIES}
    return name, s, series


# =====================================================================
# 9. Everything that takes time, computed once and cached
# =====================================================================
KEEP_SERIES = ("all", "unprotected", "barrier_only", "pairs_only", "push", "min_norm")


def _task(name):
    t0 = time.time()
    if name == "lqr":
        return name, team_lqr_study(), None, time.time() - t0
    if name == "chain":
        out = chain_study()
        return name, out, out.pop("series"), time.time() - t0
    _, s, series = run_mission(name)
    return name, s, series if name in KEEP_SERIES else None, time.time() - t0


def results(fresh=False, verbose=True):
    """Summaries and time series of every mission, of the LQR study and of the cable.
    Recomputed when this file changes."""
    import hashlib
    from multiprocessing import Pool
    source = Path(__file__).read_text().split("# 10. The checks")[0]      # the checks may change freely
    stamp = hashlib.sha1(source.encode()).hexdigest()
    if CACHE.exists() and not fresh:
        try:
            saved = pickle.loads(CACHE.read_bytes())
            if saved.get("stamp") == stamp:
                return saved
        except Exception:
            pass
    names = list(MISSIONS) + ["lqr", "chain"]
    out = dict(stamp=stamp, summary={}, series={}, wall={})
    with Pool(4) as pool:
        for name, s, series, took in pool.imap_unordered(_task, names):
            out["summary"][name], out["wall"][name] = s, took
            if series is not None:
                out["series"][name] = series
            if verbose:
                print(f"  computed {name} in {took:.0f} s", flush=True)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def from_chapter(path, names, ns):
    """Functions of an earlier chapter's script that runs everything at import."""
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
            found[node.name] = ns[node.name]
    return [found[n] for n in names]


def geomech_params(p):
    gp = {"e3": E3, "g": p.g, "m_L": p.m_L, "J_L": p.J_L}
    for i in range(p.n):
        gp.update({f"m{i + 1}": p.m[i], f"l{i + 1}": p.l[i], f"J{i + 1}": p.J[i], f"r{i + 1}": p.r[i]})
    return gp


def team_name(c, i):
    return c if i is None else f"{c}{i + 1}"


def geomech_agreement(rb, mod, layout, state, u):
    """Largest difference between geomech's linearization and ours on ker C.
    Returns (raw, tangent, reconciled): raw compares M A_ours Z with A_geomech Z;
    tangent projects the rows of each direction by I - q q^T; reconciled also removes,
    for each direction, the term -(xi x q)(q . E_xi) of eq18-projection: geomech varies
    the coefficient E_xi as printed, and the equation of motion is only its tangent
    part. The term is computed from geomech's own E_xi, not fitted. `c` is the
    coefficient a least-squares fit would give and `c_explicit` is -q . E_xi."""
    A, B, Z = linearize_fd(rb, state, u)
    Mg, Ag, Bg, E = geomech_AB(rb, mod, layout, state, u, geomech_params(rb.params), team_name)
    Rm = Mg @ A @ Z - Ag @ Z
    raw = float(np.abs(Rm).max())
    sl = slots(rb)
    m = len(sl)
    tangent_part, reconciled, cs, cx = 0.0, 0.0, [], []
    other = Rm.copy()
    for k, (kind, c, v, i) in enumerate(sl):
        if kind != "S2":
            continue
        q = _get(state, c, i)
        rows = slice(3 * (m + k), 3 * (m + k) + 3)
        Rk = (I3 - np.outer(q, q)) @ Rm[rows]
        D = np.array([np.cross(Z[3 * k:3 * k + 3, j], q) for j in range(Z.shape[1])]).T
        cs.append(float(np.sum(Rk * D) / np.sum(D * D)))
        c_explicit = -float(q @ E[k])
        cx.append(c_explicit)
        tangent_part = max(tangent_part, float(np.abs(Rk).max()))
        reconciled = max(reconciled, float(np.abs(Rk - c_explicit * D).max()))
        other[rows] = 0.0
    return dict(raw=raw, tangent=tangent_part, reconciled=max(reconciled, float(np.abs(other).max())),
                B=float(np.abs(Mg @ B - Bg).max()), c=cs, c_explicit=cx, scale=float(np.abs(Ag @ Z).max()))


# =====================================================================
# 10. The checks
# =====================================================================
def _solve_qp_v1(Q, c, A, b, tol=1e-9):
    """The first version of solve_qp in verify/ch08.py, kept here only to show what its
    guard on dependent active sets changed (see the checks of verify/ch08.py)."""
    from itertools import combinations
    n, m = len(c), len(b)
    for size in range(m + 1):
        for act in combinations(range(m), size):
            act = list(act)
            if act:
                Aa = A[act]
                K = np.block([[Q, Aa.T], [Aa, np.zeros((size, size))]])
                try:
                    sol = np.linalg.solve(K, np.concatenate([-c, b[act]]))
                except np.linalg.LinAlgError:
                    continue
                z, lam = sol[:n], sol[n:]
            else:
                z, lam = np.linalg.solve(Q, -c), np.zeros(0)
            if np.all(A @ z <= b + tol * (1 + np.abs(b))) and np.all(lam >= -tol):
                return z
    return None


def main():
    ck = Checker("ch19")
    rng = np.random.default_rng(19)
    rb = team()
    p = rb.params
    plan = Plan()
    R = results(fresh="--fresh" in sys.argv)
    S = R["summary"]

    # ---------------------------------------------------------------- the manifold
    for n in (2, 3, 4):
        ck.true(f"n = {n}: dimension 6 + 5n", team(n).dim == 6 + 5 * n, f"{team(n).dim}")
    ck.record("dim", {n: team(n).dim for n in (2, 3, 4)})
    ck.record("stored", {n: team(n).stored for n in (2, 3, 4)})
    hover_one = (p.m[0] + p.m_L) * p.g
    ck.true("one vehicle cannot hover with the payload", hover_one > F_MAX, f"{hover_one:.1f} N > {F_MAX} N")
    ck.record("hover_one", hover_one)
    ck.record("hover_share", (p.m[0] + p.m_L / 3) * p.g)

    # ---------------------------------------------------------------- the model, checked by machine
    scale = {}
    for n in (2, 3, 4):
        rbn = team(n)
        mod, layout, err, _, size = geomech_linear(rbn, f"ch19_team{n}")
        ck.true(f"n = {n}: geomech check_eom below 1e-5", err < 1e-5, f"{err:.1e}")
        worst_rest, worst_move, worst_raw, worst_B, worst_c, worst_qE_rest = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for trial in range(4):
            st = rbn.random_state(rng, speed=0.0 if trial < 2 else 0.7)
            u = (rng.uniform(10, 20, n), 0.3 * rng.normal(size=(n, 3)))
            g = geomech_agreement(rbn, mod, layout, st, u)
            worst_B = max(worst_B, g["B"])
            if trial < 2:
                worst_rest = max(worst_rest, g["raw"])
                worst_qE_rest = max(worst_qE_rest, max(abs(c) for c in g["c_explicit"]))
            else:
                worst_move, worst_raw = max(worst_move, g["reconciled"]), max(worst_raw, g["tangent"])
                worst_c = max(worst_c, max(abs(a - b) for a, b in zip(g["c"], g["c_explicit"])))
        ck.close(f"n = {n}: our linearization equals geomech's at states at rest", worst_rest, 0, tol=1e-6)
        ck.close(f"n = {n}: B equals geomech's", worst_B, 0, tol=1e-9)
        ck.close(f"n = {n}: moving states agree once -(xi x q)(q . E_xi) of eq18-projection is added", worst_move, 0, tol=1e-6)
        ck.close(f"n = {n}: a fitted coefficient equals -q . E_xi", worst_c, 0, tol=1e-6)
        ck.close(f"n = {n}: q . E_xi vanishes at states at rest", worst_qE_rest, 0, tol=1e-9)
        ck.true(f"n = {n}: without that term the tangent rows differ at moving states", worst_raw > 1e-2,
                f"{worst_raw:.2f}")
        Gm = grasp_map(rbn.params.r)
        N, rank = null_basis(Gm)
        ref = plan_point(rbn.params, Plan(length=0.0, drop=0.0, turn=0.0, s_gap=S_WIDE), 0.0)
        s0, u0 = state_of(rbn, ref), (ref["f"], ref["tau"])
        A, B, Z = linearize_fd(rbn, s0, u0)
        Ar, Br = Z.T @ A @ Z, Z.T @ B
        scale[n] = dict(dim=rbn.dim, inputs=4 * n, check_eom=err, eom_chars=size,
                        grasp_rank=rank, internal=N.shape[1], reduced=Ar.shape[0],
                        ctrb=ctrb_rank(Ar, Br), pairs=n * (n - 1) // 2,
                        qp_vars=N.shape[1], qp_cons=n + n * (n - 1) // 2, filter_vars=3 * n,
                        lin_rest=worst_rest, lin_move=worst_move, lin_raw=worst_raw, lin_c=worst_c)
    ck.true("n = 2: the grasp map has rank 5", scale[2]["grasp_rank"] == 5)
    ck.true("n = 3, 4: the grasp map has rank 6", scale[3]["grasp_rank"] == 6 and scale[4]["grasp_rank"] == 6)
    ck.true("internal forces: 3n - rank", [scale[n]["internal"] for n in (2, 3, 4)] == [1, 3, 6])
    ck.true("n = 2: the payload is most tilted at the end of the flight",
            abs(S["n2"]["tilt_end"] - S["n2"]["tilt_max"]) < 1e-6, f"{S['n2']['tilt_end']:.2f} deg")
    ck.true("n = 2: hover linearization is not controllable on ker C (30 of 32)",
            (scale[2]["reduced"], scale[2]["ctrb"]) == (32, 30))
    ck.true("n = 3: controllable on ker C (42 of 42)", (scale[3]["reduced"], scale[3]["ctrb"]) == (42, 42))
    ck.true("n = 4: controllable on ker C (52 of 52)", (scale[4]["reduced"], scale[4]["ctrb"]) == (52, 52))

    worst = 0.0
    for _ in range(10):
        st = rb.random_state(rng)
        worst = max(worst, max(np.abs(x - y).max() for x, y in zip(affine_maps(rb, st),
                                                                   affine_maps_probed(rb, st))))
    ck.close("the copy of the library's linear system equals probing the library", worst, 0, tol=1e-10)
    newton = payload = torque = 0.0
    for _ in range(20):
        st = rb.random_state(rng)
        u = (rng.uniform(5, 20, 3), rng.normal(size=(3, 3)))
        acc = rb.dynamics(st, u)
        _, _, a = vehicle_kinematics(rb, st, acc)
        T = tensions(rb, st, u)
        newton = max(newton, max(np.abs(p.m[i] * (a[i] + p.g * E3) - u[0][i] * st.R[i] @ E3
                                        - T[i] * st.q[i]).max() for i in range(3)))
        payload = max(payload, np.abs(p.m_L * (acc[0] + p.g * E3) + sum(T[i] * st.q[i] for i in range(3))).max())
        torque = max(torque, np.abs(p.J_L @ acc[1] + np.cross(st.Omega_L, p.J_L @ st.Omega_L)
                                    + sum(np.cross(p.r[i], st.R_L.T @ (T[i] * st.q[i])) for i in range(3))).max())
    ck.close("20 states: each vehicle obeys m (a + g e3) = f R e3 + T q", newton, 0, tol=1e-9)
    ck.close("20 states: the payload obeys m_L (a + g e3) = -sum T_i q_i", payload, 0, tol=1e-9)
    ck.close("20 states: the payload obeys J_L Omega_dot + Omega x J_L Omega = -sum r_i x R_L^T T_i q_i",
             torque, 0, tol=1e-9)

    # ---------------------------------------------------------------- the redundancy
    Gm = grasp_map(p.r)
    W_hover = np.r_[0.0, 0.0, p.m_L * p.g, 0.0, 0.0, 0.0]
    lam, _ = distribute(p.r, W_hover)
    ck.close("minimum norm in hover: every cable vertical with a third of the weight",
             lam, np.tile([0, 0, p.m_L * p.g / 3], (3, 1)), tol=1e-12)
    ck.record("T_hover_vertical", p.m_L * p.g / 3)
    d_vertical = np.linalg.norm(p.r[0] - p.r[1])
    ck.record("d_vertical", d_vertical)
    ck.true("vertical cables put the vehicles closer than r_min", d_vertical < R_MIN, f"{d_vertical:.3f} m")
    lam_opt, act = distribute(p.r, W_hover, l=p.l, d_min=D_PLAN, t_min=T_MIN)
    ck.close("QP: wrench balance to rounding", Gm @ lam_opt.ravel(), W_hover, tol=1e-10)
    xq = p.r + lam_opt / np.linalg.norm(lam_opt, axis=1)[:, None]
    d_opt = min(np.linalg.norm(xq[i] - xq[j]) for i in range(3) for j in range(i + 1, 3))
    lean_opt = np.degrees(np.arctan2(np.linalg.norm(lam_opt[:, :2], axis=1), lam_opt[:, 2]))
    ck.true("QP in hover: three pair constraints active, distance within 1 mm of D_PLAN",
            act == 3 and abs(d_opt - D_PLAN) < 1e-3, f"{d_opt:.4f} m")
    ck.record("qp_hover", dict(d=d_opt, lean=float(lean_opt[0]), T=float(np.linalg.norm(lam_opt[0])),
                               internal=float(np.linalg.norm(lam_opt[0, :2])),
                               f=float(np.linalg.norm(p.m[0] * p.g * E3 + lam_opt[0]))))
    # an accelerating, turning payload: minimum norm against the QP
    W_acc = np.r_[p.m_L * np.array([1.2, 0.0, p.g]), np.array([0.0, 0.0, 0.08])]
    lam_mn, _ = distribute(p.r, W_acc)
    lam_qp, act = distribute(p.r, W_acc, l=p.l, d_min=D_PLAN, t_min=T_MIN)
    for tag, lm in (("mn", lam_mn), ("qp", lam_qp)):
        ck.close(f"accelerating payload, {tag}: wrench balance", Gm @ lm.ravel(), W_acc, tol=1e-10)
        xq = p.r + lm / np.linalg.norm(lm, axis=1)[:, None]
        dd = [float(np.linalg.norm(xq[i] - xq[j])) for i in range(3) for j in range(i + 1, 3)]
        ck.record(f"acc_{tag}", dict(T=[float(v) for v in np.linalg.norm(lm, axis=1)], d=dd,
                                     norm=float(np.linalg.norm(lm))))
    ck.true("accelerating payload: all tensions positive in both", min(lam_mn[:, 2].min(), lam_qp[:, 2].min()) > T_MIN)
    Nb, _ = null_basis(Gm)
    ck.close("internal forces do not change the wrench", Gm @ Nb, 0, tol=1e-12)
    ck.close("the spreading force is an internal force", Gm @ spread_direction(p.r), 0, tol=1e-12)
    # random wrenches: balance and constraints
    worst_bal, worst_T, feas = 0.0, np.inf, 0
    for _ in range(200):
        W = W_hover + np.r_[rng.normal(size=3) * 2.0, rng.normal(size=3) * 0.15]
        lm, act = distribute(p.r, W, l=p.l, d_min=D_PLAN, t_min=T_MIN)
        if lm is None:
            continue
        feas += 1
        worst_bal, worst_T = max(worst_bal, np.abs(Gm @ lm.ravel() - W).max()), min(worst_T, lm[:, 2].min())
    ck.close("QP at 200 random wrenches: balance", worst_bal, 0, tol=1e-9)
    ck.true("QP at 200 random wrenches: every cable pulls with at least T_MIN", worst_T >= T_MIN - 1e-8,
            f"{feas} feasible, smallest {worst_T:.3f} N")
    ck.record("qp_random_feasible", feas)
    # the first version of the solver of verify/ch08.py, without its guard, on the same kind of problem
    huge, huge_now, caught, same = 0, 0, 0, 0
    rng2 = np.random.default_rng(19)
    for _ in range(200):
        W = W_hover + np.r_[rng2.normal(size=3) * 2.0, rng2.normal(size=3) * 0.15]
        lm1, _ = distribute(p.r, W, l=p.l, d_min=D_PLAN, t_min=T_MIN, solver=_solve_qp_v1)
        lm, _ = distribute(p.r, W, l=p.l, d_min=D_PLAN, t_min=T_MIN)
        bad = lm1 is not None and np.abs(lm1).max() > 1e6
        huge += int(bad)
        huge_now += int(lm is not None and np.abs(lm).max() > 1e6)
        caught += int(bad and lm is None)
        same += int(not bad and ((lm is None and lm1 is None)
                                 or (lm is not None and lm1 is not None and np.array_equal(lm, lm1))))
    ck.true("the unguarded first version returns a huge point on some of the same 200 wrenches", huge > 0,
            f"{huge} of 200")
    ck.true("... the solver of verify/ch08.py reports those infeasible, and equals the first version on the rest",
            huge_now == 0 and caught == huge and same == 200 - huge, f"{caught} infeasible, {same} equal")
    ck.record("qp_unguarded_huge", huge)

    # ---------------------------------------------------------------- the plan (flat map)
    worst, fl, fh, Tl, dl = 0.0, np.inf, 0.0, np.inf, np.inf
    for t in np.arange(0.0, T_END + 1e-9, 0.05):
        ref = plan_point(p, plan, t)
        st = state_of(rb, ref)
        acc = rb.dynamics(st, (ref["f"], ref["tau"]))
        worst = max(worst, np.abs(acc[0] - ref["a_L"]).max(), np.abs(acc[1] - ref["Omega_L_dot"]).max(),
                    np.abs(acc[2] - ref["omega_dot"]).max(), np.abs(acc[3] - ref["Omega_dot"]).max(),
                    np.abs(tensions(rb, st, (ref["f"], ref["tau"])) - ref["T"]).max())
        fl, fh, Tl = min(fl, ref["f"].min()), max(fh, ref["f"].max()), min(Tl, ref["T"].min())
        x = ref["x_Q"]
        dl = min(dl, min(np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i + 1, 3)))
    ck.close("flat map of the team: state and inputs satisfy the library's model", worst, 0, tol=1e-10)
    ck.record("flat_residual", float(worst))
    ck.record("plan", dict(f_min=fl, f_max=fh, T_min=Tl, d_min=dl, d_cruise=float(np.linalg.norm(
        plan_point(p, plan, 0.0)["x_Q"][0] - plan_point(p, plan, 0.0)["x_Q"][1])),
        speed=float(max(np.linalg.norm(plan_point(p, plan, t)["v_L"]) for t in np.arange(0, 12, 0.05))),
        acc=float(max(np.linalg.norm(plan_point(p, plan, t)["a_L"]) for t in np.arange(0, 12, 0.05)))))
    ck.true("the plan asks for thrusts within the limit and positive tensions", fh < F_MAX and Tl > T_MIN,
            f"f in [{fl:.1f}, {fh:.1f}] N, T >= {Tl:.2f} N")
    ck.true("the plan brings two vehicles closer than r_min", dl < R_MIN, f"{dl:.3f} m")
    inside = [t for t in np.arange(0.0, T_END, 0.01)
              if min(np.linalg.norm(a - b) for i, a in enumerate(plan_point(p, plan, t)["x_Q"])
                     for b in plan_point(p, plan, t)["x_Q"][i + 1:]) < R_MIN]
    ck.record("plan_inside", dict(t0=float(inside[0]), t1=float(inside[-1]), seconds=float(len(inside) * 0.01)))
    ck.record("plan_wall_margin", float(GAP_W / 2 - ARM - 0.5 * dl))
    ck.record("attitude_rates", [float(np.sqrt(GAINS["k_R"] / J)) for J in np.diag(p.J[0])])
    ends = []
    for h in (5e-3, 2.5e-3):
        ref0 = plan_point(p, plan, 2.0)
        states = rb.rollout(state_of(rb, ref0), lambda t, s: (plan_point(p, plan, t)["f"],
                                                              plan_point(p, plan, t)["tau"]),
                            h, int(round(2.0 / h)), hold=False, t0=2.0)
        ends.append(np.linalg.norm(states[-1].x_L - plan_point(p, plan, 4.0)["x_L"]))
    ck.true("open loop on the flat inputs for 2 s: fourth order", 10 < ends[0] / ends[1] < 24 or ends[0] < 1e-9,
            f"{ends[0]:.1e} m, {ends[1]:.1e} m")
    ck.record("open_loop", dict(h=ends[0], half=ends[1], ratio=ends[0] / ends[1]))
    ref = plan_point(p, plan, 4.3)
    counts = np.linalg.matrix_rank(Gm), 6 + 3 + Nb.shape[1], 4 * 3
    ck.true("flat output count: 6 + n + (3n - 6) = 4n", counts[1] == counts[2])

    # ---------------------------------------------------------------- blocks against their originals
    (orig,) = from_chapter(HERE / "ch06.py", ["omega_dot_des"],
                           dict(np=np, tangent=tangent, e_q=e_q, e_omega=e_omega, k_q=1.0, k_om=1.0))
    worst = 0.0
    for _ in range(20):
        q, qd = unit(rng.normal(size=3)), unit(rng.normal(size=3))
        om, omd, omdd = tangent(q, rng.normal(size=3)), tangent(qd, rng.normal(size=3)), rng.normal(size=3)
        worst = max(worst, np.abs(omega_dot_des(q, om, qd, omd, omdd, 7.0, 3.0)
                                  - orig(q, om, qd, omd, omdd, 7.0, 3.0)).max())
    ck.close("eq6-law-s2 here equals verify/ch06.py", worst, 0, tol=1e-13)
    from robots.quadrotor import desired_attitude as lib_Rd
    F = rng.normal(size=3) + 3 * E3
    ck.close("eq6-Rd here equals the library's", desired_attitude(F, 0.3),
             lib_Rd(F, np.array([np.cos(0.3), np.sin(0.3), 0.0])), tol=1e-14)
    # on the reference the cascade returns the flat inputs
    worst = 0.0
    casc = Cascade(rb, plan, separation=False, tension=False, d_plan=None)
    for t in (0.5, 3.1, 4.4, 6.2, 9.0):
        ref = plan_point(p, plan, t)
        f, tau = casc(t, state_of(rb, ref))
        worst = max(worst, np.abs(f - ref["f"]).max(), np.abs(tau - ref["tau"]).max())
    ck.close("on the reference the cascade returns the flat inputs", worst, 0, tol=1e-9)
    ck.record("cascade_on_reference", float(worst))
    # the filter's model of H_ddot against a finite difference along the model
    st = state_of(rb, plan_point(p, plan, 4.0))
    st = replace(st, omega=st.omega + 0.2 * np.array([tangent(q, rng.normal(size=3)) for q in st.q]))
    Fv = np.array([15.0 * E3 + rng.normal(size=3) for _ in range(3)])
    a0, Da, T0, DT = affine_maps(rb, st)
    x, v = vehicle_kinematics(rb, st)
    acc_v = (a0 + Da @ Fv.ravel()).reshape(3, 3)
    d, dv, da = x[0] - x[1], v[0] - v[1], acc_v[0] - acc_v[1]
    H_ddot = 2 * dv @ dv + 2 * d @ da
    hh = 1e-4
    fR = (np.linalg.norm(Fv, axis=1), np.zeros((3, 3)))
    st_al = replace(st, R=np.array([desired_attitude(Fi) for Fi in Fv]), Omega=np.zeros((3, 3)))

    def H_of(s):
        xx = rb.quadrotor_positions(s)
        return (xx[0] - xx[1]) @ (xx[0] - xx[1])
    sp = rb.step(st_al, fR, hh)
    sm = rb.step(st_al, fR, -hh)
    ck.close("barrier: the model's second derivative of H_ij equals a difference along the library's flow",
             (H_of(sp) - 2 * H_of(st_al) + H_of(sm)) / hh ** 2, H_ddot, tol=1e-4)

    # the QP solver of verify/ch08.py against its unguarded first version, on well-posed problems
    worst = 0.0
    for _ in range(200):
        nv, mc = 4, 3
        Aq, bq = rng.normal(size=(mc, nv)), rng.normal(size=mc) + 0.5
        cq = rng.normal(size=nv)
        z1, z2 = _solve_qp_v1(np.eye(nv), cq, Aq, bq), ch08.solve_qp(np.eye(nv), cq, Aq, bq)
        worst = max(worst, np.abs(z1 - z2).max())
    ck.close("the QP solver equals its unguarded first version on 200 well-posed problems", worst, 0, tol=1e-9)
    Aq = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    zq = ch08.solve_qp(np.eye(2), np.array([1.0, 1.0]), Aq, np.array([-2.0, -2.0, -3.0]))
    ck.close("QP solver with a repeated constraint: the right point", zq, [-2.0, -3.0], tol=1e-12)

    # ---------------------------------------------------------------- loop speeds
    order = {k: {q: S[k][q] for q in ("rms", "peak", "final", "att_max", "f_max", "T_min", "done")}
             for k in S if k.startswith("order_")}
    ck.record("order", order)
    ck.true("cascade on a safe plan: millimeter tracking", S["order_base"]["rms"] < 3e-3, f"{S['order_base']['rms'] * 1e3:.2f} mm")
    ck.true("payload attitude loop at 4 rad/s: worse, and the thrust saturates",
            S["order_payload_fast"]["rms"] > 5 * S["order_base"]["rms"],
            f"{S['order_payload_fast']['rms'] * 1e3:.2f} mm, f up to {S['order_payload_fast']['f_max']:.1f} N")
    ck.true("... and a faster vehicle attitude loop restores it", S["order_payload_fast_kappa"]["rms"] < 5e-3,
            f"{S['order_payload_fast_kappa']['rms'] * 1e3:.2f} mm")
    ck.true("no feedforward: centimeters, not millimeters", S["order_no_ff"]["rms"] > 50 * S["order_base"]["rms"],
            f"{S['order_no_ff']['rms'] * 100:.1f} cm")

    # ---------------------------------------------------------------- the mission and its ablation
    A_ = S["all"]
    ck.record("mission", {k: {q: v for q, v in s.items() if q != "seconds"} for k, s in S.items()})   # no wall-clock times
    ck.true("mission: completed, QP feasible at every step", A_["done"] and A_["infeasible"] == 0)
    ck.true("mission: payload within 5 cm of its reference throughout", A_["peak"] < 0.05, f"{A_['peak'] * 100:.1f} cm")
    ck.true("mission: every tension above T_MIN", A_["T_min"] > T_MIN, f"{A_['T_min']:.2f} N")
    ck.true("mission: vehicles never closer than r_min, every H_ij >= 0", A_["H_min"] >= 0,
            f"d {A_['d_min']:.3f} m, H {A_['H_min']:.3f}")
    ck.true("mission: thrust within the limit", A_["f_max"] < F_MAX, f"{A_['f_max']:.1f} N")
    ck.true("mission: every vehicle clears the walls of the gap", A_["wall"] > 0.02, f"{A_['wall'] * 100:.1f} cm")
    for k in ("unprotected", "pairs_only", "barrier_only", "push", "n4"):
        ck.record(f"wall_{k}", S[k]["wall"])
    ck.true("mission: manifold residual at rounding level", A_["resid"] < 1e-12, f"{A_['resid']:.1e}")
    ck.true("unprotected: the vehicles follow the reference inside r_min", S["unprotected"]["H_min"] < 0,
            f"d {S['unprotected']['d_min']:.3f} m")
    ck.true("pair constraints alone: a transient inside r_min", S["pairs_only"]["H_min"] < 0,
            f"d {S['pairs_only']['d_min']:.3f} m")
    ck.true("barrier alone: holds r_min", S["barrier_only"]["H_min"] > -1e-3,
            f"H {S['barrier_only']['H_min']:.4f}")
    ck.true("barrier alone tilts the payload more than with the pair constraints",
            S["barrier_only"]["tilt_max"] > A_["tilt_max"],
            f"{S['barrier_only']['tilt_max']:.1f} deg against {A_['tilt_max']:.1f} deg")
    ck.true("an unknown push takes two vehicles inside r_min with the barrier on", S["push"]["H_min"] < 0,
            f"d {S['push']['d_min']:.3f} m")
    ck.true("telling the barrier about the push helps", S["push_told"]["d_min"] > S["push"]["d_min"],
            f"d {S['push_told']['d_min']:.3f} m")
    for k in ("all", "min_norm", "barrier_only", "pairs_only", "unprotected", "no_tension_floor", "no_ff",
              "weak", "long_cable", "push", "push_told", "slow_barrier", "n2", "n3_safe", "n4_safe"):
        ck.true(f"{k}: stays on the manifold", S[k]["resid"] < 1e-12, f"{S[k]['resid']:.1e}")

    # ---------------------------------------------------------------- scaling
    for n, key in ((2, "n2"), (3, "n3_safe"), (4, "n4_safe")):
        scale[n].update({q: S[key][q] for q in ("rms", "peak", "T_min", "d_min", "H_min", "f_max", "tilt_max",
                                                 "done", "wall", "steps", "infeasible", "filtered")})
        scale[n]["plan_d_min"] = S[key]["plan_d_min"]
        ck.true(f"n = {n}: a plan that respects r_min is flown to millimeters with every block on",
                S[key]["done"] and S[key]["rms"] < 5e-3 and S[key]["H_min"] > 0 and S[key]["T_min"] > T_MIN,
                f"rms {S[key]['rms'] * 1e3:.2f} mm, d {S[key]['d_min']:.3f} m")
    ck.record("scaling", scale)
    ck.true("n = 4 on the reference that is too tight: the vehicles stay apart, the thrust saturates",
            S["n4"]["H_min"] >= 0 and S["n4"]["T_min"] > T_MIN and S["n4"]["f_max"] >= F_MAX - 1e-9,
            f"rms {S['n4']['rms'] * 100:.1f} cm, d {S['n4']['d_min']:.3f} m, T {S['n4']['T_min']:.2f} N")
    ck.true("payload attitude loop at 6 rad/s: tracking is lost", S["order_payload_faster"]["rms"] > 0.5,
            f"{S['order_payload_faster']['rms']:.1f} m")
    ck.true("cable loop at 14 rad/s: tracking is lost, and kappa = 2 restores it",
            S["order_cable_faster"]["rms"] > 0.5 and S["order_cable_faster_kappa"]["rms"] < 5e-3,
            f"{S['order_cable_faster']['rms']:.1f} m, {S['order_cable_faster_kappa']['rms'] * 1e3:.2f} mm")
    ck.true("a vehicle at 85% thrust: the payload is not held level", S["weak"]["tilt_max"] > 30,
            f"tilt {S['weak']['tilt_max']:.0f} deg, T down to {S['weak']['T_min']:.1f} N")

    # ---------------------------------------------------------------- LQR of the team in hover
    Lq = S["lqr"]
    ck.record("lqr", Lq)
    ck.true("team hover LQR: closed loop stable", Lq["slowest"] > 0, f"slowest pole {-Lq['slowest']:.2f} 1/s")
    ck.true("team hover LQR: recovers a 0.3 m offset", Lq["cases"]["offset 0.3 m"]["lqr"]["final"] < 1e-2)

    # ---------------------------------------------------------------- the flexible cable
    C = S["chain"]
    ck.record("chain", C)
    ck.true("cable: 3 (n_Q + 1) + 2 n degrees of freedom", C["dim"] == 3 * 3 + 2 * CHAIN_N, f"{C['dim']}")
    ck.close("cable: the flat map of thm10-chain satisfies the library's model", C["flat_residual"], 0, tol=1e-8)
    ck.close("cable: the hanging reference is an equilibrium", C["eq_residual"], 0, tol=1e-10)
    ck.true("cable: hanging equilibrium controllable on ker C", C["rank"] == C["reduced"], f"{C['rank']} of {C['reduced']}")
    ck.true("cable: LQR returns it to the hanging equilibrium", C["hold_err_end"] < 5e-3, f"{C['hold_err_end'] * 1e3:.2f} mm")
    ck.true("cable: tracking LQR beats open loop", C["track_rms_late"] < 0.1 * C["open_err_max"],
            f"{C['track_rms_late'] * 100:.2f} cm against {C['open_err_max'] * 100:.0f} cm")
    ck.true("cable: linearization predicts a perturbed simulation, error falls with eps",
            C["lin_vs_sim"][1e-4] < 1e-3 and C["lin_vs_sim"][1e-3] > 3 * C["lin_vs_sim"][1e-4],
            f"{C['lin_vs_sim'][1e-3]:.1e}, {C['lin_vs_sim'][1e-4]:.1e}")
    ck.true("cable: stays on the manifold", C["resid"] < 1e-12, f"{C['resid']:.1e}")
    from geomech import check_eom, compute_eom
    L, dW, V, inputs = chain_robot().lagrangian()
    err = float(check_eom(L, dW, V, compute_eom(L, dW, V), inputs=inputs))
    ck.true("cable: geomech check_eom below 1e-5", err < 1e-5, f"{err:.1e}")
    ck.record("chain_check_eom", err)

    # ---------------------------------------------------------------- integration
    free = team().random_state(np.random.default_rng(5), speed=0.5)
    drift = []
    for h in (4e-3, 2e-3):
        s, E0 = free, rb.energy(free)
        for _ in range(int(round(1.0 / h))):
            s = rb.step(s, rb.zero_input(), h)
        drift.append(abs(rb.energy(s) - E0) / abs(E0))
    ck.true("motors off for 1 s: energy drift falls like h^4", drift[0] < 1e-7 and 8 < drift[0] / drift[1] < 32,
            f"{drift[0]:.1e}, ratio {drift[0] / drift[1]:.1f}")
    ck.record("energy", dict(drift=drift[0], ratio=drift[0] / drift[1]))
    ck.record("constants", dict(F_MAX=F_MAX, R_MIN=R_MIN, T_MIN=T_MIN, D_PLAN=D_PLAN, ALPHA=[ALPHA1, ALPHA2],
                                S_WIDE=S_WIDE, S_GAP=S_GAP, GAP_W=GAP_W, ARM=ARM, GAINS=GAINS,
                                lean_wide=float(np.degrees(np.arctan(S_WIDE / (p.m_L * p.g / 3)))),
                                lean_gap=float(np.degrees(np.arctan(S_GAP / (p.m_L * p.g / 3))))))
    ck.finish()


if __name__ == "__main__":
    main()
