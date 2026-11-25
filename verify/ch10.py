"""Checks for Chapter 10, Differential flatness.  Run: python verify/ch10.py

The flat maps of the chapter are written once, in this file, on truncated Taylor
series ("jets"), so that every time derivative they need is exact to rounding.
The figure script and the notebook use the same functions. All dynamics and all
simulation come from the robot model library in case_studies/robots.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "verify"))
sys.path.insert(0, str(ROOT / "case_studies"))
import robots  # noqa: E402
from common import Checker, hat, vee  # noqa: E402
from robots.quadrotor import circle_reference, desired_attitude  # noqa: E402
from robots.core import attitude_tracking_torque  # noqa: E402

E1, E2, E3 = np.eye(3)
G = 9.81


# =====================================================================
# Jets: truncated Taylor series. a[k] is the k-th Taylor coefficient,
# so the k-th time derivative is k! a[k]. Axis 0 is the order.
# =====================================================================
def jmul(a, b, op=None):
    """Product of two jets (Cauchy product). op is the pointwise product:
    scalar times anything by default, or np.dot, np.cross, np.matmul."""
    K = min(len(a), len(b))
    if op is None:
        def op(x, y):
            return x * y if np.ndim(x) == 0 or np.ndim(y) == 0 or np.ndim(x) == np.ndim(y) \
                else x[..., None] * y
    out = [sum(op(a[i], b[k - i]) for i in range(k + 1)) for k in range(K)]
    return np.array(out)


def jdot(a, b):
    return jmul(a, b, np.dot)


def jcross(a, b):
    return jmul(a, b, np.cross)


def jsqrt(a):
    s = np.zeros(len(a))
    s[0] = math.sqrt(a[0])
    for k in range(1, len(a)):
        s[k] = (a[k] - sum(s[i] * s[k - i] for i in range(1, k))) / (2 * s[0])
    return s


def jdiv(a, b):
    """a / b for a scalar jet b."""
    K = min(len(a), len(b))
    d = np.zeros((K,) + np.shape(a[0]))
    for k in range(K):
        d[k] = (a[k] - sum(d[i] * b[k - i] for i in range(k))) / b[0]
    return d


def jnorm(a):
    return jsqrt(jdot(a, a))


def junit(a):
    return jdiv(a, jnorm(a))


def jder(a, times=1):
    """Jet of the time derivative: one order shorter each time."""
    for _ in range(times):
        a = np.array([(k + 1) * a[k + 1] for k in range(len(a) - 1)])
    return a


def jcossin(psi):
    """Jets of cos(psi) and sin(psi) for a scalar jet psi."""
    K = len(psi)
    c, s = np.zeros(K), np.zeros(K)
    c[0], s[0] = math.cos(psi[0]), math.sin(psi[0])
    for k in range(1, K):
        c[k] = -sum(j * psi[j] * s[k - j] for j in range(1, k + 1)) / k
        s[k] = sum(j * psi[j] * c[k - j] for j in range(1, k + 1)) / k
    return c, s


def derivs(a):
    """The time derivatives [a, a', a'', ...] of a jet."""
    return np.array([math.factorial(k) * a[k] for k in range(len(a))])


def jet_from_derivs(d):
    return np.array([np.asarray(d[k], float) / math.factorial(k) for k in range(len(d))])


# ------------------------------------------------------------ smooth test curves
class Trig:
    """x(t) = c0 + sum_j amp_j sin(rate_j t + phase_j), componentwise, with
    every derivative in closed form."""

    def __init__(self, c0, amp, rate, phase):
        self.c0 = np.asarray(c0, float)
        self.amp, self.rate, self.phase = (np.asarray(a, float) for a in (amp, rate, phase))

    def jet(self, t, order):
        out = []
        for k in range(order + 1):
            d = np.sum(self.amp * self.rate**k * np.sin(self.rate * t + self.phase + k * np.pi / 2),
                       axis=0)
            out.append((d + (self.c0 if k == 0 else 0.0)) / math.factorial(k))
        return np.array(out)

    def __call__(self, t):
        return self.jet(t, 0)[0]


def random_curve(rng, size=0.6, terms=3, shape=(3,), rate=(0.4, 1.6)):
    return Trig(rng.normal(size=shape), size * rng.normal(size=(terms,) + shape) / terms,
                rng.uniform(*rate, size=(terms,) + shape), rng.uniform(0, 2 * np.pi, size=(terms,) + shape))


def figure_eight(rate=1.0, a=1.2, b=0.6, c=0.15, z0=0.0):
    """The load path of the chapter: x_L = (a sin(rate t), b sin(2 rate t), z0 + c cos(2 rate t))."""
    return Trig([0.0, 0.0, z0], [[a, b, c]], [[rate, 2 * rate, 2 * rate]], [[0.0, 0.0, np.pi / 2]])


# =====================================================================
# Flat maps
# =====================================================================
def attitude_from_force(F, psi, J):
    """From the jet of the thrust vector F = f R e3 (order >= 2) and of the heading
    angle psi_d (order >= 2): f, R (eq6-Rd), Omega, Omega_dot and tau (eq10-quad-map)."""
    K = min(len(F), len(psi))
    F, psi = F[:K], psi[:K]
    f = jnorm(F)
    b3 = jdiv(F, f)
    c, s = jcossin(psi)
    heading = np.stack([c, s, np.zeros(K)], axis=1)
    b2 = junit(jcross(b3, heading))
    b1 = jcross(b2, b3)
    R = np.stack([b1, b2, b3], axis=2)                       # jet of matrices, columns b_i
    Rt = np.transpose(R, (0, 2, 1))
    W = jmul(Rt[:K - 1], jder(R), np.matmul)                 # hat(Omega) = R' R_dot
    Omega = np.array([vee(0.5 * (w - w.T)) for w in W])
    Omega_dot = jder(Omega)
    Om, Omd = Omega[0], Omega_dot[0]
    return dict(f=f[0], R=R[0], Omega=Om, Omega_dot=Omd, tau=J @ Omd + np.cross(Om, J @ Om),
                f_jet=f, R_jet=R, Omega_jet=Omega, F=F[0])


def quadrotor_flat(x, psi, p):
    """eq10-quad-map. x: jet of the position, order >= 4. psi: jet of psi_d, order >= 2."""
    acc = jder(x, 2)
    acc[0] = acc[0] + p.g * E3
    out = attitude_from_force(p.m * acc, psi, p.J)
    d = derivs(x)
    out.update(x=d[0], v=d[1], a=d[2])
    return out


def quadrotor_flat_closed(d, psi, psi1, psi2, p):
    """The same map in closed form, eq10-quad-map to eq10-quad-Omega3-dot.
    d = [x, x', x'', x''', x'''']."""
    F, F1, F2 = p.m * (d[2] + p.g * E3), p.m * d[3], p.m * d[4]
    f = np.linalg.norm(F)
    b3 = F / f
    c = np.array([np.cos(psi), np.sin(psi), 0.0])
    cp = np.cross(E3, c)                                     # c_dot = psi_dot cp
    b2 = np.cross(b3, c) / np.linalg.norm(np.cross(b3, c))
    b1 = np.cross(b2, b3)
    f1 = b3 @ F1
    O1, O2 = -(b2 @ F1) / f, (b1 @ F1) / f
    O3 = (O1 * (b3 @ c) + psi1 * (b2 @ cp)) / (b1 @ c)
    Om = np.array([O1, O2, O3])
    O1d = -(b2 @ F2 + 2 * f1 * O1) / f + O2 * O3
    O2d = (b1 @ F2 - 2 * f1 * O2) / f - O1 * O3
    R = np.column_stack([b1, b2, b3])
    b1d, b2d, b3d = (R @ np.cross(Om, e) for e in (E1, E2, E3))
    num = (O1d * (b3 @ c) + O1 * (b3d @ c + psi1 * (b3 @ cp)) + psi2 * (b2 @ cp)
           + psi1 * (b2d @ cp) - O3 * (b1d @ c + psi1 * (b1 @ cp)))
    Omd = np.array([O1d, O2d, num / (b1 @ c)])
    return dict(f=f, R=R, Omega=Om, Omega_dot=Omd, tau=p.J @ Omd + np.cross(Om, p.J @ Om),
                f_dot=f1, f_ddot=b3 @ F2 + f * (O1**2 + O2**2))


def load_flat(xL, psi, p):
    """eq10-load-map. xL: jet of the load position, order >= 6."""
    A = jder(xL, 2)
    A[0] = A[0] + p.g * E3
    A = p.m_L * A                                           # -T q = m_L (x_L'' + g e3)
    T = jnorm(A)
    q = -jdiv(A, T)
    K = len(q)
    xQ = xL[:K] - p.l * q
    accQ = jder(xQ, 2)
    accQ[0] = accQ[0] + p.g * E3
    F = p.m_Q * accQ + A[:K - 2]                            # f R e3
    out = attitude_from_force(F, psi, p.J)
    qd = derivs(q)
    d = derivs(xL)
    out.update(x_L=d[0], v_L=d[1], a_L=d[2], q=qd[0], omega=np.cross(qd[0], qd[1]),
               omega_dot=np.cross(qd[0], qd[2]), T=T[0], x_Q=xQ[0], q_jet=q, T_jet=T, xQ_jet=xQ)
    return out


def chain_flat(xn, psi, mass, l, J, g=G, attach=(0,), pulls=None):
    """thm10-chain. Joints 0..n, link i from joint i - 1 to joint i with direction
    q_i, a free end at joint n, quadrotors at the joints in `attach` (joint 0 always).
    xn: jet of the free end, order >= 2 n + 4. mass[k]: mass at joint k, an attached
    quadrotor included. psi: one heading jet per quadrotor (or a single jet).
    pulls[k]: jet of the tension vector T_k q_k of link k, for every attached k >= 1;
    these are flat outputs. Returns link directions, tensions, and for each
    quadrotor (in the order of `attach`) its attitude, velocity and input."""
    n = len(l)
    psis = [psi] * len(attach) if np.ndim(psi[0]) == 0 and np.ndim(psi) == 1 else list(psi)
    Js = [J] * len(attach) if np.ndim(J) == 2 else list(J)
    x, pull = xn, None                                      # pull = T_i q_i
    q, T, quads = [None] * n, [None] * n, {}

    def weight(x, k):
        A = jder(x, 2)
        A[0] = A[0] + g * E3
        return mass[k] * A

    for i in range(n, 0, -1):
        A = weight(x, i)
        below = 0.0 * A if pull is None else pull[:len(A)]
        if i in attach:                                     # the tension above is given
            pull = pulls[i]
            K = min(len(A), len(pull))
            quads[i] = A[:K] + pull[:K] - below[:K]         # f_i R_i e3
        else:                                               # T_i q_i = T_{i+1} q_{i+1} - m_i (x_i'' + g e3)
            pull = below - A
        T[i - 1] = jnorm(pull)
        q[i - 1] = jdiv(pull, T[i - 1])
        K = min(len(x), len(pull))
        x = x[:K] - l[i - 1] * q[i - 1][:K]                 # x_{i-1} = x_i - l_i q_i
    A = weight(x, 0)
    quads[0] = A - pull[:len(A)]
    att = [attitude_from_force(quads[k], psis[j], Js[j]) for j, k in enumerate(attach)]
    qd = [derivs(qi) for qi in q]
    out = dict(x_0=derivs(x)[0], v_0=derivs(x)[1], a_0=derivs(x)[2],
               q=np.array([d[0] for d in qd]),
               omega=np.array([np.cross(d[0], d[1]) for d in qd]),
               omega_dot=np.array([np.cross(d[0], d[2]) for d in qd]),
               T=np.array([Ti[0] for Ti in T]))
    for key in ("f", "R", "Omega", "Omega_dot", "tau"):
        out[key] = np.array([a[key] for a in att])
    return out


# =====================================================================
# Open-loop simulation on the library models
# =====================================================================
def simulate(robot, state, u, h, n, t0=0.0, params=None, every=1, feedback=False):
    """Integrate a library model with the library's own fourth-order step, with the
    input evaluated at each stage (times t, t + h/2, t + h/2, t + h) and not held:
    the library's rollout with hold=False. Holding a moving input over a step
    would make the method first order. u is u(t), or u(t, state) when feedback=True."""
    law = u if feedback else (lambda t, s: u(t))
    return robot.rollout(state, law, h, n, params, hold=False, t0=t0)[::every]


# =====================================================================
# Smooth curves through waypoints: minimize the integral of |x^(r)|^2
# =====================================================================
def _basis(s, k, deg):
    """Row of d^k/ds^k of (1, s, ..., s^deg)."""
    row = np.zeros(deg + 1)
    for j in range(k, deg + 1):
        row[j] = math.factorial(j) / math.factorial(j - k) * s ** (j - k)
    return row


def _constraints(n_seg, dur, r, smooth):
    """Linear constraints A c = b(w) on the stacked coefficients of one axis:
    interpolation, rest at both ends (derivatives 1..r-1 zero) and continuity of
    derivatives 1..smooth at the interior waypoints. Returns A and a function b(w)."""
    deg, rows, kinds = 2 * r - 1, [], []

    def row(i, s, k):
        z = np.zeros(n_seg * (deg + 1))
        z[i * (deg + 1):(i + 1) * (deg + 1)] = _basis(s, k, deg) / dur[i] ** k
        return z

    for i in range(n_seg):
        rows += [row(i, 0.0, 0), row(i, 1.0, 0)]
        kinds += [("w", i), ("w", i + 1)]
    for k in range(1, r):
        rows += [row(0, 0.0, k), row(n_seg - 1, 1.0, k)]
        kinds += [("0",), ("0",)]
    for i in range(n_seg - 1):
        for k in range(1, smooth + 1):
            rows.append(row(i, 1.0, k) - row(i + 1, 0.0, k))
            kinds.append(("0",))
    A = np.array(rows)

    def b(w):
        return np.array([w[kd[1]] if kd[0] == "w" else 0.0 for kd in kinds])
    return A, b


def _cost(n_seg, dur, r):
    """P with c' P c = integral of (x^(r))^2 over all segments, one axis."""
    deg = 2 * r - 1
    P = np.zeros((n_seg * (deg + 1),) * 2)
    for i in range(n_seg):
        for j in range(r, deg + 1):
            for k in range(r, deg + 1):
                cj = math.factorial(j) / math.factorial(j - r)
                ck_ = math.factorial(k) / math.factorial(k - r)
                P[i * (deg + 1) + j, i * (deg + 1) + k] = cj * ck_ / (j + k - 2 * r + 1) * dur[i] ** (1 - 2 * r)
    return P


class Spline:
    """Piecewise polynomial of degree 2 r - 1 through waypoints at given times."""

    def __init__(self, coeff, times, r):
        self.coeff, self.times, self.r = coeff, np.asarray(times, float), r   # coeff: (n_seg, deg + 1, dim)
        self.dur = np.diff(self.times)

    def jet(self, t, order):
        i = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.dur) - 1))
        s = (t - self.times[i]) / self.dur[i]
        deg = self.coeff.shape[1] - 1
        return np.array([_basis(s, k, deg) @ self.coeff[i] / self.dur[i] ** k / math.factorial(k)
                         for k in range(order + 1)])

    def jet_side(self, i, s, order):
        deg = self.coeff.shape[1] - 1
        return np.array([_basis(s, k, deg) @ self.coeff[i] / self.dur[i] ** k / math.factorial(k)
                         for k in range(order + 1)])

    def __call__(self, t):
        return self.jet(t, 0)[0]

    def cost(self):
        P = _cost(len(self.dur), self.dur, self.r)
        c = self.coeff.reshape(-1, self.coeff.shape[2])
        return float(np.trace(c.T @ P @ c))


def min_derivative_spline(waypoints, times, r=4, method="qp"):
    """The curve through the waypoints at the given times, at rest at both ends,
    that minimizes the integral of |x^(r)|^2 (r = 4: minimum snap).

    method "qp": quadratic program with continuity up to order r - 1 imposed,
    solved through its KKT system (eq10-minsnap).
    method "spline": no optimization. The square linear system that asks for
    continuity up to order 2 r - 2, which the minimizer has on its own."""
    w = np.asarray(waypoints, float)
    times = np.asarray(times, float)
    n_seg, dur, deg = len(w) - 1, np.diff(times), 2 * r - 1
    if method == "qp":
        A, b = _constraints(n_seg, dur, r, r - 1)
        P = _cost(n_seg, dur, r)
        KKT = np.block([[2 * P, A.T], [A, np.zeros((len(A), len(A)))]])
        sol = np.column_stack([np.linalg.solve(KKT, np.concatenate([np.zeros(P.shape[0]), b(w[:, d])]))
                               for d in range(w.shape[1])])
        c = sol[:P.shape[0]]
    else:
        A, b = _constraints(n_seg, dur, r, 2 * r - 2)
        c = np.column_stack([np.linalg.solve(A, b(w[:, d])) for d in range(w.shape[1])])
    return Spline(c.reshape(n_seg, deg + 1, w.shape[1]), times, r)


def chord_times(waypoints, t_f):
    """Waypoint times proportional to chord length, from 0 to t_f."""
    d = np.linalg.norm(np.diff(np.asarray(waypoints, float), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)]) * t_f / np.sum(d)


WAYPOINTS = np.array([[0.0, 0.0, 0.0], [1.5, 1.0, 0.8], [3.0, -0.8, 1.2], [4.5, 0.6, 0.5], [6.0, 0.0, 0.0]])


# =====================================================================
# The planar thought experiments
# =====================================================================
def hand_position(xw, l, g=G):
    """Where the hand must be for the weight (jet xw, order >= 2) to follow its path
    on a string of length l: x_hand = x + l (x'' + g e3)/|x'' + g e3|  (eq10-hand)."""
    A = jder(xw, 2)
    A[0] = A[0] + g * E3
    return xw[:len(A)] + l * junit(A)


def string_run(P, L, times, h=1e-3):
    """A weight on a string of length L (or a stick whose center of oscillation is
    at L from the hand). The hand is moved as eq10-hand says for the curve P, and
    the pendulum's own equation, L alpha = q x (-(a_hand + g e3)), is integrated
    with a fourth-order step. Returns the simulated positions of the weight (or
    of the point at L) at the given times, and the hand positions."""
    def hand_acc(t):
        return derivs(jder(hand_position(P.jet(t, 6), L), 2))[0]

    def rhs(t, q, w):
        return np.cross(w, q), np.cross(q, -(hand_acc(t) + G * E3)) / L

    j = P.jet(0.0, 4)
    A = jder(j, 2)
    A[0] = A[0] + G * E3
    qj = -junit(A)
    q, w = qj[0], np.cross(qj[0], qj[1])
    out, hands, t = [], [], 0.0
    for t_next in times:
        steps = max(1, int(np.ceil((t_next - t) / h - 1e-9)))
        hs = (t_next - t) / steps                            # land exactly on t_next
        for _ in range(steps):
            k1 = rhs(t, q, w)
            k2 = rhs(t + hs / 2, q + hs / 2 * k1[0], w + hs / 2 * k1[1])
            k3 = rhs(t + hs / 2, q + hs / 2 * k2[0], w + hs / 2 * k2[1])
            k4 = rhs(t + hs, q + hs * k3[0], w + hs * k3[1])
            q = q + hs / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w = w + hs / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            q = q / np.linalg.norm(q)
            t += hs
        hand = derivs(hand_position(P.jet(t, 2), L))[0]
        hands.append(hand)
        out.append(hand + L * q)
    return np.array(out), np.array(hands)


def stick_run(P, r, d, t_f, h=1e-3, aim=None):
    """A rigid stick swung in the vertical plane by its end. r: hand to center of
    mass, d = I_c/(m r), so that the point at r + d from the hand is the center of
    oscillation. The hand is moved by eq10-hand with string length `aim`
    (default r + d). Returns the largest distance between the point at `aim`
    and P(t)."""
    L = r + d
    aim = L if aim is None else aim
    times = np.arange(1, int(round(t_f / (10 * h))) + 1) * 10 * h
    if aim == L:
        pts, _ = string_run(P, L, times, h)
        return float(max(np.linalg.norm(p - P(t)) for p, t in zip(pts, times)))
    # hand moved for a string of length `aim`, stick still swings with length L
    def hand_acc(t):
        return derivs(jder(hand_position(P.jet(t, 6), aim), 2))[0]

    def rhs(t, q, w):
        return np.cross(w, q), np.cross(q, -(hand_acc(t) + G * E3)) / L
    j = P.jet(0.0, 4)
    A = jder(j, 2)
    A[0] = A[0] + G * E3
    qj = -junit(A)
    q, w, worst, t = qj[0], np.cross(qj[0], qj[1]), 0.0, 0.0
    for _ in range(int(round(t_f / h))):
        k1 = rhs(t, q, w)
        k2 = rhs(t + h / 2, q + h / 2 * k1[0], w + h / 2 * k1[1])
        k3 = rhs(t + h / 2, q + h / 2 * k2[0], w + h / 2 * k2[1])
        k4 = rhs(t + h, q + h * k3[0], w + h * k3[1])
        q = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        q = q / np.linalg.norm(q)
        t += h
        hand = derivs(hand_position(P.jet(t, 2), aim))[0]
        worst = max(worst, float(np.linalg.norm(hand + aim * q - P(t))))
    return worst


def chain_demo():
    """The three-link cable of fig10-chain: a quadrotor at joint 0, masses of
    0.2, 0.2 and 0.3 kg at joints 1 to 3, links of 0.4 m, and the curve of its free end."""
    prm = robots.flexible_cable.CableParams(n=3, attach=(0,), m=np.array([0.0, 0.2, 0.2, 0.3]),
                                            l=np.full(3, 0.4))
    return robots.get("quadrotor_cable", n=3, attach=(0,), params=prm), figure_eight(rate=CHAIN_RATE, a=1.0, b=0.5, c=0.1)


CHAIN_RATE = 1.5


# =====================================================================
# States from flat maps, and the feedforward experiment
# =====================================================================
def quad_state(rb, o):
    return rb.State(x=o["x"], v=o["v"], R=o["R"], Omega=o["Omega"])


def load_state(rb, o):
    return rb.State(x_L=o["x_L"], v_L=o["v_L"], q=o["q"], omega=o["omega"], R=o["R"], Omega=o["Omega"])


def cable_state(rb, o):
    return rb.State(x_0=o["x_0"], v_0=o["v_0"], q=o["q"], omega=o["omega"], R=o["R"], Omega=o["Omega"])


def circle_flat(t, p, rate=1.0):
    """The flat map along the reference of sec6-cascade. Returns the reference tuple
    of the library and (R_d, Omega_d, Omega_d_dot, f_d, tau_d) from eq10-quad-map."""
    ref = circle_reference(t, rate)
    psi = rate * t + np.pi / 2
    o = quadrotor_flat_closed(ref[:5], psi, rate, 0.0, p)
    return ref, o


def cascade_run(rb, mode, t_f=20.0, h=5e-3, kappa=1.0, offset=(0.05, 0.0, 0.0), stop=5.0):
    """The experiment of warn6-feedforward on the library quadrotor: eq6-Fd, eq6-Rd
    and eq6-law with Lee's gains, attitude gains scaled by (kappa^2, kappa).
    mode "zero": Omega_d = Omega_d_dot = 0.
    mode "flat": Omega_d and Omega_d_dot of the reference, from the flat map.
    mode "exact": the library's controller (derivatives of R_d along the closed loop).
    Returns times and position errors (stops when the error passes `stop`)."""
    p, k = rb.params, rb.Gains()
    gains = rb.Gains(k_R=kappa**2 * k.k_R, k_Omega=kappa * k.k_Omega)
    ref0, o0 = circle_flat(0.0, p)

    def law(t, s):
        ref, o = circle_flat(t, p)
        if mode == "exact":
            return rb.tracking_controller(s, ref, gains)
        Fd = -k.k_x * (s.x - ref[0]) - k.k_v * (s.v - ref[1]) + p.m * p.g * E3 + p.m * ref[2]
        Rd = desired_attitude(Fd, ref[5])
        Omd, Omd_dot = (o["Omega"], o["Omega_dot"]) if mode == "flat" else (np.zeros(3), np.zeros(3))
        return Fd @ (s.R @ E3), attitude_tracking_torque(s.R, s.Omega, Rd, Omd, Omd_dot, p.J,
                                                         gains.k_R, gains.k_Omega)

    c0, s0 = np.cos(np.pi / 2), np.sin(np.pi / 2)        # the start of warn6-feedforward
    R0 = np.array([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]])
    s = rb.State(x=ref0[0] + np.asarray(offset), v=ref0[1], R=R0, Omega=np.zeros(3))
    ts, es = [0.0], [float(np.linalg.norm(offset))]
    chunk = 20
    while ts[-1] < t_f - 1e-9 and es[-1] <= stop:
        s = simulate(rb, s, law, h, chunk, t0=ts[-1], feedback=True)[-1]
        ts.append(ts[-1] + chunk * h)
        es.append(float(np.linalg.norm(s.x - circle_reference(ts[-1])[0])))
    return np.array(ts), np.array(es)


# =====================================================================
# Checks
# =====================================================================
def main():
    ck = Checker("ch10")
    rng = np.random.default_rng(10)
    deg = np.degrees

    # ---------------------------------------------------------------- jets
    t = 0.37
    c = Trig([0.3, -0.2, 1.0], [[0.5, 0.4, 0.3]], [[1.3, 0.7, 2.1]], [[0.2, 1.0, 2.0]])
    j = c.jet(t, 6)
    fd = (c(t + 1e-4) - c(t - 1e-4)) / 2e-4
    ck.close("jet of a trigonometric curve: first derivative against a central difference", derivs(j)[1], fd, 1e-7)
    sj = Trig(2.0, [0.5], [1.7], [0.3]).jet(t, 6)
    ck.close("jet arithmetic: sqrt(a)^2 = a, (a/b) b = a, cos^2 + sin^2 = 1, to order 6",
             np.concatenate([jmul(jsqrt(sj), jsqrt(sj)) - sj, jmul(jdiv(sj, sj + 1.0), sj + 1.0) - sj,
                             jmul(*([jcossin(sj)[0]] * 2)) + jmul(*([jcossin(sj)[1]] * 2)) - np.eye(7)[0]]),
             0.0, 1e-9)
    u_j = junit(j)
    ck.close("jet of a unit vector: |q| = 1 to every order, and its derivative shifts the jet",
             np.concatenate([jdot(u_j, u_j) - np.eye(7)[0], derivs(jder(u_j))[0] - derivs(u_j)[1]]), 0.0, 1e-10)

    # ---------------------------------------------------------------- hand and weight (thought10-hand)
    l = 1.0
    hand = {}
    for name, w in (("slow", 2.0), ("natural", math.sqrt(G / l)), ("fast", 4.5)):
        amp = 0.3
        path = Trig([0.0, 0.0, 0.0], [[amp, 0.0, 0.0]], [[w, 1.0, 1.0]], [[np.pi / 2, 0.0, 0.0]])  # x = amp cos(w t)
        hj = hand_position(path.jet(0.0, 2), l)
        a_x = -amp * w * w
        want = np.array([amp + l * a_x / math.hypot(a_x, G), 0.0, l * G / math.hypot(a_x, G)])
        ck.close(f"hand position at the turning point, {name} swing: eq10-hand in closed form", hj[0], want, 1e-12)
        hand[name] = {"rate_rad_s": w, "weight_m": amp, "hand_m": float(hj[0][0]),
                      "string_angle_deg": float(deg(math.atan2(-a_x, G)))}
    ck.true("hand leads at low rate, nearly still at the natural rate, opposite above it",
            hand["slow"]["hand_m"] > 0.1 and abs(hand["natural"]["hand_m"]) < 0.02 and hand["fast"]["hand_m"] < -0.1,
            ", ".join(f"{v['hand_m']:+.3f}" for v in hand.values()))
    ck.record("hand", hand)
    ck.record("natural_rate_rad_s", math.sqrt(G / l))

    # the stick: the flat output is the center of oscillation (ex10-stick)
    Lr = 0.9                                                   # uniform rod held at one end
    r_c, d_c = Lr / 2, (Lr**2 / 12) / (Lr / 2)
    ck.close("uniform rod: center of oscillation at two thirds of its length", r_c + d_c, 2 * Lr / 3, 1e-14)
    P = Trig([0.0, 0.0, 0.0], [[0.4, 0.0, 0.1]], [[2.0, 1.0, 3.0]], [[0.0, 0.0, 0.5]])
    err_stick = stick_run(P, r_c, d_c, 5.0, h=2e-3)
    ck.true("stick swung by its end: the center of oscillation follows the drawn path", err_stick < 1e-9,
            f"err {err_stick:.1e} m")
    # the same recipe aimed at another point fails: move the hand as if the tip were the weight
    err_tip = stick_run(P, r_c, d_c, 5.0, h=2e-3, aim=Lr)
    ck.true("the same recipe aimed at the tip of the stick misses", err_tip > 0.02, f"err {err_tip:.3f} m")
    ck.record("stick", {"length_m": Lr, "center_of_oscillation_m": r_c + d_c, "err_m": err_stick,
                        "tip_recipe_err_m": err_tip})

    # ---------------------------------------------------------------- quadrotor (thm10-quad)
    rb = robots.get("quadrotor")
    p = rb.params
    worst_cf, worst_res, worst_fd = 0.0, 0.0, 0.0
    for _ in range(20):
        x, psi = random_curve(rng), random_curve(rng, shape=())
        t = rng.uniform(0, 10)
        o = quadrotor_flat(x.jet(t, 4), psi.jet(t, 2), p)
        d, dp = derivs(x.jet(t, 4)), derivs(psi.jet(t, 2))
        cf = quadrotor_flat_closed(d, dp[0], dp[1], dp[2], p)
        worst_cf = max(worst_cf, *(np.max(np.abs(o[k] - cf[k])) for k in ("f", "R", "Omega", "Omega_dot", "tau")))
        fj = derivs(o["f_jet"])
        worst_cf = max(worst_cf, abs(fj[1] - cf["f_dot"]), abs(fj[2] - cf["f_ddot"]))
        acc = rb.dynamics(quad_state(rb, o), (o["f"], o["tau"]))
        worst_res = max(worst_res, np.max(np.abs(acc[0] - o["a"])), np.max(np.abs(acc[1] - o["Omega_dot"])))
        e = 1e-5                                             # Omega against a finite difference of R(t)
        Rp = quadrotor_flat(x.jet(t + e, 4), psi.jet(t + e, 2), p)["R"]
        Rm = quadrotor_flat(x.jet(t - e, 4), psi.jet(t - e, 2), p)["R"]
        worst_fd = max(worst_fd, np.max(np.abs(vee(o["R"].T @ (Rp - Rm) / (2 * e)) - o["Omega"])))
        Rd = desired_attitude(o["F"], np.array([np.cos(dp[0]), np.sin(dp[0]), 0.0]))
        worst_cf = max(worst_cf, np.max(np.abs(Rd - o["R"])), abs(np.linalg.det(o["R"]) - 1))
    ck.true("quadrotor: closed forms eq10-quad-Omega, eq10-quad-Omega3, eq10-quad-Omega-dot equal the jet map; R is eq6-Rd",
            worst_cf < 1e-9, f"err {worst_cf:.1e}")
    ck.true("quadrotor: recovered state and inputs satisfy eq4-quadrotor in the library (20 random points)",
            worst_res < 1e-10, f"residual {worst_res:.1e}")
    ck.true("quadrotor: Omega equals the finite-difference derivative of R(t)", worst_fd < 1e-7, f"err {worst_fd:.1e}")

    # the reference of sec6-cascade, flown open loop
    def circle_jets(t):
        ref = circle_reference(t)
        return jet_from_derivs(ref[:5]), np.array([t + np.pi / 2, 1.0, 0.0])

    def u_circle(t):
        o = quadrotor_flat(*circle_jets(t), p)
        return o["f"], o["tau"]
    o0 = quadrotor_flat(*circle_jets(0.0), p)
    errs = {}
    for h in (0.02, 0.01):
        end = simulate(rb, quad_state(rb, o0), u_circle, h, int(round(10 / h)))[-1]
        errs[h] = float(np.linalg.norm(end.x - circle_reference(10.0)[0]))
    order = math.log2(errs[0.02] / errs[0.01])
    ck.true("quadrotor open loop on the circle of sec6-cascade: error after 10 s below 1e-7 m, fourth order in h",
            errs[0.01] < 1e-7 and 3.5 < order < 4.5, f"{errs[0.01]:.1e} m, order {order:.2f}")
    grid = np.linspace(0, 2 * np.pi, 721)
    O = [quadrotor_flat(*circle_jets(t), p) for t in grid]
    circ = {"hover_thrust_N": p.m * p.g, "f_min_N": min(o["f"] for o in O), "f_max_N": max(o["f"] for o in O),
            "tilt_max_deg": max(deg(np.arccos(o["R"][2, 2])) for o in O),
            "Omega_max_rad_s": max(np.linalg.norm(o["Omega"]) for o in O),
            "tau_max_Nm": max(np.linalg.norm(o["tau"]) for o in O),
            "open_loop_err_10s_m": errs[0.01], "open_loop_order": order}
    ck.record("circle", {k: float(v) for k, v in circ.items()})
    # at hover the map returns hover
    hov = quadrotor_flat(np.zeros((5, 3)), np.zeros(3), p)
    ck.close("hover: f = m g, R = I, Omega = 0, tau = 0", np.concatenate([[hov["f"] - p.m * p.g], (hov["R"] - np.eye(3)).ravel(), hov["Omega"], hov["tau"]]), 0.0, 1e-12)

    # derivative ladder of the quadrotor: which derivative of x each quantity first feels
    def ladder(fn, base, keys, orders, bump=0.5):
        """The highest derivative of the flat output that each quantity depends on."""
        ref = fn(base)
        last = {}
        for key in keys:
            last[key] = max(k for k in orders if np.max(np.abs(np.asarray(
                fn(_bump(base, k, bump))[key]) - np.asarray(ref[key]))) > 1e-11)
        return None, last

    def _bump(base, k, bump):
        jb = base.copy()
        jb[k] = jb[k] + bump * np.array([0.3, -0.5, 0.4]) / math.factorial(k)
        return jb
    xq = random_curve(rng)
    psiq = random_curve(rng, shape=()).jet(1.0, 2)
    _, last = ladder(lambda jx: quadrotor_flat(jx, psiq, p), xq.jet(1.0, 4), ("f", "R", "Omega", "tau"), range(5))
    ck.true("quadrotor ladder: f and R need the acceleration, Omega the jerk, tau the snap",
            last == {"f": 2, "R": 2, "Omega": 3, "tau": 4}, str(last))

    # open loop is not feedback: a one degree tilt at release, and a 1% mass error
    def drift(state0, params=None, t_end=5.0, h=0.01):
        end = simulate(rb, state0, u_circle, h, int(round(t_end / h)), params=params)[-1]
        return float(np.linalg.norm(end.x - circle_reference(t_end)[0]))
    s_tilt = quad_state(rb, o0)
    s_tilt.R = s_tilt.R @ robots.expm_so3(np.radians(1.0) * E1)
    heavy = rb.Params(m=1.01 * p.m, J=p.J)
    rob = {"tilt_1deg_err_5s_m": drift(s_tilt), "mass_1pct_err_5s_m": drift(quad_state(rb, o0), heavy)}
    ck.true("open loop, 1 degree tilt at release: more than 1 m off after 5 s", rob["tilt_1deg_err_5s_m"] > 1.0,
            f"{rob['tilt_1deg_err_5s_m']:.2f} m")
    ck.true("open loop, 1% mass error: more than 1 m off after 5 s", rob["mass_1pct_err_5s_m"] > 1.0,
            f"{rob['mass_1pct_err_5s_m']:.2f} m")
    ck.record("open_loop_fragile", rob)

    # the singularity at zero thrust
    sing = {}
    for frac in (0.5, 0.9, 0.99):
        amp = frac * G / 4.0                                   # z = amp cos(2 t): peak downward acceleration frac g
        path = Trig([0.0, 0.0, 0.0], [[0.2, 0.0, amp]], [[2.0, 1.0, 2.0]], [[0.0, 0.0, np.pi / 2]])
        O = [quadrotor_flat(path.jet(t, 4), np.zeros(3), p) for t in np.linspace(-0.6, 0.6, 2401)]
        sing[f"{frac:g}"] = {"f_min_over_hover": min(o["f"] for o in O) / (p.m * p.g),
                             "Omega_max_rad_s": max(np.linalg.norm(o["Omega"]) for o in O),
                             "tau_max_Nm": max(np.linalg.norm(o["tau"]) for o in O)}
    ck.true("near free fall the angular velocity grows like jerk over thrust",
            abs(sing["0.99"]["Omega_max_rad_s"] - 1.6 / (0.01 * G)) < 0.05 * 1.6 / (0.01 * G)
            and sing["0.99"]["Omega_max_rad_s"] > 9 * sing["0.9"]["Omega_max_rad_s"],
            ", ".join(f"{v['Omega_max_rad_s']:.2f}" for v in sing.values()))
    ck.record("singularity", sing)
    # through free fall the thrust direction flips
    path = Trig([0.0, 0.0, 0.0], [[0.0, 0.0, 1.2 * G / 4.0]], [[1.0, 1.0, 2.0]], [[0.0, 0.0, np.pi / 2]])
    b3_top = quadrotor_flat(path.jet(0.0, 4), np.zeros(3), p)["R"][:, 2]
    b3_bot = quadrotor_flat(path.jet(np.pi / 2, 4), np.zeros(3), p)["R"][:, 2]
    ck.close("past free fall the map asks for an inverted vehicle", np.concatenate([b3_top, b3_bot]), [0, 0, -1, 0, 0, 1], 1e-12)

    # ---------------------------------------------------------------- cable-suspended load (thm10-load)
    rl = robots.get("quadrotor_load")
    pl = rl.params
    worst_res, worst_T, worst_xq = 0.0, 0.0, 0.0
    for _ in range(20):
        x, psi = random_curve(rng), random_curve(rng, shape=())
        t = rng.uniform(0, 10)
        o = load_flat(x.jet(t, 6), psi.jet(t, 2), pl)
        s = load_state(rl, o)
        acc = rl.dynamics(s, (o["f"], o["tau"]))
        worst_res = max(worst_res, *(np.max(np.abs(a - b)) for a, b in zip(acc, (o["a_L"], o["omega_dot"], o["Omega_dot"]))))
        worst_T = max(worst_T, abs(rl.tension(s, (o["f"], o["tau"])) - o["T"]),
                      np.max(np.abs(-o["T"] * o["q"] - pl.m_L * (o["a_L"] + G * E3))))
        worst_xq = max(worst_xq, np.max(np.abs(rl.quadrotor_position(s) - o["x_Q"])),
                       np.max(np.abs(o["x_Q"] - derivs(hand_position(x.jet(t, 2), pl.l))[0])), rl.on_manifold(s))
    ck.true("load: recovered state and inputs satisfy eq4-load in the library (20 random points)", worst_res < 1e-9,
            f"residual {worst_res:.1e}")
    ck.true("load: T of eq10-load-map equals eq4-load-tension, and -T q = m_L (x_L'' + g e3)", worst_T < 1e-10, f"err {worst_T:.1e}")
    ck.true("load: x_Q = x_L - l q is the hand of eq10-hand, and the state is on the manifold", worst_xq < 1e-10, f"err {worst_xq:.1e}")
    xl = random_curve(rng)
    _, last = ladder(lambda jx: load_flat(jx, psiq, pl), xl.jet(1.0, 6), ("q", "T", "omega", "x_Q", "f", "R", "Omega", "tau"), range(7))
    ck.true("load ladder: q, T, x_Q order 2; omega 3; f, R 4; Omega 5; tau 6",
            last == {"q": 2, "T": 2, "x_Q": 2, "omega": 3, "f": 4, "R": 4, "Omega": 5, "tau": 6}, str(last))

    eight = figure_eight()
    psi0 = np.zeros(3)

    def u_eight(t, path=eight):
        o = load_flat(path.jet(t, 6), psi0, pl)
        return o["f"], o["tau"]
    period = 2 * np.pi
    o0 = load_flat(eight.jet(0.0, 6), psi0, pl)
    errs = {}
    for h in (0.01, 0.005):
        n = int(round(2 * period / h))
        end = simulate(rl, load_state(rl, o0), u_eight, h, n)[-1]
        errs[h] = float(np.linalg.norm(end.x_L - eight(n * h)))
    order = math.log2(errs[0.01] / errs[0.005])
    ck.true("load open loop on the figure-eight, two laps: error below 1e-6 m, fourth order in h",
            errs[0.005] < 1e-6 and 3.5 < order < 4.5, f"{errs[0.005]:.1e} m, order {order:.2f}")

    def stats(path, n=1441, span=None):
        span = span or 2 * np.pi
        O = [load_flat(path.jet(t, 6), psi0, pl) for t in np.linspace(0, span, n)]
        return {"T_min_N": min(o["T"] for o in O), "T_max_N": max(o["T"] for o in O),
                "cable_angle_max_deg": max(deg(np.arccos(-o["q"][2])) for o in O),
                "f_min_N": min(o["f"] for o in O), "f_max_N": max(o["f"] for o in O),
                "tilt_max_deg": max(deg(np.arccos(o["R"][2, 2])) for o in O),
                "Omega_max_rad_s": max(np.linalg.norm(o["Omega"]) for o in O),
                "tau_max_Nm": max(np.linalg.norm(o["tau"]) for o in O),
                "x6_max": max(np.linalg.norm(derivs(path.jet(t, 6))[6]) for t in np.linspace(0, span, 200))}
    st = stats(eight)
    st.update(period_s=period, hover_thrust_N=(pl.m_Q + pl.m_L) * pl.g, weight_N=pl.m_L * pl.g,
              open_loop_err_m=errs[0.005], open_loop_order=order,
              lead_max_m=max(np.linalg.norm((load_flat(eight.jet(t, 6), psi0, pl)["x_Q"] - eight(t))[:2])
                             for t in np.linspace(0, period, 721)))
    ck.record("eight", {k: float(v) for k, v in st.items()})
    ck.true("figure-eight: cable taut and thrust positive throughout", st["T_min_N"] > 0.5 and st["f_min_N"] > 5,
            f"T in [{st['T_min_N']:.2f}, {st['T_max_N']:.2f}] N, f in [{st['f_min_N']:.1f}, {st['f_max_N']:.1f}] N")

    # faster and faster: the tension reaches zero at the crossing when 4 c rate^2 = g
    rate_c = math.sqrt(G / (4 * 0.15))
    A0 = derivs(figure_eight(rate=rate_c).jet(0.0, 2))[2] + G * E3
    ck.close("figure-eight at the critical rate: the load is in free fall at the crossing", A0, 0.0, 1e-12)
    speed = {}
    for rate in (1.0, 1.5, 2.0, 3.0, 3.9):
        sp = stats(figure_eight(rate=rate), span=2 * np.pi / rate, n=2881)
        speed[f"{rate:g}"] = {k: float(sp[k]) for k in ("T_min_N", "cable_angle_max_deg", "tilt_max_deg", "f_min_N", "f_max_N", "tau_max_Nm")}
    ck.true("as the rate approaches the critical one the tension falls and the torque grows without bound",
            speed["3.9"]["T_min_N"] < 0.3 and speed["3.9"]["tau_max_Nm"] > 50 * speed["2"]["tau_max_Nm"],
            f"T_min {speed['3.9']['T_min_N']:.2f} N, tau {speed['2']['tau_max_Nm']:.2f} -> {speed['3.9']['tau_max_Nm']:.0f} N m")
    ck.true("at twice the rate the map asks the quadrotor to fly inverted", speed["2"]["tilt_max_deg"] > 90 > speed["1.5"]["tilt_max_deg"],
            f"largest tilt {speed['1.5']['tilt_max_deg']:.0f} deg, then {speed['2']['tilt_max_deg']:.0f} deg")
    ck.record("speed", speed)
    ck.record("critical_rate_rad_s", rate_c)
    # before the cable goes slack, the quadrotor's own thrust vanishes at the crossing
    from scipy.optimize import brentq

    def F_crossing(r):
        return load_flat(figure_eight(rate=r).jet(0.0, 6), psi0, pl)["F"]
    rate_f = brentq(lambda r: F_crossing(r)[2], 1.5, 2.0)
    s_f = load_state(rl, load_flat(figure_eight(rate=rate_f - 1e-6).jet(0.0, 6), psi0, pl))
    q_dot = np.cross(s_f.omega, s_f.q)
    ck.true("figure-eight at 1.72 rad/s: the thrust vector vanishes at the crossing, while the tension is 0.82 of the weight",
            np.linalg.norm(F_crossing(rate_f)) < 1e-9 and abs(rate_f - 1.716) < 1e-3
            and abs(pl.m_L * (G - 0.6 * rate_f**2) / (pl.m_L * G) - 0.82) < 0.005,
            f"rate {rate_f:.4f}, |F| = {np.linalg.norm(F_crossing(rate_f)):.1e} N")
    ck.close("there the centripetal term m_Q l |q_dot|^2 equals (m_Q + m_L) |x_L'' + g e3|",
             pl.m_Q * pl.l * q_dot @ q_dot, (pl.m_Q + pl.m_L) * (G - 0.6 * rate_f**2), 1e-3)
    ck.record("zero_thrust_rate_rad_s", rate_f)

    # a ripple on the load path: 1 mm at 2 Hz
    w_r = 2 * np.pi * 2.0
    ripple = Trig([0.0, 0.0, 0.0], [[1.2, 0.6, 0.15], [1e-3, 0.0, 0.0]],
                  [[1.0, 2.0, 2.0], [w_r, 1.0, 1.0]], [[0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]])
    sr = stats(ripple, n=5761)
    rip = {"tau_max_clean_Nm": st["tau_max_Nm"], "tau_max_ripple_Nm": sr["tau_max_Nm"],
           "x6_clean": st["x6_max"], "x6_ripple": 1e-3 * w_r**6, "acc_ripple": 1e-3 * w_r**2,
           "Omega_max_clean_rad_s": st["Omega_max_rad_s"], "Omega_max_ripple_rad_s": sr["Omega_max_rad_s"],
           "tilt_max_ripple_deg": sr["tilt_max_deg"]}
    ck.true("a 1 mm ripple at 2 Hz on the load path multiplies the torque by more than 50",
            rip["tau_max_ripple_Nm"] > 50 * rip["tau_max_clean_Nm"],
            f"{rip['tau_max_clean_Nm']:.3f} -> {rip['tau_max_ripple_Nm']:.2f} N m")
    ck.record("ripple", {k: float(v) for k, v in rip.items()})

    # open loop with the cable 2 degrees off at release
    s_off = load_state(rl, o0)
    s_off.q = robots.expm_so3(np.radians(2.0) * E1) @ s_off.q
    s_off.omega = s_off.omega - (s_off.omega @ s_off.q) * s_off.q
    n = int(round(2 * period / 0.01))
    run = simulate(rl, s_off, u_eight, 0.01, n, every=10)
    e_off = [float(np.linalg.norm(s.x_L - eight(k * 0.1))) for k, s in enumerate(run)]
    ck.true("load open loop, cable 2 degrees off at release: more than 0.1 m off within two laps", max(e_off) > 0.1,
            f"max {max(e_off):.2f} m")
    ck.record("load_fragile", {"cable_2deg_max_err_m": max(e_off), "laps": 2})

    # ---------------------------------------------------------------- chain (thm10-chain)
    chain = {}
    for n in (1, 2, 3):
        rc = robots.get("quadrotor_cable", n=n, attach=(0,))
        pc = rc.params
        mass = rc.joint_masses()
        worst = 0.0
        for _ in range(10):
            x, psi = random_curve(rng, size=0.5), random_curve(rng, shape=())
            t = rng.uniform(0, 10)
            o = chain_flat(x.jet(t, 2 * n + 4), psi.jet(t, 2), mass, pc.l, pc.J[0])
            s = cable_state(rc, o)
            acc = rc.dynamics(s, (o["f"], o["tau"]))
            worst = max(worst, *(np.max(np.abs(a - b)) for a, b in zip(acc, (o["a_0"], o["omega_dot"], o["Omega_dot"]))),
                        np.max(np.abs(rc.positions(s)[-1] - x(t))), rc.on_manifold(s))
        ck.true(f"chain, n = {n}: recursion from the free end satisfies the library's quadrotor_cable model", worst < 1e-8,
                f"residual {worst:.1e}")
        xc = random_curve(rng, size=0.5)
        psic = random_curve(rng, shape=()).jet(0.5, 2)
        _, last = ladder(lambda jx: {k: v for k, v in chain_flat(jx, psic, mass, pc.l, pc.J[0]).items()},
                         xc.jet(0.5, 2 * n + 4), ("x_0", "f", "Omega", "tau"), range(2 * n + 5))
        ck.true(f"chain, n = {n}: x_0 needs {2 * n} derivatives of the free end, f {2 * n + 2}, Omega {2 * n + 3}, tau {2 * n + 4}",
                last == {"x_0": 2 * n, "f": 2 * n + 2, "Omega": 2 * n + 3, "tau": 2 * n + 4}, str(last))
        chain[str(n)] = {"derivatives": 2 * n + 4, "residual": float(worst)}
    # one link is the cable-suspended load
    rc1 = robots.get("quadrotor_cable", n=1, attach=(0,), params=robots.flexible_cable.CableParams(
        n=1, attach=(0,), m=np.array([0.0, pl.m_L]), l=np.array([pl.l]), m_Q=np.array([pl.m_Q]), J=np.array([pl.J])))
    x, psi = random_curve(rng), random_curve(rng, shape=())
    o1 = chain_flat(x.jet(0.3, 6), psi.jet(0.3, 2), rc1.joint_masses(), rc1.params.l, pl.J)
    ol = load_flat(x.jet(0.3, 6), psi.jet(0.3, 2), pl)
    ck.close("chain with one link is eq10-load-map", np.concatenate([o1["q"][0] - ol["q"], o1["T"] - ol["T"], o1["tau"][0] - ol["tau"],
                                                                      [o1["f"][0] - ol["f"]], o1["x_0"] - ol["x_Q"]]), 0.0, 1e-9)
    # open loop with three links
    rc3, path3 = chain_demo()
    mass3, pc3 = rc3.joint_masses(), rc3.params

    def u_chain(t):
        o = chain_flat(path3.jet(t, 10), psi0, mass3, pc3.l, pc3.J[0])
        return o["f"], o["tau"]
    oc = chain_flat(path3.jet(0.0, 10), psi0, mass3, pc3.l, pc3.J[0])
    errs = {}
    for h in (0.01, 0.005):
        nst = int(round(2 * np.pi / CHAIN_RATE / h))
        end = simulate(rc3, cable_state(rc3, oc), u_chain, h, nst)[-1]
        errs[h] = float(np.linalg.norm(rc3.positions(end)[-1] - path3(nst * h)))
    order = math.log2(errs[0.01] / errs[0.005])
    ck.true("three links, open loop for one lap: the free end is back on its curve, fourth order in h",
            errs[0.005] < 1e-4 and 3.3 < order < 4.7, f"{errs[0.005]:.1e} m, order {order:.2f}")
    Oc = [chain_flat(path3.jet(t, 10), psi0, mass3, pc3.l, pc3.J[0]) for t in np.linspace(0, 2 * np.pi / CHAIN_RATE, 361)]
    bend = max(deg(np.arccos(np.clip(o["q"][0] @ o["q"][2], -1, 1))) for o in Oc)
    chain["three_links"] = {"open_loop_err_m": errs[0.005], "tau_max_Nm": float(max(np.linalg.norm(o["tau"]) for o in Oc)), "order": order, "T_min_N": float(min(o["T"].min() for o in Oc)),
                            "bend_max_deg": float(bend), "tilt_max_deg": float(max(deg(np.arccos(o["R"][0][2, 2])) for o in Oc)),
                            "x10_max": float(max(np.linalg.norm(derivs(path3.jet(t, 10))[10]) for t in np.linspace(0, 6.28, 100))),
                            "masses_kg": [float(m) for m in mass3], "lengths_m": [float(v) for v in pc3.l]}
    ck.true("three links: every tension positive along the lap", chain["three_links"]["T_min_N"] > 0.5,
            f"min {chain['three_links']['T_min_N']:.2f} N")
    # two quadrotors: one on top, one in the middle; the tension vector above the middle one is a flat output
    rc2 = robots.get("quadrotor_cable", n=2, attach=(0, 1))
    pc2 = rc2.params
    mass2 = rc2.joint_masses()
    worst = 0.0
    for _ in range(10):
        x, psi_a, psi_b = random_curve(rng, size=0.5), random_curve(rng, shape=()), random_curve(rng, shape=())
        wig = random_curve(rng, size=0.4)
        t = rng.uniform(0, 10)
        pulls = {1: np.array([1.5, 0.5, -4.0]) * np.eye(5)[0][:, None] + wig.jet(t, 4) * np.array([1.0] + [1.0] * 4)[:, None]}
        o = chain_flat(x.jet(t, 8), [psi_a.jet(t, 2), psi_b.jet(t, 2)], mass2, pc2.l, pc2.J, attach=(0, 1), pulls=pulls)
        s = cable_state(rc2, o)
        acc = rc2.dynamics(s, (o["f"], o["tau"]))
        worst = max(worst, *(np.max(np.abs(a - b)) for a, b in zip(acc, (o["a_0"], o["omega_dot"], o["Omega_dot"]))),
                    np.max(np.abs(rc2.positions(s)[-1] - x(t))))
    ck.true("two quadrotors on one cable: free end, two headings and one tension vector determine everything", worst < 1e-8,
            f"residual {worst:.1e}")
    # the layout of the thesis's own example: quadrotors at both ends, the tension of the last link given
    rcb = robots.get("quadrotor_cable", n=2, attach=(0, 2))
    pcb, massb, worst_b = rcb.params, rcb.joint_masses(), 0.0
    for _ in range(10):
        x, psi_a, psi_b = random_curve(rng, size=0.5), random_curve(rng, shape=()), random_curve(rng, shape=())
        wig = random_curve(rng, size=0.4)
        t = rng.uniform(0, 10)
        pull = wig.jet(t, 6)
        pull[0] = pull[0] + np.array([1.5, 0.5, -4.0])
        o = chain_flat(x.jet(t, 8), [psi_a.jet(t, 2), psi_b.jet(t, 2)], massb, pcb.l, pcb.J, attach=(0, 2), pulls={2: pull})
        s = cable_state(rcb, o)
        acc = rcb.dynamics(s, (o["f"], o["tau"]))
        worst_b = max(worst_b, *(np.max(np.abs(a - b)) for a, b in zip(acc, (o["a_0"], o["omega_dot"], o["Omega_dot"]))),
                      np.max(np.abs(rcb.positions(s)[-1] - x(t))))
    ck.true("quadrotors at both ends of a two-link cable: start position, two headings, one tension vector", worst_b < 1e-8,
            f"residual {worst_b:.1e}")
    chain["two_quadrotors_residual"] = float(max(worst, worst_b))
    ck.true("two-quadrotor residuals below 1e-12", max(worst, worst_b) < 1e-12, f"{max(worst, worst_b):.1e}")
    ck.true("count of flat outputs equals count of inputs: 3 + 3 (n_Q - 1) + n_Q = 4 n_Q", all(3 + 3 * (k - 1) + k == 4 * k for k in range(1, 6)))
    ck.record("chain", chain)

    # ---------------------------------------------------------------- a system that is not flat (warn10-recipe)
    import sympy as sp
    xs, th, vs, ws, ls, gs = sp.symbols("x theta v w l g", real=True)
    zs = sp.Matrix([xs, th, vs, ws])
    drift_f = sp.Matrix([vs, ws, 0, -gs / ls * sp.sin(th)])      # pendulum hanging from a cart, cart acceleration as input
    in_g = sp.Matrix([0, 0, 1, -sp.cos(th) / ls])

    def bracket(a, b):
        return b.jacobian(zs) * a - a.jacobian(zs) * b
    ad1 = bracket(drift_f, in_g)
    ad2 = bracket(drift_f, ad1)
    ad3 = bracket(drift_f, ad2)
    at = {th: 0.7, ws: 0.4, ls: 1.0, gs: 9.81, xs: 0.0, vs: 0.0}
    ctrb = sp.Matrix.hstack(in_g, ad1, ad2, ad3).subs(at)
    dist = sp.Matrix.hstack(in_g, ad1, ad2, bracket(in_g, ad1), bracket(ad1, ad2), bracket(in_g, ad2)).subs(at)
    ck.true("cart with a pendulum, one input: controllable, but span{g, ad_f g, ad_f^2 g} is not involutive, so not flat",
            ctrb.rank() == 4 and sp.Matrix.hstack(in_g, ad1, ad2).subs(at).rank() == 3 and dist.rank() == 4)

    # ---------------------------------------------------------------- minimum snap (sec10-trajectories)
    W = WAYPOINTS
    ms = {}
    tm = chord_times(W, 6.0)
    qp, sp = min_derivative_spline(W, tm, 4, "qp"), min_derivative_spline(W, tm, 4, "spline")
    ck.close("minimum snap: the quadratic program and the degree-7 spline system give the same curve", qp.coeff, sp.coeff, 1e-8)
    worst_w = max(np.max(np.abs(qp(tk) - wk)) for tk, wk in zip(tm, W))
    ends = np.max(np.abs(np.concatenate([derivs(qp.jet(0.0, 3))[1:], derivs(qp.jet(6.0, 3))[1:]])))
    jumps = [np.max(np.abs(derivs(qp.jet_side(i, 1.0, 7)) - derivs(qp.jet_side(i + 1, 0.0, 7))), axis=1) for i in range(3)]
    jumps = np.max(np.array(jumps), axis=0)
    ck.true("minimum snap: passes through the waypoints, at rest at both ends", worst_w < 1e-10 and ends < 1e-8,
            f"{worst_w:.1e}, {ends:.1e}")
    ck.true("minimum snap: continuity imposed to order 3, obtained to order 6, and the seventh derivative jumps",
            np.max(jumps[:7]) < 1e-6 and jumps[7] > 1.0, f"jump in orders 4 to 6: {np.max(jumps[4:7]):.1e}, order 7: {jumps[7]:.1f}")
    # optimality: any feasible perturbation raises the cost
    A, _ = _constraints(4, np.diff(tm), 4, 3)
    null = np.linalg.svd(A)[2][np.linalg.matrix_rank(A):].T
    base, raised = qp.cost(), []
    for _ in range(20):
        dc = (null @ rng.normal(size=(null.shape[1], 3))).reshape(4, 8, 3)
        raised.append(Spline(qp.coeff + 1e-2 * dc, tm, 4).cost() - base)
    ck.true("minimum snap: 20 random feasible perturbations all raise the cost", min(raised) > 0, f"smallest increase {min(raised):.1e}")
    def quad_stats(sp_, t_f, n=1201):
        O = [quadrotor_flat(sp_.jet(t, 4), np.zeros(3), p) for t in np.linspace(0, t_f, n)]
        return {"f_min_over_hover": min(o["f"] for o in O) / (p.m * p.g), "f_max_over_hover": max(o["f"] for o in O) / (p.m * p.g),
                "tilt_max_deg": max(deg(np.arccos(o["R"][2, 2])) for o in O),
                "Omega_max_rad_s": max(np.linalg.norm(o["Omega"]) for o in O),
                "tau_max_Nm": max(np.linalg.norm(o["tau"]) for o in O)}
    for t_f in (6.0, 4.0):
        spt = min_derivative_spline(W, chord_times(W, t_f), 4)
        ms[f"{t_f:g}"] = {k: float(v) for k, v in quad_stats(spt, t_f).items()}
        ms[f"{t_f:g}"]["cost"] = spt.cost()
    ck.close("time scaling: the snap cost scales as t_f^(-7)", ms["4"]["cost"] / ms["6"]["cost"], 1.5**7, 1e-6 * 1.5**7)
    ck.true("shorter time, harder flight: tilt and thrust grow", ms["4"]["tilt_max_deg"] > ms["6"]["tilt_max_deg"] + 10)
    # fly it open loop on the library
    def u_ms(t):
        o = quadrotor_flat(qp.jet(min(t, 6.0), 4), np.zeros(3), p)
        return o["f"], o["tau"]
    end = simulate(rb, quad_state(rb, quadrotor_flat(qp.jet(0.0, 4), np.zeros(3), p)), u_ms, 0.005, 1200)[-1]
    ms["open_loop_err_m"] = float(np.linalg.norm(end.x - W[-1]))
    ck.true("minimum-snap flight, open loop: arrives at the last waypoint", ms["open_loop_err_m"] < 1e-6, f"{ms['open_loop_err_m']:.1e} m")
    # the load through the same waypoints: minimize the sixth derivative
    tl = chord_times(W, 8.0)
    q6, s6 = min_derivative_spline(W, tl, 6, "qp"), min_derivative_spline(W, tl, 6, "spline")
    ck.close("sixth-derivative minimizer (degree 11): program and spline system agree", q6.coeff, s6.coeff, 1e-5)
    jl = [np.max(np.abs(derivs(q6.jet_side(i, 1.0, 11)) - derivs(q6.jet_side(i + 1, 0.0, 11))), axis=1) for i in range(3)]
    jl = np.max(np.array(jl), axis=0)
    ck.true("load curve: continuous to order 10 at the waypoints, so x_L^(6) is continuous", np.max(jl[:7]) < 1e-5,
            f"jump up to order 6: {np.max(jl[:7]):.1e}")
    Ol = [load_flat(q6.jet(t, 6), psi0, pl) for t in np.linspace(0, 8.0, 1601)]
    # a minimum-snap curve used for the load does not start from hover: x_L^(4) is free at its ends
    q4 = min_derivative_spline(W, tl, 4, "qp")
    start4, start6 = load_flat(q4.jet(0.0, 6), psi0, pl), load_flat(q6.jet(0.0, 6), psi0, pl)
    tilt4, tilt6 = deg(np.arccos(start4["R"][2, 2])), deg(np.arccos(min(1.0, start6["R"][2, 2])))
    ms["load"] = {"t_f_s": 8.0, "T_min_N": float(min(o["T"] for o in Ol)), "T_max_N": float(max(o["T"] for o in Ol)),
                  "cable_angle_max_deg": float(max(deg(np.arccos(-o["q"][2])) for o in Ol)),
                  "tilt_max_deg": float(max(deg(np.arccos(o["R"][2, 2])) for o in Ol)),
                  "tau_max_Nm": float(max(np.linalg.norm(o["tau"]) for o in Ol)),
                  "start_tilt_minsnap_deg": float(tilt4), "start_Omega_minsnap_rad_s": float(np.linalg.norm(start4["Omega"])),
                  "start_tilt_sixth_deg": float(tilt6), "start_Omega_sixth_rad_s": float(np.linalg.norm(start6["Omega"]))}
    ck.true("load at rest at both ends: the sixth-derivative curve starts in hover, the minimum-snap curve does not",
            tilt4 > 1.0 and tilt6 < 1e-5 and np.linalg.norm(start6["Omega"]) < 1e-8,
            f"tilt {tilt4:.1f} deg and {np.linalg.norm(start4['Omega']):.2f} rad/s against {tilt6:.0e} deg")
    ck.record("minsnap", ms)
    ck.record("waypoints", {"points_m": W.tolist(), "times_6s": [float(v) for v in tm]})

    # ---------------------------------------------------------------- feedforward (sec10-feedforward)
    ff = {}
    for mode, kappa in (("zero", 1.0), ("flat", 1.0), ("exact", 1.0), ("zero", 1.5), ("flat", 1.5)):
        ts, es = cascade_run(rb, mode, kappa=kappa)
        ff[f"{mode}_{kappa:g}"] = {"diverged": bool(es[-1] > 5.0), "t_end_s": float(ts[-1]),
                                   "max_err_last_5s_m": None if es[-1] > 5.0 else float(np.max(es[ts > 15.0]))}
    ck.true("zero feedforward, Lee's gains: diverges, as in warn6-feedforward", ff["zero_1"]["diverged"], f"at {ff['zero_1']['t_end_s']:.1f} s")
    ck.true("exact feedforward, Lee's gains: converges", ff["exact_1"]["max_err_last_5s_m"] < 1e-6, f"{ff['exact_1']['max_err_last_5s_m']:.1e} m")
    ck.true("flat feedforward of the reference, Lee's gains: still diverges", ff["flat_1"]["diverged"], f"at {ff['flat_1']['t_end_s']:.1f} s")
    ck.true("attitude loop 1.5 times faster: flat feedforward converges to zero, zero feedforward leaves centimeters",
            ff["flat_1.5"]["max_err_last_5s_m"] < 1e-4 and ff["zero_1.5"]["max_err_last_5s_m"] > 0.01,
            f"{ff['flat_1.5']['max_err_last_5s_m']:.1e} m against {ff['zero_1.5']['max_err_last_5s_m']:.4f} m")
    ck.record("feedforward", ff)
    # along the reference the flat feedforward is the exact one
    ref, o = circle_flat(1.3, p)
    u_lib = rb.tracking_controller(quad_state(rb, dict(x=ref[0], v=ref[1], R=o["R"], Omega=o["Omega"])), ref)
    ck.close("on the reference, the library's cascade returns the flat inputs f_d and tau_d", np.concatenate([[u_lib[0] - o["f"]], u_lib[1] - o["tau"]]), 0.0, 1e-5)

    # ------------------------------------------------- heading and yaw (sec10-quad, Appendix A)
    # R = Rz(yaw) Ry(pitch) Rx(roll) of eq-zyx. The heading psi has R e2 perpendicular to
    # Rz(psi) e1. Then tan(psi - yaw) = -sin(pitch) tan(roll).
    from common import Rx, Ry, Rz
    worst_h, worst_0 = 0.0, 0.0
    for _ in range(50):
        yaw, pitch, roll = rng.uniform(-1.2, 1.2, size=3)
        for th, ph, zero in ((pitch, roll, False), (pitch, 0.0, True), (0.0, roll, True)):
            b2 = (Rz(yaw) @ Ry(th) @ Rx(ph))[:, 1]
            psi = math.atan2(-b2[0], b2[1])
            worst_h = max(worst_h, abs(math.tan(psi - yaw) + math.sin(th) * math.tan(ph)))
            if zero:
                worst_0 = max(worst_0, abs(psi - yaw))
    ck.close("heading and yaw: tan(psi - yaw) = -sin(pitch) tan(roll)", worst_h, 0.0, 1e-12)
    ck.close("heading equals yaw when the roll or the pitch is zero", worst_0, 0.0, 1e-12)

    ck.finish()


if __name__ == "__main__":
    main()
