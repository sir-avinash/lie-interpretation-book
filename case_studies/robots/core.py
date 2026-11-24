"""Shared pieces of the robot model library.

Everything here follows the conventions of Appendix A of the book:

* the world frame has e3 up and gravity acts along -e3;
* a rotation matrix R maps body-frame components to world-frame components,
  with R_dot = R hat(Omega) and Omega in the body frame;
* a direction q in S^2 has q_dot = omega x q with omega in the world frame and
  omega . q = 0;
* errors take the actual configuration first: e_R(R, Rd), e_q(q, qd).

A robot is described by a list of *factors*. Each factor names one piece of the
configuration manifold, the field of the state that stores it and the field
that stores its velocity. The generic code in this file (the integration step,
the manifold residual, the random state) is written once against that list.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

E1, E2, E3 = np.eye(3)
I3 = np.eye(3)
GRAVITY = 9.81


# ------------------------------------------------------------------ operators
def hat(x):
    """hat(x) y = x cross y."""
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def vee(A):
    """Inverse of hat."""
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def expm_so3(x):
    """exp(hat(x)), Rodrigues' formula (prop2-rodrigues)."""
    angle = np.linalg.norm(x)
    if angle < 1e-12:
        return I3 + hat(x)
    K = hat(x / angle)
    return I3 + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def log_so3(R):
    """Rotation vector of R: log(R) with the vee map applied."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return vee(R - R.T) / 2.0
    if np.pi - angle < 1e-6:
        S = (R + I3) / 2.0
        a = np.sqrt(np.clip(np.diag(S), 0.0, None))
        k = int(np.argmax(a))
        return angle * S[:, k] / a[k]
    return angle * vee(R - R.T) / (2.0 * np.sin(angle))


def unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)


def tangent(q, x):
    """(I - q q') x: the part of x in the tangent plane at q (eq4-sph-proj)."""
    return x - (x @ q) * q


# ------------------------------------------------- errors of Chapter 3 (code names of Appendix A)
def Psi_R(R, Rd):
    return 0.5 * np.trace(I3 - Rd.T @ R)


def e_R(R, Rd):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def e_Omega(R, Omega, Rd, Omegad):
    return Omega - R.T @ Rd @ Omegad


def Psi_q(q, qd):
    return 1.0 - q @ qd


def e_q(q, qd):
    return np.cross(qd, q)


def e_omega(q, omega, omegad):
    return omega - tangent(q, omegad)


def attitude_tracking_torque(R, Omega, Rd, Omegad, Omegad_dot, J, k_R, k_Omega):
    """The attitude tracking law, eq6-law:

    tau = -k_R e_R - k_Omega e_Omega
          + Omega x J Omega - J (Omega x R'Rd Omegad - R'Rd Omegad_dot).
    """
    return (-k_R * e_R(R, Rd) - k_Omega * e_Omega(R, Omega, Rd, Omegad)
            + np.cross(Omega, J @ Omega)
            - J @ (np.cross(Omega, R.T @ Rd @ Omegad) - R.T @ Rd @ Omegad_dot))


# -------------------------------------------------------------------- factors
@dataclass(frozen=True)
class Factor:
    """One factor of a configuration manifold.

    kind      "R" (flat: a scalar or a vector), "S2" or "SO3"
    config    name of the state field that holds the configuration
    velocity  name of the state field that holds its velocity
    size      for kind "R": 1 for a scalar, 3 for a vector in R^3
    count     how many copies
    stacked   True when the fields are stacked arrays, one row per copy
    """
    kind: str
    config: str
    velocity: str
    size: int = 3
    count: int = 1
    stacked: bool = False

    @property
    def dim(self):
        return {"R": self.size, "S2": 2, "SO3": 3}[self.kind] * self.count

    @property
    def stored(self):
        return {"R": self.size, "S2": 3, "SO3": 9}[self.kind] * self.count


def _each(stacked, fn, *arrays):
    """Apply fn to one element, or row by row to stacked arrays."""
    if not stacked:
        return fn(*arrays)
    return np.array([fn(*row) for row in zip(*arrays)])


def _dexpinv(x, y, sign):
    """y + sign (x cross y)/2 + x cross (x cross y)/12.

    Rate of the rotation vector x inside a step when the angular velocity is y:
    sign = +1 for R = R_k exp(hat(x)) with body angular velocity y,
    sign = -1 for q = exp(hat(x)) q_k with world-frame angular velocity y.
    """
    return y + sign * 0.5 * np.cross(x, y) + np.cross(x, np.cross(x, y)) / 12.0


class Robot:
    """Base class. A robot module fills in the class attributes and four methods.

    Class attributes
        name      registry name
        manifold  the configuration manifold as a string, e.g. "R^3 x S^2 x SO(3)"
        labels    dict of book labels that define the model
        Params    dataclass of parameters (book symbols, SI units)
        State     dataclass of the state (Appendix A code names)
    Instance attributes
        factors   list of Factor, in the order of the manifold string
        params    the default parameters of this instance
    Methods to write
        dynamics(state, u, params)   accelerations, one per factor, in factor order
        energy(state, params)        kinetic plus potential energy
        zero_input()                 an input of the right shape, all zeros
        lagrangian()                 geomech (L, dW, variables, inputs)
    Methods provided here
        step(state, u, h, params)    one fourth-order step that stays on the manifold
        on_manifold(state)           largest violation of the manifold constraints
        random_state(rng)            a random state on the manifold
        rollout(state, u, h, n)      n steps, returns the list of states
    """
    name = ""
    manifold = ""
    labels: dict = {}
    Params: Any = None
    State: Any = None
    factors: list = []

    def __init__(self, params=None):
        self.params = params if params is not None else self.Params()

    # -- to be written by each robot ------------------------------------
    def dynamics(self, state, u, params=None):
        raise NotImplementedError

    def energy(self, state, params=None):
        raise NotImplementedError

    def zero_input(self):
        raise NotImplementedError

    def lagrangian(self):
        raise NotImplementedError

    # -- generic --------------------------------------------------------
    @property
    def dim(self):
        """Dimension of the configuration manifold (degrees of freedom)."""
        return sum(f.dim for f in self.factors)

    @property
    def stored(self):
        """How many numbers store a configuration."""
        return sum(f.stored for f in self.factors)

    def on_manifold(self, state):
        """Largest violation of |q| = 1, q . omega = 0 and R'R = I, det R = 1."""
        worst = 0.0
        for f in self.factors:
            c, v = getattr(state, f.config), getattr(state, f.velocity)
            if f.kind == "S2":
                c, v = np.reshape(c, (-1, 3)), np.reshape(v, (-1, 3))
                worst = max(worst, np.max(np.abs(np.linalg.norm(c, axis=1) - 1.0)),
                            np.max(np.abs(np.sum(c * v, axis=1))))
            elif f.kind == "SO3":
                for R in np.reshape(c, (-1, 3, 3)):
                    worst = max(worst, np.max(np.abs(R.T @ R - I3)),
                                abs(np.linalg.det(R) - 1.0))
        return float(worst)

    def _moved(self, state, increments, velocities):
        """The state reached from `state` by one increment per factor."""
        fields = {}
        for f, d, v in zip(self.factors, increments, velocities):
            c = getattr(state, f.config)
            if f.kind == "R":
                fields[f.config] = c + d
            elif f.kind == "S2":
                fields[f.config] = _each(f.stacked, lambda x, q: expm_so3(x) @ q, d, c)
            else:
                fields[f.config] = _each(f.stacked, lambda x, R: R @ expm_so3(x), d, c)
            fields[f.velocity] = v
        return replace(state, **fields)

    def _rates(self, increments, velocities):
        rates = []
        for f, d, v in zip(self.factors, increments, velocities):
            if f.kind == "R":
                rates.append(v)
            else:
                sign = -1.0 if f.kind == "S2" else 1.0
                rates.append(_each(f.stacked, lambda x, y: _dexpinv(x, y, sign), d, v))
        return rates

    def step(self, state, u, h, params=None):
        """Advance by h with the input u held constant.

        Classical fourth-order Runge-Kutta stages are run on one increment per
        factor (a displacement, or a rotation vector), and the configuration
        is updated by the exponential step eq4-lie-euler,

            R_{k+1} = R_k exp(hat(theta)),    q_{k+1} = exp(hat(phi)) q_k,

        so |q| = 1 and R'R = I hold to rounding error for any h. The velocity
        of each direction is projected onto the tangent plane of the new q at
        the end, which keeps q . omega = 0 exact as well.
        """
        params = params if params is not None else self.params
        zero = [0.0 * np.asarray(getattr(state, f.velocity), float) for f in self.factors]
        vel0 = [np.asarray(getattr(state, f.velocity), float) for f in self.factors]

        def slope(inc, vel):
            acc = self.dynamics(self._moved(state, inc, vel), u, params)
            return self._rates(inc, vel), [np.asarray(a, float) for a in acc]

        def add(a, b, s):
            return [x + s * y for x, y in zip(a, b)]

        k1 = slope(zero, vel0)
        k2 = slope(add(zero, k1[0], h / 2), add(vel0, k1[1], h / 2))
        k3 = slope(add(zero, k2[0], h / 2), add(vel0, k2[1], h / 2))
        k4 = slope(add(zero, k3[0], h), add(vel0, k3[1], h))
        inc = [h / 6 * (a + 2 * b + 2 * c + d) for a, b, c, d in zip(k1[0], k2[0], k3[0], k4[0])]
        vel = [v + h / 6 * (a + 2 * b + 2 * c + d)
               for v, a, b, c, d in zip(vel0, k1[1], k2[1], k3[1], k4[1])]
        new = self._moved(state, inc, vel)
        fix = {}
        for f in self.factors:
            if f.kind == "S2":
                fix[f.velocity] = _each(f.stacked, lambda q, w: tangent(q, w),
                                        getattr(new, f.config), getattr(new, f.velocity))
        return replace(new, **fix)

    def rollout(self, state, u, h, n, params=None):
        """n steps of size h. u is a constant input or a function u(t, state)."""
        states = [state]
        for k in range(n):
            uk = u(k * h, states[-1]) if callable(u) else u
            states.append(self.step(states[-1], uk, h, params))
        return states

    def random_state(self, rng, speed=1.0):
        """A random state on the manifold (positions and velocities of order one)."""
        fields = {}
        for f in self.factors:
            n = f.count
            if f.kind == "R":
                shape = (n, f.size) if f.stacked else (f.size,)
                c, v = rng.normal(size=shape), speed * rng.normal(size=shape)
                if f.size == 1:
                    c, v = np.abs(c) + 0.5, v       # a length: keep it positive
                    c, v = (c[:, 0], v[:, 0]) if f.stacked else (float(c[0]), float(v[0]))
            elif f.kind == "S2":
                c = np.array([unit(x) for x in rng.normal(size=(n, 3))])
                v = np.array([tangent(q, x) for q, x in zip(c, speed * rng.normal(size=(n, 3)))])
                if not f.stacked:
                    c, v = c[0], v[0]
            else:
                c = np.array([expm_so3(x) for x in rng.normal(size=(n, 3))])
                v = speed * rng.normal(size=(n, 3))
                if not f.stacked:
                    c, v = c[0], v[0]
            fields[f.config], fields[f.velocity] = c, v
        return self.State(**fields)

    def __repr__(self):
        return f"<{self.name}: {self.manifold}, dimension {self.dim}>"
