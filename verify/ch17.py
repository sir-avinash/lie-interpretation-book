"""Chapter 17 verification: one quadrotor through the whole recipe, in one mission.

The mission is assembled from the blocks that the earlier chapters verified:

* the vehicle, its step and its cascade: the robot model library (case_studies/robots);
* minimum snap and the flat map: verify/ch10.py (imported);
* L1 adaptation and the small QP solver: verify/ch08.py (imported);
* the augmented barrier: a copy of verify/ch08.py with the vehicle and the obstacle as
  arguments, checked here against the original;
* the invariant extended Kalman filter: the group operations of verify/ch14.py
  (imported), wrapped in a filter for one vehicle and checked against run_filters;
* error-state MPC: zoh, condense and box_qp of verify/ch13.py (imported).

Run:  python verify/ch17.py
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "case_studies"))

import robots  # noqa: E402
from common import Checker, expm_so3, hat, random_rotation, vee  # noqa: E402
from robots.core import Psi_R, attitude_tracking_torque, e_R, log_so3  # noqa: E402
from robots.quadrotor import desired_attitude  # noqa: E402


def _load(chapter):
    spec = importlib.util.spec_from_file_location(f"ch{chapter}_verify", HERE / f"ch{chapter}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V8, V10, V13, V14 = _load("08"), _load("10"), _load("13"), _load("14")
E1, E2, E3 = np.eye(3)
I3 = np.eye(3)


def unit(x):
    return np.asarray(x, float) / np.linalg.norm(x)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# =====================================================================
# 1. The mission
# =====================================================================
@dataclass
class Mission:
    """Every number of the mission. Change one and fly again."""
    h: float = 0.005                               # s: time step, IMU period, sampling period
    t_carry: float = 6.0                           # s: carried by hand, motors off, filter running
    t_hold: float = 2.5                            # s: after the release the reference waits at the first waypoint
    t_course: float = 6.0                          # s: duration of the minimum-snap course
    t_rest: float = 2.0                            # s: hover at the target
    waypoints: np.ndarray = field(default_factory=lambda: np.array(
        [[0.0, 0.0, 3.0], [3.0, 1.0, 3.6], [6.0, -0.6, 3.0], [9.0, 0.0, 2.0]]))
    psi_f: float = np.pi / 2                       # rad: heading at the end of the course
    # the carry: from the pad to the release point, then a throw with a flip
    x_pad: np.ndarray = field(default_factory=lambda: np.array([-1.5, -1.0, 1.0]))
    psi_pad: float = np.radians(140.0)             # heading on the pad, which the filter does not know
    x_release: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.2]))
    v_release: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, 3.0]))
    flip: float = np.radians(160.0)                # the throw turns the vehicle by this much
    flip_axis: np.ndarray = field(default_factory=lambda: unit([1.0, 0.4, 0.2]))
    t_flip: float = 0.8                            # s: duration of the flip, at the end of the carry
    # what the filter believes when it is switched on: level, at rest, at the origin, heading zero
    sig0: tuple = (1.0, 2.0, 2.5)                  # initial standard deviations: v (m/s), x (m), R (rad)
    # the package: weight known, place unknown
    m_add: float = 0.8                             # kg
    r_add: np.ndarray = field(default_factory=lambda: np.array([0.20, -0.12, -0.05]))   # m, body frame
    # the gust: a crosswind force (world frame) and a torque (body frame)
    gust_force: float = 3.0                        # N, peak
    gust_window: tuple = (1.0, 5.0)                # s, counted from the start of the course
    gust_torque: float = 2.0                       # times the gust torque of Chapter 8
    # the obstacle
    x_o: np.ndarray = field(default_factory=lambda: np.array([4.5, 0.4, 3.7]))
    r_o: float = 0.6
    beta: float = 1.8
    sigma_k: tuple = (0.25, 4.0, 0.0)              # sigma(p) = k1 tanh((k3 - k2 p) / 2)
    alpha: tuple = (2.0, 4.0)
    w_tau: float = 40.0
    # actuator limits
    f_max: float = 100.0                           # N
    tau_max: float = 8.0                           # N·m, each axis
    # sensors
    sig_g: float = 0.01                            # rad/s per sample
    sig_a: float = 0.1                             # m/s² per sample
    sig_y: float = 0.03                            # m per landmark reading
    every: int = 4                                 # a landmark reading every `every` steps
    landmarks: np.ndarray = field(default_factory=lambda: np.array(
        [[-1.0, -3.0, 0.0], [10.0, -3.0, 0.5], [10.0, 3.0, 0.0], [-1.0, 3.0, 1.0]]))
    # gains: Lee's position gains, attitude gains scaled by (kappa^2, kappa)
    kappa: float = 2.0
    # adaptation (Chapter 8)
    k_Delta: float = 4000.0
    k_f: float = 20.0
    Delta_max: float = 4.0
    c: float = 0.1
    seed: int = 17


    @property
    def t_start(self):
        return self.t_carry + self.t_hold

    @property
    def t_end(self):
        return self.t_start + self.t_course + self.t_rest


QUAD = robots.get("quadrotor")


class Carry:
    """The vehicle in the operator's hand: a prescribed motion from the pad to the release.
    Position: two quintics, the carry and the throw. Attitude:
    the pad heading with a wobble, then a flip about a body axis during the last t_flip."""

    def __init__(self, ms):
        self.ms = ms

    def position(self, t, order=0):
        """Rest to rest from the pad to a point 0.6 m under the release point, then the
        throw: from rest to the release point and the release velocity in t_flip."""
        ms = self.ms
        T1, T2 = ms.t_carry - ms.t_flip, ms.t_flip
        low = ms.x_release - 0.6 * E3

        def b(s, T):
            return [10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4) / T,
                    (60 * s - 180 * s**2 + 120 * s**3) / T**2][order]

        if t <= T1:
            return (ms.x_pad if order == 0 else 0.0) + (low - ms.x_pad) * b(t / T1, T1)
        s = (t - T1) / T2
        c = [-4 * s**3 + 7 * s**4 - 3 * s**5, (-12 * s**2 + 28 * s**3 - 15 * s**4) / T2,
             (-24 * s + 84 * s**2 - 60 * s**3) / T2**2][order]
        return (low if order == 0 else 0.0) + (ms.x_release - low) * b(s, T2) + ms.v_release * T2 * c

    def attitude(self, t):
        ms = self.ms
        u = np.clip((t - (ms.t_carry - ms.t_flip)) / ms.t_flip, 0.0, 1.5)
        wobble = 0.25 * np.sin(2.1 * t) * np.array([1.0, 0.5, 0.0])
        return (Rz(ms.psi_pad + 0.5 * np.sin(1.3 * t)) @ expm_so3(wobble)
                @ expm_so3(ms.flip * (2 * u * u - u**3) * ms.flip_axis))

    def state(self, t, d=1e-5):
        R = self.attitude(t)
        Omega = log_so3(self.attitude(t - d).T @ self.attitude(t + d)) / (2 * d)
        return QUAD.State(x=self.position(t), v=self.position(t, 1), R=R, Omega=Omega)

    def imu(self, t, h, g):
        """What a noise-free IMU reports for the step from t to t + h: the mean body angular
        velocity and the mean specific force (eq14-imu)."""
        R0, R1 = self.attitude(t), self.attitude(t + h)
        a_s = 0.5 * (R0.T @ (self.position(t, 2) + g * E3) + R1.T @ (self.position(t + h, 2) + g * E3))
        return log_so3(R0.T @ R1) / h, a_s


def believed(ms):
    """Parameters the controller believes: the total mass, and the inertia of the bare vehicle."""
    p = QUAD.params
    return replace(p, m=p.m + ms.m_add)


def true_params(ms):
    """The loaded vehicle about its own center of mass."""
    p = QUAD.params
    m = p.m + ms.m_add
    mu = p.m * ms.m_add / m                        # reduced mass
    r = ms.r_add
    return replace(p, m=m, J=p.J + mu * ((r @ r) * I3 - np.outer(r, r))), ms.m_add / m * r


def gains(ms):
    k = QUAD.Gains()
    return replace(k, k_R=ms.kappa**2 * k.k_R, k_Omega=ms.kappa * k.k_Omega)


class LoadedQuadrotor(robots.quadrotor.Quadrotor):
    """The library quadrotor with three things the controller does not know about:
    the thrust misses the center of mass by rho, a force F_e acts in the world frame
    and a torque tau_e in the body frame. With all three zero it is the library model."""
    rho = np.zeros(3)
    F_e = np.zeros(3)
    tau_e = np.zeros(3)

    def dynamics(self, state, u, params=None):
        p = params or self.params
        f, tau = u
        v_dot, Omega_dot = super().dynamics(state, (f, tau + f * np.cross(E3, self.rho) + self.tau_e), p)
        return v_dot + self.F_e / p.m, Omega_dot


def gust(ms, t):
    """Crosswind force (world frame) and gust torque (body frame) at time t."""
    a, b = ms.t_start + ms.gust_window[0], ms.t_start + ms.gust_window[1]
    s = np.clip((t - a) / 1.0, 0.0, 1.0) * np.clip((b - t) / 1.0, 0.0, 1.0)
    s = s * s * (3 - 2 * s)
    return ms.gust_force * s * np.array([0.0, -1.0, 0.0]), ms.gust_torque * s * V8.gust(t)


# =====================================================================
# 2. The plan: minimum snap in the flat outputs (Chapter 10)
# =====================================================================
class Plan:
    """x_d(t) and psi_d(t): hold, minimum-snap course, hold. The flat map gives the rest."""

    def __init__(self, ms):
        self.ms, self.p = ms, believed(ms)
        self.times = V10.chord_times(ms.waypoints, ms.t_course)
        self.spline = V10.min_derivative_spline(ms.waypoints, self.times, r=4)

    def flat_output(self, t):
        """[x, x', x'', x''', x''''] and (psi, psi', psi'')."""
        ms = self.ms
        raw = (t - ms.t_start) / ms.t_course
        s = np.clip(raw, 0.0, 1.0)
        d = V10.derivs(self.spline.jet(s * ms.t_course, 4))
        if raw < -1e-12 or raw > 1 + 1e-12:               # holding: only the snap jumps at the two ends
            d[1:] = 0.0
        T = ms.t_course
        psi = ms.psi_f * (10 * s**3 - 15 * s**4 + 6 * s**5)
        psi1 = ms.psi_f * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        psi2 = ms.psi_f * (60 * s - 180 * s**2 + 120 * s**3) / T**2
        return d, (psi, psi1, psi2)

    def __call__(self, t):
        """The reference tuple of the library's controller, and the flat map eq10-quad-map."""
        d, (psi, psi1, psi2) = self.flat_output(t)
        c, s = np.cos(psi), np.sin(psi)
        hd = np.array([c, s, 0.0])
        hp = np.array([-s, c, 0.0])
        ref = (d[0], d[1], d[2], d[3], d[4], hd, psi1 * hp, psi2 * hp - psi1**2 * hd)
        return ref, V10.quadrotor_flat_closed(d, psi, psi1, psi2, self.p)


def closed_loop_feedforward(x, v, R, Om, ref, k, p):
    """Omegad and its rate by differentiating F_d along the closed loop, as the library's
    tracking_controller does (thm6-cascade). Returns f, Rd, Omegad, Omegad_dot."""
    xd, vd, ad, jd, sd, hd, hd1, hd2 = ref
    e_v = v - vd
    Fd = -k.k_x * (x - xd) - k.k_v * e_v + p.m * p.g * E3 + p.m * ad
    b3 = R @ E3
    f = max(Fd @ b3, 0.0)
    e_v1 = f * b3 / p.m - p.g * E3 - ad
    F1 = -k.k_x * e_v - k.k_v * e_v1 + p.m * jd
    b3_dot = R @ np.cross(Om, E3)
    f1 = (F1 @ b3 + Fd @ b3_dot) if f > 0.0 else 0.0
    e_v2 = (f1 * b3 + f * b3_dot) / p.m - jd
    F2 = -k.k_x * e_v1 - k.k_v * e_v2 + p.m * sd
    return (f,) + tuple(robots.quadrotor.desired_attitude_rates(Fd, F1, F2, hd, hd1, hd2))


# =====================================================================
# 3. The safety filter: augmented barrier (Chapter 8), vehicle and obstacle as arguments
# =====================================================================
def sigma(p, k):
    """Wu's sigmoid sigma = k1 tanh((k3 - k2 p) / 2) with its first two derivatives."""
    k1, k2, k3 = k
    th = np.tanh(0.5 * (k3 - k2 * p))
    return k1 * th, -0.5 * k1 * k2 * (1 - th**2), -0.5 * k1 * k2**2 * th * (1 - th**2)


def aug_H(x, R, ms):
    """eq8-aug."""
    r = x - ms.x_o
    return r @ r - ms.beta * ms.r_o**2 - sigma(r @ R[:, 2], ms.sigma_k)[0]


def aug_terms(x, v, R, Om, m, J, ms, Delta=np.zeros(3)):
    """H, H1 = H_dot + alpha1 H, and H1_dot = a + b_f f + b_tau . tau for eq8-aug.
    Delta is a body torque that the model of the filter adds to the input (zero in Chapter 8)."""
    alpha1 = ms.alpha[0]
    r = x - ms.x_o
    b3 = R[:, 2]
    p = r @ b3
    sg, d1, d2 = sigma(p, ms.sigma_k)
    b3_dot = R @ np.cross(Om, E3)
    p_dot = v @ b3 + r @ b3_dot
    H = r @ r - ms.beta * ms.r_o**2 - sg
    H_dot = 2 * r @ v - d1 * p_dot
    Ji = np.linalg.inv(J)
    drift = Ji @ (Delta - np.cross(Om, J @ Om))
    p_dd0 = (-9.81 * b3[2] + 2 * v @ b3_dot + r @ R @ np.cross(Om, np.cross(Om, E3))
             + r @ R @ np.cross(drift, E3))
    a = alpha1 * H_dot + 2 * v @ v - 2 * 9.81 * r[2] - d2 * p_dot**2 - d1 * p_dd0
    return H, H_dot + alpha1 * H, a, (2 * p - d1) / m, -d1 * Ji @ np.cross(E3, R.T @ r)


def safety_filter(x, v, R, Om, f_d, tau_d, m, J, ms, Delta=np.zeros(3)):
    """eq8-safety-filter for the pair (f, tau): the smallest change, weighted, that keeps
    eq8-rd2 for the augmented barrier, with 0 <= f <= f_max. Returns f, tau, feasible."""
    _, H1, a, b_f, b_tau = aug_terms(x, v, R, Om, m, J, ms, Delta)
    Q = np.diag([1.0, ms.w_tau, ms.w_tau, ms.w_tau])
    c = -Q @ np.concatenate([[f_d], tau_d])
    A = np.array([np.concatenate([[-b_f], -b_tau]), [-1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    b = np.array([a + ms.alpha[1] * H1, 0.0, ms.f_max])
    z = V8.solve_qp(Q, c, A, b)
    if z is None:
        return min(max(f_d, 0.0), ms.f_max), tau_d, False
    return z[0], z[1:], True


class L1Told(V8.L1Attitude):
    """The L1 controller of Chapter 8, split in two so that something can stand between
    the torque it proposes and the torque that is applied.

    propose(...)  returns tau_d of eq8-l1-law and does not advance the controller.
    advance(...)  advances predictor, estimate and filter by one period. dtau = tau - tau_d
                  is what the safety filter and the actuator limits changed. The predictor
                  is given the same change, turned into its own frame like the estimate:
                  tau_fb_p += Xi R_p^T R dtau. With dtau = 0 this is the step of Chapter 8.
    """

    def propose(self, R, Om, Rd, Omd, Omd_dot):
        return self.torques(R, Om, Rd, Omd, Omd_dot)[0]

    def advance(self, R, Om, Rd, Omd, Omd_dot, dtau=np.zeros(3)):
        h = self.h
        _, _, tau_fb_p = self.torques(R, Om, Rd, Omd, Omd_dot)
        Xi = self.Xi(R)
        eRt, eOmt, _ = self.errors(R, Om)
        y = -(Xi.T @ eOmt + self.c * Xi.T @ self.Ji.T @ eRt)                 # eq8-adapt
        Dhat_new = self.Dhat + h * self.k_Delta * V8.proj(self.Dhat, y, self.bound)
        size = np.linalg.norm(Dhat_new)
        if size > self.bound:
            Dhat_new = Dhat_new * (self.bound / size)
        tau_fb_p = tau_fb_p + Xi @ (Dhat_new - self.Dhat) + Xi @ self.Rp.T @ R @ dtau
        self.Dhat = Dhat_new
        Omp_dot = self.Ji @ tau_fb_p - (hat(self.Omp) @ self.Rp.T @ Rd @ Omd - self.Rp.T @ Rd @ Omd_dot)
        Omp_new = self.Omp + h * Omp_dot
        self.Rp = self.Rp @ expm_so3(0.5 * h * (self.Omp + Omp_new))
        self.Omp = Omp_new
        self.Df = self.Df + (1 - np.exp(-self.k_f * h)) * (self.Dhat - self.Df)   # eq8-filter


# =====================================================================
# 4. The filter: right-invariant extended Kalman filter on SE_2(3) (def14-inekf)
# =====================================================================
class InvariantEKF:
    """One vehicle. State: Rhat, vhat, xhat and the covariance Sigma of chi_r = (v, x, R)."""

    def __init__(self, Rhat, vhat, xhat, sig_v, sig_x, sig_R, landmarks, sig_g, sig_a, sig_y, h):
        self.R, self.v, self.x = Rhat.copy(), vhat.copy(), xhat.copy()
        S0 = np.diag(np.r_[np.full(3, sig_v**2), np.full(3, sig_x**2), np.full(3, sig_R**2)])
        T0 = V14.sek_Ad(V14.sek_mat(self.R, [self.v, self.x]))     # eq14-left-right at the estimate
        T0[0:3, 0:3] = I3
        T0[3:6, 3:6] = I3
        self.Sigma = T0 @ S0 @ T0.T
        self.h, self.landmarks = h, np.asarray(landmarks, float)
        self.Sigma_n = np.zeros((9, 9))
        self.Sigma_n[0:3, 0:3] = I3 * (sig_a * h) ** 2
        self.Sigma_n[6:9, 6:9] = I3 * (sig_g * h) ** 2
        self.Sigma_y = np.eye(3 * len(self.landmarks)) * sig_y**2
        self.Phi = V14.Phi_right(h)
        self.C_y = np.concatenate([np.hstack([np.zeros((3, 3)), I3, -hat(p)]) for p in self.landmarks])
        self.nis = np.inf

    def propagate(self, Om, a_s):
        """eq14-propagate with one IMU reading."""
        Ad = V14.sek_Ad(V14.sek_mat(self.R, [self.v, self.x]))
        self.Sigma = self.Phi @ (self.Sigma + Ad @ self.Sigma_n @ Ad.T) @ self.Phi.T
        self.R, self.v, self.x = V14.imu_step(self.R, self.v, self.x, Om, a_s, self.h)

    def update(self, y):
        """eq14-innovation, eq14-gain, eq14-update with one reading of every landmark, y of shape (L, 3)."""
        ytilde = (y @ self.R.T + self.x - self.landmarks).ravel()
        S_y = self.C_y @ self.Sigma @ self.C_y.T + self.Sigma_y      # isotropic noise: R Sigma_y R^T = Sigma_y
        Ktilde = np.linalg.solve(S_y, self.C_y @ self.Sigma).T
        self.nis = float(ytilde @ np.linalg.solve(S_y, ytilde)) / len(ytilde)
        d = -Ktilde @ ytilde
        E, Jl = V14.exp_so3_b(d[6:9]), V14.jl_b(d[6:9])
        self.v = E @ self.v + Jl @ d[0:3]
        self.x = E @ self.x + Jl @ d[3:6]
        self.R = E @ self.R
        IK = np.eye(9) - Ktilde @ self.C_y
        self.Sigma = IK @ self.Sigma @ IK.T + Ktilde @ self.Sigma_y @ Ktilde.T

    def error(self, R, v, x):
        """chi_r of eq14-errors against the truth, and its NEES."""
        ER = self.R @ R.T
        w = V14.log_so3(ER)
        Ji = np.linalg.inv(V14.jl_b(w))
        chi = np.r_[Ji @ (self.v - ER @ v), Ji @ (self.x - ER @ x), w]
        return chi, float(chi @ np.linalg.solve(self.Sigma, chi)) / 9


# =====================================================================
# 5. The loop
# =====================================================================
ALL_ON = dict(estimate=True, adapt=True, safety=True, feedforward="flat", attitude="geometric")


def fly(ms=None, plan=None, t_end=None, **blocks):
    """Fly the mission. blocks switches the parts of the stack:

    estimate     True: the controller sees the filter's estimate. False: the truth.
    adapt        True: L1 adaptive control (eq8-l1-law), with the state predictor told what the
                 safety filter and the actuator limits changed. "untold": L1 exactly as in
                 Chapter 8. False: the plain controller (eq6-law).
    safety       True: the safety filter with the augmented barrier (eq8-aug), whose model
                 adds the filtered estimate Delta_f to the torque when there is one.
                 "blind": the filter exactly as in Chapter 8. False: no filter.
    feedforward  "flat": Omegad and its rate from the flat map of the reference.
                 "zero": both zero. "exact": differentiated along the closed loop,
                 as the library's controller does.
    attitude     "geometric" or "euler" (an Euler-angle law with the same linear gains).
    """
    ms = ms or Mission()
    opt = {**ALL_ON, **blocks}
    plan = plan or Plan(ms)
    rng = np.random.default_rng(ms.seed)
    h = ms.h
    n = int(round((t_end or ms.t_end) / h))
    n_rel = int(round(ms.t_carry / h))
    p0, k = believed(ms), gains(ms)
    p_true, rho = true_params(ms)
    quad = LoadedQuadrotor(p_true)
    quad.rho = rho
    carry = Carry(ms)
    state = carry.state(0.0)
    ekf = InvariantEKF(I3, np.zeros(3), np.zeros(3), *ms.sig0, ms.landmarks, ms.sig_g, ms.sig_a, ms.sig_y, h)
    Om_read = state.Omega + ms.sig_g * rng.normal(size=3)
    l1 = None
    out = {key: np.zeros((n + 1, 3)) for key in ("x", "v", "xd", "xhat", "vhat", "tau", "tau_d", "Dhat", "Df")}
    out.update({key: np.zeros(n + 1) for key in ("t", "f", "f_d", "H", "dist", "att_err", "pos_err", "vel_err",
                                                 "nees", "nis", "e_x", "Psi_pred", "Psi", "tilt", "manifold", "active")})
    out["R"], out["Rhat"], out["Rd"] = (np.zeros((n + 1, 3, 3)) for _ in range(3))
    out["feasible"] = np.ones(n + 1, bool)
    for i in range(n + 1):
        t = i * h
        ref, flat = plan(t)
        flying = i >= n_rel
        # --- what the controller sees
        if opt["estimate"]:
            x, v, R, Om = ekf.x, ekf.v, ekf.R, Om_read
        else:
            x, v, R, Om = state.x, state.v, state.R, state.Omega
        # --- the cascade: eq6-Fd, eq6-Rd, eq6-law
        f_d, tau_d, Rd, l1_args = 0.0, np.zeros(3), I3, None
        if flying:
            Fd = -k.k_x * (x - ref[0]) - k.k_v * (v - ref[1]) + p0.m * p0.g * E3 + p0.m * ref[2]
            f_d = max(Fd @ R[:, 2], 0.0)
            Rd = desired_attitude(Fd, ref[5])
            if opt["feedforward"] == "flat":
                Omd, Omd_dot = flat["Omega"], flat["Omega_dot"]
            elif opt["feedforward"] == "exact":
                _, _, Omd, Omd_dot = closed_loop_feedforward(x, v, R, Om, ref, k, p0)
            else:
                Omd, Omd_dot = np.zeros(3), np.zeros(3)
            if opt["attitude"] == "euler":
                tau_d = euler_torque(R, Om, Rd, Omd, Omd_dot, p0.J, k)
            elif opt["adapt"]:
                if l1 is None:
                    l1 = L1Told(R, Om, h, k_Delta=ms.k_Delta, k_f=ms.k_f, bound=ms.Delta_max, c=ms.c,
                                       J=p0.J, kR=k.k_R, kOm=k.k_Omega, kRt=k.k_R, kOmt=k.k_Omega)
                tau_d = l1.propose(R, Om, Rd, Omd, Omd_dot)
                l1_args = (R, Om, Rd, Omd, Omd_dot)
            else:
                tau_d = attitude_tracking_torque(R, Om, Rd, Omd, Omd_dot, p0.J, k.k_R, k.k_Omega)
        # --- the safety filter
        f, tau, ok = f_d, tau_d, True
        if flying and opt["safety"]:
            Delta = l1.Df if (l1 is not None and opt["safety"] is True) else np.zeros(3)
            f, tau, ok = safety_filter(x, v, R, Om, f_d, tau_d, p0.m, p0.J, ms, Delta)
        f = float(np.clip(f, 0.0, ms.f_max))
        tau = np.clip(tau, -ms.tau_max, ms.tau_max)
        if l1_args is not None:
            l1.advance(*l1_args, dtau=(tau - tau_d) if opt["adapt"] is True else np.zeros(3))
        # --- record (against the truth)
        o = out
        o["t"][i], o["x"][i], o["v"][i], o["R"][i], o["xd"][i] = t, state.x, state.v, state.R, ref[0]
        o["xhat"][i], o["vhat"][i], o["Rhat"][i], o["Rd"][i] = ekf.x, ekf.v, ekf.R, Rd
        o["f"][i], o["tau"][i], o["f_d"][i], o["tau_d"][i] = f, tau, f_d, tau_d
        o["H"][i] = aug_H(state.x, state.R, ms)
        o["dist"][i] = np.linalg.norm(state.x - ms.x_o)
        chi, o["nees"][i] = ekf.error(state.R, state.v, state.x)
        o["att_err"][i] = np.linalg.norm(chi[6:9])
        o["pos_err"][i] = np.linalg.norm(ekf.x - state.x)
        o["vel_err"][i] = np.linalg.norm(ekf.v - state.v)
        o["e_x"][i] = np.linalg.norm(state.x - ref[0])
        o["Psi"][i] = Psi_R(state.R, Rd)
        o["tilt"][i] = np.arccos(np.clip(state.R[2, 2], -1, 1))
        o["nis"][i], o["feasible"][i] = ekf.nis, ok
        o["manifold"][i] = quad.on_manifold(state)
        o["active"][i] = abs(f - f_d) + np.linalg.norm(tau - tau_d)
        if l1 is not None:
            o["Dhat"][i], o["Df"][i], o["Psi_pred"][i] = l1.Dhat, l1.Df, Psi_R(l1.Rp, R)
        if i == n:
            break
        if flying:
            # --- the vehicle: one step of the library, input held
            quad.F_e, quad.tau_e = gust(ms, t)
            acc0 = quad.dynamics(state, (f, tau))[0]
            new = quad.step(state, (f, tau), h)
            acc1 = quad.dynamics(new, (f, tau))[0]
            # --- the IMU: means over the step (eq14-imu)
            Om_mean = 0.5 * (state.Omega + new.Omega)
            a_s = 0.5 * (state.R.T @ (acc0 + p_true.g * E3) + new.R.T @ (acc1 + p_true.g * E3))
        else:
            Om_mean, a_s = carry.imu(t, h, p_true.g)
            new = carry.state(t + h)
        Om_read = Om_mean + ms.sig_g * rng.normal(size=3)
        ekf.propagate(Om_read, a_s + ms.sig_a * rng.normal(size=3))
        state = new
        if (i + 1) % ms.every == 0:
            y = (ms.landmarks - state.x) @ state.R + ms.sig_y * rng.normal(size=(len(ms.landmarks), 3))
            ekf.update(y)
    out["state"], out["mission"], out["plan"], out["n_rel"] = state, ms, plan, n_rel
    return out


def _from_ch06(*names):
    """verify/ch06.py runs its checks at import, so its functions are taken out by name."""
    ns, found = {"np": np}, {}
    for node in ast.parse((HERE / "ch06.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "ch06.py", "exec"), ns)
            found[node.name] = ns[node.name]
    return [found[name] for name in names]


euler_zyx, E_mat = _from_ch06("euler_zyx", "E_mat")


def euler_torque(R, Om, Rd, Omd, Omd_dot, J, k):
    """eq6-law with the error vector e_R replaced by the Euler-angle error E (Theta - Theta_d)
    of eq-zyx and eq-euler-jacobian. The two agree to first order at R = Rd."""
    Th, Thd = euler_zyx(R), euler_zyx(Rd)
    dTh = (Th - Thd + np.pi) % (2 * np.pi) - np.pi
    e_euler = E_mat(Th[1], Th[2]) @ dTh
    return attitude_tracking_torque(R, Om, Rd, Omd, Omd_dot, J, k.k_R, k.k_Omega) + k.k_R * (e_R(R, Rd) - e_euler)


def detour_window(o, tol=1e-6, after=2.0):
    """First and last step of the course at which the safety filter changes the input,
    the last one extended by `after` seconds for the return to the reference."""
    ms = o["mission"]
    c = int(round(ms.t_start / ms.h))
    act = np.nonzero(o["active"][c:] > tol)[0]
    return (int(c + act[0]), int(c + act[-1] + round(after / ms.h))) if len(act) else None


def summary(o, window=None):
    """The numbers of one flight, measured against the truth."""
    ms, j, h = o["mission"], o["n_rel"], o["mission"].h
    c = int(round(ms.t_start / h))
    e = int(round((ms.t_start + ms.t_course) / h))
    idx = np.r_[c:e] if window is None else np.r_[c:window[0], window[1]:e]
    up = np.nonzero(np.degrees(o["tilt"][j:]) < 10.0)[0]
    last = slice(len(o["t"]) - int(round(1.0 / h)), None)
    return dict(
        t_upright=float(up[0] * h) if len(up) else float("nan"),
        track_rms=float(np.sqrt(np.mean(o["e_x"][idx] ** 2))), track_max=float(o["e_x"][idx].max()),
        hover=float(np.mean(o["e_x"][last])),
        min_dist=float(o["dist"][j:].min()), min_H=float(o["H"][j:].min()),
        f_peak=float(o["f"].max()), tau_peak=float(np.abs(o["tau"]).max()),
        tau_rms=float(np.sqrt(np.mean(np.sum(o["tau"][c:e] ** 2, axis=1)))),
        tilt_max=float(np.degrees(o["tilt"][c:].max())), min_z=float(o["x"][j:, 2].min()),
        infeasible=int((~o["feasible"]).sum()), hit=bool(o["dist"][j:].min() < ms.r_o),
        pred_max=float(o["Psi_pred"][j:].max()))


# =====================================================================
# 6. Local optimality: the attitude loop as an LQR and as an error-state MPC
# =====================================================================
class AttitudeMPC:
    """Error-state MPC (def13-mpc) for a target at rest, with |tau_i| <= tau_max.
    The model is eq7-so3-lin at Omega_d = 0, sampled exactly; the QP is eq13-qp."""

    def __init__(self, J, h, N, Q, Ru, tau_max, bounded=True):
        from scipy.linalg import solve_discrete_are
        B = np.vstack([np.zeros((3, 3)), np.linalg.inv(J)])
        A, Bk = V13.zoh(V13.err_A(np.zeros(3), J), B, h)
        self.Pf = solve_discrete_are(A, Bk, h * Q, h * Ru)
        self.K = np.linalg.solve(h * Ru + Bk.T @ self.Pf @ Bk, Bk.T @ self.Pf @ A)
        self.P_UU, self.P_Us = V13.condense([A] * N, [Bk] * N, h * Q, h * Ru, self.Pf)[:2]
        self.tau_max, self.bounded, self.U = tau_max, bounded, None
        self.lo, self.hi = -tau_max * np.ones(3 * N), tau_max * np.ones(3 * N)

    def __call__(self, R, Om, Rd=I3):
        s0 = np.r_[V13.log_so3(Rd.T @ R), Om]
        if not self.bounded:
            return np.clip(-self.K @ s0, -self.tau_max, self.tau_max)
        warm = None if self.U is None else np.r_[self.U[3:], np.zeros(3)]
        self.U, _ = V13.box_qp(self.P_UU, self.P_Us @ s0, self.lo, self.hi, warm)
        return self.U[:3]


def attitude_run(law, R0, Om0, J, t_f=3.0, h=0.005, hold=1):
    """A rigid body with the torque law `law(R, Omega)` sampled every `hold` steps."""
    Ji = np.linalg.inv(J)
    R, Om, n = R0.copy(), Om0.copy(), int(round(t_f / h))
    ang, taus = np.zeros(n + 1), np.zeros((n, 3))
    for i in range(n):
        ang[i] = np.linalg.norm(log_so3(R))
        if i % hold == 0:
            tau = law(R, Om)
        taus[i] = tau
        R, Om = V8.body_step(R, Om, tau, h, J, Ji)
    ang[n] = np.linalg.norm(log_so3(R))
    return ang, taus


def settle(ang, h, tol_deg=5.0):
    bad = np.nonzero(np.degrees(ang) > tol_deg)[0]
    return float((bad[-1] + 1) * h) if len(bad) else 0.0


# =====================================================================
# Checks
# =====================================================================
def _from_robots(*names):
    """The geomech helpers of verify/robots.py, which also runs its checks at import."""
    from geomech import check_eom, compute_eom, to_standard_form
    from geomech.codegen import StateLayout, collect_params, load_module, module_source
    ns = dict(np=np, check_eom=check_eom, compute_eom=compute_eom, to_standard_form=to_standard_form,
              StateLayout=StateLayout, collect_params=collect_params, load_module=load_module,
              module_source=module_source)
    for node in ast.parse((HERE / "robots.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "robots.py", "exec"), ns)
    return [ns[name] for name in names]


def main():
    ck = Checker("ch17")
    rng = np.random.default_rng(17)
    ms = Mission()
    plan = Plan(ms)
    p0, k = believed(ms), gains(ms)
    p_true, rho = true_params(ms)
    h = ms.h
    t_all = time.time()

    # ------------------------------------------------------------ the vehicle
    p = QUAD.params
    ck.true("library quadrotor: 6 degrees of freedom, 12 numbers stored", QUAD.dim == 6 and QUAD.stored == 12)
    hover = p0.m * p0.g
    Delta_e = hover * np.cross(E3, rho)
    ck.record("vehicle", dict(m=p.m, J=np.diag(p.J).tolist(), m_add=ms.m_add, r_add=ms.r_add.tolist(),
                              m_total=p0.m, hover=hover, rho_mm=(1e3 * rho).tolist(),
                              J_true=np.round(p_true.J, 4).tolist(),
                              J_rise_percent=(100 * (np.diag(p_true.J) / np.diag(p.J) - 1)).tolist(),
                              Delta_e=Delta_e.tolist(), Delta_e_norm=float(np.linalg.norm(Delta_e)),
                              f_max=ms.f_max, f_max_over_hover=ms.f_max / hover, tau_max=ms.tau_max))
    ck.record("gains", dict(k_x=k.k_x, k_v=k.k_v, k_R=k.k_R, k_Omega=k.k_Omega, kappa=ms.kappa,
                            k_Delta=ms.k_Delta, k_f=ms.k_f, Delta_max=ms.Delta_max, c=ms.c))
    eig = np.linalg.eigvalsh(p_true.J)
    ck.true("true inertia of the loaded vehicle is symmetric positive definite", eig.min() > 0)
    # the loaded model collapses to the library model
    lq, worst = LoadedQuadrotor(p), 0.0
    for _ in range(20):
        st, u = QUAD.random_state(rng), (float(rng.uniform(0, 80)), rng.normal(size=3))
        worst = max(worst, max(np.max(np.abs(a - b)) for a, b in zip(lq.dynamics(st, u), QUAD.dynamics(st, u))))
    ck.true("loaded model with nothing unknown is the library model", worst == 0.0, f"err {worst:.1e}")
    # the offset torque: thrust through the geometric center, seen from the center of mass
    st, f = QUAD.random_state(rng), 37.0
    lq2 = LoadedQuadrotor(p_true)
    lq2.rho = rho
    Om_dot = lq2.dynamics(st, (f, np.zeros(3)))[1]
    want = np.linalg.solve(p_true.J, np.cross(-rho, f * E3) - np.cross(st.Omega, p_true.J @ st.Omega))
    ck.close("offset torque is (-rho) x f e3 about the center of mass", Om_dot, want, tol=1e-12)

    # ------------------------------------------------------------ dynamics, derived again (Chapter 5)
    derive, geomech_view, flat_input = _from_robots("derive", "geomech_view", "flat_input")
    from geomech.simulation import accelerations
    mod, layout, err = derive(QUAD, "ch17_quadrotor")
    ck.true("quadrotor: geomech check_eom < 1e-5", err < 1e-5, f"err {err:.1e}")
    worst = 0.0
    gm = {"m": p_true.m, "J": p_true.J, "g": p_true.g, "e3": E3}
    for _ in range(20):
        st, u = QUAD.random_state(rng), (float(rng.uniform(0, 80)), 3 * rng.normal(size=3))
        xg, mine = geomech_view(QUAD, layout, st, QUAD.dynamics(st, u, p_true), {})
        worst = max(worst, float(np.max(np.abs(mine - accelerations(mod, xg, flat_input(u), gm)))))
    ck.true("library dynamics equal the derived model at 20 random states (loaded inertia)", worst < 1e-9,
            f"err {worst:.1e}")
    ck.record("symbolic", dict(check_eom=err, vs_library=worst))

    # ------------------------------------------------------------ reused blocks against their originals
    # barrier: my copy with arguments against verify/ch08.py with its constants
    m8 = Mission(x_o=V8.OBS_C, r_o=V8.OBS_R, beta=V8.AUG_BETA, sigma_k=(V8.AUG_K1, V8.AUG_K2, V8.AUG_K3),
                 alpha=(3.0, 6.0))
    worst = 0.0
    for _ in range(30):
        x, v, R, Om = rng.normal(size=3), rng.normal(size=3), random_rotation(rng), rng.normal(size=3)
        mine = aug_terms(x, v, R, Om, V8.M_Q, V8.J0, replace(m8, alpha=(3.0, 6.0)))
        theirs = V8.aug_terms(x, v, R, Om, V8.J0, 3.0)
        worst = max(worst, max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(mine, theirs)))
        worst = max(worst, abs(aug_H(x, R, m8) - V8.aug_H(x, R)))
        fd, td = float(rng.uniform(5, 20)), 0.3 * rng.normal(size=3)
        a_ = safety_filter(x, v, R, Om, fd, td, V8.M_Q, V8.J0, replace(m8, f_max=30.0, w_tau=40.0))
        b_ = V8.quad_filter(x, v, R, Om, fd, td, V8.J0)
        worst = max(worst, abs(a_[0] - b_[0]), float(np.max(np.abs(a_[1] - b_[1]))))
    ck.true("augmented barrier and safety filter equal verify/ch08.py at 30 random states", worst < 1e-10,
            f"err {worst:.1e}")
    # barrier derivative by finite differences along the model, at this mission's parameters
    worst = 0.0
    for _ in range(10):
        st = QUAD.State(x=ms.x_o + rng.normal(size=3), v=rng.normal(size=3), R=random_rotation(rng),
                        Omega=rng.normal(size=3))
        u = (float(rng.uniform(20, 70)), rng.normal(size=3))
        Dl = 0.5 * rng.normal(size=3)
        H, H1, a, b_f, b_tau = aug_terms(st.x, st.v, st.R, st.Omega, p0.m, p0.J, ms, Dl)

        def H1_at(s):
            Hs, H1s, *_ = aug_terms(s.x, s.v, s.R, s.Omega, p0.m, p0.J, ms, Dl)
            return H1s
        d = 1e-5
        fwd = QUAD.step(st, (u[0], u[1] + Dl), d, p0)
        bwd = QUAD.step(st, (u[0], u[1] + Dl), -d, p0)
        fd_ = (H1_at(fwd) - H1_at(bwd)) / (2 * d)
        worst = max(worst, abs(fd_ - (a + b_f * u[0] + b_tau @ u[1])))
    ck.true("H1_dot = a + b_f f + b_tau . tau by finite differences (with a model torque Delta)", worst < 1e-6,
            f"err {worst:.1e}")
    k1, k2, _ = ms.sigma_k
    ck.true("Wu's bound on sigma: sigma_0 < (beta - 1) r_o^2", k1 < (ms.beta - 1) * ms.r_o**2,
            f"{k1} < {(ms.beta - 1) * ms.r_o**2:.3f}")
    ck.record("barrier", dict(beta=ms.beta, r_o=ms.r_o, sigma_k=list(ms.sigma_k), alpha=list(ms.alpha),
                              x_o=ms.x_o.tolist(), keep_out=float(np.sqrt(ms.beta * ms.r_o**2 - k1)),
                              bound=(ms.beta - 1) * ms.r_o**2, w_tau=ms.w_tau))

    # L1: with dtau = 0, propose and advance are the step of Chapter 8
    a8 = V8.L1Attitude(I3, np.zeros(3), 1e-3)
    mine = L1Told(I3, np.zeros(3), 1e-3)
    worst, R, Om = 0.0, expm_so3(np.array([0.2, -0.1, 0.3])), np.array([0.3, 0.1, -0.2])
    ref8 = V8.ConingReference(1e-3)
    for i in range(400):
        Rd, Omd, Omd_dot = ref8(i * 1e-3)
        t1 = a8.step(R, Om, Rd, Omd, Omd_dot)
        t2 = mine.propose(R, Om, Rd, Omd, Omd_dot)
        mine.advance(R, Om, Rd, Omd, Omd_dot)
        worst = max(worst, float(np.max(np.abs(t1 - t2))), float(np.max(np.abs(a8.Dhat - mine.Dhat))),
                    float(np.max(np.abs(a8.Rp - mine.Rp))))
        R, Om = V8.body_step(R, Om, t1 + V8.DELTA_CAM, 1e-3, V8.J_TRUE, np.linalg.inv(V8.J_TRUE))
    ck.true("L1 split into propose and advance equals the step of verify/ch08.py over 400 periods",
            worst < 1e-12, f"err {worst:.1e}")
    # eq8-pred-dyn survives an override when the predictor is told
    def pred_gap(told):
        l1 = L1Told(I3, np.zeros(3), 1e-3)
        R, Om, Ji = I3.copy(), np.zeros(3), np.linalg.inv(V8.J0)
        worst = 0.0
        for i in range(1500):
            tau_d = l1.propose(R, Om, I3, np.zeros(3), np.zeros(3))
            dtau = np.array([0.3, -0.2, 0.1]) * np.sin(4e-3 * i) * (i > 300)
            l1.advance(R, Om, I3, np.zeros(3), np.zeros(3), dtau if told else np.zeros(3))
            R, Om = V8.body_step(R, Om, tau_d + dtau, 1e-3, V8.J0, Ji)
            worst = max(worst, Psi_R(l1.Rp, R))
        return worst
    gap_told, gap_untold = pred_gap(True), pred_gap(False)
    ck.true("an override of the torque leaves the prediction error alone when the predictor is told",
            gap_told < 1e-6 * gap_untold, f"told {gap_told:.1e}, untold {gap_untold:.1e}")

    # filter: one-vehicle class against run_filters of verify/ch14.py on its scenario, same random numbers
    sc = V14.Scenario(T=4.0)
    res = V14.run_filters(sc, np.random.default_rng(5), 1, 60.0, 1.0, 1.0, which=("inv",), keep=True)
    r5 = np.random.default_rng(5)
    R0, v0, x0 = V14.initial_estimates(sc, r5, 1, 60.0, 1.0, 1.0)
    ekf = InvariantEKF(R0[0], v0[0], x0[0], 1.0, 1.0, np.radians(60.0), sc.landmarks, sc.sig_g, sc.sig_a,
                       sc.sig_y, sc.h)
    gn = r5.normal(size=(sc.n, 1, 3)) * sc.sig_g
    an = r5.normal(size=(sc.n, 1, 3)) * sc.sig_a
    worst = 0.0
    for i in range(sc.n):
        ekf.propagate(sc.Om[i] + gn[i, 0], sc.a_s[i] + an[i, 0])
        if (i + 1) % sc.every == 0:
            ekf.update(sc.meas(i + 1) + r5.normal(size=(1, len(sc.landmarks), 3))[0] * sc.sig_y)
        Rr, vr, xr, Sr = res["hist"]["inv"][i + 1]
        worst = max(worst, float(np.max(np.abs(ekf.R - Rr[0]))), float(np.max(np.abs(ekf.x - xr[0]))),
                    float(np.max(np.abs(ekf.Sigma - Sr[0]))))
    ck.true("invariant filter for one vehicle equals run_filters of verify/ch14.py over 4 s", worst < 1e-9,
            f"err {worst:.1e}")

    # MPC: my class against ErrorStateMPC of verify/ch13.py, for its body and a target at rest
    rest = lambda t: (I3, np.zeros(3), np.zeros(3))  # noqa: E731
    theirs = V13.ErrorStateMPC(rest, 0.2)
    mine = AttitudeMPC(V13.J_B, V13.H_MPC, V13.N_MPC, V13.Q_MPC, V13.RU_MPC, V13.TAU_MAX)
    worst = 0.0
    for _ in range(5):
        R, Om = expm_so3(rng.normal(size=3)), 2 * rng.normal(size=3)
        theirs.U_prev, mine.U = None, None
        worst = max(worst, float(np.max(np.abs(theirs(0, R, Om) - mine(R, Om)))))
    ck.true("attitude MPC equals ErrorStateMPC of verify/ch13.py for a target at rest", worst < 1e-8,
            f"err {worst:.1e}")
    # Euler law agrees with eq6-law to first order
    Rd = expm_so3(np.array([0.3, -0.2, 0.5]))
    gaps = []
    for eps in (1e-2, 1e-3):
        R = Rd @ expm_so3(eps * np.array([0.5, -0.7, 0.4]))
        gaps.append(np.linalg.norm(euler_torque(R, np.zeros(3), Rd, np.zeros(3), np.zeros(3), p0.J, k)
                                   - attitude_tracking_torque(R, np.zeros(3), Rd, np.zeros(3), np.zeros(3),
                                                              p0.J, k.k_R, k.k_Omega)))
    ck.true("Euler-angle law equals eq6-law to first order at R = Rd", gaps[1] < 0.02 * gaps[0],
            f"gap {gaps[0]:.1e} -> {gaps[1]:.1e}")

    # ------------------------------------------------------------ the plan (Chapter 10)
    ts = np.arange(ms.t_start, ms.t_start + ms.t_course + 1e-9, 0.01)
    flats = [plan(t) for t in ts]
    f_plan = np.array([o["f"] for _, o in flats])
    tilt_plan = np.degrees(np.array([np.arccos(o["R"][2, 2]) for _, o in flats]))
    tau_plan = np.array([np.linalg.norm(o["tau"]) for _, o in flats])
    Om_plan = np.array([np.linalg.norm(o["Omega"]) for _, o in flats])
    speed = np.array([np.linalg.norm(r[1]) for r, _ in flats])
    gap = np.array([np.linalg.norm(r[0] - ms.x_o) for r, _ in flats])
    inside = ts[gap < ms.r_o]
    length = float(np.sum(np.linalg.norm(np.diff(np.array([r[0] for r, _ in flats]), axis=0), axis=1)))
    ck.true("the reference passes through the obstacle", gap.min() < ms.r_o, f"closest {gap.min():.3f} m")
    ck.true("plan is feasible: thrust and torque inside the limits",
            f_plan.max() < ms.f_max and f_plan.min() > 0 and tau_plan.max() < ms.tau_max)
    # the flat map solves the model: residual, and an open-loop flight
    worst = 0.0
    for t in rng.uniform(ms.t_start, ms.t_start + ms.t_course, size=20):
        ref, o = plan(t)
        st = QUAD.State(x=ref[0], v=ref[1], R=o["R"], Omega=o["Omega"])
        v_dot, Om_dot = QUAD.dynamics(st, (o["f"], o["tau"]), p0)
        worst = max(worst, float(np.max(np.abs(v_dot - ref[2]))), float(np.max(np.abs(Om_dot - o["Omega_dot"]))))
    ck.true("flat map of the plan satisfies the library model at 20 instants", worst < 1e-9, f"err {worst:.1e}")
    ref, o = plan(ms.t_start)
    st = QUAD.State(x=ref[0], v=ref[1], R=o["R"], Omega=o["Omega"])
    u_open = lambda t, s: (plan(t)[1]["f"], plan(t)[1]["tau"])  # noqa: E731
    end = QUAD.rollout(st, u_open, 0.01, int(round(ms.t_course / 0.01)), p0, hold=False, t0=ms.t_start)[-1]
    open_err = float(np.linalg.norm(end.x - ms.waypoints[-1]))
    ck.true("open loop on the believed model, the plan arrives at the target", open_err < 1e-5, f"{open_err:.1e} m")
    fast = Plan(replace(ms, t_course=4.0))
    f_fast = [fast(t)[1] for t in np.arange(ms.t_start, ms.t_start + 4.0, 0.01)]
    ck.record("plan", dict(times=(ms.t_start + plan.times).tolist(), length=length, speed_max=float(speed.max()),
                           f_min=float(f_plan.min()), f_max=float(f_plan.max()),
                           f_ratio=[float(f_plan.min() / hover), float(f_plan.max() / hover)],
                           tilt_max=float(tilt_plan.max()), tau_max=float(tau_plan.max()),
                           Omega_max=float(Om_plan.max()), closest_to_center=float(gap.min()),
                           inside_from=float(inside[0]), inside_to=float(inside[-1]),
                           open_loop_error=open_err, flat_residual=worst,
                           fast_tilt=float(np.degrees(max(np.arccos(o_["R"][2, 2]) for o_ in f_fast))),
                           fast_f_ratio=float(max(o_["f"] for o_ in f_fast) / hover),
                           fast_tau=float(max(np.linalg.norm(o_["tau"]) for o_ in f_fast))))

    # ------------------------------------------------------------ the mission, everything on
    full = fly(ms, plan)
    win = detour_window(full)
    S = {"all": summary(full, win)}
    j, c = full["n_rel"], int(round(ms.t_start / h))
    ck.record("window", dict(first=win[0] * h, last=win[1] * h, length=(win[1] - win[0]) * h,
                             active_steps=int((full["active"] > 1e-6).sum()),
                             active_course=int((full["active"][c:] > 1e-6).sum())))

    def row(name, ms_=None, **kw):
        S[name] = summary(fly(ms_ or ms, plan if ms_ is None else None, **kw), win)
        return S[name]

    # release and recovery
    rel = full["state"].__class__(x=full["x"][j], v=full["v"][j], R=full["R"][j], Omega=Carry(ms).state(ms.t_carry).Omega)
    ang_rel = float(np.degrees(np.linalg.norm(log_so3(full["Rd"][j].T @ rel.R))))
    after = np.nonzero(full["e_x"][j:c] > 0.05)[0]
    ck.record("release", dict(t=ms.t_carry, tilt=float(np.degrees(full["tilt"][j])), angle_to_Rd=ang_rel,
                              Psi=float(full["Psi"][j]), e_R_norm=float(np.linalg.norm(e_R(rel.R, full["Rd"][j]))),
                              spin=float(np.linalg.norm(rel.Omega)), speed=float(np.linalg.norm(rel.v)),
                              e_x=float(full["e_x"][j]), t_upright=S["all"]["t_upright"],
                              t_thrust=float(np.nonzero(full["f"][j:] > 1.0)[0][0] * h),
                              apex=float(full["x"][j:c, 2].max()), min_z=float(full["x"][j:c, 2].min()),
                              t_5cm=float((after[-1] + 1) * h), f_peak=float(full["f"][j:c].max()),
                              tau_sat_steps=int((np.abs(full["tau"][j:c]).max(axis=1) > ms.tau_max - 1e-9).sum()),
                              e_x_at_start=float(full["e_x"][c])))
    ck.true("release: more than 140 degrees from the attitude the cascade asks for", ang_rel > 140,
            f"{ang_rel:.1f} deg")
    ck.true("recovery: upright within 0.6 s, within 5 cm of the hold point before the course",
            S["all"]["t_upright"] < 0.6 and full["e_x"][c] < 0.05)

    # ------------------------------------------------------------ estimation alone, then in the loop
    def at(o, key, t):
        return float(o[key][int(round(t / h))])
    late = np.nonzero(np.degrees(full["att_err"][:j]) > 2.0)[0]
    late_x = np.nonzero(full["pos_err"][:j] > 0.05)[0]
    fl = slice(j, None)
    upd = np.arange(len(full["t"]))[::ms.every]
    est = dict(initial_angle=float(np.degrees(full["att_err"][0])), initial_pos=float(full["pos_err"][0]),
               att={str(t): np.degrees(at(full, "att_err", t)) for t in (0.5, 1.0, 2.0, 4.0, 6.0)},
               pos={str(t): at(full, "pos_err", t) for t in (0.5, 1.0, 2.0, 4.0, 6.0)},
               nees={str(t): at(full, "nees", t) for t in (0.5, 1.0, 2.0, 4.0, 6.0)},
               t_2deg=float((late[-1] + 1) * h), t_5cm=float((late_x[-1] + 1) * h),
               nees_release=float(np.mean(full["nees"][j - 100:j])), nis_release=float(np.mean(full["nis"][j - 100:j:ms.every])),
               nees_flight=float(np.mean(full["nees"][fl])), nis_flight=float(np.mean(full["nis"][fl][::ms.every])),
               att_flight_rms=float(np.degrees(np.sqrt(np.mean(full["att_err"][fl] ** 2)))),
               att_flight_max=float(np.degrees(full["att_err"][fl].max())),
               pos_flight_rms=float(np.sqrt(np.mean(full["pos_err"][fl] ** 2))),
               pos_flight_max=float(full["pos_err"][fl].max()),
               vel_flight_rms=float(np.sqrt(np.mean(full["vel_err"][fl] ** 2))),
               rate=1.0 / (ms.every * h), landmarks=len(ms.landmarks), sig_y=ms.sig_y, sig_g=ms.sig_g, sig_a=ms.sig_a)
    ck.true("filter alone: from 140 degrees to below 2 degrees for good within 2 s", est["t_2deg"] < 2.0,
            f"{est['t_2deg']:.2f} s")
    ck.true("filter in flight: attitude error below 0.5 degrees, position below 5 cm",
            est["att_flight_max"] < 0.5 and est["pos_flight_max"] < 0.05,
            f"{est['att_flight_max']:.2f} deg, {est['pos_flight_max']:.3f} m")
    ck.true("filter in flight: mean NEES between 0.5 and 3", 0.5 < est["nees_flight"] < 3.0,
            f"NEES {est['nees_flight']:.2f}")
    ck.true("filter is overconfident during the transient: NEES above 100 at 1 s", at(full, "nees", 1.0) > 100)
    truth = row("truth", estimate=False)
    for tag, t_c in (("early", 1.2), ("earliest", 0.9)):
        o_e = fly(replace(ms, t_carry=t_c))
        e_, je = summary(o_e), o_e["n_rel"]
        est[tag] = dict(t_carry=t_c, att=float(np.degrees(o_e["att_err"][je])), pos=float(o_e["pos_err"][je]),
                        nees=float(o_e["nees"][je]), min_dist=e_["min_dist"], min_H=e_["min_H"], hover=e_["hover"],
                        min_z=e_["min_z"], t_upright=e_["t_upright"], track_max=e_["track_max"])
    ck.record("estimation", est)

    # ------------------------------------------------------------ the cascade, alone (Chapters 6 and 10)
    calm = replace(ms, r_add=np.zeros(3), gust_force=0.0, gust_torque=0.0)
    casc = {}
    for kappa in (2.0, 1.0):
        for ff in ("flat", "zero", "exact"):
            o = fly(replace(calm, kappa=kappa), estimate=False, adapt=False, safety=False, feedforward=ff)
            s_ = summary(o)
            casc[f"{ff}_{kappa:g}"] = dict(rms=s_["track_rms"], max=s_["track_max"], hover=s_["hover"],
                                          t_upright=s_["t_upright"], min_z=s_["min_z"], f_peak=s_["f_peak"])
    ck.record("cascade", casc)
    ck.true("cascade alone, kappa = 2: flat feedforward tracks the course to 3 cm", casc["flat_2"]["max"] < 0.03,
            f"max {casc['flat_2']['max'] * 100:.2f} cm")
    ck.true("cascade alone: without feedforward the error is several times larger",
            casc["zero_2"]["rms"] > 3 * casc["flat_2"]["rms"])
    # the feedforward differentiated along the closed loop is the library's controller
    worst = 0.0
    for _ in range(10):
        st = QUAD.random_state(rng)
        ref = plan(ms.t_start + rng.uniform(0, ms.t_course))[0]
        f_, Rd_, Od_, Odd_ = closed_loop_feedforward(st.x, st.v, st.R, st.Omega, ref, k, p0)
        mine = np.r_[f_, attitude_tracking_torque(st.R, st.Omega, Rd_, Od_, Odd_, p0.J, k.k_R, k.k_Omega)]
        theirs = QUAD.tracking_controller(st, ref, k, p0)
        worst = max(worst, float(np.max(np.abs(mine - np.r_[theirs[0], theirs[1]]))))
    ck.true("closed-loop feedforward with eq6-law is the library's tracking_controller", worst < 1e-12, f"err {worst:.1e}")
    # on the reference, the cascade with flat feedforward returns the flat inputs
    ref, o = plan(ms.t_start + 3.0)
    Fd = p0.m * p0.g * E3 + p0.m * ref[2]
    tau = attitude_tracking_torque(o["R"], o["Omega"], desired_attitude(Fd, ref[5]), o["Omega"], o["Omega_dot"],
                                   p0.J, k.k_R, k.k_Omega)
    ck.close("on the reference the cascade returns the inputs of the flat map", np.r_[Fd @ o["R"][:, 2], tau],
             np.r_[o["f"], o["tau"]], tol=1e-9)

    # ------------------------------------------------------------ local optimality (Chapters 7 and 13)
    Q, Ru = np.diag([1240.0] * 3 + [20.0] * 3), np.eye(3)
    from scipy.linalg import solve_continuous_are
    A7 = V13.err_A(np.zeros(3), p0.J)
    B7 = np.vstack([np.zeros((3, 3)), np.linalg.inv(p0.J)])
    K7 = np.linalg.solve(Ru, B7.T @ solve_continuous_are(A7, B7, Q, Ru))
    want = np.hstack([np.sqrt(1240.0) * I3, np.diag(np.sqrt(20.0 + 2 * np.diag(p0.J) * np.sqrt(1240.0)))])
    ck.close("LQR at rest: K = [sqrt(q_1) I, diag sqrt(q_2 + 2 J_i sqrt(q_1))]", K7, want, tol=1e-6)
    R_rel, Om_rel = rel.R, rel.Omega
    pd = lambda R, Om: np.clip(attitude_tracking_torque(R, Om, I3, np.zeros(3), np.zeros(3), p0.J, k.k_R, k.k_Omega),  # noqa: E731
                               -ms.tau_max, ms.tau_max)
    h_mpc, N_mpc = 0.02, 25
    mpc = AttitudeMPC(p0.J, h_mpc, N_mpc, Q, Ru, ms.tau_max)
    lqr = AttitudeMPC(p0.J, h_mpc, N_mpc, Q, Ru, ms.tau_max, bounded=False)
    loc = dict(K_R=np.diag(K7[:, :3]).tolist(), K_Omega=np.diag(K7[:, 3:]).tolist(), Q=[1240.0, 20.0], N=N_mpc, h=h_mpc)
    t0 = time.time()
    for name, law, hold in (("pd", pd, 1), ("lqr", lqr, 4), ("mpc", mpc, 4)):
        ang, taus = attitude_run(law, R_rel, Om_rel, p0.J, hold=hold)
        loc[name] = dict(settle=settle(ang, 0.005), peak=float(np.degrees(ang.max())),
                         saturated=float(np.mean(np.abs(taus).max(axis=1) > ms.tau_max - 1e-6) * 3.0),
                         effort=float(np.sum(taus**2) * 0.005))
    # a small error: bounds inactive, MPC and LQR coincide, and the PD is close
    R_s, Om_s = expm_so3(np.radians(8.0) * unit([1.0, -1.0, 0.5])), np.array([0.2, 0.1, -0.1])
    mpc.U = None
    gap_small = float(np.max(np.abs(mpc(R_s, Om_s) - lqr(R_s, Om_s))))
    ck.true("bounds inactive: MPC returns the LQR torque", gap_small < 1e-7, f"err {gap_small:.1e}")
    for name, law, hold in (("pd", pd, 1), ("lqr", lqr, 4), ("mpc", mpc, 4)):
        mpc.U = None
        ang, taus = attitude_run(law, R_s, Om_s, p0.J, hold=hold)
        loc[name + "_small"] = dict(settle=settle(ang, 0.005, 1.0), effort=float(np.sum(taus**2) * 0.005))
    weak = {}
    for name, law in (("lqr", AttitudeMPC(p0.J, h_mpc, N_mpc, Q, Ru, 2.0, bounded=False)),
                      ("mpc", AttitudeMPC(p0.J, h_mpc, N_mpc, Q, Ru, 2.0)),
                      ("pd", lambda R, Om: np.clip(pd(R, Om), -2.0, 2.0))):
        ang, taus = attitude_run(law, R_rel, Om_rel, p0.J, hold=1 if name == "pd" else 4)
        weak[name] = dict(settle=settle(ang, 0.005), effort=float(np.sum(taus**2) * 0.005))
    loc["weak"] = weak
    ck.record("local", loc)
    ck.true("from the release attitude, MPC with bounds settles sooner than the clipped PD and the clipped LQR",
            loc["mpc"]["settle"] < loc["pd"]["settle"] and loc["mpc"]["settle"] <= loc["lqr"]["settle"],
            f"mpc {loc['mpc']['settle']:.2f}, lqr {loc['lqr']['settle']:.2f}, pd {loc['pd']['settle']:.2f} s")

    # ------------------------------------------------------------ uncertainty (Chapter 8), no obstacle
    far = replace(ms, x_o=np.array([100.0, 0.0, 0.0]))
    unc = {}
    for name, kw in (("plain", dict(adapt=False)), ("l1", dict())):
        o = fly(far, plan, safety=False, **kw)
        e = int(round((ms.t_start + ms.t_course) / h))
        ang = np.degrees(np.arccos(np.clip(1 - o["Psi"][c:], -1, 1)))
        unc[name] = dict(rms=float(np.sqrt(np.mean(o["e_x"][c:e] ** 2))), max=float(o["e_x"][c:e].max()),
                         hover=float(np.mean(o["e_x"][-200:])), att_rms=float(np.sqrt(np.mean(ang**2))),
                         att_max=float(ang.max()), Df_end=np.mean(o["Df"][-200:], axis=0).tolist(),
                         pred_max=float(o["Psi_pred"][j:].max()),
                         tau_rate=float(np.sqrt(np.mean(np.sum(np.diff(o["tau"][c:e], axis=0) ** 2, axis=1))) / h))
    unc["Delta_e"] = Delta_e.tolist()
    unc["gust_force"], unc["gust_torque_max"] = ms.gust_force, float(max(np.linalg.norm(gust(ms, t)[1]) for t in np.arange(0, 20, 0.01)))
    unc["sag_predicted"] = ms.gust_force / k.k_x
    unc["static_tilt_plain"] = float(np.degrees(np.arcsin(np.linalg.norm(Delta_e) / k.k_R)))
    ck.record("uncertainty", unc)
    ck.true("adaptation: filtered estimate at the final hover equals the offset torque to 0.1 N m",
            np.linalg.norm(np.array(unc["l1"]["Df_end"]) - Delta_e) < 0.1)
    ck.true("adaptation: smaller attitude error and smaller hover error than the plain controller",
            unc["l1"]["att_rms"] < 0.5 * unc["plain"]["att_rms"] and unc["l1"]["hover"] < 0.5 * unc["plain"]["hover"])

    # ------------------------------------------------------------ safety and the ablations
    row("no feedforward", feedforward="zero")
    row("closed-loop feedforward", feedforward="exact")
    row("no adaptation", adapt=False)
    row("no safety filter", safety=False)
    row("euler", attitude="euler")
    row("predictor not told", adapt="untold")
    row("barrier not informed", safety="blind")
    row("chapter 8 as is", adapt="untold", safety="blind")
    row("lee gains", replace(ms, kappa=1.0))
    row("not told, no offset", replace(ms, r_add=np.zeros(3)), adapt="untold", safety="blind")
    row("pitch flip", replace(ms, flip_axis=E2), adapt=False)
    row("pitch flip euler", replace(ms, flip_axis=E2), attitude="euler")
    ck.record("ablation", S)
    throws = []
    for deg in range(0, 360, 30):
        axis = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])
        o = fly(replace(ms, flip_axis=axis), t_end=ms.t_start)
        up = np.nonzero(np.degrees(o["tilt"][j:]) < 10.0)[0]
        throws.append(dict(axis_deg=deg, t_upright=float(up[0] * h) if len(up) else float("nan"),
                           min_z=float(o["x"][j:, 2].min()), tilt0=float(np.degrees(o["tilt"][j])),
                           e_x_end=float(o["e_x"][-1])))
    ck.record("throws", throws)
    seeds = {}
    for al in ((2.0, 4.0), (3.0, 6.0)):
        res = [summary(fly(replace(ms, alpha=al, seed=sd), plan)) for sd in (1, 2, 3)]
        seeds[f"{al[0]:g}_{al[1]:g}"] = dict(min_H=[r_["min_H"] for r_ in res], min_dist=[r_["min_dist"] for r_ in res])
    ck.record("seeds", seeds)
    ck.true("three more draws of the sensor noise: H stays positive with both wires",
            min(seeds["2_4"]["min_H"]) > 0.0, f"smallest {min(seeds['2_4']['min_H']):.3f}")
    a = S["all"]
    ck.true("full mission: the ball is never entered", not a["hit"], f"closest {a['min_dist']:.3f} m > {ms.r_o} m")
    ck.true("full mission: the barrier function is never negative after the release", a["min_H"] >= 0.0,
            f"min H {a['min_H']:.3f}")
    ck.true("full mission: the safety program is feasible at every step", a["infeasible"] == 0)
    ck.true("full mission: hover at the target to 2 cm", a["hover"] < 0.02, f"{a['hover'] * 100:.2f} cm")
    ck.true("full mission: thrust and torque stay within the limits", a["f_peak"] <= ms.f_max and a["tau_peak"] <= ms.tau_max)
    ck.true("without the safety filter the vehicle enters the ball", S["no safety filter"]["hit"],
            f"closest {S['no safety filter']['min_dist']:.3f} m")
    ck.true("predictor not told: the prediction error is many times larger",
            S["chapter 8 as is"]["pred_max"] > 20 * S["barrier not informed"]["pred_max"]
            and S["not told, no offset"]["pred_max"] > 1e3 * a["pred_max"],
            f"{S['chapter 8 as is']['pred_max']:.1e} and {S['not told, no offset']['pred_max']:.1e} "
            f"against {S['barrier not informed']['pred_max']:.1e}")
    ck.true("blocks as their chapters leave them: the vehicle enters the ball", S["chapter 8 as is"]["hit"],
            f"closest {S['chapter 8 as is']['min_dist']:.3f} m")
    ck.true("barrier not informed: H goes negative", S["barrier not informed"]["min_H"] < 0.0,
            f"min H {S['barrier not informed']['min_H']:.3f}")
    ck.true("informed by an untold predictor is worse than not informed",
            S["predictor not told"]["min_H"] < S["barrier not informed"]["min_H"])
    sat = np.abs(full["tau"][c:]).max(axis=1) > ms.tau_max - 1e-9
    ck.record("mission", dict(t_end=ms.t_end, steps=len(full["t"]) - 1, h=h,
                              f_range_course=[float(full["f"][c:].min()), float(full["f"][c:].max())],
                              tau_sat_steps_course=int(sat.sum()), tau_sat_ms=float(sat.sum() * h * 1e3),
                              Df_end=np.mean(full["Df"][-200:], axis=0).tolist(),
                              tilt_max_course=a["tilt_max"]))

    # ------------------------------------------------------------ integration (Chapters 2, 4, 12)
    resid = float(full["manifold"].max())
    ck.true("the simulated state never leaves the manifold", resid < 1e-12, f"residual {resid:.1e}")
    R_naive = full["R"][j].copy()
    for i in range(j, len(full["t"]) - 1):
        Om_i = log_so3(full["R"][i].T @ full["R"][i + 1]) / h
        R_naive = R_naive + h * R_naive @ hat(Om_i)
    naive = float(np.max(np.abs(R_naive.T @ R_naive - I3)))
    drifts = []
    for hh in (0.01, 0.005):
        st = QUAD.State(x=rel.x, v=rel.v, R=rel.R, Omega=rel.Omega + np.array([0.0, 2.0, 0.5]))
        E0, worst = QUAD.energy(st, p_true), 0.0
        for _ in range(int(round(2.0 / hh))):
            st = QUAD.step(st, QUAD.zero_input(), hh, p_true)
            worst = max(worst, abs(QUAD.energy(st, p_true) - E0))
        drifts.append(worst / abs(E0))
    ck.true("motors off: energy conserved, and the drift falls by 16 when the step is halved",
            drifts[1] < 1e-9 and 10 < drifts[0] / drifts[1] < 24, f"{drifts[1]:.1e}, ratio {drifts[0] / drifts[1]:.1f}")
    ck.record("integration", dict(manifold=resid, naive=naive, energy_drift=drifts[1], energy_ratio=drifts[0] / drifts[1],
                                  filter_manifold=float(max(np.max(np.abs(Rh.T @ Rh - I3)) for Rh in full["Rhat"]))))
    ck.true("the estimate never leaves the group either", ck.values["integration"]["filter_manifold"] < 1e-12)
    ck.true("the naive update of sec2-integration would have left the manifold", naive > 1e-3, f"residual {naive:.2f}")

    notebook_checks(ck, rng)
    ck.record("runtime", round(time.time() - t_all, 1))
    ck.finish()


def notebook_checks(ck, rng):
    """The notebook carries copies of the blocks. Each copy is compared with its original here."""
    src = HERE.parent / "notebooks" / "src" / "ch17.py"
    if not src.exists():
        ck.true("notebook source exists", False)
        return
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    cells = [c for c in src.read_text().split("# %%") if not c.lstrip().startswith("[markdown]")]
    ns = {"__name__": "ch17_notebook"}
    sys.modules["ch17_notebook"] = type(sys)("ch17_notebook")
    for c in cells:
        if "# --- run" not in c:
            exec(compile(c, "notebooks/src/ch17.py", "exec"), ns)
    ms = Mission()
    nms = ns["Mission"]()
    # the model and its step
    p_true, rho = true_params(ms)
    mine, theirs = ns["LoadedQuadrotor"](ns["true_params"](nms)[0]), LoadedQuadrotor(p_true)
    worst = 0.0
    for _ in range(10):
        st = QUAD.random_state(rng)
        u = (float(rng.uniform(0, 90)), rng.normal(size=3))
        for q in (mine, theirs):
            q.rho, q.F_e, q.tau_e = rho, np.array([0.0, -2.0, 1.0]), np.array([0.1, 0.2, -0.3])
        a = mine.step(ns["State"](x=st.x, v=st.v, R=st.R, Omega=st.Omega), u, 0.01)
        b = theirs.step(st, u, 0.01)
        worst = max(worst, float(np.max(np.abs(a.x - b.x))), float(np.max(np.abs(a.v - b.v))),
                    float(np.max(np.abs(a.R - b.R))), float(np.max(np.abs(a.Omega - b.Omega))))
    ck.true("notebook: the step of its quadrotor is the library's step", worst < 1e-12, f"err {worst:.1e}")
    # the plan and the flat map
    a, b = ns["Plan"](nms), Plan(ms)
    worst = float(np.max(np.abs(a.spline.coeff - b.spline.coeff)))
    for t in rng.uniform(ms.t_start, ms.t_start + ms.t_course, size=5):
        (ra, oa), (rb, ob) = a(t), b(t)
        worst = max(worst, float(np.max(np.abs(oa["tau"] - ob["tau"]))), abs(oa["f"] - ob["f"]),
                    float(np.max(np.abs(ra[4] - rb[4]))))
    ck.true("notebook: minimum-snap plan and flat map equal those of verify/ch10.py", worst < 1e-12, f"err {worst:.1e}")
    # the whole loop: carry, release and the first second of flight, same random numbers
    t_cmp = ms.t_carry + 1.0
    worst = 0.0
    for kw in ({}, dict(adapt="untold", safety="blind"), dict(attitude="euler", feedforward="exact")):
        o1, o2 = ns["fly"](nms, t_end=t_cmp, **kw), fly(ms, t_end=t_cmp, **kw)
        worst = max(worst, max(float(np.max(np.abs(o1[key] - o2[key]))) for key in ("x", "R", "xhat", "Rhat", "tau", "f", "Df", "H")))
    ck.true("notebook: its mission loop reproduces this script through release and recovery", worst < 1e-8,
            f"err {worst:.1e}")
    # ... and through the obstacle, with everything on
    far_ = dict(t_carry=1.0, t_hold=1.0)
    o1 = ns["fly"](ns["Mission"](**far_))
    o2 = fly(Mission(**far_))
    worst = max(float(np.max(np.abs(o1[key] - o2[key]))) for key in ("x", "tau", "f", "H"))
    ck.true("notebook: a whole (shortened) mission agrees, safety filter and adaptation included", worst < 1e-6,
            f"err {worst:.1e}")


if __name__ == "__main__":
    main()
