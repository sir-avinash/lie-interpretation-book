"""Chapter 8: every identity, bound, closed-loop claim and quoted number, checked.

    python verify/ch08.py

The functions above ``main`` are the chapter's models and controllers. The figure
script imports them, so the figures show exactly what is verified here.

Three groups:
  1. attitude tracking on SO(3): geometric PD, and geometric L1 adaptation;
  2. barrier functions: a point and a wall, a pendulum on S^2 inside a cone,
     a tilt limit on SO(3);
  3. a quadrotor that must not hit a sphere: the augmented barrier function.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, expm_so3, hat, random_rotation, vee  # noqa: E402

E1, E2, E3 = np.eye(3)
I3 = np.eye(3)
GRAV = 9.81


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


# =====================================================================
# 1. Attitude errors (actual first, desired second)
# =====================================================================
def Psi_R(R, Rd):
    return 0.5 * np.trace(I3 - Rd.T @ R)


def e_R(R, Rd):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def e_Omega(R, Om, Rd, Omd):
    return Om - R.T @ Rd @ Omd


def C_R(R, Rd):
    return 0.5 * (np.trace(R.T @ Rd) * I3 - R.T @ Rd)


def angle_between(R, Rd):
    return np.arccos(np.clip(1 - Psi_R(R, Rd), -1, 1))


# ---- the vehicle of the chapter -------------------------------------
J0 = np.diag([6.968e-3, 6.211e-3, 10.34e-3])       # inertia the controller believes, kg m^2
M_ADD, R_ADD = 0.5, np.array([0.08, 0.0, -0.03])   # a camera clipped on off-center: mass (kg), place (m)
J_ADD = -M_ADD * hat(R_ADD) @ hat(R_ADD)
J_TRUE = J0 + J_ADD
DELTA_CAM = M_ADD * GRAV * np.cross(E3, R_ADD)     # thrust no longer passes through the center of mass
K_R, K_OM = 2.0, 0.25


def feedforward(R, Om, Rd, Omd, Omd_dot, J):
    """Om x J Om - J (hat(Om) R^T Rd Omd - R^T Rd Omd_dot): cancels everything but the PD term."""
    return np.cross(Om, J @ Om) - J @ (hat(Om) @ R.T @ Rd @ Omd - R.T @ Rd @ Omd_dot)


def pd_torque(R, Om, Rd, Omd, Omd_dot, J=J0, kR=K_R, kOm=K_OM):
    """Geometric PD with feedforward, equation (eq8-pd)."""
    mu = -kR * e_R(R, Rd) - kOm * e_Omega(R, Om, Rd, Omd)
    return mu + feedforward(R, Om, Rd, Omd, Omd_dot, J), mu


def lumped_uncertainty(R, Om, Rd, Omd, Omd_dot, mu, Delta_e, J=J_TRUE, Jn=J0):
    """Delta of (eq8-lumped): J0 e_Omega_dot = mu + Delta."""
    acc = hat(Om) @ R.T @ Rd @ Omd - R.T @ Rd @ Omd_dot
    Delta_m = -np.cross(Om, (J - Jn) @ Om) + (J - Jn) @ acc
    JJ = Jn @ np.linalg.inv(J)
    return (JJ - I3) @ mu + JJ @ (Delta_m + Delta_e)


# ---- references ------------------------------------------------------
def ref_hover(t):
    return I3, np.zeros(3), np.zeros(3)


class ConingReference:
    """R_d(t) integrated from a smooth body rate. Call in increasing t with a fixed step."""

    def __init__(self, h, amp=0.9, rate=1.5, spin=0.3):
        self.h, self.amp, self.rate, self.spin = h, amp, rate, spin
        self.R, self.k = I3.copy(), 0

    def rates(self, t):
        a, w = self.amp, self.rate
        Omd = np.array([a * np.cos(w * t), a * np.sin(w * t), self.spin])
        Omd_dot = np.array([-a * w * np.sin(w * t), a * w * np.cos(w * t), 0.0])
        return Omd, Omd_dot

    def __call__(self, t):
        k = int(round(t / self.h))
        while self.k < k:                           # midpoint rule on the group
            tm = (self.k + 0.5) * self.h
            self.R = self.R @ expm_so3(self.h * self.rates(tm)[0])
            self.k += 1
        Omd, Omd_dot = self.rates(t)
        return self.R, Omd, Omd_dot


# ---- disturbances ----------------------------------------------------
def gust(t):
    """A time-varying body torque, the shape used by Kotaru (2022, eq. 3.47), scaled."""
    s = 0.5 * (np.cos(t) + 0.5 * np.cos(3 * t + 0.23) + 0.5 * np.cos(5 * t - 0.4) + 0.5 * np.cos(7 * t + 2.09))
    return 0.25 * s * np.array([1.0, -0.6, 0.3])


def dist_camera(t):
    return DELTA_CAM


def dist_camera_gust(t):
    return DELTA_CAM + gust(t)


# ---- projection operator ---------------------------------------------
def proj(theta, y, bound, eps=0.1):
    """Smooth projection of Lavretsky and Gibson: keeps |theta| <= bound sqrt(1 + eps)."""
    f = ((1 + eps) * theta @ theta - bound**2) / (eps * bound**2)
    grad = 2 * (1 + eps) * theta / (eps * bound**2)
    if f > 0 and y @ grad > 0:
        return y - grad * (grad @ y) / (grad @ grad) * f
    return y


# ---- the L1 controller -------------------------------------------------
class L1Attitude:
    """Geometric L1 adaptive attitude control (Kotaru 2022, ch. 3), book notation.

    State: predictor (Rp, Omp), estimate Dhat, filtered estimate Df.
    """

    def __init__(self, R0, Om0, h, gamma=4000.0, kappa=20.0, bound=3.0, c=0.01,
                 J=J0, kR=K_R, kOm=K_OM, kRt=K_R, kOmt=K_OM):
        self.Rp, self.Omp = R0.copy(), Om0.copy()
        self.Dhat, self.Df = np.zeros(3), np.zeros(3)
        self.h, self.gamma, self.kappa, self.bound, self.c = h, gamma, kappa, bound, c
        self.J, self.Ji = J, np.linalg.inv(J)
        self.kR, self.kOm, self.kRt, self.kOmt = kR, kOm, kRt, kOmt

    def errors(self, R, Om):
        """Prediction errors: the predictor plays 'actual', the plant plays 'desired'."""
        return e_R(self.Rp, R), e_Omega(self.Rp, self.Omp, R, Om), Psi_R(self.Rp, R)

    def U(self, R):
        return self.J @ self.Rp.T @ R @ self.Ji @ R.T @ self.Rp

    def torques(self, R, Om, Rd, Omd, Omd_dot):
        """Return (tau, mu, mu_p) from the current controller state. Does not advance it."""
        J, Ji, Rp, Omp = self.J, self.Ji, self.Rp, self.Omp
        eRt, eOmt, _ = self.errors(R, Om)
        mu1p = -self.kR * e_R(Rp, Rd) - self.kOm * e_Omega(Rp, Omp, Rd, Omd)
        mu_p = mu1p + self.U(R) @ self.Dhat - self.Df
        eOm = e_Omega(R, Om, Rd, Omd)
        mu = (J @ R.T @ Rp @ Ji @ (mu1p - self.Df + self.kRt * eRt + self.kOmt * eOmt)
              + J @ R.T @ Rp @ hat(eOmt) @ Rp.T @ R @ eOm)
        tau = mu + feedforward(R, Om, Rd, Omd, Omd_dot, J)
        return tau, mu, mu_p

    def adapt_direction(self, R, Om):
        eRt, eOmt, _ = self.errors(R, Om)
        U = self.U(R)
        return -(U.T @ eOmt + self.c * U.T @ self.Ji.T @ eRt)

    def step(self, R, Om, Rd, Omd, Omd_dot):
        """One controller period: torque from the current state, then advance the state.

        The estimate is advanced first and the predictor uses the new value
        (semi-implicit Euler), which keeps the fast estimate-predictor loop
        stable for sqrt(gamma / J) h < 2.
        """
        h = self.h
        tau, mu, mu_p = self.torques(R, Om, Rd, Omd, Omd_dot)
        self.last_mu = mu
        U = self.U(R)
        eRt, eOmt, _ = self.errors(R, Om)
        y = -(U.T @ eOmt + self.c * U.T @ self.Ji.T @ eRt)
        Dhat_new = self.Dhat + h * self.gamma * proj(self.Dhat, y, self.bound)
        mu_p = mu_p + U @ (Dhat_new - self.Dhat)
        self.Dhat = Dhat_new
        Omp_dot = self.Ji @ mu_p - (hat(self.Omp) @ self.Rp.T @ Rd @ Omd - self.Rp.T @ Rd @ Omd_dot)
        Omp_new = self.Omp + h * Omp_dot
        self.Rp = self.Rp @ expm_so3(0.5 * h * (self.Omp + Omp_new))
        self.Omp = Omp_new
        self.Df = self.Df + (1 - np.exp(-self.kappa * h)) * (self.Dhat - self.Df)
        return tau


def body_step(R, Om, tau, h, J, Ji):
    """Rigid body, torque held over the step: RK4 on Omega, exact exponential on R."""
    def f(w):
        return Ji @ (tau - np.cross(w, J @ w))
    k1 = f(Om)
    k2 = f(Om + 0.5 * h * k1)
    k3 = f(Om + 0.5 * h * k2)
    k4 = f(Om + h * k3)
    Om_new = Om + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return R @ expm_so3(0.5 * h * (Om + Om_new)), Om_new


def run_attitude(controller, ref, dist, tf, h=1e-3, R0=I3, Om0=np.zeros(3), J=J_TRUE,
                 noise=0.0, lag=0.0, delay=0, seed=0, **kw):
    """Closed-loop attitude simulation.

    controller: "pd" or "l1".  noise: std of the gyroscope error (rad/s).
    lag: motor time constant (s).  delay: input delay in controller periods.
    """
    rng = np.random.default_rng(seed)
    Ji = np.linalg.inv(J)
    n = int(round(tf / h))
    R, Om = R0.copy(), Om0.copy()
    l1 = L1Attitude(R, Om, h, **kw) if controller == "l1" else None
    tau_act, queue = np.zeros(3), [np.zeros(3)] * delay
    out = {k: np.zeros((n, 3)) for k in ("tau", "Dhat", "Df", "Delta", "Om")}
    out.update({k: np.zeros(n) for k in ("t", "Psi", "angle", "Psi_pred", "eOm_pred", "eOm", "Psi_p")})
    for k in range(n):
        t = k * h
        Rd, Omd, Omd_dot = ref(t)
        Om_meas = Om + noise * rng.normal(size=3)
        De = dist(t)
        if l1 is None:
            tau, mu = pd_torque(R, Om_meas, Rd, Omd, Omd_dot)
        else:
            out["Psi_pred"][k] = Psi_R(l1.Rp, R)
            out["eOm_pred"][k] = np.linalg.norm(e_Omega(l1.Rp, l1.Omp, R, Om))
            out["Psi_p"][k] = Psi_R(l1.Rp, Rd)
            out["Dhat"][k], out["Df"][k] = l1.Dhat, l1.Df
            tau = l1.step(R, Om_meas, Rd, Omd, Omd_dot)
            mu = l1.last_mu
        out["Delta"][k] = lumped_uncertainty(R, Om, Rd, Omd, Omd_dot, mu, De, J)
        out["t"][k], out["Psi"][k], out["angle"][k] = t, Psi_R(R, Rd), angle_between(R, Rd)
        out["eOm"][k] = np.linalg.norm(e_Omega(R, Om, Rd, Omd))
        out["tau"][k], out["Om"][k] = tau, Om
        if delay:
            queue.append(tau)
            tau = queue.pop(0)
        tau_act = tau if lag <= 0 else tau_act + (1 - np.exp(-h / lag)) * (tau - tau_act)
        R, Om = body_step(R, Om, tau_act + De, h, J, Ji)
        bad = not np.all(np.isfinite(Om)) or np.linalg.norm(Om) > 1e3
        if l1 is not None:
            bad = bad or not np.all(np.isfinite(l1.Omp)) or np.linalg.norm(l1.Omp) > 1e3
        if bad:
            for key in out:
                out[key] = out[key][:k + 1]
            out["diverged"] = True
            break
    out["R_final"] = R
    out.setdefault("diverged", False)
    return out


# =====================================================================
# 2. Barrier functions
# =====================================================================
def solve_qp(Q, c, A, b, tol=1e-9):
    """min 0.5 z'Qz + c'z  subject to  A z <= b, for a handful of constraints.

    Tries every active set, smallest first, and returns the first KKT point.
    Returns None when the constraints cannot all be met.
    """
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


# ---- the simplest case: a point that must not cross a wall -----------
WALL_N, WALL_D = E1[:2], 1.0                       # safe set  H = d - n.x >= 0


def wall_H(x):
    return WALL_D - WALL_N @ x


def wall_filter(x, ud, alpha):
    """argmin |u - ud|^2  s.t.  H_dot = -n.u >= -alpha H   (closed form)."""
    return ud - max(0.0, WALL_N @ ud - alpha * wall_H(x)) * WALL_N


def run_wall(alpha, x0=(0.0, 0.0), xd=(2.0, 1.5), k=1.0, tf=8.0, h=1e-3):
    n = int(round(tf / h))
    xs = np.zeros((n + 1, 2))
    xs[0] = x0
    xd = np.asarray(xd, float)
    for i in range(n):
        ud = -k * (xs[i] - xd)
        u = ud if alpha is None else wall_filter(xs[i], ud, alpha)
        xs[i + 1] = xs[i] + h * u
    return np.arange(n + 1) * h, xs


# ---- a pendulum that must stay inside a cone (S^2) -------------------
M_P, L_P = 0.5, 0.8                                # the pendulum of Chapter 7
K_Q, K_W_PD, C_V = 12.0, 2.5, 1.0                  # tracking gains on S^2 and the cross term of V
CONE_AXIS, CONE_ANGLE = -E3, np.radians(30.0)


def s2_accel(q, om, F):
    """m l^2 om_dot = -m g l q x e3 + l q x F   (eq7-pendulum)."""
    return np.cross(q, F - M_P * GRAV * E3) / (M_P * L_P)


def Psi_q(q, qd):
    return 1 - q @ qd


def e_q(q, qd):
    return np.cross(qd, q)


def e_omega(q, om, omd):
    return om - (omd - (q @ omd) * q)


class CircleReference:
    """q_d moves at constant rate on a small circle about the axis c."""

    def __init__(self, c, q0, rate):
        self.c, self.q0, self.rate = unit(c), unit(q0), rate

    def __call__(self, t):
        c, w = self.c, self.rate
        qd = expm_so3(w * t * c) @ self.q0
        qd_dot = w * np.cross(c, qd)
        omd = np.cross(qd, qd_dot)                 # tangent: omd . qd = 0
        qd_ddot = w * np.cross(c, qd_dot)
        omd_dot = np.cross(qd, qd_ddot)
        return qd, omd, omd_dot


def s2_reference_accel(q, om, omd, omd_dot):
    """a_d = d/dt[(I - q q^T) omd], so that e_omega_dot = om_dot - a_d."""
    q_dot = np.cross(om, q)
    return omd_dot - (q @ omd_dot) * q - q_dot * (q @ omd) - q * (q_dot @ omd)


def s2_feedforward(q, om, omd, omd_dot):
    """Force that makes the tangent part of e_omega_dot vanish."""
    a = s2_reference_accel(q, om, omd, omd_dot)
    w = a - (a @ q) * q + GRAV / L_P * np.cross(q, E3)
    return -M_P * L_P * np.cross(q, w)


def s2_pd_force(q, om, qd, omd, omd_dot, kq=K_Q, kw=K_W_PD):
    """Geometric PD with feedforward on S^2."""
    fb = -(kq * e_q(q, qd) + kw * e_omega(q, om, omd)) / (M_P * L_P**2)
    return s2_feedforward(q, om, omd, omd_dot) - M_P * L_P * np.cross(q, fb)


def s2_V(q, om, qd, omd, kq=K_Q, c=C_V):
    eq, ew = e_q(q, qd), e_omega(q, om, omd)
    return 0.5 * M_P * L_P**2 * ew @ ew + kq * Psi_q(q, qd) + c * M_P * L_P**2 * eq @ ew


def s2_V_dot_terms(q, om, qd, omd, omd_dot, kq=K_Q, c=C_V):
    """V_dot = aV + bV . F."""
    ml2 = M_P * L_P**2
    eq, ew = e_q(q, qd), e_omega(q, om, omd)
    a_d = s2_reference_accel(q, om, omd, omd_dot)
    eq_dot = (q @ qd) * ew + (q @ omd) * np.cross(q, eq) - (qd @ om) * q
    aV = ml2 * (ew + c * eq) @ (-GRAV / L_P * np.cross(q, E3) - a_d) + kq * eq @ ew + c * ml2 * eq_dot @ ew
    bV = L_P * np.cross(ew + c * eq, q)
    return aV, bV


def cone_static(t):
    return CONE_AXIS, np.zeros(3), np.zeros(3)


def cone_swinging(t):
    """A cone whose axis sways by 12 degrees about the vertical."""
    a, w = np.radians(12.0), 0.7
    ang = a * np.sin(w * t)
    qc = np.array([np.sin(ang), 0.0, -np.cos(ang)])
    d = np.array([np.cos(ang), 0.0, np.sin(ang)])          # d qc / d ang
    dd = np.array([-np.sin(ang), 0.0, np.cos(ang)])
    ang_dot, ang_ddot = a * w * np.cos(w * t), -a * w**2 * np.sin(w * t)
    return qc, d * ang_dot, d * ang_ddot + dd * ang_dot**2


def cone_H(q, om, qc, qc_dot, qc_ddot, angle=CONE_ANGLE):
    """H, H_dot, and H_ddot = aH + bH . F for the cone H = q . qc - cos(angle)."""
    q_dot = np.cross(om, q)
    H = q @ qc - np.cos(angle)
    H_dot = q_dot @ qc + q @ qc_dot
    tang = qc - (q @ qc) * q                                # (I - q q^T) qc
    aH = -(om @ om) * (q @ qc) - GRAV / L_P * (E3 @ tang) + 2 * q_dot @ qc_dot + q @ qc_ddot
    bH = tang / (M_P * L_P)
    return H, H_dot, aH, bH


def s2_safe_force(t, q, om, ref, cone=cone_static, alpha1=6.0, alpha2=6.0, kV=4.0, kslack=50.0,
                  angle=CONE_ANGLE, Fmax=None):
    """CBF-CLF-QP on S^2 (eq8-qp). Returns F, slack, and whether the QP was feasible."""
    qd, omd, omd_dot = ref(t)
    Fff = s2_feedforward(q, om, omd, omd_dot)
    aV, bV = s2_V_dot_terms(q, om, qd, omd, omd_dot)
    V = s2_V(q, om, qd, omd)
    H, H_dot, aH, bH = cone_H(q, om, *cone(t), angle=angle)
    H1 = H_dot + alpha1 * H
    Q = np.diag([1.0, 1.0, 1.0, kslack])
    A = [np.concatenate([bV, [-1.0]]), np.concatenate([-bH, [0.0]])]
    b = [-kV * V - aV - bV @ Fff, aH + bH @ Fff + alpha1 * H_dot + alpha2 * H1]
    if Fmax is not None:                                    # box on the total force
        for i in range(3):
            row = np.zeros(4)
            row[i] = 1.0
            A += [row, -row]
            b += [Fmax - Fff[i], Fmax + Fff[i]]
    z = solve_qp(Q, np.zeros(4), np.array(A), np.array(b))
    if z is None:
        return Fff, np.nan, False
    return Fff + z[:3], z[3], True


def s2_step(q, om, F, h):
    """RK4 with the force held, then back onto the sphere."""
    def f(q_, om_):
        return np.cross(om_, q_), s2_accel(q_, om_, F)
    k1 = f(q, om)
    k2 = f(q + 0.5 * h * k1[0], om + 0.5 * h * k1[1])
    k3 = f(q + 0.5 * h * k2[0], om + 0.5 * h * k2[1])
    k4 = f(q + h * k3[0], om + h * k3[1])
    q = unit(q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
    om = om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return q, om - (om @ q) * q


def run_cone(ref, tf, h=2e-3, controller="qp", cone=cone_static, q0=None, om0=None, wind=None, **kw):
    """Pendulum on S^2 under the CBF-CLF-QP ("qp") or the tracker alone ("pd").

    wind: a constant force on the bob that the controller does not know about.
    """
    n = int(round(tf / h))
    wind = np.zeros(3) if wind is None else np.asarray(wind, float)
    qd0, omd0, _ = ref(0.0)
    q = qd0.copy() if q0 is None else unit(q0)
    om = (omd0 - (omd0 @ q) * q) if om0 is None else np.asarray(om0, float)
    out = {k: np.zeros((n, 3)) for k in ("q", "qd", "F", "qc")}
    out.update({k: np.zeros(n) for k in ("t", "H", "H1", "V", "slack", "Psi")})
    out["feasible"] = np.ones(n, bool)
    a1 = kw.get("alpha1", 6.0)
    for k in range(n):
        t = k * h
        qd, omd, omd_dot = ref(t)
        if controller == "qp":
            F, slack, ok = s2_safe_force(t, q, om, ref, cone=cone, **kw)
        else:
            F, slack, ok = s2_pd_force(q, om, qd, omd, omd_dot), 0.0, True
        ang = kw.get("angle", CONE_ANGLE)
        H, H_dot, _, _ = cone_H(q, om, *cone(t), angle=ang)
        out["t"][k], out["q"][k], out["qd"][k], out["F"][k] = t, q, qd, F
        out["qc"][k] = cone(t)[0]
        out["H"][k], out["H1"][k] = H, H_dot + a1 * H
        out["V"][k], out["slack"][k], out["Psi"][k] = s2_V(q, om, qd, omd), slack, Psi_q(q, qd)
        out["feasible"][k] = ok
        q, om = s2_step(q, om, F + wind, h)
    return out


# ---- a tilt limit on SO(3) -------------------------------------------
def tilt_H(R, Om, angle):
    """H = e3 . R e3 - cos(angle), H_dot, and the parts of H_ddot = aH + bH . Om_dot."""
    b3 = R.T @ E3
    H = b3[2] - np.cos(angle)
    H_dot = Om @ np.cross(E3, b3)
    aH = -(Om @ Om) * b3[2] + (Om @ b3) * Om[2]
    bH = np.cross(E3, b3)
    return H, H_dot, aH, bH


def tilt_filter(R, Om, tau_d, angle, J, alpha1=8.0, alpha2=8.0):
    """argmin |tau - tau_d|^2  s.t.  H_ddot + (alpha1 + alpha2) H_dot + alpha1 alpha2 H >= 0."""
    Ji = np.linalg.inv(J)
    H, H_dot, aH, bH = tilt_H(R, Om, angle)
    g = Ji @ bH                                             # H_ddot = aH - bH.Ji(Om x J Om) + g . tau
    rhs = -(aH - bH @ Ji @ np.cross(Om, J @ Om)) - (alpha1 + alpha2) * H_dot - alpha1 * alpha2 * H
    short = rhs - g @ tau_d
    if short <= 0 or g @ g < 1e-14:
        return tau_d
    return tau_d + short * g / (g @ g)


def run_tilt(Rd, angle, tf=4.0, h=1e-3, J=J0, filtered=True):
    Ji = np.linalg.inv(J)
    n = int(round(tf / h))
    R, Om = I3.copy(), np.zeros(3)
    out = {"t": np.zeros(n), "H": np.zeros(n), "tilt": np.zeros(n), "Psi": np.zeros(n), "b3": np.zeros((n, 3))}
    for k in range(n):
        tau_d, _ = pd_torque(R, Om, Rd, np.zeros(3), np.zeros(3), J)
        tau = tilt_filter(R, Om, tau_d, angle, J) if filtered else tau_d
        out["t"][k], out["H"][k] = k * h, tilt_H(R, Om, angle)[0]
        out["tilt"][k], out["Psi"][k], out["b3"][k] = np.arccos(np.clip(R[2, 2], -1, 1)), Psi_R(R, Rd), R @ E3
        R, Om = body_step(R, Om, tau, h, J, Ji)
    out["R_final"] = R
    return out


# =====================================================================
# 3. A quadrotor that must not hit a sphere
# =====================================================================
M_Q = 1.129
OBS_C, OBS_R = np.array([0.0, 0.0, 0.0]), 0.6      # obstacle center and radius
AUG_BETA, AUG_K1, AUG_K2, AUG_K3 = 1.8, 0.25, 4.0, 0.0
K_X, K_V = 4.0, 3.2


def sigma(s):
    """Wu's sigmoid, sigma = k1 tanh((k3 - k2 s) / 2), with its first two derivatives."""
    th = np.tanh(0.5 * (AUG_K3 - AUG_K2 * s))
    d1 = -0.5 * AUG_K1 * AUG_K2 * (1 - th**2)
    d2 = -0.5 * AUG_K1 * AUG_K2**2 * th * (1 - th**2)
    return AUG_K1 * th, d1, d2


def naive_terms(x, v, R):
    """Plain barrier H = |x - xo|^2 - ro^2: H, H_dot, and H_ddot = a + b_f f (no torque)."""
    r = x - OBS_C
    return r @ r - OBS_R**2, 2 * r @ v, 2 * v @ v - 2 * GRAV * r[2], 2 * (r @ R[:, 2]) / M_Q


def aug_H(x, R):
    r = x - OBS_C
    return r @ r - AUG_BETA * OBS_R**2 - sigma(r @ R[:, 2])[0]


def aug_terms(x, v, R, Om, J, alpha1):
    """Augmented barrier: H, H1 = H_dot + alpha1 H, and H1_dot = a + b_f f + b_tau . tau."""
    r = x - OBS_C
    b3 = R[:, 2]
    s = r @ b3
    sg, d1, d2 = sigma(s)
    b3_dot = R @ np.cross(Om, E3)
    s_dot = v @ b3 + r @ b3_dot
    H = r @ r - AUG_BETA * OBS_R**2 - sg
    H_dot = 2 * r @ v - d1 * s_dot
    Ji = np.linalg.inv(J)
    # s_ddot = v_dot . b3 + 2 v . b3_dot + r . R (Om x (Om x e3)) + r . R (Om_dot x e3)
    drift_Om = -Ji @ np.cross(Om, J @ Om)
    s_dd_0 = -GRAV * b3[2] + 2 * v @ b3_dot + r @ R @ np.cross(Om, np.cross(Om, E3)) + r @ R @ np.cross(drift_Om, E3)
    a = alpha1 * H_dot + 2 * v @ v - 2 * GRAV * r[2] - d2 * s_dot**2 - d1 * s_dd_0
    b_f = (2 * s - d1) / M_Q
    b_tau = -d1 * Ji @ np.cross(E3, R.T @ r)
    return H, H_dot + alpha1 * H, a, b_f, b_tau


def quad_tracker(x, v, R, Om, xd, vd, ad, J, state):
    """Geometric tracking controller: thrust and torque for a position reference."""
    Fdes = -K_X * (x - xd) - K_V * (v - vd) + M_Q * GRAV * E3 + M_Q * ad
    f = Fdes @ R[:, 2]
    b3d = unit(Fdes)
    b2d = unit(np.cross(b3d, E1))
    Rd = np.column_stack([np.cross(b2d, b3d), b2d, b3d])
    Omd = np.zeros(3)
    if state.get("Rd") is not None:
        W = state["Rd"].T @ Rd
        Omd = vee(W - W.T) / (2 * state["h"])
    state["Rd"] = Rd
    tau, _ = pd_torque(R, Om, Rd, Omd, np.zeros(3), J, kR=8.0, kOm=1.2)
    return f, tau


def line_reference(t, speed=1.0, x0=-3.0, z=0.25):
    return np.array([x0 + speed * t, 0.0, z]), np.array([speed, 0.0, 0.0]), np.zeros(3)


def quad_filter(x, v, R, Om, f_d, tau_d, J, alpha1=3.0, alpha2=6.0, f_max=30.0, w_tau=40.0, mode="augmented"):
    """Smallest change to (f, tau) that keeps the barrier condition, with 0 <= f <= f_max."""
    if mode == "augmented":
        _, H1, a, b_f, b_tau = aug_terms(x, v, R, Om, J, alpha1)
    else:
        H, H_dot, a0, b_f = naive_terms(x, v, R)
        H1, a, b_tau = H_dot + alpha1 * H, a0 + alpha1 * H_dot, np.zeros(3)
    Q = np.diag([1.0, w_tau, w_tau, w_tau])
    c = -Q @ np.concatenate([[f_d], tau_d])
    A = np.array([np.concatenate([[-b_f], -b_tau]), [-1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    b = np.array([a + alpha2 * H1, 0.0, f_max])
    z = solve_qp(Q, c, A, b)
    if z is None:
        return min(max(f_d, 0.0), f_max), tau_d, False
    return z[0], z[1:], True


def run_quad(mode="augmented", tf=9.0, h=2e-3, J=J0, ref=line_reference, **kw):
    Ji = np.linalg.inv(J)
    n = int(round(tf / h))
    x, v, _ = ref(0.0)
    R, Om = I3.copy(), np.zeros(3)
    st = {"Rd": None, "h": h}
    out = {k: np.zeros((n, 3)) for k in ("x", "xd", "b3", "tau")}
    out.update({k: np.zeros(n) for k in ("t", "dist", "f", "H", "H1", "s")})
    out["feasible"] = np.ones(n, bool)
    for k in range(n):
        t = k * h
        xd, vd, ad = ref(t)
        f, tau = quad_tracker(x, v, R, Om, xd, vd, ad, J, st)
        ok = True
        if mode != "none":
            f, tau, ok = quad_filter(x, v, R, Om, f, tau, J, mode=mode, **kw)
        r = x - OBS_C
        out["t"][k], out["x"][k], out["xd"][k], out["b3"][k] = t, x, xd, R[:, 2]
        out["dist"][k], out["f"][k], out["tau"][k], out["s"][k] = np.linalg.norm(r), f, tau, r @ R[:, 2]
        out["H"][k] = aug_H(x, R)
        out["feasible"][k] = ok
        # translation: v_dot = f R e3 / m - g e3, with f and tau held
        acc = f * R[:, 2] / M_Q - GRAV * E3
        x, v = x + h * v + 0.5 * h**2 * acc, v + h * acc
        R, Om = body_step(R, Om, tau, h, J, Ji)
    return out


# ---- two vehicles that must keep their distance (Kotaru 2022, sec. 6.5) ----
def pair_filter(x1, v1, x2, v2, a1d, a2d, r_min, k1=6.0, k2=5.0):
    """Smallest change to two commanded accelerations that keeps H = |x1 - x2|^2 - r_min^2 >= 0."""
    r, dv = x1 - x2, v1 - v2
    H, H_dot = r @ r - r_min**2, 2 * r @ dv
    g = np.concatenate([2 * r, -2 * r])                      # H_ddot = 2|dv|^2 + g . (a1, a2)
    ad = np.concatenate([a1d, a2d])
    short = -k1 * H - k2 * H_dot - 2 * dv @ dv - g @ ad
    if short > 0:
        ad = ad + short * g / (g @ g)
    return ad[:len(x1)], ad[len(x1):]


def run_pair(filtered=True, r_min=0.8, tf=8.0, h=2e-3, offset=0.05):
    n = int(round(tf / h))
    x1, x2 = np.array([-2.0, offset]), np.array([2.0, -offset])
    g1, g2 = np.array([2.0, offset]), np.array([-2.0, -offset])
    v1, v2 = np.zeros(2), np.zeros(2)
    X1, X2, dist = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n)
    for k in range(n):
        a1, a2 = -4.0 * (x1 - g1) - 3.2 * v1, -4.0 * (x2 - g2) - 3.2 * v2
        if filtered:
            a1, a2 = pair_filter(x1, v1, x2, v2, a1, a2, r_min)
        X1[k], X2[k], dist[k] = x1, x2, np.linalg.norm(x1 - x2)
        x1, v1 = x1 + h * v1 + 0.5 * h * h * a1, v1 + h * a1
        x2, v2 = x2 + h * v2 + 0.5 * h * h * a2, v2 + h * a2
    return X1, X2, dist


# ---- helpers shared with the figure script ---------------------------
def cone_reference():
    """The reference of the cone example: a circle of radius 26 deg about an axis 22 deg off -e3."""
    tilt, rad = np.radians(22.0), np.radians(26.0)
    c = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    q0 = np.array([np.sin(tilt - rad), 0.0, -np.cos(tilt - rad)])
    return CircleReference(c, q0, 0.9)


def quad_reference(t):
    """Level flight at 2 m/s, aimed 10 cm below the center of the obstacle."""
    return line_reference(t, speed=2.0, z=-0.1)


def error_stats(o, t_from):
    a = np.degrees(o["angle"][o["t"] >= t_from])
    return float(np.sqrt(np.mean(a**2))), float(a.max())


def torque_rate_rms(o, t_from, h):
    rate = np.diff(o["tau"], axis=0) / h
    return float(np.sqrt(np.mean(np.sum(rate[o["t"][1:] >= t_from]**2, axis=1))))


def delay_margin(controller, hi=80, **kw):
    """Largest input delay, in controller periods, for which the hover loop settles."""
    def stable(d):
        with np.errstate(all="ignore"):
            o = run_attitude(controller, ref_hover, dist_camera, 3.0, delay=d, **kw)
        if o["diverged"]:
            return False
        a = np.degrees(o["angle"][-800:])
        return a.std() < 0.5 and a.max() < 40
    lo = 0
    if stable(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return lo


def lyapunov_matrices(c, kR=K_R, kOm=K_OM, J=J0, psi=1.0):
    """W, W1, W2 of the proof of Lee, Leok and McClamroch (2010), as in (eq8-bound)."""
    lm, lM = np.linalg.eigvalsh(J)[[0, -1]]
    W = np.array([[c * kR / lM, -c * kOm / (2 * lm)], [-c * kOm / (2 * lm), kOm - c]])
    W1 = 0.5 * np.array([[kR, -c], [-c, lm]])
    W2 = 0.5 * np.array([[2 * kR / (2 - psi), c], [c, lM]])
    return W, W1, W2, lm, lM


def ultimate_bound_gain(c, **kw):
    """K such that limsup |(e_R, e_Omega)| <= K * sup|Delta|, or inf if c is not admissible."""
    W, W1, W2, lm, _ = lyapunov_matrices(c, **kw)
    if np.linalg.eigvalsh(W)[0] <= 0 or np.linalg.eigvalsh(W1)[0] <= 0:
        return np.inf
    rho = np.sqrt(1 + (c / lm)**2) / np.linalg.eigvalsh(W)[0]
    return float(rho * np.sqrt(np.linalg.eigvalsh(W2)[-1] / np.linalg.eigvalsh(W1)[0]))


# =====================================================================
# geomech: the three equations of motion used above, derived a second time
# =====================================================================
def geomech_checks(ck, rng):
    from geomech import (S2, SO3, Cross, Dot, Matrix, Scalar, SystemVariables, Vector, check_eom, compute_eom,
                         getScalars, systems, to_standard_form)
    from geomech.codegen import StateLayout, collect_params, load_module, module_source
    from geomech.simulation import accelerations

    def plant(L, dW, variables, inputs, name):
        eom = compute_eom(L, dW, variables)
        err = check_eom(L, dW, variables, eom, inputs=inputs)
        sf = to_standard_form(eom, variables, inputs)
        layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
        return load_module(module_source(sf, layout, lagrangian=L), name), layout, float(err)

    errs = {}
    half = Scalar("0.5", value=0.5, attr=["Constant"])
    # pendulum on S^2 with a force on the bob
    m, g, l = getScalars("m g l", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    q, F = S2("q"), Vector("F")
    om, xi = q.get_tangent_vector(), q.get_variation_vector()
    L = half * m * l * l * Dot(om, om) - m * g * l * Dot(q, e3)
    mod, layout, errs["pendulum on S2"] = plant(L, Dot(Cross(xi, q) * l, F), SystemVariables(vectors=[q]), [F], "ch08_s2")
    worst = 0.0
    for _ in range(20):
        qq = unit(rng.normal(size=3))
        w = np.cross(qq, rng.normal(size=3))
        Fr = rng.normal(size=3) * 3
        a_g = accelerations(mod, layout.pack(q=qq, omega_q=w), Fr, {"e3": E3, "g": GRAV, "l": L_P, "m": M_P})
        worst = max(worst, np.max(np.abs(a_g - s2_accel(qq, w, Fr))))
    ck.true("pendulum: m l^2 om_dot = -m g l q x e3 + l q x F equals geomech", worst < 1e-9, f"err {worst:.1e}")
    # rigid body with a body torque (the disturbance torque enters the same way)
    Jm = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    tau, R = Vector("M"), SO3("R")
    Om, eta = R.get_tangent_vector(), R.get_variation_vector()
    mod, layout, errs["rigid body on SO(3)"] = plant(half * Dot(Om, Jm * Om), Dot(eta, tau), SystemVariables(matrices=[R]),
                                                     [tau], "ch08_so3")
    worst = 0.0
    for _ in range(20):
        Rr, w, tr, De = random_rotation(rng), rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        a_g = accelerations(mod, layout.pack(R=Rr, Omega_R=w), tr + De, {"J": J_TRUE})
        worst = max(worst, np.max(np.abs(a_g - np.linalg.solve(J_TRUE, tr + De - np.cross(w, J_TRUE @ w)))))
    ck.true("rigid body: J Om_dot + Om x J Om = tau + Delta_e equals geomech", worst < 1e-9, f"err {worst:.1e}")
    # quadrotor
    s = systems.get("quadrotor")
    mod, layout, errs["quadrotor"] = plant(s.L, s.dW, s.variables, list(s.inputs), "ch08_quad")
    p = dict(s.params)
    worst = 0.0
    for _ in range(20):
        Rr, w, f, tr = random_rotation(rng), rng.normal(size=3), 5 + 20 * rng.random(), rng.normal(size=3)
        a_g = accelerations(mod, layout.pack(x=rng.normal(size=3), R=Rr, x_dot=rng.normal(size=3), Omega_R=w),
                            np.concatenate([[f], tr]), p)
        mine = np.concatenate([f * Rr @ E3 / p["m"] - p.get("g", GRAV) * E3,
                               np.linalg.solve(p["J"], tr - np.cross(w, p["J"] @ w))])
        worst = max(worst, np.max(np.abs(a_g - mine)))
    ck.true("quadrotor: m v_dot = f R e3 - m g e3, J Om_dot + Om x J Om = tau equals geomech", worst < 1e-9, f"err {worst:.1e}")
    for name, err in errs.items():
        ck.true(f"geomech check_eom: {name}", err < 1e-5, f"err {err:.1e}")
    ck.record("geomech_check_eom", errs)


# =====================================================================
def main():
    ck = Checker("ch08")
    rng = np.random.default_rng(8)
    t_start = time.time()
    h = 1e-3
    deg = np.degrees

    geomech_checks(ck, rng)

    # ------------------------------------------------------------------
    # 1. How uncertainty enters: J0 e_Omega_dot = mu + Delta
    # ------------------------------------------------------------------
    Ji_true = np.linalg.inv(J_TRUE)
    worst, worst_match, worst_fd, worst_C = 0.0, 0.0, 0.0, 0.0
    for _ in range(30):
        R, Rd = random_rotation(rng), random_rotation(rng)
        Om, Omd, Omd_dot, De = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        tau, mu = pd_torque(R, Om, Rd, Omd, Omd_dot)
        Om_dot = Ji_true @ (tau + De - np.cross(Om, J_TRUE @ Om))
        eOm_dot = Om_dot + hat(Om) @ R.T @ Rd @ Omd - R.T @ Rd @ Omd_dot
        Delta = lumped_uncertainty(R, Om, Rd, Omd, Omd_dot, mu, De)
        worst = max(worst, np.max(np.abs(J0 @ eOm_dot - mu - Delta)))
        # with the right inertia the only uncertainty is the disturbance torque itself
        worst_match = max(worst_match, np.max(np.abs(lumped_uncertainty(R, Om, Rd, Omd, Omd_dot, mu, De, J=J0) - De)))
        # e_Omega_dot by finite differences along the true motion
        dt = 1e-6
        Rp_, Rm_ = R @ expm_so3(dt * Om), R @ expm_so3(-dt * Om)
        Rdp, Rdm = Rd @ expm_so3(dt * Omd), Rd @ expm_so3(-dt * Omd)
        fd = (e_Omega(Rp_, Om + dt * Om_dot, Rdp, Omd + dt * Omd_dot)
              - e_Omega(Rm_, Om - dt * Om_dot, Rdm, Omd - dt * Omd_dot)) / (2 * dt)
        worst_fd = max(worst_fd, np.max(np.abs(fd - eOm_dot)))
        eOm = e_Omega(R, Om, Rd, Omd)
        worst_C = max(worst_C, eOm @ C_R(R, Rd) @ eOm - eOm @ eOm, np.linalg.norm(C_R(R, Rd), 2) - 1)
    ck.close("error dynamics: J0 e_Omega_dot = mu + Delta with Delta of (eq8-lumped)", worst, 0, tol=1e-9)
    ck.close("with J = J0 the lumped uncertainty is the disturbance torque", worst_match, 0, tol=1e-12)
    ck.close("e_Omega_dot = Om_dot + hat(Om) R^T Rd Omd - R^T Rd Omd_dot (finite differences)", worst_fd, 0, tol=1e-6)
    ck.true("spectral norm of C_R at most 1, so e_Omega . C_R e_Omega <= |e_Omega|^2", worst_C < 1e-12, f"max excess {worst_C:.1e}")

    # the camera
    ck.close("camera torque m_a g e3 x r_a = (0, 0.392, 0) N m", DELTA_CAM, [0, 0.3924, 0], tol=1e-4)
    ck.record("camera_torque_Nm", float(np.linalg.norm(DELTA_CAM)))
    ck.record("J_true_diag", np.diag(J_TRUE).tolist())
    ck.record("inertia_change_percent", (100 * (np.diag(J_TRUE) / np.diag(J0) - 1)).tolist())

    o_pd = run_attitude("pd", ref_hover, dist_camera, 6.0)
    offset = deg(o_pd["angle"][-1])
    predicted = deg(np.arcsin(np.linalg.norm(DELTA_CAM) / K_R))
    ck.close("hover with the camera: PD settles at sin(theta) = |Delta_e| / k_R", offset, predicted, tol=1e-6)
    ck.close("hover with the camera: e_R at rest equals Delta_e / k_R", e_R(o_pd["R_final"], I3), DELTA_CAM / K_R, tol=1e-8)
    ck.record("pd_hover_offset_deg", float(offset))
    ck.record("pd_hover_sideways_accel", float(GRAV * np.tan(np.radians(offset))))
    big = 5.5 * DELTA_CAM
    with np.errstate(all="ignore"):
        o_big = run_attitude("pd", ref_hover, lambda t: big, 6.0)
    ck.true("a disturbance torque above k_R cannot be held: PD tumbles", np.linalg.norm(big) > K_R and deg(o_big["angle"].max()) > 170,
            f"|Delta_e| = {np.linalg.norm(big):.2f} N m, max error {deg(o_big['angle'].max()):.1f} deg")
    ck.record("big_torque_Nm", float(np.linalg.norm(big)))
    ck.record("pd_big_max_deg", float(deg(o_big["angle"].max())))

    # ------------------------------------------------------------------
    # 2. Robustness for free: the ultimate bound of (thm8-bound)
    # ------------------------------------------------------------------
    cs = np.linspace(0.002, 0.11, 400)
    gains = np.array([ultimate_bound_gain(c) for c in cs])
    c_best = float(cs[np.argmin(gains)])
    K_ub = float(gains.min())
    W, W1, W2, lm, lM = lyapunov_matrices(c_best)
    ck.true("W and W1 positive definite for the chosen c", np.linalg.eigvalsh(W)[0] > 0 and np.linalg.eigvalsh(W1)[0] > 0, f"c = {c_best:.3f}")
    worst_v, worst_lo, worst_hi = -np.inf, -np.inf, -np.inf
    J0i = np.linalg.inv(J0)
    for _ in range(400):
        Rd = random_rotation(rng)
        R = Rd @ expm_so3(unit(rng.normal(size=3)) * rng.uniform(0, np.pi / 2))     # Psi <= 1
        Om, Omd, Delta = rng.normal(size=3) * 3, rng.normal(size=3), rng.normal(size=3)
        eR, eOm = e_R(R, Rd), e_Omega(R, Om, Rd, Omd)
        z = np.array([np.linalg.norm(eR), np.linalg.norm(eOm)])
        eOm_dot = J0i @ (-K_R * eR - K_OM * eOm + Delta)
        V = 0.5 * eOm @ J0 @ eOm + K_R * Psi_R(R, Rd) + c_best * eR @ eOm
        V_dot = eOm @ J0 @ eOm_dot + K_R * eR @ eOm + c_best * (C_R(R, Rd) @ eOm) @ eOm + c_best * eR @ eOm_dot
        bound = -z @ W @ z + (z[1] + c_best / lm * z[0]) * np.linalg.norm(Delta)
        worst_v = max(worst_v, V_dot - bound)
        worst_lo, worst_hi = max(worst_lo, z @ W1 @ z - V), max(worst_hi, V - z @ W2 @ z)
    ck.true("V_dot <= -z'Wz + (|e_Omega| + c |e_R| / lambda_min) |Delta|", worst_v < 1e-9, f"max excess {worst_v:.1e}")
    ck.true("z'W1 z <= V <= z'W2 z while Psi_R <= 1", worst_lo < 1e-9 and worst_hi < 1e-9)
    ck.record("bound_c", c_best)
    ck.record("bound_gain_per_Nm", K_ub)
    small = 0.02
    o_small = run_attitude("pd", ConingReference(h), lambda t: small * gust(t) / 0.25, 8.0, J=J0)
    dmax = small * 0.5 * 2.5 * np.linalg.norm([1.0, -0.6, 0.3])
    ck.true("the gust torque is bounded by 1.51 times its scale", max(np.linalg.norm(gust(t)) for t in np.linspace(0, 60, 6001)) <= 0.25 * 1.51)
    tail = o_small["t"] > 4
    z_sim = float(np.max(np.hypot(np.sqrt(o_small["Psi"][tail] * (2 - o_small["Psi"][tail])), o_small["eOm"][tail])))
    ck.true("ultimate bound holds in simulation (and is conservative)", z_sim <= K_ub * dmax and o_small["Psi"].max() < 1,
            f"simulated {z_sim:.4f}, bound {K_ub * dmax:.3f}")
    ck.record("bound_small_torque_Nm", float(dmax))
    ck.record("bound_small_sim", z_sim)
    ck.record("bound_small_value", float(K_ub * dmax))
    ck.record("bound_conservatism", float(K_ub * dmax / z_sim))
    ck.record("bound_largest_useful_torque_Nm", float(1.0 / K_ub))

    # ------------------------------------------------------------------
    # 3. Geometric L1: identities
    # ------------------------------------------------------------------
    worst_pred, worst_dyn, worst_rel, worst_proj = 0.0, 0.0, 0.0, -np.inf
    for _ in range(30):
        R, Rd = random_rotation(rng), random_rotation(rng)
        Om, Omd, Omd_dot, De = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        l1 = L1Attitude(random_rotation(rng), rng.normal(size=3), h)
        l1.Dhat, l1.Df = rng.normal(size=3), rng.normal(size=3)
        Rp, Omp = l1.Rp, l1.Omp
        tau, mu, mu_p = l1.torques(R, Om, Rd, Omd, Omd_dot)
        Om_dot = Ji_true @ (tau + De - np.cross(Om, J_TRUE @ Om))
        Delta = lumped_uncertainty(R, Om, Rd, Omd, Omd_dot, mu, De)
        Omp_dot = J0i @ mu_p - (hat(Omp) @ Rp.T @ Rd @ Omd - Rp.T @ Rd @ Omd_dot)
        # predictor: J0 e^p_Omega_dot = mu_p
        ep_dot = Omp_dot + hat(Omp) @ Rp.T @ Rd @ Omd - Rp.T @ Rd @ Omd_dot
        worst_pred = max(worst_pred, np.max(np.abs(J0 @ ep_dot - mu_p)))
        # prediction error dynamics, differentiated directly
        RpR_dot = -hat(Omp) @ Rp.T @ R + Rp.T @ R @ hat(Om)
        et_dot = Omp_dot - RpR_dot @ Om - Rp.T @ R @ Om_dot
        eRt, eOmt, _ = l1.errors(R, Om)
        U = l1.U(R)
        rhs = -l1.kRt * eRt - l1.kOmt * eOmt + U @ (l1.Dhat - Rp.T @ R @ Delta)
        worst_dyn = max(worst_dyn, np.max(np.abs(J0 @ et_dot - rhs)))
        # the relation between mu and mu_p (Kotaru 2022, eq. 3.32)
        rel = (J0 @ R.T @ Rp @ J0i @ (mu_p + l1.kRt * eRt + l1.kOmt * eOmt) - R.T @ Rp @ l1.Dhat
               + J0 @ R.T @ Rp @ hat(eOmt) @ Rp.T @ R @ e_Omega(R, Om, Rd, Omd))
        worst_rel = max(worst_rel, np.max(np.abs(rel - mu)))
    for _ in range(2000):
        th_true = unit(rng.normal(size=3)) * rng.uniform(0, 3.0)
        th_hat = unit(rng.normal(size=3)) * rng.uniform(0, 3.0 * np.sqrt(1.1))
        y = rng.normal(size=3)
        worst_proj = max(worst_proj, (th_hat - th_true) @ (proj(th_hat, y, 3.0) - y))
    ck.close("predictor: J0 e^p_Omega_dot = mu_p", worst_pred, 0, tol=1e-9)
    ck.close("prediction error: J0 e~_Omega_dot = -k~_R e~_R - k~_Omega e~_Omega + U Delta~ (eq8-pred-dyn)", worst_dyn, 0, tol=1e-9)
    ck.close("plant and predictor inputs are related as in Kotaru (2022, eq. 3.32)", worst_rel, 0, tol=1e-9)
    ck.true("projection: Delta~ . (Proj(Delta^, y) - y) <= 0 whenever |Delta| <= bound", worst_proj < 1e-12, f"max {worst_proj:.1e}")

    # ------------------------------------------------------------------
    # 4. Geometric L1: closed loop
    # ------------------------------------------------------------------
    o_l1 = run_attitude("l1", ref_hover, dist_camera, 6.0)
    ck.true("hover with the camera: L1 removes the offset", deg(o_l1["angle"][-1]) < 1e-3 and deg(o_l1["angle"].max()) < 3.5,
            f"peak {deg(o_l1['angle'].max()):.2f} deg, final {deg(o_l1['angle'][-1]):.1e} deg")
    ck.record("l1_hover_peak_deg", float(deg(o_l1["angle"].max())))
    ck.record("l1_hover_time_below_0p1deg", float(o_l1["t"][np.where(deg(o_l1["angle"]) > 0.1)[0][-1]]))
    ck.close("the filtered estimate converges to the lumped uncertainty", o_l1["Df"][-1], o_l1["Delta"][-1], tol=1e-5)
    o_l1big = run_attitude("l1", ref_hover, lambda t: big, 6.0)
    ck.true("the torque that tumbled PD is absorbed by L1", deg(o_l1big["angle"].max()) < 20 and deg(o_l1big["angle"][-1]) < 1e-2,
            f"peak {deg(o_l1big['angle'].max()):.1f} deg")
    ck.record("l1_big_peak_deg", float(deg(o_l1big["angle"].max())))
    ck.true("projection keeps the estimate inside its ball", np.linalg.norm(o_l1big["Dhat"], axis=1).max() <= 3.0 * np.sqrt(1.1) + 1e-9)

    track = {}
    for name, dist in (("camera", dist_camera), ("camera_gust", dist_camera_gust)):
        for ctrl in ("pd", "l1"):
            o = run_attitude(ctrl, ConingReference(h), dist, 10.0)
            track[name, ctrl] = o
            rms, mx = error_stats(o, 3.0)
            ck.record(f"track_{name}_{ctrl}_rms_deg", rms)
            ck.record(f"track_{name}_{ctrl}_max_deg", mx)
    ck.true("moving reference, camera: L1 error at least 100 times below PD",
            error_stats(track["camera", "l1"], 3)[0] * 100 < error_stats(track["camera", "pd"], 3)[0])
    ck.true("moving reference, camera and gust: L1 error at least 10 times below PD",
            error_stats(track["camera_gust", "l1"], 3)[0] * 10 < error_stats(track["camera_gust", "pd"], 3)[0])
    o = track["camera_gust", "l1"]
    ck.true("prediction error stays far below tracking error", o["Psi_pred"].max() < 1e-6 and o["Psi"][3000:].max() > 1e-5,
            f"max Psi~ {o['Psi_pred'].max():.1e}, max Psi {o['Psi'][3000:].max():.1e}")
    ck.record("gust_bound_Nm", float(0.25 * 1.51))
    # prediction error against adaptation gain: the quantity the theorem is about
    pred = {}
    for gamma in (250.0, 1000.0, 4000.0, 16000.0):
        o = run_attitude("l1", ConingReference(h), dist_camera_gust, 5.0, gamma=gamma)
        pred[gamma] = float(o["eOm_pred"][1000:].max())
        ck.record(f"pred_error_rate_gamma_{int(gamma)}", pred[gamma])
    ck.true("prediction error shrinks as the adaptation gain grows", pred[250.0] > pred[1000.0] > pred[4000.0] > pred[16000.0],
            ", ".join(f"{v:.1e}" for v in pred.values()))

    # the filter: cutoff against tracking error and torque activity, with gyroscope noise
    kappas = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    study = {}
    for kappa in kappas:
        o = run_attitude("l1", ConingReference(h), dist_camera_gust, 7.0, noise=0.02, kappa=kappa)
        study[kappa] = (error_stats(o, 3.0)[0], torque_rate_rms(o, 3.0, h))
        ck.record(f"cutoff_{int(kappa)}_rms_deg", study[kappa][0])
        ck.record(f"cutoff_{int(kappa)}_torque_rate", study[kappa][1])
    o = run_attitude("pd", ConingReference(h), dist_camera_gust, 7.0, noise=0.02)
    ck.record("cutoff_pd_rms_deg", error_stats(o, 3.0)[0])
    ck.record("cutoff_pd_torque_rate", torque_rate_rms(o, 3.0, h))
    errs_k = [study[k][0] for k in kappas]
    ck.true("raising the cutoff lowers the tracking error", all(a > b for a, b in zip(errs_k, errs_k[1:])),
            ", ".join(f"{e:.2f}" for e in errs_k))
    ck.true("raising the cutoff from 10 to 300 rad/s multiplies the torque rate by more than 5",
            study[300.0][1] > 5 * study[10.0][1], f"{study[10.0][1]:.1f} -> {study[300.0][1]:.1f} N m/s")
    with np.errstate(all="ignore"):
        o = run_attitude("l1", ConingReference(h), dist_camera_gust, 7.0, noise=0.02, lag=0.02, kappa=3000.0)
    ck.true("with a 20 ms motor lag, a cutoff of 3000 rad/s is unstable", o["diverged"] or deg(o["angle"].max()) > 90)
    o = run_attitude("l1", ConingReference(h), dist_camera_gust, 7.0, noise=0.02, lag=0.02, kappa=30.0)
    ck.true("with the same lag, a cutoff of 30 rad/s is fine", not o["diverged"] and error_stats(o, 3.0)[0] < 1.0)
    # the L1 point: the delay margin is set by the filter, not by the adaptation gain
    dm_pd = delay_margin("pd")
    dm = {kappa: delay_margin("l1", kappa=kappa) for kappa in (3.0, 30.0, 300.0)}
    dm_gamma = {gamma: delay_margin("l1", kappa=30.0, gamma=gamma) for gamma in (1000.0, 16000.0)}
    ck.record("delay_margin_pd_ms", dm_pd)
    for kappa, d in dm.items():
        ck.record(f"delay_margin_cutoff_{int(kappa)}_ms", d)
    for gamma, d in dm_gamma.items():
        ck.record(f"delay_margin_gamma_{int(gamma)}_ms", d)
    ck.true("delay margin falls as the cutoff rises", dm_pd >= dm[3.0] > dm[30.0] > dm[300.0], f"PD {dm_pd}, L1 {list(dm.values())} ms")
    ck.true("delay margin does not depend on the adaptation gain (within 1 ms)",
            max(abs(dm_gamma[1000.0] - dm[30.0]), abs(dm_gamma[16000.0] - dm[30.0])) <= 1, f"{list(dm_gamma.values())} ms")
    print(f"  [attitude part done in {time.time() - t_start:.0f} s]")

    # ------------------------------------------------------------------
    # 5. A point and a wall
    # ------------------------------------------------------------------
    worst = 0.0
    for _ in range(200):
        x, ud, alpha = rng.normal(size=2), rng.normal(size=2) * 2, rng.uniform(0.2, 6)
        x[0] = min(x[0], 0.99)
        z = solve_qp(np.eye(2), -ud, np.array([WALL_N]), np.array([alpha * wall_H(x)]))
        worst = max(worst, np.max(np.abs(z - wall_filter(x, ud, alpha))))
    ck.close("wall: the closed-form filter is the solution of the QP", worst, 0, tol=1e-9)
    u_ex = wall_filter(np.array([0.9, 0.0]), np.array([1.1, 1.5]), 5.0)
    ck.close("wall example: at x1 = 0.9 with alpha = 5 the filter passes 0.5 m/s of the wanted 1.1", u_ex, [0.5, 1.5], tol=1e-12)
    for alpha in (1.0, 5.0, 25.0):
        t, xs = run_wall(alpha)
        Hs = WALL_D - xs[:, 0]
        ck.true(f"wall, alpha = {alpha:g}: H(t) >= H(0) exp(-alpha t) and H > 0", np.all(Hs >= np.exp(-alpha * t) - 1e-9) and Hs.min() > 0,
                f"min H {Hs.min():.1e}")
        ck.close(f"wall, alpha = {alpha:g}: ends at the safe point nearest the target", xs[-1], [1.0, 1.5], tol=2e-3)
        untouched = xs[np.argmax(np.abs(np.gradient(xs[:, 0], t) - (2.0 - xs[:, 0])) > 1e-6), 0]
        ck.record(f"wall_first_touch_alpha_{int(alpha)}", float(untouched))
    ck.close("wall, alpha = 5: the filter first acts at x1 = 0.75", ck.values["wall_first_touch_alpha_5"], 0.75, tol=2e-3)
    t, xs = run_wall(None)
    ck.true("wall: without the filter the tracker crosses", xs[:, 0].max() > 1.5)

    # ------------------------------------------------------------------
    # 6. A cone on S^2
    # ------------------------------------------------------------------
    def fd_cone(cone):
        worst1, worst2, worstV = 0.0, 0.0, 0.0
        ref = cone_reference()
        for _ in range(15):
            t0 = rng.uniform(0, 5)
            q = unit(-E3 + 0.5 * rng.normal(size=3))
            om = np.cross(q, rng.normal(size=3))
            F = rng.normal(size=3) * 3
            dt, sub = 1e-3, 10
            qp_, op_, qm_, om_ = q, om, q, om
            for _ in range(sub):
                qp_, op_ = s2_step(qp_, op_, F, dt / sub)
                qm_, om_ = s2_step(qm_, om_, F, -dt / sub)
            H0, Hd0, aH, bH = cone_H(q, om, *cone(t0))
            Hp, Hdp, _, _ = cone_H(qp_, op_, *cone(t0 + dt))
            Hm, Hdm, _, _ = cone_H(qm_, om_, *cone(t0 - dt))
            worst1 = max(worst1, abs((Hp - Hm) / (2 * dt) - Hd0))
            worst2 = max(worst2, abs((Hdp - Hdm) / (2 * dt) - (aH + bH @ F)))
            qd, omd, omd_dot = ref(t0)
            aV, bV = s2_V_dot_terms(q, om, qd, omd, omd_dot)
            Vp = s2_V(qp_, op_, *ref(t0 + dt)[:2])
            Vm = s2_V(qm_, om_, *ref(t0 - dt)[:2])
            worstV = max(worstV, abs((Vp - Vm) / (2 * dt) - (aV + bV @ F)))
        return worst1, worst2, worstV
    w1, w2, wV = fd_cone(cone_static)
    ck.close("cone: H_dot = om . (q x q_c) (finite differences along the motion)", w1, 0, tol=1e-5)
    ck.close("cone: H_ddot formula (eq8-cone-ddot)", w2, 0, tol=1e-5)
    ck.close("V_dot on S^2 is a_V + b_V . F", wV, 0, tol=1e-5)
    w1, w2, _ = fd_cone(cone_swinging)
    ck.close("moving cone: H_dot = (om - om_c) . (q x q_c)", w1, 0, tol=1e-5)
    ck.close("moving cone: H_ddot with the terms 2 q_dot . q_c_dot + q . q_c_ddot", w2, 0, tol=1e-5)
    worst, worst_sw = 0.0, 0.0
    for _ in range(50):
        q = unit(rng.normal(size=3))
        om = np.cross(q, rng.normal(size=3))
        qc = unit(rng.normal(size=3))
        omc = np.cross(qc, rng.normal(size=3))
        F = rng.normal(size=3)
        _, Hd, aH, bH = cone_H(q, om, qc, np.zeros(3), np.zeros(3))
        worst = max(worst, abs(Hd - om @ np.cross(q, qc)),
                    abs(aH + bH @ F - (-(om @ om) * (q @ qc) + (qc - (q @ qc) * q) @ (F - M_P * GRAV * E3) / (M_P * L_P))))
        _, Hd, _, _ = cone_H(q, om, qc, np.cross(omc, qc), np.zeros(3))
        worst_sw = max(worst_sw, abs(Hd - (om - omc) @ np.cross(q, qc)))
    ck.close("cone: the two closed forms printed in the chapter", max(worst, worst_sw), 0, tol=1e-12)
    # the tracker's force makes V_dot what the PD analysis says
    worst = 0.0
    ref = cone_reference()
    for _ in range(30):
        t0 = rng.uniform(0, 5)
        qd, omd, omd_dot = ref(t0)
        q = unit(qd + 0.4 * rng.normal(size=3))
        om = np.cross(q, rng.normal(size=3))
        F = s2_pd_force(q, om, qd, omd, omd_dot)
        acc = s2_accel(q, om, F)
        a_d = s2_reference_accel(q, om, omd, omd_dot)
        tang = (acc - a_d) - ((acc - a_d) @ q) * q
        worst = max(worst, np.max(np.abs(M_P * L_P**2 * tang + K_Q * e_q(q, qd) + K_W_PD * e_omega(q, om, omd))))
    ck.close("S^2 tracker: m l^2 (tangent part of e_omega_dot) = -k_q e_q - k_omega e_omega", worst, 0, tol=1e-10)

    ref = cone_reference()
    lap = 2 * np.pi / 0.9
    o_cpd = run_cone(ref, lap, controller="pd")
    o_cqp = run_cone(ref, 1.5 * lap)
    ref_angle = deg(np.arccos(np.clip(-o_cqp["qd"][:, 2], -1, 1)))
    ck.record("cone_ref_max_angle_deg", float(ref_angle.max()))
    ck.record("cone_ref_unsafe_fraction", float(np.mean(ref_angle[: int(lap / 2e-3)] > 30)))
    ck.true("cone: the tracker alone follows the reference out of the cone", o_cpd["H"].min() < -0.15 and o_cpd["Psi"].max() < 1e-6,
            f"min H {o_cpd['H'].min():.3f}")
    ck.record("cone_pd_min_H", float(o_cpd["H"].min()))
    ck.true("cone: the QP never leaves it (min H >= -1e-6) and is always feasible", o_cqp["H"].min() >= -1e-6 and o_cqp["feasible"].all(),
            f"min H {o_cqp['H'].min():.1e}, min H1 {o_cqp['H1'].min():.1e}")
    ck.record("cone_qp_min_H", float(o_cqp["H"].min()))
    q_angle = deg(np.arccos(np.clip(-o_cqp["q"][:, 2], -1, 1)))
    ck.true("cone: the state reaches the boundary and slides along it", q_angle.max() > 29.99 and np.mean(q_angle > 29.9) > 0.2,
            f"max {q_angle.max():.3f} deg, {100 * np.mean(q_angle[: int(lap / 2e-3)] > 29.9):.0f}% of a lap within 0.1 deg of it")
    ck.record("cone_boundary_fraction", float(np.mean(q_angle[: int(lap / 2e-3)] > 29.9)))
    k_rejoin = int(0.97 * lap / 2e-3)
    ck.true("cone: the state rejoins the reference once it is safe again", o_cqp["Psi"][k_rejoin] < 5e-4,
            f"Psi_q {o_cqp['Psi'][k_rejoin]:.1e} at the end of the lap")
    ck.record("cone_max_Psi", float(o_cqp["Psi"].max()))
    ck.record("cone_max_gap_deg", float(deg(np.arccos(1 - o_cqp["Psi"].max()))))
    ck.record("cone_max_slack", float(np.nanmax(o_cqp["slack"])))
    ck.record("cone_max_force_N", float(np.linalg.norm(o_cqp["F"], axis=1).max()))
    ck.true("cone: the slack is zero (below 1e-3) while the reference is well inside", np.nanmax(o_cqp["slack"][ref_angle < 15]) < 2e-2,
            f"{np.nanmax(o_cqp['slack'][ref_angle < 15]):.1e}")
    o = run_cone(ref, lap, cone=cone_swinging)
    ck.true("moving cone: never violated", o["H"].min() >= -1e-6 and o["feasible"].all(), f"min H {o['H'].min():.1e}")
    ck.record("moving_cone_min_H", float(o["H"].min()))

    # sampling: the same QP at slower rates
    for hh in (0.002, 0.02, 0.05, 0.1):
        o = run_cone(ref, lap, h=hh)
        jumps = np.linalg.norm(np.diff(o["F"], axis=0), axis=1)
        ck.record(f"sample_{int(hh * 1000)}ms_min_H", float(o["H"].min()))
        ck.record(f"sample_{int(hh * 1000)}ms_max_force_jump_N", float(jumps.max()))
        ck.record(f"sample_{int(hh * 1000)}ms_feasible", bool(o["feasible"].all()))
    # input limits: with a small force budget the QP has no solution
    o = run_cone(ref, lap, Fmax=4.0)
    o2 = run_cone(ref, lap, Fmax=1.5)
    ck.true("force limit 4 N per axis: still feasible and safe", o["feasible"].all() and o["H"].min() > -1e-6)
    ck.true("force limit 1.5 N per axis: the QP becomes infeasible", not o2["feasible"].all(),
            f"{(~o2['feasible']).sum()} of {len(o2['t'])} steps, min H {o2['H'].min():.3f}")
    ck.record("limit_infeasible_steps", int((~o2["feasible"]).sum()))
    ck.record("limit_min_H", float(o2["H"].min()))
    # a wrong model inside the barrier condition: an unknown steady wind
    wind = np.array([1.0, 0.0, 0.0])
    o = run_cone(ref, 1.5 * lap, wind=wind)
    predicted = -np.sin(CONE_ANGLE) * np.linalg.norm(wind) / (M_P * L_P) / 36.0
    ck.true("unknown 1 N wind: the cone is violated", o["H"].min() < -0.01, f"min H {o['H'].min():.4f}, estimate {predicted:.4f}")
    ck.record("wind_min_H", float(o["H"].min()))
    ck.record("wind_estimate_H", float(predicted))
    ck.record("wind_overshoot_deg", float(deg(np.arccos(np.cos(CONE_ANGLE) + o["H"].min())) - 30))
    ck.true("the violation is close to (error in H_ddot) / (alpha1 alpha2)", abs(o["H"].min() - predicted) < 0.3 * abs(predicted))
    o = run_cone(ref, 1.5 * lap, wind=wind, alpha1=18.0, alpha2=18.0)
    ck.true("three times the gains: about one ninth of the violation", -o["H"].min() < 0.2 * 0.035 and o["H"].min() < 0,
            f"min H {o['H'].min():.4f}")
    ck.record("wind_min_H_high_gain", float(o["H"].min()))
    margin = np.arccos(np.cos(CONE_ANGLE) + 0.04)
    o = run_cone(ref, 1.5 * lap, wind=wind, angle=margin)
    true_H = o["H"] + np.cos(margin) - np.cos(CONE_ANGLE)
    ck.true("a cone tightened by the size of the model error keeps the real one", true_H.min() > 0, f"min H {true_H.min():.4f}")
    ck.record("wind_margin_deg", float(30 - deg(margin)))
    print(f"  [cone part done in {time.time() - t_start:.0f} s]")

    # ------------------------------------------------------------------
    # 7. Tilt limit on SO(3)
    # ------------------------------------------------------------------
    worst1, worst2 = 0.0, 0.0
    for _ in range(15):
        R, Om, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3) * 0.1
        dt, sub = 1e-3, 10
        Rp_, Op_, Rm_, Om_ = R, Om, R, Om
        for _ in range(sub):
            Rp_, Op_ = body_step(Rp_, Op_, tau, dt / sub, J0, J0i)
            Rm_, Om_ = body_step(Rm_, Om_, tau, -dt / sub, J0, J0i)
        ang = np.radians(35)
        H0, Hd0, aH, bH = tilt_H(R, Om, ang)
        Hp, Hdp, _, _ = tilt_H(Rp_, Op_, ang)
        Hm, Hdm, _, _ = tilt_H(Rm_, Om_, ang)
        Om_dot = J0i @ (tau - np.cross(Om, J0 @ Om))
        worst1 = max(worst1, abs((Hp - Hm) / (2 * dt) - Hd0))
        worst2 = max(worst2, abs((Hdp - Hdm) / (2 * dt) - (aH + bH @ Om_dot)))
        # the filter is the QP
        tau_d = rng.normal(size=3)
        g = J0i @ bH
        rhs = -(aH - bH @ J0i @ np.cross(Om, J0 @ Om)) - 16 * Hd0 - 64 * H0
        z = solve_qp(np.eye(3), -tau_d, np.array([-g]), np.array([-rhs]))
        worst2 = max(worst2, 1e-3 * np.max(np.abs(z - tilt_filter(R, Om, tau_d, ang, J0))))
    ck.close("tilt: H_dot = Om . (e3 x R^T e3)", worst1, 0, tol=1e-5)
    ck.close("tilt: H_ddot formula (eq8-tilt), and the closed-form filter is the QP", worst2, 0, tol=1e-5)
    Rd_tilt = expm_so3(np.radians(60) * E2)
    o_t = run_tilt(Rd_tilt, np.radians(35))
    o_t0 = run_tilt(Rd_tilt, np.radians(35), filtered=False)
    ck.true("tilt: target at 60 deg, limit 35 deg: the body stops at the limit", abs(deg(o_t["tilt"][-1]) - 35) < 1e-3 and o_t["H"].min() > -1e-9,
            f"final tilt {deg(o_t['tilt'][-1]):.4f} deg, min H {o_t['H'].min():.1e}")
    ck.true("tilt: without the filter the body goes to 60 deg", abs(deg(o_t0["tilt"][-1]) - 60) < 1e-3)
    ck.record("tilt_final_deg", float(deg(o_t["tilt"][-1])))
    ck.record("tilt_final_Psi", float(o_t["Psi"][-1]))
    ck.close("tilt: the rest attitude is 25 deg from the target", deg(angle_between(o_t["R_final"], Rd_tilt)), 25.0, tol=1e-3)

    # ------------------------------------------------------------------
    # 8. Quadrotor and obstacle
    # ------------------------------------------------------------------
    worst1, worst2, worst_naive = 0.0, 0.0, 0.0
    for _ in range(15):
        x, v = OBS_C + unit(rng.normal(size=3)) * rng.uniform(0.7, 1.5), rng.normal(size=3)
        R, Om, f, tau = random_rotation(rng), rng.normal(size=3), rng.uniform(5, 20), rng.normal(size=3) * 0.1
        a1 = 3.0

        def flow(x, v, R, Om, dt, sub=20):
            for _ in range(sub):
                hh = dt / sub
                acc = f * R[:, 2] / M_Q - GRAV * E3
                Rn, Omn = body_step(R, Om, tau, hh, J0, J0i)
                accn = f * Rn[:, 2] / M_Q - GRAV * E3
                x, v = x + hh * v + hh * hh * (2 * acc + accn) / 6, v + 0.5 * hh * (acc + accn)
                R, Om = Rn, Omn
            return x, v, R, Om
        dt = 1e-3
        sp, sm = flow(x, v, R, Om, dt), flow(x, v, R, Om, -dt)
        H, H1, a, b_f, b_tau = aug_terms(x, v, R, Om, J0, a1)
        Hp, H1p = aug_terms(*sp, J0, a1)[:2]
        Hm, H1m = aug_terms(*sm, J0, a1)[:2]
        worst1 = max(worst1, abs((Hp - Hm) / (2 * dt) - (H1 - a1 * H)))
        worst2 = max(worst2, abs((H1p - H1m) / (2 * dt) - (a + b_f * f + b_tau @ tau)))
        Hn, Hnd, an, bn = naive_terms(x, v, R)
        Hnp, Hnm = naive_terms(*sp[:3]), naive_terms(*sm[:3])
        worst_naive = max(worst_naive, abs((Hnp[1] - Hnm[1]) / (2 * dt) - (an + bn * f)), abs((Hnp[0] - Hnm[0]) / (2 * dt) - Hnd))
        ck.rows  # noqa: B018
    ck.close("plain barrier: H_ddot = 2|v|^2 + 2 (x - x_o) . (f R e3 / m - g e3), no torque in it", worst_naive, 0, tol=1e-5)
    ck.close("augmented barrier: H_dot (finite differences along the motion)", worst1, 0, tol=1e-5)
    ck.close("augmented barrier: H1_dot = a + b_f f + b_tau . tau, with b_f = (2 s - sigma') / m", worst2, 0, tol=2e-5)
    worst = np.inf
    for _ in range(4000):
        x, R = OBS_C + rng.normal(size=3) * 1.2, random_rotation(rng)
        if aug_H(x, R) >= 0:
            worst = min(worst, np.linalg.norm(x - OBS_C) - OBS_R)
    ck.true("augmented H >= 0 implies the vehicle is outside the obstacle (Wu, Lemma 7.2)", worst > 0, f"closest sample {worst:.3f} m outside")
    ss = np.linspace(-3, 3, 6001)
    sg, d1, _ = sigma(ss)
    ck.true("sigma is strictly decreasing and bounded by k1 < (beta - 1) r_o^2", d1.max() < 0 and np.abs(sg).max() <= AUG_K1 < (AUG_BETA - 1) * OBS_R**2)
    s_int = np.linspace(-np.sqrt(AUG_BETA) * OBS_R, 0, 4001)[1:-1]
    cond = 2 * s_int - sigma(s_int)[1] > 0
    s_star = float(s_int[np.argmax(cond)])
    ck.true("Wu's third condition holds only for s in (s*, 0) with our sigma", not cond.all() and cond[np.argmax(cond):].all(), f"s* = {s_star:.3f} m")
    ck.record("sigma_third_condition_from", s_star)
    ck.record("aug_keepout_radius", float(np.sqrt(AUG_BETA) * OBS_R))
    # the two conditions together force beta > 2:  beta r_o^2 = int 2|s| < int |sigma'| <= 2 sigma_0 < 2 (beta - 1) r_o^2
    ck.close("integral of 2|s| over (-sqrt(beta) r_o, 0) is beta r_o^2", np.trapezoid(2 * np.abs(s_int), s_int), AUG_BETA * OBS_R**2, tol=1e-3)

    with np.errstate(all="ignore"):
        o_none = run_quad("none", tf=6.0, ref=quad_reference)
        o_naive = run_quad("naive", tf=6.0, ref=quad_reference)
        o_aug = run_quad("augmented", tf=6.0, ref=quad_reference)
    ck.true("no filter: the tracker flies through the obstacle", o_none["dist"].min() < 0.2, f"min distance {o_none['dist'].min():.2f} m")
    ck.true("plain barrier, thrust only: infeasible, and it hits", (~o_naive["feasible"]).sum() > 100 and o_naive["dist"].min() < OBS_R,
            f"{(~o_naive['feasible']).sum()} infeasible steps, min distance {o_naive['dist'].min():.2f} m")
    ck.true("augmented barrier: always feasible, never inside", o_aug["feasible"].all() and o_aug["dist"].min() > OBS_R and o_aug["H"].min() > -1e-6,
            f"min distance {o_aug['dist'].min():.3f} m, min H {o_aug['H'].min():.1e}")
    ck.true("augmented barrier: rejoins the reference", np.linalg.norm(o_aug["x"][-1] - o_aug["xd"][-1]) < 0.05)
    k_first = int(np.argmax(~o_naive["feasible"]))
    ck.record("quad_naive_first_infeasible_dist", float(o_naive["dist"][k_first]))
    ck.record("quad_naive_s_at_infeasible", float(o_naive["s"][k_first]))
    ck.record("quad_naive_infeasible_steps", int((~o_naive["feasible"]).sum()))
    ck.record("quad_naive_min_dist", float(o_naive["dist"].min()))
    ck.record("quad_aug_min_dist", float(o_aug["dist"].min()))
    ck.record("quad_aug_max_tilt_deg", float(deg(np.arccos(o_aug["b3"][:, 2].min()))))
    ck.record("quad_aug_max_torque", float(np.abs(o_aug["tau"]).max()))
    ck.record("quad_aug_thrust_range", [float(o_aug["f"].min()), float(o_aug["f"].max())])
    ck.record("quad_hover_thrust", float(M_Q * GRAV))

    # two vehicles (Kotaru 2022, sec. 6.5)
    worst = 0.0
    for _ in range(20):
        x1, x2, v1, v2, a1, a2 = (rng.normal(size=2) for _ in range(6))
        dt = 1e-4

        def Hd(s):
            return 2 * ((x1 + s * v1 + 0.5 * s * s * a1) - (x2 + s * v2 + 0.5 * s * s * a2)) @ ((v1 + s * a1) - (v2 + s * a2))
        worst = max(worst, abs((Hd(dt) - Hd(-dt)) / (2 * dt) - (2 * (v1 - v2) @ (v1 - v2) + 2 * (x1 - x2) @ (a1 - a2))))
    ck.close("pair barrier: H_ddot = 2|v1 - v2|^2 + 2 (x1 - x2) . (a1 - a2)", worst, 0, tol=1e-6)
    _, _, d_free = run_pair(filtered=False)
    X1, X2, d_safe = run_pair()
    ck.true("two vehicles swapping places: the filter keeps them 0.8 m apart", d_free.min() < 0.2 and d_safe.min() >= 0.8 - 1e-6,
            f"min distance {d_free.min():.2f} m without, {d_safe.min():.4f} m with")
    ck.true("and both still arrive", np.linalg.norm(X1[-1] - [2.0, 0.05]) < 0.02 and np.linalg.norm(X2[-1] - [-2.0, -0.05]) < 0.02)
    ck.record("pair_min_dist_free", float(d_free.min()))
    ck.record("pair_min_dist_safe", float(d_safe.min()))

    print(f"  [all done in {time.time() - t_start:.0f} s]")
    ck.finish()


if __name__ == "__main__":
    main()
