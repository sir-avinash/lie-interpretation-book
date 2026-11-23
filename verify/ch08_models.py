"""Models and controllers of Chapter 8.

Imported by verify/ch08.py (which checks them) and by figures/src/ch08.py (which
draws them), so the figures show exactly what is verified.

Three groups:
  1. attitude tracking on SO(3): geometric PD, and geometric L1 adaptation;
  2. barrier functions: a point and a wall, a pendulum on S^2 inside a cone,
     a tilt limit on SO(3);
  3. a quadrotor that must not hit a sphere: the augmented barrier function.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import expm_so3, hat, vee  # noqa: E402

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
        if not np.all(np.isfinite(Om)) or np.linalg.norm(Om) > 1e3:
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


def run_cone(ref, tf, h=1e-3, controller="qp", cone=cone_static, q0=None, om0=None, **kw):
    n = int(round(tf / h))
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
        q, om = s2_step(q, om, F, h)
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
