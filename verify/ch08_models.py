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
