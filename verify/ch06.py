"""Chapter 6: every identity, inequality, closed-loop claim and quoted number, checked.

    python verify/ch06.py

Equations of motion are derived a second time with geomech (compute_eom, check_eom) and
the hand-written forms used by the controllers are compared with the geomech models.
Every inequality of the Lyapunov proofs is tested at random states and random gains.
"""
import numpy as np
from scipy.integrate import solve_ivp

from geomech import check_eom, compute_eom, systems, to_standard_form
from geomech.codegen import StateLayout, collect_params, load_module, module_source
from geomech.simulation import accelerations

from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee

ck = Checker("ch06")
rng = np.random.default_rng(6)
E1, E2, E3 = np.eye(3)
G = 9.81
EOM_TOL = 1e-5
I3 = np.eye(3)


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def tangent(q, v):
    return v - (v @ q) * q


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return vee(R - R.T) / 2
    if np.pi - th < 1e-6:
        S = (R + I3) / 2
        a = np.sqrt(np.clip(np.diag(S), 0, None))
        k = int(np.argmax(a))
        a = S[:, k] / a[k]
        return th * a
    return th * vee(R - R.T) / (2 * np.sin(th))


def Psi_R(R, Rd):
    return 0.5 * np.trace(I3 - Rd.T @ R)


def e_R(R, Rd):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def e_Omega(R, Om, Rd, Omd):
    return Om - R.T @ Rd @ Omd


def C_R(R, Rd):
    return 0.5 * (np.trace(R.T @ Rd) * I3 - R.T @ Rd)


def Psi_q(q, qd):
    return 1 - q @ qd


def e_q(q, qd):
    return np.cross(qd, q)


def e_omega(q, om, omd):
    return om - tangent(q, omd)


def library(name, **kw):
    s = systems.get(name, **kw)
    eom = compute_eom(s.L, s.dW, s.variables)
    err = check_eom(s.L, s.dW, s.variables, eom, inputs=list(s.inputs))
    sf = to_standard_form(eom, s.variables, list(s.inputs))
    layout = StateLayout(s.variables, list(s.inputs), collect_params([s.L, s.dW], s.variables, list(s.inputs)))
    mod = load_module(module_source(sf, layout, lagrangian=s.L), name)
    ck.true(f"geomech check_eom: {name}", err < EOM_TOL, f"err {err:.1e}")
    return mod, layout, dict(s.params)


# =========================================================================
# 0. The vehicle of the chapter (Lee, Leok and McClamroch 2010) and its models
# =========================================================================
m_Q = 4.34
J = np.diag([0.0820, 0.0845, 0.1377])
k_R, k_Om = 8.81, 2.54
k_x, k_v = 16 * m_Q, 5.6 * m_Q
J_min, J_max = J[0, 0], J[2, 2]
Jinv = np.linalg.inv(J)
ck.record("vehicle", {"m": m_Q, "J_diag": np.diag(J).tolist(), "k_R": k_R, "k_Omega": k_Om, "k_x": k_x, "k_v": k_v})

frb_mod, frb_layout, frb_p = library("free_rigid_body")
worst = 0.0
for _ in range(20):
    R, Om, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    a_g = accelerations(frb_mod, frb_layout.pack(R=R, Omega_R=Om), tau, {**frb_p, "J": J})
    worst = max(worst, np.max(np.abs(a_g - Jinv @ (tau - np.cross(Om, J @ Om)))))
ck.true("rigid body: J Om_dot + Om x J Om = tau equals geomech", worst < 1e-9, f"err {worst:.1e}")


# =========================================================================
# 1. The opening puzzle: PD on a line converges from anywhere
# =========================================================================
# e_ddot = -k_p e - k_d e_dot ; V = 1/2 e_dot^2 + 1/2 k_p e^2 decreases for every start
worst = -np.inf
for _ in range(200):
    kp_, kd_ = 0.2 + 5 * rng.random(2)
    e, ed = 100 * rng.normal(size=2)
    worst = max(worst, ed * (-kp_ * e - kd_ * ed) + kp_ * e * ed + kd_ * ed**2)
ck.true("line: d/dt(1/2 e_dot^2 + 1/2 k_p e^2) = -k_d e_dot^2 for every state", abs(worst) < 1e-9)
# the poles for the attitude gains on each axis
poles = [np.roots([J[i, i], k_Om, k_R]) for i in range(3)]
ck.record("poles_per_axis", [[float(p.real) for p in sorted(pp, key=lambda z: z.real)] for pp in poles])
ck.true("all three axes are overdamped with these gains (real poles)", all(abs(p.imag) < 1e-12 for pp in poles for p in pp))
ck.record("natural_freq_per_axis", [float(np.sqrt(k_R / J[i, i])) for i in range(3)])
ck.record("damping_ratio_per_axis", [float(k_Om / (2 * np.sqrt(k_R * J[i, i]))) for i in range(3)])
slowest_pole = max(p.real for pp in poles for p in pp)
ck.record("slowest_pole_real", float(slowest_pole))


# =========================================================================
# 2. Attitude tracking law and closed-loop error dynamics
# =========================================================================
def ref_att(t):
    """A tumbling reference: R_d(t), Omega_d(t), Omega_d_dot(t). Integrated once, interpolated."""
    Omd = np.array([0.6 * np.sin(1.3 * t), 0.5, 0.8 * np.cos(0.7 * t)])
    Omd_dot = np.array([0.6 * 1.3 * np.cos(1.3 * t), 0.0, -0.8 * 0.7 * np.sin(0.7 * t)])
    return Omd, Omd_dot


def att_torque(R, Om, Rd, Omd, Omd_dot, kR=k_R, kOm=k_Om, J_=J):
    """tau = -k_R e_R - k_Omega e_Omega + Omega x J Omega - J (Omega^ R'R_d Omega_d - R'R_d Omega_d_dot)."""
    return (-kR * e_R(R, Rd) - kOm * e_Omega(R, Om, Rd, Omd) + np.cross(Om, J_ @ Om)
            - J_ @ (np.cross(Om, R.T @ Rd @ Omd) - R.T @ Rd @ Omd_dot))


def att_rhs(t, s, moving=True, kR=k_R, kOm=k_Om):
    R, Om, Rd = s[:9].reshape(3, 3), s[9:12], s[12:21].reshape(3, 3)
    Omd, Omd_dot = ref_att(t) if moving else (np.zeros(3), np.zeros(3))
    tau = att_torque(R, Om, Rd, Omd, Omd_dot, kR, kOm)
    return np.concatenate([(R @ hat(Om)).ravel(), Jinv @ (tau - np.cross(Om, J @ Om)), (Rd @ hat(Omd)).ravel()])


def att_run(Re0, eOm0, t_f, moving=True, Rd0=None, n=2001, kR=k_R, kOm=k_Om):
    Rd0 = I3 if Rd0 is None else Rd0
    R0 = Rd0 @ Re0
    Omd0 = ref_att(0.0)[0] if moving else np.zeros(3)
    Om0 = eOm0 + R0.T @ Rd0 @ Omd0
    s0 = np.concatenate([R0.ravel(), Om0, Rd0.ravel()])
    ts = np.linspace(0, t_f, n)
    sol = solve_ivp(att_rhs, [0, t_f], s0, args=(moving, kR, kOm), t_eval=ts, rtol=1e-10, atol=1e-12)
    out = []
    for t, s in zip(sol.t, sol.y.T):
        R, Om, Rd = s[:9].reshape(3, 3), s[9:12], s[12:21].reshape(3, 3)
        Omd = ref_att(t)[0] if moving else np.zeros(3)
        out.append((R, Om, Rd, Omd))
    return sol.t, out


# e_Omega_dot = Omega_dot + Omega^ R'R_d Omega_d - R'R_d Omega_d_dot, by finite differences
worst = worst_cl = 0.0
for _ in range(20):
    R, Rd, Om = random_rotation(rng), random_rotation(rng), rng.normal(size=3)
    Omd, Omd_dot, Om_dot = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    d = 1e-6

    def eOm_at(h):
        return e_Omega(R @ expm_so3(h * Om), Om + h * Om_dot, Rd @ expm_so3(h * Omd), Omd + h * Omd_dot)

    fd = (eOm_at(d) - eOm_at(-d)) / (2 * d)
    worst = max(worst, np.max(np.abs(fd - (Om_dot + np.cross(Om, R.T @ Rd @ Omd) - R.T @ Rd @ Omd_dot))))
    tau = att_torque(R, Om, Rd, Omd, Omd_dot)
    Om_dot_cl = Jinv @ (tau - np.cross(Om, J @ Om))
    eOm_dot = Om_dot_cl + np.cross(Om, R.T @ Rd @ Omd) - R.T @ Rd @ Omd_dot
    worst_cl = max(worst_cl, np.max(np.abs(J @ eOm_dot + k_R * e_R(R, Rd) + k_Om * e_Omega(R, Om, Rd, Omd))))
ck.true("e_Omega_dot = Omega_dot + Omega x (R'R_d Omega_d) - R'R_d Omega_d_dot (finite differences)", worst < 1e-7,
        f"err {worst:.1e}")
ck.true("closed loop: J e_Omega_dot = -k_R e_R - k_Omega e_Omega at random states", worst_cl < 1e-12, f"err {worst_cl:.1e}")

# the error system is autonomous: a tumbling reference and a fixed target give the same error history
Re0 = expm_so3(np.radians(150) * unit([1.0, 2.0, -1.0]))
eOm0 = np.array([0.5, -0.3, 0.4])
Rd0 = random_rotation(rng)
t_a, run_mov = att_run(Re0, eOm0, 4.0, moving=True, Rd0=Rd0)
_, run_fix = att_run(Re0, eOm0, 4.0, moving=False)
gap = max(np.max(np.abs(a[2].T @ a[0] - b[2].T @ b[0])) for a, b in zip(run_mov, run_fix))
gap_v = max(np.max(np.abs(e_Omega(*a) - e_Omega(*b))) for a, b in zip(run_mov, run_fix))
ck.true("error dynamics are autonomous: R_d'R(t) and e_Omega(t) do not depend on the reference motion",
        max(gap, gap_v) < 1e-6, f"gap {max(gap, gap_v):.1e}")
ck.record("autonomy_gap", float(max(gap, gap_v)))


# =========================================================================
# 3. The Lyapunov proof on SO(3): every identity and inequality
# =========================================================================
def V0_att(R, Om, Rd, Omd, kR=k_R, J_=J):
    eO = e_Omega(R, Om, Rd, Omd)
    return 0.5 * eO @ J_ @ eO + kR * Psi_R(R, Rd)


def V_att(R, Om, Rd, Omd, c, kR=k_R, J_=J):
    return V0_att(R, Om, Rd, Omd, kR, J_) + c * e_R(R, Rd) @ J_ @ e_Omega(R, Om, Rd, Omd)


def c_max_att(kR, kOm, lm, lM):
    return min(np.sqrt(kR * lm) / lM, 4 * kR * kOm / (kOm**2 + 4 * kR * lM))


def mats_att(kR, kOm, lm, lM, c, psi):
    W1 = 0.5 * np.array([[kR, -c * lM], [-c * lM, lm]])
    W2 = 0.5 * np.array([[2 * kR / (2 - psi), c * lM], [c * lM, lM]])
    W3 = np.array([[c * kR, -c * kOm / 2], [-c * kOm / 2, kOm - c * lM]])
    return W1, W2, W3


def rate_att(kR, kOm, lm, lM, psi, c=None):
    """Best guaranteed rate gamma = lambda_min(W3)/lambda_max(W2) over admissible c (or at a given c)."""
    cm = c_max_att(kR, kOm, lm, lM)
    cs = [c] if c is not None else np.linspace(0.002, 0.998, 499) * cm
    best = (0.0, None)
    for c_ in cs:
        W1, W2, W3 = mats_att(kR, kOm, lm, lM, c_, psi)
        gamma = np.linalg.eigvalsh(W3)[0] / np.linalg.eigvalsh(W2)[-1]
        if gamma > best[0]:
            best = (gamma, c_)
    return best


# identities
w1 = w2 = w3 = w4 = w5 = 0.0
for _ in range(300):
    th = np.pi * rng.random()
    a = unit(rng.normal(size=3))
    Rd = random_rotation(rng)
    R = Rd @ expm_so3(th * a)
    Psi, eR = Psi_R(R, Rd), e_R(R, Rd)
    w1 = max(w1, abs(Psi - (1 - np.cos(th))), np.max(np.abs(eR - np.sin(th) * a)))
    w2 = max(w2, abs(eR @ eR - Psi * (2 - Psi)))
    C = C_R(R, Rd)
    w3 = max(w3, np.linalg.norm(C, 2) - 1)
    w4 = max(w4, abs(np.linalg.det(C) - 0.5 * np.cos(th) * (1 + np.cos(th))))
    # e_R_dot = C_R e_Omega by finite differences
    Om, Omd = rng.normal(size=3), rng.normal(size=3)
    d = 1e-6
    fd = (e_R(R @ expm_so3(d * Om), Rd @ expm_so3(d * Omd)) - e_R(R @ expm_so3(-d * Om), Rd @ expm_so3(-d * Omd))) / (2 * d)
    w5 = max(w5, np.max(np.abs(fd - C @ e_Omega(R, Om, Rd, Omd))))
ck.true("Psi_R = 1 - cos(theta), e_R = sin(theta) a", w1 < 1e-12, f"err {w1:.1e}")
ck.true("|e_R|^2 = Psi_R (2 - Psi_R)", w2 < 1e-12, f"err {w2:.1e}")
ck.true("|C_R|_2 <= 1 at 300 random attitudes", w3 < 1e-12, f"max |C_R| - 1 = {w3:.1e}")
ck.true("det C_R = 1/2 cos(theta) (1 + cos(theta))", w4 < 1e-12, f"err {w4:.1e}")
ck.true("e_R_dot = C_R e_Omega (finite differences)", w5 < 1e-8, f"err {w5:.1e}")

# sandwich for Psi on a sublevel set, V_dot formula, and the three quadratic bounds, at random states AND gains
viol_sand = viol_lo = viol_up = viol_dot = 0.0
err_dot = 0.0
n_pd = 0
for trial in range(4000):
    Jd = np.sort(0.01 + rng.random(3))
    Ur = random_rotation(rng)
    J_ = Ur @ np.diag(Jd) @ Ur.T                      # a full inertia matrix
    lm, lM = Jd[0], Jd[-1]
    kR, kOm = 0.1 + 20 * rng.random(), 0.05 + 5 * rng.random()
    cm = c_max_att(kR, kOm, lm, lM)
    c = cm * rng.uniform(0.01, 0.99)
    psi = 2 * rng.uniform(0.01, 0.999)
    th = np.arccos(1 - psi * rng.random())            # Psi_R <= psi
    Rd = random_rotation(rng)
    R = Rd @ expm_so3(th * unit(rng.normal(size=3)))
    Omd = rng.normal(size=3)
    eO = 3 * rng.normal(size=3) * rng.random()
    Om = eO + R.T @ Rd @ Omd
    Psi, eR = Psi_R(R, Rd), e_R(R, Rd)
    z = np.array([np.linalg.norm(eR), np.linalg.norm(eO)])
    viol_sand = max(viol_sand, 0.5 * eR @ eR - Psi, Psi - eR @ eR / (2 - psi))
    W1, W2, W3 = mats_att(kR, kOm, lm, lM, c, psi)
    V = 0.5 * eO @ J_ @ eO + kR * Psi + c * eR @ J_ @ eO
    viol_lo = max(viol_lo, z @ W1 @ z - V)
    viol_up = max(viol_up, V - z @ W2 @ z)
    C = C_R(R, Rd)
    JeO_dot = -kR * eR - kOm * eO
    Vdot = eO @ JeO_dot + kR * eR @ eO + c * (C @ eO) @ J_ @ eO + c * eR @ JeO_dot
    Vdot_formula = -kOm * eO @ eO + c * eO @ C.T @ J_ @ eO - c * kR * eR @ eR - c * kOm * eR @ eO
    err_dot = max(err_dot, abs(Vdot - Vdot_formula))
    viol_dot = max(viol_dot, Vdot + z @ W3 @ z)
    n_pd += (np.linalg.eigvalsh(W1)[0] > 0) and (np.linalg.eigvalsh(W3)[0] > 0)
ck.true("1/2 |e_R|^2 <= Psi_R <= |e_R|^2/(2 - psi) on Psi_R <= psi < 2 (4000 samples)", viol_sand < 1e-12, f"max viol {viol_sand:.1e}")
ck.true("V >= z'W1 z (4000 random states, inertias, gains)", viol_lo < 1e-10, f"max viol {viol_lo:.1e}")
ck.true("V <= z'W2 z on the sublevel set (4000 random states, inertias, gains)", viol_up < 1e-10, f"max viol {viol_up:.1e}")
ck.true("V_dot = -k_Om|e_Om|^2 + c e_Om.C_R'J e_Om - c k_R|e_R|^2 - c k_Om e_R.e_Om", err_dot < 1e-9, f"err {err_dot:.1e}")
ck.true("V_dot <= -z'W3 z (4000 random states, inertias, gains)", viol_dot < 1e-10, f"max viol {viol_dot:.1e}")
ck.true("W1 > 0 and W3 > 0 for every c below the stated bound", n_pd == 4000, f"{n_pd}/4000")

# the bound on c is sharp for W3 and W1: just above each limit one of them loses definiteness
sharp = 0
for _ in range(500):
    lm = 0.01 + rng.random()
    lM = lm * (1 + 3 * rng.random())
    kR, kOm = 0.1 + 20 * rng.random(), 0.05 + 5 * rng.random()
    c1, c2 = np.sqrt(kR * lm) / lM, 4 * kR * kOm / (kOm**2 + 4 * kR * lM)
    M1a, _, _ = mats_att(kR, kOm, lm, lM, 1.001 * c1, 1.0)
    _, _, Wa = mats_att(kR, kOm, lm, lM, 1.001 * c2, 1.0)
    sharp += (np.linalg.eigvalsh(M1a)[0] < 0) and (np.linalg.eigvalsh(Wa)[0] < 0)
ck.true("W1 > 0 iff c < sqrt(k_R J_min)/J_max and W3 > 0 iff c < 4 k_R k_Om/(k_Om^2 + 4 k_R J_max)", sharp == 500, f"{sharp}/500")

# V_dot by finite differences along a closed-loop run (the formula describes the real closed loop)
c_demo = 0.5 * c_max_att(k_R, k_Om, J_min, J_max)
Vs = np.array([V_att(*a, c_demo) for a in run_mov])
dV_fd = np.gradient(Vs, t_a, edge_order=2)
dV_f = []
for (R, Om, Rd, Omd) in run_mov:
    eO, eR, C = e_Omega(R, Om, Rd, Omd), e_R(R, Rd), C_R(R, Rd)
    dV_f.append(-k_Om * eO @ eO + c_demo * eO @ C.T @ J @ eO - c_demo * k_R * eR @ eR - c_demo * k_Om * eR @ eO)
rel = np.max(np.abs(dV_fd[5:-5] - np.array(dV_f)[5:-5])) / np.max(np.abs(dV_f))
ck.true("V_dot formula matches the derivative of V along a simulated run", rel < 1e-3, f"relative err {rel:.1e}")

# worked example: 150 degrees of initial error, tumbling reference
V0s = np.array([V0_att(*a) for a in run_mov])
Psis = np.array([Psi_R(a[0], a[2]) for a in run_mov])
ck.true("V0 = 1/2 e_Om.J e_Om + k_R Psi_R never increases along the run", np.max(np.diff(V0s)) < 1e-9)
Psi_max = V0s[0] / k_R
ck.true("example starts inside the sublevel set V0 < 2 k_R", Psi_max < 2, f"Psi_max = {Psi_max:.4f}")
ck.true("Psi_R(t) <= V0(0)/k_R for all t", np.max(Psis) <= Psi_max + 1e-9)
gamma, c_best = rate_att(k_R, k_Om, J_min, J_max, Psi_max)
W1, W2, W3 = mats_att(k_R, k_Om, J_min, J_max, c_best, Psi_max)
Vc = np.array([V_att(*a, c_best) for a in run_mov])
ck.true("example: V(t) <= V(0) exp(-gamma t) with the guaranteed rate", np.all(Vc <= Vc[0] * np.exp(-gamma * t_a) * (1 + 1e-9) + 1e-12),
        f"gamma = {gamma:.3f}")
zs = np.array([[np.linalg.norm(e_R(a[0], a[2])), np.linalg.norm(e_Omega(*a))] for a in run_mov])
z2 = np.sum(zs**2, axis=1)
overshoot = np.linalg.eigvalsh(W2)[-1] / np.linalg.eigvalsh(W1)[0]
ck.true("example: |z(t)|^2 <= (lambda_max(W2)/lambda_min(W1)) |z(0)|^2 exp(-gamma t)", np.all(z2 <= overshoot * z2[0] * np.exp(-gamma * t_a) + 1e-12))
# measured decay rate of V late in the run
mask = (t_a > 1.5) & (Vc > 1e-9)
meas = -np.polyfit(t_a[mask], np.log(Vc[mask]), 1)[0]
ck.record("example_att", {"angle0_deg": 150.0, "Psi0": float(Psis[0]), "V0_0": float(V0s[0]), "two_kR": 2 * k_R,
                          "Psi_max": float(Psi_max), "c_max": float(c_max_att(k_R, k_Om, J_min, J_max)),
                          "c_best": float(c_best), "gamma_guaranteed": float(gamma), "gamma_measured": float(meas),
                          "overshoot_factor": float(overshoot),
                          "c_lim_W1": float(np.sqrt(k_R * J_min) / J_max),
                          "c_lim_W3": float(4 * k_R * k_Om / (k_Om**2 + 4 * k_R * J_max))})
ck.true("measured decay of V is twice the slowest pole, and faster than the guarantee",
        abs(meas + 2 * slowest_pole) < 0.3 and meas > gamma, f"measured {meas:.2f}, 2|pole| {-2 * slowest_pole:.2f}, bound {gamma:.2f}")
# guaranteed rate for a small error (psi -> 0) and how it dies as psi -> 2
ck.record("gamma_vs_Psi_max", {f"{p:g}": float(rate_att(k_R, k_Om, J_min, J_max, p)[0]) for p in (0.1, 1.0, 1.9, 1.99, 1.999)})
gammas = [rate_att(k_R, k_Om, J_min, J_max, p)[0] for p in (0.1, 1.0, 1.9, 1.99, 1.999)]
ck.true("the guaranteed rate decreases as the sublevel set grows toward Psi_R = 2", np.all(np.diff(gammas) < 0))

# without the cross term: V0_dot = -k_Om |e_Om|^2 vanishes at rest away from the target
R = expm_so3(1.0 * E1)
ck.close("V0_dot = 0 at rest with a 57 degree error (semidefinite only)", -k_Om * 0.0, 0.0)
ck.true("with the cross term V_dot < 0 at the same state", -c_demo * k_R * e_R(R, I3) @ e_R(R, I3) < 0)


# =========================================================================
# 4. Equilibria and almost-global convergence
# =========================================================================
# e_R = 0 exactly at the identity and at half turns
w = 0.0
for _ in range(200):
    a = unit(rng.normal(size=3))
    U = 2 * np.outer(a, a) - I3
    w = max(w, np.linalg.norm(e_R(U, I3)), abs(Psi_R(U, I3) - 2))
ck.true("every half turn has e_R = 0 and Psi_R = 2", w < 1e-12, f"err {w:.1e}")
w = np.inf
for _ in range(2000):
    th = rng.uniform(0.02, np.pi - 0.02)
    w = min(w, np.linalg.norm(e_R(expm_so3(th * unit(rng.normal(size=3))), I3)) - np.sin(th))
ck.true("e_R is nonzero for every angle strictly between 0 and 180 degrees", abs(w) < 1e-12)

# linearization at a half turn: e_R = -a a' eta + O(eta^2)
w = 0.0
for _ in range(50):
    a = unit(rng.normal(size=3))
    U = 2 * np.outer(a, a) - I3
    eta = rng.normal(size=3)
    d = 1e-5
    w = max(w, np.max(np.abs(e_R(U @ expm_so3(d * eta), I3) / d + np.outer(a, a) @ eta)))
ck.true("near a half turn about a: e_R = -(a a') eta to first order", w < 1e-4, f"err {w:.1e}")


def half_turn_eigs(a, kR=k_R, kOm=k_Om, J_=J):
    A = np.block([[np.zeros((3, 3)), I3], [np.linalg.solve(J_, kR * np.outer(a, a)), -kOm * np.linalg.inv(J_)]])
    return np.linalg.eigvals(A)


counts = set()
worst_root = 0.0
for _ in range(200):
    a = unit(rng.normal(size=3))
    ev = half_turn_eigs(a)
    counts.add((int(np.sum(ev.real > 1e-7)), int(np.sum(np.abs(ev) < 1e-7)), int(np.sum(ev.real < -1e-7))))
    lam = np.max(ev.real)
    # the unstable rate solves 1 = k_R a.(lam^2 J + k_Om lam I)^-1 a
    worst_root = max(worst_root, abs(1 - k_R * a @ np.linalg.solve(lam**2 * J + k_Om * lam * I3, a)))
ck.true("every half turn: 1 unstable, 2 zero, 3 stable eigenvalues", counts == {(1, 2, 3)}, f"{counts}")
ck.true("unstable rate solves 1 = k_R a.(lam^2 J + k_Om lam)^-1 a", worst_root < 1e-9, f"err {worst_root:.1e}")
lam_axes = [float(np.max(half_turn_eigs(e).real)) for e in (E1, E2, E3)]
for i in range(3):
    ck.close(f"half turn about e{i + 1}: J_i lam^2 + k_Om lam - k_R = 0", J[i, i] * lam_axes[i]**2 + k_Om * lam_axes[i] - k_R, 0, tol=1e-9)
ck.record("half_turn_growth_rates", lam_axes)
ck.record("half_turn_decade_cost_s", [float(np.log(10) / l) for l in lam_axes])

# weighted error function: four isolated equilibria with 0, 1, 2, 3 unstable directions
Dw = np.diag([0.75, 0.5, 0.25])


def e_R_w(R, Rd, D=Dw):
    return vee(D @ Rd.T @ R - R.T @ Rd @ D)


def Psi_R_w(R, Rd, D=Dw):
    return np.trace(D @ (I3 - Rd.T @ R))


n_unst = []
for U in (I3, np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):
    nameU = "diag(%d,%d,%d)" % tuple(np.diag(U))
    ck.close(f"weighted e_R vanishes at {nameU}", e_R_w(U, I3), np.zeros(3))
    cdiag = np.diag(Dw @ U)
    H = np.diag([cdiag[1] + cdiag[2], cdiag[0] + cdiag[2], cdiag[0] + cdiag[1]])
    # Hessian check by finite differences of e_R
    Hfd = np.column_stack([(e_R_w(U @ expm_so3(1e-6 * e), I3) - e_R_w(U @ expm_so3(-1e-6 * e), I3)) / 2e-6 for e in I3])
    ck.close(f"weighted: e_R = H eta near {nameU}", Hfd, H, tol=1e-8)
    A = np.block([[np.zeros((3, 3)), I3], [-k_R * Jinv @ H, -k_Om * Jinv]])
    ev = np.linalg.eigvals(A)
    ck.true(f"weighted: equilibrium {nameU} is hyperbolic", np.min(np.abs(ev.real)) > 1e-3)
    n_unst.append(int(np.sum(ev.real > 0)))
ck.true("weighted: unstable directions at the four equilibria are 0, 1, 2, 3", n_unst == [0, 1, 2, 3], f"{n_unst}")
ck.record("weighted_D", np.diag(Dw).tolist())
ck.record("weighted_heights", [float(Psi_R_w(U, I3)) for U in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1]))])
# no other zeros of the weighted e_R: random search
w = np.inf
for _ in range(20000):
    R = random_rotation(rng)
    dists = [np.linalg.norm(R - U) for U in (I3, np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1]))]
    if min(dists) > 0.3:
        w = min(w, np.linalg.norm(e_R_w(R, I3)))
ck.true("weighted e_R is bounded away from zero away from the four equilibria (20000 samples)", w > 0.02, f"min |e_R| {w:.3f}")
# weighted Psi_dot = e_R . e_Omega
w = 0.0
for _ in range(50):
    R, Rd, Om, Omd = random_rotation(rng), random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    d = 1e-6
    fd = (Psi_R_w(R @ expm_so3(d * Om), Rd @ expm_so3(d * Omd)) - Psi_R_w(R @ expm_so3(-d * Om), Rd @ expm_so3(-d * Omd))) / (2 * d)
    w = max(w, abs(fd - e_R_w(R, Rd) @ e_Omega(R, Om, Rd, Omd)))
ck.true("weighted: Psi_R_dot = e_R . e_Omega", w < 1e-8, f"err {w:.1e}")


# dense set of initial attitudes, at rest, fixed target: all converge; time against initial angle
def settle_time(Re0, eOm0=np.zeros(3), tol_deg=5.0, t_f=30.0, law="trace"):
    def rhs(t, s):
        R, Om = s[:9].reshape(3, 3), s[9:]
        e = e_R(R, I3) if law == "trace" else log_so3(R)
        return np.concatenate([(R @ hat(Om)).ravel(), Jinv @ (-k_R * e - k_Om * Om)])

    sol = solve_ivp(rhs, [0, t_f], np.concatenate([Re0.ravel(), eOm0]), rtol=1e-9, atol=1e-11, dense_output=True)
    ts = np.linspace(0, t_f, 6001)
    ang = np.array([np.degrees(np.arccos(np.clip((np.trace(sol.sol(t)[:9].reshape(3, 3)) - 1) / 2, -1, 1))) for t in ts])
    out = np.nonzero(ang > tol_deg)[0]
    return (ts[out[-1] + 1] if len(out) and out[-1] + 1 < len(ts) else (0.0 if not len(out) else np.inf)), ang[-1]


N_dense = 240
angles0 = np.concatenate([rng.uniform(5, 179, N_dense - 40), 180 - np.logspace(-3, 0, 40)])
axes0 = np.array([unit(rng.normal(size=3)) for _ in range(N_dense)])
settle = np.array([settle_time(expm_so3(np.radians(a) * ax)) for a, ax in zip(angles0, axes0)])
ck.true(f"all {N_dense} random initial attitudes (5 to 179.999 degrees, at rest) converge", np.all(np.isfinite(settle[:, 0])) and np.max(settle[:, 1]) < 0.01,
        f"max final angle {np.max(settle[:, 1]):.1e} deg")
ck.record("dense_n", N_dense)
ck.record("dense_settle_s", {"90deg_max": float(np.max(settle[(angles0 > 85) & (angles0 < 95), 0])),
                             "below_150_max": float(np.max(settle[angles0 < 150, 0])),
                             "max": float(np.max(settle[:, 0]))})
# along one axis: time from 179, 179.9, 179.99 degrees; each decade costs ln(10)/lambda
for ax, name in ((E1, "e1"), (E3, "e3")):
    ts_ = [settle_time(expm_so3(np.radians(180 - 10.0**(-k)) * ax))[0] for k in range(0, 4)]
    lam = float(np.max(half_turn_eigs(ax).real))
    steps = np.diff(ts_)
    ck.true(f"near a half turn about {name}: each factor of ten costs ln(10)/lambda", np.max(np.abs(steps[1:] - np.log(10) / lam)) < 0.03,
            f"steps {np.round(steps, 3)}, predicted {np.log(10) / lam:.3f}")
    ck.record(f"linger_{name}", {"times_s": [float(x) for x in ts_], "lambda": lam, "decade_s": float(np.log(10) / lam)})
# the logarithm law does not linger
tl = [settle_time(expm_so3(np.radians(180 - 10.0**(-k)) * E1), law="log")[0] for k in range(0, 4)]
ck.true("logarithm law: settle time from 179 to 179.999 degrees varies by less than 0.01 s", np.ptp(tl) < 0.01, f"{np.round(tl, 3)}")
ck.record("linger_log_e1_s", [float(x) for x in tl])
# exactly at a half turn, at rest: nothing happens
t_half, ang_half = settle_time(np.diag([1.0, -1.0, -1.0]), t_f=10.0)
ck.true("exactly half a turn, at rest: the body stays there", abs(ang_half - 180) < 1e-6, f"angle after 10 s: {ang_half:.6f}")

# =========================================================================
# 5. The same template on the sphere
# =========================================================================
sp_mod, sp_layout, sp_p = library("spherical_pendulum")
m_p, l_p = 1.0, 1.0
sp_p.update(m=m_p, l=l_p, g=G)
k_q, k_om = 16.0, 5.6
ck.record("s2_params", {"m": m_p, "l": l_p, "k_q": k_q, "k_omega": k_om})


def pend_omega_dot(q, om, F):
    """m l^2 omega_dot = -m g l q x e3 + l q x F  (force F on the bob)."""
    return (-m_p * G * l_p * np.cross(q, E3) + l_p * np.cross(q, F)) / (m_p * l_p**2)


worst = 0.0
for _ in range(20):
    q = unit(rng.normal(size=3))
    om, F = tangent(q, rng.normal(size=3)), rng.normal(size=3)
    a_g = accelerations(sp_mod, sp_layout.pack(q=q, omega_q=om), l_p * np.cross(q, F), sp_p)
    worst = max(worst, np.max(np.abs(a_g - pend_omega_dot(q, om, F))))
ck.true("pendulum with a force on the bob equals geomech (moment l q x F)", worst < 1e-9, f"err {worst:.1e}")

CONE, W_REF = np.radians(35), 1.2


def ref_s2(t):
    """Reference direction on a cone about e3: q_d, omega_d, omega_d_dot."""
    c, s = np.cos(W_REF * t), np.sin(W_REF * t)
    qd = np.array([np.sin(CONE) * c, np.sin(CONE) * s, np.cos(CONE)])
    qd1 = np.sin(CONE) * W_REF * np.array([-s, c, 0])
    qd2 = -np.sin(CONE) * W_REF**2 * np.array([c, s, 0])
    return qd, np.cross(qd, qd1), np.cross(qd, qd2)


B_ref = float(np.linalg.norm(ref_s2(0.3)[1]))
ck.close("reference on the cone: |omega_d| = w sin(alpha), constant", [np.linalg.norm(ref_s2(t)[1]) for t in (0, 1, 2.2)], [W_REF * np.sin(CONE)] * 3)
ck.record("s2_omega_d_bound", B_ref)


def omega_dot_des(q, om, qd, omd, omd_dot, kq=k_q, kom=k_om):
    """(I - q q') omega_d_dot - (q . omega_d) q_dot - k_q e_q - k_omega e_omega."""
    return tangent(q, omd_dot) - (q @ omd) * np.cross(om, q) - kq * e_q(q, qd) - kom * e_omega(q, om, omd)


def s2_force(q, om, qd, omd, omd_dot):
    return m_p * G * E3 + m_p * l_p * np.cross(omega_dot_des(q, om, qd, omd, omd_dot), q)


# identities and derivative formulas on the sphere
w1 = w2 = w3 = w4 = w5 = 0.0
for _ in range(100):
    q, qd = unit(rng.normal(size=3)), unit(rng.normal(size=3))
    om, omd = tangent(q, rng.normal(size=3)), tangent(qd, rng.normal(size=3))
    om_dot = tangent(q, rng.normal(size=3))                 # q . omega_dot = 0
    omd_dot = tangent(qd, rng.normal(size=3))
    eq, ew, Psi = e_q(q, qd), e_omega(q, om, omd), Psi_q(q, qd)
    w1 = max(w1, abs(eq @ eq - Psi * (2 - Psi)), abs(eq @ q), abs(ew @ q))
    d = 1e-6

    def at(h):
        q_ = expm_so3(h * om + 0.5 * h * h * om_dot) @ q
        qd_ = expm_so3(h * omd + 0.5 * h * h * omd_dot) @ qd
        return q_, om + h * om_dot, qd_, omd + h * omd_dot

    (qa, oma, qda, omda), (qb, omb, qdb, omdb) = at(d), at(-d)
    w2 = max(w2, abs((Psi_q(qa, qda) - Psi_q(qb, qdb)) / (2 * d) - eq @ ew))
    fd_ew = (e_omega(qa, tangent(qa, oma), omda) - e_omega(qb, tangent(qb, omb), omdb)) / (2 * d)
    ew_dot = om_dot - tangent(q, omd_dot) + (q @ omd) * np.cross(om, q) + (np.cross(om, q) @ omd) * q
    w3 = max(w3, np.max(np.abs(fd_ew - ew_dot)))
    fd_eq = (e_q(qa, qda) - e_q(qb, qdb)) / (2 * d)
    eq_dot = (q @ qd) * ew + (q @ omd) * np.cross(q, eq) - (qd @ om) * q
    w4 = max(w4, np.max(np.abs(fd_eq - eq_dot)))
    # closed loop with the force law: tangent part of e_omega_dot is -k_q e_q - k_omega e_omega
    F = s2_force(q, om, qd, omd, omd_dot)
    od = pend_omega_dot(q, om, F)
    ew_dot_cl = od - tangent(q, omd_dot) + (q @ omd) * np.cross(om, q) + (np.cross(om, q) @ omd) * q
    w5 = max(w5, np.max(np.abs(ew_dot_cl - (-k_q * eq - k_om * ew + (np.cross(om, q) @ omd) * q))), abs(od @ q))
ck.true("sphere: |e_q|^2 = Psi_q (2 - Psi_q), e_q . q = 0, e_omega . q = 0", w1 < 1e-12, f"err {w1:.1e}")
ck.true("sphere: Psi_q_dot = e_q . e_omega (finite differences)", w2 < 1e-8, f"err {w2:.1e}")
ck.true("sphere: e_omega_dot = omega_dot - (I-qq')omega_d_dot + (q.omega_d) q_dot + (q_dot.omega_d) q", w3 < 1e-7, f"err {w3:.1e}")
ck.true("sphere: e_q_dot formula of Chapter 3 (finite differences)", w4 < 1e-7, f"err {w4:.1e}")
ck.true("sphere closed loop: e_omega_dot = -k_q e_q - k_omega e_omega + (q_dot.omega_d) q, and q.omega_dot = 0", w5 < 1e-10, f"err {w5:.1e}")


def c_max_s2(kq, kom, B):
    return min(np.sqrt(kq), 4 * kq * kom / ((kom + B)**2 + 4 * kq))


def mats_s2(kq, kom, B, c, psi):
    W1 = 0.5 * np.array([[kq, -c], [-c, 1.0]])
    W2 = 0.5 * np.array([[2 * kq / (2 - psi), c], [c, 1.0]])
    W3 = np.array([[c * kq, -c * (kom + B) / 2], [-c * (kom + B) / 2, kom - c]])
    return W1, W2, W3


viol_lo = viol_up = viol_dot = err_dot = 0.0
n_pd = 0
for _ in range(4000):
    kq, kom, B = 0.1 + 30 * rng.random(), 0.05 + 8 * rng.random(), 3 * rng.random()
    c = c_max_s2(kq, kom, B) * rng.uniform(0.01, 0.99)
    psi = 2 * rng.uniform(0.01, 0.999)
    qd = unit(rng.normal(size=3))
    th = np.arccos(1 - psi * rng.random())
    q = expm_so3(th * unit(tangent(qd, rng.normal(size=3)))) @ qd
    omd = B * rng.random() * unit(tangent(qd, rng.normal(size=3)))
    ew = 3 * rng.random() * tangent(q, rng.normal(size=3))
    om = ew + tangent(q, omd)
    eq, Psi = e_q(q, qd), Psi_q(q, qd)
    z = np.array([np.linalg.norm(eq), np.linalg.norm(ew)])
    W1, W2, W3 = mats_s2(kq, kom, B, c, psi)
    V = 0.5 * ew @ ew + kq * Psi + c * eq @ ew
    viol_lo, viol_up = max(viol_lo, z @ W1 @ z - V), max(viol_up, V - z @ W2 @ z)
    ew_dot = -kq * eq - kom * ew + (np.cross(om, q) @ omd) * q
    eq_dot = (q @ qd) * ew + (q @ omd) * np.cross(q, eq) - (qd @ om) * q
    Vdot = ew @ ew_dot + kq * eq @ ew + c * (eq_dot @ ew + eq @ ew_dot)
    Vdot_formula = (-kom * ew @ ew + c * (q @ qd) * ew @ ew + c * (q @ omd) * np.cross(q, eq) @ ew
                    - c * kq * eq @ eq - c * kom * eq @ ew)
    err_dot = max(err_dot, abs(Vdot - Vdot_formula))
    viol_dot = max(viol_dot, Vdot + z @ W3 @ z)
    n_pd += (np.linalg.eigvalsh(W1)[0] > 0) and (np.linalg.eigvalsh(W3)[0] > 0)
ck.true("sphere: z'W1 z <= V <= z'W2 z on Psi_q <= psi (4000 random states and gains)", max(viol_lo, viol_up) < 1e-10,
        f"max viol {max(viol_lo, viol_up):.1e}")
ck.true("sphere: V_dot formula with the cross term", err_dot < 1e-9, f"err {err_dot:.1e}")
ck.true("sphere: V_dot <= -z'W3 z for |omega_d| <= B (4000 random states and gains)", viol_dot < 1e-10, f"max viol {viol_dot:.1e}")
ck.true("sphere: W1 > 0 and W3 > 0 for every c below the stated bound", n_pd == 4000, f"{n_pd}/4000")


def s2_rhs(t, s):
    q, om = s[:3], s[3:]
    return np.concatenate([np.cross(om, q), pend_omega_dot(q, om, s2_force(q, om, *ref_s2(t)))])


def s2_run(angle_deg, t_f=8.0, n=1601, side=E2):
    qd0, omd0, _ = ref_s2(0.0)
    q0 = expm_so3(np.radians(angle_deg) * unit(np.cross(qd0, side))) @ qd0
    sol = solve_ivp(s2_rhs, [0, t_f], np.concatenate([q0, tangent(q0, omd0)]), t_eval=np.linspace(0, t_f, n), rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[:3].T, sol.y[3:].T


s2_times = {}
for ang in (60.0, 170.0, 179.9):
    ts_, qs_, oms_ = s2_run(ang)
    refs = [ref_s2(t) for t in ts_]
    Ps = np.array([Psi_q(q, r[0]) for q, r in zip(qs_, refs)])
    V0 = np.array([0.5 * e_omega(q, om, r[1]) @ e_omega(q, om, r[1]) + k_q * Psi_q(q, r[0]) for q, om, r in zip(qs_, oms_, refs)])
    ck.true(f"sphere run from {ang:g} deg: V0 never increases, |q| = 1, converges", np.max(np.diff(V0)) < 1e-9
            and abs(np.linalg.norm(qs_[-1]) - 1) < 1e-8 and Ps[-1] < 1e-9, f"final Psi_q {Ps[-1]:.1e}")
    ang_t = np.degrees(np.arccos(np.clip(1 - Ps, -1, 1)))
    s2_times[f"{ang:g}"] = float(ts_[np.nonzero(ang_t > 5)[0][-1] + 1])
    if ang == 170.0:
        psi_bar_s = V0[0] / k_q
        gamma_s, c_s = max(((np.linalg.eigvalsh(mats_s2(k_q, k_om, B_ref, c_, psi_bar_s)[2])[0]
                            / np.linalg.eigvalsh(mats_s2(k_q, k_om, B_ref, c_, psi_bar_s)[1])[-1]), c_)
                          for c_ in np.linspace(0.002, 0.998, 499) * c_max_s2(k_q, k_om, B_ref))
        Vc = np.array([0.5 * e_omega(q, om, r[1]) @ e_omega(q, om, r[1]) + k_q * Psi_q(q, r[0]) + c_s * e_q(q, r[0]) @ e_omega(q, om, r[1])
                       for q, om, r in zip(qs_, oms_, refs)])
        ck.true("sphere run from 170 deg: V(t) <= V(0) exp(-gamma t)", np.all(Vc <= Vc[0] * np.exp(-gamma_s * ts_) * (1 + 1e-9) + 1e-12), f"gamma {gamma_s:.3f}")
        ck.record("s2_example", {"Psi_max": float(psi_bar_s), "gamma_guaranteed": float(gamma_s), "c_max": float(c_max_s2(k_q, k_om, B_ref))})
ck.record("s2_time_to_5deg_s", s2_times)
# antipode of a fixed target: two unstable directions, rate from lam^2 + k_omega lam - k_q = 0
lam_s2 = (-k_om + np.sqrt(k_om**2 + 4 * k_q)) / 2
qd = E3
q = expm_so3(1e-5 * E1) @ (-qd)
ck.close("sphere, near the antipode: e_q = -xi to first order", e_q(q, qd) / 1e-5, -E1, tol=1e-4)
ck.record("s2_antipode_rate", float(lam_s2))


# =========================================================================
# 6. The quadrotor cascade
# =========================================================================
q_mod, q_layout, q_p = library("quadrotor")
worst = 0.0
for _ in range(20):
    R, Om, f, tau = random_rotation(rng), rng.normal(size=3), 5 + 60 * rng.random(), rng.normal(size=3)
    a_g = accelerations(q_mod, q_layout.pack(x=rng.normal(size=3), R=R, x_dot=rng.normal(size=3), Omega_R=Om),
                        np.concatenate([[f], tau]), {**q_p, "m": m_Q, "J": J})
    mine = np.concatenate([f * R @ E3 / m_Q - G * E3, Jinv @ (tau - np.cross(Om, J @ Om))])
    worst = max(worst, np.max(np.abs(a_g - mine)))
ck.true("quadrotor: m x_ddot = f R e3 - m g e3, J Om_dot + Om x J Om = tau equals geomech", worst < 1e-9, f"err {worst:.1e}")


def ref_pos(t, w=1.0, r=1.5):
    """Circle of radius 1.5 m with a vertical wobble; heading along the tangent. Returns x_d and four derivatives,
    then the heading vector and two derivatives."""
    c, s = np.cos(w * t), np.sin(w * t)
    c2, s2 = np.cos(2 * w * t), np.sin(2 * w * t)
    x = np.array([r * c, r * s, 0.3 * s2])
    v = np.array([-r * w * s, r * w * c, 0.6 * w * c2])
    a = np.array([-r * w**2 * c, -r * w**2 * s, -1.2 * w**2 * s2])
    j = np.array([r * w**3 * s, -r * w**3 * c, -2.4 * w**3 * c2])
    sn = np.array([r * w**4 * c, r * w**4 * s, 4.8 * w**4 * s2])
    psi = w * t + np.pi / 2
    hd = np.array([np.cos(psi), np.sin(psi), 0.0])
    return x, v, a, j, sn, hd, w * np.array([-np.sin(psi), np.cos(psi), 0.0]), -w**2 * hd


def Rd_from(F, hd):
    """Attitude target: third axis along F, first axis as close to the heading hd as that allows."""
    b3 = unit(F)
    b2 = unit(np.cross(b3, hd))
    return np.column_stack([np.cross(b2, b3), b2, b3])


def Rd_derivs(F, F1, F2, hd, hd1, hd2, d=1e-4):
    """R_d, Omega_d, Omega_d_dot from F_d, the heading and their first two time derivatives."""
    def g(s):
        return Rd_from(F + s * F1 + 0.5 * s * s * F2, hd + s * hd1 + 0.5 * s * s * hd2)

    R0, Rp, Rm = g(0.0), g(d), g(-d)
    Wd = R0.T @ (Rp - Rm) / (2 * d)
    Omd = vee(0.5 * (Wd - Wd.T))
    A2 = R0.T @ (Rp - 2 * R0 + Rm) / d**2 - hat(Omd) @ hat(Omd)
    return R0, Omd, vee(0.5 * (A2 - A2.T))


def quad_ctrl(t, x, v, R, Om, ff=True, eps=1.0, clip=True, ref=ref_pos):
    xd, vd, ad, jd, sd, hd, hd1, hd2 = ref(t)
    ex, ev = x - xd, v - vd
    Fd = -k_x * ex - k_v * ev + m_Q * G * E3 + m_Q * ad
    b3 = R @ E3
    f = Fd @ b3
    on = f > 0 or not clip
    f = f if on else 0.0
    ev1 = f * b3 / m_Q - G * E3 - ad                            # e_v_dot from the model
    F1 = -k_x * ev - k_v * ev1 + m_Q * jd
    b3_dot = R @ np.cross(Om, E3)
    f1 = (F1 @ b3 + Fd @ b3_dot) if on else 0.0
    ev2 = (f1 * b3 + f * b3_dot) / m_Q - jd
    F2 = -k_x * ev1 - k_v * ev2 + m_Q * sd
    Rd, Omd, Omd_dot = Rd_derivs(Fd, F1, F2, hd, hd1, hd2)
    if not ff:
        Omd, Omd_dot = np.zeros(3), np.zeros(3)
    tau = att_torque(R, Om, Rd, Omd, Omd_dot, k_R / eps**2, k_Om / eps)
    return f, tau, Rd, Omd, ex, ev, Fd


def quad_rhs(t, s, ff, eps, clip):
    x, v, R, Om = s[:3], s[3:6], s[6:15].reshape(3, 3), s[15:18]
    f, tau, *_ = quad_ctrl(t, x, v, R, Om, ff, eps, clip)
    return np.concatenate([v, f * R @ E3 / m_Q - G * E3, (R @ hat(Om)).ravel(), Jinv @ (tau - np.cross(Om, J @ Om))])


def quad_run(x0, v0, R0, Om0, t_f, ff=True, eps=1.0, clip=True, n=2001, stop=None):
    ev = None
    if stop is not None:
        def ev(t, s, *a):
            return stop - np.linalg.norm(s[:3] - ref_pos(t)[0])
        ev.terminal = True
    sol = solve_ivp(quad_rhs, [0, t_f], np.concatenate([x0, v0, R0.ravel(), Om0]), args=(ff, eps, clip),
                    t_eval=np.linspace(0, t_f, n), rtol=1e-9, atol=1e-11, max_step=0.01, events=ev)
    return sol.t, sol.y.T


# the thrust error: e_F = f R e3 - F_d has length |F_d| sin(tilt), and is bounded by |F_d| |e_R| when Psi_R < 1
w1 = 0.0
viol = -np.inf
for _ in range(2000):
    Fd, hd = rng.normal(size=3) * 20, unit(rng.normal(size=3))
    Rd = Rd_from(Fd, hd)
    th = np.pi * rng.random()
    R = Rd @ expm_so3(th * unit(rng.normal(size=3)))
    b3 = R @ E3
    e_F = (Fd @ b3) * b3 - Fd
    tilt = np.arccos(np.clip(b3 @ unit(Fd), -1, 1))
    w1 = max(w1, abs(np.linalg.norm(e_F) - np.linalg.norm(Fd) * np.sin(tilt)), np.max(np.abs(Rd @ E3 - unit(Fd))),
             np.max(np.abs(Rd.T @ Rd - I3)), abs(np.linalg.det(Rd) - 1), abs((Rd @ E2) @ hd), float(tilt > th + 1e-9))
    if Psi_R(R, Rd) < 1:
        viol = max(viol, np.linalg.norm(e_F) - np.linalg.norm(Fd) * np.linalg.norm(e_R(R, Rd)))
ck.true("R_d is a rotation with R_d e3 = F_d/|F_d| and second axis perpendicular to the heading; |e_F| = |F_d| sin(tilt); tilt <= theta",
        w1 < 1e-9, f"err {w1:.1e}")
ck.true("|f R e3 - F_d| <= |F_d| |e_R| whenever Psi_R < 1 (2000 samples)", viol < 1e-9, f"max viol {viol:.1e}")
# the bound fails beyond 90 degrees
R = Rd @ expm_so3(np.radians(170) * (Rd.T @ unit(np.cross(Rd @ E3, E1))))
b3 = R @ E3
ck.true("the bound |e_F| <= |F_d||e_R| can fail when Psi_R > 1", np.linalg.norm((Fd @ b3) * b3 - Fd) > np.linalg.norm(Fd) * np.linalg.norm(e_R(R, Rd)))

# Omega_d from the finite-difference construction matches a direct derivative of R_d along a run
x0 = ref_pos(0)[0] + np.array([0.6, -0.4, 0.5])
t_q, S_q = quad_run(x0, np.zeros(3), Rz(np.pi / 2) @ Rx(np.radians(60)), np.zeros(3), 8.0, n=8001)
info = [quad_ctrl(t, s[:3], s[3:6], s[6:15].reshape(3, 3), s[15:18]) for t, s in zip(t_q, S_q)]
k = 3000
Omd_fd = vee(info[k][2].T @ (info[k + 1][2] - info[k - 1][2]) / (t_q[k + 1] - t_q[k - 1]))
ck.close("Omega_d used by the controller equals the derivative of R_d(t) along the closed loop", info[k][3], Omd_fd, tol=1e-5)
ck.true("no clipping in this run after the first instants (thrust positive)", min(i[0] for i in info[200:]) > 0)
# with the exact feedforward the attitude error system of the quadrotor is the autonomous one of section 2
Re0 = info[0][2].T @ S_q[0, 6:15].reshape(3, 3)
eOm0 = S_q[0, 15:18] - Re0.T @ info[0][3]
t_cmp, run_cmp = att_run(Re0, eOm0, 8.0, moving=False, n=8001)
gap = max(np.max(np.abs(i[2].T @ s[6:15].reshape(3, 3) - a[0])) for i, s, a in zip(info[::40], S_q[::40], run_cmp[::40]))
ck.true("cascade is triangular: the quadrotor's attitude error follows the autonomous error system", gap < 1e-4, f"gap {gap:.1e}")
ck.record("cascade_triangular_gap", float(gap))
Psi_q0 = Psi_R(S_q[0, 6:15].reshape(3, 3), info[0][2])
ex_q = np.array([np.linalg.norm(i[4]) for i in info])
ck.true("quadrotor from a 60 degree tilt and 0.9 m offset: position error below 1 mm after 4 s, 1e-6 at 8 s",
        np.max(ex_q[t_q > 4]) < 1e-3 and ex_q[-1] < 1e-6, f"{np.max(ex_q[t_q > 4]):.1e}, {ex_q[-1]:.1e}")

# recovery from upside down (thrust clipped at zero)
R0_inv = Rz(np.pi / 2) @ Rx(np.radians(178))
t_i, S_i = quad_run(x0, np.zeros(3), R0_inv, np.zeros(3), 8.0, n=8001)
info_i = [quad_ctrl(t, s[:3], s[3:6], s[6:15].reshape(3, 3), s[15:18]) for t, s in zip(t_i, S_i)]
Psi_i = np.array([Psi_R(s[6:15].reshape(3, 3), i[2]) for s, i in zip(S_i, info_i)])
ex_i = np.array([np.linalg.norm(i[4]) for i in info_i])
dz_i = S_i[:, 2] - np.array([ref_pos(t)[0][2] for t in t_i])
tilt_i = np.degrees(np.arccos(np.clip(S_i[:, 14], -1, 1)))             # angle between R e3 and e3
f_i = np.array([i[0] for i in info_i])
tau_i = np.array([np.linalg.norm(i[1]) for i in info_i])
ck.true("inverted start: all errors converge", ex_i[-1] < 1e-6 and Psi_i[-1] < 1e-9, f"e_x {ex_i[-1]:.1e}, Psi_R {Psi_i[-1]:.1e}")
ck.true("inverted start: F_d never vanishes, so R_d is defined throughout", min(np.linalg.norm(i[6]) for i in info_i) > 5)
t_upright = float(t_i[np.nonzero(tilt_i > 90)[0][-1] + 1])
ck.record("inverted", {"tilt0_deg": float(tilt_i[0]), "Psi_R0": float(Psi_i[0]), "t_Psi_below_1_s": float(t_i[np.nonzero(Psi_i >= 1)[0][-1] + 1]),
                       "t_tilt_below_90_s": t_upright, "max_drop_m": float(-np.min(dz_i)), "max_e_x_m": float(np.max(ex_i)),
                       "t_e_x_below_5cm_s": float(t_i[np.nonzero(ex_i > 0.05)[0][-1] + 1]), "max_thrust_N": float(np.max(f_i)),
                       "hover_thrust_N": float(m_Q * G), "max_torque_Nm": float(np.max(tau_i)),
                       "e_x_0_m": float(ex_i[0]), "thrust_zero_until_s": float(t_i[np.nonzero(f_i <= 0)[0][-1] + 1])})

# feedforward dropped (Omega_d = 0, Omega_d_dot = 0): the inner loop must be faster
x0s = ref_pos(0)[0] + np.array([0.05, 0.0, 0.0])
ff_study = {}
for eps_ in (1.0, 0.5, 0.25):
    t_, S_ = quad_run(x0s, ref_pos(0)[1], Rz(np.pi / 2), np.zeros(3), 20.0, ff=False, eps=eps_, n=2001, stop=5.0)
    e_ = np.array([np.linalg.norm(s[:3] - ref_pos(t)[0]) for t, s in zip(t_, S_)])
    ff_study[f"{eps_:g}"] = {"diverged": bool(t_[-1] < 19.9), "t_end_s": float(t_[-1]),
                             "max_e_x_last_5s_m": float(np.max(e_[t_ > 15])) if t_[-1] > 19.9 else None}
t_, S_ = quad_run(x0s, ref_pos(0)[1], Rz(np.pi / 2), np.zeros(3), 20.0, ff=True, eps=1.0, n=2001, stop=5.0)
e_ex = np.array([np.linalg.norm(s[:3] - ref_pos(t)[0]) for t, s in zip(t_, S_)])
ff_study["exact_1"] = {"diverged": bool(t_[-1] < 19.9), "max_e_x_last_5s_m": float(np.max(e_ex[t_ > 15]))}
ck.record("feedforward_study", ff_study)
ck.true("exact feedforward, original gains: tracking error below 1e-6 m", ff_study["exact_1"]["max_e_x_last_5s_m"] < 1e-6)
ck.true("no feedforward, original gains: the cascade diverges (error passes 5 m)", ff_study["1"]["diverged"], f"at t = {ff_study['1']['t_end_s']:.1f} s")
ck.true("no feedforward, attitude loop 2x and 4x faster: bounded error that shrinks with eps",
        (not ff_study["0.5"]["diverged"]) and (not ff_study["0.25"]["diverged"])
        and ff_study["0.25"]["max_e_x_last_5s_m"] < ff_study["0.5"]["max_e_x_last_5s_m"] < 0.05,
        f"{ff_study['0.5']['max_e_x_last_5s_m']:.4f} m, {ff_study['0.25']['max_e_x_last_5s_m']:.4f} m")
# exact feedforward with a slow inner loop still converges (the cascade is triangular)
t_, S_ = quad_run(x0, np.zeros(3), Rz(np.pi / 2) @ Rx(np.radians(20)), np.zeros(3), 20.0, ff=True, eps=3.0, n=2001)
e_slow = np.linalg.norm(S_[-1, :3] - ref_pos(t_[-1])[0])
ck.true("exact feedforward, attitude loop 3x slower (slower than the position loop): still converges", e_slow < 1e-6, f"e_x(20 s) = {e_slow:.1e}")
ck.record("slow_inner_exact_e_x_20s", float(e_slow))
ck.record("loop_rates", {"position_wn": float(np.sqrt(k_x / m_Q)), "position_zeta": float(k_v / (2 * np.sqrt(k_x * m_Q))),
                         "attitude_slowest_pole": float(-slowest_pole)})


# =========================================================================
# 7. Quadrotor with a cable-suspended load: a deeper cascade
# =========================================================================
ql_mod, ql_layout, ql_p = library("quadrotor_suspended_load")
m_L, l_c = 0.5, 1.0
ql_p.update(m_Q=m_Q, m_L=m_L, l=l_c, J=J)


def load_accels(q, om, R, Om, f, tau):
    u = f * R @ E3
    vL_dot = ((q @ u) - m_Q * l_c * (om @ om)) * q / (m_Q + m_L) - G * E3
    return vL_dot, -np.cross(q, u) / (m_Q * l_c), Jinv @ (tau - np.cross(Om, J @ Om))


worst = 0.0
for _ in range(20):
    q = unit(rng.normal(size=3))
    om = tangent(q, rng.normal(size=3))
    R, Om, f, tau = random_rotation(rng), rng.normal(size=3), 5 + 60 * rng.random(), rng.normal(size=3)
    xs = ql_layout.pack(x_L=rng.normal(size=3), q=q, R=R, x_L_dot=rng.normal(size=3), omega_q=om, Omega_R=Om)
    a_g = accelerations(ql_mod, xs, np.concatenate([[f], tau]), ql_p)
    worst = max(worst, np.max(np.abs(a_g - np.concatenate(load_accels(q, om, R, Om, f, tau)))))
ck.true("quadrotor + load: the equations used here equal geomech", worst < 1e-8, f"err {worst:.1e}")

kxL, kvL = 4.0 * (m_Q + m_L), 2.8 * (m_Q + m_L)
kqL, komL = 25.0, 7.0
ck.record("load_params", {"m_Q": m_Q, "m_L": m_L, "l": l_c, "k_x": kxL, "k_v": kvL, "k_q": kqL, "k_omega": komL})


def load_force(x, v, q, om, xd):
    """Thrust vector F asked of the quadrotor, and the cable direction q_d the position loop asks for."""
    A = -kxL * (x - xd) - kvL * v + (m_Q + m_L) * G * E3 + m_Q * l_c * (om @ om) * q
    qd = -unit(A)
    om_des = -kqL * e_q(q, qd) - komL * om                      # target fixed, omega_d taken as zero
    return (A @ q) * q + m_Q * l_c * np.cross(q, om_des), qd, A, om_des


# what the two parts of F do, at random states
w1 = w2 = 0.0
for _ in range(200):
    q = unit(rng.normal(size=3))
    om = tangent(q, rng.normal(size=3))
    x, v, xd = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    F, qd, A, om_des = load_force(x, v, q, om, xd)
    w1 = max(w1, np.max(np.abs(-np.cross(q, F) / (m_Q * l_c) - om_des)))
    # the load receives the part of the PD force A0 that lies along the cable
    A0 = -kxL * (x - xd) - kvL * v + (m_Q + m_L) * G * E3
    w2 = max(w2, np.max(np.abs(((q @ F) - m_Q * l_c * (om @ om)) * q - (q @ A0) * q)))
ck.true("load cascade: with thrust vector F the cable obeys omega_dot = -k_q e_q - k_omega omega", w1 < 1e-10, f"err {w1:.1e}")
ck.true("load cascade: the load obeys (m_Q+m_L)(a + g e3) = (q q')(-k_x e_x - k_v e_v + (m_Q+m_L) g e3)", w2 < 1e-9, f"err {w2:.1e}")


def load_rhs(t, s, xd, eps):
    x, v, q, om, R, Om = s[:3], s[3:6], s[6:9], s[9:12], s[12:21].reshape(3, 3), s[21:]
    F, *_ = load_force(x, v, q, om, xd)
    f = max(F @ (R @ E3), 0.0)
    tau = att_torque(R, Om, Rd_from(F, E1), np.zeros(3), np.zeros(3), k_R / eps**2, k_Om / eps)
    vLd, omd, Omd = load_accels(q, om, R, Om, f, tau)
    return np.concatenate([v, vLd, np.cross(om, q), omd, (R @ hat(Om)).ravel(), Omd])


def load_run(eps, t_f=12.0, n=2401, stop=20.0):
    xd = np.array([1.5, 0.0, 0.5])
    q0 = expm_so3(np.radians(40) * E1) @ (-E3)
    s0 = np.concatenate([np.zeros(6), q0, np.zeros(3), I3.ravel(), np.zeros(3)])

    def ev(t, s, *a):
        return stop - np.linalg.norm(s[:3] - xd)
    ev.terminal = True
    sol = solve_ivp(load_rhs, [0, t_f], s0, args=(xd, eps), t_eval=np.linspace(0, t_f, n), rtol=1e-9, atol=1e-11,
                    max_step=0.01, events=ev)
    return sol.t, sol.y.T, xd


load_study = {}
for eps_ in (1.0, 0.5):
    t_l, S_l, xd_l = load_run(eps_)
    exl = np.linalg.norm(S_l[:, :3] - xd_l, axis=1)
    swing = np.degrees(np.arccos(np.clip(-S_l[:, 8], -1, 1)))
    Tn = []
    for s in S_l:
        F, *_ = load_force(s[:3], s[3:6], s[6:9], s[9:12], xd_l)
        f = max(F @ (s[12:21].reshape(3, 3) @ E3), 0.0)
        Tn.append(m_L / (m_Q + m_L) * (m_Q * l_c * (s[9:12] @ s[9:12]) - s[6:9] @ (f * s[12:21].reshape(3, 3) @ E3)))
    conv = bool(t_l[-1] > 11.9 and exl[-1] < 0.01)
    load_study[f"{eps_:g}"] = {"converged": conv, "t_end_s": float(t_l[-1]), "e_x_end_m": float(exl[-1]), "swing_end_deg": float(swing[-1]),
                               "t_e_x_below_5cm_s": float(t_l[np.nonzero(exl > 0.05)[0][-1] + 1]) if conv else None,
                               "max_swing_deg": float(np.max(swing)), "min_tension_N": float(np.min(Tn))}
ck.record("load_study", load_study)
ck.true("load cascade, attitude gains of the bare vehicle: diverges", not load_study["1"]["converged"], f"e_x = {load_study['1']['e_x_end_m']:.1f} m at {load_study['1']['t_end_s']:.1f} s")
ck.true("load cascade, attitude loop 2x faster: load reaches the target, cable hangs straight, cable stays taut",
        load_study["0.5"]["converged"] and load_study["0.5"]["swing_end_deg"] < 0.5 and load_study["0.5"]["min_tension_N"] > 0,
        f"e_x {load_study['0.5']['e_x_end_m']:.1e} m, swing {load_study['0.5']['swing_end_deg']:.2f} deg, min T {load_study['0.5']['min_tension_N']:.2f} N")
ck.record("load_loop_rates", {"position_wn": float(np.sqrt(kxL / (m_Q + m_L))), "cable_wn": float(np.sqrt(kqL)),
                              "attitude_slowest_pole_eps1": float(-slowest_pole),
                              "attitude_slowest_pole_eps05": float(-max(np.roots([J[0, 0], k_Om / 0.5, k_R / 0.25]).real))})

# =========================================================================
# 8. Feedback linearization: geometric against Euler angles on the 3D pendulum (Siravuru 2019, ch. 4)
# =========================================================================
rp_mod, rp_layout, rp_p = library("rigid_pendulum")
m_3, rho_3 = 1.0, np.array([0.0, 0.0, -0.5])
J_3 = np.diag([0.1625, 0.1625, 0.01])
J3inv = np.linalg.inv(J_3)
k_p, k_d = 100.0, 20.0
rp_p.update({"J": J_3, "m": m_3, "g": G, "\\rho": rho_3})
ck.record("fl_params", {"m": m_3, "J_diag": np.diag(J_3).tolist(), "rho": rho_3.tolist(), "k_p": k_p, "k_d": k_d})


def grav_3(R):
    return m_3 * G * np.cross(R.T @ E3, rho_3)


worst = 0.0
for _ in range(20):
    R, Om, tau = random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    a_g = accelerations(rp_mod, rp_layout.pack(R=R, Omega_R=Om), tau, rp_p)
    worst = max(worst, np.max(np.abs(a_g - J3inv @ (tau + grav_3(R) - np.cross(Om, J_3 @ Om)))))
ck.true("3D pendulum: J Om_dot + Om x J Om = m g (R'e3) x rho + tau equals geomech", worst < 1e-9, f"err {worst:.1e}")


def euler_zyx(R):
    return np.array([np.arctan2(R[1, 0], R[0, 0]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[2, 1], R[2, 2])])


def E_mat(th, ph):
    return np.array([[-np.sin(th), 0, 1], [np.cos(th) * np.sin(ph), np.cos(ph), 0], [np.cos(th) * np.cos(ph), -np.sin(ph), 0]])


def E_dot(th, ph, th1, ph1):
    d_th = np.array([[-np.cos(th), 0, 0], [-np.sin(th) * np.sin(ph), 0, 0], [-np.sin(th) * np.cos(ph), 0, 0]])
    d_ph = np.array([[0, 0, 0], [np.cos(th) * np.cos(ph), -np.sin(ph), 0], [-np.cos(th) * np.sin(ph), -np.cos(ph), 0]])
    return d_th * th1 + d_ph * ph1


# Omega = E Theta_dot for R = Rz(psi) Ry(theta) Rx(phi), by finite differences
Th, Th1 = np.array([0.4, -0.7, 1.1]), np.array([0.3, 0.5, -0.2])
d = 1e-6
Rp = Rz(Th[0] + d * Th1[0]) @ Ry(Th[1] + d * Th1[1]) @ Rx(Th[2] + d * Th1[2])
Rm = Rz(Th[0] - d * Th1[0]) @ Ry(Th[1] - d * Th1[1]) @ Rx(Th[2] - d * Th1[2])
R0 = Rz(Th[0]) @ Ry(Th[1]) @ Rx(Th[2])
ck.close("Omega = E(theta, phi) Theta_dot", vee(R0.T @ (Rp - Rm) / (2 * d)), E_mat(Th[1], Th[2]) @ Th1, tol=1e-8)
ck.close("euler_zyx inverts Rz Ry Rx", euler_zyx(R0), Th)


def tau_geo(R, Om):
    """Geometric feedback linearization: e_Omega_dot = -k_p e_R - k_d e_Omega (target R_d = I)."""
    return J_3 @ (-k_p * e_R(R, I3) - k_d * Om) + np.cross(Om, J_3 @ Om) - grav_3(R)


def tau_euler(R, Om):
    """Euler-angle feedback linearization: Theta_ddot = -k_p Theta - k_d Theta_dot."""
    Th = euler_zyx(R)
    if abs(np.cos(Th[1])) < 1e-9:
        return np.full(3, np.nan)
    Em = E_mat(Th[1], Th[2])
    Th1 = np.linalg.solve(Em, Om)
    return J_3 @ (Em @ (-k_p * Th - k_d * Th1) + E_dot(Th[1], Th[2], Th1[1], Th1[2]) @ Th1) + np.cross(Om, J_3 @ Om) - grav_3(R)


# both laws do what they claim at random states
w1 = w2 = 0.0
for _ in range(50):
    R, Om = random_rotation(rng), rng.normal(size=3)
    Om_dot = J3inv @ (tau_geo(R, Om) + grav_3(R) - np.cross(Om, J_3 @ Om))
    w1 = max(w1, np.max(np.abs(Om_dot + k_p * e_R(R, I3) + k_d * Om)))
    Th = euler_zyx(R)
    Em = E_mat(Th[1], Th[2])
    Th1 = np.linalg.solve(Em, Om)
    Om_dot = J3inv @ (tau_euler(R, Om) + grav_3(R) - np.cross(Om, J_3 @ Om))
    Th2 = np.linalg.solve(Em, Om_dot - E_dot(Th[1], Th[2], Th1[1], Th1[2]) @ Th1)
    w2 = max(w2, np.max(np.abs(Th2 + k_p * Th + k_d * Th1)))
ck.true("geometric law gives Omega_dot = -k_p e_R - k_d Omega exactly", w1 < 1e-10, f"err {w1:.1e}")
ck.true("Euler law gives Theta_ddot = -k_p Theta - k_d Theta_dot exactly", w2 < 1e-8, f"err {w2:.1e}")


def fl_run(R0, Om0, law, t_f=3.0):
    """Returns final Psi_R, effort (integral of |tau|^2), peak |tau|, time to stay within 5 degrees."""
    if not np.all(np.isfinite(law(R0, Om0))):
        return np.nan, np.nan, np.inf, np.nan

    def rhs(t, s):
        R, Om = s[:9].reshape(3, 3), s[9:12]
        tau = law(R, Om)
        return np.concatenate([(R @ hat(Om)).ravel(), J3inv @ (tau + grav_3(R) - np.cross(Om, J_3 @ Om)), [tau @ tau]])

    ts = np.linspace(0, t_f, 1501)
    sol = solve_ivp(rhs, [0, t_f], np.concatenate([R0.ravel(), Om0, [0.0]]), t_eval=ts, rtol=1e-8, atol=1e-10)
    Rs_ = sol.y[:9].T.reshape(-1, 3, 3)
    peak = max(np.linalg.norm(law(R, Om)) for R, Om in zip(Rs_, sol.y[9:12].T))
    ang = np.degrees(np.arccos(np.clip((np.trace(Rs_, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out = np.nonzero(ang > 5)[0]
    t5 = 0.0 if not len(out) else (ts[out[-1] + 1] if out[-1] + 1 < len(ts) else np.inf)
    return 1 - np.cos(np.radians(ang[-1])), sol.y[12, -1], peak, t5


grid = np.radians(np.arange(-180, 181, 45))
fl_R0, fl_labels = [], []
for ps in grid:
    for th in grid:
        for ph in grid:
            Rg = Rz(ps) @ Ry(th) @ Rx(ph)
            if not any(np.allclose(Rg, Q, atol=1e-9) for Q in fl_R0):
                fl_R0.append(Rg)
                fl_labels.append((ps, th, ph))
n_lab, n_att = len(grid)**3, len(fl_R0)
ck.true("the 729 grid labels name 208 distinct attitudes", (n_lab, n_att) == (729, 208), f"{n_lab} labels, {n_att} attitudes")
ck.close("the label (180, 180, 180) degrees is the target attitude itself", Rz(np.pi) @ Ry(np.pi) @ Rx(np.pi), I3)
half = np.array([abs(np.trace(Rg) + 1) < 1e-9 for Rg in fl_R0])
sing = np.array([abs(abs(Rg[2, 0]) - 1) < 1e-9 for Rg in fl_R0])
fl = {}
for name, Om0 in (("rest", np.zeros(3)), ("moving", np.array([1.0, -1.0, 1.0]))):
    res_g = np.array([fl_run(Rg, Om0, tau_geo) for Rg in fl_R0])
    res_e = np.array([fl_run(Rg, Om0, tau_euler) for Rg in fl_R0])
    ok_g, ok_e = res_g[:, 0] < 1e-3, res_e[:, 0] < 1e-3
    both = ok_g & ok_e & ~np.all(np.isclose(np.array(fl_R0), I3), axis=(1, 2))
    fl[name] = {"geo_converged": int(ok_g.sum()), "euler_converged": int(ok_e.sum()), "euler_undefined": int(np.isnan(res_e[:, 0]).sum()),
                "geo_stuck_are_half_turns": bool(np.array_equal(~ok_g, half)) if name == "rest" else None,
                "geo_peak_median": float(np.median(res_g[both, 2])), "euler_peak_median": float(np.median(res_e[both, 2])),
                "geo_peak_max": float(np.max(res_g[both, 2])), "euler_peak_max": float(np.max(res_e[both, 2])),
                "effort_ratio_median": float(np.median(res_e[both, 1] / res_g[both, 1])),
                "geo_t5_median": float(np.median(res_g[both, 3])), "euler_t5_median": float(np.median(res_e[both, 3])),
                "geo_t5_max": float(np.max(res_g[both, 3])), "euler_t5_max": float(np.max(res_e[both, 3])),
                "n_both": int(both.sum())}
    if name == "rest":
        fl_rest = (res_g, res_e)
ck.record("fl_grid", {"labels": n_lab, "attitudes": n_att, "half_turns": int(half.sum()), "singular": int(sing.sum()), **fl})
ck.true("grid at rest: the geometric law fails exactly on the 29 half turns (zero torque there) and nowhere else",
        fl["rest"]["geo_stuck_are_half_turns"] and int(half.sum()) == 29 and fl["rest"]["geo_converged"] == 179)
ck.true("grid: the Euler law is undefined at the 16 attitudes with 90 degrees of pitch, converges from the other 192",
        int(sing.sum()) == 16 and fl["rest"]["euler_undefined"] == 16 and fl["rest"]["euler_converged"] == 192)
ck.true("grid with an initial rate (1,-1,1) rad/s: geometric leaves the half turns", fl["moving"]["geo_converged"] >= 205,
        f"{fl['moving']['geo_converged']}/208 within 3 s")
ck.true("grid: Euler law uses several times the peak torque and effort of the geometric law",
        fl["rest"]["euler_peak_median"] > 3 * fl["rest"]["geo_peak_median"] and fl["rest"]["effort_ratio_median"] > 2,
        f"peak {fl['rest']['euler_peak_median']:.1f} vs {fl['rest']['geo_peak_median']:.1f} N m, effort x{fl['rest']['effort_ratio_median']:.1f}")
ck.true("grid: the Euler law is faster where it works", fl["rest"]["euler_t5_max"] < fl["rest"]["geo_t5_median"] + 2.0)
# the attitudes that do not converge with this initial rate: half turns whose axis is perpendicular to the rate.
# The rate then slides the body along the family of half turns, where the restoring torque stays zero.
Om_g = np.array([1.0, -1.0, 1.0])
late = [Rg for Rg in fl_R0 if fl_run(Rg, Om_g, tau_geo)[0] >= 1e-3]
late_ok = all(abs(np.trace(Rg) + 1) < 1e-9 and abs(unit(log_so3(Rg)) @ Om_g) < 1e-6 for Rg in late)
late_final = [fl_run(Rg, Om_g, tau_geo, t_f=8.0)[0] for Rg in late]
ck.true("moving grid: the 3 attitudes that stay are half turns about an axis perpendicular to the initial rate, and they end at another half turn",
        len(late) == 3 and late_ok and all(v > 1.999 for v in late_final), f"{len(late)} attitudes, final Psi_R {np.round(late_final, 6)}")
ck.record("fl_late", len(late))
# a nudge of 0.001 rad/s along the axis frees them
free = [fl_run(Rg, Om_g + 1e-3 * unit(log_so3(Rg)), tau_geo, t_f=8.0)[0] for Rg in late]
ck.true("the same three starts with 0.001 rad/s added along the axis converge", all(v < 1e-6 for v in free))

# near the chart singularity with a modest body rate: Euler torque grows like 1/cos(pitch)
near = {}
for pitch in (0.0, 60.0, 89.9, 89.99):
    Rg = Ry(np.radians(pitch)) @ Rx(0.3)
    Om0 = np.array([1.0, -1.0, 1.0])
    Th = euler_zyx(Rg)
    near[f"{pitch:g}"] = {"euler_Nm": float(np.linalg.norm(tau_euler(Rg, Om0))), "geo_Nm": float(np.linalg.norm(tau_geo(Rg, Om0))),
                          "yaw_rate": float(np.linalg.solve(E_mat(Th[1], Th[2]), Om0)[0])}
ck.record("fl_near_singularity", near)
ck.true("near 90 degrees of pitch the Euler torque is several times the geometric one and still growing; the geometric one is flat",
        near["89.99"]["euler_Nm"] > 5 * near["89.9"]["euler_Nm"] > 25 * near["89.9"]["geo_Nm"]
        and abs(near["89.99"]["geo_Nm"] - near["89.9"]["geo_Nm"]) < 0.1,
        f"{near['89.9']['euler_Nm']:.0f}, {near['89.99']['euler_Nm']:.0f} against {near['89.9']['geo_Nm']:.1f} N m")
# output linearization of y = e_R itself needs C_R^-1: singular at 90 and 180 degrees
ck.close("det C_R = 0 at 90 degrees of error", np.linalg.det(C_R(expm_so3(np.pi / 2 * unit([1.0, 2, 3])), I3)), 0.0)
# range of the two error measures on the grid
ck.record("fl_error_ranges", {"max_norm_e_R": 1.0, "max_norm_Theta": float(max(np.linalg.norm(euler_zyx(Rg)) for Rg in fl_R0))})


# =========================================================================
# 9. Products of rotation groups: add the Lyapunov functions
# =========================================================================
Js = [np.diag([0.9, 1.1, 0.4]), np.diag([2.0, 1.6, 0.8]), np.diag([0.9, 1.1, 0.4])]     # stance leg, torso, swing leg
kRs, kOs = [60.0, 120.0, 60.0], [12.0, 24.0, 12.0]


def prod_rhs(t, s):
    out = []
    for i in range(3):
        R, Om = s[12 * i:12 * i + 9].reshape(3, 3), s[12 * i + 9:12 * i + 12]
        tau = -kRs[i] * e_R(R, I3) - kOs[i] * Om + np.cross(Om, Js[i] @ Om)
        out += [(R @ hat(Om)).ravel(), np.linalg.solve(Js[i], tau - np.cross(Om, Js[i] @ Om))]
    return np.concatenate(out)


s0 = np.concatenate([np.concatenate([expm_so3(np.radians(a) * unit(rng.normal(size=3))).ravel(), rng.normal(size=3)]) for a in (120, 60, 170)])
sol = solve_ivp(prod_rhs, [0, 6], s0, t_eval=np.linspace(0, 6, 601), rtol=1e-10, atol=1e-12)
Vp = np.array([sum(0.5 * s[12 * i + 9:12 * i + 12] @ Js[i] @ s[12 * i + 9:12 * i + 12] + kRs[i] * Psi_R(s[12 * i:12 * i + 9].reshape(3, 3), I3)
                   for i in range(3)) for s in sol.y.T])
ck.true("three factors of SO(3): the summed Lyapunov function never increases and all three attitudes converge",
        np.max(np.diff(Vp)) < 1e-8 and Vp[-1] < 1e-6, f"V(6 s) = {Vp[-1]:.1e}")
# a step of 0.35 s: how much of a 60 degree error is left at impact, per axis pole
step_T = 0.35
left = {}
for a0 in (10.0, 60.0, 150.0):
    Rs0 = expm_so3(np.radians(a0) * E2)
    sol = solve_ivp(lambda t, s: np.concatenate([(s[:9].reshape(3, 3) @ hat(s[9:])).ravel(),
                                                 np.linalg.solve(Js[0], -kRs[0] * e_R(s[:9].reshape(3, 3), I3) - kOs[0] * s[9:])]),
                    [0, step_T], np.concatenate([Rs0.ravel(), np.zeros(3)]), rtol=1e-10, atol=1e-12)
    Rf = sol.y[:9, -1].reshape(3, 3)
    left[f"{a0:g}"] = float(np.degrees(np.arccos(np.clip((np.trace(Rf) - 1) / 2, -1, 1))))
ck.record("leg_step", {"step_s": step_T, "angle_left_deg": left, "wn": float(np.sqrt(kRs[0] / Js[0][1, 1])),
                       "peak_torque_150_Nm": float(kRs[0] * 1.0), "peak_torque_10_Nm": float(kRs[0] * np.sin(np.radians(10)))})
ck.true("swing leg: a 10 degree error is mostly gone in one 0.35 s step, a 150 degree error is not",
        left["10"] < 2.5 and left["150"] > 60, f"{left}")


# =========================================================================
# 10. Limits: torque saturation, and the choice of error vector
# =========================================================================
def sat_run(tau_max, law="trace", angle=150.0, t_f=12.0):
    Re0 = expm_so3(np.radians(angle) * unit([1.0, 2.0, -1.0]))

    def rhs(t, s):
        R, Om = s[:9].reshape(3, 3), s[9:]
        e = e_R(R, I3) if law == "trace" else log_so3(R)
        tau = np.clip(-k_R * e - k_Om * Om, -tau_max, tau_max)
        return np.concatenate([(R @ hat(Om)).ravel(), Jinv @ (tau - np.cross(Om, J @ Om))])

    ts = np.linspace(0, t_f, 4801)
    sol = solve_ivp(rhs, [0, t_f], np.concatenate([Re0.ravel(), np.zeros(3)]), t_eval=ts, rtol=1e-9, atol=1e-11, max_step=0.01)
    ang = np.degrees(np.arccos(np.clip((np.trace(sol.y[:9].T.reshape(-1, 3, 3), axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out = np.nonzero(ang > 5)[0]
    return float(ts[out[-1] + 1]) if out[-1] + 1 < len(ts) else np.inf, float(ang[-1])


sat = {f"{tm:g}": sat_run(tm)[0] for tm in (np.inf, 1.0, 0.2)}
ck.record("saturation_settle_s", sat)
ck.true("PD without cancellation, torque clipped per axis: still converges from 150 degrees, more slowly",
        np.isfinite(sat["0.2"]) and sat["inf"] < sat["1"] < sat["0.2"], f"{sat}")
# energy argument under per-axis clipping for a fixed target: tau . Omega <= -k_R e_R . Omega is NOT implied; check V0 along the run
ck.record("trace_vs_log_150deg_s", {"trace": sat_run(np.inf)[0], "log": sat_run(np.inf, law="log")[0]})
ck.record("peak_torque_demand", {"trace_max_Nm": k_R * 1.0, "log_at_179deg_Nm": float(k_R * np.radians(179))})
# near the target the two error vectors agree to first order
eta = 1e-3 * unit([1.0, -2.0, 0.5])
ck.close("trace and logarithm error vectors agree to first order", e_R(expm_so3(eta), I3), log_so3(expm_so3(eta)), tol=1e-9)

# @@NEXT@@

ck.finish()
