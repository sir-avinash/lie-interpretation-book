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
lam_m, lam_M = J[0, 0], J[2, 2]
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
    M1 = 0.5 * np.array([[kR, -c * lM], [-c * lM, lm]])
    M2 = 0.5 * np.array([[2 * kR / (2 - psi), c * lM], [c * lM, lM]])
    W = np.array([[c * kR, -c * kOm / 2], [-c * kOm / 2, kOm - c * lM]])
    return M1, M2, W


def rate_att(kR, kOm, lm, lM, psi, c=None):
    """Best guaranteed rate beta = lambda_min(W)/lambda_max(M2) over admissible c (or at a given c)."""
    cm = c_max_att(kR, kOm, lm, lM)
    cs = [c] if c is not None else np.linspace(0.002, 0.998, 499) * cm
    best = (0.0, None)
    for c_ in cs:
        M1, M2, W = mats_att(kR, kOm, lm, lM, c_, psi)
        beta = np.linalg.eigvalsh(W)[0] / np.linalg.eigvalsh(M2)[-1]
        if beta > best[0]:
            best = (beta, c_)
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
    M1, M2, W = mats_att(kR, kOm, lm, lM, c, psi)
    V = 0.5 * eO @ J_ @ eO + kR * Psi + c * eR @ J_ @ eO
    viol_lo = max(viol_lo, z @ M1 @ z - V)
    viol_up = max(viol_up, V - z @ M2 @ z)
    C = C_R(R, Rd)
    JeO_dot = -kR * eR - kOm * eO
    Vdot = eO @ JeO_dot + kR * eR @ eO + c * (C @ eO) @ J_ @ eO + c * eR @ JeO_dot
    Vdot_formula = -kOm * eO @ eO + c * eO @ C.T @ J_ @ eO - c * kR * eR @ eR - c * kOm * eR @ eO
    err_dot = max(err_dot, abs(Vdot - Vdot_formula))
    viol_dot = max(viol_dot, Vdot + z @ W @ z)
    n_pd += (np.linalg.eigvalsh(M1)[0] > 0) and (np.linalg.eigvalsh(W)[0] > 0)
ck.true("1/2 |e_R|^2 <= Psi_R <= |e_R|^2/(2 - psi) on Psi_R <= psi < 2 (4000 samples)", viol_sand < 1e-12, f"max viol {viol_sand:.1e}")
ck.true("V >= z'M1 z (4000 random states, inertias, gains)", viol_lo < 1e-10, f"max viol {viol_lo:.1e}")
ck.true("V <= z'M2 z on the sublevel set (4000 random states, inertias, gains)", viol_up < 1e-10, f"max viol {viol_up:.1e}")
ck.true("V_dot = -k_Om|e_Om|^2 + c e_Om.C_R'J e_Om - c k_R|e_R|^2 - c k_Om e_R.e_Om", err_dot < 1e-9, f"err {err_dot:.1e}")
ck.true("V_dot <= -z'W z (4000 random states, inertias, gains)", viol_dot < 1e-10, f"max viol {viol_dot:.1e}")
ck.true("M1 > 0 and W > 0 for every c below the stated bound", n_pd == 4000, f"{n_pd}/4000")

# the bound on c is sharp for W and M1: just above each limit one of them loses definiteness
sharp = 0
for _ in range(500):
    lm = 0.01 + rng.random()
    lM = lm * (1 + 3 * rng.random())
    kR, kOm = 0.1 + 20 * rng.random(), 0.05 + 5 * rng.random()
    c1, c2 = np.sqrt(kR * lm) / lM, 4 * kR * kOm / (kOm**2 + 4 * kR * lM)
    M1a, _, _ = mats_att(kR, kOm, lm, lM, 1.001 * c1, 1.0)
    _, _, Wa = mats_att(kR, kOm, lm, lM, 1.001 * c2, 1.0)
    sharp += (np.linalg.eigvalsh(M1a)[0] < 0) and (np.linalg.eigvalsh(Wa)[0] < 0)
ck.true("M1 > 0 iff c < sqrt(k_R lam_m)/lam_M and W > 0 iff c < 4 k_R k_Om/(k_Om^2 + 4 k_R lam_M)", sharp == 500, f"{sharp}/500")

# V_dot by finite differences along a closed-loop run (the formula describes the real closed loop)
c_demo = 0.5 * c_max_att(k_R, k_Om, lam_m, lam_M)
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
psi_bar = V0s[0] / k_R
ck.true("example starts inside the sublevel set V0 < 2 k_R", psi_bar < 2, f"psi_bar = {psi_bar:.4f}")
ck.true("Psi_R(t) <= V0(0)/k_R for all t", np.max(Psis) <= psi_bar + 1e-9)
beta, c_best = rate_att(k_R, k_Om, lam_m, lam_M, psi_bar)
M1, M2, W = mats_att(k_R, k_Om, lam_m, lam_M, c_best, psi_bar)
Vc = np.array([V_att(*a, c_best) for a in run_mov])
ck.true("example: V(t) <= V(0) exp(-beta t) with the guaranteed rate", np.all(Vc <= Vc[0] * np.exp(-beta * t_a) * (1 + 1e-9) + 1e-12),
        f"beta = {beta:.3f}")
zs = np.array([[np.linalg.norm(e_R(a[0], a[2])), np.linalg.norm(e_Omega(*a))] for a in run_mov])
z2 = np.sum(zs**2, axis=1)
kappa = np.linalg.eigvalsh(M2)[-1] / np.linalg.eigvalsh(M1)[0]
ck.true("example: |z(t)|^2 <= (lam_max(M2)/lam_min(M1)) |z(0)|^2 exp(-beta t)", np.all(z2 <= kappa * z2[0] * np.exp(-beta * t_a) + 1e-12))
# measured decay rate of V late in the run
mask = (t_a > 1.5) & (Vc > 1e-9)
meas = -np.polyfit(t_a[mask], np.log(Vc[mask]), 1)[0]
ck.record("example_att", {"angle0_deg": 150.0, "Psi0": float(Psis[0]), "V0_0": float(V0s[0]), "two_kR": 2 * k_R,
                          "psi_bar": float(psi_bar), "c_max": float(c_max_att(k_R, k_Om, lam_m, lam_M)),
                          "c_best": float(c_best), "beta_guaranteed": float(beta), "beta_measured": float(meas),
                          "overshoot_factor": float(kappa),
                          "c_lim_M1": float(np.sqrt(k_R * lam_m) / lam_M),
                          "c_lim_W": float(4 * k_R * k_Om / (k_Om**2 + 4 * k_R * lam_M))})
ck.true("measured decay of V is twice the slowest pole, and faster than the guarantee",
        abs(meas + 2 * slowest_pole) < 0.3 and meas > beta, f"measured {meas:.2f}, 2|pole| {-2 * slowest_pole:.2f}, bound {beta:.2f}")
# guaranteed rate for a small error (psi -> 0) and how it dies as psi -> 2
ck.record("beta_vs_psi", {f"{p:g}": float(rate_att(k_R, k_Om, lam_m, lam_M, p)[0]) for p in (0.1, 1.0, 1.9, 1.99, 1.999)})
betas = [rate_att(k_R, k_Om, lam_m, lam_M, p)[0] for p in (0.1, 1.0, 1.9, 1.99, 1.999)]
ck.true("the guaranteed rate decreases as the sublevel set grows toward Psi_R = 2", np.all(np.diff(betas) < 0))

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

# @@NEXT@@

ck.finish()
