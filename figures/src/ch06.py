"""Figures for Chapter 6.  Run: python figures/src/ch06.py [name ...]

Every curve is a simulated closed-loop trajectory of the controllers in the chapter,
with the same parameters as verify/ch06.py.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, OCHRE, PAPER, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, great_circle, halo, new_plot, phase, save, small_circle, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
EYE = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
G = 9.81

# the vehicle of the chapter (Lee, Leok and McClamroch 2010)
m_Q = 4.34
J = np.diag([0.0820, 0.0845, 0.1377])
Jinv = np.linalg.inv(J)
k_R, k_Om = 8.81, 2.54
k_x, k_v = 16 * m_Q, 5.6 * m_Q
J_min, J_max = J[0, 0], J[2, 2]


def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return EYE + hat(w)
    K = hat(np.asarray(w) / th)
    return EYE + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-9:
        return vee(R - R.T) / 2
    if np.pi - th < 1e-6:
        S = (R + EYE) / 2
        a = np.sqrt(np.clip(np.diag(S), 0, None))
        k = int(np.argmax(a))
        return th * S[:, k] / a[k]
    return th * vee(R - R.T) / (2 * np.sin(th))


def Rx(a):
    return expm_so3(a * E1)


def Ry(a):
    return expm_so3(a * E2)


def Rz(a):
    return expm_so3(a * E3)


def e_R(R, Rd):
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def Psi_R(R, Rd):
    return 0.5 * np.trace(EYE - Rd.T @ R)


def tangent(q, v):
    return v - (v @ q) * q


# ------------------------------------------------------------- simulations
def att_error_run(Re0, eOm0, t_f, n=801, law="trace"):
    """The closed-loop error system: Re_dot = Re hat(e_Omega), J e_Omega_dot = -k_R e - k_Omega e_Omega."""
    def rhs(t, s):
        Re, eO = s[:9].reshape(3, 3), s[9:]
        e = e_R(Re, EYE) if law == "trace" else log_so3(Re)
        return np.concatenate([(Re @ hat(eO)).ravel(), Jinv @ (-k_R * e - k_Om * eO)])

    ts = np.linspace(0, t_f, n)
    sol = solve_ivp(rhs, [0, t_f], np.concatenate([Re0.ravel(), eOm0]), t_eval=ts, rtol=1e-9, atol=1e-11)
    return ts, sol.y[:9].T.reshape(-1, 3, 3), sol.y[9:].T


def angle_of(Rs):
    return np.degrees(np.arccos(np.clip((np.trace(Rs, axis1=1, axis2=2) - 1) / 2, -1, 1)))


# ---------------------------------------------------------- 1. opening picture
def puzzle():
    s = Scene(width=6.6, height=3.3, azim=-58, elev=16)
    # left: the wire
    c0 = -2.55 * s.right
    L = 1.35
    s.curve(np.array([c0 - L * s.right, c0 + L * s.right]), color=INK, lw=1.7, alpha=1.0, zorder=20)
    for x0, y in ((-1.2, 0), (-0.75, 0), (0.55, 0), (1.15, 0)):
        p = c0 + x0 * s.right
        tip = c0 + (x0 - np.sign(x0) * min(0.42, 0.55 * abs(x0))) * s.right
        s.arrow(p + 0.16 * s.up, tip + 0.16 * s.up, color=INDIGO, lw=1.3, head=0.09)
        s.dot(p, color=INDIGO, size=6)
    s.dot(c0, color=INK, size=6.5)
    s.label(c0, "target", dy=-13, size=9.5, **SANS)
    s.text2d(c0 @ s.right, 1.25, "a bead on a wire", fontsize=10.5, **SANS)
    s.text2d(c0 @ s.right, -1.33, "every start arrives", fontsize=10, color=INDIGO, **SANS)

    # right: the ball of rotations, radius pi drawn as 1
    ctr = 0.75 * s.right
    s.shadow(ctr - 1.1 * E3, 1.0, strength=0.16)
    s.sphere(ctr, 1.0, color=PERIWINKLE, alpha=0.5)
    for nrm in (E3, E1):
        s.sphere_curve(ctr + great_circle(nrm), center=ctr, color=INK, lw=0.6)
    rng = np.random.default_rng(3)
    axes = [unit(v) for v in ([1, 0.2, 0.5], [-0.6, 0.9, 0.3], [0.1, -0.8, 0.9], [-0.9, -0.5, -0.2], [0.5, 0.9, -0.6],
                              [0.3, -0.9, -0.5], [-0.3, 0.3, 1.0])]
    for a in axes:
        eO = 2.5 * rng.normal(size=3)
        ts, Rs, _ = att_error_run(expm_so3(np.radians(160) * a), eO, 3.0, n=400)
        pts = np.array([log_so3(R) for R in Rs]) / np.pi
        s.curve(ctr + pts, color=INDIGO, lw=1.2, alpha=0.95, zorder=24)
        s.dot(ctr + pts[0], color=INDIGO, size=4.5, zorder=25)
        k = 55
        s.arrow(ctr + pts[k], ctr + pts[k + 6], color=INDIGO, lw=1.2, head=0.085, zorder=25)
    # a start very close to the boundary: it lingers
    a = unit([0.75, -0.25, -0.62])
    ts, Rs, _ = att_error_run(expm_so3(np.radians(179.95) * a), np.zeros(3), 6.0, n=600)
    pts = np.array([log_so3(R) for R in Rs]) / np.pi
    s.curve(ctr + pts, color=TERRACOTTA, lw=1.5, alpha=1.0, zorder=26)
    s.dot(ctr + pts[0], color=TERRACOTTA, size=6, zorder=27)
    s.label(ctr + pts[0], "half a turn away:\nno torque", dx=12, dy=-12, size=9.5, color=TERRACOTTA, ha="left", **SANS)
    s.pad_to([ctr + 2.35 * s.right])
    s.dot(ctr, color=INK, size=6.5, zorder=28)
    s.label(ctr, "target", dx=20, dy=10, size=9.5, **SANS)
    s.text2d(ctr @ s.right, 1.25, "a rigid body, same law", fontsize=10.5, **SANS)
    s.text2d(ctr @ s.right, -1.33, "every start? Not quite", fontsize=10, color=TERRACOTTA, **SANS)
    s.label(ctr + unit([-0.2, 0.9, 0.75]), r"boundary: $\theta = 180^\circ$", dx=30, dy=14, size=9.5, color=MUTED, **SANS)
    s.save(OUT / "ch06-puzzle.svg", pad=0.06)


# ------------------------------------------------- 2. the Lyapunov function
def c_max(kR, kOm):
    return min(np.sqrt(kR * J_min) / J_max, 4 * kR * kOm / (kOm**2 + 4 * kR * J_max))


def W_mats(c, psi):
    W1 = 0.5 * np.array([[k_R, -c * J_max], [-c * J_max, J_min]])
    W2 = 0.5 * np.array([[2 * k_R / (2 - psi), c * J_max], [c * J_max, J_max]])
    W3 = np.array([[c * k_R, -c * k_Om / 2], [-c * k_Om / 2, k_Om - c * J_max]])
    return W1, W2, W3


def best_rate(psi):
    best = (0.0, 0.0)
    for c in np.linspace(0.002, 0.998, 499) * c_max(k_R, k_Om):
        _, W2, W3 = W_mats(c, psi)
        g = np.linalg.eigvalsh(W3)[0] / np.linalg.eigvalsh(W2)[-1]
        best = max(best, (g, c))
    return best


def lyapunov():
    Re0 = expm_so3(np.radians(150) * unit([1.0, 2.0, -1.0]))
    eO0 = np.array([0.5, -0.3, 0.4])
    ts, Rs, eOs = att_error_run(Re0, eO0, 3.0, n=1201)
    V0 = np.array([0.5 * eO @ J @ eO + k_R * Psi_R(R, EYE) for R, eO in zip(Rs, eOs)])
    psi_max = V0[0] / k_R
    gam, c = best_rate(psi_max)
    V = np.array([0.5 * eO @ J @ eO + k_R * Psi_R(R, EYE) + c * e_R(R, EYE) @ J @ eO for R, eO in zip(Rs, eOs)])

    fig, (ax, bx) = new_plot(width=6.6, height=2.9, ncols=2, gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.34))
    ax.semilogy(ts, V, color=INDIGO, lw=1.7)
    ax.semilogy(ts, V[0] * np.exp(-gam * ts), color=TERRACOTTA, lw=1.4)
    k0 = 600
    slope = np.polyfit(ts[k0:], np.log(V[k0:]), 1)
    ax.semilogy(ts[400:], np.exp(np.polyval(slope, ts[400:])) * 3, color=MUTED, lw=0.9, ls=(0, (3.5, 3)))
    ax.text(1.95, 0.05, f"slope {-slope[0]:.1f} s$^{{-1}}$:\ntwice the slowest pole", fontsize=9, color=MUTED, **SANS)
    ax.text(1.05, 30, f"guaranteed: $V(0)\\,e^{{-\\gamma t}}$,  $\\gamma$ = {gam:.3f} s$^{{-1}}$", fontsize=9.5, color=TERRACOTTA,
            path_effects=halo(), **SANS)
    ax.text(0.5, 0.03, "simulated $V(t)$", fontsize=9.5, color=INDIGO, ha="right", **SANS)
    ax.set_ylim(1e-7, 300)
    ax.set_xlim(0, 3)
    ax.set_xlabel("time (s)", fontsize=10, **SANS)
    ax.set_ylabel("$V$", fontsize=11)

    psis = np.linspace(0.02, 1.995, 80)
    bx.plot(psis, [best_rate(p)[0] for p in psis], color=TERRACOTTA, lw=1.7)
    bx.plot([psi_max], [gam], "o", ms=5, color=TERRACOTTA)
    bx.annotate("the example", (psi_max, gam), xytext=(-8, 22), textcoords="offset points", fontsize=9, ha="right",
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), **SANS)
    bx.axvline(2.0, color=MUTED, lw=0.8, ls=(0, (3.5, 3)))
    bx.text(1.96, 0.42, "half a turn", rotation=90, fontsize=9, color=MUTED, ha="right", va="center", **SANS)
    bx.set_xlim(0, 2.05)
    bx.set_ylim(0, 0.55)
    bx.set_xlabel(r"size of the sublevel set, $\Psi_{\max}$", fontsize=10, **SANS)
    bx.set_ylabel(r"guaranteed rate $\gamma$ (s$^{-1}$)", fontsize=10, **SANS)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.96)
    save(fig, OUT / "ch06-lyapunov.svg")


# ----------------------------------------- 3. settling time against angle
def settle_time(Re0, law, t_f=14.0):
    ts, Rs, _ = att_error_run(Re0, np.zeros(3), t_f, n=5601, law=law)
    ang = angle_of(Rs)
    out = np.nonzero(ang > 5)[0]
    return 0.0 if not len(out) else ts[min(out[-1] + 1, len(ts) - 1)]


def settle():
    rng = np.random.default_rng(6)
    n = 240
    gaps = np.concatenate([180 - rng.uniform(5, 179, n - 60), np.logspace(-3, 0.3, 60)])     # degrees short of 180
    axes = [unit(rng.normal(size=3)) for _ in range(n)]
    t_tr = np.array([settle_time(expm_so3(np.radians(180 - g) * a), "trace") for g, a in zip(gaps, axes)])
    t_lg = np.array([settle_time(expm_so3(np.radians(180 - g) * a), "log") for g, a in zip(gaps, axes)])

    fig, (ax, bx) = new_plot(width=6.6, height=2.9, ncols=2, gridspec_kw=dict(width_ratios=[1, 1], wspace=0.3))
    ang = 180 - gaps
    ax.plot(ang, t_tr, "o", ms=2.6, color=INDIGO, mew=0)
    ax.plot(ang, t_lg, "o", ms=2.6, color=TEAL, mew=0)
    ax.set_xlim(0, 181)
    ax.set_ylim(0, 5)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("initial error angle (degrees)", fontsize=10, **SANS)
    ax.set_ylabel("time to stay inside 5° (s)", fontsize=10, **SANS)
    ax.text(100, 1.75, "trace error", color=INDIGO, fontsize=9.5, **SANS)
    ax.text(118, 0.28, "logarithm error", color=TEAL, fontsize=9.5, **SANS)

    bx.semilogx(gaps, t_tr, "o", ms=2.6, color=INDIGO, mew=0)
    bx.semilogx(gaps, t_lg, "o", ms=2.6, color=TEAL, mew=0)
    lam = (-k_Om + np.sqrt(k_Om**2 + 4 * k_R * J[0, 0])) / (2 * J[0, 0])
    gg = np.array([1e-3, 1.0])
    bx.semilogx(gg, 2.34 - np.log(gg) / lam, color=MUTED, lw=0.9, ls=(0, (3.5, 3)))
    bx.text(150, 4.45, r"0.73 s per factor of ten", fontsize=9, color=MUTED, **SANS)
    bx.set_xlim(200, 5e-4)
    bx.set_ylim(0, 5)
    bx.set_xlabel("degrees short of half a turn", fontsize=10, **SANS)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.96)
    save(fig, OUT / "ch06-settle.svg")


# --------------------------------------------------- 4. the sphere, animated
k_q, k_om = 16.0, 5.6
CONE, W_REF = np.radians(35), 1.2


def ref_s2(t):
    c, s = np.cos(W_REF * t), np.sin(W_REF * t)
    qd = np.array([np.sin(CONE) * c, np.sin(CONE) * s, np.cos(CONE)])
    qd1 = np.sin(CONE) * W_REF * np.array([-s, c, 0])
    qd2 = -np.sin(CONE) * W_REF**2 * np.array([c, s, 0])
    return qd, np.cross(qd, qd1), np.cross(qd, qd2)


def s2_rhs(t, s):
    q, om = s[:3], s[3:]
    qd, omd, omd_dot = ref_s2(t)
    e_q, e_omega = np.cross(qd, q), om - tangent(q, omd)
    # the closed loop of the chapter's law: this is the omega_dot it commands
    return np.concatenate([np.cross(om, q), tangent(q, omd_dot) - (q @ omd) * np.cross(om, q) - k_q * e_q - k_om * e_omega])


def s2_run(angle_deg, side, t_f, n):
    qd0, omd0, _ = ref_s2(0.0)
    q0 = expm_so3(np.radians(angle_deg) * unit(np.cross(qd0, side))) @ qd0
    sol = solve_ivp(s2_rhs, [0, t_f], np.concatenate([q0, tangent(q0, omd0)]), t_eval=np.linspace(0, t_f, n), rtol=1e-10, atol=1e-12)
    return sol.y[:3].T / np.linalg.norm(sol.y[:3].T, axis=1)[:, None]


S2_TF, S2_N = 5.2, 521
_s2_cache = {}


def s2_data():
    if not _s2_cache:
        _s2_cache["runs"] = [(s2_run(60.0, E2, S2_TF, S2_N), TEAL, "60°"),
                             (s2_run(170.0, E1, S2_TF, S2_N), OCHRE, "170°"),
                             (s2_run(179.9, E2, S2_TF, S2_N), TERRACOTTA, "179.9°")]
    return _s2_cache["runs"]


def sphere_frame(t):
    s = Scene(width=6.4, height=3.9, azim=-64, elev=38, xlim=(-2.05, 2.05), ylim=(-1.42, 1.32))
    tt = S2_TF * phase(t, 0.04, 0.96)
    k = int(round(tt / S2_TF * (S2_N - 1)))
    s.shadow((0, 0, -1.12), 1.0, strength=0.16)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6)
    s.sphere_curve(small_circle(E3, CONE), color=INDIGO, lw=0.9)
    s.sphere_curve(small_circle(-E3, CONE), color=MUTED, lw=0.7)
    qd = ref_s2(tt)[0]
    for run, col, name in s2_data():
        trail = run[max(0, k - 150):k + 1]
        if len(trail) > 1:
            s.sphere_curve(1.003 * trail, color=col, lw=1.6, show_hidden=True, zorder=24)
        vis = s.point_visible(run[k])
        s.dot(1.003 * run[k], color=col, size=6.5, hollow=not vis, zorder=40)
    s.dot(1.003 * qd, color=INDIGO, size=7.5, hollow=not s.point_visible(qd), zorder=42)
    s.label(1.003 * qd, r"$q_d$", dx=0, dy=13, size=12, color=INDIGO)
    s.dot(-1.003 * qd, color=MUTED, size=5.5, hollow=True, zorder=38)
    s.label(-1.003 * qd, r"$-q_d$", dx=0, dy=-13, size=11, color=MUTED)
    ang = [np.degrees(np.arccos(np.clip(run[k] @ qd, -1, 1))) for run, _, _ in s2_data()]
    if tt < 0.02:
        msg = "three starts: 60°, 170° and 179.9° from the reference"
    elif ang[2] > 150:
        msg = "two are on their way. The third waits beside the antipode"
    elif ang[2] > 5:
        msg = "the antipode moves on, and the third start falls after the reference"
    else:
        msg = "all three ride with the reference"
    s.text2d(0.0, -1.3, msg, fontsize=10, color=INK, **SANS)
    s.text2d(-1.62, 1.18, f"t = {tt:.1f} s", fontsize=10, color=MUTED, **SANS)
    s.text2d(1.0, 1.18, "error", fontsize=9.5, color=MUTED, **SANS)
    for j, ((_, col, _), a) in enumerate(zip(s2_data(), ang)):
        s.text2d(1.32 + 0.3 * j, 1.18, f"{a:.1f}°" if a > 179.5 else f"{a:.0f}°", fontsize=10, color=col, **SANS)
    return s


def sphere():
    animate(sphere_frame, OUT / "ch06-sphere.gif", OUT / "ch06-sphere-strip.png", n=84, fps=12, width_px=640,
            stills=(0.0, 0.3, 0.62, 1.0), strip_px=760, strip_cols=2)


# ------------------------------------------------- 5. the cascade, animated
def ref_pos(t, w=1.0, r=1.5):
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
    b3 = unit(F)
    b2 = unit(np.cross(b3, hd))
    return np.column_stack([np.cross(b2, b3), b2, b3])


def Rd_derivs(F, F1, F2, hd, hd1, hd2, d=1e-4):
    def g(s):
        return Rd_from(F + s * F1 + 0.5 * s * s * F2, hd + s * hd1 + 0.5 * s * s * hd2)

    R0, Rp, Rm = g(0.0), g(d), g(-d)
    Wd = R0.T @ (Rp - Rm) / (2 * d)
    Omd = vee(0.5 * (Wd - Wd.T))
    A2 = R0.T @ (Rp - 2 * R0 + Rm) / d**2 - hat(Omd) @ hat(Omd)
    return R0, Omd, vee(0.5 * (A2 - A2.T))


def quad_ctrl(t, x, v, R, Om):
    xd, vd, ad, jd, sd, hd, hd1, hd2 = ref_pos(t)
    Fd = -k_x * (x - xd) - k_v * (v - vd) + m_Q * G * E3 + m_Q * ad
    b3 = R @ E3
    f = Fd @ b3
    on = f > 0
    f = f if on else 0.0
    ev1 = f * b3 / m_Q - G * E3 - ad
    F1 = -k_x * (v - vd) - k_v * ev1 + m_Q * jd
    b3_dot = R @ np.cross(Om, E3)
    f1 = (F1 @ b3 + Fd @ b3_dot) if on else 0.0
    ev2 = (f1 * b3 + f * b3_dot) / m_Q - jd
    F2 = -k_x * ev1 - k_v * ev2 + m_Q * sd
    Rd, Omd, Omd_dot = Rd_derivs(Fd, F1, F2, hd, hd1, hd2)
    e_Omega = Om - R.T @ Rd @ Omd
    tau = (-k_R * e_R(R, Rd) - k_Om * e_Omega + np.cross(Om, J @ Om)
           - J @ (np.cross(Om, R.T @ Rd @ Omd) - R.T @ Rd @ Omd_dot))
    return f, tau, Fd


def quad_rhs(t, s):
    x, v, R, Om = s[:3], s[3:6], s[6:15].reshape(3, 3), s[15:18]
    f, tau, _ = quad_ctrl(t, x, v, R, Om)
    return np.concatenate([v, f * R @ E3 / m_Q - G * E3, (R @ hat(Om)).ravel(), Jinv @ (tau - np.cross(Om, J @ Om))])


Q_TF, Q_N = 4.0, 801
_q_cache = {}


def quad_data():
    if not _q_cache:
        x0 = ref_pos(0)[0] + np.array([0.6, -0.4, 0.5])
        s0 = np.concatenate([x0, np.zeros(3), (Rz(np.pi / 2) @ Rx(np.radians(178))).ravel(), np.zeros(3)])
        ts = np.linspace(0, Q_TF, Q_N)
        sol = solve_ivp(quad_rhs, [0, Q_TF], s0, t_eval=ts, rtol=1e-9, atol=1e-11, max_step=0.01)
        _q_cache["t"], _q_cache["S"] = ts, sol.y.T
        _q_cache["F"] = [quad_ctrl(t, s[:3], s[3:6], s[6:15].reshape(3, 3), s[15:18]) for t, s in zip(ts, sol.y.T)]
    return _q_cache["t"], _q_cache["S"], _q_cache["F"]


def draw_vehicle(s, x, R, color=INK, arm=0.26):
    for a in (R[:, 0], R[:, 1]):
        s.curve(np.array([x - arm * a, x + arm * a]), color=color, lw=2.2, alpha=1.0, zorder=34)
        for sg in (-1, 1):
            c = x + sg * arm * a
            th = np.linspace(0, 2 * np.pi, 30)
            ring = c + 0.085 * (np.outer(np.cos(th), R[:, 0]) + np.outer(np.sin(th), R[:, 1])) + 0.02 * R[:, 2]
            s.curve(ring, color=color, lw=0.9, alpha=1.0, zorder=34)


def cascade_frame(t):
    s = Scene(width=6.4, height=3.9, azim=-52, elev=24, xlim=(-2.75, 2.75), ylim=(-1.98, 1.37))
    ts, S, info = quad_data()
    # more time on the first second
    u = phase(t, 0.04, 0.97)
    tt = Q_TF * (0.3 * u + 0.7 * u**2.2)
    k = int(round(tt / Q_TF * (Q_N - 1)))
    ground = -1.25
    th = np.linspace(0, 2 * np.pi, 241)
    circ = np.array([ref_pos(a)[0] for a in th])
    s.curve(np.column_stack([circ[:, 0], circ[:, 1], np.full(len(th), ground)]), color=MUTED, lw=0.6, alpha=0.5)
    s.curve(circ, color=INDIGO, lw=1.0, alpha=0.9, zorder=20)
    xd = ref_pos(tt)[0]
    x, R = S[k, :3], S[k, 6:15].reshape(3, 3)
    f, _, Fd = info[k]
    s.shadow((x[0], x[1], ground), 0.3, strength=0.2)
    s.curve(S[:k + 1, :3], color=TEAL, lw=1.5, alpha=0.95, zorder=26)
    s.dot(xd, color=INDIGO, size=7, zorder=30)
    s.label(xd, r"$x_d$", dx=-12, dy=-12, size=12, color=INDIGO)
    draw_vehicle(s, x, R)
    s.arrow(x, x + 0.5 * R[:, 2], color=INK, lw=1.6, head=0.1, zorder=36)
    s.arrow(x, x + 0.34 * Fd / (m_Q * G), color=TERRACOTTA, lw=1.5, head=0.1, zorder=35)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    err = np.linalg.norm(x - xd)
    if tt < 0.02:
        msg = "released upside down, 0.88 m from the reference"
    elif f <= 0:
        msg = "thrust cut: it falls and turns toward the wanted force"
    elif tilt > 25 and tt < 0.6:
        msg = "thrust back on: it rights itself and brakes the fall"
    elif err > 0.05:
        msg = "position loop closes the gap"
    else:
        msg = "locked on to the reference"
    s.text2d(0.0, -1.86, msg, fontsize=10, color=INK, **SANS)
    s.text2d(-2.2, 1.24, f"t = {tt:.2f} s", fontsize=10, color=MUTED, **SANS)
    s.text2d(1.55, 1.24, f"position error {err:.2f} m   tilt {tilt:.0f}°", fontsize=9.5, color=MUTED, **SANS)
    s.text2d(-2.05, 0.75, r"$F_d$: wanted force", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.text2d(-2.05, 0.97, r"$R\,e_3$: thrust axis", fontsize=9.5, color=INK, **SANS)
    return s


def cascade():
    animate(cascade_frame, OUT / "ch06-cascade.gif", OUT / "ch06-cascade-strip.png", n=90, fps=12, width_px=640,
            stills=(0.0, 0.2, 0.5, 1.0), strip_px=760, strip_cols=2)


# ------------------------------------------ 6. the grid of initial attitudes
def grid():
    s = Scene(width=6.4, height=3.9, azim=-58, elev=18)
    s.shadow((0, 0, -1.12), 1.0, strength=0.16)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE, alpha=0.45)
    for nrm in (E3, E1, E2):
        s.sphere_curve(great_circle(nrm), color=INK, lw=0.55)
    g = np.radians(np.arange(-180, 181, 45))
    Rs = []
    for ps in g:
        for th in g:
            for ph in g:
                R0 = Rz(ps) @ Ry(th) @ Rx(ph)
                if not any(np.allclose(R0, Q, atol=1e-9) for Q in Rs):
                    Rs.append(R0)
    pts = np.array([log_so3(R) for R in Rs]) / np.pi
    order = np.argsort(pts @ s.view)
    n_sing = n_half = 0
    for i in order:
        R0, p = Rs[i], pts[i]
        sing = abs(abs(R0[2, 0]) - 1) < 1e-9
        half = abs(np.trace(R0) + 1) < 1e-9
        n_sing += sing
        n_half += half
        depth = 0.5 + 0.5 * (p @ s.view)
        col = TERRACOTTA if sing else INDIGO
        size = (5.6 if (sing or half) else 3.4) * (0.8 + 0.35 * depth)
        for pp in ([p, -p] if half else [p]):                        # a half turn is one attitude at two antipodal points
            s.dot(pp, color=col, size=size, hollow=half, zorder=30 + 5 * (pp @ s.view))
    s.dot((0, 0, 0), color=INK, size=7, zorder=45)
    s.label((0, 0, 0), "target", dx=-24, dy=-11, size=9.5, **SANS)
    s.text2d(2.05, 0.75, "pitch 90°:\nEuler law undefined", fontsize=9.5, color=TERRACOTTA, ha="left", **SANS)
    s.dot(1.72 * s.right + 0.75 * s.up, color=TERRACOTTA, size=6)
    s.text2d(2.05, 0.2, "half turn at rest:\ngeometric law stays", fontsize=9.5, color=INDIGO, ha="left", **SANS)
    s.dot(1.72 * s.right + 0.2 * s.up, color=INDIGO, size=6, hollow=True)
    s.text2d(2.05, -0.3, "both laws converge", fontsize=9.5, color=INDIGO, ha="left", **SANS)
    s.dot(1.72 * s.right - 0.3 * s.up, color=INDIGO, size=4)
    s.text2d(2.05, -0.8, "hollow and terracotta:\nboth fail", fontsize=9.5, color=MUTED, ha="left", **SANS)
    s.dot(1.72 * s.right - 0.8 * s.up, color=TERRACOTTA, size=6, hollow=True)
    s.pad_to([3.6 * s.right, -1.3 * s.right])
    s.save(OUT / "ch06-grid.svg", pad=0.06)
    print(f"grid: {len(Rs)} attitudes, {n_sing} singular, {n_half} half turns")


FIGS = dict(puzzle=puzzle, lyapunov=lyapunov, settle=settle, sphere=sphere, cascade=cascade, grid=grid)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
