"""Figures for Chapter 12.  Run: python figures/src/ch12.py [name ...]"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (DASH, INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, axis_through_box, box, caption, geodesic, halo, hat, new_plot, save,
                    style_axes, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
I3 = np.eye(3)
G = 9.81
SANS = dict(fontfamily="Source Sans 3")
PALE_TERRA = "#f3d9cc"
_CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def expm_so3(w):
    th = np.sqrt(w @ w)
    if th < 1e-10:
        return I3 + hat(w)
    K = hat(w / th)
    return I3 + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    th = np.arccos(np.clip(0.5 * (np.trace(R) - 1), -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return v if th < 1e-9 else th / np.sin(th) * v


def dexpinv(th, w):
    t = np.sqrt(th @ th)
    if t < 1e-6:
        return w + 0.5 * cross(th, w) + cross(th, cross(th, w)) / 12
    c = (1 - 0.5 * t / np.tan(0.5 * t)) / t**2
    return w + 0.5 * cross(th, w) + c * cross(th, cross(th, w))


# ------------------------------------------------------------------ planar pendulum
GL = G
TH0, H_P, T_P = np.radians(60.0), 0.05, 10.0


def explicit_euler(th, w, h):
    return th + h * w, w - h * GL * np.sin(th)


def implicit_euler(th, w, h):
    th1 = th
    for _ in range(60):
        th1 -= (th1 - th - h * (w - h * GL * np.sin(th1))) / (1 + h * h * GL * np.cos(th1))
    return th1, w - h * GL * np.sin(th1)


def verlet(th, w, h):
    wh = w - 0.5 * h * GL * np.sin(th)
    th1 = th + h * wh
    return th1, wh - 0.5 * h * GL * np.sin(th1)


def planar_runs():
    out = {}
    n = int(round(T_P / H_P))
    for name, step in (("explicit", explicit_euler), ("implicit", implicit_euler), ("verlet", verlet)):
        z = np.empty((n + 1, 2))
        z[0] = TH0, 0.0
        for k in range(n):
            z[k + 1] = step(z[k, 0], z[k, 1], H_P)
        out[name] = z
    return out


def planar_energy(z):
    return 0.5 * z[..., 1] ** 2 + GL * (1 - np.cos(z[..., 0]))


PLANAR = (("explicit", "explicit Euler", TERRACOTTA), ("implicit", "implicit Euler", TEAL), ("verlet", "Störmer-Verlet", INDIGO))


def phase():
    runs = cached("planar", planar_runs)
    fig, axes = new_plot(6.6, 2.7, ncols=3, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.88, wspace=0.08)
    E0 = planar_energy(np.array([TH0, 0.0]))
    th = np.linspace(-TH0, TH0, 400)
    pt = np.sqrt(np.maximum(2 * (E0 - GL * (1 - np.cos(th))), 0))
    for ax, (key, name, col) in zip(axes, PLANAR):
        ax.plot(np.degrees(np.r_[th, th[::-1]]), np.r_[pt, -pt[::-1]], color=MUTED, lw=0.9, ls=DASH)
        z = runs[key]
        ax.plot(np.degrees(z[:, 0]), z[:, 1], color=col, lw=0.7)
        ax.plot(np.degrees(z[:, 0]), z[:, 1], ".", color=col, ms=2.6)
        ax.plot([60], [0], "o", ms=4.5, color=INK, zorder=5)
        ax.set_xlim(-200, 200)
        ax.set_ylim(-8.5, 8.5)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_title(name, fontsize=10.5, color=col, **SANS)
        ax.set_xlabel("angle θ (degrees)", fontsize=9.5, **SANS)
    axes[0].set_ylabel("momentum p (kg·m²/s)", fontsize=9.5, **SANS)
    axes[0].text(105, 4.6, "over the top\nat 5.4 s", fontsize=9, color=TERRACOTTA, ha="center", va="center", path_effects=halo(),
                 linespacing=1.2, **SANS)
    axes[0].text(0, -7.3, "spirals out", fontsize=9, color=TERRACOTTA, ha="center", path_effects=halo(), **SANS)
    axes[1].text(0, 5.6, "spirals in", fontsize=9, color=TEAL, ha="center", path_effects=halo(), **SANS)
    axes[2].text(0, 5.6, "closed curve", fontsize=9, color=INDIGO, ha="center", path_effects=halo(), **SANS)
    axes[2].annotate("start", (60, 0), xytext=(8, 10), textcoords="offset points", fontsize=9, color=INK, path_effects=halo(), **SANS)
    save(fig, OUT / "ch12-phase.svg")


def three_frame(t):
    runs = cached("planar", planar_runs)
    n = len(runs["verlet"]) - 1
    k = int(round(t * n))
    s = Scene(width=7.0, height=3.9, xlim=(-3.6, 3.6), ylim=(-2.35, 1.66))
    E0 = planar_energy(np.array([TH0, 0.0]))
    for i, (key, name, col) in enumerate(PLANAR):
        cx = -2.3 + 2.3 * i
        z = runs[key]
        th = z[k, 0]
        bob = np.array([cx + np.sin(th), -np.cos(th)])
        arc = np.linspace(-np.pi, np.pi, 200)
        s.ax.plot(cx + np.sin(arc), -np.cos(arc), color=MUTED, lw=0.5, ls=DASH, zorder=2)
        # where the true pendulum turns round
        for sg in (1, -1):
            s.ax.plot([cx + sg * 0.93 * np.sin(TH0), cx + sg * 1.07 * np.sin(TH0)], [-0.93 * np.cos(TH0), -1.07 * np.cos(TH0)],
                      color=INK, lw=0.9, zorder=3)
        # the last second of the bob
        tr = z[max(0, k - 20):k + 1, 0]
        s.ax.plot(cx + np.sin(tr), -np.cos(tr), color=col, lw=1.1, alpha=1.0, zorder=4)
        s.ax.plot([cx, bob[0]], [0, bob[1]], color=INK, lw=1.7, zorder=5, solid_capstyle="round")
        s.ax.plot([bob[0]], [bob[1]], "o", ms=11, color=col, zorder=6, markeredgecolor=INK, markeredgewidth=0.8)
        s.ax.plot([cx], [0], "o", ms=4, color=INK, zorder=7)
        s.ax.text(cx, 1.42, name, fontsize=10.5, color=col, ha="center", va="center", **SANS)
        # energy bar, horizontal, under the pendulum
        x0, w, y0, hh = cx - 0.95, 1.9, -1.72, 0.17
        Emax = 3.0 * E0
        e = min(planar_energy(z[k]), Emax)
        s.ax.fill([x0, x0 + w, x0 + w, x0], [y0, y0, y0 + hh, y0 + hh], color=PAPER, ec=INK, lw=0.7, zorder=3)
        s.ax.fill([x0, x0 + w * e / Emax, x0 + w * e / Emax, x0], [y0, y0, y0 + hh, y0 + hh], color=col, lw=0, zorder=4)
        xt = x0 + w * E0 / Emax
        s.ax.plot([xt, xt], [y0 - 0.07, y0 + hh + 0.07], color=INK, lw=1.2, zorder=5)
        val = planar_energy(z[k])
        s.ax.text(cx, y0 - 0.2, f"energy {val:.1f} J" + (" (off scale)" if val > Emax else ""), fontsize=9, color=col, ha="center", va="center", **SANS)
    s.ax.text(-2.3 - 0.95 + 1.9 / 3, -1.42, "true: 4.9 J", fontsize=8.5, color=INK, ha="center", va="center", **SANS)
    caption(s, f"t = {k * H_P:4.1f} s.  Same pendulum, same step h = 0.05 s, three rules for the step", y=-2.2)
    return s


def three_anim():
    animate(three_frame, OUT / "ch12-three.gif", OUT / "ch12-three-strip.png", n=121, fps=12, width_px=700,
            stills=(0.0, 0.3, 0.6, 1.0), strip_px=900, strip_cols=2, hold=2.0)


# ------------------------------------------------------------------ discrete path of the ball
def path():
    m, tf, N = 1.0, 1.0, 5
    h = tf / N
    t = np.linspace(0, tf, N + 1)
    x = 0.5 * G * t * (tf - t)

    def S(xx):
        return sum(0.5 * m * (xx[k + 1] - xx[k]) ** 2 / h - 0.5 * h * m * G * (xx[k] + xx[k + 1]) for k in range(N))

    fig, axes = new_plot(6.6, 2.9, ncols=2, gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.95, wspace=0.3)
    ax = axes[0]
    tt = np.linspace(0, tf, 200)
    ax.plot(tt, 0.5 * G * tt * (tf - tt), color=MUTED, lw=0.9, ls=DASH)
    for eps in (0.25, -0.25):
        xb = x.copy()
        xb[2] += eps
        ax.plot(t[1:4], xb[1:4], color=TEAL, lw=1.1)
        ax.plot([t[2]], [xb[2]], "o", ms=5, color=PAPER, markeredgecolor=TEAL, markeredgewidth=1.1, zorder=4)
    ax.plot(t, x, color=INDIGO, lw=1.7)
    ax.plot(t, x, "o", ms=5.5, color=INDIGO, zorder=5)
    ax.annotate("", (t[2], x[2] + 0.25), (t[2], x[2] - 0.25), arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.0), zorder=3)
    ax.text(t[2] + 0.035, x[2] + 0.33, "move one knot", fontsize=9, color=TEAL, ha="left", va="center", path_effects=halo(), **SANS)
    ax.text(0.78, 1.02, "parabola", fontsize=9, color=MUTED, ha="left", path_effects=halo(), **SANS)
    for k in range(N + 1):
        ax.text(t[k], x[k] - 0.13 if k in (0, 5) else x[k] + (0.12 if k != 2 else -0.14), f"$x_{k}$", fontsize=10.5, color=INDIGO,
                ha="center", va="center", path_effects=halo())
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.25, 1.75)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{v:.1f}" for v in t])
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("height (m)", fontsize=9.5, **SANS)
    ax = axes[1]
    ee = np.linspace(-0.3, 0.3, 121)
    Sv = np.array([S(x + e * np.eye(N + 1)[2]) for e in ee])
    ax.plot(ee, Sv, color=TEAL, lw=1.7)
    ax.plot([0], [S(x)], "o", ms=5.5, color=INDIGO, zorder=5)
    for e in (-0.1, 0.1):
        ax.plot([e], [S(x) + m * e * e / h], "o", ms=4.5, color=PAPER, markeredgecolor=TEAL, markeredgewidth=1.1, zorder=5)
    ax.annotate("+0.05 J·s at ±10 cm", (0.1, S(x) + 0.05), xytext=(0.0, S(x) + 0.3), fontsize=9, color=TEAL, ha="center",
                path_effects=halo(), arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7), **SANS)
    ax.text(0, S(x) - 0.045, "stationary", fontsize=9, color=INDIGO, ha="center", va="top", path_effects=halo(), **SANS)
    ax.set_xlim(-0.32, 0.32)
    ax.set_ylim(S(x) - 0.12, S(x) + 0.5)
    ax.set_xlabel("displacement of $x_2$ (m)", fontsize=9.5, **SANS)
    ax.set_ylabel("action sum $S_h$ (J·s)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch12-path.svg")


# ------------------------------------------------------------------ rigid body
J_P = np.diag([0.20, 0.25, 0.10])
J_PINV = np.linalg.inv(J_P)
M_P, RHO_P = 1.0, np.array([0.0, 0.0, -0.4])
R0_P, OM0_P = expm_so3(np.array([2.0, 0.0, 0.0])), np.array([0.0, 0.5, 6.0])
E_REST = M_P * G * RHO_P[2]


def torque_pend(R):
    return M_P * G * cross(R.T @ E3, RHO_P)


def energy_pend(R, p):
    return 0.5 * p @ (J_PINV @ p) + M_P * G * E3 @ (R @ RHO_P)


def step_momentum(Om, h, J):
    th = h * np.sqrt(Om @ Om)
    a, b = (1 - th * th / 6, 0.5 - th * th / 24) if th < 1e-4 else (np.sin(th) / th, (1 - np.cos(th)) / th**2)
    JO = J @ Om
    return a * JO + b * h * cross(Om, JO)


def solve_step(pb, h, J):
    Om = np.linalg.solve(J, pb)
    for _ in range(50):
        r = step_momentum(Om, h, J) - pb
        if np.max(np.abs(r)) < 1e-14 * (1 + np.max(np.abs(pb))):
            break
        Om = Om - np.linalg.solve(J + 0.5 * h * (hat(Om) @ J - hat(J @ Om)), r)
    return Om


def lgvi_step(R, p, h, J=J_P, torque=torque_pend):
    pb = p + 0.5 * h * torque(R)
    U = expm_so3(h * solve_step(pb, h, J))
    R1 = R @ U
    return R1, U.T @ pb + 0.5 * h * torque(R1)


def accel_pend(R, Om):
    return J_PINV @ (torque_pend(R) - cross(Om, J_P @ Om))


def rk4_lie(R, Om, h, acc=accel_pend):
    def f(th_, Om_):
        return dexpinv(th_, Om_), acc(R @ expm_so3(th_), Om_)
    z = np.zeros(3)
    k1 = f(z, Om)
    k2 = f(h / 2 * k1[0], Om + h / 2 * k1[1])
    k3 = f(h / 2 * k2[0], Om + h / 2 * k2[1])
    k4 = f(h * k3[0], Om + h * k3[1])
    return R @ expm_so3(h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])), Om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


HOUR = 3600.0
WINDOWS = ((0.0, "the first seconds"), (600.0, "after 10 minutes"), (1800.0, "after 30 minutes"), (3597.0, "after one hour"))
WIN_LEN = 3.0
H_RK, H_VI = 0.08, 0.02


def hour_runs():
    """Both hour-long simulations: energy histories, and the attitudes inside the four windows of the animation."""
    cache = Path(os.environ.get("CH12_CACHE", "/tmp/shots")) / "ch12_hour.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    E0 = energy_pend(R0_P, J_P @ OM0_P)
    out = {}
    for name, h in (("rk", H_RK), ("rk_fine", 0.02), ("vi", H_VI)):
        n = int(round(HOUR / h))
        R, v = R0_P.copy(), (OM0_P.copy() if name != "vi" else J_P @ OM0_P)
        dE = np.empty(n + 1)
        dE[0] = 0.0
        keep = {int(round((t0 + j * h) / h)) for t0, _ in WINDOWS for j in range(int(round(WIN_LEN / h)) + 1)}
        Rw = {0: R.copy()}
        for k in range(1, n + 1):
            if name == "vi":
                R, v = lgvi_step(R, v, h)
                dE[k] = energy_pend(R, v) - E0
            else:
                R, v = rk4_lie(R, v, h)
                dE[k] = energy_pend(R, J_P @ v) - E0
            if k in keep:
                Rw[k] = R.copy()
        out[f"dE_{name}"] = dE
        if name != "rk_fine":
            out[f"R_{name}"] = np.array([Rw[k] for k in sorted(Rw)])
            out[f"k_{name}"] = np.array(sorted(Rw))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def draw_pendulum(s, R, origin, tag_color, sc=0.88):
    dims = sc * np.array([0.50, 0.34, 0.84])
    o = np.asarray(origin, float)
    ctr = o + R @ np.array([0.0, 0.0, -sc])
    cols = {(0, 1): PERIWINKLE, (0, -1): PERIWINKLE, (1, 1): SAND, (1, -1): SAND, (2, 1): PEACH, (2, -1): PEACH}
    rod_z = 40 + (4 if (R @ -E3) @ s.view < 0 else -4)
    s.curve(np.array([o, o + R @ np.array([0, 0, -0.6 * sc])]), color=INK, lw=2.0, alpha=1.0, zorder=rod_z - 0.5)
    box(s, ctr, R, dims, cols, zorder=40)
    axis_through_box(s, ctr, R, dims, R[:, 0], 0.55, tag_color)
    s.dot(o, color=INK, size=8, zorder=60, hollow=True)
    s.dot(o, color=INK, size=3.2, zorder=61)


def hour_frame(t):
    d = cached("hour", hour_runs)
    nwin = len(WINDOWS)
    w = min(int(t * nwin), nwin - 1)
    local = (t * nwin - w) if t < 1 else 1.0
    now = WINDOWS[w][0] + local * WIN_LEN
    s = Scene(width=7.2, height=3.9, azim=-62, elev=13, xlim=(-3.85, 3.85), ylim=(-2.3, 1.87))
    E_start = energy_pend(R0_P, J_P @ OM0_P) - E_REST
    for name, h, off, col, title in (("rk", H_RK, -1.9, TERRACOTTA, "Runge-Kutta, order 4, h = 0.08 s"),
                                     ("vi", H_VI, 2.05, INDIGO, "variational, order 2, h = 0.02 s")):
        k = int(round(now / h))
        idx = int(np.searchsorted(d[f"k_{name}"], k))
        R = d[f"R_{name}"][min(idx, len(d[f"R_{name}"]) - 1)]
        origin = off * s.right + 0.35 * s.up
        draw_pendulum(s, R, origin, col)
        cx = s.project(origin)[0][0]
        s.ax.text(cx, 1.62, title, fontsize=10, color=col, ha="center", va="center", **SANS)
        # energy bar
        e = E_start + d[f"dE_{name}"][min(k, len(d[f"dE_{name}"]) - 1)]
        x0, y0, bw, bh = cx - 1.78, -1.3, 0.2, 2.45
        s.ax.fill([x0, x0 + bw, x0 + bw, x0], [y0, y0, y0 + bh, y0 + bh], color=PAPER, ec=INK, lw=0.7, zorder=3)
        top = y0 + bh * e / (1.1 * E_start)
        s.ax.fill([x0, x0 + bw, x0 + bw, x0], [y0, y0, top, top], color=col, lw=0, zorder=4)
        yt = y0 + bh / 1.1
        s.ax.plot([x0 - 0.07, x0 + bw + 0.07], [yt, yt], color=INK, lw=1.2, zorder=5)
        s.ax.text(x0 + bw / 2, y0 - 0.17, f"{e:.2f} J", fontsize=9.5, color=col, ha="center", va="center", **SANS)
        s.ax.text(x0 + bw / 2 + (0.12 if name == "rk" else 0.0), y0 + bh + 0.15, "tick: 7.39 J", fontsize=8.5, color=INK, ha="center", va="center", **SANS)
    mins = int(now // 60)
    caption(s, f"{WINDOWS[w][1]}:  t = {mins:d} min {now - 60 * mins:4.1f} s.  Bars: energy above rest", y=-2.1)
    return s


def hour_anim():
    animate(hour_frame, OUT / "ch12-hour.gif", OUT / "ch12-hour-strip.png", n=144, fps=12, width_px=720,
            stills=(0.1, 0.36, 0.61, 0.97), strip_px=900, strip_cols=2, hold=2.0)


def energy():
    d = cached("hour", hour_runs)
    fig, ax = new_plot(6.4, 3.0)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.17, top=0.95)

    def envelope(dE, h, nb=160):
        n = len(dE) - 1
        edges = np.unique(np.round(np.logspace(0, np.log10(n), nb)).astype(int))
        tt, env = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            tt.append(b * h)
            env.append(np.abs(dE[a:b + 1]).max())
        return np.array(tt), np.array(env)
    for key, h, col, lab, xy in (("rk", H_RK, TERRACOTTA, "Runge-Kutta, h = 0.08 s", (38, 1.1)),
                                 ("rk_fine", 0.02, MUTED, "Runge-Kutta, h = 0.02 s", (300, 2.2e-4)),
                                 ("vi", H_VI, INDIGO, "variational, h = 0.02 s", (2.2, 0.07))):
        tt, env = envelope(d[f"dE_{key}"], h)
        ax.loglog(tt, env, color=col, lw=1.8 if key != "rk_fine" else 1.3)
        ax.text(*xy, lab, fontsize=9.5, color=col, path_effects=halo(), **SANS)
    ax.axhline(7.388, color=INK, lw=0.6, ls=DASH)
    ax.text(1.2, 9.5, "all the energy there is: 7.39 J", fontsize=9, color=INK, path_effects=halo(), **SANS)
    for x, lab in ((60, "1 min"), (3600, "1 h")):
        ax.axvline(x, color=INK, lw=0.4, alpha=0.5)
        ax.text(x * 0.93, 3e-7, lab, fontsize=8.5, color=INK, ha="right", **SANS)
    ax.set_xlim(1, 3600)
    ax.set_ylim(1e-7, 40)
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("size of the energy error (J)", fontsize=9.5, **SANS)
    ax.minorticks_off()
    save(fig, OUT / "ch12-energy.svg")


# ------------------------------------------------------------------ momentum sphere of the book
J_B = np.array([0.445e-3, 1.0225e-3, 1.4225e-3])


def sphere():
    Jb = np.diag(J_B)
    d0 = unit(np.array([0.50, 0.0, 0.866]))                  # 30 deg from e3 toward e1
    p0 = 0.0128 * d0
    h = 0.02
    zero = lambda R: np.zeros(3)
    R, p = I3.copy(), p0.copy()
    P_vi = [p.copy()]
    for _ in range(400):
        R, p = lgvi_step(R, p, h, J=Jb, torque=zero)
        P_vi.append(p.copy())
    p = p0.copy()
    P_ex = [p.copy()]
    for _ in range(260):
        p = expm_so3(h * p / J_B).T @ p
        P_ex.append(p.copy())
    P_vi, P_ex = np.array(P_vi) / 0.0128, np.array(P_ex) / 0.0128
    # the true curve of constant energy through the start, by a fine variational run
    R, p = I3.copy(), p0.copy()
    P_true = [p.copy()]
    for _ in range(4000):
        R, p = lgvi_step(R, p, 0.0005, J=Jb, torque=zero)
        P_true.append(p.copy())
    P_true = np.array(P_true) / 0.0128
    s = Scene(width=5.4, height=3.7, azim=-38, elev=28)
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(P_true, color=INDIGO, lw=1.0, show_hidden=False)
    s.sphere_curve(P_ex, color=TERRACOTTA, lw=0.8, show_hidden=False)
    for P, col, ms in ((P_ex, TERRACOTTA, 3.0), (P_vi[:46], INDIGO, 3.2)):
        for q in P:
            if s.point_visible(q):
                s.dot(q, color=col, size=ms, zorder=32)
    s.dot(d0, color=INK, size=4.6, zorder=40)
    s.label(d0, "start", dx=-20, dy=4, size=9.5, **SANS)
    for axis, name, dx, dy in ((E1, r"$e_1$", 10, -6), (E3, r"$e_3$", 0, 12), (E2, r"$e_2$", 12, 0)):
        if s.point_visible(axis):
            s.dot(axis, color=INK, size=3.6, hollow=True)
            s.label(axis, name, dx=dx, dy=dy, size=12)
    s.text2d(1.75, 0.75, "variational:\non one level curve", fontsize=9.5, color=INDIGO, ha="center", linespacing=1.25, **SANS)
    s.text2d(1.8, -0.35, "explicit exponential\nupdate: crosses\nthe level curves", fontsize=9.5, color=TERRACOTTA, ha="center",
             linespacing=1.25, **SANS)
    s.text2d(0.0, -1.38, "body angular momentum $p_k$ on its sphere,  $J_1 < J_2 < J_3$", fontsize=9.5, color=MUTED, **SANS)
    s._lims = ((-1.35, 2.65), (-1.55, 1.25))
    s.save(OUT / "ch12-sphere.svg")
    return P_ex


# ------------------------------------------------------------------ collocation
def colloc():
    f = lambda x: x * (1 - x)
    sol = lambda t, x0: 1 / (1 + (1 / x0 - 1) * np.exp(-t))
    h = 2.0
    fig, axes = new_plot(6.6, 2.8, ncols=2, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.9, wspace=0.08)
    for ax, xb, title in ((axes[0], 0.72, "knots that violate the model"), (axes[1], float(sol(h, 0.15)), "knots on a solution")):
        xa = 0.15
        fa, fb = f(xa), f(xb)
        c = np.linalg.solve(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, h, h**2, h**3], [0, 1, 2 * h, 3 * h**2]]), [xa, fa, xb, fb])
        tt = np.linspace(0, h, 100)
        ax.plot(tt, sol(tt, 0.15), color=MUTED, lw=0.9, ls=DASH)
        ax.plot(tt, np.polyval(c[::-1], tt), color=INDIGO, lw=1.7)
        xm = np.polyval(c[::-1], h / 2)
        sm_poly, sm_model = np.polyval(np.polyder(c[::-1]), h / 2), f(xm)
        for t0, x0, sl, col in ((0, xa, fa, INK), (h, xb, fb, INK), (h / 2, xm, sm_model, TERRACOTTA), (h / 2, xm, sm_poly, INDIGO)):
            dt = 0.32
            ax.plot([t0 - dt, t0 + dt], [x0 - dt * sl, x0 + dt * sl], color=col, lw=1.3 if col != INDIGO else 1.0, zorder=4)
        ax.plot([0, h], [xa, xb], "o", ms=5.5, color=INK, zorder=5)
        ax.plot([h / 2], [xm], "o", ms=5, color=PAPER, markeredgecolor=INDIGO, markeredgewidth=1.2, zorder=6)
        ax.set_title(title, fontsize=10, color=INK, **SANS)
        ax.set_xticks([0, h / 2, h])
        ax.set_xticklabels(["$t_k$", "midpoint", "$t_{k+1}$"])
        ax.set_xlim(-0.45, h + 0.45)
        ax.set_ylim(0.0, 0.9)
    axes[0].set_ylabel("state x", fontsize=9.5, **SANS)
    axes[0].text(1.42, 0.3, "slope the\nmodel asks for", fontsize=9, color=TERRACOTTA, ha="left", va="center", path_effects=halo(), linespacing=1.2, **SANS)
    axes[0].text(0.7, 0.5, "slope of\nthe cubic", fontsize=9, color=INDIGO, ha="right", va="center", path_effects=halo(), linespacing=1.2, **SANS)
    axes[0].text(0.12, 0.06, "model slope at the knot", fontsize=9, color=INK, ha="left", va="center", path_effects=halo(), linespacing=1.2, **SANS)
    axes[1].text(1.2, 0.2, "the two slopes agree:\nzero defect", fontsize=9, color=INDIGO, ha="left", va="center", path_effects=halo(), linespacing=1.2, **SANS)
    axes[1].text(-0.3, 0.78, "dashed: true solution", fontsize=9, color=MUTED, ha="left", path_effects=halo(), **SANS)
    save(fig, OUT / "ch12-colloc.svg")


def chord():
    s = Scene(width=5.2, height=3.6, azim=45, elev=52)
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    qa, qb = E1, E2
    mid = 0.5 * (qa + qb)
    s.sphere_curve(geodesic(qa, qb, 80), color=INDIGO, lw=1.8)
    s.curve(np.array([qa, qb]), color=TERRACOTTA, lw=1.5, alpha=1.0, zorder=26)
    s.curve(np.array([[0, 0, 0], mid]), color=TERRACOTTA, lw=0.9, alpha=1.0, zorder=25, linestyle=DASH)
    s.curve(np.array([mid, unit(mid)]), color=INK, lw=0.7, alpha=1.0, zorder=25, linestyle=DASH)
    s.dot(qa, color=INK, size=4.8)
    s.dot(qb, color=INK, size=4.8)
    s.dot(mid, color=TERRACOTTA, size=4.8, zorder=40)
    s.dot(unit(mid), color=INDIGO, size=4.8, zorder=40)
    s.dot((0, 0, 0), color=INK, size=2.5, zorder=24)
    s.label(qa, r"$q_k$", dx=-13, dy=-6, size=12)
    s.label(qb, r"$q_{k+1}$", dx=19, dy=-6, size=12)
    s.label(mid, "entrywise midpoint:\nlength 0.707", dx=0, dy=30, size=9.5, color=TERRACOTTA, linespacing=1.25, **SANS)
    s.label(unit(mid), "midpoint of the arc:\nlength 1", dx=0, dy=-24, size=9.5, color=INDIGO, linespacing=1.25, **SANS)
    s.label((0, 0, 0), "center", dx=0, dy=10, size=9, color=INK, **SANS)
    s._lims = ((-2.3, 2.3), (-1.4, 1.2))
    s.save(OUT / "ch12-chord.svg")


# ------------------------------------------------------------------ the optimization
def opt():
    d = np.load(ROOT / "verify" / "ch12_opt.npz")
    N, h = 20, 0.1
    t = np.arange(N + 1) * h
    Rf = expm_so3(np.radians(150.0) * np.ones(3) / np.sqrt(3))

    def replay(tau, n_sub=40):
        R, Om = I3.copy(), np.zeros(3)
        tt, err = [0.0], [np.degrees(np.linalg.norm(log_so3(Rf.T @ R)))]
        hs = h / n_sub
        for k in range(N):
            for j in range(n_sub):
                def f(th_, Om_, sfrac):
                    tq = tau[k] + (tau[k + 1] - tau[k]) * (j + sfrac) / n_sub
                    return dexpinv(th_, Om_), J_PINV @ (tq - cross(Om_, J_P @ Om_))
                z = np.zeros(3)
                k1 = f(z, Om, 0.0)
                k2 = f(hs / 2 * k1[0], Om + hs / 2 * k1[1], 0.5)
                k3 = f(hs / 2 * k2[0], Om + hs / 2 * k2[1], 0.5)
                k4 = f(hs * k3[0], Om + hs * k3[1], 1.0)
                R = R @ expm_so3(hs / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
                Om = Om + hs / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            tt.append((k + 1) * h)
            err.append(R.copy())
        return err[1:]

    fig, axes = new_plot(6.6, 2.9, ncols=2, gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.94, wspace=0.27)
    ax = axes[0]
    ax.axhspan(0.5, 0.62, color=PALE_TERRA, lw=0)
    ax.axhspan(-0.62, -0.5, color=PALE_TERRA, lw=0)
    for y in (0.5, -0.5):
        ax.axhline(y, color=INK, lw=0.9)
    for key, col, mk in (("chart", TEAL, "s"), ("variational", INDIGO, "o")):
        tau = d[f"tau_{key}"]
        for i in range(3):
            ax.plot(t, tau[:, i], color=col, lw=1.3, marker=mk, ms=2.8)
    tv = d["tau_variational"]
    order = np.argsort(tv[2])
    for rank, i in enumerate(order):
        kx, dx, dy = ((2, 0.0, -0.065), (2, -0.06, -0.075), (3, 0.09, 0.05))[rank]
        ax.text(kx * h + dx, tv[kx, i] + dy, rf"$\tau_{i + 1}$", fontsize=10.5, color=INK, ha="center", va="center", path_effects=halo())
    ax.text(1.3, 0.3, "squares: chart\ndots: variational", fontsize=9, color=INK, ha="left", va="center", linespacing=1.3, **SANS)
    ax.text(1.0, 0.555, r"bound $\tau_{\max}$", fontsize=9, color=INK, ha="center", va="center", **SANS)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.62, 0.62)
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("torque (N·m)", fontsize=9.5, **SANS)
    ax = axes[1]
    for key, col, mk, lab, yoff in (("chart", TEAL, "s", "chart, exact kinematics", 0.12), ("variational", INDIGO, "o", "variational", 0.1)):
        Rr = replay(d[f"tau_{key}"])
        Rk = d[f"R_{key}"]
        e = [0.0] + [np.degrees(np.linalg.norm(log_so3(Rk[k + 1].T @ Rr[k]))) for k in range(N)]
        ax.plot(t, e, color=col, lw=1.5, marker=mk, ms=3)
        ax.text(1.95 if key == "chart" else 1.55, e[-1] + yoff, lab, fontsize=9.5, color=col, ha="right", va="center", path_effects=halo(), **SANS)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.7)
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("replay minus plan, attitude (degrees)", fontsize=9, **SANS)
    save(fig, OUT / "ch12-opt.svg")


ALL = dict(phase=phase, three=three_anim, path=path, hour=hour_anim, energy=energy, sphere=sphere, colloc=colloc, chord=chord, opt=opt)

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
