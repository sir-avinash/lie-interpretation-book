"""Figures for Chapter 3.  Run: python figures/src/ch03.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, OCHRE, PAPER, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, geodesic, great_circle, halo, new_plot, phase, rot, save, small_circle, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")


def spin_arc(s, center, axis, radius, a0, a1, color=INK, lw=1.1, head=0.08):
    """A circular arrow around `axis`, drawn as real 3D geometry."""
    axis = unit(axis)
    a = unit(np.cross(axis, [0.31, 0.52, 0.8]))
    b = np.cross(axis, a)
    t = np.linspace(a0, a1, 60)
    pts = np.asarray(center) + radius * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))
    s.curve(pts[:-3], color=color, lw=lw, alpha=1.0, zorder=31)
    s.arrow(pts[-4], pts[-1], color=color, lw=lw, head=head, zorder=31)


# ------------------------------------------------------------ opening puzzle
def _vec(v):
    """A reading as the text prints it: (0, 1, -1)."""
    return "(" + ",\\,".join(f"{x:.0f}".replace("-", "{-}") for x in np.round(v, 6) + 0.0) + ")"


def frames_frame(t):
    """Two bodies that spin as one. Their gyroscopes disagree; the transported comparison does not."""
    s = Scene(width=6.6, height=3.5, azim=-62, elev=23, xlim=(-3.5, 3.5), ylim=(-2.1, 1.61))
    psi = np.radians(-17) - 8.0 * (1 - t)                       # real time: 1 rad/s for the 8 s of the loop
    Rd = rot(E3, psi)                                          # reference attitude
    R = Rd @ rot(E1, np.pi / 2)                                # body frame, mounted a quarter turn away
    w = E3                                                     # the one angular velocity, in the world frame
    Om_d, Om = Rd.T @ w, R.T @ w                               # what each gyroscope reads
    naive = Om - Om_d
    carried = R.T @ Rd @ Om_d
    cd, c = -1.7 * s.right, 1.7 * s.right
    LEN, WLEN = 0.72, 1.12

    def body(o, Rm, name):
        s.shadow(o - 1.02 * E3, 0.8, strength=0.14)
        s.curve(np.array([o - 0.95 * E3, o + 1.5 * E3]), color=MUTED, lw=0.8, linestyle=(0, (3.5, 3)), alpha=0.9)
        spin_arc(s, o + 1.3 * E3, E3, 0.2, psi - np.radians(280), psi, color=INK)
        s.arrow(o, o + WLEN * w, color=INK, lw=1.5, head=0.1, zorder=29)
        s.frame(o, Rm, length=LEN)
        cols = (TERRACOTTA, TEAL, INDIGO)
        for i in range(3):
            up = abs(Rm[2, i]) > 0.9
            s.label(o + (LEN + 0.13) * Rm[:, i], rf"$b_{i + 1}$", color=cols[i], dx=-10 if up else 0, dy=-6 if up else 8)
        s.dot(o, color=INK, size=4)
        s.text2d(o @ s.right, -1.3, name, fontsize=10, **SANS)

    body(cd, Rd, "reference frame")
    body(c, R, "body frame, mounted a quarter turn away")
    s.text2d(cd @ s.right, -1.58, rf"gyroscope reads $\Omega_d = {_vec(Om_d)}$", fontsize=11.5, color=INDIGO)
    s.text2d(c @ s.right, -1.58, rf"gyroscope reads $\Omega = {_vec(Om)}$", fontsize=11.5, color=TEAL)

    sub = phase(t, 0.30, 0.36)                                 # beat 2: subtract the readings
    out = phase(t, 0.60, 0.64)
    go = ease(phase(t, 0.66, 0.84))                            # beat 3: carry, then subtract
    if t < 0.30:
        msg, col = "the two bodies turn as one, at 1 rad/s about the vertical", MUTED
    elif t < 0.64:
        # the numbers of the reference, read along the body's axes, are a different arrow in space
        a = min(sub, 1 - out)
        ghost = c + WLEN * R @ Om_d
        s.arrow(c, ghost, color=TERRACOTTA, lw=1.2, head=0.09, zorder=31, dashed=True, alpha=a)
        s.arrow(ghost, c + WLEN * R @ Om, color=TERRACOTTA, lw=1.6, head=0.1, zorder=32, alpha=a)
        if a > 0.5:
            s.text2d(0.0, 0.62, rf"the numbers ${_vec(Om_d)}$" + "\nlaid along the body's axes\n(dashed) are another arrow",
                     fontsize=9.5, color=TERRACOTTA, linespacing=1.3, **SANS)
        msg = rf"subtract the readings:  $\Omega - \Omega_d = {_vec(naive)}$,  an error of {np.linalg.norm(naive):.2f} rad/s"
        col = TERRACOTTA
    else:
        o = cd + go * (c - cd)                                 # the world-frame vector R_d Omega_d, slid across
        s.arrow(o, o + WLEN * Rd @ Om_d, color=INDIGO, lw=2.4, head=0.12, zorder=33)
        if go < 1:
            s.label(o + 0.6 * WLEN * E3, r"$R_d\,\Omega_d$", dx=19, size=11.5, color=INDIGO)
            msg = "carry the reference's angular velocity to the body first"
        else:
            s.label(c + 0.75 * WLEN * E3, rf"$R^{{\!\top}}\! R_d\,\Omega_d = {_vec(carried)}$", dx=58, dy=4, size=11.5, color=INDIGO)
            err = np.linalg.norm(Om - carried)
            msg = rf"then subtract:  $e_\Omega = \Omega - R^{{\!\top}}\! R_d\,\Omega_d = {_vec(Om - carried)}$,  an error of {err:.0f} rad/s"
        col = INDIGO
    s.text2d(0.0, -1.93, msg, fontsize=10, color=col, **SANS)
    return s


def frames():
    animate(frames_frame, OUT / "ch03-frames.gif", OUT / "ch03-frames-strip.png", n=112, fps=14,
            width_px=720, stills=(0.215, 0.411, 0.75, 1.0), strip_px=900, strip_cols=2)


# ------------------------------------------------- a variation is a nearby curve
def variation_frame(t):
    s = Scene(width=6.4, height=3.84, azim=-28, elev=22, xlim=(-1.38, 3.5), ylim=(-1.66, 1.268))
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)

    def q_ref(u):                                  # a curve across the face of the sphere
        return unit(s.view + 1.25 * u * s.right + (-0.5 + 0.2 * np.sin(2.2 * u)) * s.up)

    def tangent(u):
        return unit(q_ref(u + 1e-4) - q_ref(u - 1e-4))

    def xi(u):                                     # xi . q = 0, chosen so that delta q pushes up and along
        q = q_ref(u)
        push = np.cross(q, tangent(u)) * (0.66 + 0.26 * np.cos(1.9 * u - 0.5)) + 0.3 * tangent(u)
        return np.cross(q, push)

    def q_eps(u, eps):                             # q_eps = exp(eps hat(xi)) q
        return rot(xi(u), eps * np.linalg.norm(xi(u))) @ q_ref(u)

    us = np.linspace(-0.8, 0.62, 120)
    HI, LO = 0.8, -0.45
    # eps sweeps out, back through zero to the other side, and out again
    eps = HI * ease(phase(t, 0.10, 0.38)) - (HI - LO) * ease(phase(t, 0.42, 0.66)) + (HI - LO) * ease(phase(t, 0.70, 0.92))
    seen_hi = HI * ease(phase(t, 0.10, 0.38))
    seen_lo = (LO - HI) * ease(phase(t, 0.42, 0.66)) + HI if t > 0.42 else 0.0
    for e in (LO, LO / 2, HI / 3, 2 * HI / 3, HI):            # members of the family already visited
        if (0 < e <= seen_hi + 1e-9) or (seen_lo - 1e-9 <= e < 0):
            s.sphere_curve(np.array([q_eps(u, e) for u in us]), color=TERRACOTTA, lw=0.7, alpha=0.45, zorder=16)
    s.sphere_curve(np.array([q_ref(u) for u in us]), color=INDIGO, lw=2.0, zorder=17)
    if abs(eps) > 1e-6:
        s.sphere_curve(np.array([q_eps(u, eps) for u in us]), color=TERRACOTTA, lw=1.7, zorder=18)
    marks = [-0.62, -0.1, 0.42]
    for k, u in enumerate(marks):
        q = q_ref(u)
        dq = np.cross(xi(u), q)
        s.arrow(1.003 * q, 1.003 * q + 0.9 * HI * dq, color=INK, lw=1.4, head=0.09, zorder=32)
        s.dot(1.003 * q, color=INDIGO, size=4.2)
        if abs(eps) > 1e-6:
            s.dot(1.003 * q_eps(u, eps), color=TERRACOTTA, size=4.2, zorder=33)
        if k == 0:
            s.arrow(1.003 * q, 1.003 * q + 0.55 * xi(u), color=TEAL, lw=1.4, head=0.09, zorder=32)
            s.label(q + 0.55 * xi(u), r"$\xi$", dx=-4, dy=-11, color=TEAL, size=12.5)
        if k == 2:
            s.label(q + 0.45 * HI * dq, r"$\mathrm{\delta} q = \xi \times q$", dx=39, dy=0, size=12)
    s.label(q_ref(0.62), "reference curve $q(t)$", dx=18, dy=-19, color=INDIGO, size=10, **SANS)
    if eps > 0.62:
        s.label(q_eps(0.3, eps), r"nearby curve $q_\epsilon(t)$", dx=0, dy=13, color=TERRACOTTA, size=10, **SANS)

    # the same nudge at the middle instant, seen edge on: the circle is the sphere cut by the plane of q and delta q
    u = marks[1]
    nx = np.linalg.norm(xi(u))                                # = |delta q|, because xi is perpendicular to q
    c, r = np.array([2.4, -0.9]), 1.0
    ang = np.linspace(-0.95, 0.95, 80)
    ax = s.ax
    ax.plot(c[0] + r * np.sin(ang), c[1] + r * np.cos(ang), color=INK, lw=1.7, solid_capstyle="round", zorder=20)
    ax.plot([c[0] - 0.62 * r, c[0] + 0.95 * r], [c[1] + r, c[1] + r], color=MUTED, lw=0.8, ls=(0, (3.5, 3)), zorder=14)
    top = c + [0, r]
    on = c + r * np.array([np.sin(eps * nx), np.cos(eps * nx)])   # exp(eps hat(xi)) q stays on the circle
    off = top + [r * eps * nx, 0]                                 # q + eps delta q slides along the tangent line
    tip = top + [0.9 * HI * nx * r, 0]
    ax.annotate("", tip, xytext=top, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=9, shrinkA=0, shrinkB=0),
                zorder=30)
    ax.plot(*top, "o", ms=4.2, color=INDIGO, zorder=33)
    if abs(eps) > 1e-6:
        ax.plot(*np.array([c + r * (off - c) / np.linalg.norm(off - c), off]).T, color=TERRACOTTA, lw=0.8, zorder=13)
        ax.plot(*off, "o", ms=4.6, mfc=PAPER, mec=TERRACOTTA, mew=1.1, zorder=34)
        ax.plot(*on, "o", ms=4.6, color=TERRACOTTA, zorder=34)
    kw = dict(path_effects=halo(), zorder=50, ha="center", va="center")
    ax.text(top[0] - 0.02, top[1] + 0.14, "$q$", color=INDIGO, fontsize=12, **kw)
    if eps > 0.3:
        ax.text(*(on - 0.17 * (on - c)), r"$q_\epsilon$", color=TERRACOTTA, fontsize=12, **kw)
        ax.text(off[0] + 0.03, off[1] + 0.16, r"$q + \epsilon\,\mathrm{\delta} q$", color=TERRACOTTA, fontsize=11.5, **kw)
    ax.text(c[0], c[1] + 0.36, "the sphere, seen edge on\nat the middle instant", color=MUTED, fontsize=9, linespacing=1.25, **SANS, **kw)
    ax.text(c[0], 1.08, f"$\\epsilon = {eps:+.2f}$".replace("-", "{-}").replace("+", "{+}"), fontsize=12, **kw)
    ax.text(c[0], 0.82, rf"$\|q_\epsilon\| = {np.linalg.norm(q_eps(u, eps)):.2f}$", fontsize=11.5, color=TERRACOTTA, **kw)
    ax.text(c[0], 0.57, rf"$\|q + \epsilon\,\mathrm{{\delta}} q\| = {np.linalg.norm(q_ref(u) + eps * np.cross(xi(u), q_ref(u))):.2f}$",
            fontsize=11.5, color=TERRACOTTA, **kw)

    if t < 0.10:
        msg = "a reference curve on the sphere, and at each instant a small rotation axis ξ"
    elif t < 0.42:
        msg = "turn each point a little about its axis: a nearby curve, still on the sphere"
    elif t < 0.92:
        msg = "sweep ε: a family of curves on the sphere. Adding ε δq instead leaves the sphere"
    else:
        msg = "the variation δq is the direction in which the family opens at ε = 0"
    s.text2d(1.02, -1.5, msg, fontsize=10, color=INK, **SANS)
    return s


def variation():
    animate(variation_frame, OUT / "ch03-variation.gif", OUT / "ch03-variation-strip.png", n=108, fps=14,
            width_px=720, stills=(0.05, 0.38, 0.66, 1.0), strip_px=900, strip_cols=2)


# ------------------------------------------------------ heights and slopes
def heights():
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.9, wspace=0.24)
    deg = np.linspace(0, 360, 721)
    th = np.radians(deg)
    wrapped = np.where(th <= np.pi, th, th - 2 * np.pi)
    fold = np.minimum(th, 2 * np.pi - th)             # distance to the target, in [0, pi]

    def broken(ax, y, **kw):
        y = np.array(y, float)
        y[np.isclose(deg, 180)] = np.nan
        left, right = deg < 180, deg > 180
        ax.plot(deg[left], y[left], **kw)
        ax.plot(deg[right], y[right], **kw)

    ax0.plot(deg, 1 - np.cos(th), color=INDIGO, lw=1.8)
    ax0.plot(deg, 0.5 * fold**2, color=TEAL, lw=1.8)
    ax0.plot(deg, 4 * (1 - np.cos(fold / 2)), color=TERRACOTTA, lw=1.8)
    ax0.text(8, 5.05, r"logarithm:  $\frac{1}{2}\theta^2$", color=TEAL, ha="left", fontsize=10.5)
    ax0.text(8, 4.35, r"trace:  $1-\cos\theta$", color=INDIGO, ha="left", fontsize=10.5)
    ax0.text(8, 3.65, "quaternion:", color=TERRACOTTA, ha="left", fontsize=10.5)
    ax0.text(8, 3.0, r"$4(1-\cos\frac{\theta}{2})$", color=TERRACOTTA, ha="left", fontsize=10.5)
    ax0.set_ylim(0, 5.6)
    ax0.set_title("height  Ψ", fontsize=10, loc="left", color=INK, **SANS)

    ax1.axhline(0, color=INK, lw=0.6)
    ax1.plot(deg, np.sin(th), color=INDIGO, lw=1.8)
    broken(ax1, wrapped, color=TEAL, lw=1.8)
    broken(ax1, np.sign(np.cos(th / 2)) * 2 * np.sin(th / 2), color=TERRACOTTA, lw=1.8)
    for y, col in [(np.pi, TEAL), (-np.pi, TEAL), (2, TERRACOTTA), (-2, TERRACOTTA)]:
        ax1.plot([180], [y], "o", ms=4, mfc=PAPER, mec=col, mew=1.0, zorder=5)
    ax1.plot([180], [0], "o", ms=4.5, color=INDIGO, zorder=6)
    ax1.plot([180, 180], [-np.pi, np.pi], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax1.text(8, -1.75, r"logarithm:  $\theta$", color=TEAL, fontsize=10.5, ha="left")
    ax1.text(8, -2.4, r"trace:  $\sin\theta$", color=INDIGO, fontsize=10.5, ha="left")
    ax1.text(8, -3.05, r"quaternion:  $2\sin\frac{\theta}{2}$", color=TERRACOTTA, fontsize=10.5, ha="left")
    ax1.annotate("continuous,\nso it must vanish", (180, 0), xytext=(215, 1.2), fontsize=9, color=INDIGO, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INDIGO, lw=0.7, shrinkB=3), linespacing=1.2, **SANS)
    ax1.text(215, 2.75, "never vanish,\nso they must jump", fontsize=9, color=MUTED, ha="left", va="center", linespacing=1.2, **SANS)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_title("slope  = restoring torque", fontsize=10, loc="left", color=INK, **SANS)
    for ax in (ax0, ax1):
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("error angle θ about a fixed axis (degrees)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch03-heights.svg")


# --------------------------------------------------------------- lingering
KP, KD = 4.0, 2 * np.sqrt(2)
LAWS = {                                   # name: (restoring torque, height, color, label)
    "log": (lambda a: a, lambda a: 0.5 * a**2, TEAL, "logarithm"),
    "quat": (lambda a: 2 * np.sin(a / 2), lambda a: 4 * (1 - np.cos(a / 2)), TERRACOTTA, "quaternion"),
    "trace": (np.sin, lambda a: 1 - np.cos(a), INDIGO, "trace"),
}
_RUNS = {}


def linger_runs():
    """The simulations of the example, integrated and timed exactly as in verify/ch03.py."""
    if not _RUNS:
        for law, th0 in [("log", 179.0), ("quat", 179.0), ("trace", 179.0), ("trace", 179.9)]:
            f = LAWS[law][0]
            sol = solve_ivp(lambda t, y: [y[1], -KP * f(y[0]) - KD * y[1]], [0, 40.0], [np.radians(th0), 0.0],
                            rtol=1e-10, atol=1e-12, dense_output=True)
            tt = np.linspace(0, 40.0, 40001)
            outside = np.nonzero(np.abs(np.degrees(sol.sol(tt)[0])) > 5.0)[0]
            _RUNS[law, th0] = (sol.sol, float(tt[outside[-1] + 1]))
    return _RUNS


def linger_frame(t):
    runs = linger_runs()
    now = 10.0 * t                                                # simulated seconds, at a steady rate
    fig, (ax0, ax1) = new_plot(6.6, 3.3, ncols=2)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.27, top=0.91, wspace=0.2)
    th = np.linspace(0, np.pi, 300)
    for law, (_, height, col, _) in LAWS.items():
        ax0.plot(np.degrees(th), height(th), color=col, lw=1.8, zorder=3)
    ax0.plot([0], [0], "o", ms=4.5, color=INK, zorder=6, clip_on=False)
    ax0.text(7, 0.42, "target", fontsize=9.5, ha="left", **SANS)
    ts = np.linspace(0, max(now, 1e-3), 400)
    for (law, th0), (sol, settle) in runs.items():
        _, height, col, name = LAWS[law]
        far = th0 > 179.5
        a_now = float(sol(now)[0])
        ax0.plot([np.degrees(a_now)], [height(a_now)], "o", ms=7.5, mfc=PAPER if far else col, mec=col, mew=1.3,
                 zorder=8 if far else 9, clip_on=False)
        ax1.plot(ts, np.degrees(sol(ts)[0]), color=col, lw=1.8, ls=(0, (4, 2.5)) if far else "-", zorder=3)
        ax1.plot([now], [np.degrees(a_now)], "o", ms=5.5, mfc=PAPER if far else col, mec=col, mew=1.2, zorder=8, clip_on=False)
    # legend that fills in as each run comes inside 5 degrees for good
    rows = [("log", 179.0, r"logarithm  $\frac{1}{2}\theta^2$"), ("quat", 179.0, r"quaternion  $4(1-\cos\frac{\theta}{2})$"),
            ("trace", 179.0, r"trace  $1-\cos\theta$"), ("trace", 179.9, "trace, from 179.9°")]
    ax0.text(6, 5.25, "inside 5° for good after", fontsize=9, color=MUTED, ha="left", **SANS)
    for k, (law, th0, text) in enumerate(rows):
        col, settle = LAWS[law][2], runs[law, th0][1]
        y = 4.62 - 0.62 * k
        ax0.plot([10], [y + 0.02], "o", ms=5.5, mfc=PAPER if th0 > 179.5 else col, mec=col, mew=1.2)
        ax0.text(18, y, text, fontsize=9.5, color=col, ha="left", va="center", **SANS)
        if now >= settle:
            ax0.text(126, y, f"{settle:.1f} s", fontsize=9.5, color=col, ha="right", va="center", fontweight="semibold", **SANS)
    ax0.set_xlim(-12, 180)
    ax0.set_ylim(0, 5.6)
    ax0.set_xticks([0, 45, 90, 135, 180])
    ax0.set_xlabel("error angle θ (degrees)", fontsize=9.5, **SANS)
    ax0.set_title("height  Ψ", fontsize=10, loc="left", color=INK, **SANS)

    ax1.axhline(0, color=INK, lw=0.6)
    ax1.axhline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax1.text(9.95, 11, "5°", fontsize=8.5, color=MUTED, ha="right", **SANS)
    ax1.text(9.95, 181, f"t = {now:.1f} s", fontsize=10, ha="right", **SANS)
    t_a, t_b = runs["trace", 179.0][1], runs["trace", 179.9][1]
    if now >= t_b:                                                # the gap between the two trace runs, measured at 60 degrees
        grid = np.linspace(0, 10, 10001)
        c_a, c_b = (grid[np.argmax(np.degrees(runs["trace", s0][0](grid)[0]) < 60)] for s0 in (179.0, 179.9))
        ax1.annotate("", (c_b, 60), xytext=(c_a, 60), arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=0.8, mutation_scale=8,
                                                                      shrinkA=1, shrinkB=1))
        ax1.text(c_b + 0.25, 62, f"{c_b - c_a:.1f} s later", color=MUTED, fontsize=9, ha="left", va="center", **SANS)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-12, 195)
    ax1.set_yticks([0, 45, 90, 135, 180])
    ax1.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax1.set_title("error angle θ (degrees)", fontsize=10, loc="left", color=INK, **SANS)

    if now < 0.7:
        msg, col = "three error vectors, the same gains, released at rest 179° from the target", INK
    elif now < 3.0:
        msg, col = "the logarithm and quaternion heights are steep at the top: they leave at once", INK
    elif now < t_a:
        msg, col = "the trace height is flat at the top, so its restoring torque fades there and the body lingers", INDIGO
    elif now < t_b:
        msg, col = "released a tenth of a degree from half a turn (hollow), the trace error lingers longer still", INDIGO
    else:
        msg, col = f"each factor of ten closer to 180° costs the trace error {np.log(10) / ((-KD + np.sqrt(KD**2 + 4 * KP)) / 2):.1f} s more", INDIGO
    fig.text(0.5, 0.045, msg, fontsize=10, color=col, ha="center", va="center", **SANS)
    return fig


def linger():
    animate(linger_frame, OUT / "ch03-linger.gif", OUT / "ch03-linger-strip.png", n=121, fps=13,
            width_px=760, stills=(0.12, 0.36, 0.7, 1.0), strip_px=900, strip_cols=2, hold=1.6)


# ------------------------------------------- carrying an arrow round an octant
def _colat(c, lon):
    c, lon = np.radians(c), np.radians(lon)
    return np.array([np.sin(c) * np.cos(lon), np.sin(c) * np.sin(lon), np.cos(c)])


LOOPS = [[E3, E1, E2],                               # one eighth of the sphere
         [E3, _colat(60, 0), _colat(60, 90)]]        # a smaller triangle with the same first corner


def _loop_legs(P):
    """Great-circle legs of a closed triangle: (start, axis, angle) for each."""
    legs = []
    for k in range(3):
        a, b = P[k], P[(k + 1) % 3]
        legs.append((a, unit(np.cross(a, b)), np.arccos(np.clip(a @ b, -1, 1))))
    return legs


def _carry(P, prog):
    """Parallel transport round the triangle P, as far as `prog` in [0, 3] legs.

    Starts at P[0] with the unit tangent toward P[1]. Along a great circle a vector stays
    parallel when it turns with the circle, about the circle's axis. Returns the position,
    the carried arrow, and the arrows left behind at regular intervals.
    """
    legs = _loop_legs(P)
    v = unit(np.cross(legs[0][1], P[0]))
    trail = []
    for k, (a, axis, ang) in enumerate(legs):
        f = np.clip(prog - k, 0.0, 1.0)
        for g in (0.0, 1 / 3, 2 / 3):
            if (k > 0 or g > 0) and (k < 2 or g < 0.5) and prog - k > g + 0.04:
                trail.append((rot(axis, g * ang) @ a, rot(axis, g * ang) @ v))
        if prog <= k + 1:
            return rot(axis, f * ang) @ a, rot(axis, f * ang) @ v, trail
        v = rot(axis, ang) @ v
    return P[0], v, trail


def _turned(P):
    """Angle by which the arrow comes home rotated, about the outward normal at P[0]."""
    legs = _loop_legs(P)
    v0 = unit(np.cross(legs[0][1], P[0]))
    _, v1, _ = _carry(P, 3.0)
    return float(np.arctan2(P[0] @ np.cross(v0, v1), v0 @ v1))


def _solid_angle(P):
    """Area of a spherical triangle from its corners alone (Van Oosterom and Strackee)."""
    a, b, c = P
    return float(2 * np.arctan2(abs(a @ np.cross(b, c)), 1 + a @ b + b @ c + c @ a))


def globe_frame(t):
    s = Scene(width=6.4, height=3.4, azim=45, elev=44, xlim=(-2.8, 2.8), ylim=(-1.68, 1.3))
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    for lat in np.radians([30, 60]):
        s.sphere_curve(small_circle(E3, np.pi / 2 - lat), color=INK, lw=0.45, alpha=0.35, show_hidden=False)
    for lon in np.radians([0, 30, 60, 90, 120, 150]):
        s.sphere_curve(great_circle([-np.sin(lon), np.cos(lon), 0]), color=INK, lw=0.45, alpha=0.35, show_hidden=False)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6, alpha=0.5)

    # timeline: each leg is walked with a pause at the corners, where the walker turns and the arrow does not
    spans = [[(0.03, 0.17), (0.19, 0.33), (0.35, 0.49)], [(0.60, 0.70), (0.72, 0.82), (0.84, 0.94)]]
    prog = [sum(ease(phase(t, a, b)) for a, b in sp) for sp in spans]
    active = 0 if t < 0.57 else 1
    L = 0.36
    turned = [_turned(P) for P in LOOPS]
    areas = [_solid_angle(P) for P in LOOPS]
    assert np.allclose(turned, areas, atol=1e-9)             # the rotation is the enclosed area

    for i, P in enumerate(LOOPS):
        if i > active:
            continue
        legs = _loop_legs(P)
        edge = np.vstack([np.array([rot(ax, f * ang) @ a for f in np.linspace(0, 1, 40)]) for a, ax, ang in legs])
        done = prog[i] >= 3 - 1e-9
        if done:                                             # shade what the loop encloses
            s.ax.fill(*s.project(edge).T, fc=OCHRE, ec="none", alpha=0.2 if i == 0 else 0.3, zorder=13)
        s.sphere_curve(edge, color=INDIGO, lw=0.8, alpha=0.5, zorder=16)
        for k, (a, ax, ang) in enumerate(legs):
            f = np.clip(prog[i] - k, 0, 1)
            if f > 0:
                s.sphere_curve(np.array([rot(ax, g * ang) @ a for g in np.linspace(0, f, 40)]), color=INDIGO,
                               lw=2.0 if i == active else 1.3, zorder=17)
        pos, v, trail = _carry(P, prog[i])
        if i == active:
            for tp, tv in trail:
                s.arrow(1.004 * tp, 1.004 * tp + L * tv, color=TERRACOTTA, lw=1.0, head=0.08, zorder=32, alpha=0.45)
        if not done:
            s.arrow(1.004 * pos, 1.004 * pos + L * v, color=TERRACOTTA, lw=1.8, head=0.1, zorder=34)
            s.dot(1.004 * pos, color=TERRACOTTA, size=5, zorder=35)
        else:                                                # home again: the arrow, and the angle it turned by
            s.arrow(1.004 * E3, 1.004 * E3 + L * v, color=TERRACOTTA, lw=1.6, head=0.09, zorder=34,
                    alpha=1.0 if i == active else 0.45)
            r = 0.2 if i == 0 else 0.29
            arc = np.array([E3 + r * rot(E3, g) @ E1 for g in np.linspace(0, turned[i], 30)])
            s.curve(1.004 * arc, color=TERRACOTTA, lw=0.9, alpha=1.0, zorder=33)
            if i == active:
                mid = E3 + 0.5 * rot(E3, turned[i] / 2) @ E1 if i == 0 else E3 + (L + 0.13) * v
                s.label(mid, f"{np.degrees(turned[i]):.0f}°", size=9.5, color=TERRACOTTA, **SANS)
    s.arrow(1.004 * E3, 1.004 * E3 + L * E1, color=INK, lw=1.6, head=0.09, zorder=34)
    s.dot(1.004 * E3, color=INK, size=4.5, zorder=35)
    s.label(E3, "pole", dy=13, size=10, color=MUTED, **SANS)
    s.label(E3 + L * E1, "start", dx=-19, dy=-1, size=10, **SANS)

    # the two results, side by side with what produced them
    names = ["one eighth of the sphere", "a smaller triangle"]
    for i in range(2):
        if prog[i] >= 3 - 1e-9:
            y = 0.62 - 0.72 * i
            s.text2d(-1.98, y, f"{names[i]}\nencloses area {areas[i]:.2f}", fontsize=9.5, color=INDIGO, linespacing=1.3, **SANS)
            s.text2d(1.98, y, f"the arrow comes home\nturned by {turned[i]:.2f} rad = {np.degrees(turned[i]):.0f}°",
                     fontsize=9.5, color=TERRACOTTA, linespacing=1.3, **SANS)

    leg = int(min(prog[active], 2.999))
    if active == 0 and prog[0] < 3:
        msg = ["walk from the pole down a meridian: the arrow points straight ahead",
               "a quarter of the way round the equator: the arrow points south, never turned",
               "back up to the pole, the arrow still held parallel"][leg]
    elif active == 0:
        msg = "home again: the arrow has turned a quarter turn, and nobody turned it"
    elif prog[1] < 3:
        msg = "now a smaller triangle, carried the same way"
    else:
        msg = "the arrow turns by exactly the area that the path encloses"
    s.text2d(0.0, -1.5, msg, fontsize=10, color=INK if active == 0 and prog[0] < 3 else TERRACOTTA, **SANS)
    return s


def globe():
    animate(globe_frame, OUT / "ch03-globe.gif", OUT / "ch03-globe-strip.png", n=120, fps=14,
            width_px=720, stills=(0.26, 0.55, 0.78, 1.0), strip_px=900, strip_cols=2)


# -------------------------------------------------- transport map on the sphere
def transport_frame(t):
    s = Scene(width=6.4, height=3.42, azim=-36, elev=22, xlim=(-2.75, 2.75), ylim=(-1.66, 1.28))
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    def facing(tilt, turn):                                      # a point `tilt` degrees away from the line of sight
        tilt, turn = np.radians(tilt), np.radians(turn)
        return np.cos(tilt) * s.view + np.sin(tilt) * (np.cos(turn) * s.right + np.sin(turn) * s.up)

    qd, far = facing(50, 168), facing(50, 302.7)                 # a quarter circle apart, both facing the reader
    along_d = unit(far - (far @ qd) * qd)                        # direction of travel, tangent at q_d
    across = np.cross(qd, along_d)                               # axis of the great circle through q_d and q
    # q slides a quarter circle away from q_d, then comes halfway back
    theta = np.radians(25 + 65 * ease(phase(t, 0.08, 0.48)) - 38 * ease(phase(t, 0.58, 0.78)))
    q = rot(across, theta) @ qd
    along_q = np.cross(across, q)
    arc = geodesic(qd, q)
    s.sphere_curve(great_circle(across), color=INK, lw=0.5, alpha=0.4, show_hidden=False)
    s.sphere_curve(arc, color=TEAL, lw=1.6, zorder=16)

    def plane(p, size):
        al = np.cross(across, p)
        c = 1.002 * p
        s.patch([c + size * (al + across), c + size * (-al + across), c + size * (-al - across), c + size * (al - across)],
                color=SAND, alpha=0.78)

    plane(qd, 0.34)
    plane(q, 0.38)
    A, B = 0.55 * 0.52, 0.83 * 0.52                              # parts of the reference velocity along and across the circle
    qd_dot = A * along_d + B * across
    T = np.cross(np.cross(qd, qd_dot), q)                        # the transport map: (q_d x q_d_dot) x q
    par = rot(across, theta) @ qd_dot                            # parallel transport along the great circle
    ratio = (T @ across) / (par @ across)
    assert abs(ratio - np.cos(theta)) < 1e-12 and abs(T @ along_q - par @ along_q) < 1e-12
    s.arrow(qd, qd + qd_dot, color=INDIGO, zorder=32)
    s.curve(np.array([q + A * along_q, q + par]), color=MUTED, lw=0.7, alpha=0.9, zorder=30)   # the part across the circle
    s.arrow(q, q + par, color=MUTED, lw=1.0, head=0.08, zorder=31, dashed=True)
    s.arrow(q, q + T, color=INDIGO, zorder=33)
    s.dot(qd, color=INK, size=4.5)
    s.dot(q, color=INK, size=4.5)
    s.label(qd, r"$q_d$", dx=-13, dy=-3, size=12)
    s.label(q, r"$q$", dx=-11, dy=-8, size=12)
    s.label(qd + qd_dot, r"$\dot q_d$", dx=9, dy=7, color=INDIGO, size=12)
    s.label(arc[len(arc) // 2], r"$\theta$", dx=-9, dy=-7, color=TEAL, size=12)

    final = t >= 0.82
    if final:                                                    # the actual velocity, and the difference
        q_dot = 0.52 * (-0.62 * along_q + 0.35 * across)
        s.arrow(q, q + q_dot, color=INK, zorder=33)
        s.arrow(q + T, q + q_dot, color=TERRACOTTA, lw=1.4, head=0.09, zorder=34)
        s.label(q + q_dot, r"$\dot q$", dx=-10, dy=5, size=12)
        s.label(q + 0.5 * (T + q_dot), r"$e_{\dot q}$", dx=-2, dy=12, color=TERRACOTTA, size=12.5)

    s.text2d(-1.98, 0.62, f"$\\theta = {np.degrees(theta):.0f}°$", fontsize=12.5, color=TEAL)
    s.text2d(-1.98, 0.14, "along the circle:\nkept whole", fontsize=9.5, color=INDIGO, linespacing=1.3, **SANS)
    s.text2d(-1.98, -0.5, "across the circle:\nscaled by " + rf"$\cos\theta = {abs(ratio) if abs(ratio) < 0.005 else ratio:.2f}$",
             fontsize=9.5, color=INDIGO, linespacing=1.3, **SANS)
    s.text2d(1.98, 0.5, "solid: $\\dot q_d$ carried to $q$\nby the transport map", fontsize=9.5, color=INDIGO, linespacing=1.3, **SANS)
    s.text2d(1.98, -0.14, "dashed: parallel transport\nalong the great circle", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    if t < 0.08:
        msg, col = "the reference velocity lives in the tangent plane at the reference", INK
    elif t < 0.50:
        msg, col = "carried to the tangent plane at q, its sideways part shrinks as q moves away", INDIGO
    elif t < 0.58:
        msg, col = "a quarter circle away the sideways part is gone. Parallel transport (dashed) keeps it", INDIGO
    elif not final:
        msg, col = "wherever q sits, the carried velocity is tangent there", INDIGO
    else:
        msg, col = "now both arrows live in one plane: subtract them to get the velocity error", TERRACOTTA
    s.text2d(0.0, -1.5, msg, fontsize=10, color=col, **SANS)
    return s


def transport():
    animate(transport_frame, OUT / "ch03-transport.gif", OUT / "ch03-transport-strip.png", n=108, fps=14,
            width_px=720, stills=(0.04, 0.3, 0.54, 1.0), strip_px=900, strip_cols=2)


if __name__ == "__main__":
    frames()
    variation()
    heights()
    linger()
    globe()
    transport()
