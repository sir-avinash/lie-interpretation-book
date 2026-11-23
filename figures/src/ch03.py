"""Figures for Chapter 3.  Run: python figures/src/ch03.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, OCHRE, PAPER, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, geodesic, great_circle, halo, new_plot, rot, save, small_circle, unit)

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
def frames():
    s = Scene(width=6.6, height=3.0, azim=-62, elev=17)
    Roff = rot(E1, np.pi / 2)
    yaw = rot(E3, np.radians(48))

    def body(offset, R, name, reading, col):
        c = offset * s.right
        s.shadow(c - 1.02 * E3, 0.8, strength=0.14)
        s.curve(np.array([c - 0.95 * E3, c + 1.25 * E3]), color=MUTED, lw=0.8, linestyle=(0, (3.5, 3)), alpha=0.9)
        spin_arc(s, c + 1.08 * E3, E3, 0.2, np.radians(-30), np.radians(250), color=INK)
        s.frame(c, R, length=0.78, labels=[r"$b_1$", r"$b_2$", r"$b_3$"])
        s.dot(c, color=INK, size=4)
        s.label(c - 1.32 * E3, name, size=10.5, **SANS)
        s.label(c - 1.62 * E3, reading, size=11.5, color=col)
        return c

    c = body(-1.7, yaw, "reference frame", r"gyro reads $\Omega_d = (0,\,0,\,1)$", INDIGO)
    s.label(c + 1.45 * E3, "both spin at 1 rad/s about the vertical", dx=118, size=10, color=MUTED, **SANS)
    body(1.7, yaw @ Roff, "body frame, mounted a quarter turn away", r"gyro reads $\Omega = (0,\,1,\,0)$", TEAL)
    s.save(OUT / "ch03-frames.svg", pad=0.08)


# ------------------------------------------------- a variation is a nearby curve
def variation():
    s = Scene(width=6.4, height=3.6, azim=-28, elev=22)
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)

    def q_ref(t):                                  # a curve across the face of the sphere
        return unit(s.view + 1.25 * t * s.right + (-0.5 + 0.2 * np.sin(2.2 * t)) * s.up)

    def tangent(t):
        return unit(q_ref(t + 1e-4) - q_ref(t - 1e-4))

    def xi(t):                                     # xi . q = 0, chosen so that delta q pushes up and along
        q = q_ref(t)
        push = np.cross(q, tangent(t)) * (0.66 + 0.26 * np.cos(1.9 * t - 0.5)) + 0.3 * tangent(t)
        return np.cross(q, push)

    def q_eps(t, eps):
        return rot(xi(t), eps * np.linalg.norm(xi(t))) @ q_ref(t)

    ts = np.linspace(-0.8, 0.62, 160)
    EPS = 0.8
    for eps, lw, al in [(EPS / 3, 0.7, 0.5), (2 * EPS / 3, 0.7, 0.5), (EPS, 1.6, 1.0)]:
        s.sphere_curve(np.array([q_eps(t, eps) for t in ts]), color=TERRACOTTA, lw=lw, alpha=al, zorder=16)
    s.sphere_curve(np.array([q_ref(t) for t in ts]), color=INDIGO, lw=2.0, zorder=17)
    for k, t in enumerate([-0.62, -0.1, 0.42]):
        q = q_ref(t)
        dq = np.cross(xi(t), q)
        s.arrow(1.003 * q, 1.003 * q + 0.9 * EPS * dq, color=INK, lw=1.4, head=0.09, zorder=32)
        s.dot(1.003 * q, color=INDIGO, size=4.2)
        if k == 0:
            s.arrow(1.003 * q, 1.003 * q + 0.55 * xi(t), color=TEAL, lw=1.4, head=0.09, zorder=32)
            s.label(q + 0.55 * xi(t), r"$\xi$", dx=-4, dy=-11, color=TEAL, size=12.5)
        if k == 1:
            s.label(q, r"$q(t)$", dx=6, dy=-13, color=INDIGO, size=12)
        if k == 2:
            s.label(q + 0.45 * EPS * dq, r"$\delta q = \xi \times q$", dx=39, dy=0, size=12)
    s.label(q_ref(0.62), "reference curve", dx=14, dy=-16, color=INDIGO, size=10, **SANS)
    s.label(q_eps(0.3, EPS), r"nearby curve $q_\epsilon(t)$", dx=0, dy=13, color=TERRACOTTA, size=10.5, **SANS)
    s.text2d(-1.95, 0.55, "a variation is the\nfirst-order gap between\na curve and its neighbors",
             fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.text2d(1.95, -0.1, r"$\xi$ is the axis of the" + "\nsmall rotation that\ncarries $q$ to $q_\\epsilon$",
             fontsize=9.5, color=TEAL, linespacing=1.3, **SANS)
    s.save(OUT / "ch03-variation.svg", pad=0.1)


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
    ax0.plot(deg, 0.5 * fold**2, color=TERRACOTTA, lw=1.8)
    ax0.plot(deg, 4 * (1 - np.cos(fold / 2)), color=TEAL, lw=1.8)
    ax0.text(8, 5.05, r"log:  $\frac{1}{2}\theta^2$", color=TERRACOTTA, ha="left", fontsize=10.5)
    ax0.text(8, 4.35, r"trace:  $1-\cos\theta$", color=INDIGO, ha="left", fontsize=10.5)
    ax0.text(8, 3.65, "quaternion:", color=TEAL, ha="left", fontsize=10.5)
    ax0.text(8, 3.0, r"$4(1-\cos\frac{\theta}{2})$", color=TEAL, ha="left", fontsize=10.5)
    ax0.set_ylim(0, 5.6)
    ax0.set_title("height  Ψ", fontsize=10, loc="left", color=INK, **SANS)

    ax1.axhline(0, color=INK, lw=0.6)
    ax1.plot(deg, np.sin(th), color=INDIGO, lw=1.8)
    broken(ax1, wrapped, color=TERRACOTTA, lw=1.8)
    broken(ax1, np.sign(np.cos(th / 2)) * 2 * np.sin(th / 2), color=TEAL, lw=1.8)
    for y, col in [(np.pi, TERRACOTTA), (-np.pi, TERRACOTTA), (2, TEAL), (-2, TEAL)]:
        ax1.plot([180], [y], "o", ms=4, mfc=PAPER, mec=col, mew=1.0, zorder=5)
    ax1.plot([180], [0], "o", ms=4.5, color=INDIGO, zorder=6)
    ax1.plot([180, 180], [-np.pi, np.pi], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax1.text(8, -1.75, r"log:  $\theta$", color=TERRACOTTA, fontsize=10.5, ha="left")
    ax1.text(8, -2.4, r"trace:  $\sin\theta$", color=INDIGO, fontsize=10.5, ha="left")
    ax1.text(8, -3.05, r"quaternion:  $2\sin\frac{\theta}{2}$", color=TEAL, fontsize=10.5, ha="left")
    ax1.annotate("continuous,\nso it must vanish", (180, 0), xytext=(215, 1.2), fontsize=9, color=INDIGO, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INDIGO, lw=0.7, shrinkB=3), linespacing=1.2, **SANS)
    ax1.text(215, 2.75, "never vanish,\nso they must jump", fontsize=9, color=MUTED, ha="left", va="center", linespacing=1.2, **SANS)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_title("slope  = restoring action", fontsize=10, loc="left", color=INK, **SANS)
    for ax in (ax0, ax1):
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("error angle θ about a fixed axis (degrees)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch03-heights.svg")


# --------------------------------------------------------------- lingering
def linger():
    fig, ax = new_plot(6.4, 2.8)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.95)
    kp, kd = 4.0, 2 * np.sqrt(2)
    laws = [("trace", np.sin, INDIGO), ("log", lambda a: a, TERRACOTTA), ("quat", lambda a: 2 * np.sin(a / 2), TEAL)]
    tt = np.linspace(0, 10, 1500)
    for name, f, col in laws:
        for th0, ls in [(179.0, "-"), (179.9, (0, (4, 2.5)))]:
            if name != "trace" and th0 > 179.5:
                continue
            sol = solve_ivp(lambda t, y: [y[1], -kp * f(y[0]) - kd * y[1]], [0, 10], [np.radians(th0), 0.0],
                            t_eval=tt, rtol=1e-10, atol=1e-12)
            ax.plot(tt, np.degrees(sol.y[0]), color=col, lw=1.8, ls=ls)
    ax.axhline(0, color=INK, lw=0.6)
    ax.text(0.95, 60, r"log:  $\theta$", color=TERRACOTTA, fontsize=10.5, ha="right", path_effects=halo())
    ax.text(1.62, 96, r"quaternion:  $2\sin\frac{\theta}{2}$", color=TEAL, fontsize=10.5, ha="left", path_effects=halo())
    ax.text(3.3, 148, r"trace:  $\sin\theta$,  from 179°", color=INDIGO, fontsize=10.5, ha="left", path_effects=halo())
    ax.text(7.25, 100, "from 179.9°", color=INDIGO, fontsize=10.5, ha="left", path_effects=halo())
    ax.annotate("", (6.42, 40), xytext=(4.2, 40), arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=0.8, mutation_scale=8))
    ax.text(5.31, 53, "2.2 s later", color=MUTED, fontsize=9, ha="center", path_effects=halo(), **SANS)
    ax.set_xlim(0, 10)
    ax.set_ylim(-25, 190)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("error angle (degrees)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch03-linger.svg")


# ------------------------------------------- carrying an arrow round an octant
def globe():
    s = Scene(width=6.4, height=3.8, azim=45, elev=33)
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    for lat in np.radians([30, 60]):
        s.sphere_curve(small_circle(E3, np.pi / 2 - lat), color=INK, lw=0.45, alpha=0.35, show_hidden=False)
    for lon in np.radians([0, 30, 60, 90, 120, 150]):
        s.sphere_curve(great_circle([-np.sin(lon), np.cos(lon), 0]), color=INK, lw=0.45, alpha=0.35, show_hidden=False)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6, alpha=0.5)
    legs = [(E3, E1), (E1, E2), (E2, E3)]
    v = E1.copy()                                 # at the pole, pointing down the first meridian
    L = 0.36
    for k, (a, b) in enumerate(legs):
        s.sphere_curve(geodesic(a, b, n=90), color=INDIGO, lw=2.0, zorder=17)
        axis = unit(np.cross(a, b))
        for frac in ([0.4, 0.74] if k == 0 else [0.0, 0.34, 0.67]):
            Q = rot(axis, frac * np.pi / 2)
            s.arrow(1.004 * Q @ a, 1.004 * Q @ a + L * Q @ v, color=TERRACOTTA, lw=1.4, head=0.09, zorder=33)
        v = rot(axis, np.pi / 2) @ v              # parallel transport along a great circle
        s.arrow(1.004 * rot(axis, 0.84 * np.pi / 2) @ a, 1.004 * rot(axis, 0.92 * np.pi / 2) @ a,
                color=INDIGO, lw=1.2, head=0.11, zorder=18)
    s.arrow(1.004 * E3, 1.004 * E3 + L * E1, color=INK, lw=1.6, head=0.09, zorder=34)
    s.arrow(1.004 * E3, 1.004 * E3 + L * v, color=TERRACOTTA, lw=1.6, head=0.09, zorder=34)
    s.dot(1.004 * E3, color=INK, size=4.5, zorder=35)
    s.label(E3 + L * E1, "start", dx=-19, dy=-1, size=10, **SANS)
    s.label(E3 + L * v, "finish", dx=22, dy=-1, size=10, color=TERRACOTTA, **SANS)
    s.label(E3, "pole", dy=13, size=10, color=MUTED, **SANS)
    s.label(E1, "1", dx=-11, dy=4, size=10, color=INDIGO, **SANS)
    s.label(geodesic(E1, E2)[45], "2", dy=9, size=10, color=INDIGO, **SANS)
    s.label(E2, "3", dx=11, dy=4, size=10, color=INDIGO, **SANS)
    s.text2d(1.95, 0.3, "the arrow never turns\nrelative to the surface,\nyet it comes home\nrotated by 90°",
             fontsize=9.5, color=TERRACOTTA, linespacing=1.3, **SANS)
    s.text2d(-1.95, 0.3, "three quarter-circles\nenclose one eighth\nof the sphere",
             fontsize=9.5, color=INDIGO, linespacing=1.3, **SANS)
    s.save(OUT / "ch03-globe.svg", pad=0.1)


# -------------------------------------------------- transport map on the sphere
def transport():
    s = Scene(width=6.4, height=3.8, azim=-36, elev=22)
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    qd = unit(s.view - 0.55 * s.right + 0.5 * s.up)
    q = unit(s.view + 0.3 * s.right - 0.28 * s.up)
    across = unit(np.cross(qd, q))
    theta = np.arccos(q @ qd)
    arc = geodesic(qd, q)
    s.sphere_curve(great_circle(across), color=INK, lw=0.5, alpha=0.4, show_hidden=False)
    s.sphere_curve(arc, color=TEAL, lw=1.6, zorder=16)

    def plane(p, size):
        along = unit(np.cross(across, p))
        c = 1.002 * p
        s.patch([c + size * (along + across), c + size * (-along + across), c + size * (-along - across),
                 c + size * (along - across)], color=SAND, alpha=0.78)
        return along

    along_d = plane(qd, 0.34)
    plane(q, 0.38)
    qd_dot = 0.52 * (0.55 * along_d + 0.83 * across)             # reference velocity at q_d
    T = np.cross(np.cross(qd, qd_dot), q)                        # (q_d x q_d_dot) x q
    par = rot(across, theta) @ qd_dot                            # parallel transport, for comparison
    along_q = unit(np.cross(across, q))
    q_dot = 0.52 * (-0.62 * along_q + 0.35 * across)
    s.arrow(qd, qd + qd_dot, color=INDIGO, zorder=32)
    s.arrow(q, q + par, color=MUTED, lw=1.0, head=0.08, zorder=31, dashed=True)
    s.arrow(q, q + T, color=INDIGO, zorder=33)
    s.arrow(q, q + q_dot, color=INK, zorder=33)
    s.arrow(q + T, q + q_dot, color=TERRACOTTA, lw=1.4, head=0.09, zorder=34)
    s.dot(qd, color=INK, size=4.5)
    s.dot(q, color=INK, size=4.5)
    s.label(qd, r"$q_d$", dx=-13, dy=-3, size=12)
    s.label(q, r"$q$", dx=-11, dy=-6, size=12)
    s.label(qd + qd_dot, r"$\dot q_d$", dx=11, dy=5, color=INDIGO, size=12)
    s.label(q + T, r"$\dot q_d$ carried to $q$", dx=4, dy=-13, color=INDIGO, size=10.5, ha="left", **SANS)
    s.label(q + q_dot, r"$\dot q$", dx=-10, dy=7, size=12)
    s.label(q + 0.5 * (T + q_dot), r"$e_{\dot q}$", dx=9, dy=11, color=TERRACOTTA, size=12.5)
    s.label(q + par, "parallel\ntransport", dx=7, dy=3, color=MUTED, size=9, ha="left", linespacing=1.15, **SANS)
    s.label(arc[36], r"$\theta$", dx=-10, dy=-5, color=TEAL, size=12)
    s.text2d(-1.95, 0.4, "two velocities,\ntwo tangent planes:\ncarry one across,\nthen subtract",
             fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.text2d(1.95, -0.45, "the difference is\nthe velocity error",
             fontsize=9.5, color=TERRACOTTA, linespacing=1.3, **SANS)
    s.save(OUT / "ch03-transport.svg", pad=0.1)


if __name__ == "__main__":
    frames()
    variation()
    heights()
    linger()
    globe()
    transport()
