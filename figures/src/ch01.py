"""Figures for Chapter 1.  Run: python figures/src/ch01.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, OCHRE, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, geodesic, great_circle, halo, new_plot, rot, save, small_circle, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)


def circle_in_plane(a, b, radius, n=160):
    t = np.linspace(0, 2 * np.pi, n)
    return radius * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))


# ------------------------------------------------------------------ gimbals
def gimbals():
    s = Scene(width=6.6, height=3.3, azim=-58, elev=16)

    def assembly(offset, pitch, tag, caption):
        c = offset * s.right
        yaw, roll = np.radians(-38), np.radians(25)
        Rz, Ry, Rx = rot(E3, yaw), rot(E2, pitch), rot(E1, roll)
        y1 = Rz @ E2                 # pitch axis, carried by the outer ring
        x2 = Rz @ Ry @ E1            # roll axis, carried by the middle ring
        Rb = Rz @ Ry @ Rx
        rings = [
            (c + circle_in_plane(E3, y1, 1.00), INDIGO, 2.6),
            (c + circle_in_plane(y1, x2, 0.80), TEAL, 2.6),
            (c + circle_in_plane(x2, Rb @ E2, 0.60), TERRACOTTA, 2.6),
        ]
        axes = [
            (np.array([c - 1.42 * E3, c - 1.0 * E3]), INDIGO), (np.array([c + 1.0 * E3, c + 1.42 * E3]), INDIGO),
            (np.array([c - 1.0 * y1, c - 0.8 * y1]), TEAL), (np.array([c + 0.8 * y1, c + 1.0 * y1]), TEAL),
            (np.array([c - 0.8 * x2, c - 0.6 * x2]), TERRACOTTA), (np.array([c + 0.6 * x2, c + 0.8 * x2]), TERRACOTTA),
        ]
        s.ribbons(rings + [(np.linspace(a[0], a[1], 12), col, 1.3) for a, col in axes])
        plate = [c + Rb @ (0.24 * np.array(v)) for v in [(1.3, 1, 0), (-1.3, 1, 0), (-1.3, -1, 0), (1.3, -1, 0)]]
        s.patch(plate, color=SAND, alpha=0.95, zorder=40)
        s.arrow(c, c + 0.36 * Rb @ E3, color=INK, lw=1.2, head=0.09, zorder=41)
        s.shadow(c - 1.5 * E3, 0.95, strength=0.16)
        s.label(c + 1.56 * E3, "yaw axis", color=INDIGO, size=9.5, fontfamily="Source Sans 3")
        s.label(c + 1.12 * y1, "pitch", dx=14, dy=2, color=TEAL, size=9.5, fontfamily="Source Sans 3")
        s.label(c - 1.9 * E3, f"({tag}) {caption}", size=10.5, fontfamily="Source Sans 3")
        return c, x2

    c, x2 = assembly(-1.55, np.radians(28), "a", "pitch 28°: three independent axes")
    s.label(c + 0.92 * x2, "roll", dx=-10, dy=-8, color=TERRACOTTA, size=9.5, fontfamily="Source Sans 3")
    c, x2 = assembly(1.55, np.radians(90), "b", "pitch 90°: roll axis lands on yaw axis")
    s.label(c - 0.7 * E3, "roll", dx=22, dy=-4, color=TERRACOTTA, size=9.5, fontfamily="Source Sans 3")
    s.save(OUT / "ch01-gimbals.svg", pad=0.05)


# --------------------------------------------------------- Euler rate blow-up
def euler_rates():
    fig, ax = new_plot(6.4, 2.7)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.93)
    th = np.radians(np.linspace(0, 89.9, 800))
    phi, Om = np.radians(45), np.array([0.0, 0.5, 0.5])
    yaw_rate = (Om[1] * np.sin(phi) + Om[2] * np.cos(phi)) / np.cos(th)
    ax.semilogy(np.degrees(th), yaw_rate, color=TERRACOTTA, lw=1.8)
    ax.axhline(np.linalg.norm(Om), color=INDIGO, lw=1.5)
    ax.plot([89], [(Om[1] * np.sin(phi) + Om[2] * np.cos(phi)) / np.cos(np.radians(89))], "o", ms=5, color=TERRACOTTA)
    ax.annotate("40.5 rad/s at 89°", (89, 40.5), xytext=(-12, 0), textcoords="offset points", ha="right",
                va="center", fontsize=10, color=TERRACOTTA, path_effects=halo())
    ax.text(30, 1.4, r"yaw rate $\dot\psi$ reported by the Euler chart", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax.text(30, 0.47, r"what the body is doing: $\|\Omega\| = 0.71$ rad/s", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax.set_xlim(0, 90)
    ax.set_ylim(0.3, 600)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("pitch angle θ (degrees)", fontfamily="Source Sans 3", fontsize=9.5)
    ax.set_ylabel("rad/s", fontfamily="Source Sans 3", fontsize=9.5)
    save(fig, OUT / "ch01-euler-rates.svg")


# ---------------------------------------------------------------- unwinding
def unwinding():
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2, gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.17, top=0.95, wspace=0.18)
    ax0.set_aspect("equal")
    ax0.axis("off")
    t = np.linspace(0, 2 * np.pi, 300)
    ax0.plot(np.cos(t), np.sin(t), color=INK, lw=1.7)
    half = np.radians(-5)                              # quaternion angle is half the rotation angle
    qp = np.array([np.cos(half), np.sin(half)])
    qm = -qp
    short = np.linspace(half, 0, 30)
    long_ = np.linspace(half + np.pi, 2 * np.pi, 200)
    ax0.plot(1.0 * np.cos(short), 1.0 * np.sin(short), color=INDIGO, lw=4, solid_capstyle="round")
    ax0.plot(1.13 * np.cos(long_), 1.13 * np.sin(long_), color=TERRACOTTA, lw=1.8)
    tip = 1.13 * np.array([np.cos(long_[-1]), np.sin(long_[-1])])
    ax0.annotate("", tip, xytext=1.13 * np.array([np.cos(long_[-12]), np.sin(long_[-12])]),
                 arrowprops=dict(arrowstyle="-|>", color=TERRACOTTA, lw=1.8, mutation_scale=11))
    for pt, col, hollow in [((1, 0), INK, False), ((-1, 0), INK, True), (qp, INDIGO, False), (qm, TERRACOTTA, False)]:
        ax0.plot(*pt, "o", ms=6, mfc=PAPER if hollow else col, mec=col, mew=1.2, zorder=5)
    kw = dict(fontsize=11, path_effects=halo())
    ax0.text(1.3, 0.13, r"$+\mathbf{1}$", ha="left", **kw)
    ax0.text(1.3, -0.12, "target", ha="left", fontsize=9, fontfamily="Source Sans 3", color=MUTED)
    ax0.text(-1.3, 0.13, r"$-\mathbf{1}$", ha="right", **kw)
    ax0.text(-1.3, -0.12, "same attitude", ha="right", fontsize=9, fontfamily="Source Sans 3", color=MUTED)
    ax0.text(qp[0] - 0.2, qp[1] - 0.02, r"$\mathfrak{q}$", ha="center", color=INDIGO, **kw)
    ax0.text(qm[0] + 0.25, qm[1] + 0.0, r"$-\mathfrak{q}$", ha="center", color=TERRACOTTA, **kw)
    ax0.text(0, 0.12, r"$(\cos\frac{\theta}{2},\ \sin\frac{\theta}{2})$", ha="center", fontsize=11)
    ax0.text(0, -0.2, "fixed axis", ha="center", fontsize=9, fontfamily="Source Sans 3", color=MUTED)
    ax0.text(0, -1.42, "350° the long way round", ha="center", color=TERRACOTTA, fontsize=9.5, fontfamily="Source Sans 3")
    ax0.set_xlim(-2.15, 2.15)
    ax0.set_ylim(-1.6, 1.3)

    J3, kp, kd = 0.10, 2.0, 0.6

    def f(_, s):
        return [s[1], (-kp * np.sin(s[0] / 2) - kd * s[1]) / J3]

    tt = np.linspace(0, 6, 600)
    for th0, col, lab in [(np.radians(-10), INDIGO, r"start at $\mathfrak{q}$"),
                          (np.radians(350), TERRACOTTA, r"start at $-\mathfrak{q}$")]:
        sol = solve_ivp(f, [0, 6], [th0, 0.0], t_eval=tt, rtol=1e-9, atol=1e-11)
        travelled = np.degrees(np.abs(sol.y[0] - th0))
        ax1.plot(tt, travelled, color=col, lw=1.8)
        ax1.text(3.0, travelled[-1] + (22 if th0 > 0 else 18), lab, color=col, fontsize=10.5, path_effects=halo())
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 400)
    ax1.set_yticks([0, 90, 180, 270, 360])
    ax1.set_xlabel("time (s)", fontfamily="Source Sans 3", fontsize=9.5)
    ax1.set_ylabel("angle turned (degrees)", fontfamily="Source Sans 3", fontsize=9.5)
    save(fig, OUT / "ch01-unwinding.svg")


# ------------------------------------------------- ball model of SO(3)
def so3_ball():
    s = Scene(width=6.4, height=3.5, azim=-40, elev=18)
    s.shadow((0, 0, -1.08), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE, alpha=0.5)
    for nrm in (E3, E1):
        s.sphere_curve(great_circle(nrm), color=INK, lw=0.6, alpha=0.5)
    u = unit([0.25, -0.75, 0.6])
    p_exit, p_enter = u, -u
    s.curve(np.array([-u, u]), color=MUTED, lw=0.7, hidden=True)
    s.arrow((0, 0, 0), 0.62 * u, color=INDIGO)
    s.dot((0, 0, 0), color=INK)
    s.label((0, 0, 0), "identity", dx=-26, dy=-9, size=9.5, fontfamily="Source Sans 3")
    s.label(0.62 * u, r"$\theta\,u$", dx=14, dy=-2, color=INDIGO)
    s.dot(p_exit, color=TERRACOTTA)
    s.dot(p_enter, color=TERRACOTTA, hollow=not s.point_visible(p_enter))
    s.label(p_exit, r"$\pi u$", dx=13, dy=7, color=TERRACOTTA)
    s.label(p_enter, r"$-\pi u$", dx=-17, dy=-8, color=TERRACOTTA)
    # A rotation that keeps turning about u: leaves through pi u, re-enters at -pi u.
    s.arrow(0.66 * u, 0.97 * u, color=TERRACOTTA, lw=1.2, head=0.08)
    s.arrow(-0.99 * u, -0.6 * u, color=TERRACOTTA, lw=1.2, head=0.08)
    s.text2d(1.75, 0.55, "radius π", fontsize=9.5, fontfamily="Source Sans 3", color=MUTED)
    s.text2d(1.9, -0.2, "antipodal boundary points\nare the same rotation",
             fontsize=9.5, fontfamily="Source Sans 3", color=TERRACOTTA, linespacing=1.3)
    s.text2d(-1.95, 0.25, "direction = axis\nlength = angle", fontsize=9.5,
             fontfamily="Source Sans 3", color=INDIGO, linespacing=1.3)
    s.save(OUT / "ch01-so3-ball.svg", pad=0.1)


# --------------------------------------------------- obstruction, S1 and S2
def obstruction():
    s = Scene(width=6.6, height=3.4, azim=-30, elev=20)
    # (a) a circle, drawn flat in the camera plane
    c = -2.7 * s.right
    ring = c + np.array([np.cos(t) * s.right + np.sin(t) * s.up for t in np.linspace(0, 2 * np.pi, 200)])
    s.curve(ring, color=INK, lw=1.7, alpha=1.0, zorder=20)
    for sign in (1, -1):
        for ang in np.radians([-140, -95, -50]):
            a0 = np.pi / 2 + sign * ang
            a1 = a0 - sign * np.radians(-16)
            p0, p1 = (c + 1.16 * (np.cos(a) * s.right + np.sin(a) * s.up) for a in (a0, a1))
            s.arrow(p0, p1, color=INDIGO, lw=1.3, head=0.1)
    s.dot(c + s.up, color=INDIGO, size=6)
    s.dot(c - s.up, color=TERRACOTTA, size=6, hollow=True)
    s.label(c + s.up, "goal", dy=13, size=10, color=INDIGO, fontfamily="Source Sans 3")
    s.label(c - s.up, "a second equilibrium is forced", dy=-14, size=10, color=TERRACOTTA, fontfamily="Source Sans 3")
    s.label(c, r"$S^1$", size=13)
    # (b) the sphere with the flow of  omega = -k e_q
    qd = unit([0.1, -0.55, 0.83])
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    side = unit(np.cross(qd, E1))
    for k in range(8):
        start = rot(qd, k * np.pi / 4) @ side
        merid = np.array([np.cos(a) * qd + np.sin(a) * start for a in np.linspace(0.12, np.pi - 0.1, 90)])
        s.sphere_curve(merid, color=INK, lw=0.7)
        for a in (1.75, 0.95):
            p0 = np.cos(a) * qd + np.sin(a) * start
            p1 = np.cos(a - 0.22) * qd + np.sin(a - 0.22) * start
            if s.point_visible(p0) and s.point_visible(p1):
                s.arrow(1.004 * p0, 1.004 * p1, color=INDIGO, lw=1.2, head=0.09)
    s.dot(qd, color=INDIGO, size=6)
    s.dot(-qd, color=TERRACOTTA, size=6, hollow=True)
    s.label(qd, r"$q_d$", dx=12, dy=10, color=INDIGO, size=12)
    s.label(-qd, r"$-q_d$", dx=-14, dy=-11, color=TERRACOTTA, size=12)
    s.text2d(1.25, -0.95, r"$S^2$", fontsize=13)
    s.save(OUT / "ch01-obstruction.svg", pad=0.08)


# --------------------------------------------------------- error on S2
def s2_error():
    s = Scene(width=6.4, height=3.6, azim=-32, elev=20)
    qd = unit([0.25, -0.2, 0.9])
    q = unit([0.8, -0.6, 0.12])
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    for polar in np.radians([25, 62, 100, 140]):
        s.sphere_curve(small_circle(qd, polar), color=INK, lw=0.6, alpha=0.55)
    a, b = s.tangent_plane(1.002 * q, size=0.4, color=SAND, alpha=0.8)
    s.sphere_curve(geodesic(qd, q), color=TEAL, lw=1.9, zorder=16)
    e_q = np.cross(qd, q)
    qdot = qd - (q @ qd) * q                 # direction of motion under omega = -k e_q
    s.arrow(q, q + 0.36 * unit(qdot), color=INDIGO, zorder=31)
    s.arrow(q, q + 0.36 * unit(e_q), color=TERRACOTTA, zorder=31)
    s.dot(qd, color=INK, size=5)
    s.dot(q, color=INK, size=5)
    s.label(qd, r"$q_d$", dx=-12, dy=9, size=12)
    s.label(q, r"$q$", dx=9, dy=-10, size=12)
    s.label(q + 0.36 * unit(e_q), r"$e_q = q_d \times q$", dx=44, dy=0, color=TERRACOTTA)
    s.label(q + 0.36 * unit(qdot), r"$\dot q$", dx=-11, dy=-2, color=INDIGO)
    mid = geodesic(qd, q)[22]
    s.label(mid, r"$\theta$", dx=10, dy=0, color=TEAL)
    s.label(q - 0.4 * (a + b), r"$T_qS^2$", dx=-20, dy=-9, size=10.5)
    s.text2d(-1.9, 0.45, r"level sets of $\Psi_q = 1 - q\cdot q_d$", fontsize=9.5,
             fontfamily="Source Sans 3", color=MUTED)
    s.save(OUT / "ch01-s2-error.svg", pad=0.1)


if __name__ == "__main__":
    gimbals()
    euler_rates()
    unwinding()
    so3_ball()
    obstruction()
    s2_error()
