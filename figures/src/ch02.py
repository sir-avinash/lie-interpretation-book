"""Figures for Chapter 2.  Run: python figures/src/ch02.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, geodesic, great_circle, halo, hat, new_plot, rot, save, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")


# ----------------------------------------------------------------- helpers
def box(s, center, R, dims, face_colors, zorder=40, lw=0.9):
    """Shaded convex box. Draws only the faces that look at the camera.

    `dims` are the full edge lengths along the body axes (columns of R).
    `face_colors` maps (axis, sign) to a fill color. Returns the visible faces
    as {(axis, sign): (face center, in-plane half vectors)} for decoration.
    """
    c = np.asarray(center, float)
    half = 0.5 * np.asarray(dims, float)
    seen = {}
    for axis in range(3):
        for sign in (1, -1):
            n = sign * R[:, axis]
            if n @ s.view <= 1e-9:
                continue
            a, b = [(half[k] * R[:, k]) for k in range(3) if k != axis]
            fc = c + half[axis] * n
            s.patch([fc + a + b, fc - a + b, fc - a - b, fc + a - b],
                    color=face_colors[(axis, sign)], alpha=1.0, lw=lw, zorder=zorder + 2 * (n @ s.view))
            seen[(axis, sign)] = (fc, a, b)
    return seen


def book(s, center, R, zorder=40):
    """A book: body x across the cover, y along the spine, z out of the front cover."""
    dims = (0.80, 1.10, 0.22)
    cols = {(2, 1): PERIWINKLE, (2, -1): SAND, (0, -1): PEACH, (0, 1): PAPER, (1, 1): PAPER, (1, -1): PAPER}
    seen = box(s, center, R, dims, cols, zorder=zorder)
    if (2, 1) in seen:                       # a title block near the top of the front cover
        fc = seen[(2, 1)][0] + 0.002 * R[:, 2]
        ax, ay = 0.24 * R[:, 0], 0.07 * R[:, 1]
        o = fc + 0.30 * R[:, 1]
        s.patch([o + ax + ay, o - ax + ay, o - ax - ay, o + ax - ay], color=INDIGO, alpha=0.85, lw=0.0, zorder=zorder + 3)
    return seen


def arc_arrow(s, center, axis, start, angle, radius, color, n=40, head=0.1, lw=1.2):
    """Circular arrow about `axis`, starting from direction `start`."""
    pts = np.array([center + radius * rot(axis, t) @ unit(start) for t in np.linspace(0, angle, n)])
    s.curve(pts[:-3], color=color, lw=lw, zorder=45, alpha=1.0)
    s.arrow(pts[-4], pts[-1], color=color, lw=lw, head=head, zorder=45)


# -------------------------------------------------- tangent plane to S2
def tangent():
    s = Scene(width=6.4, height=3.7, azim=-34, elev=20)
    q = unit([0.35, -0.10, 0.93])
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6, alpha=0.5)
    # In-plane direction chosen so that q_dot, omega and the normal project well apart.
    a0 = unit(np.cross(q, E3))
    phi = np.radians(120)
    d = np.cos(phi) * a0 + np.sin(phi) * np.cross(q, a0)
    qdot = 0.62 * d
    omega = np.cross(q, qdot)
    # A curve on the sphere through q with that velocity: a small circle about a tilted axis.
    axis = unit(unit(omega) + 0.45 * q)
    ts = np.linspace(-1.5, 1.2, 160)
    path = np.array([rot(axis, t) @ q for t in ts])
    s.sphere_curve(path, color=TEAL, lw=1.9, zorder=16)
    a, b = s.tangent_plane(1.002 * q, size=0.72, color=SAND, alpha=0.82, spin=phi)
    s.curve(np.array([[0, 0, 0], q]), color=MUTED, hidden=True)
    s.dot((0, 0, 0), color=MUTED, size=3)
    s.arrow(q, q + qdot, color=INDIGO, zorder=31)
    s.arrow(q, q + omega, color=TERRACOTTA, zorder=31)
    s.arrow(q, q + 0.45 * q, color=INK, lw=1.1, head=0.09, zorder=31)
    s.dot(q, color=INK, size=5)
    s.label(q, r"$q$", dx=-11, dy=-5, size=12)
    s.label(q + qdot, r"$\dot q$", dx=9, dy=-7, color=INDIGO, size=12)
    s.label(q + omega, r"$\omega = q \times \dot q$", dx=40, dy=4, color=TERRACOTTA)
    s.label(q + 0.45 * q, "normal", dx=-22, dy=3, size=9.5, **SANS)
    s.label(path[14], r"$q(t)$", dx=-14, dy=8, color=TEAL)
    s.label(q - 0.72 * (a + b), r"$T_qS^2$", dx=-22, dy=0, size=11)
    s.project(-2.9 * s.right)
    s.text2d(-2.0, 0.35, "every velocity of a curve\nthrough $q$ lies in this plane", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.text2d(-2.0, -0.55, r"$q\cdot\dot q = 0$", fontsize=11.5)
    s.save(OUT / "ch02-tangent.svg", pad=0.1)


# ------------------------------------------- exp map: circle and sphere
def expmap():
    s = Scene(width=6.6, height=3.5, azim=-30, elev=22)
    # (a) circle in the camera plane, tangent line at the top
    c = -3.25 * s.right - 0.1 * s.up
    s.project(c - 1.75 * s.right)

    def P(x, y):
        return c + x * s.right + y * s.up

    ring = np.array([P(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 240)])
    s.curve(ring, color=INK, lw=1.7, alpha=1.0, zorder=20)
    s.curve(np.array([P(-1.55, 1), P(1.55, 1)]), color=INK, lw=0.8)
    for t, col in [(1.3, INDIGO), (-0.8, TEAL)]:
        arc = np.array([P(np.sin(a), np.cos(a)) for a in np.linspace(0, t, 60)])
        s.curve(np.array([P(0, 1.0), P(t, 1.0)]), color=col, lw=2.6, alpha=1.0, zorder=22)
        s.curve(arc, color=col, lw=2.6, alpha=1.0, zorder=22)
        s.dot(P(t, 1.0), color=col, size=4.5, hollow=True)
        s.dot(P(np.sin(t), np.cos(t)), color=col, size=4.5)
        # the wrapping motion: tip of the tangent segment swings down onto the circle
        k = np.linspace(0, 1, 30)
        wrap = np.array([P(np.sin(t * kk) + (1 - kk) * t * np.cos(t * kk), np.cos(t * kk) - (1 - kk) * t * np.sin(t * kk)) for kk in k])
        s.curve(wrap, color=col, lw=0.8, hidden=True)
    s.dot(P(0, 1), color=INK, size=5)
    s.label(P(0, 1), "identity", dy=12, size=9.5, **SANS)
    s.label(P(1.3, 1.0), r"$\theta$", dx=4, dy=11, color=INDIGO, size=12)
    s.label(P(np.sin(1.3), np.cos(1.3)), r"$\exp(\theta J)$", dx=-32, dy=-6, color=INDIGO)
    s.label(P(-1.1, 1), "tangent line", dx=0, dy=11, size=9.5, color=MUTED, **SANS)
    s.label(P(0, 0), r"$\mathrm{SO}(2)$", size=12)
    s.label(P(0, -1.32), "equal lengths, wrapped", size=9.5, color=MUTED, **SANS)

    # (b) sphere with a tangent plane at q
    q = unit([0.35, -0.55, 0.76])
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    a, b = s.tangent_plane(1.002 * q, size=0.9, color=SAND, alpha=0.6)
    for ang, length, col, name, off in [(3.5, 1.2, INDIGO, r"$v$", (9, 7)), (1.2, 0.75, TEAL, None, None)]:
        d = np.cos(ang) * a + np.sin(ang) * b
        v = length * d
        end = np.cos(length) * q + np.sin(length) * d
        s.sphere_curve(geodesic(q, end), color=col, lw=2.2, zorder=16)
        s.arrow(q, q + v, color=col, zorder=31)
        s.dot(end, color=col, size=4.5)
        k = np.linspace(0, 1, 30)
        wrap = np.array([np.cos(length * kk) * q + np.sin(length * kk) * d
                         + (1 - kk) * length * (-np.sin(length * kk) * q + np.cos(length * kk) * d) for kk in k])
        s.curve(wrap, color=col, lw=0.8, hidden=True, zorder=25)
        if name:
            s.label(q + v, name, dx=off[0], dy=off[1], color=col, size=12)
            s.label(end, r"$\exp_q(v)$", dx=-30, dy=-8, color=col)
    s.dot(q, color=INK, size=5)
    s.label(q, r"$q$", dx=-4, dy=12, size=12)
    s.label(q - 0.9 * (a - b), r"$T_qS^2$", dx=-18, dy=4, size=10.5)
    s.text2d(1.2, -1.02, r"$S^2$", fontsize=13)
    s.save(OUT / "ch02-expmap.svg", pad=0.08)


# ---------------------------------------- two quarter turns, two orders
def noncommute():
    s = Scene(width=6.6, height=4.3, azim=-62, elev=24)
    Ax, Az = rot(E1, np.pi / 2), rot(E3, np.pi / 2)
    rows = [
        (1.25, [np.eye(3), Ax, Az @ Ax], ["90° about $x$", "90° about $z$"], [E1, E3], "cover faces right, spine upright", INDIGO),
        (-1.25, [np.eye(3), Az, Ax @ Az], ["90° about $z$", "90° about $x$"], [E3, E1], "cover faces you, spine level", TERRACOTTA),
    ]
    xs = [-2.6, 0.0, 2.6]
    for y0, Rs, names, axes, verdict, col in rows:
        for k, R in enumerate(Rs):
            c = xs[k] * s.right + y0 * s.up
            s.shadow(c - 0.72 * E3, 0.62, strength=0.14)
            book(s, c, R)
        for k in range(2):
            m = 0.5 * (xs[k] + xs[k + 1]) * s.right + y0 * s.up
            s.arrow(m - 0.42 * s.right, m + 0.42 * s.right, color=col, lw=1.3)
            s.label(m, names[k], dy=13, color=col, size=10, **SANS)
        s.label(xs[2] * s.right + (y0 - 0.98) * s.up, verdict, color=col, size=10, **SANS)
    s.label(xs[0] * s.right + (1.25 - 0.98) * s.up, "start", size=10, color=MUTED, **SANS)
    s.label(xs[0] * s.right + (-1.25 - 0.98) * s.up, "start", size=10, color=MUTED, **SANS)
    # fixed axes, drawn once
    o = -3.95 * s.right + 0.0 * s.up
    s.project([o - 0.75 * s.right, 4.3 * s.right])
    for e, name, off in [(E1, "$x$", (7, -3)), (E2, "$y$", (6, 5)), (E3, "$z$", (0, 9))]:
        s.arrow(o, o + 0.5 * e, color=INK, lw=1.1, head=0.09)
        s.label(o + 0.5 * e, name, dx=off[0], dy=off[1], size=11)
    s.label(o, "axes fixed\nto the desk", dy=-22, size=9, color=MUTED, linespacing=1.2, **SANS)
    s.save(OUT / "ch02-noncommute.svg", pad=0.1)


# ------------------------------------- quadrotor with a suspended load
def quadload():
    s = Scene(width=6.6, height=4.2, azim=-50, elev=22)
    L = 1.55
    xQ = np.array([0.0, 0.0, 0.0])
    q = unit([0.42, -0.25, -0.87])
    xL = xQ + L * q
    R = rot(E3, np.radians(-8)) @ rot(E2, np.radians(-14)) @ rot(E1, np.radians(10))

    s.shadow(xQ - (L + 0.12) * E3, L, strength=0.13)
    s.sphere(xQ, L, color=PERIWINKLE, alpha=0.38, outline=False)
    ring = np.array([xQ + L * (np.cos(t) * s.right + np.sin(t) * s.up) for t in np.linspace(0, 2 * np.pi, 240)])
    s.curve(ring, color=INK, lw=0.9, alpha=0.7)
    s.sphere_curve(xQ + L * great_circle(E3), center=xQ, color=INK, lw=0.6, alpha=0.5)
    s.sphere_curve(xQ + L * great_circle(np.cross(q, E3)), center=xQ, color=INK, lw=0.6, alpha=0.5)

    # airframe: two arms and four rotor discs, in the body x-y plane
    arm = 0.62
    strands = []
    for d in (R[:, 0], R[:, 1]):
        strands.append((np.linspace(xQ - arm * d, xQ + arm * d, 30), INK, 2.2))
    for d in (R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]):
        cen = xQ + arm * d + 0.05 * R[:, 2]
        disc = np.array([cen + 0.24 * (np.cos(t) * R[:, 0] + np.sin(t) * R[:, 1]) for t in np.linspace(0, 2 * np.pi, 60)])
        s.patch(disc, color=SAND, alpha=0.9, lw=0.9, zorder=34 + 3 * (cen @ s.view))
    s.ribbons(strands, zorder=33)
    # cable, load, and the direction q
    s.curve(np.array([xQ, xL]), color=INK, lw=1.3, alpha=1.0, zorder=28)
    s.arrow(xQ, xQ + 0.62 * q, color=TEAL, lw=2.0, zorder=36)
    s.dot(xL, color=TERRACOTTA, size=9, zorder=40)
    s.dot(xQ, color=INK, size=4, zorder=41)
    # body frame
    for i, (colr, name, off) in enumerate([(MUTED, None, None), (MUTED, None, None), (INDIGO, r"$R\,e_3$", (16, 4))]):
        tip = xQ + (0.5 if i < 2 else 0.78) * R[:, i]
        if i == 2:
            s.arrow(xQ, tip, color=colr, zorder=42)
            s.label(tip, name, dx=off[0], dy=off[1], color=colr)
    # world frame
    o = xQ - 2.9 * s.right - 1.35 * s.up
    for e, name, off in [(E1, "$e_1$", (9, -4)), (E2, "$e_2$", (9, 4)), (E3, "$e_3$", (0, 9))]:
        s.arrow(o, o + 0.5 * e, color=INK, lw=1.1, head=0.09)
        s.label(o + 0.5 * e, name, dx=off[0], dy=off[1], size=10.5)
    s.label(o, "world", dx=-4, dy=-11, size=9.5, color=MUTED, **SANS)
    # annotations
    s.label(xQ + 0.62 * q, r"$q \in S^2$", dx=30, dy=2, color=TEAL, size=12)
    s.label(xL, r"$x_L \in \mathbb{R}^3$", dx=36, dy=-6, color=TERRACOTTA, size=12)
    s.label(xQ + arm * R[:, 0], r"$R \in \mathrm{SO}(3)$", dx=44, dy=6, color=INDIGO, size=12)
    s.label(xQ - arm * R[:, 0], r"$x_Q = x_L - l\,q$", dx=-46, dy=24, size=11)
    mid = xQ + 0.5 * L * q
    s.label(xQ + 0.8 * L * q, r"$l$", dx=-9, dy=2, size=11.5)
    s.text2d(2.75, 1.05, "attitude: 3", fontsize=9.5, color=INDIGO, **SANS)
    s.text2d(2.75, 0.75, "cable direction: 2", fontsize=9.5, color=TEAL, **SANS)
    s.text2d(2.75, 0.45, "load position: 3", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.text2d(2.75, 0.10, "8 degrees of freedom", fontsize=10, color=INK, **SANS)
    s.project(-3.6 * s.right)
    s.text2d(-2.6, 1.0, "the load stays on a sphere\nof radius $l$ about the quadrotor", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.save(OUT / "ch02-quadload.svg", pad=0.1)


# ------------------------------------- one Euler step vs one exp step
def steps():
    fig, ax = new_plot(6.4, 3.3)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    t = np.linspace(-0.25, np.pi / 2 + 0.5, 300)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=1.7, zorder=3)
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    hw, n = 0.42, 4                           # h * omega, exaggerated so the drift is visible
    q0 = np.array([np.cos(0.12), np.sin(0.12)])
    qe, qx = [q0], [q0]
    for _ in range(n):
        qe.append(qe[-1] + hw * J @ qe[-1])
        c, sn = np.cos(hw), np.sin(hw)
        qx.append(np.array([[c, -sn], [sn, c]]) @ qx[-1])
    qe, qx = np.array(qe), np.array(qx)
    for k in range(n):
        ax.annotate("", qe[k + 1], xytext=qe[k], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=TERRACOTTA, lw=1.6, mutation_scale=11, shrinkA=0, shrinkB=0))
        a0 = 0.12 + k * hw
        arc = np.linspace(a0, a0 + hw, 30)
        ax.plot(np.cos(arc), np.sin(arc), color=INDIGO, lw=3.2, solid_capstyle="butt", zorder=4)
        ax.plot([0, qe[k + 1][0]], [0, qe[k + 1][1]], color=TERRACOTTA, lw=0.6, ls=(0, (3.5, 3)), alpha=0.6, zorder=2)
    ax.plot(qx[:, 0], qx[:, 1], "o", ms=5, color=INDIGO, zorder=7)
    ax.plot(qe[1:, 0], qe[1:, 1], "o", ms=5, mfc=PAPER, mec=TERRACOTTA, mew=1.2, zorder=7)
    ax.plot(*q0, "o", ms=6, color=INK, zorder=8)
    ax.plot(0, 0, "o", ms=3, color=MUTED)
    kw = dict(path_effects=halo(), zorder=9)
    ax.text(q0[0] + 0.09, q0[1] - 0.02, r"$q_0$", fontsize=12, ha="left", va="center", **kw)
    ax.text(qe[1][0] + 0.07, qe[1][1] + 0.03, r"$q_0 + h\,\dot q_0$", fontsize=11, color=TERRACOTTA, ha="left", va="center", **kw)
    ax.text(qe[-1][0] - 0.1, qe[-1][1] + 0.07, f"Euler: radius {np.linalg.norm(qe[-1]):.2f} after four steps", fontsize=9.5, color=TERRACOTTA,
            ha="right", va="center", fontfamily="Source Sans 3", **kw)
    ax.text(qx[-1][0] - 0.34, qx[-1][1] - 0.2, "exponential step:\nradius 1.00", fontsize=9.5, color=INDIGO,
            ha="right", va="center", fontfamily="Source Sans 3", linespacing=1.3, **kw)
    ax.text(1.25, 1.2, "each Euler step runs along\nthe tangent, so it always\nlands outside the circle", fontsize=9.5, color=MUTED,
            ha="left", va="center", fontfamily="Source Sans 3", linespacing=1.3, **kw)
    ax.set_xlim(-2.3, 2.9)
    ax.set_ylim(-0.25, 1.75)
    save(fig, OUT / "ch02-steps.svg")
    return float(np.linalg.norm(qe[-1]))


# ------------------------------------------ drift off SO(3), simulated
def drift():
    def Omega_of(t):
        return np.array([np.cos(t), np.sin(0.7 * t), 0.5])

    def run(kind, h, T=10.0, mid=False):
        n = round(T / h)
        R = np.eye(3)
        Rs = [R]
        for k in range(n):
            Om = Omega_of((k + 0.5) * h) if mid else Omega_of(k * h)
            R = R + h * R @ hat(Om) if kind == "euler" else R @ rot(Om, h * np.linalg.norm(Om))
            Rs.append(R)
        return np.array(Rs)

    h = 0.01
    tt = np.arange(0, 10 + h / 2, h)
    eu, ex, exm = run("euler", h), run("exp", h), run("exp", h, mid=True)
    fine = run("exp", 1e-4, mid=True)[::100]

    def orth(Rs):
        return np.array([np.linalg.norm(R.T @ R - np.eye(3)) for R in Rs])

    def ang(Rs):
        return np.degrees([np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)) for A, B in zip(fine, Rs)])

    fig, (ax0, ax1) = new_plot(6.6, 2.8, ncols=2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.9, wspace=0.3)
    ax0.semilogy(tt[1:], orth(eu)[1:], color=TERRACOTTA, lw=1.8)
    ax0.semilogy(tt[1:], np.maximum(orth(ex)[1:], 1e-17), color=INDIGO, lw=1.2)
    ax0.set_ylim(1e-17, 10)
    ax0.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
    ax0.text(5.0, 2e-3, "Euler step", color=TERRACOTTA, fontsize=10.5, ha="center", path_effects=halo())
    ax0.text(5.0, 1.5e-13, "exponential step", color=INDIGO, fontsize=10.5, ha="center", path_effects=halo())
    ax0.set_title(r"distance from the manifold, $\|R^{\!\top}\! R - I\|$", fontsize=10, **SANS)
    ax1.semilogy(tt[1:], np.maximum(ang(ex)[1:], 1e-7), color=INDIGO, lw=1.8)
    ax1.semilogy(tt[1:], np.maximum(ang(exm)[1:], 1e-7), color=TEAL, lw=1.8)
    ax1.set_ylim(1e-6, 10)
    ax1.text(9.8, 1.2, r"$\Omega$ sampled at the start of the step", color=INDIGO, fontsize=10, ha="right", path_effects=halo())
    ax1.text(9.8, 1.2e-5, r"$\Omega$ sampled at the midpoint", color=TEAL, fontsize=10, ha="right", path_effects=halo())
    ax1.set_title("attitude error of the exponential step (degrees)", fontsize=10, **SANS)
    for ax in (ax0, ax1):
        ax.set_xlim(0, 10)
        ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
        ax.minorticks_off()
    save(fig, OUT / "ch02-drift.svg")


if __name__ == "__main__":
    tangent()
    expmap()
    noncommute()
    quadload()
    r = steps()
    print(f"steps figure: Euler radius after four steps = {r:.4f}")
    drift()
