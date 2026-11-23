"""Figures for Chapter 2.  Run: python figures/src/ch02.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (DASH, ICE, INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, geodesic, great_circle, halo, hat, new_plot, phase, quadrotor, rot, save, unit)

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


def axis_through_box(s, center, R, dims, d, length, color, zorder=40, lw=1.5, head=0.11):
    """Arrow from the center of a box along the unit vector d, with the right occlusion.

    If d points toward the camera the arrow is drawn from where it leaves the box, on top.
    If it points away it is drawn from the center, underneath the box. Returns the tip.
    """
    c = np.asarray(center, float)
    half = 0.5 * np.asarray(dims, float)
    tip = c + length * d
    if d @ s.view > 0:
        leave = min(half[k] / max(abs(d @ R[:, k]), 1e-9) for k in range(3))
        s.arrow(c + min(leave, 0.95 * length) * d, tip, color=color, zorder=zorder + 4, lw=lw, head=head)
    else:
        s.arrow(c, tip, color=color, zorder=zorder - 4, lw=lw, head=head)
    return tip


def caption(s, x, y, msg, color=MUTED, size=10):
    """The line of text inside an animation frame that says what is happening now."""
    s.text2d(x, y, msg, fontsize=size, color=color, **SANS)


_CACHE = {}


def cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


# ---------------------------------- opening: the cable that grows (animation)
PUZ_H, PUZ_W, PUZ_T, PUZ_CONE = 0.01, 2.0, 10.0, np.radians(50)


def puzzle_run(phi_end=0.0):
    """The simulator of the opening puzzle, on a load that swings steadily round a cone.

    The true direction q(t) circles the downward vertical at the half-angle PUZ_CONE with
    |omega| = PUZ_W, omega = q x q_dot. The simulator takes Euler steps q <- q + h (omega x q).
    Returns the times, the true directions and the simulated ones. `phi_end` is the azimuth of
    the true direction at the final time.
    """
    n = round(PUZ_T / PUZ_H)
    rate = PUZ_W / np.sin(PUZ_CONE)                       # precession rate that makes |q_dot| = PUZ_W
    ts = PUZ_H * np.arange(n + 1)
    phi = phi_end + rate * (ts - PUZ_T)
    sa, ca = np.sin(PUZ_CONE), np.cos(PUZ_CONE)
    true = np.stack([sa * np.cos(phi), sa * np.sin(phi), -ca * np.ones_like(phi)], axis=1)
    vel = rate * np.stack([-sa * np.sin(phi), sa * np.cos(phi), np.zeros_like(phi)], axis=1)
    omega = np.cross(true, vel)
    sim = [true[0]]
    for k in range(n):
        sim.append(sim[-1] + PUZ_H * np.cross(omega[k], sim[-1]))
    return ts, true, np.array(sim)


def puzzle_frame(t):
    s = Scene(width=6.6, height=3.75, azim=-40, elev=17, xlim=(-4.1, 2.7), ylim=(-2.25, 1.6))
    L = 1.3                                               # one meter of cable, in scene units
    xQ = np.zeros(3)
    # End the run where the load shows clear of the sphere's outline: q perpendicular to the view, on the right.
    az_view = np.arctan2(s.view[1], s.view[0])
    phi_end = az_view + np.arccos(np.tan(np.radians(17)) / np.tan(PUZ_CONE))
    ts, true, sim = cached(("puzzle", round(phi_end, 9)), lambda: puzzle_run(phi_end))
    tau = PUZ_T * phase(t, 0.06, 0.90)
    k = int(round(tau / PUZ_H))
    length = float(np.linalg.norm(sim[k]))

    s.shadow(xQ - (L + 0.22) * E3, 1.25 * L, strength=0.12)
    s.sphere(xQ, L, color=PERIWINKLE, alpha=0.38, outline=False)
    ring = np.array([xQ + L * (np.cos(a) * s.right + np.sin(a) * s.up) for a in np.linspace(0, 2 * np.pi, 240)])
    s.curve(ring, color=INK, lw=0.9, alpha=0.7)
    s.sphere_curve(xQ + L * great_circle(E3), center=xQ, color=INK, lw=0.6, alpha=0.5)
    # where the load should be: a circle on the sphere, and a hollow marker on a dashed cable
    circle = np.array([[np.sin(PUZ_CONE) * np.cos(a), np.sin(PUZ_CONE) * np.sin(a), -np.cos(PUZ_CONE)]
                       for a in np.linspace(0, 2 * np.pi, 200)])
    s.sphere_curve(xQ + L * circle, center=xQ, color=MUTED, lw=0.8, alpha=1.0)
    # the simulated load: its trail leaves the sphere
    if k > 1:
        trail = xQ + L * sim[: k + 1 : 2]
        front = sim[: k + 1 : 2] @ s.view >= 0
        start = 0
        for i in range(1, len(trail) + 1):
            if i == len(trail) or front[i] != front[start]:
                seg = trail[max(start - 1, 0):min(i + 1, len(trail))]
                if front[start]:
                    s.curve(seg, color=TERRACOTTA, lw=1.0, alpha=1.0, zorder=22)
                else:
                    s.curve(seg, color=TERRACOTTA, lw=0.7, alpha=1.0, zorder=5)
                start = i
    s.curve(np.array([xQ, xQ + L * true[k]]), color=MUTED, lw=0.9, alpha=1.0, linestyle=DASH, zorder=26)
    s.dot(xQ + L * true[k], color=MUTED, size=6.5, hollow=True, zorder=27)
    s.curve(np.array([xQ, xQ + L * sim[k]]), color=TERRACOTTA, lw=1.5, alpha=1.0, zorder=28)
    s.dot(xQ + L * sim[k], color=TERRACOTTA, size=8, zorder=40)
    quadrotor(s, xQ, rot(E3, np.radians(20)), arm=0.52)
    s.dot(xQ, color=INK, size=3.5, zorder=41)

    # readouts
    x0 = -2.85
    s.text2d(x0, 1.12, "the simulator's update", fontsize=9.5, color=MUTED, **SANS)
    s.text2d(x0, 0.80, r"$q \leftarrow q + h\,(\omega \times q)$", fontsize=12.5, color=TERRACOTTA)
    s.text2d(x0, 0.22, f"time  {tau:4.1f} s", fontsize=10, color=INK, **SANS)
    s.text2d(x0, -0.12, "simulated cable length", fontsize=9.5, color=MUTED, **SANS)
    s.text2d(x0, -0.46, f"{length:.2f} m", fontsize=15, color=TERRACOTTA, **SANS)
    # a gauge: one meter in ink, the excess in terracotta
    g0, g1, gy = x0 - 0.6, x0 + 0.4, -0.84
    s.ax.plot([g0, g1], [gy, gy], color=INK, lw=3.0, solid_capstyle="butt", zorder=40)
    s.ax.plot([g1, g1 + (length - 1.0) * (g1 - g0)], [gy, gy], color=TERRACOTTA, lw=3.0, solid_capstyle="butt", zorder=40)
    for gx in (g0, g1):
        s.ax.plot([gx, gx], [gy - 0.07, gy + 0.07], color=INK, lw=0.8, zorder=41)
    s.text2d(0.5 * (g0 + g1), gy - 0.24, "1 m", fontsize=9, color=MUTED, **SANS)
    s.text2d(1.98, 1.2, "the load must stay\non this sphere", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    if tau <= 0:
        msg, col = "a load swings steadily at 2 rad/s on a cable one meter long", MUTED
    elif tau < PUZ_T:
        msg, col = "each step is correct at that instant, and lands a little outside the sphere", TERRACOTTA
    else:
        msg, col = "after ten seconds the load hangs 22 cm farther away than the cable allows", TERRACOTTA
    caption(s, -0.7, -2.05, msg, color=col)
    return s


def puzzle():
    animate(puzzle_frame, OUT / "ch02-puzzle.gif", OUT / "ch02-puzzle-strip.png", n=110, fps=12,
            width_px=720, stills=(0.0, 0.34, 0.62, 1.0), strip_px=900, strip_cols=2, hold=2.0)


# -------------------------------------------------- tangent plane to S2 (animation)
def outward(s, base, tip, dist):
    """Label offset in points that continues the projected direction from `base` to `tip`."""
    d = (np.asarray(tip, float) - np.asarray(base, float))
    v = np.array([d @ s.right, d @ s.up])
    v = v / max(np.hypot(*v), 1e-9)
    return dict(dx=dist * v[0], dy=dist * v[1])


def tangent_frame(t):
    s = Scene(width=6.4, height=3.7, azim=-34, elev=27, xlim=(-3.45, 2.1), ylim=(-1.66, 1.549))

    def cam(x, y):
        return x * s.right + y * s.up + np.sqrt(1 - x * x - y * y) * s.view

    # A curve on the sphere from near the pole to near the equator: the small circle through three
    # points chosen to face the camera, traversed at constant speed.
    A, M, B = cam(-0.30, 0.60), cam(0.16, 0.42), cam(0.50, -0.10)
    axis = unit(np.cross(M - A, B - M))
    q0 = A
    sweep = lambda p: np.arctan2(np.cross(q0 - (q0 @ axis) * axis, p - (p @ axis) * axis) @ axis,
                                 (q0 - (q0 @ axis) * axis) @ (p - (p @ axis) * axis))
    rate = 0.55 / np.linalg.norm(np.cross(axis, q0))
    tau0, tau1 = 0.0, float(sweep(B))
    tau = tau0 + (tau1 - tau0) * ease(phase(t, 0.10, 0.88))

    def state(u):
        q = rot(axis, u) @ q0
        qdot = rate * np.cross(axis, q)
        return q, qdot, np.cross(q, qdot)

    def plane(q, qdot, size=0.56):
        a = unit(qdot)
        b = np.cross(q, a)
        p = 1.002 * q
        return [p + size * (a + b), p + size * (-a + b), p + size * (-a - b), p + size * (a - b)]

    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6, alpha=0.5)
    path = np.array([rot(axis, u) @ q0 for u in np.linspace(tau0 - 0.6, tau1 + 0.3, 160)])
    s.sphere_curve(path, color=TEAL, lw=1.9, zorder=13, alpha=1.0)
    s.label(path[0], r"$q(t)$", dx=-19, dy=1, color=TEAL)
    # the plane where the walk started stays behind as a dashed outline
    qs, qds, _ = state(tau0)
    if tau > tau0 + 0.02:
        ghost = np.array(plane(qs, qds) + plane(qs, qds)[:1])
        s.curve(ghost, color=MUTED, lw=0.8, alpha=1.0, linestyle=DASH, zorder=15)
        s.arrow(qs, qs + qds, color=MUTED, lw=0.9, head=0.08, dashed=True, zorder=16)
        s.dot(qs, color=MUTED, size=4.5, hollow=True, zorder=16.5)
    q, qdot, omega = state(tau)
    corners = plane(q, qdot)
    s.patch(corners, color=SAND, alpha=0.82, zorder=17)
    s.curve(np.array([[0, 0, 0], q]), color=MUTED, hidden=True)
    s.dot((0, 0, 0), color=MUTED, size=3)
    s.arrow(q, q + qdot, color=INDIGO, zorder=31)
    s.arrow(q, q + omega, color=TERRACOTTA, zorder=31)
    s.arrow(q, q + 0.45 * q, color=INK, lw=1.1, head=0.09, zorder=31)
    s.dot(q, color=INK, size=5)
    s.label(q, r"$q$", size=12, **outward(s, 0.5 * (qdot + omega), 0 * q, 11))
    s.label(q + qdot, r"$\dot q$", color=INDIGO, size=12, **outward(s, q, q + qdot, 10))
    s.label(q + omega, r"$\omega$", color=TERRACOTTA, size=12, **outward(s, q, q + omega, 10))
    if np.hypot(q @ s.right, q @ s.up) > 0.55:
        s.label(q + 0.45 * q, "normal", size=9.5, **outward(s, 0 * q, q, 17), **SANS)
    s.label(corners[1], r"$T_qS^2$", size=11, **outward(s, 1.002 * q, corners[1], 15))
    s.text2d(-2.4, 0.62, "every velocity of a curve\nthrough $q$ lies in this plane", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.text2d(-2.4, 0.02, r"$q\cdot\dot q = 0$", fontsize=11.5)
    s.text2d(-2.4, -0.34, r"$\omega = q \times \dot q$", fontsize=11.5, color=TERRACOTTA)
    if tau <= tau0:
        msg, col = "at $q$ the velocity lies in the tangent plane", MUTED
    elif tau < tau1:
        msg, col = "the plane rides along with $q$, and tilts as $q$ moves", INDIGO
    else:
        msg, col = "another point, another plane: the tangent space has changed", INDIGO
    caption(s, -0.68, -1.5, msg, color=col)
    return s


def tangent():
    animate(tangent_frame, OUT / "ch02-tangent.gif", OUT / "ch02-tangent-strip.png", n=84, fps=12,
            width_px=700, stills=(0.0, 0.42, 0.62, 1.0), strip_px=880, strip_cols=2, hold=2.0)


# ----------------------------- zooming in: a sphere looks flat near a point (animation)
def flat_frame(t):
    s = Scene(width=6.4, height=3.75, azim=-34, elev=27, xlim=(-3.5, 1.9), ylim=(-2.2, 0.964))
    lens = ((-0.06, -0.42), 1.3)                         # a round window of fixed size on the page
    half = 0.6                                           # half-width of the patch; it keeps this size on the page
    rho = float(np.exp(np.log(40.0) * ease(phase(t, 0.10, 0.88))))      # radius of the sphere: zooming in makes it grow
    n = 0.12 * s.right + 0.52 * s.up + np.sqrt(1 - 0.12**2 - 0.52**2) * s.view   # outward normal where you stand
    a = unit(np.cross(n, E3))
    b = np.cross(n, a)
    center = -rho * n                                    # you stand at the origin of the scene

    def on_sphere(x, y):
        """The point reached by walking from the origin along the great circle toward x a + y b, for its length."""
        r = np.hypot(x, y)
        if r < 1e-12:
            return np.zeros(3)
        return center + rho * (np.cos(r / rho) * n + np.sin(r / rho) * (x * a + y * b) / r)

    s.sphere(center, rho, color=PERIWINKLE, res=600)
    us = np.linspace(-half, half, 60)
    ticks = np.linspace(-half, half, 7)
    for c in ticks:
        lw = 1.3 if abs(abs(c) - half) < 1e-9 else 0.8
        s.sphere_curve(np.array([on_sphere(c, u) for u in us]), center=center, color=INDIGO, lw=lw, alpha=1.0, zorder=15)
        s.sphere_curve(np.array([on_sphere(u, c) for u in us]), center=center, color=INDIGO, lw=lw, alpha=1.0, zorder=15)
    # Everything so far is seen through a round window of fixed size, so the page does not fill with sphere.
    from matplotlib.patches import Circle
    window = Circle(*lens, transform=s.ax.transData)
    for art in list(s.ax.images) + list(s.ax.patches) + list(s.ax.lines):
        art.set_clip_path(window)
    s.ax.add_patch(Circle(*lens, fill=False, ec=INK, lw=0.9, zorder=21))
    # the tangent plane with the same grid, flat
    s.patch([half * (a + b), half * (-a + b), half * (-a - b), half * (a - b)], color=SAND, alpha=0.55, lw=1.0, zorder=17)
    for c in ticks[1:-1]:
        s.curve(np.array([c * a - half * b, c * a + half * b]), color=INK, lw=0.5, alpha=1.0, zorder=18)
        s.curve(np.array([-half * a + c * b, half * a + c * b]), color=INK, lw=0.5, alpha=1.0, zorder=18)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):   # how far the plane's corners float above the surface
        s.curve(np.array([half * (sx * a + sy * b), on_sphere(half * sx, half * sy)]), color=TERRACOTTA, lw=1.0, alpha=1.0, zorder=19)
    s.arrow(np.zeros(3), 0.42 * n, color=INK, lw=1.1, head=0.09, zorder=31)
    s.dot(np.zeros(3), color=INK, size=5)
    s.label(np.zeros(3), r"$q$", dx=-10, dy=-6, size=12)
    low = min((half * (sx * a + sy * b) for sx in (1, -1) for sy in (1, -1)), key=lambda p: p @ s.up)
    s.label(low, r"$T_qS^2$", size=11, **outward(s, np.zeros(3), low, 17))
    gap = np.hypot(rho, half * np.sqrt(2)) - rho         # distance from a corner of the plane to the sphere
    x0 = -2.5
    s.text2d(x0, 0.42, "radius of the sphere", fontsize=9.5, color=MUTED, **SANS)
    s.text2d(x0, 0.12, f"{rho / (2 * half):.1f} patch widths", fontsize=12, color=INDIGO, **SANS)
    s.text2d(x0, -0.42, "gap at the corners", fontsize=9.5, color=MUTED, **SANS)
    s.text2d(x0, -0.72, f"{100 * gap / (2 * half):.1f} % of the width", fontsize=12, color=TERRACOTTA, **SANS)
    if rho <= 1.0:
        msg, col = "a patch of the sphere around $q$, and the tangent plane at $q$", MUTED
    elif rho < 40.0 - 1e-9:
        msg, col = "zoom in on $q$: the patch keeps its size on the page, so the sphere grows", INDIGO
    else:
        msg, col = "seen from close enough, the sphere and its tangent plane agree", INDIGO
    caption(s, -0.8, -2.03, msg, color=col)
    return s


def flat():
    animate(flat_frame, OUT / "ch02-flat.gif", OUT / "ch02-flat-strip.png", n=84, fps=12,
            width_px=700, stills=(0.0, 0.36, 0.56, 1.0), strip_px=880, strip_cols=2, hold=2.0)


# ------------------------------- body and world angular velocity (animation)
TT_FRAMES, TT_FPS, TT_RATE = 108, 12, 0.5
PHONE_DIMS = (0.42, 0.84, 0.07)


def turntable_angle(t):
    """Rate and angle of the turntable at frame time t in [0, 1], played in real time.

    It spins up smoothly to TT_RATE rad/s and then holds it. The angle is the integral of the rate.
    """
    def table():
        u = np.linspace(0, 1, 2001)
        rate = TT_RATE * np.array([ease(phase(x, 0.08, 0.30)) for x in u])
        seconds = u * (TT_FRAMES - 1) / TT_FPS
        angle = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(seconds))])
        return u, rate, angle

    u, rate, angle = cached("turntable", table)
    return float(np.interp(t, u, rate)), float(np.interp(t, u, angle))


def phone(s, center, R, zorder):
    """A phone: body x across the screen, y along its long side, z out of the screen."""
    cols = {(2, 1): PERIWINKLE, (2, -1): SAND, (0, 1): PAPER, (0, -1): PAPER, (1, 1): PAPER, (1, -1): PAPER}
    seen = box(s, center, R, PHONE_DIMS, cols, zorder=zorder)
    if (2, 1) in seen:
        fc = seen[(2, 1)][0] + 0.002 * R[:, 2]
        ax, ay = 0.16 * R[:, 0], 0.30 * R[:, 1]
        o = fc + 0.04 * R[:, 1]
        s.patch([o + ax + ay, o - ax + ay, o - ax - ay, o + ax - ay], color=ICE, alpha=1.0, lw=0.5, zorder=zorder + 3)
        ring = np.array([fc - 0.345 * R[:, 1] + 0.03 * (np.cos(a) * R[:, 0] + np.sin(a) * R[:, 1]) for a in np.linspace(0, 2 * np.pi, 24)])
        s.curve(ring, color=INK, lw=0.5, alpha=1.0, zorder=zorder + 3.2)


def turntable_frame(t):
    s = Scene(width=6.8, height=3.7, azim=-38, elev=24, xlim=(-3.3, 3.3), ylim=(-1.95, 1.641))
    rate, angle = turntable_angle(t)
    spin = rot(E3, angle)
    omega = rate * E3                                           # world-frame angular velocity of both phones
    axis_cols = (TERRACOTTA, TEAL, INDIGO)
    cases = [                                                   # center, attitude at rest, where it sits, name
        (-1.62 * s.right, rot(E3, np.radians(25)), np.array([0.11, -0.43, 0.5 * PHONE_DIMS[2]]), "lying flat"),
        (1.62 * s.right, rot(E3, np.radians(12)) @ rot(E1, np.pi / 2),
         0.56 * rot(E3, np.radians(12)) @ E1 + 0.5 * PHONE_DIMS[1] * E3, "standing upright"),
    ]
    for c, R0, seat, name in cases:
        # the platter, with a mark that shows it turning
        rad, thick = 0.95, 0.07
        circ = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 90)])
        s.shadow(c - 0.25 * E3, 1.0, strength=0.12)
        s.patch(c - thick * E3 + rad * circ, color=SAND, alpha=1.0, lw=0.9, zorder=11)
        for sgn in (1, -1):
            edge = c + sgn * rad * s.right
            s.curve(np.array([edge, edge - thick * E3]), color=INK, lw=0.9, alpha=0.8, zorder=11.5)
        s.patch(c + rad * circ, color=SAND, alpha=1.0, lw=0.9, zorder=12)
        for k in range(12):
            d = spin @ np.array([np.cos(k * np.pi / 6), np.sin(k * np.pi / 6), 0.0])
            s.curve(np.array([c + (0.80 if k else 0.62) * rad * d, c + 0.93 * rad * d]), color=INK if k == 0 else MUTED,
                    lw=1.4 if k == 0 else 0.6, alpha=1.0, zorder=13)
        # the phone rides on the platter, off the spindle
        R = spin @ R0
        ctr = c + spin @ seat
        near = (ctr - c) @ s.view > 0
        zp = 44 if near else 22
        phone(s, ctr, R, zp)
        for i, nm in enumerate(("$x$", "$y$", "$z$")):
            tip = axis_through_box(s, ctr, R, PHONE_DIMS, R[:, i], 0.5, axis_cols[i], zorder=zp, lw=1.3, head=0.09)
            if np.hypot((tip - ctr) @ s.right, (tip - ctr) @ s.up) > 0.2 and R[:, i] @ s.view > -0.5:
                s.label(tip, nm, color=axis_cols[i], size=11, **outward(s, ctr, tip, 8))
        # world-frame angular velocity, on the spindle
        if rate > 0.02 * TT_RATE:
            tip = c + 2.1 * rate * E3
            s.arrow(c, tip, color=INK, lw=1.9, zorder=33)
            if rate > 0.3 * TT_RATE:
                s.label(tip, r"$\omega$", dx=-9, dy=2, size=12.5)
        s.dot(c, color=INK, size=3.2, zorder=33)
        s.label(c + 1.26 * s.up, name, size=10, color=MUTED, **SANS)
        # the gyroscope: Omega = R^T omega, component by component
        Omega = R.T @ omega
        cx = c @ s.right
        base = -1.33
        s.text2d(cx - 0.92, base + 0.21, "gyroscope\nreads $\\Omega$", fontsize=9.5, color=MUTED, linespacing=1.25, **SANS)
        s.ax.plot([cx - 0.36, cx + 0.92], [base, base], color=INK, lw=0.8, zorder=40)
        for i, nm in enumerate(("$x$", "$y$", "$z$")):
            bx = cx - 0.12 + 0.42 * i
            top = base + 0.9 * Omega[i]
            s.ax.fill([bx - 0.1, bx + 0.1, bx + 0.1, bx - 0.1], [base, base, top, top], color=axis_cols[i], zorder=41, lw=0)
            s.text2d(bx, base - 0.15, nm, fontsize=10.5, color=axis_cols[i])
            s.text2d(bx, max(top, base) + 0.13, f"{abs(Omega[i]) + 0.0:.2f}", fontsize=8.5, color=axis_cols[i], **SANS)
    s.text2d(0.0, 1.5, f"world frame, both phones:  $\\omega$ = (0, 0, {rate:.2f}) rad/s", fontsize=10, color=INK, **SANS)
    if rate <= 0:
        msg, col = "the same phone on two turntables, both at rest", MUTED
    elif rate < TT_RATE - 1e-9:
        msg, col = "both turntables spin up about the vertical", INK
    else:
        msg, col = "one spin, two readings: along the phone's $z$ when flat, along its $y$ when upright", INK
    caption(s, 0.0, -1.78, msg, color=col)
    return s


def turntable():
    animate(turntable_frame, OUT / "ch02-turntable.gif", OUT / "ch02-turntable-strip.png", n=TT_FRAMES, fps=TT_FPS,
            width_px=720, stills=(0.0, 0.3, 0.52, 1.0), strip_px=900, strip_cols=2, hold=1.5)


# ------------------------------------------- exp map: circle and sphere (animation)
def expmap_frame(t):
    s = Scene(width=6.6, height=3.5, azim=-30, elev=22, xlim=(-5.12, 1.72), ylim=(-1.86, 1.767))
    k = ease(phase(t, 0.12, 0.86))                       # fraction of each tangent vector rolled onto the manifold
    ks = np.linspace(0, k, 40)
    # (a) circle in the camera plane, tangent line at the top
    c = -3.25 * s.right - 0.1 * s.up

    def P(x, y):
        return c + x * s.right + y * s.up

    ring = np.array([P(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 240)])
    s.curve(ring, color=INK, lw=1.7, alpha=1.0, zorder=20)
    s.curve(np.array([P(-1.55, 1), P(1.55, 1)]), color=INK, lw=0.8)
    for th, col in [(1.3, INDIGO), (-0.8, TEAL)]:
        # Roll the segment onto the circle: the part already rolled is an arc of the same length,
        # the rest is still straight and tangent where it touches.
        def roll(kk):
            return P(np.sin(th * kk) + (1 - kk) * th * np.cos(th * kk), np.cos(th * kk) - (1 - kk) * th * np.sin(th * kk))

        s.curve(np.array([P(0, 1.0), P(th, 1.0)]), color=col, lw=0.9, alpha=1.0, linestyle=DASH, zorder=21)
        s.dot(P(th, 1.0), color=col, size=4.5, hollow=True)
        arc = np.array([P(np.sin(a), np.cos(a)) for a in np.linspace(0, th * k, 60)])
        s.curve(arc, color=col, lw=2.6, alpha=1.0, zorder=22)
        s.curve(np.array([P(np.sin(th * k), np.cos(th * k)), roll(k)]), color=col, lw=2.6, alpha=1.0, zorder=22)
        if k > 0:
            s.curve(np.array([roll(kk) for kk in ks]), color=col, lw=0.8, hidden=True)
        s.dot(roll(k), color=col, size=4.5)
    s.dot(P(0, 1), color=INK, size=5)
    s.label(P(0, 1), "identity", dy=12, size=9.5, **SANS)
    s.label(P(1.3, 1.0), r"$\theta$", dx=4, dy=11, color=INDIGO, size=12)
    s.label(P(-1.1, 1), "tangent line", dx=0, dy=11, size=9.5, color=MUTED, **SANS)
    s.label(P(0, 0), r"$\mathrm{SO}(2)$", size=12)
    if k >= 1:
        s.label(P(np.sin(1.3), np.cos(1.3)), r"$\exp(\theta \Lambda)$", dx=-32, dy=-6, color=INDIGO)

    # (b) sphere with a tangent plane at q
    q = unit([0.35, -0.55, 0.76])
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    a, b = s.tangent_plane(1.002 * q, size=0.9, color=SAND, alpha=0.6)
    for ang, length, col, name, off in [(3.5, 1.2, INDIGO, r"$v$", (9, 7)), (1.2, 0.75, TEAL, None, None)]:
        d = np.cos(ang) * a + np.sin(ang) * b
        v = length * d

        def roll(kk):
            return (np.cos(length * kk) * q + np.sin(length * kk) * d
                    + (1 - kk) * length * (-np.sin(length * kk) * q + np.cos(length * kk) * d))

        touch = np.cos(length * k) * q + np.sin(length * k) * d
        if k > 0:
            s.arrow(q, q + v, color=col, lw=0.9, head=0.08, dashed=True, zorder=29)
            s.sphere_curve(geodesic(q, touch), color=col, lw=2.2, zorder=16, alpha=1.0)
            s.curve(np.array([roll(kk) for kk in ks]), color=col, lw=0.8, hidden=True, zorder=25)
        if k < 1:
            s.arrow(touch, roll(k), color=col, zorder=31)
        s.dot(roll(k), color=col, size=4.5)
        if name:
            s.label(q + v, name, dx=off[0], dy=off[1], color=col, size=12)
            if k >= 1:
                s.label(touch, r"$\exp_q(v)$", dx=-30, dy=-8, color=col)
    s.dot(q, color=INK, size=5)
    s.label(q, r"$q$", dx=-4, dy=12, size=12)
    s.label(q - 0.9 * (a - b), r"$T_qS^2$", dx=-18, dy=4, size=10.5)
    s.text2d(1.2, -1.02, r"$S^2$", fontsize=13)
    if k <= 0:
        msg, col = "tangent vectors at the base point: straight segments in a flat space", MUTED
    elif k < 1:
        msg, col = "roll each one onto the manifold without stretching it", INDIGO
    else:
        msg, col = "the exponential map: the same lengths, now measured along the manifold", INDIGO
    caption(s, -1.7, -1.68, msg, color=col)
    return s


def expmap():
    animate(expmap_frame, OUT / "ch02-expmap.gif", OUT / "ch02-expmap-strip.png", n=84, fps=12,
            width_px=720, stills=(0.0, 0.42, 0.62, 1.0), strip_px=900, strip_cols=2, hold=2.0)


# ---------------------------------------- two quarter turns, two orders (animation)
BOOK_DIMS = (0.80, 1.10, 0.22)


def book_relative():
    """Axis and angle of the single rotation, about an axis fixed to the desk, that carries the
    result of "x then z" onto the result of "z then x"."""
    Ax, Az = rot(E1, np.pi / 2), rot(E3, np.pi / 2)
    G = (Ax @ Az) @ (Az @ Ax).T
    angle = np.arccos(np.clip((np.trace(G) - 1) / 2, -1, 1))
    axis = np.array([G[2, 1] - G[1, 2], G[0, 2] - G[2, 0], G[1, 0] - G[0, 1]]) / (2 * np.sin(angle))
    return axis, angle


def noncommute_frame(t):
    s = Scene(width=6.8, height=3.6, azim=-62, elev=24, xlim=(-3.85, 2.45), ylim=(-1.83, 1.505))
    a1 = 0.5 * np.pi * ease(phase(t, 0.07, 0.29))
    a2 = 0.5 * np.pi * ease(phase(t, 0.36, 0.58))
    n, total = book_relative()
    a3 = total * ease(phase(t, 0.70, 0.95))
    cL, cR = -1.3 * s.right + 0.05 * s.up, 1.3 * s.right + 0.05 * s.up
    RL = rot(n, a3) @ rot(E3, a2) @ rot(E1, a1)          # x first, then z, about axes fixed to the desk
    RR = rot(E1, a2) @ rot(E3, a1)                        # z first, then x

    def turning(c, R, axis, angle, name, col, full):
        """The axis of the turn in progress, through the book, with an arc that grows with the angle."""
        tip = axis_through_box(s, c, R, BOOK_DIMS, axis, 1.0, col, lw=1.3)
        s.label(tip, name, dx=9 if axis @ s.right > 0.5 else 8, dy=-2 if axis @ s.right > 0.5 else 5, color=col, size=11.5)
        if angle > 0.02 * full:
            side = unit(np.cross(axis, s.view))
            arc_arrow(s, c + 0.78 * axis, axis, side, angle, 0.26, col, head=0.09, lw=1.2)
        return tip

    for c, R in ((cL, RL), (cR, RR)):
        s.shadow(c - 0.72 * E3, 0.62, strength=0.14)
        book(s, c, R)
    if 0.07 <= t < 0.33:
        turning(cL, RL, E1, a1, "$x$", INDIGO, np.pi / 2)
        turning(cR, RR, E3, a1, "$z$", TERRACOTTA, np.pi / 2)
        msg, col = "first turn, 90°: the left book about $x$, the right book about $z$", INK
    elif 0.36 <= t < 0.62:
        turning(cL, RL, E3, a2, "$z$", INDIGO, np.pi / 2)
        turning(cR, RR, E1, a2, "$x$", TERRACOTTA, np.pi / 2)
        msg, col = "second turn, 90°: the left book about $z$, the right book about $x$", INK
    elif t < 0.07:
        msg, col = "two copies of one book: cover up, spine on the left", MUTED
    elif t < 0.36:
        msg, col = "after the first turn: left about $x$, right about $z$", MUTED
    elif t < 0.70:
        msg, col = "after both turns: the same two turns, two different attitudes", INK
    else:
        tip = turning(cL, RL, n, a3, "", TEAL, total)
        s.label(tip, "120°", dx=17, dy=-3, color=TEAL, size=10, **SANS)
        msg, col = "one turn of 120° about this axis carries the left result onto the right one", TEAL
    s.label(cL + 1.22 * s.up, "$x$ first, then $z$", color=INDIGO, size=10.5, **SANS)
    s.label(cR + 1.22 * s.up, "$z$ first, then $x$", color=TERRACOTTA, size=10.5, **SANS)
    # axes fixed to the desk, drawn once
    o = -3.3 * s.right - 0.25 * s.up
    for e, name, off in [(E1, "$x$", (7, -3)), (E2, "$y$", (6, 5)), (E3, "$z$", (0, 9))]:
        s.arrow(o, o + 0.5 * e, color=INK, lw=1.1, head=0.09)
        s.label(o + 0.5 * e, name, dx=off[0], dy=off[1], size=11)
    s.label(o, "axes fixed\nto the desk", dy=-22, size=9, color=MUTED, linespacing=1.2, **SANS)
    caption(s, -0.2, -1.62, msg, color=col)
    return s


def noncommute():
    animate(noncommute_frame, OUT / "ch02-noncommute.gif", OUT / "ch02-noncommute-strip.png", n=110, fps=12,
            width_px=720, stills=(0.0, 0.33, 0.64, 0.86), strip_px=900, strip_cols=2, hold=2.0)


# ------------------------------------- quadrotor with a cable-suspended load
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

    # airframe, drawn as in every chapter
    arm = 0.62
    quadrotor(s, xQ, R, arm=arm)
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
    s.label(o, "world frame", dx=-6, dy=-12, size=9.5, color=MUTED, **SANS)
    # annotations
    s.label(xQ + 0.62 * q, r"$q \in S^2$", dx=30, dy=2, color=TEAL, size=12)
    s.label(xL, r"$x_L \in \mathbb{R}^3$", dx=36, dy=-6, color=TERRACOTTA, size=12)
    s.label(xQ + arm * R[:, 0], r"$R \in \mathrm{SO}(3)$", dx=44, dy=6, color=INDIGO, size=12)
    s.label(xQ - arm * R[:, 0], r"$x_Q = x_L - l\,q$", dx=-52, dy=-3, size=11)
    mid = xQ + 0.5 * L * q
    s.label(xQ + 0.8 * L * q, r"$l$", dx=-9, dy=2, size=11.5)
    s.text2d(2.75, 1.05, "attitude: 3", fontsize=9.5, color=INDIGO, **SANS)
    s.text2d(2.75, 0.75, "cable direction: 2", fontsize=9.5, color=TEAL, **SANS)
    s.text2d(2.75, 0.45, "load position: 3", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.text2d(2.75, 0.10, "8 degrees of freedom", fontsize=10, color=INK, **SANS)
    s.project(-3.6 * s.right)
    s.text2d(-2.6, 1.0, "the load stays on a sphere\nof radius $l$ about the quadrotor", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.save(OUT / "ch02-quadload.svg", pad=0.1)


# ------------------------------------- Euler steps vs exponential steps (animation)
def steps_frame(t):
    fig, ax = new_plot(6.4, 3.3)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    arcs = np.linspace(-0.1, np.pi / 2 + 0.5, 300)
    ax.plot(np.cos(arcs), np.sin(arcs), color=INK, lw=1.7, zorder=3)
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    hw, n = 0.42, 4                           # h * omega, exaggerated so the drift is visible
    q0 = np.array([np.cos(0.12), np.sin(0.12)])
    qe, qx = [q0], [q0]
    for _ in range(n):
        qe.append(qe[-1] + hw * J @ qe[-1])
        c, sn = np.cos(hw), np.sin(hw)
        qx.append(np.array([[c, -sn], [sn, c]]) @ qx[-1])
    qe, qx = np.array(qe), np.array(qx)
    progress = n * phase(t, 0.06, 0.90)
    done = int(np.floor(progress + 1e-9))     # steps completed
    for k in range(n):
        f = ease(np.clip(progress - k, 0.0, 1.0))
        if f <= 0:
            break
        tip = qe[k] + f * (qe[k + 1] - qe[k])
        ax.annotate("", tip, xytext=qe[k], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=TERRACOTTA, lw=1.6, mutation_scale=11, shrinkA=0, shrinkB=0))
        a0 = 0.12 + k * hw
        arc = np.linspace(a0, a0 + f * hw, 30)
        ax.plot(np.cos(arc), np.sin(arc), color=INDIGO, lw=3.2, solid_capstyle="butt", zorder=4)
        if f < 1:
            ax.plot(np.cos(arc[-1]), np.sin(arc[-1]), "o", ms=5, color=INDIGO, zorder=7)
        else:
            ax.plot([0, qe[k + 1][0]], [0, qe[k + 1][1]], color=TERRACOTTA, lw=0.6, ls=(0, (3.5, 3)), alpha=1.0, zorder=2)
    ax.plot(qx[: done + 1, 0], qx[: done + 1, 1], "o", ms=5, color=INDIGO, zorder=7)
    ax.plot(qe[1: done + 1, 0], qe[1: done + 1, 1], "o", ms=5, mfc=PAPER, mec=TERRACOTTA, mew=1.2, zorder=7)
    ax.plot(*q0, "o", ms=6, color=INK, zorder=8)
    ax.plot(0, 0, "o", ms=3, color=MUTED)
    kw = dict(path_effects=halo(), zorder=9)
    sans = dict(fontfamily="Source Sans 3", fontsize=9.5, linespacing=1.3, **kw)
    ax.text(q0[0] + 0.09, q0[1] - 0.02, r"$q_0$", fontsize=12, ha="left", va="center", **kw)
    if done >= 1:
        ax.text(qe[1][0] + 0.07, qe[1][1] + 0.03, r"$q_0 + h\,\dot q_0$", fontsize=11, color=TERRACOTTA, ha="left", va="center", **kw)
    count = ["", " after one step", " after two steps", " after three steps", " after four steps"][done]
    ax.text(-1.8, 1.62, f"Euler step: radius {np.linalg.norm(qe[done]):.2f}{count}", color=TERRACOTTA, ha="left", va="center", **sans)
    ax.text(-1.8, 1.44, f"exponential step: radius {np.linalg.norm(qx[done]):.2f}", color=INDIGO, ha="left", va="center", **sans)
    ax.text(1.05, 1.3, "each Euler step runs along\nthe tangent, so it always\nlands outside the circle", color=MUTED, ha="left", va="center", **sans)
    if done == 0 and progress <= 0:
        msg, col = "two ways to step along the circle from $q_0$", MUTED
    elif done < n:
        msg, col = "Euler steps leave along the tangent. Exponential steps follow the circle", INK
    else:
        msg, col = "after four steps the Euler radius has grown from 1 to 1.38", TERRACOTTA
    ax.text(0.2, -0.3, msg, color=col, ha="center", va="center", fontfamily="Source Sans 3", fontsize=10, **kw)
    ax.set_xlim(-1.9, 2.3)
    ax.set_ylim(-0.42, 1.746)
    return fig


def steps():
    animate(steps_frame, OUT / "ch02-steps.gif", OUT / "ch02-steps-strip.png", n=84, fps=12,
            width_px=700, stills=(0.0, 0.32, 0.6, 1.0), strip_px=880, strip_cols=2, hold=2.0)


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


# ------------------------------------------------------ animation: combing
def _fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * k / n)
    azim = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.sin(polar) * np.cos(azim), np.sin(polar) * np.sin(azim), np.cos(polar)], axis=1)


def _comb_field(P, c, blend):
    """Tangent field on the unit sphere at the points P (rows).

    blend = 0: the sphere combed flat toward the direction c. It vanishes at +c and -c.
    blend = 1: a field with a single zero, at +c (a constant field on the plane,
               carried to the sphere by stereographic projection from c).
    """
    two = c - (P @ c)[:, None] * P
    d = unit(np.cross(c, E3))
    one = (1 - P @ c)[:, None] * d + (P @ d)[:, None] * (c - P)
    return (1 - blend) * two + blend * 1.4 * one


def _cowlicks(c, blend):
    """Zeros of the field, found numerically: coarse search, then local refinement."""
    P = _fibonacci_sphere(6000)
    mag = np.linalg.norm(_comb_field(P, c, blend), axis=1)
    found = []
    for _ in range(3):
        i = int(np.argmin(mag))
        if mag[i] > 0.08:
            break
        p = P[i]
        for radius in (0.06, 0.02, 0.006):
            a = unit(np.cross(p, [0.3, 0.5, 0.8]))
            b = np.cross(p, a)
            g = np.linspace(-radius, radius, 15)
            cand = np.array([unit(p + x * a + y * b) for x in g for y in g])
            p = cand[np.argmin(np.linalg.norm(_comb_field(cand, c, blend), axis=1))]
        if all(np.arccos(np.clip(p @ f, -1, 1)) > 0.25 for f in found):
            found.append(p)
        mag[np.arccos(np.clip(P @ P[i], -1, 1)) < 0.45] = np.inf
    return found


def comb_frame(t):
    from figkit import ease, phase

    s = Scene(width=6.6, height=3.5, azim=-30, elev=20, xlim=(-4.05, 1.75), ylim=(-1.62, 1.46))
    # --- left: a group. Carry one arrow around the circle by multiplication.
    ctr = -2.75 * s.right
    ring = ctr + np.array([np.cos(a) * s.right + np.sin(a) * s.up for a in np.linspace(0, 2 * np.pi, 200)])
    s.curve(ring, color=INK, lw=1.7, alpha=1.0, zorder=20)
    sweep = 2 * np.pi * ease(phase(t, 0.0, 0.32))

    def on_ring(a, r=1.0):
        return ctr + r * (np.cos(a) * s.right + np.sin(a) * s.up)

    def tangent_arrow(a, color, lw=1.3, alpha=1.0):
        tip = on_ring(a, 1.17) + 0.36 * (-np.sin(a) * s.right + np.cos(a) * s.up)
        s.arrow(on_ring(a, 1.17), tip, color=color, lw=lw, head=0.11, alpha=alpha)

    for a in np.arange(0, 2 * np.pi - 1e-9, np.pi / 6):
        if a <= sweep + 1e-9:
            tangent_arrow(a, INDIGO)
    if 0 < sweep < 2 * np.pi - 1e-6:
        tangent_arrow(sweep, TERRACOTTA, lw=1.7)
        s.dot(on_ring(sweep), color=TERRACOTTA, size=6)
    s.dot(on_ring(0.0), color=INK, size=5)
    s.label(on_ring(0.0), "identity", dx=-27, dy=0, size=9.5, **SANS)
    s.label(ctr, r"$S^1$", size=13, dy=10)
    s.label(ctr, "a group", size=9.5, dy=-9, color=MUTED, **SANS)
    done = sweep >= 2 * np.pi - 1e-6
    s.text2d(ctr @ s.right, -1.44, "arrows everywhere, none of them zero" if done else "carry one arrow by multiplication",
             fontsize=10, color=INDIGO, **SANS)

    # --- right: a sphere. Comb it, and watch the cowlicks.
    comb = ease(phase(t, 0.32, 0.62))
    blend = ease(phase(t, 0.62, 0.97))
    c = unit(rot(E3, -0.2 + 1.5 * comb) @ rot(E2, 0.35 * np.sin(np.pi * comb)) @ unit([-0.1, -0.85, 0.5]))
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    if t >= 0.30:
        P = _fibonacci_sphere(300)
        V = _comb_field(P, c, blend)
        for p, v in zip(P, V):
            n = np.linalg.norm(v)
            if n < 0.05 or not s.point_visible(p) or p @ s.view < 0.12:
                continue
            step = 0.17 * min(1.0, n / 0.75) * v / n
            s.arrow(1.004 * p, 1.004 * unit(p + step), color=INDIGO, lw=1.0, head=0.055, zorder=30)
        zeros = _cowlicks(c, blend)
        for z in zeros:
            vis = s.point_visible(z)
            s.dot(1.004 * z, color=TERRACOTTA, size=7, hollow=not vis, zorder=45)
        nvis = [z for z in zeros if s.point_visible(z)]
        if nvis:
            s.label(1.004 * nvis[0], "cowlick", dx=0, dy=13, size=9.5, color=TERRACOTTA, **SANS)
        if blend <= 0:
            msg = "comb it flat: one cowlick here, one on the far side"
        elif len(zeros) > 1:
            msg = "comb differently: the cowlicks move and merge"
        else:
            msg = "you can merge them. You cannot remove them"
        s.text2d(0.0, -1.44, msg, fontsize=10, color=TERRACOTTA, **SANS)
    else:
        s.text2d(0.0, -1.44, "now try the same on a sphere", fontsize=10, color=MUTED, **SANS)
    s.text2d(1.42, 1.2, r"$S^2$", fontsize=13)
    return s


def comb():
    from figkit import animate

    animate(comb_frame, OUT / "ch02-comb.gif", OUT / "ch02-comb-strip.png", n=96, fps=14,
            width_px=760, stills=(0.16, 0.45, 0.8, 1.0), strip_px=900, strip_cols=2)


if __name__ == "__main__":
    figures = dict(puzzle=puzzle, tangent=tangent, flat=flat, turntable=turntable, expmap=expmap, noncommute=noncommute,
                   quadload=quadload, steps=steps, drift=drift, comb=comb)
    for name in sys.argv[1:] or figures:
        figures[name]()
