"""Figures for Chapter 1.  Run: python figures/src/ch01.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, OCHRE, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, geodesic, great_circle, halo, hat, new_plot, phase, rot, save,
                    small_circle, unit)
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")


# ------------------------------------------------------------ shared helpers
def flat_scene(width, height, xlim, ylim):
    """A Scene that looks straight at the x-z plane: the 3D point (x, 0, y) lands at canvas (x, y)."""
    return Scene(width=width, height=height, azim=-90, elev=0, xlim=xlim, ylim=ylim)


def P(x, y):
    """Canvas point (x, y) of a flat_scene, as the 3D point the kit expects."""
    return np.array([x, 0.0, y])


def arc2(center, radius, a0, a1, n=120):
    """Arc in the canvas plane of a flat_scene. Angles in radians, counterclockwise from +x."""
    a = np.linspace(a0, a1, max(2, int(n)))
    return np.stack([center[0] + radius * np.cos(a), np.zeros_like(a), center[1] + radius * np.sin(a)], axis=1)


def tint(color, shade):
    """A pastel fill, darkened a little where it turns away from the light."""
    base = np.array(to_rgb(color))
    return tuple(base * (0.80 + 0.20 * shade))


def world_light(s):
    return s.light_cam[0] * s.right + s.light_cam[1] * s.up + s.light_cam[2] * s.view


def paint(s, pieces, zorder=30.0):
    """Painter's algorithm: draw (depth, draw(z)) pieces from far to near."""
    pieces.sort(key=lambda piece: piece[0])
    for k, (_, draw) in enumerate(pieces):
        draw(zorder + 1e-3 * k)


def poly(s, corners, fc, z, ec=None, lw=0.0, alpha=1.0):
    xy = s.project(corners)
    s.ax.add_patch(Polygon(xy, closed=True, fc=fc, ec=ec if ec else fc, lw=lw if ec else 0.3, alpha=alpha,
                           zorder=z, joinstyle="round"))


def box_pieces(s, center, R, dims, face_colors, lw=0.9):
    """Shaded convex box as painter pieces: only the faces that look at the camera."""
    c, half, out = np.asarray(center, float), 0.5 * np.asarray(dims, float), []
    light = world_light(s)
    for axis in range(3):
        for sign in (1, -1):
            n = sign * R[:, axis]
            if n @ s.view <= 1e-9:
                continue
            a, b = [(half[k] * R[:, k]) for k in range(3) if k != axis]
            fc = c + half[axis] * n
            col = tint(face_colors[(axis, sign)], max(n @ light, 0.0))
            out.append(((fc @ s.view), lambda z, fc=fc, a=a, b=b, col=col: poly(
                s, [fc + a + b, fc - a + b, fc - a - b, fc + a - b], col, z, ec=INK, lw=lw)))
    return out


def circle_in_plane(a, b, radius, n=160):
    t = np.linspace(0, 2 * np.pi, n)
    return radius * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))


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
    ax.text(4, 2.6, r"yaw rate $\dot\psi$ reported by the Euler chart", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax.text(4, 0.36, r"what the body is doing: $\|\Omega\| = 0.71$ rad/s", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax.set_xlim(0, 90)
    ax.set_ylim(0.22, 600)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("pitch angle θ (degrees)", fontfamily="Source Sans 3", fontsize=9.5)
    ax.set_ylabel("rad/s", fontfamily="Source Sans 3", fontsize=9.5)
    save(fig, OUT / "ch01-euler-rates.svg")


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
    s.label(0.62 * u, r"$\theta\,a$", dx=14, dy=-2, color=INDIGO)
    s.dot(p_exit, color=TERRACOTTA)
    s.dot(p_enter, color=TERRACOTTA, hollow=not s.point_visible(p_enter))
    s.label(p_exit, r"$\pi a$", dx=13, dy=7, color=TERRACOTTA)
    s.label(p_enter, r"$-\pi a$", dx=-17, dy=-8, color=TERRACOTTA)
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
    s.label(c + s.up, "target", dy=13, size=10, color=INDIGO, fontfamily="Source Sans 3")
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


# ------------------------------------------------- animation: the plate trick
def quat_to_R(q):
    """Rotation matrix of the unit quaternion q = (q0, qv), equation eq-quat-to-R."""
    return np.eye(3) + 2 * q[0] * hat(q[1:]) + 2 * hat(q[1:]) @ hat(q[1:])


def arm_quaternions(turn, swing, sigma):
    """Quaternions along an arm from a fixed base (quaternion +1) to a level plate turned by `turn`.

    The plate's quaternion is (cos(turn/2), 0, 0, sin(turn/2)). Together with +1 it
    lies on the two-sphere of quaternions w + x i + z k. The arm follows the arc
    from +1 to the plate's quaternion on the circle cut from that sphere by a plane
    through the two points. `swing` turns the plane about the chord between them:
    swing = 0 is uniform twist about the vertical, and swing = pi is the arc that
    goes the other way round. For a turn of 4 pi and swing = pi the arc is a point:
    the arm is untwisted. `sigma` in [0, 1] is the position along the arm.
    """
    north = np.array([1.0, 0.0, 0.0])                          # coordinates (w, z, x)
    plate = np.array([np.cos(turn / 2), np.sin(turn / 2), 0.0])
    chord = plate - north
    if np.linalg.norm(chord) < 1e-9:
        pts = np.tile(north, (len(sigma), 1))
    else:
        chord = chord / np.linalg.norm(chord)
        n0 = np.array([0.0, 0.0, 1.0])
        n = np.cos(swing) * n0 + np.sin(swing) * np.cross(chord, n0)
        center = (n @ north) * n
        ea = north - center
        radius = np.linalg.norm(ea)
        ea = ea / radius
        eb = np.cross(n, ea)
        ang = np.arctan2((plate - center) @ eb, (plate - center) @ ea) % (2 * np.pi)
        pts = center + radius * (np.outer(np.cos(ang * sigma), ea) + np.outer(np.sin(ang * sigma), eb))
    return np.stack([pts[:, 0], pts[:, 2], np.zeros(len(pts)), pts[:, 1]], axis=1)   # (q0, q1, q2, q3)


PLATE_LEN, PLATE_STIFF = 1.75, (0.08, 0.36)      # ribbon length; rigid fractions at the base and at the plate


def plate_arm(turn, swing, n=220):
    """Frames and centerline of the ribbon. Its tangent is the third axis of each frame."""
    s = np.linspace(0, 1, n)
    a, b = PLATE_STIFF
    sigma = np.clip((s - a) / (1 - a - b), 0, 1)
    F = np.array([quat_to_R(q) for q in arm_quaternions(turn, swing, sigma)])
    T = F[:, :, 2]
    X = np.vstack([np.zeros(3), np.cumsum(0.5 * (T[1:] + T[:-1]), axis=0)]) * PLATE_LEN / (n - 1)
    return X, F


def plate_schedule(t):
    """Timeline: (turn of the plate, swing of the arm, caption)."""
    first = ease(phase(t, 0.05, 0.36))
    second = ease(phase(t, 0.50, 0.90))
    turn = 2 * np.pi * (first + second)
    swing = np.pi * second
    if t < 0.05:
        msg = "a level plate on a ribbon: ribbon flat, quaternion at +1"
    elif t < 0.36:
        msg = "first full turn about the vertical"
    elif t < 0.50:
        msg = "one turn: plate back where it began, ribbon twisted, quaternion at −1"
    elif t < 0.90:
        msg = "second turn, same direction: the ribbon swings round the plate"
    else:
        msg = "two turns: ribbon flat again, quaternion back at +1"
    return turn, swing, msg


def plate_frame(t):
    turn, swing, msg = plate_schedule(t)
    s = Scene(width=6.6, height=3.7, azim=-64, elev=20, xlim=(-2.25, 3.25), ylim=(-1.42, 1.66))
    light = world_light(s)
    base = np.array([0.0, 0.0, -0.92])
    X, F = plate_arm(turn, swing)
    X = X + base
    pieces = []

    # pedestal and floor shadow
    s.shadow(base - 0.02 * E3, 0.62, strength=0.2)
    ped = [base + 0.2 * (c * E1 + d * E2) for c, d in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
    s.patch(ped, color=MUTED, alpha=0.9, zorder=12)

    # the ribbon: one face periwinkle, the other peach, so a twist shows
    half_w = 0.17
    for i in range(len(X) - 1):
        quad = [X[i] - half_w * F[i, :, 0], X[i] + half_w * F[i, :, 0],
                X[i + 1] + half_w * F[i + 1, :, 0], X[i + 1] - half_w * F[i + 1, :, 0]]
        nrm = F[i, :, 1]
        facing = nrm @ s.view
        col = tint(PERIWINKLE if facing < 0 else PEACH, abs(nrm @ light))
        depth = 0.5 * (X[i] + X[i + 1]) @ s.view

        def draw(z, quad=quad, col=col):
            poly(s, quad, col, z)
            for a, b in ((quad[0], quad[3]), (quad[1], quad[2])):
                xy = s.project([a, b])
                s.ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=0.9, solid_capstyle="round", zorder=z + 5e-4)
        pieces.append((depth, draw))

    # the plate: level, turned by `turn` about the vertical, carried by the end of the ribbon
    Rp = rot(E3, turn)
    c = X[-1] + 0.03 * E3
    th = np.linspace(0, 2 * np.pi, 90)
    rim = lambda r, dz=0.0: c + dz * E3 + r * (np.outer(np.cos(th), Rp[:, 0]) + np.outer(np.sin(th), Rp[:, 1]))  # noqa: E731
    wedge = np.radians(np.linspace(-13, 13, 12))
    mark = np.vstack([c + 0.2 * Rp[:, 0], c + 0.56 * (np.outer(np.cos(wedge), Rp[:, 0]) + np.outer(np.sin(wedge), Rp[:, 1]))])

    def draw_plate(z):
        poly(s, rim(0.56, -0.05), tint(SAND, 0.2), z, ec=INK, lw=1.0)
        poly(s, rim(0.56), tint(SAND, 1.0), z + 1e-4, ec=INK, lw=1.2)
        xy = s.project(rim(0.36))
        s.ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=0.6, alpha=0.6, zorder=z + 2e-4)
        poly(s, mark, TERRACOTTA, z + 3e-4)
    pieces.append((c @ s.view + 0.05, draw_plate))
    paint(s, pieces)

    # dials, drawn in the plane of the page
    def cam(x, y):
        return x * s.right + y * s.up

    def dial(cx, cy, r, angle, color, title, readout):
        a = np.linspace(0, 2 * np.pi, 200)
        s.curve(np.array([cam(cx + r * np.cos(u), cy + r * np.sin(u)) for u in a]), color=INK, lw=1.3, alpha=1.0)
        if angle > 1e-6:
            sweep = np.linspace(0, angle, max(8, int(60 * angle)))
            k = 0.5 * r / (4 * np.pi)                           # a spiral, so a second lap does not hide the first
            s.curve(np.array([cam(cx + (0.38 * r + k * u) * np.cos(u), cy + (0.38 * r + k * u) * np.sin(u)) for u in sweep]),
                    color=color, lw=1.6, alpha=1.0, zorder=22)
        s.curve(np.array([cam(cx, cy), cam(cx + r * np.cos(angle), cy + r * np.sin(angle))]), color=color, lw=1.2, alpha=1.0)
        s.dot(cam(cx + r * np.cos(angle), cy + r * np.sin(angle)), color=color, size=6.5, zorder=40)
        s.text2d(cx, cy + r + 0.2, title, fontsize=9.5, color=INK, **SANS)
        s.text2d(cx, cy - r - 0.2, readout, fontsize=9.5, color=color, **SANS)

    deg = np.degrees(turn)
    dial(2.3, 0.95, 0.38, turn, TERRACOTTA, "plate", f"turned {deg:.0f}°")
    dial(2.3, -0.40, 0.38, turn / 2, INDIGO, "quaternion", f"half angle {deg / 2:.0f}°")
    s.dot(cam(2.3 + 0.38, -0.40), color=INK, size=4.5, zorder=38)
    s.dot(cam(2.3 - 0.38, -0.40), color=INK, size=4.5, zorder=38, hollow=True)
    s.text2d(2.3 + 0.62, -0.40, r"$+\mathbf{1}$", fontsize=10.5)
    s.text2d(2.3 - 0.64, -0.40, r"$-\mathbf{1}$", fontsize=10.5)
    s.text2d(0.5, -1.27, msg, fontsize=10, color=INK, **SANS)
    return s


def plate():
    animate(plate_frame, OUT / "ch01-plate.gif", OUT / "ch01-plate-strip.png", n=110, fps=14,
            width_px=720, stills=(0.0, 0.43, 0.70, 1.0), strip_px=900, strip_cols=2)


# ------------------------------------------------ animation: the heading puzzle
def heading_frame(t):
    s = flat_scene(6.6, 3.4, xlim=(-3.3, 3.3), ylim=(-1.75, 1.65))
    C, r = np.array([-1.85, 0.12]), 1.0
    start, target = 350.0, 10.0
    long_p, short_p = ease(phase(t, 0.06, 0.48)), ease(phase(t, 0.62, 0.84))
    second_act = t >= 0.60

    def on(h, rr=r):                                   # compass heading: clockwise from north
        a = np.radians(90.0 - h)
        return P(C[0] + rr * np.cos(a), C[1] + rr * np.sin(a))

    def X(h):                                          # the same heading on a line from 0 to 360
        return 0.45 + 2.5 * h / 360.0

    # compass
    s.curve(arc2(C, r, 0, 2 * np.pi, 240), color=INK, lw=1.7, alpha=1.0, zorder=20)
    for h in range(0, 360, 30):
        s.curve(np.array([on(h, 1.0), on(h, 1.07 if h % 90 else 1.1)]), color=INK, lw=0.8, alpha=1.0)
    for h, lab in ((90, "90°"), (270, "270°")):
        s.label(on(h, 1.32), lab, size=9, color=MUTED, **SANS)
    s.label(on(0, 1.3), "north", size=9, color=MUTED, **SANS)
    s.dot(on(start), color=INK, size=5.5, hollow=True, zorder=42)
    s.dot(on(target), color=INK, size=5.5, zorder=42)
    s.label(on(start, 1.2), "start 350°", dx=-30, dy=-7, size=9.5, **SANS)
    s.label(on(target, 1.2), "target 10°", dx=31, dy=-7, size=9.5, **SANS)

    # line
    y0 = 0.42
    s.curve(np.array([P(X(0), y0), P(X(360), y0)]), color=INK, lw=1.7, alpha=1.0, zorder=20)
    for h in (0, 90, 180, 270, 360):
        s.curve(np.array([P(X(h), y0 - 0.07), P(X(h), y0 + 0.07)]), color=INK, lw=0.8, alpha=1.0)
        s.label(P(X(h), y0 - 0.24), f"{h}°", size=9, color=MUTED, **SANS)
    s.dot(P(X(start), y0), color=INK, size=5.5, hollow=True, zorder=42)
    s.dot(P(X(target), y0), color=INK, size=5.5, zorder=42)
    s.label(P(X(180), y0 + 0.78), "the same headings on a line", size=10, **SANS)

    # act 1: subtraction, the long way round
    h_long = start - 340.0 * long_p
    fade = 0.38 if second_act else 1.0
    if long_p > 0:
        s.curve(np.array([on(h, 0.86) for h in np.linspace(start, h_long, 200)]), color=TERRACOTTA, lw=2.2,
                alpha=fade, zorder=24)
        s.curve(np.array([P(X(start), y0 + 0.2), P(X(h_long), y0 + 0.2)]), color=TERRACOTTA, lw=2.2, alpha=fade, zorder=24)
        if long_p >= 1:
            s.arrow(P(X(target) + 0.2, y0 + 0.2), P(X(target), y0 + 0.2), color=TERRACOTTA, lw=0.1, head=0.14, alpha=fade)
            s.arrow(on(target + 9, 0.86), on(target, 0.86), color=TERRACOTTA, lw=0.1, head=0.14, alpha=fade)
        s.label(P(X(180), y0 + 0.42), "10° − 350° = −340°", size=10, color=TERRACOTTA,
                alpha=fade, **SANS)
    # act 2: on the circle, the short way through north
    h_short = start + 20.0 * short_p
    if second_act and short_p > 0:
        s.curve(np.array([on(h, 0.74) for h in np.linspace(start, h_short, 40)]), color=INDIGO, lw=2.6, alpha=1.0, zorder=25)
        s.curve(np.array([P(X(start), y0 - 0.5), P(X(min(h_short, 360.0)), y0 - 0.5)]), color=INDIGO, lw=2.6, alpha=1.0)
        if h_short > 360:
            s.curve(np.array([P(X(0), y0 - 0.5), P(X(h_short - 360), y0 - 0.5)]), color=INDIGO, lw=2.6, alpha=1.0)
    if second_act:
        # the two ends of the line are one heading
        a = np.linspace(0, np.pi, 80)
        mid, half = 0.5 * (X(0) + X(360)), 0.5 * (X(360) - X(0))
        s.curve(np.array([P(mid + half * np.cos(u), y0 - 0.5 - 0.62 * np.sin(u)) for u in a]), color=MUTED, lw=0.8,
                hidden=True, alpha=0.9)
        s.label(P(mid, y0 - 1.26), "0° and 360° are the same heading", size=9.5, color=MUTED, **SANS)

    # the robot, seen from above, and its marker on the line
    h_now = h_short if second_act else h_long
    col = INDIGO if second_act else TERRACOTTA
    body = arc2(C, 0.2, 0, 2 * np.pi, 60)
    s.patch(body, color=SAND, alpha=1.0, lw=1.1, zorder=31)
    s.arrow(P(*C), on(h_now, 0.66), color=INK, lw=1.7, head=0.15, zorder=33)
    s.dot(on(h_now), color=col, size=7, zorder=44)
    s.dot(P(X(h_now % 360.0), y0), color=col, size=7, zorder=44)
    turned = 20.0 * short_p if second_act else 340.0 * long_p
    s.label(P(C[0], C[1] - 1.3), f"turned {turned:.0f}°", size=9.5, color=col, **SANS)

    if t < 0.06:
        msg, mc = "heading 350°, target 10°: what is the error?", INK
    elif t < 0.48:
        msg, mc = "subtract the numbers and the robot swings the long way round", TERRACOTTA
    elif t < 0.60:
        msg, mc = "340° of turning to end up 20° from where it began", TERRACOTTA
    elif t < 0.84:
        msg, mc = "on the circle the target is 20° away, through north", INDIGO
    else:
        msg, mc = "20°, not 340°: headings live on a circle, and subtraction is for lines", INDIGO
    s.text2d(0.0, -1.55, msg, fontsize=10, color=mc, **SANS)
    return s


def heading():
    animate(heading_frame, OUT / "ch01-heading.gif", OUT / "ch01-heading-strip.png", n=96, fps=14,
            width_px=720, stills=(0.0, 0.30, 0.54, 1.0), strip_px=900, strip_cols=2)


# ---------------------------------------------------- animation: gimbal lock
def bump(p):
    """Out and back: 0 at p = 0 and p = 1, 1 in the middle, smooth at both ends."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.clip(p, 0.0, 1.0)))


GIMBAL_SWEEP = np.radians(65.0)


def gimbal_body(yaw, pitch, roll):
    return rot(E3, yaw) @ rot(E2, pitch) @ rot(E1, roll)


def gimbal_frame(t):
    s = Scene(width=6.6, height=3.9, azim=-58, elev=16, xlim=(-2.55, 3.65), ylim=(-1.98, 1.69))
    c = np.zeros(3)
    yaw0, roll0 = np.radians(-23), np.radians(40)
    pitch = np.radians(28 + 62 * ease(phase(t, 0.10, 0.36)))
    p_yaw, p_roll = phase(t, 0.42, 0.64), phase(t, 0.68, 0.90)
    yaw = yaw0 - GIMBAL_SWEEP * bump(p_yaw)
    roll = roll0 + GIMBAL_SWEEP * bump(p_roll)
    Rz, Ry = rot(E3, yaw), rot(E2, pitch)
    y1 = Rz @ E2                 # pitch axle, carried by the outer ring
    x2 = Rz @ Ry @ E1            # roll axle, carried by the middle ring
    Rb = gimbal_body(yaw, pitch, roll)
    rings = [
        (c + circle_in_plane(E3, y1, 1.00), INDIGO, 2.6),
        (c + circle_in_plane(y1, x2, 0.80), TEAL, 2.6),
        (c + circle_in_plane(x2, Rb @ E2, 0.60), TERRACOTTA, 2.6),
    ]
    axles = [
        (np.array([c - 1.42 * E3, c - 1.0 * E3]), INDIGO), (np.array([c + 1.0 * E3, c + 1.42 * E3]), INDIGO),
        (np.array([c - 1.0 * y1, c - 0.8 * y1]), TEAL), (np.array([c + 0.8 * y1, c + 1.0 * y1]), TEAL),
        (np.array([c - 0.8 * x2, c - 0.6 * x2]), TERRACOTTA), (np.array([c + 0.6 * x2, c + 0.8 * x2]), TERRACOTTA),
    ]
    s.ribbons(rings + [(np.linspace(a[0], a[1], 12), col, 1.3) for a, col in axles])
    faces = {(2, 1): SAND, (2, -1): SAND, (0, 1): PEACH, (0, -1): PEACH, (1, 1): PERIWINKLE, (1, -1): PERIWINKLE}
    paint(s, box_pieces(s, c, Rb, (0.5, 0.42, 0.16), faces), zorder=40)
    tip = 0.5
    s.arrow(c + 0.21 * Rb @ E2, c + tip * Rb @ E2, color=INK, lw=1.4, head=0.1, zorder=43)
    s.shadow(c - 1.5 * E3, 0.95, strength=0.16)
    s.label(c + 1.56 * E3, "yaw", color=INDIGO, size=9.5, **SANS)
    s.label(c + 1.0 * y1, "pitch", dx=17, dy=3, color=TEAL, size=9.5, **SANS)
    s.label(c + 0.7 * x2, "roll", dx=19, dy=-6, color=TERRACOTTA, size=9.5, **SANS)

    # where each command carries the tip of the arrow fixed on the body
    locked = np.radians(90)
    if p_yaw > 0:
        reach = bump(p_yaw) if p_yaw < 0.5 else 1.0
        arc = [c + tip * gimbal_body(yaw0 - GIMBAL_SWEEP * u, locked, roll0) @ E2 for u in np.linspace(0, reach, 40)]
        s.curve(np.array(arc), color=INDIGO, lw=4.2, alpha=0.45, zorder=41)
    if p_roll > 0:
        reach = bump(p_roll) if p_roll < 0.5 else 1.0
        arc = [c + tip * gimbal_body(yaw0, locked, roll0 + GIMBAL_SWEEP * u) @ E2 for u in np.linspace(0, reach, 40)]
        s.curve(np.array(arc), color=TERRACOTTA, lw=1.6, alpha=1.0, zorder=42)
    if t >= 0.90:
        lost = rot(E3, yaw0) @ E1                      # at 90 degrees of pitch no axle has a component along this
        s.curve(np.array([c - 1.5 * lost, c + 1.5 * lost]), color=TERRACOTTA, hidden=True, alpha=0.95, lw=1.1, zorder=44)
        s.text2d(2.45, -0.62, "dashed: no ring turns\nthe body about this axis", fontsize=9.5, color=TERRACOTTA,
                 linespacing=1.25, **SANS)

    # readouts
    x0 = 2.45
    s.text2d(x0, 1.2, f"pitch {np.degrees(pitch):.0f}°", fontsize=10, color=TEAL, **SANS)
    s.text2d(x0, 0.9, rf"$\det E = -\cos\theta = {-np.cos(pitch):.2f}$".replace("-0.00", "0.00"), fontsize=10.5)
    if p_yaw > 0:
        s.text2d(x0, 0.35, "yaw command", fontsize=10, color=INDIGO, alpha=1.0 if p_yaw < 1 else 0.55, **SANS)
    if p_roll > 0:
        s.text2d(x0, 0.08, "roll command", fontsize=10, color=TERRACOTTA, **SANS)
    if t < 0.10:
        msg = "three rings, three axles: yaw, pitch, roll"
    elif t < 0.40:
        msg = "pitch the middle ring up: the roll axle swings toward the yaw axle"
    elif t < 0.66:
        msg = "at 90° of pitch, command yaw: the body turns about the vertical"
    elif t < 0.90:
        msg = "command roll instead: the body makes the same motion"
    else:
        msg = "two rings do one job, and one direction of rotation is lost"
    s.text2d(0.55, -1.8, msg, fontsize=10, color=INK, **SANS)
    return s


def gimbals():
    animate(gimbal_frame, OUT / "ch01-gimbals.gif", OUT / "ch01-gimbals-strip.png", n=110, fps=14,
            width_px=720, stills=(0.0, 0.40, 0.53, 1.0), strip_px=900, strip_cols=2)


# ------------------------------------------------------ animation: unwinding
J_BODY, KP, KD = np.diag([0.05, 0.05, 0.10]), 2.0, 0.6      # the inertia and gains of the unwinding example
UNWIND_T = 7.0


def quat_mul(a, b):
    return np.concatenate([[a[0] * b[0] - a[1:] @ b[1:]], a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])])


def unwinding_closed_loop(_, state):
    """Quaternion kinematics and Euler's equation under tau = -kp qv - kd Omega."""
    q, Om = state[:4], state[4:]
    tau = -KP * q[1:] - KD * Om
    dq = 0.5 * quat_mul(q, np.concatenate([[0.0], Om]))
    dOm = np.linalg.solve(J_BODY, tau - np.cross(Om, J_BODY @ Om))
    return np.concatenate([dq, dOm])


_UNWIND = {}


def unwinding_runs():
    """The two runs of the unwinding example: from q and from -q, the same attitude."""
    if not _UNWIND:
        a0 = np.radians(-10.0)
        q_plus = np.array([np.cos(a0 / 2), 0, 0, np.sin(a0 / 2)])
        tt = np.linspace(0, UNWIND_T, 1401)
        for key, q_init in (("plus", q_plus), ("minus", -q_plus)):
            sol = solve_ivp(unwinding_closed_loop, [0, UNWIND_T], np.concatenate([q_init, np.zeros(3)]),
                            rtol=1e-10, atol=1e-12, t_eval=tt)
            half = np.unwrap(np.arctan2(sol.y[3], sol.y[0]))            # angle of (q0, q3) on the quaternion circle
            dt = tt[1] - tt[0]
            turned = np.concatenate([[0.0], np.cumsum(0.5 * (np.abs(sol.y[6, 1:]) + np.abs(sol.y[6, :-1])) * dt)])
            _UNWIND[key] = dict(t=tt, half=half, turned=np.degrees(turned))
    return _UNWIND


def unwinding_frame(t):
    runs = unwinding_runs()
    time = UNWIND_T * phase(t, 0.07, 0.93)
    s = Scene(width=6.6, height=3.7, azim=-62, elev=34, xlim=(-3.3, 3.3), ylim=(-1.9, 1.8))

    def cam(x, y):
        return x * s.right + y * s.up

    faces = {(2, 1): PERIWINKLE, (2, -1): SAND, (0, 1): PEACH, (0, -1): PAPER, (1, 1): PAPER, (1, -1): PAPER}
    readout = {}
    for key, cx, col, name in (("plus", -2.25, INDIGO, r"estimator reports $\mathfrak{q}$"),
                               ("minus", -0.42, TERRACOTTA, r"estimator reports $-\mathfrak{q}$")):
        run = runs[key]
        half = float(np.interp(time, run["t"], run["half"]))
        turned = float(np.interp(time, run["t"], run["turned"]))
        readout[key] = (half, turned)
        c = cam(cx, 0.05)
        R = rot(E3, 2 * half)                                   # the attitude R(q): a turn by twice the half angle
        s.shadow(c - 0.12 * E3, 0.7, strength=0.18)
        # the turning done so far, drawn on the floor under the nose
        a = np.linspace(2 * run["half"][0], 2 * half, max(2, int(abs(turned)) + 2))
        trail = c - 0.1 * E3 + 0.8 * (np.outer(np.cos(a), E1) + np.outer(np.sin(a), E2))
        full = c - 0.1 * E3 + circle_in_plane(E1, E2, 0.8)
        s.curve(full, color=MUTED, lw=0.6, alpha=0.45, zorder=11)
        s.curve(trail, color=col, lw=2.4, alpha=1.0, zorder=13)
        s.curve(np.array([c - 0.1 * E3, c - 0.1 * E3 + 0.8 * E1]), color=MUTED, hidden=True, alpha=0.9, zorder=12)
        s.dot(c - 0.1 * E3 + 0.8 * E1, color=INK, size=4.5, zorder=14)
        s.label(c - 0.1 * E3 + 0.8 * E1, "target", dx=20, dy=-6, size=9, **SANS)
        pieces = box_pieces(s, c, R, (0.78, 0.5, 0.2), faces)
        paint(s, pieces, zorder=30)
        s.arrow(c + 0.101 * E3, c + 0.101 * E3 + 0.8 * R @ E1, color=INK, lw=1.6, head=0.13, zorder=36)
        s.text2d(cx, 1.22, name, fontsize=10.5, color=col)
        s.text2d(cx, -1.08, f"turned {turned:.0f}°", fontsize=10, color=col, **SANS)

    # the quaternion circle: (q0, q3) = (cos, sin) of half the rotation angle
    dc, dr = np.array([2.05, 0.1]), 0.72
    u = np.linspace(0, 2 * np.pi, 200)
    s.curve(np.array([cam(dc[0] + dr * np.cos(a), dc[1] + dr * np.sin(a)) for a in u]), color=INK, lw=1.5, alpha=1.0, zorder=20)
    for key, col, rr in (("minus", TERRACOTTA, 1.14), ("plus", INDIGO, 1.0)):
        half0, half = runs[key]["half"][0], readout[key][0]
        a = np.linspace(half0, half, max(2, int(abs(np.degrees(half - half0))) + 2))
        s.curve(np.array([cam(dc[0] + rr * dr * np.cos(v), dc[1] + rr * dr * np.sin(v)) for v in a]), color=col,
                lw=3.2 if key == "plus" else 1.8, alpha=1.0, zorder=22)
        s.dot(cam(dc[0] + dr * np.cos(half), dc[1] + dr * np.sin(half)), color=col, size=6.5, zorder=41)
    s.dot(cam(dc[0] + dr, dc[1]), color=INK, size=4.5, zorder=38)
    s.dot(cam(dc[0] - dr, dc[1]), color=INK, size=4.5, zorder=38, hollow=True)
    s.text2d(dc[0] + dr + 0.24, dc[1] + 0.02, r"$+\mathbf{1}$", fontsize=10.5)
    s.text2d(dc[0] - dr - 0.27, dc[1] + 0.02, r"$-\mathbf{1}$", fontsize=10.5)
    s.text2d(dc[0], dc[1] + 0.13, r"$(\cos\frac{\theta}{2},\ \sin\frac{\theta}{2})$", fontsize=10)
    s.text2d(dc[0], dc[1] - 0.16, "quaternion circle", fontsize=9, color=MUTED, **SANS)
    s.text2d(dc[0], 1.22, f"time {time:.1f} s", fontsize=10, color=INK, **SANS)
    s.text2d(dc[0], -1.08, "both signs are the same attitude", fontsize=9, color=MUTED, **SANS)
    if t < 0.07:
        msg = "one body, at rest 10° from the target. The estimator may report either sign"
    elif t < 0.93:
        msg = "the same law pulls each quaternion toward +1"
    else:
        msg = "10° for one sign, 350° the long way round for the other: unwinding"
    s.text2d(0.0, -1.68, msg, fontsize=10, color=INK, **SANS)
    return s


def unwinding():
    animate(unwinding_frame, OUT / "ch01-unwinding.gif", OUT / "ch01-unwinding-strip.png", n=104, fps=14,
            width_px=720, stills=(0.0, 0.30, 0.45, 1.0), strip_px=900, strip_cols=2)


# ------------------------------------------- animation: the flow on the sphere
def flow_frame(t):
    s = Scene(width=6.4, height=3.6, azim=-30, elev=20, xlim=(-2.9, 2.9), ylim=(-1.75, 1.5))
    qd = unit([0.1, -0.55, 0.83])
    k_tau = 4.6 * ease(phase(t, 0.06, 0.94))                    # k times the elapsed time
    s.shadow((0, 0, -1.1), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    side = unit(np.cross(qd, E1))
    for k in range(8):
        start = rot(qd, k * np.pi / 4) @ side
        merid = np.array([np.cos(a) * qd + np.sin(a) * start for a in np.linspace(0.02, np.pi - 0.02, 90)])
        s.sphere_curve(merid, color=INK, lw=0.6, alpha=0.5)
    # points released on rings around the antipode. Each obeys theta_dot = -k sin(theta) on its great circle.
    for th0, count, shift in ((3.02, 8, 0.0), (2.6, 12, 0.13), (2.05, 16, 0.0), (1.45, 16, 0.2), (0.85, 12, 0.0)):
        for j in range(count):
            start = rot(qd, shift + 2 * np.pi * j / count) @ side
            angle = lambda kt: 2 * np.arctan(np.tan(th0 / 2) * np.exp(-kt))  # noqa: E731
            tail = np.array([np.cos(a) * qd + np.sin(a) * start
                             for a in angle(np.linspace(max(k_tau - 0.3, 0.0), k_tau, 10))])
            head = tail[-1]
            if k_tau > 0:
                s.sphere_curve(1.003 * tail, color=INDIGO, lw=1.5, show_hidden=False, zorder=24)
            if s.point_visible(head):
                s.dot(1.003 * head, color=INDIGO, size=3.6, zorder=31)
    s.dot(qd, color=INK, size=6, zorder=45)
    s.dot(-qd, color=TERRACOTTA, size=6, hollow=True, zorder=45)
    s.label(qd, r"$q_d$", dx=12, dy=11, size=12)
    s.label(-qd, r"$-q_d$", dx=-19, dy=-13, color=TERRACOTTA, size=12)
    s.text2d(-2.05, 0.55, "target", fontsize=9.5, color=INK, **SANS)
    s.curve(np.array([-1.72 * s.right + 0.55 * s.up, qd - 0.1 * s.right]), color=MUTED, lw=0.5, alpha=0.8, zorder=44)
    s.text2d(2.05, -0.55, "the antipode\nnever moves", fontsize=9.5, color=TERRACOTTA, linespacing=1.25, **SANS)
    s.text2d(2.05, 0.75, rf"$k\,t = {k_tau:.1f}$", fontsize=10.5)
    if t < 0.06:
        msg = r"release points all over the sphere under $\omega = -k\,e_q$"
    elif t < 0.5:
        msg = "each one slides along its great circle. Near the antipode the push is weak"
    elif t < 0.94:
        msg = "the push peaks a quarter turn from the target, then fades again"
    else:
        msg = r"every point reaches $q_d$ except the one that sits at $-q_d$"
    s.text2d(0.0, -1.57, msg, fontsize=10, color=INK, **SANS)
    return s


def flow():
    animate(flow_frame, OUT / "ch01-flow.gif", OUT / "ch01-flow-strip.png", n=90, fps=14,
            width_px=680, stills=(0.0, 0.36, 0.6, 1.0), strip_px=900, strip_cols=2)


def preview(make_frame, name, times):
    """Development aid: write single frames of an animation to /tmp/shots."""
    from figkit import _render
    for t in times:
        _render(make_frame(t), 760).save(f"/tmp/shots/ch01-{name}-t{int(round(100 * t)):03d}.png")


FIGURES = dict(heading=heading, so3_ball=so3_ball, gimbals=gimbals, euler_rates=euler_rates, plate=plate,
               unwinding=unwinding, obstruction=obstruction, s2_error=s2_error, flow=flow)

if __name__ == "__main__":
    for name in sys.argv[1:] or list(FIGURES):       # python figures/src/ch01.py [name ...]
        FIGURES[name]()
