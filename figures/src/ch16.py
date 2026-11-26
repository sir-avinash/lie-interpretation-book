"""Figures for Chapter 16.  Run: python figures/src/ch16.py

Every motion is integrated by the robot model library of case_studies/robots,
with the same `step` for every robot.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "case_studies"))
from figkit import (DASH, INDIGO, INK, LW_VECTOR, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL,  # noqa: E402
                    TERRACOTTA, Scene, animate, axis_through_box, box, caption, halo, quadrotor,
                    rot, save, unit)

import robots  # noqa: E402
from robots.core import attitude_tracking_torque, tangent  # noqa: E402
from robots.reaction_mass_biped import PI  # noqa: E402

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
TRIAD = (TERRACOTTA, TEAL, INDIGO)
FACES = {(0, 1): PERIWINKLE, (0, -1): PERIWINKLE, (1, 1): SAND, (1, -1): SAND, (2, 1): PEACH, (2, -1): PEACH}
N_GALLERY, T_GALLERY, H = 61, 5.0, 2.5e-3
_cache = {}


def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def sample(robot, state, control, t_f=T_GALLERY, n=N_GALLERY, h=H):
    """Simulate with the library's step and keep n equally spaced states."""
    every = int(round(t_f / (n - 1) / h))
    out, t = [state], 0.0
    for k in range((n - 1) * every):
        state = robot.step(state, control(t, state), h)
        t += h
        if (k + 1) % every == 0:
            out.append(state)
    return out


# ------------------------------------------------------------- the seven runs
def run_pendulum():
    pend = robots.get("spherical_pendulum")
    q0 = unit([0.85, 0.0, -0.3])
    return sample(pend, pend.State(q=q0, omega=tangent(q0, [0.6, 2.4, 2.2])), lambda t, s: pend.zero_input())


def run_body():
    body = robots.get("pendulum_3d", params=robots.rigid_body.Params(rho=np.zeros(3)))
    return sample(body, body.State(Omega=np.array([2.2, 0.25, 0.3])), lambda t, s: body.zero_input())


def run_quadrotor():
    quad = robots.get("quadrotor")
    return sample(quad, quad.State(), lambda t, s: quad.tracking_controller(s, robots.quadrotor.circle_reference(t)))


LOAD_TARGET = np.array([1.5, 0.0, 0.5])


def run_load():
    robot = robots.get("quadrotor_load", params=robots.quadrotor_load.chapter6_params())
    swing = robots.expm_so3(np.radians(40) * E1) @ -E3
    return robot, sample(robot, robot.State(q=swing), lambda t, s: robot.tracking_controller(s, LOAD_TARGET))


def level_torque(R, Omega, J):
    """eq6-law with a level target: holds each quadrotor's attitude."""
    return attitude_tracking_torque(R, Omega, np.eye(3), np.zeros(3), np.zeros(3), J, 8.0, 2.0)


def run_team():
    team = robots.get("multi_quadrotor_payload", n=3)
    p = team.params
    f = (p.m + p.m_L / 3) * p.g                               # each cable carries a third of the payload
    state = team.State(x_L=np.zeros(3), v_L=np.array([0.25, 0.0, 0.0]), R_L=np.eye(3),
                       Omega_L=np.array([0.0, 0.0, 1.6]), q=np.tile(-E3, (3, 1)), omega=np.zeros((3, 3)),
                       R=np.array([np.eye(3)] * 3), Omega=np.zeros((3, 3)))

    def control(t, s):
        return f, np.array([level_torque(R, Om, J) for R, Om, J in zip(s.R, s.Omega, p.J)])
    return team, sample(team, state, control)


def run_cable():
    cable = robots.get("quadrotor_cable", n=5, attach=(0,))
    p = cable.params
    f = np.array([cable.joint_masses().sum() * p.g])
    q0 = np.array([rot(E2, np.radians(18 + 9 * k)) @ -E3 for k in range(5)])
    state = cable.State(x_0=np.zeros(3), v_0=np.zeros(3), q=q0,
                        omega=np.array([tangent(q, [0.0, 0.0, 0.9]) for q in q0]),
                        R=np.eye(3)[None], Omega=np.zeros((1, 3)))

    def control(t, s):
        return f, level_torque(s.R[0], s.Omega[0], p.J[0])[None]
    return cable, sample(cable, state, control)


def run_biped():
    biped = robots.get("reaction_mass_biped")
    p, zero = biped.params, np.zeros(3)
    state = biped.State(l=0.5, l_dot=0.0, R1=np.eye(3), Omega1=zero, RP=np.eye(3), OmegaP=zero, R2=np.eye(3),
                        Omega2=zero, slide=np.full(3, 0.10), slide_dot=zero)

    def control(t, s):
        """eq6-law on each of the three bodies, a spring on the leg length and on the sliding masses."""
        R2d = rot(E2, 0.55 * np.sin(2.4 * t))
        RPd = rot(E3, 0.5 * np.sin(1.3 * t))
        R1d = rot(E1, 0.14 * np.sin(1.6 * t)) @ rot(E2, 0.12 * np.cos(1.6 * t))
        tau_2 = attitude_tracking_torque(s.R2, s.Omega2, R2d, zero, zero, p.J_0, 40.0, 9.0)
        JP = biped.torso_inertia(s.slide)
        tau_P = attitude_tracking_torque(s.RP, s.OmegaP, RPd, zero, zero, JP, 25.0, 6.0) + s.RP.T @ s.R2 @ tau_2
        J1 = p.J_0 + p.m * s.l**2 * PI
        tau_1 = (attitude_tracking_torque(s.R1, s.Omega1, R1d, zero, zero, J1, 160.0, 36.0)
                 - p.m * p.g * s.l * np.cross(s.R1.T @ E3, E3) + s.R1.T @ s.RP @ tau_P)
        f = p.m * (p.g * E3 @ (s.R1 @ E3) - s.l * s.Omega1 @ PI @ s.Omega1 - 60 * (s.l - 0.5) - 14 * s.l_dot)
        want = 0.11 + 0.06 * np.sin(2.0 * t + np.array([0.0, 2.0, 4.0]))
        f_s = 2 * p.m_r * (-80 * (s.slide - want) - 16 * s.slide_dot)
        return f, tau_1, tau_P, tau_2, f_s
    return biped, sample(biped, state, control)


# ------------------------------------------------------------- drawing pieces
def triad(s, o, R, length=0.36, lw=1.2, zorder=44):
    s.frame(o, R, length=length, colors=TRIAD, lw=lw, head=0.085, zorder=zorder)


def ball(s, c, r):
    """A factor S^2: a small sphere at the point a direction is attached to."""
    s.sphere(c, r, color=PERIWINKLE, res=90, alpha=1.0, outline=False)
    th = np.linspace(0, 2 * np.pi, 40)
    xy = s.project(c)[0]
    s.ax.plot(xy[0] + r * np.cos(th), xy[1] + r * np.sin(th), color=INK, lw=0.7, zorder=20)


def trail(s, pts, color=INDIGO, lw=1.0):
    if len(pts) > 1:
        s.curve(np.asarray(pts), color=color, lw=lw, alpha=1.0, zorder=18)


def draw_pendulum(s, o, k):
    states = cached("pendulum", run_pendulum)
    r = 0.95
    Q = np.array([st.q for st in states[:k + 1]])
    s.sphere(o, r, color=PERIWINKLE, res=220, alpha=0.5)
    s.sphere_curve(o + r * Q, center=o, color=INDIGO, lw=1.1, alpha=1.0, zorder=22)
    bob = o + r * Q[-1]
    front = (bob - o) @ s.view >= 0
    s.curve(np.array([o, bob]), color=INK, lw=1.2, alpha=1.0, zorder=23 if front else 13)
    s.dot(o, color=INK, size=3.0, zorder=24)
    s.dot(bob, color=INDIGO, size=5.5, hollow=not front, zorder=40)


def draw_body(s, o, k):
    R = cached("body", run_body)[k].R
    dims = (0.62, 0.46, 1.0)
    for i in range(3):
        if R[:, i] @ s.view <= 0:
            axis_through_box(s, o, R, dims, R[:, i], 0.95, TRIAD[i], lw=1.3, head=0.09)
    box(s, o, R, dims, FACES)
    for i in range(3):
        if R[:, i] @ s.view > 0:
            axis_through_box(s, o, R, dims, R[:, i], 0.95, TRIAD[i], lw=1.3, head=0.09)


def draw_quadrotor(s, o, k):
    states = cached("quadrotor", run_quadrotor)
    sc, c = 0.62, np.array([0.0, 0.0, 0.0])
    tt = np.linspace(0, 2 * np.pi, 120)
    ref = np.array([robots.quadrotor.circle_reference(t)[0] for t in tt])
    s.curve(o + sc * (ref - c), color=MUTED, lw=0.8, alpha=1.0, linestyle=DASH, zorder=13)
    xd = robots.quadrotor.circle_reference(k * T_GALLERY / (N_GALLERY - 1))[0]
    s.dot(o + sc * (xd - c), color=MUTED, size=4.0, hollow=True, zorder=20)
    trail(s, [o + sc * (st.x - c) for st in states[:k + 1]])
    x = o + sc * (states[k].x - c)
    quadrotor(s, x, states[k].R, arm=0.3, lw=1.6)
    triad(s, x, states[k].R, length=0.42)


def draw_load(s, o, k):
    robot, states = cached("load", run_load)
    sc, c = 0.62, np.array([0.75, 0.0, 0.75])
    st = states[k]
    xL, xQ = o + sc * (st.x_L - c), o + sc * (robot.quadrotor_position(st) - c)
    trail(s, [o + sc * (a.x_L - c) for a in states[:k + 1]])
    s.dot(o + sc * (LOAD_TARGET - c), color=INK, size=3.2, zorder=20)
    ball(s, xQ, 0.13)
    s.curve(np.array([xQ, xL]), color=INK, lw=1.1, alpha=1.0, zorder=24)
    s.dot(xL, color=INDIGO, size=5.5, zorder=40)
    quadrotor(s, xQ, st.R, arm=0.3, lw=1.6)
    triad(s, xQ, st.R, length=0.4)


def draw_team(s, o, k):
    team, states = cached("team", run_team)
    sc, st = 0.8, states[k]
    c = np.array([0.6, 0.0, 0.45])
    xL = o + sc * (st.x_L - c)
    xQ = o + sc * (team.quadrotor_positions(st) - c)
    att = xL + sc * team.params.r @ st.R_L.T
    box(s, xL, st.R_L, (0.8, 0.8, 0.09), FACES, zorder=30)
    triad(s, xL, st.R_L, length=0.36, zorder=37)
    for a, b, R in zip(att, xQ, st.R):
        s.curve(np.array([a, b]), color=INK, lw=1.0, alpha=1.0, zorder=38)
        ball(s, b, 0.09)
        quadrotor(s, b, R, arm=0.2, lw=1.3, zorder=40)


def draw_cable(s, o, k):
    cable, states = cached("cable", run_cable)
    sc, st = 1.25, states[k]
    c = np.array([0.15, 0.0, -0.5])
    X = o + sc * (cable.positions(st) - c)
    trail(s, [o + sc * (cable.positions(a)[-1] - c) for a in states[:k + 1]])
    for j in range(5):
        ball(s, X[j], 0.07)
    s.curve(X, color=INK, lw=1.2, alpha=1.0, zorder=24)
    s.dot(X[-1], color=INDIGO, size=5.5, zorder=40)
    quadrotor(s, X[0], st.R[0], arm=0.3, lw=1.6)
    triad(s, X[0], st.R[0], length=0.4)


def draw_biped(s, o, k, st=None, sc=2.1, triads=True):
    biped, states = cached("biped", run_biped) if st is None else (robots.get("reaction_mass_biped"), None)
    st = states[k] if st is None else st
    foot = o - np.array([0.0, 0.0, 0.5 * sc])
    hip = foot + sc * st.l * st.R1 @ E3
    toe = hip - sc * 0.5 * st.R2 @ E3
    dims = (0.34, 0.26, 0.42)
    s.curve(np.array([foot - 0.22 * E1, foot + 0.22 * E1]), color=MUTED, lw=2.2, alpha=1.0, zorder=14)
    s.curve(np.array([foot, hip]), color=INK, lw=2.0, alpha=1.0, zorder=24)
    s.curve(np.array([hip, toe]), color=INK, lw=2.0, alpha=1.0, zorder=24)
    s.dot(toe, color=INK, size=3.0, zorder=25)
    if triads:
        triad(s, 0.5 * (foot + hip), st.R1, length=0.26, lw=1.0, zorder=26)
        triad(s, 0.5 * (hip + toe), st.R2, length=0.26, lw=1.0, zorder=26)
    for i in range(3):                                         # the sliding reaction masses
        a = sc * st.slide[i] * st.RP[:, i]
        back = a @ s.view < 0
        for sign in (1, -1):
            behind = (sign * a) @ s.view < 0
            s.curve(np.array([hip, hip + sign * a]), color=TEAL, lw=1.0, alpha=1.0, zorder=28 if behind else 52)
            s.dot(hip + sign * a, color=TEAL, size=4.2, zorder=29 if behind else 53)
        del back
    box(s, hip, st.RP, dims, FACES, zorder=40)
    if triads:
        for i in range(3):
            if st.RP[:, i] @ s.view > 0:
                axis_through_box(s, hip, st.RP, dims, st.RP[:, i], 0.62, TRIAD[i], lw=1.2, head=0.085, zorder=46)


# ------------------------------------------------------------- the gallery
GALLERY = [
    (draw_pendulum, (-4.1, 2.05), "spherical pendulum", r"$S^2$"),
    (draw_body, (0.0, 2.05), "rigid body", r"$\mathrm{SO}(3)$"),
    (draw_quadrotor, (4.1, 2.05), "quadrotor", r"$\mathbb{R}^3\times\mathrm{SO}(3)$"),
    (draw_load, (-4.75, -1.6), "cable-suspended load", r"$\mathbb{R}^3\times S^2\times\mathrm{SO}(3)$"),
    (draw_team, (-1.55, -1.6), "three quadrotors, rigid payload",
     r"$\mathbb{R}^3\times\mathrm{SO}(3)\times(S^2\times\mathrm{SO}(3))^3$"),
    (draw_cable, (1.85, -1.6), "flexible cable, five links", r"$\mathbb{R}^3\times(S^2)^5\times\mathrm{SO}(3)$"),
    (draw_biped, (4.75, -1.6), "Reaction Mass Biped", r"$\mathbb{R}\times\mathrm{SO}(3)^3\times\mathbb{R}^3$"),
]
XLIM, YLIM = (-6.55, 6.55), (-4.25, 3.3)


def gallery_frame(t):
    k = int(round(t * (N_GALLERY - 1)))
    s = Scene(width=7.6, height=7.6 * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0]), azim=-62, elev=16,
              xlim=XLIM, ylim=YLIM)
    for draw, (cx, cy), name, manifold in GALLERY:
        draw(s, cx * s.right + cy * s.up, k)
        s.ax.text(cx, cy - 1.36, name, fontsize=8.6, ha="center", va="center", color=INK, zorder=60,
                  path_effects=halo(), **SANS)
        s.ax.text(cx, cy - 1.70, manifold, fontsize=10.5, ha="center", va="center", color=INK, zorder=60,
                  path_effects=halo())
    caption(s, f"seven models, one integration step: t = {k * T_GALLERY / (N_GALLERY - 1):.1f} s", y=YLIM[0] + 0.22,
            size=9.5)
    return s


def gallery():
    animate(gallery_frame, OUT / "ch16-gallery.gif", OUT / "ch16-gallery-strip.png", n=N_GALLERY, fps=12,
            width_px=860, stills=(0.7,), strip_px=1500, hold=1.0, colors=96)


# ------------------------------------------------------------- the pipeline
def pipeline():
    fig = plt.figure(figsize=(7.0, 2.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 2.85)
    top = [("configuration\nmanifold", "Chapter 2", INK), ("errors and\ntransport", "Chapter 3", INK),
           ("equations\nof motion", "Chapter 4", INK), ("symbolic\ncheck", "Chapter 5", TERRACOTTA)]
    bottom = [("tracking", "Chapter 6", INDIGO), ("linearization\nand LQR", "Chapter 7", INDIGO),
              ("uncertainty\nand safety", "Chapter 8", INDIGO), ("planning and\nintegration", "Chapters 10, 12, 13", TEAL),
              ("estimation", "Chapter 14", TEAL)]
    w, h = 1.16, 0.74

    def row(items, y, x0, gap):
        centers = []
        for k, (title, chap, color) in enumerate(items):
            x = x0 + k * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                        fc="#f6f3ea", ec=color, lw=1.2, zorder=3))
            ax.text(x + w / 2, y + 0.09, title, ha="center", va="center", fontsize=9.6, color=INK, zorder=4,
                    linespacing=1.05, **SANS)
            ax.text(x + w / 2, y - 0.25, chap, ha="center", va="center", fontsize=7.6, color=color, zorder=4, **SANS)
            centers.append(x + w / 2)
        for a, b in zip(centers[:-1], centers[1:]):
            ax.annotate("", (b - w / 2 - 0.02, y), xytext=(a + w / 2 + 0.02, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_VECTOR * 0.8, mutation_scale=9))
        return centers

    y1, y2 = 2.22, 0.62
    ct = row(top, y1, 0.55, 0.42)
    cb = row(bottom, y2, 0.14, 0.23)
    # from the checked model down to everything that uses it
    xm, ym = ct[-1], 1.42
    ax.plot([xm, xm], [y1 - h / 2 - 0.02, ym], color=INK, lw=LW_VECTOR * 0.8, zorder=2)
    ax.plot([cb[0], cb[-1]], [ym, ym], color=INK, lw=LW_VECTOR * 0.8, zorder=2)
    for x in cb:
        ax.annotate("", (x, y2 + h / 2 + 0.02), xytext=(x, ym),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_VECTOR * 0.8, mutation_scale=9))
    ax.text(0.14, ym + 0.13, "a model you can trust", ha="left", va="center", fontsize=8.6, color=MUTED, **SANS)
    ax.text(0.14, 2.74, "modeling", ha="left", va="center", fontsize=8.6, color=MUTED, **SANS)
    ax.text(0.14, 0.1, "what you do with the model", ha="left", va="center", fontsize=8.6, color=MUTED, **SANS)
    save(fig, OUT / "ch16-pipeline.svg")


# ------------------------------------------------------------- the skater
N_SKATE, T_SKATE = 41, 0.8


def run_skater():
    biped = robots.get("reaction_mass_biped")
    p, zero = biped.params, np.zeros(3)
    state = biped.State(l=0.5, l_dot=0.0, R1=np.eye(3), Omega1=zero, RP=np.eye(3),
                        OmegaP=np.array([0.0, 0.0, 3.0]), R2=np.eye(3), Omega2=zero,
                        slide=np.full(3, 0.05), slide_dot=zero)
    u = (p.m * p.g, zero, zero, zero, zero)
    return biped, sample(biped, state, lambda t, s: u, t_f=T_SKATE, n=N_SKATE, h=1e-3)


def skater_frame(t):
    biped, states = cached("skater", run_skater)
    k = int(round(t * (N_SKATE - 1)))
    st = states[k]
    xlim, ylim = (-2.3, 3.5), (-1.45, 1.5)
    s = Scene(width=6.4, height=6.4 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), azim=-55, elev=24, xlim=xlim, ylim=ylim)
    o = -0.55 * s.right + 0.12 * s.up
    draw_biped(s, o, k, st=st, sc=1.9, triads=False)
    hip = o - np.array([0.0, 0.0, 0.95]) + 1.9 * st.l * st.R1 @ E3
    for i in (0, 1):                                              # where the two horizontal pairs have been
        for sign in (1, -1):
            pts = [hip + sign * 1.9 * a.slide[i] * a.RP[:, i] for a in states[:k + 1]]
            trail(s, pts, color=TEAL, lw=0.8)
    tip = axis_through_box(s, hip, st.RP, (0.34, 0.26, 0.42), E3, 0.3 * st.OmegaP[2], INDIGO, lw=1.6, zorder=46)
    s.label(tip, r"$\Omega$", dx=10, dy=2, color=INDIGO, size=11)
    # bars
    mom = lambda a: (a.RP @ biped.torso_inertia(a.slide) @ a.OmegaP)[2]
    x0, wmax = 1.55, 1.7
    for y, value, first, color, name, text in (
            (0.88, st.OmegaP[2], states[0].OmegaP[2], INDIGO, "spin rate of the torso", f"{st.OmegaP[2]:.2f} rad/s"),
            (0.05, mom(st), mom(states[0]), TEAL, "angular momentum of the torso", f"{mom(st):.3f} N·m·s"),
            (-0.78, st.slide[0], 0.30, MUTED, "distance of the masses from the hip", f"{100 * st.slide[0]:.0f} cm")):
        s.ax.add_patch(plt.Rectangle((x0, y - 0.09), wmax * value / first, 0.18, fc=color, ec="none", zorder=20))
        s.ax.plot([x0 + wmax, x0 + wmax], [y - 0.16, y + 0.16], color=INK, lw=0.8, zorder=21)
        s.ax.plot([x0, x0], [y - 0.16, y + 0.16], color=INK, lw=0.8, zorder=21)
        s.ax.text(x0, y + 0.27, name, fontsize=8.6, ha="left", va="center", color=INK, **SANS)
        s.ax.text(x0, y - 0.27, text, fontsize=8.6, ha="left", va="center", color=color, **SANS)
    caption(s, f"nothing holds the masses: they slide out and the torso slows, t = {k * T_SKATE / (N_SKATE - 1):.2f} s",
            size=9.5)
    return s


def skater():
    animate(skater_frame, OUT / "ch16-skater.gif", OUT / "ch16-skater-strip.png", n=N_SKATE, fps=12, width_px=720,
            stills=(0.0, 1.0), strip_px=900, strip_cols=1, hold=1.6)


if __name__ == "__main__":
    pipeline()
    skater()
    gallery()
