"""Figures for Chapter 7.  Run: python figures/src/ch07.py

Every curve comes from the models and controllers in verify/ch07.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "verify"))
from matplotlib.patches import Rectangle  # noqa: E402

from figkit import (DASH, INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, geodesic, great_circle, halo, new_plot, phase, rot, save, style_axes, unit)

_spec = importlib.util.spec_from_file_location("ch07_verify", ROOT / "verify" / "ch07.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")


PALE_TEAL = "#b5d6d2"
_CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def box(s, center, R, dims, face_colors, zorder=40, lw=0.9):
    """Shaded convex box: only the faces that look at the camera. `dims` are full edge lengths along the columns of R."""
    c = np.asarray(center, float)
    half = 0.5 * np.asarray(dims, float)
    for axis in range(3):
        for sign in (1, -1):
            n = sign * R[:, axis]
            if n @ s.view <= 1e-9:
                continue
            a, b = [(half[k] * R[:, k]) for k in range(3) if k != axis]
            fc = c + half[axis] * n
            s.patch([fc + a + b, fc - a + b, fc - a - b, fc + a - b],
                    color=face_colors[(axis, sign)], alpha=1.0, lw=lw, zorder=zorder + 2 * (n @ s.view))


def wire_box(s, center, R, dims, color=MUTED, lw=0.8, zorder=46):
    """A target or reference attitude: the box of the body as thin dashed gray edges."""
    c = np.asarray(center, float)
    half = 0.5 * np.asarray(dims, float)
    corner = lambda sx, sy, sz: c + R @ (half * np.array([sx, sy, sz]))
    for u in (1, -1):
        for v in (1, -1):
            for a, b in ((corner(1, u, v), corner(-1, u, v)), (corner(u, 1, v), corner(u, -1, v)),
                         (corner(u, v, 1), corner(u, v, -1))):
                s.curve(np.array([a, b]), color=color, lw=lw, alpha=1.0, zorder=zorder, linestyle=DASH)


BODY_DIMS = np.array([1.25, 0.85, 0.14])
BODY_FACES = {(2, 1): PERIWINKLE, (2, -1): PEACH, (1, 1): PALE_TEAL, (1, -1): PALE_TEAL, (0, 1): PAPER, (0, -1): PAPER}


def tip_label(s, base, tip, text, off=11, **kw):
    """Label just beyond the tip of an arrow, pushed along the arrow as the camera sees it."""
    a, b = s.project([base, tip])
    d = b - a
    n = np.hypot(*d)
    d = d / n if n > 1e-6 else np.array([0.0, 1.0])
    s.label(tip, text, dx=off * d[0], dy=off * d[1], **kw)


def rot_vec(qd, q):
    """Axis times angle of the smallest rotation that turns the direction qd into q."""
    c = np.cross(qd, q)
    return np.arctan2(np.linalg.norm(c), qd @ q) * unit(c)


# ------------------------------------- opening: three ways to write a small deviation
PUZ_ETA = np.radians(10.0) * unit([1.0, 1.0, 1.0])            # one small rotation, fixed in the body
PUZ_SHEAR = np.array([[0.3, 0.9, 0.0], [0.9, -0.6, 0.3], [0.0, 0.3, 0.5]])   # a change of the entries that is no rotation


def puzzle_frame(t):
    a, b = ease(phase(t, 0.07, 0.6)), ease(phase(t, 0.68, 0.9))
    pitch = 89.0 * a
    Rd = V.Ry(np.radians(pitch))
    R = Rd @ V.expm_so3(PUZ_ETA)
    d_theta = np.degrees(V.wrap(V.euler_of(R) - V.euler_of(Rd)))
    eta = np.degrees(V.log_so3(Rd.T @ R))
    X = R + 0.4 * b * Rd @ PUZ_SHEAR                         # the nine entries, changed freely

    s = Scene(width=7.0, height=3.9, azim=-38, elev=22, xlim=(-1.9, 5.1), ylim=(-1.95, 1.95))
    at = lambda x, y: x * s.right + y * s.up
    ctr = at(-0.3, 0.1)
    box(s, ctr, R, 1.5 * BODY_DIMS, BODY_FACES)
    wire_box(s, ctr, Rd, 1.5 * BODY_DIMS)
    s.text2d(-0.3, 1.72, "a body 10° from its reference", fontsize=9.5, color=MUTED, **SANS)
    s.text2d(-0.3, -1.42, f"reference pitch {pitch:.0f}°", fontsize=10, color=TERRACOTTA if pitch > 80 else INK, **SANS)
    s.text2d(-0.3, 1.47, "dashed: reference", fontsize=9, color=MUTED, **SANS)

    x_lab, x0, per_deg, x_num = 1.62, 3.0, 1.45 / 180.0, 5.0
    left = dict(ha="left", fontsize=9.5, **SANS)

    def bars(y_top, names, values, col):
        for k, (name, val) in enumerate(zip(names, values)):
            y = y_top - 0.22 * k
            s.text2d(x_lab + 0.1, y, name, color=INK, **left)
            s.ax.plot([x0, x0 + 180 * per_deg], [y, y], color=MUTED, lw=0.5, alpha=0.5, zorder=20)
            s.ax.add_patch(Rectangle((x0, y - 0.065), abs(val) * per_deg, 0.13, fc=col, ec="none", zorder=30))
            s.text2d(x_num, y, f"{val:.1f}°".replace("-", "−"), color=col, ha="right", fontsize=9.5, **SANS)

    # 1. the deviation in yaw, pitch and roll
    s.text2d(x_lab, 1.72, r"Euler angles: $\Theta - \Theta_d$", color=TERRACOTTA, **left)
    bars(1.44, ("yaw", "pitch", "roll"), d_theta, TERRACOTTA)
    # 2. the nine entries, drawn as the three columns and the cube they span
    sheared = b > 0.02
    col9 = TERRACOTTA if sheared else INK
    s.text2d(x_lab, 0.62, "the nine entries of $R$" + (", changed freely" if sheared else ""), color=col9, **left)
    o = at(2.2, -0.12) - 0.19 * X @ np.ones(3)
    L = 0.46
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for u in (0, 1):
            for v in (0, 1):
                if u or v:
                    a0 = o + L * (u * X[:, j] + v * X[:, k])
                    s.curve(np.array([a0, a0 + L * X[:, i]]), color=col9, lw=0.7, alpha=1.0, zorder=27)
    s.frame(o, X, length=L, lw=1.6, head=0.09)
    for i in range(3):
        for j in range(3):
            s.text2d(3.75 + 0.62 * j, 0.3 - 0.22 * i, f"{X[i, j]:.2f}".replace("-", "−"), color=col9, ha="right",
                     fontsize=9.5, **SANS)
    det = np.linalg.det(X)
    s.text2d(x_num, -0.42, f"det = {det:.2f}" + (": not a rotation" if sheared else ""), color=col9, ha="right",
             fontsize=9.5, **SANS)
    # 3. the variation
    s.text2d(x_lab, -0.82, r"variation $\eta$", color=INDIGO, **left)
    bars(-1.08, (r"$\eta_1$", r"$\eta_2$", r"$\eta_3$"), eta, INDIGO)

    if t < 0.07:
        msg, col = "one small deviation of an attitude, written three ways", INK
    elif t < 0.64:
        msg, col = "pitch the reference up: near 90° the same deviation is a huge change of yaw and roll", TERRACOTTA
    elif t < 0.92:
        msg, col = "change the nine entries freely and the three axes stop being a rotation", TERRACOTTA
    else:
        msg, col = "the variation is three numbers, the same at every attitude", INDIGO
    s.text2d(1.6, -1.78, msg, fontsize=10, color=col, **SANS)
    return s


def puzzle():
    animate(puzzle_frame, OUT / "ch07-puzzle.gif", OUT / "ch07-puzzle-strip.png", n=96, fps=13, width_px=760,
            stills=(0.0, 0.45, 0.62, 0.95), strip_px=900, strip_cols=2, hold=2.0)


# ------------------------------------------------- variation on the sphere
def swing_pair(T, eps):
    """Free swing, a neighbor started `eps` away, and the linearized flow between them."""
    q0, w0 = V.s2_swing()
    axis = unit(np.cross(q0, [0.3, 1.0, 0.2]))
    free = lambda t, q, w: np.zeros(3)
    ref = V.s2_sim(q0, w0, free, T)
    qe, we, s0 = V.s2_neighbour(q0, w0, axis, eps)
    return ref, V.s2_sim(qe, we, free, T), V.s2_linear_flow(ref, s0, T)


def pair20():
    """The free swing and its neighbor released 20° away, used by both sphere animations."""
    return cached("pair20", lambda: swing_pair(1.8, np.radians(20.0))[:2])


def swing_scene(azim, elev, t_start, now):
    """Sphere with the two swings drawn from t_start up to `now`. Returns the scene and the two states."""
    ref, nb = pair20()
    s = Scene(width=6.2, height=5.0, azim=azim, elev=elev, xlim=(-2.3, 2.3), ylim=(-2.0, 1.71))
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.5, alpha=0.4)
    tt = np.linspace(t_start, max(now, t_start + 1e-3), max(3, int(220 * (now - t_start))))
    s.sphere_curve(ref.sol(tt)[:3].T, color=INDIGO, lw=2.0, zorder=16, alpha=1.0, show_hidden=False)
    s.sphere_curve(nb.sol(tt)[:3].T, color=TERRACOTTA, lw=1.5, zorder=16, alpha=1.0, show_hidden=False)
    return s, ref.sol(now), nb.sol(now)


VAR_T0, VAR_T1, VAR_SLOW, VAR_FPS = 0.78, 1.74, 6.0, 12


def variation_frame(t):
    now = VAR_T0 + (VAR_T1 - VAR_T0) * t
    s, xd, x = swing_scene(-112, 17, VAR_T0, now)
    qd, q = xd[:3], x[:3]
    xi = rot_vec(qd, q)
    dq = np.cross(xi, qd)
    a1, a2 = s.tangent_plane(1.004 * qd, size=0.52, color=SAND, alpha=0.82)
    s.sphere_curve(geodesic(qd, q, 30), color=MUTED, lw=0.9, zorder=15, alpha=1.0)
    s.arrow(qd, qd + dq, color=TERRACOTTA, zorder=32)
    s.arrow(qd, qd + xi, color=TEAL, zorder=32)
    s.dot(qd, color=INDIGO, size=4.8)
    s.dot(q, color=TERRACOTTA, size=4.8)
    tip_label(s, qd, qd + xi, r"$\xi$", off=10, color=TEAL, size=13)
    tip_label(s, qd, qd + dq, r"$\xi \times q_d$", off=20, color=TERRACOTTA)
    back = -unit(s.project(xi + dq)[0])
    s.label(qd, r"$q_d$", dx=13 * back[0], dy=13 * back[1], color=INDIGO)
    s.text2d(-1.55, 1.5, "reference", fontsize=9.5, color=INDIGO, **SANS)
    s.text2d(-1.55, 1.3, "neighbor", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.ax.plot([-2.15, -1.95], [1.5, 1.5], color=INDIGO, lw=2.0)
    s.ax.plot([-2.15, -1.95], [1.3, 1.3], color=TERRACOTTA, lw=1.5)
    s.text2d(1.75, 1.5, r"$\xi \cdot q_d = 0$", fontsize=11.5, color=TEAL)
    s.text2d(1.75, 1.27, rf"$\|\xi\|$ = {np.degrees(np.linalg.norm(xi)):.0f}°", fontsize=10.5, color=TEAL)
    if t < 0.3:
        msg = r"at each instant the neighbor is the reference turned about $\xi$"
    elif t < 0.7:
        msg = "the tangent plane rides along with the reference"
    else:
        msg = r"$\xi$ and the displacement $\xi \times q_d$ never leave it, at right angles to each other"
    s.text2d(0.0, -1.62, msg, fontsize=10, **SANS)
    s.text2d(0.0, -1.85, f"t = {now:.2f} s, shown {VAR_SLOW:.0f} times slower", fontsize=9.5, color=MUTED, **SANS)
    return s


def variation():
    n = int(round((VAR_T1 - VAR_T0) * VAR_SLOW * VAR_FPS)) + 1
    animate(variation_frame, OUT / "ch07-variation.gif", OUT / "ch07-variation-strip.png", n=n, fps=VAR_FPS,
            width_px=620, stills=(0.08, 0.4, 0.7, 1.0), strip_px=760, strip_cols=2, hold=1.6)


# ---------------------------------- two tangent planes: the second constraint
PL_GAP0, PL_GAP1 = np.radians(58.0), np.radians(26.0)


def planes_frame(t):
    carry = ease(phase(t, 0.1, 0.36))
    built = t >= 0.42
    close = ease(phase(t, 0.56, 0.94))
    ang = PL_GAP0 + (PL_GAP1 - PL_GAP0) * close
    s = Scene(width=6.4, height=3.9, azim=-70, elev=27, xlim=(-3.1, 3.1), ylim=(-1.78, 1.998))
    fwd = unit(np.array([s.view[0], s.view[1], 0.0]))    # horizontal, toward the viewer
    qd = E3
    side = unit(s.right + 0.45 * fwd)
    q = np.cos(ang) * E3 + np.sin(ang) * side
    ax = unit(np.cross(qd, side))
    along = np.cross(ax, qd)                             # unit tangent at q_d pointing toward q
    wd = 0.66 * unit(-along + 0.55 * ax)
    w = rot(ax, ang) @ wd                                # the same sideways twirl, at the other direction
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(geodesic(qd, np.cos(PL_GAP0) * E3 + np.sin(PL_GAP0) * side), color=INK, lw=0.8, alpha=0.7)
    c = 1.004 * qd
    s.patch([c + 0.5 * along + 0.55 * ax, c - 0.8 * along + 0.55 * ax, c - 0.8 * along - 0.55 * ax, c + 0.5 * along - 0.55 * ax],
            color=SAND, alpha=0.85)
    wq = unit(w)
    cq, bq = 1.004 * q, np.cross(q, wq)
    s.patch([cq + 0.78 * wq + 0.3 * bq, cq - 0.2 * wq + 0.3 * bq, cq - 0.2 * wq - 0.3 * bq, cq + 0.78 * wq - 0.3 * bq],
            color=PEACH, alpha=0.72, zorder=13.5)
    s.arrow(qd, qd + wd, color=INDIGO, zorder=33)
    s.arrow(q, q + w, color=TERRACOTTA, zorder=33)
    if carry > 0:
        p = unit((1 - carry) * q + carry * qd) if carry < 1 else qd
        s.arrow(p, p + w, color=TERRACOTTA, lw=1.1, dashed=True, zorder=32)          # omega redrawn on the way to q_d
    if built:
        foot = qd + w - (w @ qd) * qd
        s.curve(np.array([foot, qd + w]), color=TEAL, lw=1.9, zorder=34, alpha=1.0)
        s.curve(np.array([qd, foot]), color=MUTED, lw=0.7, hidden=True, zorder=31)
        s.arrow(qd + wd, qd + w, color=INK, lw=1.2, head=0.09, zorder=33)
        s.label(qd + 0.5 * (wd + w), r"$\delta\omega$", dx=-13, dy=7, size=12)
        s.label(qd + w, r"$q_d\cdot\delta\omega$", dx=30, dy=9, color=TEAL)
    s.dot(qd, color=INK, size=4.5)
    s.dot(q, color=INK, size=4.5)
    s.label(qd, r"$q_d$", dx=9, dy=-10, size=12)
    s.label(q, r"$q$", dx=-10, dy=-9, size=12)
    s.label(qd + wd, r"$\omega_d$", dx=-14, dy=-2, color=INDIGO, size=12)
    s.label(q + w, r"$\omega$", dx=12, dy=-3, color=TERRACOTTA, size=12)
    s.text2d(-2.15, 0.55, r"$\omega_d$ lies in the" + "\n" + r"tangent plane at $q_d$", fontsize=9.5, color=INDIGO,
             linespacing=1.3, **SANS)
    s.text2d(2.25, -0.45, r"$\omega$ lies in the" + "\n" + r"tangent plane at $q$", fontsize=9.5, color=TERRACOTTA,
             linespacing=1.3, **SANS)
    if carry > 0:
        s.text2d(2.05, 1.45, r"dashed: $\omega$ redrawn at $q_d$", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.text2d(-2.15, -0.45, f"between $q_d$ and $q$: {np.degrees(ang):.0f}°", fontsize=9.5, color=INK, **SANS)
    if t < 0.1:
        msg, col = "two directions, each twirled sideways: each angular velocity lies in its own tangent plane", INK
    elif t < 0.42:
        msg, col = r"carry $\omega$ over to $q_d$: it sticks out of the plane there", TERRACOTTA
    elif t < 0.56:
        msg, col = r"so $\delta\omega = \omega - \omega_d$ has a component along $q_d$", TEAL
    else:
        msg, col = "bring the directions together and that component shrinks: it is not free", TEAL
    s.text2d(0.0, -1.6, msg, fontsize=10, color=col, **SANS)
    return s


def two_planes():
    animate(planes_frame, OUT / "ch07-planes.gif", OUT / "ch07-planes-strip.png", n=84, fps=13,
            width_px=680, stills=(0.0, 0.27, 0.5, 1.0), strip_px=820, strip_cols=2, hold=1.6)


# --------------------------------- nonlinear error against linear prediction
LIN_T = 1.2
LIN_SHOWN = ((1, INDIGO), (10, TEAL), (30, TERRACOTTA))


def linear_data():
    """Error between two simulated swings and its linear prediction, for every initial angle of the figure."""
    tt = np.linspace(0, LIN_T, 361)
    ref, _, lin = swing_pair(LIN_T, 0.0)
    Qd, Xi = ref.sol(tt)[:3].T, lin.sol(tt)[:3].T
    out = dict(t=tt, band=np.linalg.norm(Xi, axis=1), runs={})
    for deg in [d for d, _ in LIN_SHOWN]:
        e = np.radians(deg)
        eq = np.cross(Qd, swing_pair(LIN_T, e)[1].sol(tt)[:3].T)
        out["runs"][deg] = dict(ratio=np.linalg.norm(eq, axis=1) / e,
                                rel=np.linalg.norm(eq - e * Xi, axis=1) / np.linalg.norm(e * Xi, axis=1))
    return out


def linear_frame(t):
    D = cached("linear", linear_data)
    now = LIN_T * phase(t, 0.0, 0.9)
    k = max(1, int(round(now / LIN_T * (len(D["t"]) - 1))))
    done = now >= LIN_T - 1e-9
    fig, (ax0, ax1) = new_plot(6.9, 3.4, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.3, top=0.92, wspace=0.3)
    tt = D["t"][: k + 1]
    ax0.plot(D["t"], D["band"], color=MUTED, alpha=0.16, lw=5.5, zorder=4, solid_capstyle="round")
    ax0.plot(tt, D["band"][: k + 1], color=MUTED, alpha=0.5, lw=5.5, zorder=5, solid_capstyle="round")
    for (deg, col), off in zip(LIN_SHOWN, (9, 0, -8)):
        y = D["runs"][deg]["ratio"][: k + 1]
        ax0.plot(tt, y, color=col, lw=1.6, zorder=6)
        ax0.plot(tt[-1:], y[-1:], "o", ms=4.5, color=col, zorder=7)
        ax0.annotate(f"{deg}°", (tt[-1], y[-1]), xytext=(7, off), textcoords="offset points", color=col, fontsize=9.5,
                     va="center", **SANS)
    ax0.text(0.04, 1.47, r"gray band: linear prediction $\|\xi(t)\|$", fontsize=10.5, color=MUTED, path_effects=halo())
    ax0.text(0.04, 0.1, r"lines: simulated $\|e_q(t)\|$ / initial angle", fontsize=10.5, path_effects=halo())
    ax0.set_xlim(0, LIN_T * 1.12)
    ax0.set_ylim(0, 1.6)
    ax0.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax0.set_ylabel("error per unit of initial error", fontsize=9.5, **SANS)

    # right: how far off the prediction is right now, one bar per swing
    for row, (deg, col) in enumerate(LIN_SHOWN):
        rel = 100 * D["runs"][deg]["rel"][k]
        ax1.barh(2 - row, rel, height=0.5, color=col, zorder=3)
        ax1.text(rel + 1.2, 2 - row, f"{rel:.1f}%" if rel < 10 else f"{rel:.0f}%", color=col, fontsize=10, va="center",
                 fontweight="semibold" if done else "normal", **SANS)
    ax1.set_yticks([2, 1, 0])
    ax1.set_yticklabels(["1°", "10°", "30°"])
    ax1.set_ylim(-0.6, 2.6)
    ax1.set_xlim(0, 58)
    ax1.set_xticks([0, 10, 20, 30, 40, 50])
    ax1.set_xticklabels(["0", "10%", "20%", "30%", "40%", "50%"])
    ax1.set_xlabel("relative error of the prediction now", fontsize=9.5, **SANS)
    ax1.set_ylabel("initial angle", fontsize=9.5, **SANS)
    ax1.text(57, 2.45, f"t = {now:.2f} s", fontsize=10, ha="right", **SANS)
    if now < 0.2:
        msg, col = "three neighbors of one swing, released 1°, 10° and 30° away, and one linear prediction for all", INK
    elif now < 0.5:
        msg, col = "the 1° and 10° errors stay on the prediction", INK
    elif not done:
        msg, col = "the 30° error peels away", TERRACOTTA
    else:
        msg, col = "after 1.2 s the prediction is off by 1.6%, 16% and 48%: in proportion to the initial angle", INK
    fig.text(0.5, 0.06, msg, fontsize=10, color=col, ha="center", va="center", **SANS)
    return fig


def linear_vs_nonlinear():
    animate(linear_frame, OUT / "ch07-linear.gif", OUT / "ch07-linear-strip.png", n=84, fps=12, width_px=760,
            stills=(0.15, 0.5, 0.75, 1.0), strip_px=900, strip_cols=2, hold=2.0)


# ---------------------------- holding a nose-up attitude: geometric vs Euler
HOLD_T, HOLD_SLOW, HOLD_FPS = 1.5, 5.0, 12


def hold_frame(t):
    h = cached("hold85", lambda: V.hold_compare(85.0, 20.0, (1, 1, 1)))
    now = HOLD_T * t
    k = int(round(now / h["geometric"]["t"][1]))
    s = Scene(width=7.0, height=3.6, azim=24, elev=22, xlim=(-1.25, 5.75), ylim=(-1.95, 1.65))
    at = lambda x, y: x * s.right + y * s.up
    for name, title, col, x in (("geometric", "geometric LQR", INDIGO, -0.2), ("euler", "Euler-angle LQR", TERRACOTTA, 1.75)):
        ctr = at(x, 0.1)
        box(s, ctr, h[name]["R"][k], BODY_DIMS, BODY_FACES)
        wire_box(s, ctr, h["Rd"], BODY_DIMS)
        s.text2d(x, 1.42, title, fontsize=10.5, color=col)
        s.text2d(x, -1.0, f"error {h[name]['angle'][k]:.1f}°", fontsize=10, color=col, **SANS)
    s.text2d(0.78, -1.28, "dashed: the target, at 85° of pitch", fontsize=9, color=MUTED, **SANS)

    ax = s.fig.add_axes([0.6, 0.3, 0.385, 0.58])
    style_axes(ax)
    for name, col, lw in (("euler", TERRACOTTA, 1.7), ("geometric", INDIGO, 2.0)):
        ax.plot(h[name]["t"][: k + 1], h[name]["angle"][: k + 1], color=col, lw=lw)
        ax.plot(h[name]["t"][k: k + 1], h[name]["angle"][k: k + 1], "o", ms=4.8, color=col, zorder=6, clip_on=False)
    k_peak = int(np.argmax(h["euler"]["angle"]))
    if k >= k_peak:
        ax.annotate(f"{h['euler']['angle'][k_peak]:.1f}°", (h["euler"]["t"][k_peak], h["euler"]["angle"][k_peak]),
                    xytext=(8, -2), textcoords="offset points", color=TERRACOTTA, fontsize=9.5, va="center", **SANS)
    ax.set_xlim(0, HOLD_T)
    ax.set_ylim(0, 190)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_title("attitude error angle (degrees)", fontsize=9.5, loc="left", **SANS)
    if now < 0.06:
        msg, col = "the same body, the same weights, the same start 20° from the target", INK
    elif k < k_peak:
        msg, col = "near the pole of its chart the Euler-angle LQR sees a large error and spends torque on it", TERRACOTTA
    elif now < 0.95:
        msg, col = "it rolls the body almost all the way over before it comes back", TERRACOTTA
    else:
        msg, col = "the geometric LQR went straight to the target, as it does at every attitude", INDIGO
    s.text2d(2.25, -1.56, msg, fontsize=10, color=col, **SANS)
    s.text2d(2.25, -1.8, f"t = {now:.2f} s, shown {HOLD_SLOW:.0f} times slower", fontsize=9.5, color=MUTED, **SANS)
    return s


def hold_figure():
    n = int(round(HOLD_T * HOLD_SLOW * HOLD_FPS)) + 1
    animate(hold_frame, OUT / "ch07-hold.gif", OUT / "ch07-hold-strip.png", n=n, fps=HOLD_FPS, width_px=760,
            stills=(0.0, 0.23, 0.4, 1.0), strip_px=900, strip_cols=2, hold=1.8)


# -------------------------------------- gain history along the flip reference
GAIN_SLOW, GAIN_FPS = 4.0, 12


def gain_data():
    ref = V.Flip()
    lq = V.flip_lqr(ref)
    fe = V.flip_euler_lqr(ref, lq["P_inf"])
    return dict(ref=ref, tg=fe["tg"], pitch=np.degrees(fe["euler"][:, 1]), ts=lq["ts"],
                geo=np.linalg.norm(lq["Ks"], axis=(1, 2)), tsE=fe["ts"], eul=np.linalg.norm(fe["Ks"], axis=(1, 2)))


def gain_frame(t):
    D = cached("gain", gain_data)
    ref = D["ref"]
    now = ref.T * t
    pitch = float(np.interp(now, D["tg"], D["pitch"]))
    near = abs(pitch) > 80
    s = Scene(width=7.0, height=3.6, azim=62, elev=20, xlim=(-1.35, 5.65), ylim=(-1.95, 1.65))
    at = lambda x, y: x * s.right + y * s.up
    box(s, at(-0.05, 0.15), ref(now)[0], 1.3 * BODY_DIMS, BODY_FACES)
    s.text2d(-0.05, 1.42, "the reference attitude $R_d(t)$", fontsize=10.5, color=INDIGO)
    s.text2d(-0.05, -1.0, f"reference pitch {int(round(pitch))}°".replace("-", "−"), fontsize=10, color=TERRACOTTA if near else INK, **SANS)

    ax0 = s.fig.add_axes([0.46, 0.7, 0.52, 0.2])
    ax1 = s.fig.add_axes([0.46, 0.3, 0.52, 0.34])
    for ax in (ax0, ax1):
        style_axes(ax)
        ax.set_xlim(0, ref.T)
        ax.axvline(now, color=MUTED, lw=0.6, zorder=1)
    m = D["tg"] <= now + 1e-9
    ax0.plot(D["tg"][m], D["pitch"][m], color=INK, lw=1.5)
    ax0.plot([now], [pitch], "o", ms=4.5, color=TERRACOTTA if near else INK, zorder=6, clip_on=False)
    for sgn in (1, -1):
        ax0.axhline(sgn * 90, color=TERRACOTTA, lw=0.8, ls=(0, (3.5, 3)))
    ax0.text(0.84, 48, "chart singularity at ±90°", color=TERRACOTTA, fontsize=9, **SANS)
    ax0.set_ylim(-100, 100)
    ax0.set_yticks([-90, 0, 90])
    ax0.set_xticklabels([])
    ax0.set_title("reference pitch (degrees)", fontsize=9.5, loc="left", pad=3, **SANS)
    for ts, y, col, lw in ((D["tsE"], D["eul"], TERRACOTTA, 1.7), (D["ts"], D["geo"], INDIGO, 2.0)):
        m = ts <= now + 1e-9
        ax1.semilogy(ts[m], y[m], color=col, lw=lw)
        ax1.plot([now], [np.interp(now, ts, y)], "o", ms=4.5, color=col, zorder=6, clip_on=False)
    ax1.text(0.57, 22, "gain designed in Euler angles", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax1.text(0.62, 2.55, r"gain designed in $(\eta, \delta\Omega)$", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax1.set_ylim(2, 30)
    ax1.set_yticks([2, 5, 10, 20])
    ax1.set_yticklabels(["2", "5", "10", "20"])
    ax1.minorticks_off()
    ax1.set_xlabel("time along the flip (s)", fontsize=9.5, **SANS)
    ax1.set_ylabel(r"$\|K(t)\|$", fontsize=10.5)
    if now < 0.2:
        msg, col = "one full turn in 1.5 s, and a Riccati gain for it designed in two linear states", INK
    elif now < 0.56:
        msg, col = "as the reference pitch climbs toward 90°, the gain designed in Euler angles swells", TERRACOTTA
    elif now < 1.0:
        msg, col = "and it swells again before the second pass, at −85° of pitch", TERRACOTTA
    else:
        msg, col = r"in $(\eta, \delta\Omega)$ the gain stays between 3.83 and 4.09 for the whole flip", INDIGO
    s.text2d(2.15, -1.56, msg, fontsize=10, color=col, **SANS)
    s.text2d(2.15, -1.8, f"t = {now:.2f} s, shown {GAIN_SLOW:.0f} times slower", fontsize=9.5, color=MUTED, **SANS)
    return s


def gain_history():
    n = int(round(V.Flip().T * GAIN_SLOW * GAIN_FPS)) + 1
    animate(gain_frame, OUT / "ch07-gain.gif", OUT / "ch07-gain-strip.png", n=n, fps=GAIN_FPS, width_px=760,
            stills=(0.1, 0.34, 0.62, 1.0), strip_px=900, strip_cols=2, hold=1.8)


# ---------------------------------------------------- where it stops working
def large_errors():
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.9, wspace=0.26)
    inv = V.s2_inverted_lqr()
    K = inv["K"]
    kp = K[0, 1]
    th_sin = np.degrees(np.arccos(V.M_P * V.GRAV / kp))
    for mode, col, bound in (("sin", INDIGO, th_sin), ("log", TEAL, None)):
        degs = np.r_[np.arange(5, 70, 5), np.arange(70, 80, 0.25)]
        ts = np.array([V.s2_settle(K, np.radians(d), mode) for d in degs])
        ok = np.isfinite(ts)
        ax0.plot(degs[ok], ts[ok], color=col, lw=1.8)
        ax0.axvline(degs[ok][-1] + 0.125, color=col, lw=0.9, ls=(0, (3.5, 3)))
    ax0.axvspan(78, 90, color=PEACH, alpha=0.5, lw=0)
    ax0.text(84, 2.0, "falls", color=TERRACOTTA, fontsize=10, ha="center", rotation=90, **SANS)
    ax0.text(30, 1.75, r"error $e_q$", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax0.text(33, 0.95, "logarithm error", color=TEAL, fontsize=10.5, path_effects=halo())
    ax0.annotate("72.2°", (th_sin, 3.72), xytext=(-5, 0), textcoords="offset points", ha="right", color=INDIGO,
                 fontsize=9.5, **SANS)
    ax0.annotate("77.2°", (77.2, 3.72), xytext=(4, 0), textcoords="offset points", ha="left", color=TEAL,
                 fontsize=9.5, **SANS)
    ax0.set_xlim(0, 90)
    ax0.set_ylim(0, 4)
    ax0.set_xticks([0, 30, 60, 90])
    ax0.set_title("inverted pendulum, force on the bob", fontsize=9.5, loc="left", **SANS)
    ax0.set_xlabel("initial angle from upright (degrees)", fontsize=9.5, **SANS)
    ax0.set_ylabel("time to settle within 1° (s)", fontsize=9.5, **SANS)

    _, _, K0 = V.hover_lqr()
    degs = np.r_[np.arange(10, 170, 10), [170, 175, 178, 179, 179.5, 179.8, 179.9, 179.97, 179.99]]
    for mode, col in (("sin", INDIGO), ("log", TEAL)):
        ax1.plot(degs, [V.so3_settle(K0, np.radians(d), mode) for d in degs], color=col, lw=1.8)
    ax1.text(98, 1.75, r"error $e_R$", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax1.text(120, 0.72, "logarithm error", color=TEAL, fontsize=10.5, path_effects=halo())
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 4)
    ax1.set_xticks([0, 45, 90, 135, 180])
    ax1.set_title("attitude hold, torque in the body frame", fontsize=9.5, loc="left", **SANS)
    ax1.set_xlabel("initial error angle (degrees)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch07-large-errors.svg")


if __name__ == "__main__":
    puzzle()
    variation()
    two_planes()
    linear_vs_nonlinear()
    hold_figure()
    gain_history()
    large_errors()
