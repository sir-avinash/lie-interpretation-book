"""Figures for Chapter 8.  Run: python figures/src/ch08.py

Every curve is a simulation from verify/ch08.py, the script that checks the chapter.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "verify"))
import ch08 as m  # noqa: E402
from figkit import (INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, geodesic, halo, new_plot, phase, rot, save, small_circle, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
DASHED = (0, (3.5, 3))


def circle3(center, normal, radius, n=60):
    normal = unit(normal)
    a = unit(np.cross(normal, [0.31, 0.52, 0.8]))
    b = np.cross(normal, a)
    t = np.linspace(0, 2 * np.pi, n)
    return np.asarray(center) + radius * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))


def draw_quad(s, x, R, arm=0.5, color=INK, lw=1.5, alpha=1.0, dashed=False, zorder=24):
    """A quadrotor drawn from its real pose: two arms and four rotor disks."""
    kw = dict(color=color, lw=lw, alpha=alpha, zorder=zorder)
    if dashed:
        kw["linestyle"] = DASHED
    for d in (R[:, 0], R[:, 1]):
        s.curve(np.array([x - arm * d, x + arm * d]), **kw)
    for d in (R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]):
        s.curve(circle3(x + arm * d + 0.04 * arm * R[:, 2], R[:, 2], 0.36 * arm), **{**kw, "lw": 0.75 * lw})


# ------------------------------------------------------------ opening picture
def puzzle():
    s = Scene(width=6.6, height=3.3, azim=-64, elev=14)
    o = m.run_attitude("pd", m.ref_hover, m.dist_camera, 6.0)
    R = o["R_final"]
    off = np.degrees(o["angle"][-1])
    # --- left: the vehicle that will not level
    x0 = np.array([-1.55, 0.0, 0.0])
    s.shadow(x0 - 1.0 * E3, 0.75, strength=0.16)
    draw_quad(s, x0, np.eye(3), arm=0.62, color=MUTED, lw=1.0, dashed=True, zorder=18)
    draw_quad(s, x0, R, arm=0.62)
    cam = x0 + R @ (m.R_ADD * 0.62 / 0.25)
    s.curve(np.array([x0 + R @ np.array([m.R_ADD[0] * 0.62 / 0.25, 0, 0]), cam]), color=TERRACOTTA, lw=1.4, alpha=1.0, zorder=26)
    s.dot(cam, color=TERRACOTTA, size=8.5, zorder=27)
    s.label(cam, "0.5 kg camera,\n8 cm off-center", dx=4, dy=-21, size=9.5, color=TERRACOTTA, linespacing=1.15, **SANS)
    s.arrow(x0, x0 + 1.05 * E3, color=MUTED, lw=1.1, head=0.09, dashed=True)
    s.arrow(x0, x0 + 1.05 * R[:, 2], color=INDIGO, lw=1.7, head=0.11)
    s.label(x0 + 1.05 * E3, r"$R_d\,e_3$", dx=-19, dy=3, color=MUTED, size=11.5)
    s.label(x0 + 1.05 * R[:, 2], r"$R\,e_3$", dx=17, dy=2, color=INDIGO, size=11.5)
    arc = np.array([x0 + 0.78 * (np.cos(a) * E3 + np.sin(a) * unit(R[:, 2] - R[2, 2] * E3)) for a in np.linspace(0, np.radians(off), 20)])
    s.curve(arc, color=INK, lw=0.9, alpha=1.0, zorder=31)
    s.label(x0 + 0.78 * unit(E3 + R[:, 2]), f"{off:.1f}°", dx=27, dy=-3, size=10, **SANS)
    s.label(x0 - 1.0 * E3, "the tracking error settles here, and stays", dy=-14, size=10, **SANS)
    # --- right: the wall nobody told the tracker about
    w0 = np.array([1.75, 0.0, 0.0])
    s.shadow(w0 + np.array([-0.95, 0, -1.0]), 0.6, strength=0.13)
    s.patch([w0 + np.array([0, -0.9, -0.95]), w0 + np.array([0, 0.9, -0.95]), w0 + np.array([0, 0.9, 0.95]), w0 + np.array([0, -0.9, 0.95])],
            color=SAND, alpha=0.85, zorder=22)
    xq = w0 + np.array([-0.95, 0.0, 0.05])
    Rq = rot(E2, np.radians(9))
    draw_quad(s, xq, Rq, arm=0.4, zorder=28)
    path = np.array([w0 + np.array([x, 0.0, 0.05 + 0.1 * np.sin(1.3 * x)]) for x in np.linspace(-2.0, 0.0, 40)])
    s.curve(path[:22], color=INDIGO, lw=1.4, alpha=1.0, linestyle=DASHED, zorder=17)
    s.curve(path[27:], color=INDIGO, lw=1.4, alpha=1.0, linestyle=DASHED, zorder=30)
    beyond = np.array([w0 + np.array([x, 0.0, 0.05 + 0.1 * np.sin(1.3 * x)]) for x in np.linspace(0.0, 0.9, 20)])
    s.curve(beyond[:-2], color=TERRACOTTA, lw=1.4, alpha=1.0, linestyle=DASHED, zorder=16)
    s.arrow(beyond[-3], beyond[-1], color=TERRACOTTA, lw=1.4, head=0.1, zorder=16)
    s.label(w0 + np.array([0, 0.0, 0.95]), "wall", dx=-3, dy=9, size=10, **SANS)
    s.label(beyond[-1], r"$x_d(t)$", dx=14, dy=8, color=TERRACOTTA, size=11.5)
    s.label(w0 + np.array([-0.55, 0, -1.0]), "the reference goes straight through", dy=-14, size=10, **SANS)
    s.save(OUT / "ch08-puzzle.svg", pad=0.1)


# ------------------------------------------------------ nominal against L1
def l1():
    h = 1e-3
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.88, wspace=0.27)
    for ctrl, col in (("pd", TERRACOTTA), ("l1", INDIGO)):
        o = m.run_attitude(ctrl, m.ConingReference(h), m.dist_camera_gust, 10.0)
        ax0.semilogy(o["t"][5:], np.degrees(o["angle"][5:]), color=col, lw=1.5)
    ax0.set_ylim(3e-3, 60)
    ax0.set_xlim(0, 10)
    ax0.set_yticks([0.01, 0.1, 1, 10])
    ax0.set_yticklabels(["0.01°", "0.1°", "1°", "10°"])
    ax0.text(9.8, 24, "geometric PD", color=TERRACOTTA, ha="right", fontsize=10, **SANS)
    ax0.text(9.8, 0.062, "with L1 adaptation", color=INDIGO, ha="right", fontsize=10, **SANS, path_effects=halo())
    ax0.set_title("attitude error, camera and gusts", fontsize=10, loc="left", color=INK, **SANS)
    ax0.set_xlabel("time (s)", fontsize=9.5, **SANS)

    o = m.run_attitude("l1", m.ConingReference(h), m.dist_camera_gust, 5.0, noise=0.02, kappa=10.0)
    sel = (o["t"] > 1.0) & (o["t"] < 5.0)
    ax1.plot(o["t"][sel], o["Dhat"][sel, 1], color=MUTED, lw=0.5, alpha=0.9)
    ax1.plot(o["t"][sel], o["Delta"][sel, 1], color=INK, lw=1.6)
    ax1.plot(o["t"][sel], o["Df"][sel, 1], color=TEAL, lw=1.6)
    ax1.set_xlim(1, 5)
    ax1.set_title("second component of the uncertainty (N·m)", fontsize=10, loc="left", color=INK, **SANS)
    ax1.set_xlabel("time (s)", fontsize=9.5, **SANS)
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo, hi + 0.22 * (hi - lo))
    y0 = hi + 0.17 * (hi - lo)
    ax1.text(1.1, y0, r"true $\Delta$", color=INK, fontsize=10, va="center")
    ax1.text(2.05, y0, r"raw estimate", color=MUTED, fontsize=10, va="center", **SANS)
    ax1.text(3.55, y0, r"after the filter", color=TEAL, fontsize=10, va="center", **SANS)
    save(fig, OUT / "ch08-l1.svg")


# ------------------------------------------------------------ the filter
def filter_tradeoff():
    import json
    vals = json.loads((ROOT / "verify" / "ch08_values.json").read_text())
    kap = np.array([1, 3, 10, 30, 100, 300])
    err = np.array([vals[f"cutoff_{k}_rms_deg"] for k in kap])
    rate = np.array([vals[f"cutoff_{k}_torque_rate"] for k in kap])
    fig, (ax0, ax1) = new_plot(6.6, 2.8, ncols=2)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.86, wspace=0.3)
    ax0.loglog(kap, err, "o-", color=INDIGO, lw=1.6, ms=4)
    ax0.set_ylim(0.03, 30)
    ax0.set_yticks([0.1, 1, 10])
    ax0.set_yticklabels(["0.1°", "1°", "10°"])
    ax0.axhline(vals["cutoff_pd_rms_deg"], color=TERRACOTTA, lw=1.0, ls=DASHED)
    ax0.text(1.0, vals["cutoff_pd_rms_deg"] * 1.25, "no adaptation", color=TERRACOTTA, fontsize=9.5, **SANS)
    ax0.text(18, 1.0, "RMS attitude error", color=INDIGO, fontsize=9.5, ha="left", **SANS)
    axr = ax0.twinx()
    axr.loglog(kap, rate, "s-", color=TEAL, lw=1.6, ms=4)
    axr.set_ylim(3, 3000)
    for side in ("top", "left"):
        axr.spines[side].set_visible(False)
    axr.spines["right"].set_color(TEAL)
    axr.tick_params(colors=TEAL, width=0.8, length=3, labelsize=9, which="both")
    for lab in axr.get_yticklabels():
        lab.set_fontfamily("Source Sans 3")
    ax0.text(1.0, 0.062, "torque rate (N·m/s, right axis)", color=TEAL, fontsize=9.5, ha="left", va="center", **SANS)
    ax0.set_xlabel("filter cutoff  κ  (rad/s)", fontsize=9.5, **SANS)
    ax0.set_title("too low is slow, too high is rough", fontsize=10, loc="left", color=INK, **SANS)

    ks = np.array([3, 30, 300])
    dm = np.array([vals[f"delay_margin_cutoff_{k}_ms"] for k in ks])
    ax1.semilogx(ks, dm, "o-", color=INDIGO, lw=1.6, ms=5)
    ax1.plot([30, 30], [vals["delay_margin_gamma_1000_ms"], vals["delay_margin_gamma_16000_ms"]], "o", ms=9, mfc="none", mec=TEAL, mew=1.1)
    ax1.axhline(vals["delay_margin_pd_ms"], color=TERRACOTTA, lw=1.0, ls=DASHED)
    ax1.text(1.2, vals["delay_margin_pd_ms"] + 1.5, "no adaptation", color=TERRACOTTA, fontsize=9.5, **SANS)
    ax1.annotate("same for adaptation gains\nγ = 1000, 4000, 16000", (30, dm[1]), xytext=(1.25, 10), fontsize=9.5, color=TEAL, ha="left",
                 va="center", linespacing=1.15, arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7, shrinkB=5), **SANS)
    ax1.set_xlim(1, 500)
    ax1.set_ylim(0, 54)
    ax1.set_xlabel("filter cutoff  κ  (rad/s)", fontsize=9.5, **SANS)
    ax1.set_title("input delay the loop survives (ms)", fontsize=10, loc="left", color=INK, **SANS)
    save(fig, OUT / "ch08-filter.svg")


# ------------------------------------------------------- a point and a wall
def wall():
    fig, (ax0, ax1) = new_plot(6.6, 3.0, ncols=2)
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.06, top=0.9, wspace=0.12)
    # left: velocity space at x = (0.9, 0), alpha = 5
    x, ud, alpha = np.array([0.9, 0.0]), np.array([1.1, 1.5]), 5.0
    u = m.wall_filter(x, ud, alpha)
    lim = alpha * m.wall_H(x)
    ax0.fill_betweenx([-0.5, 2.1], lim, 1.9, color=PEACH, alpha=0.55, lw=0)
    ax0.plot([lim, lim], [-0.5, 2.1], color=INK, lw=1.2)
    ax0.plot([0, 0], [-0.5, 2.1], color=MUTED, lw=0.5)
    ax0.plot([-0.9, 1.9], [0, 0], color=MUTED, lw=0.5)
    ax0.annotate("", ud, (0, 0), arrowprops=dict(arrowstyle="-|>", color=TERRACOTTA, lw=1.6, shrinkA=0, shrinkB=0))
    ax0.annotate("", u, (0, 0), arrowprops=dict(arrowstyle="-|>", color=INDIGO, lw=1.8, shrinkA=0, shrinkB=0))
    ax0.plot([u[0], ud[0]], [u[1], ud[1]], color=INK, lw=0.8, ls=DASHED)
    ax0.plot(*u, "o", ms=4, color=INDIGO)
    ax0.text(ud[0] + 0.06, ud[1] + 0.05, r"$u_d$", color=TERRACOTTA, fontsize=12)
    ax0.text(u[0] - 0.08, u[1] + 0.07, r"$u$", color=INDIGO, fontsize=12, ha="right")
    ax0.text(1.22, 0.45, "too fast\ntoward\nthe wall", color=TERRACOTTA, fontsize=9.5, ha="center", va="center", linespacing=1.2, **SANS)
    ax0.text(lim - 0.05, -0.38, r"$n\cdot u = \alpha H$", fontsize=11, ha="right", path_effects=halo())
    ax0.text(-0.85, 1.9, "allowed velocities", color=INK, fontsize=9.5, **SANS)
    ax0.set_xlim(-0.9, 1.9)
    ax0.set_ylim(-0.5, 2.1)
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.set_title("the nearest allowed velocity", fontsize=10, loc="left", color=INK, **SANS)
    # right: paths
    ax1.fill_betweenx([-0.25, 1.9], 1.0, 2.25, color=PEACH, alpha=0.55, lw=0)
    ax1.plot([1, 1], [-0.25, 1.9], color=INK, lw=1.7)
    t, xs = m.run_wall(None)
    ax1.plot(xs[:, 0], xs[:, 1], color=TERRACOTTA, lw=1.3, ls=DASHED)
    for alpha, lw in ((1.0, 1.1), (5.0, 1.8), (25.0, 1.1)):
        t, xs = m.run_wall(alpha)
        ax1.plot(xs[:, 0], xs[:, 1], color=INDIGO, lw=lw)
    ax1.plot([0], [0], "o", ms=4.5, color=INK)
    ax1.plot([2], [1.5], "o", ms=5, mfc=PAPER, mec=TERRACOTTA, mew=1.1)
    ax1.plot([1], [1.5], "o", ms=4.5, color=INDIGO)
    ax1.text(2.0, 1.63, r"target $x_d$", color=TERRACOTTA, fontsize=10.5, ha="center")
    ax1.text(0.0, -0.14, "start", fontsize=9.5, ha="center", **SANS)
    ax1.text(-0.25, 1.35, r"$\alpha = 1,\ 5,\ 25$", color=INDIGO, fontsize=11, ha="left")
    ax1.text(-0.25, 1.17, "from left to right", color=INDIGO, fontsize=9.5, ha="left", **SANS)
    ax1.text(1.62, 0.9, "no filter", color=TERRACOTTA, fontsize=9.5, ha="center", rotation=37, **SANS)
    ax1.text(1.06, -0.14, "wall", fontsize=9.5, ha="left", **SANS)
    ax1.set_xlim(-0.3, 2.25)
    ax1.set_ylim(-0.25, 1.9)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("the paths that result", fontsize=10, loc="left", color=INK, **SANS)
    save(fig, OUT / "ch08-wall.svg")


# ------------------------------------------------- the cone on the sphere
_CONE = {}


def _cone_data():
    if not _CONE:
        ref = m.cone_reference()
        lap = 2 * np.pi / 0.9
        o = m.run_cone(ref, lap)
        _CONE.update(o=o, lap=lap)
    return _CONE["o"], _CONE["lap"]


def cone_frame(t):
    o, lap = _cone_data()
    s = Scene(width=6.4, height=3.9, azim=-58, elev=-66, xlim=(-1.75, 1.75), ylim=(-1.36, 1.08))
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    n = len(o["t"])
    k = int(round((0.02 + 0.98 * (0.15 * ease(t) + 0.85 * t)) * (n - 1)))
    ref_angle = np.degrees(np.arccos(np.clip(-o["qd"][:, 2], -1, 1)))
    # the cone
    for ang, lw in ((10, 0.5), (20, 0.5)):
        s.sphere_curve(small_circle(-E3, np.radians(ang)), color=MUTED, lw=lw, show_hidden=False, alpha=0.5)
    s.sphere_curve(1.002 * small_circle(-E3, m.CONE_ANGLE), color=INK, lw=1.7, zorder=18)
    s.dot(-1.002 * E3, color=INK, size=3.5)
    s.label(-E3, r"$q_c$", dx=-10, dy=-8, size=12)
    # the whole reference, faint: safe part and unsafe part
    full = 1.003 * o["qd"]
    bad = ref_angle > 30
    idx = np.where(np.diff(bad.astype(int)) != 0)[0]
    a, b = (idx[0], idx[1]) if len(idx) >= 2 else (0, 0)
    s.sphere_curve(full[: a + 1], color=MUTED, lw=0.9, linestyle=DASHED, zorder=15)
    s.sphere_curve(full[b:], color=MUTED, lw=0.9, linestyle=DASHED, zorder=15)
    s.sphere_curve(full[a:b + 1], color=TERRACOTTA, lw=1.1, linestyle=DASHED, zorder=15)
    # what has happened so far
    s.sphere_curve(1.004 * o["q"][: k + 1], color=INDIGO, lw=2.2, zorder=19, alpha=1.0)
    q, qd = o["q"][k], o["qd"][k]
    if o["Psi"][k] > 2e-4:
        s.sphere_curve(1.004 * geodesic(q, qd, n=30), color=TERRACOTTA, lw=1.0, zorder=19)
    s.dot(1.004 * qd, color=TERRACOTTA if bad[k] else MUTED, size=6.5, hollow=True, zorder=40)
    s.dot(1.004 * q, color=INDIGO, size=6.5, zorder=41)
    s.label(1.004 * qd, r"$q_d$", dx=13, dy=-9, size=12, color=TERRACOTTA if bad[k] else MUTED)
    s.label(1.004 * q, r"$q$", dx=-11, dy=9, size=12, color=INDIGO)
    edge = small_circle(-E3, m.CONE_ANGLE)[150]
    s.label(edge, r"$H = 0$", dx=-26, dy=-4, size=11)
    s.label(unit(-E3 + 0.38 * s.up - 0.12 * s.right), "safe", size=9.5, color=INK, **SANS)
    s.label(unit(-E3 + 1.25 * s.up - 0.3 * s.right), "must never be here", size=9.5, color=TERRACOTTA, **SANS)
    if not bad[k] and k < a:
        msg, col = "the reference is safe: the state follows it", INDIGO
    elif bad[k]:
        msg, col = "the reference leaves the cone: the state slides along the boundary", TERRACOTTA
    elif o["Psi"][k] > 2e-4:
        msg, col = "the reference is back inside: the state rejoins it", INDIGO
    else:
        msg, col = "tracking again, as if nothing had happened", INDIGO
    s.text2d(0.0, -1.24, msg, fontsize=10.5, color=col, **SANS)
    return s


def cone():
    animate(cone_frame, OUT / "ch08-cone.gif", OUT / "ch08-cone-strip.png", n=90, fps=13,
            width_px=720, stills=(0.1, 0.42, 0.7, 1.0), strip_px=900, strip_cols=2)


# -------------------------------------------- a quadrotor and an obstacle
_QUAD = {}


def _quad_data():
    if not _QUAD:
        with np.errstate(all="ignore"):
            _QUAD["naive"] = m.run_quad("naive", tf=6.0, ref=m.quad_reference)
            _QUAD["aug"] = m.run_quad("augmented", tf=6.0, ref=m.quad_reference)
    return _QUAD["naive"], _QUAD["aug"]


def glyph(s, x, b3, color, size=0.2, lw=1.5):
    """Side view of a quadrotor: a bar for the body, a tick for the thrust axis."""
    side = np.cross(E2, b3)
    s.curve(np.array([x - size * side, x + size * side]), color=color, lw=lw + 0.6, alpha=1.0, zorder=34)
    s.arrow(x, x + 1.5 * size * b3, color=color, lw=lw - 0.3, head=0.07, zorder=34)


def obstacle_frame(t):
    naive, aug = _quad_data()
    s = Scene(width=6.6, height=3.3, azim=-90, elev=0, xlim=(-2.7, 4.3), ylim=(-1.6, 1.9))
    s.sphere(m.OBS_C, m.OBS_R, color=PEACH)
    ring = np.array([m.OBS_C + np.sqrt(m.AUG_BETA) * m.OBS_R * (np.cos(a) * E1 + np.sin(a) * E3) for a in np.linspace(0, 2 * np.pi, 120)])
    s.curve(ring, color=MUTED, lw=0.8, linestyle=DASHED, alpha=0.9)
    s.curve(np.array([[-2.65, 0, -0.1], [4.25, 0, -0.1]]), color=MUTED, lw=0.9, linestyle=DASHED, zorder=15)
    s.label((3.5, 0, -0.1), r"reference $x_d(t)$", dy=-10, size=10, color=MUTED)
    n = len(aug["t"])
    tt = 0.12 * ease(t) + 0.88 * t
    k = int(round((0.05 + 0.55 * tt) * (n - 1)))
    # the plain barrier: stops being drawn where it enters the obstacle
    hit = int(np.argmax(naive["dist"] < m.OBS_R))
    kn = min(k, hit)
    s.curve(naive["x"][: kn + 1], color=TERRACOTTA, lw=1.6, alpha=1.0, zorder=25)
    glyph(s, naive["x"][kn], naive["b3"][kn], TERRACOTTA)
    if k >= hit:
        s.dot(naive["x"][hit], color=TERRACOTTA, size=7, zorder=45)
        s.label(naive["x"][hit], "hits", dx=-15, dy=-10, size=9.5, color=TERRACOTTA, **SANS)
    s.curve(aug["x"][: k + 1], color=INDIGO, lw=1.9, alpha=1.0, zorder=26)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(aug["x"], axis=0), axis=1))])
    for mark in np.arange(1.0, arc[k] - 0.3, 0.55):
        j = int(np.searchsorted(arc, mark))
        glyph(s, aug["x"][j], aug["b3"][j], INDIGO, size=0.12, lw=0.9)
    glyph(s, aug["x"][k], aug["b3"][k], INDIGO)
    s.label(m.OBS_C, "obstacle", size=10, **SANS, halo="none")
    s.text2d(-1.55, 1.68, "plain barrier, thrust only", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.text2d(-1.55, 1.42, "augmented barrier, thrust and torque", fontsize=9.5, color=INDIGO, **SANS)
    d = aug["dist"][k]
    xk = aug["x"][k, 0]
    if xk < -1.45:
        msg, col = "both fly level toward the obstacle, 10 cm below its center", INK
    elif k < hit:
        msg, col = "thrust alone cannot push away: it points the wrong way", TERRACOTTA
    elif xk < 1.6:
        msg, col = "torque tilts the thrust axis away first, and the vehicle goes around", INDIGO
    else:
        msg, col = "past the obstacle: returning to the reference", INDIGO
    s.text2d(0.8, -1.42, msg, fontsize=10.5, color=col, **SANS)
    return s


def obstacle():
    animate(obstacle_frame, OUT / "ch08-obstacle.gif", OUT / "ch08-obstacle-strip.png", n=84, fps=13,
            width_px=720, stills=(0.1, 0.4, 0.62, 1.0), strip_px=900, strip_cols=2)


if __name__ == "__main__":
    puzzle()
    l1()
    filter_tradeoff()
    wall()
    cone()
    obstacle()
