"""Figures for Chapter 14.  Run: python figures/src/ch14.py

Every curve comes from the filters and simulations in verify/ch14.py.
Convention of the chapter: the truth is a thin dashed gray line with a hollow
marker, an estimate is a solid line with a filled marker, indigo for the filter
built on the geometry and terracotta for the conventional one.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "verify"))

from figkit import (DASH, INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, axis_through_box, box, caption, halo, new_plot, phase, quadrotor,
                    save, style_axes, unit)

_spec = importlib.util.spec_from_file_location("ch14_verify", ROOT / "verify" / "ch14.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
_CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def ellipse2(S, n=120, k=2.0):
    """Points of the k-sigma ellipse of a 2x2 covariance."""
    w, U = np.linalg.eigh(S)
    th = np.linspace(0, 2 * np.pi, n)
    return (U * np.sqrt(np.maximum(w, 0))) @ np.array([np.cos(th), np.sin(th)]) * k


# ------------------------------------------------ opening: two filters, one bad start
OPEN_T, OPEN_RUN = 12.0, 7


def opening_data():
    sc = V.Scenario()
    r = V.run_filters(sc, np.random.default_rng(4), 8, 170, 1.0, 1.0, fixed_angle=True, keep=True)
    out = {"sc": sc}
    for f in ("inv", "std"):
        out[f] = dict(R=np.array([h[0][OPEN_RUN] for h in r["hist"][f]]),
                      x=np.array([h[2][OPEN_RUN] for h in r["hist"][f]]),
                      att=np.degrees(r[f]["att"][:, OPEN_RUN]))
    return out


def opening_frame(t):
    D = cached("opening", opening_data)
    sc = D["sc"]
    now = OPEN_T * phase(t, 0.0, 0.94)
    k = int(round(now / sc.h))
    s = Scene(width=7.2, height=4.3, azim=-62, elev=38, xlim=(-10.5, 8.5), ylim=(-5.6, 7.0))
    g = 4.6
    s.patch([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]], color=SAND, alpha=0.45, lw=0.5)
    for p in sc.landmarks:
        s.curve(np.array([[p[0], p[1], 0.0], p + 0.9 * E3]), color=INK, lw=1.6, alpha=1.0)
        s.dot(p + 0.9 * E3, color=INK, size=5)
    s.label(sc.landmarks[0] + 0.9 * E3, "landmark", dy=10, size=9, **SANS)
    kl = int(round(10.0 / sc.h))
    s.curve(sc.x[: kl + 1], color=MUTED, lw=0.9, linestyle=DASH, alpha=1.0)
    for f, col in (("std", TERRACOTTA), ("inv", INDIGO)):
        s.curve(D[f]["x"][: k + 1], color=col, lw=1.7, alpha=1.0, zorder=30)
        quadrotor(s, D[f]["x"][k], D[f]["R"][k], arm=0.55, color=col, lw=1.6, zorder=36)
        s.arrow(D[f]["x"][k], D[f]["x"][k] + 0.9 * D[f]["R"][k][:, 2], color=col, lw=1.3, head=0.25, zorder=37)
    quadrotor(s, sc.x[k], sc.R[k], arm=0.55, color=MUTED, lw=1.2, dashed=True, zorder=34)
    s.dot(sc.x[k], color=MUTED, size=4.5, hollow=True, zorder=40)
    s.text2d(-9.0, 6.5, f"t = {now:4.1f} s", fontsize=10, **SANS)
    s.text2d(5.4, 6.5, f"invariant: {D['inv']['att'][k]:5.1f}° off", fontsize=10, color=INDIGO, **SANS)
    s.text2d(5.4, 5.9, f"conventional: {D['std']['att'][k]:5.1f}° off", fontsize=10, color=TERRACOTTA, **SANS)
    if now < 0.6:
        msg, col = "both filters start 170° wrong in attitude: the estimates hang upside down", INK
    elif now < 3.0:
        msg, col = "same readings, same covariance: the conventional estimate drifts away", TERRACOTTA
    elif now < 7.0:
        msg, col = "the invariant estimate has turned upright and joins the true path (dashed)", INDIGO
    else:
        msg, col = "the conventional estimate comes back the long way round", TERRACOTTA
    caption(s, msg, color=col)
    return s


def opening():
    animate(opening_frame, OUT / "ch14-opening.gif", OUT / "ch14-opening-strip.png", n=96, fps=12, width_px=720,
            stills=(0.0, 0.2, 0.45, 1.0), strip_px=900, strip_cols=2, hold=2.0)


# ------------------------------------------------------------ compass at the wrap
def compass():
    t, truth, hist = V.compass_demo()
    fig, ax = new_plot(6.4, 3.0)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.17, top=0.95)
    deg = lambda a: np.degrees(a)
    ax.axhline(180, color=INK, lw=0.6)
    ax.plot(t, deg(truth), color=MUTED, lw=1.0, linestyle=DASH)
    ax.plot(t, deg(hist["subtract"]), color=TERRACOTTA, lw=1.7)
    ax.plot(t, deg(hist["wrap"]), color=INDIGO, lw=1.7)
    ax.text(0.3, 186, "180°: the compass wraps to −180°", fontsize=9, **SANS)
    ax.text(9.0, 196, "truth (dashed) and the filter on the circle", fontsize=9.5, color=INDIGO, ha="center", va="top",
            path_effects=halo(), **SANS)
    ax.text(7.6, 20, "the filter that subtracts", fontsize=9.5, color=TERRACOTTA, ha="left", path_effects=halo(), **SANS)
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("heading estimate (degrees)", fontsize=9.5, **SANS)
    ax.set_xlim(0, 12)
    ax.set_ylim(-110, 240)
    ax.set_yticks([-90, 0, 90, 180])
    save(fig, OUT / "ch14-compass.svg")


# --------------------------------------------------------------------- banana
def banana_frame(t):
    b = cached("banana", lambda: V.banana(noisy=False, sig0=(0.25, 0.25, 0.4)))
    n = b["chi"].shape[0] - 1
    k = int(round(n * phase(t, 0.05, 0.9)))
    P, chi, S = b["poses"][k], b["chi"][k], b["cov"][k]
    bar = b["bar"][k]
    fig, (ax0, ax1) = new_plot(7.2, 3.7, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.9, wspace=0.24)
    sub = slice(0, 900)
    ax0.plot(b["bar"][: k + 1, 1], b["bar"][: k + 1, 2], color=MUTED, lw=0.9, linestyle=DASH)
    ax0.plot(P[sub, 1], P[sub, 2], ".", ms=2.2, color=INK, alpha=1.0, zorder=3)
    m, C = P[:, 1:].mean(axis=0), np.cov(P[:, 1:].T)
    e = ellipse2(C)
    ax0.plot(m[0] + e[0], m[1] + e[1], color=TERRACOTTA, lw=1.6, zorder=5)
    ax0.plot([m[0]], [m[1]], "o", ms=4.5, color=TERRACOTTA, zorder=6)
    # image of the covariance ellipse (two largest axes) from exponential coordinates
    w, U = np.linalg.eigh(S)
    th = np.linspace(0, 2 * np.pi, 160)
    cc = 2.0 * (np.outer(np.cos(th), U[:, 2] * np.sqrt(w[2])) + np.outer(np.sin(th), U[:, 1] * np.sqrt(w[1])))
    img = V.se2_mul((bar[0:1], bar[1:2], bar[2:3]), V.se2_exp(cc))
    ax0.plot(img[1], img[2], color=INDIGO, lw=1.8, zorder=6)
    ax0.plot([bar[1]], [bar[2]], "o", ms=4.5, color=INDIGO, zorder=7)
    ax0.set_xlim(-2.5, 12.5)
    ax0.set_ylim(-8.5, 8.5)
    ax0.set_aspect("equal")
    ax0.set_xlabel("position, forward (m)", fontsize=9.5, **SANS)
    ax0.set_ylabel("position, sideways (m)", fontsize=9.5, **SANS)
    ax0.set_title("positions of 900 of the samples", fontsize=10, **SANS)
    ax1.plot(np.degrees(chi[sub, 2]), chi[sub, 1], ".", ms=2.2, color=INK, alpha=1.0, zorder=3)
    e = ellipse2(S[np.ix_([2, 1], [2, 1])])
    ax1.plot(np.degrees(e[0]), e[1], color=INDIGO, lw=1.8, zorder=5)
    ax1.set_xlim(-75, 75)
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel("heading component of $\\chi$ (degrees)", fontsize=9.5, **SANS)
    ax1.set_ylabel("sideways component of $\\chi$ (m)", fontsize=9.5, **SANS)
    ax1.set_title("the same samples in exponential coordinates", fontsize=10, **SANS)
    dist = k * b["h"]
    if k == 0:
        msg, col = "start: 0.25 m of position and 23° of heading uncertainty", INK
    elif dist < 5:
        msg, col = f"after {dist:.0f} m of dead reckoning the cloud of positions starts to bend", INK
    elif k < n:
        msg, col = f"after {dist:.0f} m: the fitted Gaussian (terracotta) has its mean where few samples are", TERRACOTTA
    else:
        msg, col = "after 10 m: a banana in position, an ellipse in exponential coordinates (indigo)", INDIGO
    fig.text(0.5, 0.045, msg, fontsize=10, color=col, ha="center", **SANS)
    return fig


def banana():
    animate(banana_frame, OUT / "ch14-banana.gif", OUT / "ch14-banana-strip.png", n=72, fps=12, width_px=760,
            stills=(0.0, 0.4, 0.65, 1.0), strip_px=900, strip_cols=2, hold=2.0)


# ---------------------------------------------------------- autonomous error
def autonomous():
    two = V.two_trajectories()
    t = two["t"]
    c = 1                                    # the world-frame component shown
    fig, (ax0, ax1) = new_plot(7.0, 3.0, ncols=2)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.9, wspace=0.26)
    ax0.plot(t, two["figure"]["Er"][:, c, 3], color=INDIGO, lw=2.4)
    ax0.plot(t, two["hover"]["Er"][:, c, 3], color=PAPER, lw=1.0, linestyle=(0, (2, 2.5)))
    ax0.set_title(r"right-invariant: second component of $\hat v - E_R\,v$", fontsize=10)
    ax0.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax0.set_ylabel("m/s", fontsize=9.5, **SANS)
    ax0.text(1.2, -4.0, "loop and hover:\none curve", fontsize=9.5, color=INDIGO, va="top", **SANS)
    for name, ls in (("figure", "-"), ("hover", (0, (2, 2.5)))):
        ax1.plot(t, two[name]["add_v"][:, c], color=TERRACOTTA, lw=1.9, linestyle=ls)
        ax1.plot(t, two[name]["lin_v"][:, c], color=MUTED, lw=0.9, linestyle=ls)
    ax1.text(0.9, -19.5, "loop (solid)\nhover (dashed)", fontsize=9.5, color=TERRACOTTA, ha="right", va="top", **SANS)
    ax1.text(2.95, 7.6, "thin: predicted by the linearization", fontsize=9, color=MUTED, ha="right", va="top", **SANS)
    ax1.set_ylim(-27, 8)
    ax0.set_ylim(-27, 8)
    ax1.set_title(r"additive: second component of $\hat v - v$", fontsize=10)
    ax1.set_xlabel("time (s)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch14-autonomous.svg")
    return ax0, ax1


# ---------------------------------------------------------------- convergence
def convergence():
    sc, rows = cached("mc", V.mc_table)
    row = rows[170]
    fig, ax = new_plot(6.4, 3.1)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.16, top=0.95)
    for f, col, name in (("std", TERRACOTTA, "conventional"), ("inv", INDIGO, "invariant")):
        ax.fill_between(sc.t, row[f]["curve50"], row[f]["curve90"], color=col, alpha=0.18, lw=0)
        ax.plot(sc.t, row[f]["curve90"], color=col, lw=0.7)
        ax.plot(sc.t, row[f]["curve50"], color=col, lw=1.9)
    ax.axhline(2.0, color=INK, lw=0.6, linestyle=DASH)
    ax.text(19.8, 2.25, "2°", fontsize=9, ha="right", **SANS)
    ax.text(2.3, 0.55, "invariant", fontsize=10, color=INDIGO, **SANS)
    ax.text(11.5, 28, "conventional", fontsize=10, color=TERRACOTTA, **SANS)
    ax.set_yscale("log")
    ax.set_ylim(0.08, 250)
    ax.set_xlim(0, 20)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1°", "1°", "10°", "100°"])
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("attitude error", fontsize=9.5, **SANS)
    save(fig, OUT / "ch14-convergence.svg")


# -------------------------------------------------------------------- contact
def contact():
    s = Scene(width=6.0, height=3.9, azim=-58, elev=16)
    R = V.expm_so3(np.array([0.10, 0.16, 0.25]))
    c = np.array([0.0, 0.0, 1.0])
    foot = np.array([0.42, -0.25, 0.0])
    knee = np.array([0.52, -0.12, 0.52])
    dims = (0.62, 0.42, 0.34)
    g = 1.1
    s.patch([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]], color=SAND, alpha=0.5, lw=0.5)
    hip = c + R @ np.array([0.12, -0.12, -0.17])
    s.curve(np.array([hip, knee, foot]), color=INK, lw=2.4, alpha=1.0, zorder=20)
    s.dot(knee, color=INK, size=4.5, zorder=21)
    faces = {(0, 1): PERIWINKLE, (0, -1): PERIWINKLE, (1, 1): SAND, (1, -1): SAND, (2, 1): PEACH, (2, -1): PEACH}
    box(s, c, R, dims, faces)
    for i, col in enumerate((TERRACOTTA, TEAL, INDIGO)):
        axis_through_box(s, c, R, dims, R[:, i], 0.62, col, lw=1.3, head=0.07)
    s.arrow(c + 0.02 * s.right, foot + 0.02 * s.right, color=TEAL, lw=1.6, head=0.08, dashed=True, zorder=60)
    s.dot(foot, color=INK, size=6, zorder=61)
    s.label(0.45 * c + 0.55 * foot, "$y$", dx=12, dy=4, color=TEAL)
    s.label(foot, "$x_c$", dx=14, dy=-9)
    s.label(c + 0.32 * R[:, 2], "body, with the IMU", dx=-64, dy=12, size=9.5, **SANS)
    s.label(foot, "contact: fixed in the world, position unknown", dx=0, dy=-24, size=9.5, **SANS)
    s.label(c, "$(R, v, x)$", dx=-70, dy=-6)
    s.save(OUT / "ch14-contact.svg")


# --------------------------------------------------------------------- sphere
SPH = dict(sig0=0.45, sig_y=0.2, steps=(0, 3))


def sphere():
    truth = V.s2_truth(T=1.2)
    r = V.s2_filters(truth, 104, sig0=SPH["sig0"], sig_y=SPH["sig_y"], keep_cov=True)
    t, qs, _ = truth
    qv, qn, covs = r["var"]["q"], r["naive"]["q"], r["var"]["cov"]
    view = unit(qv[3] + 0.35 * E3 - 0.5 * np.cross(E3, qv[3]))
    az, el = np.degrees(np.arctan2(view[1], view[0])), np.degrees(np.arcsin(view[2]))
    s = Scene(width=6.2, height=4.6, azim=az, elev=el)
    s.sphere(alpha=1.0)
    s.sphere_curve(qs, color=MUTED, lw=1.1, linestyle=DASH)
    s.sphere_curve(qv[:100], color=INDIGO, lw=1.6)
    for k in SPH["steps"]:
        q = qv[k]
        P3 = covs[k][:3, :3]
        a = unit(np.cross(q, E3))
        b = np.cross(q, a)
        S2 = np.array([[a @ P3 @ a, a @ P3 @ b], [b @ P3 @ a, b @ P3 @ b]])
        e = ellipse2(S2, k=1.0)
        half = 1.25 * np.sqrt(np.max(np.linalg.eigvalsh(S2)))
        s.tangent_plane(q * 1.002, size=half, color=SAND, alpha=0.6)
        pts = q * 1.004 + np.cross(np.outer(e[0], a) + np.outer(e[1], b), q)   # xi turns q: the spread is xi x q
        s.curve(pts, color=INDIGO, lw=1.5, alpha=1.0, zorder=45)
        s.dot(q * 1.004, color=INDIGO, size=4.8, zorder=46)
        s.dot(qs[k], color=MUTED, size=4.5, hollow=True, zorder=44)
    s.curve(qn[:40], color=TERRACOTTA, lw=1.3, alpha=1.0, zorder=47)
    kn = int(np.argmax(np.abs(np.linalg.norm(qn[:40], axis=1) - 1)))
    s.dot(qn[kn], color=TERRACOTTA, size=4.8, zorder=48)
    info = dict(kn=kn, norm=float(np.linalg.norm(qn[kn])), err0=float(np.degrees(np.arccos(qv[0] @ qs[0]))))
    s.label(qn[kn], f"length {info['norm']:.2f}", dx=40, dy=6, size=9.5, color=TERRACOTTA, **SANS)
    s.label(qv[0], "first estimate", dx=-46, dy=10, size=9.5, color=INDIGO, **SANS)
    s.label(qs[110], "true direction", dx=40, dy=-6, size=9.5, color=MUTED, **SANS)
    s.save(OUT / "ch14-sphere.svg")
    print(info)
    return info


FIGS = dict(opening=opening, compass=compass, banana=banana, autonomous=autonomous,
            convergence=convergence, contact=contact, sphere=sphere)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
