"""Figures for Chapter 19: three quadrotors carry a payload, two carry a cable.

Every picture is drawn from the simulations of verify/ch19.py, read from its cache
(verify/ch19_cache.pkl; the cache is rebuilt when that script changes).

Run:  python figures/src/ch19.py [name ...]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "verify"))
_spec = importlib.util.spec_from_file_location("ch19_verify", ROOT / "verify" / "ch19.py")
m = importlib.util.module_from_spec(_spec)
sys.modules["ch19_verify"] = m
_spec.loader.exec_module(m)

from figkit import (DASH, INDIGO, INK, MUTED, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA, Scene,  # noqa: E402
                    animate, box, caption, new_plot, quadrotor, save, style_axes)

OUT = ROOT / "figures"
SANS = dict(fontfamily="Source Sans 3")
FACES = {(0, 1): PERIWINKLE, (0, -1): PERIWINKLE, (1, 1): SAND, (1, -1): SAND, (2, 1): PEACH, (2, -1): PEACH}
E3 = np.array([0.0, 0.0, 1.0])
_R = {}


def data():
    if not _R:
        _R.update(m.results(verbose=False))
    return _R


def team():
    return m.team()


# ------------------------------------------------------------------ shared drawing
def draw_payload(s, x_L, R_L, zorder=30):
    box(s, x_L + R_L @ np.array([0.1, 0.0, 0.0]), R_L, (0.62, 0.72, 0.05), FACES, zorder=zorder)


def draw_team(s, x_L, R_L, x_Q, R, p, tension=None, arm=0.25, zorder=30, cable=INK):
    draw_payload(s, x_L, R_L, zorder)
    for i in range(len(x_Q)):
        a = x_L + R_L @ p.r[i]
        col = cable if tension is None else (TERRACOTTA if tension[i] < m.T_MIN else cable)
        s.curve(np.array([a, x_Q[i]]), color=col, lw=1.1, alpha=1.0, zorder=zorder + 6)
        s.dot(a, color=INK, size=2.6, zorder=zorder + 7)
        quadrotor(s, x_Q[i], R[i], arm=arm, lw=1.5, zorder=zorder + 8)


def draw_gap(s, z0=0.0, z1=2.9, depth=0.12):
    """The two walls of the gap: the forbidden side is pale terracotta, the edge ink."""
    for sign in (1, -1):
        y0, y1 = sign * m.GAP_W / 2, sign * (m.GAP_W / 2 + 0.55)
        corners = [(m.GAP_X, y0, z0), (m.GAP_X, y1, z0), (m.GAP_X, y1, z1), (m.GAP_X, y0, z1)]
        s.patch(corners, color=PEACH, alpha=0.55, edge=INK, lw=0.8, zorder=8 if sign > 0 else 60)
        s.curve(np.array([(m.GAP_X, y0, z0), (m.GAP_X, y0, z1)]), color=INK, lw=1.5, alpha=1.0,
                zorder=9 if sign > 0 else 61)


def floor(s, x0=-0.9, x1=7.0, y=1.5):
    s.patch([(x0, -y, 0), (x1, -y, 0), (x1, y, 0), (x0, y, 0)], color="#f4f1e8", alpha=1.0, edge=MUTED,
            lw=0.4, zorder=1)


def scene_mission(width=7.0, height=3.5):
    return Scene(width=width, height=height, azim=-66, elev=19, xlim=(-1.6, 7.3), ylim=(-0.75, 3.7))


# ------------------------------------------------------------------ 1. the mission
def mission():
    p = team().params
    plan = m.Plan()
    s = scene_mission()
    floor(s)
    ts = np.linspace(0, m.T_END, 200)
    path = np.array([plan.x_L(t)[0] for t in ts])
    s.curve(path, color=INDIGO, lw=1.2, alpha=1.0, zorder=20, linestyle=DASH)
    draw_gap(s)
    for t in (0.6, 5.0, 11.6):
        ref = m.plan_point(p, plan, t)
        draw_team(s, ref["x_L"], ref["R_L"], ref["x_Q"], ref["R"], p, zorder=30 if t < 5.5 else 70)
    s.label((0.0, 0, 2.95), "cruise: cables spread", dy=4, size=9.5, **SANS)
    s.label((3.0, 0, 3.25), "the gap: 1.45 m wide", dy=2, size=9.5, **SANS)
    s.label((6.0, 0, 2.05), "set down, level", dy=6, size=9.5, **SANS)
    s.label((1.55, 0, 1.6), "payload path", dy=-11, size=9, color=INDIGO, **SANS)
    s.dot((6.0, 0.0, 0.6), color=INK, size=4.5, zorder=90)
    s.label((6.0, 0.0, 0.6), "target", dx=22, dy=-8, size=9.5, **SANS)
    s.save(OUT / "ch19-mission.svg", pad=0.05)


# ------------------------------------------------------------------ 2. the tension distribution
def tensions():
    p = team().params
    W = np.r_[0.0, 0.0, p.m_L * p.g, 0.0, 0.0, 0.0]
    lam_mn, _ = m.distribute(p.r, W)
    lam_qp, _ = m.distribute(p.r, W, l=p.l, d_min=m.D_PLAN, t_min=m.T_MIN)
    lam_wide = lam_mn + (m.S_WIDE * m.spread_direction(p.r)).reshape(3, 3)
    s = Scene(width=7.0, height=3.3, azim=-58, elev=24, xlim=(-1.25, 4.55), ylim=(-0.62, 1.75))
    titles = ("minimum norm", "with the pair constraints", "cruise")
    for k, lam in enumerate((lam_mn, lam_qp, lam_wide)):
        x_L = np.array([1.75 * k, -0.95 * k, 0.0])
        Rz = m.expm_so3(np.radians(90.0) * E3)
        q = -lam / np.linalg.norm(lam, axis=1)[:, None]
        x_Q = x_L + p.r @ Rz.T - q @ Rz.T
        draw_team(s, x_L, Rz, x_Q, np.array([np.eye(3)] * 3), p, arm=0.2)
        for i in range(3):
            a = x_L + Rz @ p.r[i]
            s.arrow(a, a + 0.17 * (Rz @ lam[i]), color=INDIGO, lw=1.6, head=0.07, zorder=50)
            inner = np.array([lam[i, 0], lam[i, 1], 0.0])
            if np.linalg.norm(inner) > 1e-6:
                s.arrow(a, a + 0.45 * (Rz @ inner), color=TEAL, lw=1.5, head=0.06, zorder=51)
        for i in range(3):
            j = (i + 1) % 3
            d = np.linalg.norm(x_Q[i] - x_Q[j])
            col = TERRACOTTA if d < m.R_MIN else MUTED
            s.curve(np.array([x_Q[i], x_Q[j]]), color=col, lw=0.8, alpha=1.0, zorder=28, linestyle=DASH)
        d = np.linalg.norm(x_Q[0] - x_Q[1])
        top = x_L + np.array([0.0, 0.0, 1.0])
        s.label(top, f"{d:.2f} m apart", dy=24, size=9.5, color=TERRACOTTA if d < m.R_MIN else INK, **SANS)
        s.label(x_L, titles[k], dy=-27, size=9.5, **SANS)
        s.label(x_L, f"T = {np.linalg.norm(lam[0]):.2f} N each", dy=-40, size=9, color=MUTED, **SANS)
    s.label((0.0, 0.0, 0.0) + np.array([0.0, 0.42, 0.33]), "force of a cable\non the payload", dx=-52, dy=6,
            size=8.5, color=INDIGO, **SANS)
    a = np.array([3.5, -1.9, 0.0]) + m.expm_so3(np.radians(90.0) * E3) @ p.r[0]
    s.label(a + np.array([0.0, 0.5, 0.0]), "internal force", dx=34, dy=-14, size=8.5, color=TEAL, **SANS)
    s.save(OUT / "ch19-tensions.svg", pad=0.04)


# ------------------------------------------------------------------ 3. the flight (animation)
def flight_frame(u):
    o = data()["series"]["all"]
    p = team().params
    k = min(int(round(u * (len(o["t"]) - 1))), len(o["t"]) - 1)
    t = float(o["t"][k])
    s = scene_mission(width=7.0, height=3.9)
    s._lims = ((-1.6, 7.3), (-1.2, 3.75))
    floor(s)
    s.curve(o["x_d"], color=MUTED, lw=0.9, alpha=1.0, zorder=10, linestyle=DASH)
    draw_gap(s)
    s.curve(o["x_L"][:k + 1], color=INDIGO, lw=1.6, alpha=1.0, zorder=22)
    s.dot(o["x_d"][k], color=MUTED, size=5, zorder=24, hollow=True)
    for i in range(3):
        s.dot(o["x_Qd"][k][i], color=MUTED, size=3.6, zorder=25, hollow=True)
    front = o["x_L"][k][0] > m.GAP_X + 0.1
    draw_team(s, o["x_L"][k], o["R_L"][k], o["x_Q"][k], o["R"][k], p, tension=o["T"][k], zorder=70 if front else 30)
    active = bool(o["filtered"][k])
    if active:
        x = o["x_Q"][k]
        for i in range(3):
            for j in range(i + 1, 3):
                if np.linalg.norm(x[i] - x[j]) < m.R_MIN + 0.08:
                    s.curve(np.array([x[i], x[j]]), color=TERRACOTTA, lw=1.3, alpha=1.0, zorder=95)
    s.text2d(-1.45, 3.5, f"t = {t:4.1f} s", fontsize=10, ha="left", **SANS)
    s.text2d(-1.45, 3.2, "shown at real speed", fontsize=8.5, ha="left", color=MUTED, **SANS)
    # live tension bars
    x0, y0, scale = 5.35, 3.5, 0.42
    s.text2d(x0 - 0.1, y0, "tensions", fontsize=9, ha="right", **SANS)
    for i in range(3):
        y = y0 - 0.2 * i
        s.ax.plot([x0, x0 + scale * float(o["T"][k][i])], [y, y], color=INDIGO, lw=4.2, alpha=1.0,
                  solid_capstyle="butt", zorder=96)
        s.text2d(x0 + scale * 3.6 + 0.08, y, f"{float(o['T'][k][i]):.2f} N", fontsize=8.5, ha="left", **SANS)
    s.ax.plot([x0 + scale * m.T_MIN] * 2, [y0 + 0.1, y0 - 0.5], color=INK, lw=0.8, alpha=1.0, zorder=97)
    hk, dk = float(o["H"][k].min()), float(o["dist"][k].min())
    s.text2d(1.0, 3.5, f"closest pair {dk:4.2f} m", fontsize=10, ha="left", **SANS)
    s.text2d(1.0, 3.2, f"smallest barrier function H = {hk:4.2f}", fontsize=10, ha="left",
             color=TERRACOTTA if active else INK, **SANS)
    err = 100 * float(np.linalg.norm(o["x_L"][k] - o["x_d"][k]))
    if t < 1.0:
        msg, col = "hover in the cruise formation", INDIGO
    elif active:
        msg, col = "the separation barrier acts on the thrust vectors", TERRACOTTA
    elif int(o["active"][k]) > 0:
        msg, col = "the tension distribution holds the pairs apart: internal force only", TEAL
    elif t < 7.4:
        msg, col = f"cascade with flat feedforward: payload {err:.1f} cm from its reference", INDIGO
    elif t < 11.5:
        msg, col = "descending to set the payload down", INDIGO
    else:
        msg, col = f"at the target, tilted {float(o['tilt'][k]):.2f}°", INDIGO
    caption(s, msg, color=col, y=-1.0)
    return s


def flight():
    animate(flight_frame, OUT / "ch19-flight.gif", OUT / "ch19-flight-strip.png", n=132, fps=11,
            width_px=700, stills=(0.08, 0.33, 0.44, 1.0), strip_px=900, strip_cols=2, colors=48)


# ------------------------------------------------------------------ 4. what each protection does
def protect():
    S = data()["series"]
    fig, axes = new_plot(width=7.0, height=2.9, ncols=2)
    rows = (("unprotected", "no protection", MUTED, DASH), ("pairs_only", "pair constraints only", TEAL, "-"),
            ("barrier_only", "barrier only", TERRACOTTA, "-"), ("all", "both", INDIGO, "-"))
    for key, name, col, ls in rows:
        o = S[key]
        axes[0].plot(o["t"], o["dist"].min(axis=1), color=col, lw=1.5 if ls == "-" else 1.0, linestyle=ls, label=name)
        axes[1].plot(o["t"], o["tilt"], color=col, lw=1.5 if ls == "-" else 1.0, linestyle=ls)
    axes[0].axhline(m.R_MIN, color=INK, lw=1.0)
    axes[0].axhspan(0.70, m.R_MIN, color=PEACH, alpha=0.45, lw=0)
    axes[0].text(8.9, 0.79, "$r_{\\min}$", fontsize=10, va="top")
    axes[0].set_xlim(2.5, 9.5), axes[0].set_ylim(0.70, 1.15), axes[1].set_xlim(2.5, 9.5)
    axes[0].set_xlabel("time (s)"), axes[1].set_xlabel("time (s)")
    axes[0].set_ylabel("closest pair of vehicles (m)"), axes[1].set_ylabel("payload tilt (deg)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper center")
    for ax in axes:
        style_axes(ax)
    save(fig, OUT / "ch19-protect.svg")


# ------------------------------------------------------------------ 5. the cable (animation)
def cable_frame(u):
    C = data()["series"]["chain"]["track"]
    k = min(int(round(u * (len(C["t"]) - 1))), len(C["t"]) - 1)
    t = float(C["t"][k])
    s = Scene(width=6.6, height=3.6, azim=-72, elev=17, xlim=(-1.15, 1.45), ylim=(0.55, 2.0))
    x, xd = C["x"][k], C["x_d"][k]
    tail = C["x_d"][:, -1]
    s.curve(tail, color=MUTED, lw=0.8, alpha=1.0, zorder=8, linestyle=DASH)
    s.curve(xd, color=MUTED, lw=1.0, alpha=1.0, zorder=12, linestyle=DASH)
    for pt in xd[[0, -1]]:
        s.dot(pt, color=MUTED, size=4.2, zorder=13, hollow=True)
    s.curve(x, color=INDIGO, lw=2.0, alpha=1.0, zorder=30)
    for pt in x[1:-1]:
        s.dot(pt, color=INDIGO, size=3.2, zorder=31)
    for j, end in enumerate((0, -1)):
        quadrotor(s, x[end], C["R"][k][j], arm=0.16, lw=1.4, zorder=40)
    s.text2d(-1.08, 1.93, f"t = {t:3.1f} s", fontsize=10, ha="left", **SANS)
    s.text2d(-1.08, 1.86, "shown at real speed", fontsize=8.5, ha="left", color=MUTED, **SANS)
    s.text2d(1.38, 1.93, f"largest joint error {100 * float(C['err'][k]):4.1f} cm", fontsize=10, ha="right", **SANS)
    if t < 1.2:
        msg, col = "start: every link 12° off, one end 8 cm off", TERRACOTTA
    elif t < 3.0:
        msg, col = "the LQR about the flat reference pulls the cable onto it", INDIGO
    else:
        msg, col = "five links and two vehicles on a reference drawn for one joint", INDIGO
    caption(s, msg, color=col, y=0.62)
    return s


def cable():
    animate(cable_frame, OUT / "ch19-cable.gif", OUT / "ch19-cable-strip.png", n=96, fps=12,
            width_px=640, stills=(0.0, 0.12, 0.5, 1.0), strip_px=860, strip_cols=2, colors=40)


if __name__ == "__main__":
    which = sys.argv[1:] or ["mission", "tensions", "protect", "flight", "cable"]
    for name in which:
        globals()[name]()
