"""Figures for Chapter 17.  Run: python figures/src/ch17.py

Every curve and every pose is a simulation from verify/ch17.py, the script that checks
the chapter. The flights take a few seconds each and are kept in a temporary file.
"""
import importlib.util
import pickle
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "verify"))
_spec = importlib.util.spec_from_file_location("ch17_verify", ROOT / "verify" / "ch17.py")
m = importlib.util.module_from_spec(_spec)
sys.modules["ch17_verify"] = m
_spec.loader.exec_module(m)
from figkit import (DASH, INDIGO, INK, MUTED, OCHRE, PEACH, TEAL, TERRACOTTA, Scene, animate,  # noqa: E402
                    caption, halo, new_plot, quadrotor, save)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
MS = m.Mission()
PLAN = m.Plan(MS)
KEEP = ("t", "x", "xd", "R", "xhat", "Rhat", "H", "e_x", "att_err", "pos_err", "Psi", "Psi_pred", "Df", "f",
        "tilt", "dist", "n_rel")
FLIGHTS = {
    "full": (MS, {}),
    "plain": (replace(MS, x_o=np.array([100.0, 0.0, 0.0])), dict(adapt=False, safety=False)),
    "l1": (replace(MS, x_o=np.array([100.0, 0.0, 0.0])), dict(safety=False)),
    "neither": (MS, dict(adapt="untold", safety="blind")),
    "told": (MS, dict(safety="blind")),
    "informed": (MS, dict(adapt="untold")),
}
_cache = {}


def flight(name):
    """One flight of verify/ch17.py, computed once."""
    path = Path(tempfile.gettempdir()) / "ch17_flights.pkl"
    stamp = (ROOT / "verify" / "ch17.py").stat().st_mtime
    if not _cache and path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved.get("stamp") == stamp:
            _cache.update(saved)
    if name not in _cache:
        ms, blocks = FLIGHTS[name]
        o = m.fly(ms, **blocks)
        _cache[name] = {key: o[key] for key in KEEP}
        _cache["stamp"] = stamp
        path.write_bytes(pickle.dumps(_cache))
    return _cache[name]


def reference_curve(t0, t1, n=400):
    return np.array([PLAN(t)[0][0] for t in np.linspace(t0, t1, n)])


def ground_shadow(s, pts, **kw):
    """The curve dropped to the ground, to show height."""
    flat = np.array(pts, float).copy()
    flat[:, 2] = 0.0
    s.curve(flat, color=MUTED, lw=0.6, alpha=1.0, zorder=3, **kw)


def scene_course(width=7.0):
    s = Scene(width=width, height=width * 6.4 / 13.6, azim=-76, elev=17, xlim=(-2.9, 10.7), ylim=(-1.15, 5.25))
    for p in MS.landmarks:
        s.curve(np.array([[p[0], p[1], 0.0], p]), color=INK, lw=0.6, alpha=1.0, zorder=4)
        s.dot(p, color=INK, size=4.5, zorder=12)
    return s


def ball(s):
    s.sphere(MS.x_o, MS.r_o, color=PEACH, res=260)
    s.curve(np.array([[MS.x_o[0], MS.x_o[1], 0.0], MS.x_o - MS.r_o * E3]), color=MUTED, lw=0.6, alpha=1.0,
            zorder=3, linestyle=DASH)


# ------------------------------------------------------------ opening picture
def mission():
    o = flight("full")
    j, h = o["n_rel"], MS.h
    s = scene_course()
    carry = m.Carry(MS)
    hand = np.array([carry.position(t) for t in np.linspace(0, MS.t_carry, 120)])
    s.curve(hand, color=INK, lw=0.9, alpha=1.0, zorder=14, linestyle=(0, (1, 2.2)))
    ground_shadow(s, hand)
    ref = reference_curve(MS.t_start, MS.t_start + MS.t_course)
    ground_shadow(s, ref)
    inside = np.linalg.norm(ref - MS.x_o, axis=1) < MS.r_o
    first, last = np.nonzero(inside)[0][[0, -1]]
    s.curve(ref[:first + 1], color=INDIGO, lw=1.7, alpha=1.0, zorder=20)
    ball(s)
    s.curve(ref[first:last + 1], color=TERRACOTTA, lw=1.3, alpha=1.0, zorder=31, linestyle=DASH)
    s.curve(ref[last:], color=INDIGO, lw=1.7, alpha=1.0, zorder=32)
    for w in MS.waypoints[:-1]:
        s.dot(w, color=INDIGO, size=5.5, zorder=33, hollow=True)
    s.dot(MS.waypoints[-1], color=INK, size=5.5, zorder=33)
    # the vehicle: on the pad, at the release, and upright again
    quadrotor(s, carry.position(0.0), carry.attitude(0.0), arm=0.42, zorder=36)
    quadrotor(s, o["x"][j], o["R"][j], arm=0.42, zorder=36)
    k = j + int(round(1.2 / h))
    quadrotor(s, o["x"][k], o["R"][k], arm=0.42, color=MUTED, lw=1.2, zorder=30)
    s.arrow(o["x"][j], o["x"][j] + 0.75 * o["R"][j][:, 2], color=TERRACOTTA, lw=1.5, head=0.09, zorder=37)
    # the package, drawn at its place on the frame (to the scale of the arms)
    scale = 0.42 / 0.315
    for idx in (0, j):
        Rk = carry.attitude(0.0) if idx == 0 else o["R"][j]
        xk = carry.position(0.0) if idx == 0 else o["x"][j]
        s.dot(xk + scale * Rk @ MS.r_add, color=TERRACOTTA, size=4.2, zorder=38)
    # the crosswind
    for xw in (5.6, 6.6, 7.6):
        a = np.array([xw, 2.6, 3.9])
        s.arrow(a, a - 1.1 * E2, color=TEAL, lw=1.2, head=0.08, zorder=16)
    s.label((6.6, 2.6, 3.9), "crosswind, 3 N", dx=30, dy=11, size=9.5, color=TEAL, **SANS)
    s.label(carry.position(0.0), "pad: the filter starts\n140° and 2 m wrong", dx=-8, dy=-27, size=9.5,
            linespacing=1.15, **SANS)
    s.label(hand[70], "carried, motors off", dx=-46, dy=4, size=9.5, **SANS)
    s.label(o["x"][j], "thrown upside down", dx=-38, dy=30, size=9.5, color=TERRACOTTA, **SANS)
    s.label(MS.x_o, "ball, not\nin the plan", size=8.5, halo="none", linespacing=1.05, **SANS)
    s.label(ref[40], r"reference $x_d(t)$", dx=16, dy=17, size=10, color=INDIGO, **SANS)
    s.label(MS.waypoints[-1], "target", dx=20, dy=-9, size=9.5, **SANS)
    s.label(MS.landmarks[2], "landmark", dx=-2, dy=11, size=9.5, **SANS)
    s.label(o["x"][j] + scale * o["R"][j] @ MS.r_add, "package", dx=27, dy=-3, size=9, color=TERRACOTTA, **SANS)
    s.save(OUT / "ch17-mission.svg", pad=0.02)


# ------------------------------------------------------------ the plan
def plan():
    hover = m.believed(MS).m * 9.81
    fig, axes = new_plot(6.8, 2.5, ncols=3)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.86, wspace=0.3)
    for T, col in ((5.0, TERRACOTTA), (MS.t_course, INDIGO)):
        ms = replace(MS, t_course=T)
        pl = m.Plan(ms)
        ts = np.linspace(ms.t_start, ms.t_start + T, 500)
        fl = [pl(t)[1] for t in ts]
        axes[0].plot(ts - ms.t_start, [o["f"] / hover for o in fl], color=col, lw=1.5)
        axes[1].plot(ts - ms.t_start, np.degrees([np.arccos(o["R"][2, 2]) for o in fl]), color=col, lw=1.5)
    ts = np.linspace(MS.t_start, MS.t_start + MS.t_course, 600)
    gap = np.array([np.linalg.norm(PLAN(t)[0][0] - MS.x_o) for t in ts])
    axes[2].plot(ts - MS.t_start, gap, color=INDIGO, lw=1.5)
    axes[2].plot((ts - MS.t_start)[gap < MS.r_o], gap[gap < MS.r_o], color=TERRACOTTA, lw=2.2)
    axes[2].axhline(MS.r_o, color=INK, lw=0.8)
    axes[2].axhspan(0, MS.r_o, color=PEACH, alpha=0.45, lw=0)
    axes[2].set_ylim(0, 3.0)
    axes[2].text(7.4, 0.3, "inside the ball", fontsize=9, ha="right", va="center", color=TERRACOTTA, **SANS)
    axes[0].axhline(1.0, color=MUTED, lw=0.7, linestyle=DASH)
    axes[0].text(7.4, 1.5, "course of 5 s", color=TERRACOTTA, fontsize=9, ha="right", **SANS)
    axes[0].text(7.4, 0.82, "course of 7.5 s", color=INDIGO, fontsize=9, ha="right", **SANS)
    axes[1].set_yticks([0, 20, 40])
    axes[1].set_yticklabels(["0°", "20°", "40°"])
    for ax, title in zip(axes, ("thrust over hover thrust", "tilt", "distance to the center (m)")):
        ax.set_title(title, fontsize=9.5, loc="left", color=INK, **SANS)
        ax.set_xlabel("time on the course (s)", fontsize=9, **SANS)
        ax.set_xlim(0, 7.5)
    save(fig, OUT / "ch17-plan.svg")


# ------------------------------------------------------------ adaptation
def adapt():
    fig, (ax0, ax1) = new_plot(6.8, 2.8, ncols=2)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.88, wspace=0.22)
    for name, col in (("plain", TERRACOTTA), ("l1", INDIGO)):
        o = flight(name)
        j = o["n_rel"]
        ang = np.degrees(np.arccos(np.clip(1 - o["Psi"][j:], -1, 1)))
        ax0.semilogy(o["t"][j:], np.maximum(ang, 1e-3), color=col, lw=1.2)
    ax0.set_xlim(MS.t_carry, MS.t_end)
    ax0.set_ylim(0.02, 400)
    ax0.set_yticks([0.1, 1, 10, 100])
    ax0.set_yticklabels(["0.1°", "1°", "10°", "100°"])
    ax0.text(17.8, 9.0, "plain controller", color=TERRACOTTA, ha="right", fontsize=9.5, **SANS, path_effects=halo())
    ax0.text(17.8, 0.035, "with adaptation", color=INDIGO, ha="right", fontsize=9.5, **SANS, path_effects=halo())
    ax0.axvspan(MS.t_start + MS.gust_window[0], MS.t_start + MS.gust_window[1], color=TEAL, alpha=0.1, lw=0)
    ax0.text(12.2, 180, "gust", color=TEAL, fontsize=9.5, ha="center", **SANS)
    ax0.set_title(r"attitude error between $R$ and $R_d$", fontsize=9.5, loc="left", color=INK, **SANS)
    o = flight("l1")
    j = o["n_rel"]
    rho = m.true_params(MS)[1]
    truth = np.outer(o["f"][j:], np.cross(E3, rho))
    for i, col in ((0, TEAL), (1, INDIGO)):
        ax1.plot(o["t"][j:], truth[:, i], color=MUTED, lw=0.9, linestyle=DASH)
        ax1.plot(o["t"][j:], o["Df"][j:, i], color=col, lw=1.3)
    ax1.set_xlim(MS.t_carry, MS.t_end)
    ax1.set_ylim(-1.2, 3.4)
    ax1.text(17.8, 2.05, "second component", color=INDIGO, ha="right", fontsize=9.5, **SANS, path_effects=halo())
    ax1.text(17.8, 0.42, "first component", color=TEAL, ha="right", fontsize=9.5, **SANS, path_effects=halo())
    ax1.text(6.5, 3.05, "dashed: torque of the thrust about the center of mass", color=MUTED, fontsize=8.5, **SANS)
    ax1.set_title(r"filtered estimate $\Delta_f$ (N·m)", fontsize=9.5, loc="left", color=INK, **SANS)
    for ax in (ax0, ax1):
        ax.set_xlabel("time (s)", fontsize=9, **SANS)
    save(fig, OUT / "ch17-adapt.svg")


# ------------------------------------------------------------ the two wires
def compose():
    fig, (ax0, ax1) = new_plot(6.8, 3.0, ncols=2)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.88, wspace=0.24)
    rows = (("neither", TERRACOTTA, "neither wire"), ("informed", OCHRE, "barrier informed only"),
            ("told", TEAL, "predictor told only"), ("full", INDIGO, "both wires"))
    t0, t1 = 10.2, 13.2
    for name, col, _ in rows:
        o = flight(name)
        sel = (o["t"] >= t0) & (o["t"] <= t1)
        ax0.plot(o["t"][sel], o["H"][sel], color=col, lw=1.5)
        ax1.semilogy(o["t"][sel], np.maximum(o["Psi_pred"][sel], 1e-9), color=col, lw=1.5)
    ax0.axhline(0.0, color=INK, lw=0.8)
    ax0.axhspan(-0.7, 0.0, color=PEACH, alpha=0.45, lw=0)
    ax0.set_ylim(-0.62, 1.6)
    ax0.set_xlim(t0, t1)
    ax1.set_xlim(t0, t1)
    ax1.set_ylim(1e-9, 3)
    ax1.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    for y, (name, col, text) in zip((1.47, 1.3, 1.13, 0.96), rows):
        ax0.text(13.15, y, text, color=col, fontsize=9.5, ha="right", **SANS, path_effects=halo())
    ax0.text(10.3, -0.5, "outside the safe set", color=TERRACOTTA, fontsize=9, **SANS)
    ax0.set_title(r"barrier function $H$", fontsize=9.5, loc="left", color=INK, **SANS)
    ax1.set_title(r"prediction error $\tilde\Psi$", fontsize=9.5, loc="left", color=INK, **SANS)
    for ax in (ax0, ax1):
        ax.set_xlabel("time (s)", fontsize=9, **SANS)
    save(fig, OUT / "ch17-compose.svg")


# ------------------------------------------------------------ animation: the estimate converges
def estimate_frame(u):
    o = flight("full")
    h = MS.h
    t = 6.6 * u
    k = int(round(t / h))
    s = Scene(width=6.6, height=6.6 * 4.3 / 5.8, azim=-62, elev=18, xlim=(-3.3, 2.5), ylim=(-0.95, 3.35))
    lm = MS.landmarks[0]
    s.curve(np.array([[lm[0], lm[1], 0.0], lm]), color=INK, lw=0.6, alpha=1.0, zorder=4)
    s.dot(np.array([-1.0, -3.0, 0.0]), color=INK, size=5, zorder=12)
    s.label(np.array([-1.0, -3.0, 0.0]), "landmark", dx=0, dy=-11, size=9, **SANS)
    # the truth: dashed gray, like a reference
    s.curve(o["x"][:k + 1], color=MUTED, lw=0.9, alpha=1.0, zorder=14, linestyle=DASH)
    quadrotor(s, o["x"][k], o["R"][k], arm=0.36, color=MUTED, lw=1.3, dashed=True, zorder=20)
    s.dot(o["x"][k], color=MUTED, size=4.5, zorder=21, hollow=True)
    # the estimate: solid indigo
    k0 = max(0, k - int(round(1.5 / h)))
    s.curve(o["xhat"][k0:k + 1], color=INDIGO, lw=1.3, alpha=1.0, zorder=24)
    quadrotor(s, o["xhat"][k], o["Rhat"][k], arm=0.36, color=INDIGO, lw=1.7, zorder=30)
    s.dot(o["xhat"][k], color=INDIGO, size=4.5, zorder=31)
    s.text2d(-3.2, 3.12, "truth", fontsize=10, color=MUTED, ha="left", **SANS)
    s.text2d(-3.2, 2.86, "estimate", fontsize=10, color=INDIGO, ha="left", **SANS)
    s.text2d(2.4, 3.12, f"t = {t:3.1f} s", fontsize=10, ha="right", **SANS)
    s.text2d(2.4, 2.86, f"attitude error {np.degrees(o['att_err'][k]):5.1f}°", fontsize=10, ha="right", **SANS)
    s.text2d(2.4, 2.6, f"position error {o['pos_err'][k]:4.2f} m", fontsize=10, ha="right", **SANS)
    if t < 0.25:
        msg = "the estimate starts level at the origin, 140° and 2 m wrong"
    elif t < 1.6:
        msg = "IMU and landmark readings pull it onto the vehicle"
    elif t < MS.t_carry - MS.t_flip:
        msg = "carried by hand: the filter needs no model of the vehicle"
    elif t < MS.t_carry:
        msg = "the throw: the estimate flips with the vehicle"
    else:
        msg = "released upside down, with an estimate good to 0.05°"
    caption(s, msg, y=-0.72)
    return s


def estimate():
    animate(estimate_frame, OUT / "ch17-estimate.gif", OUT / "ch17-estimate-strip.png", n=100, fps=15,
            width_px=640, stills=(0.0, 0.06, 0.5, 0.955), strip_px=820, strip_cols=2, colors=48)


# ------------------------------------------------------------ animation: the flight
def flight_frame(u):
    o = flight("full")
    h, j = MS.h, o["n_rel"]
    t = MS.t_carry + (MS.t_end - MS.t_carry) * u
    k = min(int(round(t / h)), len(o["t"]) - 1)
    s = scene_course(width=7.0)
    ref = reference_curve(MS.t_start, MS.t_start + MS.t_course, 300)
    s.curve(ref, color=MUTED, lw=0.9, alpha=1.0, zorder=10, linestyle=DASH)
    ball(s)
    ring = MS.x_o + np.sqrt(MS.beta) * MS.r_o * np.array(
        [np.cos(a) * s.right + np.sin(a) * s.up for a in np.linspace(0, 2 * np.pi, 90)])
    s.curve(ring, color=MUTED, lw=0.7, alpha=1.0, zorder=31, linestyle=(0, (1, 2)))
    behind = np.linalg.norm(ref - MS.x_o, axis=1) < MS.r_o
    s.curve(ref[behind], color=TERRACOTTA, lw=1.0, alpha=1.0, zorder=32, linestyle=DASH)
    s.dot(MS.waypoints[-1], color=INK, size=5, zorder=33)
    s.label(MS.waypoints[-1], "target", dx=20, dy=-9, size=9.5, **SANS)
    s.curve(o["x"][j:k + 1], color=INDIGO, lw=1.7, alpha=1.0, zorder=34)
    ground_shadow(s, o["x"][j:k + 1])
    s.dot(o["xd"][k], color=MUTED, size=5, zorder=35, hollow=True)
    quadrotor(s, o["x"][k], o["R"][k], arm=0.42, zorder=40)
    s.arrow(o["x"][k], o["x"][k] + 0.012 * o["f"][k] * o["R"][k][:, 2], color=INDIGO, lw=1.3, head=0.08, zorder=41)
    s.text2d(-2.8, 4.95, f"t = {t:4.1f} s", fontsize=10, ha="left", **SANS)
    s.text2d(-2.8, 4.6, "shown at 1.2 times real speed", fontsize=8.5, ha="left", color=MUTED, **SANS)
    hk = o["H"][k]
    s.text2d(10.6, 4.95, f"barrier function H = {hk:5.2f}" if hk < 9.99 else "barrier function H > 10",
             fontsize=10, ha="right", color=INK, **SANS)
    s.text2d(10.6, 4.6, f"distance from the reference {o['e_x'][k]:4.2f} m", fontsize=10, ha="right", **SANS)
    tr = t - MS.t_carry
    if tr < 0.22:
        msg, col = "thrown upside down: thrust off, turning", TERRACOTTA
    elif tr < 0.9:
        msg, col = "upright after half a second", INDIGO
    elif t < MS.t_start:
        msg, col = "joining the reference, which waits at the first waypoint", INDIGO
    elif t < 10.17:
        msg, col = "on the course: the cascade with flat feedforward", INDIGO
    elif t < 12.15:
        msg, col = "the safety filter bends the flight around the ball", TERRACOTTA
    elif t < 14.3:
        msg, col = "back to the reference, in a crosswind of 3 N", INDIGO
    elif t < 16.0:
        msg, col = "last leg, heading still turning", INDIGO
    else:
        msg, col = "hover at the target, 1 cm off", INDIGO
    caption(s, msg, color=col, y=-0.93)
    return s


def flight_anim():
    animate(flight_frame, OUT / "ch17-flight.gif", OUT / "ch17-flight-strip.png", n=120, fps=12,
            width_px=700, stills=(0.012, 0.19, 0.43, 1.0), strip_px=900, strip_cols=2, colors=48)


if __name__ == "__main__":
    which = sys.argv[1:] or ["mission", "plan", "adapt", "compose", "estimate", "flight_anim"]
    for name in which:
        globals()[name]()
