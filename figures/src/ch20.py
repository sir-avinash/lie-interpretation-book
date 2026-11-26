"""Figures for Chapter 20.  Run: python figures/src/ch20.py

Every pose and every curve is a simulation. The walks, the impact map and the motion
primitive come from the robot model library (case_studies/robots/biped_walking.py); the
turn plan, the skater and the pendulum experiment from verify/ch20.py, the script that
checks the chapter. The runs take some seconds each and are kept in verify/ch20_cache.pkl.
"""
import importlib.util
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "case_studies"))
_spec = importlib.util.spec_from_file_location("ch20_verify", ROOT / "verify" / "ch20.py")
m = importlib.util.module_from_spec(_spec)
sys.modules["ch20_verify"] = m
_spec.loader.exec_module(m)
from figkit import (DASH, INDIGO, INK, MUTED, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA, Scene,  # noqa: E402
                    animate, axis_through_box, box, caption, halo, new_plot, save)

import robots  # noqa: E402
from robots import biped_walking as bw  # noqa: E402

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
TRIAD = (TERRACOTTA, TEAL, INDIGO)
FACES = {(0, 1): PERIWINKLE, (0, -1): PERIWINKLE, (1, 1): SAND, (1, -1): SAND, (2, 1): PEACH, (2, -1): PEACH}
LEG_FACES = {key: SAND for key in FACES}
GHOST = "#9a9da6"
GROUND = "#f4f1e8"
GAIT = bw.Gait()
biped = robots.get("reaction_mass_biped")
_cache = {}
_lims = {}
KEEP = ("t", "step", "hip", "foot", "swing", "R1", "RP", "R2", "l", "slide", "a1", "aP", "a2", "V", "F", "tau_1",
        "e_l", "H_foot", "ts")


def _runs():
    return {
        "straight": lambda: bw.walk([0.0] * 10),
        "turn": lambda: bw.walk(*[m.turn_plan()[0]], rolls=m.turn_plan()[1]),
        "straight90": lambda: bw.walk([np.pi / 2] * 8),
        "book": lambda: bw.walk([0.0] * 8, law=bw.book_law, record=0),
        "book90": lambda: bw.walk([np.pi / 2] * 8, law=bw.book_law, record=0),
    }


def run(name):
    """One run, computed once and kept until verify/ch20.py or the library's walking code changes."""
    path = ROOT / "verify" / "ch20_cache.pkl"
    stamp = tuple(f.stat().st_mtime for f in (ROOT / "verify" / "ch20.py", Path(bw.__file__),
                                               ROOT / "case_studies" / "robots" / "reaction_mass_biped.py",
                                               ROOT / "case_studies" / "robots" / "core.py"))
    if not _cache and path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved.get("stamp") == stamp:
            _cache.update(saved)
    if name not in _cache:
        if name == "skater":
            _cache[name] = m.skater()
        elif name == "pendulum":
            ns = m.chapter6_pendulum()
            rng = np.random.default_rng(20)
            _cache[name] = {ang: m.pendulum_deadline(ns, ang, 30, 0.35, rng) for ang in (10, 30, 60, 90, 120, 150)}
        else:
            o, steps = _runs()[name]()
            keep = {k: o[k] for k in KEEP if k in o and len(o[k])}
            keep["steps"] = [{k: v for k, v in s.items()} for s in steps]
            _cache[name] = keep
        _cache["stamp"] = stamp
        path.write_bytes(pickle.dumps(_cache))
    return _cache[name]


# ------------------------------------------------------------------ drawing
def draw_biped(s, foot, l, R1, RP, R2, slide, l_0=GAIT.l_0, triads=False, ghost=False, z=30, masses=True):
    """The Reaction Mass Biped: two legs and a torso drawn with figkit.box, and the three
    pairs of reaction masses on their rails. `foot` is the stance foot."""
    hip = foot + l * R1 @ E3
    toe = hip - l_0 * R2 @ E3
    if ghost:
        for a, b in ((foot, hip), (hip, toe)):
            s.curve(np.array([a, b]), color=GHOST, lw=0.9, alpha=1.0, zorder=z - 8, linestyle=DASH)
        s.dot(toe, color=GHOST, size=3.0, zorder=z - 8, hollow=True)
        return hip, toe
    legs = [(0.5 * (foot + hip), R1, l), (0.5 * (hip + toe), R2, l_0)]
    legs.sort(key=lambda g: g[0] @ s.view)
    for k, (c, R, length) in enumerate(legs):
        box(s, c, R, (0.07, 0.07, length), LEG_FACES, zorder=z + 2 * k, lw=0.7)
    s.dot(toe, color=INK, size=3.0, zorder=z + 6)
    s.dot(foot, color=INK, size=3.0, zorder=z + 6)
    dims = (0.26, 0.20, 0.32)
    if masses:
        for i in range(3):
            a = slide[i] * RP[:, i]
            for sign in (1, -1):
                behind = (sign * a) @ s.view < 0
                s.curve(np.array([hip, hip + sign * (a + 0.02 * RP[:, i])]), color=TEAL, lw=1.0, alpha=1.0,
                        zorder=z + 7 if behind else z + 22)
                s.dot(hip + sign * a, color=TEAL, size=3.8, zorder=z + 8 if behind else z + 23)
    box(s, hip, RP, dims, FACES, zorder=z + 12)
    if triads:
        s.frame(0.5 * (foot + hip), R1, length=0.22, colors=TRIAD, lw=1.1, head=0.06, zorder=z + 30)
        s.frame(0.5 * (hip + toe), R2, length=0.22, colors=TRIAD, lw=1.1, head=0.06, zorder=z + 30)
        for i in range(3):
            if RP[:, i] @ s.view > 0:
                axis_through_box(s, hip, RP, dims, RP[:, i], 0.5, TRIAD[i], lw=1.2, head=0.07, zorder=z + 20)
    return hip, toe


def ground(s, x0, x1, y0, y1, z=2):
    s.patch([np.array([x0, y0, 0.0]), np.array([x1, y0, 0.0]), np.array([x1, y1, 0.0]), np.array([x0, y1, 0.0])],
            color=GROUND, alpha=1.0, edge=MUTED, lw=0.6, zorder=z)


def footprints(s, steps, upto=None, z=6):
    pts = [st["foot"] for st in steps[:upto]]
    for pnt in pts:
        th = np.linspace(0, 2 * np.pi, 24)
        ring = pnt + 0.05 * np.c_[np.cos(th), np.sin(th), 0 * th]
        s.curve(ring, color=MUTED, lw=0.9, alpha=1.0, zorder=z)


def pose(o, k):
    return dict(foot=o["foot"][k], l=o["l"][k], R1=o["R1"][k], RP=o["RP"][k], R2=o["R2"][k], slide=o["slide"][k])


# ------------------------------------------------------------------ 1. the opening picture
def opening():
    o = run("turn")
    steps = o["steps"]
    k = int(np.argmin(np.abs(o["t"] - (sum(st["step_time"] for st in steps[:7]) + 0.55))))
    s = Scene(width=6.2, height=4.6, azim=-22, elev=16)
    pp = pose(o, k)
    c = pp["foot"]
    ground(s, c[0] - 1.25, c[0] + 1.0, c[1] - 1.45, c[1] + 1.0)
    near = [st for st in steps if abs(st["foot"][0] - c[0] + 0.125) < 1.05 and abs(st["foot"][1] - c[1] + 0.225) < 1.15]
    footprints(s, near)
    hip, toe = draw_biped(s, **pp, triads=True)
    mid1 = 0.5 * (pp["foot"] + hip)
    mid2 = 0.5 * (hip + toe)
    s.label(mid1, r"$R_1$", dx=-16, dy=-14, color=INK)
    s.label(mid2, r"$R_2$", dx=16, dy=-14, color=INK)
    s.label(hip, r"$R_P$", dx=-34, dy=22, color=INK)
    s.label(pp["foot"], "stance foot", dx=4, dy=-13, size=9, **SANS)
    s.label(toe, "swing foot", dx=-34, dy=2, size=9, **SANS)
    s.label(0.5 * (pp["foot"] + mid1), r"$l$", dx=11, dy=0, color=INK)
    far = hip + pp["slide"][1] * pp["RP"][:, 1]
    s.label(far, "reaction masses", dx=50, dy=10, size=9, color=TEAL, **SANS)
    s.label(near[0]["foot"], "footprints", dx=0, dy=-13, size=9, color=MUTED, **SANS)
    s.save(OUT / "ch20-biped.svg", pad=0.2)


# ------------------------------------------------------------------ 2. the pendulum with a deadline
def deadline():
    data = run("pendulum")
    angles = sorted(data)
    fig, axes = new_plot(width=6.6, height=2.9, ncols=2)
    for ax, (ig, ie), ylabel in ((axes[0], (0, 3), "angle left at the deadline (degrees)"),
                                 (axes[1], (1, 4), "peak torque (N·m)")):
        for col, color, name, dx in ((ig, INDIGO, "geometric law", -1.6), (ie, TERRACOTTA, "Euler-angle law", 1.6)):
            med = [np.median(data[a][:, col]) for a in angles]
            lo = [np.min(data[a][:, col]) for a in angles]
            hi = [np.max(data[a][:, col]) for a in angles]
            x = np.array(angles) + dx
            ax.vlines(x, lo, hi, color=color, lw=1.2)
            ax.plot(x, med, "o-", color=color, ms=4, lw=1.3, label=name)
        ax.set_xticks(angles)
        ax.set_xlabel("size of the initial error (degrees)", fontsize=9, **SANS)
        ax.set_ylabel(ylabel, fontsize=9, **SANS)
    axes[1].set_yscale("log")
    axes[0].axvspan(5, 35, color=SAND, alpha=0.6, lw=0, zorder=0)
    axes[1].axvspan(5, 35, color=SAND, alpha=0.6, lw=0, zorder=0)
    axes[0].text(20, axes[0].get_ylim()[1] * 0.9, "walking", fontsize=9, ha="center", va="top", color=INK, **SANS)
    axes[0].legend(frameon=False, prop={"family": "Source Sans 3", "size": 9}, loc="upper left", bbox_to_anchor=(0.28, 0.82))
    fig.tight_layout()
    save(fig, OUT / "ch20-deadline.svg")


# ------------------------------------------------------------------ 3. the impact
def impact_figure():
    o = run("straight")
    last = o["steps"][-1]
    pre, post = last["pre"], last["post"]
    s = Scene(width=6.6, height=3.5, azim=-72, elev=12)
    gap = np.array([2.3, 0.0, 0.0])
    sc_v, sc_i = 1.2, 0.42
    side_off = 0.42 * s.right + 0.1 * E3
    for side, origin in (("before", np.zeros(3)), ("after", gap)):
        foot = origin
        ground(s, origin[0] - 0.5, origin[0] + 1.15, -0.5, 0.5)
        hip, toe = draw_biped(s, foot, pre.l, pre.R1, pre.RP, pre.R2, pre.slide)
        if side == "before":
            v = bw.hip_velocity(pre.l, pre.l_dot, pre.R1, pre.Omega1)
            s.arrow(hip - side_off, hip - side_off + sc_v * v, color=INDIGO, zorder=80)
            s.label(hip - side_off, f"hip velocity\n{np.linalg.norm(v):.2f} m/s", dx=-30, dy=16, size=9, color=INDIGO, **SANS)
            s.label(foot, "stance foot", dx=-26, dy=-13, size=9, **SANS)
            s.label(toe, "swing foot arrives", dx=34, dy=-13, size=9, **SANS)
            s.label(hip + 0.75 * E3, "just before", dx=0, dy=0, size=10, **SANS)
        else:
            v = bw.hip_velocity(post.l, post.l_dot, post.R1, post.Omega1)
            s.arrow(hip - side_off, hip - side_off + sc_v * v, color=INDIGO, zorder=80, head=0.05)
            s.label(hip - side_off, f"hip velocity\n{np.linalg.norm(v):.2f} m/s", dx=-30, dy=16, size=9, color=INDIGO, **SANS)
            s.arrow(toe, toe + sc_i * last["impulse"], color=TERRACOTTA, zorder=80)
            s.label(toe + sc_i * last["impulse"], f"impulse\n{np.linalg.norm(last['impulse']):.2f} N·s", dx=26, dy=2, size=9, color=TERRACOTTA, **SANS)
            s.label(toe, "new stance foot", dx=34, dy=-13, size=9, **SANS)
            s.label(foot, "new swing foot", dx=-30, dy=-13, size=9, **SANS)
            s.label(hip + 0.75 * E3, "just after", dx=0, dy=0, size=10, **SANS)
    s.save(OUT / "ch20-impact.svg", pad=0.25)


# ------------------------------------------------------------------ 4. the walk with a turn (animation)
N_TURN = 150


def turn_frame(t):
    o = run("turn")
    steps = o["steps"]
    T = o["t"][-1]
    k = int(np.argmin(np.abs(o["t"] - t * T)))
    if "lims" not in _lims:
        probe = Scene(azim=-128, elev=24)
        pts = np.vstack([o["hip"] * np.array([1, 1, 0]) + np.array([dx, dy, 0]) for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)]
                        + [o["hip"] + np.array([0, 0, 0.45])])
        xy = probe.project(pts)
        plt.close(probe.fig)
        _lims["lims"] = ((xy[:, 0].min() - 0.15, xy[:, 0].max() + 0.15), (xy[:, 1].min() - 0.55, xy[:, 1].max() + 0.75))
    xlim, ylim = _lims["lims"]
    s = Scene(width=6.6, height=6.6 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), azim=-128, elev=24, xlim=xlim, ylim=ylim)
    ground(s, -0.6, 4.4, -0.7, 4.7)
    n_done = int(o["step"][k])
    footprints(s, steps, upto=n_done + 1)
    s.curve((o["hip"][:k + 1:3] * np.array([1, 1, 0])) if k > 3 else o["hip"][:2] * np.array([1, 1, 0]),
            color=INDIGO, lw=0.9, alpha=1.0, zorder=7)
    pp = pose(o, k)
    hd = m.turn_plan()[0]
    d = bw.primitive(o["ts"][k], hd[n_done], GAIT)
    draw_biped(s, pp["foot"], d["l"], d["R1"], d["RP"], d["R2"], d["slide"], ghost=True)
    draw_biped(s, **pp)
    # readout: the Lyapunov function of the tracking law
    tr = s.ax.transAxes
    Vmax = 6.0
    val = min(o["V"][k], Vmax)
    s.ax.add_patch(plt.Rectangle((0.03, 0.795), 0.3 * val / Vmax, 0.022, fc=TERRACOTTA, ec="none", zorder=90, transform=tr))
    s.ax.plot([0.03, 0.03], [0.785, 0.827], color=INK, lw=0.8, zorder=91, transform=tr)
    s.ax.text(0.03, 0.855, "tracking error, measured by the Lyapunov function", fontsize=8.6, ha="left", va="center",
              color=INK, zorder=92, transform=tr, **SANS)
    s.ax.text(0.03, 0.755, f"{o['V'][k]:.2f} J", fontsize=8.6, ha="left", va="center", color=TERRACOTTA, zorder=92,
              transform=tr, **SANS)
    s.ax.text(0.03, 0.955, f"step {n_done + 1} of {len(steps)}", fontsize=9.5, ha="left", va="center", color=INK,
              transform=tr, **SANS)
    s.ax.text(0.03, 0.91, f"heading of the reference {np.degrees(hd[n_done]):.0f}°", fontsize=9.5, ha="left", va="center",
              color=INK, transform=tr, **SANS)
    phase = ("walking straight" if n_done < 4 else "turning: the heading jumps 15° at each touchdown" if n_done < 10
             else "walking straight again")
    caption(s, f"{phase}, t = {o['t'][k]:.1f} s (shown 1.1 times faster)", size=9.5)
    return s


def turn():
    animate(turn_frame, OUT / "ch20-turn.gif", OUT / "ch20-turn-strip.png", n=N_TURN, fps=12, width_px=720,
            stills=(0.12, 0.42, 0.62, 1.0), strip_px=620, strip_cols=2, hold=1.6, colors=64)


# ------------------------------------------------------------------ 5. the skater (animation)
N_SKATE = 76


def skater_frame(t):
    o = run("skater")
    k = int(round(t * (len(o["t"]) - 1)))
    xlim, ylim = (-1.5, 2.5), (-0.4, 1.75)
    s = Scene(width=6.4, height=6.4 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), azim=-60, elev=22, xlim=xlim, ylim=ylim)
    origin = -0.55 * s.right
    ground(s, origin[0] - 0.55, origin[0] + 0.55, origin[1] - 0.55, origin[1] + 0.55)
    RP, slide = o["RP"][k], o["slide"][k]
    hip, _ = draw_biped(s, origin, 0.9, np.eye(3), RP, np.eye(3), slide)
    for i in (0, 1):
        for sign in (1, -1):
            pts = np.array([hip + sign * o["slide"][j][i] * o["RP"][j][:, i] for j in range(0, k + 1, 10)] + [hip + sign * slide[i] * RP[:, i]])
            if len(pts) > 1:
                s.curve(pts, color=TEAL, lw=0.7, alpha=1.0, zorder=20)
    tip = axis_through_box(s, hip, RP, (0.26, 0.20, 0.32), E3, 0.14 * o["spin"][k], INDIGO, lw=1.6, zorder=75)
    s.label(tip, r"$\Omega_P$", dx=12, dy=2, color=INDIGO, size=11)
    x0, wmax = 0.85, 1.25
    for y, value, full, color, name, text in (
            (1.38, o["spin"][k], o["spin"][-1], INDIGO, "spin rate of the torso", f"{o['spin'][k]:.2f} rad/s"),
            (0.82, o["mom"][k], o["mom"][0], TEAL, "angular momentum of the torso", f"{o['mom'][k]:.3f} N·m·s"),
            (0.26, slide[0], o["slide"][0][0], MUTED, "distance of the masses from the hip", f"{100 * slide[0]:.0f} cm")):
        s.ax.add_patch(plt.Rectangle((x0, y - 0.06), wmax * value / full, 0.12, fc=color, ec="none", zorder=90))
        s.ax.plot([x0, x0], [y - 0.11, y + 0.11], color=INK, lw=0.8, zorder=91)
        s.ax.plot([x0 + wmax, x0 + wmax], [y - 0.11, y + 0.11], color=INK, lw=0.8, zorder=91)
        s.ax.text(x0, y + 0.19, name, fontsize=8.6, ha="left", va="center", color=INK, **SANS)
        s.ax.text(x0, y - 0.19, text, fontsize=8.6, ha="left", va="center", color=color, **SANS)
    tt = o["t"][k]
    what = "the masses are out" if tt < 0.3 else "forces along the rails pull the masses in" if tt < 0.9 else "the masses are in"
    caption(s, f"{what}: no torque acts on the torso, t = {tt:.2f} s (4 times slower)", size=9.5, y=ylim[0] + 0.09)
    return s


def skater():
    animate(skater_frame, OUT / "ch20-skater.gif", OUT / "ch20-skater-strip.png", n=N_SKATE, fps=12, width_px=720,
            stills=(0.0, 1.0), strip_px=900, strip_cols=1, hold=1.6, colors=64)


# ------------------------------------------------------------------ 6. the walks as data
def errors():
    fig, axes = new_plot(width=6.6, height=5.2, ncols=2, nrows=3, sharex="col")
    for col, name, title in ((0, "straight", "walking straight"), (1, "turn", "four steps, a turn of 90°, four steps")):
        o = run(name)
        t = o["t"]
        ax = axes[0, col]
        ax.plot(t, np.degrees(o["a1"]), color=INDIGO, lw=1.0, label="stance leg")
        ax.plot(t, np.degrees(o["a2"]), color=TEAL, lw=1.0, label="swing leg")
        ax.plot(t, np.degrees(o["aP"]), color=TERRACOTTA, lw=1.0, label="torso")
        ax.set_title(title, fontsize=9.5, **SANS)
        ax.set_ylim(0, 36)
        ax = axes[1, col]
        F = o["F"]
        ax.plot(t, np.linalg.norm(F[:, :2], axis=1) / F[:, 2], color=INDIGO, lw=1.0)
        ax.axhline(0.6, color=INK, lw=0.8)
        ax.axhspan(0.6, 0.7, color=PEACH, alpha=0.5, lw=0)
        ax.set_ylim(0, 0.7)
        ax = axes[2, col]
        for i, (color, lab) in enumerate(((TERRACOTTA, "about $e_1$"), (INDIGO, "about $e_2$"), (TEAL, "about $e_3$"))):
            ax.plot(t, o["tau_1"][:, i], color=color, lw=0.9, label=lab)
        ax.set_xlabel("time (s)", fontsize=9, **SANS)
        ax.set_ylim(-22, 62)
    axes[0, 0].set_ylabel("attitude error (degrees)", fontsize=9, **SANS)
    axes[1, 0].set_ylabel("friction ratio", fontsize=9, **SANS)
    axes[2, 0].set_ylabel("ankle torque (N·m)", fontsize=9, **SANS)
    axes[0, 1].legend(frameon=False, prop={"family": "Source Sans 3", "size": 8.5}, ncol=3, loc="upper center",
                      columnspacing=1.0, handlelength=1.2)
    axes[2, 1].legend(frameon=False, prop={"family": "Source Sans 3", "size": 8.5}, ncol=3, loc="upper center",
                      columnspacing=1.0, handlelength=1.2)
    axes[1, 1].text(0.3, 0.63, "friction ratio 0.6", fontsize=8.5, color=INK, va="center", **SANS)
    fig.tight_layout()
    save(fig, OUT / "ch20-errors.svg")


def gait_figure():
    fig, ax = new_plot(width=5.2, height=2.8)
    n = np.arange(1, 9)
    for name, color, marker, lab in (("straight", INDIGO, "o", "thesis law, heading 0°"),
                                     ("straight90", TERRACOTTA, "s", "thesis law, heading 90°"),
                                     ("book", TEAL, "^", "scalar gains, heading 0°"),
                                     ("book90", TEAL, "v", "scalar gains, heading 90°")):
        L = [st["step_length"] for st in run(name)["steps"]][:8]
        ax.plot(n, np.array(L) * 100, marker + "-", color=color, ms=4.5, lw=1.1, label=lab,
                mfc="none" if name == "book90" else color)
    ax.axhline(200 * GAIT.l_0 * np.sin(GAIT.theta_s), color=GHOST, lw=0.9, linestyle=DASH)
    ax.text(8.1, 200 * GAIT.l_0 * np.sin(GAIT.theta_s) + 0.12, "reference", fontsize=8.5, color=MUTED, ha="right", va="bottom", **SANS)
    ax.set_xlabel("step", fontsize=9, **SANS)
    ax.set_ylabel("step length (cm)", fontsize=9, **SANS)
    ax.legend(frameon=False, prop={"family": "Source Sans 3", "size": 8.5}, loc="center right")
    fig.tight_layout()
    save(fig, OUT / "ch20-gait.svg")


# ------------------------------------------------------------------ 7. Cassie, a schematic
def cassie_figure():
    cas = robots.get("cassie")
    p = cas.params
    s = Scene(width=5.6, height=4.0, azim=-62, elev=14)
    x = np.array([0.0, 0.0, 0.95])
    R = np.eye(3)
    R_r, R_l = bw.Rx(np.pi - 0.07) @ bw.Ry(0.1), bw.Rx(np.pi + 0.07) @ bw.Ry(-0.1)      # leg axes point down from the hips
    ground(s, -0.55, 0.55, -0.55, 0.55)
    wide = 1.6                                   # the hips are drawn wider apart than they are
    for side, R_leg, l, name in ((-1.0, R_r, 0.97, "r"), (1.0, R_l, 0.96, "l")):
        hipj = x + side * wide * p.d * R @ E2
        knee, footp = hipj + 0.5 * l * R_leg @ E3, hipj + l * R_leg @ E3
        box(s, knee, R_leg, (0.06, 0.06, l), LEG_FACES, zorder=30 if side < 0 else 24, lw=0.7)
        s.dot(hipj, color=INK, size=4.0, zorder=60)
        s.dot(knee, color=INK, size=3.4, zorder=61)
        s.frame(knee, R_leg, length=0.22, colors=TRIAD, lw=1.0, head=0.06, zorder=62)
        s.label(knee, rf"$R_{name}$", dx=22 * side, dy=-16, color=INK)
        s.label(0.5 * (knee + footp), rf"$l_{name}$", dx=15 * side, dy=-4, color=INK)
        box(s, footp - 0.03 * E3, np.eye(3), (0.26, 0.12, 0.05), {k: PERIWINKLE for k in FACES}, zorder=20 if side > 0 else 34, lw=0.7)
    dims = (0.2, 2 * wide * p.d, 0.24)
    box(s, x, R, dims, FACES, zorder=44)
    for i in range(3):
        if R[:, i] @ s.view > 0:
            axis_through_box(s, x, R, dims, R[:, i], 0.42, TRIAD[i], lw=1.2, head=0.07, zorder=50)
    s.label(x + 0.42 * E3, r"$(x,\ R)$", dx=0, dy=11, color=INK)
    s.label(x + 0.62 * E3, "pelvis", dx=0, dy=10, size=9, **SANS)
    s.label(np.array([0.0, -wide * p.d, 0.0]), "Hovershoes", dx=-40, dy=-12, size=9, color=MUTED, **SANS)
    s.label(np.array([0.55, -0.55, 0.0]), "a schematic: not simulated in this book", dx=0, dy=-16, size=9, color=MUTED, **SANS)
    s.save(OUT / "ch20-cassie.svg", pad=0.3)


if __name__ == "__main__":
    which = sys.argv[1:] or ["opening", "deadline", "impact", "turn", "skater", "errors", "gait", "cassie"]
    table = dict(opening=opening, deadline=deadline, impact=impact_figure, turn=turn, skater=skater, errors=errors,
                 gait=gait_figure, cassie=cassie_figure)
    for name in which:
        table[name]()
