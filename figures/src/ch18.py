"""Figures for Chapter 18.  Run: python figures/src/ch18.py [name ...]

Every curve is a flight of verify/ch18.py on the robot model library, or the
flat map along one of its plans. The flights are read from the small cache
that `python verify/ch18.py` leaves (verify/ch18_cache.npz), and flown again
when it is missing or stale.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (DASH, INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, box, caption, halo, new_plot, quadrotor, save, small_circle)

_spec = importlib.util.spec_from_file_location("verify_ch18", ROOT / "verify" / "ch18.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["verify_ch18"] = V
_spec.loader.exec_module(V)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
PALE = "#f6ddd0"                      # the forbidden side of a constraint
FACES = {(0, 1): SAND, (0, -1): SAND, (1, 1): SAND, (1, -1): SAND, (2, 1): PEACH, (2, -1): PEACH}
_cache = {}


def runs():
    if "runs" not in _cache:
        _cache["runs"] = V.load_runs()
    return _cache["runs"]


def plans():
    if "plans" not in _cache:
        _cache["plans"] = V.plans()
    return _cache["plans"]


def at(o, t):
    """Index of the logged sample nearest to time t."""
    return int(np.argmin(np.abs(o["t"] - t)))


def rot_of(o, k):
    return o["R"][k].reshape(3, 3)


# ------------------------------------------------------------ the room
VIEW = dict(azim=-74, elev=17)
LIMS = dict(xlim=(-0.75, 6.2), ylim=(-0.62, 2.72))
T_SET = V.T_START + V.T_TRAVEL


def room(s, labels=True):
    """Floor line of the course, the wall and the platform."""
    plan = plans()[0]
    w = V.WALL
    y0, y1 = w["y"]
    s.patch([[w["x"], y0, 0], [w["x"], y1, 0], [w["x"], y1, w["height"]], [w["x"], y0, w["height"]]],
            color=SAND, alpha=1.0, zorder=12)
    box(s, np.array([5.0, 1.8, 0.2]), np.eye(3), (0.7, 0.7, 0.4), FACES, zorder=13)
    tt = np.linspace(V.T_START, T_SET, 200)
    xs = np.array([plan.flat(t)["x_L"] for t in tt])
    s.curve(np.column_stack([xs[:, 0], xs[:, 1], np.zeros(len(tt))]), color=MUTED, lw=0.6, alpha=1.0, zorder=9)
    if labels:
        s.label([w["x"], y0, w["height"]], "wall", dx=-14, dy=8, size=9.5, **SANS)
        s.label([5.0, 1.8, 0.0], "platform", dx=-4, dy=-22, size=9.5, **SANS)
    return xs


def cone_below(s, x_Q, deg=V.CONE_DEG, l=1.0, lw=0.8, axis=True):
    """The cone the cable must stay in, hung under the quadrotor."""
    th = np.radians(deg)
    ph = np.linspace(0, 2 * np.pi, 80)
    ring = x_Q + l * (-np.cos(th) * E3 + np.sin(th) * (np.outer(np.cos(ph), E1) + np.outer(np.sin(ph), E2)))
    s.curve(ring, color=INK, lw=lw, alpha=1.0, zorder=22)
    side = s.right - (s.right @ E3) * E3
    side = side / np.linalg.norm(side)
    for sg in (-1, 1):
        s.curve(np.array([x_Q, x_Q + l * (-np.cos(th) * E3 + sg * np.sin(th) * side)]), color=INK, lw=lw,
                alpha=1.0, zorder=22)
    if axis:
        s.curve(np.array([x_Q, x_Q - l * np.cos(th) * E3]), color=INK, lw=0.6, alpha=1.0, linestyle=DASH, zorder=22)


def vehicle(s, x_Q, R, x_L, color=INK, load=INDIGO, arm=0.19, hollow=False, z=30):
    s.curve(np.array([x_Q, x_L]), color=color, lw=1.0, alpha=1.0, zorder=z)
    quadrotor(s, x_Q, R, arm=arm, lw=1.4, color=color, zorder=z + 4)
    s.dot(x_L, color=load, size=6.0, zorder=z + 6, hollow=hollow)


# ----------------------------------------------------------- 1. the mission
def mission():
    plan = plans()[0]
    s = Scene(width=6.6, height=3.5, **VIEW)
    xs = room(s)
    tt = np.linspace(V.T_START, T_SET, 200)
    xq = np.array([plan.flat(t)["x_Q"] for t in tt])
    s.curve(xq, color=MUTED, lw=0.8, alpha=1.0, zorder=18)
    s.curve(xs, color=INDIGO, lw=2.0, alpha=1.0, zorder=24)
    shots = V.T_START + np.arange(7) * V.T_TRAVEL / 6
    for k, t in enumerate(shots):
        o = plan.flat(t)
        vehicle(s, o["x_Q"], o["R"], o["x_L"])
        if k == 2:
            cone_below(s, o["x_Q"])
            s.label(o["x_Q"] - 0.55 * E3, r"$q_c$", dx=-8, dy=0, size=11)
            s.label(o["x_Q"], "the cable must stay\nin this cone", dx=-10, dy=30, size=9.5, **SANS)
        if k == 4:
            s.label(o["x_Q"], r"$x_Q$", dx=16, dy=9, size=12)
            s.label(o["x_L"], r"$x_L$", dx=4, dy=-15, size=12, color=INDIGO)
            s.label(0.5 * (o["x_Q"] + o["x_L"]), r"$q$", dx=9, dy=2, size=12)
    s.label(xs[0], "start", dx=6, dy=-14, size=9.5, **SANS)
    s.label(xs[95], "course of the load", dx=44, dy=-16, size=9.5, color=INDIGO, **SANS)
    s.label(xq[112], "path of the quadrotor", dx=0, dy=13, size=9.5, color=MUTED, **SANS)
    s.save(OUT / "ch18-mission.svg", pad=0.1)


# -------------------------------------------------------------- 2. the plan
def plan():
    p_plan, p_hurry, _ = plans()
    fig, axes = new_plot(6.4, 3.9, nrows=2, sharex=True)
    hover_f = (V.P0.m_Q + V.P0.m_L) * V.P0.g
    hover_T = V.P0.m_L * V.P0.g
    for pl, col, name in ((p_plan, INDIGO, f"{V.T_TRAVEL:.0f} s"), (p_hurry, TERRACOTTA, f"{V.T_HURRY:.0f} s")):
        r = V.plan_report(pl)
        t = r["t"] - pl.start
        axes[0].plot(t, r["swing"], color=col, lw=1.8)
        axes[1].plot(t, r["T"] / hover_T, color=col, lw=1.8)
        axes[1].plot(t, r["f"] / hover_f, color=col, lw=1.0, linestyle=DASH)
        pos = (3.95, 21.5) if col == INDIGO else (2.2, 32.5)
        axes[0].text(*pos, f"course in {name}", color=col, fontsize=9.5, path_effects=halo(), **SANS)
    axes[0].axhspan(V.CONE_DEG, 36, color=PALE, lw=0)
    axes[0].axhline(V.CONE_DEG, color=INK, lw=1.2)
    axes[0].text(0.15, V.CONE_DEG + 1.2, "outside the cone", fontsize=9.5, color=TERRACOTTA, **SANS)
    axes[0].set_ylim(0, 36)
    axes[0].set_ylabel("cable angle (degrees)", fontsize=9.5, **SANS)
    axes[1].axhline(1.0, color=MUTED, lw=0.6)
    axes[1].set_ylim(0.5, 1.35)
    axes[1].set_ylabel("fraction of hover value", fontsize=9.5, **SANS)
    axes[1].set_xlabel("time since the course began (s)", fontsize=9.5, **SANS)
    axes[1].text(0.15, 1.24, "solid: tension     dashed: thrust", fontsize=9.5, color=INK, **SANS)
    axes[1].set_xlim(0, V.T_TRAVEL)
    fig.tight_layout()
    save(fig, OUT / "ch18-plan.svg")


# ------------------------------------------------- 3. feedforward (animation)
FF_END = 8.0


def feedforward_frame(t):
    R_ = runs()
    good, bad = R_["nominal"], R_["nominal_no_ff"]
    s = Scene(width=6.6, height=3.5, **VIEW, **LIMS)
    room(s, labels=False)
    tt = t * FF_END
    k = at(good, tt)
    s.curve(good["x_d"], color=MUTED, lw=0.9, alpha=1.0, linestyle=DASH, zorder=16)
    s.curve(bad["x_L"][: k + 1], color=TERRACOTTA, lw=1.6, alpha=1.0, zorder=24)
    s.curve(good["x_L"][: k + 1], color=INDIGO, lw=1.9, alpha=1.0, zorder=25)
    vehicle(s, bad["x_Q"][k], rot_of(bad, k), bad["x_L"][k], color=TERRACOTTA, load=TERRACOTTA, z=28)
    vehicle(s, good["x_Q"][k], rot_of(good, k), good["x_L"][k], z=34)
    s.dot(good["x_d"][k], color=MUTED, size=6.5, hollow=True, zorder=44)
    s.text2d(0.45, 2.52, "with the flat feedforward", fontsize=9.5, color=INDIGO, ha="left", **SANS)
    s.text2d(0.45, 2.3, "the same gains, feedforward set to zero", fontsize=9.5, color=TERRACOTTA, ha="left", **SANS)
    err = 100 * bad["e_x"][k]
    if tt < V.T_START:
        msg = "both hover, waiting for the course to begin"
    elif tt < T_SET:
        msg = f"without feedforward the cable lags: the load is {err:.0f} cm from its reference"
    else:
        msg = f"the reference has stopped. Without feedforward the load is still {err:.0f} cm away"
    caption(s, f"t = {tt:.1f} s (shown at real speed).  " + msg, y=-0.47, size=9.5)
    return s


def feedforward():
    animate(feedforward_frame, OUT / "ch18-feedforward.gif", OUT / "ch18-feedforward-strip.png", n=80, fps=10,
            width_px=700, stills=(0.3, 0.5, 0.7, 0.95), strip_px=900, strip_cols=2)


# ------------------------------------------------------------------ 4. LQR
def lqr():
    R_ = runs()
    fig, axes = new_plot(6.4, 2.6, ncols=3, sharey=True)
    for ax, deg in zip(axes, (10, 30, 50)):
        for name, col in ((f"cas_off{deg}", INDIGO), (f"lqr_off{deg}", TEAL)):
            o = R_[name]
            ax.plot(o["t"], 100 * o["e_x"], color=col, lw=1.7)
        ax.axhline(5.0, color=MUTED, lw=0.6, linestyle=DASH)
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 130)
        ax.set_title(f"cable swung {deg}°", fontsize=9.5, **SANS)
        ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    axes[0].set_ylabel("load error (cm)", fontsize=9.5, **SANS)
    axes[0].text(2.2, 26, "cascade", color=INDIGO, fontsize=9.5, **SANS)
    axes[0].text(0.75, 58, "LQR", color=TEAL, fontsize=9.5, **SANS)
    axes[0].text(4.6, 9, "5 cm", color=MUTED, fontsize=9, **SANS)
    axes[2].text(3.1, 112, "LQR: lost", color=TEAL, fontsize=9.5, **SANS)
    fig.tight_layout()
    save(fig, OUT / "ch18-lqr.svg")


# ----------------------------------------------------------- 5. adaptation
def adapt():
    R_ = runs()
    plain, ad = R_["plain"], R_["truth"]
    fig, axes = new_plot(6.4, 2.7, ncols=2)
    for ax in axes:
        ax.axvspan(*V.GUST, color=SAND, lw=0)
        ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
        ax.set_xlim(0, V.T_END)
    axes[0].plot(plain["t"], 100 * plain["e_x"], color=TERRACOTTA, lw=1.7)
    axes[0].plot(ad["t"], 100 * ad["e_x"], color=INDIGO, lw=1.7)
    axes[0].set_ylabel("load error (cm)", fontsize=9.5, **SANS)
    axes[0].set_ylim(0, 30)
    axes[0].text(5.4, 22.5, "no adaptation", color=TERRACOTTA, fontsize=9.5, **SANS)
    axes[0].text(5.7, 8.5, "adaptation", color=INDIGO, fontsize=9.5, **SANS)
    axes[0].text(np.mean(V.GUST), 27.5, "gust", color=INK, fontsize=9.5, ha="center", **SANS)
    axes[1].plot(ad["t"], ad["Delta"][:, 2], color=MUTED, lw=1.0, linestyle=DASH)
    axes[1].plot(ad["t"], ad["Delta_f"][:, 2], color=INDIGO, lw=1.7)
    axes[1].set_ylabel("vertical lumped force (N)", fontsize=9.5, **SANS)
    axes[1].set_ylim(-2.0, 0.2)
    axes[1].text(5.3, -1.25, "filtered estimate", color=INDIGO, fontsize=9.5, **SANS)
    axes[1].text(5.3, -1.72, "truth", color=MUTED, fontsize=9.5, **SANS)
    fig.tight_layout()
    save(fig, OUT / "ch18-adapt.svg")


# ------------------------------------------------- 6. the cone (animation)
def cone_frame(t):
    R_ = runs()
    free, safe = R_["hurry_free"], R_["hurry_safe"]
    s = Scene(width=6.4, height=3.9, azim=-90, elev=-90, xlim=(-1.25, 1.25), ylim=(-0.87, 0.65))
    ang = np.radians(V.CONE_DEG)
    outer, inner = small_circle(-E3, np.radians(36)), small_circle(-E3, ang)
    ring = np.vstack([s.project(outer), s.project(inner)[::-1]])
    s.ax.add_patch(Polygon(ring, closed=True, fc=PALE, ec="none", zorder=5))
    s.ax.add_patch(Polygon(s.project(inner), closed=True, fc=PAPER, ec="none", zorder=6))
    for a in (10, 20):
        s.curve(small_circle(-E3, np.radians(a)), color=MUTED, lw=0.5, alpha=1.0, zorder=8)
    s.curve(inner, color=INK, lw=1.7, alpha=1.0, zorder=18)
    s.dot(-E3, color=INK, size=3.5)
    s.label(-E3, r"$q_c$", dx=12, dy=-9, size=12)
    n = len(safe["t"])
    t_end = V.T_START + V.T_HURRY + 1.5
    k = min(at(safe, t * t_end), n - 1)
    qr = safe["qd_plan"]
    out = np.degrees(np.arccos(np.clip(-qr[:, 2], -1, 1))) > V.CONE_DEG
    s.curve(qr, color=MUTED, lw=0.9, alpha=1.0, linestyle=DASH, zorder=15)
    seg = np.where(out, 1.0, np.nan)[:, None] * qr
    s.curve(seg, color=TERRACOTTA, lw=1.1, alpha=1.0, linestyle=DASH, zorder=16)
    s.curve(free["q"][: k + 1], color=TERRACOTTA, lw=1.5, alpha=1.0, zorder=19)
    s.curve(safe["q"][: k + 1], color=INDIGO, lw=2.1, alpha=1.0, zorder=20)
    s.dot(qr[k], color=TERRACOTTA if out[k] else MUTED, size=6.5, hollow=True, zorder=40)
    s.dot(free["q"][k], color=TERRACOTTA, size=5.5, zorder=41)
    s.dot(safe["q"][k], color=INDIGO, size=6.5, zorder=42)
    edge = small_circle(-E3, ang)[np.argmax(s.project(small_circle(-E3, ang))[:, 0])]
    s.label(edge, r"$H = 0$", dx=0, dy=0, size=11)
    s.text2d(-1.2, 0.56, "no filter", fontsize=9.5, color=TERRACOTTA, ha="left", **SANS)
    s.text2d(-1.2, 0.46, "safety filter", fontsize=9.5, color=INDIGO, ha="left", **SANS)
    s.text2d(-1.2, 0.36, "cable of the reference", fontsize=9.5, color=MUTED, ha="left", **SANS)
    s.text2d(1.2, 0.56, "must never be here:", fontsize=9.5, color=TERRACOTTA, ha="right", **SANS)
    s.text2d(1.2, 0.46, "the shaded ring", fontsize=9.5, color=TERRACOTTA, ha="right", **SANS)
    tt = safe["t"][k]
    if out[k]:
        msg = "the reference is outside the cone: the filtered cable turns back at the edge"
    elif safe["e_x"][k] > 0.1:
        msg = f"the reference is inside, but the filtered load is {100 * safe['e_x'][k]:.0f} cm behind it"
    else:
        msg = "the reference is inside the cone: both cables follow it"
    caption(s, f"t = {tt:.1f} s (real speed), seen from below.  " + msg, y=-0.79, size=9.5)
    return s


def cone():
    animate(cone_frame, OUT / "ch18-cone.gif", OUT / "ch18-cone-strip.png", n=84, fps=12,
            width_px=700, stills=(0.3, 0.48, 0.66, 1.0), strip_px=900, strip_cols=2)


# ------------------------------------------------------------ 7. estimate
def estimate():
    R_ = runs()
    wrong, truth = R_["est_wrong"], R_["truth"]
    fig, axes = new_plot(6.4, 3.7, nrows=2)
    ok = np.isfinite(wrong["y_err"])
    axes[0].plot(wrong["t"][ok], wrong["y_err"][ok], "o", ms=2.2, color=MUTED, mec="none")
    axes[0].plot(np.r_[0.0, wrong["t"]], np.r_[40.0, wrong["e_qhat"]], color=INDIGO, lw=1.7)
    axes[0].plot([0.0], [40.0], "o", ms=5, color=INDIGO, clip_on=False, zorder=5)
    axes[0].text(0.06, 36, "first guess, 40° wrong", color=INDIGO, fontsize=9.5, va="center", **SANS)
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 3.0)
    axes[0].set_ylim(0.05, 60)
    axes[0].set_yticks([0.1, 1, 10])
    axes[0].set_yticklabels(["0.1", "1", "10"])
    axes[0].set_ylabel("direction error (degrees)", fontsize=9.5, **SANS)
    axes[0].text(1.6, 9.0, "one reading", color=MUTED, fontsize=9.5, **SANS)
    axes[0].text(0.3, 0.075, "the estimate", color=INDIGO, fontsize=9.5, **SANS)
    axes[1].axvspan(*V.GUST, color=SAND, lw=0)
    axes[1].plot(truth["t"], truth["e_omegahat"], color=INDIGO, lw=1.2)
    axes[1].set_xlim(0, V.T_END)
    axes[1].set_ylim(0, 0.2)
    axes[1].set_ylabel("angular velocity error (rad/s)", fontsize=9.5, **SANS)
    axes[1].set_xlabel("time (s)", fontsize=9.5, **SANS)
    axes[1].text(np.mean(V.GUST), 0.18, "gust", color=INK, fontsize=9.5, ha="center", **SANS)
    fig.tight_layout()
    save(fig, OUT / "ch18-estimate.svg")


# ------------------------------------------- 8. the whole flight (animation)
def bar(s, x, y0, height, lo, hi, value, floor, name, unit_text):
    """A vertical gauge in canvas units: forbidden part shaded, the value as an indigo bar."""
    w = 0.13

    def yy(v):
        return y0 + height * (np.clip(v, lo, hi) - lo) / (hi - lo)
    s.ax.add_patch(Rectangle((x, y0), w, yy(floor) - y0, fc=PALE, ec="none", zorder=40))
    s.ax.add_patch(Rectangle((x, yy(max(min(value, floor), lo))), w, max(yy(value) - yy(floor), 0.0), fc=INDIGO,
                             ec="none", zorder=41))
    s.ax.add_patch(Rectangle((x, y0), w, height, fill=False, ec=INK, lw=0.8, zorder=42))
    s.ax.plot([x - 0.04, x + w + 0.04], [yy(floor)] * 2, color=INK, lw=1.2, zorder=43)
    s.ax.text(x + w / 2, y0 + height + 0.1, name, fontsize=10, ha="center", va="center", color=INK, zorder=50)
    s.ax.text(x + w / 2, y0 - 0.1, unit_text, fontsize=8.5, ha="center", va="center", color=INK, zorder=50, **SANS)


def flight_frame(t):
    o = runs()["full"]
    s = Scene(width=6.6, height=3.5, **VIEW, **LIMS)
    room(s, labels=False)
    tt = t * V.T_END
    k = at(o, tt)
    s.curve(o["x_d"], color=MUTED, lw=0.9, alpha=1.0, linestyle=DASH, zorder=16)
    s.curve(o["x_L"][: k + 1], color=INDIGO, lw=1.9, alpha=1.0, zorder=25)
    x_Q, R = o["x_Q"][k], rot_of(o, k)
    cone_below(s, x_Q, lw=0.6, axis=False)
    s.curve(np.array([x_Q, x_Q + V.P0.l * o["qhat"][k]]), color=INDIGO, lw=2.2, alpha=1.0, zorder=29)
    s.curve(np.array([x_Q, o["x_L"][k]]), color=MUTED, lw=0.8, alpha=1.0, linestyle=DASH, zorder=31)
    quadrotor(s, x_Q, R, arm=0.19, lw=1.4, zorder=34)
    s.dot(o["x_L"][k], color=INDIGO, size=6.0, zorder=36)
    s.dot(o["x_d"][k], color=MUTED, size=6.5, hollow=True, zorder=44)
    gust_now = o["d"][k]
    if np.linalg.norm(gust_now) > 0.05:
        s.arrow(x_Q - 0.45 * gust_now, x_Q - 0.08 * gust_now, color=TEAL, lw=1.6, head=0.09, zorder=38)
        s.label(x_Q - 0.45 * gust_now, "gust", dx=4, dy=10, size=9.5, color=TEAL, **SANS)
    bar(s, -0.52, 1.72, 0.72, 0.0, 6.0, o["T"][k], V.T_MIN, r"$T$", f"{o['T'][k]:.1f} N")
    bar(s, -0.02, 1.72, 0.72, -0.02, 0.1, o["H"][k], 0.0, r"$H$", f"{o['H'][k]:.3f}")
    if tt < V.T_START:
        msg = "hovering with a load half as heavy again as the manifest says"
    elif np.linalg.norm(gust_now) > 0.05:
        msg = f"crosswind of {np.linalg.norm(gust_now):.1f} N on the quadrotor; load {100 * o['e_x'][k]:.0f} cm off"
    elif tt < T_SET:
        msg = f"load {100 * o['e_x'][k]:.0f} cm from its reference; estimated cable {o['e_qhat'][k]:.1f}° from the true one"
    else:
        msg = f"set down: {100 * o['e_x'][k]:.1f} cm from the target, cable {o['swing'][k]:.1f}° from the vertical"
    caption(s, f"t = {tt:.1f} s (shown at real speed).  " + msg, y=-0.47, size=9.5)
    return s


def flight():
    animate(flight_frame, OUT / "ch18-flight.gif", OUT / "ch18-flight-strip.png", n=90, fps=10,
            width_px=700, stills=(0.22, 0.42, 0.6, 0.95), strip_px=900, strip_cols=2)


FIGURES = dict(mission=mission, plan=plan, feedforward=feedforward, lqr=lqr, adapt=adapt, cone=cone,
               estimate=estimate, flight=flight)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()
