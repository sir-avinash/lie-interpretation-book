"""Figures for Chapter 10.  Run: python figures/src/ch10.py [name ...]

Every curve comes from the flat maps of verify/ch10.py and from open-loop or
closed-loop simulations of the robot model library (case_studies/robots).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (DASH, INDIGO, INK, MUTED, PAPER, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, caption, halo, new_plot, phase, quadrotor, save)

_spec = importlib.util.spec_from_file_location("verify_ch10", ROOT / "verify" / "ch10.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
robots = V.robots

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
G = 9.81
PSI0 = np.zeros(3)

RL = robots.get("quadrotor_load")
PL = RL.params
EIGHT = V.figure_eight()
LAP = 2 * np.pi
_cache = {}


def eight_run():
    """Two laps of the figure-eight, open loop on the library model."""
    if "eight" not in _cache:
        h, every = 0.005, 4
        n = int(round(LAP / h))

        def u(t):
            o = V.load_flat(EIGHT.jet(t, 6), PSI0, PL)
            return o["f"], o["tau"]
        s0 = V.load_state(RL, V.load_flat(EIGHT.jet(0.0, 6), PSI0, PL))
        states = V.simulate(RL, s0, u, h, n, every=every)
        ts = np.arange(len(states)) * h * every
        err = max(np.linalg.norm(s.x_L - EIGHT(t)) for s, t in zip(states, ts))
        print(f"eight: open-loop error over the lap {err:.1e} m")
        _cache["eight"] = (ts, states)
    return _cache["eight"]


# ----------------------------------------------------------------- 1. opening
EIGHT_VIEW = dict(azim=-68, elev=17)


def opening():
    s = Scene(width=6.4, height=3.5, **EIGHT_VIEW)
    tt = np.linspace(0, LAP, 361)
    O = [V.load_flat(EIGHT.jet(t, 6), PSI0, PL) for t in tt]
    xL = np.array([o["x_L"] for o in O])
    xQ = np.array([o["x_Q"] for o in O])
    ground = -0.75
    s.curve(np.column_stack([xL[:, 0], xL[:, 1], np.full(len(tt), ground)]), color=MUTED, lw=0.6, alpha=0.5)
    s.curve(xQ, color=MUTED, lw=0.9, alpha=0.9, zorder=18)
    s.curve(xL, color=INDIGO, lw=2.0, alpha=1.0, zorder=24)
    shots = np.arange(7) * LAP / 7 + 0.35
    k_right = int(np.argmax([V.load_flat(EIGHT.jet(t, 6), PSI0, PL)["x_Q"] @ s.right for t in shots]))
    for k, t in enumerate(shots):
        o = V.load_flat(EIGHT.jet(t, 6), PSI0, PL)
        s.curve(np.array([o["x_Q"], o["x_L"]]), color=INK, lw=1.0, alpha=1.0, zorder=28)
        quadrotor(s, o["x_Q"], o["R"], arm=0.17, lw=1.5, zorder=34)
        s.dot(o["x_L"], color=INDIGO, size=6.5, zorder=36)
        if k == k_right:
            s.label(o["x_L"], r"$x_L$", dx=17, dy=-5, size=12, color=INDIGO)
            s.label(o["x_Q"], r"$x_Q$", dx=36, dy=4, size=12)
            s.label(0.45 * o["x_L"] + 0.55 * o["x_Q"], r"$q$", dx=10, dy=0, size=12)
    s.text2d(-1.45, -0.42, "drawn:\nthe curve of the load", fontsize=10, color=INDIGO, ha="right", **SANS)
    s.text2d(-1.45, 0.95, "computed: cable,\nquadrotor path, tilt,\nthrust, torque", fontsize=10, color=INK,
             ha="right", **SANS)
    s.pad_to([-2.75 * s.right, 2.0 * s.right])
    s.save(OUT / "ch10-opening.svg", pad=0.08)


# ----------------------------------------------------------------- 2. the hand
HAND_RATES = (("slow: 2 rad/s", 2.0), ("its own rate: 3.13 rad/s", np.sqrt(G)), ("fast: 4.5 rad/s", 4.5))
HAND_TF = 2 * np.pi
HAND_N = 100


def hand_data():
    if "hand" not in _cache:
        times = np.linspace(0, HAND_TF, HAND_N)[1:]
        runs = []
        for _, w in HAND_RATES:
            path = V.Trig([0.0, 0.0, 0.0], [[0.3, 0.0, 0.0]], [[w, 1.0, 1.0]], [[np.pi / 2, 0.0, 0.0]])
            pts, hands = V.string_run(path, 1.0, times, h=1e-3)
            p0, h0 = path(0.0), V.derivs(V.hand_position(path.jet(0.0, 2), 1.0))[0]
            runs.append((path, np.vstack([p0, pts]), np.vstack([h0, hands])))
            print(f"hand, {w:.2f} rad/s: simulated weight within "
                  f"{max(np.linalg.norm(p - path(t)) for p, t in zip(pts, times)):.1e} m of its line")
        _cache["hand"] = runs
    return _cache["hand"]


def hand_frame(t):
    s = Scene(width=6.4, height=3.3, azim=-90, elev=0, xlim=(-1.84, 1.58), ylim=(-0.47, 1.26))
    k = int(round(phase(t, 0.0, 1.0) * (HAND_N - 1)))
    tt = k * HAND_TF / (HAND_N - 1)
    for (name, w), (path, pts, hands), cx in zip(HAND_RATES, hand_data(), (-1.08, 0.0, 1.08)):
        off = np.array([cx, 0.0, 0.0])
        s.curve(np.array([[-0.3, 0, 0], [0.3, 0, 0]]) + off, color=INDIGO, lw=1.4, alpha=1.0, zorder=10)
        s.curve(np.array([[-0.3, 0, 1.0], [0.3, 0, 1.0]]) + off, color=MUTED, lw=0.6, alpha=1.0, zorder=8,
                linestyle=DASH)
        lo, hi = hands[:, 0].min(), hands[:, 0].max()                # how far the hand travels
        s.curve(np.array([[lo, 0, 1.09], [hi, 0, 1.09]]) + off, color=INK, lw=2.4, alpha=1.0, zorder=9)
        xw, xh = pts[k] + off, hands[k] + off
        A = V.derivs(path.jet(tt, 2))[2] + G * E3
        s.curve(np.array([xh, xw]), color=INK, lw=1.2, alpha=1.0, zorder=20)
        s.arrow(xw, xw + 0.42 * A / G, color=TERRACOTTA, lw=1.7, head=0.07, zorder=24)
        s.dot(xw, color=INDIGO, size=8, zorder=30)
        s.dot(xh, color=INK, size=7, hollow=True, zorder=30)
        s.text2d(cx, -0.17, name, fontsize=9.5, color=INK, **SANS)
    s.text2d(-1.8, 1.0, "hand", fontsize=9.5, color=INK, ha="left", **SANS)
    s.text2d(-1.8, 1.17, "its range", fontsize=9.5, color=INK, ha="left", **SANS)
    s.text2d(-1.8, 0.0, "weight", fontsize=9.5, color=INDIGO, ha="left", **SANS)
    s.text2d(-0.13, -0.38, f"t = {tt:.1f} s: the string lies along the missing force (terracotta)", fontsize=10, **SANS)
    return s


def hand():
    animate(hand_frame, OUT / "ch10-hand.gif", OUT / "ch10-hand-strip.png", n=HAND_N, fps=14, width_px=640,
            stills=(0.0, 0.085, 0.17, 0.25), strip_px=760, strip_cols=2)


# ----------------------------------------------------------------- 3. the ladder
def ladder():
    fig, axes = new_plot(width=6.4, height=3.0, ncols=2, gridspec_kw=dict(width_ratios=[5, 7], wspace=0.08))
    rungs = (
        ("quadrotor", r"x", [(0, r"$x$", INDIGO), (1, r"$\dot{x}$", INK), (2, r"$f,\ R$", INK),
                             (3, r"$\Omega$", INK), (4, r"$\tau$", TERRACOTTA)]),
        ("quadrotor with a cable-suspended load", r"x_L",
         [(0, r"$x_L$", INDIGO), (1, r"$\dot{x}_L$", INK), (2, r"$q,\ T,\ x_Q$", INK), (3, r"$\omega$", INK),
          (4, r"$f,\ R$", INK), (5, r"$\Omega$", INK), (6, r"$\tau$", TERRACOTTA)]),
    )
    for ax, (title, sym, items) in zip(axes, rungs):
        n = len(items)
        for k, text, col in items:
            ax.plot([k - 0.5, k + 0.5], [k, k], color=col, lw=2.2, solid_capstyle="round")
            if k < n - 1:
                ax.plot([k + 0.5, k + 0.5], [k, k + 1], color=MUTED, lw=0.8)
            ax.text(k, k + 0.22, text, ha="center", va="bottom", fontsize=11, color=col, path_effects=halo())
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(-0.6, 7.3)
        ax.set_xticks(range(n))
        names = [f"${sym}$", rf"$\dot{{{sym}}}$", rf"$\ddot{{{sym}}}$"] + [f"${sym}^{{({k})}}$" for k in range(3, 7)]
        ax.set_xticklabels(names[:n], fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=10, color=INK, loc="left", **SANS)
        ax.set_xlabel("highest derivative of the flat output it needs", fontsize=9, color=INK, **SANS)
    axes[1].text(5.9, 0.4, "state", fontsize=9.5, color=INK, ha="right", **SANS)
    axes[1].text(5.9, -0.25, "input (the thrust $f$ is one too)", fontsize=9.5, color=TERRACOTTA, ha="right", **SANS)
    axes[1].text(5.9, 1.05, "flat output", fontsize=9.5, color=INDIGO, ha="right", **SANS)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.9, bottom=0.2)
    save(fig, OUT / "ch10-ladder.svg")


# ----------------------------------------------------------------- 4. the figure-eight, open loop
def eight_frame(t):
    s = Scene(width=6.4, height=3.6, xlim=(-1.72, 1.72), ylim=(-0.72, 1.22), **EIGHT_VIEW)
    ts, states = eight_run()
    k = int(round(phase(t, 0.03, 0.97) * (len(ts) - 1)))
    tt, st = ts[k], states[k]
    lap = np.array([EIGHT(a) for a in np.linspace(0, LAP, 241)])
    s.curve(lap, color=MUTED, lw=1.0, alpha=1.0, zorder=14, linestyle=DASH)
    trace = np.array([x.x_L for x in states[:k + 1]])
    s.curve(trace, color=INDIGO, lw=1.9, alpha=1.0, zorder=22)
    xQ = RL.quadrotor_position(st)
    o = V.load_flat(EIGHT.jet(tt, 6), PSI0, PL)
    s.curve(np.array([xQ, st.x_L]), color=INK, lw=1.1, alpha=1.0, zorder=26)
    A = o["a_L"] + G * E3
    s.arrow(st.x_L, st.x_L + 0.42 * A / G, color=TERRACOTTA, lw=1.8, head=0.08, zorder=30)
    quadrotor(s, xQ, st.R, arm=0.2, lw=1.6, zorder=34)
    s.dot(EIGHT(tt), color=MUTED, size=9, hollow=True, zorder=31)
    s.dot(st.x_L, color=INDIGO, size=5.5, zorder=36)
    tilt = np.degrees(np.arccos(np.clip(st.R[2, 2], -1, 1)))
    s.text2d(-1.66, 1.12, f"t = {tt:4.1f} s", fontsize=10, color=MUTED, ha="left", **SANS)
    s.text2d(1.66, 1.12, f"tension {o['T']:.2f} N   thrust {o['f']:.1f} N   tilt {tilt:2.0f}°", fontsize=9.5,
             color=MUTED, ha="right", **SANS)
    msg = ("no feedback: thrust and torque were computed from the dashed curve alone" if tt < 0.55 * LAP
           else "the cable stays on the missing force of the load (terracotta)")
    caption(s, msg, y=-0.62)
    return s


def eight():
    animate(eight_frame, OUT / "ch10-eight.gif", OUT / "ch10-eight-strip.png", n=96, fps=12, width_px=640,
            stills=(0.12, 0.38, 0.64, 0.95), strip_px=760, strip_cols=2)


# ----------------------------------------------------------------- 5. limits
def limits():
    from scipy.optimize import brentq

    def F_crossing(r):
        return V.load_flat(V.figure_eight(rate=r).jet(0.0, 6), PSI0, PL)["F"][2]
    rate_f = brentq(F_crossing, 1.5, 2.0)                      # the thrust vanishes at the crossing
    rate_c = np.sqrt(G / 0.6)                                  # the tension vanishes at the crossing
    rates = np.unique(np.concatenate([np.linspace(0.6, 3.6, 61), np.linspace(3.65, 4.0, 8),
                                      rate_f + np.array([-0.03, -0.012, -0.004, 0.004, 0.012, 0.03])]))
    T_min, f_min, tau_max = [], [], []
    for r in rates:
        path = V.figure_eight(rate=r)
        O = [V.load_flat(path.jet(t, 6), PSI0, PL) for t in np.linspace(0, 2 * np.pi / r, 721)]
        T_min.append(min(o["T"] for o in O) / (PL.m_L * G))
        f_min.append(min(o["f"] for o in O) / ((PL.m_Q + PL.m_L) * G))
        tau_max.append(max(np.linalg.norm(o["tau"]) for o in O))
    print(f"limits: zero thrust at rate {rate_f:.3f}, zero tension at rate {rate_c:.3f}")
    fig, (a1, a2) = new_plot(width=6.4, height=3.8, nrows=2, sharex=True)
    a1.plot(np.append(rates, rate_c), np.append(T_min, 0.0), color=INDIGO, lw=1.8)
    a1.plot(rates, f_min, color=TEAL, lw=1.8)
    a1.plot([rate_c, rate_f], [0.0, 0.0], "o", ms=5, mfc=PAPER, mec=INK, zorder=5, clip_on=False)
    a1.text(3.05, 0.62, "smallest tension", fontsize=9.5, color=INDIGO, ha="left", **SANS)
    a1.text(0.62, 0.52, "smallest thrust", fontsize=9.5, color=TEAL, ha="left", **SANS)
    a1.annotate("zero tension:\nslack cable", (rate_c, 0.0), xytext=(-4, 34), textcoords="offset points",
                fontsize=9, color=INK, ha="center", va="center", **SANS)
    a1.annotate("zero thrust", (rate_f, 0.0), xytext=(6, 12), textcoords="offset points",
                fontsize=9, color=INK, ha="left", va="center", **SANS)
    a1.set_ylabel("fraction of the\nvalue in hover", fontsize=9.5, **SANS)
    a1.set_ylim(0.0, 1.5)
    a2.semilogy(rates, tau_max, color=TERRACOTTA, lw=1.8)
    a2.axvline(rate_f, color=MUTED, lw=0.7, linestyle=DASH)
    a1.axvline(rate_f, color=MUTED, lw=0.7, linestyle=DASH)
    i1 = int(np.argmin(np.abs(rates - 1.0)))
    a2.plot([1.0], [tau_max[i1]], "o", ms=4.5, color=INK, zorder=5)
    a2.annotate("the example", (1.0, tau_max[i1]), xytext=(0, 14), textcoords="offset points", fontsize=9,
                color=INK, ha="center", **SANS)
    a2.text(rate_f + 0.06, 3e-3, "beyond: asked to fly inverted at the crossing", fontsize=9, color=MUTED,
            va="center", ha="left", **SANS)
    a2.set_ylabel("largest torque (N·m)", fontsize=9.5, **SANS)
    a2.set_xlabel("rate of the figure-eight (rad/s)", fontsize=9.5, **SANS)
    a2.set_xlim(0.55, 4.2)
    a2.set_ylim(1e-3, 1e6)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.12, hspace=0.12)
    save(fig, OUT / "ch10-limits.svg")


# ----------------------------------------------------------------- 6. the chain
def chain_data():
    if "chain" not in _cache:
        rc, path = V.chain_demo()
        mass, pc = rc.joint_masses(), rc.params
        t0, h, every = 0.55, 0.0025, 8
        lap = 2 * np.pi / V.CHAIN_RATE

        def flat(t):
            return V.chain_flat(path.jet(t, 10), PSI0, mass, pc.l, pc.J[0])

        def u(t):
            o = flat(t)
            return o["f"], o["tau"]
        states = V.simulate(rc, V.cable_state(rc, flat(t0)), u, h, int(round(lap / h)), t0=t0, every=every)
        ts = t0 + np.arange(len(states)) * h * every
        err = max(np.linalg.norm(rc.positions(s)[-1] - path(t)) for s, t in zip(states, ts))
        print(f"chain: open-loop error of the free end over the lap {err:.1e} m")
        _cache["chain"] = (rc, path, flat, ts, states, t0)
    return _cache["chain"]


CHAIN_STEPS = (
    (0.00, "given: the curve of the free end, and its derivatives"),
    (0.07, "the last link lies along the missing force of the last mass"),
    (0.19, "one joint up: the link balances everything that hangs below it"),
    (0.31, "and again, to the top of the cable"),
    (0.43, "what is left at the top is the thrust, and the quadrotor's map takes over"),
    (0.56, "now let time run: the whole cable follows from the one curve"),
)


def chain_frame(t):
    s = Scene(width=6.4, height=4.0, azim=-68, elev=14, xlim=(-1.6, 1.6), ylim=(-0.66, 1.34))
    rc, path, flat, ts, states, t0 = chain_data()
    pc, mass = rc.params, rc.joint_masses()
    lap = np.array([path(a) for a in np.linspace(0, 2 * np.pi / V.CHAIN_RATE, 241)])
    s.curve(lap, color=MUTED, lw=1.0, alpha=1.0, zorder=14, linestyle=DASH)
    run = phase(t, 0.56, 0.98)
    k = int(round(run * (len(ts) - 1)))
    st, tt = states[k], ts[k]
    o = flat(tt)
    X = rc.positions(st)                                      # joints 0..3
    n = 3
    # how much of the construction is visible: links n, n-1, ..., 1 appear in turn
    grow = [phase(t, 0.07 + 0.12 * j, 0.15 + 0.12 * j) for j in range(n)]       # j = 0: last link
    top = phase(t, 0.43, 0.5)
    if k > 0:
        s.curve(np.array([rc.positions(x)[-1] for x in states[:k + 1]]), color=INDIGO, lw=1.9, alpha=1.0, zorder=22)
    pull = o["T"][:, None] * o["q"]                           # tension vectors T_i q_i
    scale = 0.06                                              # meters of arrow per newton
    for j in range(n):
        i = n - j                                             # link i, between joints i - 1 and i
        if grow[j] <= 0:
            break
        s.curve(np.array([X[i], X[i] + grow[j] * (X[i - 1] - X[i])]), color=INK, lw=1.3, alpha=1.0, zorder=26)
        if grow[j] >= 1 and i > 1:
            s.dot(X[i - 1], color=INK, size=4.5, zorder=33)
    if run <= 0 and top <= 0:
        # the tension vector of the link being placed, drawn at its lower joint: it balances
        # the missing forces of every mass from there down
        j_now = max([j for j in range(n) if grow[j] > 0], default=0)
        i = n - j_now
        s.arrow(X[i], X[i] - scale * pull[i - 1], color=TERRACOTTA, lw=1.7, head=0.07, zorder=30)
    if top > 0:
        quadrotor(s, X[0], st.R[0], arm=0.2, lw=1.6, zorder=34)
        if run <= 0:
            s.arrow(X[0], X[0] + 0.25 * scale * o["f"][0] * st.R[0][:, 2], color=TERRACOTTA, lw=1.7, head=0.07,
                    zorder=36)
    s.dot(path(tt), color=MUTED, size=9, hollow=True, zorder=31)
    s.dot(X[n], color=INDIGO, size=6, zorder=36)
    msg = [m for a, m in CHAIN_STEPS if t >= a][-1]
    caption(s, msg, y=-0.56)
    s.text2d(-1.55, 1.25, f"t = {tt:4.2f} s", fontsize=10, color=MUTED, ha="left", **SANS)
    return s


def chain():
    animate(chain_frame, OUT / "ch10-chain.gif", OUT / "ch10-chain-strip.png", n=110, fps=12, width_px=640,
            stills=(0.04, 0.17, 0.5, 0.8), strip_px=760, strip_cols=2)


# ----------------------------------------------------------------- 7. waypoints
def waypoints():
    rb = robots.get("quadrotor")
    p = rb.params
    W = V.WAYPOINTS
    s = Scene(width=6.4, height=3.3, azim=-72, elev=38, xlim=(-0.9, 11.0), ylim=(-1.95, 2.1))
    sp6 = V.min_derivative_spline(W, V.chord_times(W, 6.0), 4)
    tt = np.linspace(0, 6.0, 301)
    X = np.array([sp6(t) for t in tt])
    ground = -0.8
    s.curve(np.column_stack([X[:, 0], X[:, 1], np.full(len(tt), ground)]), color=MUTED, lw=0.6, alpha=0.5)
    for w in W:
        s.curve(np.array([[w[0], w[1], ground], w]), color=MUTED, lw=0.5, alpha=0.6, linestyle=DASH)
    s.curve(X, color=INDIGO, lw=1.9, alpha=1.0, zorder=22)
    for t in np.arange(0, 6.01, 0.6):
        o = V.quadrotor_flat(sp6.jet(t, 4), PSI0, p)
        quadrotor(s, o["x"], o["R"], arm=0.24, lw=1.2, zorder=30)
    for w in W:
        s.dot(w, color=INK, size=5, zorder=36)
    s.label(W[0], "start", dx=-22, dy=6, size=9.5, **SANS)
    s.label(W[3], "waypoint", dx=30, dy=12, size=9.5, **SANS)
    s.label(W[-1], "end", dx=16, dy=-10, size=9.5, **SANS)
    fig = s.finalize()
    for row, (key, lab) in enumerate((("f", "thrust / hover"), ("tilt", "tilt (deg)"))):
        ax = fig.add_axes([0.715, 0.6 - 0.44 * row, 0.27, 0.33])
        from figkit import style_axes
        style_axes(ax)
        for t_f, col in ((6.0, INDIGO), (4.0, TERRACOTTA)):
            sp = V.min_derivative_spline(W, V.chord_times(W, t_f), 4)
            ts = np.linspace(0, t_f, 401)
            O = [V.quadrotor_flat(sp.jet(t, 4), PSI0, p) for t in ts]
            y = [o["f"] / (p.m * p.g) for o in O] if key == "f" else [np.degrees(np.arccos(o["R"][2, 2])) for o in O]
            ax.plot(ts, y, color=col, lw=1.5)
            if row == 1:
                ax.text(*((4.3, 44.0) if t_f < 5 else (4.9, 31.0)), f"$t_f$ = {t_f:g} s", fontsize=9, color=col, ha="left")
        ax.set_ylabel(lab, fontsize=9, **SANS)
        ax.set_xlim(0, 6.1)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)", fontsize=9, **SANS)
    save(fig, OUT / "ch10-waypoints.svg")


# ----------------------------------------------------------------- 8. feedforward
def feedforward():
    rb = robots.get("quadrotor")
    fig, axes = new_plot(width=6.4, height=2.9, ncols=2, sharey=True)
    names = {"zero": (r"zero feedforward", TERRACOTTA), "flat": ("flat feedforward of the reference", INDIGO),
             "exact": ("exact feedforward", TEAL)}
    for ax, kappa in zip(axes, (1.0, 1.5)):
        for mode in ("zero", "flat", "exact"):
            ts, es = V.cascade_run(rb, mode, kappa=kappa)
            lab, col = names[mode]
            ax.semilogy(ts, np.maximum(es, 1e-12), color=col, lw=1.6, label=lab)
            print(f"feedforward: {mode}, kappa {kappa}: ends at t = {ts[-1]:.1f} s with error {es[-1]:.1e} m")
        ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
        ax.set_xlim(0, 20)
        ax.set_ylim(1e-11, 20)
        ax.set_title("Lee's gains" if kappa == 1.0 else "attitude loop 1.5 times faster", fontsize=10, loc="left", **SANS)
    axes[0].set_ylabel("position error (m)", fontsize=9.5, **SANS)
    axes[0].text(10.2, 1.2, "zero", fontsize=9.5, color=TERRACOTTA, ha="right", **SANS)
    axes[0].text(5.4, 1.2, "flat", fontsize=9.5, color=INDIGO, ha="right", **SANS)
    axes[0].text(12, 2e-9, "exact", fontsize=9.5, color=TEAL, **SANS)
    axes[1].text(12, 5e-2, "zero: 1.9 cm remains", fontsize=9.5, color=TERRACOTTA, **SANS)
    axes[1].text(9.5, 3e-5, "flat", fontsize=9.5, color=INDIGO, **SANS)
    axes[1].text(10.5, 3e-9, "exact", fontsize=9.5, color=TEAL, **SANS)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.17, wspace=0.08)
    save(fig, OUT / "ch10-feedforward.svg")


FIGS = dict(opening=opening, hand=hand, ladder=ladder, eight=eight, limits=limits, chain=chain,
            waypoints=waypoints, feedforward=feedforward)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
