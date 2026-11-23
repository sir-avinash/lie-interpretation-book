"""Figures for Chapter 13.  Run: python figures/src/ch13.py

Every curve comes from the solvers and controllers in verify/ch13.py.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "verify"))

_spec = importlib.util.spec_from_file_location("ch13_verify", ROOT / "verify" / "ch13.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

import numpy as np  # noqa: E402
from figkit import (INDIGO, INK, MUTED, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, animate, ease, geodesic, great_circle, halo, new_plot, phase, save, small_circle, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")
DASH = (0, (3.5, 3))


def arc(p, q, n=40):
    return geodesic(p, q, n) if np.linalg.norm(np.cross(p, q)) > 1e-9 else np.array([p, q])


# ---------------------------------------------- opening: a step that stays on
def opening():
    """Newton's method for the mean of four directions, drawn where it happens."""
    _, pts = V.mean_data()
    qs, steps = V.s2_mean_newton(V.opening_start(pts), pts, 5)
    q0, q1 = qs[0], qs[1]
    dq = np.cross(steps[0], q0)                           # the Newton step as a displacement
    a1 = unit(dq)
    a2 = np.cross(q0, a1)
    view = unit(0.62 * q0 - 0.50 * a2 - 0.42 * a1)       # look across the step, not down on it
    s = Scene(width=6.4, height=3.6, azim=np.degrees(np.arctan2(view[1], view[0])), elev=np.degrees(np.arcsin(view[2])))
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    c = 1.003 * q0
    L = np.linalg.norm(dq)
    s.patch([c + (L + 0.12) * a1 + 0.34 * a2, c - 0.2 * a1 + 0.34 * a2, c - 0.2 * a1 - 0.34 * a2,
             c + (L + 0.12) * a1 - 0.34 * a2], color=SAND, alpha=0.8)
    for p in pts:
        s.sphere_curve(arc(q0, p), color=MUTED, lw=0.7, zorder=15)
    s.sphere_curve(1.002 * arc(q0, q1, 60), color=INDIGO, lw=2.3, zorder=31)
    tip = q0 + dq
    s.curve(np.array([tip, unit(tip)]), color=TERRACOTTA, lw=0.9, linestyle=DASH, zorder=29, alpha=1.0)
    s.arrow(q0, tip, color=TERRACOTTA, zorder=32)
    vis = [p for p in pts if s.point_visible(p)]
    for p in vis:
        s.dot(1.002 * p, color=INK, size=4.6)
    s.label(vis[0], "readings", dx=4, dy=-13, size=9.5, **SANS)
    s.dot(tip, color=TERRACOTTA, size=5.2)
    s.dot(1.002 * q0, color=INDIGO, size=5.4)
    s.dot(1.002 * q1, color=INDIGO, size=5.4)
    s.label(q0, r"$q$", dx=-13, dy=-4, color=INDIGO)
    s.label(tip, r"$q + \xi\times q$", dx=12, dy=7, color=TERRACOTTA, ha="left")
    s.label(tip, "a plain step leaves the sphere", dx=12, dy=-7, color=TERRACOTTA, size=9.5, ha="left", **SANS)
    s.label(q1, "the same step, wrapped onto the sphere", dx=14, dy=1, color=INDIGO, size=9.5, ha="left", **SANS)
    s.pad_to([tip + 2.1 * s.right, q0 - 1.9 * s.right])
    s.save(OUT / "ch13-opening.svg")
    return np.degrees(L)


# ----------------------------------------------------- Newton on SO(3)
def newton():
    Rs, _ = V.mean_data()
    R0 = Rs[0]
    R_mean = V.so3_mean_iterate(R0, Rs, "newton", 12)[-1]
    dist = lambda R: np.degrees(np.linalg.norm(V.log_so3(R_mean.T @ V.project_so3(R))))        # noqa: E731
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.17, top=0.93, wspace=0.3)
    floor = 3e-13
    spec = (("gradient", TEAL, "gradient step", (5.2, 3e-3), 1.5), ("project", MUTED, "Newton, then project", (3.55, 1e-7), 1.3),
            ("newton", INDIGO, "Newton,\nexponential", (0.15, 4e-10), 2.0))
    for name, col, text, pos, lw in spec:
        e = np.array([max(dist(R), floor) for R in V.so3_mean_iterate(R0, Rs, name, 8)])
        ax0.semilogy(range(len(e)), e, "-o", color=col, lw=lw, ms=3.8, zorder=6 if name == "newton" else 5)
        ax0.text(*pos, text, color=col, fontsize=9.5, path_effects=halo(), **SANS)
    ax0.axhspan(1e-14, 2e-12, color=SAND, alpha=0.7, lw=0)
    ax0.text(7.9, 4.5e-14, "rounding", color=MUTED, fontsize=9, ha="right", **SANS)
    ax0.set_ylim(1e-14, 500)
    ax0.set_xlim(-0.2, 8.2)
    ax0.set_yticks([1e2, 1e-2, 1e-6, 1e-10])
    ax0.minorticks_off()
    ax0.set_xlabel("iteration", fontsize=9.5, **SANS)
    ax0.set_ylabel("distance to the mean (degrees)", fontsize=9.5, **SANS)

    off_add = [np.linalg.norm(R.T @ R - np.eye(3)) for R in V.so3_mean_iterate(R0, Rs, "additive", 8)]
    off_exp = [max(np.linalg.norm(R.T @ R - np.eye(3)), 1e-16) for R in V.so3_mean_iterate(R0, Rs, "newton", 8)]
    ax1.semilogy(range(9), np.maximum(off_add, 1e-16), "-o", color=TERRACOTTA, lw=1.6, ms=3.8)
    ax1.semilogy(range(9), off_exp, "-o", color=INDIGO, lw=2.0, ms=3.8)
    ax1.text(2.0, 12, r"$R \leftarrow R + R\,\eta\!{}^\wedge$" if False else "additive update", color=TERRACOTTA, fontsize=9.5,
             path_effects=halo(), **SANS)
    ax1.text(2.0, 1.2e-14, "exponential update", color=INDIGO, fontsize=9.5, path_effects=halo(), **SANS)
    ax1.set_ylim(1e-17, 300)
    ax1.set_xlim(-0.2, 8.2)
    ax1.set_yticks([1, 1e-5, 1e-10, 1e-15])
    ax1.minorticks_off()
    ax1.set_xlabel("iteration", fontsize=9.5, **SANS)
    ax1.set_ylabel(r"distance from the group, $\|R^\top R - I\|$", fontsize=9.5, **SANS)
    save(fig, OUT / "ch13-newton.svg")


# --------------------------------------------- closed loop on the flip
_FLIP = {}


def flip_runs():
    if not _FLIP:
        sc = V.scenario_flip()
        _FLIP.update(sc=sc, **{w: V.run_scenario(sc, w) for w in ("mpc", "lqr", "euler")})
    return _FLIP


def tracking():
    d = flip_runs()
    sc, h = d["sc"], V.H_MPC
    fig, (ax0, ax1) = new_plot(6.6, 3.9, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.12, top=0.97, hspace=0.14)
    t = d["mpc"]["t"]
    ax0.plot(t, d["lqr"]["angle"], color=TERRACOTTA, lw=1.6)
    ax0.plot(t, d["euler"]["angle"], color=TEAL, lw=1.3)
    ax0.plot(t, d["mpc"]["angle"], color=INDIGO, lw=2.0)
    ax0.text(0.93, 39, "LQR, torque clipped afterwards", color=TERRACOTTA, fontsize=9.5, path_effects=halo(), **SANS)
    ax0.text(1.5, 12.5, "MPC in Euler angles", color=TEAL, fontsize=9.5, path_effects=halo(), **SANS)
    ax0.text(0.50, 8.5, "error-state MPC", color=INDIGO, fontsize=9.5, path_effects=halo(), **SANS)
    ax0.set_ylim(0, 50)
    ax0.set_ylabel("attitude error (degrees)", fontsize=9.5, **SANS)
    tm = t[:-1] + 0.5 * h
    ff = np.array([sc["ref"](x)[2][1] for x in tm])
    for sgn in (1, -1):
        ax1.axhline(sgn * sc["tau_max"], color=INK, lw=0.8, ls=DASH)
    ax1.fill_between(tm, sc["tau_max"], np.maximum(ff, sc["tau_max"]), color=PEACH, alpha=0.8, lw=0)
    ax1.fill_between(tm, -sc["tau_max"], np.minimum(ff, -sc["tau_max"]), color=PEACH, alpha=0.8, lw=0)
    ax1.plot(tm, ff, color=MUTED, lw=1.0)
    ax1.step(tm, d["lqr"]["tau"][:, 1], where="mid", color=TERRACOTTA, lw=1.3)
    ax1.step(tm, d["mpc"]["tau"][:, 1], where="mid", color=INDIGO, lw=1.8)
    ax1.text(1.62, 0.66, "bound", fontsize=9.5, **SANS)
    ax1.text(0.335, 0.86, "the reference asks for more", color=MUTED, fontsize=9.5, ha="center", **SANS)
    ax1.set_ylim(-0.95, 0.98)
    ax1.set_xlim(0, sc["T_end"])
    ax1.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax1.set_ylabel(r"pitch torque $\tau_2$ (N·m)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch13-tracking.svg")


# ---------------------------------------- animation: the receding horizon
def horizon_data():
    d = flip_runs()
    if "plans" in d:
        return d
    sc, h, N = d["sc"], V.H_MPC, V.N_MPC
    ctrl = V.ErrorStateMPC(sc["ref"], sc["T_end"], tau_max=sc["tau_max"])
    run = d["mpc"]
    plans, plan_tau = [], []
    for k in range(45):
        As = [ctrl.AB[k + i][0] for i in range(N)]
        Bs = [ctrl.AB[k + i][1] for i in range(N)]
        P_UU, P_Us, Sx, Su = V.condense(As, Bs, ctrl.Qh, ctrl.Rh, ctrl.Pf)
        s0 = ctrl.error_state(k, run["R"][k], run["Om"][k])
        ff = ctrl.tau_d[k:k + N].ravel()
        U, _ = V.box_qp(P_UU, P_Us @ s0, -sc["tau_max"] - ff, sc["tau_max"] - ff)
        pred = (Sx @ s0 + Su @ U).reshape(N, 6)
        axis = [run["R"][k] @ E3] + [sc["ref"]((k + i + 1) * h)[0] @ V.expm_so3(pred[i, :3]) @ E3 for i in range(N)]
        plans.append(np.array(axis))
        plan_tau.append((ff + U).reshape(N, 3)[:, 1])
    d.update(plans=plans, plan_tau=plan_tau)
    return d


def horizon_frame(t):
    d = horizon_data()
    sc, h, N = d["sc"], V.H_MPC, V.N_MPC
    run = d["mpc"]
    kmax = 40
    u = ease(phase(t, 0.04, 0.96)) * kmax
    k = min(int(u), kmax)
    s = Scene(width=6.6, height=3.5, azim=-28, elev=14, xlim=(-1.25, 3.55), ylim=(-1.38, 1.22))
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    ref_axis = np.array([sc["ref"](x)[0] @ E3 for x in np.linspace(0, 1.5, 400)])
    s.sphere_curve(ref_axis, color=MUTED, lw=0.9, linestyle=DASH)
    past = run["R"][:k + 1, :, 2]
    if k >= 1:
        s.sphere_curve(1.002 * past, color=INDIGO, lw=2.3, zorder=31, show_hidden=False)
    s.sphere_curve(1.002 * d["plans"][k], color=TEAL, lw=1.4, zorder=30, show_hidden=False)
    s.dot(1.003 * d["plans"][k][-1], color=TEAL, size=3.2)
    qd = sc["ref"](k * h)[0] @ E3
    s.dot(1.003 * qd, color=MUTED, size=5.5, hollow=True)
    s.dot(1.004 * past[-1], color=INDIGO, size=6)
    s.label(qd, "reference", dx=-12, dy=3, size=9.5, ha="right", color=MUTED, **SANS)
    s.label(past[-1], r"$R\,e_3$", dx=13, dy=-6, color=INDIGO, ha="left")
    # --- the torque strip, drawn in canvas coordinates
    x0, x1, y0, yr = 1.62, 3.42, -0.05, 0.62            # y = y0 + yr * tau
    tx = lambda tt: x0 + (x1 - x0) * np.asarray(tt) / 1.32                 # noqa: E731
    ty = lambda v: y0 + yr * np.asarray(v)                                 # noqa: E731
    ax = s.ax
    tm = (np.arange(66) + 0.5) * h
    ff = np.array([sc["ref"](x)[2][1] for x in tm])
    ax.fill_between(tx(tm), ty(sc["tau_max"]), ty(np.maximum(ff, sc["tau_max"])), color=PEACH, alpha=0.85, lw=0, zorder=40)
    ax.fill_between(tx(tm), ty(-sc["tau_max"]), ty(np.minimum(ff, -sc["tau_max"])), color=PEACH, alpha=0.85, lw=0, zorder=40)
    for sgn in (1, -1):
        ax.plot([x0, x1], [ty(sgn * sc["tau_max"])] * 2, color=INK, lw=0.8, ls=DASH, zorder=41)
    ax.plot([x0, x1], [ty(0)] * 2, color=INK, lw=0.5, alpha=0.5, zorder=41)
    ax.plot([x0, x0], [ty(-1.0), ty(1.0)], color=INK, lw=0.8, zorder=41)
    ax.plot(tx(tm), ty(ff), color=MUTED, lw=1.0, zorder=42)
    tk = (k + np.arange(N) + 0.5) * h
    ax.step(tx(tk), ty(d["plan_tau"][k]), where="mid", color=TEAL, lw=1.3, zorder=43)
    if k >= 1:
        ax.step(tx(tm[:k]), ty(run["tau"][:k, 1]), where="mid", color=INDIGO, lw=2.0, zorder=44)
    ax.plot([tx(k * h)] * 2, [ty(-0.95), ty(0.95)], color=INDIGO, lw=0.7, zorder=42)
    s.text2d(x1 - 0.02, ty(sc["tau_max"]) + 0.075, "bound", fontsize=9, ha="right", **SANS)
    s.text2d(x0 + 0.05, ty(1.0) + 0.02, r"pitch torque $\tau_2$", fontsize=10, ha="left")
    s.text2d(tx(1.18), ty(-0.93), "asked", fontsize=9, color=MUTED, ha="center", **SANS)
    s.text2d(0.5 * (x0 + x1), ty(-1.0) - 0.07, "time", fontsize=9, ha="center", **SANS)
    if t < 0.04:
        msg = "start 20° from the reference"
    elif t > 0.96:
        msg = "the torque never left its bounds"
    else:
        msg = f"t = {k * h:.2f} s: plan {N} steps ahead (teal), apply the first torque, plan again"
    s.text2d(1.15, -1.25, msg, fontsize=10, color=INDIGO, **SANS)
    return s


def horizon():
    animate(horizon_frame, OUT / "ch13-horizon.gif", OUT / "ch13-horizon-strip.png", n=88, fps=11,
            width_px=760, stills=(0.0, 0.42, 0.6, 1.0), strip_px=900, strip_cols=2)


# ------------------------- animation: interior-point method around a cone
_IPM = {}


def ipm_data():
    if not _IPM:
        prob = V.Reorient()
        snaps = []
        sol = V.interior_point(prob, prob.initial_guess(0.0),
                               log=lambda it, x, mu: snaps.append((x["R"][:, :, 0].copy(), mu)))
        over = V.interior_point(prob, prob.initial_guess(-0.3))
        _IPM.update(prob=prob, sol=sol, snaps=snaps, over=over)
    return _IPM


def cone_scene(xlim=None, ylim=None, width=6.4, height=3.7):
    d = ipm_data()
    prob = d["prob"]
    s = Scene(width=width, height=height, azim=50, elev=27, xlim=xlim, ylim=ylim)
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.5, alpha=0.35)
    cap = small_circle(prob.q_c, np.radians(prob.theta_max_deg), 200)
    xy = s.project(1.001 * cap)
    from matplotlib.patches import Polygon
    s.ax.add_patch(Polygon(xy, closed=True, fc=PEACH, ec="none", alpha=0.9, zorder=13))
    s.sphere_curve(1.001 * cap, color=INK, lw=1.7, zorder=16)
    s.arrow(prob.q_c, 1.42 * prob.q_c, color=INK, zorder=32)
    s.label(1.42 * prob.q_c, r"$q_c$", dx=10, dy=4)
    a, b = prob.R0 @ E1, prob.RN @ E1
    s.dot(1.003 * a, color=INK, size=5)
    s.dot(1.003 * b, color=INK, size=5)
    s.label(a, "start", dx=-4, dy=-13, size=9.5, **SANS)
    s.label(b, "target", dx=6, dy=-13, size=9.5, **SANS)
    return s, prob


def draw_path(s, pts, color, lw, dots=True, zorder=30, **kw):
    s.sphere_curve(1.003 * np.vstack([arc(pts[i], pts[i + 1], 8) for i in range(len(pts) - 1)]), color=color, lw=lw,
                   zorder=zorder, **kw)
    if dots:
        for p in pts[1:-1]:
            if s.point_visible(p):
                s.dot(1.004 * p, color=color, size=2.6, zorder=zorder + 1)


CONE_LIM = dict(xlim=(-1.3, 1.3), ylim=(-1.4, 1.3))


def cone_frame(t):
    d = ipm_data()
    snaps, hist = d["snaps"], d["sol"]["hist"]
    s, prob = cone_scene(**CONE_LIM, width=6.0, height=6.0 * 2.7 / 2.6)
    n = len(snaps) - 1
    u = ease(phase(t, 0.10, 0.92)) * n
    i = min(int(u), n - 1) if n > 0 else 0
    f = u - i
    A, B = snaps[i][0], snaps[min(i + 1, n)][0]
    pts = np.array([unit((1 - f) * a + f * b) for a, b in zip(A, B)])
    draw_path(s, snaps[0][0], MUTED, 0.9, dots=False, zorder=28)
    inside = pts @ prob.q_c > prob.cos_theta_max + 1e-9
    col = INDIGO if t >= 0.92 else TEAL
    draw_path(s, pts, col, 2.0)
    for p in pts[inside]:
        s.dot(1.004 * p, color=TERRACOTTA, size=3.4, zorder=33)
    it = min(int(round(u)), n)
    if t < 0.10:
        msg = "first guess: the shortest turn, straight through the cone"
    elif t >= 0.92:
        msg = f"iteration {n}: minimum effort, grazing the cone"
    else:
        msg = f"iteration {it}:  $\\mu$ = {snaps[it][1]:.0e},  residual {hist[it]['E0']:.0e}"
    s.text2d(0.0, -1.3, msg, fontsize=10, color=INDIGO if t >= 0.92 else INK, **SANS)
    s.text2d(0.0, 1.2, r"the axis $R_k e_1$ at every step of the horizon", fontsize=10.5)
    return s


def cone():
    animate(cone_frame, OUT / "ch13-cone.gif", OUT / "ch13-cone-strip.png", n=84, fps=10,
            width_px=620, stills=(0.0, 0.3, 0.55, 1.0), strip_px=760, strip_cols=2)


# ----------------------------------- the interior-point method in numbers
def ipm():
    d = ipm_data()
    prob, sol = d["prob"], d["sol"]
    hist = sol["hist"]
    fig, (ax0, ax1) = new_plot(6.6, 3.0, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.16, top=0.93, wspace=0.22)
    it = np.arange(len(hist))
    ax0.semilogy(it, [hh["mu"] for hh in hist], drawstyle="steps-post", color=TERRACOTTA, lw=1.4)
    ax0.semilogy(it, [hh["E0"] for hh in hist], "-o", color=INDIGO, lw=1.9, ms=3.6)
    ax0.text(8.3, 1.2e-1, "KKT residual", color=INDIGO, fontsize=9.5, path_effects=halo(), **SANS)
    ax0.text(0.4, 3e-7, r"barrier parameter $\mu$", color=TERRACOTTA, fontsize=10, path_effects=halo())
    ax0.set_xlabel("iteration", fontsize=9.5, **SANS)
    ax0.set_ylim(3e-10, 5)
    ax0.set_yticks([1, 1e-3, 1e-6, 1e-9])
    ax0.minorticks_off()
    # the reduced KKT matrix, ordered by stage
    x, nu_r, nu_c, w = sol["x"], sol["nu_r"], sol["nu_c"], sol["w"]
    Ac, Ar = prob.ineq_jac(x), prob.eq_jac(x)
    K = np.block([[prob.lag_hess(x, nu_r, nu_c) + Ac.T @ ((nu_c / w)[:, None] * Ac), Ar.T], [Ar, np.zeros((prob.n_eq,) * 2)]])
    order = V.stage_order(prob)
    P = np.abs(K[np.ix_(order, order)]) > 1e-10
    from matplotlib.colors import ListedColormap
    ax1.imshow(P, cmap=ListedColormap([PAPER, INK]), interpolation="nearest")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for side in ("top", "right"):
        ax1.spines[side].set_visible(True)
        ax1.spines[side].set_linewidth(0.8)
    ax1.set_xlabel(f"{K.shape[0]} unknowns, ordered by stage", fontsize=9.5, **SANS)
    ax1.xaxis.set_label_position("bottom")
    save(fig, OUT / "ch13-ipm.svg")


if __name__ == "__main__":
    which = sys.argv[1:] or ["opening", "newton", "tracking", "horizon", "cone", "ipm"]
    for name in which:
        globals()[name]()
