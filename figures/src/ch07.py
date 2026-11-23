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
from figkit import (INDIGO, INK, MUTED, OCHRE, PAPER, PEACH, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, geodesic, great_circle, halo, new_plot, rot, save, small_circle, unit)

_spec = importlib.util.spec_from_file_location("ch07_verify", ROOT / "verify" / "ch07.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
SANS = dict(fontfamily="Source Sans 3")


def on_sphere(scene, p):
    return scene.point_visible(p)


# ------------------------------------------------- variation on the sphere
def swing_pair(T, eps):
    """Free swing, a neighbor started `eps` away, and the linearized flow between them."""
    q0, w0 = V.s2_swing()
    axis = unit(np.cross(q0, [0.3, 1.0, 0.2]))
    free = lambda t, q, w: np.zeros(3)
    ref = V.s2_sim(q0, w0, free, T)
    qe, we, s0 = V.s2_neighbour(q0, w0, axis, eps)
    return ref, V.s2_sim(qe, we, free, T), V.s2_linear_flow(ref, s0, T)


def variation():
    s = Scene(width=6.4, height=3.6, azim=-112, elev=17)
    eps = np.radians(20.0)
    ref, nb, lin = swing_pair(1.8, eps)
    tt = np.linspace(0.78, 1.74, 200)
    P, Pn = ref.sol(tt)[:3].T, nb.sol(tt)[:3].T
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.5, alpha=0.4)
    s.sphere_curve(P, color=INDIGO, lw=2.1, zorder=16)
    s.sphere_curve(Pn, color=TERRACOTTA, lw=1.6, zorder=16)
    tk = 1.02                                         # the linear state at one instant
    for t in (0.84, tk, 1.3, 1.5, 1.68):
        a, b = ref.sol(t)[:3], nb.sol(t)[:3]
        if t != tk:
            s.sphere_curve(geodesic(a, b, 30), color=MUTED, lw=0.8, zorder=15)
        s.dot(a, color=INDIGO, size=3.6)
        s.dot(b, color=TERRACOTTA, size=3.6)
    qd, qn = ref.sol(tk)[:3], nb.sol(tk)[:3]
    xi = eps * lin.sol(tk)[:3]
    dq = np.cross(xi, qd)
    a1 = unit(xi)
    a2 = np.cross(qd, a1)
    c = 1.004 * qd
    s.patch([c + 0.5 * a1 + 0.42 * a2, c - 0.2 * a1 + 0.42 * a2, c - 0.2 * a1 - 0.62 * a2, c + 0.5 * a1 - 0.62 * a2],
            color=SAND, alpha=0.82)
    s.arrow(qd, qd + dq, color=TERRACOTTA, zorder=32)
    s.arrow(qd, qd + xi, color=TEAL, zorder=32)
    s.dot(qd, color=INDIGO, size=4.6)
    s.dot(qn, color=TERRACOTTA, size=4.6)
    s.label(qd + xi, r"$\xi$", dx=-10, dy=4, color=TEAL, size=13)
    s.label(qd + 0.5 * dq, r"$\delta q = \xi \times q_d$", dx=20, dy=-16, color=TERRACOTTA)
    s.label(qd, r"$q_d(t)$", dx=-22, dy=-6, color=INDIGO)
    s.label(qn, r"$q(t)$", dx=19, dy=9, color=TERRACOTTA)
    s.label(P[-8], "reference", dx=31, dy=-2, size=9.5, color=INDIGO, **SANS)
    s.label(Pn[-8], "neighbor", dx=-32, dy=-6, size=9.5, color=TERRACOTTA, **SANS)
    s.label(c + 0.5 * a1 + 0.42 * a2, r"$T_{q_d}S^2$", dx=-4, dy=10, size=10.5)
    s.text2d(2.1, 0.35, "the tangent plane rides\nalong with the reference", fontsize=9.5, color=MUTED,
             linespacing=1.35, **SANS)
    s.text2d(-2.05, -0.35, r"$\xi \cdot q_d = 0$", fontsize=11.5, color=TEAL)
    s.save(OUT / "ch07-variation.svg", pad=0.08)


# ---------------------------------- two tangent planes: the second constraint
def two_planes():
    s = Scene(width=6.4, height=3.6, azim=-70, elev=27)
    right = s.right
    fwd = unit(np.array([s.view[0], s.view[1], 0.0]))  # horizontal, toward the viewer
    qd = E3
    ang = np.radians(58)
    q = np.cos(ang) * E3 + np.sin(ang) * unit(right + 0.45 * fwd)
    ax = unit(np.cross(qd, q))
    along = np.cross(ax, qd)                          # unit tangent at q_d pointing toward q
    wd = 0.66 * unit(-along + 0.55 * ax)
    w = rot(ax, ang) @ wd                             # the same spin, carried along the great circle to q
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(geodesic(qd, q), color=INK, lw=0.8, alpha=0.7)
    c = 1.004 * qd
    s.patch([c + 0.5 * along + 0.55 * ax, c - 0.8 * along + 0.55 * ax, c - 0.8 * along - 0.55 * ax, c + 0.5 * along - 0.55 * ax],
            color=SAND, alpha=0.85)
    s.tangent_plane(1.004 * q, size=0.4, color=PEACH, alpha=0.72, spin=0.15)
    s.arrow(qd, qd + wd, color=INDIGO, zorder=33)
    s.arrow(q, q + w, color=TERRACOTTA, zorder=33)
    s.arrow(qd, qd + w, color=TERRACOTTA, lw=1.1, dashed=True, zorder=32)      # omega redrawn at q_d
    foot = qd + w - (w @ qd) * qd
    s.curve(np.array([foot, qd + w]), color=TEAL, lw=1.7, zorder=34, alpha=1.0)
    s.curve(np.array([qd, foot]), color=MUTED, lw=0.7, hidden=True, zorder=31)
    s.arrow(qd + wd, qd + w, color=INK, lw=1.2, head=0.09, zorder=33)
    s.dot(qd, color=INK, size=4.5)
    s.dot(q, color=INK, size=4.5)
    s.label(qd, r"$q_d$", dx=9, dy=-10, size=12)
    s.label(q, r"$q$", dx=-10, dy=-9, size=12)
    s.label(qd + wd, r"$\omega_d$", dx=-14, dy=-2, color=INDIGO, size=12)
    s.label(q + w, r"$\omega$", dx=11, dy=5, color=TERRACOTTA, size=12)
    s.label(qd + 0.5 * (wd + w), r"$\delta\omega$", dx=-13, dy=7, size=12)
    s.label(0.5 * (foot + qd + w), r"$q_d\cdot\delta\omega$", dx=27, dy=2, color=TEAL)
    s.text2d(-2.1, 0.55, r"$\omega_d$ lies in the" + "\n" + r"tangent plane at $q_d$", fontsize=9.5, color=INDIGO,
             linespacing=1.3, **SANS)
    s.text2d(2.2, -0.45, r"$\omega$ lies in the" + "\n" + r"tangent plane at $q$", fontsize=9.5, color=TERRACOTTA,
             linespacing=1.3, **SANS)
    s.text2d(1.85, 1.3, r"dashed: $\omega$ redrawn at $q_d$", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.save(OUT / "ch07-two-planes.svg", pad=0.08)


# --------------------------------- nonlinear error against linear prediction
def linear_vs_nonlinear():
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.94, wspace=0.3)
    T = 1.2
    tt = np.linspace(0, T, 300)
    ref, _, lin = swing_pair(T, 0.0)
    ax0.plot(tt, np.linalg.norm(lin.sol(tt)[:3].T, axis=1), color="#b9bcc6", lw=5.5, zorder=5, solid_capstyle="round")
    for deg, col, off in ((1, INDIGO, 9), (10, TEAL, 0), (30, TERRACOTTA, -8)):
        nb = swing_pair(T, np.radians(deg))[1]
        y = np.linalg.norm(np.cross(ref.sol(tt)[:3].T, nb.sol(tt)[:3].T), axis=1) / np.radians(deg)
        ax0.plot(tt, y, color=col, lw=1.5, zorder=6, ls="-")
        ax0.annotate(f"{deg}°", (tt[-1], y[-1]), xytext=(5, off), textcoords="offset points", color=col, fontsize=9.5,
                     va="center", **SANS)
    ax0.text(0.04, 1.47, r"gray band: linear prediction $\|\xi(t)\|$", fontsize=10.5, color=MUTED, path_effects=halo())
    ax0.text(0.04, 0.1, r"lines: simulated $\|e_q(t)\|$ / initial angle", fontsize=10.5, path_effects=halo())
    ax0.set_xlim(0, T * 1.1)
    ax0.set_ylim(0, 1.6)
    ax0.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax0.set_ylabel("error per unit of initial error", fontsize=9.5, **SANS)

    degs = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])
    rel = []
    for d in degs:
        e = np.radians(d)
        eq = np.cross(ref.sol(T)[:3], swing_pair(T, e)[1].sol(T)[:3])
        rel.append(np.linalg.norm(eq - e * lin.sol(T)[:3]) / np.linalg.norm(e * lin.sol(T)[:3]))
    ax1.loglog(degs, rel[0] * degs / degs[0], color=MUTED, lw=0.9, ls=(0, (3.5, 3)))
    ax1.loglog(degs, rel, "o", color=TERRACOTTA, ms=4.5)
    ax1.text(1.5, 0.011, "slope 1", color=MUTED, fontsize=9.5, **SANS)
    ax1.set_xlabel("initial angle between $q$ and $q_d$ (degrees)", fontsize=9.5, **SANS)
    ax1.set_ylabel("relative error of the prediction", fontsize=9.5, **SANS)
    ax1.set_xticks([0.25, 1, 4, 16])
    ax1.set_xticklabels(["0.25", "1", "4", "16"])
    ax1.set_yticks([0.01, 0.1])
    ax1.set_yticklabels(["1%", "10%"])
    ax1.minorticks_off()
    save(fig, OUT / "ch07-linear-vs-nonlinear.svg")


# ---------------------------- holding a nose-up attitude: geometric vs Euler
def hold_figure():
    """Left: path of one body axis on the sphere. Right: error angle against time."""
    from figkit import style_axes
    h = V.hold_compare(85.0, 20.0, (1, 1, 1))
    s = Scene(width=6.6, height=3.3, azim=58, elev=30, xlim=(-1.3, 4.25), ylim=(-1.52, 1.2))
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    pe, pg = h["euler"]["R"][:, :, 1], h["geometric"]["R"][:, :, 1]
    s.sphere_curve(pe[:420], color=TERRACOTTA, lw=1.7, zorder=16)
    s.sphere_curve(pg, color=INDIGO, lw=3.0, zorder=17)
    far = [k for k in range(400) if s.point_visible(pe[k]) and pe[k] @ h["Rd"][:, 1] < 0.6]
    for k in (far[len(far) // 5], far[(3 * len(far)) // 5], far[-3]):
        step = unit(pe[k + 1] - pe[k])
        s.arrow(1.004 * pe[k] - 0.07 * step, 1.004 * pe[k] + 0.07 * step, color=TERRACOTTA, lw=1.2, head=0.11)
    target = h["Rd"][:, 1]
    s.dot(target, color=INK, size=5)
    s.dot(pg[0], color=INK, size=4.2, hollow=True)
    s.label(target, "target", dx=4, dy=-12, size=9.5, **SANS)
    s.label(pg[0], "start", dx=-4, dy=11, size=9.5, **SANS)
    s.text2d(0.0, -1.38, "path of the body axis $Re_2$", fontsize=9.5, color=MUTED, **SANS)
    ax = s.fig.add_axes([0.555, 0.2, 0.425, 0.72])
    style_axes(ax)
    for name, col, lw in (("euler", TERRACOTTA, 1.7), ("geometric", INDIGO, 2.0)):
        ax.plot(h[name]["t"], h[name]["angle"], color=col, lw=lw)
    ax.text(0.5, 150, "Euler-angle LQR", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax.text(0.3, 32, "geometric LQR", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 185)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("attitude error angle (degrees)", fontsize=9.5, **SANS)
    s.save(OUT / "ch07-hold.svg", pad=0.05)


# -------------------------------------- gain history along the flip reference
def gain_history():
    ref = V.Flip()
    lq = V.flip_lqr(ref)
    fe = V.flip_euler_lqr(ref, lq["P_inf"])
    tg, eul, tsE, KsE = fe["tg"], fe["euler"], fe["ts"], fe["Ks"]

    fig, (ax0, ax1) = new_plot(6.4, 3.5, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1.7]})
    fig.subplots_adjust(left=0.1, right=0.975, bottom=0.14, top=0.96, hspace=0.16)
    pitch = np.degrees(eul[:, 1])
    ax0.plot(tg, pitch, color=INK, lw=1.5)
    for sgn in (1, -1):
        ax0.axhline(sgn * 90, color=TERRACOTTA, lw=0.8, ls=(0, (3.5, 3)))
    ax0.text(0.02, 68, "chart singularity at ±90°", color=TERRACOTTA, fontsize=9.5, **SANS)
    ax0.text(0.86, 20, "reference pitch", fontsize=10.5, path_effects=halo())
    ax0.set_ylim(-100, 100)
    ax0.set_yticks([-90, 0, 90])
    ax0.set_ylabel("degrees", fontsize=9.5, **SANS)
    ax1.semilogy(tsE, np.linalg.norm(KsE, axis=(1, 2)), color=TERRACOTTA, lw=1.7)
    ax1.semilogy(lq["ts"], np.linalg.norm(lq["Ks"], axis=(1, 2)), color=INDIGO, lw=2.0)
    ax1.text(0.57, 22, "gain designed in Euler angles", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax1.text(0.62, 2.55, r"gain designed in $(\eta, \delta\Omega)$", color=INDIGO, fontsize=10.5, path_effects=halo())
    ax1.set_ylim(2, 30)
    ax1.set_yticks([2, 5, 10, 20])
    ax1.set_yticklabels(["2", "5", "10", "20"])
    ax1.minorticks_off()
    ax1.set_xlim(0, ref.T)
    ax1.set_xlabel("time along the flip (s)", fontsize=9.5, **SANS)
    ax1.set_ylabel(r"$\|K(t)\|$", fontsize=10.5)
    save(fig, OUT / "ch07-gain-history.svg")


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
    ax0.text(47, 0.95, "log error", color=TEAL, fontsize=10.5, path_effects=halo())
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
    ax1.text(120, 0.72, "log error", color=TEAL, fontsize=10.5, path_effects=halo())
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 4)
    ax1.set_xticks([0, 45, 90, 135, 180])
    ax1.set_title("attitude hold, torque in the body frame", fontsize=9.5, loc="left", **SANS)
    ax1.set_xlabel("initial error angle (degrees)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch07-large-errors.svg")


if __name__ == "__main__":
    variation()
    two_planes()
    linear_vs_nonlinear()
    hold_figure()
    gain_history()
    large_errors()
