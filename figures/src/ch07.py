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
def variation():
    s = Scene(width=6.4, height=3.7, azim=-62, elev=14)
    q0, w0 = V.s2_swing()
    T = 0.95
    ref = V.s2_sim(q0, w0, lambda t, q, w: np.zeros(3), T)
    axis = unit(np.cross(q0, [0.3, 1.0, 0.2]))
    eps = np.radians(16.0)
    qe = V.expm_so3(eps * axis) @ q0
    we = w0 - (w0 @ qe) * qe
    nb = V.s2_sim(qe, we, lambda t, q, w: np.zeros(3), T)
    lin = V.s2_linear_flow(ref, np.r_[axis, np.zeros(3)], T)
    tt = np.linspace(0, T, 200)
    P, Pn = ref.sol(tt)[:3].T, nb.sol(tt)[:3].T

    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.5, alpha=0.4)
    s.sphere_curve(P, color=INDIGO, lw=2.0, zorder=16)
    s.sphere_curve(Pn, color=TERRACOTTA, lw=1.6, zorder=16)
    for t in (0.0, 0.3, 0.62, 0.95):
        a, b = ref.sol(t)[:3], nb.sol(t)[:3]
        s.sphere_curve(geodesic(a, b, 30), color=TEAL, lw=0.9, zorder=15)
        s.dot(a, color=INDIGO, size=4)
        s.dot(b, color=TERRACOTTA, size=4)
    # the linearized state at one instant, drawn in the tangent plane at q_d
    tk = 0.62
    qd, qn = ref.sol(tk)[:3], nb.sol(tk)[:3]
    xi = eps * lin.sol(tk)[:3]
    dq = np.cross(xi, qd)
    spin = np.arctan2(*(np.array([unit(dq) @ c for c in (np.cross(qd, unit(np.cross(qd, E3))), unit(np.cross(qd, E3)))])))
    s.tangent_plane(1.003 * qd, size=0.46, color=SAND, alpha=0.82, spin=spin)
    scale = 1.0
    s.arrow(qd, qd + scale * dq, color=TERRACOTTA, zorder=32)
    s.arrow(qd, qd + scale * xi, color=TEAL, zorder=32)
    s.label(qd + scale * xi, r"$\xi$", dx=9, dy=6, color=TEAL, size=12)
    s.label(qd + scale * dq, r"$\delta q = \xi \times q_d$", dx=-40, dy=9, color=TERRACOTTA)
    s.label(qd, r"$q_d(t)$", dx=-20, dy=-9, color=INDIGO)
    s.label(qn, r"$q(t)$", dx=-13, dy=11, color=TERRACOTTA)
    s.label(ref.sol(0.0)[:3], "reference", dx=34, dy=4, size=9.5, color=INDIGO, **SANS)
    s.label(nb.sol(0.0)[:3], "neighbour", dx=-30, dy=6, size=9.5, color=TERRACOTTA, **SANS)
    s.text2d(1.95, 0.55, "the linear state lives in\nthe tangent plane at $q_d(t)$,\nwhich moves with the reference",
             fontsize=9.5, color=MUTED, linespacing=1.35, **SANS)
    s.text2d(-1.95, -0.55, r"$\xi \cdot q_d = 0$", fontsize=11, color=TEAL)
    s.save(OUT / "ch07-variation.svg", pad=0.08)


# ---------------------------------- two tangent planes: the second constraint
def two_planes():
    s = Scene(width=6.4, height=3.6, azim=-48, elev=20)
    qd = unit([0.05, -0.35, 0.93])
    ax = unit(np.cross(qd, [1.0, 0.25, 0.0]))
    q = rot(ax, np.radians(48)) @ qd
    wd = 0.62 * unit(np.cross(qd, [0.2, 1.0, 0.3]))
    w = rot(ax, np.radians(48)) @ wd                  # the same spin, carried to the neighbour
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(geodesic(qd, q), color=INK, lw=0.7, alpha=0.6)
    s.tangent_plane(1.003 * qd, size=0.5, color=SAND, alpha=0.8)
    s.tangent_plane(1.003 * q, size=0.42, color=PEACH, alpha=0.7)
    s.arrow(qd, qd + wd, color=INDIGO, zorder=33)
    s.arrow(q, q + w, color=TERRACOTTA, zorder=33)
    # omega carried over to q_d: it pokes out of the plane there
    s.arrow(qd, qd + w, color=TERRACOTTA, lw=1.1, dashed=True, zorder=32)
    normal = (w @ qd) * qd
    s.curve(np.array([qd + w - normal, qd + w]), color=TEAL, lw=1.4, zorder=34, alpha=1.0)
    s.curve(np.array([qd, qd + w - normal]), color=MUTED, lw=0.7, hidden=True, zorder=31)
    s.arrow(qd + wd, qd + w, color=INK, lw=1.1, head=0.08, zorder=33)
    s.dot(qd, color=INK, size=4.5)
    s.dot(q, color=INK, size=4.5)
    s.label(qd, r"$q_d$", dx=-12, dy=-8, size=12)
    s.label(q, r"$q$", dx=10, dy=-9, size=12)
    s.label(qd + wd, r"$\omega_d$", dx=-13, dy=5, color=INDIGO, size=12)
    s.label(q + w, r"$\omega$", dx=11, dy=3, color=TERRACOTTA, size=12)
    s.label(qd + 0.5 * (wd + w), r"$\delta\omega$", dx=2, dy=12, size=12)
    s.label(qd + w - 0.5 * normal, r"$q_d\!\cdot\delta\omega \neq 0$", dx=40, dy=-2, color=TEAL)
    s.text2d(-1.95, 0.75, r"$\omega_d$ is tangent at $q_d$", fontsize=9.5, color=INDIGO, **SANS)
    s.text2d(2.05, -0.55, r"$\omega$ is tangent at $q$," + "\na different plane", fontsize=9.5, color=TERRACOTTA,
             linespacing=1.3, **SANS)
    s.save(OUT / "ch07-two-planes.svg", pad=0.08)


# --------------------------------- nonlinear error against linear prediction
def linear_vs_nonlinear():
    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.94, wspace=0.3)
    q0, w0 = V.s2_swing()
    T = 1.2
    ref = V.s2_sim(q0, w0, lambda t, q, w: np.zeros(3), T)
    axis = unit(np.cross(q0, [0.3, 1.0, 0.2]))
    lin = V.s2_linear_flow(ref, np.r_[axis, np.zeros(3)], T)
    tt = np.linspace(0, T, 300)
    xi = lin.sol(tt)[:3].T

    def neighbour(e):
        qe = V.expm_so3(e * axis) @ q0
        return V.s2_sim(qe, w0 - (w0 @ qe) * qe, lambda t, q, w: np.zeros(3), T)

    ax0.plot(tt, np.linalg.norm(xi, axis=1), color=INK, lw=2.4, zorder=5)
    for deg, col in ((1, INDIGO), (10, TEAL), (30, TERRACOTTA)):
        nb = neighbour(np.radians(deg))
        eq = np.cross(ref.sol(tt)[:3].T, nb.sol(tt)[:3].T)
        y = np.linalg.norm(eq, axis=1) / np.radians(deg)
        ax0.plot(tt, y, color=col, lw=1.5, zorder=6, ls=(0, (4, 2)) if deg == 1 else "-")
        ax0.annotate(f"{deg}°", (tt[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=col, fontsize=9.5,
                     va="center", **SANS)
    ax0.text(0.05, 1.42, r"linear prediction $\|\xi(t)\|$", fontsize=10.5, path_effects=halo())
    ax0.text(0.05, 0.12, r"simulated $\|e_q(t)\|\,/\,$initial angle", fontsize=10.5, color=MUTED, path_effects=halo())
    ax0.set_xlim(0, T * 1.09)
    ax0.set_ylim(0, 1.6)
    ax0.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax0.set_ylabel("error per unit of initial error", fontsize=9.5, **SANS)

    degs = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])
    rel = []
    for d in degs:
        e = np.radians(d)
        eq = np.cross(ref.sol(T)[:3], neighbour(e).sol(T)[:3])
        rel.append(np.linalg.norm(eq - e * lin.sol(T)[:3]) / np.linalg.norm(e * lin.sol(T)[:3]))
    ax1.loglog(degs, rel, "o-", color=TERRACOTTA, lw=1.6, ms=4)
    ax1.loglog(degs, rel[0] * degs / degs[0], color=MUTED, lw=0.9, ls=(0, (3.5, 3)))
    ax1.text(1.0, 0.012, "slope 1", color=MUTED, fontsize=9.5, **SANS)
    ax1.set_xlabel("initial angle between q and $q_d$ (degrees)", fontsize=9.5, **SANS)
    ax1.set_ylabel("relative error of the prediction", fontsize=9.5, **SANS)
    ax1.set_xticks([0.25, 1, 4, 16])
    ax1.set_xticklabels(["0.25", "1", "4", "16"])
    ax1.minorticks_off()
    save(fig, OUT / "ch07-linear-vs-nonlinear.svg")


# ---------------------------- holding a nose-up attitude: geometric vs Euler
def hold():
    h = V.hold_compare(85.0, 20.0, (1, 1, 1))
    s = Scene(width=6.6, height=3.5, azim=-24, elev=24, xlim=(-1.45, 4.2))
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.5, alpha=0.4)
    pe, pg = h["euler"]["R"][:, :, 1], h["geometric"]["R"][:, :, 1]
    s.sphere_curve(pe[:420], color=TERRACOTTA, lw=1.7, zorder=16)
    s.sphere_curve(pg, color=INDIGO, lw=2.6, zorder=17)
    for k in (60, 85):
        if s.point_visible(pe[k]):
            s.arrow(1.004 * pe[k], 1.004 * pe[k + 4], color=TERRACOTTA, lw=1.2, head=0.1)
    target = h["Rd"][:, 1]
    s.dot(target, color=INK, size=5)
    s.dot(pg[0], color=INK, size=4, hollow=True)
    s.label(target, "target", dx=-22, dy=-9, size=9.5, **SANS)
    s.label(pg[0], "start", dx=16, dy=9, size=9.5, **SANS)
    s.label(pe[150], "Euler-angle LQR", dx=-4, dy=-13, size=9.5, color=TERRACOTTA, **SANS)
    s.label(pg[8], "geometric LQR", dx=-44, dy=14, size=9.5, color=INDIGO, **SANS)
    s.text2d(0.0, -1.33, "path of the body axis $Re_2$", fontsize=9.5, color=MUTED, **SANS)
    s.save(OUT / "ch07-hold.svg", pad=0.05)
    return h


def hold_figure():
    """Left: the sphere scene. Right: error angle against time, on inset axes."""
    h = V.hold_compare(85.0, 20.0, (1, 1, 1))
    s = Scene(width=6.6, height=3.3, azim=-24, elev=24, xlim=(-1.3, 4.25), ylim=(-1.5, 1.22))
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.5, alpha=0.4)
    pe, pg = h["euler"]["R"][:, :, 1], h["geometric"]["R"][:, :, 1]
    s.sphere_curve(pe[:420], color=TERRACOTTA, lw=1.7, zorder=16)
    s.sphere_curve(pg, color=INDIGO, lw=2.8, zorder=17)
    for k in (58, 84):
        if s.point_visible(pe[k]) and s.point_visible(pe[k + 4]):
            s.arrow(1.004 * pe[k], 1.004 * pe[k + 4], color=TERRACOTTA, lw=1.2, head=0.11)
    target = h["Rd"][:, 1]
    s.dot(target, color=INK, size=5)
    s.dot(pg[0], color=INK, size=4.2, hollow=True)
    s.label(target, "target", dx=-22, dy=-9, size=9.5, **SANS)
    s.label(pg[0], "start", dx=17, dy=8, size=9.5, **SANS)
    s.text2d(0.0, -1.36, "path of the body axis $Re_2$ on the sphere", fontsize=9.5, color=MUTED, **SANS)
    ax = s.fig.add_axes([0.555, 0.2, 0.425, 0.72])
    from figkit import style_axes
    style_axes(ax)
    for name, col, lw in (("euler", TERRACOTTA, 1.7), ("geometric", INDIGO, 2.0)):
        ax.plot(h[name]["t"], h[name]["angle"], color=col, lw=lw)
    ax.text(0.5, 150, "Euler-angle LQR", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax.text(0.33, 30, "geometric LQR", color=INDIGO, fontsize=10.5, path_effects=halo())
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
    tg = np.linspace(0, ref.T, 1501)
    eul = np.array([V.euler_of(ref(t)[0]) for t in tg])
    eul[:, 0], eul[:, 2] = np.unwrap(eul[:, 0]), np.unwrap(eul[:, 2])
    from scipy.interpolate import interp1d
    AE = interp1d(tg, np.array([V.euler_A(np.r_[e, ref(t)[1]], ref(t)[2]) for e, t in zip(eul, tg)]), axis=0)
    tsE, _, KsE = V.riccati_ltv(lambda t: AE(t), lambda t: V.SO3_B, V.Q_ATT, V.R_ATT, lq["P_inf"], ref.T, 1501)

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
    ax1.text(0.62, 13, "gain designed in Euler angles", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
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
