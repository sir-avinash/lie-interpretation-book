"""Figures for Chapter 4.  Run: python figures/src/ch04.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (INDIGO, INK, MUTED, PERIWINKLE, SAND, TEAL, TERRACOTTA,  # noqa: E402
                    Scene, great_circle, halo, hat, new_plot, rot, save, style_axes, unit)

OUT = ROOT / "figures"
E1, E2, E3 = np.eye(3)
G = 9.81
SANS = dict(fontfamily="Source Sans 3")


def tangent(q, v):
    return v - (v @ q) * q


# ------------------------------------------------- Euler mass matrix conditioning
J_P = np.diag([0.20, 0.25, 0.10])
M_P, RHO_P = 1.0, np.array([0.0, 0.0, -0.4])


def E_mat(th, phi):
    return np.array([[-np.sin(th), 0, 1],
                     [np.cos(th) * np.sin(phi), np.cos(phi), 0],
                     [np.cos(th) * np.cos(phi), -np.sin(phi), 0]])


def euler_cond():
    fig, ax = new_plot(6.4, 2.8)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.19, top=0.94)
    th = np.radians(np.linspace(0, 89.95, 900))
    phi = np.radians(20.0)
    cond = np.array([np.linalg.cond(E_mat(t, phi).T @ J_P @ E_mat(t, phi)) for t in th])
    ax.semilogy(np.degrees(th), cond, color=TERRACOTTA, lw=1.8)
    ax.axhline(np.linalg.cond(J_P), color=INDIGO, lw=1.6)
    c89 = np.linalg.cond(E_mat(np.radians(89), phi).T @ J_P @ E_mat(np.radians(89), phi))
    ax.plot([89], [c89], "o", ms=5, color=TERRACOTTA)
    ax.annotate(r"$2.4\times 10^{4}$ at 89°", (89, c89), xytext=(-10, 2), textcoords="offset points", ha="right",
                va="center", fontsize=10, color=TERRACOTTA, path_effects=halo())
    ax.text(20, 60, r"Euler-angle mass matrix $E^{\top}\! J E$", color=TERRACOTTA, fontsize=10.5, path_effects=halo())
    ax.text(20, 0.85, r"body-frame mass matrix $J$: condition number 2.5 at every attitude", color=INDIGO, fontsize=10.5,
            path_effects=halo())
    ax.set_xlim(0, 90)
    ax.set_ylim(0.5, 3e6)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("pitch angle θ (degrees)", fontsize=9.5, **SANS)
    ax.set_ylabel("condition number", fontsize=9.5, **SANS)
    save(fig, OUT / "ch04-euler-cond.svg")


# --------------------------------------------- Hamilton's principle on the sphere
def pendulum_path(q0, w0, T, n=200, l=1.0):
    def rhs(_, s):
        q, w = s[:3], s[3:]
        return np.concatenate([np.cross(w, q), -(G / l) * np.cross(q, E3)])
    sol = solve_ivp(rhs, [0, T], np.concatenate([q0, tangent(q0, w0)]), t_eval=np.linspace(0, T, n), rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[:3].T, sol.y[3:].T


def cam(s, a, b):
    """Unit vector on the camera-facing side of the unit sphere, at canvas position (a, b)."""
    return a * s.right + b * s.up + np.sqrt(1 - a * a - b * b) * s.view


def variation_s2():
    s = Scene(width=6.4, height=3.7, azim=-38, elev=10)
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6, alpha=0.45)
    q0 = cam(s, -0.66, 0.02)
    v0 = 2.5 * unit(tangent(q0, s.right + 0.62 * s.up))
    t, Q, W = pendulum_path(q0, np.cross(q0, v0), 0.60)
    T = t[-1]
    bump = np.sin(np.pi * t / T)
    Qd = np.cross(W, Q)
    xi = np.array([-b * unit(qd) for b, qd in zip(bump, Qd)])          # xi along the path, so delta q is across it
    for eps, alpha in ((0.44, 0.95), (0.22, 0.6), (-0.22, 0.6), (-0.44, 0.6)):
        Qe = np.array([rot(x, eps * np.linalg.norm(x)) @ q if np.linalg.norm(x) > 1e-9 else q for x, q in zip(xi, Q)])
        s.sphere_curve(Qe, color=TEAL, lw=1.1, alpha=alpha, show_hidden=False, zorder=15)
        if eps == 0.44:
            Qtop = Qe
    s.sphere_curve(Q, color=INDIGO, lw=2.2, zorder=17, show_hidden=False)
    for k in (35, 67, 100, 133, 165):
        dq = np.cross(xi[k], Q[k])
        s.arrow(1.003 * Q[k], 1.003 * (Q[k] + 0.42 * dq), color=TERRACOTTA, lw=1.3, head=0.085)
    s.dot(Q[0], color=INK, size=5)
    s.dot(Q[-1], color=INK, size=5)
    s.label(Q[0], r"$q(0)$", dx=-19, dy=-4, size=11.5)
    s.label(Q[-1], r"$q(t_f)$", dx=20, dy=3, size=11.5)
    s.label(Q[150], r"$q(t)$", dx=-14, dy=-10, color=INDIGO, size=11.5)
    s.text2d(-2.0, 0.55, r"$\delta q = \xi \times q$", fontsize=11.5, color=TERRACOTTA)
    s.text2d(-2.0, 0.22, "tangent to the sphere", fontsize=9.5, color=TERRACOTTA, **SANS)
    s.text2d(2.05, 0.62, r"$q_\epsilon(t) = \exp(\epsilon\,\xi^{\wedge})\,q(t)$", fontsize=11.5, color=TEAL)
    s.text2d(2.05, 0.22, "every varied curve\nstays on the sphere", fontsize=9.5, color=TEAL, linespacing=1.3, **SANS)
    s.text2d(2.05, -0.5, "endpoints are\nnot varied", fontsize=9.5, color=MUTED, linespacing=1.3, **SANS)
    s.save(OUT / "ch04-variation-s2.svg", pad=0.1)


# ----------------------------------------- spherical pendulum: forces and projection
def pendulum_forces():
    az = -32.0
    s = Scene(width=6.4, height=3.8, azim=az, elev=6)
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE, alpha=0.5)
    s.sphere_curve(great_circle(E3), color=INK, lw=0.6, alpha=0.45)
    phi, polar = np.radians(az + 42), np.radians(52)       # bob 52° from straight down
    q = np.array([np.sin(polar) * np.cos(phi), np.sin(polar) * np.sin(phi), -np.cos(polar)])
    a, b = s.tangent_plane(1.002 * q, size=0.56, color=SAND, alpha=0.85)
    s.curve(np.array([[0, 0, 0], q]), color=INK, lw=1.6, alpha=1.0, zorder=22)       # the rod
    s.dot((0, 0, 0), color=INK, size=4.5)
    s.label((0, 0, 0), "pivot", dx=-18, dy=7, size=9.5, **SANS)
    grav = -0.66 * E3
    g_tan = tangent(q, grav)
    s.arrow(q, q + grav, color=TERRACOTTA, zorder=32)
    s.arrow(q, q + g_tan, color=INDIGO, zorder=33)
    s.curve(np.array([q + grav, q + g_tan]), color=MUTED, lw=0.8, hidden=True, zorder=31)
    s.arrow(q, q - 0.5 * q, color=TEAL, zorder=32)
    s.dot(q, color=INK, size=6.5, zorder=35)
    s.label(q, r"$l\,q$", dx=13, dy=7, size=12)
    s.label(q + grav, r"$-m g\,e_3$", dx=24, dy=-3, color=TERRACOTTA)
    s.label(q + g_tan, r"$-(I - q q^{\top})\,m g\,e_3$", dx=-12, dy=-12, color=INDIGO, ha="right")
    s.label(q - 0.5 * q, r"$-T q$", dx=17, dy=4, color=TEAL)
    corners = [q + 0.56 * (i * a + j * b) for i in (1, -1) for j in (1, -1)]
    top_right = max(corners, key=lambda c: s.project(c)[0].sum())
    s.label(top_right, r"$T_qS^2$", dx=17, dy=2, size=10.5)
    s.text2d(2.2, 0.55, "tension: whatever keeps\nthe bob on the sphere", fontsize=9.5, color=TEAL, linespacing=1.3, **SANS)
    s.text2d(-2.15, -0.3, "only the part of gravity\nin the tangent plane\nmoves the bob", fontsize=9.5, color=INDIGO,
             linespacing=1.3, **SANS)
    s.save(OUT / "ch04-pendulum-forces.svg", pad=0.1)


# ------------------------------------------------ polhodes and the tumbling book
M_B, DIMS = 0.3, np.array([0.20, 0.13, 0.03])
J_B = M_B / 12 * np.array([DIMS[1]**2 + DIMS[2]**2, DIMS[0]**2 + DIMS[2]**2, DIMS[0]**2 + DIMS[1]**2])


def body_momentum_path(p0, T, n=1500):
    """Unit body angular momentum p = J Omega / |J Omega|: p_dot = p x (J^-1 p), time rescaled."""
    sol = solve_ivp(lambda _, p: np.cross(p, p / J_B) * J_B[1], [0, T], unit(p0), t_eval=np.linspace(0, T, n),
                    rtol=1e-10, atol=1e-12)
    return sol.y.T


def polhodes():
    s = Scene(width=6.6, height=3.4, azim=-52, elev=22)
    s.ax.set_position([0.0, 0.0, 0.47, 1.0])
    s.shadow((0, 0, -1.12), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    # closed polhodes around the smallest (e1) and largest (e3) axes
    for axis in (E1, -E1, E3, -E3):
        side = E2 if abs(axis @ E3) < 0.5 else E1
        for ang in np.radians([18, 36, 54]) if abs(axis @ E1) > 0.5 else np.radians([20, 40, 60, 76]):
            p0 = np.cos(ang) * axis + np.sin(ang) * side
            P = body_momentum_path(p0, 40.0, 3000)
            # keep one period
            d = np.linalg.norm(P - P[0], axis=1)
            back = np.where((d[1:-1] < d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < 0.02))[0]
            P = P[: back[0] + 3] if len(back) else P
            s.sphere_curve(P, color=INK, lw=0.8, alpha=0.7, show_hidden=False)
    # separatrices: the two great circles through +-e2 on which the energy equals that of spin about e2
    k = np.sqrt((1 / J_B[0] - 1 / J_B[1]) / (1 / J_B[1] - 1 / J_B[2]))
    for sign in (1, -1):
        s.sphere_curve(great_circle([k, 0, -sign]), color=TERRACOTTA, lw=1.1, alpha=0.75, show_hidden=False)
    # the simulated book: spin about e2 with a 1% tilt
    P = body_momentum_path(J_B * np.array([0.01, 1.0, 0.01]), 60.0, 6000)
    s.sphere_curve(P, color=TERRACOTTA, lw=2.0, zorder=18, show_hidden=False)
    for axis, name, dx, dy in ((E1, r"$e_1$", 12, -2), (-E2, r"$-e_2$", -15, 5), (E3, r"$e_3$", 8, 10)):
        s.dot(axis, color=INK if name != r"$-e_2$" else TERRACOTTA, size=4.5, hollow=name == r"$-e_2$")
        s.label(axis, name, dx=dx, dy=dy, size=12, color=INK if name != r"$-e_2$" else TERRACOTTA)
    s.text2d(0.0, -1.42, "body angular momentum $J\\Omega$ on its sphere,  $J_1 < J_2 < J_3$", fontsize=9.5, color=MUTED, **SANS)

    s.ax.set_xlim(-1.3, 1.3)
    s.ax.set_ylim(-1.6, 1.25)
    s._lims = ((-1.3, 1.3), (-1.6, 1.25))

    # time histories from Euler's equation
    ax = s.fig.add_axes([0.555, 0.19, 0.425, 0.74])
    style_axes(ax)
    spin = 2 * np.pi * 2.0

    def run(Om0):
        sol = solve_ivp(lambda _, w: np.cross(J_B * w, w) / J_B, [0, 4.0], Om0, t_eval=np.linspace(0, 4, 1600),
                        rtol=1e-10, atol=1e-12)
        return sol.t, sol.y
    t, w = run(spin * np.array([0.01, 1.0, 0.01]))
    ax.plot(t, w[1] / spin, color=TERRACOTTA, lw=1.8)
    t, w = run(spin * np.array([1.0, 0.01, 0.01]))
    ax.plot(t, w[0] / spin, color=INDIGO, lw=1.6)
    ax.axhline(0, color=INK, lw=0.5, alpha=0.5)
    ax.text(0.95, 1.08, r"about $e_1$ or $e_3$: stays", color=INDIGO, fontsize=10, path_effects=halo())
    ax.text(1.62, -0.62, "about $e_2$:\nflips every 1.5 s", color=TERRACOTTA, fontsize=10, path_effects=halo(), linespacing=1.25,
            ha="center")
    ax.set_xlim(0, 4)
    ax.set_ylim(-1.2, 1.3)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    ax.set_ylabel("spin component / initial spin", fontsize=9.5, **SANS)
    s.save(OUT / "ch04-polhodes.svg")


# ------------------------------------------------ quadrotor with a cable-suspended load
def circle3(c, u, v, r, n=80):
    t = np.linspace(0, 2 * np.pi, n)
    return c + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def quad_load():
    s = Scene(width=6.4, height=3.9, azim=-70, elev=20)
    R = rot(E2, np.radians(22)) @ rot(E1, np.radians(-8))
    xQ = np.array([0.0, 0.0, 1.0])
    q = unit([-0.80, 0.0, -0.60])
    l = 1.9
    xL = xQ + l * q
    b1, b2, b3 = R[:, 0], R[:, 1], R[:, 2]
    arm, rad = 0.78, 0.36
    d1, d2 = unit(b1 + b2), unit(b2 - b1)                 # arms at 45° to the body axes
    strands = []
    for d in (d1, d2):
        strands.append((np.linspace(xQ - arm * d, xQ + arm * d, 40), INK, 2.6))
    for d in (d1, -d1, d2, -d2):
        c = xQ + arm * d + 0.07 * b3
        strands.append((circle3(c, b1, b2, rad), INDIGO, 1.5))
    strands.append((np.linspace(xQ, xL, 60), INK, 1.1))
    s.ribbons(strands)
    for d in (d1, -d1, d2, -d2):
        s.patch(circle3(xQ + arm * d + 0.07 * b3, b1, b2, rad, 40), color=PERIWINKLE, alpha=0.5, edge=PERIWINKLE, lw=0.1, zorder=12)
    s.arrow(xQ, xQ + 1.15 * b3, color=TERRACOTTA, zorder=36)
    s.label(xQ + 1.15 * b3, r"$f\,R\,e_3$", dx=24, dy=0, color=TERRACOTTA, size=12)
    s.arrow(xQ, xQ + 0.62 * q, color=INDIGO, zorder=37)
    s.label(xQ + 0.62 * q, r"$q$", dx=2, dy=-12, color=INDIGO, size=13)
    s.arrow(xL, xL - 0.55 * q, color=TEAL, zorder=37)
    s.label(xL - 0.55 * q, r"$-T q$", dx=-4, dy=12, color=TEAL)
    s.arrow(xL, xL - 0.5 * E3, color=MUTED, lw=1.3, head=0.1)
    s.label(xL - 0.5 * E3, r"$-m_L g\,e_3$", dx=32, dy=0, color=MUTED, size=10.5)
    s.arrow(xQ, xQ - 0.5 * E3, color=MUTED, lw=1.3, head=0.1, zorder=34)
    s.label(xQ - 0.5 * E3, r"$-m_Q g\,e_3$", dx=-2, dy=-11, color=MUTED, size=10.5)
    s.dot(xQ, color=INK, size=5, zorder=40)
    for d in (d1, -d1, d2, -d2):
        s.dot(xQ + arm * d + 0.07 * b3, color=INK, size=3, zorder=33)
    s.dot(xL, color=INK, size=8, zorder=40)
    s.label(xL, r"$x_L$", dx=-14, dy=-4, size=12)
    s.label(xQ + 0.72 * l * q, r"$l$", dx=4, dy=-10, size=11.5)
    top = s.project(xQ + arm * d2 + 0.07 * b3)[0]
    s.text2d(top[0] - 1.25, top[1] + 0.42, r"quadrotor at $x_Q = x_L - l\,q$", fontsize=11)
    s.text2d(top[0] - 1.25, top[1] + 0.14, "attitude $R$, body torque $\\tau$", fontsize=11)
    o = xL + np.array([3.3, 0.0, -0.35])
    s.frame(o, np.eye(3), length=0.42, colors=(INK, INK, INK), lw=1.1, head=0.08)
    s.label(o + 0.42 * E1, r"$e_1$", dx=10, dy=-2, size=10.5)
    s.label(o + 0.42 * E2, r"$e_2$", dx=9, dy=5, size=10.5)
    s.label(o + 0.42 * E3, r"$e_3$", dx=0, dy=9, size=10.5)
    s.save(OUT / "ch04-quad-load.svg", pad=0.15)


# ------------------------------------------------------ integrators on SO(3)
def expm(w):
    th = np.linalg.norm(w)
    return np.eye(3) + hat(w) if th < 1e-12 else rot(w, th)


def pend_accel(R, Om):
    return np.linalg.solve(J_P, -np.cross(Om, J_P @ Om) + M_P * G * np.cross(R.T @ E3, RHO_P))


def pend_energy(R, Om):
    return 0.5 * Om @ J_P @ Om + M_P * G * E3 @ (R @ RHO_P)


def dexpinv(th, Om):
    return Om + 0.5 * np.cross(th, Om) + np.cross(th, np.cross(th, Om)) / 12


def rk4(f, a, b, h):
    k1 = f(a, b)
    k2 = f(a + h / 2 * k1[0], b + h / 2 * k1[1])
    k3 = f(a + h / 2 * k2[0], b + h / 2 * k2[1])
    k4 = f(a + h * k3[0], b + h * k3[1])
    return a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), b + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


def integrators():
    h, T = 0.02, 60.0
    R0, Om0 = expm(np.array([2.0, 0.0, 0.0])), np.array([0.0, 0.5, 6.0])
    n = int(round(T / h))
    t = h * np.arange(1, n + 1)
    out = {}
    for name in ("naive", "lie"):
        R, Om = R0.copy(), Om0.copy()
        E0 = pend_energy(R, Om)
        rows = []
        for _ in range(n):
            if name == "naive":
                R, Om = rk4(lambda R_, Om_: (R_ @ hat(Om_), pend_accel(R_, Om_)), R, Om, h)
            else:
                th, Om = rk4(lambda th_, Om_, R=R: (dexpinv(th_, Om_), pend_accel(R @ expm(th_), Om_)), np.zeros(3), Om, h)
                R = R @ expm(th)
            rows.append((np.linalg.norm(R.T @ R - np.eye(3)), abs(pend_energy(R, Om) - E0)))
        out[name] = np.array(rows)
    fig, (ax0, ax1) = new_plot(6.6, 2.8, ncols=2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.9, wspace=0.3)
    ax0.semilogy(t, out["naive"][:, 0], color=TERRACOTTA, lw=1.6)
    ax0.semilogy(t, np.maximum(out["lie"][:, 0], 1e-17), color=INDIGO, lw=1.2)
    ax0.set_ylim(1e-17, 1e-1)
    ax0.set_yticks([1e-16, 1e-12, 1e-8, 1e-4])
    ax0.text(30, 3e-3, "RK4 on the nine entries of $R$", color=TERRACOTTA, fontsize=10, ha="center", path_effects=halo())
    ax0.text(30, 6e-13, "RK4 with exponential update", color=INDIGO, fontsize=10, ha="center", path_effects=halo())
    ax0.set_title(r"distance from the manifold  $\|R^{\top}\! R - I\|$", fontsize=10, **SANS)
    ax1.semilogy(t, out["naive"][:, 1], color=TERRACOTTA, lw=1.4)
    ax1.semilogy(t, out["lie"][:, 1], color=INDIGO, lw=1.4)
    ax1.set_ylim(1e-8, 1e-1)
    ax1.text(30, 6e-3, "both drift: neither conserves energy", color=INK, fontsize=10, ha="center", path_effects=halo())
    ax1.set_title("energy error (J)", fontsize=10, **SANS)
    for ax in (ax0, ax1):
        ax.set_xlim(0, T)
        ax.set_xlabel("time (s)", fontsize=9.5, **SANS)
    save(fig, OUT / "ch04-integrators.svg")


if __name__ == "__main__":
    euler_cond()
    variation_s2()
    pendulum_forces()
    polhodes()
    quad_load()
    integrators()
