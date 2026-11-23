"""Figures for Chapter 5.  Run: python figures/src/ch05.py

Every tree, expression and trajectory drawn here is produced by geomech at run time.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from figkit import (DASH, INDIGO, INK, LW_CONTOUR, LW_SILHOUETTE, LW_VECTOR, MUTED, OCHRE, PAPER, PEACH,  # noqa: E402
                    PERIWINKLE, SAND, TEAL, TERRACOTTA, Scene, animate, ease, halo, new_plot, phase, plt, quadrotor,
                    save, unit)
from matplotlib.colors import to_rgb  # noqa: E402

from geomech import (S2, SO3, Cross, Dot, EOMTrace, Matrix, Scalar, SystemVariables, Vector,  # noqa: E402
                     check_eom, compute_eom, evaluate, getScalars, to_latex, to_standard_form)
from geomech.codegen import StateLayout, collect_params, load_module, module_source  # noqa: E402
from geomech.simulation import simulate  # noqa: E402

OUT = ROOT / "figures"
SANS = "Source Sans 3"
E3 = np.array([0.0, 0.0, 1.0])
half = Scalar("0.5", value=0.5, attr=["Constant"])


def col(v):
    return np.asarray(v, float).reshape(3, 1)


_CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def mix(c0, c1, a):
    """Blend two colors: a = 0 gives c0, a = 1 gives c1."""
    c0, c1 = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(c0 + (c1 - c0) * float(np.clip(a, 0, 1)))


# ------------------------------------------------------------------ systems
def pendulum():
    m, g, l = getScalars("m g l", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    q, tau = S2("q"), Vector("\\tau")
    omega, xi = q.get_tangent_vector(), q.get_variation_vector()
    L = half * m * l * l * Dot(omega, omega) - m * g * l * Dot(q, e3)
    dW = Dot(xi, tau)
    return L, dW, SystemVariables(vectors=[q])


def quadrotor_load():
    mQ, mL, g, l = getScalars("m_Q m_L g l", attr=["Constant"])
    J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    e3 = Vector("e3", attr=["Constant"])
    xL, q, R = Vector("x_L"), S2("q"), SO3("R")
    f, tau = Scalar("f"), Vector("\\tau")
    omega, xi = q.get_tangent_vector(), q.get_variation_vector()
    Omega, eta = R.get_tangent_vector(), R.get_variation_vector()
    xQ = xL - q * l
    vQ, vL = xQ.t_diff(), xL.t_diff()
    L = (half * mQ * Dot(vQ, vQ) + half * mL * Dot(vL, vL) + half * Dot(Omega, J * Omega)
         - mQ * g * Dot(e3, xQ) - mL * g * Dot(e3, xL))
    dW = Dot(xL.delta() - Cross(xi, q) * l, (R * e3) * f) + Dot(eta, tau)
    variables, inputs = SystemVariables(vectors=[xL, q], matrices=[R]), [f, tau]
    eom = compute_eom(L, dW, variables)
    sym = dict(mQ=mQ, mL=mL, g=g, l=l, e3=e3, xL=xL, q=q, R=R, f=f, omega=omega, xi=xi)
    return L, dW, variables, inputs, eom, sym


PARAMS = {"m_Q": 1.5, "m_L": 0.3, "l": 1.0, "g": 9.81,
          "J": np.diag([0.02, 0.025, 0.04]), "e3": E3.copy()}


# ------------------------------------------------------------ expression tree
LEAF_TEX = {"0.5": r"$\frac{1}{2}$", "m": "$m$", "g": "$g$", "l": "$l$", "\\omega_{q}": r"$\omega$",
            "q": "$q$", "e3": "$e_3$", "(-1)": "$-1$"}


def leaf_style(node):
    kind = type(node).__name__
    if kind == "S2":
        return PERIWINKLE, INDIGO
    if kind == "TS2":
        return "#cfe3e0", TEAL
    return SAND, "#8a7340"


def layout_tree(node, depth, xs, out, parent=None):
    """Tidy layout: leaves get consecutive x slots, a parent sits over the middle of its children."""
    kids = getattr(node, "nodes", ())
    me = {"node": node, "depth": depth, "parent": parent, "leaf": not kids}
    if not kids:
        me["x"] = xs[0]
        xs[0] += 1.0
    else:
        placed = [layout_tree(k, depth + 1, xs, out, me) for k in kids]
        me["x"] = 0.5 * (placed[0]["x"] + placed[-1]["x"])
    out.append(me)
    return me


def tree():
    L, _, _ = pendulum()
    items, xs = [], [0.0]
    layout_tree(L, 0, xs, items)
    depth = max(it["depth"] for it in items)
    fig = plt.figure(figsize=(6.6, 3.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    dy = 1.0

    def pos(it):
        return np.array([it["x"], -it["depth"] * dy])

    for it in items:
        if it["parent"] is not None:
            a, b = pos(it["parent"]), pos(it)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=LW_CONTOUR, zorder=2, solid_capstyle="round")
    for it in items:
        x, y = pos(it)
        if it["leaf"]:
            face, edge = leaf_style(it["node"])
            ax.add_patch(FancyBboxPatch((x - 0.36, y - 0.27), 0.72, 0.54, boxstyle="round,pad=0,rounding_size=0.24",
                                        fc=face, ec=edge, lw=1.1, zorder=3))
            ax.text(x, y - 0.02, LEAF_TEX[str(it["node"])], ha="center", va="center", fontsize=12, zorder=4)
        else:
            ax.add_patch(FancyBboxPatch((x - 0.4, y - 0.24), 0.8, 0.48, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=PAPER, ec=INK, lw=1.1, zorder=3))
            ax.text(x, y - 0.01, type(it["node"]).__name__, ha="center", va="center", fontsize=9.5,
                    fontfamily=SANS, zorder=4)
    # annotations for the two halves of L
    root = next(it for it in items if it["depth"] == 0)
    kin, pot = (it for it in items if it["depth"] == 1)
    ax.text(kin["x"] - 0.75, -dy + 0.02, r"$\frac{1}{2} m l^2\, \omega\cdot\omega$", ha="right", va="center",
            fontsize=11.5, color=MUTED)
    ax.text(pot["x"] + 0.75, -dy + 0.02, r"$-\,m g l\; q\cdot e_3$", ha="left", va="center", fontsize=11.5,
            color=MUTED)
    ax.text(root["x"], 0.55, r"$L$", ha="center", va="center", fontsize=13)
    # legend
    lx, ly = -0.55, 0.5
    for k, (face, edge, text) in enumerate([(PERIWINKLE, INDIGO, "point on $S^2$"),
                                            ("#cfe3e0", TEAL, "tangent vector at that point"),
                                            (SAND, "#8a7340", "constant")]):
        yy = ly - 0.5 * k
        ax.add_patch(FancyBboxPatch((lx, yy - 0.14), 0.42, 0.28, boxstyle="round,pad=0,rounding_size=0.13",
                                    fc=face, ec=edge, lw=1.0, zorder=3))
        ax.text(lx + 0.6, yy - 0.01, text, ha="left", va="center", fontsize=9.5, fontfamily=SANS, color=INK)
    ax.set_xlim(-0.8, xs[0] - 0.2)
    ax.set_ylim(-depth * dy - 0.5, 0.95)
    save(fig, OUT / "ch05-tree.svg")


# ------------------------------------------------------------------ pipeline
def tidy(tex):
    """The library's LaTeX with the subscript q dropped and e3 set as e_3; nothing else changes."""
    tex = tex.replace("\\omega_{q}", "\\omega").replace("\\xi_{q}", "\\xi").replace("e3", "e_3")
    tex = tex.replace("\\delta ", "\\mathrm{\\delta}")
    tex = tex.replace("m l l", "m\\,l\\,l").replace("m g l", "m\\,g\\,l")
    if tex.startswith("\\left(") and tex.endswith("\\right)"):
        tex = tex[len("\\left("):-len("\\right)")]
    return "$" + tex + "$"


def pipeline():
    L, dW, variables = pendulum()
    tr = EOMTrace()
    compute_eom(L, dW, variables, trace=tr)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.summary(chars=10)
    nodes = {ln.split()[0]: int(ln.split()[1]) for ln in buf.getvalue().splitlines() if " nodes " in ln}
    key = "equation[\\xi_{q}]"
    rows = [
        ("dS", "vary", "product rule; $\\mathrm{\\delta} q = \\xi\\times q$ at once, $\\mathrm{\\delta}\\omega$ left alone", INK),
        ("manifold_rules", "manifold rule", "$\\mathrm{\\delta}\\omega \\;\\to\\; \\dot\\xi - 2\\,\\omega\\times\\xi$", TEAL),
        ("simplify", "simplify", "$\\omega\\cdot(\\omega\\times\\xi) = 0$, numbers folded", INK),
        ("ibp", "integrate by parts", "$\\dot\\xi\\cdot f \\;\\to\\; -\\,\\xi\\cdot\\dot f$", TERRACOTTA),
        (key, "extract", "read off what multiplies $\\xi$", INDIGO),
    ]
    fig = plt.figure(figsize=(6.6, 4.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.62)
    x_box, w_box, h_box = 1.42, 5.02, 0.44
    y0, step = 4.1, 0.75
    nmax = max(nodes.values())

    def box(y, text, edge=INK, face=PAPER, size=11.5):
        ax.add_patch(FancyBboxPatch((x_box, y - h_box / 2), w_box, h_box, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=face, ec=edge, lw=1.1, zorder=3))
        ax.text(x_box + 0.14, y - 0.01, text, ha="left", va="center", fontsize=size, zorder=4)

    def count(y, n, color=MUTED):
        ax.plot([1.22 - 1.0 * n / nmax, 1.22], [y, y], color=color, lw=5, solid_capstyle="butt", alpha=0.55)
        ax.text(1.22 - 1.0 * n / nmax - 0.06, y, f"{n}", ha="right", va="center", fontsize=9, fontfamily=SANS,
                color=color)

    box(y0, "$L \\;=\\; $" + tidy(to_latex(L)), face="#f6f3ea")
    count(y0, len_nodes(L))
    ax.text(0.2, y0 + 0.36, "nodes in the tree", ha="left", va="center", fontsize=9, fontfamily=SANS, color=MUTED)
    for k, (name, verb, note, color) in enumerate(rows):
        y = y0 - (k + 1) * step
        ya, yb = y + step - h_box / 2, y + h_box / 2
        ax.annotate("", (x_box + 0.5, yb + 0.02), xytext=(x_box + 0.5, ya - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=LW_VECTOR, mutation_scale=10))
        ax.text(x_box + 0.66, 0.5 * (ya + yb), verb, ha="left", va="center", fontsize=9.5, fontfamily=SANS,
                color=color, fontweight="semibold")
        ax.text(x_box + 2.05, 0.5 * (ya + yb), note, ha="left", va="center", fontsize=9.5, color=MUTED)
        last = name == key
        lhs = "$E \\;=\\; $" if last else "$\\mathrm{\\delta} S \\;=\\; $"
        box(y, lhs + tidy(to_latex(tr[name])), edge=INDIGO if last else INK, face=PERIWINKLE if last else PAPER,
            size=10.2 if name == "manifold_rules" else 11.5)
        count(y, nodes[name], color=INDIGO if last else MUTED)
    save(fig, OUT / "ch05-pipeline.svg")


def len_nodes(expr):
    return 1 + sum(len_nodes(k) for k in getattr(expr, "nodes", ()))


# ------------------------------------------------------------- tangent plane
def tangent():
    """The coefficient E of xi for the cable, evaluated at a real state, split at q."""
    L, dW, variables, inputs, eom, sym = quadrotor_load()
    s = Scene(width=6.4, height=3.5, azim=-52, elev=17)
    tilt = np.radians(58)
    q = unit(np.cos(tilt) * s.view + np.sin(tilt) * (0.42 * s.right + 0.91 * s.up))
    w = 1.5 * unit(np.cross(q, s.up))
    d = unit(np.cross(q, np.cross(-0.85 * s.right - 0.5 * s.up, q)))       # a tangent direction, down-left
    env = {"q": col(q), str(sym["omega"]): col(w), str(sym["xL"].t_diff()): col(-1.1 * w / 1.5 + 0.3 * d),
           str(sym["xL"].t_diff().t_diff()): col(5.0 * np.cross(d, q) - 9.81 * E3),
           str(sym["omega"].t_diff()): col(np.zeros(3)), "e3": col(E3), "R": np.eye(3),
           "f": 0.0, "m_Q": 1.5, "m_L": 0.3, "l": 1.0, "g": 9.81}
    E = evaluate(eom["\\xi_{q}"][1], env).ravel()
    scale = 1.25 / np.linalg.norm(E)
    En, Et = (E @ q) * q * scale, (E - (E @ q) * q) * scale
    s.shadow((0, 0, -1.08), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.tangent_plane(q, size=0.8, spin=0.25, color=SAND, alpha=0.82)
    tip = q + Et + En
    s.curve(np.array([q + Et, tip]), color=MUTED, lw=0.8, hidden=True, zorder=29)
    s.curve(np.array([q + En, tip]), color=MUTED, lw=0.8, hidden=True, zorder=29)
    s.arrow(q, q + En, color=TERRACOTTA, lw=1.7)
    s.arrow(q, q + Et, color=INDIGO, lw=1.9)
    s.arrow(q, tip, color=INK, lw=1.2)
    s.dot(q, color=INK)
    s.label(q, "$q$", dx=11, dy=-9, size=12)
    s.label(tip, r"$E_\xi$", dx=-9, dy=8, size=12)
    s.label(q + Et, r"$(I - qq^{\top})E_\xi$", dx=-6, dy=-13, color=INDIGO, size=11.5)
    s.label(q + En, r"$(q\cdot E_\xi)\,q$", dx=42, dy=-9, color=TERRACOTTA, size=11.5)
    s.text2d(-2.15, 0.1, "the part a variation $\\xi \\perp q$\ncan test: the equation of motion", fontsize=9.5,
             fontfamily=SANS, color=INDIGO, linespacing=1.3)
    s.text2d(2.15, 1.25, "the part along $q$: no variation\nreaches it, so it says nothing", fontsize=9.5,
             fontfamily=SANS, color=TERRACOTTA, linespacing=1.3)
    s.text2d(2.0, -0.15, "tangent plane at $q$,\nwhere $\\xi$ and $\\omega$ live", fontsize=9.5, fontfamily=SANS,
             color=MUTED, linespacing=1.3)
    s.save(OUT / "ch05-tangent.svg", pad=0.1)


# ---------------------------------------------------------------- sign error
def sign_error():
    L, dW, variables, inputs, eom, sym = quadrotor_load()
    mQ, mL, g, l, e3, xL, q, R, f, omega = (sym[k] for k in "mQ mL g l e3 xL q R f omega".split())
    aL, domega = xL.t_diff().t_diff(), omega.t_diff()

    def load_equation(sign):
        return ((aL + e3 * g) * (mQ + mL) * (-1) + Cross(domega, q) * (mQ * l)
                + q * (Dot(omega, omega) * mQ * l) * sign + (R * e3) * f)

    seeds = np.arange(20)
    errs = {}
    for sign in (-1, +1):
        hand = dict(eom)
        hand[str(xL.delta())] = (xL.delta(), load_equation(sign))
        errs[sign] = [check_eom(L, dW, variables, hand, inputs=inputs, seed=int(sd)) for sd in seeds]

    fig, (ax0, ax1) = new_plot(6.6, 2.9, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.1, right=0.975, bottom=0.18, top=0.93, wspace=0.34)
    ax0.semilogy(seeds, errs[-1], "o", ms=4.5, color=INDIGO)
    ax0.semilogy(seeds, errs[+1], "o", ms=4.5, color=TERRACOTTA)
    ax0.axhline(1e-5, color=INK, lw=0.8, ls=(0, (3.5, 3)))
    ax0.text(19.4, 2.4e-5, "accept below $10^{-5}$", ha="right", va="bottom", fontsize=9.5, fontfamily=SANS,
             color=INK, path_effects=halo())
    ax0.text(9.5, 6e-7, "sign as derived: finite-difference noise", ha="center", fontsize=9.5, fontfamily=SANS,
             color=INDIGO, path_effects=halo())
    ax0.text(9.5, 0.035, "sign flipped by hand", ha="center", fontsize=9.5, fontfamily=SANS, color=TERRACOTTA,
             path_effects=halo())
    ax0.set_ylim(1e-9, 30)
    ax0.set_xlim(-0.8, 19.8)
    ax0.set_xticks([0, 5, 10, 15])
    ax0.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax0.set_xlabel("random seed of the test state", fontfamily=SANS, fontsize=9.5)
    ax0.set_ylabel("check_eom error", fontfamily=SANS, fontsize=9.5)

    # Size of the discrepancy between the two hand equations as the cable swings faster.
    qv = np.array([np.sin(0.4), 0.0, -np.cos(0.4)])
    rates = np.linspace(0, 3, 61)
    gap = []
    for wmag in rates:
        env = {"q": col(qv), str(omega): col([0, wmag, 0]), "e3": col(E3), "R": np.eye(3), "f": 17.0,
               str(aL): col([0.1, 0.2, 0.3]), str(domega): col([0, 0.5, 0]), "m_Q": 1.5, "m_L": 0.3, "l": 1.0,
               "g": 9.81}
        gap.append(np.linalg.norm(evaluate(load_equation(+1), dict(env)) - evaluate(load_equation(-1), dict(env))))
    ax1.plot(rates, gap, color=TERRACOTTA, lw=1.8)
    ax1.axhline(0.3 * 9.81, color=MUTED, lw=0.8, ls=(0, (3.5, 3)))
    ax1.text(2.95, 0.3 * 9.81 + 0.7, "weight of the load", ha="right", fontsize=9.5, fontfamily=SANS, color=MUTED,
             path_effects=halo())
    ax1.plot([0], [0], "o", ms=5.5, color=INDIGO, zorder=5, clip_on=False)
    ax1.annotate("hover: the two\nmodels agree", (0, 0), xytext=(0.28, 7.4), fontsize=9.5, fontfamily=SANS,
                 color=INDIGO, linespacing=1.3, path_effects=halo(),
                 arrowprops=dict(arrowstyle="-", color=INDIGO, lw=0.8, shrinkA=2, shrinkB=4))
    ax1.text(2.3, 19.5, r"$2\,m_Q l\,\|\omega\|^2$", ha="right", fontsize=11, color=TERRACOTTA, path_effects=halo())
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 28)
    ax1.set_xlabel("swing rate of the cable (rad/s)", fontfamily=SANS, fontsize=9.5)
    ax1.set_ylabel("force the wrong sign invents (N)", fontfamily=SANS, fontsize=9.5)
    save(fig, OUT / "ch05-sign-error.svg")


# ----------------------------------------------------- simulated quadrotor + load
def circle3(center, a, b, r, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    return center + r * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))


def quadload_run():
    L, dW, variables, inputs, eom, _ = quadrotor_load()
    sf = to_standard_form(eom, variables, inputs)
    layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
    plant = load_module(module_source(sf, layout, lagrangian=L), "quadrotor_load")
    p = PARAMS
    q0 = np.array([np.sin(0.8), 0.0, -np.cos(0.8)])
    w0 = np.cross(q0, np.cross([0.0, 0.0, 1.6], q0))
    v_cm = np.array([0.45, 0.0, 0.0])
    v0 = v_cm + p["m_Q"] * p["l"] / (p["m_Q"] + p["m_L"]) * np.cross(w0, q0)
    x0 = layout.pack(x_L=np.zeros(3), q=q0, R=np.eye(3), omega_q=w0, x_L_dot=v0)
    hover = np.r_[(p["m_Q"] + p["m_L"]) * p["g"], 0.0, 0.0, 0.0]
    traj = simulate(plant, x0, 6.0, p, u=hover)
    xLs, qs = traj["x_L"], traj["q"]
    xQs = xLs - p["l"] * qs
    com = (p["m_Q"] * xQs + p["m_L"] * xLs) / (p["m_Q"] + p["m_L"])
    floor = xLs[:, 2].min() - 0.55
    return dict(t=traj.t, xL=xLs, xQ=xQs, R=traj["R"], com=com, floor=floor)


def view_limits(azim, elev, pts, left=0.2, right=0.2, bottom=0.2, top=0.2):
    """Canvas limits that hold these 3D points for the whole of an animation."""
    probe = Scene(azim=azim, elev=elev)
    xy = probe.project(pts)
    plt.close(probe.fig)
    return (xy[:, 0].min() - left, xy[:, 0].max() + right), (xy[:, 1].min() - bottom, xy[:, 1].max() + top)


QL_FPS = 14


def quadload_frame(t):
    D = cached("quadload", quadload_run)
    xL, xQ, com, floor = D["xL"], D["xQ"], D["com"], D["floor"]
    k = int(round(t * (len(D["t"]) - 1)))
    shadow = np.c_[xL[:, :2], np.full(len(xL), floor)]
    xlim, ylim = cached("quadload_lims", lambda: view_limits(-78, 17, np.vstack([xL, xQ + 0.5 * E3, shadow]),
                                                             left=0.5, right=0.5, bottom=0.75, top=0.3))
    s = Scene(width=6.9, height=6.9 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), azim=-78, elev=17, xlim=xlim, ylim=ylim)
    s.curve(com[[0, -1]], color=MUTED, lw=0.8, hidden=True)
    s.curve(shadow[:k + 1:10], color=MUTED, lw=0.8, alpha=1.0, zorder=5)
    s.shadow(shadow[k], 0.16, strength=0.3)
    s.curve(xQ[:k + 1:10], color=MUTED, lw=1.0, alpha=1.0, zorder=15)
    s.curve(xL[:k + 1:10], color=INDIGO, lw=1.5, alpha=1.0, zorder=16)
    s.curve(com[:k + 1:10], color=TEAL, lw=1.4, alpha=1.0, zorder=17)
    s.curve(np.array([xQ[k], xL[k]]), color=INK, lw=1.1, alpha=1.0, zorder=32)
    quadrotor(s, xQ[k], D["R"][k], arm=0.2, lw=1.7)
    s.arrow(xQ[k], xQ[k] + 0.42 * D["R"][k][:, 2], color=TERRACOTTA, lw=1.4, head=0.09, zorder=36)
    s.label(xQ[k] + 0.42 * D["R"][k][:, 2], r"$f\,R\,e_3$", dx=20, dy=2, size=10.5, color=TERRACOTTA)
    s.dot(xL[k], color=INDIGO, size=6.5, zorder=38)
    s.dot(com[k], color=TEAL, size=4.5, zorder=39)
    s.label(com[0], "center of mass", dx=-8, dy=0, size=9.5, fontfamily=SANS, color=TEAL, ha="right")
    s.label(xL[k], "$x_L$", dx=0, dy=-13, size=11, color=INDIGO)
    s.label(shadow[0], "shadow of the load", dx=-6, dy=-36, size=9.5, fontfamily=SANS, color=MUTED)
    xm, ym = 0.5 * (xlim[0] + xlim[1]), ylim[0]
    s.text2d(xlim[1] - 0.1, ylim[1] - 0.16, f"t = {D['t'][k]:.1f} s, real time", fontsize=9.5, color=MUTED, ha="right",
             fontfamily=SANS)
    if t < 0.45:
        msg = "the thrust cancels the total weight, so the center of mass coasts on a straight line"
    else:
        msg = "the light load swings wide below; the heavy quadrotor wobbles with one fifth of the radius"
    s.text2d(xm, ym + 0.17, msg, fontsize=10, color=INK, fontfamily=SANS)
    return s


def quadload():
    n = int(round(6.0 * QL_FPS)) + 1
    animate(quadload_frame, OUT / "ch05-quadload.gif", OUT / "ch05-quadload-strip.png", n=n, fps=QL_FPS,
            width_px=720, stills=(0.08, 0.4, 0.7, 1.0), strip_px=900, strip_cols=2, hold=1.8)


# ----------------------------------------------------------------- n-link chain
def chain_run():
    from geomech.indexed import Family, IndexedVariables, Sum, compute_indexed_eom, indices, n

    i, j = indices("i j")
    g, e3 = Scalar("g", attr=["Constant"]), Vector("e3", attr=["Constant"])
    q = Family(S2, "q")
    m = Family(Scalar, "m", attr=["Constant"])
    l = Family(Scalar, "l", attr=["Constant"])

    def x(k):
        return Sum(j, 1, k, q[j] * l[j])

    v = x(i).t_diff()
    L = Sum(i, 1, n, 0.5 * m[i] * v.dot(v)) - Sum(i, 1, n, m[i] * g * x(i).dot(e3))
    fam = IndexedVariables(families=[q])
    eom = compute_indexed_eom(L, None, fam)
    size = 4
    members, L4 = fam.members(size), L.instantiate(size)
    sf = to_standard_form(eom.instantiate(size), members, [])
    layout = StateLayout(members, [], collect_params([L4], members, []))
    plant = load_module(module_source(sf, layout, lagrangian=L4), "chain4")
    p = {"g": 9.81, "e3": E3.copy()}
    p.update({f"m_{{{k}}}": 0.25 for k in range(1, size + 1)})
    p.update({f"l_{{{k}}}": 0.25 for k in range(1, size + 1)})
    x0 = layout.pack(**{f"q_{k}": [1.0, 0.0, 0.0] for k in range(1, size + 1)},
                     **{f"omega_q_{k}": [0.0, 0.0, 0.6 * k] for k in range(1, size + 1)})
    traj = simulate(plant, x0, 2.0, p)
    joints = np.zeros((len(traj.t), size + 1, 3))
    for k in range(1, size + 1):
        joints[:, k] = joints[:, k - 1] + 0.25 * traj[f"q_{k}"]

    return traj.t, joints


CHAIN_END, CHAIN_SLOW, CHAIN_FPS = 1.3, 6.0, 12


def chain_frame(t):
    tt, joints = cached("chain", chain_run)
    last = int(round(CHAIN_END * 1000))
    k = int(round(t * last))
    floor = -1.22
    tip = joints[:last + 1, -1]
    shadow = np.c_[tip[:, :2], np.full(len(tip), floor)]
    xlim, ylim = cached("chain_lims", lambda: view_limits(-64, 16, np.vstack([joints[:last + 1].reshape(-1, 3), shadow]),
                                                          left=0.35, right=0.35, bottom=0.42, top=0.3))
    s = Scene(width=6.6, height=6.6 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), azim=-64, elev=16, xlim=xlim, ylim=ylim)
    s.curve(np.array([[0, 0, 0], [0, 0, floor]]), color=MUTED, lw=0.7, hidden=True)
    s.curve(joints[0], color=MUTED, lw=0.8, hidden=True)                    # where it was released
    s.dot(joints[0, -1], color=MUTED, size=4, hollow=True, zorder=18)
    s.label(joints[0, -1], "released here", dx=0, dy=11, size=9.5, fontfamily=SANS, color=MUTED)
    s.curve(shadow[:k + 1:4], color=MUTED, lw=0.8, alpha=1.0, zorder=5)
    s.shadow(shadow[k], 0.09, strength=0.3)
    s.curve(tip[:k + 1:4], color=INDIGO, lw=1.3, alpha=1.0, zorder=19)
    s.curve(joints[k], color=INK, lw=1.9, alpha=1.0, zorder=30)
    for jt in joints[k, 1:-1]:
        s.dot(jt, color=INK, size=4.2, zorder=31)
    s.dot(joints[k, -1], color=INDIGO, size=5.5, zorder=32)
    s.dot((0, 0, 0), color=INK, size=5, hollow=True, zorder=60)
    s.label((0, 0, 0), "pivot", dx=-4, dy=11, size=9.5, fontfamily=SANS)
    d = unit(s.project(joints[k, -1])[0] - s.project(joints[k, -2])[0] + 1e-9)       # away from the last link
    s.label(joints[k, -1], "mass 4", dx=24 * d[0], dy=15 * d[1], size=9.5, fontfamily=SANS, color=INDIGO)
    s.text2d(xlim[1] - 0.06, ylim[1] - 0.1, f"t = {tt[k]:.2f} s, shown {CHAIN_SLOW:g} times slower", fontsize=9.5,
             color=MUTED, ha="right", fontfamily=SANS)
    if t < 0.4:
        msg = "four links, from the equation for link $i$ at $n = 4$: released level, turning about the vertical"
    else:
        msg = "the chain whips through and folds back; the indigo curve is the path of mass 4"
    s.text2d(0.5 * (xlim[0] + xlim[1]), ylim[0] + 0.1, msg, fontsize=10, color=INK, fontfamily=SANS)
    return s


def chain():
    n = int(round(CHAIN_END * CHAIN_SLOW * CHAIN_FPS)) + 1
    animate(chain_frame, OUT / "ch05-chain.gif", OUT / "ch05-chain-strip.png", n=n, fps=CHAIN_FPS, width_px=700,
            stills=(0.0, 0.35, 0.68, 1.0), strip_px=900, strip_cols=2, hold=1.8)


# --------------------------------------------- animation: the tree, rewritten
# Every tree drawn is a stage the library recorded in its trace. Two things are
# folded for the eye, and only for the eye: a chain of binary products is drawn
# as one Mul node, and a run of named constants inside it as one leaf.
OP_NAME = {"Add": "Add", "VAdd": "Add", "Mul": "Mul", "SVMul": "Mul", "Dot": "Dot", "Cross": "Cross",
           "TimeDerivative": "d/dt", "Variation": "$\\mathrm{\\delta}$"}
PRODUCTS = ("Mul", "SVMul")
RW_TEX = dict(LEAF_TEX, **{"\\xi_{q}": r"$\xi$", "\\tau": r"$\tau$", "(-2)": "$-2$"})
RW_W, RW_H, RW_DY, RW_TOP = 7.4, 4.95, 0.45, 4.95 - 0.78
RW_KEY = "equation[\\xi_{q}]"
RW_STAGES = [   # trace key, name of the step, left side of the formula, what the clerk's card says
    (None, "Lagrangian", "$L \\;=\\; $", "the Lagrangian, as the library stores it"),
    ("dS", "vary", "$\\mathrm{\\delta} S \\;=\\; $",
     "rule: product rule on every product; a point $q$ on $S^2$ varies as $\\xi\\times q$; add $\\xi\\cdot\\tau$"),
    ("manifold_rules", "manifold rule", "$\\mathrm{\\delta} S \\;=\\; $",
     "rule: every $\\mathrm{\\delta}\\omega$ becomes $\\dot\\xi - 2\\,\\omega\\times\\xi$"),
    ("simplify", "simplify", "$\\mathrm{\\delta} S \\;=\\; $",
     "rule: $\\omega\\cdot(\\omega\\times\\xi) = 0$; then the two equal halves merge"),
    ("ibp", "integrate by parts", "$\\mathrm{\\delta} S \\;=\\; $",
     "rule: move the dot from $\\xi$ to its partner and flip the sign"),
    (RW_KEY, "extract", "$E_\\xi \\;=\\; $",
     "rule: turn the triple product, then collect what multiplies $\\xi$"),
]


def factors(node):
    """The factors of a chain of binary products, in the order the library stores them."""
    if type(node).__name__ in PRODUCTS:
        return [f for k in node.nodes for f in factors(k)]
    return [node]


def named_constant(node):
    return (not getattr(node, "nodes", None) and type(node).__name__ == "Scalar"
            and "Constant" in (node.attr or ()) and str(node) not in ("(-1)", "(-2)"))


def display_tree(node):
    """The library's tree with product chains flattened and runs of named constants folded into one leaf."""
    kind = type(node).__name__
    kids = getattr(node, "nodes", None) or ()
    if not kids:
        if kind in ("S2", "TS2"):
            face, edge = leaf_style(node)
        elif "Constant" in (node.attr or ()):
            face, edge = SAND, "#8a7340"
        else:
            face, edge = PAPER, INK
        return dict(label=RW_TEX[str(node)], key=str(node), leaf=True, face=face, edge=edge, kids=[], w=0.33)
    if kind in PRODUCTS:
        out, run = [], []

        def flush():
            if run:
                tex = "$" + "\\,".join(RW_TEX[str(c)].strip("$") for c in run) + "$"
                out.append(dict(label=tex, key=" ".join(str(c) for c in run), leaf=True, face=SAND, edge="#8a7340",
                                kids=[], w=0.33 if len(run) == 1 else 0.2 + 0.12 * len(run)))
                run.clear()

        for f in factors(node):
            if named_constant(f):
                run.append(f)
            else:
                flush()
                out.append(display_tree(f))
        flush()
        return dict(label="Mul", key="Mul", leaf=False, kids=out)
    return dict(label=OP_NAME[kind], key=OP_NAME[kind], leaf=False, kids=[display_tree(k) for k in kids])


def place_tree(root, gap=0.055):
    """Leaves side by side in the stored order, every operation over the middle of its children."""
    nodes, x = [], [0.0]

    def walk(d, depth, parent):
        d.update(depth=depth, parent=parent, id=len(nodes))
        nodes.append(d)
        if d["leaf"]:
            d["x"] = x[0] + 0.5 * d["w"]
            x[0] += d["w"] + gap
            d["leaves"] = {d["id"]}
        else:
            for k in d["kids"]:
                walk(k, depth + 1, d)
            d["x"] = 0.5 * (d["kids"][0]["x"] + d["kids"][-1]["x"])
            d["leaves"] = set().union(*(k["leaves"] for k in d["kids"]))

    walk(root, 0, None)
    shift = 0.5 * RW_W - 0.5 * (x[0] - gap)
    for d in nodes:
        d["pos"] = np.array([d["x"] + shift, RW_TOP - RW_DY * d["depth"]])
    return nodes


def match_trees(A, B):
    """Which nodes of tree A are still there in tree B. Returns {id in A: id in B}.

    Leaves are aligned in reading order (same symbol, and preferably the same kind of
    parent). An operation is matched to the operation of the same kind that sits over
    the most of the same leaves. What is left unmatched was removed or created by the rule.
    """
    la, lb = [d for d in A if d["leaf"]], [d for d in B if d["leaf"]]

    def up(d, k):
        for _ in range(k):
            d = d["parent"] if d is not None else None
        return d["key"] if d is not None else None

    def score(a, b):
        if a["key"] != b["key"]:
            return -1.0
        return 1.0 + 0.5 * (up(a, 1) == up(b, 1)) + 0.25 * (up(a, 2) == up(b, 2))

    n, m = len(la), len(lb)
    best = np.zeros((n + 1, m + 1))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            sc = score(la[i], lb[j])
            take = best[i + 1, j + 1] + sc if sc > 0 else -1.0
            best[i, j] = max(best[i + 1, j], best[i, j + 1], take)
    pairs, i, j = {}, 0, 0
    while i < n and j < m:
        sc = score(la[i], lb[j])
        if sc > 0 and abs(best[i, j] - (best[i + 1, j + 1] + sc)) < 1e-12:
            pairs[la[i]["id"]] = lb[j]["id"]
            i, j = i + 1, j + 1
        elif best[i + 1, j] >= best[i, j + 1]:
            i += 1
        else:
            j += 1
    # A symbol that occurs once on each side among the leftovers has only changed places.
    rest_a = [d for d in la if d["id"] not in pairs]
    rest_b = [d for d in lb if d["id"] not in pairs.values()]
    for a in rest_a:
        same_a = [d for d in rest_a if d["key"] == a["key"]]
        same_b = [d for d in rest_b if d["key"] == a["key"]]
        if len(same_a) == 1 and len(same_b) == 1:
            pairs[a["id"]] = same_b[0]["id"]
    ops_a, ops_b = [d for d in A if not d["leaf"]], [d for d in B if not d["leaf"]]
    cands = []
    for a in ops_a:
        mapped = {pairs[i] for i in a["leaves"] if i in pairs}
        for b in ops_b:
            if a["key"] == b["key"]:
                common = len(mapped & b["leaves"])
                if common:
                    jac = common / (len(a["leaves"]) + len(b["leaves"]) - common)
                    cands.append((-jac, abs(a["depth"] - b["depth"]), a["id"], b["id"]))
    for _, _, ia, ib in sorted(cands):
        if ia not in pairs and ib not in pairs.values():
            pairs[ia] = ib
    # An operation whose parent survived, and which is the only one of its kind there, has moved.
    for a in ops_a:
        if a["id"] in pairs or a["parent"] is None or a["parent"]["id"] not in pairs:
            continue
        pb = B[pairs[a["parent"]["id"]]]
        free_b = [k for k in pb["kids"] if k["key"] == a["key"] and not k["leaf"] and k["id"] not in pairs.values()]
        free_a = [k for k in a["parent"]["kids"] if k["key"] == a["key"] and not k["leaf"] and k["id"] not in pairs]
        if len(free_a) == 1 and len(free_b) == 1:
            pairs[a["id"]] = free_b[0]["id"]
    return pairs


def rewrite_data():
    L, dW, variables = pendulum()
    tr = EOMTrace()
    compute_eom(L, dW, variables, trace=tr)
    stages = []
    for key, name, lhs, card in RW_STAGES:
        expr = L if key is None else tr[key]
        stages.append(dict(name=name, card=card, nodes=place_tree(display_tree(expr)), count=len_nodes(expr),
                           formula=lhs + tidy(to_latex(expr))))
    matches = [match_trees(a["nodes"], b["nodes"]) for a, b in zip(stages[:-1], stages[1:])]
    # A time derivative that survives but changes sides is what integration by parts is about: mark it.
    hops = [{i for i, j in mt.items() if a["nodes"][i]["key"] == "d/dt"
             and not {mt.get(leaf) for leaf in a["nodes"][i]["leaves"]} & b["nodes"][j]["leaves"]}
            for a, b, mt in zip(stages[:-1], stages[1:], matches)]
    # the timeline: (what happens, which stage or transition, how many frames)
    plan = [("show", 0, 9)]
    for k in range(len(matches)):
        plan += [("announce", k, 8), ("out", k, 4), ("move", k, 8), ("in", k, 4), ("settle", k + 1, 8)]
    plan.append(("show", len(stages) - 1, 6))
    return stages, matches, hops, plan


def rewrite_time(plan, index, u):
    """The time t in [0, 1] at which segment `index` of the plan has run a fraction u."""
    total = sum(p[2] for p in plan)
    return (sum(p[2] for p in plan[:index]) + u * plan[index][2]) / total


def draw_node(ax, d, pos, alpha=1.0, hot=0.0):
    """One node of a tree. `hot` blends its outline toward terracotta: touched by the rule in hand."""
    if alpha <= 0.01:
        return
    x, y = pos
    lw = 1.1 + 0.7 * hot
    if d["leaf"]:
        w, h = d["w"], 0.3
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.13",
                                    fc=d["face"], ec=mix(d["edge"], TERRACOTTA, hot), lw=lw, alpha=alpha, zorder=3))
        ax.text(x, y - 0.012, d["label"], ha="center", va="center", fontsize=10.5, alpha=alpha, zorder=4)
    else:
        w, h = 0.44 if d["label"] == "Cross" else 0.36, 0.25
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=PAPER, ec=mix(INK, TERRACOTTA, hot), lw=lw, alpha=alpha, zorder=3))
        ax.text(x, y - 0.008, d["label"], ha="center", va="center", fontsize=8.5 if d["label"][0] != "$" else 10.5,
                fontfamily=SANS, alpha=alpha, zorder=4, color=mix(INK, TERRACOTTA, hot))


def draw_edge(ax, a, b, alpha=1.0, hot=0.0):
    if alpha > 0.01:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=mix(INK, TERRACOTTA, hot), lw=LW_CONTOUR, alpha=alpha, zorder=2,
                solid_capstyle="round")


def rewrite_frame(t):
    stages, matches, hops, plan = cached("rewrite", rewrite_data)
    f = t * sum(p[2] for p in plan)
    for what, k, frames in plan:
        if f <= frames:
            break
        f -= frames
    u = float(np.clip(f / frames, 0, 1))

    fig = plt.figure(figsize=(RW_W, RW_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, RW_W)
    ax.set_ylim(0, RW_H)

    if what in ("show", "settle"):
        now, shown, hot_caption = stages[k], k, what == "settle"
        fresh = set()
        if what == "settle":
            fresh = ({d["id"] for d in now["nodes"]} - set(matches[k - 1].values())
                     | {matches[k - 1][i] for i in hops[k - 1]})
        hot = 1 - ease(phase(u, 0.45, 1.0)) if what == "settle" else 0.0
        for d in now["nodes"]:
            h = hot if d["id"] in fresh else 0.0
            if d["parent"] is not None:
                draw_edge(ax, d["parent"]["pos"], d["pos"], hot=h)
            draw_node(ax, d, d["pos"], hot=h)
    else:
        A, B, mt = stages[k]["nodes"], stages[k + 1]["nodes"], matches[k]
        back = {v: key for key, v in mt.items()}
        shown, hot_caption = (k + 1 if what == "in" else k), True
        kept_edge = lambda d: (d["id"] in mt and d["parent"]["id"] in mt
                               and B[mt[d["id"]]]["parent"] is B[mt[d["parent"]["id"]]])
        if what in ("announce", "out"):
            hot = ease(u) if what == "announce" else 1.0
            alpha = 1.0 if what == "announce" else 1 - ease(u)
            for d in A:
                gone = d["id"] not in mt
                if d["parent"] is not None:
                    keep = kept_edge(d)
                    draw_edge(ax, d["parent"]["pos"], d["pos"], alpha=1.0 if keep else alpha, hot=0.0 if keep else hot)
                draw_node(ax, d, d["pos"], alpha=alpha if gone else 1.0, hot=hot if gone or d["id"] in hops[k] else 0.0)
        else:
            e = ease(u) if what == "move" else 1.0

            def where(d):                    # a surviving node of A on its way to its place in B
                p0, p1 = d["pos"], B[mt[d["id"]]]["pos"]
                p = p0 + (p1 - p0) * e
                if d["id"] in hops[k]:                                     # the derivative hops over
                    p = p + np.array([0.0, 0.2 * np.sin(np.pi * e)])
                return p

            for d in A:
                if d["id"] in mt:
                    if d["parent"] is not None and kept_edge(d):
                        draw_edge(ax, where(d["parent"]), where(d))
                    draw_node(ax, d, where(d), hot=1.0 if d["id"] in hops[k] else 0.0)
            if what == "in":
                a = ease(u)

                for d in B:                  # what the rule created appears in its place
                    new = d["id"] not in back
                    par = d["parent"]
                    if par is not None and not (not new and par["id"] in back and kept_edge(A[back[d["id"]]])):
                        draw_edge(ax, par["pos"], d["pos"], alpha=a, hot=1.0)
                    if new:
                        draw_node(ax, d, d["pos"], alpha=a, hot=1.0)

    # the steps, the size of the stored tree, the expression, and the card in the clerk's hand
    xs = np.cumsum([0.0] + [0.3 + 0.082 * len(st["name"]) for st in stages])
    x0 = 0.5 * RW_W - 0.5 * xs[-1]
    for i, st in enumerate(stages):
        on = i == shown
        ax.text(x0 + 0.5 * (xs[i] + xs[i + 1]), RW_H - 0.2, st["name"], ha="center", va="center", fontsize=9.5,
                fontfamily=SANS, color=INDIGO if on else MUTED, fontweight="semibold" if on else "normal")
        if i:
            ax.text(x0 + xs[i], RW_H - 0.2, "›", ha="center", va="center", fontsize=10, fontfamily=SANS, color=MUTED)
    cx = x0 + 0.5 * (xs[shown] + xs[shown + 1])
    half_w = 0.5 * (xs[shown + 1] - xs[shown]) - 0.14
    ax.plot([cx - half_w, cx + half_w], [RW_H - 0.36] * 2, color=INDIGO, lw=1.6, solid_capstyle="round")
    ax.text(RW_W - 0.15, RW_H - 0.66, f"{stages[shown]['count']} nodes", ha="right", va="center", fontsize=9.5,
            fontfamily=SANS, color=INK)
    ax.text(RW_W - 0.15, RW_H - 0.86, "in the stored tree", ha="right", va="center", fontsize=9, fontfamily=SANS,
            color=MUTED)
    for i, (face, edge, text) in enumerate([(PERIWINKLE, INDIGO, "point on $S^2$"), ("#cfe3e0", TEAL, "tangent vector"),
                                            (SAND, "#8a7340", "constant"), (PAPER, INK, "input")]):
        yy = RW_H - 0.62 - 0.2 * i
        ax.add_patch(FancyBboxPatch((0.15, yy - 0.07), 0.24, 0.14, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=face, ec=edge, lw=0.9, zorder=3))
        ax.text(0.47, yy - 0.005, text, ha="left", va="center", fontsize=9, fontfamily=SANS, color=INK)
    ax.text(0.5 * RW_W, 0.56, stages[shown]["formula"], ha="center", va="center", fontsize=11.5, color=INK)
    card = stages[k + 1]["card"] if what in ("announce", "out", "move", "in") else stages[k]["card"]
    tone = TERRACOTTA if hot_caption else INK
    if what == "show" and k == len(stages) - 1:
        card, tone = "what is left multiplies $\\xi$: the equation of motion, $E_\\xi = 0$", INDIGO
    ax.text(0.5 * RW_W, 0.2, card, ha="center", va="center", fontsize=10, fontfamily=SANS, color=tone)
    return fig


def rewrite():
    stages, matches, hops, plan = cached("rewrite", rewrite_data)
    n = sum(p[2] for p in plan) + 1
    settle = [i for i, p in enumerate(plan) if p[0] == "settle"]
    stills = [rewrite_time(plan, 0, 0.5)] + [rewrite_time(plan, i, 0.2) for i in settle]
    animate(rewrite_frame, OUT / "ch05-rewrite.gif", OUT / "ch05-rewrite-strip.png", n=n, fps=11, width_px=740,
            stills=stills, strip_px=1000, strip_cols=2, hold=3.0)


# ------------------------------------- the planted sign error, flown and checked
SIGN_SLOW, SIGN_FPS = 2.5, 12


def sci(x):
    """4.1e-08 as mathtext."""
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def sign_runs():
    """The load equation with each sign, checked against the Lagrangian and then flown from the same hover.

    Thrust is held at the total weight. Two open-loop pitch pulses (tilt forward, then tilt back
    to brake) set the load swinging. The same inputs go to both models.
    """
    L, dW, variables, inputs, eom, sym = quadrotor_load()
    mQ, mL, g, l, e3, xL, q, R, f, omega = (sym[k] for k in "mQ mL g l e3 xL q R f omega".split())
    aL, domega = xL.t_diff().t_diff(), omega.t_diff()

    def load_equation(sign):
        return ((aL + e3 * g) * (mQ + mL) * (-1) + Cross(domega, q) * (mQ * l)
                + q * (Dot(omega, omega) * mQ * l) * sign + (R * e3) * f)

    layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
    p = PARAMS
    weight = (p["m_Q"] + p["m_L"]) * p["g"]
    amp, T, gap = 0.12, 0.2, 0.4

    def u(t, _):
        tau = 0.0
        for t0, sg in ((1.0, 1), (1.0 + 2 * T + gap, -1), (1.0 + 4 * T + gap, -1), (1.0 + 6 * T + 2 * gap, 1)):
            if t0 <= t < t0 + T:
                tau = sg * amp
            elif t0 + T <= t < t0 + 2 * T:
                tau = -sg * amp
        return np.r_[weight, 0.0, tau, 0.0]

    x0 = layout.pack(x_L=np.zeros(3), q=[0.0, 0.0, -1.0], R=np.eye(3))
    out = {"t_pulse": 1.0, "t_level": 1.0 + 8 * T + 2 * gap}
    for sign in (-1, +1):
        hand = dict(eom)
        hand[str(xL.delta())] = (xL.delta(), load_equation(sign))
        plant = load_module(module_source(to_standard_form(hand, variables, inputs), layout), f"hand{sign + 1}")
        traj = simulate(plant, x0, 3.6, p, u=u)
        out[sign] = dict(t=traj.t, xL=traj["x_L"], q=traj["q"], R=traj["R"], om=traj["omega_q"],
                         xQ=traj["x_L"] - p["l"] * traj["q"],
                         check=check_eom(L, dW, variables, hand, inputs=inputs))
    # what the two equations differ by, evaluated along the flight of the flipped one
    wrong = out[+1]
    force = np.zeros(len(wrong["t"]))
    for k in range(0, len(force), 10):
        env = {"q": col(wrong["q"][k]), str(omega): col(wrong["om"][k]), "e3": col(E3), "R": wrong["R"][k],
               "f": weight, str(aL): col(np.zeros(3)), str(domega): col(np.zeros(3)), "m_Q": p["m_Q"],
               "m_L": p["m_L"], "l": p["l"], "g": p["g"]}
        force[k:k + 10] = np.linalg.norm(evaluate(load_equation(+1), dict(env)) - evaluate(load_equation(-1), dict(env)))
    out["force"] = force
    # the derivation as a page: the terms of the variation of the action before and after integration by parts
    tr = EOMTrace()
    compute_eom(L, dW, variables, trace=tr)
    lines = []
    for stage in ("simplify", "ibp"):
        for term in tr[stage].nodes:
            tex = to_latex(term)
            for a, b in (("\\omega_{q}", "\\omega"), ("\\xi_{q}", "\\xi"), ("\\eta_{R}", "\\eta"),
                         ("\\Omega_{R}", "\\Omega"), ("e3", "e_3"), ("\\delta ", "\\mathrm{\\delta}")):
                tex = tex.replace(a, b)
            sign_, body = ("-", tex[1:]) if tex.startswith("-") else ("+", tex)
            lines.append(dict(sign=sign_, body=body, stage=stage,
                              disputed=stage == "ibp" and str(term).count("Dot(\\omega_{q},\\omega_{q})") == 1))
    assert sum(ln["disputed"] for ln in lines) == 1
    out["lines"] = lines
    return out


def draw_vehicle(s, run, k, color, zorder, arm=0.3, load=6.5):
    s.curve(np.array([run["xQ"][k], run["xL"][k]]), color=color, lw=1.1, alpha=1.0, zorder=zorder)
    quadrotor(s, run["xQ"][k], run["R"][k], arm=arm, color=color, zorder=zorder + 1)
    s.dot(run["xL"][k], color=color, size=load, zorder=zorder + 2)


def flight_frame(t):
    D = cached("sign", sign_runs)
    right, wrong = D[-1], D[+1]
    t0, t1 = 0.4, 3.3
    now = t0 + (t1 - t0) * phase(t, 0.0, 0.93)
    k = int(round(now * 1000))
    s = Scene(width=7.0, height=4.8, azim=-90, elev=8, xlim=(-1.25, 5.6), ylim=(-3.15, 1.55))
    a = int(t0 * 1000)
    s.curve(wrong["xL"][a:k + 1], color=TERRACOTTA, lw=1.4, alpha=1.0, zorder=16)
    s.curve(right["xL"][a:k + 1], color=INDIGO, lw=1.6, alpha=1.0, zorder=17)
    draw_vehicle(s, wrong, k, TERRACOTTA, 24)
    draw_vehicle(s, right, k, INK, 34)
    s.dot(right["xL"][k], color=INDIGO, size=6.5, zorder=37)
    split = np.linalg.norm(wrong["xL"][k] - right["xL"][k])
    s.label(right["xQ"][k], "sign as derived", dx=0, dy=17, size=9.5, fontfamily=SANS, color=INK)
    if split > 0.45:
        s.label(wrong["xQ"][k], "sign flipped", dx=-44, dy=4, size=9.5, fontfamily=SANS, color=TERRACOTTA)
    force = D["force"][k]
    if force > 0.25:                                           # the invented force, drawn at the load it acts on
        tip = wrong["xL"][k] + 0.11 * force * wrong["q"][k]
        s.arrow(wrong["xL"][k], tip, color=TERRACOTTA, lw=1.6, head=0.1, zorder=30)
        if force > 1.2:
            s.label(tip, r"$2\,m_Q l\,\|\omega\|^2 q$", dx=0, dy=-11, size=10.5, color=TERRACOTTA)

    # the size of the invented force, against the weight of the load
    bx, by, scale = 3.85, -1.75, 0.3                           # scene units per newton
    w_load = PARAMS["m_L"] * PARAMS["g"]
    s.ax.plot([bx - 0.2, bx + 0.2], [by, by], color=INK, lw=0.9, zorder=20)
    s.ax.fill([bx - 0.11, bx + 0.11, bx + 0.11, bx - 0.11], [by, by, by + scale * force, by + scale * force],
              color=TERRACOTTA, zorder=19, lw=0)
    s.ax.plot([bx - 0.2, bx + 0.2], [by + scale * w_load] * 2, color=MUTED, lw=0.8, linestyle=DASH, zorder=21)
    s.text2d(bx + 0.28, by + scale * w_load, f"weight of the load, {w_load:.1f} N", fontsize=9.5, color=MUTED,
             ha="left", fontfamily=SANS)
    s.text2d(bx + 0.28, by, "0", fontsize=9.5, color=MUTED, ha="left", fontfamily=SANS)
    s.text2d(bx - 0.3, by - 0.24, f"force the wrong sign invents: {force:.1f} N", fontsize=9.5, color=TERRACOTTA,
             ha="left", fontfamily=SANS)
    # what the check said about each equation, before either flew
    s.text2d(3.55, 1.3, "the check, at a random state", fontsize=9.5, color=MUTED, ha="left", fontfamily=SANS)
    s.text2d(3.55, 1.02, "sign as derived:", fontsize=9.5, color=INDIGO, ha="left", fontfamily=SANS)
    s.text2d(5.25, 1.02, sci(right["check"]), fontsize=10.5, color=INDIGO, ha="right")
    s.text2d(3.55, 0.76, "sign flipped:", fontsize=9.5, color=TERRACOTTA, ha="left", fontfamily=SANS)
    s.text2d(5.25, 0.76, f"${wrong['check']:.1f}$", fontsize=10.5, color=TERRACOTTA, ha="right")
    s.text2d(5.45, -2.4, f"t = {now:.1f} s, shown {SIGN_SLOW:g} times slower", fontsize=9.5, color=MUTED,
             ha="right", fontfamily=SANS)
    if now < D["t_pulse"]:
        msg, colr = "hover: the cable hangs still, the disputed term is zero, and the two models fly as one", INK
    elif split < 0.3:
        msg, colr = "pitch pulses, the same for both models, set the load swinging", INK
    else:
        msg, colr = "the load swings: the flipped sign invents a force along the cable, and its model falls away", TERRACOTTA
    s.text2d(2.17, -2.9, msg, fontsize=10, color=colr, fontfamily=SANS)
    return s


def sign_flight():
    n = int(round((3.3 - 0.4) * SIGN_SLOW * SIGN_FPS / 0.93)) + 1
    animate(flight_frame, OUT / "ch05-sign-flight.gif", OUT / "ch05-sign-flight-strip.png", n=n, fps=SIGN_FPS,
            width_px=720, stills=(0.1, 0.5, 0.72, 0.93), strip_px=900, strip_cols=2, hold=2.0)


# ------------------------------------------------------- opening illustration
def opening_frame(t):
    """The hook: one sign in a page of algebra, two flights that agree in hover, and a check that needs no flight."""
    D = cached("sign", sign_runs)
    right, wrong, lines = D[-1], D[+1], D["lines"]
    s = Scene(width=7.2, height=4.07, azim=-90, elev=8, xlim=(-5.6, 3.5), ylim=(-3.55, 1.6))
    ax = s.ax
    sans = dict(fontfamily=SANS, fontsize=9.5)

    # 1. the page
    x0, x1, y0, y1 = -5.42, -1.45, -1.12, 1.48
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06", fc="#f6f3ea",
                                ec=INK, lw=0.9, zorder=2))
    ax.text(x0 + 0.14, y1 - 0.2, "$\\mathrm{\\delta}S$ of the quadrotor and its load, term by term", ha="left",
            va="center", color=MUTED, zorder=5, **sans)
    target = next(i for i, ln in enumerate(lines) if ln["disputed"])
    pitch, top, rows = 0.262, y1 - 0.52, 8
    scroll = (target - 4) * ease(phase(t, 0.02, 0.22))
    flip = ease(phase(t, 0.25, 0.30))
    for i, ln in enumerate(lines):
        r = i - scroll
        if r < -0.3 or r > rows - 0.6:
            continue
        alpha = float(np.clip(min(r + 0.3, rows - 0.6 - r) / 0.4, 0, 1))
        y = top - pitch * r
        hot = ln["disputed"] and t > 0.2
        colr = mix(INK, TERRACOTTA, ease(phase(t, 0.2, 0.25))) if hot else INK
        if hot:
            ax.text(x0 + 0.2, y, "$-$", ha="center", va="center", fontsize=10.5, color=colr, alpha=1 - flip, zorder=6)
            ax.text(x0 + 0.2, y, "$+$", ha="center", va="center", fontsize=10.5, color=colr, alpha=flip, zorder=6)
            ax.add_patch(FancyBboxPatch((x0 + 0.07, y - 0.115), 0.26, 0.23, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc="none", ec=TERRACOTTA, lw=1.3, alpha=ease(phase(t, 0.2, 0.25)), zorder=6))
        else:
            ax.text(x0 + 0.2, y, f"${ln['sign']}$", ha="center", va="center", fontsize=9, color=colr,
                    alpha=0.8 * alpha, zorder=5)
        ax.text(x0 + 0.42, y, f"${ln['body']}$", ha="left", va="center", fontsize=10.5 if hot else 9, color=colr,
                alpha=1.0 if hot else 0.8 * alpha, zorder=5)
    if flip > 0:
        ax.text(x1 - 0.14, top - pitch * 4, "copied with the wrong sign", ha="right", va="center", color=TERRACOTTA,
                alpha=flip, zorder=6, **sans)

    # 2. two flights from the same hover with the same inputs
    ta, tb = 0.6, 3.3
    now = ta + (tb - ta) * phase(t, 0.32, 0.86)
    k = int(round(now * 1000))
    a = int(ta * 1000)
    s.curve(wrong["xL"][a:k + 1], color=TERRACOTTA, lw=1.4, alpha=1.0, zorder=16)
    s.curve(right["xL"][a:k + 1], color=INDIGO, lw=1.6, alpha=1.0, zorder=17)
    draw_vehicle(s, wrong, k, TERRACOTTA, 24, arm=0.33)
    draw_vehicle(s, right, k, INK, 34, arm=0.33)
    s.dot(right["xL"][k], color=INDIGO, size=6.5, zorder=37)
    split = np.linalg.norm(wrong["xL"][k] - right["xL"][k])
    s.label(right["xQ"][k], "sign as derived", dx=0, dy=17, size=9.5, fontfamily=SANS, color=INK)
    if split > 0.45:
        s.label(wrong["xQ"][k], "sign flipped", dx=-44, dy=4, size=9.5, fontfamily=SANS, color=TERRACOTTA)
    if 0.32 < t:
        ax.text(3.38, 1.4, f"t = {now:.1f} s, shown 2 times slower", ha="right", va="center", color=MUTED, **sans)

    # 3. the check
    lit = ease(phase(t, 0.88, 0.97))
    cy, lo, hi = -1.62, -3.7, -1.75                     # the error axis runs from 1e-9 to 10, a decade per 0.195
    xe = lambda err: lo + (np.log10(err) + 9) / 10 * (hi - lo)
    ax.text(x0, cy, "the check: error of each equation against the Lagrangian", ha="left", va="center",
            color=INK if lit > 0 else MUTED, zorder=5, **sans)
    for row, (run, name, colr) in enumerate(((right, "sign as derived", INDIGO), (wrong, "sign flipped", TERRACOTTA))):
        y = cy - 0.42 - 0.4 * row
        ax.text(x0, y, name, ha="left", va="center", color=colr if lit > 0 else MUTED, zorder=5, **sans)
        ax.plot([lo, hi], [y, y], color=MUTED, lw=0.6, alpha=0.6, zorder=3)
        if lit > 0:
            end = lo + (xe(run["check"]) - lo) * lit
            ax.fill([lo, end, end, lo], [y - 0.07, y - 0.07, y + 0.07, y + 0.07], color=colr, lw=0, zorder=4)
            val = sci(run["check"]) if run["check"] < 1e-3 else f"${run['check']:.1f}$"
            ax.text(hi + 0.12, y, val, ha="left", va="center", fontsize=10.5, color=colr, alpha=lit, zorder=5)
    xt = xe(1e-5)
    ax.plot([xt, xt], [cy - 0.25, cy - 0.99], color=INK, lw=0.8, linestyle=DASH, zorder=6)
    ax.text(xt, cy - 1.18, "accept below $10^{-5}$", ha="center", va="center", color=INK if lit > 0 else MUTED,
            zorder=5, **sans)

    if t < 0.2:
        msg, colr = "a derivation by hand runs to pages of terms like these", INK
    elif t < 0.32:
        msg, colr = "copy one sign wrong, and nothing on the page objects", TERRACOTTA
    elif now < D["t_pulse"]:
        msg, colr = "fly both models: in hover the cable hangs still and they agree exactly", INK
    elif split < 0.3 and t < 0.88:
        msg, colr = "then the same pitch pulses for both set the load swinging", INK
    elif t < 0.88:
        msg, colr = "only when the load swings does the model with the flipped sign fall away", TERRACOTTA
    else:
        msg, colr = "the check finds it with no flight at all: seven orders of magnitude apart", INDIGO
    ax.text(-1.05, -3.28, msg, ha="center", va="center", fontsize=10, color=colr, fontfamily=SANS, zorder=50)
    return s


def opening():
    animate(opening_frame, OUT / "ch05-opening.gif", OUT / "ch05-opening-strip.png", n=121, fps=12, width_px=740,
            stills=(0.12, 0.31, 0.75, 1.0), strip_px=900, strip_cols=2, hold=2.5)


# ------------------------------------------------ animation: pushing along the rod
def rod_frame(t):
    """A push of fixed size on the bob, turned from across the rod to straight along it, and split at q."""
    s = Scene(width=6.9, height=4.0, azim=-52, elev=17, xlim=(-2.25, 3.55), ylim=(-1.42, 1.942))
    tilt = np.radians(58)
    q = unit(np.cos(tilt) * s.view + np.sin(tilt) * (0.42 * s.right + 0.91 * s.up))
    d = unit(np.cross(q, np.cross(-0.85 * s.right - 0.5 * s.up, q)))       # a direction across the rod
    alpha = 0.5 * np.pi * ease(phase(t, 0.1, 0.84))
    F = 1.05 * (np.cos(alpha) * d - np.sin(alpha) * q)                     # at the end it points at the pivot
    Fn, Ft = (F @ q) * q, F - (F @ q) * q
    s.shadow((0, 0, -1.08), 1.0)
    s.sphere((0, 0, 0), 1.0, color=PERIWINKLE)
    s.curve(np.array([[0, 0, 0], q]), color=INK, lw=1.2, hidden=True)      # the rod, inside the sphere
    s.dot((0, 0, 0), color=INK, size=4, hollow=True, zorder=13)
    s.tangent_plane(q, size=0.8, spin=0.25, color=SAND, alpha=0.82)
    tail = q - F                                                           # every arrow ends on the bob: a push
    if np.linalg.norm(Fn) > 0.03 and np.linalg.norm(Ft) > 0.03:
        s.curve(np.array([q - Ft, tail]), color=MUTED, lw=0.8, hidden=True, zorder=29)
        s.curve(np.array([q - Fn, tail]), color=MUTED, lw=0.8, hidden=True, zorder=29)
    s.arrow(q - Fn, q, color=TERRACOTTA, lw=1.7)
    s.arrow(q - Ft, q, color=INDIGO, lw=1.9)
    s.arrow(tail, q, color=INK, lw=1.2, zorder=31)
    s.dot(q, color=INK, size=6, zorder=40)
    s.label(q, "$q$", dx=12, dy=-9, size=12)
    s.label(tail, "$F$", dx=9, dy=8, size=12)
    s.text2d(-1.25, 1.76, "pivot at the center, bob at $q$", fontsize=9.5, fontfamily=SANS, color=MUTED)
    # the two parts as bars, in units of the push
    bx, w = 1.6, 1.4
    for y, part, colr, name in ((1.05, np.linalg.norm(Ft), INDIGO, "part across the rod: moves the bob"),
                                (0.35, np.linalg.norm(Fn), TERRACOTTA, "part along the rod: the rod takes it")):
        frac = part / np.linalg.norm(F)
        s.text2d(bx, y + 0.24, name, fontsize=9.5, fontfamily=SANS, color=colr, ha="left")
        s.ax.plot([bx, bx + w], [y, y], color=MUTED, lw=0.6, alpha=0.6, zorder=20)
        s.ax.fill([bx, bx + w * frac, bx + w * frac, bx], [y - 0.07, y - 0.07, y + 0.07, y + 0.07], color=colr, lw=0,
                  zorder=21)
        s.text2d(bx + w + 0.1, y, f"{100 * frac:.0f}%", fontsize=9.5, fontfamily=SANS, color=colr, ha="left")
    s.text2d(bx, -0.25, "tangent plane at $q$: the only\ndirections the bob can move in", fontsize=9.5, fontfamily=SANS,
             color=MUTED, ha="left", linespacing=1.3)
    if t < 0.1:
        msg, colr = "push the bob across the rod: the whole push moves it", INDIGO
    elif t < 0.86:
        msg, colr = "turn the same push toward the rod: only its tangent part still moves the bob", INK
    else:
        msg, colr = "push straight along the rod: nothing moves, the rod takes it all", TERRACOTTA
    s.text2d(0.65, -1.25, msg, fontsize=10, color=colr, fontfamily=SANS)
    return s


def rod():
    animate(rod_frame, OUT / "ch05-rod.gif", OUT / "ch05-rod-strip.png", n=78, fps=12, width_px=700,
            stills=(0.0, 0.42, 0.6, 1.0), strip_px=900, strip_cols=2, hold=2.0)


if __name__ == "__main__":
    opening()
    tree()
    pipeline()
    rewrite()
    rod()
    tangent()
    sign_error()
    sign_flight()
    quadload()
    chain()
