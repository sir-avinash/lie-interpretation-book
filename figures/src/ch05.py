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
from figkit import (INDIGO, INK, LW_CONTOUR, LW_SILHOUETTE, LW_VECTOR, MUTED, OCHRE, PAPER, PEACH,  # noqa: E402
                    PERIWINKLE, SAND, TEAL, TERRACOTTA, Scene, halo, new_plot, plt, save, unit)

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


def quadload():
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

    s = Scene(width=6.6, height=3.3, azim=-78, elev=17)
    floor = xLs[:, 2].min() - 0.55
    s.curve(np.c_[xLs[:, :2], np.full(len(xLs), floor)], color=MUTED, lw=0.8, alpha=0.5)
    s.curve(com[[0, -1]], color=MUTED, lw=0.8, hidden=True)
    s.ribbons([(xLs[::10], INDIGO, 1.5), (xQs[::10], TERRACOTTA, 1.5)], gap=1.8)
    ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    for k, idx in enumerate([0, 1500, 3000, 4500, 6000]):
        xq, xl = xQs[idx], xLs[idx]
        s.shadow((xl[0], xl[1], floor), 0.16, strength=0.3)
        s.curve(np.array([xq, xl]), color=INK, lw=1.1, alpha=1.0, zorder=32)
        for arm in (ex, ey):
            s.curve(np.array([xq - 0.17 * arm, xq + 0.17 * arm]), color=INK, lw=1.7, alpha=1.0, zorder=33)
            for sg in (-1, 1):
                s.curve(circle3(xq + sg * 0.17 * arm, ex, ey, 0.075), color=INK, lw=0.9, alpha=1.0, zorder=33)
        s.dot(xl, color=INDIGO, size=6, zorder=34)
        s.arrow(xq, xq + 0.36 * E3, color=TEAL, lw=1.3, head=0.08, zorder=31)
        s.label(xq + 0.36 * E3, f"{traj.t[idx]:.1f} s", dx=0, dy=9, size=9.5, fontfamily=SANS, color=INK)
    s.label(com[-1], "center of mass", dx=44, dy=0, size=9.5, fontfamily=SANS, color=MUTED)
    s.label(xQs[0], "quadrotor $x_Q$", dx=-50, dy=-12, size=9.5, fontfamily=SANS, color=TERRACOTTA)
    s.label(xLs[0], "load $x_L$", dx=-32, dy=15, size=9.5, fontfamily=SANS, color=INDIGO)
    s.label(xQs[-1] + 0.2 * E3, "thrust $f R e_3$", dx=42, dy=6, size=9.5, fontfamily=SANS, color=TEAL)
    s.label(np.r_[xLs[3000, :2], floor], "shadow of the load path", dx=0, dy=-27, size=9.5, fontfamily=SANS,
            color=MUTED)
    lo = s.project(xQs[0])[0]
    hi = s.project(com[-1])[0]
    low = s.project(np.r_[xLs[3000, :2], floor])[0]
    s._pts.append(np.array([[lo[0] - 1.2, low[1] - 0.42], [hi[0] + 1.3, hi[1]]]))
    s.save(OUT / "ch05-quadload.svg", pad=0.1)


# ----------------------------------------------------------------- n-link chain
def chain():
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

    s = Scene(width=6.4, height=3.6, azim=-64, elev=16)
    times = [0, 170, 320, 450, 570, 700, 860]
    last = times[-1]
    floor = -1.22
    s.curve(np.c_[joints[:last:4, -1, :2], np.full(len(joints[:last:4]), floor)], color=MUTED, lw=0.8, alpha=0.5)
    s.curve(np.array([[0, 0, 0], [0, 0, floor]]), color=MUTED, lw=0.7, hidden=True)
    s.curve(joints[:last:4, -1], color=INDIGO, lw=1.0, alpha=0.6)
    for k, idx in enumerate(times):
        shade = 0.22 + 0.78 * k / (len(times) - 1)
        colr = tuple(1 - shade * (1 - c) for c in (0.10, 0.10, 0.10))
        s.curve(joints[idx], color=colr, lw=1.9 if idx == last else 1.4, alpha=1.0, zorder=20 + k)
        for jt in joints[idx, 1:]:
            s.dot(jt, color=colr, size=4.2 if idx == last else 3.4, zorder=21 + k)
    s.dot((0, 0, 0), color=INK, size=5, hollow=True, zorder=60)
    s.label((0, 0, 0), "pivot", dx=-4, dy=11, size=9.5, fontfamily=SANS)
    s.label(joints[0, -1], "$t = 0$", dx=4, dy=11, size=9.5, fontfamily=SANS, color=MUTED)
    s.label(joints[last, -1], f"$t = {traj.t[last]:.2f}$ s", dx=-30, dy=-8, size=9.5, fontfamily=SANS)
    s.label(0.5 * (joints[0, 1] + joints[0, 2]), "$q_2$", dx=0, dy=10, size=11, color=MUTED)
    s.label(joints[320, -1], "path of mass 4", dx=40, dy=-6, size=9.5, fontfamily=SANS, color=INDIGO)
    s.label(np.r_[joints[570, -1, :2], floor], "its shadow on the floor", dx=-10, dy=-15, size=9.5, fontfamily=SANS,
            color=MUTED)
    lo = s.project(joints[0, -1])[0]
    s._pts.append(np.array([[lo[0] + 0.5, lo[1]], [lo[0] - 2.6, lo[1]]]))
    s.save(OUT / "ch05-chain.svg", pad=0.14)


if __name__ == "__main__":
    tree()
    pipeline()
    tangent()
    sign_error()
    quadload()
    chain()
