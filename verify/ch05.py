"""Chapter 5: every listing, every claim about geomech and every quoted number.

    python verify/ch05.py

The blocks between "# --- listing" and "# --- end" are the code printed in the chapter.
"""
import contextlib
import dataclasses
import io
import logging
import time

import numpy as np

import geomech
from common import Checker

ck = Checker("ch05")
rng = np.random.default_rng(5)
ck.record("geomech_version", geomech.__version__)


def captured(fn, *a, **kw):
    """Run fn and return (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def error_text(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__, str(exc)
    return None, ""


# =========================================================================
# 1. The spherical pendulum, stage by stage
# =========================================================================
# --- listing: pendulum
from geomech import *

half = Scalar("0.5", value=0.5, attr=["Constant"])
m, g, l = getScalars("m g l", attr=["Constant"])
e3 = Vector("e3", attr=["Constant"])
q, tau = S2("q"), Vector("\\tau")               # rod direction, torque at the pivot
omega, xi = q.get_tangent_vector(), q.get_variation_vector()

L = half * m * l * l * Dot(omega, omega) - m * g * l * Dot(q, e3)
dW = Dot(xi, tau)
variables = SystemVariables(vectors=[q])

trace = EOMTrace()
eom = compute_eom(L, dW, variables, trace=trace)
# --- end

pend = dict(L=L, dW=dW, variables=variables, eom=eom, q=q, tau=tau, omega=omega, xi=xi)
_, summary = captured(trace.summary, chars=200)
print(summary)
stages = dict(trace.stages)
ck.true("pendulum: trace has 13 stages", len(trace.stages) == 13, f"{len(trace.stages)}")
nodes = {ln.split()[0]: int(ln.split()[1]) for ln in summary.splitlines() if " nodes " in ln}
for name, want in [("dS", 34), ("manifold_rules", 46), ("simplify", 27), ("ibp", 29), ("equation[\\xi_{q}]", 23)]:
    ck.true(f"pendulum: stage {name} has {want} nodes", nodes[name] == want, f"{nodes[name]}")
ck.record("pendulum_stage_nodes", nodes)

# What the kinematics of the leaf types are.
ck.true("q.t_diff() is omega x q", str(q.t_diff()) == "Cross(\\omega_{q},q)")
ck.true("q.delta() is xi x q", str(q.delta()) == "Cross(\\xi_{q},q)")
ck.true("the tangent vector is typed TS2", type(omega).__name__ == "TS2")
ck.true("L is an Add node with two operands", type(L).__name__ == "Add" and len(L.nodes) == 2)

# The stage strings quoted in the walk-through table.
tex = {k: to_latex(v) for k, v in stages.items()}
ck.true("stage dS keeps delta omega unexpanded", tex["dS"].count("\\delta \\omega_{q}") == 2)
ck.true("stage manifold_rules substitutes xi_dot - 2 omega x xi",
        "\\dot{\\xi_{q}} -2 \\left(\\omega_{q} \\times \\xi_{q}\\right)" in tex["manifold_rules"])
ck.true("stage simplify: m l l (xi_dot . omega) survives, the factor-2 term is gone",
        tex["simplify"].startswith("\\left(m l l \\left(\\dot{\\xi_{q}} \\cdot \\omega_{q}\\right)")
        and "\\times \\xi_{q}\\right)\\right)" not in tex["simplify"])
ck.true("stage ibp: -m l l (xi . omega_dot)",
        tex["ibp"].startswith("\\left(-m l l \\left(\\xi_{q} \\cdot \\dot{\\omega_{q}}\\right)"))
final = str(eom["\\xi_{q}"][1])
ck.true("pendulum equation string",
        final == "(\\frac{d}{dt}(\\omega_{q})*m*l*l*(-1)+Cross(q,e3)*m*g*l*(-1)+\\tau)", final)
_, printed = captured(print_eom, eom)
ck.true("print_eom gives the LaTeX integral form",
        printed.strip() == "\\int \\xi_{q} \\cdot \\left(-m l l \\dot{\\omega_{q}} -m g l \\left(q \\times e3\\right)"
                           " + \\tau\\right) \\, dt = 0", printed.strip())

# Each stage can be rerun by hand from the one before it.
ck.true("apply_manifold_rules(dS) reproduces the stage",
        str(apply_manifold_rules(stages["dS"], variables)) == str(stages["manifold_rules"]))
ck.true("full_simplify(manifold_rules) reproduces the stage",
        str(full_simplify(stages["manifold_rules"])) == str(stages["simplify"]))

# Simplification must not change the value of delta S (on the constraint set).
qn = rng.normal(size=3)
qn /= np.linalg.norm(qn)
wn = np.cross(qn, rng.normal(size=3))
xin = np.cross(qn, rng.normal(size=3))
col = lambda v: np.asarray(v, float).reshape(3, 1)  # noqa: E731
env = {"q": col(qn), str(omega): col(wn), str(xi): col(xin), str(xi.t_diff()): col(rng.normal(size=3)),
       str(omega.t_diff()): col(np.cross(qn, rng.normal(size=3))), "m": 1.3, "g": 9.81, "l": 0.7,
       "e3": col([0, 0, 1]), "\\tau": col(rng.normal(size=3))}
vals = [evaluate(stages[k], dict(env)) for k in ("manifold_rules", "simplify")]
ck.close("simplify preserves the value of delta S", vals[0], vals[1], tol=1e-12)
ck.close("omega . (omega x xi) = 0, the identity that kills the factor-2 term", wn @ np.cross(wn, xin), 0.0)
a_, b_, c_ = rng.normal(size=(3, 3))
ck.close("(a x b) . c = a . (b x c)", np.cross(a_, b_) @ c_, a_ @ np.cross(b_, c_))

# The rule log.
buf = io.StringIO()
handler = logging.StreamHandler(buf)
logging.getLogger("geomech.rules").addHandler(handler)
set_trace(True)
compute_eom(L, dW, variables)
set_trace(False)
logging.getLogger("geomech.rules").removeHandler(handler)
rule_lines = buf.getvalue().splitlines()
ck.true("pendulum: 44 rule firings are logged", len(rule_lines) == 44, f"{len(rule_lines)}")
ck.record("pendulum_rule_firings", len(rule_lines))
names = [ln.split(":")[0] for ln in rule_lines]
ck.true("rule manifold.s2_variation_omega fires twice", names.count("manifold.s2_variation_omega") == 2)
ck.true("rule vector.dot_cross_orthogonal fires twice", names.count("vector.dot_cross_orthogonal") == 2)
ck.true("rule ibp.dot fires once", names.count("ibp.dot") == 1)
ck.true("rule simplify.mvmul_hat_to_cross fires", "simplify.mvmul_hat_to_cross" in names)
ck.true("the logged S2 rule line",
        rule_lines[0] == "manifold.s2_variation_omega: δ(v:\\omega_{q}) ==> (d/dt(v:\\xi_{q}) + "
                         "cross(v:\\omega_{q}, v:\\xi_{q})·(-2))", rule_lines[0])

pend_err = check_eom(L, dW, variables, eom, inputs=[tau])
ck.true("pendulum: check_eom < 1e-5", pend_err < 1e-5, f"{pend_err:.1e}")
ck.record("pendulum_check_eom", pend_err)

# Nodes are frozen.
try:
    L.nodes = ()
    frozen = False
except dataclasses.FrozenInstanceError:
    frozen = True
ck.true("expression nodes are frozen dataclasses", frozen)

# Standard form of the pendulum: M = -(physical inertia).
sf_p = to_standard_form(eom, variables, [tau])["\\xi_{q}"]
ck.true("pendulum standard form: mass block prints I*m*l*l*(-1)",
        str(sf_p.M["\\frac{d}{dt}(\\omega_{q})"]) == "I*m*l*l*(-1)", str(sf_p.M["\\frac{d}{dt}(\\omega_{q})"]))

# The 3D pendulum of the exercises, and what an undeclared symmetry costs.
Jm = Matrix("J", attr=["Constant", "SymmetricMatrix"])
rho = Vector("\\rho", attr=["Constant"])
Rp, Mp = SO3("R"), Vector("\\tau")
Omp = Rp.get_tangent_vector()
Lp = half * Dot(Omp, Jm * Omp) - m * g * Dot(e3, Rp * rho)
Vp = SystemVariables(matrices=[Rp])
eom_p = compute_eom(Lp, Dot(Rp.get_variation_vector(), Mp), Vp)
s3 = str(eom_p["\\eta_{R}"][1])
ck.true("3D pendulum: -J Omega_dot - Omega x J Omega - m g rho x R^T e3 + M",
        s3 == "(J*\\frac{d}{dt}(\\Omega_{R})*(-1)+Cross(\\Omega_{R},J*\\Omega_{R})*(-1)"
              "+Cross(\\rho,(R)'*e3)*m*g*(-1)+\\tau)", s3)
ck.true("3D pendulum: check_eom < 1e-5",
        check_eom(Lp, Dot(Rp.get_variation_vector(), Mp), Vp, eom_p, inputs=[Mp]) < 1e-5)
Jn = Matrix("J", attr=["Constant"])
Ln = half * Dot(Omp, Jn * Omp)
eom_n = compute_eom(Ln, None, Vp)
ck.true("without SymmetricMatrix the result keeps J and J^T apart", "(J)'" in str(eom_n["\\eta_{R}"][1]))
ck.true("... and is still correct", check_eom(Ln, None, Vp, eom_n) < 1e-5)

# The S1 row of the leaf table: a planar pendulum about the axis e2.
e2 = Vector("e2", attr=["Constant", "UnitNorm"])
qc = S1("q", axis=e2)
wc, gam = qc.get_tangent_vector(), qc.get_variation_vector()
Lc = half * m * l * l * wc * wc - m * g * l * Dot(qc, e3)
Vc = SystemVariables(vectors=[qc])
eom_c = compute_eom(Lc, None, Vc)
ck.true("S1: q.t_diff() is omega (e x q), with a scalar omega",
        str(qc.t_diff()) == "Cross(e2,q)*\\omega_{q}" and type(wc).__name__ == "TS1", str(qc.t_diff()))
ck.true("S1: q.delta() is gamma (e x q)", str(qc.delta()) == "Cross(e2,q)*\\gamma_{q}", str(qc.delta()))
ck.true("S1: planar pendulum gives one scalar equation that passes the check",
        list(eom_c) == ["\\gamma_{q}"] and check_eom(Lc, None, Vc, eom_c) < 1e-5, str(eom_c["\\gamma_{q}"][1]))

# =========================================================================
# 2. Failure messages quoted in the chapter
# =========================================================================
def no_constant():
    e3_ = Vector("e3")
    L_ = half * m * l * l * Dot(omega, omega) - m * g * l * Dot(q, e3_)
    compute_eom(L_, None, variables)


kind, text = error_text(no_constant)
ck.true("forgotten Constant raises ValueError naming the symbol",
        kind == "ValueError" and text.startswith(
            "compute_eom: δS still contains the variation symbols '\\delta{e3}' of the symbols 'e3', which are "
            "neither entries of SystemVariables nor declared in params"), text[:90])

kind, text = error_text(lambda: Scalar("m", attr=["constant"]))
ck.true("attribute typo raises ValueError",
        kind == "ValueError" and text.startswith("attr: unknown attribute 'constant' for symbol 'm'"), text[:60])

kind, text = error_text(lambda: compute_eom(L, None, SystemVariables(vectors=[omega])))
ck.true("listing omega as a variable raises TypeError",
        kind == "TypeError" and "is a tangent or variation vector, not a configuration variable; list the manifold "
                                "point 'q' instead" in text, text[:60])


def second_order():
    x_ = Vector("x")
    a_ = x_.t_diff().t_diff()
    compute_eom(half * m * Dot(a_, a_), None, SystemVariables(vectors=[x_]))


kind, text = error_text(second_order)
ck.true("acceleration-dependent Lagrangian raises AlgebraicError",
        kind == "AlgebraicError" and "the Lagrangian depends on accelerations" in text
        and "higher-order Lagrangians are not supported" in text, text[:60])

x_ = Vector("x")
v_ = x_.t_diff()
Lq = half * m * Dot(v_, v_) * Dot(v_, v_)
eq4 = compute_eom(Lq, None, SystemVariables(vectors=[x_]))
ck.true("a quartic-in-velocity Lagrangian is accepted and correct",
        check_eom(Lq, None, SystemVariables(vectors=[x_]), eq4) < 1e-5)

kind, text = error_text(lambda: to_standard_form(eom, variables, [g]))
ck.true("a constant listed as input raises ValueError",
        kind == "ValueError" and text == "to_standard_form: input 'g' is constant: its variation vanishes, so it "
                                         "cannot be an input", text)

# =========================================================================
# 3. Quadrotor with a cable-suspended load
# =========================================================================
t0 = time.perf_counter()
# --- listing: quadrotor
from geomech import *

half = Scalar("0.5", value=0.5, attr=["Constant"])
mQ, mL, g, l = getScalars("m_Q m_L g l", attr=["Constant"])
J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
e3 = Vector("e3", attr=["Constant"])
xL, q, R = Vector("x_L"), S2("q"), SO3("R")     # load, cable direction, attitude
f, tau = Scalar("f"), Vector("\\tau")          # thrust, torque
omega, xi = q.get_tangent_vector(), q.get_variation_vector()
Omega, eta = R.get_tangent_vector(), R.get_variation_vector()

xQ = xL - q * l                                 # quadrotor: one cable length up
vQ, vL = xQ.t_diff(), xL.t_diff()
L = (half * mQ * Dot(vQ, vQ) + half * mL * Dot(vL, vL)
     + half * Dot(Omega, J * Omega) - mQ * g * Dot(e3, xQ) - mL * g * Dot(e3, xL))
dW = Dot(xL.delta() - Cross(xi, q) * l, (R * e3) * f) + Dot(eta, tau)

variables, inputs = SystemVariables(vectors=[xL, q], matrices=[R]), [f, tau]
eom = compute_eom(L, dW, variables)
err = check_eom(L, dW, variables, eom, inputs=inputs)
# --- end
elapsed = time.perf_counter() - t0
ck.true("quadrotor-load: derivation and check take under a second", elapsed < 1.0, f"{elapsed:.2f} s")
ck.true("quadrotor-load: check_eom < 1e-5", err < 1e-5, f"{err:.1e}")
ck.record("quadload_check_eom", err)

kx, kq, kR = "\\delta{x_L}", "\\xi_{q}", "\\eta_{R}"
ck.true("quadrotor-load: three equations, keyed by variation", list(eom) == [kx, kq, kR])
sx, sq, sR = (str(eom[k][1]) for k in (kx, kq, kR))
ck.true("load equation string", sx == (
    "(\\frac{d}{dt}(\\frac{d}{dt}(x_L))*m_Q*(-1)+Cross(\\frac{d}{dt}(\\omega_{q}),q)*m_Q*l"
    "+q*Dot(\\omega_{q},\\omega_{q})*m_Q*l*(-1)+\\frac{d}{dt}(\\frac{d}{dt}(x_L))*m_L*(-1)"
    "+e3*m_Q*g*(-1)+e3*m_L*g*(-1)+R*e3*f)"), sx)
ck.true("cable equation has seven terms", len(eom[kq][1].nodes) == 7, f"{len(eom[kq][1].nodes)}")
ck.true("attitude equation string",
        sR == "(J*\\frac{d}{dt}(\\Omega_{R})*(-1)+Cross(\\Omega_{R},J*\\Omega_{R})*(-1)+\\tau)", sR)

worst = [check_eom(L, dW, variables, eom, inputs=inputs, seed=s) for s in range(20)]
ck.true("correct equation: error below 1e-6 for 20 seeds", max(worst) < 1e-6, f"max {max(worst):.1e}")
ck.record("quadload_check_eom_20_seeds", [min(worst), max(worst)])

# ---- the planted sign error ----------------------------------------------
# --- listing: sign error
aL, domega = xL.t_diff().t_diff(), omega.t_diff()


def load_equation(sign):            # the x_L row, typed in from a page of algebra
    return ((aL + e3 * g) * (mQ + mL) * (-1) + Cross(domega, q) * (mQ * l)
            + q * (Dot(omega, omega) * mQ * l) * sign + (R * e3) * f)


for sign in (-1, +1):
    hand = dict(eom)
    hand[str(xL.delta())] = (xL.delta(), load_equation(sign))
    print(sign, check_eom(L, dW, variables, hand, inputs=inputs))
# --- end
errs = {}
for sign in (-1, +1):
    hand = dict(eom)
    hand[kx] = (xL.delta(), load_equation(sign))
    errs[sign] = check_eom(L, dW, variables, hand, inputs=inputs)
    seeds = [check_eom(L, dW, variables, hand, inputs=inputs, seed=s) for s in range(20)]
    ck.record(f"hand_equation_sign_{sign:+d}_20_seeds", [min(seeds), max(seeds)])
    if sign == +1:
        ck.true("planted sign error: error above 0.1 for all 20 seeds", min(seeds) > 0.1, f"min {min(seeds):.2f}")
ck.true("hand equation with the right sign passes", errs[-1] < 1e-5, f"{errs[-1]:.1e}")
ck.true("hand equation with the planted sign fails by seven orders", errs[+1] > 1.0, f"{errs[+1]:.2f}")
ck.record("planted_sign_error_check_eom", errs[+1])
hand[kx] = (xL.delta(), load_equation(+1))
kind, text = error_text(lambda: assert_eom(L, dW, variables, hand, inputs=inputs))
ck.true("assert_eom names the wrong equation",
        kind == "AssertionError" and text.startswith(
            "equation for '\\delta{x_L}' disagrees with the Euler-Lagrange residual: relative error 1.903e+00"), text)

# What the wrong sign does physically: zero at hover, 2 m_Q l |omega|^2 along the cable otherwise.
pn = {"m_Q": 1.5, "m_L": 0.3, "l": 1.0, "g": 9.81}


def gap(w_mag):
    qv = np.array([np.sin(0.4), 0.0, -np.cos(0.4)])
    wv = w_mag * np.array([0.0, 1.0, 0.0])
    env_ = {"q": col(qv), str(omega): col(wv), "e3": col([0, 0, 1]), "R": np.eye(3), "f": 17.0,
            str(aL): col([0.1, 0.2, 0.3]), str(domega): col([0.0, 0.5, 0.0]), **pn}
    return (evaluate(load_equation(+1), dict(env_)) - evaluate(load_equation(-1), dict(env_))).ravel(), qv


d0, _ = gap(0.0)
d2, qv = gap(2.0)
ck.close("wrong sign is invisible when the cable does not swing", d0, np.zeros(3))
ck.close("wrong sign adds 2 m_Q l |omega|^2 q", d2, 2 * 1.5 * 1.0 * 4.0 * qv, tol=1e-12)
ck.record("sign_error_force_at_2radps_N", float(np.linalg.norm(d2)))
ck.record("sign_error_accel_at_2radps", float(np.linalg.norm(d2) / 1.8))
ck.record("load_weight_N", 0.3 * 9.81)

# ---- tangent plane ----------------------------------------------------------
qv = rng.normal(size=3)
qv /= np.linalg.norm(qv)
wv = np.cross(qv, rng.normal(size=3))
vv = rng.normal(size=3)
ck.close("the three stragglers of the cable equation sum to -q (omega . x_L_dot)",
         np.cross(np.cross(wv, qv), vv) - 2 * np.cross(wv, np.cross(qv, vv)) + np.cross(qv, np.cross(wv, vv)),
         -qv * (wv @ vv), tol=1e-12)
Rn = np.linalg.qr(rng.normal(size=(3, 3)))[0]
Rn *= np.sign(np.linalg.det(Rn))
env_q = {"q": col(qv), str(omega): col(wv), str(vL): col(vv), str(aL): col(rng.normal(size=3)),
         str(domega): col(np.cross(qv, rng.normal(size=3))), "e3": col([0, 0, 1]), "R": Rn, "f": 17.0, **pn}
Eq = evaluate(eom[kq][1], dict(env_q)).ravel()
ck.true("the derived cable equation has a component along q", abs(Eq @ qv) > 1e-3, f"q.E = {Eq @ qv:.3f}")
ck.close("... equal to -m_Q l (omega . x_L_dot)", Eq @ qv, -1.5 * 1.0 * (wv @ vv), tol=1e-12)

# A wrong term along q in the cable equation cannot be seen, and should not be.
hand = dict(eom)
hand[kq] = (xi, eom[kq][1] + q * (Dot(omega, vL) * mQ * l) * 5)
ck.true("adding a multiple of q to the cable equation leaves check_eom unchanged",
        check_eom(L, dW, variables, hand, inputs=inputs) < 1e-5)

# ---- standard form -----------------------------------------------------------
sf = to_standard_form(eom, variables, inputs)
Mxx = str(sf[kx].M["\\frac{d}{dt}(\\frac{d}{dt}(x_L))"])
ck.true("standard form: M[x_L_ddot] = -(m_Q + m_L) I", Mxx == "(I*m_Q*(-1)+I*m_L*(-1))", Mxx)
ck.true("standard form: M[Omega_dot] = -J", str(sf[kR].M["\\frac{d}{dt}(\\Omega_{R})"]) == "J*(-1)")
ck.true("standard form: G[f] of the load row is R e3", str(sf[kx].G["f"]) == "R*e3")
ck.true("standard form: S2 mass block is hat(q) hat(q) m_Q l^2",
        str(sf[kq].M["\\frac{d}{dt}(\\omega_{q})"]) == "Hat(q)*Hat(q)*m_Q*l*l", str(sf[kq].M["\\frac{d}{dt}(\\omega_{q})"]))

# --- listing: simulate
from geomech.codegen import StateLayout, collect_params, load_module, module_source
from geomech.simulation import simulate

sf = to_standard_form(eom, variables, inputs)
layout = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
plant = load_module(module_source(sf, layout, lagrangian=L), "quadrotor_load")

p = {"m_Q": 1.5, "m_L": 0.3, "l": 1.0, "g": 9.81,
     "J": np.diag([0.02, 0.025, 0.04]), "e3": np.array([0.0, 0.0, 1.0])}
q0 = np.array([np.sin(0.8), 0.0, -np.cos(0.8)])     # cable swung out by 0.8 rad
w0 = np.cross(q0, np.cross([0.0, 0.0, 1.6], q0))    # circling about the vertical
v_cm = np.array([0.45, 0.0, 0.0])                   # centre-of-mass velocity
v0 = v_cm + p["m_Q"] * p["l"] / (p["m_Q"] + p["m_L"]) * np.cross(w0, q0)
x0 = layout.pack(x_L=np.zeros(3), q=q0, R=np.eye(3), omega_q=w0, x_L_dot=v0)

hover = np.r_[(p["m_Q"] + p["m_L"]) * p["g"], 0.0, 0.0, 0.0]     # u = [f, tau]
traj = simulate(plant, x0, 6.0, p, u=hover)
free = simulate(plant, x0, 6.0, p)                  # no inputs: energy is conserved
E = free.energy()
print(np.max(np.abs(E - E[0])) / abs(E[0]), free.max_drift())
# --- end
src = module_source(sf, layout, lagrangian=L)
ck.true("generated module has mass, bias, input_matrix, kinematics, energy",
        all(hasattr(plant, n) for n in ("mass", "bias", "input_matrix", "kinematics", "energy")))
ck.true("generated module imports only numpy and copy",
        [ln for ln in src.splitlines() if ln.startswith("import")] == ["import copy", "import numpy as np"])
ck.record("generated_module_lines", len(src.splitlines()))
ck.true("state has 24 numbers, 9 accelerations, 4 inputs",
        (plant.LAYOUT["n_state"], plant.LAYOUT["n_accel"], plant.LAYOUT["n_inputs"]) == (24, 9, 4))
ck.true("default integrator is rkmk4 with dt = 1e-3", free.method == "rkmk4" and abs(free.t[1] - 1e-3) < 1e-12,
        free.method)
drift = float(np.max(np.abs(E - E[0])) / abs(E[0]))
ck.true("unforced run: relative energy drift below 1e-10", drift < 1e-10, f"{drift:.1e}")
ck.true("unforced run: constraint drift below 1e-12", free.max_drift() < 1e-12, f"{free.max_drift():.1e}")
ck.record("free_energy_drift", drift)
ck.record("free_constraint_drift", free.max_drift())
ck.record("free_initial_energy_J", float(E[0]))

Ef = traj.energy()
xQn = traj["x_L"] - p["l"] * traj["q"]
work = hover[0] * (xQn[:, 2] - xQn[0, 2])
ck.true("forced run: energy changes (thrust does work)", np.ptp(Ef) > 0.1, f"range {np.ptp(Ef):.2f} J")
ck.close("forced run: E - f e3.x_Q is conserved", Ef - work, Ef[0] * np.ones_like(Ef), tol=1e-9)
ck.record("forced_energy_range_J", float(np.ptp(Ef)))
ck.record("forced_modified_energy_drift_J", float(np.max(np.abs(Ef - work - Ef[0]))))
com = (p["m_Q"] * xQn + p["m_L"] * traj["x_L"]) / (p["m_Q"] + p["m_L"])
ck.close("forced run: centre of mass coasts on a straight line at v_cm", com, com[0] + np.outer(traj.t, v_cm), tol=1e-9)
ck.true("forced run: constraint drift below 1e-12", traj.max_drift() < 1e-12, f"{traj.max_drift():.1e}")
ck.record("forced_com_line_error_m", float(np.max(np.abs(com - com[0] - np.outer(traj.t, v_cm)))))
ck.record("forced_quad_wobble_m", float(np.max(np.linalg.norm(xQn - com, axis=1))))
ck.record("forced_load_radius_m", float(np.max(np.linalg.norm(traj["x_L"] - com, axis=1))))

# Step-size study for the figure and the text: fourth order until round-off.
drifts = {}
for dt in (0.04, 0.02, 0.01):
    tr_ = simulate(plant, x0, 6.0, p, dt=dt)
    E_ = tr_.energy()
    drifts[dt] = float(np.max(np.abs(E_ - E_[0])) / abs(E_[0]))
order = np.log2(drifts[0.04] / drifts[0.02])
ck.true("energy error falls like dt^4 or faster", order > 3.5, f"observed order {order:.2f}")
ck.record("energy_drift_vs_dt", {str(k): v for k, v in drifts.items()})

# ---- linearization exists and uses the variation coordinates (developed in Chapter 7) ----
lin = linearize(eom, variables, inputs)
ck.true("linearize returns a LinearizedSystem", type(lin).__name__ == "LinearizedSystem", type(lin).__name__)
print("linearized system attributes:", [a_ for a_ in dir(lin) if not a_.startswith("_")])

# ---- comparison with the hand-derived model, Wu (2017) eq. (3.4) -------------
from geomech.simulation import accelerations


def wu_accelerations(xLd, qv, wv, Rm, Om, fv, Mv):
    """(m_Q + m_L)(vL_dot + g e3) = (q.f R e3 - m_Q l qdot.qdot) q;  m_Q l omega_dot = -q x f R e3;
    J Omega_dot + Omega x J Omega = M."""
    mQ_, mL_, l_, g_, J_ = p["m_Q"], p["m_L"], p["l"], p["g"], p["J"]
    F = fv * Rm[:, 2]
    qd = np.cross(wv, qv)
    aL_ = (qv @ F - mQ_ * l_ * qd @ qd) * qv / (mQ_ + mL_) - g_ * p["e3"]
    dw = -np.cross(qv, F) / (mQ_ * l_)
    dOm = np.linalg.solve(J_, Mv - np.cross(Om, J_ @ Om))
    return np.r_[aL_, dw, dOm]


worst = 0.0
for _ in range(50):
    qv = rng.normal(size=3)
    qv /= np.linalg.norm(qv)
    wv = np.cross(qv, rng.normal(size=3))
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Rm *= np.sign(np.linalg.det(Rm))
    Om, xLd = rng.normal(size=3), rng.normal(size=3)
    uv = np.r_[rng.uniform(5, 30), rng.normal(size=3)]
    state = layout.pack(x_L=rng.normal(size=3), q=qv, R=Rm, x_L_dot=xLd, omega_q=wv, Omega_R=Om)
    a_lib = accelerations(plant, state, uv, p)
    worst = max(worst, float(np.max(np.abs(a_lib - wu_accelerations(xLd, qv, wv, Rm, Om, uv[0], uv[1:])))))
ck.true("accelerations agree with Wu (2017) eq. (3.4) at 50 random states", worst < 1e-10, f"max diff {worst:.1e}")
ck.record("wu_comparison_max_diff", worst)

# =========================================================================
# 4. n bodies
# =========================================================================
# --- listing: chain
from geomech.indexed import (Family, IndexedVariables, Sum, compute_indexed_eom,
                             indices, n)

i, j = indices("i j")
g, e3 = Scalar("g", attr=["Constant"]), Vector("e3", attr=["Constant"])
q = Family(S2, "q")                               # q_1, ..., q_n
m = Family(Scalar, "m", attr=["Constant"])
l = Family(Scalar, "l", attr=["Constant"])


def x(k):                                         # position of mass k
    return Sum(j, 1, k, q[j] * l[j])


v = x(i).t_diff()
L = Sum(i, 1, n, 0.5 * m[i] * v.dot(v)) - Sum(i, 1, n, m[i] * g * x(i).dot(e3))
chain = IndexedVariables(families=[q])
eom = compute_indexed_eom(L, None, chain)

for size in range(1, 7):
    members, eq = chain.members(size), eom.instantiate(size)
    print(size, check_eom(L.instantiate(size), None, members, eq))
# --- end
generic = str(eom["q"])
print(generic)
ck.true("generic chain equation: gravity sum from j = i to n",
        generic.startswith("Σ_{j=i}^{n} (Cross(q_{i},e3)*l_{i}*m_{j}*g*(-1))"), generic[:60])
ck.true("generic chain equation: double sum with k from max(i, j) to n",
        "Σ_{j=1}^{n} Σ_{k=max(i, j)}^{n}" in generic)
ck.true("generic chain equation: five terms in the double sum", generic.count("m_{k}*l_{i}*l_{j}") == 5)
chain_errs = {}
for size in range(1, 7):
    chain_errs[size] = check_eom(L.instantiate(size), None, chain.members(size), eom.instantiate(size))
    ck.true(f"chain, n = {size}: check_eom < 1e-5", chain_errs[size] < 1e-5, f"{chain_errs[size]:.1e}")
ck.record("chain_check_eom", {str(k): v_ for k, v_ in chain_errs.items()})
ck.true("n = 6 gives six equations", len(eom.instantiate(6)) == 6)

# The tidy form printed in the chapter, against the instantiated library equation (n = 4).
size = 4
eq4 = eom.instantiate(size)
mv, lv = rng.uniform(0.5, 1.5, size), rng.uniform(0.5, 1.5, size)
qs = rng.normal(size=(size, 3))
qs /= np.linalg.norm(qs, axis=1)[:, None]
ws = np.cross(qs, rng.normal(size=(size, 3)))
dws = np.cross(qs, rng.normal(size=(size, 3)))
env_c = {"g": 9.81, "e3": col([0, 0, 1])}
for k in range(size):
    qk = q[k + 1]
    env_c.update({f"m_{{{k + 1}}}": mv[k], f"l_{{{k + 1}}}": lv[k], str(qk): col(qs[k]),
                  str(qk.get_tangent_vector()): col(ws[k]),
                  str(qk.get_tangent_vector().t_diff()): col(dws[k])})
worst = 0.0
for a in range(size):
    key = str(q[a + 1].get_variation_vector())
    E_lib = evaluate(eq4[key][1], dict(env_c)).ravel()
    E_tidy = -mv[a:].sum() * 9.81 * lv[a] * np.cross(qs[a], [0, 0, 1])
    for b in range(size):
        Mab = mv[max(a, b):].sum() * lv[a] * lv[b]
        E_tidy += Mab * (-np.cross(qs[a], np.cross(dws[b], qs[b])) + (ws[b] @ ws[b]) * np.cross(qs[a], qs[b]))
    P = np.eye(3) - np.outer(qs[a], qs[a])
    worst = max(worst, float(np.max(np.abs(P @ (E_lib - E_tidy)))))
ck.true("tidy chain equation equals the library's on each tangent plane (n = 4)", worst < 1e-10, f"{worst:.1e}")

# Simulate a four-link chain released from the horizontal.
members = chain.members(size)
L4 = L.instantiate(size)
sf4 = to_standard_form(eq4, members, [])
lay4 = StateLayout(members, [], collect_params([L4], members, []))
plant4 = load_module(module_source(sf4, lay4, lagrangian=L4), "chain4")
p4 = {"g": 9.81, "e3": np.array([0.0, 0.0, 1.0])}
p4.update({f"m_{{{k}}}": 0.25 for k in range(1, size + 1)})
p4.update({f"l_{{{k}}}": 0.25 for k in range(1, size + 1)})
ck.true("chain slot names", lay4.slot_names()[:2] == ["q_1", "q_2"] and lay4.slot_names()[4] == "omega_q_1",
        str(lay4.slot_names()))
x04 = lay4.pack(**{f"q_{k}": [1.0, 0.0, 0.0] for k in range(1, size + 1)},
                **{f"omega_q_{k}": [0.0, 0.0, 0.6 * k] for k in range(1, size + 1)})
tr4 = simulate(plant4, x04, 2.0, p4)
E4 = tr4.energy()
drift4 = float(np.max(np.abs(E4 - E4[0])) / max(abs(E4[0]), 1.0))
ck.true("four-link chain: energy drift below 1e-6 over 2 s", drift4 < 1e-6, f"{drift4:.1e}")
ck.record("chain4_energy_drift", drift4)
ck.record("chain4_constraint_drift", tr4.max_drift())

# Several quadrotors carrying a rigid body (Wu 2017, section 3.2.2), from the library's catalogue.
import geomech.systems

rigid = geomech.systems.get("multi_quadrotor_rigid_load", n_quads=3)
t0 = time.perf_counter()
eom_r = compute_eom(rigid.L, rigid.dW, rigid.variables)
t_rigid = time.perf_counter() - t0
err_r = check_eom(rigid.L, rigid.dW, rigid.variables, eom_r, inputs=rigid.inputs)
ck.true("three quadrotors with a rigid load: 21 degrees of freedom", rigid.dof == 21, f"{rigid.dof}")
ck.true("three quadrotors with a rigid load: check_eom < 1e-5", err_r < 1e-5, f"{err_r:.1e}, {t_rigid:.1f} s")
ck.record("multi_quad_rigid_check_eom", err_r)
ck.record("multi_quad_rigid_equations", len(eom_r))
ck.record("catalogue_size", len(geomech.systems.list_systems()))

ck.finish()
