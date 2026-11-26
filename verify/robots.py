"""The robot model library of case_studies/robots, checked.

    python verify/robots.py

For every robot of the library:
  1. the equations of motion are derived with geomech from the robot's own
     Lagrangian, and check_eom must be below 1e-5;
  2. the library's hand-written `dynamics` is compared with the accelerations
     of the geomech model at random states;
  3. `dynamics`, the controllers and the linearizations are compared with the
     functions that the chapters verified, taken from verify/chNN.py;
  4. a simulation with `step` must stay on the manifold to rounding error,
     conserve energy to fourth order in the step, and conserve the momenta
     that the model conserves;
  5. the sized models are checked for several sizes.
The listings of chapters/ch16_case_studies.md are the blocks marked
"# --- listing" below.
"""
import ast
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "case_studies"))

from geomech import check_eom, compute_eom, systems, to_standard_form  # noqa: E402
from geomech.codegen import StateLayout, collect_params, load_module, module_source  # noqa: E402
from geomech.simulation import accelerations  # noqa: E402

import robots  # noqa: E402
from common import Checker, Rx, Ry, Rz, expm_so3, hat, random_rotation, vee  # noqa: E402
from robots.core import E1, E2, E3, tangent, unit  # noqa: E402

ck = Checker("robots")
rng = np.random.default_rng(16)
EOM_TOL, MATCH_TOL = 1e-5, 1e-9
G = 9.81
summary = {}


# =========================================================================
# Helpers
# =========================================================================
def chapter_functions(chapter, names, **given):
    """Named top-level functions of verify/<chapter>.py, compiled from its source.

    Importing a chapter script would run all of its checks. Here only the
    function definitions are taken, and the module-level constants they use
    are passed in `given`, with the values that the chapter uses.
    """
    path = HERE / f"{chapter}.py"
    ns = {"np": np, "hat": hat, "vee": vee, "expm_so3": expm_so3, "Rx": Rx, "Ry": Ry, "Rz": Rz,
          "E1": E1, "E2": E2, "E3": E3, "I3": np.eye(3), "G": G, "GRAV": G,
          "unit": unit, "tangent": tangent}
    ns.update(given)
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
            found[node.name] = ns[node.name]
    missing = set(names) - set(found)
    assert not missing, f"{chapter}: {missing} not found"
    return [found[n] for n in names]


def derive(robot, tag=None):
    """geomech model of a robot: (module, layout, check_eom error)."""
    L, dW, variables, inputs = robot.lagrangian()
    eom = compute_eom(L, dW, variables)
    err = float(check_eom(L, dW, variables, eom, inputs=inputs) if inputs
                else check_eom(L, None, variables, eom))
    sf = to_standard_form(eom, variables, inputs)
    exprs = [L] if dW is None else [L, dW]
    layout = StateLayout(variables, inputs, collect_params(exprs, variables, inputs))
    mod = load_module(module_source(sf, layout, lagrangian=L), (tag or robot.name).replace("-", "_"))
    return mod, layout, err


def geomech_view(robot, layout, state, acc, names):
    """The library's state and accelerations, arranged in geomech's layout."""
    slots, flat = {}, np.zeros(max(sl.stop for sl in layout.accel_slices.values()))
    for f, acc_f in zip(robot.factors, acc):
        gm = names.get(f.config, [f.config])
        c, v = getattr(state, f.config), getattr(state, f.velocity)
        rows = zip(gm, c, v, acc_f) if (f.stacked or len(gm) > 1) else [(gm[0], c, v, acc_f)]
        for name, ci, vi, ai in rows:
            vel, key = {"R": (f"{name}_dot", f"\\frac{{d}}{{dt}}(\\frac{{d}}{{dt}}({name}))"),
                        "S2": (f"omega_{name}", f"\\frac{{d}}{{dt}}(\\omega_{{{name}}})"),
                        "SO3": (f"Omega_{name}", f"\\frac{{d}}{{dt}}(\\Omega_{{{name}}})")}[f.kind]
            slots[name], slots[vel] = ci, vi
            flat[layout.accel_slices[key]] = ai
    return layout.pack(**slots), flat


def flat_input(u):
    """(f, tau) -> [f, tau] for one vehicle, interleaved rows for several."""
    if u is None:
        return np.zeros(0)
    if isinstance(u, tuple) and len(u) == 2 and np.ndim(u[0]) == 1 and np.ndim(u[1]) == 2:
        return np.concatenate([np.r_[fi, ti] for fi, ti in zip(*u)])
    return np.concatenate([np.ravel(x) for x in (u if isinstance(u, tuple) else (u,))])


def random_input(robot, scale=1.0):
    u = robot.zero_input()
    if u is None:
        return None
    if isinstance(u, tuple):
        return tuple(scale * rng.normal(size=np.shape(x)) if np.ndim(x) else float(scale * rng.normal())
                     for x in u)
    return scale * rng.normal(size=np.shape(u))


def against_geomech(robot, gm_params, names=None, tag=None, trials=20):
    """Checks 1 and 2. Returns (check_eom error, worst difference of the accelerations)."""
    tag = tag or robot.name
    mod, layout, err = derive(robot, tag)
    ck.true(f"{tag}: geomech check_eom < 1e-5", err < EOM_TOL, f"err {err:.1e}")
    worst = 0.0
    for _ in range(trials):
        state, u = robot.random_state(rng), random_input(robot, 3.0)
        x, mine = geomech_view(robot, layout, state, robot.dynamics(state, u), names or {})
        theirs = accelerations(mod, x, flat_input(u), gm_params)
        worst = max(worst, float(np.max(np.abs(mine - theirs))))
    ck.true(f"{tag}: library dynamics equal geomech at {trials} random states", worst < MATCH_TOL,
            f"err {worst:.1e}")
    summary.setdefault(tag, {}).update(check_eom=err, vs_geomech=worst)
    return err, worst


def simulate(robot, state, u, h, t_f, watch=None):
    """Zero-order-hold run. Returns (energy drift, manifold residual, spread of watch)."""
    E0, resid, seen = robot.energy(state), 0.0, []
    drift = 0.0
    for _ in range(int(round(t_f / h))):
        state = robot.step(state, u, h)
        drift = max(drift, abs(robot.energy(state) - E0))
        resid = max(resid, robot.on_manifold(state))
        if watch is not None:
            seen.append(watch(state))
    spread = float(np.max(np.ptp(np.array(seen), axis=0))) if seen else 0.0
    return drift / max(abs(E0), 1.0), resid, spread


def conservation(robot, tag=None, h=4e-3, t_f=0.5, watch=None, watch_name="", speed=0.6, state=None):
    """Check 4: free motion from a random state."""
    tag = tag or robot.name
    state = state if state is not None else robot.random_state(rng, speed=speed)
    u = robot.zero_input()
    d1, r1, w1 = simulate(robot, state, u, h, t_f, watch)
    d2, r2, _ = simulate(robot, state, u, h / 2, t_f)
    order = float(np.log2(d1 / d2)) if d2 > 1e-15 else 4.0
    ck.true(f"{tag}: manifold residual at rounding level over {t_f:g} s", max(r1, r2) < 1e-12,
            f"{max(r1, r2):.1e}")
    ck.true(f"{tag}: relative energy drift below 1e-6 at h = {h:g} s", d1 < 1e-6, f"{d1:.1e}")
    ck.true(f"{tag}: energy drift falls like h^4 (order between 3.3 and 5)",
            3.3 < order < 5.0 or d1 < 1e-11, f"h: {d1:.1e}, h/2: {d2:.1e}, order {order:.2f}")
    if watch is not None:
        scale = max(1.0, float(np.max(np.abs(watch(state)))))
        ck.true(f"{tag}: {watch_name} conserved", w1 / scale < 1e-6, f"relative spread {w1 / scale:.1e}")
    summary.setdefault(tag, {}).update(energy_drift=d1, energy_order=order, manifold_residual=max(r1, r2))


# =========================================================================
# 0. The interface
# =========================================================================
names_all = robots.list_robots()
ck.true("registry lists twelve robots", len(names_all) == 12, str(names_all))
table = {"spherical_pendulum": (2, 3), "rigid_body": (3, 9), "pendulum_3d": (3, 9), "quadrotor": (6, 12),
         "quadrotor_load": (8, 15), "telescoping_leg": (4, 10), "reaction_mass_biped": (13, 31),
         "cassie": (14, 32)}
for name, (dim, stored) in table.items():
    r = robots.get(name)
    ck.true(f"{name}: dimension {dim}, {stored} numbers stored", (r.dim, r.stored) == (dim, stored),
            f"{r.manifold}: {r.dim}, {r.stored}")
r = robots.get("multi_quadrotor_payload", n=3)
ck.true("three quadrotors with a rigid payload: 6 + 5n = 21, 12 + 12n = 48 stored",
        (r.dim, r.stored) == (21, 48), f"{r.dim}, {r.stored}")
r = robots.get("multi_quadrotor_load", n=3)
ck.true("three quadrotors with a point-mass load: 3 + 5n = 18", r.dim == 18)
r = robots.get("quadrotor_cable", n=5, attach=(0, 5))
ck.true("cable of five links between two quadrotors: 3 (n_Q + 1) + 2n = 19", r.dim == 19)
for name in names_all:
    r = robots.get(name)
    s = r.random_state(rng)
    ok = (r.on_manifold(s) < 1e-13 and np.isfinite(r.energy(s)) and isinstance(r.labels, dict)
          and all(hasattr(r, a) for a in ("manifold", "factors", "params", "dynamics", "step", "lagrangian")))
    ck.true(f"{name}: interface complete, random state on the manifold", ok, r.manifold)
try:
    robots.get("no_such_robot")
    ck.true("unknown robot raises KeyError", False)
except KeyError:
    ck.true("unknown robot raises KeyError", True)


# =========================================================================
# 1. Spherical pendulum (eq4-sph)
# =========================================================================
pend = robots.get("spherical_pendulum")
p = pend.params
against_geomech(pend, {"m": p.m, "l": p.l, "g": p.g, "e3": E3})

# against the chapters' functions, each at its own parameter values
sph_omega_dot, sph_qddot = chapter_functions("ch04", ["sph_omega_dot", "sph_qddot"], m_s=0.7, l_s=1.3)
(pend_omega_dot,) = chapter_functions("ch06", ["pend_omega_dot"], m_p=1.0, l_p=1.0)
(s2_f,) = chapter_functions("ch07", ["s2_f"], M_P=0.5, L_P=0.8)
(s2_accel,) = chapter_functions("ch08", ["s2_accel"], M_P=0.5, L_P=0.8)
p04, p06, p07 = pend.Params(m=0.7, l=1.3), pend.Params(m=1.0, l=1.0), pend.Params(m=0.5, l=0.8)
w = np.zeros(5)
for _ in range(20):
    s, F = pend.random_state(rng), rng.normal(size=3)
    q, om = s.q, s.omega
    w[0] = max(w[0], np.max(np.abs(pend.dynamics(s, F, p04)[0] - sph_omega_dot(q, om, p04.l * np.cross(q, F)))))
    od = pend.dynamics(s, np.zeros(3), p04)[0]
    qd = np.cross(om, q)
    w[1] = max(w[1], np.max(np.abs(np.cross(od, q) + np.cross(om, qd) - sph_qddot(q, qd))))
    w[2] = max(w[2], np.max(np.abs(pend.dynamics(s, F, p06)[0] - pend_omega_dot(q, om, F))))
    w[3] = max(w[3], np.max(np.abs(pend.dynamics(s, F, p07)[0] - s2_f(q, om, F)[1])))
    w[4] = max(w[4], np.max(np.abs(pend.dynamics(s, F, p07)[0] - s2_accel(q, om, F))))
for err, where in zip(w, ("ch04 sph_omega_dot (eq4-sph)", "ch04 sph_qddot (eq4-sph-qdd)", "ch06 pend_omega_dot",
                          "ch07 s2_f (eq7-pendulum)", "ch08 s2_accel")):
    ck.true(f"spherical_pendulum: dynamics equal {where}", err < 1e-12, f"err {err:.1e}")
summary["spherical_pendulum"]["vs_chapter"] = float(w.max())

# tracking law eq6-law-s2 and linearization eq7-s2-lin against the chapters
e_q6, e_omega6, omega_dot_des, s2_force = None, None, None, None
e_q6, e_omega6 = chapter_functions("ch06", ["e_q", "e_omega"])
omega_dot_des, s2_force = chapter_functions("ch06", ["omega_dot_des", "s2_force"], e_q=e_q6, e_omega=e_omega6,
                                            k_q=16.0, k_om=5.6, m_p=1.0, l_p=1.0)
s2_A, s2_B, s2_C = chapter_functions("ch07", ["s2_A", "s2_B", "s2_C"], M_P=0.5, L_P=0.8)
w1 = w2 = 0.0
for _ in range(20):
    s = pend.random_state(rng)
    qd = unit(rng.normal(size=3))
    omd, omd_dot = tangent(qd, rng.normal(size=3)), rng.normal(size=3)
    w1 = max(w1, np.max(np.abs(pend.tracking_controller(s, qd, omd, omd_dot, params=p06)
                               - s2_force(s.q, s.omega, qd, omd, omd_dot))))
    Fd = rng.normal(size=3)
    A, B, C = pend.linearize(qd, omd, Fd, p07)
    w2 = max(w2, np.max(np.abs(A - s2_A(qd, omd, Fd))), np.max(np.abs(B - s2_B(qd))),
             np.max(np.abs(C - s2_C(qd, omd))))
ck.true("spherical_pendulum: tracking_controller equals ch06 s2_force (eq6-law-s2)", w1 < 1e-12, f"err {w1:.1e}")
ck.true("spherical_pendulum: linearize equals ch07 s2_A, s2_B, s2_C (eq7-s2-lin)", w2 < 1e-12, f"err {w2:.1e}")
conservation(pend, watch=lambda s: s.omega[2], watch_name="e3 . omega (vertical angular momentum)")

# --- listing: pendulum
pend = robots.get("spherical_pendulum")                  # m = 1 kg, l = 1 m
state = pend.State(q=np.array([1.0, 0.0, 0.0]), omega=np.zeros(3))
swing = pend.rollout(state, pend.zero_input(), 1e-3, 1000)
print(max(pend.tension(s) for s in swing))   # 29.43 N: three times the weight
print(max(pend.on_manifold(s) for s in swing))           # 2e-15
# --- end
T_max = max(pend.tension(s) for s in swing)
ck.close("listing pendulum: released from the horizontal, the largest tension is 3 m g", T_max, 3 * G, tol=1e-4)
ck.true("listing pendulum: manifold residual at rounding level", max(pend.on_manifold(s) for s in swing) < 1e-14)
ck.record("pendulum_max_tension_N", float(T_max))

# closed loop: the tracking law brings the pendulum onto a moving reference
CONE, W_REF = np.radians(35), 1.2
(ref_s2,) = chapter_functions("ch06", ["ref_s2"], CONE=CONE, W_REF=W_REF)
state = pend.State(q=unit([0.2, -0.6, -0.5]), omega=np.zeros(3))
for k in range(4000):
    state = pend.step(state, pend.tracking_controller(state, *ref_s2(k * 2e-3)), 2e-3)
err_s2 = float(np.degrees(np.arccos(np.clip(state.q @ ref_s2(8.0)[0], -1, 1))))
ck.true("spherical_pendulum: tracking law follows the cone reference of ch06 (below 0.5 deg after 8 s)",
        err_s2 < 0.5, f"{err_s2:.3f} deg")


# =========================================================================
# 2. Rigid body and 3D pendulum (eq4-euler-free, eq4-pendulum)
# =========================================================================
body, pend3 = robots.get("rigid_body"), robots.get("pendulum_3d")
for r in (body, pend3):
    p = r.params
    against_geomech(r, {"J": p.J, "m": p.m, "g": p.g, "rho": p.rho, "e3": E3})
p = pend3.params
(pendulum_accel,) = chapter_functions("ch04", ["pendulum_accel"], J_p=p.J, m_p=p.m, rho_p=p.rho)
J_B = np.diag([0.03, 0.05, 0.07])
so3_acc, so3_A = chapter_functions("ch07", ["so3_acc", "so3_A"], ZERO3=np.zeros(3), J_B=J_B)
(body_step,) = chapter_functions("ch08", ["body_step"])
e_R6, e_Omega6 = chapter_functions("ch06", ["e_R", "e_Omega"])
J6 = np.diag([0.0820, 0.0845, 0.1377])
(att_torque,) = chapter_functions("ch06", ["att_torque"], e_R=e_R6, e_Omega=e_Omega6, k_R=8.81, k_Om=2.54, J=J6)
p07 = pend3.Params(J=J_B, m=0.8, rho=np.array([0.05, -0.02, -0.3]))
mgr = p07.m * p07.g * p07.rho
p06 = pend3.Params(J=J6, m=0.0, rho=np.zeros(3))
w = np.zeros(5)
for _ in range(20):
    s, tau = pend3.random_state(rng), rng.normal(size=3)
    w[0] = max(w[0], np.max(np.abs(pend3.dynamics(s, tau)[0] - pendulum_accel(s.R, s.Omega, tau))))
    w[1] = max(w[1], np.max(np.abs(pend3.dynamics(s, tau, p07)[0] - so3_acc(s.R, s.Omega, tau, mgr))))
    Rd, Omd, Omd_dot = random_rotation(rng), rng.normal(size=3), rng.normal(size=3)
    A, B = pend3.linearize(Rd, Omd, p07)
    w[2] = max(w[2], np.max(np.abs(A - so3_A(Rd, Omd, mgr))),
               np.max(np.abs(B - np.vstack([np.zeros((3, 3)), np.linalg.inv(J_B)]))))
    w[3] = max(w[3], np.max(np.abs(body.tracking_controller(s, Rd, Omd, Omd_dot, params=p06)
                                   - att_torque(s.R, s.Omega, Rd, Omd, Omd_dot))))
    # one step of the library against one step of ch08's integrator (both fourth order)
    s1 = body.step(s, tau, 1e-3, p06)
    R8, Om8 = body_step(s.R, s.Omega, tau, 1e-3, J6, np.linalg.inv(J6))
    w[4] = max(w[4], np.max(np.abs(s1.Omega - Om8)), np.max(np.abs(s1.R - R8)))
for err, where, tol in zip(w, ("dynamics equal ch04 pendulum_accel (eq4-pendulum)", "dynamics equal ch07 so3_acc (eq7-body)",
                               "linearize equals ch07 so3_A, SO3_B (eq7-so3-lin)",
                               "tracking_controller equals ch06 att_torque (eq6-law)",
                               "one step agrees with ch08 body_step to 1e-7"), (1e-12, 1e-12, 1e-12, 1e-12, 1e-7)):
    ck.true(f"pendulum_3d / rigid_body: {where}", err < tol, f"err {err:.1e}")
summary["pendulum_3d"]["vs_chapter"] = float(w[:2].max())
summary["rigid_body"]["vs_chapter"] = float(w[3])

# linearization against finite differences of the library's own dynamics
Rd, Omd = random_rotation(rng), rng.normal(size=3)
A, B = pend3.linearize(Rd, Omd)
taud = pend3.params.J @ rng.normal(size=3)
base = pend3.dynamics(pend3.State(R=Rd, Omega=Omd), taud)[0]
num = np.zeros((3, 6))
for k in range(3):
    d = 1e-6 * np.eye(3)[k]
    num[:, k] = (pend3.dynamics(pend3.State(R=Rd @ expm_so3(d), Omega=Omd), taud)[0] - base) / 1e-6
    num[:, 3 + k] = (pend3.dynamics(pend3.State(R=Rd, Omega=Omd + d), taud)[0] - base) / 1e-6
ck.close("pendulum_3d: lower rows of A equal finite differences of dynamics", A[3:], num, tol=1e-5)

conservation(body, watch=lambda s: body.momentum(s), watch_name="world-frame angular momentum R J Omega")
conservation(pend3, watch=lambda s: pend3.momentum(s)[2], watch_name="e3 . R J Omega")

# --- listing: book
book = robots.get("rigid_body")   # the paperback of Chapter 4
spin = 2 * np.pi * 2.0                                   # two turns per second
state = book.State(R=np.eye(3), Omega=spin * np.array([0.01, 1.0, 0.01]))
flight = book.rollout(state, book.zero_input(), 1e-3, 6000)
flip = next(k for k, s in enumerate(flight) if s.Omega[1] < 0) * 1e-3
print(flip)   # 0.88 s: the first half flip
print(np.ptp([book.momentum(s) for s in flight], axis=0).max())    # 4e-12
# --- end
ck.true("listing book: first reversal of Omega_2 at 0.88 s, as in Chapter 4", abs(flip - 0.8805) < 2e-3, f"{flip:.4f} s")
mom_spread = float(np.ptp([book.momentum(s) for s in flight], axis=0).max())
ck.true("listing book: R J Omega constant over 6 s", mom_spread < 1e-9, f"spread {mom_spread:.1e}")
ck.record("book_first_flip_s", float(flip))
ck.record("book_momentum_spread", mom_spread)


# =========================================================================
# 3. Quadrotor (eq4-quadrotor)
# =========================================================================
quad = robots.get("quadrotor")
p = quad.params
against_geomech(quad, {"m": p.m, "J": p.J, "g": p.g, "e3": E3})
k = quad.Gains()
Rd_from, Rd_derivs, ref_pos = chapter_functions("ch06", ["Rd_from", "Rd_derivs", "ref_pos"])
(quad_ctrl,) = chapter_functions("ch06", ["quad_ctrl"], ref_pos=ref_pos, Rd_derivs=Rd_derivs, att_torque=att_torque,
                                 k_x=k.k_x, k_v=k.k_v, k_R=k.k_R, k_Om=k.k_Omega, m_Q=p.m)
(quad_rhs,) = chapter_functions("ch06", ["quad_rhs"], quad_ctrl=quad_ctrl, m_Q=p.m, J=p.J, Jinv=np.linalg.inv(p.J))
w1 = w2 = w3 = 0.0
for _ in range(20):
    s, t = quad.random_state(rng), 10 * rng.random()
    ref = robots.quadrotor.circle_reference(t)
    w1 = max(w1, max(np.max(np.abs(a - b)) for a, b in zip(ref, ref_pos(t))))
    f, tau = quad.tracking_controller(s, ref)
    f6, tau6, *_ = quad_ctrl(t, s.x, s.v, s.R, s.Omega)
    w2 = max(w2, abs(f - f6), np.max(np.abs(tau - tau6)))
    rhs = quad_rhs(t, np.concatenate([s.x, s.v, s.R.ravel(), s.Omega]), True, 1.0, True)
    v_dot, Omega_dot = quad.dynamics(s, (f, tau))
    w3 = max(w3, np.max(np.abs(rhs[3:6] - v_dot)), np.max(np.abs(rhs[15:] - Omega_dot)))
ck.true("quadrotor: circle_reference equals ch06 ref_pos", w1 < 1e-12, f"err {w1:.1e}")
ck.true("quadrotor: tracking_controller equals ch06 quad_ctrl (eq6-Fd, eq6-Rd, eq6-law)", w2 < 1e-9, f"err {w2:.1e}")
ck.true("quadrotor: closed-loop accelerations equal ch06 quad_rhs (eq4-quadrotor)", w3 < 1e-9, f"err {w3:.1e}")
summary["quadrotor"]["vs_chapter"] = float(max(w2, w3))
conservation(quad, watch=lambda s: s.R @ quad.params.J @ s.Omega, watch_name="R J Omega")
free = quad.rollout(quad.State(v=np.array([1.0, 0.0, 2.0])), quad.zero_input(), 1e-2, 50)
ck.close("quadrotor: with f = 0 the center of mass is on a parabola", free[-1].x,
         np.array([0.5, 0.0, 1.0 - 0.5 * G * 0.25]), tol=1e-12)

# --- listing: quadrotor
quad = robots.get("quadrotor")                           # the vehicle of Chapter 6
state = quad.State()                                     # at rest at the origin
for k in range(4000):
    ref = robots.quadrotor.circle_reference(k * 2e-3)
    state = quad.step(state, quad.tracking_controller(state, ref), 2e-3)
print(np.linalg.norm(state.x - ref[0]))                  # 3 mm after 8 s
# --- end
quad_err = float(np.linalg.norm(state.x - ref[0]))
ck.true("listing quadrotor: from 1.5 m away, on the circle to better than 1 cm after 8 s", quad_err < 1e-2,
        f"{quad_err:.1e} m")
ck.record("quadrotor_tracking_error_m", quad_err)


# =========================================================================
# 4. Quadrotor with a cable-suspended load (eq4-load)
# =========================================================================
ql = robots.get("quadrotor_load")
p = ql.params
against_geomech(ql, {"m_Q": p.m_Q, "m_L": p.m_L, "l": p.l, "J": p.J, "g": p.g, "e3": E3})
(load_accels4,) = chapter_functions("ch04", ["load_accels"], mQ=p.m_Q, mL=p.m_L, ll=p.l, JQ=p.J)
(wu_accelerations,) = chapter_functions(
    "ch05", ["wu_accelerations"], p={"m_Q": p.m_Q, "m_L": p.m_L, "l": p.l, "g": p.g, "J": p.J, "e3": E3})
p6 = robots.quadrotor_load.chapter6_params()
g6 = ql.Gains()
(load_accels6,) = chapter_functions("ch06", ["load_accels"], m_Q=p6.m_Q, m_L=p6.m_L, l_c=p6.l, J=p6.J,
                                    Jinv=np.linalg.inv(p6.J))
(load_force,) = chapter_functions("ch06", ["load_force"], e_q=e_q6, kxL=g6.k_x, kvL=g6.k_v, kqL=g6.k_q,
                                  komL=g6.k_omega, m_Q=p6.m_Q, m_L=p6.m_L, l_c=p6.l)
w = np.zeros(5)
for _ in range(20):
    s, u = ql.random_state(rng), (5 + 20 * rng.random(), rng.normal(size=3))
    mine = np.concatenate(ql.dynamics(s, u))
    w[0] = max(w[0], np.max(np.abs(mine - np.concatenate(load_accels4(s.q, s.omega, s.R, s.Omega, *u)))))
    w[1] = max(w[1], np.max(np.abs(mine - wu_accelerations(s.v_L, s.q, s.omega, s.R, s.Omega, *u))))
    w[2] = max(w[2], np.max(np.abs(np.concatenate(ql.dynamics(s, u, p6))
                                   - np.concatenate(load_accels6(s.q, s.omega, s.R, s.Omega, *u)))))
    # the controller, as ch06 load_rhs assembles it for a fixed target
    xd = rng.normal(size=3)
    f, tau = ql.tracking_controller(s, xd, gains=ql.Gains(k_R=8.81, k_Omega=2.54), params=p6)
    F, *_ = load_force(s.x_L, s.v_L, s.q, s.omega, xd)
    f6 = max(F @ (s.R @ E3), 0.0)
    tau6 = att_torque(s.R, s.Omega, Rd_from(F, E1), np.zeros(3), np.zeros(3), 8.81, 2.54)
    w[3] = max(w[3], abs(f - f6), np.max(np.abs(tau - tau6)))
    # Newton's law for each body with the tension eq4-load-tension
    T = ql.tension(s, u)
    v_L_dot, omega_dot, _ = ql.dynamics(s, u)
    q_dd = np.cross(omega_dot, s.q) + np.cross(s.omega, np.cross(s.omega, s.q))
    w[4] = max(w[4], np.max(np.abs(p.m_L * (v_L_dot + G * E3) + T * s.q)),
               np.max(np.abs(p.m_Q * (v_L_dot - p.l * q_dd + G * E3) - u[0] * s.R @ E3 - T * s.q)))
for err, where in zip(w, ("dynamics equal ch04 load_accels (eq4-load)", "dynamics equal ch05 wu_accelerations (eq5-wu)",
                          "dynamics equal ch06 load_accels", "tracking_controller equals ch06 load_force, att_torque (eq6-load)",
                          "tension satisfies Newton's law for both bodies (eq4-load-tension)")):
    ck.true(f"quadrotor_load: {where}", err < 1e-10, f"err {err:.1e}")
summary["quadrotor_load"]["vs_chapter"] = float(w[:3].max())
hover = ql.State()
ck.close("quadrotor_load: hover is an equilibrium", np.concatenate(ql.dynamics(hover, ql.hover_input())), np.zeros(9))
ck.close("quadrotor_load: hover tension is m_L g = 2.94 N", ql.tension(hover, ql.hover_input()), p.m_L * G)


def load_momentum(s):
    """Horizontal linear momentum of the pair."""
    v_Q = s.v_L - ql.params.l * np.cross(s.omega, s.q)
    return (ql.params.m_Q * v_Q + ql.params.m_L * s.v_L)[:2]


conservation(ql, watch=load_momentum, watch_name="horizontal momentum")

# --- listing: load
params = robots.quadrotor_load.chapter6_params()         # 4.34 kg carrying 0.5 kg
robot = robots.get("quadrotor_load", params=params)
swing = robots.expm_so3(np.radians(40) * robots.E1) @ -robots.E3
state, target = robot.State(q=swing), np.array([1.5, 0.0, 0.5])
T_min = np.inf
for k in range(6000):
    u = robot.tracking_controller(state, target)         # eq6-load, eq6-Rd, eq6-law
    T_min = min(T_min, robot.tension(state, u))          # eq4-load-tension
    state = robot.step(state, u, 2e-3)
print(np.linalg.norm(state.x_L - target), T_min)         # 1.0 mm, 4.17 N
print(robot.on_manifold(state))                          # 3e-14
# --- end
load_err = float(np.linalg.norm(state.x_L - target))
ck.true("listing load: from a 40 deg swing the load is 1 mm from the target after 12 s, as in ch06",
        abs(load_err - 1.0563e-3) < 5e-5, f"{load_err:.4e} m")
ck.true("listing load: the cable stays taut, smallest tension 4.17 N as in ch06", abs(T_min - 4.1714) < 5e-3, f"{T_min:.4f} N")
ck.true("listing load: manifold residual at rounding level", robot.on_manifold(state) < 1e-12,
        f"{robot.on_manifold(state):.1e}")
ck.record("load_min_tension_N", float(T_min))
ck.record("load_final_error_m", load_err)


# =========================================================================
# 5. Several quadrotors: point-mass load (eq4-multi) and rigid payload
# =========================================================================
def team_params(r):
    p = r.params
    out = {"m_L": p.m_L, "g": p.g, "e3": E3}
    for i in range(r.n):
        out.update({f"m{i + 1}": p.m[i], f"l{i + 1}": p.l[i], f"J{i + 1}": p.J[i]})
        if hasattr(p, "r"):
            out[f"r{i + 1}"] = p.r[i]
    if hasattr(p, "J_L"):
        out["J_L"] = p.J_L
    return out


def team_names(r):
    return {"q": [f"q{i + 1}" for i in range(r.n)], "R": [f"R{i + 1}" for i in range(r.n)]}


def team_momentum(r):
    def watch(s):
        p = r.params
        v_Q = s.v_L - p.l[:, None] * np.cross(s.omega, s.q)
        if hasattr(p, "r"):
            v_Q = v_Q + np.cross(s.Omega_L, p.r) @ s.R_L.T
        return (p.m_L * s.v_L + p.m @ v_Q)[:2]
    return watch


for n in (1, 2, 3):
    # unequal masses and lengths, so that no symmetry hides an index error
    team = robots.get("multi_quadrotor_load", n=n)
    team.params.m, team.params.l = 1.0 + 0.3 * np.arange(n), 0.8 + 0.2 * np.arange(n)
    against_geomech(team, team_params(team), team_names(team), tag=f"multi_quadrotor_load-n{n}", trials=10)
    conservation(team, tag=f"multi_quadrotor_load-n{n}", watch=team_momentum(team), watch_name="horizontal momentum")

# the chapter's own check of eq4-multi is written inline in ch04.py (two quadrotors); the same lines, retyped
team = robots.get("multi_quadrotor_load", n=2)
p = team.params
worst = 0.0
for _ in range(10):
    s, u = team.random_state(rng), (5 + 20 * rng.random(2), rng.normal(size=(2, 3)))
    us = [u[0][i] * s.R[i] @ E3 for i in range(2)]
    A = p.m_L * np.eye(3) + sum(p.m[i] * np.outer(s.q[i], s.q[i]) for i in range(2))
    rhs = sum(((s.q[i] @ us[i]) - p.m[i] * p.l[i] * (s.omega[i] @ s.omega[i])) * s.q[i] for i in range(2))
    acc = np.linalg.solve(A, rhs)
    omds = [np.cross(s.q[i], p.m[i] * acc - us[i]) / (p.m[i] * p.l[i]) for i in range(2)]
    v_L_dot, omega_dot, _ = team.dynamics(s, u)
    worst = max(worst, np.max(np.abs(v_L_dot - (acc - G * E3))), np.max(np.abs(omega_dot - np.array(omds))))
ck.true("multi_quadrotor_load: dynamics equal the lines of ch04.py that check eq4-multi", worst < 1e-12, f"err {worst:.1e}")
summary["multi_quadrotor_load-n2"]["vs_chapter"] = float(worst)

# n = 1 collapses to the quadrotor with a cable-suspended load
one = robots.get("multi_quadrotor_load", n=1)
one.params.m_L, one.params.m, one.params.l = ql.params.m_L, np.array([ql.params.m_Q]), np.array([ql.params.l])
worst = 0.0
for _ in range(10):
    s, u = ql.random_state(rng), (5 + 20 * rng.random(), rng.normal(size=3))
    s1 = one.State(x_L=s.x_L, v_L=s.v_L, q=s.q[None], omega=s.omega[None], R=s.R[None], Omega=s.Omega[None])
    a1 = one.dynamics(s1, (np.array([u[0]]), u[1][None]))
    worst = max(worst, np.max(np.abs(np.concatenate([np.ravel(x) for x in a1]) - np.concatenate(ql.dynamics(s, u)))),
                abs(one.tensions(s1, (np.array([u[0]]), u[1][None]))[0] - ql.tension(s, u)),
                abs(one.energy(s1) - ql.energy(s)))
ck.true("multi_quadrotor_load: n = 1 equals quadrotor_load (dynamics, tension, energy)", worst < 1e-10, f"err {worst:.1e}")

for n in (2, 3):
    team = robots.get("multi_quadrotor_payload", n=n)
    team.params.m, team.params.l = 1.0 + 0.3 * np.arange(n), 0.8 + 0.2 * np.arange(n)
    team.params.r = team.params.r + 0.1 * rng.normal(size=(n, 3))
    names = {**team_names(team), "R_L": ["R_L"]}
    against_geomech(team, team_params(team), names, tag=f"multi_quadrotor_payload-n{n}", trials=10)
    conservation(team, tag=f"multi_quadrotor_payload-n{n}", watch=team_momentum(team), watch_name="horizontal momentum")

# the rigid-payload model against the Lagrangian of geomech's own catalog (Wu 2017, sec. 3.2.2)
cat = systems.get("multi_quadrotor_rigid_load", n_quads=2)
eom = compute_eom(cat.L, cat.dW, cat.variables)
cat_err = float(check_eom(cat.L, cat.dW, cat.variables, eom, inputs=list(cat.inputs)))
cat_layout = StateLayout(cat.variables, list(cat.inputs), collect_params([cat.L, cat.dW], cat.variables, list(cat.inputs)))
cat_mod = load_module(module_source(to_standard_form(eom, cat.variables, list(cat.inputs)), cat_layout, lagrangian=cat.L),
                      "catalog_rigid_load")
cp = dict(cat.params)
team = robots.get("multi_quadrotor_payload", n=2)
tp = team.params
tp.m_L, tp.J_L, tp.r = cp["m_L"], cp["J_L"], np.array([cp["\\rho_1"], cp["\\rho_2"]])
tp.m, tp.l, tp.J = np.array([cp["m1"], cp["m2"]]), np.array([cp["l1"], cp["l2"]]), np.array([cp["J1"], cp["J2"]])
worst = 0.0
for _ in range(10):
    s, u = team.random_state(rng), random_input(team, 3.0)
    x, mine = geomech_view(team, cat_layout, s, team.dynamics(s, u), {**team_names(team), "R_L": ["R_L"]})
    worst = max(worst, float(np.max(np.abs(mine - accelerations(cat_mod, x, flat_input(u), cp)))))
ck.true("multi_quadrotor_payload: equals geomech's catalog model multi_quadrotor_rigid_load (n = 2)",
        cat_err < EOM_TOL and worst < MATCH_TOL, f"check_eom {cat_err:.1e}, err {worst:.1e}")

# a payload with every cable attached at its center of mass moves like a point-mass load
team = robots.get("multi_quadrotor_payload", n=3)
team.params.r = np.zeros((3, 3))
point = robots.get("multi_quadrotor_load", n=3)
point.params.m_L = team.params.m_L
s, u = team.random_state(rng), random_input(team, 3.0)
u = (np.abs(u[0]) + 5, u[1])
a_rigid = team.dynamics(s, u)
a_point = point.dynamics(point.State(x_L=s.x_L, v_L=s.v_L, q=s.q, omega=s.omega, R=s.R, Omega=s.Omega), u)
ck.close("multi_quadrotor_payload: with r_i = 0 it equals multi_quadrotor_load",
         np.concatenate([a_rigid[0], a_rigid[2].ravel()]), np.concatenate([a_point[0], a_point[1].ravel()]), tol=1e-10)

# --- listing: team
team = robots.get("multi_quadrotor_payload", n=3)        # three quadrotors
state = team.random_state(np.random.default_rng(1), speed=0.5)
E0 = team.energy(state)
for k in range(1000):
    state = team.step(state, team.zero_input(), 1e-3)    # motors off: free fall
print(team.manifold, team.dim)   # R^3 x SO(3) x (S^2 x SO(3))^3 21
print(abs(team.energy(state) - E0) / E0)                 # 6e-14
print(team.on_manifold(state))                           # 9e-15
# --- end
team_drift = float(abs(team.energy(state) - E0) / E0)
ck.true("listing team: relative energy drift below 1e-9 over 1 s of free fall", team_drift < 1e-9, f"{team_drift:.1e}")
ck.true("listing team: manifold residual at rounding level", team.on_manifold(state) < 1e-13, f"{team.on_manifold(state):.1e}")
ck.record("team_energy_drift", team_drift)


# =========================================================================
# 6. Flexible cable: hanging chain (eq5-chain) and cable carried by quadrotors
# =========================================================================
def chain_params(r):
    out = {"g": r.params.g, "e3": E3}
    for i in range(r.n):
        out.update({f"m{i + 1}": r.params.m[i], f"l{i + 1}": r.params.l[i]})
    return out


for n in (1, 2, 3, 4, 5):
    chain = robots.get("flexible_cable", n=n)
    chain.params.m, chain.params.l = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, n)
    against_geomech(chain, chain_params(chain), {"q": [f"q{i + 1}" for i in range(n)]}, tag=f"flexible_cable-n{n}", trials=10)
    conservation(chain, tag=f"flexible_cable-n{n}", h=1e-3, speed=0.3, watch=lambda s, c=chain: c.momentum(s)[2],
                 watch_name="e3 . angular momentum about the pivot")

# eq5-chain itself, as ch05.py types it for its own check: the residual vanishes at the library's accelerations
chain = robots.get("flexible_cable", n=4)
chain.params.m, chain.params.l = rng.uniform(0.5, 1.5, 4), rng.uniform(0.5, 1.5, 4)
mv, lv = chain.params.m, chain.params.l
worst = 0.0
for _ in range(10):
    s = chain.random_state(rng)
    qs, ws, dws = s.q, s.omega, chain.dynamics(s)[0]
    for a in range(4):
        E_tidy = -mv[a:].sum() * 9.81 * lv[a] * np.cross(qs[a], [0, 0, 1])
        for b in range(4):
            Mab = mv[max(a, b):].sum() * lv[a] * lv[b]
            E_tidy += Mab * (-np.cross(qs[a], np.cross(dws[b], qs[b])) + (ws[b] @ ws[b]) * np.cross(qs[a], qs[b]))
        worst = max(worst, float(np.max(np.abs(E_tidy))), abs(float(qs[a] @ dws[a])))
ck.true("flexible_cable: eq5-chain, as typed in ch05.py, vanishes at the library's accelerations (n = 4)",
        worst < 1e-10, f"residual {worst:.1e}")
summary["flexible_cable-n4"]["vs_chapter"] = float(worst)

# one link is the spherical pendulum
chain = robots.get("flexible_cable", n=1)
chain.params.m, chain.params.l = np.array([1.0]), np.array([1.0])
s = pend.random_state(rng)
ck.close("flexible_cable: n = 1 equals spherical_pendulum",
         chain.dynamics(chain.State(q=s.q[None], omega=s.omega[None]))[0][0], pend.dynamics(s, np.zeros(3))[0], tol=1e-12)

# the default chain is the four-link chain of Chapter 5, released from the horizontal while turning
chain = robots.get("flexible_cable", n=4)
state = chain.State(q=np.tile([1.0, 0.0, 0.0], (4, 1)), omega=np.array([[0.0, 0.0, 0.6 * k] for k in range(1, 5)]))
d4, r4, _ = simulate(chain, state, None, 1e-3, 2.0)
ck.true("flexible_cable: the chain of fig5-chain, relative energy drift below 1e-6 over 2 s", d4 < 1e-6, f"{d4:.1e}")
ck.record("chain4_energy_drift", d4)


def cable_params(r):
    mass = r.joint_masses()
    out = {"g": r.params.g, "e3": E3}
    out.update({f"m{k}": mass[k] for k in range(r.n + 1)})
    out.update({f"l{i + 1}": r.params.l[i] for i in range(r.n)})
    out.update({f"J{k}": r.params.J[j] for j, k in enumerate(r.params.attach)})
    return out


def cable_names(r):
    return {"q": [f"q{i + 1}" for i in range(r.n)], "R": [f"R{k}" for k in r.params.attach]}


def cable_momentum(r):
    return lambda s: (r.joint_masses() @ r.velocities(s))[:2]


for n, attach in ((1, (0,)), (3, (0,)), (3, (0, 3)), (4, (0, 2, 4)), (5, (5,))):
    cable = robots.get("quadrotor_cable", n=n, attach=attach)
    cable.params.m = rng.uniform(0.05, 0.3, n + 1)
    cable.params.l = rng.uniform(0.2, 0.5, n)
    cable.params.m_Q = 1.0 + 0.3 * np.arange(len(attach))
    tag = f"quadrotor_cable-n{n}-at{''.join(map(str, attach))}"
    against_geomech(cable, cable_params(cable), cable_names(cable), tag=tag, trials=10)
    conservation(cable, tag=tag, h=2e-3, speed=0.5, watch=cable_momentum(cable), watch_name="horizontal momentum")

# one link under one quadrotor is the quadrotor with a cable-suspended load: x_L = x_0 + l q
cable = robots.get("quadrotor_cable", n=1, attach=(0,))
cp_ = cable.params
cp_.m, cp_.l, cp_.m_Q, cp_.J = np.array([0.0, ql.params.m_L]), np.array([ql.params.l]), np.array([ql.params.m_Q]), ql.params.J[None]
worst = 0.0
for _ in range(10):
    s, u = ql.random_state(rng), (5 + 20 * rng.random(), rng.normal(size=3))
    s1 = cable.State(x_0=s.x_L - ql.params.l * s.q, v_0=s.v_L - ql.params.l * np.cross(s.omega, s.q),
                     q=s.q[None], omega=s.omega[None], R=s.R[None], Omega=s.Omega[None])
    v0_dot, omega_dot, Omega_dot = cable.dynamics(s1, (np.array([u[0]]), u[1][None]))
    v_L_dot, omega_dot_L, Omega_dot_L = ql.dynamics(s, u)
    q_dd = np.cross(omega_dot_L, s.q) + np.cross(s.omega, np.cross(s.omega, s.q))
    worst = max(worst, np.max(np.abs(v0_dot - (v_L_dot - ql.params.l * q_dd))), np.max(np.abs(omega_dot[0] - omega_dot_L)),
                np.max(np.abs(Omega_dot[0] - Omega_dot_L)), abs(cable.energy(s1) - ql.energy(s)))
ck.true("quadrotor_cable: one link under one quadrotor equals quadrotor_load (eq4-load)", worst < 1e-10, f"err {worst:.1e}")

# a cable whose joint 0 is very heavy: that joint is in free fall, and the links below it move like a
# hanging chain without gravity, which accelerates both ends equally (thought4-cut)
cable = robots.get("quadrotor_cable", n=3, attach=())
chain = robots.get("flexible_cable", n=3)
cable.params.m = np.r_[1e9, chain.params.m]
cable.params.l = chain.params.l
s = chain.random_state(rng)
s1 = cable.State(x_0=np.zeros(3), v_0=np.zeros(3), q=s.q, omega=s.omega, R=np.zeros((0, 3, 3)), Omega=np.zeros((0, 3)))
a_cable = cable.dynamics(s1, cable.zero_input())
ck.close("quadrotor_cable: under a very heavy joint 0 the links obey eq5-chain with g = 0",
         a_cable[1], chain.dynamics(s, None, chain.Params(n=3, g=0.0))[0], tol=1e-6)
ck.close("quadrotor_cable: ... and joint 0 is in free fall", a_cable[0], -G * E3, tol=1e-6)

# --- listing: cable
cable = robots.get("quadrotor_cable", n=5, attach=(0, 5))  # five links
state = cable.random_state(np.random.default_rng(2), speed=0.5)
E0 = cable.energy(state)
for k in range(1000):
    state = cable.step(state, cable.zero_input(), 1e-3)
print(cable.manifold, cable.dim)   # R^3 x (S^2)^5 x SO(3)^2 19
print(abs(cable.energy(state) - E0) / abs(E0))           # 3e-11
# --- end
cable_drift = float(abs(cable.energy(state) - E0) / abs(E0))
ck.true("listing cable: relative energy drift below 1e-8 over 1 s", cable_drift < 1e-8, f"{cable_drift:.1e}")
ck.record("cable_energy_drift", cable_drift)


# =========================================================================
# 7. Legged robots: telescoping leg (eq4-leg), Reaction Mass Biped, Cassie
# =========================================================================
leg = robots.get("telescoping_leg")
p = leg.params
against_geomech(leg, {"m": p.m, "J_0": p.J_0, "g": p.g, "e3": E3})
# eq4-leg as ch04.py types it for its own check (inline there, with the same m and J_0)
K = -hat(E3) @ hat(E3)
worst = 0.0
for _ in range(20):
    s, f, tau = leg.random_state(rng), 50 * rng.normal(), rng.normal(size=3)
    rho, rhod, R, Om = s.l, s.l_dot, s.R, s.Omega
    Jr = p.J_0 + p.m * rho**2 * K
    rhodd = rho * Om @ K @ Om - G * E3 @ (R @ E3) + f / p.m
    Omd = np.linalg.solve(Jr, tau + p.m * G * rho * np.cross(R.T @ E3, E3) - np.cross(Om, Jr @ Om) - 2 * p.m * rho * rhod * K @ Om)
    l_dd, Omega_dot = leg.dynamics(s, (f, tau))
    worst = max(worst, abs(l_dd - rhodd), np.max(np.abs(Omega_dot - Omd)))
ck.true("telescoping_leg: dynamics equal the lines of ch04.py that check eq4-leg", worst < 1e-12, f"err {worst:.1e}")
summary["telescoping_leg"]["vs_chapter"] = float(worst)
conservation(leg, watch=lambda s: (s.R @ (p.J_0 + p.m * s.l**2 * K) @ s.Omega)[2], watch_name="e3 . R J(l) Omega")

biped = robots.get("reaction_mass_biped")
p = biped.params
ck.close("reaction_mass_biped: total mass 2 m_leg + 6 m_r = 3.684 kg", p.m, 3.684)
against_geomech(biped, {"m_leg": p.m_leg, "m_r": p.m_r, "J_0": p.J_0, "J_P": p.J_P, "g": p.g,
                        "e1": E1, "e2": E2, "e3": E3}, {"slide": ["s1", "s2", "s3"]})
# with no hip torque the stance leg is the telescoping leg
worst = 0.0
leg_p = leg.Params(m=p.m, J_0=p.J_0)
for _ in range(10):
    s, u = biped.random_state(rng), random_input(biped, 3.0)
    u = (u[0], u[1], np.zeros(3), u[3], u[4])
    a = biped.dynamics(s, u)
    b = leg.dynamics(leg.State(l=s.l, l_dot=s.l_dot, R=s.R1, Omega=s.Omega1), (u[0], u[1]), leg_p)
    worst = max(worst, abs(a[0] - b[0]), np.max(np.abs(a[1] - b[1])))
ck.true("reaction_mass_biped: with tau_P = 0 the stance leg obeys eq4-leg", worst < 1e-12, f"err {worst:.1e}")
summary["reaction_mass_biped"]["vs_chapter"] = float(worst)
# the sign of the centrifugal term on the reaction masses (thesis eq. 5.14 prints the opposite sign)
s = biped.random_state(rng)
s.slide, s.slide_dot = np.array([0.1, 0.2, 0.3]), np.zeros(3)
ck.true("reaction_mass_biped: a spinning torso throws its reaction masses outward",
        np.all(biped.dynamics(s, biped.zero_input())[4] > 0))


def biped_momentum(s):
    """e3 component of the total angular momentum about the stance foot."""
    J1 = p.J_0 + p.m * s.l**2 * K
    return (s.R1 @ J1 @ s.Omega1 + s.RP @ biped.torso_inertia(s.slide) @ s.OmegaP + s.R2 @ p.J_0 @ s.Omega2)[2]


conservation(biped, watch=biped_momentum, watch_name="e3 . angular momentum about the foot")

# --- listing: biped
biped = robots.get("reaction_mass_biped")
p, I = biped.params, np.eye(3)
zero = np.zeros(3)
state = biped.State(l=0.5, l_dot=0.0, R1=I, Omega1=zero, RP=I,
                    OmegaP=np.array([0.0, 0.0, 3.0]), R2=I, Omega2=zero,
                    slide=np.full(3, 0.05), slide_dot=zero)
u = (p.m * p.g, zero, zero, zero, zero)                  # the leg holds the weight
momentum = lambda s: s.RP @ biped.torso_inertia(s.slide) @ s.OmegaP
mom0 = momentum(state)
for k in range(800):
    state = biped.step(state, u, 1e-3)
print(state.slide[0], state.OmegaP[2])                   # 0.26 m, 2.21 rad/s
print(np.abs(momentum(state) - mom0).max())              # 7e-14
# --- end
ck.true("listing biped: the masses slide out and the torso slows down",
        abs(state.slide[0] - 0.257) < 2e-3 and abs(state.OmegaP[2] - 2.21) < 5e-3, f"slide {state.slide[0]:.3f} m, spin {state.OmegaP[2]:.3f} rad/s")
ck.true("listing biped: the torso's angular momentum is unchanged", np.abs(momentum(state) - mom0).max() < 1e-10,
        f"{np.abs(momentum(state) - mom0).max():.1e}")
ck.record("biped_slide_m", float(state.slide[0]))
ck.record("biped_spin_rad_s", float(state.OmegaP[2]))

cassie = robots.get("cassie")
p = cassie.params
against_geomech(cassie, {"m_b": p.m_b, "m_h": p.m_h, "m_k": p.m_k, "d": p.d, "J_b": p.J_b, "J_leg": p.J_leg,
                         "g": p.g, "e2": E2, "e3": E3})


def cassie_momentum(s):
    """Horizontal linear momentum of the floating-base model."""
    total = p.m_b * s.v
    for side, R_leg, Om, l, l_dot in cassie._legs(s):
        v_hip = s.v + side * p.d * s.R @ np.cross(s.Omega, E2)
        total = total + p.m_h * v_hip + p.m_k * (v_hip + 0.5 * l_dot * R_leg @ E3 + 0.5 * l * R_leg @ np.cross(Om, E3))
    return total[:2]


conservation(cassie, watch=cassie_momentum, watch_name="horizontal momentum (free fall, no contact)")
s = cassie.random_state(rng)
feet = cassie.feet(s)
ck.close("cassie: feet are a hip width apart when both legs point the same way",
         np.linalg.norm(cassie.feet(cassie.State(**{**s.__dict__, "R_l": s.R_r, "l_l": s.l_r}))[1]
                        - cassie.feet(cassie.State(**{**s.__dict__, "R_l": s.R_r, "l_l": s.l_r}))[0]), 2 * p.d, tol=1e-12)

# =========================================================================
# rollout with a moving input: held over the step, or evaluated at every stage
# =========================================================================
pend = robots.get("spherical_pendulum")
push = lambda t, s: np.array([1.5 * np.sin(3.0 * t), 2.0 * np.cos(2.0 * t), 0.5 * t])   # a force that moves
start = pend.State(q=unit([0.6, 0.2, -0.77]), omega=np.zeros(3))
fine = pend.rollout(start, push, 1.25e-4, 8000, hold=False)[-1].q                          # 1 s
miss = {hold: [np.linalg.norm(pend.rollout(start, push, h, round(1.0 / h), hold=hold)[-1].q - fine)
               for h in (0.02, 0.01)] for hold in (True, False)}
order = {hold: float(np.log2(e[0] / e[1])) for hold, e in miss.items()}
ck.true("rollout, hold=False: a moving input evaluated at every stage keeps the fourth order",
        order[False] > 3.7, f"order {order[False]:.2f}, error {miss[False][1]:.1e}")
ck.true("rollout, hold=True: the same input held over the step is first order",
        0.8 < order[True] < 1.3, f"order {order[True]:.2f}, error {miss[True][1]:.1e}")
const = pend.zero_input()
ck.close("rollout: for a constant input the two options give the same states",
         pend.rollout(start, lambda t, s: const, 1e-2, 50, hold=False)[-1].q,
         pend.rollout(start, const, 1e-2, 50)[-1].q, tol=1e-15)
one = pend.step(start, push, 1e-2, t=0.3)
two = pend.rollout(start, push, 1e-2, 1, hold=False, t0=0.3)[-1]
ck.close("step(..., t=t) is one step of rollout(..., hold=False, t0=t)", one.q, two.q, tol=0.0)
ck.record("rollout_order", {"held": order[True], "staged": order[False]})


# =========================================================================
# 8. Walking the Reaction Mass Biped (robots/biped_walking.py, Chapter 20)
# =========================================================================
# A generator of its own, so that the checks above keep their random states.
import json  # noqa: E402

from geomech import linearize as geomech_linearize  # noqa: E402

from robots import biped_walking as bw  # noqa: E402

rng_w = np.random.default_rng(20)
biped = robots.get("reaction_mass_biped")
p = biped.params
worst = dict(kkt=0.0, stick=0.0, mom=0.0, carnot=0.0, loss=np.inf, relabel=0.0)
for _ in range(50):
    s, l_0 = biped.random_state(rng_w), 0.6 + rng_w.random()
    new, imp, _ = biped.impact(s, l_0)
    v, v_plus, imp_kkt, D, A = bw.impact_kkt(s, l_0)
    v_new = np.concatenate([new.Omega1, new.OmegaP, new.slide_dot, new.Omega2,
                            bw.hip_velocity(new.l, new.l_dot, new.R1, new.Omega1)])
    # eq20-impact against eq20-kkt: same impulse, same velocities after eq20-relabel
    worst["kkt"] = max(worst["kkt"], np.abs(imp - imp_kkt).max(), np.abs(bw.relabel(v_plus) - v_new).max())
    worst["stick"] = max(worst["stick"], np.abs(A @ v_plus).max())
    c = -l_0 * s.R2 @ E3                                # the landing foot, seen from the hip
    H_minus = bw.angular_momentum_about(c, s, v[12:15], s.Omega1, s.OmegaP, s.Omega2)
    H_plus = bw.angular_momentum_about(c, s, v_plus[12:15], v_plus[0:3], v_plus[3:6], v_plus[9:12])
    worst["mom"] = max(worst["mom"], np.abs(H_plus - H_minus).max())
    T_minus = bw.kinetic_floating(s, v[12:15], s.Omega1, s.OmegaP, s.Omega2, s.slide_dot)
    T_plus = bw.kinetic_floating(s, v_plus[12:15], v_plus[0:3], v_plus[3:6], v_plus[9:12], v_plus[6:9])
    dv = v_plus - v
    worst["carnot"] = max(worst["carnot"], abs((T_minus - T_plus) - 0.5 * dv @ D @ dv))
    worst["loss"] = min(worst["loss"], T_minus - T_plus)
    worst["relabel"] = max(worst["relabel"], np.abs(bw.relabel(bw.relabel(v)) - v).max())
ck.true("biped_walking: impact closed form eq20-impact equals the 18 x 18 system eq20-kkt (with eq20-relabel)",
        worst["kkt"] < 1e-10, f"err {worst['kkt']:.1e}")
ck.true("biped_walking: the landing foot is at rest after the impact", worst["stick"] < 1e-12, f"err {worst['stick']:.1e}")
ck.true("biped_walking: angular momentum about the landing foot, summed mass by mass, is conserved by the impact",
        worst["mom"] < 1e-11, f"err {worst['mom']:.1e}")
ck.true("biped_walking: the kinetic energy lost is that of the velocity jump, and never negative",
        worst["carnot"] < 1e-11 and worst["loss"] >= 0, f"err {worst['carnot']:.1e}, smallest loss {worst['loss']:.2e} J")
ck.true("biped_walking: relabeling the legs twice is the identity", worst["relabel"] == 0.0)

# linearize against the linearization that geomech derives from the robot's Lagrangian
L, dW, variables, inputs = biped.lagrangian()
eom = compute_eom(L, dW, variables)
lay_b = StateLayout(variables, inputs, collect_params([L, dW], variables, inputs))
gm_b = load_module(module_source(to_standard_form(eom, variables, inputs), lay_b, lagrangian=L), "biped_walking_stance")
gl_b = load_module(module_source(geomech_linearize(eom, variables, inputs), lay_b), "biped_walking_lin")
gm_p = {"m_leg": p.m_leg, "m_r": p.m_r, "J_0": p.J_0, "J_P": p.J_P, "g": p.g, "e1": E1, "e2": E2, "e3": E3}
wA = wB = 0.0
for _ in range(3):
    s, u = biped.random_state(rng_w, speed=0.5), random_input(biped, 3.0)
    x, _ = geomech_view(biped, lay_b, s, biped.dynamics(s, u), {"slide": ["s1", "s2", "s3"]})
    uf = flat_input(u)
    a = accelerations(gm_b, x, uf, gm_p)
    Mg = np.asarray(gl_b.M_lin(x, a, uf, gm_p))
    Ag = np.linalg.solve(Mg, np.asarray(gl_b.A_lin(x, a, uf, gm_p)))
    Bg = np.linalg.solve(Mg, np.asarray(gl_b.B_lin(x, a, uf, gm_p)))
    A, B = biped.linearize(s, u)
    wA = max(wA, np.abs(Ag - A).max() / (1 + np.abs(Ag).max()))
    wB = max(wB, np.abs(Bg - B).max())
ck.true("reaction_mass_biped: linearize equals geomech linearize at random stance states, all 26 rows",
        wA < 1e-7 and wB < 1e-9, f"A rel {wA:.1e}, B {wB:.1e}")

# the tracking controller: thesis law by default, eq6-law per factor with BookGains; equal on the primitive
gait = bw.Gait()
s_d, d = bw.primitive_state(0.37, 0.4, gait), bw.primitive(0.37, 0.4, gait)
ck.close("reaction_mass_biped: on the primitive the two tracking laws give the same input (eq20-law, eq6-law)",
         bw.pack_u(biped.tracking_controller(s_d, d)), bw.pack_u(biped.tracking_controller(s_d, d, bw.BookGains())),
         tol=1e-10)

# a short walk reproduces the step lengths that verify/ch20.py records for the chapter
quoted = json.loads((HERE / "ch20_values.json").read_text())["walk_straight"]
_, walked = bw.walk([0.0] * 2, record=0)
got = [st["step_length"] for st in walked]
ck.true("biped_walking: two steps of walk reproduce the step lengths of ch20_values.json exactly",
        got == quoted["step_lengths"][:2] and walked[0]["step_time"] == quoted["first_step_time"],
        f"{got[0]:.6f} m, {got[1]:.6f} m")

ck.record("summary", summary)
print("\n  robot                              check_eom  vs geomech  vs chapter  manifold  energy drift (order)")
for tag, row in summary.items():
    vc = f"{row['vs_chapter']:.1e}" if "vs_chapter" in row else "   -   "
    print(f"  {tag:34s} {row['check_eom']:.1e}    {row['vs_geomech']:.1e}     {vc}     "
          f"{row.get('manifold_residual', float('nan')):.0e}    {row.get('energy_drift', float('nan')):.1e} "
          f"({row.get('energy_order', float('nan')):.1f})")
ck.finish()
