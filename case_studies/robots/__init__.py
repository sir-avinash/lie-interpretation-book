"""The robot model library of the book's case studies.

Every robot of the book is one module here, the single place where its model is
written down. All of them have the same small interface:

    import robots
    robot = robots.get("quadrotor_load")          # see robots.list_robots()

    robot.manifold, robot.factors, robot.dim      # "R^3 x S^2 x SO(3)", its factors, 8
    robot.params                                  # dataclass: book symbols, SI units
    robot.labels                                  # book labels that define the model
    state = robot.State(...)                      # dataclass: x_L, v_L, q, omega, R, Omega
    state = robot.random_state(rng)               # or a random state on the manifold

    acc = robot.dynamics(state, u, params)        # accelerations, one per factor
    state = robot.step(state, u, h, params)       # one step that stays on the manifold
    robot.energy(state, params)                   # kinetic plus potential energy
    robot.on_manifold(state)                      # largest constraint violation
    robot.lagrangian()                            # geomech (L, dW, variables, inputs)

Where the book has them, a robot also has `tracking_controller(...)` (Chapter 6)
and `linearize(...)` (Chapter 7). `params` may be omitted everywhere: the robot
then uses its own `robot.params`.

Conventions (Appendix A): world frame with e3 up, gravity along -e3;
R_dot = R hat(Omega) with Omega in the body frame; q_dot = omega x q with omega
in the world frame; a cable direction q points from the quadrotor to the load,
x_Q = x_L - l q; an input is u = (f, tau), thrust and body torque.

Only NumPy is needed to run a model. geomech is imported inside `lagrangian()`.
"""
from . import (cassie, flexible_cable, multi_quadrotor_payload, quadrotor, quadrotor_load,
               reaction_mass_biped, rigid_body, spherical_pendulum)
from .core import (E1, E2, E3, Factor, Robot, Psi_q, Psi_R, e_omega, e_Omega, e_q, e_R, expm_so3,
                   hat, log_so3, tangent, unit, vee)

_REGISTRY = {
    "spherical_pendulum": spherical_pendulum.SphericalPendulum,
    "rigid_body": rigid_body.FreeRigidBody,
    "pendulum_3d": rigid_body.Pendulum3D,
    "quadrotor": quadrotor.Quadrotor,
    "quadrotor_load": quadrotor_load.QuadrotorLoad,
    "multi_quadrotor_load": multi_quadrotor_payload.MultiQuadrotorLoad,
    "multi_quadrotor_payload": multi_quadrotor_payload.MultiQuadrotorPayload,
    "flexible_cable": flexible_cable.HangingChain,
    "quadrotor_cable": flexible_cable.QuadrotorCable,
    "telescoping_leg": reaction_mass_biped.TelescopingLeg,
    "reaction_mass_biped": reaction_mass_biped.ReactionMassBiped,
    "cassie": cassie.Cassie,
}


def list_robots():
    """Names accepted by `get`."""
    return list(_REGISTRY)


def get(name, **kwargs):
    """A robot by name. Sized robots take their size: get("flexible_cable", n=6),
    get("multi_quadrotor_load", n=3), get("quadrotor_cable", n=5, attach=(0, 5))."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown robot {name!r}; choose from {list_robots()}")
    return _REGISTRY[name](**kwargs)
