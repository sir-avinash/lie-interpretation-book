---
title: Symbolic derivation
short: Symbolic derivation
id: ch-symbolic
summary: The variational recipe is term rewriting, so a program can run it on the manifold. How it works, how to check it, and how to turn its output into a simulation.
notebook: notebooks/ch05_geomech_symbolic_pipeline.ipynb
verify: verify/ch05.py
---

Here are the equations of motion of a quadrotor carrying a load on a cable. They are copied from a PhD thesis on aerial transportation (Wu, 2017), where every controller is designed on them.

$$
\begin{aligned}
(m_Q + m_L)\,(\ddot x_L + g\,e_3) &= \big(q \cdot f R e_3 \;-\; m_Q\, l\; \dot q \cdot \dot q\big)\, q, \\
m_Q\, l\; \dot\omega &= -\,q \times f R e_3, \\
J \dot\Omega + \Omega \times J\Omega &= \tau .
\end{aligned}
\label{eq5-wu}
$$

The load is at $x_L$. The unit vector $q$ points from the quadrotor down the cable to the load, with $\dot q = \omega \times q$. The quadrotor has attitude $R$, thrust $f$ and torque $\tau$. Now one question. Look at the term $m_Q\, l\; \dot q \cdot \dot q$ in the first line. It is printed with a minus sign. Should it be a plus?

Try to decide before reading on. Staring does not help. You could re-derive the equations, as [](#ch-dynamics) does over several pages. Or you could fly. In hover the cable hangs still, $\dot q = 0$, and the term is zero. A model with the wrong sign matches every hover log you will ever record. The error only appears when the load swings. For a 1.5 kg quadrotor on a 1 m cable swinging at 2 rad/s, the wrong sign invents a force of 12 N along the cable. The load weighs 2.9 N.

So one sign can pass every bench test and fail in the first aggressive manoeuvre. This chapter offers a third option. Let a program do the derivation, since a program does not drop signs. Then test its answer with a check that shares none of its reasoning.

## Seventeen lines {#sec5-seventeen}

The program is `geomech`, a Python library. You give it a Lagrangian and the virtual work of the inputs. It returns the equations of motion. Here is the whole quadrotor with its load.

```python
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
```

Read it like the first page of a derivation. Constants are flagged so that their variations vanish. The three configuration variables are declared by the manifold they live on: a position in $\R^3$, a direction on $S^2$, an attitude on $\SO(3)$. The library hands back the matching velocities and variations: `omega` is the $\omega$ of $\dot q = \omega \times q$, `xi` is the $\xi$ of $\delta q = \xi \times q$, and `Omega`, `eta` are the $\Omega$ and $\eta$ of $\dot R = R\,\Omega^\wedge$ and $\delta R = R\,\eta^\wedge$. The Lagrangian is kinetic minus potential energy. The virtual work says the thrust $fRe_3$ acts at the quadrotor, whose virtual displacement is $\delta x_Q = \delta x_L - l\,\xi \times q$, and the torque pairs with $\eta$.

The derivation and the check together run in well under a second. The result is one equation per variation. Two of them, in our notation:

$$
\begin{aligned}
\delta x_L:&\quad -(m_Q + m_L)(\ddot x_L + g\,e_3) + m_Q\, l\; \dot\omega \times q \;-\; m_Q\, l\,(\omega\cdot\omega)\, q + f R e_3 = 0, \\
\eta:&\quad -J\dot\Omega - \Omega \times J\Omega + \tau = 0 .
\end{aligned}
\label{eq5-quad-rows}
$$

There is the disputed term, with a minus sign. (Since $\omega \perp q$, $\dot q\cdot\dot q = \omega\cdot\omega$.) The third equation, for the cable, has seven terms and waits in [](#sec5-verify). The last line sets `err` to $4.1\times10^{-8}$. That number will matter more than the equations.

Two things should bother you. Why believe a program over a thesis? And what did the program actually do?

## Why not ask a computer algebra system? {#sec5-trees}

Take the second question first. Computer algebra systems have differentiated Lagrangians for decades. Why a separate tool?

A general system differentiates functions of real variables. Ask it for $\partial L / \partial R$ and it wants to know what $R$ is a function of. You can answer with three angles, and you are back in [](#ch-why) with a singularity. You can answer with nine matrix entries and six constraints, and you now carry Lagrange multipliers for constraints the geometry already knows. Either way you have left the manifold to please the software.

Now look again at the recipe of [](#ch-dynamics). It never differentiates with respect to $R$. It does four things. Apply $\delta$ with the product rule. Replace the variation of each manifold quantity with a known expression. Integrate by parts. Read off what multiplies each variation. None of these steps needs coordinates. Each one replaces a pattern of symbols with another pattern.

So the right object is not a function of numbers. It is a **tree** ([](#fig5-tree)). Interior nodes are operations: `Add`, `Mul`, `Dot`, `Cross`. Leaves are symbols, and every leaf carries a type that says where it lives.

![The Lagrangian of the spherical pendulum as the library stores it, drawn from the stored object. Products are binary, so $\tfrac12 m l^2$ is a chain of three `Mul` nodes. The colour of a leaf is its type. That type is all the program knows about $q$, and it is enough.](figures/ch05-tree.svg){#fig5-tree}

The type is what lets a rewrite rule do geometry. The program does not know what a pendulum is. It knows that the variation of a leaf of type "point on $S^2$" is $\xi \times q$, that the variation of a leaf flagged constant is zero, and so on down [](#tbl5-leaves).

::: {.table #tbl5-leaves}
| Declared as | Lives on | Velocity | Variation | Variation of the velocity |
| :-- | :-- | :-- | :-- | :-- |
| `Scalar("s")` | $\R$ | $\dot s$ | $\delta s$ | $\tfrac{\dd}{\dd t}\delta s$ |
| `Vector("x")` | $\R^3$ | $\dot x$ | $\delta x$ | $\tfrac{\dd}{\dd t}\delta x$ |
| `S1("q", axis=e)` | circle about $e$ | $\dot q = \omega\, e \times q$ | $\delta q = \gamma\, e\times q$ | $\delta\omega = \dot\gamma$ |
| `S2("q")` | $S^2$ | $\dot q = \omega \times q$ | $\delta q = \xi \times q$ | $\delta\omega = \dot\xi - 2\,\omega\times\xi$ |
| `SO3("R")` | $\SO(3)$ | $\dot R = R\,\Omega^\wedge$ | $\delta R = R\,\eta^\wedge$ | $\delta\Omega = \dot\eta + \Omega\times\eta$ |

The five leaf types and the rules attached to them. The last column is the content of [](#ch-variations), stored as data.
:::

Nodes cannot be modified once built. A rewrite never edits a tree. It builds a new one. So every intermediate stage of a derivation still exists afterwards, and you can ask to see it.

## One pendulum, every stage {#sec5-pipeline}

The quadrotor is too big to watch. Take the smallest system that has a curved configuration space: a mass $m$ on a rigid rod of length $l$, pivoting freely about the origin. The rod direction is $q \in S^2$. A torque $\tau$ acts at the pivot.

$$
L = \tfrac12\, m\, l^2\; \omega\cdot\omega \;-\; m\, g\, l\; q\cdot e_3, \qquad \delta W = \xi \cdot \tau .
$$

```python
m, g, l = getScalars("m g l", attr=["Constant"])
e3 = Vector("e3", attr=["Constant"])
q, tau = S2("q"), Vector("\\tau")               # rod direction, torque at the pivot
omega, xi = q.get_tangent_vector(), q.get_variation_vector()

L = half * m * l * l * Dot(omega, omega) - m * g * l * Dot(q, e3)
dW = Dot(xi, tau)
variables = SystemVariables(vectors=[q])

trace = EOMTrace()
eom = compute_eom(L, dW, variables, trace=trace)
```

The `trace` argument asks the pipeline to keep every stage. It records thirteen. [](#fig5-pipeline) shows the five that change the expression, exactly as the library prints them. Follow them with a pencil.

![The spherical pendulum going through the pipeline. Each box is the library's own output for that stage. The bars count tree nodes: the expression grows when the manifold rule is substituted, then shrinks as identities fire. The last box is the equation of motion.](figures/ch05-pipeline.svg){#fig5-pipeline}

**Vary.** The product rule runs down the tree. The kinetic term gives $\delta\omega\cdot\omega + \omega\cdot\delta\omega$, kept as two terms because nothing has yet said the dot product is symmetric. The point $q$ is replaced on the spot by $\xi\times q$. The variation of the velocity, $\delta\omega$, is left as an unevaluated symbol.

**Apply the manifold rule.** Every $\delta\omega$ becomes $\dot\xi - 2\,\omega\times\xi$, the $S^2$ row of [](#tbl5-leaves). The tree is now at its largest, 46 nodes. On paper you would write the same line. This is also where hand derivations go wrong: a flipped cross product, a forgotten factor of 2.

**Simplify.** Small identities are applied until nothing changes. One of them is that a vector is perpendicular to its own cross products, $\omega\cdot(\omega\times\xi) = 0$. It deletes the term that was just introduced. For this Lagrangian the factor of 2 was never going to matter, and the program found that out by trying. Then the two halves merge and the $\tfrac12$ is absorbed. On paper you would have jumped here "by inspection". The program cannot inspect. Over the whole derivation it logged 44 rule applications, each with a name. Two of them:

```text
vector.dot_cross_orthogonal: <cross(v:\omega_{q}, v:\xi_{q}), v:\omega_{q}> ==> 0
ibp.dot: <d/dt(v:\xi_{q}), v:\omega_{q}> ==> <-v:\xi_{q}, d/dt(v:\omega_{q})>
```

**Integrate by parts.** The term $m l^2\, \dot\xi\cdot\omega$ becomes $-m l^2\, \xi\cdot\dot\omega$. The boundary term is dropped, because variations vanish at the end times. For a larger system this stage first has to rearrange each term until $\dot\xi$ stands alone on one side of a dot product. It uses $(a\times b)\cdot c = a\cdot(b\times c)$ and $(Af)\cdot w = f\cdot(A\T w)$ to get there.

**Extract.** Every term is now $\xi$ dotted with something. The gravity term needs one last turn of the triple product, $(\xi\times q)\cdot e_3 = \xi\cdot(q\times e_3)$. Collect what multiplies $\xi$:

$$
E_\xi \;=\; -\,m\, l^2\, \dot\omega \;-\; m\, g\, l\; q\times e_3 \;+\; \tau \;=\; 0 . \label{eq5-pendulum}
$$

Sanity check: hang the rod straight down, $q = -e_3$, and $q\times e_3 = 0$. No torque, no acceleration.

::: {.thought #thought5-clerk name="The clerk who knows no mechanics"}
Hand the five boxes of [](#fig5-pipeline) to a careful clerk who has never seen a pendulum. Give them one card with the rules: the product rule, the $S^2$ row of [](#tbl5-leaves), $a\cdot(a\times b)=0$, the triple product, and "move the dot from $\xi$ to its partner and flip the sign". Ask them to check that each box follows from the one above.

They can. Nothing on the card mentions energy, force or a sphere. That is what it means for the derivation to be mechanical, and it is why a machine can do it. It is also a warning. The clerk will apply a wrong card just as carefully as a right one.
:::

Notice the form of [](#eq5-pendulum). Everything is on one side and the inertia term is negative, because $E_\xi$ is literally the coefficient of $\xi$ in $\delta S$. The library does not rearrange.

The clerk's warning is the open problem. Exact is not the same as right. A rule in the library could be wrong. Or the equation you are holding came from somewhere else. What would convince a sceptic?

## Trust, but verify {#sec5-verify}

A sceptic wants a second derivation that shares nothing with the first. There is one. The Euler–Lagrange equations can be evaluated numerically from $L$ alone, with no algebra at all.

That is what `check_eom` does. It draws a random state: a random point on each manifold, random velocities, random accelerations, random inputs, random values for every constant. It evaluates the number $L$ at nearby states and forms $\tfrac{\dd}{\dd t}\,\partial L/\partial v - \partial L/\partial q$ by central differences, in the constrained forms that hold on $S^2$ and $\SO(3)$. It adds the forcing read from $\delta W$. Then it evaluates the symbolic $E$ at the same state and compares. No product rule, no manifold rule, no integration by parts. The only thing the two computations share is $L$. The function returns the worst relative error over five random states.

For the quadrotor it returned $4.1\times10^{-8}$. That is the noise floor of a finite difference. Now run the experiment from the opening. Type in the load equation as if from a page of algebra, once with each sign.

```python
aL, domega = xL.t_diff().t_diff(), omega.t_diff()


def load_equation(sign):            # the x_L row, typed in from a page of algebra
    return ((aL + e3 * g) * (mQ + mL) * (-1) + Cross(domega, q) * (mQ * l)
            + q * (Dot(omega, omega) * mQ * l) * sign + (R * e3) * f)


for sign in (-1, +1):
    hand = dict(eom)
    hand[str(xL.delta())] = (xL.delta(), load_equation(sign))
    print(sign, check_eom(L, dW, variables, hand, inputs=inputs))
```

```text
-1 4.059809332153562e-08
1 1.9027858460945652
```

Seven orders of magnitude apart ([](#fig5-sign-error), left). The companion function `assert_eom` says which equation is at fault: `equation for '\delta{x_L}' disagrees with the Euler-Lagrange residual: relative error 1.903e+00 >= 1.0e-05`.

![Left: the check applied to the hand-typed load equation at 20 different random seeds. With the correct sign the error stays at finite-difference noise. With the sign flipped it is of order one. Right: why flight testing in hover would not have found it. The spurious force grows with the square of the swing rate and is zero when the cable hangs still.](figures/ch05-sign-error.svg){#fig5-sign-error}

::: {.example #ex5-sign name="Why the random state matters"}
With $m_Q = 1.5$ kg and $l = 1$ m, the two versions of the load equation differ by $2\,m_Q\, l\,\|\omega\|^2\, q$. At $\omega = 0$ that is nothing. At 2 rad/s it is 12 N, or 6.7 m/s² of load acceleration, against a load weight of 2.9 N. A test at hover checks the equation at the one state where the wrong term is silent. The check draws $\omega$ at random, so every term is exercised at once. Over 20 seeds the wrong equation scored between 0.30 and 2.8. The right one never exceeded $9.0\times10^{-8}$.
:::

Now the cable equation, which we skipped. The program returns

$$
\begin{aligned}
E_\xi \;=\;& m_Q\, l\; q\times\ddot x_L \;-\; m_Q\, l^2\; q\times(\dot\omega\times q) \;+\; m_Q\, g\, l\; q\times e_3 \;-\; l f\; q\times R e_3 \\
&+\; m_Q\, l\,\big[(\omega\times q)\times\dot x_L \;-\; 2\,\omega\times(q\times\dot x_L) \;+\; q\times(\omega\times\dot x_L)\big].
\end{aligned}
\label{eq5-cable}
$$

The first line has a counterpart in the thesis. The bracket has none. Three terms in the load's velocity, which [](#eq5-wu) never mentions. Is the program wrong, or the thesis?

::: {.thought #thought5-rod name="Pushing along the rod"}
Hold a pendulum by its pivot and push the bob straight along the rod, toward the pivot. Nothing happens. The rod takes the push. Only the part of a force that lies across the rod can move the bob.

The equations inherit this. A variation $\xi$ is always perpendicular to $q$, so $\xi\cdot E$ only ever tests the part of $E$ that lies in the tangent plane. Whatever $E$ has along $q$ is multiplied by zero in $\delta S$. It is not an equation. It is not anything.
:::

![The coefficient $E$ of $\xi$ for the cable, evaluated at one state and drawn at $q$. Only its tangent part is an equation of motion. The program does not remove the part along $q$, and two correct answers may differ there.](figures/ch05-tangent.svg){#fig5-tangent}

Apply the Jacobi identity to the bracket in [](#eq5-cable) and it collapses to $-(\omega\cdot\dot x_L)\,q$. It points along $q$ ([](#fig5-tangent)). It is the push along the rod. The simplifier has no Jacobi rule, so the three terms survive, and they are harmless. The check compares tangent parts only. Add any multiple of $q$ to the cable equation and it still passes, as it should.

::: {.warning #warn5-conventions name="Two conventions that will bite"}
*Signs.* Every equation is $E = 0$ with $E = -(\text{Euler–Lagrange expression}) + \text{forcing}$. A point mass gives $-m\ddot x + F = 0$. If you copy an equation into a paper, move the inertia across yourself.

*Tangent plane.* For a variable on $S^2$ the equation is $(I - qq\T)\,E = 0$, together with $q\cdot\dot\omega = 0$. Never solve $E = 0$ as three equations in three unknowns. The answer will have $q\cdot\dot\omega \neq 0$ and the trajectory will leave the sphere.
:::

Some mistakes never reach the check. The library refuses, with a message that names the culprit. Two you will meet in your first week:

```text
ValueError: compute_eom: δS still contains the variation symbols '\delta{e3}' of the
symbols 'e3', which are neither entries of SystemVariables nor declared in params; ...

TypeError: compute_eom: SystemVariables.vectors entry '\omega_{q}' is a tangent or
variation vector, not a configuration variable; list the manifold point 'q' instead
```

The first means you forgot `attr=["Constant"]` on $e_3$, so the program varied it. The second means you listed a velocity where a configuration was wanted. A forgotten flag does not always raise an error. Leave `"SymmetricMatrix"` off the inertia and you get a correct equation with $\tfrac12(J + J\T)$ throughout. The program will not assume what you did not declare.

The check has two blind spots. It measures error relative to the size of the whole equation, so a wrong term that is tiny next to its neighbours can hide. Leave parameters symbolic and they are drawn of order one, which avoids this. And it compares the equation with the Lagrangian, not the Lagrangian with the robot. Type the wrong kinetic energy and you get the right equations for the wrong machine, verified to eight digits.

Verified equations are still only symbols. To see whether they describe a quadrotor, they have to move.

## From equations to motion {#sec5-simulate}

An integrator needs accelerations. Each $E$ is linear in them and in the inputs, so it splits as

$$
M(q)\,a \;+\; b(q, v) \;+\; G(q)\,u \;=\; 0, \label{eq5-standard}
$$

with $a = (\ddot x_L, \dot\omega, \dot\Omega)$ and $u = (f, \tau)$. The function `to_standard_form` performs the split symbolically. Because of the sign convention, $M$ is minus the physical inertia: the load row has $-(m_Q + m_L) I$, the attitude row has $-J$. The cable row has $m_Q l^2\, q^\wedge q^\wedge = -m_Q l^2 (I - qq\T)$, which has rank two. That is the tangent plane again, showing up as a singular block.

```python
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
```

`module_source` writes [](#eq5-standard) out as a plain NumPy module, 311 lines for this system, with functions for $M$, $b$, $G$, the kinematics and the energy. It no longer needs the symbolic library. `simulate` integrates it with a fourth-order Runge–Kutta–Munthe-Kaas scheme. The idea is a familiar one: do not add increments to $q$ and $R$. Each step moves $q$ by a rotation and $R$ by a matrix exponential, so the state cannot leave the manifold. The $S^2$ rows are projected as the pitfall above demands.

The run in the listing holds the thrust at the total weight, with zero torque. The cable starts 0.8 rad from vertical and circling, and the centre of mass coasts at 0.45 m/s ([](#fig5-quadload)).

![The simulated trajectory, five snapshots over 6 s. The thrust cancels the total weight, so the centre of mass moves on a straight line. The light load swings wide below, and the heavy quadrotor wobbles about the same line with one fifth of the radius.](figures/ch05-quadload.svg){#fig5-quadload}

A trajectory figure proves little. Conserved quantities prove more.

::: {.example #ex5-conservation name="Four checks on one simulation"}
*Constraints.* Over 6000 steps, $\|q\| - 1$, $q\cdot\omega$ and $R\T R - I$ stay below $3\times10^{-13}$.

*Energy.* With no inputs the system falls freely and energy must be constant. The relative drift is $1.0\times10^{-12}$. Coarser steps of 0.04, 0.02 and 0.01 s give $1.0\times10^{-8}$, $6.5\times10^{-10}$ and $4.1\times10^{-11}$: a factor of 16 per halving, the signature of a fourth-order method.

*Work.* With thrust on, energy is not conserved. It varies over a range of 0.75 J. But a constant vertical force has a potential, so $E - f\,e_3\cdot x_Q$ should be constant. It is, to $7\times10^{-12}$ J.

*Momentum.* The centre of mass stays on its straight line to $1.3\times10^{-10}$ m. The quadrotor circles it at 0.167 m, the load at 0.833 m. The ratio is $m_Q/m_L = 5$.
:::

Now settle the opening question. Equation [](#eq5-wu) is solved for the accelerations. The program's rows [](#eq5-quad-rows) and [](#eq5-cable) are coupled. Substitute one into the other and two lines of algebra give the thesis form. Numerically, the accelerations from the generated code and from [](#eq5-wu) agree to $2.5\times10^{-14}$ at 50 random states. The thesis is right. The sign is a minus.

A controller needs one more thing: the dynamics linearized about a trajectory. That is one more variation. Apply $\delta$ to $E$ itself, with the same rules, and collect terms. The function `linearize` does this and returns $M\,\delta\dot x = A\,\delta x + B\,\delta u$ in the variation coordinates $(\delta x_L, \xi, \eta)$. [](#ch-linearization) develops it.

## More bodies {#sec5-scaling}

Nothing in the pipeline cared that the quadrotor lives on a product, $\R^3\times S^2\times\SO(3)$. We listed three variables and got three equations. Larger products cost typing and little else. The second model of Wu's thesis has several quadrotors carrying one rigid body by cables. It is among the 30 systems in the library's catalogue. With three quadrotors it has 21 degrees of freedom and eight coupled equations. They derive in a fraction of a second and check at $9.0\times10^{-8}$.

A harder request: a chain of $n$ spherical pendulums, where $n$ is a symbol. There are $n$ equations and you cannot type them. The `geomech.indexed` module lets you write the Lagrangian with sums.

```python
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
```

The result is a single equation for link $i$, with sums whose limits depend on $i$. On the tangent plane of $q_i$ it reads

$$
\sum_{j=1}^{n} M_{ij}\,\Big[-\,q_i\times(\dot\omega_j\times q_j) + (\omega_j\cdot\omega_j)\; q_i\times q_j\Big]
\;-\; \Big(\sum_{k=i}^{n} m_k\Big)\, g\, l_i\; q_i\times e_3 \;=\; 0,
\qquad
M_{ij} = \Big(\sum_{k=\max(i,j)}^{n} m_k\Big)\, l_i\, l_j . \label{eq5-chain}
$$

Read $M_{ij}$ aloud: the mass hanging below both link $i$ and link $j$, times the two lengths. Nobody typed the $\max(i,j)$. The Lagrangian contains $\sum_i m_i\,\|\sum_{j\le i} l_j \dot q_j\|^2$, and exchanging the order of summation produces that bound. The program does the exchange. (Its raw output also carries three terms per pair that point along $q_i$, the same Jacobi leftovers as before.)

The loop at the end of the listing instantiates the generic equation for $n = 1$ to $6$ and checks each against the Lagrangian expanded for the same $n$. The errors run from $6.2\times10^{-9}$ to $1.1\times10^{-7}$. [](#fig5-chain) simulates $n = 4$.

![A chain of four equal links, released from the horizontal while turning about the vertical, simulated from the generic equation instantiated at $n=4$. Seven instants are overlaid, light to dark. Energy drifts by $2.6\times10^{-8}$ over 2 s at the default step: the chain whips, and a fixed step feels it.](figures/ch05-chain.svg){#fig5-chain}

How does a sum fit into a tree? It does not. A sum is kept as a list of ordinary trees whose leaves carry an index, with the inequalities between indices stored beside them. Each tree goes through the same five stages. The inequalities become the limits of the sums that remain.

## What it cannot do {#sec5-limits}

A tool you trust is a tool whose edges you know.

The Lagrangian must depend on configurations and velocities only. Put an acceleration in it and the library stops with "the Lagrangian depends on accelerations ... higher-order Lagrangians are not supported". (Any power of the velocity is fine. A quartic kinetic term derives and checks.)

The manifolds are $\R$, $\R^3$, $S^1$, $S^2$, $\SO(3)$ and their products. $\SE(3)$ exists only as $\R^3\times\SO(3)$ with the inertial velocity $\dot x$. There is no way to declare a new manifold from outside.

Constraints must be built into the configuration space, as the cable is built into $S^2$. Rolling constraints, closed kinematic chains and contact are out of scope. The cable here is always taut. A cable that goes slack is a different model with a different configuration space, and switching between them is your job.

The simplifier is not canonical. Two correct answers can print differently, as the bracket in [](#eq5-cable) showed. Compare equations by value, with the check, never by eye.

And once more: the check certifies the equations against $L$. It cannot certify $L$ against the world.

## Exercises {#sec5-exercises .unnumbered}

::: {.exercise #exr5-sign-physical}
Settle the sign in [](#eq5-wu) by physics alone. Cut the thrust, $f = 0$, so that both bodies fall freely while whirling about their centre of mass. Which way must the cable pull the load? Compare with the first line of [](#eq5-wu) and with the tension the thesis gives, $T = \frac{m_L}{m_Q + m_L}\,(m_Q\, l\,\dot q\cdot\dot q - q\cdot fRe_3)$.
:::

::: {.exercise #exr5-factor-two}
Derive the $S^2$ rule $\delta\omega = \dot\xi - 2\,\omega\times\xi$ from $\omega = q\times\dot q$ and $\delta q = \xi\times q$ with $\xi\cdot q = 0$. Show that the second term points along $q$. Then find a Lagrangian for which it does not drop out, derive its equations in the notebook, and locate the stage of the trace where the term survives.
:::

::: {.exercise #exr5-rigid-pendulum}
Walk the 3D pendulum, $L = \tfrac12\,\Omega\cdot J\Omega - m g\, e_3\cdot R\rho$ with $\delta W = \eta\cdot\tau$, through the pipeline by hand, one box per stage as in [](#fig5-pipeline). Then run it with a trace and compare. You should reach $-J\dot\Omega - \Omega\times J\Omega - m g\,\rho\times R\T e_3 + \tau = 0$. Which term comes from $\delta\Omega = \dot\eta + \Omega\times\eta$, and why does it not vanish here as it did on the sphere?
:::

::: {.exercise #exr5-jacobi}
Prove that the bracket in [](#eq5-cable) equals $-(\omega\cdot\dot x_L)\,q$. Then eliminate $\dot\omega$ between [](#eq5-cable) and the first row of [](#eq5-quad-rows) to recover the first line of [](#eq5-wu).
:::

::: {.exercise #exr5-plant}
Plant your own error. In the notebook, change one sign or one factor in the cable equation and run `check_eom`. Find a change that the check cannot see, and explain in one sentence why it is not an error.
:::

::: {.exercise #exr5-elastic}
Replace the rigid cable by a spring: add a scalar variable for the cable length, with elastic energy $\tfrac12 k (l - l_0)^2$. Derive, check and simulate. How many equations are there now? Then use [](#eq5-chain) to write the mass matrix of a three-link chain and compare it with `to_standard_form` at $n = 3$.
:::

## Notes and sources {#sec5-notes .unnumbered}

The variational method that the pipeline automates, the rules of [](#tbl5-leaves) and the chain of pendulums behind [](#eq5-chain) are set out by Lee, Leok and McClamroch (2018). Bittner and Sreenath (2016) were the first to automate the derivation of coordinate-free equations on manifolds, with a package written in Scala. `geomech` (Kotaru, 2026) is a Python library in the same line. It adds the numerical check, linearization, code generation, simulation and sums over a symbolic number of bodies. The numbers in this chapter were produced with version 1.1.0.

The quadrotor with a suspended load, [](#eq5-wu), and the multi-quadrotor rigid-payload model are taken from Wu (2017, ch. 3). The single-quadrotor model was first derived by Sreenath, Lee and Kumar (2013). For a sense of what the automation replaces, see the reaction mass biped of Siravuru (2019, ch. 5): thirteen degrees of freedom, with three attitudes on $\SO(3)$, leg lengths and sliding reaction masses, derived by hand from a variational principle over several pages. The integrator is due to Munthe-Kaas (1999).

- B. Bittner and K. Sreenath, "Symbolic computation of dynamics on smooth manifolds," *Workshop on the Algorithmic Foundations of Robotics (WAFR)*, 2016.
- P. Kotaru, *geomech: symbolic equations of motion on smooth manifolds*, version 1.1.0, 2026. https://github.com/vkotaru/geomech
- T. Lee, M. Leok and N. H. McClamroch, *Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds*, Springer, 2018.
- H. Munthe-Kaas, "High order Runge–Kutta methods on manifolds," *Applied Numerical Mathematics* 29(1), 1999.
- A. Siravuru, *Geometric Control and Learning for Dynamic Legged Robots*, PhD thesis, Carnegie Mellon University, 2019.
- K. Sreenath, T. Lee and V. Kumar, "Geometric control and differential flatness of a quadrotor UAV with a cable-suspended load," *IEEE Conference on Decision and Control*, 2013.
- G. Wu, *Safety-critical Geometric Control Design with Application to Aerial Transportation*, PhD thesis, Carnegie Mellon University, 2017.
