---
title: Filtering on Lie groups
short: Filtering
id: ch-filtering
summary: An estimation error whose dynamics do not depend on the estimate, the invariant extended Kalman filter built on it, and what to do when the state is a sphere and not a group.
notebook: notebooks/ch14_invariant_filtering.ipynb
verify: verify/ch14.py
---

A drone is switched on in someone's hands. Its filter is told that it sits level and at rest. In fact it is being carried upside down, 170° from where the filter believes it is.

The extended Kalman filter now does what it always does. It linearizes the model about its current estimate. It pushes a covariance through that linearization. It weighs the next measurement by that covariance. Every one of these steps takes the estimate's word for where the vehicle is, and the estimate is wrong. So the linearization is wrong, so the covariance is wrong, so the gain is wrong. The filter consults a mistaken model to decide how much to trust itself. [](#fig14-opening) shows what follows.

Here is a question that sounds like wishful thinking. Could the estimation error be defined so that its dynamics do not involve the estimate at all? Then the covariance would be computed from a model that is right however wrong the estimate is. For inertial navigation the answer is yes. It takes the right group and the right error.

![Two filters, one bad start. A vehicle flies a banked loop past three landmarks, and both filters begin 170° wrong in attitude with the same covariance and the same readings. The truth is the dashed gray path, here and in every figure of this chapter. The invariant filter (indigo) is on the path within a few seconds. The conventional one (terracotta) wanders meters away first and comes back the long way round.](figures/ch14-opening-strip.png){#fig14-opening anim="figures/ch14-opening.gif"}

## A compass and a gyroscope {#sec14-circle}

Start with one angle. A robot turns at 4°/s. A gyroscope reads the turn rate fifty times a second with a noise of 0.5°/s, and a compass reads the heading five times a second with a noise of 3°. A Kalman filter with one state blends them. Between compass readings it integrates the gyroscope and lets its variance grow. At each reading it forms the **innovation**, the predicted reading minus the actual one, and removes a fraction of it from the estimate. The fraction is the **Kalman gain**.

Let the true heading start at 170° and pass 180° after 2.5 s. The compass, which reports in $(-180^\circ, 180^\circ]$, jumps from 179° to −179°. The estimate says 181°. A filter that subtracts gets an innovation of 360°, and its gain, tuned for noise of a few degrees, starts dragging the estimate down through 90° toward −179° the long way round ([](#fig14-compass)). The heading error reaches 179°. When the run ends, 9.5 s after the wrap, it is still 73°.

![A heading filter at the wrap. The truth (dashed gray) passes 180° at 2.5 s. The filter that subtracts headings (terracotta) is pulled the long way round toward the same heading written as a negative number. The filter that takes the innovation on the circle (indigo) does not notice the wrap and stays within 2.0° of the truth.](figures/ch14-compass.svg){#fig14-compass}

You know the patch from [](#fig-heading): wrap the innovation into $(-180^\circ, 180^\circ]$. With it the error never exceeds 2.0°. As in [](#ch-why), the patch is geometry in disguise. Write the heading as a rotation of the plane. The wrapped difference of $\hat\theta$ and $y$ is the logarithm of the rotation $R_z(\hat\theta)\,R_z(y)\T$, and the corrected estimate is $\exp$ of a small angle times the old one. Mean and correction live on the circle. The innovation, the variance and the gain live on its tangent line.

On the circle that is the whole story, because rotations of the plane commute. A rigid body in space has three angles that do not. There are then two different errors, $\hat X X^{-1}$ and $X^{-1}\hat X$, and it is not obvious what a covariance of either would mean.

## Uncertainty lives in the tangent space {#sec14-gaussian}

Let $X$ be the true state, an element of a matrix Lie group, and $\hat X$ the estimate. A hat marks an estimate throughout. [](#def3-invariant-errors) offers two errors, and we take the estimate in the first slot, as [](#ex3-dead-reckoning) did:

$$
E_r = \hat X\,X^{-1} = \exp(\chi_r^\wedge), \qquad E_\ell = X^{-1}\hat X = \exp(\chi^\wedge). \label{eq14-errors}
$$

The vector $\chi_r$ is the logarithm of the right-invariant error, and $\chi$ that of the left-invariant error, as in [](#eq3-lie-algebra-error). Each is an ordinary vector with as many components as the group has dimensions. A vector can have a covariance.

::: {.definition #def14-gaussian name="Concentrated Gaussian"}
The estimate $\hat X$ with covariance $\Sigma$ stands for the random group element
$$
X = \exp(-\chi_r^\wedge)\,\hat X, \qquad \chi_r \sim \mathcal{N}(0, \Sigma).
$$
This is a **concentrated Gaussian** on the group, in its right-invariant form. The left-invariant form is $X = \hat X\exp(-\chi^\wedge)$ with $\chi \sim \mathcal{N}(0, \Sigma)$.
:::

The mean is a point of the group. The covariance is a symmetric matrix on the Lie algebra, $3\times3$ for an attitude and $9\times9$ for the navigation state below. The exponential wraps the Gaussian onto the group. Nothing is subtracted between group elements and nothing has to be normalized afterward.

What does such a distribution look like? Not like an ellipse.

::: {.thought #thought14-banana name="A walk with closed eyes"}
Stand in a field, pick a tree ten meters ahead, close your eyes and walk ten meters. Your stride is reliable, so you cover ten meters. Your heading was off by an unknown amount from the first step. Where are you?

Somewhere on an arc of radius ten meters about your starting point. If a thousand people try, they stand along a curved band. Fit a Gaussian to their positions and its mean lies behind the band, where nobody is standing.
:::

[](#fig14-banana) runs the experiment for a planar robot whose pose lies in $\SE(2)$. Its initial pose is a concentrated Gaussian with 0.25 m of position and 23° of heading uncertainty, and it then drives 10 m by exact dead reckoning. The cloud of positions bends into the shape that roboticists call the banana. Its forward coordinate has a skewness of −2.3. In exponential coordinates the same samples have a skewness below 0.03 in every component, and their covariance agrees with the one a linear equation predicts to within 2%, which is sampling error for 4000 samples. Long et al. (2012) put it in a title: the banana distribution is Gaussian. The next section shows why it is exactly so here, and where the exactness ends.

![The banana is Gaussian. Left: 4000 dead-reckoned poses of a planar robot, with the ellipse a Gaussian fitted to the positions would draw (terracotta) and the image of the covariance ellipse from exponential coordinates (indigo). Right: the same samples in exponential coordinates, where the cloud stays an ellipse and only shears.](figures/ch14-banana-strip.png){#fig14-banana anim="figures/ch14-banana.gif"}

The two errors of [](#eq14-errors) describe the same cloud in two frames. [](#thought3-remount) said what each ignores: $E_r$ does not notice a remounted sensor, and $E_\ell$ does not notice a moved world frame. Their logarithms are related by the adjoint of [](#eq2-ad-se3),

$$
\chi_r = \Ad_X\,\chi, \qquad\text{so a covariance } \Sigma \text{ of } \chi \text{ is the covariance } \Ad_{\hat X}\,\Sigma\,\Ad_{\hat X}\T \text{ of } \chi_r \text{, to first order.} \label{eq14-left-right}
$$

::: {.thought #thought14-lever name="One degree, a hundred meters out"}
A robot stands 100 m east of the origin of the map. Its position is known to the centimeter and its heading to within 1°.

In left-invariant coordinates this is a covariance with one large entry: a rotation about the robot's own axis. In right-invariant coordinates $\exp(\chi_r^\wedge)$ multiplies on the left, so a pure rotation $\chi_r$ turns the robot about the origin of the map and carries it 1.75 m sideways. To say "heading uncertain, position certain" in these coordinates you need a rotation and a compensating translation, perfectly correlated. That correlation is the block $x^\wedge R$ of the adjoint. Forget it when you initialize a right-invariant filter and you have told the filter something false about the position.
:::

So a covariance always belongs to one of the two errors, and converting between them is the job of [](#eq14-left-right). Which error should a filter use? The one whose dynamics are simplest. We need to look at how each moves.

## An error that does not care where you are {#sec14-autonomous}

An inertial measurement unit (IMU) is a gyroscope and an accelerometer in one package. The gyroscope reads the body angular velocity $\Omega$. The accelerometer reads the **specific force** $a_s$: the acceleration minus gravity, in the body frame. A vehicle at rest reads $a_s = g\,e_3$ rotated into its body frame. With attitude $R$, velocity $v$ and position $x$,

$$
\dot R = R\,\Omega^\wedge, \qquad \dot v = R\,a_s - g\,e_3, \qquad \dot x = v. \label{eq14-imu}
$$

Dead reckoning integrates [](#eq14-imu) from an estimate $(\hat R, \hat v, \hat x)$ with the same readings.

::: {.thought #thought14-level name="The spirit level that lies"}
Suppose the only thing wrong with the estimate is a tilt of 10°. The filter rotates every accelerometer reading into the world with $\hat R$ and then subtracts gravity along $e_3$. But its $e_3$ is not the vehicle's. The gravity it removes is 10° off the gravity that was measured, and the leftover, $2g\sin 5^\circ = 1.71$ m/s², accumulates as velocity.

Does the leftover depend on what the vehicle is doing? It seems it must: a hard maneuver produces large readings, and rotating a large vector by the wrong matrix gives a large mistake. Hold that thought.
:::

Stack the state in a $5\times5$ matrix,

$$
X = \begin{bmatrix} R & v & x \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \in \SE_2(3), \qquad
\chi^\wedge = \begin{bmatrix} \chi_R^\wedge & \chi_v & \chi_x \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \quad \chi = (\chi_v, \chi_x, \chi_R). \label{eq14-X}
$$

This is the group $\SE_2(3)$ of [](#sec2-se3). The vector of an algebra element lists the linear parts first, in the order of the columns, and the rotation last, like a twist. The right-invariant error $\hat X X^{-1}$ has the three parts

$$
E_R = \hat R R\T, \qquad \eta_v = \hat v - E_R\,v, \qquad \eta_x = \hat x - E_R\,x .
$$

Differentiate them along [](#eq14-imu). The attitude part is constant, as in [](#ex3-dead-reckoning). For the velocity part, $\hat R\,a_s - E_R R\,a_s = 0$. The readings cancel, and what remains is

$$
\dot E_R = 0, \qquad \dot\eta_v = -g\,(I - E_R)\,e_3, \qquad \dot\eta_x = \eta_v . \label{eq14-autonomous}
$$

Neither the state nor the estimate appears. Not even the readings appear. The error falls freely under the difference between gravity and misrotated gravity, whatever the vehicle does. The intuition of [](#thought14-level) was wrong, and it was wrong because it measured the velocity error as $\hat v - v$. That difference does depend on the maneuver. The quantity $\hat v - E_R\,v$ compares the two velocities after turning one into the frame of the other, and the maneuver drops out.

::: {.example #ex14-two-flights name="One mistake on two flights"}
Take an initial error of 114.6° (2 rad) in attitude, with 1 m/s and 1.2 m in the other parts. Apply it to two flights: the banked loop of [](#fig14-opening) and a hover at another place and heading. Dead-reckon both for 3 s.

The right-invariant errors of the two flights agree to $3\times10^{-13}$ at every step ([](#fig14-autonomous)). The additive velocity errors $\hat v - v$ differ by 2.7 m/s after 2 s. A conventional filter predicts its error with a Jacobian evaluated at the estimate. On the loop, after 2 s, that prediction of the velocity error is off by 97% of its size.
:::

![The same mistake on two flights. Left: the velocity part of the right-invariant error on the loop (solid) and in hover (dashed) is one line, the straight line that [](#eq14-autonomous) predicts. Right: the additive velocity error $\hat v - v$ differs between the flights, and the linearized prediction of a conventional filter (thin lines) follows neither. One world-frame component is shown.](figures/ch14-autonomous.svg){#fig14-autonomous}

Something stronger is true. Equation [](#eq14-autonomous) is nonlinear in $E_R$. Its logarithm obeys a linear equation, and exactly:

$$
\dot\chi_r = A\,\chi_r, \qquad A = \begin{bmatrix} 0 & 0 & -g\,e_3^\wedge \\ I & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \qquad
\exp(A h) = \begin{bmatrix} I & 0 & -g h\,e_3^\wedge \\ h I & I & -\tfrac12 g h^2 e_3^\wedge \\ 0 & 0 & I \end{bmatrix}. \label{eq14-A-right}
$$

$A$ is constant and $A^3 = 0$. In the example, $\exp\big((\exp(At)\,\chi_r(0))^\wedge\big)$ reproduces the error matrix to $5\times10^{-13}$, at 114.6° of error. This is no approximation for small errors.

The property behind this was identified by Barrau and Bonnabel (2017). We state it as Hartley et al. (2018) do in their Theorems 1 and 2.

::: {.definition #def14-affine name="Group-affine dynamics"}
Dynamics $\dot X = f_u(X)$ on a matrix Lie group, with input $u$, are **group-affine** if for all group elements $X_1$, $X_2$
$$
f_u(X_1 X_2) = f_u(X_1)\,X_2 + X_1\,f_u(X_2) - X_1\,f_u(I)\,X_2 . \label{eq14-affine}
$$
:::

::: {.theorem #thm14-loglinear name="Autonomous error and log-linear property"}
Let the true state and the estimate both follow group-affine dynamics with the same input. Then each invariant error obeys an equation that involves neither trajectory:
$$
\dot E_r = f_u(E_r) - E_r\,f_u(I), \qquad \dot E_\ell = f_u(E_\ell) - f_u(I)\,E_\ell . \label{eq14-error-dynamics}
$$
Let $A(t)$ be the matrix with $f_u(\exp(\chi^\wedge)) - \exp(\chi^\wedge)\,f_u(I) = (A\,\chi)^\wedge + O(\|\chi\|^2)$, and let $\chi_r(t)$ solve $\dot\chi_r = A(t)\,\chi_r$. If $E_r(0) = \exp(\chi_r(0)^\wedge)$, then
$$
E_r(t) = \exp\big(\chi_r(t)^\wedge\big) \quad\text{for all } t \ge 0, \label{eq14-loglinear}
$$
for an initial error of any size. The same holds for $E_\ell$ with its own matrix.
:::

The dynamics [](#eq14-imu) satisfy [](#eq14-affine), which you can check by multiplying out, and [](#eq14-A-right) is the matrix of the theorem. For the left-invariant error the matrix is

$$
\dot\chi = \begin{bmatrix} -\Omega^\wedge & 0 & -a_s^\wedge \\ I & -\Omega^\wedge & 0 \\ 0 & 0 & -\Omega^\wedge \end{bmatrix}\chi, \label{eq14-A-left}
$$

which contains the readings and still not the state. Both were checked against integration of [](#eq14-imu) with random inputs and errors near 100°.

Compare the conventional choice. An error-state filter of the usual kind, built on a quaternion or a rotation matrix, uses $\hat R = R\exp(\delta\theta^\wedge)$ together with the differences $\hat v - v$ and $\hat x - x$. Its velocity error obeys $\ddt(\hat v - v) = -\hat R\,a_s^\wedge\,\delta\theta$ to first order. The estimate $\hat R$ sits in the Jacobian. [](#eq3-lie-algebra-error) was the control version of the same contrast: one linear model along a whole reference. Here it is one linear model for every estimate, including the wrong ones.

The banana of [](#fig14-banana) is now explained. Driving at a commanded twist is group-affine, the initial $\chi$ was Gaussian, and a linear map of a Gaussian is Gaussian. The cloud in exponential coordinates is an ellipse at all times, exactly.

The theorem is about a model without noise. A filter has noise in its readings and corrects with measurements. Does the estimate stay out of those steps too?

## The invariant extended Kalman filter {#sec14-inekf}

**Noise.** The gyroscope reads $\Omega + n_g$ and the accelerometer $a_s + n_a$, with covariances $\Sigma_g$ and $\Sigma_a$ per sample. Noise is added in the body frame, on the right of $X$, and the right-invariant error lives on the left. Moving it across costs an adjoint. Over a step $h$, to first order,

$$
\chi_{r,k+1} = \exp(A h)\,\big(\chi_{r,k} + \Ad_{\hat X_k} n_k\big), \qquad n_k = h\,(n_a, 0, n_g), \label{eq14-noise}
$$

with $\Ad_X$ the $9\times9$ matrix whose diagonal blocks are $R$ and whose last column of blocks is $(v^\wedge R,\ x^\wedge R,\ R)$. So the estimate does enter, through the direction in which the noise is injected. It does not enter the deterministic part.

**Measurement.** A measurement suits the right-invariant error when it has the form

$$
\begin{bmatrix} y_k \\ 0 \\ 1 \end{bmatrix} = X_k^{-1}\,d + \begin{bmatrix} n_y \\ 0 \\ 0 \end{bmatrix} \label{eq14-obs}
$$

for a known vector $d$. A landmark at a known position $p$, seen from the vehicle, is the case $d = (p, 0, 1)$: then $y = R\T(p - x) + n_y$. Multiply the reading by the estimate. Since $\hat X X^{-1} d = E_r\,d$, the innovation depends on the error alone:

$$
\tilde y_k = \hat R_k\,y_k + \hat x_k - p = C_y\,\chi_r + \hat R_k\,n_y + O(\|\chi_r\|^2), \qquad C_y = \begin{bmatrix} 0 & I & -p^\wedge \end{bmatrix}. \label{eq14-innovation}
$$

$C_y$ is constant. A measurement of the form $X d + n_y$, such as a position fix, pairs with the left-invariant error in the same way ([](#exr14-left)).

::: {.definition #def14-inekf name="Right-invariant extended Kalman filter"}
The state of the filter is an estimate $\hat X$ and the covariance $\Sigma$ of $\chi_r$. With $\Sigma_n$ the covariance of $n_k$ and $\Sigma_y$ that of $n_y$:

1. **Propagate.** Integrate [](#eq14-imu) with the readings to get $\hat X_{k+1}$, and
$$
\Sigma_{k+1} = \exp(A h)\,\big(\Sigma_k + \Ad_{\hat X_k}\,\Sigma_n\,\Ad_{\hat X_k}\T\big)\exp(A h)\T . \label{eq14-propagate}
$$
2. **Gain.** With the innovation $\tilde y$ of [](#eq14-innovation) and its covariance $\Sigma_{\tilde y} = C_y\,\Sigma\,C_y\T + \hat R\,\Sigma_y\,\hat R\T$,
$$
\tilde K = \Sigma\,C_y\T\,\Sigma_{\tilde y}^{-1} . \label{eq14-gain}
$$
3. **Correct**, on the group and on the side of the error:
$$
\hat X^+ = \exp\big(-(\tilde K\tilde y)^\wedge\big)\,\hat X, \qquad \Sigma^+ = (I - \tilde K C_y)\,\Sigma\,(I - \tilde K C_y)\T + \tilde K\,\hat R\,\Sigma_y\,\hat R\T\,\tilde K\T . \label{eq14-update}
$$
:::

Several landmarks stack their innovations and their $C_y$. The minus sign in [](#eq14-update) follows from [](#eq14-errors): the error is the estimate relative to the truth, so a correction removes it, as a feedback $-K\,s$ removes a tracking error in [](#eq7-riccati).

::: {.remark}
The filtering literature writes $P$, $Q$, $N$ or $R$, $K$ and $H$ for the covariance, the two noise covariances, the gain and the measurement matrix. In this book those letters are the Riccati solution, a weight, the attitude, a feedback gain and a barrier function. We write $\Sigma$, $\Sigma_n$, $\Sigma_y$, $\tilde K$ and $C_y$, with the tilde that marks every prediction error and the gain that acts on one. Hartley et al. define the innovation with the opposite sign and correct with $\exp(+\cdot)$. The filters are the same.
:::

Count where the estimate appears. Not in $A$. Not in $C_y$. Only in the adjoint that turns the sensor noise and in the rotation of the measurement noise. If both noises were zero, the gain sequence could be computed before the flight, as for a linear system. That is as close to a Kalman filter as a nonlinear problem gets.

::: {.warning name="Correct on the side of the error"}
The right-invariant error is removed by multiplying on the left. Writing $\hat X\exp(\cdot)$ in [](#eq14-update), with everything else unchanged, is a one-character mistake. It is not a small one. In the simulation below that filter diverged from a 2° initial error, because a correction meant for the world frame was applied in the body frame of a vehicle meters from the origin. The same care applies at the start: a covariance stated as "attitude, velocity and position, independently" must be converted with [](#eq14-left-right) before a right-invariant filter may use it.
:::

The argument so far promises a better transient from a bad estimate. It promises nothing when the estimate is good. How much does it matter in numbers?

## Navigation with landmarks {#sec14-worked}

The vehicle of [](#fig14-opening) flies an elliptical loop with semi-axes of 2.5 m and 2 m every 10 s, banking like a multicopter, for 20 s. Its IMU runs at 100 Hz with noise of 0.01 rad/s and 0.1 m/s² per sample. Ten times a second it sees three landmarks in its body frame, each with 0.1 m of noise. The truth is generated by the same discrete model the filters integrate, so the only errors are the initial estimate and the noise.

Two filters run on identical readings. The first is [](#def14-inekf). The second is the conventional error-state filter of [](#sec14-autonomous), with a multiplicative attitude error and additive velocity and position errors, which is how a quaternion-based extended Kalman filter is built. Both start from the same estimate. Both are given the same initial covariance, written in each filter's own coordinates through [](#eq14-left-right). Each row of [](#tbl14-mc) is 200 runs with the initial attitude error of the stated size about a random axis.

::: {.table #tbl14-mc}

| Initial attitude error | Filter | Median error at 2 s | Settled, median | Settled, 90% of runs | NEES, last 5 s |
| :-: | :-- | :-: | :-: | :-: | :-: |
| 5° | invariant | 0.36° | 0.1 s | 0.2 s | 1.01 |
| | conventional | 0.37° | 0.1 s | 0.2 s | 1.01 |
| 60° | invariant | 1.1° | 0.8 s | 1.7 s | 1.01 |
| | conventional | 1.7° | 1.8 s | 3.2 s | 1.01 |
| 120° | invariant | 4.7° | 3.2 s | 3.7 s | 1.02 |
| | conventional | 15.8° | 4.8 s | 8.0 s | 1.24 |
| 170° | invariant | 44.8° | 5.2 s | 7.1 s | 1.34 |
| | conventional | 103.8° | 17.9 s | not by 20 s | 69.7 |

: Monte Carlo comparison, 200 runs per row. "Settled" is the time after which the attitude error stays below 2°. NEES is the normalized estimation error squared, $\chi_r\cdot\Sigma^{-1}\chi_r$ divided by 9 and averaged: 1 for a filter whose covariance tells the truth. The initial velocity and position errors are 0.2 m/s and 0.2 m in the first row and 1 m/s and 1 m in the others.

:::

Read the first row first. From a good start the two filters are indistinguishable: the same error to three digits and the same consistency. Near the truth every reasonable linearization is accurate, and invariance has nothing to add. Hartley et al. (2018) report the same.

The rows below are the reason for the theory. From 60° the invariant filter settles in less than half the time. From 170° it is on the truth after 5.2 s in the median run, while the conventional filter needs 17.9 s and in one run in four has not settled when the flight ends. Its covariance is by then far too small for its error, with an NEES of 70. It is the confidently mistaken filter of the opening ([](#fig14-convergence)).

![Convergence from 170° of initial attitude error over 200 runs: median (lines) and the band up to the 90th percentile. The invariant filter (indigo) and the conventional one (terracotta) receive identical readings.](figures/ch14-convergence.svg){#fig14-convergence}

The covariance itself can be tested without any measurement. Dead-reckon 4000 noisy copies of the first 4 s and compare the sample covariance of $\chi_r$ with [](#eq14-propagate). The nine standard deviations agree within 2%, and the matrices within 5% in norm, which is the sampling error.

Landmarks make this problem easy: three of them fix the pose. A legged robot has nothing of the kind.

## Legs: the contact point joins the group {#sec14-legged}

A walking robot has an IMU, joint encoders and feet. While a foot stands on the ground, the leg is a measuring rod from the body to a fixed point. Nobody knows where that point is. Hartley et al. (2018) make it part of the state. With the contact position $x_c$ as one more column,

$$
X = \begin{bmatrix} R & v & x & x_c \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \in \SE_3(3), \qquad \dot x_c = 0 \ \text{ up to noise}, \label{eq14-contact}
$$

the dynamics are still group-affine, and the matrix of the error equation is [](#eq14-A-right) with a row and a column of zeros for the contact. Each additional foot adds a column. The paper writes $\SE_{N+2}(3)$ for $N$ contacts. The assumption that a foot in contact does not move is softened by white noise on its velocity, which covers some slip.

The leg kinematics give the position of the foot relative to the body, in the body frame: $y = R\T(x_c - x) + n_y$. That is [](#eq14-obs) with $d = (0, 0, 1, -1)$. The innovation and its matrix are

$$
\tilde y = \hat R\,y - (\hat x_c - \hat x), \qquad C_y = \begin{bmatrix} 0 & I & -I & 0 \end{bmatrix} \label{eq14-fk}
$$

in the order $(\chi_v, \chi_x, \chi_c, \chi_R)$. In the paper the noise $n_y$ comes from the encoders through the leg's Jacobian.

![Contact-aided estimation. The body carries an IMU. The leg measures the vector $y$ from the body to the foot, in the body frame. The contact position $x_c$ is unknown, constant while the foot stands, and estimated with the rest of the state.](figures/ch14-contact.svg){#fig14-contact}

With constant $A$ and $C_y$, observability is a rank computation that holds along every trajectory. The observability matrix has rank 8 of 12. The four missing directions are a common shift of $x$ and $x_c$ and a rotation about $e_3$: absolute position and yaw. Hartley et al. find the same, and stress that the invariant filter cannot acquire false information about these directions through its linearization, because the linearization does not move.

A small experiment shows the split. This is our simplified demonstration, not the paper's: a body sways above one fixed contact for 6 s, with the IMU and contact noise levels of the paper's Table I and 1 cm of noise on the leg measurement, and 100 runs start 20° wrong in tilt, 30° in yaw and 1 m/s in velocity. After 1 s the median tilt error is 0.13° and the body-frame velocity error has fallen from 1.4 m/s to 0.01 m/s. The yaw error is still 29° at the end and the position error still 0.11 m. The filter learns what gravity and a planted foot can tell it, and nothing else.

**Biases.** Real gyroscopes and accelerometers have slowly varying offsets $b_g$, $b_a$ that must be estimated. Hartley et al. note, citing Barrau, that no Lie group contains the biases and keeps the dynamics group-affine. They append the bias errors $\hat b_g - b_g$ and $\hat b_a - b_a$ to $\chi_r$ as ordinary vectors and call the result an "imperfect" invariant filter. A bias error enters the error equation like sensor noise in [](#eq14-noise), through $-\Ad_{\hat X}$. So the matrix $A$ acquires columns that contain $\hat R$, $\hat v^\wedge\hat R$ and $\hat x^\wedge\hat R$. The estimate is back in the linearization, but only in the columns that multiply the bias error: when that error is zero, the rest of [](#eq14-A-right) is unchanged.

What the paper reports is this. In simulation of a Cassie biped, 100 runs with initial Euler-angle errors up to 30° and velocity errors up to 1 m/s: both filters converge, the invariant one considerably faster. On the real robot walking at about 0.3 m/s, with bias estimation on and the same spread of initial errors, the invariant filter converged faster and more reliably in all 100 runs. Initialized close to the truth, the two performed alike. This is the pattern of [](#tbl14-mc).

Everything so far used a group. A cable is not a group element.

## When the state is not a group {#sec14-s2}

The direction $q \in S^2$ of a pendulum or a cable has no multiplication. There is no $\hat q\,q^{-1}$, no invariant error and no theorem like [](#thm14-loglinear). The first half of the idea survives: keep the mean on the manifold and the uncertainty in a tangent space. Kotaru (2022, ch. 4) builds an extended Kalman filter this way from the variation-based linearization of [](#ch-linearization).

Take the spherical pendulum of [](#eq7-pendulum) with no applied force. The estimate is a direction $\hat q$ of unit length and an angular velocity perpendicular to it. A hat on $\omega$ would read as the hat map, so the estimated angular velocity is written $\omega_{\hat q}$: it is a tangent vector at $\hat q$, with $\omega_{\hat q}\cdot\hat q = 0$. The error is the linear state $s = (\xi, \delta\omega)$ of [](#sec7-variation), and a finite $s$ is applied by

$$
(q, \omega) \oplus s = \Big(\exp(\xi^\wedge)\,q,\ \ \exp(\xi^\wedge)\big(\omega + (I - q q\T)\,\delta\omega\big)\Big). \label{eq14-oplus}
$$

The direction turns about $\xi$. The angular velocity is corrected in the tangent plane and carried along by the same turn, which for $\xi\cdot q = 0$ is parallel transport along the great circle (the dashed arrow of [](#fig3-transport)). The inverse operation returns the $s$ between two states. These are the mapping operators $\oplus$ and $\ominus$ of Kotaru's Appendix B.1, and [](#eq14-oplus) agrees with his formulas (B.12) and (B.14) to (B.17).

::: {.thought #thought14-flat name="A covariance with no thickness"}
The covariance of $s$ is a $6\times6$ matrix. But $s$ is confined to the constraint subspace $\ker C$ of [](#eq7-s2-constraint), which has dimension four: an error along $\hat q$ would lengthen the rod, and the filter must be certain that the rod has not grown. So $\Sigma$ has rank four, with $C\,\Sigma = 0$. Picture a pancake in six dimensions.

A pancake cannot be inverted. And the plane it lies in turns with the estimate: after every correction $\hat q$ has moved, and yesterday's pancake pokes out of today's tangent plane.
:::

You can store $\Sigma$ either way: as the $6\times6$ matrix of rank four, or as a $4\times4$ matrix in a basis $Z$ of $\ker C$, as [](#def7-constrained) did for controllability. We keep the $6\times6$ form, as Kotaru does, and need two facts about it.

First, the filter never has to invert $\Sigma$. The gain is $\tilde K = \Sigma\,C_y\T\,\Sigma_{\tilde y}^{-1}$, and $\Sigma_{\tilde y}$ is invertible because the measurement noise is. The correction $-\tilde K\tilde y$ is a combination of the columns of $\Sigma$, so it lies in $\ker C$ by itself. Second, the matrix

$$
E_C = \begin{bmatrix} I - q q\T & 0 \\ -q\,(q\times\omega)\T & I - q q\T \end{bmatrix} \label{eq14-E}
$$

is a projector onto $\ker C$ at $(q, \omega)$. It puts a covariance into the current constraint subspace.

::: {.definition #def14-vekf name="Variation-based extended Kalman filter on the sphere"}
The state of the filter is $(\hat q, \omega_{\hat q})$ and a covariance $\Sigma$ of $s$ with $C\,\Sigma = 0$.

1. **Propagate.** Integrate the pendulum to the next sample. With $A$ of [](#eq7-s2-lin) at the estimate, $A_k = \exp(A h)$ and $E_C$ at the new estimate, $\Sigma \leftarrow E_C\,(A_k\,\Sigma\,A_k\T + \Sigma_n)\,E_C\T$.
2. **Correct.** For a position reading $y = l\,q + n_y$: $\tilde y = l\,\hat q - y$, $C_y = [\,-l\,\hat q^\wedge\ \ 0\,]$, the gain of [](#eq14-gain), $s^+ = -\tilde K\tilde y$ and $(\hat q, \omega_{\hat q}) \leftarrow (\hat q, \omega_{\hat q})\oplus s^+$.
3. **Move the covariance.** After the usual covariance update, turn both blocks by the rotation $\exp(\xi^\wedge)$ of the correction, where $\xi$ is the first half of $s^+$, and apply $E_C$ at the new estimate.
:::

Step 3 is the transport of [](#def3-transport) applied to an uncertainty: the ellipse is carried along the arc to its new tangent plane ([](#fig14-sphere)). It is exact to first order in the size of the correction.

Kotaru's algorithm is organized differently, and the difference is worth knowing. He measures the variation from the previous estimate, updates that variation, and then applies a separate constraint update that projects it onto $\ker C$ with a weight $\Sigma^{-1}$. He observes that two eigenvalues of the covariance are zero and that the estimates converge without the constraint update. Both observations follow from the two facts above. His appendix selects the sign of the rotation axis by a comparison that, as printed in (B.6), has the same value on both sides. With the axis $q_{k-1}\times q_k$ and the angle $\arccos(q_{k-1}\cdot q_k)$ the positive sign is always the right one.

![The estimate lives on the sphere, its uncertainty in the tangent plane. The true direction (dashed gray) and the estimate (indigo) of a swinging pendulum, with the one-standard-deviation ellipse of the direction drawn in the tangent plane at the first estimate and three readings later. The plane and the ellipse travel with the estimate. For this picture the readings are noisier (0.2 m) than in [](#ex14-pendulum), so that the ellipses can be seen. The estimate of the filter on six numbers (terracotta) follows closely, but its length reaches 1.08.](figures/ch14-sphere.svg){#fig14-sphere}

::: {.example #ex14-pendulum name="The spherical pendulum, estimated two ways"}
We use Kotaru's setup: $l = 1$ m, an initial covariance equal to the identity, measurement noise of variance $10^{-3}$ on each coordinate of $l\,q$ and process noise of $10^{-5}$. The pendulum is released from horizontal with a sideways velocity and observed at 100 Hz for 5 s. In 100 runs the initial estimate is drawn from the initial covariance: a median of 71° wrong in direction and at most 156°.

The variation-based filter converges in every run. After the first second its direction error has a root mean square of 0.82°. $\|\hat q\| - 1$ and $\hat q\cdot\omega_{\hat q}$ never exceed $10^{-15}$, $C\,\Sigma$ stays below $10^{-12}$ with no constraint update, and the NEES in the four constrained dimensions is 1.01.

The naive filter treats $(q, \omega)$ as six unconstrained numbers and adds its corrections. Its direction error, 0.87° after normalizing, is about as good, because the measurement sees $q$ directly. But $\|\hat q\|$ ranges from 0.91 to 1.08 over the runs, and from 0.97 to 1.03 after the first second. Its $\hat q\cdot\omega_{\hat q}$ reaches 2.9 rad/s. Its estimate is not the state of any pendulum, and whatever uses it has to repair it first.
:::

Kotaru reports the corresponding results: convergence in 100 runs from random initial states, some with nearly the largest possible attitude error, a unit norm that a standard filter does not keep, and singularities in a filter written in spherical coordinates. He then validates the filter in experiments on a pendulum and on a quadrotor with a cable-suspended load.

Notice what was lost on the way from the group to the sphere. $A$ and $C_y$ in [](#def14-vekf) depend on the estimate. This is a careful extended Kalman filter, not an invariant one.

## What invariance does not give you {#sec14-limits}

**It needs the right dynamics.** The theorem covers group-affine systems. Kinematics driven by measured velocities and accelerations qualify. Add a drag force along a body axis and [](#eq14-affine) fails ([](#exr14-affine)). Add a bias and the estimate returns to the matrix. [](#sec14-legged) showed how much survives.

**It needs the right measurements.** [](#eq14-obs) is a vector known in one frame and measured in the other. A range, a bearing or a camera pixel is a nonlinear function of that vector, and $C_y$ then depends on the estimate again.

**Noise is not invariant.** Body-frame noise enters a right-invariant filter through $\Ad_{\hat X}$. The covariance is exact only as far as the estimate is.

**A Gaussian in the algebra is still an approximation.** With process noise the cloud is only nearly Gaussian in exponential coordinates. Repeat [](#fig14-banana) with noisy odometry and no initial uncertainty: the lateral and heading parts match the linear prediction, but the forward variance comes out 2.8 times the predicted one, a second-order effect that the linear equation cannot see. And a Gaussian has one peak. At half a turn the logarithm is not unique. That is the cut locus of [](#sec7-local), and a single mean with a covariance cannot say "this attitude or the opposite one". The 170° row of [](#tbl14-mc) is slow for this reason, in both filters.

**It is local.** Barrau and Bonnabel (2017) prove that the invariant filter, used as an observer, is locally stable around any trajectory under observability conditions. That is a local statement with an unusually wide reach, not a global one, and [](#thm-obstruction) suggests why nobody should expect more from a single smooth estimate on a rotation group.

**Unobservable stays unobservable.** No filter recovers yaw from gravity and a foothold.

Other filters respect the same geometry: unscented filters on manifolds (Brossard, Barrau and Bonnabel 2020) and smoothers that optimize over a whole trajectory with the steps of [](#def13-retraction). They are not covered here. What they share with this chapter is the rule from its second section: the estimate is a point of the manifold, and the uncertainty is a vector in a tangent space.

## Exercises {#sec14-exercises .unnumbered}

::: {.exercise #exr14-circle}
On $\SO(2)$ show that $E_r = E_\ell$ and that the wrapped difference of two headings is $\log(R_z(\hat\theta)R_z(y)\T)^\vee$. Run the compass filter of the notebook with a compass noise of 60°. Is the scalar Kalman update still a sensible average of two headings? What does the definition of a concentrated Gaussian assume that fails here?
:::

::: {.exercise #exr14-affine}
Multiply out [](#eq14-affine) for the IMU dynamics [](#eq14-imu). Then add to $\dot v$ (a) a linear drag $-c_1 v$ and (b) a drag along the first body axis, $-c_1\,(R e_1)(R e_1\cdot v)$, with a positive constant $c_1$. One of them keeps the dynamics group-affine. Which, and what is the new $A$?
:::

::: {.exercise #exr14-left}
Derive [](#eq14-A-left) from [](#eq14-error-dynamics). A position fix reads $y = x + n_y$. Write it as $X d + n_y$, form an innovation that depends only on $E_\ell$, and find its constant $C_y$. On which side does the correction multiply?
:::

::: {.exercise #exr14-leak}
Use [](#eq14-A-right) to find how long a tilt error of 1° takes to produce a position error of 1 m in a dead-reckoned estimate. (You should get 3.4 s.) Why does a yaw error of 1° produce none? Relate this to the unobservable directions of [](#sec14-legged).
:::

::: {.exercise #exr14-banana}
In the notebook, give the planar robot noisy odometry and no initial uncertainty. Compare the sample covariance of $\chi$ with the linear prediction, component by component. Then halve the noise and repeat. How does the mismatch in the forward variance scale, and what does that say about its order?
:::

::: {.exercise #exr14-two-feet}
Extend [](#eq14-contact) to two feet in contact. Write $C_y$ for the two leg measurements and compute the rank of the observability matrix. Do two feet make yaw observable? What would a single landmark at a known position add?
:::

::: {.exercise #exr14-pancake}
Show that $E_C$ in [](#eq14-E) satisfies $C E_C = 0$ and $E_C^2 = E_C$, and that $E_C\,s = s$ for every $s$ in $\ker C$. Then show that if $C\,\Sigma = 0$ before the correction of [](#def14-vekf), then $C\,s^+ = 0$ and $C\,\Sigma^+ = 0$ after it, before the estimate is moved. Why is step 3 still needed?
:::

## Notes and sources {#sec14-notes .unnumbered}

The group-affine property, the autonomous error and the log-linear property are due to Barrau and Bonnabel (2017), who also introduced $\SE_2(3)$ for inertial navigation and proved the stability of the filter as an observer. Their 2018 review is the most accessible introduction. [](#thm14-loglinear) is stated as in Hartley et al. (2018, Theorems 1 and 2), whose contact-aided filter, forward-kinematic measurement, observability analysis, bias augmentation and experiments on Cassie are the subject of [](#sec14-legged). Hartley et al. (2020) is the extended journal version. We order the algebra as $(\chi_v, \chi_x, \chi_R)$ and define the innovation as predicted minus measured. The paper uses $(\xi_R, \xi_v, \xi_p)$ and the opposite sign.

Wang and Chirikjian (2006) studied how errors propagate on the Euclidean group. Long et al. (2012) showed that the banana-shaped cloud of dead-reckoned poses is modeled well by a Gaussian in exponential coordinates. Barfoot (2017) treats uncertainty on $\SO(3)$ and $\SE(3)$ at textbook length, and Solà, Deray and Atchuthan (2018) give the Jacobians a practitioner needs.

The variation-based extended Kalman filter on $S^2$, its mapping operators and its experiments are from Kotaru and Sreenath (2019) and Kotaru (2022, ch. 4 and Appendix B.1), built on the linearization of Wu and Sreenath (2015). [](#def14-vekf) re-centers the variation at the current estimate at every step and replaces the constraint update by the projector [](#eq14-E). The simulations of this chapter, including the comparison in [](#tbl14-mc) and the contact demonstration, are ours and are smaller than the experiments of the papers.

- A. Barrau and S. Bonnabel, ["The invariant extended Kalman filter as a stable observer"](https://doi.org/10.1109/TAC.2016.2594085), *IEEE Transactions on Automatic Control* 62(4), 2017.
- A. Barrau and S. Bonnabel, ["Invariant Kalman filtering"](https://doi.org/10.1146/annurev-control-060117-105010), *Annual Review of Control, Robotics, and Autonomous Systems* 1, 2018.
- T. D. Barfoot, [*State Estimation for Robotics*](https://www.cambridge.org/core/product/identifier/9781316671528/type/book), Cambridge University Press, 2017.
- M. Brossard, A. Barrau and S. Bonnabel, ["A code for unscented Kalman filtering on manifolds (UKF-M)"](https://arxiv.org/abs/2002.00878), *IEEE International Conference on Robotics and Automation*, 2020. arXiv:2002.00878.
- R. Hartley, M. Ghaffari Jadidi, J. W. Grizzle and R. M. Eustice, ["Contact-aided invariant extended Kalman filtering for legged robot state estimation"](https://doi.org/10.15607/RSS.2018.XIV.050), *Robotics: Science and Systems*, 2018. arXiv:1805.10410.
- R. Hartley, M. Ghaffari, R. M. Eustice and J. W. Grizzle, ["Contact-aided invariant extended Kalman filtering for robot state estimation"](https://arxiv.org/abs/1904.09251), *International Journal of Robotics Research*, 2020. arXiv:1904.09251.
- P. Kotaru, [*Dynamics & Control for Collaborative Aerial Manipulation*](https://escholarship.org/uc/item/1qf6q0n5), PhD thesis, University of California, Berkeley, 2022.
- P. Kotaru and K. Sreenath, ["Variation based extended Kalman filter on S²"](https://hybrid-robotics.berkeley.edu/publications/ECC2019_VariationEKF_S2.pdf), *European Control Conference*, 2019.
- A. W. Long, K. C. Wolfe, M. J. Mashner and G. S. Chirikjian, ["The banana distribution is Gaussian: a localization study with exponential coordinates"](https://doi.org/10.15607/RSS.2012.VIII.034), *Robotics: Science and Systems*, 2012.
- J. Solà, J. Deray and D. Atchuthan, ["A micro Lie theory for state estimation in robotics"](https://arxiv.org/abs/1812.01537), arXiv:1812.01537, 2018.
- Y. Wang and G. S. Chirikjian, ["Error propagation on the Euclidean group with applications to manipulator kinematics"](https://doi.org/10.1109/TRO.2006.878978), *IEEE Transactions on Robotics* 22(4), 2006.
- G. Wu and K. Sreenath, ["Variation-based linearization of nonlinear systems evolving on SO(3) and S²"](https://doi.org/10.1109/ACCESS.2015.2477880), *IEEE Access* 3, 2015.
