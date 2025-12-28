"""
Title: Coupled Pendulum System — RK4 Integration
Physics: Classical Mechanics / Nonlinear Dynamics
Method: Fourth-order Runge–Kutta (RK4)

System:
Two coupled nonlinear pendulums interacting through a linear coupling
term. Each pendulum experiences a nonlinear restoring torque due to
gravity, and the coupling allows energy exchange between them.

Equations of Motion:
θ̈₁ = −ω₀² sin(θ₁) − k (θ₁ − θ₂)
θ̈₂ = −ω₀² sin(θ₂) − k (θ₂ − θ₁)

Description:
The equations of motion are integrated numerically using the RK4 method.
By varying the coupling strength and initial conditions, the system
exhibits energy transfer, normal-mode behavior, and nonlinear dynamics.
For stronger coupling or larger amplitudes, the motion can become
highly sensitive to initial conditions.

Units:
Time and angles are expressed in dimensionless units.
The parameter ω₀² is absorbed into the numerical coefficient.

Author: Aprameyan Veerereghavan
"""

import math as m
import matplotlib.pyplot as plt

# --------------------------------------------------
# Physical parameter
# --------------------------------------------------

coupling_k = 0   # coupling strength between the two pendulums

# --------------------------------------------------
# Equations of motion
# x = angle, v = angular velocity
# --------------------------------------------------

def dv1(theta1, omega1, theta2, omega2, t):
    # Angular acceleration of pendulum 1
    return -5 * m.sin(theta1) - coupling_k * (theta1 - theta2)

def dx1(theta1, omega1, theta2, omega2, t):
    # Angular velocity of pendulum 1
    return omega1

def dv2(theta1, omega1, theta2, omega2, t):
    # Angular acceleration of pendulum 2
    return -5 * m.sin(theta2) - coupling_k * (theta2 - theta1)

def dx2(theta1, omega1, theta2, omega2, t):
    # Angular velocity of pendulum 2
    return omega2

# --------------------------------------------------
# Time and storage arrays
# --------------------------------------------------

n_steps = 1500
dt = 0.01

theta1 = [0.0] * (n_steps + 1)
omega1 = [0.0] * (n_steps + 1)

theta2 = [0.0] * (n_steps + 1)
omega2 = [0.0] * (n_steps + 1)

time = [0.0] * (n_steps + 1)

# --------------------------------------------------
# Initial conditions
# --------------------------------------------------

theta1[0] = 1
omega1[0] = 1

theta2[0] = -1
omega2[0] = -1

# --------------------------------------------------
# RK4 integration loop
# --------------------------------------------------

for i in range(n_steps):

    # k1 terms
    X11 = dx1(theta1[i], omega1[i], theta2[i], omega2[i], time[i])
    V11 = dv1(theta1[i], omega1[i], theta2[i], omega2[i], time[i])

    X21 = dx2(theta1[i], omega1[i], theta2[i], omega2[i], time[i])
    V21 = dv2(theta1[i], omega1[i], theta2[i], omega2[i], time[i])

    # k2 terms
    X12 = dx1(theta1[i] + X11*dt/2, omega1[i] + V11*dt/2,
              theta2[i] + X21*dt/2, omega2[i] + V21*dt/2, time[i] + dt/2)

    V12 = dv1(theta1[i] + X11*dt/2, omega1[i] + V11*dt/2,
              theta2[i] + X21*dt/2, omega2[i] + V21*dt/2, time[i] + dt/2)

    X22 = dx2(theta1[i] + X11*dt/2, omega1[i] + V11*dt/2,
              theta2[i] + X21*dt/2, omega2[i] + V21*dt/2, time[i] + dt/2)

    V22 = dv2(theta1[i] + X11*dt/2, omega1[i] + V11*dt/2,
              theta2[i] + X21*dt/2, omega2[i] + V21*dt/2, time[i] + dt/2)

    # k3 terms
    X13 = dx1(theta1[i] + X12*dt/2, omega1[i] + V12*dt/2,
              theta2[i] + X22*dt/2, omega2[i] + V22*dt/2, time[i] + dt/2)

    V13 = dv1(theta1[i] + X12*dt/2, omega1[i] + V12*dt/2,
              theta2[i] + X22*dt/2, omega2[i] + V22*dt/2, time[i] + dt/2)

    X23 = dx2(theta1[i] + X12*dt/2, omega1[i] + V12*dt/2,
              theta2[i] + X22*dt/2, omega2[i] + V22*dt/2, time[i] + dt/2)

    V23 = dv2(theta1[i] + X12*dt/2, omega1[i] + V12*dt/2,
              theta2[i] + X22*dt/2, omega2[i] + V22*dt/2, time[i] + dt/2)

    # k4 terms
    X14 = dx1(theta1[i] + X13*dt, omega1[i] + V13*dt,
              theta2[i] + X23*dt, omega2[i] + V23*dt, time[i] + dt)

    V14 = dv1(theta1[i] + X13*dt, omega1[i] + V13*dt,
              theta2[i] + X23*dt, omega2[i] + V23*dt, time[i] + dt)

    X24 = dx2(theta1[i] + X13*dt, omega1[i] + V13*dt,
              theta2[i] + X23*dt, omega2[i] + V23*dt, time[i] + dt)

    V24 = dv2(theta1[i] + X13*dt, omega1[i] + V13*dt,
              theta2[i] + X23*dt, omega2[i] + V23*dt, time[i] + dt)

    # RK4 weighted averages
    theta1_avg = (X11 + 2*X12 + 2*X13 + X14) / 6.0
    omega1_avg = (V11 + 2*V12 + 2*V13 + V14) / 6.0

    theta2_avg = (X21 + 2*X22 + 2*X23 + X24) / 6.0
    omega2_avg = (V21 + 2*V22 + 2*V23 + V24) / 6.0

    # Update state
    omega1[i+1] = omega1[i] + omega1_avg * dt
    theta1[i+1] = theta1[i] + theta1_avg * dt

    omega2[i+1] = omega2[i] + omega2_avg * dt
    theta2[i+1] = theta2[i] + theta2_avg * dt

    time[i+1] = time[i] + dt

# --------------------------------------------------
# Plot angles vs time
# --------------------------------------------------

plt.plot(time, theta1, label="Pendulum 1")
plt.plot(time, theta2, label="Pendulum 2")
plt.xlabel("Time")
plt.ylabel("Angle")
plt.legend()
plt.show()
