"""
Title: Transient Time vs Damping in a Driven Oscillator
Physics: Classical Mechanics / Nonlinear Dynamics
Method: Fourth-order Runge–Kutta (RK4)

System:
A driven, damped oscillator governed by
x'' + d x' + ω₀² x = sin(Ω t).

Description:
The equation of motion is integrated numerically for different damping
constants. The transient time is estimated by detecting the onset of
steady-state periodic motion. The dependence of transient duration on
damping strength is then analyzed.

This program highlights relaxation dynamics and the role of damping in
driven systems.

Units:
All quantities are expressed in dimensionless units.

Author: Aprameyan Veerereghavan
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Equations of motion
# x'' + d x' + 2x = sin(2t)
# --------------------------------------------------

def acceleration(x, v, t, damping):
    # dv/dt
    return m.sin(2 * t) - damping * v - 2 * x

def velocity(x, v, t):
    # dx/dt
    return v


# --------------------------------------------------
# Time grid and storage
# --------------------------------------------------

n_steps = 15000
dt = 0.01

x = [0.0] * (n_steps + 1)   # position
v = [0.0] * (n_steps + 1)   # velocity
t = [0.0] * (n_steps + 1)   # time

# Initial condition
v[0] = 4

# Range of damping constants
damping_values = np.arange(0.1, 0.5, 0.1)

transient_times = []

# --------------------------------------------------
# Loop over damping values
# --------------------------------------------------

for damping in damping_values:

    # Time integration using RK4
    for i in range(n_steps):

        k1x = velocity(x[i], v[i], t[i])
        k1v = acceleration(x[i], v[i], t[i], damping)

        k2x = velocity(x[i] + k1x * dt / 2, v[i] + k1v * dt / 2, t[i] + dt / 2)
        k2v = acceleration(x[i] + k1x * dt / 2, v[i] + k1v * dt / 2, t[i] + dt / 2, damping)

        k3x = velocity(x[i] + k2x * dt / 2, v[i] + k2v * dt / 2, t[i] + dt / 2)
        k3v = acceleration(x[i] + k2x * dt / 2, v[i] + k2v * dt / 2, t[i] + dt / 2, damping)

        k4x = velocity(x[i] + k3x * dt, v[i] + k3v * dt, t[i] + dt)
        k4v = acceleration(x[i] + k3x * dt, v[i] + k3v * dt, t[i] + dt, damping)

        x_avg = (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        v_avg = (k1v + 2*k2v + 2*k3v + k4v) / 6.0

        v[i+1] = v[i] + v_avg * dt
        x[i+1] = x[i] + x_avg * dt
        t[i+1] = t[i] + dt

    # --------------------------------------------------
    # Detect steady-state periodic motion
    # --------------------------------------------------

    candidate_times = []

    # 314 ≈ one driving period (2π / 2 ≈ π)
    for i in range(14000):
        if round(x[i], 3) == round(x[i + 314], 3):
            if round(x[i + 1], 3) == round(x[i + 315], 3):
                if round(x[i + 2], 3) == round(x[i + 316], 3):
                    if round(x[i + 3], 3) == round(x[i + 317], 3):
                        candidate_times.append(t[i])

    # First time when periodicity is detected
    transient_times.append(candidate_times[0])


# --------------------------------------------------
# Plot transient time vs damping
# --------------------------------------------------

plt.plot(damping_values, transient_times)
plt.xlabel("Damping constant")
plt.ylabel("Transient time")
plt.title("Transient time vs damping in driven oscillator")
plt.show()
