"""
Title: Phase Space of Simple Harmonic Oscillator
Physics: Classical Mechanics
Method: Fourth-order Runge–Kutta (RK4)

System:
Simple harmonic oscillator governed by
d²x/dt² + ω² x = 0.

Description:
The equations of motion are integrated using the RK4 method for a range
of initial conditions. Phase-space trajectories (x, v) are plotted to
illustrate closed orbits and energy conservation in harmonic motion.

This program visualizes the structure of phase space for a linear
dynamical system.

Units:
All quantities are dimensionless.

Author: Aprameyan Veerereghavan
"""

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# System definition: Simple Harmonic Oscillator
# d²x/dt² + ω² x = 0
# --------------------------------------------------

OMEGA_SQ = 2.0   # ω² parameter (change this!)

def dx_dt(x, y):
    return y

def dy_dt(x, y):
    return -OMEGA_SQ * x


# --------------------------------------------------
# RK4 integrator
# --------------------------------------------------

def rk4_step(x, y, dt):
    k1x = dx_dt(x, y)
    k1y = dy_dt(x, y)

    k2x = dx_dt(x + 0.5 * k1x * dt, y + 0.5 * k1y * dt)
    k2y = dy_dt(x + 0.5 * k1x * dt, y + 0.5 * k1y * dt)

    k3x = dx_dt(x + 0.5 * k2x * dt, y + 0.5 * k2y * dt)
    k3y = dy_dt(x + 0.5 * k2x * dt, y + 0.5 * k2y * dt)

    k4x = dx_dt(x + k3x * dt, y + k3y * dt)
    k4y = dy_dt(x + k3x * dt, y + k3y * dt)

    x_next = x + (dt / 6) * (k1x + 2*k2x + 2*k3x + k4x)
    y_next = y + (dt / 6) * (k1y + 2*k2y + 2*k3y + k4y)

    return x_next, y_next


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

dt = 0.01
n_steps = 300


# --------------------------------------------------
# Phase-space trajectories
# --------------------------------------------------

for x0 in range(-4, 5):
    for y0 in range(-4, 5):

        x = np.zeros(n_steps)
        y = np.zeros(n_steps)

        x[0] = x0
        y[0] = y0

        for i in range(n_steps - 1):
            x[i+1], y[i+1] = rk4_step(x[i], y[i], dt)

        plt.plot(x, y, color='blue', linewidth=0.6)

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Phase Space of SHO (RK4), ω² = {OMEGA_SQ}")
plt.grid(True)
plt.show()
