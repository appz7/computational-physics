"""
Title: Forced and Damped Oscillator — Resonance Curve
Physics: Classical Mechanics
Method: Fourth-order Runge–Kutta (RK4)

System:
A driven, damped harmonic oscillator governed by the equation
x'' + d x' + ω₀² x = sin(ω t), where d is the damping constant and
ω is the driving frequency.

Description:
The equation of motion is integrated numerically using the RK4 method
for a range of driving frequencies and damping constants. After
discarding transient dynamics, the steady-state oscillation amplitude
is extracted to construct resonance curves.

This program illustrates resonance behavior, damping effects, and
frequency response in driven systems.

Units:
All quantities are expressed in dimensionless units.

Author: Aprameyan Veerereghavan
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Equation of motion:
# x'' + d x' + 3x = sin(ω t)
# --------------------------------------------------

def acceleration(x, v, t, damping, omega):
    return -natural_freq*x - damping*v + np.sin(omega*t)

def velocity(x, v, t):
    return v


# --------------------------------------------------
# Time grid and storage
# --------------------------------------------------
natural_freq = 3
num_steps = 20000
dt = 0.01

x = [0.0] * (num_steps + 1)
v = [0.0] * (num_steps + 1)
t = [0.0] * (num_steps + 1)

# --------------------------------------------------
# Parameter ranges
# --------------------------------------------------

omega_values = np.arange(natural_freq-1, natural_freq+1, 0.01)     # driving frequency
damping_values = [0.05, 0.1, 0.2]              # damping constants

# --------------------------------------------------
# Loop over damping values
# --------------------------------------------------

for damping in damping_values:

    resonance_amplitudes = []

    # Loop over driving frequencies
    for omega in omega_values:

        displacement_samples = []

        # Initial conditions (reset for each frequency)
        x[0] = 5
        v[0] = 0
        t[0] = 0

        # --------------------------------------------------
        # RK4 time integration
        # --------------------------------------------------

        for i in range(num_steps):

            k1x = velocity(x[i], v[i], t[i])
            k1v = acceleration(x[i], v[i], t[i], damping, omega)

            k2x = velocity(x[i] + k1x*dt/2, v[i] + k1v*dt/2, t[i] + dt/2)
            k2v = acceleration(x[i] + k1x*dt/2, v[i] + k1v*dt/2, t[i] + dt/2, damping, omega)

            k3x = velocity(x[i] + k2x*dt/2, v[i] + k2v*dt/2, t[i] + dt/2)
            k3v = acceleration(x[i] + k2x*dt/2, v[i] + k2v*dt/2, t[i] + dt/2, damping, omega)

            k4x = velocity(x[i] + k3x*dt, v[i] + k3v*dt, t[i] + dt)
            k4v = acceleration(x[i] + k3x*dt, v[i] + k3v*dt, t[i] + dt, damping, omega)

            x_avg = (k1x + 2*k2x + 2*k3x + k4x) / 6.0
            v_avg = (k1v + 2*k2v + 2*k3v + k4v) / 6.0

            v[i+1] = v[i] + v_avg * dt
            x[i+1] = x[i] + x_avg * dt
            t[i+1] = t[i] + dt

            # Collect data after transient dies out
            if i > 19000:
                displacement_samples.append(x[i])

        # Reset arrays (kept exactly as in original logic)
        x = [0.0] * (num_steps + 1)
        v = [0.0] * (num_steps + 1)
        t = [0.0] * (num_steps + 1)

        # Resonant amplitude = max steady-state displacement
        resonance_amplitudes.append(max(displacement_samples))

    # --------------------------------------------------
    # Plot resonance curve for this damping
    # --------------------------------------------------

    plt.plot(omega_values, resonance_amplitudes, label=f"d = {damping}")
    plt.xlabel("Driving frequency ω")
    plt.ylabel("Resonant amplitude")
    plt.legend()


plt.show()
