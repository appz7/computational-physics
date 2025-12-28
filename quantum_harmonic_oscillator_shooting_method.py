"""
Title: Quantum Harmonic Oscillator — Shooting Method
Physics: Quantum Mechanics
Method: Finite-difference shooting method with matching condition

System:
One-dimensional quantum harmonic oscillator with potential
V(x) = (1/2) m ω² x² (with constants absorbed into units).

Description:
The time-independent Schrödinger equation is solved numerically by
integrating the wavefunction from both sides of the domain toward the
classical turning point. A matching condition is used to define an
eigenvalue equation, which is solved using the secant method.

The resulting energy eigenvalue and corresponding eigenfunction
are obtained and visualized.

Units:
Natural units are used such that ħ² / (2m) = 1.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Spatial domain (used implicitly)
# --------------------------------------------------

x_left = -4
x_right = 4

# --------------------------------------------------
# Right-to-left integration of Schrödinger equation
# --------------------------------------------------

def integrate_right(energy, turning_point):
    h = 0.001                      # spatial step size
    x = -4                         # start from left boundary

    psi_prev = 0                   # ψ(x-h)
    psi_curr = 5                   # ψ(x)
    psi_values = [psi_prev, psi_curr]

    while x <= turning_point + h:
        psi_next = (2 - h**2 * (energy - 3 * x**2)) * psi_curr - psi_prev
        psi_values.append(psi_next)

        psi_prev = psi_curr
        psi_curr = psi_next
        x = x + h

    return psi_values


# --------------------------------------------------
# Left-to-right integration of Schrödinger equation
# --------------------------------------------------

def integrate_left(energy, turning_point):
    h = 0.001                      # spatial step size
    x = 4                          # start from right boundary

    psi_prev = 0
    psi_curr = 5
    psi_values = [psi_prev, psi_curr]

    while x >= turning_point - h:
        psi_next = (2 - h**2 * (energy - 3 * x**2)) * psi_curr - psi_prev
        psi_values.append(psi_next)

        psi_prev = psi_curr
        psi_curr = psi_next
        x = x - h

    return psi_values


# --------------------------------------------------
# Energy scan to locate eigenvalues
# --------------------------------------------------

energy_range = np.arange(1, 15, 0.1)
mismatch_values = []

for energy in energy_range:

    turning_point = (energy / 3)**0.5   # classical turning point

    psi_right = np.array(integrate_right(energy, turning_point))
    psi_left = np.array(integrate_left(energy, turning_point))

    # Scale left solution to match right solution
    scale_factor = psi_right[-2] / psi_left[-2]
    psi_left = psi_left * scale_factor

    # Difference used for matching condition
    difference = psi_left[-1] - psi_right[-3]
    mismatch_values.append(difference)


# --------------------------------------------------
# Helper function for root finding
# --------------------------------------------------

def mismatch(energy):
    turning_point = (energy / 3)**0.5

    psi_right = np.array(integrate_right(energy, turning_point))
    psi_left = np.array(integrate_left(energy, turning_point))

    scale_factor = psi_right[-2] / psi_left[-2]
    psi_left = psi_left * scale_factor

    difference = psi_left[-1] - psi_right[-3]
    return difference


# --------------------------------------------------
# Plot mismatch vs energy
# --------------------------------------------------

plt.plot(energy_range, mismatch_values)
plt.xlabel("Energy")
plt.ylabel("Wavefunction mismatch")
plt.title("Eigenvalue condition via shooting method")
plt.show()


# --------------------------------------------------
# Compute eigenfunction for final eigenvalue
# --------------------------------------------------

def compute_eigenfunction(energy):
    h = 0.001
    x = -3.998

    psi_prev = 0
    psi_curr = 1

    x_values = [x, -3.999]
    psi_values = [psi_prev, psi_curr]

    while x < 3:
        psi_next = (2 - h**2 * (energy - 3 * x**2)) * psi_curr - psi_prev

        psi_values.append(psi_next)
        psi_prev = psi_curr
        psi_curr = psi_next

        x_values.append(x)
        x = x + h

    return x_values, psi_values


# --------------------------------------------------
# Secant method to refine eigenenergy
# --------------------------------------------------

energy_1 = 2
energy_2 = 2.5

mismatch_1 = mismatch(energy_1)
mismatch_2 = mismatch(energy_2)

energy_3 = (energy_2 * mismatch_1 - energy_1 * mismatch_2) / (mismatch_1 - mismatch_2)

while abs(energy_3 - energy_2) > 1e-5:
    energy_1 = energy_2
    energy_2 = energy_3

    mismatch_2 = mismatch(energy_2)
    mismatch_1 = mismatch(energy_1)

    energy_3 = (energy_2 * mismatch_1 - energy_1 * mismatch_2) / (mismatch_1 - mismatch_2)


# --------------------------------------------------
# Plot final eigenfunction
# --------------------------------------------------

x_vals, psi_vals = compute_eigenfunction(energy_3)

plt.plot(x_vals, psi_vals)
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Eigenfunction of 1D Quantum Harmonic Oscillator")
plt.show()
