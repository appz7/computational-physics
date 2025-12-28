"""
Title: 1D Infinite Potential Well — Shooting Method
Physics: Time-independent Schrödinger equation
Method: Finite-difference shooting method + secant method
System: Particle in a 1D infinite square well

Description:
This program solves the 1D Schrödinger equation for an infinite
potential well using a shooting method. The wavefunction is
propagated using a second-order finite-difference recurrence relation,
and the energy eigenvalue is obtained via the secant method by enforcing
the boundary condition ψ(L) = 0.

Units:
Natural units are used such that ħ² / (2m) = 1.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Spatial grid
# --------------------------------------------------

dx = 0.001
x_grid = np.arange(0, 0.102, dx)

# --------------------------------------------------
# Shooting method for 1D Schrödinger equation
# ψ_{n+1} = (2 - E) ψ_n - ψ_{n-1}
# --------------------------------------------------

def compute_wavefunction(energy):
    psi_prev = 0.0          # ψ(0) = 0
    psi_curr = 2.0          # arbitrary initial slope

    psi_values = [psi_prev, psi_curr]

    # Propagate wavefunction
    for _ in range(100):
        psi_next = (2 - energy) * psi_curr - psi_prev
        psi_values.append(psi_next)

        psi_prev = psi_curr
        psi_curr = psi_next

    return psi_values


# --------------------------------------------------
# Secant method to find eigenvalue
# --------------------------------------------------

energy_1 = 0.01
psi_1 = compute_wavefunction(energy_1)

energy_2 = 0.02
psi_2 = compute_wavefunction(energy_2)

energy_3 = (
    energy_2 * psi_1[-1] - energy_1 * psi_2[-1]
) / (psi_1[-1] - psi_2[-1])

# Iterate until convergence
while abs(energy_3 - energy_2) > 1e-5:
    energy_1 = energy_2
    energy_2 = energy_3

    psi_1 = compute_wavefunction(energy_1)
    psi_2 = compute_wavefunction(energy_2)

    energy_3 = (
        energy_2 * psi_1[-1] - energy_1 * psi_2[-1]
    ) / (psi_1[-1] - psi_2[-1])

# --------------------------------------------------
# Final normalized eigenfunction
# --------------------------------------------------

psi = np.array(compute_wavefunction(energy_3))

# Correct quantum normalization
normalization = np.sqrt(dx * np.sum(psi**2))
psi_normalized = psi / normalization

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.plot(x_grid, psi_normalized)
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Normalized eigenfunction (shooting method)")
plt.show()
