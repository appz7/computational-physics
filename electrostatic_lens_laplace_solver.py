"""
Title: Electrostatic Lens — Laplace Equation Solver
Physics: Electrodynamics
Method: Finite Difference Method with Successive Over-Relaxation (SOR)

System:
An axisymmetric electrostatic lens geometry defined on a cylindrical
(r, z) grid. Fixed electrode potentials are imposed as boundary
conditions.

Description:
Laplace’s equation is solved iteratively on a two-dimensional grid
using finite-difference approximations. Successive over-relaxation
is employed to accelerate convergence. Symmetry is enforced by
mirroring the solution in both radial and axial directions.

The resulting potential distribution is visualized through
equipotential contour lines.

Units:
Potential is expressed in arbitrary voltage units.
Spatial coordinates are dimensionless grid units.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Grid parameters
# --------------------------------------------------

step = 0.01              # Spatial step size
num_r = 100              # Radial grid points
num_z = 40               # Axial grid points

# --------------------------------------------------
# Initialize potential array
# --------------------------------------------------

potential = np.zeros([num_r, num_z])

# --------------------------------------------------
# Boundary conditions (electrode geometry)
# --------------------------------------------------

# Horizontal electrode at z = 20, r = 20 → 99
for r in range(20, 100):
    potential[r][20] = 1000

# Vertical electrode at r = 20, z = 20 → 39
for z in range(20, 40):
    potential[20][z] = 1000

# Linear potential ramp at r = 99, z = 0 → 19
voltage = 0
for z in range(20):
    potential[99][z] = voltage
    voltage += 50

# Top electrode at z = 39, r = 0 → 19
for r in range(20):
    potential[r][39] = 1000


# --------------------------------------------------
# Iterative solution of Laplace's equation (SOR)
# --------------------------------------------------

for iteration in range(200):
    for r in range(99):

        # Different axial extent depending on radial position
        if r < 20:
            z_max = 39
        else:
            z_max = 20

        for z in range(1, z_max):

            # Axis condition (r = 0)
            if r == 0:
                phi_new = (1/6) * (
                    4 * potential[r+1][z]
                    + potential[r][z+1]
                    + potential[r][z-1]
                )

            # General cylindrical Laplace update
            else:
                phi_new = (1/4) * (
                    potential[r+1][z]
                    + potential[r-1][z]
                    + potential[r][z+1]
                    + potential[r][z-1]
                ) + (1/(8*r)) * (
                    potential[r+1][z]
                    - potential[r-1][z]
                )

            # Successive over-relaxation (SOR)
            potential[r][z] += 1.5 * (phi_new - potential[r][z])


# --------------------------------------------------
# Enforce symmetry by mirroring
# --------------------------------------------------

# Mirror in radial direction
potential_mirror_r = np.zeros([num_r, num_z])
for r in range(num_r):
    potential_mirror_r[r] = potential[num_r - 1 - r]

potential = np.vstack((potential_mirror_r, potential))

# Mirror in axial direction
potential_mirror_z = np.zeros([2 * num_r, num_z])
potential_negative = -potential

for z in range(num_z):
    potential_mirror_z[:, z] = potential_negative[:, num_z - 1 - z]

potential = np.concatenate((potential_mirror_z, potential), axis=1)


# --------------------------------------------------
# Plot equipotential contours
# --------------------------------------------------

levels = [-1000, -800, -600, -400, -200, 200, 400, 600, 800, 1000]

for level in levels:
    x_vals = []
    y_vals = []

    for r in range(200):
        for z in range(80):
            if level - 5 < potential[r][z] < level + 5:
                x_vals.append(z)
                y_vals.append(r)

    plt.plot(x_vals, y_vals, color='b')

plt.title("Equipotential lines of electrostatic lens")
plt.show()
