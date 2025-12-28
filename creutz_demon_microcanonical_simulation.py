"""
Title: Creutz Demon Algorithm — Microcanonical Ensemble
Physics: Statistical Mechanics
Method: Monte Carlo (Creutz demon algorithm)

System:
A classical many-particle system simulated in the microcanonical
ensemble using the Creutz demon algorithm. The demon exchanges energy
with the system while conserving the total energy.

Description:
The algorithm performs random trial updates to particle velocities.
Energy changes are accepted only if the demon can supply the required
energy. At equilibrium, the demon energy distribution provides a
measure of the system temperature.

This implementation tracks convergence, average system energy,
mean velocity, and demon energy as a function of Monte Carlo steps.

Units:
Mass is taken as m = 1.
Energy and temperature are expressed in dimensionless units.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt

# --------------------------------------------------
# User input
# --------------------------------------------------

num_particles = int(input("Input number of particles: "))
total_energy = float(input("Input total system energy: "))

# --------------------------------------------------
# Initial velocities (equal energy per particle)
# --------------------------------------------------

initial_velocity = np.sqrt(2 * total_energy / num_particles)
velocities = np.full(num_particles, initial_velocity)

# --------------------------------------------------
# Demon parameters
# --------------------------------------------------

demon_energy = 0.0
max_velocity_change = 2.0

# --------------------------------------------------
# Monte Carlo bookkeeping
# --------------------------------------------------

accepted_moves = 0
max_iterations = 1_000_000

cumulative_energy = total_energy
previous_avg_energy = total_energy
demon_energy_sum = 0.0

convergence_counter = 0
iteration_counter = 0

# --------------------------------------------------
# Data storage for plotting
# --------------------------------------------------

time_steps = []
system_energy_avg = []
mean_velocity = []
demon_energy_avg = []
energy_difference = []

# --------------------------------------------------
# Monte Carlo loop (Creutz demon algorithm)
# --------------------------------------------------

for step in range(1, max_iterations + 1):

    avg_energy = cumulative_energy / step
    system_energy_avg.append(avg_energy)
    time_steps.append(step)

    avg_demon_energy = demon_energy_sum / step
    demon_energy_avg.append(avg_demon_energy)

    mean_velocity.append((2 * (avg_energy / num_particles))**0.5)

    energy_diff = abs(previous_avg_energy - avg_energy)
    energy_difference.append(energy_diff)

    # Convergence check
    if energy_diff <= 1.0e-4:
        convergence_counter += 1
    else:
        convergence_counter = 0

    if convergence_counter >= 10:
        print("Equilibrium reached at iteration", step)
        break

    previous_avg_energy = avg_energy

    # ----------------------------------------------
    # Trial velocity updates
    # ----------------------------------------------

    for _ in range(num_particles):

        delta_v = (2 * np.random.rand() - 1) * max_velocity_change
        particle_index = r.randint(0, num_particles - 1)

        trial_velocity = velocities[particle_index] + delta_v
        delta_energy = 0.5 * (trial_velocity**2 - velocities[particle_index]**2)

        if delta_energy <= demon_energy:
            velocities[particle_index] = trial_velocity
            accepted_moves += 1
            demon_energy -= delta_energy
            total_energy += delta_energy

        iteration_counter += 1

    cumulative_energy += total_energy
    demon_energy_sum += demon_energy


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

acceptance_ratio = accepted_moves / (num_particles * max_iterations)
print("Acceptance Ratio =", acceptance_ratio)

# --------------------------------------------------
# Plots
# --------------------------------------------------

plt.plot(time_steps, system_energy_avg, label="Average system energy")
plt.legend()
plt.show()

plt.plot(time_steps, mean_velocity, label="Mean velocity")
plt.legend()
plt.show()

plt.plot(time_steps, demon_energy_avg, label="Mean demon energy")
plt.legend()
plt.show()

print("Equilibrium mean velocity per particle:", mean_velocity[-1])
print("Mean demon energy (≈ temperature):", demon_energy_avg[-1])
