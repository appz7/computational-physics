"""
Title: Metropolis Algorithm — Canonical Ensemble
Physics: Statistical Mechanics
Method: Metropolis Monte Carlo

System:
Canonical ensemble simulations for:
1) A single-particle system with fluctuating energy
2) A multi-particle system with energy exchange between particles

Description:
The Metropolis algorithm is used to sample energy configurations
according to the Boltzmann distribution at a fixed temperature.
For the single-particle case, the resulting energy distribution
approaches the expected exponential (Boltzmann) form.
For the multi-particle system, the total energy fluctuates while
individual particle energies evolve stochastically.

Energy distributions are computed and visualized for both cases.

Units:
Boltzmann constant k_B is set to 1.
Energy and temperature are dimensionless.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# PART 1: Metropolis algorithm for ONE particle
# Canonical ensemble (fixed temperature Beta_1)
# Energy is allowed to fluctuate
# ==================================================

num_particles_1 = 1          # Single particle
Beta_1 = 1                  # Inverse temperature (β = 1 / kT, kB = 1)
Energy_1 = 5                # Initial energy
Num_mcs_1 = 10000           # Number of Monte Carlo steps
Energy_values_1 = [Energy_1]
Max_energy_change_1 = 2     # Maximum trial energy change

# Metropolis Monte Carlo loop
for i in range(1, Num_mcs_1):

    # Propose a random energy change
    energy_change = (2 * np.random.rand() - 1) * Max_energy_change_1
    energy_trial = Energy_1 + energy_change
    delta_energy = energy_trial - Energy_1

    # Enforce positivity of energy
    if energy_trial > 0:

        # Always accept downhill moves
        if delta_energy < 0:
            Energy_1 = energy_trial
            Energy_values_1.append(Energy_1)

        # Accept uphill moves with Boltzmann probability
        else:
            w = np.exp(-Beta_1 * delta_energy)
            r = np.random.rand()
            if r < w:
                Energy_1 = energy_trial
                Energy_values_1.append(Energy_1)
            else:
                Energy_values_1.append(Energy_1)


# --------------------------------------------------
# Energy distribution for one particle
# Should follow Boltzmann distribution
# --------------------------------------------------

X_1 = np.linspace(0, 10, 101)
Probability_1 = []

for i in range(len(X_1) - 1):
    count = 0
    for j in Energy_values_1:
        if j < X_1[i + 1] and j > X_1[i]:
            count += 1
    Probability_1.append(count)

X_1 = np.delete(X_1, [-1])

plt.plot(X_1, Probability_1)
plt.xlabel('Energy')
plt.ylabel('Distribution')
plt.title('Energy Distribution for 1 Particle (Canonical Ensemble)')
plt.show()


# ==================================================
# PART 2: Metropolis algorithm for MULTI-particle system
# Canonical ensemble with fixed temperature Beta_2
# Total energy fluctuates
# ==================================================

num_particles_2 = 10          # Number of particles
Beta_2 = 0.4                 # Inverse temperature
Total_energy_2 = 0           # Initial total energy

# Each particle starts with equal energy
Configurations_2 = np.full(num_particles_2, Total_energy_2 / num_particles_2)

Num_mcs_2 = 100000            # Monte Carlo steps
Energy_values_2 = [Total_energy_2]
Max_energy_change_2 = 2       # Maximum trial energy change

# Metropolis Monte Carlo loop
for i in range(1, Num_mcs_2):

    # Attempt updates for each particle
    for j in range(1, num_particles_2 + 1):

        # Random trial energy change
        energy_change = (2 * np.random.rand() - 1) * Max_energy_change_2
        iparticle = np.random.randint(0, num_particles_2)

        energy_trial = Configurations_2[iparticle] + energy_change
        delta_energy = energy_trial - Configurations_2[iparticle]

        # Enforce positivity of individual particle energy
        if energy_trial > 0:

            # Always accept downhill moves
            if delta_energy <= 0:
                Configurations_2[iparticle] = energy_trial
                Total_energy_2 += delta_energy

            # Accept uphill moves with Boltzmann probability
            else:
                w = np.exp(-Beta_2 * delta_energy)
                r = np.random.rand()
                if r < w:
                    Configurations_2[iparticle] = energy_trial
                    Total_energy_2 += delta_energy

    # Store total energy after each Monte Carlo sweep
    Energy_values_2.append(Total_energy_2)


# --------------------------------------------------
# Energy distribution for multi-particle system
# --------------------------------------------------

X_2 = np.linspace(0, 100, 101)
Probability_2 = []

for i in range(len(X_2) - 1):
    count = 0
    for j in Energy_values_2:
        if j < X_2[i + 1] and j > X_2[i]:
            count += 1
    Probability_2.append(count)

X_2 = np.delete(X_2, [-1])

plt.plot(X_2, Probability_2)
plt.xlabel('Energy')
plt.ylabel('Distribution')
plt.title('Energy Distribution for Multi-Particle System (Canonical Ensemble)')
plt.show()
