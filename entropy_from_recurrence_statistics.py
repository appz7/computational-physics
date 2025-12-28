"""
Title: Entropy from Recurrence Statistics
Physics: Statistical Mechanics / Information Theory
Method: Recurrence-based entropy estimation

System:
A discrete system with a fixed number of binary states, where a given
number of states are occupied randomly in each trial.

Description:
Random configurations are generated for different numbers of occupied
states. Recurrence events (repeated configurations) are counted, and
the entropy is estimated using the logarithm of the mean recurrence
time.

This approach illustrates the connection between entropy and the
statistics of state recurrences in finite systems.

Entropy Measure:
S ∝ ln(τ), where τ is the recurrence time.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# System parameters
# --------------------------------------------------

num_states = 10                    # Number of binary slots
num_trials = 200                   # Number of random experiments
total_comparisons = (num_trials * (num_trials - 1)) / 2

occupied_counts = []
entropy_values = []

# --------------------------------------------------
# Loop over number of occupied slots
# --------------------------------------------------

for num_occupied in range(num_states + 1):

    occupied_counts.append(num_occupied)
    configurations = np.zeros(num_trials)

    # Generate random configurations
    for trial in range(num_trials):

        occupancy = np.zeros(num_states)
        count = 0

        while count <= num_occupied - 1:
            index = np.random.randint(0, num_states)
            if occupancy[index] != 1:
                occupancy[index] = 1
                count += 1

        # Encode configuration as integer
        config_code = 0
        for k in range(len(occupancy)):
            if occupancy[k] == 1:
                config_code += (2 ** k)

        configurations[trial] = config_code

    # --------------------------------------------------
    # Identify unique configurations
    # --------------------------------------------------

    unique_configs = []
    for i in range(len(configurations)):
        is_new = True
        for j in range(i):
            if configurations[i] == configurations[j]:
                is_new = False
                break
        if is_new:
            unique_configs.append(configurations[i])

    # --------------------------------------------------
    # Count recurrence events
    # --------------------------------------------------

    coincidence_count = 0
    for config in unique_configs:
        occurrences = 0
        for sample in configurations:
            if config == sample:
                occurrences += 1
        coincidence_count += (occurrences * (occurrences - 1)) / 2

    # --------------------------------------------------
    # Recurrence-based entropy
    # --------------------------------------------------

    recurrence_ratio = coincidence_count / total_comparisons
    recurrence_time = 1 / recurrence_ratio
    entropy_values.append(np.log(recurrence_time))


# --------------------------------------------------
# Plot entropy vs number of occupied states
# --------------------------------------------------

plt.plot(occupied_counts, entropy_values, marker='o')
plt.xlabel("Number of occupied states")
plt.ylabel("Entropy (ln τ)")
plt.title("Entropy from recurrence statistics")
plt.show()
