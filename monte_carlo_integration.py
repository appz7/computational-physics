"""
Title: Monte Carlo Integration — Hit-or-Miss and Mean-Value Methods
Physics: Numerical Methods / Computational Physics
Method: Monte Carlo Integration

Problem:
Numerical evaluation of the definite integral
∫ sin²(x) dx over a finite interval.

Description:
Two Monte Carlo techniques are implemented:
1) Hit-or-miss Monte Carlo integration
2) Mean-value Monte Carlo integration

A convergence study is performed by varying the number of random
samples, and the error is analyzed relative to the exact analytical
result. The expected N^{-1/2} Monte Carlo error scaling is illustrated
on a log–log plot.

Units:
All quantities are dimensionless.

Author: Aprameyan Veerereghavan
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Function to integrate
# --------------------------------------------------

def integrand(x):
    return np.sin(x)**2


# --------------------------------------------------
# Hit-or-miss Monte Carlo integration
# --------------------------------------------------

def monte_carlo_hit_or_miss(x_min, x_max, n_samples):
    count_inside = 0

    # Used only to determine y-range
    x_grid = np.arange(x_min, x_max, 0.01)
    y_min = np.min(integrand(x_grid))
    y_max = np.max(integrand(x_grid))

    for _ in range(n_samples):
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)

        if y_rand <= integrand(x_rand):
            count_inside += 1

    integral_estimate = (count_inside / n_samples) * (x_max - x_min)
    return integral_estimate


# --------------------------------------------------
# Integration limits and sample size
# --------------------------------------------------

a = 0
b = 2 * np.pi
n_samples = 1000

I_estimate = monte_carlo_hit_or_miss(a, b, n_samples)
print("Monte Carlo (hit-or-miss) result:", I_estimate)


# --------------------------------------------------
# Simple (mean-value) Monte Carlo integration
# --------------------------------------------------

def monte_carlo_mean_value(x_min, x_max, n_samples):
    x_random = np.random.uniform(x_min, x_max, n_samples)
    integral_estimate = (x_max - x_min) * np.mean(integrand(x_random))
    return integral_estimate


# --------------------------------------------------
# Convergence study
# --------------------------------------------------

sample_sizes = np.logspace(1, 5, num=50, dtype=int)

integral_values = []
for n in sample_sizes:
    integral_values.append(monte_carlo_mean_value(a, b, n))

# Exact value of the integral
exact_value = np.pi

error = np.abs(np.array(integral_values) - exact_value)

plt.loglog(sample_sizes, error)
plt.xlabel("Number of samples (N)")
plt.ylabel("Absolute error")
plt.title("Monte Carlo Integration Error Scaling")
plt.show()
