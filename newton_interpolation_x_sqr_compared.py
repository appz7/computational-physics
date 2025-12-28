"""
Title: Newton’s Divided Difference Interpolation
Physics: Numerical Methods
Method: Polynomial interpolation (Newton form)

Problem:
Interpolation of a function using Newton’s divided difference method,
with comparison against a known analytical function.

Description:
Given a set of data points, the Newton divided difference table is
constructed and used to evaluate the interpolation polynomial at
new points. The interpolated values are compared with the exact
function to illustrate interpolation accuracy and behavior.

This program demonstrates the construction and evaluation of
Newton interpolation polynomials.

Author: Aprameyan Veerereghavan
"""

import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# Newton's Divided Difference Interpolation
# ==================================================

# Number of given data points
n = int(input('enter number of data points'))

# x-array stores the x-values of data points
x = np.zeros(n)

# y-table stores divided differences
# y[j][i] = i-th divided difference starting at j
y = np.zeros([n, n])

# --------------------------------------------------
# Input data points
# --------------------------------------------------

for i in range(n):
    x0 = float(input('enter the x value '))
    y0 = float(input('enter the y value '))
    x[i] = x0
    y[i][0] = y0       # Zeroth divided difference = y value

# --------------------------------------------------
# Construct divided difference table
# --------------------------------------------------

for i in range(1, n):
    for j in range(n - i):
        y[j][i] = (y[j+1][i-1] - y[j][i-1]) / (x[j+i] - x[j])

# --------------------------------------------------
# Evaluate interpolation polynomial
# --------------------------------------------------

# Points where interpolation is evaluated
X1 = np.arange(-4, 4, 1)

Y1 = []   # Interpolated values
Y2 = []   # Exact function values (for comparison)

for k in range(len(X1)):
    ans = 0

    # Newton interpolation formula
    for i in range(n):
        coefficient = 1
        for j in range(i):
            coefficient = coefficient * (X1[k] - x[j])
        ans = ans + coefficient * y[0][i]

    Y1.append(ans)

    # Exact function used for comparison
    o = X1[k]
    q = o * o - 0.3
    Y2.append(q)

# --------------------------------------------------
# Plot interpolated vs exact function
# --------------------------------------------------

plt.plot(X1, Y1, label='Newton interpolation')
plt.plot(X1, Y2, label='Exact function')
plt.legend()
plt.show()
