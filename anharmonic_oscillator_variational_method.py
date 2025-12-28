"""
Title: Anharmonic Oscillator — Variational Method
Physics: Quantum Mechanics (Time-independent Schrödinger equation)
Method: Variational approach using finite harmonic oscillator basis
         and Hamiltonian matrix diagonalisation (Jacobi rotation)

System:
A one-dimensional anharmonic oscillator with potential
V(x) = (1/2) ω² x² + λ x⁴ (quartic term optional).

Description:
The Hamiltonian is constructed in a truncated harmonic oscillator
basis by evaluating matrix elements of the potential numerically.
The resulting Hamiltonian matrix is diagonalised using the Jacobi
rotation method to obtain approximate energy eigenvalues.
This is a variational calculation, and the eigenvalues serve as
upper bounds to the true energies.

Units:
Natural units are used such that ħ² / (2m) = 1.

Notes:
- Basis size is truncated (3 × 3), so results are qualitative.
- Accuracy improves with larger basis size.
- The quartic term can be enabled by uncommenting the x⁴ contribution.

Author: Aprameyan Veerereghavan
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m

# --------------------------------------------------
# Physical parameters
# --------------------------------------------------

omega = 0.5   # Harmonic oscillator frequency

# --------------------------------------------------
# Harmonic oscillator basis functions ψ_n(x)
# --------------------------------------------------

def ho_basis(n):
    x_grid = np.arange(-50, 50, 0.01)

    gaussian = np.exp(-omega * x_grid**2 / 2)
    normalization = 1 / (2**n * m.factorial(n))
    psi = normalization * gaussian

    if n == 0:
        psi = psi
    elif n == 1:
        psi = psi * (2 * x_grid)
    elif n == 2:
        psi = psi * (4 * x_grid**2 - 2)
    elif n == 3:
        psi = psi * (8 * x_grid**3 - 12 * x_grid)

    return psi


# --------------------------------------------------
# Spatial grid and potential
# --------------------------------------------------

x_grid = np.arange(-50, 50, 0.01)

# Anharmonic potential: V(x) = ½ ω² x² + 0.1 x⁴
potential = 0.5 * omega**2 * x_grid**2 + 0.1 * (x_grid**4)
#+ 0.1 * x_grid**4

# --------------------------------------------------
# Hamiltonian matrix construction
# --------------------------------------------------

V_matrix = np.zeros([3, 3])   # Potential energy matrix
T_matrix = np.zeros([3, 3])   # Kinetic + HO part
H_matrix = np.zeros([3, 3])   # Total Hamiltonian

# Potential energy matrix elements
for i in range(3):
    for j in range(3):
        integrand = ho_basis(i) * potential * ho_basis(j)
        integral = (0.01 / 2) * (2 * sum(integrand) - integrand[0] - integrand[-1])
        V_matrix[i][j] = integral

# Harmonic oscillator Hamiltonian matrix elements
for i in range(3):
    for j in range(3):
        if i == j:
            T_matrix[i][i] = (i + 0.5) * omega * 0.5
        if abs(i - j) == 2:
            if j > i:
                T_matrix[i][j] = -0.25 * omega * ((i + 1) * j)**0.5
            else:
                T_matrix[i][j] = T_matrix[j][i]

# Total Hamiltonian
H_matrix = V_matrix + T_matrix

# --------------------------------------------------
# Sanity check: ⟨0|0⟩ normalization
# --------------------------------------------------

psi0 = ho_basis(0) * ho_basis(0)
norm_check = (0.01 / 2) * (2 * sum(psi0) - psi0[0] - psi0[-1])
print(norm_check)   # should be finite


# --------------------------------------------------
# Jacobi rotation method for diagonalization
# --------------------------------------------------

def rotate_matrix(A, J):
    size = len(J)
    JT = J.T

    temp = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            for k in range(size):
                temp[i][j] += A[i][k] * J[k][j]

    rotated = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            for k in range(size):
                rotated[i][j] += JT[i][k] * temp[k][j]

    return rotated


# --------------------------------------------------
# Jacobi iteration
# --------------------------------------------------

off_diag_sum = 0
diag_sum = 0

for i in range(len(H_matrix)):
    for j in range(len(H_matrix)):
        if i != j:
            off_diag_sum += abs(H_matrix[i][j])**2
        else:
            diag_sum += abs(H_matrix[i][j])

tolerance = diag_sum / 100000
iterations = 0

while off_diag_sum > tolerance:

    J = np.identity(len(H_matrix))

    max_val = H_matrix[0][1]
    p = 0
    q = 1

    for i in range(len(H_matrix)):
        for j in range(len(H_matrix)):
            if i != j and H_matrix[i][j] > max_val:
                max_val = H_matrix[i][j]
                p = i
                q = j

    theta = (H_matrix[p][p] - H_matrix[q][q]) / (2 * H_matrix[p][q])

    if theta >= 100:
        t = 1 / (2 * theta)
    elif theta < 0:
        t = -1 / (abs(theta) + (theta**2 + 1)**0.5)
    elif 0 < theta < 100:
        t = 1 / (abs(theta) + (theta**2 + 1)**0.5)
    else:
        t = 0

    c = 1 / (t**2 + 1)**0.5
    s = t * c

    J[p][p] = c
    J[q][q] = c
    J[p][q] = s
    J[q][p] = -s

    H_matrix = rotate_matrix(H_matrix, J)

    off_diag_sum = 0
    for i in range(len(H_matrix)):
        for j in range(len(H_matrix)):
            if i != j:
                off_diag_sum += abs(H_matrix[i][j])**2

    iterations += 1
    if iterations > 4000:
        break


# --------------------------------------------------
# Final eigenvalues
# --------------------------------------------------

for i in range(len(H_matrix)):
    print(f"Eigenvalue {i} ≈ {round(H_matrix[i][i], 2)}")
