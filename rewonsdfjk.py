import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root_scalar

# Constants
hbar = 1.0
m = 1.0

# Potential parameters
a = 2.0     # width of the well
V0 = 20.0   # height of potential outside
L = 4.0     # integration range (-L to L)

# Define potential function
def V(x):
    return 0 if abs(x) < a / 2 else V0

# Schrödinger equation as a system of first-order ODEs
def schrodinger(y, x, E):
    psi, phi = y  # y[0] = ψ, y[1] = ψ'
    dpsi_dx = phi
    dphi_dx = 2 * m / hbar**2 * (V(x) - E) * psi
    return [dpsi_dx, dphi_dx]

# Runge-Kutta integration (via odeint)
def solve_psi(E):
    x = np.linspace(-L, L, 1000)
    y0 = [0.0, 1.0]  # Initial guess: ψ(0)=0, ψ'(0)=1 (odd symmetry)
    sol = odeint(schrodinger, y0, x, args=(E,))
    psi = sol[:, 0]
    return x, psi

# Objective function: match ψ at the boundary (e.g., at x=L, should go to 0)
def boundary_condition(E):
    x, psi = solve_psi(E)
    return psi[-1]  # Value at x = L

# Find energy that satisfies boundary condition
res = root_scalar(boundary_condition, bracket=[0.01, V0 - 0.01], method='bisect')
E_found = res.root

# Get final solution
x, psi = solve_psi(E_found)

# Normalize
norm = np.trapz(psi**2, x)
psi /= np.sqrt(norm)

# Plot
plt.plot(x, psi, label=f'E = {E_found:.2f}')
plt.plot(x, [V(x)/40 for x in x], 'k--', label='V(x)/40')
plt.axhline(0, color='gray', lw=0.5)
plt.title('Wavefunction from Shooting Method')
plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.legend()
plt.grid(True)
plt.show()
