
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def find_energy(n):
    return (n ** 2) * (np.pi ** 2) / 2

def V(x):
    return np.where((0 <= x) & (x <= 1), 0, np.inf)

def schrodinger(r, x, V, E):

    hbar = 1
    m = 1
    psi, phi = r
    dpsi = phi
    dphi = (2 * m / hbar**2) * (V(x) - E) * psi
    return [dpsi, dphi]

def solver(n):
    x_vals = np.linspace(0, 1, 100)
    initial = [0, 1]

    E = find_energy(n)

    solution = odeint(schrodinger, initial, x_vals, args=(V, E))
    psi_values = solution[:, 0]
    norm = np.trapezoid(psi_values ** 2, x_vals)
    psi_values /= np.sqrt(norm)

    return x_vals, psi_values

def plot_schrodinger(n):
    x_vals, psi_values = solver(n)

    fig, ax = plt.subplots()
    plt.figure()
    plt.subplot(211)
    plt.plot(x_vals, psi_values)
    plt.xlabel("Position")
    plt.ylabel("Wavefunction")

    plt.subplot(212)
    plt.plot(x_vals, psi_values ** 2)
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

