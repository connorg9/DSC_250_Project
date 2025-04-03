import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def find_energy(n, a, V0):
    hbar = 1
    m = 1
    return ((n**2) * (np.pi ** 2) * (hbar ** 2)) / ((2 * m) * ((2 * a)**2)) - V0

def V(x, a, V0):
    return np.where((-a/2 <= x) & (x <= a/2), 0, V0)

def schrodinger(r, x, V, E, a, V0):
    hbar = 1
    m = 1
    psi, phi = r

    dpsi = phi
    dphi = (2 * m / hbar**2) * (V(x, a, V0) - E) * psi

    return [dpsi, dphi]

def solver(n, a, V0):
    resolution = 500
    min = a - (2 * a) - (a/2)
    max = a + (a/2)
    x_vals = np.linspace(min, max, resolution)

    initial = [0, 1]

    E = find_energy(n, a, V0)

    solution = odeint(schrodinger, initial, x_vals, args=(V, E, a, V0))
    psi_vals = solution[:, 0]

    norm = np.trapezoid(psi_vals ** 2, x_vals)
    psi_vals /= np.sqrt(norm)

    return x_vals, psi_vals, E

def plot_schrodinger(n, a, V0):
    fig, ax = plt.subplots(len(n), 2, figsize=(12, 1.75 * len(n)))

    for i, n in enumerate(n):

        x_vals, psi_vals, E = solver(n, a, V0)

        left_bound = x_vals <= -0.5
        middle = (x_vals > -0.5) & (x_vals < 0.5)
        right_bound = x_vals >= 0.5

        ax[i, 0].plot(x_vals[left_bound], [V0] * len(x_vals[left_bound]), linestyle='--', color='k')
        ax[i, 0].plot(x_vals[right_bound], [V0] * len(x_vals[right_bound]), linestyle='--', color='k')
        ax[i, 0].plot(x_vals[left_bound], E + psi_vals[left_bound], color='r')
        ax[i, 0].plot(x_vals[middle], E + psi_vals[middle], color='g')
        ax[i, 0].plot(x_vals[right_bound], E + psi_vals[right_bound], color='r')
        ax[i, 0].set_title("Wavefunction of n = {0}".format(n))
        ax[i, 0].axvline(-0.5, color='k', linestyle='--')
        ax[i, 0].axvline(0.5, color='k', linestyle='--')

        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("Psi")
        ax[i, 0].grid(True)

        ax[i, 1].plot(x_vals[left_bound], [V0] * len(x_vals[left_bound]), linestyle='--', color='k')
        ax[i, 1].plot(x_vals[right_bound], [V0] * len(x_vals[right_bound]), linestyle='--', color='k')
        ax[i, 1].plot(x_vals[left_bound], E + (psi_vals[left_bound]**2), color='r')
        ax[i, 1].plot(x_vals[middle], E + (psi_vals[middle]**2), color='g')
        ax[i, 1].plot(x_vals[right_bound], E + (psi_vals[right_bound]**2), color='r')
        ax[i, 1].set_title("Probability of n = {0}".format(n))
        ax[i, 1].axvline(-0.5, color='k', linestyle='--')
        ax[i, 1].axvline(0.5, color='k', linestyle='--')

        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Prob. Density")
        ax[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

plot_schrodinger([1, 2, 3, 4, 5], 1, 10)