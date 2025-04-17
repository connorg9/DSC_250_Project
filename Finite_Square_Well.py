import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import brentq

hbar = 1
m = 1
def find_energy(a, V0):

    return


def V(x, a, V0):
    return np.where((-a / 2 <= x) & (x <= a / 2), 0, V0)


def schrodinger(r, x, V, E, a, V0):
    psi, dpsi = r

    # dpsi = dpsi
    d2psi = (2 * m / hbar ** 2) * (V(x, a, V0) - E) * psi

    return [dpsi, d2psi]


def solver(n, a, V0):
    resolution = 1000
    x_min = -a - (a / 2)
    x_max = a + (a / 2)
    x_vals = np.linspace(x_min, x_max, resolution)

    initial = [0, 1e-10]
    energies, roots = find_energy(a, V0)
    E = energies[n]
    print(E)

    solution = odeint(schrodinger, initial, x_vals, args=(V, E, a, V0))
    psi_vals = solution[:, 0]

    norm = np.trapezoid(psi_vals ** 2, x_vals)
    psi_vals /= np.sqrt(norm)

    return x_vals, psi_vals, E


def plot_schrodinger(n, a, V0):
    fig, ax = plt.subplots(len(n), 2, figsize=(12, 1.75 * len(n)))

    for i, n in enumerate(n):

        x_vals, psi_vals, E = solver(n, a, V0)

        left_bound = x_vals <= -a / 2
        right_bound = x_vals >= a / 2

        ax[i, 0].plot(x_vals[left_bound], [V0] * len(x_vals[left_bound]), linestyle='--', color='k')
        ax[i, 0].plot(x_vals[right_bound], [V0] * len(x_vals[right_bound]), linestyle='--', color='k')
        ax[i, 0].plot(x_vals, E + psi_vals)
        ax[i, 0].set_title("Wavefunction of n = {0}".format(n))

        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("Psi")
        ax[i, 0].grid(True)

        ax[i, 1].plot(x_vals[left_bound], [V0] * len(x_vals[left_bound]), linestyle='--', color='k')
        ax[i, 1].plot(x_vals[right_bound], [V0] * len(x_vals[right_bound]), linestyle='--', color='k')
        ax[i, 1].plot(x_vals, E + (psi_vals**2))
        ax[i, 1].set_title("Probability of n = {0}".format(n))

        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Prob. Density")
        ax[i, 1].grid(True)

    plt.tight_layout()
    plt.show()


plot_schrodinger([1, 2], 2, 20)
#This kinda works, but doesn't graph properly
