
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def find_energies(n):
    hbar = 1
    omega = 1
    return (n + 0.5) * hbar * omega

def V(x):
    m = 1
    omega = 1
    return 0.5 * m * (omega ** 2) * (x ** 2)

def schrodinger(r, x, V, E):
    hbar = 1
    m = 1
    psi, dpsi = r

    #dpsi = dpsi
    d2psi = (2 * m / hbar**2) * (V(x) - E) * psi

    return [dpsi, d2psi]

def solver(n):
    resolution = 500
    min = -5
    max = 5
    x_vals = np.linspace(min, max, resolution)

    initial = [0, 1]

    E = find_energies(n)

    solution = odeint(schrodinger, initial, x_vals, args=(V, E))
    psi_vals = solution[:, 0]

    norm = np.trapezoid(psi_vals ** 2, x_vals)
    psi_vals /= np.sqrt(norm)

    return x_vals, psi_vals, E

def plot_schrodinger(n):

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    energy_limit = []
    for i, n in enumerate(n):

        x_vals, psi_vals, E = solver(n)
        energy_limit.append(E)
        bound_vals = []
        for x in x_vals:
            bound_vals.append(0.5 * x ** 2)

        ax[0].plot(x_vals, bound_vals, linestyle='--', color='k', label = "Boundary")
        ax[0].plot(x_vals, [E] * len(x_vals), linestyle='--', color='m', alpha=0.5, label = "Energy Level")
        ax[0].plot(x_vals, psi_vals + E, label = "Wavefunction/Probability")
        ax[0].set_title("Wavefunction of n = {0}".format(n))
        ax[0].set_xlabel("Position")
        ax[0].set_ylabel("$\psi(x)$")
        ax[0].fill_between(x_vals, bound_vals, color='gray', alpha=0.3)
        ax[0].set_ylim(0, max(energy_limit) + 0.25 * max(bound_vals))
        ax[0].grid(True)

        ax[1].plot(x_vals, bound_vals, linestyle='--', color='k', label="Boundary")
        ax[1].plot(x_vals, [E]*len(x_vals), linestyle='--', color='m', alpha=0.5, label="EnergyLevel")
        ax[1].plot(x_vals, E + (psi_vals**2), label="Wavefunction/Probability")
        ax[1].set_title("Probability of n={0}".format(n))
        ax[1].set_xlabel("Position")
        ax[1].set_ylabel("Prob. Density")
        ax[1].fill_between(x_vals,bound_vals, color='gray', alpha=0.3)
        ax[1].fill_between(x_vals, [E] * len(x_vals), E + (psi_vals**2), color="green", alpha=0.3, label="Allowed area")
        ax[1].set_ylim(0, max(energy_limit) + 0.25 * max(bound_vals))
        ax[1].grid(True)

    plt.tight_layout()
    plt.show()

plot_schrodinger([1, 2, 3, 4, 5, 6])