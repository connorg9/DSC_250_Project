
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

    fig, ax = plt.subplots(len(n), 2, figsize=(12, 3 * len(n)))

    for i, n in enumerate(n):

        x_vals, psi_vals, E = solver(n)

        bound_vals = []
        for x in x_vals:
            bound_vals.append(0.5 * x ** 2)

        ax[i, 0].plot(x_vals, bound_vals, linestyle='--', color='k', label = "Boundary")
        ax[i, 0].plot(x_vals, [E] * len(x_vals), linestyle='--', color='m', alpha=0.5, label = "Energy Level")
        ax[i, 0].plot(x_vals, psi_vals + E, label = "Wavefunction/Probability")
        ax[i, 0].set_title("Wavefunction of n = {0}".format(n))
        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("$\psi(x)$")
        ax[i, 0].fill_between(x_vals, bound_vals, color='gray', alpha=0.3)
        ax[i, 0].set_ylim(0, 6.5)
        ax[i, 0].grid(True)

        ax[i, 1].plot(x_vals, bound_vals, linestyle='--', color='k', label = "Boundary")
        ax[i, 1].plot(x_vals, [E] * len(x_vals), linestyle='--', color='m', alpha=0.5, label = "Energy Level")
        ax[i, 1].plot(x_vals, E + (psi_vals ** 2), label = "Wavefunction/Probability")
        ax[i, 1].set_title("Probability of n = {0}".format(n))
        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Prob. Density")
        ax[i, 1].fill_between(x_vals, bound_vals, color='gray', alpha=0.3)
        ax[i, 1].fill_between(x_vals, [E] * len(x_vals), E + (psi_vals ** 2), color = "green", alpha = 0.3, label = "Allowed area")
        ax[i, 1].set_ylim(0, 6.5)
        ax[i, 1].grid(True)

    handles, labels = ax[1, 1].get_legend_handles_labels()  # Get labels from one subplot
    fig.legend(handles, labels, bbox_to_anchor=(0.58, 0.57), fontsize='x-small')
    plt.tight_layout()
    plt.savefig("schrodinger_{}.pdf".format(n))
    plt.show()

plot_schrodinger([1, 2])