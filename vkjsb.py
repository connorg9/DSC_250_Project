
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def find_energy(n):
    return (n ** 2) * (np.pi ** 2) / 2

def V(x):
    return np.where((0 <= x) & (x <= 1), 0, 1e10)

def schrodinger(r, x, V, E):
    hbar = 1
    m = 1
    psi, phi = r

    if 0 <= x <= 1:
        dpsi = phi
        dphi = (2 * m / hbar**2) * (V(x) - E) * psi
    else:
        dpsi = 0
        dphi = 0
    return [dpsi, dphi]

def solver(n):
    resolution = 500
    min = -1
    max = 2
    x_vals = np.linspace(min, max, resolution)
    valid_x = [x for x in x_vals if 0 <= x <= 1]

    initial = [0, 1]

    E = find_energy(n)

    solution = odeint(schrodinger, initial, valid_x, args=(V, E))
    psi_vals = solution[:, 0]

    norm = np.trapezoid(psi_vals ** 2, valid_x)
    psi_vals /= np.sqrt(norm)

    zeros = int((len(x_vals) - len(valid_x)) / 2)

    beginning = np.zeros(zeros)
    psi_vals = np.concatenate((beginning, psi_vals))
    psi_vals = np.append(psi_vals, [0] * zeros)

    return x_vals, psi_vals, E

def plot_schrodinger(n_values):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    energy_limit = [] #used for graph y-lim later

    for n in n_values:
        x_vals, psi_vals, E = solver(n)
        energy_limit.append(E)

        left_bound = x_vals <= 0
        middle = (x_vals > 0) & (x_vals < 1)
        right_bound = x_vals >= 1

        ax[0].plot(x_vals[left_bound], E + psi_vals[left_bound], color='r', alpha=0.5)
        ax[0].plot(x_vals[middle], E + psi_vals[middle], label=f"n = {n}", linewidth=2)
        ax[0].plot(x_vals[right_bound],E + psi_vals[right_bound], color='r', alpha=0.5)

        psi_squared = psi_vals ** 2
        ax[1].plot(x_vals[left_bound], E + psi_squared[left_bound], color='r', alpha=0.5)
        ax[1].plot(x_vals[middle], E + psi_squared[middle], label=f"n = {n}", linewidth=2)
        ax[1].plot(x_vals[right_bound], E + psi_squared[right_bound], color='r', alpha=0.5)

    ax[0].set_title("Wavefunctions")
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel(r"$\psi(x)$")
    ax[0].set_ylim(0, max(energy_limit) + .2 * max(energy_limit))
    ax[0].axvline(0, color='k', linestyle='--')
    ax[0].axvline(1, color='k', linestyle='--')
    ax[0].fill_between(x_vals, 0, max(energy_limit) + .2 * max(energy_limit), where=(x_vals <= 0) | (x_vals >= 1), color='gray', alpha=0.3)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title("Probability Densities")
    ax[1].set_xlabel("Position")
    ax[1].set_ylabel("Probability Density")
    ax[1].set_ylim(0, max(energy_limit) + .2 * max(energy_limit))
    ax[1].axvline(0, color='k', linestyle='--')
    ax[1].axvline(1, color='k', linestyle='--')
    ax[1].fill_between(x_vals, 0, max(energy_limit) + .2 * max(energy_limit), where=(x_vals <= 0) | (x_vals >= 1), color='gray', alpha=0.3)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

plot_schrodinger([1, 2, 3, 4, 5])


# Example usage
