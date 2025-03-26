
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
    psi_values = solution[:, 0]

    norm = np.trapezoid(psi_values ** 2, valid_x)
    psi_values /= np.sqrt(norm)

    zeros = len(x_vals) - len(valid_x)

    psi_values = np.append(psi_values, [0] * zeros)

    print(psi_values)
    return x_vals, valid_x, psi_values



def plot_schrodinger(n):

    fig, ax = plt.subplots(len(n), 2, figsize=(12, 3 * len(n)))

    for i, n in enumerate(n):

        x_vals, valid_x, psi_values = solver(n)

        ax[i, 0].plot(valid_x, psi_values)
        ax[i, 0].set_title("Wavefunction of n = {0}".format(n))
        ax[i, 0].axvline(0, color='k', linestyle='--')
        ax[i, 0].axvline(1, color='k', linestyle='--')
        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("Psi")
        ax[i, 0].grid(True)

        ax[i, 1].plot(valid_x, psi_values ** 2)
        ax[i, 1].set_title("Probability of n = {0}".format(n))
        ax[i, 1].axvline(0, color='k', linestyle='--')
        ax[i, 1].axvline(1, color='k', linestyle='--')
        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Prob. Density")
        ax[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

plot_schrodinger([1, 2, 3, 4, 5])