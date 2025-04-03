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
    resolution = 50
    min = -2
    max = 2
    x_vals = np.linspace(min, max, resolution)
    y_vals = np.linspace(min, max, resolution)

    valid_x = [x for x in x_vals if 0 <= x <= 1]
    valid_y = [y for y in y_vals if 0 <= y <= 1]

    X, Y = np.meshgrid(valid_x, valid_y)
    initial = [[0, 1]

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

def plot_schrodinger(n):
    fig, ax = plt.subplots(len(n), 2)

    for i, n in enumerate(n):

        x_vals, psi_vals, E = solver(n)

        left_bound = x_vals <= 0
        middle = (x_vals > 0) & (x_vals < 1)
        right_bound = x_vals >= 1

        ax[i, 0].plot(x_vals[left_bound], psi_vals[left_bound], color='r')
        ax[i, 0].plot(x_vals[middle], psi_vals[middle], color='g')
        ax[i, 0].plot(x_vals[right_bound], psi_vals[right_bound], color='r')

        ax[i, 0].set_ylim(-2, 2)

        ax[i, 0].set_title("Wavefunction of n = {0}".format(n))
        ax[i, 0].axvline(0, color='k', linestyle='--')
        ax[i, 0].axvline(1, color='k', linestyle='--')
        ax[i, 0].fill_between(x_vals[left_bound], 2, alpha=0.3, color='gray')
        ax[i, 0].fill_between(x_vals[left_bound], -2, alpha=0.3, color='gray')
        ax[i, 0].fill_between(x_vals[right_bound], 2, alpha=0.3, color='gray')
        ax[i, 0].fill_between(x_vals[right_bound], -2, alpha=0.3, color='gray')
        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("Psi")
        ax[i, 0].grid(True)

        ax[i, 1].plot(x_vals[left_bound], psi_vals[left_bound] ** 2, color='r')
        ax[i, 1].plot(x_vals[middle], psi_vals[middle] ** 2, color='g')
        ax[i, 1].plot(x_vals[right_bound], psi_vals[right_bound] ** 2, color='r')

        ax[i, 1].set_ylim(-0.5, 2.5)

        ax[i, 1].set_title("Probability of n = {0}".format(n))
        ax[i, 1].axvline(0, color='k', linestyle='--')
        ax[i, 1].axvline(1, color='k', linestyle='--')
        ax[i, 1].fill_between(x_vals[left_bound], 2.5, alpha=0.3, color='gray')
        ax[i, 1].fill_between(x_vals[left_bound], -0.5, alpha=0.3, color='gray')
        ax[i, 1].fill_between(x_vals[right_bound], 2.5, alpha=0.3, color='gray')
        ax[i, 1].fill_between(x_vals[right_bound], -0.5, alpha=0.3, color='gray')
        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Prob. Density")
        ax[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

plot_schrodinger([1, 2, 3, 4, 5])