import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def find_energy(nx, ny):
    return (nx ** 2 + ny ** 2) * (np.pi ** 2) / 2

def V(x, y):
    return np.where((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1), 0, 1e10)

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

def solver(nx, ny):
    resolution = 50
    min = -2
    max = 2
    x_vals = np.linspace(min, max, resolution)
    y_vals = np.linspace(min, max, resolution)

    valid_x = [x for x in x_vals if 0 <= x <= 1]
    valid_y = [y for y in y_vals if 0 <= y <= 1]

    X, Y = np.meshgrid(valid_x, valid_y)

    initial = [[0, 1], [0, 1]]

    E = find_energy(nx, ny)

    solution = odeint(schrodinger, initial, valid_x, args=(V, E))
    psi_vals = solution[:, 0]

    norm = np.trapezoid(psi_vals ** 2, valid_x)
    psi_vals /= np.sqrt(norm)

    zeros = int((len(x_vals) - len(valid_x)) / 2)

    beginning = np.zeros(zeros)
    psi_vals = np.concatenate((beginning, psi_vals))
    psi_vals = np.append(psi_vals, [0] * zeros)

    return x_vals, psi_vals, E

def plot_schrodinger(states):
    fig, ax = plt.subplots(len(states), 2)

    for i, (nx, ny) in enumerate(states):

        x_vals, psi_vals, E = solver(nx, ny)



        ax[i, 0].set_title("Wavefunction of n = {0}".format(nx, ny))
        ax[i, 0].axvline(0, color='k', linestyle='--')
        ax[i, 0].axvline(1, color='k', linestyle='--')

        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("Psi")
        ax[i, 0].grid(True)



        ax[i, 1].set_ylim(-0.5, 2.5)

        ax[i, 1].set_title("Probability of n = {0}".format(nx, ny))
        ax[i, 1].axvline(0, color='k', linestyle='--')
        ax[i, 1].axvline(1, color='k', linestyle='--')

        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Prob. Density")
        ax[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

plot_schrodinger([(1, 1), (2, 1), (1, 2)])