import numpy as np
import matplotlib.pyplot as plt


def find_energy(n_x, n_y):
    return (np.pi ** 2 / 2) * (n_x ** 2 + n_y ** 2)


def psi(n, x):
    return np.sqrt(2) * np.sin(n * np.pi * x)


def solver(n_x, n_y):
    resolution = 100
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    psi_vals = psi(n_x, X) * psi(n_y, Y)
    E = find_energy(n_x, n_y)

    return X, Y, psi_vals, E


def plot_schrodinger(n_values):
    fig, axes = plt.subplots(len(n_values), 2, figsize=(10, 4 * len(n_values)))
    if len(n_values) == 1:
        axes = [axes]

    for i, (n_x, n_y) in enumerate(n_values):
        X, Y, psi_vals, E = solver(n_x, n_y)

        im1 = axes[i][0].contourf(X, Y, psi_vals, cmap='coolwarm')
        fig.colorbar(im1, ax=axes[i][0])
        axes[i][0].set_title(f"Wavefunction for (n_x, n_y) = ({n_x}, {n_y}), E = {E:.2f}")

        im2 = axes[i][1].contourf(X, Y, psi_vals ** 2, cmap='inferno')
        fig.colorbar(im2, ax=axes[i][1])
        axes[i][1].set_title(f"Probability Density for (n_x, n_y) = ({n_x}, {n_y})")

    plt.tight_layout()
    plt.show()


plot_schrodinger([(1, 1), (1, 2), (2, 2), (2, 3)])
