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
    dpsi = phi
    dphi = (2 * m / hbar ** 2) * (V(x) - E) * psi
    return [dpsi, dphi]


def solve_schrodinger(n):
    x_vals = np.linspace(0, 1, 1000)  # Increased resolution
    initial = [0, 1]

    E = find_energy(n)

    solution = odeint(schrodinger, initial, x_vals, args=(V, E))
    print(solution)
    psi_values = solution[:, 0]

    # More robust normalization
    prob_density = psi_values ** 2
    print(prob_density)
    total_probability = np.trapezoid(prob_density, x_vals)
    print(total_probability)
    prob_density /= total_probability

    total_probability = np.trapezoid(prob_density, x_vals)


    return x_vals, psi_values


def plot_schrodinger_states(quantum_numbers):
    """
    Create subplots for wavefunctions and probability distributions.

    Args:
        quantum_numbers (list): List of quantum numbers to plot
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(len(quantum_numbers), 2, figsize=(12, 4 * len(quantum_numbers)))

    # Flatten axs if only one quantum number is provided
    if len(quantum_numbers) == 1:
        axs = axs.reshape(1, -1)

    # Iterate through quantum numbers
    for i, n in enumerate(quantum_numbers):
        # Solve Schrödinger equation
        x_vals, psi = solve_schrodinger(n)

        # Plot wavefunction
        axs[i, 0].plot(x_vals, psi)
        axs[i, 0].set_title(f'Wavefunction (n = {n})')
        axs[i, 0].set_xlabel('Position')
        axs[i, 0].set_ylabel('Ψ(x)')
        axs[i, 0].grid(True)

        # Plot probability distribution
        axs[i, 1].plot(x_vals, psi ** 2)
        axs[i, 1].set_title(f'Probability Distribution (n = {n})')
        axs[i, 1].set_xlabel('Position')
        axs[i, 1].set_ylabel('Probability Density')
        axs[i, 1].grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Demonstrate for multiple quantum numbers
plot_schrodinger_states([1, 2, 3])