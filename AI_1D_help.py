import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def find_energies(n):
    """Calculate the energy levels of the quantum harmonic oscillator."""
    hbar = 1
    omega = 1
    return (n + 0.5) * hbar * omega

def V(x):
    """Potential energy function for the harmonic oscillator."""
    m = 1
    omega = 1
    return 0.5 * m * omega**2 * x**2

def schrodinger(r, x, E):
    """Schrödinger equation as a system of first-order ODEs."""
    hbar = 1
    m = 1
    psi, phi = r  # psi is the wavefunction, phi is its derivative
    dpsi = phi
    dphi = (2 * m / hbar**2) * (V(x) - E) * psi
    return [dpsi, dphi]

def solver(n):
    """Solves the Schrödinger equation numerically for the given quantum number n."""
    x_vals = np.linspace(-5, 5, 1000)  # More points for better resolution
    initial = [0, 1e-6]  # Small nonzero initial condition to avoid singularity
    E = find_energies(n)

    solution = odeint(schrodinger, initial, x_vals, args=(E,))
    psi_vals = solution[:, 0]

    # Normalize the wavefunction
    norm = np.trapz(psi_vals**2, x_vals)
    psi_vals /= np.sqrt(norm)

    return x_vals, psi_vals

def plot_schrodinger(n_values):
    """Plots the wavefunction and probability density for given quantum numbers."""
    fig, ax = plt.subplots(len(n_values), 2, figsize=(12, 1.75 * len(n_values)))

    for i, n in enumerate(n_values):
        x_vals, psi_vals = solver(n)

        # Potential function for visualization
        bound_vals = V(x_vals)

        # Normalize potential for plotting
        bound_vals = bound_vals / np.max(bound_vals) * np.max(np.abs(psi_vals)) * 0.8

        ax[i, 0].plot(x_vals, psi_vals, color='g')
        ax[i, 0].set_title(f"Wavefunction for n = {n}")
        ax[i, 0].set_xlabel("Position")
        ax[i, 0].set_ylabel("Psi")
        ax[i, 0].grid(True)

        ax[i, 1].plot(x_vals, psi_vals**2, color='g')
        ax[i, 1].plot(x_vals, bound_vals, color='b', linestyle='dashed', label="Potential")
        ax[i, 1].set_title(f"Probability Density for n = {n}")
        ax[i, 1].set_xlabel("Position")
        ax[i, 1].set_ylabel("Probability Density")
        ax[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

# Run the plotting function for given energy levels
plot_schrodinger([1, 2, 3, 4, 5])
