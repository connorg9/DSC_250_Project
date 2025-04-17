import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def find_energy(nx, ny):
    # Expected energy for 2D case (for comparison)
    return (nx ** 2 + ny ** 2) * (np.pi ** 2) / 2


def V(x, y):
    # 2D potential
    return np.where(((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1)), 0, 1e10)


def schrodinger_2d(x, y, psi, E):
    # Set up 2D Schrödinger equation
    # -∇²ψ + V(x,y)ψ = Eψ
    hbar = 1
    m = 1

    # Second derivatives approximation with finite differences
    # We'll use a 5-point stencil for the Laplacian
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize Laplacian result
    lap_psi = np.zeros_like(psi)

    # Interior points (excluding boundaries)
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            # 5-point stencil for Laplacian
            lap_psi[j, i] = (psi[j, i - 1] - 2 * psi[j, i] + psi[j, i + 1]) / dx ** 2 + \
                            (psi[j - 1, i] - 2 * psi[j, i] + psi[j + 1, i]) / dy ** 2

    # Calculate right side of Schrödinger equation: (V-E)ψ
    potential_term = np.zeros_like(psi)
    for i in range(len(x)):
        for j in range(len(y)):
            potential_term[j, i] = (V(x[i], y[j]) - E) * psi[j, i]

    # Full Schrödinger equation: -ℏ²/(2m)∇²ψ + Vψ = Eψ
    # Rearranged: ∇²ψ = (2m/ℏ²)(V-E)ψ
    return lap_psi - (2 * m / hbar ** 2) * potential_term


def solve_2d_eigenvalue(nx, ny, resolution=50):
    # Define grid
    min_val = 0
    max_val = 1
    x = np.linspace(min_val, max_val, resolution)
    y = np.linspace(min_val, max_val, resolution)
    X, Y = np.meshgrid(x, y)

    # Initial guess for the energy (using analytical result)
    E_guess = find_energy(nx, ny)

    # Initial guess for wavefunction
    # We use a simple sine product to get started in the right direction
    psi_guess = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            if 0 <= x[i] <= 1 and 0 <= y[j] <= 1:
                psi_guess[j, i] = np.sin(nx * np.pi * x[i]) * np.sin(ny * np.pi * y[j])

    # Boundary conditions are zero at edges
    psi_guess[0, :] = 0
    psi_guess[-1, :] = 0
    psi_guess[:, 0] = 0
    psi_guess[:, -1] = 0

    # Iterative solution using relaxation method
    max_iterations = 1000
    tolerance = 1e-6
    alpha = 0.1  # Relaxation parameter

    for iteration in range(max_iterations):
        # Calculate new psi using the Schrödinger equation
        lap_psi = schrodinger_2d(x, y, psi_guess, E_guess)

        # Update psi using relaxation method (but preserve boundary conditions)
        new_psi = psi_guess.copy()
        new_psi[1:-1, 1:-1] = psi_guess[1:-1, 1:-1] + alpha * lap_psi[1:-1, 1:-1]

        # Normalize
        norm = np.sum(new_psi ** 2) * (x[1] - x[0]) * (y[1] - y[0])
        if norm > 0:
            new_psi /= np.sqrt(norm)

        # Check convergence
        diff = np.max(np.abs(new_psi - psi_guess))
        psi_guess = new_psi

        if diff < tolerance:
            print(f"Converged after {iteration} iterations")
            break

    # Calculate actual energy from the final wavefunction
    lap_psi = np.zeros_like(psi_guess)
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            lap_psi[j, i] = (psi_guess[j, i - 1] - 2 * psi_guess[j, i] + psi_guess[j, i + 1]) / ((x[1] - x[0]) ** 2) + \
                            (psi_guess[j - 1, i] - 2 * psi_guess[j, i] + psi_guess[j + 1, i]) / ((y[1] - y[0]) ** 2)

    # E = <ψ|H|ψ> = <ψ|-∇²/2 + V|ψ>
    potential_energy = np.sum(
        np.array([[V(x[i], y[j]) * psi_guess[j, i] ** 2 for i in range(len(x))] for j in range(len(y))])) * (
                                   x[1] - x[0]) * (y[1] - y[0])
    kinetic_energy = -0.5 * np.sum(lap_psi * psi_guess) * (x[1] - x[0]) * (y[1] - y[0])
    calculated_E = kinetic_energy + potential_energy

    print(f"Expected E({nx},{ny}) = {E_guess:.4f}, Calculated E = {calculated_E:.4f}")

    return X, Y, psi_guess, calculated_E


def plot_schrodinger_2d(quantum_states):
    num_states = len(quantum_states)
    fig = plt.figure(figsize=(15, 5 * num_states))

    for i, (nx, ny) in enumerate(quantum_states):
        print(f"\nSolving for state ({nx}, {ny}):")
        X, Y, psi_vals, E = solve_2d_eigenvalue(nx, ny)
        prob_density = psi_vals ** 2

        # 3D surface plot for wavefunction
        ax1 = fig.add_subplot(num_states, 3, 3 * i + 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, psi_vals, cmap=cm.viridis, linewidth=0)
        ax1.set_title(f"Wavefunction for nx={nx}, ny={ny}")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.set_zlabel("Psi")

        # 3D surface plot for probability density
        ax2 = fig.add_subplot(num_states, 3, 3 * i + 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, prob_density, cmap=cm.plasma, linewidth=0)
        ax2.set_title(f"Probability Density for nx={nx}, ny={ny}")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.set_zlabel("Probability Density")

        # 2D contour plot for probability density
        ax3 = fig.add_subplot(num_states, 3, 3 * i + 3)
        contour = ax3.contourf(X, Y, prob_density, cmap=cm.plasma)
        ax3.set_title(f"Contour Plot for nx={nx}, ny={ny}, E={E:.2f}")
        ax3.set_xlabel("X Position")
        ax3.set_ylabel("Y Position")
        ax3.grid(True)
        fig.colorbar(contour, ax=ax3)

        # Draw the well boundaries
        for ax in [ax3]:
            ax.axvline(0, color='k', linestyle='--')
            ax.axvline(1, color='k', linestyle='--')
            ax.axhline(0, color='k', linestyle='--')
            ax.axhline(1, color='k', linestyle='--')

    plt.tight_layout()
    plt.show()


# Plot several quantum states with different nx and ny values
quantum_states = [(1, 1), (2, 1), (1, 2)]
plot_schrodinger_2d(quantum_states)