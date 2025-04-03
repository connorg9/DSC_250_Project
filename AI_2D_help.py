import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh


def solve_schrodinger_2d(nx, ny, Nx=50, Ny=50):
    """
    Solve 2D Schrödinger equation using finite difference method

    Parameters:
    -----------
    nx : int
        Quantum number in x direction
    ny : int
        Quantum number in y direction
    Nx : int, optional
        Number of grid points in x direction (default 50)
    Ny : int, optional
        Number of grid points in y direction (default 50)

    Returns:
    --------
    tuple
        x and y meshgrid, wavefunction
    """
    # Grid parameters
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    # Grid spacing
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    # 1D Kinetic energy matrices
    def create_kinetic_matrix(N, h):
        return diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / (2 * h ** 2)

    Tx = create_kinetic_matrix(Nx, hx)
    Ty = create_kinetic_matrix(Ny, hy)

    # Potential energy (infinite square well)
    V = np.ones((Nx, Ny)) * 1e10
    V[(x >= 0) & (x <= 1), :][:, (y >= 0) & (y <= 1)] = 0

    # Identity matrices
    Ix = eye(Nx)
    Iy = eye(Ny)

    # Construct Hamiltonian using Kronecker product
    H = (kron(Iy, -Tx) +
         kron(-Tx, Iy) +
         diags(V.flatten(), 0))

    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SM')

    # Reshape and normalize
    psi = eigenvectors[:, 0].reshape(Nx, Ny)
    psi /= np.sqrt(np.sum(psi ** 2 * hx * hy))

    return X, Y, psi


def plot_schrodinger_2d_3d(quantum_states):
    """
    Plot 2D and 3D Schrödinger wave functions and probability densities

    Parameters:
    -----------
    quantum_states : list of tuples
        List of (nx, ny) quantum number pairs
    """
    fig = plt.figure(figsize=(16, 4 * len(quantum_states)))

    for i, (nx, ny) in enumerate(quantum_states):
        X, Y, psi = solve_schrodinger_2d(nx, ny)

        # 2D Wave function plot
        ax1 = fig.add_subplot(len(quantum_states), 3, 3 * i + 1)
        im1 = ax1.pcolormesh(X, Y, psi, cmap='viridis', shading='auto')
        ax1.set_title(f'2D Wave Function (n_x={nx}, n_y={ny})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)

        # 2D Probability density plot
        ax2 = fig.add_subplot(len(quantum_states), 3, 3 * i + 2)
        im2 = ax2.pcolormesh(X, Y, psi ** 2, cmap='plasma', shading='auto')
        ax2.set_title(f'2D Probability Density (n_x={nx}, n_y={ny})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)

        # 3D Surface plot
        ax3 = fig.add_subplot(len(quantum_states), 3, 3 * i + 3, projection='3d')
        surf = ax3.plot_surface(X, Y, psi, cmap='viridis',
                                linewidth=0, antialiased=False)
        ax3.set_title(f'3D Wave Function Surface (n_x={nx}, n_y={ny})')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('Ψ')
        fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()


# Example usage
plot_schrodinger_2d_3d([(0, 0), (1, 0), (0, 1), (1, 1)])