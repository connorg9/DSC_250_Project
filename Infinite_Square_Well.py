import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root_scalar


def V(x, V0=50, a=1):
    """
    Potential energy function for a finite square well.

    Parameters:
    - x: position
    - V0: height of potential outside well
    - a: width of well
    """
    if np.iterable(x):
        return np.array([V(xi, V0, a) for xi in x])
    elif np.abs(x) < a / 2:
        return 0
    else:
        return V0


def fk(E, V0=50, a=1, n=1):
    """
    Returns the error in the eigenvalue equation for the finite square well.
    Used to find energy eigenvalues analytically.

    Parameters:
    - E: energy to evaluate
    - V0: height of potential outside well
    - a: width of well
    - n: quantum number (1 for even parity, 2 for odd parity)
    """
    k1 = np.sqrt(2 * E)
    k2 = np.sqrt(2 * (V0 - E))
    if n % 2:  # Odd n (even parity solutions)
        return k2 - k1 * np.tan(k1 * a / 2)
    else:  # Even n (odd parity solutions)
        return k2 + k1 / np.tan(k1 * a / 2)


def find_energy(V0=50, a=1, pts=100, max_states=5):
    """
    Finds energy eigenvalues for the finite square well.

    Parameters:
    - V0: height of potential outside well
    - a: width of well
    - pts: number of points to evaluate
    - max_states: maximum number of energy states to find

    Returns:
    - Sorted list of energy eigenvalues
    """
    Ei = np.linspace(0.0, V0, pts)
    roots = []
    for n in [1, 2]:  # Check both even and odd parity solutions
        for i in range(pts - 1):
            try:
                soln = root_scalar(fk, args=(V0, a, n), x0=Ei[i], x1=Ei[i + 1])
                if soln.converged and np.around(soln.root, 9) not in roots:
                    roots.append(np.around(soln.root, 9))
            except ValueError:
                continue

    return np.sort(roots)[:max_states]


def schrodinger(r, x, V, E):
    """
    Defines the Schrödinger equation as a system of first-order ODEs.

    Parameters:
    - r: vector [psi, dpsi/dx]
    - x: position
    - V: potential function
    - E: energy eigenvalue

    Returns:
    - [dpsi/dx, d²psi/dx²]
    """
    hbar = 1
    m = 1
    psi, phi = r

    dpsi = phi
    dphi = (2 * m / hbar ** 2) * (V(x) - E) * psi

    return [dpsi, dphi]


def solver(E, V0=50, a=1):
    """
    Solves the Schrödinger equation for given energy E.

    Parameters:
    - E: energy eigenvalue
    - V0: height of potential outside well
    - a: width of well

    Returns:
    - x_vals: position values
    - psi_vals: normalized wavefunction values
    - E: energy eigenvalue
    """
    resolution = 500
    x_min = -2
    x_max = 2
    x_vals = np.linspace(x_min, x_max, resolution)

    # Determine initial conditions based on parity
    # For simplicity we'll start at left boundary with small value
    x_solve = np.linspace(x_min, x_max, resolution)
    initial = [0.0001, 0.001]  # Small non-zero values

    # Solve Schrödinger equation
    solution = odeint(schrodinger, initial, x_solve, args=(lambda x: V(x, V0, a), E))
    psi_vals = solution[:, 0]

    # Normalize wavefunction
    well_region = np.abs(x_vals) <= a / 2 + 0.5  # Include a bit outside the well
    norm = np.trapezoid(psi_vals[well_region] ** 2, x_vals[well_region])
    if norm > 0:  # Avoid division by zero
        psi_vals /= np.sqrt(norm)

    return x_vals, psi_vals, E


def plot_schrodinger(n_values=None, V0=50, a=1):
    """
    Plots wavefunctions and probability densities for the finite square well.

    Parameters:
    - n_values: list of energy levels to plot (if None, will use first 5 eigenvalues)
    - V0: height of potential outside well
    - a: width of well
    """
    # Find energy eigenvalues
    eigenvalues = find_energy(V0, a, max_states=max(n_values) if n_values else 5)

    # If n_values not specified, use all found eigenvalues
    if n_values is None:
        n_values = range(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    energy_limit = []  # Used for graph y-lim later

    # Plot potential function
    potential_x = np.linspace(-2, 2, 1000)
    potential_y = V(potential_x, V0, a)
    for i in range(2):
        ax[i].plot(potential_x, potential_y, 'k--', alpha=0.3)
        # Fill potential barriers
        ax[i].fill_between(potential_x, 0, V0,
                           where=(np.abs(potential_x) >= a / 2),
                           color='gray', alpha=0.2)

    # Plot wavefunctions and probability densities for each energy level
    for i, n in enumerate(n_values):
        if i < len(eigenvalues):
            E = eigenvalues[i]
            energy_limit.append(E)

            x_vals, psi_vals, _ = solver(E, V0, a)

            # Separate regions inside and outside the well
            inside_well = (np.abs(x_vals) < a / 2)
            outside_well = ~inside_well

            # Plot wavefunction (shifted to its energy level)
            #ax[0].plot(x_vals[outside_well], E + psi_vals[outside_well], 'r', alpha=0.5)
            ax[0].plot(x_vals[inside_well], E + psi_vals[inside_well], label=f"n = {n}", linewidth=2)
            ax[0].axhline(y=E, xmin=0.25, xmax=0.75, linestyle='--', color=f'C{i}')

            # Plot probability density
            prob = psi_vals ** 2
            # ax[1].plot(x_vals[outside_well], E + prob[outside_well], 'r', alpha=0.5)
            ax[1].plot(x_vals[inside_well], E + prob[inside_well], label=f"n = {n}", linewidth=2)
            ax[1].axhline(y=E, xmin=0.25, xmax=0.75, linestyle='--', color=f'C{i}')

    # Set plot properties
    ax[0].set_title("Wavefunctions")
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel(r"$\psi(x)$ + Energy")
    ax[0].set_ylim(0, V0 if V0 < max(energy_limit) + 5 else max(energy_limit) + 5)
    ax[0].axvline(-a / 2, color='k', linestyle='--')
    ax[0].axvline(a / 2, color='k', linestyle='--')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title("Probability Densities")
    ax[1].set_xlabel("Position")
    ax[1].set_ylabel("Probability Density + Energy")
    ax[1].set_ylim(0, V0 if V0 < max(energy_limit) + 5 else max(energy_limit) + 5)
    ax[1].axvline(-a / 2, color='k', linestyle='--')
    ax[1].axvline(a / 2, color='k', linestyle='--')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


V0 = 50  # Height of potential
a = 1  # Width of well
plot_schrodinger([1, 2, 3, 4, 5], V0, a)