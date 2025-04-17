import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
hbar = 1.0
m = 1.0


def find_energy_levels(a, V0, n_max=5):
    """Find energy eigenvalues using the transcendental equations."""
    # Parameter related to well depth
    z0 = np.sqrt(2 * m * V0) * a / (2 * hbar)

    # Arrays to store results
    energies = []

    # Loop through possible values to find roots
    for n in range(1, 20):  # Try to find enough states
        # First guess for energy (based on infinite well)
        E_guess = (n ** 2 * np.pi ** 2 * hbar ** 2) / (2 * m * a ** 2)
        if E_guess >= V0:
            continue  # Skip unbound states

        # Calculate parameters
        k = np.sqrt(2 * m * E_guess) / hbar  # wavenumber inside well
        kappa = np.sqrt(2 * m * (V0 - E_guess)) / hbar  # decay constant outside well

        # Check if even or odd
        if n % 2 == 1:  # odd n means even wavefunction (cos)
            # Even function: equation is k*tan(k*a/2) = kappa
            def f(E):
                if E >= V0 or E <= 0:
                    return np.inf
                k = np.sqrt(2 * m * E) / hbar
                kappa = np.sqrt(2 * m * (V0 - E)) / hbar
                return k * np.tan(k * a / 2) - kappa

            parity = "even"
        else:  # even n means odd wavefunction (sin)
            # Odd function: equation is -k*cot(k*a/2) = kappa
            def f(E):
                if E >= V0 or E <= 0:
                    return np.inf
                k = np.sqrt(2 * m * E) / hbar
                kappa = np.sqrt(2 * m * (V0 - E)) / hbar
                return -k / np.tan(k * a / 2) - kappa

            parity = "odd"

        # Find energy by solving the transcendental equation
        try:
            # Try to bracket the solution
            E_low = 0.01
            E_high = min(V0, (n ** 2 * np.pi ** 2 * hbar ** 2) / (2 * m * a ** 2)) * 0.99

            # Check if solution exists in this bracket
            y_low, y_high = f(E_low), f(E_high)
            if np.sign(y_low) == np.sign(y_high):
                continue  # No root in this interval

            # Find root
            E = root_scalar(f, bracket=[E_low, E_high]).root
            energies.append((E, parity, n))

            if len(energies) >= n_max:
                break
        except:
            continue

    # Sort by energy
    energies.sort(key=lambda x: x[0])
    return energies[:n_max]


def wavefunction(x, n, a, V0):
    """Compute the wavefunction for energy level n."""
    # Get energies and symmetry
    energies = find_energy_levels(a, V0, n_max=max(3, n))
    if n > len(energies):
        raise ValueError(f"Only {len(energies)} energy levels found, requested n={n}")

    E, parity, _ = energies[n - 1]  # n is 1-indexed

    # Calculate parameters
    k = np.sqrt(2 * m * E) / hbar  # wavenumber inside well
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar  # decay constant outside well

    # Function to evaluate wavefunction at each point
    psi = np.zeros_like(x)

    if parity == "even":
        # Even function: psi(x) = A*cos(k*x) inside, B*exp(-kappa*|x|) outside
        A = 1.0  # will normalize later

        # Inside well
        mask_inside = np.abs(x) <= a / 2
        psi[mask_inside] = A * np.cos(k * x[mask_inside])

        # Outside well - match value at boundary for continuity
        B = A * np.cos(k * a / 2)
        mask_outside = np.abs(x) > a / 2
        psi[mask_outside] = B * np.exp(-kappa * (np.abs(x[mask_outside]) - a / 2))

    else:  # parity == "odd"
        # Odd function: psi(x) = A*sin(k*x) inside, B*sgn(x)*exp(-kappa*|x|) outside
        A = 1.0  # will normalize later

        # Inside well
        mask_inside = np.abs(x) <= a / 2
        psi[mask_inside] = A * np.sin(k * x[mask_inside])

        # Outside well - match value at boundary for continuity
        B = A * np.sin(k * a / 2) / np.exp(-kappa * 0)
        mask_outside = np.abs(x) > a / 2
        psi[mask_outside] = B * np.sign(x[mask_outside]) * np.exp(-kappa * (np.abs(x[mask_outside]) - a / 2))

    # Normalize
    norm = np.sqrt(np.trapz(psi ** 2, x))
    psi /= norm

    return psi, E


def plot_solutions(n_values, a, V0):
    """Plot wavefunctions and probability densities for given energy levels."""
    x_min = -4
    x_max = 4
    x = np.linspace(x_min, x_max, 2000)  # Use more points for smoother curves

    # Define potential function
    def V(x):
        return np.where(np.abs(x) <= a / 2, 0, V0)

    pot = V(x)

    fig, axs = plt.subplots(len(n_values), 2, figsize=(12, 3 * len(n_values)))

    for i, n in enumerate(n_values):
        # Get wavefunction and energy
        psi, E = wavefunction(x, n, a, V0)
        prob = psi ** 2

        # Plot wavefunction
        if len(n_values) == 1:
            ax1 = axs[0]
            ax2 = axs[1]
        else:
            ax1 = axs[i, 0]
            ax2 = axs[i, 1]

        # Plot wavefunction
        ax1.plot(x, psi, 'b-', label=r'$\psi(x)$')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Add horizontal line at E
        ax1.axhline(y=E / 20, color='r', linestyle='--', label=f'E={E:.2f}')

        # Plot potential
        ax1.plot(x, pot / 20, 'k--', alpha=0.5, label='V(x)')

        ax1.set_title(f'Wavefunction of n = {n}')
        ax1.set_xlabel('Position')
        ax1.set_ylabel(r'$\psi(x)$')
        ax1.grid(True)
        ax1.legend()

        # Plot probability density
        ax2.plot(x, prob, 'b-', label=r'$|\psi(x)|^2$')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Plot potential
        ax2.plot(x, pot / 20, 'k--', alpha=0.5, label='V(x)')

        ax2.set_title(f'Probability Density of n = {n}')
        ax2.set_xlabel('Position')
        ax2.set_ylabel(r'$|\psi(x)|^2$')
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.show()


# Test with a=2, V0=20
plot_solutions([1, 2, 3], 2, 20)