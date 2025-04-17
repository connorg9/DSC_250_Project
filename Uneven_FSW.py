import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
hbar = 1.0
m = 1.0
def find_energy_levels(a, V0):
    energies = []

    # Loop through possible values to find roots
    for n in range(1, 20):  # Try to find enough states
        # First guess for energy (based on infinite well)
        E_guess = (n ** 2 * np.pi ** 2 * hbar ** 2) / (2 * m * a ** 2)
        if E_guess >= V0:
            continue  # Basically, there are no bound states for a well with these dimensions

        if n % 2 == 1:  # odd n means even wavefunction (cos)
            # Even function: equation is k*tan(k*a/2) = kappa
            parity = "even"
            def f(E):
                if E >= V0 or E <= 0: #exclude non-bound states or negative energy (not possible)
                    return np.inf
                k = np.sqrt(2 * m * E) / hbar
                kappa = np.sqrt(2 * m * (V0 - E)) / hbar
                return (k * np.tan(k * a / 2)) - kappa


        else: #sin function
            parity = "odd"
            # Odd function: equation is -k*cot(k*a/2) = kappa
            def f(E):
                if E >= V0 or E <= 0: #exclude non-physical energies again
                    return np.inf
                k = np.sqrt(2 * m * E) / hbar
                kappa = np.sqrt(2 * m * (V0 - E)) / hbar
                return (-k / np.tan(k * a / 2)) - kappa
        try:
            # Try to bracket solution
            E_low = 0.01
            E_high = min(V0, (n ** 2 * np.pi ** 2 * hbar ** 2) / (2 * m * a ** 2)) * 0.99
            #because we are still in "n" loop, this is segmenting the functions to find each solution instead of all of them at once

            # Check if solution exists in this bracket
            y_low, y_high = f(E_low), f(E_high)
            if np.sign(y_low) == np.sign(y_high):
                continue  # No root in this interval since signs are same (IVT)

            # Find root (intersections)
            else:
                E = root_scalar(f, bracket=[E_low, E_high]).root
                energies.append((E, parity, n))

        except:
            continue
        #I don't want to have to worry about the cases where there are no intersections (which will throw an error)
        #so I put a try/except block to just keep going past those scenarios
    return energies


def wavefunction(x, n, a, V0):
    energies = find_energy_levels(a, V0) #find energies first, that way if more states requested than actually exist, stop the calculations
    if n > len(energies):
        raise ValueError("Only {0} energy levels found, requested n={1}".format(len(energies), n))

    E, parity, n = energies[n - 1]  # n is 1-indexed

    k = np.sqrt(2 * m * E) / hbar  # wavenumber inside well
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar  # decay constant outside well

    psi = np.zeros_like(x) # Function to evaluate wavefunction at each point, start with 0s

    if parity == "even":
        # Even function: psi(x) = A*cos(k*x) inside, B*exp(-kappa*|x|) outside
        A = 1.0  # will normalize later

        mask_inside = np.abs(x) <= a / 2
        psi[mask_inside] = A * np.cos(k * x[mask_inside])

        # Outside well - match value at boundary to make function continuous
        B = A * np.cos(k * a / 2)
        mask_outside = np.abs(x) > a / 2
        psi[mask_outside] = B * np.exp(-kappa * (np.abs(x[mask_outside]) - a / 2))

    else:
        # Odd function: psi(x) = A*sin(k*x) inside, B*sgn(x)*exp(-kappa*|x|) outside
        #do same steps as even just with different equations
        A = 1.0

        mask_inside = np.abs(x) <= a / 2
        psi[mask_inside] = A * np.sin(k * x[mask_inside])

        B = A * np.sin(k * a / 2)
        mask_outside = np.abs(x) > a / 2
        psi[mask_outside] = B * np.sign(x[mask_outside]) * np.exp(-kappa * (np.abs(x[mask_outside]) - a / 2))

    # Normalize
    norm = np.sqrt(np.trapezoid(psi**2, x))
    psi /= norm

    return psi, E

def plot_solutions(n_values, a, V0):
    x_min = -a - a/2
    x_max = a + a/2
    x = np.linspace(x_min, x_max, 1000)

    potential = np.where(np.abs(x) <= a / 2, 0, V0)

    fig, axs = plt.subplots(len(n_values), 2, figsize=(12, 4 * len(n_values)))

    for i, n in enumerate(n_values):
        psi, E = wavefunction(x, n, a, V0)
        prob = psi ** 2

        if len(n_values) == 1:
            ax1 = axs[0]
            ax2 = axs[1]
        else:
            ax1 = axs[i, 0]
            ax2 = axs[i, 1]

        ax1.plot(x, E + psi, 'b-', label=r'$\psi(x)$')
        ax1.plot(x, [E] * len(x), linestyle='--', color = "m")
        ax1.plot(x, potential, 'k--', alpha=0.5, label='V(x)')

        ax1.set_title(f'Wavefunction of n = {n}')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('$\psi(x)$')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(x, E + prob, 'b-', label=r'$|\psi(x)|^2$')
        ax2.plot(x, [E] * len(x), linestyle='--', color = "m")
        ax2.fill_between(x, [E] * len(x), E + prob, alpha=0.3, color="green")
        ax2.plot(x, potential, 'k--', alpha=0.5, label='V(x)')

        ax2.set_title(f'Probability Density of n = {n}')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('$\psi(x)^2$')
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.show()

plot_solutions([1, 2], 2, 10)