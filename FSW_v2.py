import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
hbar = 1.0
m = 1.0
def find_energy_levels(a, V0):
    energies = []

    # Loop through possible values to find roots
    for n in range(1, 20):
        # First guess for energy (based on infinite well)
        E_guess = (n ** 2 * np.pi ** 2 * hbar ** 2) / (2 * m * a ** 2)
        if E_guess >= V0:
            continue  # Basically, there are no bound states for a well with these dimensions (too much energy to be bound)d

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

    E, parity, n = energies[n - 1]

    k = np.sqrt(2 * m * E) / hbar  # wavenumber inside well
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar  # decay constant outside well

    psi = np.zeros_like(x) # Function to evaluate wavefunction at each point, start with 0s

    if parity == "even":
        # Even function: psi(x) = A*cos(k*x) inside, B*exp(-kappa*|x|) outside
        A = 1 # This is just a leading coefficient, will normalize later - it literally doesn't matter what this is (except 0)

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
    norm = np.trapezoid(psi**2, x)
    psi /= np.sqrt(norm)

    return psi, E

def plot_schrodinger(n_values, a, V0):
    x_min = -a - a/2
    x_max = a + a/2
    x = np.linspace(x_min, x_max, 1000)

    potential = np.where(np.abs(x) <= a / 2, 0, V0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    energy_limit = []

    for i, n in enumerate(n_values):
        psi, E = wavefunction(x, n, a, V0)
        energy_limit.append(E)
        prob = psi ** 2
        
        ax[0].plot(x, E + psi, label=f'n = {n}')
        ax[0].plot(x, [E] * len(x), linestyle='--', color = "m")
        ax[0].plot(x, potential, 'k--', alpha=0.5)

        ax[1].plot(x, E + prob)
        ax[1].plot(x, [E] * len(x), linestyle='--', color="m")
        ax[1].fill_between(x, [E] * len(x), E + prob, alpha=0.3, color="green")
        ax[1].plot(x, potential, 'k--', alpha=0.5)

    ax[0].set_title(f'Wavefunction')
    ax[0].set_xlabel('Position')
    ax[0].set_ylabel(r'$\psi(x)$')
    ax[0].set_ylim(0, V0 + 0.2 * V0)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title(f'Probability Density')
    ax[1].set_xlabel('Position')
    ax[1].set_ylabel(r'$\psi(x)^2$')
    ax[1].set_ylim(0, V0 + 0.2 * V0)
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

plot_schrodinger([1, 2, 3, 4], 2, 20)