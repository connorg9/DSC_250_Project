import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root_scalar

def V(x, V0, a, w):
    if -a/2 < x < a/2 or x > a/2 + w:
        return 0 #inside well
    elif x < -a/2:
        return V0
    elif a/2 < x < a/2 + w:
        return 0 #outside well

def fk(E, V0, a, n):
    if E <= 0 or E >= V0:
        return 1e6  # Arbitrary large error b/c otherwise get error
    k1 = np.sqrt(2 * E)
    k2 = np.sqrt(2 * (V0 - E))
    if n % 2:  # Odd n = even parity
        return k2 - k1 * np.tan(k1 * a / 2)
    else:  # Even n = odd parity
        return k2 + k1 / np.tan(k1 * a / 2)

def find_energy(V0, a):
    n = 500
    Ei = np.linspace(0.0, V0, n)
    roots = []
    for n in [1, 2]:  # Check both even and odd parity solutions - used in fk function
        for i in range(len(Ei - 1)):
            try:
                solution = root_scalar(fk, args=(V0, a, n), x0=Ei[i - 1], x1=Ei[i])
                if solution.converged and np.around(solution.root, 9) not in roots:
                    roots.append(np.around(solution.root, 9))
            except ValueError: # try/except loop to make sure things don't go wrong
                continue

    return np.sort(roots)

def schrodinger(r, x, V, E):
    hbar = 1
    m = 1
    psi, phi = r

    dpsi = phi
    dphi = (2 * m / hbar ** 2) * (V(x) - E) * psi

    return [dpsi, dphi]

def solver(E, V0, a, w):

    resolution = 500
    x_min = -2
    x_max = 5
    x_vals = np.linspace(x_min, x_max, resolution)

    initial = [0.001, 0.001]

    solution = odeint(schrodinger, initial, x_vals, args=(lambda x: V(x, V0, a, w), E)) #need lambda in args since V iterates on x - using x_vals results in "ambiguous truth value"
    psi_vals = solution[:, 0]

    well_region = np.abs(x_vals) <= a / 2 + 1
    norm = np.trapezoid(psi_vals[well_region] ** 2, x_vals[well_region]) #similar to ISW, but need to include some area outside well area to account for decay
    psi_vals /= np.sqrt(norm)

    return x_vals, psi_vals, E

def plot_schrodinger(n_values, V0, a, w):

    eigenvalues = find_energy(V0, a) #valid eigenvalues for well with given size

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    energy_limit = []  # Used for graph y-lim later

    for i, n in enumerate(n_values):
        if i < len(eigenvalues):
            E = eigenvalues[i]
            energy_limit.append(E) #need max evergy to make sure everything is graphed (setting y-lim)

            x_vals, psi_vals, energy = solver(E, V0, a, w)

            ax[0].plot(x_vals, energy + psi_vals, label="n = {0}".format(n))
            ax[0].axhline(E, linestyle='--', color=f'C{i}')

            prob = psi_vals ** 2
            ax[1].plot(x_vals, energy + prob, label="n = {0}".format(n))
            ax[1].fill_between(x_vals, energy, energy + prob, color="green", alpha=0.3)
            ax[1].axhline(E, linestyle='--', color=f'C{i}')

        else:
            print("Too many states requested, given inputs only has {0} state(s)".format(len(energy_limit)))
            break

    x_left = np.linspace(min(x_vals), -a / 2, 100)
    x_right = np.linspace(a / 2, a/2 + w, 100)

    ax[0].set_title("Wavefunctions")
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel(r"$\psi(x)$ + Energy")
    ax[0].set_ylim(0, max(energy_limit) + 0.25 * max(energy_limit))
    ax[0].fill_between(x_left, 0, V0, color='gray', alpha=0.3)
    ax[0].fill_between(x_right, 0, V0, color='gray', alpha=0.3)

    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title("Probability Densities")
    ax[1].set_xlabel("Position")
    ax[1].set_ylabel("Probability Density + Energy")
    ax[1].set_ylim(0, max(energy_limit) + 0.25 * max(energy_limit))
    ax[1].fill_between(x_left, 0, V0, color='gray', alpha=0.3)
    ax[1].fill_between(x_right, 0, V0, color='gray', alpha=0.3)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

V0 = 50
a = 1
plot_schrodinger([1, 2, 3, 4], V0, a, .25) #making w skinnier lets all states escape, not just high energies

#double square well, double harmonic, morse potential,


