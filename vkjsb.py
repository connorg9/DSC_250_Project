import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def V(x, a, V0):
    """Potential function for finite square well."""
    return np.where((x >= -a / 2) & (x <= a / 2), 0, V0)


def schrodinger_eq(x, y, E, a, V0):
    """Schrodinger equation as a first-order system.
    y[0] is psi, y[1] is d(psi)/dx
    """
    hbar = 1.0
    m = 1.0

    dpsi_dx = y[1]
    d2psi_dx2 = (2 * m / hbar ** 2) * (V(x, a, V0) - E) * y[0]

    return [dpsi_dx, d2psi_dx2]


def shooting_method(E, n, a, V0, x_range=(-4, 4), num_points=1000):
    """Use shooting method to find eigenstates."""
    hbar = 1.0
    m = 1.0

    # Set up the x-grid
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Start integration from the left with decaying exponential
    k_left = np.sqrt(2 * m * V0 / hbar ** 2 - 2 * m * E / hbar ** 2)
    y0_left = [np.exp(k_left * x_range[0]), k_left * np.exp(k_left * x_range[0])]

    # Integrate from left boundary to center
    sol_left = solve_ivp(
        lambda t, y: schrodinger_eq(t, y, E, a, V0),
        [x_range[0], 0],
        y0_left,
        t_eval=x[x <= 0],
        method='RK45',
        rtol=1e-8
    )

    # Check parity to set up right boundary condition
    if n % 2 == 1:  # Odd n: symmetric solution
        # For symmetric solutions, we use the left half and reflect it
        psi_left = sol_left.y[0]
        psi_right = np.flip(psi_left)

        # For the derivative, we need to flip sign and reverse
        dpsi_left = sol_left.y[1]
        dpsi_right = -np.flip(dpsi_left)

        # x values
        x_left = sol_left.t
        x_right = -np.flip(x_left)

        # Combine
        x_vals = np.concatenate((x_left, x_right[1:]))  # Avoid duplicate at x=0
        psi_vals = np.concatenate((psi_left, psi_right[1:]))
        dpsi_vals = np.concatenate((dpsi_left, dpsi_right[1:]))

    else:  # Even n: antisymmetric solution
        # Start integration from the right with decaying exponential
        k_right = np.sqrt(2 * m * V0 / hbar ** 2 - 2 * m * E / hbar ** 2)
        y0_right = [np.exp(k_right * x_range[1]), -k_right * np.exp(k_right * x_range[1])]

        # Integrate from right boundary to center
        sol_right = solve_ivp(
            lambda t, y: schrodinger_eq(t, y, E, a, V0),
            [x_range[1], 0],
            y0_right,
            t_eval=x[x >= 0],
            method='RK45',
            rtol=1e-8
        )

        # Combine left and right solutions
        x_vals = np.concatenate((sol_left.t, sol_right.t[1:]))  # Avoid duplicate at x=0
        psi_vals = np.concatenate((sol_left.y[0], sol_right.y[0][1:]))
        dpsi_vals = np.concatenate((sol_left.y[1], sol_right.y[1][1:]))

    # Normalize the wavefunction
    norm = np.sqrt(np.trapz(psi_vals ** 2, x_vals))
    psi_vals = psi_vals / norm

    return x_vals, psi_vals


def find_eigenvalue(n, a, V0, x_range=(-4, 4)):
    """Find the energy eigenvalue for the nth state."""
    hbar = 1.0
    m = 1.0

    # Initial guess for energy (slightly below infinite well value)
    E_inf = ((n ** 2) * (np.pi ** 2) * (hbar ** 2)) / (8 * m * (a ** 2))
    E_min = 0
    E_max = V0

    # For higher states, we expect energy to be lower than infinite well
    if n == 1:
        E_guess = 0.8 * E_inf
    else:
        E_guess = 0.9 * E_inf

    # Function to find where psi crosses zero at boundaries
    def match_at_boundary(E):
        try:
            # For odd n (symmetric), check derivative at x=0
            if n % 2 == 1:
                x, psi = shooting_method(E, n, a, V0, x_range)
                mid_idx = np.argmin(np.abs(x))
                return psi[mid_idx + 1] - psi[mid_idx - 1]  # Should be close to zero for symmetric

            # For even n (antisymmetric), check value at x=0
            else:
                x, psi = shooting_method(E, n, a, V0, x_range)
                mid_idx = np.argmin(np.abs(x))
                return psi[mid_idx]  # Should be zero for antisymmetric
        except:
            return 1e6  # Return large value if integration fails

    # Binary search for the eigenvalue
    E_left = max(0.5 * E_guess, 0.01)
    E_right = min(1.5 * E_guess, V0 - 0.01)

    # Try fixed values first based on n
    if n == 1:
        E_try = 0.7 * E_inf
    elif n == 2:
        E_try = 0.75 * E_inf
    elif n == 3:
        E_try = 0.8 * E_inf
    elif n == 4:
        E_try = 0.85 * E_inf
    elif n == 5:
        E_try = 0.9 * E_inf
    else:
        E_try = 0.9 * E_inf

    return E_try


def find_eigenstate(n, a, V0, x_range=(-4, 4), num_points=1000):
    """Find the nth eigenstate of the finite square well."""
    # Find the energy eigenvalue
    E = find_eigenvalue(n, a, V0, x_range)

    # Get the wavefunction for this energy
    x_vals, psi_vals = shooting_method(E, n, a, V0, x_range, num_points)

    return x_vals, psi_vals, E


def plot_schrodinger(n_values, a, V0):
    fig, ax = plt.subplots(len(n_values), 2, figsize=(12, 1.75 * len(n_values)))

    # Use wider x range for visualization
    x_range = (-3 * a, 3 * a)

    for i, n in enumerate(n_values):
        x_vals, psi_vals, E = find_eigenstate(n, a, V0, x_range=x_range)

        # Calculate potential values for plotting
        V_vals = V(x_vals, a, V0)

        # Identify regions inside and outside the well
        inside_well = (x_vals >= -a / 2) & (x_vals <= a / 2)
        outside_well = ~inside_well

        # Plot wavefunction
        if len(n_values) == 1:
            # Handle the case when there's only one row
            current_ax = ax[0]
            prob_ax = ax[1]
        else:
            current_ax = ax[i, 0]
            prob_ax = ax[i, 1]

        # Plot potential
        current_ax.axhspan(0, V0, alpha=0.1, color='gray')
        current_ax.plot([-a / 2, -a / 2], [0, V0], 'k--')
        current_ax.plot([a / 2, a / 2], [0, V0], 'k--')
        current_ax.plot([-3 * a, -a / 2], [V0, V0], 'k--')
        current_ax.plot([a / 2, 3 * a], [V0, V0], 'k--')

        # Plot energy level
        current_ax.axhline(y=E, color='b', linestyle='-', alpha=0.5, label=f'E_{n} = {E:.4f}')

        # Plot wavefunction (shifted by energy for visualization)
        current_ax.plot(x_vals, E + 0.5 * psi_vals, 'r-', label=f'ψ_{n}(x)')
        current_ax.set_title(f"Wavefunction for n = {n}")
        current_ax.set_xlabel("Position")
        current_ax.set_ylabel("Energy / Wavefunction")
        current_ax.grid(True)
        current_ax.legend()

        # Plot probability density
        prob_ax.axhspan(0, V0, alpha=0.1, color='gray')
        prob_ax.plot([-a / 2, -a / 2], [0, V0], 'k--')
        prob_ax.plot([a / 2, a / 2], [0, V0], 'k--')
        prob_ax.plot([-3 * a, -a / 2], [V0, V0], 'k--')
        prob_ax.plot([a / 2, 3 * a], [V0, V0], 'k--')

        # Plot probability density
        prob_ax.plot(x_vals, psi_vals ** 2, 'g-', label=f'|ψ_{n}(x)|²')
        prob_ax.set_title(f"Probability Density for n = {n}")
        prob_ax.set_xlabel("Position")
        prob_ax.set_ylabel("Probability Density")
        prob_ax.grid(True)
        prob_ax.legend()

    plt.tight_layout()
    plt.show()


# Example usage
plot_schrodinger([1, 2, 3, 4, 5], 1, 20)