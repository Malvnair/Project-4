import numpy as np
import matplotlib.pyplot as plt

# Code modified from NairMalavika_SamoylovaAlona_Lab10.py
def max_abs_eigenvalue(A):
    """
    Computes the maximum absolute eigenvalue of a given matrix.
    
    Parameters:
        A: A square numpy array representing the matrix.
    
    Returns:
        The maximum absolute eigenvalue of the matrix.
    """
    eigenvalues, _ = np.linalg.eig(A)
    return max(abs(eigenvalues))


def sch_eqn(nspace, ntime, tau, method, length=200, potential=[], wparam=[10, 0, 0.5]):
    """
    Solves the time-dependent Schrödinger equation for a given spatial and temporal discretization.

    Parameters:
        nspace: Number of spatial grid points.
        ntime: Number of time steps.
        tau: Time step size.
        method: Numerical method to use ('ftcs' or 'crank').
        length: Physical length of the spatial domain (default: 200).
        potential: List of spatial indices where the potential is applied (default: []).
        wparam: List of parameters for the initial wave packet [sigma0, x0, k0] (default: [10, 0, 0.5]).

    Returns:
        psi_xt: Complex numpy array representing the wavefunction at each time step.
        x: Spatial grid points.
        t: Time grid points.
        total_probability: 1D array of total probability at each time step.
    """
    # Define constants for the problem
    h_bar = 1
    mass = 0.5

    # Discretize spatial domain and create spatial grid
    h = length / (nspace - 1)
    x = np.linspace(-length / 2, length / 2, nspace)

    # Discretize temporal domain
    t = np.linspace(0, ntime * tau, ntime + 1)

    # Define potential energy profile
    V = np.zeros(nspace)
    for potential_index in potential:
        if 0 <= potential_index < nspace:
            V[potential_index] = 1.0
    
    # Define initial wave packet
    sigma0, x0, k0 = wparam
    Norm = 1.0 / (np.sqrt(sigma0 * np.sqrt(np.pi)))
    psi0 = Norm * np.exp(-(x - x0) ** 2 / (2 * sigma0 ** 2)) * np.exp(1j * k0 * x)

    # Create the Hamiltonian matrix
    ham = np.zeros((nspace, nspace))  
    coeff = -h_bar ** 2 / (2 * mass * h ** 2)

    # Fill the Hamiltonian matrix with tridiagonal entries
    for i in range(1, nspace - 1):
        ham[i, i - 1] = coeff
        ham[i, i] = -2 * coeff
        ham[i, i + 1] = coeff

    # Apply periodic boundary conditions
    ham[0, -1] = coeff;   ham[0, 0] = -2 * coeff;   ham[0, 1] = coeff
    ham[-1, -2] = coeff;  ham[-1, -1] = -2 * coeff; ham[-1, 0] = coeff

    # Add potential to the Hamiltonian
    H = ham + np.diag(V)
    
    # Normalize H
    eigenvalues_H, _ = np.linalg.eig(H)
    H_normalized = H / max(abs(eigenvalues_H))

    # Initialize wavefunction storage
    psi_xt = np.zeros((nspace, ntime + 1), dtype=complex)
    psi_xt[:, 0] = psi0
    psi = psi0.copy()

    # Array to track total probability
    total_probability = np.zeros(ntime + 1)
    total_probability[0] = np.sum(np.abs(psi0) ** 2)

    # Choose numerical method
    if method.lower() == 'ftcs':
        # Construct the evolution matrix
        M = np.eye(H.shape[0]) + (-1j * tau / h_bar) * H_normalized

        # Check for stability of the FTCS method
        radius = max_abs_eigenvalue(M)
        if radius - 1 > 1e-8:
            print(f"FTCS scheme is unstable. Spectral radius: {radius:.2f}")
            return psi_xt, x, t, total_probability

    elif method.lower() == 'crank':
        # Construct Crank-Nicolson matrices
        A = np.eye(nspace, dtype=complex) + 0.5j * tau * H
        B = np.eye(nspace, dtype=complex) - 0.5j * tau * H
        A_inv = np.linalg.inv(A)
    else:
        raise ValueError("Method must be 'ftcs' or 'crank'.")

    # Perform time-stepping
    for itime in range(1, ntime + 1):
        if method.lower() == 'ftcs':    
            psi = psi + (-1j * tau) * np.dot(H, psi)
        else:
            rhs = np.dot(B, psi)    
            psi = np.dot(A_inv, rhs)  

        psi_xt[:, itime] = psi
        total_probability[itime] = np.sum(np.abs(psi) ** 2)

    return psi_xt, x, t, total_probability


def schro_plot(x, t, psi_xt, plot_type, time=None):
    """
    Visualizes the solution of the Schrödinger equation at a specified time.

    Parameters:
        x: Array of spatial grid points.
        t: Array of time grid points.
        psi_xt: Array containing the wavefunction at all time steps.
        plot_type: Type of plot ('psi' for real part or 'prob' for probability density).
        time: The specific time at which to plot the solution.

    Returns:
        None. Displays the specified plot.
    """
    if time is None:
        raise ValueError("You must specify a time for the plot.")
    
    # Determine the closest time index
    time_index = np.abs(t - time).argmin()

    plt.figure()
    if plot_type.lower() == 'psi':
        # Plot the real part of the wavefunction
        plt.plot(x, np.real(psi_xt[:, time_index]), label='Real')
        plt.title(f'Real Part of Wavefunction at t={t[time_index]:.3f}')
        plt.ylabel('Re[\u03c8(x)]')
    elif plot_type.lower() == 'prob':
        # Plot the probability density
        prob_density = np.abs(psi_xt[:, time_index]) ** 2
        plt.plot(x, prob_density, label='|\u03c8|\u00b2')
        plt.title(f'Probability Density at t={t[time_index]:.3f}')
        plt.ylabel('Probability density')
    else:
        raise ValueError("Invalid plot_type. Use 'psi' or 'prob'.")
    
    # Configure plot appearance
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()
    plt.show()

def total_probability_plot(t, total_probability):
    """
    Plots the total probability as a function of time.

    Parameters:
        t: Array of time grid points.
        total_probability: Array of total probability at each time step.

    Returns:
        None. Displays the specified plot.
    """
    plt.figure()
    plt.plot(t, total_probability, label='Total Probability')
    plt.xlabel('Time')
    plt.ylabel('Total Probability')
    plt.title('Total Probability vs. Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# User Input for Method, Plot Type, and Time Index
method = input("Enter the numerical method ('ftcs' or 'crank'): ").strip().lower()
plot_type = input("Enter the plot type ('psi' for real part of the wavefunction or 'prob' for probability density): ").strip().lower()

# Simulation parameters
nspace = 2000
ntime = 300
tau = 0.0001
length = 200
potential = []
wparam = [10, 0, 0.5]
time = 0.3

# Solve the Schrödinger equation
psi_xt, x, t, total_probability = sch_eqn(nspace, ntime, tau, method, length=length, potential=potential, wparam=wparam)

# Plot the solution
schro_plot(x, t, psi_xt, plot_type=plot_type, time=time)

# Plot the total probability
total_probability_plot(t, total_probability)