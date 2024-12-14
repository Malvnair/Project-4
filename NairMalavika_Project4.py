import numpy as np
import matplotlib.pyplot as plt

# Constants for the problem
h_bar = 1
mass = 0.5

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

# Code is modified from the schro.py code from the NM4P programs
def create_spatial_grid(nspace, length):
    """
    Creates a spatial grid.
    
    Parameters:
        nspace: Number of spatial grid points.
        length: Physical length of the spatial domain.
        
    Returns:
        x: Spatial grid points.
        h: Grid spacing.
    """
    h = length / (nspace - 1)
    x = np.linspace(-length / 2, length / 2, nspace)
    return x, h

# Code is modified from the schro.py code from the NM4P programs
def initialize_wave_packet(x, wparam, h):
    """
    Initializes the Gaussian wave packet with proper normalization.

    Parameters:
        x: Spatial grid points.
        wparam: List of parameters [sigma0, x0, k0].
        h: Grid spacing.

    Returns:
        psi0: Normalized initial wavefunction.
    """
    sigma0, x0, k0 = wparam
    
    # Construct the unnormalized Gaussian wave packet
    unnormalized_psi0 = np.exp(-(x - x0) ** 2 / (2 * sigma0 ** 2)) * np.exp(1j * k0 * x)

    # Normalize the wavefunction
    norm = 1.0 / np.sqrt(np.sum(np.abs(unnormalized_psi0) ** 2) * h)
    psi0 = norm * unnormalized_psi0
        
    return psi0


def create_potential(nspace, potential_indices):
    """
    Creates the potential energy array.
    
    Parameters:
        nspace: Number of spatial grid points.
        potential_indices: List of indices where potential is applied.
        
    Returns:
        V: Potential energy array.
    """
    
    # Initialize potential array
    V = np.zeros(nspace)
    for index in potential_indices:
        # Apply potential only in valid indices
        if 0 <= index < nspace:
            V[index] = 1.0
    return V

# Code is modified from the schro.py code from the NM4P programs
def create_hamiltonian(nspace, h, potential):
    """
    Creates the Hamiltonian matrix with periodic boundary conditions.
    
    Parameters:
        nspace: Number of spatial grid points.
        h: Grid spacing.
        potential: Potential energy array.
        
    Returns:
        H: Hamiltonian matrix.
    """
    coeff = -h_bar ** 2 / (2 * mass * h ** 2)

    ham = np.zeros((nspace, nspace), dtype=complex)
    
    # Fill tridiagonal entries
    for i in range(1, nspace - 1):
        ham[i, i - 1] = coeff
        ham[i, i] = -2 * coeff
        ham[i, i + 1] = coeff

    # Periodic boundary conditions
    ham[0, -1] = coeff;   ham[0, 0] = -2 * coeff;   ham[0, 1] = coeff
    ham[-1, -2] = coeff;  ham[-1, -1] = -2 * coeff; ham[-1, 0] = coeff

    # Add potential
    H = ham + np.diag(potential)
    # Normalize H
    eigenvalues_H, _ = np.linalg.eig(H)
    H_normalized = H / max(abs(eigenvalues_H))
    return H_normalized


def solve_schrodinger(psi0, H_normalized, nspace, ntime, tau, method, h):
    """
    Solves the Schrödinger equation using the specified method.
    
    Parameters:
        psi0: Initial wavefunction.
        H: Hamiltonian matrix.
        nspace: Number of spatial grid points.
        ntime: Number of time steps.
        tau: Time step size.
        method: Numerical method ('ftcs' or 'crank').
        
    Returns:
        psi_xt: Complex array representing the wavefunction over time.
        total_prob: Array of total probabilities at each timestep.
    """
    
    # Initialize wavefunction evolution and probability array
    psi_xt = np.zeros((nspace, ntime + 1), dtype=complex)
    total_prob = np.zeros(ntime + 1)
    
    # Set initial conditions
    psi_xt[:, 0] = psi0
    total_prob[0] = np.sum(np.abs(psi0) ** 2) * h  # Include h
    psi = psi0.copy()
    
    # Choose numerical method
    if method.lower() == 'ftcs':
        # Construct the evolution matrix
        M = np.eye(H_normalized.shape[0]) + (-1j * tau / h_bar) * H_normalized

        # Check for stability of the FTCS method
        radius = max_abs_eigenvalue(M)
        if radius - 1 > 1e-3:
            raise ValueError(f"FTCS scheme is unstable. Spectral radius: {radius:.2f}.")

        H_coeff = (-1j * tau / h_bar) * H_normalized
        
    elif method.lower() == 'crank':
        A = np.eye(nspace, dtype=complex) + 0.5j * tau * H_normalized
        B = np.eye(nspace, dtype=complex) - 0.5j * tau * H_normalized
        A_inv = np.linalg.inv(A)
    else:
        raise ValueError("Invalid method. Choose 'ftcs' or 'crank'.")
    
    # Time evolution loop
    for itime in range(1, ntime + 1):
        if method.lower() == 'ftcs':
            # Update wavefunction using FTCS method
            psi = psi + np.dot(H_coeff, psi)
        else:  # Crank-Nicholson
            # Solve matrix equation for next timestep
            rhs = np.dot(B, psi)
            psi = np.dot(A_inv, rhs)
        psi_xt[:, itime] = psi
        total_prob[itime] = np.sum(np.abs(psi) ** 2) * h 
    
    return psi_xt, total_prob


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    """
    Solves the time-dependent Schrödinger equation using specified parameters.

    Parameters:
        nspace: Number of spatial grid points.
        ntime: Number of time steps to be evolved.
        tau : Time step size.
        method: Numerical method for time evolution ('ftcs' or 'crank'). Default is 'ftcs'.
        length: Physical length of the spatial domain. Default is 200.
        potential: List of spatial indices where the potential is applied. Default is an empty list (no potential).
        wparam: Parameters for the initial wave packet [sigma0, x0, k0]. Default is [10, 0, 0.5].

    Returns:
        psi_xt: Complex 2D array representing the wavefunction at each time step.
        x: Array of spatial grid points.
        t: Array of time grid points.
        total_prob: 1D array representing the total probability at each timestep.
    """
    
    # Generate spatial grid, potential, Hamiltonian, and initial wavefunction
    x, h = create_spatial_grid(nspace, length)
    V = create_potential(nspace, potential)
    H_normalized = create_hamiltonian(nspace, h, V)
    psi0 = initialize_wave_packet(x, wparam, h)
    
    # Solve the Schrödinger equation and return results
    psi_xt, total_prob = solve_schrodinger(psi0, H_normalized, nspace, ntime, tau, method, h)
    t = np.linspace(0, ntime * tau, ntime + 1)
    return psi_xt, x, t, total_prob

# Code is modified from the schro.py code from the NM4P programs
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
        Displays the specified plot.
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
        plt.ylabel('Re[ψ(x)]')
    elif plot_type.lower() == 'prob':
        # Plot the probability density
        prob_density = np.abs(psi_xt[:, time_index]) ** 2
        plt.plot(x, prob_density, label='|ψ|²')
        plt.title(f'Probability Density at t={t[time_index]:.3f}')
        plt.ylabel('Probability density')
    else:
        raise ValueError("Invalid plot_type. Use 'psi' or 'prob'.")
    
    # Plot
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()
    plt.show()

# User Input for Method, Plot Type, and Time Index
method = input("Enter the numerical method ('ftcs' or 'crank'): ").strip().lower()
plot_type = input("Enter the plot type ('psi' for real part of the wavefunction or 'prob' for probability density): ").strip().lower()
# Extra
show_plot = input("Display total probability plot? (y/n): ").strip().lower()

# Simulation parameters
nspace = 2000
ntime = 300
tau = 0.0001
length = 200
potential = []
wparam = [10, 0, 0.5]
# tmax = ntime*tau
time = 0.03

# Solve the Schrödinger equation
psi_xt, x, t, total_prob = sch_eqn(nspace, ntime, tau, method, length=length, potential=potential, wparam=wparam)

# Plot the solution
schro_plot(x, t, psi_xt, plot_type=plot_type, time=time)

# Quick plot for total probability conservation
if show_plot == 'y':
    plt.figure()
    plt.plot(t, total_prob, label="Total Probability")
    plt.axhline(y=1.0, color='r', linestyle='--', label="Expected Value (1.0)")
    plt.title("Total Probability Conservation")
    plt.xlabel("Time (t)")
    plt.ylabel("Total Probability")
    plt.yticks([-1, 0, 1, 2])
    plt.grid(True)
    plt.legend()
    plt.show()
