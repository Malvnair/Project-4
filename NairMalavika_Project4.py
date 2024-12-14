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


# Constants for the problem
h_bar = 1
mass = 0.5


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
def initialize_wave_packet(x, wparam):
    """
    Initializes the Gaussian wave packet.
    
    Parameters:
        x: Spatial grid points.
        wparam: List of parameters [sigma0, x0, k0].
        
    Returns:
        psi0: Initial wavefunction.
    """
    sigma0, x0, k0 = wparam
    norm = 1.0 / (np.sqrt(sigma0 * np.sqrt(np.pi)))
    psi0 = norm * np.exp(-(x - x0) ** 2 / (2 * sigma0 ** 2)) * np.exp(1j * k0 * x)
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
    V = np.zeros(nspace)
    for index in potential_indices:
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
    return H



def solve_schrodinger(psi0, H, nspace, ntime, tau, method):
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
    psi_xt = np.zeros((nspace, ntime + 1), dtype=complex)
    total_prob = np.zeros(ntime + 1)
    psi_xt[:, 0] = psi0
    total_prob[0] = np.sum(np.abs(psi0) ** 2)
    psi = psi0.copy()
    
    if method == 'ftcs':
        H_coeff = (-1j * tau / h_bar) * H
    elif method == 'crank':
        A = np.eye(nspace, dtype=complex) + 0.5j * tau * H
        B = np.eye(nspace, dtype=complex) - 0.5j * tau * H
        A_inv = np.linalg.inv(A)
    else:
        raise ValueError("Invalid method. Choose 'ftcs' or 'crank'.")
    
    for itime in range(1, ntime + 1):
        if method == 'ftcs':
            psi = psi + np.dot(H_coeff, psi)
        else:  # Crank-Nicholson
            rhs = np.dot(B, psi)
            psi = np.dot(A_inv, rhs)
        psi_xt[:, itime] = psi
        total_prob[itime] = np.sum(np.abs(psi) ** 2)
    
    return psi_xt, total_prob




def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    """
    Solves the time-dependent Schrödinger equation using specified parameters.

    Parameters:
        nspace (int): Number of spatial grid points.
        ntime (int): Number of time steps to be evolved.
        tau (float): Time step size.
        method (str): Numerical method for time evolution ('ftcs' or 'crank'). Default is 'ftcs'.
        length (float): Physical length of the spatial domain. Default is 200.
        potential (list): List of spatial indices where the potential is applied. Default is an empty list (no potential).
        wparam (list): Parameters for the initial wave packet [sigma0, x0, k0]. Default is [10, 0, 0.5].

    Returns:
        psi_xt (ndarray): Complex 2D array representing the wavefunction at each time step.
        x (ndarray): Array of spatial grid points.
        t (ndarray): Array of time grid points.
        total_prob (ndarray): 1D array representing the total probability at each timestep.
    """
    x, h = create_spatial_grid(nspace, length)
    V = create_potential(nspace, potential)
    H = create_hamiltonian(nspace, h, V)
    psi0 = initialize_wave_packet(x, wparam)
    psi_xt, total_prob = solve_schrodinger(psi0, H, nspace, ntime, tau, method)
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
        plt.ylabel('Re[ψ(x)]')
    elif plot_type.lower() == 'prob':
        # Plot the probability density
        prob_density = np.abs(psi_xt[:, time_index]) ** 2
        plt.plot(x, prob_density, label='|ψ|²')
        plt.title(f'Probability Density at t={t[time_index]:.3f}')
        plt.ylabel('Probability density')
    else:
        raise ValueError("Invalid plot_type. Use 'psi' or 'prob'.")
    
    # Configure plot appearance
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()
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
psi_xt, x, t, _ = sch_eqn(nspace, ntime, tau, method, length=length, potential=potential, wparam=wparam)

# Plot the solution
schro_plot(x, t, psi_xt, plot_type=plot_type, time=time)
