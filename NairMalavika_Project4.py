import numpy as np
import matplotlib.pyplot as plt


def max_abs_eigenvalue(A):
    eigenvalues, _ = np.linalg.eig(A)
    return max(abs(eigenvalues))


def sch_eqn(nspace, ntime, tau, method, length=200, potential=[], wparam=[10, 0, 0.5]):
    h_bar = 1
    mass = 0.5
    
    h = length / (nspace - 1)
    x = np.linspace(-length / 2, length / 2, nspace)

    t = np.linspace(0, ntime * tau, ntime + 1)

    V = np.zeros(nspace)
    for potential_index in potential:
        if 0 <= potential_index < nspace:
            V[potential_index] = 1.0
            
    sigma0, x0, k0 = wparam
    Norm = 1.0 / (np.sqrt(sigma0 * np.sqrt(np.pi)))
    psi0 = Norm * np.exp(-(x - x0) ** 2 / (2 * sigma0 ** 2)) * np.exp(1j * k0 * x)

    ham = np.zeros((nspace, nspace))  
    coeff = -h_bar ** 2 / (2 * mass * h ** 2)

    for i in range(1, nspace - 1):
        ham[i, i - 1] = coeff
        ham[i, i] = -2 * coeff
        ham[i, i + 1] = coeff

    ham[0, -1] = coeff;   ham[0, 0] = -2 * coeff;   ham[0, 1] = coeff
    ham[-1, -2] = coeff;  ham[-1, -1] = -2 * coeff; ham[-1, 0] = coeff

    H = ham + np.diag(V)
    
    psi_xt = np.zeros((nspace, ntime + 1), dtype=complex)
    psi_xt[:, 0] = psi0
    psi = psi0.copy()

    if method.lower() == 'ftcs':
        # Construct the evolution matrix
        M = np.eye(H.shape[0]) + (-1j * tau / h_bar) * H

        # Perform spectral radius check
        radius = max_abs_eigenvalue(M)
        if radius > 1:
            print(f"FTCS scheme is unstable. Spectral radius: {radius:.2f}")
            return psi_xt, x, t

    elif method.lower() == 'crank':
        # Crank-Nicolson matrices
        A = np.eye(nspace, dtype=complex) + 0.5j * tau * H
        B = np.eye(nspace, dtype=complex) - 0.5j * tau * H
        A_inv = np.linalg.inv(A)
    else:
        raise ValueError("Method must be 'ftcs' or 'crank'.")

    # Time-stepping loop
    for itime in range(1, ntime + 1):
        if method.lower() == 'ftcs':    
            psi = psi + (-1j * tau) * np.dot(H, psi)
        else:
            rhs = np.dot(B, psi)    
            psi = np.dot(A_inv, rhs)  

        psi_xt[:, itime] = psi

    return psi_xt, x, t


def schro_plot(x, t, psi_xt, plot_type, time=None):
    if time is None:
        raise ValueError("You must specify a time for the plot.")
    
    time_index = np.abs(t - time).argmin()

    plt.figure()
    if plot_type.lower() == 'psi':
        plt.plot(x, np.real(psi_xt[:, time_index]), label='Real')
        plt.title(f'Real Part of Wavefunction at t={t[time_index]:.3f}')
        plt.ylabel('Re[ψ(x)]')
    elif plot_type.lower() == 'prob':
        prob_density = np.abs(psi_xt[:, time_index]) ** 2
        plt.plot(x, prob_density, label='|ψ|²')
        plt.title(f'Probability Density at t={t[time_index]:.3f}')
        plt.ylabel('Probability density')
    else:
        raise ValueError("Invalid plot_type. Use 'psi' or 'prob'.")
    
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()
    plt.show()


# User Input for Method, Plot Type, and Time Index
method = input("Enter the numerical method ('ftcs' or 'crank'): ").strip().lower()
plot_type = input("Enter the plot type ('psi' for real part of the wavefunction or 'prob' for probability density): ").strip().lower()

nspace = 2000
ntime = 300
tau = 0.001
length = 200
potential = []
wparam = [10, 0, 0.5]
time = 0.3

psi_xt, x, t = sch_eqn(nspace, ntime, tau, method, length=length, potential=potential, wparam=wparam)

schro_plot(x, t, psi_xt, plot_type=plot_type, time=time)
