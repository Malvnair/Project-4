import numpy as np
import matplotlib.pyplot as plt


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    h_bar = 1.0
    mass = 0.5  
    
    
    h = length / (nspace - 1)
    x = np.linspace(-length / 2, length / 2, nspace)

    t = np.linspace(0, ntime*tau, ntime+1)

    V = np.zeros(nspace)
    for potential_index in potential:
        if 0 <= potential_index < nspace:
            V[potential_index] = 1.0
            
            
    sigma0, x0, k0 = wparam
    Norm = 1.0/(np.sqrt(sigma0*np.sqrt(np.pi)))
    psi0 = Norm * np.exp(-(x - x0)**2/(2*sigma0**2)) * np.exp(1j*k0*x)

    ham = np.zeros((nspace, nspace), dtype=complex)  
    coeff = -1.0/h**2  

    for i in range(1, nspace-1):
        ham[i, i-1] = coeff
        ham[i, i]   = -2*coeff
        ham[i, i+1] = coeff

    ham[0, -1] = coeff;   ham[0, 0] = -2*coeff;   ham[0, 1] = coeff
    ham[-1, -2] = coeff;  ham[-1, -1] = -2*coeff; ham[-1, 0] = coeff

    H = ham + np.diag(V)
    
    psi_xt = np.zeros((nspace, ntime+1), dtype=complex)
    psi_xt[:,0] = psi0
    psi = psi0.copy()

    if method.lower() == 'ftcs':
            if tau >= h**2:
                print("FTCS scheme unstable.")
                prob = np.array([np.sum(np.abs(psi0)**2)])
                return psi_xt, x, t, prob
    elif method.lower() == 'crank':
        # Crank-Nicolson matrices
        A = np.eye(nspace, dtype=complex) + 0.5j*tau*H
        B = np.eye(nspace, dtype=complex) - 0.5j*tau*H
        A_inv = np.linalg.inv(A)
    else:
        raise ValueError("Method must be 'ftcs' or 'crank'.")

    prob = np.zeros(ntime+1)
    prob[0] = np.sum(np.abs(psi)**2)

    # Time-stepping loop
    for itime in range(1, ntime+1):
        if method.lower() == 'ftcs':    
            psi = psi + (-1j * tau) * np.dot(H, psi)
        else:
            rhs = np.dot(B, psi)    
            psi = np.dot(A_inv, rhs)  

        psi_xt[:, itime] = psi

        prob[itime] = np.sum(np.abs(psi)**2)

    return psi_xt, x, t, prob

def schro_plot(x, t, psi_xt, plot_type='psi', time_index=0):
    plt.figure()

    if plot_type.lower() == 'psi':
        plt.plot(x, np.real(psi_xt[:, time_index]), label='Real')
        plt.plot(x, np.imag(psi_xt[:, time_index]), '--', label='Imag')
        plt.title(f'Wavefunction at t={t[time_index]:.3f}')
        plt.ylabel('Wavefunction')
    elif plot_type.lower() == 'prob':
        plt.plot(x, np.abs(psi_xt[:, time_index])**2, label='|ψ|²')
        plt.title(f'Probability Density at t={t[time_index]:.3f}')
        plt.ylabel('Probability density')

    plt.xlabel('x')
    plt.grid(True)
    plt.legend()

    plt.savefig("NairMalavika_Project4_Fig1")

    plt.show()

nspace = 400
ntime = 1000
tau = 0.1

psi_xt_ftcs, x_ftcs, t_ftcs, prob_ftcs = sch_eqn(nspace, ntime, tau, method='ftcs')
psi_xt_crank, x_crank, t_crank, prob_crank = sch_eqn(nspace, ntime, tau, method='crank')
schro_plot(x_ftcs, t_ftcs, psi_xt_ftcs, plot_type='psi', time_index=-1)

schro_plot(x_ftcs, t_ftcs, psi_xt_ftcs, plot_type='prob', time_index=-1)

schro_plot(x_crank, t_crank, psi_xt_crank, plot_type='psi', time_index=-1)

schro_plot(x_crank, t_crank, psi_xt_crank, plot_type='prob', time_index=-1)
