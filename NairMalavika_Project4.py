import numpy as np
import matplotlib.pyplot as plt


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    h_bar = 1.0
    mass = 0.5  
    
    
    N = int(input('Enter number of grid points: '))
    h = length/(nspace-1)
    x = np.arange(N)*h - length/2. 
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
