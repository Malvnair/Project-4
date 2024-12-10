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
    
    psi_xt = np.zeros((nspace, ntime+1), dtype=complex)
    psi_xt[:,0] = psi0
    psi = psi0.copy()

    if method.lower() == 'ftcs':
        # Compute eigenvalues for stability check
        eigvals = np.linalg.eigvals(H)
        # Check stability condition: |1 - iτλ| ≤ 1 for all λ
        stable = True
        for eigenvalue in eigvals:
            U = 1.0 - 1j*tau*eigenvalue
            if np.abs(U) > 1.0:
                stable = False
                break
        if not stable:
            print("FTCS scheme unstable")
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
