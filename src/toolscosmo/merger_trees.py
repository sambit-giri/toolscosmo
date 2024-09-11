import numpy as np
import matplotlib.pyplot as plt
from .scipy_func import *
from .cosmo import variance, rbin_to_mbin, growth_factor

# Constants and parameters (to be calibrated based on simulation data)
A = 0.7  # Fitting parameter A (example value, should be calibrated)
B = 0.4  # Fitting parameter B (example value, should be calibrated)

def sigma_squared(M, param):
    '''
    Function to calculate the density variance (sigma^2(M)).
    '''
    rbin, var, dlnvardlnr = variance(param)
    mbin = rbin_to_mbin(rbin, param)
    tck = splrep(mbin, var)
    return splev(M, tck)

def mass_derivative_of_sigma_squared(M, param):
    '''
    Function to calculate the mass derivative of density variance.
    '''
    Mmin = param.code.Mmin
    Mmax = param.code.Mmax 
    NM = param.code.NM
    Mbin = 10**np.linspace(np.log10(Mmin),np.log10(Mmax),NM)
    Sbin = sigma_squared(Mbin, param)
    dSbin_dMbin = Sbin*np.gradient(np.log(Sbin))/(Mbin*np.gradient(np.log(Mbin)))
    tck = splrep(Mbin, dSbin_dMbin)
    return splev(M, tck)

def delcrit_SC(z, param):
    assert param.code.zmin<=z<=param.code.zmax, f"Provided redshift z={z} is outside param range. Update param.code.zmin and param.code.zmax values."
    zmin = param.code.zmin if param.code.zmin>0 else 0.001
    zmax = param.code.zmax
    zbin = 10**np.linspace(np.log10(zmin),np.log10(zmax),param.code.Nz)
    delc = 1.686/growth_factor(zbin, param)
    return splev(z, splrep(zbin,delc))

def progenitor_conditional_probability_function(M1, z1, M0, z0, param, collapse_model='SC'):
    '''
    Function to compute the conditional probability function for progenitors (Eq. 1 in Jiang & van den Bosch 2014).
    '''
    dX = lambda X1,X2: (X1-X2)
    if collapse_model.upper() in ['SC', 'SPHERICAL COLLAPSE', 'SPHERICAL_COLLAPSE']:
        fcoll = lambda S1,w1,S0,w0: 1/(2*np.pi)*dX(w1,w0)/dX(S1,S0)**(1.5)*np.exp(-dX(w1,w0)**2/(2*dX(S1,S0)))
    elif collapse_model.upper() in ['EC', 'ELLIPSOIDAL COLLAPSE', 'ELLIPSOIDAL_COLLAPSE']:
        nu = lambda S,w: w**2/S
        A0 = lambda S0,w0: 0.8661*(1-0.133*nu(S0,w0)**(-0.615))
        A1 = lambda S0,w0: 0.308*nu(S0,w0)**(-0.115)
        A2 = lambda S0,w0: 0.0373*nu(S0,w0)**(-0.115)
        A3 = lambda S1,w1,S0,w0: A0(S0,w0)**2+2*A0(S0,w0)*A1(S0,w0)*np.sqrt(dX(S1,S0)*Stilde(S1,S0))/dX(w1,w0)
        Stilde = lambda S1,S0: dX(S1,S0)/S0
        fcoll = lambda S1,w1,S0,w0: A0(S0,w0)/(2*np.pi)*dX(w1,w0)/dX(S1,S0)**(1.5)*np.exp(-A1(S0,w0)**2/2*Stilde(S1,S0))\
                                    *(np.exp(-A3(S1,w1,S0,w0)*dX(w1,w0)**2/2/dX(S1,S0))+A2(S0,w0)*Stilde(S1,S0)**1.5*(1+2*A1(S0,w0)*np.sqrt(Stilde(S1,S0)/np.pi)))
    else:
        assert False, f"{collapse_model} is not implemented in progenitor_conditional_probability_function module."
    Mmin = param.code.Mmin
    Mmax = param.code.Mmax 
    NM = param.code.NM
    Mbin = 10**np.linspace(np.log10(Mmin),np.log10(Mmax),NM)
    dSdM_abs = np.abs(mass_derivative_of_sigma_squared(Mbin,param))
    dS1dM1_abs = splev(M1, splrep(Mbin,dSdM_abs))
    S1 = sigma_squared(M1, param) 
    S0 = sigma_squared(M0, param)
    w1 = delcrit_SC(z1, param)
    w0 = delcrit_SC(z0, param)
    p_CPF = fcoll(S1,w1,S0,w0)*dS1dM1_abs
    return p_CPF

def progenitor_mass_function(M1, z1, M0, z0, param, collapse_model='SC'):
    '''
    Function to compute the mass function for progenitors (Eq. 3 in Jiang & van den Bosch 2014).
    '''
    p_CPF = progenitor_conditional_probability_function(M1, z1, M0, z0, param, collapse_model=collapse_model)
    n_EPS = (M0/M1)*p_CPF
    return n_EPS


if __name__ == '__main__':
    # Example usage:
    
    import toolscosmo
    from toolscosmo import merger_trees

    param = toolscosmo.par('cpl')
    param.code.zmin = 0.0
    M0 = 1e13
    z0 = 0
    w0 = merger_trees.delcrit_SC(z0, param)
    dw = 0.002
    w1 = w0+dw 
    Mratio_bin = np.arange(0.1,1,0.01)
    M1s = Mratio_bin*M0 
    n_EPS = merger_trees.progenitor_mass_function(M1s,w1,M0,w0,param)



    M_final = 1e12  # Final halo mass in solar masses
    z_final = 0     # Final redshift
    delta_z = 0.1   # Redshift step
    min_mass = 1e8  # Minimum halo mass to consider

    # Build the merger tree and track the mass history
    merger_history = build_merger_tree(M_final, z_final, delta_z, min_mass, param)

    # Convert the history to arrays for plotting
    redshifts, masses = zip(*merger_history)

    # Plot the merger history M(z)
    plt.figure(figsize=(10, 6))
    plt.plot(redshifts, masses, marker='o', linestyle='-', color='blue')
    plt.yscale('log')  # Masses are often plotted on a log scale
    plt.gca().invert_xaxis()  # Redshift typically decreases with time, so invert x-axis
    plt.xlabel('Redshift (z)')
    plt.ylabel('Halo Mass (M_sun)')
    plt.title('Merger History M(z)')
    plt.grid(True)
    plt.show()
