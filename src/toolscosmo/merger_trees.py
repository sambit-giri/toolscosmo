import numpy as np
import matplotlib.pyplot as plt
from .scipy_func import *
from .cosmo import variance, rbin_to_mbin

# Constants and parameters (to be calibrated based on simulation data)
A = 0.7  # Fitting parameter A (example value, should be calibrated)
B = 0.4  # Fitting parameter B (example value, should be calibrated)

# Function to calculate the variance of the density field, sigma^2(M)
def sigma_squared(M, param):
    rbin, var, dlnvardlnr = variance(param)
    mbin = rbin_to_mbin(rbin, param)
    tck = splrep(mbin, var)
    return splev(M, tck)

# Function to compute the mass function f(S, S_f, ΔS) for progenitors
def progenitor_mass_function(S, S_f, delta_S):
    return A * (delta_S / S_f) * np.exp(-B * (delta_S / S_f))

# Function to draw progenitor masses based on the mass function
def draw_progenitors(M_f, z_f, delta_z, num_progenitors=100, param=None):
    S_f = sigma_squared(M_f, param)
    progenitor_masses = []
    
    for _ in range(num_progenitors):
        delta_S = np.random.uniform(0, S_f)
        S = S_f - delta_S
        M = np.power(sigma_squared(S, param), -2.0)
        probability = progenitor_mass_function(S, S_f, delta_S)
        if np.random.rand() < probability:
            progenitor_masses.append(M)
    
    total_mass = np.sum(progenitor_masses)
    progenitor_masses = [M * M_f / total_mass for M in progenitor_masses]
    
    return progenitor_masses

# Recursive function to build the merger tree and track M(z)
def build_merger_tree(M_f, z_f, delta_z, min_mass=1e8, param=None):
    history = [(z_f, M_f)]
    
    if M_f < min_mass:
        return history
    
    progenitors = draw_progenitors(M_f, z_f, delta_z, param=param)
    
    for M_p in progenitors:
        history.extend(build_merger_tree(M_p, z_f + delta_z, delta_z, min_mass, param))
    
    return history

# Example usage:
if __name__ == '__main__':
    import toolscosmo

    M_final = 1e12  # Final halo mass in solar masses
    z_final = 0     # Final redshift
    delta_z = 0.1   # Redshift step
    min_mass = 1e8  # Minimum halo mass to consider
    param = toolscosmo.par('cpl')

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
