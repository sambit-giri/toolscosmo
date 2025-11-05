import numpy as np 
import matplotlib.pyplot as plt 
import toolscosmo
import scipy
from scipy.interpolate import splrep,splev
import pickle, os
from copy import deepcopy
import pyhmcode
from pyhmcode.hmcode import HMcode2020

lstyles = ['-', '--', '-.', ':']

## set parameters
par = toolscosmo.par()
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100
par.code.verbose = True

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.As = 2.126e-09 #par.cosmo.s8 = 0.83 
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72

par1 = deepcopy(par)
par2 = deepcopy(par)
par3 = deepcopy(par)
par4 = deepcopy(par)
par1.file.ps = "CLASS"
par2.file.ps = "CAMB"
par3.file.ps = "CLASSemu" 
par4.file.ps = "BACCOemu" 

# Get linear power spectra
par1.cosmo.plin = toolscosmo.get_Plin(par1)
par2.cosmo.plin = toolscosmo.get_Plin(par2)
par3.cosmo.plin = toolscosmo.get_Plin(par3)
par4.cosmo.plin = toolscosmo.get_Plin(par4)

# Function to compute non-linear power spectrum using HMcode
def get_nonlinear_power(par, z=0.0):
    """Compute non-linear power spectrum using HMcode."""
    # Set up cosmology for HMcode
    cosmo_hm = pyhmcode.Cosmology()
    cosmo_hm.omega_m = par.cosmo.Om
    cosmo_hm.omega_b = par.cosmo.Ob
    cosmo_hm.h = par.cosmo.h0
    cosmo_hm.ns = par.cosmo.ns
    cosmo_hm.sigma_8 = par.cosmo.sigma8 if hasattr(par.cosmo, 'sigma8') else None
    cosmo_hm.A = par.cosmo.As
    cosmo_hm.T_cmb = par.cosmo.Tcmb
    
    # Get linear power spectrum
    k = par.cosmo.plin['k']
    Pk_lin = par.cosmo.plin['P']
    
    # Initialize HMcode
    hm = HMcode2020(cosmo_hm)
    
    # Compute non-linear power spectrum
    Pk_nonlin = hm.power_spectrum(k, z, Pk_lin)
    
    return {'k': k, 'P': Pk_nonlin}

# Compute non-linear power spectra
par1.cosmo.pnonlin = get_nonlinear_power(par1)
par2.cosmo.pnonlin = get_nonlinear_power(par2)
par3.cosmo.pnonlin = get_nonlinear_power(par3)
par4.cosmo.pnonlin = get_nonlinear_power(par4)

# Create plots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Linear power spectra
axs[0, 0].loglog(par1.cosmo.plin['k'], par1.cosmo.plin['P'], lw=4.0, ls='-', c='C0',
                label=f'{par1.file.ps}')
axs[0, 0].loglog(par2.cosmo.plin['k'], par2.cosmo.plin['P'], lw=4.0, ls='--', c='C1',
                label=f'{par2.file.ps}')
axs[0, 0].loglog(par3.cosmo.plin['k'], par3.cosmo.plin['P'], lw=3.0, ls='-.', c='C2',
                label=f'{par3.file.ps}')
axs[0, 0].loglog(par4.cosmo.plin['k'], par4.cosmo.plin['P'], lw=3.0, ls=':', c='C3',
                label=f'{par4.file.ps}')
axs[0, 0].axis([1e-3, 151, 1e-2, 8e4])
axs[0, 0].legend()
axs[0, 0].set_xlabel(r'k ($h$/Mpc)', fontsize=16)
axs[0, 0].set_ylabel(r'P$_{\mathrm{lin}}$(k)', fontsize=16)
axs[0, 0].set_title('Linear Power Spectra', fontsize=16)

# Non-linear power spectra
axs[0, 1].loglog(par1.cosmo.pnonlin['k'], par1.cosmo.pnonlin['P'], lw=4.0, ls='-', c='C0',
                label=f'{par1.file.ps}')
axs[0, 1].loglog(par2.cosmo.pnonlin['k'], par2.cosmo.pnonlin['P'], lw=4.0, ls='--', c='C1',
                label=f'{par2.file.ps}')
axs[0, 1].loglog(par3.cosmo.pnonlin['k'], par3.cosmo.pnonlin['P'], lw=3.0, ls='-.', c='C2',
                label=f'{par3.file.ps}')
axs[0, 1].loglog(par4.cosmo.pnonlin['k'], par4.cosmo.pnonlin['P'], lw=3.0, ls=':', c='C3',
                label=f'{par4.file.ps}')
axs[0, 1].axis([1e-3, 151, 1e-2, 8e4])
axs[0, 1].legend()
axs[0, 1].set_xlabel(r'k ($h$/Mpc)', fontsize=16)
axs[0, 1].set_ylabel(r'P$_{\mathrm{nl}}$(k)', fontsize=16)
axs[0, 1].set_title('Non-linear Power Spectra', fontsize=16)

# Ratio plots
f_pk = lambda k, pa: 10**splev(np.log10(k), splrep(np.log10(pa.cosmo.plin['k']), np.log10(pa.cosmo.plin['P'])))

# Linear ratio
axs[1, 0].semilogx(par1.cosmo.plin['k'], par1.cosmo.plin['P']/f_pk(par1.cosmo.plin['k'], par1), lw=4.0, ls='-', c='C0',
                  label=f'{par1.file.ps}')
axs[1, 0].semilogx(par2.cosmo.plin['k'], par2.cosmo.plin['P']/f_pk(par2.cosmo.plin['k'], par1), lw=4.0, ls='--', c='C1',
                  label=f'{par2.file.ps}')
axs[1, 0].semilogx(par3.cosmo.plin['k'], par3.cosmo.plin['P']/f_pk(par3.cosmo.plin['k'], par1), lw=3.0, ls='-.', c='C2',
                  label=f'{par3.file.ps}')
axs[1, 0].semilogx(par4.cosmo.plin['k'], par4.cosmo.plin['P']/f_pk(par4.cosmo.plin['k'], par1), lw=3.0, ls=':', c='C3',
                  label=f'{par4.file.ps}')
axs[1, 0].axis([1e-3, 151, 0.75, 1.25])
axs[1, 0].set_xlabel(r'k ($h$/Mpc)', fontsize=16)
axs[1, 0].set_ylabel(r'P$_{\mathrm{lin}}$(k)/P$_{\mathrm{CLASS}}$(k)', fontsize=16)
axs[1, 0].set_title('Linear Power Spectrum Ratios', fontsize=16)

# Non-linear ratio
f_pk_nl = lambda k, pa: 10**splev(np.log10(k), splrep(np.log10(pa.cosmo.pnonlin['k']), np.log10(pa.cosmo.pnonlin['P'])))

axs[1, 1].semilogx(par1.cosmo.pnonlin['k'], par1.cosmo.pnonlin['P']/f_pk_nl(par1.cosmo.pnonlin['k'], par1), lw=4.0, ls='-', c='C0',
                  label=f'{par1.file.ps}')
axs[1, 1].semilogx(par2.cosmo.pnonlin['k'], par2.cosmo.pnonlin['P']/f_pk_nl(par2.cosmo.pnonlin['k'], par1), lw=4.0, ls='--', c='C1',
                  label=f'{par2.file.ps}')
axs[1, 1].semilogx(par3.cosmo.pnonlin['k'], par3.cosmo.pnonlin['P']/f_pk_nl(par3.cosmo.pnonlin['k'], par1), lw=3.0, ls='-.', c='C2',
                  label=f'{par3.file.ps}')
axs[1, 1].semilogx(par4.cosmo.pnonlin['k'], par4.cosmo.pnonlin['P']/f_pk_nl(par4.cosmo.pnonlin['k'], par1), lw=3.0, ls=':', c='C3',
                  label=f'{par4.file.ps}')
axs[1, 1].axis([1e-3, 151, 0.75, 1.25])
axs[1, 1].set_xlabel(r'k ($h$/Mpc)', fontsize=16)
axs[1, 1].set_ylabel(r'P$_{\mathrm{nl}}$(k)/P$_{\mathrm{CLASS,nl}}$(k)', fontsize=16)
axs[1, 1].set_title('Non-linear Power Spectrum Ratios', fontsize=16)

plt.tight_layout()
plt.show()