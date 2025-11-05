import numpy as np 
import matplotlib.pyplot as plt 
import toolscosmo
import scipy
from scipy.interpolate import splrep,splev
import pickle, os
from copy import deepcopy

lstyles = ['-', '--', '-.', ':']

## set parameters
par = toolscosmo.par()
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100
par.code.verbose = True

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.As = 2.126e-09
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72

par.file.ps = "CLASSemu"
plin_CLASSemu = toolscosmo.get_Plin(par)

par.file.ps = "CLASS"
plin_CLASS = toolscosmo.get_Plin(par)

fig, axs = plt.subplots(2,1, figsize=(5,7))
axs[0].loglog(plin_CLASS['k'], plin_CLASS['P'], ls='-', label='CLASS')
axs[0].loglog(plin_CLASSemu['k'], plin_CLASSemu['P'], ls='--', label='CLASSemu')
axs[0].legend()
axs[0].set_xlabel('k ($h$/Mpc)', fontsize=16)
axs[0].set_ylabel('P(k)', fontsize=16)
axs[1].semilogx(plin_CLASS['k'], plin_CLASS['P']/plin_CLASS['P'], ls='-', label='CLASS')
axs[1].semilogx(plin_CLASS['k'], plin_CLASSemu['P']/plin_CLASS['P'], ls='--', label='CLASSemu')
axs[1].set_xlabel('k ($h$/Mpc)', fontsize=16)
axs[1].set_ylabel('Ratio', fontsize=16)
axs[1].set_ylim(0.95,1.05)
plt.tight_layout()
plt.show()

def sigma_squared(M, param):
    rbin, var, dlnvardlnr = toolscosmo.variance(param)
    mbin = toolscosmo.rbin_to_mbin(rbin, param)
    tck = splrep(mbin, var)
    return splev(M, tck)

Mbins = 10**np.linspace(6,15)
par.file.ps = "CLASSemu"
sigma2_CLASSemu = sigma_squared(Mbins, par)
par.file.ps = "CLASS"
sigma2_CLASS = sigma_squared(Mbins, par)

fig, axs = plt.subplots(2,1, figsize=(5,7))
axs[0].loglog(Mbins, sigma2_CLASS, ls='-', label='CLASS')
axs[0].loglog(Mbins, sigma2_CLASSemu, ls='--', label='CLASSemu')
axs[0].legend()
axs[1].set_xlabel('M ($h^{-1}M_\odot$)', fontsize=16)
axs[0].set_ylabel('$\sigma^2(M)$', fontsize=16)
axs[1].semilogx(Mbins, sigma2_CLASS/sigma2_CLASS, ls='-', label='CLASS')
axs[1].semilogx(Mbins, sigma2_CLASSemu/sigma2_CLASS, ls='--', label='CLASSemu')
axs[1].set_xlabel('M ($h^{-1}M_\odot$)', fontsize=16)
axs[1].set_ylabel('Ratio', fontsize=16)
axs[1].set_ylim(0.9,1.1)
plt.tight_layout()
plt.show()