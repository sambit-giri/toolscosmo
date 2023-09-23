import numpy as np 
import matplotlib.pyplot as plt 
import tools_cosmo
import scipy
import pickle, os
from copy import deepcopy

lstyles = ['-', '--', '-.', ':']

## set parameters
par = tools_cosmo.par()
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100
par.code.verbose = True

par.file.ps = "CLASS" #"CDM_Planck15_pk.dat"

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.s8 = 0.83 
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72
k_per_decade_for_pk = 5
par.cosmo.plin = tools_cosmo.get_Plin(par, k_per_decade_for_pk=k_per_decade_for_pk)

par1 = deepcopy(par)
par2 = deepcopy(par)
par3 = deepcopy(par)
par1.cosmo.Om = 0.20
par2.cosmo.Om = 0.30
par3.cosmo.Om = 0.40
par1.cosmo.plin = tools_cosmo.get_Plin(par1, k_per_decade_for_pk=k_per_decade_for_pk)
par2.cosmo.plin = tools_cosmo.get_Plin(par2, k_per_decade_for_pk=k_per_decade_for_pk)
par3.cosmo.plin = tools_cosmo.get_Plin(par3, k_per_decade_for_pk=k_per_decade_for_pk)

par4 = deepcopy(par)
par5 = deepcopy(par)
par6 = deepcopy(par)
par4.cosmo.s8 = 0.70
par5.cosmo.s8 = 0.80
par6.cosmo.s8 = 0.90
par4.cosmo.plin = tools_cosmo.get_Plin(par4, k_per_decade_for_pk=k_per_decade_for_pk)
par5.cosmo.plin = tools_cosmo.get_Plin(par5, k_per_decade_for_pk=k_per_decade_for_pk)
par6.cosmo.plin = tools_cosmo.get_Plin(par6, k_per_decade_for_pk=k_per_decade_for_pk)

W_kR = lambda k,R: 3*scipy.special.j1(k*R)/(k*R)
sig8_window = lambda k, Pk: k**2/2/np.pi * Pk * W_kR(k,8)**2

fig, axs = plt.subplots(1,2,figsize=(10,4.5))
axs[0].loglog(par1.cosmo.plin['k'], par1.cosmo.plin['P'], ls='-', c='C0',
                    label='$\Omega_m={:.1f}$'.format(par1.cosmo.Om))
axs[0].loglog(par2.cosmo.plin['k'], par2.cosmo.plin['P'], ls='-', c='C1',
                    label='$\Omega_m={:.1f}$'.format(par2.cosmo.Om))
axs[0].loglog(par3.cosmo.plin['k'], par3.cosmo.plin['P'], ls='-', c='C2',
                    label='$\Omega_m={:.1f}$'.format(par3.cosmo.Om))
axs[0].axis([1e-3,1e1,1,8e4])
axs[0].legend()
axs[0].set_xlabel('k ($h$/Mpc)')
axs[0].set_ylabel('P(k)')
axs[1].loglog(par4.cosmo.plin['k'], par4.cosmo.plin['P'], ls='-', c='C0',
                    label='$\sigma_8={:.1f}$'.format(par4.cosmo.s8))
axs[1].loglog(par5.cosmo.plin['k'], par5.cosmo.plin['P'], ls='-', c='C1',
                    label='$\sigma_8={:.1f}$'.format(par5.cosmo.s8))
axs[1].loglog(par6.cosmo.plin['k'], par6.cosmo.plin['P'], ls='-', c='C2',
                    label='$\sigma_8={:.1f}$'.format(par6.cosmo.s8))
axs[1].plot(par5.cosmo.plin['k'], sig8_window(par5.cosmo.plin['k'],par5.cosmo.plin['P']), 
                    c='k', alpha=0.3)
axs[1].axis([1e-3,1e1,1,8e4])
axs[1].legend()
axs[1].set_xlabel('k ($h$/Mpc)')
axs[1].set_ylabel('P(k)')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,2,figsize=(10,4.5))
axs[0].loglog(par1.cosmo.plin['k'], par1.cosmo.plin['P']*par1.cosmo.plin['k']**3/(2*np.pi**2), 
                    ls='-', c='C0', label='$\Omega_m={:.1f}$'.format(par1.cosmo.Om))
axs[0].loglog(par2.cosmo.plin['k'], par2.cosmo.plin['P']*par2.cosmo.plin['k']**3/(2*np.pi**2), 
                    ls='-', c='C1', label='$\Omega_m={:.1f}$'.format(par2.cosmo.Om))
axs[0].loglog(par3.cosmo.plin['k'], par3.cosmo.plin['P']*par3.cosmo.plin['k']**3/(2*np.pi**2), 
                    ls='-', c='C2', label='$\Omega_m={:.1f}$'.format(par3.cosmo.Om))
axs[0].axis([1e-3,1e1,2e-6,4e1])
axs[0].legend()
axs[0].set_xlabel('k ($h$/Mpc)')
axs[0].set_ylabel('P(k) k$^3$/(2$\pi^2$)')
axs[1].loglog(par4.cosmo.plin['k'], par4.cosmo.plin['P']*par4.cosmo.plin['k']**3/(2*np.pi**2), 
                    ls='-', c='C0', label='$\sigma_8={:.1f}$'.format(par4.cosmo.s8))
axs[1].loglog(par5.cosmo.plin['k'], par5.cosmo.plin['P']*par5.cosmo.plin['k']**3/(2*np.pi**2), 
                    ls='-', c='C1', label='$\sigma_8={:.1f}$'.format(par5.cosmo.s8))
axs[1].loglog(par6.cosmo.plin['k'], par6.cosmo.plin['P']*par6.cosmo.plin['k']**3/(2*np.pi**2), 
                    ls='-', c='C2', label='$\sigma_8={:.1f}$'.format(par6.cosmo.s8))
axs[1].plot(par5.cosmo.plin['k'], sig8_window(par5.cosmo.plin['k'],par5.cosmo.plin['P']), 
                    c='k', alpha=0.3)
axs[1].axis([1e-3,1e1,2e-6,4e1])
axs[1].legend()
axs[1].set_xlabel('k ($h$/Mpc)')
axs[1].set_ylabel('P(k) k$^3$/(2$\pi^2$)')
plt.tight_layout()
plt.show()