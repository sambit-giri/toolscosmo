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

par.file.ps = "CDM_Planck15_pk.dat"

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.As = 2.126e-09 #par.cosmo.s8 = 0.83 
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72
k_per_decade_for_pk = 5
par.cosmo.plin = toolscosmo.get_Plin(par, k_per_decade_for_pk=k_per_decade_for_pk)

par1 = deepcopy(par)
par2 = deepcopy(par)
par3 = deepcopy(par)
par4 = deepcopy(par)
par1.file.ps = "CDM_Planck15_pk.dat"
par2.file.ps = "CLASS"
par3.file.ps = "CAMB" 
par4.file.ps = "BACCO" 
par1.cosmo.plin = toolscosmo.get_Plin(par1, k_per_decade_for_pk=k_per_decade_for_pk)
par2.cosmo.plin = toolscosmo.get_Plin(par2, k_per_decade_for_pk=k_per_decade_for_pk)
par3.cosmo.plin = toolscosmo.get_Plin(par3, k_per_decade_for_pk=k_per_decade_for_pk)
par4.cosmo.plin = toolscosmo.get_Plin(par4, k_per_decade_for_pk=k_per_decade_for_pk)

fig, axs = plt.subplots(1,2,figsize=(13,5))
axs[0].loglog(par1.cosmo.plin['k'], par1.cosmo.plin['P'], lw=4.0, ls='-', c='C0',
                    label=f'{par1.file.ps}')
axs[0].loglog(par2.cosmo.plin['k'], par2.cosmo.plin['P'], lw=4.0, ls='--', c='C1',
                    label=f'{par2.file.ps}')
axs[0].loglog(par3.cosmo.plin['k'], par3.cosmo.plin['P'], lw=3.0, ls='-.', c='C2',
                    label=f'{par3.file.ps}')
axs[0].loglog(par4.cosmo.plin['k'], par4.cosmo.plin['P'], lw=3.0, ls=':', c='C3',
                    label=f'{par4.file.ps}')
axs[0].axis([1e-3,51,1e-2,8e4])
f_pk = lambda k,pa: 10**splev(np.log10(k),splrep(np.log10(pa.cosmo.plin['k']), np.log10(pa.cosmo.plin['P'])))
axs[1].semilogx(par1.cosmo.plin['k'], par1.cosmo.plin['P']/f_pk(par1.cosmo.plin['k'],par2), lw=4.0, ls='-', c='C0',
                    label=f'{par1.file.ps}')
axs[1].semilogx(par2.cosmo.plin['k'], par2.cosmo.plin['P']/f_pk(par2.cosmo.plin['k'],par2), lw=4.0, ls='--', c='C1',
                    label=f'{par2.file.ps}')
axs[1].semilogx(par3.cosmo.plin['k'], par3.cosmo.plin['P']/f_pk(par3.cosmo.plin['k'],par2), lw=3.0, ls='-.', c='C2',
                    label=f'{par3.file.ps}')
axs[1].semilogx(par4.cosmo.plin['k'], par4.cosmo.plin['P']/f_pk(par4.cosmo.plin['k'],par2), lw=3.0, ls=':', c='C3',
                    label=f'{par4.file.ps}')
axs[1].axis([1e-3,51,0.5,1.5])
axs[0].legend()
axs[0].set_xlabel('k ($h$/Mpc)', fontsize=16)
axs[0].set_ylabel('P(k)', fontsize=16)
plt.tight_layout()
plt.show()
