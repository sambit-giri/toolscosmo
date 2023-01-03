import numpy as np 
import matplotlib.pyplot as plt 
import GalaxyTools
from copy import deepcopy

lstyles = ['-', '--', '-.', ':']

## set parameters
par = GalaxyTools.par()
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100
par.code.verbose = True

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.s8 = 0.83 
# par.cosmo.As = 2.089e-9
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72

## power spectrum
par.file.ps = "CDM_Planck15_pk.dat"
ps0 = GalaxyTools.read_powerspectrum(par)
par.file.ps = "CLASS"
ps1 = GalaxyTools.read_powerspectrum(par)

## Plot
fig, axs = plt.subplots(2,3,figsize=(15,7))
axs[0,0].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='CLASS')
axs[0,0].loglog(ps0['k'], ps0['P']*ps0['k']**3/2/np.pi**2, ls=lstyles[1], label='File')
axs[0,1].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='$\Omega_m={:.2f}$'.format(par.cosmo.Om))
axs[0,2].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='$\Omega_b={:.2f}$'.format(par.cosmo.Ob))
axs[1,0].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='$h={:.2f}$'.format(par.cosmo.h0))
axs[1,1].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='$n_s={:.2f}$'.format(par.cosmo.ns))
par.cosmo.Om = 0.26; ps2 = GalaxyTools.read_powerspectrum(par)
axs[0,1].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[1], label='$\Omega_m={:.2f}$'.format(par.cosmo.Om))
par.cosmo.Om = 0.35; ps2 = GalaxyTools.read_powerspectrum(par)
axs[0,1].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[2], label='$\Omega_m={:.2f}$'.format(par.cosmo.Om))
par.cosmo.Om = 0.315
par.cosmo.Ob = 0.01; ps2 = GalaxyTools.read_powerspectrum(par)
axs[0,2].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[1], label='$\Omega_b={:.2f}$'.format(par.cosmo.Ob))
par.cosmo.Ob = 0.09; ps2 = GalaxyTools.read_powerspectrum(par)
axs[0,2].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[2], label='$\Omega_b={:.2f}$'.format(par.cosmo.Ob))
par.cosmo.Ob = 0.049
par.cosmo.h0 = 0.50; ps2 = GalaxyTools.read_powerspectrum(par)
axs[1,0].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[1], label='$h={:.2f}$'.format(par.cosmo.h0))
par.cosmo.h0 = 0.90; ps2 = GalaxyTools.read_powerspectrum(par)
axs[1,0].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[2], label='$h={:.2f}$'.format(par.cosmo.h0))
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.80; ps2 = GalaxyTools.read_powerspectrum(par)
axs[1,1].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[1], label='$n_s={:.2f}$'.format(par.cosmo.ns))
par.cosmo.ns = 1.10; ps2 = GalaxyTools.read_powerspectrum(par)
axs[1,1].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[2], label='$n_s={:.2f}$'.format(par.cosmo.ns))
par.cosmo.ns = 0.963
# axs[1,2].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='$A_s={:.2e}$'.format(par.cosmo.As))
# par.cosmo.As = 1e-10; ps2 = GalaxyTools.read_powerspectrum(par)
# axs[1,2].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[1], label='$A_s={:.2e}$'.format(par.cosmo.As))
# par.cosmo.As = 1e-8; ps2 = GalaxyTools.read_powerspectrum(par)
# axs[1,2].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[2], label='$A_s={:.2e}$'.format(par.cosmo.As))
# par.cosmo.As = 2.089e-9
axs[1,2].loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[0], label='$s_8={:.2f}$'.format(par.cosmo.s8))
par.cosmo.s8 = 0.68; ps2 = GalaxyTools.read_powerspectrum(par)
axs[1,2].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[1], label='$s_8={:.2f}$'.format(par.cosmo.s8))
par.cosmo.s8 = 0.98; ps2 = GalaxyTools.read_powerspectrum(par)
axs[1,2].loglog(ps2['k'], ps2['P']*ps2['k']**3/2/np.pi**2, ls=lstyles[2], label='$s_8={:.2f}$'.format(par.cosmo.s8))
par.cosmo.s8 = 0.83
for ax in axs.flatten():
    ax.axis([3e-2,30,3e-2,3e1])
    ax.legend()
plt.tight_layout()
plt.show()
