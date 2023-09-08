import numpy as np 
import matplotlib.pyplot as plt 
import GalaxyTools

## set parameters
par = GalaxyTools.par()

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.s8 = 0.83 
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72

zs = 10**np.linspace(-2,2,100)

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Normalised Hubble', fontsize=15)
par.cosmo.Ogamma = 0
ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls='-', label='$\Omega_\gamma={}$'.format(par.cosmo.Ogamma))
par.cosmo.Ogamma = 5.4e-5
ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls='--', label='$\Omega_\gamma={}$'.format(par.cosmo.Ogamma))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Normalised Hubble', fontsize=15)
par.DE.w = -0.5
ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls='-', label='$\omega={}$'.format(par.DE.w))
par.DE.w = -1.0
ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls='--', label='$\omega={}$'.format(par.DE.w))
par.DE.w = -1.5
ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls=':', label='$\omega={}$'.format(par.DE.w))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)
plt.tight_layout()
plt.show()
