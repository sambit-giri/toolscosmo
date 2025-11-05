import numpy as np 
import matplotlib.pyplot as plt 
import toolscosmo

## set parameters
par_lcdm = toolscosmo.par('lcdm')
par = toolscosmo.par('wcdm')
par_growing_nu = toolscosmo.par('growing_neutrino_mass')

# zs = 10**np.linspace(-2,2,100)
zs = 10**np.linspace(-2,np.log10(7),100)

# fig, ax = plt.subplots(1,1,figsize=(6,5))
# ax.set_title('Normalised Hubble', fontsize=15)
# par.cosmo.Ogamma = 0
# ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls='-', label='$\Omega_\gamma={}$'.format(par.cosmo.Ogamma))
# par.cosmo.Ogamma = 5.4e-5
# ax.loglog(zs, GalaxyTools.Ez_model(par)(zs), lw=3, ls='--', label='$\Omega_\gamma={}$'.format(par.cosmo.Ogamma))
# ax.legend()
# ax.set_xlabel('$z$', fontsize=15)
# ax.set_ylabel('$E(z)$', fontsize=15)
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Normalised Hubble', fontsize=15)
ax.loglog(zs, toolscosmo.Ez_model(par_lcdm)(zs), lw=3, ls='-', label='$\Lambda$CDM'.format(par.DE.w))
par.DE.w = -1.0
ax.loglog(zs, toolscosmo.Ez_model(par)(zs), lw=3, ls='--', label='$\omega={}$'.format(par.DE.w))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Normalised Hubble', fontsize=15)
par.DE.w = -0.5
ax.loglog(zs, toolscosmo.Ez_model(par)(zs), lw=3, ls='-', label='$\omega={}$'.format(par.DE.w))
par.DE.w = -1.0
ax.loglog(zs, toolscosmo.Ez_model(par)(zs), lw=3, ls='--', label='$\omega={}$'.format(par.DE.w))
par.DE.w = -1.5
ax.loglog(zs, toolscosmo.Ez_model(par)(zs), lw=3, ls=':', label='$\omega={}$'.format(par.DE.w))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Normalised Hubble', fontsize=15)
par.DE.w = -1.0
ax.loglog(zs, toolscosmo.Ez_model(par)(zs), lw=3, ls='-', label='$\omega={}$,WDM'.format(par.DE.w))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.1
ax.loglog(zs, toolscosmo.Ez_model(par_growing_nu)(zs), lw=3, ls='--', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.2
ax.loglog(zs, toolscosmo.Ez_model(par_growing_nu)(zs), lw=3, ls=':', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
# ax.axhline(0.74/par.cosmo.h0, color='k')
# ax.axhline(0.673/par.cosmo.h0, color='k')
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Comoving distance', fontsize=15)
par.DE.w = -1.0
ax.loglog(zs, toolscosmo.comoving_distance(zs,par), lw=3, ls='-', label='$\omega={}$,WDM'.format(par.DE.w))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.1
ax.loglog(zs, toolscosmo.comoving_distance(zs,par_growing_nu), lw=3, ls='--', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.2
ax.loglog(zs, toolscosmo.comoving_distance(zs,par_growing_nu), lw=3, ls=':', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$D_L(z)$', fontsize=15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Distance modulus', fontsize=15)
par.DE.w = -1.0
ax.plot(zs, toolscosmo.distance_modulus(zs,par), lw=3, ls='-', label='$\omega={}$,WDM'.format(par.DE.w))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.1
ax.plot(zs, toolscosmo.distance_modulus(zs,par_growing_nu), lw=3, ls='--', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.2
ax.plot(zs, toolscosmo.distance_modulus(zs,par_growing_nu), lw=3, ls=':', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$\mu(z)$', fontsize=15)
plt.tight_layout()
plt.show()