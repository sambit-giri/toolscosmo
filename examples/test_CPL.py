import numpy as np 
import matplotlib.pyplot as plt 
import tools_cosmo

par = tools_cosmo.par('cpl')

zs = 10**np.linspace(-2,np.log10(7),100)

fig, axs = plt.subplots(1,3,figsize=(13,4))

ax = axs[0] 
ax.set_title('Normalised Hubble', fontsize=15)
par.DE.w0, par.DE.wa = -1.0, 0.0
ax.loglog(zs, tools_cosmo.Ez_model(par)(zs), lw=3, ls='-', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -1.0, -1.0
ax.loglog(zs, tools_cosmo.Ez_model(par)(zs), lw=3, ls='--', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -0.5, 0.0
ax.loglog(zs, tools_cosmo.Ez_model(par)(zs), lw=3, ls='-.', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -0.5, -1.0
ax.loglog(zs, tools_cosmo.Ez_model(par)(zs), lw=3, ls=':', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)

ax = axs[1] 
ax.set_title('Luminosity Distance', fontsize=15)
par.DE.w0, par.DE.wa = -1.0, 0.0
ax.loglog(zs, tools_cosmo.luminosity_distance(zs,par), lw=3, ls='-', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -1.0, -1.0
ax.loglog(zs, tools_cosmo.luminosity_distance(zs,par), lw=3, ls='--', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -0.5, 0.0
ax.loglog(zs, tools_cosmo.luminosity_distance(zs,par), lw=3, ls='-.', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -0.5, -1.0
ax.loglog(zs, tools_cosmo.luminosity_distance(zs,par), lw=3, ls=':', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$D_\mathrm{L}(z)$', fontsize=15)

ax = axs[2] 
ax.set_title('Distance Modulus', fontsize=15)
par.DE.w0, par.DE.wa = -1.0, 0.0
ax.semilogx(zs, tools_cosmo.distance_modulus(zs,par), lw=3, ls='-', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -1.0, -1.0
ax.semilogx(zs, tools_cosmo.distance_modulus(zs,par), lw=3, ls='--', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -0.5, 0.0
ax.semilogx(zs, tools_cosmo.distance_modulus(zs,par), lw=3, ls='-.', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
par.DE.w0, par.DE.wa = -0.5, -1.0
ax.semilogx(zs, tools_cosmo.distance_modulus(zs,par), lw=3, ls=':', label=f'$\omega_0,\omega_a={par.DE.w0},{par.DE.wa}$')
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$\mu(z)$', fontsize=15)

plt.tight_layout()
plt.show()