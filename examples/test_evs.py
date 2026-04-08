import numpy as np 
import matplotlib.pyplot as plt 
import toolscosmo
import pickle, os
from copy import deepcopy
import astropy.units as u

## set parameters
par = toolscosmo.par(DM='wdm', DE='lambda')
par.DM.m_wdm  = 100.0
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100
par.code.NM   = 90
par.code.Nz   = 50
par.code.verbose = True

par.file.ps = "CAMB" # "CLASS" # "CDM_Planck15_pk.dat"
par.mf.window = 'smoothk'  # [sharpk,smoothk,tophat]
par.mf.dc = 1.686          # delta_c
par.mf.p  = 0.3             # p par of f(nu) [0.3,0.3,1] for [ST,smoothk,PS]
par.mf.q  = 1.0             # q par of f(nu) [0.707,1,1] for [ST,smoothk,PS]
par.mf.c  = 3.3
par.mf.beta = 4.8

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.s8 = 0.83 
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72

ms, zs, dndlnm = toolscosmo.massfct.dndlnm(par)
Ms = 10**(np.log10(ms[::2])+np.random.normal(0.0,0.1,size=len(ms[::2])))*u.Msun

zplots = [5,7,9]
h0 = par.cosmo.h0
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Halo Mass Function')
for ii,zi in enumerate(zplots):
    ax.loglog(ms*h0, dndlnm[np.abs(zs-zi).argmin(),:]/h0**3, ls='-', alpha=0.3, label='$z={:.1f}$'.format(zi), c='C{}'.format(ii))
ax.axis([1e6,3e14,3e-16,8e3])
ax.set_ylabel(r'$\frac{dn}{dlnM}$', fontsize=25)
ax.set_xlabel(r'$M~~[\mathrm{M}_\odot]$', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()


redshift = 0.0
plt.title(f'z={redshift:.1f}')
survey_volume = 33510.321*u.Mpc**3 #(100*u.Mpc)**3 #
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'V={survey_volume.value:.1f}')
survey_volume = (100*u.Mpc)**3 #33510.321*u.Mpc**3 #
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'V={survey_volume.value:.1f}')
survey_volume = (550*u.Mpc)**3 #(100*u.Mpc)**3 #33510.321*u.Mpc**3 #
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'V={survey_volume.value:.1f}')
plt.xlabel(r'log$_\mathrm{10}\left(\frac{M}{M_\odot}\right)$')
plt.ylabel(r'$\phi_\mathrm{max}$')
plt.legend()
plt.tight_layout()
plt.show()

survey_volume = (100*u.Mpc)**3 #33510.321*u.Mpc**3 #
plt.title(f'V={survey_volume.value:.1f}')
redshift = 0.0
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'z={redshift:.1f}')
redshift = 4.0
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'z={redshift:.1f}')
redshift = 8.0
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'z={redshift:.1f}')
plt.xlabel(r'log$_\mathrm{10}\left(\frac{M}{M_\odot}\right)$')
plt.ylabel(r'$\phi_\mathrm{max}$')
plt.legend()
plt.tight_layout()
plt.show()

survey_volume = (100*u.Mpc)**3 #33510.321*u.Mpc**3 #
redshift = 8.0
plt.title(f'V={survey_volume.value:.1f}, z={redshift:.1f}')
par.DM.m_wdm  = 100.0
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'm_wdm={par.DM.m_wdm:.1f}')
par.DM.m_wdm  = 3.0
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'm_wdm={par.DM.m_wdm:.1f}')
par.DM.m_wdm  = 1.5
phi_max, log10m = toolscosmo.evs_hypersurface_pdf(param=par, V=survey_volume, z=redshift, mmin=8, mmax=18)
plt.plot(log10m, phi_max, label=f'm_wdm={par.DM.m_wdm:.1f}')
plt.xlabel(r'log$_\mathrm{10}\left(\frac{M}{M_\odot}\right)$')
plt.ylabel(r'$\phi_\mathrm{max}$')
plt.legend()
plt.tight_layout()
plt.show()
