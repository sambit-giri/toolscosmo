import numpy as np 
import matplotlib.pyplot as plt 
import GalaxyTools

lstyles = ['-', '--', '-.', ':']

#set parameters
par = GalaxyTools.par()
par.file.ps = "CDM_Planck15_pk.dat"

par.code.zmin = 5
par.code.zmax = 40
par.code.Nz   = 50
par.code.dz_prime_lyal = 0.01
par.code.dz_prime_xray = 0.1

par.code.Mmin = 5e5  #5e4
par.code.Mmax = 2e15
par.code.NM   = 100

par.code.kmin = 0.001
par.code.kmax = 100
par.code.Nk   = 90

par.code.Emin = 500
par.code.Emax = 2000
par.code.NE   = 40

par.mf.window = 'tophat'
par.mf.c      = 2.5
par.mf.q      = 0.85
par.mf.p      = 0.3

par.code.MA = 'EXP' #'EPS'

par.lyal.f0 = 0.05
par.lyal.g1 = -0.5
par.lyal.g2 = -0.5

par.xray.f0 = 0.05
par.xray.g1 = -0.5
par.xray.g2 = -0.5

par.code.Mdark = 1e5

par.lyal.Nal = 5000 
par.lyal.pl_sed = 0.0

par.xray.pl_sed = 1.5
par.xray.cX  = 3.4e40
par.xray.Emin_norm = 500
par.xray.Emax_norm = 2000
par.xray.fX  = 1.0

par.reio.Nion = 2000
par.reio.fesc = 1.0

par.lf.Muv_min = -23.
par.lf.Muv_max = -15.
par.lf.NMuv  = 10
par.lf.sig_M = 0.2
par.lf.eps_sys = 1.0

## HMF
hmf = GalaxyTools.mass_fct(par)

z_plot = [6,7,8]
fig, ax = plt.subplots(1,1,figsize=(6,4))
for ii,zi in enumerate(z_plot):
    z_idx = np.abs(hmf['z']-zi).argmin() 
    ax.loglog(hmf['m'], hmf['dndlnm'][z_idx,:], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
ax.legend()
ax.axis([3e6,3e12,5e-6,8e2])
ax.set_xlabel(r'$M$ [$h^{-1}$M$_\odot$]', fontsize=13)
ax.set_ylabel(r'$\frac{\mathrm{d}n}{\mathrm{d}lnM}$', fontsize=15)
plt.tight_layout()
plt.show()

## UV LFs
M0 = 51.6
kappa  = 1.15e-28  # Msun yr^-1 /(erg s^-1 Hz^-1)
fstars = GalaxyTools.fstar(hmf['z'], hmf['m'], 'xray', par)

z_plot = [6,7,8]
fig, ax = plt.subplots(1,1,figsize=(6,4))
for ii,zi in enumerate(z_plot):
    z_idx = np.abs(hmf['z']-zi).argmin() 
    ax.loglog(hmf['m'], fstars[z_idx,:], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
ax.legend()
ax.axis([5e5,3e14,5e-6,8e-2])
ax.set_xlabel(r'$M$ [$h^{-1}$M$_\odot$]', fontsize=13)
ax.set_ylabel(r'$\frac{\mathrm{d}n}{\mathrm{d}lnM}$', fontsize=15)
plt.tight_layout()
plt.show()

m_ac = GalaxyTools.mass_accr(par, output=hmf)
m_AB = GalaxyTools.absolute_UV_magnitude(par, output=hmf)
