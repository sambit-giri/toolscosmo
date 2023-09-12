import numpy as np 
import matplotlib.pyplot as plt 
import GalaxyTools
import pickle, os
from copy import deepcopy

## set parameters
par = GalaxyTools.par()
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100
par.code.NM   = 90
par.code.Nz   = 50
par.code.verbose = True

par.file.ps = "CDM_Planck15_pk.dat"
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

ms, zs, dndlnm = GalaxyTools.massfct.dndlnm(par)

zplots = [5,7,9]
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Halo Mass Function')
for ii,zi in enumerate(zplots):
    ax.loglog(ms, dndlnm[np.abs(zs-zi).argmin(),:], ls='--', label='$z={:.1f}$'.format(zi), c='C{}'.format(ii))
ax.axis([1e6,3e14,3e-16,8e3])
ax.set_ylabel(r'$\frac{dn}{dlnM}$', fontsize=25)
ax.set_xlabel(r'$M~~[h^{-1}\mathrm{M}_\odot]$', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()

par.DE.w = -0.5
ms0, zs0, dndlnm0 = GalaxyTools.massfct.dndlnm(par)
par.DE.w = -1.0
ms1, zs1, dndlnm1 = GalaxyTools.massfct.dndlnm(par)
par.DE.w = -1.5
ms2, zs2, dndlnm2 = GalaxyTools.massfct.dndlnm(par)
zplots = [5,7,9]
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Halo Mass Function')
for ii,zi in enumerate(zplots):
    ax.loglog(ms0, dndlnm0[np.abs(zs0-zi).argmin(),:], ls='-', label='$z={:.1f}$'.format(zi), c='C{}'.format(ii))
    ax.loglog(ms1, dndlnm1[np.abs(zs1-zi).argmin(),:], ls='--', c='C{}'.format(ii))
    ax.loglog(ms2, dndlnm2[np.abs(zs2-zi).argmin(),:], ls='-.', c='C{}'.format(ii))
ax.axis([1e6,3e14,3e-16,8e3])
ax.set_ylabel(r'$\frac{dn}{dlnM}$', fontsize=25)
ax.set_xlabel(r'$M~~[h^{-1}\mathrm{M}_\odot]$', fontsize=15)
# ax.legend()
line_labels = ['$z={:.1f}$'.format(zi) for ii,zi in enumerate(zplots)]
line_colors = ['C{}'.format(ii) for ii,zi in enumerate(zplots)]
line_legend = ax.legend(
    handles=[plt.Line2D([], [], color=color, linestyle='-', label=label) for color, label in zip(line_colors, line_labels)],
    title=None, loc=1)
ax.add_artist(line_legend)
line_labels = ['$\omega={}$'.format(w) for w in [-0.5,-1.0,-1.5]]
line_styles = ['-', '--', '-.']
line_legend = ax.legend(
    handles=[plt.Line2D([], [], color='black', linestyle=style, label=label) for style, label in zip(line_styles, line_labels)],
    title=None, loc=3)
ax.add_artist(line_legend)
plt.tight_layout()
plt.show()