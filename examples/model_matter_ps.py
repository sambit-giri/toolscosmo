import numpy as np 
import matplotlib.pyplot as plt 
import GalaxyTools

lstyles = ['-', '--', '-.', ':']

## set parameters
par = GalaxyTools.par()
par.code.kmin = 0.001
par.code.kmax = 200
par.code.Nk   = 100

par.cosmo.Om = 0.315
par.cosmo.Ob = 0.049
par.cosmo.s8 = 0.83
par.cosmo.h0 = 0.673
par.cosmo.ns = 0.963
par.cosmo.Tcmb = 2.72

## power spectrum
par.file.ps = "CDM_Planck15_pk.dat"
ps0 = GalaxyTools.read_powerspectrum(par)
par.file.ps = "CLASS"
ps1 = GalaxyTools.read_powerspectrum(par)

## Plot
fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.loglog(ps0['k'], ps0['P']*ps0['k']**3/2/np.pi**2, ls=lstyles[0])
ax.loglog(ps1['k'], ps1['P']*ps1['k']**3/2/np.pi**2, ls=lstyles[1])
ax.axis([3e-2,30,3e-2,3e1])
plt.tight_layout()
plt.show()