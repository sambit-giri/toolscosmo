import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from time import time

import toolscosmo
from toolscosmo import merger_trees

param = toolscosmo.par('lcdm')
param.code.zmin = 0.0
param.code.kmax = 50 #500
param.code.Nrbin = 300
param.cosmo.Om = 0.25
param.cosmo.Or = 0.0
param.cosmo.Ob = 0.045
param.cosmo.h0 = 0.73
param.cosmo.ns = 1.0
param.cosmo.As = 2.089e-09 #None #
param.cosmo.s8 = 0.9
param.cosmo.solver = 'toolscosmo' #'astropy'
param.mf.dc = 1.686
param.file.ps = 'CLASSemu' #'CLASS' #'CAMB' #
param.file.ps = toolscosmo.calc_Plin(param)

fig, ax = plt.subplots(1,1,figsize=(8,6))
rbin, mbin, var = merger_trees.sigma_squared_table(param)
ax.loglog(mbin, var, c='k')
ax.axhline(param.cosmo.s8**2, c='C1', ls='--')
ax.axvline(splev(8, splrep(rbin, mbin)), c='C1', ls='--')
ax.set_xlabel('$M$ $[h^{-1}\mathrm{M}_\odot]$')
ax.set_ylabel('$\sigma^2(M)$')
plt.tight_layout()
plt.show()

# Zhang2008a = {
#     'ZMF2008a':np.array([[0.00107, 183.39727],
#                         [0.00495, 10.85711],
#                         [0.02819, 0.55090],
#                         [0.14660, 0.04674],
#                         [0.39371, 0.01853],
#                         [0.58039, 0.01951],
#                         [0.80426, 0.04674],
#                         [0.90413, 0.11788],
#                         ]),
#     'ZH2006': np.array([[0.00107, 141.83575],
#                         [0.00426, 12.66714],
#                         [0.01909, 1.02077],
#                         [0.09930, 0.07814],
#                         [0.45717, 0.01760],
#                         [0.75844, 0.03098],
#                         [0.90413, 0.11788],
#                         ]),
#     'ST2002': np.array([[0.00107, 1000.00000],
#                         [0.00524, 30.34889],
#                         [0.03917, 0.44852],
#                         [0.16527, 0.04006],
#                         [0.43067, 0.01760],
#                         [0.75823, 0.03433],
#                         [0.90413, 0.11788],
#                         ]),
# }

# M0 = 1e13*param.cosmo.h0
# z0 = 0
# w0 = merger_trees.delcrit_SC(z0, param)
# # dw = 0.002
# # w1 = w0+dw 
# z1 = 0.01
# w1 = merger_trees.delcrit_SC(z1, param)
# Mratio_bin = np.arange(0.001,1,0.01)
# M1s = Mratio_bin*M0 
# n_EPS = merger_trees.progenitor_mass_function(M1s,w1,M0,w0,param,collapse_model='EC')

# fig, ax = plt.subplots(1,1,figsize=(8,6))
# ax.loglog(Mratio_bin, M0*n_EPS, c='C2', label='toolscosmo', alpha=0.6)
# ax.loglog(Zhang2008a['ST2002'][:,0], Zhang2008a['ST2002'][:,1], c='b', ls='--', label='ST2002')
# ax.loglog(Zhang2008a['ZH2006'][:,0], Zhang2008a['ZH2006'][:,1], c='k', ls=':', label='ZH2006')
# ax.loglog(Zhang2008a['ZMF2008a'][:,0], Zhang2008a['ZMF2008a'][:,1], c='r', ls='-', label='ZMF2008a')
# ax.set_xscale('linear')
# ax.legend()
# ax.set_xlabel('$M/M_0 (M_0=10^{13}M_\odot$)')
# ax.set_ylabel('$M_0 n_\mathrm{EPS}$')
# plt.tight_layout()
# plt.show()

Jiang2014 = {
    'fig1': np.array([[0.00260, 1.75097],
                    [0.00606, 0.73930],
                    [0.01645, -0.27237],
                    [0.05801, -1.17510],
                    [0.12035, -1.68872],
                    [0.25887, -2.07782],
                    [0.42165, -2.21790],
                    [0.62597, -2.17121],
                    [0.83723, -1.79767],
                    [0.96537, -0.83268],
                    [0.99307, 0.07004],
                    [1.00000, 0.94163],
                    [1.00000, 1.90661],
                    ]),
    }


M0 = 1e13 #Msun/h
z0 = 0
w0 = merger_trees.delcrit_SC(z0, param)
dw = 0.002
w1 = w0+dw 
# z1 = 0.01
# w1 = merger_trees.delcrit_SC(z1, param)
Mratio_bin = np.linspace(0.0001,0.9999,100)
M1s = Mratio_bin*M0 
n_EPS = merger_trees.progenitor_mass_function(M1s,w1,M0,w0,param,collapse_model='SC')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.loglog(Mratio_bin, M0*n_EPS, c='C2', label='toolscosmo', alpha=0.6)
ax.loglog(Jiang2014['fig1'][:,0], 10**(Jiang2014['fig1'][:,1]), c='r', ls='-', label='EPS')
ax.axhline(1e-2, c='C1', ls='--')
ax.set_xscale('linear')
ax.legend()
ax.set_xlabel('$M/M_0 (M_0=10^{13}h^{-1}M_\odot$)')
ax.set_ylabel('$M_0 n_\mathrm{EPS}$')
ax.axis([0,1,1e-3,1e2])
plt.tight_layout()
plt.show()


Mtree_sim = merger_trees.ParkinsonColeHelly2008(param)
m_tree = Mtree_sim.run(1e12, 0, 3, 1e4, max_tree_length=5000000)

exit()

cosmo = toolscosmo.cython_ParkinsonColeHelly2008.CosmoParams(
            Om = param.cosmo.Om, 
            Ob = param.cosmo.Ob, 
            Or = param.cosmo.Or, 
            Ok = param.cosmo.Ok, 
            Ode = 1-param.cosmo.Om, 
            h0  = param.cosmo.h0, 
            )

print(Mtree_sim.Dz(0, 1e12, 1e8/1e12))
lnsigma_tck, alpha_tck = toolscosmo.cython_ParkinsonColeHelly2008.prepare_sigmaM(mbin, np.sqrt(var))
print(toolscosmo.cython_ParkinsonColeHelly2008.Dz(0, 1e12, 1e8/1e12, lnsigma_tck, alpha_tck, cosmo, 0.1, 0.1, 0.57, 0.38, -0.01))
# exit()

z_tree, M_tree, z_subh, M_subh = toolscosmo.cython_ParkinsonColeHelly2008.ParkinsonColeHelly2008_run(
                                    1e12, 0, 3, 
                                    cosmo, 
                                    mbin, 
                                    np.sqrt(var),
                                    M_res=1e4, 
                                    e1=0.1, 
                                    e2=0.1, 
                                    G0=0.57, 
                                    g1=0.38, 
                                    g2=-0.01
                                    )
exit()

t0 = time()
# D0 = toolscosmo.growth_factor(1,param); print(f'D(z)={D0:.3f}')
Mtree_sim.prepare_sigmaM() 
# s0 = Mtree_sim.sigma(1e12); print(f'sigma(M)={s0:.3f}')
# a0 = Mtree_sim.alpha(1e12); print(f'alpha(M)={a0:.3f}')
Mtree_sim.prepare_Jfit()
j0 = Mtree_sim.J(1); print(f'J(u)={j0:.3f}')
dt0 = time()-t0
print(f'Runtime: {dt0:.6f} s')

t1 = time()
# D1 = toolscosmo.cython_ParkinsonColeHelly2008.GrowthFactor(1, cosmo); print(f'D(z)={D1:.3f}')
# lnsigma_tck, alpha_tck = toolscosmo.cython_ParkinsonColeHelly2008.prepare_sigmaM(mbin, np.sqrt(var))
# s1 = toolscosmo.cython_ParkinsonColeHelly2008.sigma(1e12, lnsigma_tck); print(f'sigma(M)={s1:.3f}')
# a1 = toolscosmo.cython_ParkinsonColeHelly2008.alpha(1e12, alpha_tck); print(f'alpha(M)={a1:.3f}')
J_tck = toolscosmo.cython_ParkinsonColeHelly2008.prepare_Jfit()
j1 = toolscosmo.cython_ParkinsonColeHelly2008.J(1, J_tck); print(f'J(u)={j1:.3f}')
dt1 = time()-t1
print(f'Runtime: {dt1:.6f} s | {dt0/dt1:.3f} times faster')

