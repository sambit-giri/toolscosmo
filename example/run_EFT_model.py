import numpy as np
from numpy import sqrt
from time import time
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm
from scipy.interpolate import interp1d

import toolscosmo

par = toolscosmo.par()
plin = toolscosmo.get_Plin(par)
par.file.ps = plin
eft = toolscosmo.EFTformalism_Anastasiou2024(par, save_folder='work/IntegralMatrices', verbose=False)
eft.compute_tracer_intergrals()
ps_fit = eft.fit_P_linear(plin=plin)

fig, ax = plt.subplots(1,1,figsize=(7,5))
k, P_L = plin['k'], plin['P']
ax.loglog(k, P_L, lw=3, ls='-', label='$P_\mathrm{lin}$')
PlinSpline = interp1d(plin['k'], plin['P'], kind='cubic', fill_value='extrapolate')
k_sample = np.logspace(-3, np.log10(2), 150)
ax.loglog(k_sample, PlinSpline(k_sample), lw=3, ls='--', label='Spline')
P_Lfit = ps_fit['Plin_fit']
ax.loglog(k, P_Lfit, lw=3, ls=':', label='fit')
ax.set_xlabel(f'$k [h/\mathrm{{Mpc}}]$')
ax.set_ylabel(f'$P(k)$')
plt.tight_layout()
plt.show()

k_mod = np.logspace(-3, np.log10(2), 150)
redshift = 9
b1 = 1.0  
b2 = 0.5
bG2 = -1
bGamma3 = 0.2
R2 = 0.1
Pshot = 1
cs2 = 0.01

fig, axs = plt.subplots(2,4,figsize=(15,5))
cmap = plt.get_cmap('jet')
axs[0,0].set_title('Redshift')
param_array = np.linspace(6,12,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, b2, bG2, bGamma3, R2, Pshot, cs2, i0)
    axs[0,0].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[0,0], orientation='vertical', pad=0.01)
axs[0,1].set_title('$b_1$')
param_array = np.linspace(-20,20,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, i0, b2, bG2, bGamma3, R2, Pshot, cs2, redshift)
    axs[0,1].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[0,1], orientation='vertical', pad=0.01)
axs[0,2].set_title('$b_2$')
param_array = np.linspace(-5,5,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, i0, bG2, bGamma3, R2, Pshot, cs2, redshift)
    axs[0,2].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[0,2], orientation='vertical', pad=0.01)
axs[0,3].set_title('$b_{G_2}$')
param_array = np.linspace(-20,20,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, b2, i0, bGamma3, R2, Pshot, cs2, redshift)
    axs[0,3].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[0,3], orientation='vertical', pad=0.01)
axs[1,0].set_title('$b_{\Gamma_3}$')
param_array = np.linspace(-20,20,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, b2, bG2, i0, R2, Pshot, cs2, redshift)
    axs[1,0].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[1,0], orientation='vertical', pad=0.01)
axs[1,1].set_title('$R2$')
param_array = np.linspace(-20,20,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, b2, bG2, bGamma3, i0, Pshot, cs2, redshift)
    axs[1,1].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[1,1], orientation='vertical', pad=0.01)
axs[1,2].set_title('$P_{shot}$')
param_array = np.linspace(-20,20,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, b2, bG2, bGamma3, R2, i0, cs2, redshift)
    axs[1,2].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[1,2], orientation='vertical', pad=0.01)
axs[1,3].set_title('$c^2_{s}$')
param_array = np.linspace(-20,20,10)
norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for i0 in param_array:
    pmod = eft.model_P_tracer(k_mod, b1, b2, bG2, bGamma3, R2, Pshot, i0, redshift)
    axs[1,3].loglog(k_mod, pmod*k_mod**3/2/np.pi**2, color=cmap(norm(i0)))
cbar = fig.colorbar(sm, ax=axs[1,3], orientation='vertical', pad=0.01)
for ax in axs.flatten():
    ax.set_xlabel(f'$k [h/\mathrm{{Mpc}}]$')
    ax.set_ylabel(f'$P(k)$')
plt.tight_layout()
plt.show()