import numpy as np 
import matplotlib.pyplot as plt 
import tools_cosmo
import pickle, os, glob

from scipy.stats import binned_statistic
from scipy.interpolate import splev, splrep, interp1d
from scipy.interpolate import bisplev, bisplrep

chdir = './'

def set_param(**kwargs):
    ## set parameters
    par = tools_cosmo.par(DE='cpl')

    # par.file.ps = chdir+"CDM_Planck15_pk.dat"
    par.file.ps  = kwargs.get('ps', chdir+"CDM_Planck15_pk.dat") # "CLASS"
    Odm = kwargs.get('Odm', 0.266)
    Ob  = kwargs.get('Ob', 0.049)
    par.cosmo.Ob = Ob
    par.cosmo.Om = kwargs.get('Om', Odm+Ob) 
    par.cosmo.s8 = kwargs.get('s8', 0.830)
    par.cosmo.ns = kwargs.get('ns', 0.963)
    par.cosmo.h0 = kwargs.get('h0', 0.673) 
    par.DE.w0 = kwargs.get('w0', -1)
    par.DE.wa = kwargs.get('wa', 0)

    MA = kwargs.get('MA', 'EXP')
    if MA.upper()=='EXP':
        par.code.MA = 'EXP'
        par.MA.alpha_EXP = kwargs.get('MA_param', 0.79)
    elif MA.upper()=='EPS':
        par.code.MA = 'EPS'
        par.MA.Q_EPS = kwargs.get('MA_param', 0.6)
    else:
        par.code.MA = 'HUBBLEscale' #'21cmfast'
        par.MA.t_star = kwargs.get('MA_param', 0.5)

    par.code.NM = 80
    par.code.Nz = 100
    par.code.kmin = 1e-5
    par.code.kmax = 5e2
    par.code.Mmin = 1e5
    par.code.Mmax = 3e15 #3e15
    par.code.zmin = 5.0
    par.code.zmax = 50.0
    par.code.verbose = kwargs.get('verbose', True)

    par.lf.Muv_min = -23 #-24.25
    par.lf.Muv_max = -5 #-7.75
    par.lf.NMuv  = 33
    par.lf.sig_M = kwargs.get('sig_M', 0.56)  #scatter in UV magnitude for fixed halo mas
    par.lf.eps_sys = kwargs.get('eps_sys', 1)
    par.lf.f0_sfe  = kwargs.get('f0_sfe', 0.1) #0.14#
    par.lf.Mp_sfe  = kwargs.get('Mp_sfe', 10**11.28)
    par.lf.g1_sfe  = kwargs.get('g1_sfe', 0.49) #0.49
    par.lf.g2_sfe  = kwargs.get('g2_sfe', -0.60) #-0.60 
    par.lf.Mt_sfe  = kwargs.get('Mt_sfe', 10**4)
    par.lf.g3_sfe  = kwargs.get('g3_sfe', 5.0)
    par.lf.g4_sfe  = kwargs.get('g4_sfe', -5.0)
    par.lf.f0_sfe_nu = kwargs.get('f0_sfe_nu', -0.58) #0.0 #-0.8 #
    par.lf.Mp_sfe_nu = 0.0
    par.lf.Mt_sfe_nu = 0.0
    par.lf.g1_sfe_nu = 0.0
    par.lf.g2_sfe_nu = 0.0
    par.lf.g3_sfe_nu = 0.0
    par.lf.g4_sfe_nu = 0.0
    
    return par

def model_uvlfs(**kwargs):
    par = kwargs.get('param', set_param(**kwargs))    
    print_cosmo = kwargs.get('print_cosmo', False)
    if print_cosmo: 
        print(par.cosmo.__dict__)
        print(par.DE.__dict__)
    # print(kwargs)

    ## UV LFs
    f_duty = kwargs.get('f_duty', 1)
    uvlf = tools_cosmo.UVLF(par)
    out_lf = uvlf.UV_luminosity(f_duty=f_duty)
    # print(out_lf.keys())
    return out_lf

lf_EXP = model_uvlfs(MA='EXP', g1_sfe=-0.5, g2_sfe=-0.5, g3_sfe=0, g4_sfe=0,
                    Mt_sfe=3.365e8*2, Mp_sfe=6.73e9, f_duty='EXP')
lf_21cmfast = model_uvlfs(MA='21cmfast', MA_param=0.3, g1_sfe=-0.5, g2_sfe=-0.5, g3_sfe=0, g4_sfe=0,
                    Mt_sfe=3.365e8*2, Mp_sfe=6.73e9, f_duty='EXP')

Mplot = [1e7,1e9,1e11,1e13]
zplot = [5,8,10] #[6,9,12,15]

fig, ax = plt.subplots(1,1,figsize=(6,5))
for jj,zj in enumerate(zplot):
    z_jdx = np.abs(lf_EXP['z']-zj).argmin()
    ax.loglog(lf_EXP['m'], lf_EXP['fstar'][z_jdx,:], 
            label=f'z={zj}', lw=5, c=f'C{jj}', alpha=0.3, ls='-')
    ax.loglog(lf_21cmfast['m'], lf_21cmfast['fstar'][z_jdx,:], 
            label=f'z={zj}', lw=3, c=f'C{jj}', alpha=1.0, ls=':')
ax.legend()
ax.axis([1e5,3e15,5e-5,8e-1])
ax.set_ylabel('$f_*$')
ax.set_xlabel('M [$h^{-1}M_\odot$]')
plt.tight_layout()
plt.show()

# exit()

# lf_EXP = model_uvlfs(MA='EXP')
# # lf_EPS = model_uvlfs(MA='EPS')
# lf_21cmfast = model_uvlfs(MA='21cmfast')

Mplot = [1e7,1e9,1e11,1e13]
fig, axs = plt.subplots(1,2,figsize=(13,5))
for jj,mj in enumerate(Mplot):
    m_jdx = np.abs(lf_EXP['m']-mj).argmin()
    ax = axs[0]
    ax.semilogy(lf_EXP['z'], lf_EXP['M_accr'][:,m_jdx], 
                lw=5, alpha=0.3, c=f'C{jj}', ls='-',
                label='EXP accretion' if jj==0 else None)
    # ax.semilogy(lf_EPS['z'], lf_EPS['M_accr'][:,m_jdx], 
    #             lw=4, alpha=0.7, c=f'C{jj}', ls='--',
    #             label='EPS accretion' if jj==0 else None)
    ax.semilogy(lf_21cmfast['z'], lf_21cmfast['M_accr'][:,m_jdx], 
                lw=3, alpha=1.0, c=f'C{jj}', ls=':',
                label='Hubble scale accretion' if jj==0 else None)
    ax = axs[1]
    ax.semilogy(lf_EXP['z'], lf_EXP['dMdt_accr'][:,m_jdx], 
                lw=5, alpha=0.3, c=f'C{jj}', ls='-',
                label='EXP accretion' if jj==0 else None)
    # ax.semilogy(lf_EPS['z'], lf_EPS['dMdt_accr'][:,m_jdx], 
    #             lw=4, alpha=0.7, c=f'C{jj}', ls='--',
    #             label='EPS accretion' if jj==0 else None)
    ax.semilogy(lf_21cmfast['z'], lf_21cmfast['dMdt_accr'][:,m_jdx], 
                lw=3, alpha=1.0, c=f'C{jj}', ls=':',
                label='Hubble scale accretion' if jj==0 else None)
axs[0].legend()
axs[0].set_ylabel('M$_\mathrm{accr}$')
axs[1].set_ylabel('dM$_\mathrm{accr}$/dt')
for ax in axs: ax.set_xlabel('z')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,2,figsize=(13,5))
for jj,zj in enumerate(zplot):
    z_jdx = np.abs(lf_EXP['z']-zj).argmin()
    ax = axs[0]
    ax.semilogx(lf_EXP['m'], lf_EXP['M_AB'][z_jdx,:], 
            label=f'z={zj}', lw=5, c=f'C{jj}', alpha=0.3, ls='-')
    ax.semilogx(lf_21cmfast['m'], lf_21cmfast['M_AB'][z_jdx,:], 
            label=f'z={zj}', lw=3, c=f'C{jj}', alpha=1.0, ls=':')
    ax = axs[1]
    ax.semilogx(lf_21cmfast['m'], lf_21cmfast['M_AB'][z_jdx,:]/lf_EXP['M_AB'][z_jdx,:], 
            label=f'z={zj}', lw=3, c=f'C{jj}', alpha=1.0, ls=':')
axs[0].legend()
axs[0].axis([1e5,3e15,-26,9])
axs[1].axis([1e5,3e15,-6,9])
axs[0].set_ylabel('$M_\mathrm{AB}$')
axs[1].set_ylabel('Ratio')
for ax in axs: ax.set_xlabel('M [$h^{-1}M_\odot$]')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,2,figsize=(13,5))
for jj,zj in enumerate(zplot):
    z_jdx = np.abs(lf_EXP['z']-zj).argmin()
    ax = axs[0]
    ax.plot(lf_EXP['uvlf']['Muv_mean'], lf_EXP['uvlf']['phi_uv'][z_jdx,:], 
            label=f'z={zj}', lw=5, c=f'C{jj}', alpha=0.3, ls='-')
    ax.plot(lf_21cmfast['uvlf']['Muv_mean'], lf_21cmfast['uvlf']['phi_uv'][z_jdx,:], 
            label=f'z={zj}', lw=3, c=f'C{jj}', alpha=1.0, ls=':')
    ax = axs[1]
    ax.plot(lf_21cmfast['uvlf']['Muv_mean'], lf_21cmfast['uvlf']['phi_uv'][z_jdx,:]/lf_EXP['uvlf']['phi_uv'][z_jdx,:], 
            label=f'z={zj}', lw=3, c=f'C{jj}', alpha=1.0, ls=':')
axs[0].legend()
axs[0].set_yscale('log')
axs[0].axis([-23.8,-6.8,1e-8,5e2]) #([-23.8,-15.8,1e-8,5e-2])
axs[1].axis([-23.8,-6.8,-0.5,15]) #([-23.8,-15.8,-0.5,15])
axs[0].set_ylabel('$\phi_\mathrm{UV}$')
axs[1].set_ylabel('Ratio')
for ax in axs: ax.set_xlabel('$M_\mathrm{UV}$')
plt.tight_layout()
plt.show()


