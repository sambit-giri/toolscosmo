import numpy as np 
import matplotlib.pyplot as plt 
import toolscosmo

lstyles = ['-', '--', '-.', ':']

#set parameters
par = toolscosmo.par()
par.file.ps = "CDM_Planck15_pk.dat"

par.code.zmin = 5
par.code.zmax = 40
par.code.Nz   = 50
par.code.dz_prime_lyal = 0.01
par.code.dz_prime_xray = 0.1

par.code.Mmin = 5e5  #5e4
par.code.Mmax = 2e15
par.code.NM   = 100

par.mf.window = 'tophat'
par.mf.c      = 2.5
par.mf.q      = 0.85
par.mf.p      = 0.3

par.code.MA = 'EXP' #'EPS'


par.lf.f0_sfe = 0.1 #0.05  #0.20
par.lf.g1_sfe = 0.49  #0.55  #-0.5
par.lf.g2_sfe = -0.61 #-1.03 #-0.5
par.lf.g3_sfe = 5  #0
par.lf.g4_sfe = -5 #0
par.lf.Mp_sfe = 2.0e11
par.lf.Mt_sfe = 7e8 #3.5e8 #7.0e7

par.lf.f0_sfe_nu = -0.1  #0.20
par.lf.g1_sfe_nu = 0
par.lf.g2_sfe_nu = 0
par.lf.g3_sfe_nu = 0
par.lf.g4_sfe_nu = 0
par.lf.Mp_sfe_nu = 0
par.lf.Mt_sfe_nu = 0

par.lf.Muv_min = -23.
par.lf.Muv_max = -8.0 #-15
par.lf.NMuv  = 20
par.lf.sig_M = 0.2
par.lf.eps_sys = 1.0

# print(par.xray.__dict__)
print(par.lf.__dict__)

## HMF
hmf = toolscosmo.mass_fct(par)

## UV LFs
M0 = 51.6
kappa  = 1.15e-28  # Msun yr^-1 /(erg s^-1 Hz^-1)
fstars = toolscosmo.fstar(hmf['z'], hmf['m'], 'lf', par)

m_ac = toolscosmo.mass_accr(par, output=hmf)
uvlf = toolscosmo.UVLF(par)
# out_lf = uvlf.Muv(); print(out_lf.keys())
out_lf = uvlf.UV_luminosity(); # print(out_lf.keys())

# fig, axs = plt.subplots(1,3,figsize=(14,4))
# z_plot = [6,7,8]
# ax = axs[0] # fig, ax = plt.subplots(1,1,figsize=(6,4))
# for ii,zi in enumerate(z_plot):
#     z_idx = np.abs(hmf['z']-zi).argmin() 
#     ax.loglog(hmf['m'], hmf['dndlnm'][z_idx,:], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
# ax.legend()
# ax.axis([3e6,3e12,5e-6,8e2])
# ax.set_xlabel(r'$M$ [$h^{-1}$M$_\odot$]', fontsize=13)
# ax.set_ylabel(r'$\frac{\mathrm{d}n}{\mathrm{d}lnM}$', fontsize=15)
# # plt.tight_layout()
# # plt.show()

# z_plot = [6,7,8]
# ax = axs[1] # fig, ax = plt.subplots(1,1,figsize=(6,4))
# for ii,zi in enumerate(z_plot):
#     z_idx = np.abs(hmf['z']-zi).argmin() 
#     ax.loglog(hmf['m'], fstars[z_idx,:], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
# ax.legend()
# ax.axis([5e5,3e14,5e-6,8e-2])
# ax.set_xlabel(r'$M$ [$h^{-1}$M$_\odot$]', fontsize=13)
# ax.set_ylabel(r'$f_\star$', fontsize=15)
# # plt.tight_layout()
# # plt.show()


# B2015_data = np.loadtxt('data/Bouwen2015_in_SZ21.txt')

# z_plot = [6,7,8]
# ax = axs[2] # fig, ax = plt.subplots(1,1,figsize=(6,4))
# B2015_z6_mean = B2015_data[:,1]/2.+B2015_data[:,2]/2.
# B2015_z6_std  = np.abs(B2015_data[:,2]-B2015_z6_mean)
# ax.errorbar(B2015_data[:,0], B2015_z6_mean, yerr=B2015_z6_std, c='r', ls=' ')
# for ii,zi in enumerate(z_plot):
#     z_idx = np.abs(out_lf['z']-zi).argmin() 
#     ax.plot(out_lf['uvlf']['Muv_mean'], out_lf['uvlf']['phi_uv'][z_idx,:], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
# ax.set_yscale('log')
# ax.legend()
# ax.axis([-23.2,-14.8,5e-7,4e-2])
# ax.set_xlabel(r'$M_{\rm UV}$', fontsize=13)
# ax.set_ylabel(r'$\phi (M_{\rm UV})$', fontsize=15)
# plt.tight_layout()
# plt.show()


# z_plot = [6,7,8]
# fig0, ax01 = plt.subplots(1,2,figsize=(9,4))
# ax0, ax1 = ax01
# for ii,zi in enumerate(z_plot):
#     z_idx = np.abs(out_lf['z']-zi).argmin() 
#     ax0.plot(out_lf['M_AB'][z_idx,:], out_lf['m'], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
#     ax1.loglog(out_lf['m'], out_lf['fstar'][z_idx,:], lw=3, ls=lstyles[ii], label='z={:.1f}'.format(zi))
# ax0.set_yscale('log')
# ax0.legend()
# # ax0.axis([-26,-7,2e5,3e15])
# ax0.set_xlabel(r'$M_{\rm UV}$', fontsize=13)
# ax0.set_ylabel(r'$M_{\rm h}$', fontsize=15)
# ax1.legend()
# ax1.axis([5e5,3e14,5e-6,8e-2])
# ax1.set_xlabel(r'$M$ [$h^{-1}$M$_\odot$]', fontsize=13)
# ax1.set_ylabel(r'$f_\star$', fontsize=15)
# plt.tight_layout()
# # plt.show()

Park19_Bouwens_data   = np.loadtxt('data/Park19_Bouwens.txt')
Park19_Atek_data      = np.loadtxt('data/Park19_Atek.txt')
Park19_Livermore_data = np.loadtxt('data/Park19_Livermore.txt')
Park19_Ishigaki_data  = np.loadtxt('data/Park19_Ishigaki.txt')
Park19_Oesch_data     = np.loadtxt('data/Park19_Oesch.txt')

fig, axs = plt.subplots(1,4,figsize=(15,4))
# j,zj = 0, 6
for j,zj in enumerate([6,7,8,10]):
    axs[j].set_title('$z\sim {}$'.format(zj))
    az = np.where(Park19_Bouwens_data[:,-1]==zj)
    xx, yy = Park19_Bouwens_data[az,0].squeeze(), Park19_Bouwens_data[az,1].squeeze()
    yl, yp = Park19_Bouwens_data[az,2].squeeze(), Park19_Bouwens_data[az,3].squeeze()
    axs[j].errorbar(xx, yy, yerr=[yy-yl,yp-yy], #np.max([yy-yl,yp-yy], axis=0), 
                    ls=' ', marker='o', markeredgecolor='k', color='coral', 
                    label='Bouwens+2015,2017' if j==0 else None)
    az = np.where(Park19_Atek_data[:,-1]==zj)
    xx, yy = Park19_Atek_data[az,0].squeeze(), Park19_Atek_data[az,1].squeeze()
    yl, yp = Park19_Atek_data[az,2].squeeze(), Park19_Atek_data[az,3].squeeze()
    axs[j].errorbar(xx, yy, yerr=[yy-yl,yp-yy], #np.max([yy-yl,yp-yy], axis=0), 
                    ls=' ', marker='D', markeredgecolor='k', color='orange', 
                    label='Atek+2018' if j==0 else None)
    az = np.where(Park19_Livermore_data[:,-1]==zj)
    xx, yy = Park19_Livermore_data[az,0].squeeze(), Park19_Livermore_data[az,1].squeeze()
    yl, yp = Park19_Livermore_data[az,2].squeeze(), Park19_Livermore_data[az,3].squeeze()
    axs[j].errorbar(xx, yy, yerr=[yy-yl,yp-yy], #np.max([yy-yl,yp-yy], axis=0), 
                    ls=' ', marker='s', markeredgecolor='k', color='lightblue', 
                    label='Livermore+2017' if j==2 else None)
    az = np.where(Park19_Ishigaki_data[:,-1]==zj)
    xx, yy = Park19_Ishigaki_data[az,0].squeeze(), Park19_Ishigaki_data[az,1].squeeze()
    yl, yp = Park19_Ishigaki_data[az,2].squeeze(), Park19_Ishigaki_data[az,3].squeeze()
    axs[j].errorbar(xx, yy, yerr=[yy-yl,yp-yy], #np.max([yy-yl,yp-yy], axis=0), 
                    ls=' ', marker='p', markeredgecolor='k', color='violet', 
                    label='Ishigaki+2017' if j==2 else None)
    az = np.where(Park19_Oesch_data[:,-1]==zj)
    xx, yy = Park19_Oesch_data[az,0].squeeze(), Park19_Oesch_data[az,1].squeeze()
    yl, yp = Park19_Oesch_data[az,2].squeeze(), Park19_Oesch_data[az,3].squeeze()
    axs[j].errorbar(xx, yy, yerr=[yy-yl,yp-yy], #np.max([yy-yl,yp-yy], axis=0), 
                    ls=' ', marker='h', markeredgecolor='k', color='green', 
                    label='Oesch+2017' if j==3 else None)
    axs[j].set_yscale('log')
    axs[j].axis([-23,-9,7e-6,3e2])
    z_jdx = np.abs(out_lf['z']-zj).argmin() 
    axs[j].plot(out_lf['uvlf']['Muv_mean'], out_lf['uvlf']['phi_uv'][z_jdx,:], 
                    lw=3, ls='-', color='k', label='Model' if j==1 else None)
    axs[j].legend()
    axs[j].set_xlabel(r'$M_{\rm UV}$', fontsize=13)
    axs[j].set_ylabel(r'$\phi (M_{\rm UV})$', fontsize=15)
plt.tight_layout()
plt.show()