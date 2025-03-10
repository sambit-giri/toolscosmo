import numpy as np
import matplotlib.pyplot as plt
import toolscosmo as tcm
import tools21cm as t2c

grid_size = 512 #128
res_factor = 4 #2
box_size  = 500 #Mpc/h 
param = tcm.par()
param.file.ps = tcm.get_Plin(param)
iSeed = 314159265

z1 = 149.
z2 = 9.

delta_grf, kx, ky, kz = tcm.generate_gaussian_random_field(grid_size*res_factor, box_size, param=param, random_seed=iSeed)
particle_pos_z1 = tcm.generate_initial_conditions(grid_size*res_factor, box_size, z1, param, LPT=1, delta_grf=delta_grf, filter_aliasing=1)['positions']
particle_pos_z2 = tcm.generate_initial_conditions(grid_size*res_factor, box_size, z2, param, LPT=1, delta_grf=delta_grf, filter_aliasing=1)['positions']
delta_z1 = tcm.particles_on_grid(particle_pos_z1, grid_size, box_size)
delta_z2 = tcm.particles_on_grid(particle_pos_z2, grid_size, box_size)

fig, axs = plt.subplots(1,3,figsize=(12,4))
axs[0].set_title('Gaussian Random Field')
xx = np.linspace(0,box_size,grid_size*res_factor)
axs[0].pcolor(xx, xx, delta_grf[:,:,grid_size*res_factor//2])
xx = np.linspace(0,box_size,grid_size)
axs[1].set_title(f'$z={z1}$')
axs[1].pcolor(xx, xx, delta_z1[:,:,grid_size//2])
axs[2].set_title(f'$z={z2}$')
axs[2].pcolor(xx, xx, delta_z2[:,:,grid_size//2])
for ax in axs.flatten():
    ax.set_xlabel('[Mpc/$h$]')
    ax.set_ylabel('[Mpc/$h$]')
plt.tight_layout()
plt.show()

ps0 = t2c.power_spectrum_1d(delta_grf, kbins=20, box_dims=box_size)
ps1 = t2c.power_spectrum_1d(delta_z1, kbins=20, box_dims=box_size)
ps2 = t2c.power_spectrum_1d(delta_z2, kbins=20, box_dims=box_size)
p0, k0 = ps0
p1, k1 = ps1[0]/tcm.growth_factor(z1, param)**2, ps1[1]
p2, k2 = ps2[0]/tcm.growth_factor(z2, param)**2, ps2[1]

fig, ax = plt.subplots(1,1,figsize=(5,4))
kk, pp = param.file.ps['k'], param.file.ps['P']
ax.loglog(kk, pp, c='k', ls='-', label='$P_\mathrm{{lin}}$')
ax.loglog(k0, p0, c='r', ls='--', label='$P^{{grid}}_\mathrm{{lin}}$')
ax.loglog(k1, p1, c='C0', ls='-.', label=f'$z={z1}$')
ax.loglog(k2, p2, c='C1', ls=':', label=f'$z={z2}$')
ax.set_xlabel('[$h$/Mpc]')
ax.set_ylabel('$P(k)$')
ax.axis([8e-3,3,3,4e4])
ax.legend()
plt.tight_layout()
plt.show()