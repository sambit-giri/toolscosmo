import numpy as np
import matplotlib.pyplot as plt
import toolscosmo

grid_size = 128
box_size  = 100 #Mpc/h 
param = toolscosmo.par()

z1 = 149.
z2 = 9.

delta_grf, kx, ky, kz = toolscosmo.generate_gaussian_random_field(grid_size, box_size, param=param)
particle_pos_z1 = toolscosmo.generate_initial_conditions(grid_size, box_size, z1, param, LPT=2, delta_grf=delta_grf)
particle_pos_z2 = toolscosmo.generate_initial_conditions(grid_size, box_size, z2, param, LPT=2, delta_grf=delta_grf)
delta_z1 = toolscosmo.particles_on_grid(particle_pos_z1, grid_size, box_size)
delta_z2 = toolscosmo.particles_on_grid(particle_pos_z2, grid_size, box_size)

fig, axs = plt.subplots(1,3,figsize=(12,4))
xx = np.linspace(0,box_size,grid_size)
axs[0].set_title('Gaussian Random Field')
axs[0].pcolor(xx, xx, delta_grf[:,:,64])
axs[1].set_title(f'$z={z1}$')
axs[1].pcolor(xx, xx, delta_z1[:,:,64])
axs[2].set_title(f'$z={z2}$')
axs[2].pcolor(xx, xx, delta_z2[:,:,64],50)
for ax in axs.flatten():
    ax.set_xlabel('[Mpc/$h$]')
    ax.set_ylabel('[Mpc/$h$]')
plt.tight_layout()
plt.show()