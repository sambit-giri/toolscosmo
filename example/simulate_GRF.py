import numpy as np
import matplotlib.pyplot as plt
import toolscosmo as tcm
import tools21cm as t2c

def simulate_delta_GRF(Om=0.315, Ob=0.049, Or=5.4e-05, As=2.089e-09, h0=0.673, ns=0.963, iSeed=42, grid_size=128, box_size=500):
    """
    Generate Gaussian Random Field (GRF) from matter power spectrum.

    Parameters:
    -----------
    Om : float
        Matter density parameter.
    Ob : float
        Baryon density parameter.
    Or : float
        Radiation density parameter.
    As : float
        Amplitude of the primordial power spectrum.
    h0 : float
        Hubble parameter.
    ns : float
        Spectral index of the primordial power spectrum.
    iSeed : int, optional
        Random seed for reproducibility (default is 42).
    grid_size : int, optional
        Size of the grid for the simulation (default is 128).
    box_size : int, optional
        Size of the simulation box (default is 500 Mpc/h).

    Returns:
    --------
    dict
        A dictionary containing:
        - delta_lin : np.ndarray
            Linear density field in real space.
        - PS : np.ndarray
            Matter power spectrum.
    """
    param = tcm.par()
    param.cosmo.Om = Om
    param.cosmo.Ob = Ob
    param.cosmo.Or = Or
    param.cosmo.Ok = 0.
    param.cosmo.As = As
    param.cosmo.h0 = h0
    param.cosmo.ns = ns
    param.file.ps = 'CAMB' # 'CLASS' 'CLASSemu'
    param.file.ps = tcm.get_Plin(param)
    delta_lin = tcm.generate_gaussian_random_field(grid_size, box_size, param=param, random_seed=iSeed)['delta_lin']
    return {'delta_lin': delta_lin, 'PS': param.file.ps}

def simulate_21cm_GRF(z=9., Om=0.315, Ob=0.049, Or=5.4e-05, As=2.089e-09, h0=0.673, ns=0.963, iSeed=42, grid_size=128, box_size=500):
    """
    Generate Gaussian Random Field (GRF) from matter power spectrum.

    Parameters:
    -----------
    z : float
        Redshift.
    Om : float
        Matter density parameter.
    Ob : float
        Baryon density parameter.
    Or : float
        Radiation density parameter.
    As : float
        Amplitude of the primordial power spectrum.
    h0 : float
        Hubble parameter.
    ns : float
        Spectral index of the primordial power spectrum.
    iSeed : int, optional
        Random seed for reproducibility (default is 42).
    grid_size : int, optional
        Size of the grid for the simulation (default is 128).
    box_size : int, optional
        Size of the simulation box (default is 500 Mpc/h).

    Returns:
    --------
    dict
        A dictionary containing:
        - delta_lin : np.ndarray
            Linear density field in real space.
        - PS : np.ndarray
            Matter power spectrum.
        - dt21 : np.ndarray
            21cm brightness temperature.
    """
    grf_mod = simulate_delta_GRF(Om=Om, Ob=Ob, Or=Or, As=As, h0=h0, ns=ns, iSeed=iSeed, grid_size=grid_size, box_size=box_size)
    dt21 = t2c.mean_dt(z)*(1+grf_mod['delta_lin'])
    grf_mod['dt21'] = dt21
    return grf_mod

if __name__ == "__main__":
    grid_size = 128
    box_size  = 500 #Mpc/h

    Om_list = [0.27, 0.30]
    iSeed_list = [42, 64]

    fig, axs = plt.subplots(2,3,figsize=(13,8))
    xx = np.linspace(0,box_size,grid_size)
    for i,Om in enumerate(Om_list):
        for j,iSeed in enumerate(iSeed_list):
            grf_mod = simulate_21cm_GRF(Om=Om, iSeed=iSeed, grid_size=grid_size, box_size=box_size)
            ps, ks = t2c.power_spectrum_1d(grf_mod['delta_lin'], kbins=15, box_dims=box_size)
            axs[i,j].set_title(r'$\Omega$=%.3f, Random_seed=%d'%(Om,iSeed))
            im = axs[i,j].pcolor(xx, xx, grf_mod['delta_lin'][10])
            axs[i,j].set_xlabel('X [Mpc/h]')
            axs[i,j].set_ylabel('Y [Mpc/h]')
            fig.colorbar(im, ax=axs[i,j])
            axs[i,-1].set_title(r'$\Omega$=%.3f'%(Om))
            axs[i,-1].loglog(ks, ps*ks**3/2/np.pi**2, c=f'C{j}', label=r'Random_seed=%d'%iSeed, zorder=3)
        ps, ks = grf_mod['PS']['P'], grf_mod['PS']['k']
        axs[i,-1].loglog(ks, ps*ks**3/2/np.pi**2, c='k', label=r'CLASSemu', zorder=1)
        axs[i,-1].axis([8e-3,3,3e-3,4])
        axs[i,-1].set_xlabel('k [h/Mpc]')
        axs[i,-1].set_ylabel(r'$\Delta^2$ [h/Mpc]')
    plt.tight_layout()
    plt.show()
            