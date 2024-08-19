import numpy as np
from scipy.integrate import cumtrapz, trapz, quad
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt 
import param 

c        = 2.99792e5                # Speed of light [km/s]
hP       = 4.1357e-15               # Planck constant [eV/Hz]
m_p      = 8.4119e-58               # Proton mass in [Msun]
kB       = 1.380649e-16             # Boltzmann constsant [erg/K]
rhoc0    = 2.755e11                 # Critical density at z=0 [h^2 Msun/Mpc^3]

def Ez_model(param):
    """
    Normalised Hubble parameter.
    Exotic dark energy models can be defined here.
    """
    Om = param.cosmo.Om
    Ogamma = param.cosmo.Ogamma
    Ol = 1.0-Om-Ogamma # Flat universe assumption
    if param.DE.name.lower()=='wcdm':
        w  = param.DE.w
        Ez = lambda z: (Om*(1+z)**3 + Ogamma*(1+z)**4 + Ol*(1+z)**(3*(1+w)))**0.5
    elif param.DE.name.lower()=='growing_neutrino_mass':
        Onu  = param.DE.Onu
        Oede = param.DE.Oede
        Ods0 = Ol #1-param.cosmo.Om
        z2a  = lambda z: 1/(1+z)
        Ods1 = lambda z: (Ods0*z2a(z)**3+2*Onu*(z2a(z)**1.5-z2a(z)**3))/(1-Ods0*(1-z2a(z)**3)+2*Onu*(z2a(z)**1.5-z2a(z)**3))
        Ods  = np.vectorize(lambda z: Ods1(z) if Oede<Ods1(z)<1 else Oede)
        Ez = lambda z: (Om*z2a(z)**-1/(1-Ods(z)))**0.5
    else:
        Ez = lambda z: (Om*(1+z)**3 + Ogamma*(1+z)**4 + Ol)**0.5
    return Ez

def hubble(z,param):
    """
    Hubble parameter
    """
    H0 = 100.0*param.cosmo.h0
    Ez = Ez_model(param)
    return H0 * Ez(z)
    

def growth_factor(z, param):
    """
    Growth factor from Longair textbook (Eq. 11.56)
    z: array of redshifts from zmin to zmax
    """
    Om = param.cosmo.Om

    D0 = hubble(0,param) * (5.0*Om/2.0) * quad(lambda a: (a*hubble(1/a-1,param))**(-3), 0.01, 1, epsrel=5e-3, limit=100)[0]
    Dz = []
    for i in range(len(z)):
        Dz += [hubble(z[i],param) * (5.0*Om/2.0) * quad(lambda a: (a*hubble(1/a-1,param))**(-3), 0.01, 1/(1+z[i]), epsrel=5e-3, limit=100)[0]]
    Dz = np.array(Dz)
    return Dz/D0


def comoving_distance(z,param):
    """
    Comoving distance between z[0] and z[-1]
    """
    return cumtrapz(c/hubble(z,param),z,initial=0)  # [Mpc]

def luminosity_distance(z,param):
    """
    Luminosity distance between z[0] and z[-1]
    """
    return comoving_distance(z,param)*(1+z)         # [Mpc]

distance_modulus = lambda z, par: 5*np.log10(luminosity_distance(z,par)/10)+25

## set parameters
par_wcdm = param.par('wcdm')
par_growing_nu = param.par('growing_neutrino_mass')

zs = 10**np.linspace(-2,np.log10(7),100)

## Plot

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Normalised Hubble', fontsize=15)
par_wcdm.DE.w = -1.0
ax.loglog(zs, Ez_model(par_wcdm)(zs), lw=3, ls='-', label='$\omega={}$,WDM'.format(par_wcdm.DE.w))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.1
ax.loglog(zs, Ez_model(par_growing_nu)(zs), lw=3, ls='--', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.2
ax.loglog(zs, Ez_model(par_growing_nu)(zs), lw=3, ls=':', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$E(z)$', fontsize=15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Comoving distance', fontsize=15)
par_wcdm.DE.w = -1.0
ax.loglog(zs, comoving_distance(zs,par_wcdm), lw=3, ls='-', label='$\omega={}$,WDM'.format(par_wcdm.DE.w))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.1
ax.loglog(zs, comoving_distance(zs,par_growing_nu), lw=3, ls='--', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.2
ax.loglog(zs, comoving_distance(zs,par_growing_nu), lw=3, ls=':', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$D_L(z)$', fontsize=15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Distance modulus', fontsize=15)
par_wcdm.DE.w = -1.0
ax.plot(zs, distance_modulus(zs,par_wcdm), lw=3, ls='-', label='$\omega={}$,WDM'.format(par_wcdm.DE.w))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.1
ax.plot(zs, distance_modulus(zs,par_growing_nu), lw=3, ls='--', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
par_growing_nu.DE.Onu  = 0.2
par_growing_nu.DE.Oede = 0.2
ax.plot(zs, distance_modulus(zs,par_growing_nu), lw=3, ls=':', 
            label=r'$\Omega_\nu={}$,$\Omega_e={}$'.format(par_growing_nu.DE.Onu,par_growing_nu.DE.Oede))
ax.legend()
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$\mu(z)$', fontsize=15)
plt.tight_layout()
plt.show()