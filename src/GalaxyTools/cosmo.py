"""

FUNCTIONS RELATED TO COSMOLOGY

"""
import numpy as np
from scipy.integrate import cumtrapz, trapz, quad
from scipy.interpolate import splrep,splev
from .constants import rhoc0,c
from .run_BoltzmannSolver import *

def rhoc_of_z(z,param):
    """
    Redshift dependence of critical density
    (in comoving units where rho_b=const; same as in AHF)
    """
    Om = param.cosmo.Om
    Ol = 1.0-Om
    return rhoc0*(Om*(1.0+z)**3.0 + Ol)/(1.0+z)**3.0


def hubble(z,param):
    """
    Hubble parameter
    """
    Om = param.cosmo.Om
    Ol = 1.0-Om
    H0 = 100.0*param.cosmo.h0
    return H0 * (Om*(1+z)**3 + (1.0 - Om - Ol)*(1+z)**2 + Ol)**0.5
    

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


def delta_comoving_distance(z0,z1,param):
    """
    Comoving distance between z0 and z1
    if z0 and z1 are close together (no integral)
    """
    zh = (z0+z1)/2
    return (z1-z0)*c/hubble(zh,param)


def T_cmb(z,param):
    """
    CMB temperature
    """
    Tcmb0 = param.cosmo.Tcmb
    return Tcmb0*(1+z)


def read_powerspectrum(param):
    """
    Linear power spectrum from file
    """
    try:
        names='k, P'
        PS = np.genfromtxt(param.file.ps,usecols=(0,1),comments='#',dtype=None, names=names)
    except:
        # print(param.file.ps)
        if param.file.ps.lower()=='camb':
            r = run_camb(param)
        elif param.file.ps.lower()=='class':
            class_ = run_class(param)
            PS = {'k': class_.k, 'P': class_.pk_lin}
        else:
            print('Either choose between CAMB or CLASS Boltmann Solvers or provide a file containing the linear power spectrum.')
            PS = None
    return PS

