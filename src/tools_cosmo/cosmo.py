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

def ez_grownu(z, om, omega_rad, omega_nu, oe):
        # om, omega_nu, oe, h0 = theta
        #omega_rad = 2.469e-5 * (H0/100.)**-2. * (1+0.2271*3.04)
        rad_term = omega_rad * (1+z)**4.
        ods = 1 - om - omega_rad
        a = 1./(1+z)
        aval = (1-oe)*(ods - 2*omega_nu)
        cval = oe*(ods - 1)
        bval = 2*omega_nu*(1 - oe)
        nume = -bval + np.sqrt(bval**2. - 4.*aval*cval)
        denom = 2.*aval
        a_trans = pow(nume/denom, 2./3)
        mat_term = om*(1+z)**3.
        # print(omega_nu,oe,a)
        if a > a_trans:
            ttop = ods*a**3. + 2*omega_nu*(pow(a, 3./2) - a**3.)
            tbot = 1 - ods *(1 - a**3) + 2 * omega_nu * (pow(a, 3./2) - a**3)
            ods_a = ttop/tbot
        elif a <=a_trans:
            ods_a = oe
        ezsq = (mat_term + rad_term)/(1-ods_a)
        return np.sqrt(ezsq) #1./np.sqrt(ezsq)

def Ez_growing_nu(z, Om0=0.315, Ok0=0.0, Or0=5.4e-5, Onu0=0.01, Oe0=0.01):
     return np.vectorize(ez_grownu)(z, Om0, Or0, Onu0, Oe0)

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
        # Onu  = param.DE.Onu
        # Oede = param.DE.Oede
        # Ods0 = Ol #1-param.cosmo.Om
        # z2a  = lambda z: 1/(1+z)
        # Ods1 = lambda z: (Ods0*z2a(z)**3+2*Onu*(z2a(z)**1.5-z2a(z)**3))/(1-Ods0*(1-z2a(z)**3)+2*Onu*(z2a(z)**1.5-z2a(z)**3))
        # Ods  = np.vectorize(lambda z: Ods1(z) if Oede<Ods1(z)<1 else Oede)
        # Ez = lambda z: Om*z2a(z)**-1/(1-Ods(z))
        Ez = lambda z: Ez_growing_nu(z, Om0=Om, Ok0=0.0, Or0=Ogamma, Onu0=param.DE.Onu, Oe0=param.DE.Oede)
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

def distance_modulus(z,param): 
    """
    Distance modulus between z[0] and z[-1]
    """
    return 5*np.log10(luminosity_distance(z,param)/10)+25


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


def read_powerspectrum(param, **info):
    """
    Linear power spectrum from file
    """
    try:
        names='k, P'
        PS = np.genfromtxt(param.file.ps,usecols=(0,1),comments='#',dtype=None, names=names)
    except:
        PS = calc_Plin(param, **info)
    return PS

def calc_Plin(param, **info):
    # print(param.file.ps)
    if param.file.ps.lower()=='camb':
        r = run_camb(param)
    elif param.file.ps.lower()=='class':
        class_ = run_class(param, **info)
        PS = {'k': class_.k, 'P': class_.pk_lin}
    else:
        print('Either choose between CAMB or CLASS Boltmann Solvers or provide a file containing the linear power spectrum.')
        PS = None 
    return PS 

def get_Plin(param, **info):
    return read_powerspectrum(param, **info)

def wf(y,param):
    """
    Window function
    """
    window = param.mf.window
    if (window=='tophat'):
        w = 3.0*(np.sin(y) - y*np.cos(y))/y**3.0
        w[y>100] = 0
    elif (window=='sharpk'):
        w = np.ones(y)
        w[y>1]=0
    elif (window=='gaussian'):
        w = np.exp(-y**2.0/2.0)
    elif (window=='smoothk'):
        beta = param.mf.beta
        w = 1/(1+y**beta)
    else:
        print("ERROR: undefined window function!")
        exit()
    return w


def dwf(y,param):
    """
    Derivative Of window function
    dwf = dwf(kR)/dln(kR)
    """
    window = param.mf.window
    if (window == 'tophat'):
        dw = 3.0*((y**2.0 - 3.0)*np.sin(y) + 3.0*y*np.cos(y))/y**3.0
        dw[y>100] = 0
    elif (window == 'sharpk'):
        """
        delta function (must be accounted for in main code)
        """
        dw = 0.0
    elif (window == 'gaussian'):
        dw = - y**2.0*np.exp(-y**2.0/2.0)
    elif (window=='smoothk'):
        beta = param.mf.beta
        dw = - beta*y**beta/(1+y**beta)**2
    else:
        print("ERROR: undefined window function!")
        exit()
    return dw


def variance(param):
    """
    variance of density perturbations at z=0
    """
    #window function
    window = param.mf.window

    #read in linear power spectrum
    try:
        PS = param.cosmo.plin
        kmin  = min(PS['k'])
        kmax  = max(PS['k'])
    except:
        # names='k, P'
        # PS = np.genfromtxt(param.file.psfct,usecols=(0,1),comments='#',dtype=None, names=names)
        PS = read_powerspectrum(param)
        kmin  = min(PS['k'])
        kmax  = max(PS['k'])

    #set binning
    Nrbin = param.code.Nrbin
    rmin  = param.code.rmin
    rmax  = param.code.rmax
    rbin  = np.logspace(np.log(rmin),np.log(rmax),Nrbin,base=np.e)

    #calculate variance and derivative
    if (window == 'tophat' or window == 'gaussian' or window == 'smoothk'):
        var = []
        dlnvardlnr = []
        for i in range(Nrbin):
            #var
            itd_var = PS['k']**2 * PS['P'] * wf(PS['k']*rbin[i],param)**2
            var += [trapz(itd_var,PS['k'])/(2*np.pi**2)]
            #dlnvar/dlnr
            itd_dvar = PS['k']**2 * PS['P'] * wf(PS['k']*rbin[i],param) * dwf(PS['k']*rbin[i],param)
            dlnvardlnr += [2*np.trapz(itd_dvar,PS['k'])/(2*np.pi**2*var[i])]
        var = np.array(var)
        dlnvardlnr = np.array(dlnvardlnr)
    elif (window == 'sharpk'):
        #var
        Plin_tck = splrep(PS['k'],PS['P'])
        kbin = 1/rbin
        kbin = kbin[::-1]
        var = cumtrapz(kbin**2 * splev(kbin,Plin_tck),kbin,initial=1e-5) / (2*np.pi**2)
        var = var[::-1]
        #dlnvar/dlnr
        dlnvardlnr = -1/(2*np.pi**2*var) * splev(1/rbin,Plin_tck)/rbin**3.0
    else:
        print("ERROR: undefined window function!")
        exit()

    #write varfct to file
    try:
        np.savetxt(param.file.varfct, np.vstack((rbin, var, dlnvardlnr)).T)
    except IOError:
        print('IOERROR: cannot write varfct!')
        exit()

    return rbin, var, dlnvardlnr

