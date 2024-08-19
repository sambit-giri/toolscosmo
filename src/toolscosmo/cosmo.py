"""

FUNCTIONS RELATED TO COSMOLOGY

"""
import numpy as np
from astropy import cosmology, units
from.scipy_func import *

from .constants import rhoc0,c
from .run_BoltzmannSolver import *
from .emulate_BoltmannSolver import *

def prepare_cosmo_solver(param):
    if param.code.verbose: print('Preparing cosmological solvers...')
    if param.cosmo.solver.lower()=='astropy':
        solver_estimator = astropy_cosmo(param).cosmo
        if param.code.verbose: print('astropy will be used.')
    elif param.cosmo.solver.lower()=='camb':
        cosmo_camb = run_camb(param)
        solver_estimator = cosmo_camb['results']
        if param.file.ps.lower()=='camb':
            PS = {'k': cosmo_camb['k'], 'P': cosmo_camb['P']}
            param.file.ps = PS
        else:
            print('CAMB is used for cosmological calculations.')
            print(f'Using CAMB instead of {param.file.ps} for modelling linear power spectrum will avoid running another Boltzmann solver.')
        if param.code.verbose: print('CAMB will be used.')
    elif param.cosmo.solver.lower()=='class':
        cosmo_class = run_class(param)
        solver_estimator = cosmo_class.class_module
        if param.file.ps.lower()=='class':
            PS = {'k': cosmo_class.k, 'P': cosmo_class.pk_lin}
            param.file.ps = PS
        else:
            print('CLASS is used for cosmological calculations.')
            print(f'Using CLASS instead of {param.file.ps} for modelling linear power spectrum will avoid running another Boltzmann solver.')
        if param.code.verbose: print('CLASS will be used.')
    else: 
        solver_estimator = None 
    param.cosmo.solver_estimator = solver_estimator
    if param.code.verbose: print('...done')
    return param

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
    try: 
        cosmo = param.cosmo.solver_estimator
    except:
        param = prepare_cosmo_solver(param)
        cosmo = param.cosmo.solver_estimator

    if param.cosmo.solver.lower()=='astropy':
        Ez = lambda z: cosmo.efunc(z)
        return Ez
    elif param.cosmo.solver.lower()=='camb':
        Ez = lambda z: cosmo.hubble_parameter(z)/cosmo.hubble_parameter(0)
        return Ez
    elif param.cosmo.solver.lower()=='class':
        Ez = np.vectorize(lambda z: cosmo.Hubble(z)/cosmo.Hubble(0))
        return Ez
    elif param.cosmo.solver.lower()=='tools_cosmo':
        pass
    else:
        print(f'{param.cosmo.solver} is unknown and, therefore, set to tools_cosmo.')

    Om = param.cosmo.Om
    Or = param.cosmo.Or
    Ol = 1.0-Om-Or # Flat universe assumption
    # print(param.DE.__dict__)
    Olz = lambda z: Omega_DE(z, param)
    if param.DE.name.lower() in ['wcdm','cpl','lcdm']:
        Ez = lambda z: (Om*(1+z)**3 + Or*(1+z)**4 + Olz)**0.5
    elif param.DE.name.lower()=='growing_neutrino_mass':
        Ez = lambda z: Ez_growing_nu(z, Om0=Om, Ok0=0.0, Or0=Or, Onu0=param.DE.Onu, Oe0=param.DE.Oede)
    else:
        Ez = lambda z: (Om*(1+z)**3 + Or*(1+z)**4 + Ol)**0.5
    return Ez

def hubble(z,param):
    """
    Hubble parameter
    """
    try: 
        cosmo = param.cosmo.solver_estimator
    except:
        param = prepare_cosmo_solver(param)
        cosmo = param.cosmo.solver_estimator

    if param.cosmo.solver.lower()=='astropy':
        return cosmo.H(z).value
    elif param.cosmo.solver.lower()=='camb':
        return cosmo.hubble_parameter(z)
    elif param.cosmo.solver.lower()=='class':
        return np.vectorize(lambda z0: cosmo.Hubble(z0)/cosmo.Hubble(0)*cosmo.h()*100)(z)
    elif param.cosmo.solver.lower()=='tools_cosmo':
        pass
    else:
        print(f'{param.cosmo.solver} is unknown and, therefore, set to tools_cosmo.')

    H0 = 100.0*param.cosmo.h0
    Ez = Ez_model(param)
    return H0 * Ez(z)
    

def growth_factor(z, param):
    """
    Growth factor from Longair textbook (Eq. 11.56).
    Also see arXiv:astro-ph/0006089

    z: array of redshifts from zmin to zmax
    """
    if param.code.Dz_solver.lower() in ['linder2005','linder(2005)','linder (2005)']:
        # print('Linder (2005) fitting function')
        return growth_factor_Linder2005(z, param)
    elif param.code.Dz_solver.lower() in ['solveode','ode']:
        # print('Solving the ODE')
        return growth_factor_solveODE(z, param)
    else:
        # print('Hamilton (2000) fitting function')
        Om = param.cosmo.Om
        D0 = hubble(0,param) * (5.0*Om/2.0) * quad(lambda a: (a*hubble(1/a-1,param))**(-3), 0.01, 1, epsrel=5e-3, limit=100)[0]
        Dz = []
        for i in range(len(z)):
            Dz += [hubble(z[i],param) * (5.0*Om/2.0) * quad(lambda a: (a*hubble(1/a-1,param))**(-3), 0.01, 1/(1+z[i]), epsrel=5e-3, limit=100)[0]]
        Dz = np.array(Dz)
        return Dz/D0

def w_DE(z, param):
    if param.DE.name.lower()=='lcdm':
        w = -1   
    elif param.DE.name.lower()=='wcdm':
        w = param.DE.w 
    elif param.DE.name.lower()=='cpl':
        w = param.DE.w0 + param.DE.wa*z/(1+z)
    else:
        wDE = param.DE.wDE
        if wDE is None: 
            print(f'Dark energy equation of state w(z) for {param.DE.name} should be provided through param.DE.wDE variable.')
        w = wDE(z)
    return w 

def Omega_DE(z, param):
    '''
    Evolution of dark energy density parameter.
    '''
    a = 1/(1+z)
    # Define cosmological parameters
    # H0 = param.cosmo.h0*100    # Hubble constant in km/s/Mpc
    Omega_m = param.cosmo.Om   # Matter density parameter
    Omega_k = param.cosmo.Ok   # Curvature density parameter
    Omega_r = param.cosmo.Or   # Radiation density parameter
    Omega_L = param.cosmo.Ode  # Dark energy density parameter
    if Omega_L is None: Omega_L = 1-Omega_m-Omega_k-Omega_r

    if param.DE.name.lower()=='lcdm':
        w = -1
        return Omega_L * a**(3*(1+w))
    elif param.DE.name.lower()=='wcdm':
        w = param.DE.w 
        return Omega_L * a**(3*(1+w))
    elif param.DE.name.lower()=='cpl':
        w0, wa = param.DE.w0, param.DE.wa
        return Omega_L * a**(3*(1+w0+wa)) * np.exp(-wa*a)
    else:
        wDE = param.DE.wDE
        integrand = lambda a_prime: (1 + wDE(a_prime)) / a_prime
        integral, _ = quad(integrand, a, 1)
        return Omega_L * np.exp(3 * integral)

def growth_factor_solveODE(z, param):
    # Define cosmological parameters
    H0 = param.cosmo.h0*100    # Hubble constant in km/s/Mpc
    Omega_m = param.cosmo.Om   # Matter density parameter
    Omega_k = param.cosmo.Ok   # Curvature density parameter
    Omega_r = param.cosmo.Or   # Radiation density parameter
    Omega_L = param.cosmo.Ode  # Dark energy density parameter
    if Omega_L is None: Omega_L = 1-Omega_m-Omega_k-Omega_r

    w = lambda a: w_DE(1/a-1, param)
    Omega_DE_a = lambda a: Omega_DE(1/a-1, param)
    
    def E(a):
        return np.sqrt(Omega_m * a**-3 + Omega_r * a**-4 + Omega_k * a**-2 + Omega_DE_a(a))

    def dEda(a):
        # Derivative of Hubble parameter w.r.t scale factor
        return - (3/2) * Omega_m * a**-4 / E(a) - 2 * Omega_r * a**-5 / E(a) - Omega_k * a**-3 / E(a) + (3 * (1 + w(a)) * Omega_DE_a(a)) / (2 * a * E(a))

    def diff_Da(y, a):
        # Define the growth factor differential equation
        D, dDda = y
        dD2da2 = - (3/(2 * a**2 * E(a)**2)) * Omega_m * D - (3/a + (1/E(a)) * (dEda(a))) * dDda
        return [dDda, dD2da2]

    # Initial conditions
    a_init = 1e-2  # Initial scale factor (early universe)
    D_init = a_init  # Initial growth factor D(a) ≈ a
    dDda_init = 1.0  # Initial derivative dD/da ≈ 1
    y0 = [D_init, dDda_init]

    # Scale factor range to solve over
    a_vals = np.linspace(a_init, 1, 1000)
    z_vals = 1/a_vals-1
    # Solve the differential equation
    sol = odeint(diff_Da, y0, a_vals)
    # Extract the growth factor solution
    D_vals = sol[:, 0]

    Dz = interp1d(z_vals,D_vals)
    return Dz(z)/Dz(0)

def growth_factor_Linder2005(z, param):
    """
    A fit for growth factor from Linder (2005, PhRvD, 72, 043529)
    that should work for most DDE models.

    z: array of redshifts from zmin to zmax
    """
    Om = param.cosmo.Om
    H0 = hubble(0,param)
    Ha = lambda a: hubble(1/a-1,param)
    Oa = lambda a: Om*a**(-3)/(Ha(a)/H0)**2
    wz = lambda z: w_DE(z, param)
    w1 = wz(1)
    gamma = 0.55+0.05*(1+w1) if w1>=-1 else 0.55+0.02*(1+w1)
    D0 = np.exp(quad(lambda a: (Oa(a)**gamma-1)/a, 0.01, 1, epsrel=5e-3, limit=100)[0])
    Dz = np.array([])
    for i in range(len(z)):
        ln_Da = quad(lambda a: (Oa(a)**gamma-1)/a, 0.01, 1/(1+z[i]), epsrel=5e-3, limit=100)[0]
        Dz = np.append(Dz,np.exp(ln_Da))
    return Dz/D0/(1+z)


def comoving_distance(z,param):
    """
    Comoving distance between z=0 and z.
    """
    try: 
        cosmo = param.cosmo.solver_estimator
    except:
        param = prepare_cosmo_solver(param)
        cosmo = param.cosmo.solver_estimator

    if param.cosmo.solver.lower()=='astropy':
        # cosmo = astropy_cosmo(param).cosmo
        return cosmo.comoving_distance(z).to('Mpc').value 
    elif param.cosmo.solver.lower()=='camb':
        return cosmo.comoving_radial_distance(z)
    elif param.cosmo.solver.lower()=='class':
        return np.vectorize(lambda z0: cosmo.comoving_distance(z0))(z)
    elif param.cosmo.solver.lower()=='tools_cosmo':
        pass
    else:
        print(f'{param.cosmo.solver} is unknown and, therefore, set to tools_cosmo.')
    
    if isinstance(z,list): z = np.array(z)
    # dcom = cumtrapz(c/hubble(z,param),z,initial=0)  # [Mpc]
    # dcom = lambda z: quad(lambda x: c/hubble(x,param), 0, z)[0]
    zspace = lambda z: np.logspace(-3,np.log10(z))
    dcom = lambda z: trapz(c/hubble(zspace(z),param), zspace(z)) if z>0 else 0
    return np.vectorize(dcom)(z)

def luminosity_distance(z,param):
    """
    Luminosity distance between z=0 and z.
    """
    try: 
        cosmo = param.cosmo.solver_estimator
    except:
        param = prepare_cosmo_solver(param)
        cosmo = param.cosmo.solver_estimator

    if param.cosmo.solver.lower()=='astropy':
        # cosmo = astropy_cosmo(param).cosmo
        return cosmo.luminosity_distance(z).to('Mpc').value 
    elif param.cosmo.solver.lower()=='camb':
        return cosmo.luminosity_distance(z)
    elif param.cosmo.solver.lower()=='class':
        return np.vectorize(lambda z0: cosmo.luminosity_distance(z0))(z)
    elif param.cosmo.solver.lower()=='tools_cosmo':
        pass
    else:
        print(f'{param.cosmo.solver} is unknown and, therefore, set to tools_cosmo.')

    if isinstance(z,list): z = np.array(z)
    return comoving_distance(z,param)*(1+z)         # [Mpc]

def distance_modulus(z,param): 
    """
    Distance modulus between z=0 and z.
    """
    try: 
        cosmo = param.cosmo.solver_estimator
    except:
        param = prepare_cosmo_solver(param)
        cosmo = param.cosmo.solver_estimator

    if param.cosmo.solver.lower()=='astropy':
        # cosmo = astropy_cosmo(param).cosmo
        return cosmo.distmod(z).to('mag').value 
    elif param.cosmo.solver.lower()=='camb':
        return 5*np.log10(cosmo.luminosity_distance(z))+25
    elif param.cosmo.solver.lower()=='class':
        D_L = np.vectorize(lambda z0: cosmo.luminosity_distance(z0))(z)
        return 5*np.log10(D_L)+25
    elif param.cosmo.solver.lower()=='tools_cosmo':
        pass
    else:
        print(f'{param.cosmo.solver} is unknown and, therefore, set to tools_cosmo.')

    if isinstance(z,list): z = np.array(z)
    return 5*np.log10(luminosity_distance(z,param))+25


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
        PS = {'k': r['k'], 'P': r['P']}
    elif param.file.ps.lower()=='class':
        class_ = run_class(param, **info)
        PS = {'k': class_.k, 'P': class_.pk_lin}
    elif param.file.ps.lower() in ['bacco', 'baccoemu']:
        PS = run_bacco(param, **info)
    elif param.file.ps.lower() in ['classemu']:
        PS = emulate_camb(param, **info)
    else:
        print('Provide linear power spectrum via param.file.ps')
        print('Option: file, CLASS, CAMB, BACCO')
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

class astropy_cosmo:
    '''
    A class to use astropy as cosmological calculation.
    '''
    def __init__(self, param):
        self.param = param
        cosmo = self.set_model() 

    def set_model(self, param=None):
        if param is None: param = self.param
        Om     = param.cosmo.Om
        Ob     = param.cosmo.Ob
        Or     = param.cosmo.Or
        Ode    = 'flat' if param.cosmo.Ode is None else param.cosmo.Ode
        h0     = param.cosmo.h0 
        if param.DE.name.lower()=='wcdm':
            w  = param.DE.w
            if Ode.lower()=='flat': cosmo = cosmology.FlatwCDM(h0*100, Om, w0=w)
            else: cosmo = cosmology.wCDM(h0*100, Om, Ode, w0=w)
        elif param.DE.name.lower() in ['cpl','w0wa']:
            w0 = param.DE.w0
            wa = param.DE.wa
            # print(w0, wa)
            if Ode.lower()=='flat': cosmo = cosmology.Flatw0waCDM(h0*100, Om, w0=w0, wa=wa)
            else: cosmo = cosmology.w0waCDM(h0*100, Om, Ode, w0=w0, wa=wa)
        elif param.DE.name.lower()=='growing_neutrino_mass':
            cosmo = None
        elif param.DE.name.lower()=='lcdm':
            if Ode.lower()=='flat': cosmo = cosmology.FlatLambdaCDM(h0*100, Om)
            else: cosmo = cosmology.LambdaCDM(h0*100, Om, Ode)
        else:
            cosmo = cosmology.FlatLambdaCDM(h0*100, Om)
            print(f'Flat-LambdaCDM is assumed as {param.DE.name} is unknown.')
        self.cosmo = cosmo 
        return cosmo 
        