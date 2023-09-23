import numpy as np 
from scipy.interpolate import splrep,splev
from time import time 

# Third-party libraries
# import pyhmcode as hmcode

import camb
# CAMB verbosity level
camb.set_feedback_level(0)

def run_camb(param, **info):
    # Cosmological parameters
    h  = param.cosmo.h0
    omc = param.cosmo.Om 
    omb = param.cosmo.Ob 
    omk = param.cosmo.Ok
    mnu = param.cosmo.mnu
    w0  = param.cosmo.w0 
    wa = param.cosmo.wa 
    ns = param.cosmo.ns 
    As = param.cosmo.As 

    # Redshifts
    zmin = param.code.zmin
    zmax = param.code.zmax
    nz = param.code.Nz 
    zs = np.linspace(zmin, zmax, nz)

    # Wavenumbers [h/Mpc]
    k_max = param.code.kmax

    # Use CAMB to get the linear power spectrum #

    # Get linear power spectrum from CAMB
    p = camb.CAMBparams(WantTransfer=True, 
                        WantCls=False, 
                        Want_CMB_lensing=False, 
                        DoLensing=False,
                        NonLinearModel=camb.nonlinear.Halofit(halofit_version='mead2020'),
                        )
    p.set_cosmology(
                H0=h*100,
                ombh2=omb*h**2,
                omch2=omc*h**2,
                omk=omk,
                num_massive_neutrinos=1,
                mnu=mnu,
                )
    p.set_dark_energy(w=w0, wa=wa)
    p.set_initial_power(camb.InitialPowerLaw(As=As, ns=ns))
    p.set_matter_power(redshifts=zs, kmax=k_max, nonlinear=True)

    # Compute CAMB results
    r = camb.get_results(p)
    # ks, zs, Pk_lin = r.get_linear_matter_power_spectrum(nonlinear=False)
    # Pk_nl_CAMB_interpolator = r.get_matter_power_interpolator()
    # Pk_nl = Pk_nl_CAMB_interpolator.P(zs, ks, grid=True)
    return r

class ClassModule:
    '''
    Running the classy, the python wrapper of class.

    Class takes the following cosmology:
    {'omega_cdm': Om_cdm*h**2, 'omega_b': Om_b*h**2, 'ln10^{10}A_s': 'ln10^{10}A_s', 'n_s':n_s, 'h':h}
    '''
    def __init__(self, cosmo, z=0, k=10**np.linspace(-5,2.75,300), inputs_class=None, lin=True, non_lin=False, save_data=False, verbose=True):
        self.inputs_class = inputs_class if inputs_class is not None else {'P_k_max_h/Mpc': 50 if 50>k.max() else k.max(), 'z_max_pk': 0.5 if z<0.5 else z, 'non linear': 'halofit', 'output': 'mPk', 'k_per_decade_for_pk': 10}
        self.cosmo = cosmo
        self.z = z
        self.k = k 
        self.verbose   = verbose
        self.lin       = lin 
        self.non_lin   = non_lin
        self.save_data = save_data 
        try:
            from classy import Class
        except:
            print('Install classy to use this module.')
            print('See https://github.com/lesgourg/class_public/wiki/Installation')

    def compute_Plin(self, cosmo=None, z=None, k=None):
        tstart = time()
        if cosmo is not None: self.cosmo = cosmo
        if z is not None: self.z = z 
        if k is not None: self.k = k 

        from classy import Class

        # instantiate Class
        class_module = Class()

        # set cosmology
        class_module.set(self.cosmo)

        # set basic configurations for Class
        class_module.set(self.inputs_class)

        # compute the important quantities
        class_module.compute()
        # print('s8, ns =', class_module.sigma8(), class_module.n_s())

        # Class needs k is in Mpc^-1
        k_module = 10**np.linspace(-5,np.log10(self.inputs_class['P_k_max_h/Mpc']),500) * class_module.h()
        self.full_Pk  = self.k

        # calculate the non-linear matter power spectrum
        if self.non_lin:
            pk_non = np.array([class_module.pk(ki, self.z) for ki in k_module]) * class_module.h()**3
            self.pk_non = splev(self.k*class_module.h(), self.pk_non_tck)
        
        # calculate the linear matter power spectrum
        if self.lin:
            pk_lin = np.array([class_module.pk_lin(ki, self.z) for ki in k_module]) * class_module.h()**3
            pk_lin_tck  = splrep(np.log10(k_module),np.log10(pk_lin))
            self.pk_lin = 10**splev(np.log10(self.k*class_module.h()), pk_lin_tck)
            self.full_Pk = np.vstack((self.full_Pk,self.pk_lin))

        # empty Class module - unnecessarily accumulates memory
        class_module.struct_cleanup()
        class_module.empty()
        # self.class_module = class_module
        if self.verbose: print('CLASS runtime: {:.2f} s'.format(time()-tstart))

        if self.save_data:
            np.savetxt(self.save_data, self.full_Pk.T)

def run_class(param, **info):
    inputs_class = info.get('inputs_class', None)
    if param.code.verbose: print('Using CLASS to estimate linear power spectrum.')
    cosmo = {
            'omega_cdm': (param.cosmo.Om-param.cosmo.Ob)*param.cosmo.h0**2, 
            'omega_b': param.cosmo.Ob*param.cosmo.h0**2, 
            'h': param.cosmo.h0, 
            'n_s': param.cosmo.ns, 
            'tau_reio': param.cosmo.tau_reio,
            'YHe': param.cosmo.YHe,
            #'N_ur': 3.044,
            #'N_ncdm':1,
            #'m_ncdm': param.cosmo.mnu,
             }
    if param.cosmo.As is not None: cosmo['A_s'] = param.cosmo.As
    else: cosmo['sigma8'] = param.cosmo.s8
    if inputs_class is None: 
        inputs_class = {
                        'P_k_max_h/Mpc': param.code.kmax, 
                        'z_max_pk': param.code.zmax, 
                        'output': 'mPk', 
                        'k_per_decade_for_pk': info.get('k_per_decade_for_pk', 10)
                        }
    k = 10**np.linspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
    class_ = ClassModule(cosmo, k=k, inputs_class=inputs_class, verbose=param.code.verbose)
    class_.compute_Plin()
    return class_

def _class_run(k: float, omega_cdm: float, z: float = 4.66, z0: float = 0.0) -> float:
    '''
    Function the factor 1 + q(k,z).

    Inputs
    ------
    k (float) - the wavenumber

    omega_cdm (float) - cold dark matter component, this is, omega_cdm h^2

    z (float) - the redshift at which the power spectrum is calculated (default : 4.66)

    z0 (float) - the reference redshift (default: 0.0)

    Returns
    -------
    ratio (float) - the factor 1 + q(k,z)
    '''

    cosmo = {'omega_cdm': omega_cdm, 'omega_b': par[1], 'ln10^{10}A_s': par[2], 'n_s': par[3], 'h': par[4]}

    # instantiate Class
    class_module = Class()

    # set cosmology
    class_module.set(cosmo)

    # set basic configurations for Class
    class_module.set(inputs_class)

    # compute the important quantities
    class_module.compute()

    # k is in Mpc^-1
    # calculate the non-linear matter power spectrum
    pk_non = class_module.pk(k * par[4], z)
    
    # calculate the linear matter power spectrum
    pk_lin = class_module.pk_lin(k * par[4], z0)

    # get the factor A
    a_fact = class_module.scale_independent_growth_factor(z)**2

    # calculate ratio
    ratio = pk_non / (a_fact * pk_lin)

    # empty Class module - unnecessarily accumulates memory
    class_module.struct_cleanup()
    class_module.empty()

    return ratio
