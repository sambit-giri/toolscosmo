import numpy as np 
from scipy.interpolate import splrep,splev
from time import time 
import warnings

# Third-party libraries
# import pyhmcode as hmcode

import camb
# CAMB verbosity level
camb.set_feedback_level(0)

def wdm_transfer_function(k, wdm_mass, h0=0.67, Owdm=0.25):
    '''
    This is a simple approximation for the WDM transfer function from 1112.0330
    '''
    mu = 1.12
    alpha = 0.049 * (wdm_mass / 1.0)**(-1.11) * (Owdm/0.25)**0.11 * (h0/0.7)**(1.22) #Mpc/h
    return (1 + (alpha * k)**(2*mu))**(-5/mu)

def run_camb(param, **info):
    tstart = time()
    if param.code.verbose: print('Using CAMB to estimate linear power spectrum.')

    # Cosmological parameters
    h  = param.cosmo.h0
    omc = param.cosmo.Om - param.cosmo.Ob
    omb = param.cosmo.Ob 
    omk = 0.0 #param.cosmo.Ok # Assuming flat cosmology
    mnu = param.cosmo.mnu
    ns = param.cosmo.ns 
    As = param.cosmo.As 
    if As is None and param.cosmo.s8:
        print(f'Provide As as normalisation with sigma8 not implemented.')
        return None

    # Redshifts
    # zmin = param.code.zmin
    # zmax = param.code.zmax
    # nz = param.code.Nz 
    # zs = np.linspace(zmin, zmax, nz)
    zs = info.get('z', [0.0])

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
    
    # Dark energy
    if param.DE.name.lower() in ['lcdm', 'lambda']:
        pass
    elif param.DE.name.lower() in ['cpl', 'w0wa']:
        w0 = param.DE.w0 
        wa = param.DE.wa 
        if (w0 < -1 - 1e-6 or 1 + w0 + wa < - 1e-6):
            wa = 0.0
        p.set_dark_energy(w=w0, wa=wa)
        if param.code.verbose: print(f'{param.DE.name}: w0,wa={w0},{wa}')
    elif param.DE.name.lower() in ['axioneffectivefluid', 'axion_effective_fluid', 'ula', 'ultralightaxion', 'ultra_light_axion']:
        n   = param.DE.n
        w_n = (n-1)/(n+1) if param.DE.w_n is None else param.DE.w_n
        fde_zc = param.DE.fde_zc
        zc = param.DE.zc 
        theta_i = param.DE.theta_i
        p.DarkEnergy =  camb.dark_energy.AxionEffectiveFluid(w_n=w_n, 
                                                             fde_zc=fde_zc,
                                                             zc=zc,
                                                             theta_i=theta_i,)
    else:
        print(f'{param.DE.name} is an unknown dark energy model for CAMB.')
    p.set_initial_power(camb.InitialPowerLaw(As=As, ns=ns))

    # Dark matter
    if param.DM.name.lower() in ['lcdm']:
        Tk_wdm = lambda kh: 1.0
    elif param.DM.name.lower() in ['wdm', 'warm_dark_matter']:
        wdm_mass = param.DM.m_wdm
        Tk_wdm = lambda kh: wdm_transfer_function(kh, wdm_mass)
    else:
        print(f'{param.DM.name} is an unknown dark matter model for CAMB.')

    p.set_matter_power(redshifts=zs, kmax=k_max, nonlinear=True)

    # Compute CAMB results
    p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    k_h, z, pk_h = r.get_matter_power_spectrum(
            minkh=param.code.kmin, 
            maxkh=param.code.kmax, 
            npoints=200, #npoints,
            var1=7,
            var2=7,
        )
    # ks, zs, Pk_lin = r.get_linear_matter_power_spectrum(nonlinear=False)
    # Pk_nl_CAMB_interpolator = r.get_matter_power_interpolator()
    # Pk_nl = Pk_nl_CAMB_interpolator.P(zs, ks, grid=True)
    if param.code.verbose: 
        print(f'sigma_8={r.get_sigma8_0():.3f}')
        print('CAMB runtime: {:.2f} s'.format(time()-tstart))
    out = {
        'k': k_h.squeeze(), 
        'P': pk_h.squeeze()*Tk_wdm(k_h), 
        'results': r,
        }
    # ## get dictionary of CAMB power spectra
    # powers =r.get_cmb_power_spectra(p, CMB_unit='muK')
    # # for name in powers: print(name)
    # out['CMB_Cls'] = powers
    return out

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
        # class_module.struct_cleanup()
        # class_module.empty()
        self.class_module = class_module
        if self.verbose: print('CLASS runtime: {:.2f} s'.format(time()-tstart))

        if self.save_data:
            np.savetxt(self.save_data, self.full_Pk.T)

def run_class(param, **info):
    inputs_class = info.get('inputs_class', None)
    if param.code.verbose: print('Using CLASS to estimate linear power spectrum.')
    cosmo = {
            'Omega_b': param.cosmo.Ob, 
            'h'  : param.cosmo.h0, 
            'n_s': param.cosmo.ns, 
            'tau_reio': param.cosmo.tau_reio,
            'YHe': param.cosmo.YHe,
            'N_ur': param.cosmo.N_ur,
            #'N_ncdm': 1,
            #'m_ncdm': param.cosmo.mnu,
             }
    if param.cosmo.As is not None: cosmo['A_s'] = param.cosmo.As
    else: cosmo['sigma8'] = param.cosmo.s8
    if inputs_class is None: 
        inputs_class = {
                        'P_k_max_h/Mpc': param.code.kmax, 
                        'z_max_pk': param.code.zmax, 
                        'output': 'mPk', 
                        #'k_per_decade_for_pk': info.get('k_per_decade_for_pk', 10),
                        #'tol_perturbations_integration': info.get('tol_perturbations_integration', 1e-6),
                        }

    # Dark Energy
    if param.DE.name.lower() in ['cpl', 'w0wa']:
        w0, wa = param.DE.w0 , param.DE.wa
        if param.code.verbose: print(f'{param.DE.name}: w0,wa={w0},{wa}')
        inputs_class['fluid_equation_of_state'] = 'CLP'
        inputs_class['w0_fld'] = w0 
        inputs_class['wa_fld'] = wa
        inputs_class['cs2_fld'] = 1
        inputs_class['Omega_Lambda'] = 0.0
    
    # Dark matter
    if param.DM.name.lower() in ['lcdm', 'cold_dark_matter']:
        if param.code.verbose: print(f'{param.DM.name}')
        inputs_class['Omega_cdm'] = (param.cosmo.Om-param.cosmo.Ob)
    elif param.DM.name.lower() in ['wdm', 'warm_dark_matter']:
        if param.code.verbose: print(f'{param.DM.name}: {param.DM.m_wdm} keV')
        inputs_class['ncdm_fluid_approximation'] = 3
        inputs_class['N_ncdm'] = 1
        inputs_class['m_ncdm'] = M_sterile_nu(param)*1000 # eV 
        inputs_class['Omega_ncdm'] = (param.cosmo.Om-param.cosmo.Ob)
        inputs_class['Omega_cdm'] = 0.0
    elif param.DM.name.lower() in ['cwdm', 'cold_warm_dark_matter', 'wcdm', 'warm_cold_dark_matter']:
        if param.code.verbose: print(f'{param.DM.name}: {param.DM.m_wdm} keV')
        assert 0<=param.DM.f_wdm<=1, f'The value for fraction of non-cold dark matter should be between 0 and 1, but the value provided is {param.DM.f_wdm}'
        inputs_class['ncdm_fluid_approximation'] = 3
        inputs_class['N_ncdm'] = 1
        inputs_class['m_ncdm'] = M_sterile_nu(param)*1000 # eV 
        inputs_class['Omega_ncdm'] = (param.cosmo.Om-param.cosmo.Ob)*param.DM.f_wdm
        inputs_class['Omega_cdm'] = (param.cosmo.Om-param.cosmo.Ob)*(1-param.DM.f_wdm)
    else:
        print(f'{param.DM.name} is an unknown dark matter model for CLASS.')

    inputs_class.update(info)
    # print(inputs_class.keys())

    k = 10**np.linspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
    class_ = ClassModule(cosmo, k=k, inputs_class=inputs_class, verbose=param.code.verbose)
    class_.compute_Plin()
    return class_

def M_sterile_nu(param):
	"""
	Sterile neutrino mass.

	Parameters
	----------
	param: Bunch
		Object containing parameter values. WDM mass in keV.
	"""
	Mwdm = param.DM.m_wdm
	Owdm = (param.cosmo.Om-param.cosmo.Ob)*param.cosmo.h0**2
	m_nu  = 4.43*Mwdm**(4./3)*(Owdm/0.1225)**(-1./3)
	return m_nu 

def run_bacco(param, **info):
    try:
        import baccoemu
        warnings.filterwarnings("ignore")
    except:
        print('To use this function, install baccoemu that can be found at:')
        print('https://baccoemu.readthedocs.io/')
        return None 
    
    tstart = time()
    if param.code.verbose: print('Using BACCO emulator to estimate linear power spectrum.')
    
    param_bacco = {
            #'omega_cold'    :  0.315,
            #'sigma8_cold'   :  0.83, # if A_s is not specified
            'omega_matter'   : param.cosmo.Om,
            'omega_baryon'  :  param.cosmo.Ob,
            'ns'            :  param.cosmo.ns,
            'hubble'        :  param.cosmo.h0,
            'neutrino_mass' :  0.0,
            # 'w0'            : -1.0,
            # 'wa'            :  0.0,
            'expfactor'     :  1
        }
    if param.cosmo.As is not None:
        param_bacco['A_s'] = param.cosmo.As 
    else:
        param_bacco['sigma8_cold'] = param.cosmo.s8 
    
    if param.DE.name.lower() in ['lcdm']:
        param_bacco['w0'] = -1.0
        param_bacco['wa'] = 0.0
    elif param.DE.name.lower() in ['cpl', 'w0wa']:
        param_bacco['w0'] = param.DE.w0 
        param_bacco['wa'] = param.DE.wa 
    else:
        print(f'{param.DE.name} is an unknown dark energy model.')
    
    emulator = baccoemu.Matter_powerspectrum(verbose=False)
    kmin = param.code.kmin if param.code.kmin>0.0001 else 0.0001
    kmax = param.code.kmax if param.code.kmax<50 else 50.0 
    k_i = np.logspace(np.log10(kmin), np.log10(kmax), num=100)
    k_h, pk_lin_total = emulator.get_linear_pk(k=k_i, cold=False, **param_bacco)

    if param.code.verbose: print('BACCO emulator runtime: {:.2f} s'.format(time()-tstart))
    return {'k': k_h, 'P': pk_lin_total}