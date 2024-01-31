import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import splev, splrep, interp1d
import matplotlib.pyplot as plt

from . import constants as const
from .source_model import fstar, fstar_tilde, fesc, fesc_mean, collapsed_fraction, eps_xray, eps_lyal
from .mass_accretion import mass_accretion
from .cosmo import hubble, get_Plin
from .constants import *

def mass_fct(param, output={}):
    zz, mm, var, dlnvardlnm, dndlnm, bias, fcoll_xray, dfcolldz_xray, sfrd_from_fcoll_xray = collapsed_fraction('lf',param)
    out = {
            'z': zz,
            'm': mm,   # h^-1 Msun
            'var': var, 
            'dlnvardlnm': dlnvardlnm, 
            'dndlnm': dndlnm, 
            'bias': bias
            }
    output.update(out)
    return output

def mass_accr(param, output={}):
    if param.code.verbose: print('MA is modelled with the {} method'.format(param.code.MA))
    try:
        M_accr, dMdt_accr = mass_accretion(output,param)
    except:
        hmf = mass_fct(param)
        output.update(hmf)
        M_accr, dMdt_accr = mass_accretion(hmf,param)
    out = {'M_accr': M_accr,      # h^-1 Msun
           'dMdt_accr': dMdt_accr # h^-1 Msun yr^-1
            }
    output.update(out)
    return output


class UVLF:
    def __init__(self, param, **info):
        if param.cosmo.plin is None: param.cosmo.plin = get_Plin(param, **info)
        self.param = param; # print(self.param.cosmo.Om)
        self.output = {'plin': param.cosmo.plin}

    def Muv(self, M0=51.6, kappa=1.15e-28):
        '''
        See eq (1) in Sahlen & Zackrisson (2021) or 
        eqs (12,13) in Park et al. (2019).
        '''
        param  = self.param
        output = self.output
        # M0 = 51.6
        # kappa  = 1.15e-28  # Msun yr^-1 /(erg s^-1 Hz^-1)
        hmf = mass_fct(param)
        output.update(hmf)
        output.update(mass_accr(param,output=output))
        fstars = fstar(output['z'], output['m'], 'lf', param)
        # fstars[fstars<1e-5] = 0
        output.update({'fstar': fstars})
        # idx_abv = np.argmin(np.abs(np.log10(M0[:,None]/M_accr[i,:])),axis=1)

        # print(f'{param.code.MA} in M_AB')
        # M_accr, dMdt_accr = mass_accretion(output,param)
        M_accr, dMdt_accr = output['M_accr'], output['dMdt_accr']
        dMhdt_dot = np.zeros((output['z'].size,output['m'].size))
        for i,zi in enumerate(output['z']):
            log_dMdt_accr_fct = interp1d(np.log10(M_accr[i,:]), np.log10(dMdt_accr[i,:]), fill_value='extrapolate')
            dMhdt_dot[i,:] = 10**log_dMdt_accr_fct(np.log10(output['m']))
        # dMhdt_dot = param.MA.alpha_EXP * output['m'] * (output['z'][:,None]+1) * hubble(output['z'][:,None],param) * sec_per_yr / km_per_Mpc
        # print(dMhdt_dot.shape)

        M_AB = M0 - 2.5*(np.log10(fstars) 
                       + np.log10(param.cosmo.Ob/param.cosmo.Om) 
                       + np.log10(dMhdt_dot) 
                       #+ np.log10(output['dMdt_accr']) 
                       - np.log10(kappa) - np.log10(param.cosmo.h0)
                )
        output.update({'M_AB': M_AB})
        self.output = output
        return output
    
    def UV_luminosity_SZ21_eq1(self):
        param  = self.param
        output = self.output
        try: M_AB = output['M_AB']
        except:
            output = self.Muv(M0=51.6, kappa=1.15e-28)
            M_AB = output['M_AB']
        zz = output['z']
        mm = output['m']
        dndlnm = output['dndlnm']
        Muv_edges = np.linspace(param.lf.Muv_min,param.lf.Muv_max,param.lf.NMuv+1)
        Muv_mean  = Muv_edges[1:]/2.+Muv_edges[:-1]/2.
        phi_uv = np.zeros((len(zz),len(Muv_mean)))
        W_tophat  = lambda Muv,Ma,Mb: (Ma<=Muv)*(Muv<=Mb)
        p_scatter = lambda M,Mmean: 1./np.sqrt(2*np.pi)/param.lf.sig_M * np.exp(-(M-Mmean)**2/2/param.lf.sig_M**2)
        for i in range(Muv_mean.size):
            Ma, Mb = Muv_edges[i], Muv_edges[i+1]
            Wth = W_tophat(M_AB, Ma, Mb)
            pst = p_scatter(M_AB,Muv_mean[i])
            itg = Wth*pst*dndlnm
            phi_uv[:,i] = param.lf.eps_sys*simps(itg, mm)
        output['uvlf'] = {'Muv_mean': Muv_mean, 'phi_uv': phi_uv}
        self.output = output
        return output 

    def UV_luminosity_def(self, **kwargs):
        param  = self.param
        output = self.output
        try: M_AB = output['M_AB']
        except:
            output = self.Muv(M0=51.6, kappa=1.15e-28)
            M_AB = output['M_AB']
        zz = output['z']
        mm = output['m']
        f_duty = kwargs.get('f_duty')
        if f_duty=='EXP': f_duty = lambda M: np.exp(-param.lf.Mt_sfe/M)
        else: f_duty = lambda M: np.ones_like(M)
        output['f_duty'] = f_duty
        dndlnm = output['dndlnm']*f_duty(output['m'])
        # print(output.keys())
        
        Muv_edges = np.linspace(param.lf.Muv_min,param.lf.Muv_max,param.lf.NMuv+1)
        Muv_mean  = Muv_edges[1:]/2.+Muv_edges[:-1]/2.
        phi_uv = np.zeros((len(zz),len(Muv_mean)))
        dMab = np.diff(M_AB)
        dMh  = np.diff(mm)
        dMhdMab = -dMh[None,:]/dMab #dMh[None,:]/dMab
        for i in range(zz.size):
            dndm_fct   = interp1d(M_AB[i,:], dndlnm[i,:]/mm, fill_value='extrapolate')
            dmdMuv_fct = interp1d(M_AB[i,1:]/2+M_AB[i,:-1]/2, dMhdMab[i,:], fill_value='extrapolate')
            phi_uv_fct  = lambda muv: param.lf.eps_sys*dndm_fct(muv)*dmdMuv_fct(muv)
            phi_uv[i,:] = phi_uv_fct(Muv_mean)
        output['uvlf'] = {'Muv_mean': Muv_mean, 'phi_uv': phi_uv,}
        self.output = output
        return output 

    def UV_luminosity(self, **kwargs):
        # return self.UV_luminosity_SZ21_eq1(**kwargs)
        return self.UV_luminosity_def(**kwargs)




