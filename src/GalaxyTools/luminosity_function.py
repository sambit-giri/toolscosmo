import numpy as np
from scipy.integrate import quad, simps

from . import constants as const
from .source_model import fstar, fstar_tilde, fesc, fesc_mean, collapsed_fraction, eps_xray, eps_lyal
from .mass_accretion import mass_accretion

def mass_fct(param, output={}):
    zz, mm, var, dlnvardlnm, dndlnm, bias, fcoll_xray, dfcolldz_xray, sfrd_from_fcoll_xray = collapsed_fraction('xray',param)
    out = {'z': zz,
            'm': mm,   # h^-1 Msun
            'var': var, 
            'dlnvardlnm': dlnvardlnm, 
            'dndlnm': dndlnm, 
            'bias': bias
            }
    output.update(out)
    return output

def mass_accr(param, output={}):
    try:
        M_accr, dMdt_accr = mass_accretion(output,param)
    except:
        hmf = mass_fct(param)
        M_accr, dMdt_accr = mass_accretion(hmf,param)
    out = {'M_accr': M_accr,      # h^-1 Msun
           'dMdt_accr': dMdt_accr # h^-1 Msun yr^-1
            }
    output.update(out)
    return output

def absolute_UV_magnitude(param, M0=51.6, kappa=1.15e-28, output={}):
    # M0 = 51.6
    # kappa  = 1.15e-28  # Msun yr^-1 /(erg s^-1 Hz^-1)
    hmf = mass_fct(param)
    output.update(hmf)
    output.update(mass_accr(param,output=output))
    fstars = fstar(output['z'], output['m'], 'xray', param)
    output.update({'fstar': fstars})
    M_AB = M0 - 2.5*(np.log10(fstars) + np.log10(param.cosmo.Ob/param.cosmo.Om) +\
        np.log10(output['dMdt_accr']) - np.log10(kappa) )
    output.update({'M_AB': M_AB})
    return output

def UV_luminosity(param, output={}):
    try: M_AB = output['M_AB']
    except:
        output = absolute_UV_magnitude(param, M0=51.6, kappa=1.15e-28, output=output)
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
    return output 




