"""

FUNCTIONS TO CALCULATE THE STAR-FORMATION RATE DENSITY
FROM THE HALO MASS FUNCTION (E.G. GENMASSFCT.PY)

"""

import numpy as np
# import genmassfct as gmf
from . import massfct as gmf

from scipy.integrate import cumtrapz, trapz, quad
from scipy.interpolate import splrep,splev
from scipy.optimize import fsolve
from .cosmo import hubble
from .constants import *

'''
def fstar(zz,m,type_of_flux,param):
    """
    Ross et al 2019 
    """
    M178m_ov_M200c = 1.0 #1.4
    if (type_of_flux=='xray'):
        fstar0 = param.xray.f0_sfe
        fstar = np.full((len(zz),len(m)),fstar0)
        Mcut = 1e8*param.cosmo.h0/M178m_ov_M200c
        fstar[:,np.where(m<Mcut)[0]]=0
    elif (type_of_flux=='lyal'):
        fstar0 = param.lyal.f0_sfe
        fstar = np.full((len(zz),len(m)),fstar0)
        Mcut1 = 1e9*param.cosmo.h0/M178m_ov_M200c
        incr = 4.176470 #7.1/1.7 see Ross19
        fstar[:,np.where(m<Mcut1)[0]]=incr*fstar0
        Mcut2 = 1e8*param.cosmo.h0/M178m_ov_M200c
        fstar[:,np.where(m<Mcut2)[0]]=0
    return fstar
'''

def fstar(zz,m,type_of_flux,param):
    """
    Star formation efficiency: fstar = (dMstar/dt)/(dM/dt)
    fstar = 2*fstar0/((m/Mp)**gamma1 + (m/Mp)**gamma2) * trunc  
    trunc = (1 + (m/Mt)**gamma3)**gamma4    
    """
    if (type_of_flux=='xray'):
        fstar0 = param.xray.f0_sfe
        gamma1 = param.xray.g1_sfe
        gamma2 = param.xray.g2_sfe
        gamma3 = param.xray.g3_sfe
        gamma4 = param.xray.g4_sfe
        Mpivot = param.xray.Mp_sfe
        Mtrunc = param.xray.Mt_sfe
    elif (type_of_flux=='lyal'):
        fstar0 = param.lyal.f0_sfe
        gamma1 = param.lyal.g1_sfe
        gamma2 = param.lyal.g2_sfe
        gamma3 = param.lyal.g3_sfe
        gamma4 = param.lyal.g4_sfe
        Mpivot = param.lyal.Mp_sfe
        Mtrunc = param.lyal.Mt_sfe
    elif (type_of_flux.lower()=='lf'):
        fstar0 = param.lf.f0_sfe*(1+zz)**param.lf.f0_sfe_nu
        gamma1 = param.lf.g1_sfe*(1+zz)**param.lf.g1_sfe_nu
        gamma2 = param.lf.g2_sfe*(1+zz)**param.lf.g2_sfe_nu
        gamma3 = param.lf.g3_sfe*(1+zz)**param.lf.g3_sfe_nu
        gamma4 = param.lf.g4_sfe*(1+zz)**param.lf.g4_sfe_nu
        Mpivot = param.lf.Mp_sfe*(1+zz)**param.lf.Mp_sfe_nu
        Mtrunc = param.lf.Mt_sfe*(1+zz)**param.lf.Mt_sfe_nu
    else:
        print("ERROR: type of flux has to be either lf, xray or lyal!")
        exit()
    Mmin   = param.code.Mdark

    fstar = np.zeros((len(zz),len(m)))
    for i in range(len(zz)):
        try:
            g1, g2, g3, g4 = gamma1[i], gamma2[i], gamma3[i], gamma4[i]
            f0, Mp, Mt = fstar0[i], Mpivot[i], Mtrunc[i]
        except:
            g1, g2, g3, g4 = gamma1, gamma2, gamma3, gamma4
            f0, Mp, Mt = fstar0, Mpivot, Mtrunc

        denom1 = (m/Mp)**g1
        denom2 = (m/Mp)**g2

        trunc  = (1+(Mt/m)**g3)**g4

        fstar[i,:] = 2*f0/(denom1+denom2)*trunc
        fstar[i,np.where(fstar[i,:]>1)] = 1

        fstar[i,np.where(m<Mmin)] = 0
    
    # print(g1, g2, g3, g4, fstar.shape)

    return fstar


def fstar_tilde(m,fstar):
    """
    Stellar-mass to halo-mass ratio: fstar_tilde = Mstar/M
    """
    ftilde = fstar
    #ftilde = cumtrapz(fstar*dMaccdt,z,axis=0,initial=0.0)/m    

    return ftilde


#def fstar_tilde_mean(z,m,dndlnm,type_of_flux,param):
#
#    fs = fstar(z,m,type_of_flux,param)
#    fr = fstar_tilde(z,fs)
#    mean_fstar = [np.trapz(fr[i,:]*dndlnm[i,:]/m,x=m)/np.trapz(dndlnm[i,:]/m,x=m) for i in range(len(z))]
#    mean_fstar = np.array(mean_fstar)
#    return mean_fstar


def fesc(zz,m,param):
    """
    Escape fraction for ionising radiation
    """
    f0   = param.reio.f0_esc
    Mp   = param.reio.Mp_esc
    pl   = param.reio.pl_esc
    Mmin = param.code.Mdark
    
    fesc = np.zeros((len(zz),len(m)))
    for i in range(len(zz)):
        fesc[i,:] = f0 * (Mp/m)**pl
        fesc[i,np.where(fesc[i,:]>1)] = 1

    return fesc


def fesc_mean(z,m,dndlnm,param):
    """
    mean escape fraction per collapsed mass
    """
    fr = fesc(z,m,param)
    mean_fesc = [np.trapz(fr[i,:]*dndlnm[i,:],x=m)/np.trapz(dndlnm[i,:],x=m) for i in range(len(z))]
    #mean_fesc = [np.trapz(fr[i,:]*dndlnm[i,:]/m,x=m)/np.trapz(dndlnm[i,:]/m,x=m) for i in range(len(z))]
    mean_fesc = np.array(mean_fesc)
    
    return mean_fesc

def fstarfesc_mean(z,m,dndlnm,type_of_flux,param):

    fs = fstar(z,m,type_of_flux,param)
    fe = fesc(z,m,param)

    mean_ff = [np.trapz(fe[i,:]*fs[i,:]*dndlnm[i,:],x=m)/np.trapz(dndlnm[i,:],x=m) for i in range(len(z))]
    mean_ff = np.array(mean_ff)

    #import matplotlib.pyplot as plt
    #f = plt.figure(figsize=(6,4),dpi=140)
    #plt.loglog(m, fs[10,:]*fe[10,:], color='black',ls='-')
    #plt.loglog(m, fs[10,:], color='black',ls=':')
    #plt.loglog(m, fe[10,:], color='black',ls=':')
    #plt.ylabel(r'f')
    #plt.xlabel(r'M')
    #plt.axis([1e5,1e12,0.000001,2])
    #plt.show()
    #exit()
    
    return mean_ff

    
def collapsed_fraction(type_of_flux,param):
    """
    Star-formation rate density (uses genmassfct module)
    output:    zbin    = array of redshifts
               fcoll   = fraction of stars in haloes 
               dfcoldz = change of fraction per redshift
               sfrd    = rho_m*dfcoldt (star-formation-rate-density)

    fcoll    = (Om_b/Om_m)/rho_m * int dM * fstar(M) * dn/dlnM 
    dfcolldt = dfcoll/dt = drho_star/dt
    """
    #parameters used below
    Om   = param.cosmo.Om
    Ob   = param.cosmo.Ob
    h0   = param.cosmo.h0

    #parameters of gmf
    # par = gmf.par()
    par = param
    par.file.psfct    = param.file.ps
    # par.window.window = param.mf.window
    par.mf.c          = param.mf.c
    par.mf.q          = param.mf.q
    par.mf.p          = param.mf.p
    par.cosmo.Om      = Om
    par.cosmo.Ob      = Ob
    par.cosmo.h0      = h0
    par.cosmo.zmin    = param.code.zmin
    par.cosmo.zmax    = param.code.zmax
    par.cosmo.Nz      = param.code.Nz
    par.cosmo.zbin    = 'log'
    par.code.Nrbin    = param.code.NM
    if (param.mf.window == 'tophat'):
        par.code.rmin = (3*param.code.Mmin/(4*np.pi*Om*rhoc0))**(1/3)
        par.code.rmax = (3*param.code.Mmax/(4*np.pi*Om*rhoc0))**(1/3)
    elif (param.mf.window == 'sharpk'):
        par.code.rmin = (3*param.code.Mmin/(4*np.pi*Om*rhoc0))**(1/3)/param.mf.c
        par.code.rmax = (3*param.code.Mmax/(4*np.pi*Om*rhoc0))**(1/3)/param.mf.c
    elif (param.mf.window == 'smoothk'):
        par.code.rmin = (3*param.code.Mmin/(4*np.pi*Om*rhoc0))**(1/3)/param.mf.c
        par.code.rmax = (3*param.code.Mmax/(4*np.pi*Om*rhoc0))**(1/3)/param.mf.c
    else:
        print("ERROR: Window for massfct does not exist!")
        exit()
    
    #calculate mass function with gmf for all redshifts
    mm, zz, dndlnm = gmf.dndlnm(par)

    #calculate mass dependent bias with gmf for all redshifts
    mm, zz, bias = gmf.halo_bias(par)

    #calculate variance with gmf for all redshifts
    rr, var, dlnvardlnr = gmf.variance(par)
    dlnvardlnm = 3*dlnvardlnr

    ######
    #from .bubbles import bubble_massfct
    #bubbles = bubble_massfct(zz,mm,var,dlnvardlnm,dndlnm,GS,param)
    #exit()
    #####
    
    #star-formation efficiency
    f_star = fstar(zz,mm,type_of_flux,param)
    f_tilde = fstar_tilde(mm,f_star)
    
    #collapesed fraction of stars
    fcoll = [1/(Om*rhoc0) * trapz((Ob/Om)*f_tilde[i,:]*dndlnm[i],mm) for i in range(len(zz))]
    fcoll = np.array(fcoll)

    #dfcoll/dt
    fcoll_tck = splrep(zz,fcoll)
    dfcolldz  = splev(zz,fcoll_tck,der=1)
    #dfcolldt  = -dfcolldz * (1+zz) * hubble(zz,param)          #dz/dt = (1+z)*H(z)
    sfrd = - (Om*rhoc0) * dfcolldz * (1+zz) * hubble(zz,param)  #[ (km/s)/Mpc * (Msun/h)/(Mpc/h)^3]
    sfrd = sfrd * sec_per_yr / km_per_Mpc                       # [(Msun/h)/(Mpc/h)^3/yr]
    
    return zz, mm, var, dlnvardlnm, dndlnm, bias, fcoll, dfcolldz, sfrd



def eps_xray(nu,param):
    """
    Spectral distribution function of x-ray emission
    See Eq.2 in arXiv:1406.4120
    """
    h0   = param.cosmo.h0
    fX   = param.xray.fX
    alS  = param.xray.pl_sed
    cX   = param.xray.cX/h0        # [(erg/s) * (yr*h/Msun)]
    
    #energy band within which SED is normalised
    Emin_norm =	param.xray.Emin_sed
    Emax_norm = param.xray.Emax_sed
    nu_min_norm = Emin_norm/hP
    nu_max_norm = Emax_norm/hP

    #normalised SED (following 1406.4120, Eq. 2)
    hPnu0 = 1000.0                                     # [eV]
    nu0   = hPnu0/hP                                   # [Hz]
    Anorm = (1-alS)/nu0/((nu_max_norm/nu0)**(1-alS)-(nu_min_norm/nu0)**(1-alS))
    Inu   = lambda nu: Anorm * (nu/nu0)**(-alS)        # [1/Hz]

    eps_X = fX * cX * eV_per_erg * Inu(nu) / (nu*hP)   # [1/s/Hz*(yr*h/Msun)]

    return eps_X


def eps_lyal(nu,param):
    """
    Spectral distibution function of Ly-al emission.
    eps_al = N_al*I_al(nu)/(m_p*h0) where
    I_al(nu) =  A_al*nu**al_al normalised so that 
    the integral between nu_min and nu_max is 1
    N_al = number of pbhotons per baryon
    """
    h0    = param.cosmo.h0
    N_al  = param.lyal.N_ph
    alS = param.lyal.pl_sed

    nu_min_norm  = nu_al
    nu_max_norm  = nu_LL

    Anorm = (1-alS)/(nu_max_norm**(1-alS) - nu_min_norm**(1-alS))
    Inu   = lambda nu: Anorm * nu**(-alS)        # [1/Hz]

    eps_alpha = Inu(nu)*N_al/(m_p*h0)

    return eps_alpha


def eps_reio(param):
    """
    Spectral distribution of UV emission
    """
    h0    = param.cosmo.h0
    N_ion = param.reio.N_ph
    f_esc = param.reio.f0_esc

    #Anorm = 1/(nu_max-nu_min)
    #Inu   = Anorm
    
    eps_reio = f_esc * N_ion /(m_p*h0)

    return eps_reio
