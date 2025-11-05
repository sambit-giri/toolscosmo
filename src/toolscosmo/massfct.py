"""

CALCULATE HALO MASS FUNCTIONS

"""

import numpy as np
from .scipy_func import *
from .cosmo import hubble, growth_factor, variance, read_powerspectrum
from .bias import halo_bias

def fnu(nu,param):
    """
    First crossing distribution with
    free parametersv (A,p,q)
    f(nu) = A*(2*q*nu/pi)**0.5 * (1+(q*nu)**(-p)) * exp(-q*nu/2)
    Press-Schechter: (A,p,q) = (0.5,0,1)
    Ellipsoidal collapse: (A,p,q) = (0.3222,0.3,1)
    Sheth-Tormen: (A,p,q) = (0.3222,0.3,0.707)
    """
    p = param.mf.p
    q = param.mf.q

    A = 1/(1 + 2**(-p)*gamma(0.5-p)/np.pi**0.5)

    f = A*(2.0*q*nu/np.pi)**0.5 * (1.0+(q*nu)**(-p)) * np.exp(-q*nu/2.0)
    return f


def fnu_cond(d,var,d0,var0):
    """
    Conditional Mass function (Giocoli etal 2008, eq. 59)
    """
    ddelta = d-d0
    dvar     = var-var0
    fcond = (1/2/np.pi)**0.5 * ddelta/dvar**1.5 * np.exp(-ddelta**2/dvar/2)
    return fcond


def dndlnm(param):
    """
    Halo mass function dn/dlnM 
    """
    window = param.mf.window
    cc     = param.mf.c
    Om     = param.cosmo.Om
    rhoc   = param.cosmo.rhoc
    dc     = param.mf.dc
    
    rbin, var, dlnvardlnr = variance(param)
    
    if (window == 'tophat' or window == 'gaussian'):
        mbin = 4*np.pi*Om*rhoc*rbin**3/3

    elif (window == 'sharpk' or window == 'smoothk'):
        mbin = 4*np.pi*Om*rhoc*(cc*rbin)**3/3

    if (param.code.zbin=='lin'):
        zz = np.linspace(param.code.zmin,param.code.zmax,param.code.Nz)
    elif (param.code.zbin=='log'):
        zz = np.logspace(np.log(param.code.zmin),np.log(param.code.zmax),param.code.Nz,base=np.e)
    else:
        print("ERROR: neither lin nor log for binning. Abort")
        exit()

    Dz = growth_factor(zz, param)

    dndlnm = []
    for i in range(len(zz)):
        nu = dc**2.0/var/Dz[i]**2.0
        dndlnm += [- 1/6 * Om * rhoc / mbin * fnu(nu,param) * dlnvardlnr]
    dndlnm = np.array(dndlnm)
    return mbin, zz, dndlnm

"""
def pdf(k,kc,si):
    return np.exp(-(k-kc)**2/(2*si**2))/(2*np.pi*si**2)**0.5

def dndlnm_smoothed(param):
    wf   = param.code.window
    cc   = param.mf.c
    Om   = param.cosmo.Om
    rhoc = param.cosmo.rhoc
    dc   = param.mf.dc

    rbin, var, dlnvardlnr = variance(param)

    if (wf == 'tophat' or wf == 'gaussian'):
        mbin = 4*np.pi*Om*rhoc*rbin**3/3

    elif (wf == 'sharpk'):
        mbin = 4*np.pi*Om*rhoc*(cc*rbin)**3/3

    if (param.cosmo.zbin=='lin'):
        zz = np.linspace(param.cosmo.zmin,param.cosmo.zmax,param.cosmo.Nz)
    elif (param.cosmo.zbin=='log'):
    zz = np.logspace(np.log(param.cosmo.zmin),np.log(param.cosmo.zmax),param.cosmo.Nz,base=np.e)
    else:
    print("ERROR: neither lin nor log for binning. Abort")
    exit()

    Dz = growth_factor(zz, param)

    ps = read_powerspectrum(param)
    P_tck = splrep(ps['k'],ps['P'])

    dndlnm = []
    sigk = 1/10
    for i in range(len(zz)):
        nu = dc**2.0/var/Dz[i]**2.0
        print(1/12/np.pi**2 * Om * rhoc / mbin * fnu(nu,param)/var * splev(1/rbin,P_tck)/rbin**3)
        #dndlnm += [1/12/np.pi**2 * Om * rhoc / mbin * fnu(nu,param)/var * splev(1/rbin,P_tck)/rbin**3]
        prefac = 1/12/np.pi**2 * Om * rhoc / mbin * fnu(nu,param)/var
        #for j in range(len(rbin)):
        dndlnm_smoothed = -prefac*np.array([np.trapz(splev(1/rbin,P_tck)*pdf(1/rbin,1/rbin[j],sigk),x=(1/rbin)) for j in range(len(rbin))])
        print(dndlnm_smoothed)
        dndlnm += [dndlnm_smoothed]
        exit()
    return mbin, zz, dndlnm
"""

def massfct_table(param):
    """
    Writes file with logm, Dgrowth, dndm, ngtm, mgtm, z, var
    Can be used for ares
    """
    window = param.mf.window
    cc     = param.mf.c
    Om     = param.cosmo.Om
    rhoc   = param.cosmo.rhoc
    dc     = param.mf.dc

    rbin, var, dlnvardlnr = variance(param)

    if (window == 'tophat' or window == 'gaussian'):
        mbin = 4*np.pi*Om*rhoc*rbin**3/3
    elif (window == 'sharpk' or window == 'smoothk'):
        mbin = 4*np.pi*Om*rhoc*(cc*rbin)**3/3

    if (param.cosmo.zbin=='lin'):
        zz = np.linspace(param.cosmo.zmin,param.cosmo.zmax,param.cosmo.Nz)
    elif (param.cosmo.zbin=='log'):
        zz = np.logspace(np.log(param.cosmo.zmin),np.log(param.cosmo.zmax),param.cosmo.Nz,base=np.e)
    else:
        print("ERROR: neither lin nor log for binning. Abort")
        exit()

    Dz = growth_factor(zz, param)

    dndm = []
    ngtm = []
    mgtm = []
    for i in range(len(zz)):
        nu = dc**2.0/var/Dz[i]**2.0
        dndm += [- 1/6 * Om * rhoc / mbin**2 * fnu(nu,param) * dlnvardlnr]
        ngtm += [trapz(dndm[i], mbin) - cumtrapz(dndm[i],mbin,initial=0.0)]
        mgtm += [trapz(dndm[i]*mbin, mbin) - cumtrapz(dndm*mbin,mbin,initial=0.0)]

    #write to file
    np.savez(param.file.mf_table, logM=np.log10(mbin), growth=Dz, dndM=dndm, ngtM=ngtm, mgtM=mgtm, z=zz, sigma=var)


def fstar(m,param):
    """
    Star formation efficiency
    fstar = fstar0/((m/Mp)**gamma1 + (m/Mp)**gamma2) * trunc  
    trunc = (1 + (m/Mt)**gamma3)**gamma4    
    m : array of halo masses 
    """
    fstar0 = param.sfe.fstar0
    gamma1 = param.sfe.gamma1
    gamma2 = param.sfe.gamma2
    gamma3 = param.sfe.gamma3
    gamma4 = param.sfe.gamma4
    Mpivot = param.sfe.Mpivot
    Mtrunc = param.sfe.Mtrunc

    denom1 = (m/Mpivot)**gamma1
    denom2 = (m/Mpivot)**gamma2

    trunc  = (1+(Mtrunc/m)**gamma3)**gamma4

    fstar = fstar0/(denom1+denom2)*trunc
    fstar[np.where(fstar>=1)] = 1
    return fstar



def sfrd(param):
    """
    output:    zbin    = array of redshifts
               fcoll   = frasction of matter in haloes 
               dfcoldt = star formation rate density

    fcoll    = (Om_b/Om_m)/rho_m * int dM * fstar(M) * dn/dlnM 
    dfcolldt = dfcoll/dt = drho_star/dt
    """

    window = param.mf.window
    cc     = param.mf.c
    Om     = param.cosmo.Om
    Ob     = param.cosmo.Ob
    h0     = param.cosmo.h0
    rhoc   = param.cosmo.rhoc
    dc     = param.mf.dc

    #variance and its derivative
    rbin, var, dlnvardlnr = variance(param)

    #Halo mass bin
    if (window == 'tophat' or window == 'gaussian'):
        mbin = 4*np.pi*Om*rhoc*rbin**3/3
    elif (window == 'sharpk' or window == 'smoothk'):
        mbin = 4*np.pi*Om*rhoc*(cc*rbin)**3/3

    #Redshift binning
    if (param.cosmo.zbin=='lin'):
        zz = np.linspace(param.cosmo.zmin,param.cosmo.zmax,param.cosmo.Nz)
    elif (param.cosmo.zbin=='log'):
        zz = np.logspace(np.log(param.cosmo.zmin),np.log(param.cosmo.zmax),param.cosmo.Nz,base=np.e)
    else:
        print("ERROR: neither lin nor log for binning. Abort")
        exit()

    #Growth factor
    Dz = growth_factor(zz, param)

    #star-formation efficiency
    f_star = fstar(mbin,param)

    #fcoll
    fcoll = []
    for i in range(len(zz)):
        nu = dc**2.0/var/Dz[i]**2.0
        dndlnm = - 1/6 * Om * rhoc / mbin * fnu(nu,param) * dlnvardlnr
        fcoll += [trapz(f_star*dndlnm,mbin)]
    fcoll = (Ob/Om)/(Om*rhoc) * np.array(fcoll)

    #dfcoll/dt (approx)
    fcoll_tck = splrep(zz,fcoll)
    dfcolldz  = splev(zz,fcoll_tck,der=1)
    #dfcolldt  = -dfcolldz * (1+zz) * hubble(zz,param)  #dx/dt = (1+z)*H(z)
    sfrd = - (Ob*rhoc) * dfcolldz * (1+zz) * hubble(zz,param)  #[ (km/s)/Mpc * (Msun/h)/(Mpc/h)^3]
    sfrd = sfrd * 1.023e-12                                    # [1/yr * (Msun/h)/(Mpc/h)^3] --> 1km/s = 1.023e-12Mpc/yr 
    return zz, fcoll, dfcolldz, sfrd
