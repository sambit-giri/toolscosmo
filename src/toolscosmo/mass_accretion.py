"""

DIFFERENT MODELS FOR THE AVERAGE HALO MASS ACCRETION AS
A FUNCTION OF MASS. THIS IS REQUIRED FOR THE HALO MODEL
CALCULATION

"""

import numpy as np
from .scipy_func import *
from .cosmo import hubble, growth_factor
from .constants import *

import matplotlib.pyplot as plt
from matplotlib import cm



def mass_accretion(GS,param):
    """
    Model selection
    """
    model = param.code.MA

    if(model == 'EPS'):
        M_accr, dMdt_accr = mass_accretion_EPS(GS,param)
    # elif(model == 'LC'):
    #     M_accr, dMdt_accr = mass_accretion_LC(GS,param)
    elif (model == 'EXP'):
        M_accr, dMdt_accr = mass_accretion_EXP(GS,param)
    elif (model == 'AM'):
        M_accr, dMdt_accr = mass_accretion_AM(GS,param)
    elif (model == 'EXPt'):
        M_accr, dMdt_accr = mass_accretion_EXPt(GS,param)
    elif (model == '21cmfast') or model.lower()=='hubblescale':
        M_accr, dMdt_accr = mass_accretion_HUBBLEscale(GS,param)
    else:
        print("ABORT: selected mass accretion model does not exist!")

    return M_accr, dMdt_accr

def mass_accretion_EPS(GS,param):
    """
    Assuming EPS formula
    (see Eq. 6 in 1409.5228)
    """
    Ob = param.cosmo.Ob
    Om = param.cosmo.Om
    Mmin = param.code.Mdark
    
    zz = GS['z']
    mm = GS['m']
    M0 = mm[np.where(mm>Mmin)]

    Dgrowth = growth_factor(GS['z'],param)
    dDgrowthdz = np.gradient(Dgrowth,zz,edge_order=1)

    var0 = GS['var'][np.where(mm>Mmin)]
    var_tck = splrep(M0,var0)

    #free parameter
    fM = param.MA.Q_EPS #0.6 #0.5
    if (Mmin<fM*param.code.Mmin):
        print("WARNING: choose smaller Mmin or larger Mdark")
    
    fracM = np.full(len(M0),fM)
    frac = interp1d(M0,fracM,axis=0,fill_value='extrapolate')
    #frac = interp1d(M0,fracM,axis=0,fill_value=fM)

    Dg_tck = splrep(zz,Dgrowth)
    D = lambda z: splev(z,Dg_tck)
    dDdz = lambda z: splev(z,Dg_tck,der=1)

    source = lambda M,z: (2/np.pi)**0.5 * M / (splev(frac(M)*M,var_tck,ext=1)-splev(M,var_tck,ext=1))**0.5 * 1.686/D(z)**2 * dDdz(z)
    Maccr = odeint(source,M0,zz)
    
    Raccr = Maccr/M0[None,:]
    dMaccrdz = np.gradient(Maccr,zz,axis=0,edge_order=1)
    dMaccrdt = - dMaccrdz * (1+zz)[:,None] * hubble(zz,param)[:,None] * sec_per_yr / km_per_Mpc

    #remove NaN
    Raccr[np.isnan(dMaccrdz)] = 0.0
    dMaccrdz[np.isnan(dMaccrdz)] = 0.0
    dMaccrdt[np.isnan(dMaccrdt)] = 0.0

    return Raccr*M0, dMaccrdt


def mass_accretion_EXP(GS,param):
    """
    Assuming exp growth with redshift (Eq. 5 in Furlanetto et al 2016)
    M(z) = M0*exp(-alpha(z-z0)).
    """
    Ob = param.cosmo.Ob
    Om = param.cosmo.Om
    Mmin = param.code.Mdark

    zz = GS['z']
    mm = GS['m']
    M0 = mm[np.where(mm>Mmin)]

    alpha = param.MA.alpha_EXP # 0.79

    Raccr    = np.zeros((len(zz),len(M0)))
    dMaccrdz = np.zeros((len(zz),len(M0)))
    dMaccrdt = np.zeros((len(zz),len(M0)))
    for i in range(len(zz)):
        Raccr[i,:] = np.exp(-alpha*(zz[i]-zz[0]))
        dMaccrdz[i,:] = -alpha*M0*Raccr[i,:]
        dMaccrdt[i,:] = - dMaccrdz[i,:] * (1+zz[i]) * hubble(zz[i],param) * sec_per_yr / km_per_Mpc

    return Raccr*M0, dMaccrdt

def mass_accretion_EXPt(GS,param):
    """
    Assuming accretion to depend on halo mass
    dMdt = alpha * M
    """
    Ob = param.cosmo.Ob
    Om = param.cosmo.Om
    Mmin = param.code.Mdark

    zz = GS['z']
    mm = GS['m']
    M0 = mm[np.where(mm>Mmin)]

    alpha = param.MA.alpha_EXPt # 1e-7  #[1/yr]

    dMaccrdt = np.full((len(zz),len(M0)),alpha)*M0[None,:]
    Raccr = np.ones((len(zz),len(M0)))

    #Raccr = 
    #Raccr = - trapz(alpha/(1+zz)/hubble(zz,param),zz) * km_per_Mpc / sec_per_yr
    #Raccr = Raccr[:,None]
    
    return Raccr*M0, dMaccrdt


def mass_accretion_AM(GS,param):
    """
    Halo mass accretion obtained via abundance matching the 
    halo mass functions at different redshifts
    (see also Furlanetto et al 2016)
    """
    
    Mmin = param.code.Mmin
    Ob   = param.cosmo.Ob
    Om   = param.cosmo.Om
    zstar = param.code.zstar
    
    zz = GS['z']
    mm = GS['m']
    M0 = mm[np.where(mm>Mmin)]

    dndlnm = GS['dndlnm']
    dndlnm = dndlnm[:,np.where(mm>Mmin)[0]]

    cumn = cumtrapz(dndlnm/M0, M0, initial=0.0, axis=1)
    ncum = cumn[:,-1][:,None] - cumn[:,:]

    M_accretion = np.zeros((len(zz),len(M0)))
    dMdt_accretion = np.zeros((len(zz),len(M0)))
    nbin = ncum[0,:]
    for j in range(len(M0)):
        Maccr = np.zeros(len(zz))
        for i in range(len(zz)):
            jabv = len(M0) - np.searchsorted(ncum[i,:][::-1],nbin[j],side='left')-1
            jblw = len(M0) - np.searchsorted(ncum[i,:][::-1],nbin[j],side='left')
            jabv = jabv if jabv>=0 else 0
            jblw = jblw if jblw>=0 else 0
            jabv = jabv if jabv<(len(M0)) else len(M0)-1
            jblw = jblw if jblw<len(M0) else len(M0)-1

            #interpolation
            Mabv = M0[jabv]
            Mblw = M0[jblw]
            nabv = ncum[i,jabv]
            nblw = ncum[i,jblw]
            if (jabv==jblw and jabv==0):
                Minterp = 0.0
            elif(jabv==jblw and jabv==(len(M0)-1)):
                Minterp = M_accretion[i,j-1]
            else:
                Minterp = (nbin[j]*(Mabv-Mblw) + (nabv*Mblw-nblw*Mabv))/(nabv-nblw)

            Maccr[i] = Minterp

        #smoothing
        Maccr = savgol_filter(Maccr,15,1)
        M_accretion[:,j] = Maccr   #[Msun/h]

        #calculate derivative
        Maccr_tck = splrep(zz,M_accretion[:,j],k=1)
        dMdz = splev(zz,Maccr_tck,der=1)

        dMdt = - dMdz * (1+zz) * hubble(zz,param) # [Msun/h (km/s)/Mpc]
        dMdt = dMdt * sec_per_yr / km_per_Mpc     # [(Msun/h)/yr]  

        dMdt_accretion[:,j] = dMdt
    
    #enforce monotonic increase
    for i in range(len(zz)):
        if (zz[i]<zstar):
            if(np.argmax(M_accretion[i,:])<(len(M0)-1) and np.all(M_accretion[i,:])>0):
                print(np.argmax(M_accretion[i,:]),"WARNING: force monotonic increase of M_accr (massaccretion.py)")
            monoboost = np.linspace(1,1.01,len(M_accretion[i,np.argmax(M_accretion[i,:]):]))
            M_accretion[i,np.argmax(M_accretion[i,:]):] = M_accretion[i,np.argmax(M_accretion[i,:])]*monoboost[:]

        #remove  averything abov zstar
        else:
            M_accretion[i,:] = np.zeros(len(M0))
            dMdt_accretion[i,:] = np.zeros(len(M0))
    
    return M_accretion, dMdt_accretion


def mass_accretion_HUBBLEscale(GS,param):
    """
    Park et al. (2019) model accretion by parameterising it in terms of
    the Hubble timescale (1/H(z)), which is implemented in 21cmfast.

    21cmfast: dMaccdt = H(z)/tstar * Mh
              dMaccdz = -1/(1+z)/tstar * Mh
              dM/M = - 1/tstar * dz/(1+z)
              ln(M/M0) = -1/tstar * ln(z/z0)
              M/M0 = (z/z0)**(-1/tstar)
    """
    Ob = param.cosmo.Ob
    Om = param.cosmo.Om
    Mmin = param.code.Mdark

    zz = GS['z']
    mm = GS['m']
    M0 = mm[np.where(mm>Mmin)]

    tstar = param.MA.t_star #0.5

    Raccr    = np.zeros((len(zz),len(M0)))
    dMaccrdz = np.zeros((len(zz),len(M0)))
    dMaccrdt = np.zeros((len(zz),len(M0)))
    for i in range(len(zz)):
        Raccr[i,:] = ((1+zz[0])/(1+zz[i]))**(1/tstar)
        dMaccrdz[i,:] = -1/tstar * 1/(1+zz[i]) * M0*Raccr[i,:]
        dMaccrdt[i,:] = - dMaccrdz[i,:] * (1+zz[i]) * hubble(zz[i],param) * sec_per_yr / km_per_Mpc
        
    return Raccr*M0, dMaccrdt
