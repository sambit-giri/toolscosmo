"""
External Parameters
"""

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def code_par():
    par = {
        "verbose": True,
        "zmin": 5.0,                # min redshift
        "zmax": 50.0,               # max redshift (not tested below 40)
        "zbin": 'log',              #lin or log
        "Nz": 100,                  # Nb of redshift bins
        "zstar": 35,                # redshift of first star formation
        "Mdark": 1e5,               # lowest mass of star formation 
        "dz_prime_lyal": 0.01,      # z_prime binning lyal (should not be larger than 0.01)
        "dz_prime_xray": 0.1,       # z_prime binning xray (should not be larger than 0.1)
        "kmin": 0.001,              # min k-value
        "kmax": 10,                 # max k-value
        "rmin": 0.002,              # min k-value
        "rmax": 25,                 # max k-value
        "Nrbin": 100,               # Nb of r-bins
        "Nk": 100,                  # Nb of k-bins
        "Emin": 200,                # min heating energy,
        "Emax": 2000,               # max heating energy,
        "NE": 40,                   # Nb of E-bins
        "Mmin": 1e5,                # min halo mass 
        "Mmax": 3e15,               # max halo mass (this should be above 1e15 for the EPS mass accretion)
        "NM": 100,                  # Nb of M-bins
        "rectrunc": 23,             # sum trunc of Ly-series 
        "approx_heating": 'False',  # approx xray heating
        "MA": 'EPS',                # mass accretion model [EPS,EXP,AM]
        "sfrd_from_MA": True,       # True/False: calculate sfrd using mass accretion / time derivative of fcoll 
        "bias": 'ellipsoidal',      # [ellipsoidal/spherical/tinker/jing]
        "Dz_solver": 'Hamilton2000',# Default: 'Hamilton2000', Options: 'solveODE','Hamilton2000','Linder2005'
        }
    return Bunch(par)

def cosmo_par():
    par = {
        "solver": 'astropy',       # solver/package for cosmological parameter. Options: 'astropy', 'camb', 'tools_cosmo'.

        "Om" : 0.315,
        "Ob" : 0.049,
        "Or" : 5.4e-5,
        "Ok" : 0.0,
        "Ode": None,  # Flat universe is assumed if None or 'flat'.
        "s8" : 0.83,
        "As" : None, #2.089e-9,
        "h0" : 0.673,
        "ns" : 0.963,
        "YHe": 0.2425,
        "mnu": 0.06,

        "tau_reio": 0.052,
        "Tcmb": 2.72,
        "rhoc": 2.775e11, #critical density at z=0 [[h^2 Msun/Mpc^3] 
        "plin": None,
        }
    return Bunch(par)

def de_par(name):
    if name.lower()=='lcdm':
        par = {
            "name": name,
            }
    elif name.lower()=='wcdm':
        par = {
            "name": name,
            "w": -1.0,
            }
    elif name.lower()=='cpl':
        par = {
            "name": name,
            "w0": -1.0,
            "wa": 0.00,
            }
    elif name.lower()=='growing_neutrino_mass':
        par = {
            "name": name,
            "Onu" : 0.0,
            "Oede": 0.0,
            }
    elif name.lower()=='ultra_light_axion':
        par = {
            "name"   : name,
            "n"      : 3,    # Axion field dilutes faster than radiation.
            "w_n"    : None, # (n-1)/(n+1)
            "fde_zc" : 0.07, 
            "zc"     : 10**3,
            "theta_i": 3.14-0.1,
            }
    else:
        par = {
            "name"   : name,
            "wDE"    : None
            }
    return Bunch(par) 

def mf_par():
    par = {
        "window": 'tophat',   # [sharpk,smoothk,tophat]
        "dc": 1.675,          # delta_c
        "p": 0.3,             # p par of f(nu) [0.3,0.3,1] for [ST,smoothk,PS]
        "q": 0.85,            # q par of f(nu) [0.707,1,1] for [ST,smoothk,PS]
        "c": 2.5,             # prop constant for mass [only read for sharpk,smoothk]
        "beta": 4.0,          # additional free param for smoothk
        }
    return Bunch(par)

def mar_par():
    par = {
        "alpha_EXP" : 0.79,
        "alpha_EXPt": 1e-7,
        "Q_EPS" : 0.6,
        "t_star": 0.5,
        }
    return Bunch(par)

def lyal_par():
    par = {
        "N_ph": 9690.0,          # Nb of photons
        "pl_sed": 0.0,           # power law of sed
        "f0_sfe": 0.05,         
        "Mp_sfe": 2.0e11,
        "g1_sfe": 0.49,
        "g2_sfe": -0.61,
        "Mt_sfe": 0.0,
        "g3_sfe": 2.0,
        "g4_sfe": -1.0,
        }
    return Bunch(par)

def xray_par():
    par = {
        "fX": 0.2,               # fraction of energy
        "cX": 3.4e40,         # [(erg/s) * (yr/Msun)] L0=cX*(1-al)/(nu0*h)/((numax/nu0)**(1-al)-(numin/nu0)**(1-al)) -> (astro-ph/0607234 eq22)   
        "pl_sed": 1.5,           # power law index
        "Emin_sed": 500,        # min energy for normalisation
        "Emax_sed": 8000,       # max energy for normalisation
        "f0_sfe": 0.05,
        "Mp_sfe": 2.0e11,
        "g1_sfe": 0.49,
        "g2_sfe": -0.61,
        "Mt_sfe": 0.0,
        "g3_sfe": 2.0,
        "g4_sfe": -1.0,
        }
    return Bunch(par)

def reio_par():
    par = {
        "N_ph": 2665.0,          # Nb of photons (1509.07868 assumes: Nion/Nal~0.275, see text below Eq15)
        "f0_esc": 0.15,          # photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc
        "Mp_esc": 1e10,
        "pl_esc": 0.0,           # pl_esc = 0.5 is reasonable (Park+18)
        }
    return Bunch(par)

def lf_par():
    par = {
        "Muv_min": -25.0,            # min redshift
        "Muv_max": -8.0,             # max redshift (not tested below 40)
        "NMuv": 30,                  # Nb of redshift bins
        "sig_M": 0.2,
        "eps_sys": 1,  
        # fstar
        "f0_sfe": 0.05,
        "Mp_sfe": 2.0e11,
        "g1_sfe": 0.49,
        "g2_sfe": -0.61,
        "Mt_sfe": 7e8,
        "g3_sfe": 5.0,
        "g4_sfe": -5.0, 
        "f0_sfe_nu": -0.1,
        "Mp_sfe_nu": 0.0,
        "Mt_sfe_nu": 0.0,
        "g1_sfe_nu": 0.0,
        "g2_sfe_nu": 0.0,
        "g3_sfe_nu": 0.0,
        "g4_sfe_nu": 0.0, 
        }
    return Bunch(par)

def io_files():
    par = {
        "ps" : 'CAMB', # Plin. Option: file, CLASS, CAMB, BACCO, 'CDM_Planck15_pk.dat'.
        "varfct": 'var.dat',
        "tau": 'tabulated_tau.npz',
        "cmb": False,  # If True, the Boltzmann solver will provide Cls. 
        }
    return Bunch(par)


def par(DE='LCDM'):
    par = Bunch({
        "cosmo": cosmo_par(),
        "file": io_files(),
        #"sfe": sfe_par(),
        "mf": mf_par(),
        #"sed": sed_par(),
        "code": code_par(),
        "lyal": lyal_par(),
        "xray": xray_par(),
        "reio": reio_par(),
        "lf": lf_par(),
        "MA": mar_par(),
        "DE": de_par(DE),
        })
    return par
