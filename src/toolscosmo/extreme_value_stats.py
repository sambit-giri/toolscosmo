import numpy as np
from tqdm import tqdm
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
try:
    from scipy.integrate import trapz, cumtrapz
except:
    from scipy.integrate import trapezoid as trapz
    from scipy.integrate import cumulative_trapezoid as cumtrapz
import astropy.units as u
from astropy.cosmology import Planck13
from . import massfct, source_model
from .cosmo import prepare_cosmo_solver
from .param import par

try:
    from evstats import evs
except:
    print('To use, extreme value statistics (EVS), install evstats package found at https://github.com/christopherlovell/evstats')

def evs_hypersurface_pdf(param=None, V=33510.321, z=0.0, mmin=12, mmax=18):
    """
    Calculate extreme value probability density function for the dark matter
    halo population on a spatial hypersurface (fixed redshift).

    Parameters
    ----------
    param : parameter object
    V     : volume (default: a sphere with radius 20 Mpc)
    cosmo : the cosmology object from astropy

    Returns
    -------
    phi:

    """
    mf = hmf_param(param=param, Mmin=mmin, Mmax=mmax)
    if z>0.0:
        mf.update(z=z)
    if isinstance(V, (u.quantity.Quantity)): V = V.to('Mpc^3').value
    phi_max = evs.evs_hypersurface_pdf(mf=mf, V=V)
    return phi_max, np.log10(mf.m[:-1])

def evs_bin_pdf(param=None, zmin=0., zmax=0.1, dz=0.01, mmin=12, mmax=18, dm = 0.01, fsky=1.):
    """
    Calculate EVS in redshift and mass bin

    Parameters
    ----------
    param : parameter object
    zmin : z minimum
    zmax : z maximum
    dz: delta z
    mmin: mass minimum (log10 (h^{-1} M_{\sol}) )
    mmax: mass maximum (log10 (h^{-1} M_{\sol}) )
    dm: delta m (log10 (h^{-1} M_{\sol}) )
    fsky: fraction of sky
    cosmo : the cosmology object from astropy

    Returns
    -------
    phi: probability density function
    ln10m_range: corresponding mass values for PDF (log10 (h^{-1} M_{\sol}) )
    
    """
    mf = hmf_param(param=param)

    N, f, F, ln10m_range = _evs_bin(mf=mf, zmin=zmin, zmax=zmax, dz=dz, mmin=mmin, mmax=mmax, dm=dm)

    phi = evs._apply_fsky(N, f, F, fsky)

    return phi, ln10m_range

def _evs_bin(mf=None, zmin=0., zmax=0.1, dz=0.01, mmin=12, mmax=18, dm = 0.01):
    """
    Calculate EVS (ignoring fsky dependence). Worker function for `evs_bin_pdf`

    Parameters
    ----------
    zmin : z minimum
    zmax : z maximum
    dz: delta z
    mmin: mass minimum (log10 (h^{-1} M_{\sol}) )
    mmax: mass maximum (log10 (h^{-1} M_{\sol}) )
    dm: delta m (log10 (h^{-1} M_{\sol}) )

    Returns
    -------
    N (float)
    f (array)
    F (array)
    ln10m_range (array)
    
    """

    mf.update(Mmin=mmin, Mmax=mmax, dlog10m=dm)

    N = evs._computeNinbin(mf=mf, zmin=zmin, zmax=zmax, dz=dz)

    # need to set lower limit slightly higher otherwise hmf complains.
    # should have no impact on F if set sufficiently low.
    ln10m_range = np.log10(mf.m[np.log10(mf.m) >= mmin+1])

    F = np.array([evs._computeNinbin(mf=mf, zmin=zmin, zmax=zmax, lnmax=lnmax, dz=dz) \
                            for lnmax in tqdm(ln10m_range)])

    f = np.gradient(F, mf.dlog10m)
    
    return N, f, F, ln10m_range

class hmf_param:
    def __init__(self, param=None, z=0.0, Mmin=12, Mmax=18, dlog10m=0.01):
        if param is None: param = par()
        param.code.verbose = False
        param = prepare_cosmo_solver(param)
        self.param = param
        self.cosmo = param.cosmo.solver_estimator
        self.z = z
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.update(z=z, Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m)

    def update(self, **kwargs):
        z = kwargs.get('z', self.z)
        self.z = z 
        self.Mmin = kwargs.get('Mmin', self.Mmin)
        self.Mmax = kwargs.get('Mmax', self.Mmax)
        self.dlog10m = kwargs.get('dlog10m', self.dlog10m)
        self.mass_function()

    def mass_function(self):
        param = self.param
        param.code.Nz   = 10
        param.code.zmin = self.z if self.z>0 else 1e-5 
        param.code.zmin = param.code.zmin+1
        param.code.zbin = 'log' #'linear' #
        param.code.Mmin = 10**self.Mmin if self.Mmin<30 else self.Mmin
        param.code.Mmax = 10**self.Mmax if self.Mmax<30 else self.Mmax
        param.code.NM   = 500
        ms, zs, dndlnm = massfct.dndlnm(param)
        dlog10m = self.dlog10m
        log10m_bins = np.arange(np.log10(param.code.Mmin),np.log10(param.code.Mmax),dlog10m)
        xx, yy = np.log10(ms), np.log10(dndlnm[0,:])
        tck = splrep(xx[np.isfinite(yy)], yy[np.isfinite(yy)])
        self.dndlnm = 10**splev(log10m_bins, tck)
        self.dndlog10m = self.dndlnm*np.log(10)
        self.m = 10**log10m_bins
        self.param = param