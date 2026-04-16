import numpy as np
from numpy import sqrt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from time import time
import matplotlib.pyplot as plt
from functools import lru_cache
import pandas as pd
import os, sys
from tqdm import tqdm
from abc import ABC, abstractmethod

from .cosmo import growth_factor, get_Plin, hubble
from .generate_ic import *

try:
    import tools21cm as t2c
    _TOOLS21CM_AVAILABLE = True
except ImportError:
    t2c = None
    _TOOLS21CM_AVAILABLE = False

__all__ = [
    'EFTBase',
    'EFTmodel',
    'EFTformalism_Anastasiou2024',
    'EFTformalism_Qin2022',
    'D',
    'P_linear_bias',
    'linear_bias_21',
    'compute_EPT_delta',
    'compute_21cm_EFT_tracer',
    'compute_tidal_field',
]

def D(z, par):
  '''
  Growth factor
  '''
  if isinstance(z,(int,float)): z = [z]
  Dz = growth_factor(z, par)
  return Dz

def P_linear_bias(z, par, b, **kwargs):
  '''
  Model the biased tracer power spectra with linear bias.
  delta_tracer = b*delta_m
  P_tracer = b**2* P_linear
  '''
  plin = get_Plin(par, **kwargs)
  P = plin['P']
  return {'k': plin['k'], 'P': b**2*P/D(z,par)**2}

def linear_bias_21(z, par):
  '''
  Linear bias for 21-cm
  '''
  if not _TOOLS21CM_AVAILABLE:
    raise ImportError(
      "tools21cm is required for linear_bias_21. "
      "Install it with: pip install tools21cm"
    )
  t2c.set_hubble_h(par.cosmo.h0)
  t2c.set_ns(par.cosmo.ns)
  t2c.set_omega_baryon(par.cosmo.Ob)
  t2c.set_omega_matter(par.cosmo.Om)
  t2c.set_sigma_8(par.cosmo.s8)
  b21_lin = t2c.mean_dt(z)
  return b21_lin

def compute_EPT_delta(param, grid_size, box_size, z=0.0, random_seed=42,
                      order=3, MAS='CIC', anti_aliasing=True,
                      deconvolve=True, cs=1.0, verbose=True):
    """
    Compute the EFT-of-LSS density field with customizable MAS, anti-aliasing, deconvolution, and counterterms.

    Parameters:
    -----------
    param : dict
        Cosmological parameters (e.g., {'Omega_m': 0.3, 'h': 0.7}).
    grid_size : int
        Number of grid points per dimension.
    box_size : float
        Physical box size (Mpc/h).
    random_seed : int, optional
        Seed for Gaussian random field generation. Default: 42.
    order : int, optional
        Maximum LPT order (1, 2, or 3). Default: 3.
    MAS : str, optional
        Mass assignment scheme: 'NGP', 'CIC' (default), 'TSC', or 'PCS'.
    anti_aliasing : bool, optional
        Apply anti-aliasing filter. Default: True.
    deconvolve : bool, optional
        Deconvolve MAS window function. Default: True.
    cs : float, optional
        EFT sound speed coefficient (counterterm amplitude). Default: 1.0.
    verbose : bool, optional
        Print progress messages. Default: True.

    Returns:
    --------
    delta_EPT : np.ndarray
        EFT density field (delta = rho/mean_rho - 1).
    """
    # --- Step 1: Generate initial Gaussian field ---
    if verbose: print("Generating Gaussian random field...")
    gen_field = generate_gaussian_random_field(
        grid_size, box_size, param=param, random_seed=random_seed, verbose=verbose
    )
    delta_k = rfftn(gen_field['delta_lin'])  # Linear delta in Fourier space (rFFT)

    # --- Step 2: Compute LPT displacements ---
    if verbose: print("Computing LPT displacements...")
    grad_kernel, k_squared = create_gradient_kernel_rfft3(grid_size, box_size)
    a = 1.0 / (1.0 + z)

    # 1LPT (Zel'dovich)
    D1 = D1_growth_factor(a, param)
    disp_x1, disp_y1, disp_z1 = first_order_lpt(delta_k, D1, grad_kernel, k_squared, 1)

    # 2LPT
    if order >= 2:
        D2 = D2_growth_factor(a, param)
        disp_x2, disp_y2, disp_z2 = second_order_lpt(delta_k, D2, grad_kernel, k_squared, 1)

    # 3LPT
    if order >= 3:
        D3a = D3a_growth_factor(a, param)
        D3b = D3b_growth_factor(a, param)
        disp_x3, disp_y3, disp_z3 = third_order_lpt(delta_k, D3a, D3b, grad_kernel, k_squared, 1)

    # --- Step 3: Move particles ---
    if verbose: print("Computing total displacement...")
    pos = np.stack(np.meshgrid(
        np.arange(grid_size), 
        np.arange(grid_size), 
        np.arange(grid_size), 
        indexing='ij'
    ), axis=-1).astype(np.float32) * (box_size / grid_size)

    # Sum displacements
    disp_tot = np.zeros_like(pos)
    disp_tot[..., 0] += disp_x1
    disp_tot[..., 1] += disp_y1
    disp_tot[..., 2] += disp_z1
    if order >= 2:
        disp_tot[..., 0] += disp_x2
        disp_tot[..., 1] += disp_y2
        disp_tot[..., 2] += disp_z2
    if order >= 3:
        disp_tot[..., 0] += disp_x3
        disp_tot[..., 1] += disp_y3
        disp_tot[..., 2] += disp_z3

    pos += disp_tot
    pos %= box_size  # Enforce periodic boundary

    # --- Step 4: Compute density field ---
    if verbose: print("Computing density field...")
    delta_EPT = particles_on_grid(
        pos.reshape(-1, 3), grid_size, box_size, 
        MAS=MAS, 
        anti_aliasing=anti_aliasing, 
        deconvolve=deconvolve, 
        verbose=verbose
    )

    # --- Step 5: Add EFT counterterms ---
    if cs != 0:  # Skip if cs=0
        if verbose: print(f"Adding EFT counterterms (cs={cs})...")
        kx, ky, kz = np.meshgrid(
            2*np.pi * np.fft.fftfreq(grid_size, d=box_size/grid_size),
            2*np.pi * np.fft.fftfreq(grid_size, d=box_size/grid_size),
            2*np.pi * np.fft.rfftfreq(grid_size, d=box_size/grid_size),
            indexing='ij'
        )
        k2 = kx**2 + ky**2 + kz**2
        alpha = -cs * k2 / (1 + k2)  # Counterterm kernel
        delta_k = rfftn(delta_EPT)
        delta_k *= (1 + alpha)
        delta_EPT = irfftn(delta_k, s=(grid_size, grid_size, grid_size)).real

    return delta_EPT

def compute_21cm_EFT_tracer(param, grid_size, box_size, z, 
                           MAS='CIC', anti_aliasing=True, deconvolve=True,
                           order_lpt=3, cs=1.0, b1=1.5, b2=0.5, bs=0.1, 
                           beta=0.5, sigma_v=3.0, random_seed=42, verbose=True):
    """
    Compute the 21-cm brightness temperature contrast using EFT-based tracer model.

    Parameters:
    -----------
    param : dict
        Cosmological parameters (e.g., {'Omega_m': 0.3, 'h': 0.7}).
    grid_size : int
        Grid resolution.
    box_size : float
        Box size (Mpc/h).
    z : float
        Redshift.
    MAS : str, optional
        Mass assignment scheme ('NGP', 'CIC', 'TSC', 'PCS'). Default: 'CIC'.
    anti_aliasing : bool, optional
        Apply anti-aliasing filter. Default: True.
    deconvolve : bool, optional
        Deconvolve MAS window. Default: True.
    order_lpt : int, optional
        LPT order (1, 2, or 3). Default: 3.
    cs : float, optional
        EFT sound speed coefficient. Default: 1.0.
    b1, b2, bs : float, optional
        Linear, quadratic, and shear biases for 21-cm. Default: 1.5, 0.5, 0.1.
    beta : float, optional
        RSD parameter (beta = f/b1, where f is growth rate). Default: 0.5.
    sigma_v : float, optional
        Velocity dispersion for Fingers-of-God damping (Mpc/h). Default: 3.0.
    random_seed : int, optional
        Random seed for initial conditions. Default: 42.
    verbose : bool, optional
        Print progress. Default: True.

    Returns:
    --------
    delta_Tb : np.ndarray
        21-cm brightness temperature contrast (delta_Tb/Tb_mean).
    """
    # --- Step 1: Generate LPT density field ---
    delta_EPT = compute_EPT_delta(
        param=param, grid_size=grid_size, box_size=box_size, z=z,
        random_seed=random_seed, order=order_lpt, MAS=MAS,
        anti_aliasing=anti_aliasing, deconvolve=deconvolve, cs=cs, verbose=verbose
    )

    # --- Step 2: Compute bias expansion terms ---
    if verbose: print("Computing bias terms...")
    # First-order term (linear bias)
    delta_1 = delta_EPT

    # Second-order terms (b2 and bs)
    delta_2 = delta_1**2 - np.mean(delta_1**2)
    s_ij = compute_tidal_field(delta_1, box_size)  # shape (3, 3, N, N, N)
    s2 = np.sum(s_ij**2, axis=(0, 1)) - np.mean(np.sum(s_ij**2, axis=(0, 1)))

    # Combine bias terms
    delta_bias = b1 * delta_1 + 0.5 * b2 * delta_2 + bs * s2

    # --- Step 3: Add redshift-space distortions (RSD) ---
    if verbose: print("Adding RSD...")
    kx, ky, kz_r = np.meshgrid(
        2*np.pi * np.fft.fftfreq(grid_size, d=box_size/grid_size),
        2*np.pi * np.fft.fftfreq(grid_size, d=box_size/grid_size),
        2*np.pi * np.fft.rfftfreq(grid_size, d=box_size/grid_size),
        indexing='ij'
    )
    k2 = kx**2 + ky**2 + kz_r**2
    k2[0, 0, 0] = 1.0  # Avoid division by zero

    # Apply Kaiser effect (large-scale RSD)
    if beta != 0:
        delta_k = np.fft.rfftn(delta_bias)
        delta_k *= (1 + beta * kz_r**2 / k2)  # Kaiser formula
        delta_bias = np.fft.irfftn(delta_k, s=(grid_size, grid_size, grid_size)).real

    # Apply Fingers-of-God damping (small-scale RSD)
    if sigma_v > 0:
        damping = np.exp(-0.5 * (sigma_v * kz_r)**2)
        delta_k = np.fft.rfftn(delta_bias) * damping
        delta_bias = np.fft.irfftn(delta_k, s=(grid_size, grid_size, grid_size)).real

    # --- Step 4: 21-cm specific effects ---
    # (Placeholder for X-ray heating, Ly-alpha coupling, etc.)
    delta_Tb = delta_bias  # Simplified for illustration

    return delta_Tb

def compute_tidal_field(delta, box_size):
    """Compute tidal tensor s_ij = (∂i∂j/∇^2 - δ_ij/3) delta."""
    delta_k = np.fft.rfftn(delta)
    kx, ky, kz = np.meshgrid(
        2*np.pi * np.fft.fftfreq(delta.shape[0], d=box_size/delta.shape[0]),
        2*np.pi * np.fft.fftfreq(delta.shape[1], d=box_size/delta.shape[1]),
        2*np.pi * np.fft.rfftfreq(delta.shape[2], d=box_size/delta.shape[2]),
        indexing='ij'
    )
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # Avoid division by zero

    s_ij = []
    for ki in [kx, ky, kz]:
        for kj in [kx, ky, kz]:
            s_k = (ki * kj / k2 - (ki == kj)/3) * delta_k
            s_ij.append(np.fft.irfftn(s_k, s=delta.shape).real)
    return np.array(s_ij).reshape(3, 3, *delta.shape)

class EFTformalism_Anastasiou2024:
  '''
  An implementation of the formalism developed in 2212.07421.
  '''
  def __init__(self, param, CHOP_TOL=1e-30, k_min=0.02, k_max=0.4,
               nBW=12,  # Number of Breit-Wigner terms
               verbose=True,
               save_folder='IntegralMatrices',
               ):
    self.param = param
    self.CHOP_TOL = CHOP_TOL  # Precision used as tolerance
    self.coef_dim_gen_cache = {}  # Cache for memoization
    self.k_min = k_min
    self.k_max = k_max
    self.nBW = nBW
    self.verbose = verbose
    self.save_folder = save_folder

    #Parameters used in fitting functions
    self.k0 = 1/20   #in h/Mpc
    #Units of h^2/Mpc^2
    self.k2peak1 = np.complex128(-3.40e-2) #mpc(-3.4*10**(-2))
    self.k2peak2 = np.complex128(-1.00e-3) #mpc(-1*10**(-3))
    self.k2peak3 = np.complex128(-7.60e-5) #mpc(-7.6*10**(-5))
    self.k2peak4 = np.complex128(-1.56e-5) #mpc(-1.56*10**(-5))
    self.k2UV0 = np.complex128(1.00e-4) #mpc(1*10**(-4))
    self.k2UV1 = np.complex128(6.90e-2) #mpc(6.9*10**(-2))
    self.k2UV2 = np.complex128(8.20e-3) #mpc(8.2*10**(-3))
    self.k2UV3 = np.complex128(1.30e-3) #mpc(1.3*10**(-3))
    self.k2UV4 = np.complex128(1.35e-5) #mpc(1.35*10**(-5))
    self.M0 = np.complex128(self.k2UV0)
    self.M1 = self.k_to_M(self.k2peak1, self.k2UV1)
    self.M2 = self.k_to_M(self.k2peak2, self.k2UV2)
    self.M3 = self.k_to_M(self.k2peak3, self.k2UV3)
    self.M4 = self.k_to_M(self.k2peak4, self.k2UV4)

    self.kappa = np.array([
          [1j / 2, 0, 0, 0, 0],
          [1j / 4, -1 / 4, 0, 0, 0],
          [3 * 1j / 16, -3 / 16, -1j / 8, 0, 0],
          [5 * 1j / 32, -5 / 32, -1j / 8, 1 / 16, 0],
          [35 * 1j / 256, -35 / 256, -15 * 1j / 128, 5 / 64, 1j / 32]
      ]).T
    
    self.k2UVlist = [self.k2UV0, self.k2UV1, self.k2UV1, self.k2UV1, self.k2UV2, self.k2UV2, self.k2UV2, self.k2UV2,
                self.k2UV3, self.k2UV3, self.k2UV3, self.k2UV3, self.k2UV4, self.k2UV4, self.k2UV4, self.k2UV4]
    self.Mlist = [self.M0, self.M1, self.M1, self.M1, self.M2, self.M2, self.M2, self.M2,
             self.M3, self.M3, self.M3, self.M3, self.M4, self.M4, self.M4, self.M4]

    # defines a cache for memoization
    self.coef_dim_gen_cache = {}

    self.tracer_matrices = None

  def k1dotk2(self, k2_1, k2_2, ksum2):
    """
    Calculates the dot product k1.k2 given k1^2, k2^2, and (k1+k2)^2.
    """
    return (ksum2 - k2_1 - k2_2) / 2

  @lru_cache(None)  # Cache results for efficiency
  def binomial_c(self, n, k):
    """
    Calculates n choose k using recursion with memoization.
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    if 2 * k > n:  # Use symmetry property
        return self.binomial_c(n, n - k)
    return n * self.binomial_c(n - 1, k - 1) // k
  
  def trinomial(self, n, k, i):
    """
    Returns coefficient of x^k y^i z^(n-k-i) in (x + y + z)^n.
    """
    return self.binomial_c(n, k + i) * self.binomial_c(k + i, i)

  def get_coef_simple(self, n1, kmq2exp):
    """
    Get coefficient of the expansion of ((k-q)^2 + m)^n1.
    """
    return self.binomial_c(n1, kmq2exp)

  def get_coef_simple(self, n1, kmq2exp):
    # get coefficient of the expansion of ((k-q)^2+m)^n1
    # that has the form ((k-q)^2)^kmq2exp * m^(n1-kmq2exp)
    # (aka just binomial expansion)
    return self.binomial_c(n1, kmq2exp)

  def get_coef(self, n1, k2exp, q2exp, kmq=True):
    """
    Get coefficient of the expansion of (k-q)^2n1 in terms of k^2, q^2, and (k.q).
    """
    kqexp = n1 - k2exp - q2exp
    sign = (-1)**kqexp if kmq else 1
    return sign * self.trinomial(n1, k2exp, q2exp) * (2**kqexp)

  def num_terms(self, n1, kmq=True):
    """
    Expands terms of the type (k-q)^(2n1) or (k+q)^(2n1).
    Returns:
    - term_list: Coefficients
    - exp_list: List of [k^2, q^2, k.q] exponents
    """
    list_length = (n1 + 1) * (n1 + 2) // 2
    term_list = np.zeros(int(list_length), dtype=np.longlong)
    exp_list = np.zeros((int(list_length), 3), dtype=np.longlong)

    idx = 0
    for k2exp in range(int(n1 + 1)):
        for q2exp in range(int(n1 - k2exp + 1)):
            term_list[idx] = self.get_coef(n1, k2exp, q2exp, kmq)
            exp_list[idx] = [k2exp, q2exp, n1 - k2exp - q2exp]
            idx += 1

    return term_list, exp_list

  def expand_massive_num(self, n1):
    """
    Expand terms of the form ((k-q)^2 + m)^n1.
    Returns:
    - term_list: Coefficients
    - exp_list: List of [exp_kmq2, exp_mNum].
    """
    term_list = np.array([self.get_coef_simple(n1, i) for i in range(n1 + 1)], dtype=np.longlong)
    exp_list = np.column_stack((np.arange(n1 + 1, dtype=np.longlong), np.arange(n1, -1, -1, dtype=np.longlong)))

    return term_list, exp_list
  
  def construct_design_matrix(self, kh):
    #Parameters used in fitting functions
    k0 = self.k0
    #Units of h^2/Mpc^2
    k2peak1 = self.k2peak1
    k2peak2 = self.k2peak2
    k2peak3 = self.k2peak3
    k2peak4 = self.k2peak4
    k2UV0 = self.k2UV0
    k2UV1 = self.k2UV1
    k2UV2 = self.k2UV2
    k2UV3 = self.k2UV3
    k2UV4 = self.k2UV4
    k2 = kh**2
    design_matrix = np.array([1/(1+k2/k2UV0),
            self.f_basis(k2, k2peak1, k2UV1, 0, 1, k0=k0),
            self.f_basis(k2, k2peak1, k2UV1, 0, 2, k0=k0),
            self.f_basis(k2, k2peak1, k2UV1, 0, 3, k0=k0),
            self.f_basis(k2, k2peak2, k2UV2, 1, 1, k0=k0),
            self.f_basis(k2, k2peak2, k2UV2, 1, 2, k0=k0),
            self.f_basis(k2, k2peak2, k2UV2, 1, 3, k0=k0),
            self.f_basis(k2, k2peak2, k2UV2, 1, 4, k0=k0),
            self.f_basis(k2, k2peak3, k2UV3, 0, 2, k0=k0),
            self.f_basis(k2, k2peak3, k2UV3, 0, 3, k0=k0),
            self.f_basis(k2, k2peak3, k2UV3, 0, 4, k0=k0),
            self.f_basis(k2, k2peak3, k2UV3, 0, 5, k0=k0),
            self.f_basis(k2, k2peak4, k2UV4, 0, 1, k0=k0),
            self.f_basis(k2, k2peak4, k2UV4, 0, 2, k0=k0),
            self.f_basis(k2, k2peak4, k2UV4, 0, 3, k0=k0),
            self.f_basis(k2, k2peak4, k2UV4, 0, 4, k0=k0)
            ]).T
    
    return design_matrix

  def construct_design_matrixBW(self, kh):
    '''
    Build the design matrix using the Breit-Wigner basis functions
    '''
    nBW = self.nBW
    design_matrixBW = np.vstack([self.breit_wigner_basis(kh, 1, nBW)]).T

    # Add each of the basis functions to the design matrix
    for i in range(2, nBW+1):  # We have nBW terms in total
        basis_function = self.breit_wigner_basis(kh, i, nBW)
        design_matrixBW = np.hstack([design_matrixBW, np.vstack([basis_function]).T])
    
    return design_matrixBW
  
  def breit_wigner_basis(self, k, i, n):
    '''
    Generate the Breit-Wigner basis function
    '''
    k_min_BW = 0.06  # h/Mpc
    k_max_BW = 0.38  # h/Mpc

    # Calculate m_BW,i and delta_BW_i
    m_BW_i = k_max_BW * np.exp(i / n)
    delta_BW_i = (0.02 * k_max_BW / k_min_BW) * np.exp(i / n)
    
    # Breit-Wigner function for the basis
    numerator = k**2
    denominator = ((k**2 - m_BW_i**2)**2 + (m_BW_i**2 * delta_BW_i**2))**2
    return numerator / denominator

  def k_to_M(self, k2peak, k2UV):
    return -k2peak+1.j*k2UV

  def f_basis(self, k2, k2peak, k2UV, i, j, k0=1/20, M=None):
    #Defining relevant fitting variables
    #Values for kappa_n,j in partial fraction decomposition of fitting functions
    kappa = self.kappa
    M = self.k_to_M(k2peak, k2UV) if M is None else M

    fval = 0 if isinstance(k2, (float,int)) else np.zeros_like(k2)+1.j
    for n0 in range(0,j):
      n = n0+1
      fval += k2UV**(n) * (k2/k0**2)**i * (kappa[n-1,j-1]/(k2+M)**n+np.conjugate(kappa)[n-1,j-1]/(k2+np.conjugate(M))**n)
      # print(f'j={j}, n={n} | f={fval}')
      # print(k2, M, kappa[n-1,j-1], (kappa[n-1,j-1]/(k2+M)**n+np.conjugate(kappa)[n-1,j-1]/(k2+np.conjugate(M))**n))
    
    return fval

  def fit_P_linear(self, plin=None):
    verbose = self.verbose
    par = self.param
    # from gmpy2 import mpc
    if plin is None:
      plin = get_Plin(par)
    k = plin['k']
    P = plin['P']
    # k2 = k**2

    if verbose: 
      tstart = time()
      print(f'Fitting to the linear matter power spectrum...') 
    # cost = lambda an: np.sum((np.log10(P)-np.log10(pfit_alpha(k2,an)))**2)
    # method = 'L-BFGS-B' #'Nelder-Mead' #None #'Nelder-Mead'
    # x0 = np.ones(16)
    # res = minimize(cost, x0, method=method, tol=1e-10)
    # alpha_n = res.x
    design_matrix = self.construct_design_matrix(k)
    design_matrixBW = self.construct_design_matrixBW(k)
    alpha_n, residuals, rank, s = np.linalg.lstsq(design_matrix, P, rcond=None) 
    pk_fitted = design_matrix @ alpha_n
    delta_P = P - pk_fitted 
    design_matrixBW = self.construct_design_matrixBW(k)
    design_matrixBW = np.array(design_matrixBW, dtype=np.complex128)
    alpha_n_BW, residuals, rank, s = np.linalg.lstsq(design_matrixBW, delta_P, rcond=None) 
    if verbose: 
      print(f'...done in {time()-tstart:.2f} s')
    return {
      'alpha_n'   : alpha_n,
      'alpha_n_BW': alpha_n_BW,
      'Plin_fit': pk_fitted, #pfit_alpha(k**2,alpha_n),
      'k'     : k,
      'Plin'  : P,
      'design_matrix'  : design_matrix,
      'design_matrixBW': design_matrixBW,
      }
  
  def BW_fitting_params(self, k_min_BW=0.06, k_max_BW=0.38):
    nBW = self.nBW
    mBW = lambda i: k_max_BW * np.exp(i / nBW)
    deltaBW = lambda i: (0.02 * k_max_BW / k_min_BW) * np.exp(i / nBW)

    IBW = [1]
    JBW = [2]
    k2UVlistBW = [np.complex128(mBW(i)*deltaBW(i)) for i in range(nBW)]
    MlistBW = [np.complex128(mBW(i)+1j*mBW(i)*deltaBW(i)) for i in range(nBW)]

    return {
        'k2UVlistBW': k2UVlistBW,
        'MlistBW': MlistBW, 
        'IBW': IBW,
        'JBW': JBW,
      }

  @lru_cache(None)
  def coef_dim_gen(self, expnum, expden):
    """
    Calculate coefficient in dim_gen without Gamma functions using recursion.
    """
    # Base cases
    if expden == 1:
        return (-1) ** (expnum + 1)
    if expden < 1:
        return 0

    # Check cache
    if (expnum, expden) in self.coef_dim_gen_cache:
        return self.coef_dim_gen_cache[(expnum, expden)]

    # Recursive calculation
    val = (
        self.coef_dim_gen(expnum, expden - 1) *
        (5 - 2 * expden + 2 * expnum) /
        (2 - 2 * expden)
    )
    self.coef_dim_gen_cache[(expnum, expden)] = val
    return val
    
  def dim_gen(self, expnum, expden, M):
    """
    Calculates the 3D integral of (q^2)^expnum / (q^2 + m)^expden.
    """
    if M == 0:
        return 0
    return (2 * sqrt(np.pi)) * self.coef_dim_gen(expnum, expden) * M**(expnum - expden + 1) * sqrt(M)

  def dim_result(self, expkmq2, expden, k2, mDen, kmq=True):
    """
    Computes integrals of the type ((k-q)^2)^expnum / (q^2 + mDen)^expden.
    """
    # Get term_list and exp_list
    term_list, exp_list = self.num_terms(expkmq2, kmq)

    # Extract exponents for k^2, q^2, and k.q
    k2exp = exp_list[:, 0]
    q2exp = exp_list[:, 1]
    kqexp = exp_list[:, 2]

    # Only consider even kqexp (since terms with odd kqexp are 0)
    even_mask = (kqexp % 2 == 0)
    k2exp = k2exp[even_mask]
    q2exp = q2exp[even_mask]
    kqexp = kqexp[even_mask]
    term_list = term_list[even_mask]

    # Compute contributions for each valid term
    kq_factor = 1 / (1 + kqexp)  # Avoid division by zero
    dim_gen_terms = np.array([
        self.dim_gen(q + k / 2, expden, mDen) for q, k in zip(q2exp, kqexp)
    ])
    k2_factors = k2**(k2exp + kqexp / 2)

    # Combine all terms
    result = np.sum(term_list * dim_gen_terms * k2_factors * kq_factor)
    return result

  def compute_massive_num(self, expnum, expden, k2, mNum, mDen, kmq=True):
    """
    Computes integrals of the type ((k-q)^2 + mNum)^expnum / (q^2 + mDen)^expden
    without a for loop.
    """
    if mNum == 0:
        return self.dim_result(expnum, expden, k2, mDen)

    # Expand the terms
    term_list, exp_list = self.expand_massive_num(expnum)

    # Extract kmq2exp and mNumexp from exp_list
    kmq2exp = exp_list[:, 0]
    mNumexp = exp_list[:, 1]

    # Compute the contributions in a vectorized way
    factors = (mNum**mNumexp) * np.array(
        [self.dim_result(k, expden, k2, mDen, kmq) for k in kmq2exp]
    )
    result = np.sum(term_list * factors)
    return result

  def tadpole_integral(self, n, M):
    '''
    Computes the integral of (q^2 + M)^(-n).
    '''
    if M == 0 or n <= 0:
        return 0
    elif n == 1:
        return -2 * np.sqrt(np.pi) * np.sqrt(M)
    return self.dim_gen(0, n, M)
  
  def bubble_master(self, k2, M1, M2, **kwargs):
    '''
    Computing the Bubble master integrals
    '''
    m1 = M1/k2
    m2 = M2/k2
    A0 = 2*np.sqrt(m2)+1.j*(m1-m2+1)
    A1 = 2*np.sqrt(m1)+1.j*(m1-m2-1)
    bub_mast = np.sqrt(np.pi/k2)*1.j * (np.log(A1) - np.log(A0) - 2*np.pi*1.j*np.heaviside(A1.imag,0)*np.heaviside(-A0.imag,0))
    return bub_mast
  
  def bubble_integral(self, n1, n2, k2, M1, M2):
    '''
    # function to calculate general Bubble integral.
    # returns the value of the integral of
    # (kmq^2 + M1)^(-n1) * (q^2 + M2)^(-n2)
    # calculated recursively from IBP
    # for this function memoization is done with lru_cache because m1 and m2 can be switched
    '''

    # Base cases
    if n1==0:
      return self.tadpole_integral(n2, M2)
    if n2==0:
      return self.tadpole_integral(n1, M1)
    if n1==1 and n2==1:
      return self.bubble_master(k2, M1, M2)

    k1s = k2 + M1 + M2
    jac = k1s**2 - 4*M1*M2
    dim = 3

    if n1>1:
      nu1 = n1-1
      nu2 = n2
      Ndim = dim - nu1 - nu2

      cpm0 = k1s
      cmp0 = -2*(M2/nu1)*nu2
      c000 = (2*M2-k1s)/nu1*Ndim - 2*M2 + (k1s*nu2)/nu1

    elif n2>1:
      nu1 = n1
      nu2 = n2-1
      Ndim = dim - nu1 - nu2

      cpm0 = -2*M1/nu2*nu1
      cmp0 = k1s
      c000 = (2*M1 - k1s)/nu2*Ndim + (k1s*nu1)/nu2 - 2*M1

    #Code to deal with numerators
    if n1 < 0 or n2 < 0:
      if M1 == 0 and M2 == 0:
        return 0
      if n1 < 0 and n2 > 0:
        # m1 is the mass in the numerator
        # m2 is the mass in the denominator
        return self.compute_massive_num(-n1,n2,k2,M1,M2)
      elif n2 < 0 and n1 > 0:
        # m2 is the mass in the numerator
        # m1 is the mass in the denominator
        return self.compute_massive_num(-n2,n1,k2,M2,M1)
      else:
        # case of NO DENOMINATOR
        return 0

    c000 = c000/jac
    cmp0 = cmp0/jac
    cpm0 = cpm0/jac

    result = (c000*self.bubble_integral(nu1,nu2,k2,M1,M2) + 
              cpm0*self.bubble_integral(nu1 + 1, nu2 - 1, k2, M1, M2) + 
              cmp0*self.bubble_integral(nu1 - 1, nu2 + 1, k2, M1, M2))
    return result
  
  def Prefactor(self, a, y1, y2):
    '''
    Computes prefactor that shows up in Fint
    '''
    CHOP_TOL = self.CHOP_TOL

    y2re = y2.real
    y1re = y1.real

    if abs(y2.imag) < CHOP_TOL and abs(y1.imag) < CHOP_TOL:
      if abs(y1re) >= CHOP_TOL and abs(y2re) >= CHOP_TOL:
        return sqrt(-y1re)*sqrt(-y2re)/(sqrt(a*(y1re)*(y2re)))

      if abs(y1re) < CHOP_TOL and abs(y2re) >= CHOP_TOL:
        return sqrt(-y2re)/sqrt(-a*y2re)

      if abs(y1re) >= CHOP_TOL and abs(y2re) < CHOP_TOL:
        return sqrt(-y1re)/sqrt(-a*(y1re))

      if abs(y1re) < CHOP_TOL and abs(y2re) < CHOP_TOL:
        return 1/sqrt(a)

    elif abs(y2.imag) >= CHOP_TOL and abs(y1.imag) < CHOP_TOL:
      if abs(y1re) >= CHOP_TOL:
        return sqrt(-y1re)*sqrt(-y2)/sqrt(a*y1re*y2)

      if abs(y1re) < CHOP_TOL:
        return sqrt(-y2)/sqrt(-a*y2)

    elif abs(y2.imag) < CHOP_TOL and abs(y1.imag) >= CHOP_TOL:
      if abs(y2re) > CHOP_TOL:
        return sqrt(-y1)*sqrt(-y2re)/(sqrt(a*y1*y2re))

      if abs(y2re) < CHOP_TOL:
        return sqrt(-y1)/sqrt(-a*y1)

    else:
      # case where abs(y2.imag) >= CHOP_TOL and abs(y1.imag) >= CHOP_TOL
      return sqrt(-y1)*sqrt(-y2)/sqrt(a*y1*y2) #Maybe not -y1 and -y2, note for later

  def Antideriv(self, x, y1, y2, x0):
    '''
    Calculates the antiderivative (equal to arctan times a prefactor)
    '''
    CHOP_TOL = self.CHOP_TOL
    almosteq = lambda z1, z2, tol: np.abs(z1 - z2) < tol

    if almosteq(x0,y2,CHOP_TOL)==1:
      # case where x0 = y2 = 0 or 1
      if almosteq(x,y2,CHOP_TOL)==1:
        return 0
      else:
        return 2.*sqrt(x-y1)/(-x0+y1)/sqrt(x-y2) #Gives an error when y1=x0=1 for some reason...SG: this is because (-x0+y1) is in the denominator.

    if abs(x0-y1) < CHOP_TOL:
      print('WARNING: switching var in Antideriv')
      #x0 = y2 = 0 or 1
      return self.Antideriv(x,y2,y1,x0)

    prefac = 2/(sqrt(-x0+y1)*sqrt(x0-y2))
    temp = sqrt(x-y1)*sqrt(x0-y2)/sqrt(-x0+y1)
    LimArcTan = 0

    if x == 1 and almosteq(1, y2, CHOP_TOL)==1:
      LimArcTan = 1j * sqrt(-temp**2) * np.pi/(2*temp)
      return  prefac * LimArcTan
    if x == 0 and almosteq(0, y2, CHOP_TOL):
      LimArcTan = sqrt(temp**2) * np.pi/(2*temp)
      return  prefac * LimArcTan

    return prefac*np.atan(temp/sqrt(x-y2))
  
  def Fint(self, aa, y1, y2, x0):
    '''
    Functions that calculates Fint appearing in the triangle master integral
    '''
    CHOP_TOL = self.CHOP_TOL

    #In a lot of places the authors use sqrt(-x) instead of sqrt(x)
    #May be because their choice of sqrt uses different branch
    #So maybe I would have to change it back...

    if abs(y2.imag) < CHOP_TOL:
      y2 = y2.real
    if abs(y1.imag) < CHOP_TOL:
      y1 = y1.real
    if abs(x0.imag) < CHOP_TOL:
      x0 = x0.real

    rey1 = y1.real
    imy1 = y1.imag
    rey2 = y2.real
    imy2 = y2.imag
    rex0 = x0.real
    imx0 = x0.imag

    numbranchpoints = 0
    signx0 = 0
    sign = 0

    xsol = []
    xbranch = []
    atanarglist = []
    abscrit = []
    recrit = []
    derivcrit = []

    c = imy1**2*imy2*rex0 - imy1*imy2**2*rex0-imx0**2*imy2*rey1 + imx0*imy2**2*rey1-imy2*rex0**2*rey1 + imy2*rex0*rey1**2 + imx0**2*imy1*rey2-imx0*imy1**2*rey2+imy1*rex0**2*rey2-imx0*rey1**2*rey2-imy1*rex0*rey2**2+imx0*rey1*rey2**2
    a = imy1*rex0-imy2*rex0-imx0*rey1+imy2*rey1+imx0*rey2-imy1*rey2
    b = -imx0**2*imy1 + imx0*imy1**2+imx0**2*imy2-imy1**2*imy2-imx0*imy2**2+imy1*imy2**2-imy1*rex0**2+imy2*rex0**2+imx0*rey1**2-imy2*rey1**2-imx0*rey2**2+imy1*rey2**2

    #The case of when x0 is real and 0 < x0 < 1 (B=0). This gives a cut
    if 0 < rex0 < 1 and np.abs(imx0) < CHOP_TOL:
      # Derivative of A(x, y1, y2, x0) w.r.t x
      derivcritx0 = (y1 - y2) / 2 / np.sqrt(-(rex0 - y1)**2) / (rex0 - y2)
      if derivcritx0.real < 0:
          signx0 = 1
      else:
          signx0 = -1
      cutx0 = signx0 * np.pi / (np.sqrt(-rex0 + y1) * np.sqrt(rex0 - y2))

    else:
      cutx0 = 0

    if abs(a) < CHOP_TOL:
      if b != 0:
        xsol = [- c / b]
      else:
        xsol = []

    else:
      if b**2-4*a*c > 0:
        xsol = [(-b + np.sqrt(b**2-4*a*c))/(2*a),(-b - np.sqrt(b**2-4*a*c))/(2*a)]
      else:
        #case where there is no intersection of the real axis (includes double zero)
        xsol = []

    if len(xsol) > 0:

      atanarglist = [np.sqrt(x-y1)*np.sqrt(x0-y2)/(np.sqrt(-x0+y1)*np.sqrt(x-y2)) for x in xsol]
      abscrit = [abs(atanarg) for atanarg in atanarglist]
      recrit = [atanarg.real for atanarg in atanarglist]

      for i in range(len(xsol)):
        if abscrit[i] > 1 and abs(recrit[i])<CHOP_TOL:
          numbranchpoints += 1
          xbranch.append(xsol[i])

    if numbranchpoints == 1:
      derivcrit = [np.sqrt(x0-y2)/np.sqrt(-x0+y1)*(1/(2*np.sqrt(x-y1)*np.sqrt(x-y2)) -np.sqrt(x-y1)/(2*(x-y2)*np.sqrt(x-y2))) for x in xbranch]
      if derivcrit[0].real < 0:
        sign = 1
      else:
        sign = -1
      cut = sign*np.pi*2/(np.sqrt(-x0+y1)*np.sqrt(x0-y2))
    else:
      cut = 0

    prefac0 = self.Prefactor(aa,y1,y2)
    result = prefac0*(np.sqrt(np.pi)/2)*(cut + cutx0 + self.Antideriv(1,y1,y2,x0) - self.Antideriv(0,y1,y2,x0))

    return result
  
  def TrMxy(self, y, k2_1, k2_2, k2_3, M1, M2, M3):
    CHOP_TOL = self.CHOP_TOL
    k21, k22, k23 = k2_1, k2_2, k2_3
    Diakr = lambda a, b, c: b**2-4*a*c

    Num1 = 4*k22*y+2*k21-2*k22-2*k23
    Num0 = -4*k22*y+2*M2-2*M3+2*k22
    DeltaR2 = -k21*y+k23*y-k23
    DeltaR1 = -M2*y+M3*y+k21*y-k23*y+M1-M3+k23
    DeltaR0 = M2*y-M3*y+M3
    DeltaS2 = -k21**2+2*k21*k22+2*k21*k23-k22**2+2*k22*k23-k23**2
    DeltaS1 = -4*M1*k22-2*M2*k21+2*M2*k22+2*M2*k23+2*M3*k21+2*M3*k22-2*M3*k23-2*k21*k22+2*k22**2-2*k22*k23
    DeltaS0 = -M2**2+2*M2*M3-2*M2*k22-M3**2-2*M3*k22-k22**2

    DiakrS = np.sqrt(Diakr(DeltaS2, DeltaS1, DeltaS0))
    solS1 = (-DeltaS1+DiakrS)/2/DeltaS2
    solS2 = (-DeltaS1-DiakrS)/2/DeltaS2

    cf2 = -(Num1*solS2+Num0)/DiakrS
    cf1 = (Num1*solS1+Num0)/DiakrS

    DiakrR = np.sqrt(Diakr(DeltaR2, DeltaR1, DeltaR0))
    solR1 = ((-DeltaR1+DiakrR)/2)/DeltaR2
    solR2 = ((-DeltaR1-DiakrR)/2)/DeltaR2

    if abs(cf1) < CHOP_TOL:
      # neglect cf1
      return cf2*self.Fint(DeltaR2, solR1, solR2, solS2)
    elif abs(cf2) < CHOP_TOL:
      # neglect cf2
      return cf1*self.Fint(DeltaR2, solR1, solR2, solS1)
    else:
      return cf2*self.Fint(DeltaR2, solR1, solR2, solS2)+cf1*self.Fint(DeltaR2, solR1, solR2, solS1)

  def TriaMasterZeroMasses(self, k2_1, k2_2, k2_3):
    '''
    Case for triangle master integrals where all masses vanish
    '''
    return np.pi*np.sqrt(np.pi/(k2_1*k2_2*k2_3))

  def triangle_master(self, k2_1, k2_2, k2_3, M1, M2, M3):
    '''
    --- masses are squared as in the paper ---
    Calculates triangle master integral
    '''
    if M1 == 0 and M2 == 0 and M3 == 0:
      return self.TriaMasterZeroMasses(k2_1, k2_2, k2_3)
    return self.TrMxy(1, k2_1, k2_2, k2_3, M1, M2, M3)-self.TrMxy(0, k2_1, k2_2, k2_3, M1, M2, M3)
  
  def num_one_pow(self, d1, d2, denk2, M1, M2):
    # integral of k.q/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by sqrt(k^2) 
    # coef in front of k_i
    coef = (self.bubble_integral(d1,d2-1,denk2,M1,M2) - 
            self.bubble_integral(d1-1,d2,denk2,M1,M2) - 
            (denk2 + M2 - M1)*self.bubble_integral(d1,d2,denk2,M1,M2))
    coef = coef/(2*denk2)
    return coef

  def num_two_pow(self, d1, d2, denk2, m1, m2):
    # integral of (k.q)^2/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^2
    # denk2 are magnitudes of external momenta
    coef1 = -(self.bubble_integral(d1,d2-2,denk2,m1,m2) - 2*(denk2 + m2 - m1)*self.bubble_integral(d1,d2-1,denk2,m1,m2) + (denk2 + m2 - m1)**2*self.bubble_integral(d1,d2,denk2,m1,m2)
      -2*self.bubble_integral(d1-1,d2-1,denk2,m1,m2) + 2*(denk2 + m2 - m1)*self.bubble_integral(d1-1,d2,denk2,m1,m2) + self.bubble_integral(d1-2,d2,denk2,m1,m2))/(8*denk2) + self.bubble_integral(d1-1,d2,denk2,m1,m2)/2 - m1*self.bubble_integral(d1,d2,denk2,m1,m2)/2
    coef2 = self.bubble_integral(d1-1,d2,denk2,m1,m2) - m1*self.bubble_integral(d1,d2,denk2,m1,m2) - 3*coef1
    return coef1, coef2
  
  def num_three_pow(self, d1, d2, denk2, m1, m2):
    # integral of (k.q)^3/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^3
    # coefficients were generated in Mathematica
    BubN = self.bubble_integral
    aux0=((3*(m1-m2)*(m1-m2)+(2.*(denk2*(m1+m2))))-(denk2**2))*(BubN(-1 + d1, d2, denk2, m1, m2))
    aux1=((denk2**2)+((((m1-m2)*(m1-m2)))+(2.*(denk2*(m1+m2)))))*(BubN(d1, d2, denk2, m1, m2))
    aux2=(3*(((denk2+m2)-m1)*(BubN(d1, -2 + d2, denk2, m1, m2))))+(((denk2+m2)-m1)*aux1)
    aux3=(-2.*((denk2+((-3*m1)+(3*m2)))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))))+(aux0+aux2)
    aux4=(-3*(BubN(-2 + d1, -1 + d2, denk2, m1, m2)))+((3*(BubN(-1 + d1, -2 + d2, denk2, m1, m2)))+aux3)
    aux5=((3*(denk2**2))+((-2.*(denk2*(m1+(-3*m2))))+(3*(((m1-m2)*(m1-m2))))))*(BubN(d1, -1 + d2, denk2, m1, m2))
    aux6=((((BubN(-3 + d1, d2, denk2, m1, m2))+aux4)-aux5)-(BubN(d1, -3 + d2, denk2, m1, m2)))-((denk2+((3*m1)+(-3*m2)))*(BubN(-2 + d1, d2, denk2, m1, m2)))
    coef1=3*aux6/(16 * denk2 * sqrt(denk2))
    coef2 = 1/(2*sqrt(denk2))*(BubN(d1-1,d2-1,denk2,m1,m2) - BubN(d1-2,d2,denk2,m1,m2)
      -(denk2 + m2 - 2*m1)*BubN(d1-1,d2,denk2,m1,m2) - m1*BubN(d1,d2-1,denk2,m1,m2)
      +(denk2 + m2 - m1)*m1*BubN(d1,d2,denk2,m1,m2))-5*coef1/3
    return coef1, coef2
  
  def num_four_pow(self, d1, d2, denk2, m1, m2):
    # integral of (k.q)^4/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^3
    # coefficients were generated in Mathematica
    BubN = self.bubble_integral

    aux0=((3*(((denk2+m1)**2)))+((-2.*((denk2+(3*m1))*m2))+(3*(m2**2))))*(BubN(-2 + d1, d2, denk2, m1, m2))
    aux1=((denk2**2)+((-3*(((m1-m2)**2)))+(-2.*(denk2*(m1+m2)))))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))
    aux2=((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))*(BubN(-1 + d1, d2, denk2, m1, m2))
    aux3=((3*(denk2**2))+((-2.*(denk2*(m1+(-3*m2))))+(3*(((m1-m2)**2)))))*(BubN(d1, -2 + d2, denk2, m1, m2))
    aux4=((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))*(BubN(d1, -1 + d2, denk2, m1, m2))
    aux5=((((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))**2))*(BubN(d1, d2, denk2, m1, m2))
    aux6=(-4.*(((denk2+m2)-m1)*(BubN(d1, -3 + d2, denk2, m1, m2))))+((2.*aux3)+((-4.*(((denk2+m2)-m1)*aux4))+aux5))
    aux7=(4.*aux1)+((-4.*(((denk2+m1)-m2)*aux2))+((BubN(d1, -4 + d2, denk2, m1, m2))+aux6))
    aux8=(4.*((denk2+((-3*m1)+(3*m2)))*(BubN(-1 + d1, -2 + d2, denk2, m1, m2))))+aux7
    aux9=(4.*((denk2+((3*m1)+(-3*m2)))*(BubN(-2 + d1, -1 + d2, denk2, m1, m2))))+((2.*aux0)+((-4.*(BubN(-1 + d1, -3 + d2, denk2, m1, m2)))+aux8))
    aux10=(-4.*(((denk2+m1)-m2)*(BubN(-3 + d1, d2, denk2, m1, m2))))+((6.*(BubN(-2 + d1, -2 + d2, denk2, m1, m2)))+aux9)
    aux11=(BubN(-4 + d1, d2, denk2, m1, m2))+((-4.*(BubN(-3 + d1, -1 + d2, denk2, m1, m2)))+aux10)
    coef1=3*aux11/(128*denk2*denk2)

    aux0=((denk2**2)+((-15.*(((m1-m2)**2)))+(-6.*(denk2*(m1+m2)))))*(BubN(-2 + d1, d2, denk2, m1, m2))
    aux1=-12.*(((3*denk2)+((-5.*m1)+(5.*m2)))*(BubN(-1 + d1, -2 + d2, denk2, m1, m2)))
    aux2=((denk2**2)+((-2.*(denk2*(m1+(-3*m2))))+(5.*(((m1-m2)**2)))))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))
    aux3=((denk2**3)+((5.*((m1-m2)**3))+(3*(denk2*((m1-m2)*(m1+(3*m2)))))))-((denk2**2)*(m1+(3*m2)))
    aux4=((5.*(denk2**2))+((5.*(((m1-m2)**2)))+(denk2*((-6.*m1)+(10.*m2)))))*(BubN(d1, -2 + d2, denk2, m1, m2))
    aux5=((5.*(denk2**2))+((5.*(((m1-m2)**2)))+(2.*(denk2*(m1+(5.*m2))))))*(BubN(d1, -1 + d2, denk2, m1, m2))
    aux6=(20.*(((denk2+m2)-m1)*(BubN(d1, -3 + d2, denk2, m1, m2))))+((-6.*aux4)+(4.*(((denk2+m2)-m1)*aux5)))
    aux7=(4.*(aux3*(BubN(-1 + d1, d2, denk2, m1, m2))))+((-5.*(BubN(d1, -4 + d2, denk2, m1, m2)))+aux6)
    aux8=(2.*aux0)+((20.*(BubN(-1 + d1, -3 + d2, denk2, m1, m2)))+(aux1+((12.*aux2)+aux7)))
    aux9=(12.*((denk2+((-5.*m1)+(5.*m2)))*(BubN(-2 + d1, -1 + d2, denk2, m1, m2))))+aux8
    aux10=(4.*((denk2+((5.*m1)+(-5.*m2)))*(BubN(-3 + d1, d2, denk2, m1, m2))))+((-30.*(BubN(-2 + d1, -2 + d2, denk2, m1, m2)))+aux9)
    aux11=(-5.*(BubN(-4 + d1, d2, denk2, m1, m2)))+((20.*(BubN(-3 + d1, -1 + d2, denk2, m1, m2)))+aux10)
    aux12=((5.*(denk2**2))+((5.*(((m1-m2)**2)))+(denk2*((-6.*m1)+(10.*m2)))))*(BubN(d1, d2, denk2, m1, m2))
    aux13=(denk2**-2.)*(aux11-(((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))*aux12))
    coef2=3*aux13/64

    aux0=-60.*(((3*denk2)+((-7.*m1)+(7.*m2)))*(BubN(-2 + d1, -1 + d2, denk2, m1, m2)))
    aux1=((3*(denk2**2))+((-10.*(denk2*(m1+(-3*m2))))+(35.*(((m1-m2)**2)))))*(BubN(-2 + d1, d2, denk2, m1, m2))
    aux2=((3*(denk2**2))+((7.*(((m1-m2)**2)))+(denk2*((-6.*m1)+(10.*m2)))))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))
    aux3=(-9.*((denk2**2)*(m1+(-5.*m2))))+((15.*(denk2*((m1+(-5.*m2))*(m1-m2))))+(-35.*((m1-m2)**3)))
    aux4=((7.*(denk2**2))+((7.*(((m1-m2)**2)))+(2.*(denk2*((-5.*m1)+(7.*m2))))))*(BubN(d1, -2 + d2, denk2, m1, m2))
    aux5=((7.*(denk2**2))+((-2.*(denk2*(m1+(-7.*m2))))+(7.*(((m1-m2)**2)))))*(((denk2+m2)-m1)*(BubN(d1, -1 + d2, denk2, m1, m2)))
    aux6=(35.*((m1-m2)**4.))+(6.*((denk2**2)*((3*(m1**2))+((-30.*(m1*m2))+(35.*(m2**2))))))
    aux7=(-20.*((denk2**3)*(m1+(-7.*m2))))+((-20.*(denk2*((m1+(-7.*m2))*(((m1-m2)**2)))))+aux6)
    aux8=(30.*aux4)+((-20.*aux5)+(((35.*(denk2**4.))+aux7)*(BubN(d1, d2, denk2, m1, m2))))
    aux9=(35.*(BubN(d1, -4 + d2, denk2, m1, m2)))+((-140.*(((denk2+m2)-m1)*(BubN(d1, -3 + d2, denk2, m1, m2))))+aux8)
    aux10=(-60.*aux2)+((4.*(((5.*(denk2**3))+aux3)*(BubN(-1 + d1, d2, denk2, m1, m2))))+aux9)
    aux11=(60.*(((5.*denk2)+((-7.*m1)+(7.*m2)))*(BubN(-1 + d1, -2 + d2, denk2, m1, m2))))+aux10
    aux12=(210.*(BubN(-2 + d1, -2 + d2, denk2, m1, m2)))+(aux0+((6.*aux1)+((-140.*(BubN(-1 + d1, -3 + d2, denk2, m1, m2)))+aux11)))
    aux13=(-140.*(BubN(-3 + d1, -1 + d2, denk2, m1, m2)))+((20.*((denk2+((-7.*m1)+(7.*m2)))*(BubN(-3 + d1, d2, denk2, m1, m2))))+aux12)
    coef3=((35.*(BubN(-4 + d1, d2, denk2, m1, m2)))+aux13)/(128*denk2*denk2)

    return coef1, coef2, coef3
  
  def tri_dim(self, n1, d1, d2, numk2, denk2, ksum2, m1, m2):
    # integral of (numk-q)^2n1/(q^2+m1)^2d1/((denk+q)^2+m2)^2d2
    # numerator (numk-q)^2n1 is massless
    # m1 is mass of d1 propagator, which is (q^2+m1)^2d1,
    # m2 is mass of d2 propagator, which is ((denk+q)^2+m2)^2d2
    BubN = self.bubble_integral
    num_one_pow = self.num_one_pow
    num_two_pow = self.num_two_pow
    num_three_pow = self.num_three_pow
    num_four_pow = self.num_four_pow
    k1dotk2 = self.k1dotk2

    list_length = int((1 + n1)*(2 + n1)/2)
    k2exp = 0
    q2exp = 0
    kqexp = 0
    term_list = np.zeros((list_length,), dtype = np.longlong)
    exp_list = np.zeros((list_length,3), dtype = np.longlong)
    term_list, exp_list = self.num_terms(n1,True)

    res_list = 0
    term = 0

    for i in range(list_length):
      k2exp = exp_list[i][0]
      q2exp = exp_list[i][1]
      kqexp = exp_list[i][2]
      if kqexp == 0:
        # in this case our numerator is just (q2)^q2exp
        if q2exp == 0:
          term = BubN(d1,d2,denk2,m1,m2)
        elif q2exp == 1:
          term = BubN(d1-1,d2,denk2,m1,m2)-m1*BubN(d1,d2,denk2,m1,m2)
        elif q2exp == 2:
          term = BubN(d1-2,d2,denk2,m1,m2) -2*m1*BubN(d1-1,d2,denk2,m1,m2) + m1**2*BubN(d1,d2,denk2,m1,m2)
        elif q2exp == 3:
          term = BubN(d1-3,d2,denk2,m1,m2) - 3*m1*BubN(d1-2,d2,denk2,m1,m2) + 3*m1**2*BubN(d1-1,d2,denk2,m1,m2) - m1**3*BubN(d1,d2,denk2,m1,m2)
        elif q2exp == 4:
          term = BubN(d1-4,d2,denk2,m1,m2) - 4*m1*BubN(d1-3,d2,denk2,m1,m2) + 6*m1**2*BubN(d1-2,d2,denk2,m1,m2) - 4*m1**3*BubN(d1-1,d2,denk2,m1,m2) + m1**4*BubN(d1,d2,denk2,m1,m2)
        else:
          print('exceeded calculable power')

      elif kqexp == 1:
        if q2exp == 0:
          term = num_one_pow(d1,d2,denk2,m1,m2)*k1dotk2(numk2,denk2,ksum2)
        elif q2exp == 1:
          term = (num_one_pow(d1-1,d2,denk2,m1,m2)-m1*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
        elif q2exp == 2:
          term = (num_one_pow(d1-2,d2,denk2,m1,m2) - 2*m1*num_one_pow(d1-1,d2,denk2,m1,m2) + m1**2*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
        elif q2exp == 3:
          term = (num_one_pow(d1-3,d2,denk2,m1,m2) - 3*m1*num_one_pow(d1-2,d2,denk2,m1,m2) + 3*m1**2*num_one_pow(d1-1,d2,denk2,m1,m2) - m1**3*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
        else:
          print('exceeded calculable power')

        # print('term after second if', term)
      elif kqexp == 2:
        delta_coef, dkcoef = num_two_pow(d1,d2,denk2,m1,m2)
        if q2exp == 0:
          term = (numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/denk2*dkcoef)
        elif q2exp == 1:
          delta_coef2, dkcoef2 = num_two_pow(d1-1,d2,denk2,m1,m2)
          term = -m1*(numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef)
          term += (numk2*delta_coef2 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef2)
        elif q2exp == 2:
          delta_coef2, dkcoef2 = num_two_pow(d1-1,d2,denk2,m1,m2)
          delta_coef3, dkcoef3 = num_two_pow(d1-2,d2,denk2,m1,m2)
          term = (numk2*delta_coef3 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef3)
          term += -2*m1*(numk2*delta_coef2 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef2)
          term += m1**2*(numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/denk2*dkcoef)
        else:
          print('exceeded calculable power')

      elif kqexp == 3:
        delta_coef, dkcoef = num_three_pow(d1,d2,denk2,m1,m2)
        if q2exp == 0:
          term = (numk2*delta_coef*k1dotk2(numk2,denk2,ksum2)/(sqrt(denk2)) + dkcoef*k1dotk2(numk2,denk2,ksum2)**3/(denk2*sqrt(denk2)))
        elif q2exp == 1:
          delta_coef2, dkcoef2 = num_three_pow(d1-1,d2,denk2,m1,m2)
          term = (numk2*delta_coef2*k1dotk2(numk2,denk2,ksum2)/(sqrt(denk2)) + dkcoef2*k1dotk2(numk2,denk2,ksum2)**3/(denk2*sqrt(denk2)))
          term += -m1*(numk2*delta_coef*k1dotk2(numk2,denk2,ksum2)/(sqrt(denk2)) + dkcoef*k1dotk2(numk2,denk2,ksum2)**3/(denk2*sqrt(denk2)))
        else:
          print('exceeded calculable power')

      elif kqexp == 4:
        # print('using power 4')
        if q2exp == 0:
          coef1, coef2, coef3 = num_four_pow(d1,d2,denk2,m1,m2)
          term = coef1*numk2**2 + numk2*k1dotk2(numk2,denk2,ksum2)**2*coef2/denk2 + coef3*k1dotk2(numk2,denk2,ksum2)**4/denk2**2
        else:
          print(kqexp, q2exp, 'kqexp, q2exp')
          print('exceeded calculable power')

      if kqexp > 4:
        print(kqexp, q2exp, 'kqexp, q2exp')
        print('exceeded calculable power')

      res_list += term*term_list[i]*numk2**(k2exp)
    return res_list
  
  def tri_dim_two(self, n1,  n2,  d1, numk21,  numk22,  ksum2, dm):
    # integral of (k1 - q)^2^n1 (k2 + q)^2^n2/(q2+dm)^d1

    # term_list1 are the coefficients of (k1 - q)^2^n1 corresponding to the exponents in exp_list
    # exp_list1 are the exponents of (k1 - q)^2^n1 of the form k1^2^k2exp1*q^2^q2exp1*(k.q)^kqexp1,
    # written as (k2exp1, q2exp1, kqexp1)

    # term_list2 are the coefficients of (k2 + q)^2^n2 corresponding to the exponents in exp_list
    # exp_list2 are the exponents of (k2 + q)^2^n2 of the form k2^2^k2exp2*q^2^q2exp2*(k.q)^kqexp2,
    # written as (k2exp2, q2exp2, kqexp2)

    num_terms = self.num_terms
    dim_gen = self.dim_gen
    k1dotk2 = self.k1dotk2


    list_length_n1 = int((1 + n1)*(2 + n1)/2)
    list_length_n2 = int((1 + n2)*(2 + n2)/2)
    k2exp1 = 0
    q2exp1 = 0
    k2exp2 = 0
    q2exp2 = 0
    k2exp = 0
    q2exp = 0
    kqexp1 = 0
    kqexp2 = 0
    kqexp = 0

    term_list1_arr = np.zeros((list_length_n1,), dtype = np.longlong)
    term_list1 = term_list1_arr

    exp_list1_arr = np.zeros((list_length_n1,3), dtype = np.longlong)
    exp_list1 = exp_list1_arr

    term_list2 = np.zeros((list_length_n2,), dtype = np.longlong)
    exp_list2 = np.zeros((list_length_n2,3), dtype = np.longlong)

    term_list1, exp_list1 = num_terms(n1,True)
    term_list2, exp_list2 = num_terms(n2,False)

    res_list = 0
    term = 0

    for i1 in range(list_length_n1):
      for i2 in range(list_length_n2):
        k2exp1 = exp_list1[i1][0]
        k2exp2 = exp_list2[i2][0]
        q2exp = exp_list1[i1][1] + exp_list2[i2][1]
        kqexp1 = exp_list1[i1][2]
        kqexp2 = exp_list2[i2][2]
        kqexp = kqexp1 + kqexp2

        # term = 0
        if kqexp%2 == 0:
          # if kqexp is odd then the integral vanishes by symmetry q -> -q
          # if kqexp == 8:
          #	print('using power 8')
          if kqexp != 0:
            #cases where kqexp == 2
            if kqexp1 == 2 and kqexp2 == 0:
              term = dim_gen(q2exp+1,d1,dm)*(numk21)**(kqexp1/2)/3
            elif kqexp1 == 0 and kqexp2 == 2:
              term = dim_gen(q2exp+1,d1,dm)*(numk22)**(kqexp2/2)/3
            elif kqexp1 == 1 and kqexp2 == 1:
              term = dim_gen(q2exp+1,d1,dm)*(k1dotk2(numk21,numk22,ksum2))/3

            # cases where kqexp == 4
            elif kqexp1 == 0 and kqexp2 == 4:
              term = dim_gen(q2exp+2,d1,dm)*(numk22**2)/5
            elif kqexp1 == 4 and kqexp2 == 0:
              term = dim_gen(q2exp+2,d1,dm)*(numk21**2)/5
            elif kqexp1 == 1 and kqexp2 == 3:
              term = dim_gen(q2exp+2,d1,dm)*(k1dotk2(numk21,numk22,ksum2)*numk22)/5
            elif kqexp1 == 3 and kqexp2 == 1:
              term = dim_gen(q2exp+2,d1,dm)*(k1dotk2(numk21,numk22,ksum2)*numk21)/5
            elif kqexp1 == 2 and kqexp2 == 2:
              term = dim_gen(q2exp+2,d1,dm)*(numk21*numk22 + 2*(k1dotk2(numk21,numk22,ksum2))**2)/15


            # cases where kqexp == 6
            elif kqexp1 == 6 and kqexp2 == 0:
              term = dim_gen(q2exp + 3, d1, dm)*numk21**3/7
            elif kqexp1 == 0 and kqexp2 == 6:
              term = dim_gen(q2exp + 3, d1, dm)*numk22**3/7
            elif kqexp1 == 5 and kqexp2 == 1:
              term = dim_gen(q2exp + 3, d1, dm)*numk21**2*k1dotk2(numk21,numk22,ksum2)/7
            elif kqexp1 == 1 and kqexp2 == 5:
              term = dim_gen(q2exp + 3, d1, dm)*numk22**2*k1dotk2(numk21,numk22,ksum2)/7
            elif kqexp1 == 4 and kqexp2 == 2:
              term = dim_gen(q2exp + 3,d1,dm)*(numk21**2*numk22 + 4*(k1dotk2(numk21,numk22,ksum2))**2*numk21)/35
            elif kqexp1 == 3 and kqexp2 == 3:
              term = dim_gen(q2exp + 3,d1,dm)*(3*numk21*numk22*k1dotk2(numk21,numk22,ksum2) + 2*(k1dotk2(numk21,numk22,ksum2))**3)/35
            elif kqexp1 == 2 and kqexp2 == 4:
              term = dim_gen(q2exp + 3,d1,dm)*(numk22**2*numk21 + 4*(k1dotk2(numk21,numk22,ksum2))**2*numk22)/35

            # cases where kqexp == 8
            elif kqexp1 == 4 and kqexp2 == 4:
              term = dim_gen(q2exp + 4,d1,dm)*(3*numk21**2*numk22**2 + 24*numk21*numk22*k1dotk2(numk21,numk22,ksum2)**2 + 8*(k1dotk2(numk21,numk22,ksum2))**4)/315

            else:
              print('ERROR: case not considered', kqexp, q2exp, kqexp1, kqexp2)
          else:
            # case where kqexp == 0
            term = dim_gen(q2exp,d1,dm)

          res_list += term*term_list2[i2]*term_list1[i1]*(numk21)**(k2exp1)*(numk22)**(k2exp2)

    return res_list
  
  def TrianKinem(self, k21, k22, k23, m1, m2, m3):
    # utility function to generate the special variables that simplify the recursion relations

    k1s = k21 + m1 + m2
    k2s = k22 + m2 + m3
    k3s = k23 + m3 + m1

    jac = -4*m1*m2*m3 + k1s**2*m3 + k2s**2*m1 + k3s**2*m2 - k1s*k2s*k3s
    jac = 2*jac

    ks11 = (-4*m1*m2 + k1s**2)/jac
    ks12 = (-2*k3s*m2 + k1s*k2s)/jac
    ks22 = (-4*m2*m3 + k2s**2)/jac
    ks23 = (-2*k1s*m3 + k2s*k3s)/jac
    ks31 = (-2*k2s*m1+k1s*k3s)/jac
    ks33 = (-4*m1*m3+k3s**2)/jac

    #cdef mpc[:] kinems = np.array([jac,ks11,ks22,ks33,ks12,ks23,ks31], dtype = mpc)
    #kinems = [jac,ks11,ks22,ks33,ks12,ks23,ks31]
    return  np.array([jac,ks11,ks22,ks33,ks12,ks23,ks31])
  
  def triangle_integral(self, n1, n2, n3, k21, k22, k23, m1, m2, m3):
    # calculates the general triangle integral given by
    # (k1mq^2 + m1)^(-n1) * (q^2 + m2)^(-n2) * (k2pq^2 + m3)^(-n3)
    # mi_ind are integer indices to indicate the complex mass in the cache
    BubN = self.bubble_integral
    TriaMaster = self.triangle_master
    tri_dim = self.tri_dim
    tri_dim_two = self.tri_dim_two
    TrianKinem = self.TrianKinem

    if n1 == 0:
      return BubN(n2, n3, k22, m2, m3)
    if n2 == 0:
      return BubN(n3, n1, k23, m3, m1)
    if n3 == 0:
      return BubN(n1, n2, k21, m1, m2)

    if n1 == 1 and n2 == 1 and n3 == 1:
      return TriaMaster(k21, k22, k23, m1, m2, m3)

    # deal with a numerator
    if n1 < 0 or n2 < 0 or n3 < 0:
      if n1 < -4 or n2 < -4 or n3 < -4:
        print('ERROR: case not considered -  n1, n2, n3', n1,n2,n3)
      if n1 < 0:
        if n2 > 0 and n3 > 0:
          result = tri_dim(-n1,n2,n3,k21,k22,k23,m2,m3)
          return result
        elif n2 < 0:
          result = tri_dim_two(-n2,-n1,n3,k22,k23,k21,m3)
          return result
        else:
          result = tri_dim_two(-n1,-n3,n2,k21,k22,k23,m2)
          return result
      if n2 < 0:
        if n1 > 0 and n3 > 0:
          result = tri_dim(-n2,n1,n3,k21,k23,k22,m1,m3)
          return result
        if n3 < 0:
          result =  tri_dim_two(-n3,-n2,n1,k23,k21,k22,m1)
          return result
      if n3 < 0:
        if n1 > 0 and n2 > 0:
          result = tri_dim(-n3,n1,n2,k23,k21,k22,m1,m2)
          return result
        print('ERROR: case not considered')

    #recursion relations using IBP
    kinem = TrianKinem(k21, k22, k23, m1, m2, m3)

    #jac = kinem[0]
    ks11 = kinem[1]
    ks22 = kinem[2]
    ks33 = kinem[3]
    ks12 = kinem[4]
    ks23 = kinem[5]
    ks31 = kinem[6]
    dim = 3

    if n1 > 1:
      nu1 = n1 - 1
      nu2 = n2
      nu3 = n3

      Ndim = dim - nu1 - nu2 - nu3

      cpm0 = -ks23
      cmp0 = (ks22*nu2)/nu1
      cm0p = (ks22*nu3)/nu1
      cp0m = -ks12
      c0pm = -(ks12*nu2)/nu1
      c0mp = -(ks23*nu3)/nu1
      c000 = (-nu3+Ndim)*ks12/nu1 - (-nu1+Ndim)*ks22/nu1 + (-nu2+Ndim)*ks23/nu1

    elif n2 > 1:
      nu1 = n1
      nu2 = n2 - 1
      nu3 = n3

      Ndim = dim - nu1 - nu2 - nu3

      cpm0 = (ks33*nu1)/nu2
      cmp0 = -ks23
      cm0p = -(ks23*nu3)/nu2
      cp0m = -(ks31*nu1)/nu2
      c0pm = -ks31
      c0mp = (ks33*nu3)/nu2
      c000 = (-nu1 + Ndim)*ks23/nu2 + (-nu3 + Ndim)*ks31/nu2 - (-nu2 + Ndim)*ks33/nu2

    elif n3 > 1:
      nu1 = n1
      nu2 = n2
      nu3 = n3 - 1

      Ndim = dim - nu1 - nu2 - nu3


      cpm0 = -(ks31*nu1)/nu3
      cmp0 = -(ks12*nu2)/nu3
      cm0p = -ks12
      cp0m = (ks11*nu1)/nu3
      c0pm = (ks11*nu2)/nu3
      c0mp = -ks31
      c000 = -(-nu3 + Ndim)*ks11/nu3 + (-nu1 + Ndim)*ks12/nu3 + (-nu2 + Ndim)*ks31/nu3

    result = (c000*self.triangle_integral(nu1, nu2, nu3, k21,k22,k23,m1,m2,m3)
        + c0mp*self.triangle_integral(nu1, nu2-1, nu3+1, k21,k22,k23,m1,m2,m3)
        + c0pm*self.triangle_integral(nu1, nu2+1, nu3-1, k21,k22,k23,m1,m2,m3)
        + cm0p*self.triangle_integral(nu1-1, nu2, nu3+1, k21,k22,k23,m1,m2,m3)
        + cp0m*self.triangle_integral(nu1+1, nu2, nu3-1, k21,k22,k23,m1,m2,m3)
        + cmp0*self.triangle_integral(nu1-1, nu2+1, nu3, k21,k22,k23,m1,m2,m3)
        + cpm0*self.triangle_integral(nu1+1, nu2-1, nu3, k21,k22,k23,m1,m2,m3))

    return result

  @lru_cache(None)
  def L_recursion(self, n1, d1, n2, d2, n3, d3, k21, k22, k23, m1, m2, m3): 
    # Most general integral to solve
    # calculates the L function: integral of
    # k1mq^2n1 * q^2n2 * k2pq^2n3
    # / (k1mq^2+m1)^d1 / (q^2+m2)^d2 / (k2pq^2+m3)^d3

    #mi_ind are integers that represent the masses in the cache

    #note that using hash gives a bug in python version 3.7. Need to use version 3.8 or above

    TriaN = self.triangle_integral

    result = 0
    if n1 == 0 and n2 == 0 and n3 == 0:
      result = TriaN(d1,d2,d3,k21,k22,k23,m1,m2,m3)
      return result
    if d1 == 0 and n1 != 0:
      result = self.L_recursion(0,-n1,n2,d2,n3,d3,k21,k22,k23,0,m2,m3)
      return result
    if d2 == 0 and n2 != 0:
      result = self.L_recursion(n1,d1,0,-n2,n3,d3,k21,k22,k23,m1,0,m3)
      return result
    if d3 == 0 and n3 != 0:
      result = self.L_recursion(n1,d1,n2,d2,0,-n3,k21,k22,k23,m1,m2,0)
      return result
    if n1 > 0:
      result = self.L_recursion(n1-1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3) - m1*self.L_recursion(n1-1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3)
      return result
    if n2 > 0:
      result = self.L_recursion(n1,d1,n2-1,d2-1,n3,d3,k21,k22,k23,m1,m2,m3) - m2*self.L_recursion(n1,d1,n2-1,d2,n3,d3,k21,k22,k23,m1,m2,m3)
      return result
    if n3 > 0:
      result = self.L_recursion(n1,d1,n2,d2,n3-1,d3-1,k21,k22,k23,m1,m2,m3) - m3*self.L_recursion(n1,d1,n2,d2,n3-1,d3,k21,k22,k23,m1,m2,m3)
      return result
    if n1 < 0 :
      result = (self.L_recursion(n1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3) - self.L_recursion(n1+1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3))/m1
      return result
    if n2 < 0 :
      result = (self.L_recursion(n1,d1,n2,d2-1,n3,d3,k21,k22,k23,m1,m2,m3) - self.L_recursion(n1,d1,n2+1,d2,n3,d3,k21,k22,k23,m1,m2,m3))/m2
      return result
    if n3 < 0:
      result = (self.L_recursion(n1,d1,n2,d2,n3,d3-1,k21,k22,k23,m1,m2,m3) - self.L_recursion(n1,d1,n2,d2,n3+1,d3,k21,k22,k23,m1,m2,m3))/m3
      return result
    else:
      print("Error: case not considered in L_recursion")

  #Computing redshift terms
  def redshift_term(self, qzexp, kmq2exp, q2exp, d1, d2, M1, M2, k2, kz):
    #computes integral of (q.z)^(qzexp) * (k-q)^(2*kmq2exp)*q2^(q2exp)*((k-q)^2 + M1)^(-d1)*(q^2 + M2)^(-d2)

    PI = np.pi
    Ltrian = self.L_recursion

    #in the case of no redshift 
    if qzexp == 0: 
      answer = Ltrian(kmq2exp, d1, q2exp, d2, 0, 0, k2, 0, 0, M1, M2, 0)

      return answer*(PI**(3/2)/(2*PI)**3)

    #Use L-recursion so either q2exp and/or d2 equals 0 and kmq2exp and/or d1 equals 0.
    if q2exp > 0 and d2 > 0:
      return self.redshift_term(qzexp, kmq2exp, q2exp-1, d1, d2-1, M1, M2, k2, kz) - M2*self.redshift_term(qzexp, kmq2exp, q2exp-1, d1, d2, M1, M2, k2, kz)

    if q2exp < 0 and d2 > 0:
      return (1/M2)*(self.redshift_term(qzexp, kmq2exp, q2exp, d1, d2-1, M1, M2, k2, kz) - self.redshift_term(qzexp, kmq2exp, q2exp+1, d1, d2, M1, M2, k2, kz))

    if kmq2exp > 0 and d1 > 0:
      return self.redshift_term(qzexp, kmq2exp-1, q2exp, d1-1, d2, M1, M2, k2, kz) - M1*self.redshift_term(qzexp, kmq2exp-1, q2exp, d1, d2, M1, M2, k2, kz)

    if kmq2exp < 0 and d1 > 0:
      return (1/M1)*(self.redshift_term(qzexp, kmq2exp, q2exp, d1-1, d2, M1, M2, k2, kz) - self.redshift_term(qzexp, kmq2exp+1, q2exp, d1, d2, M1, M2, k2, kz))

    result = 0 
    #Base cases when either q2exp=0 or/and d1=0 and kmq2exp=0 or/and d2=0.
    if (kmq2exp == 0 and d1 >= 0) and (q2exp == 0 and d2 >= 0): #TC2 type integral
      #This is integral of (q.z)^(qzexp)*((k-q)^2 + M1)^(-d1)*(q^2 + M2)^(-d2)
        
      if qzexp == 1:
        coef1 = self.num_one_pow(d2, d1, k2, M2, M1) 
        term = -1*(kz*coef1) #Minus sign since num_()_pow calcultes with kpq and not kmq
        
      if qzexp == 2:
        coef1, coef2 = self.num_two_pow(d2, d1, k2, M2, M1)
        term = (kz**2/k2)*coef2 + coef1
        
      if qzexp == 3:
        coef1, coef2 = self.num_three_pow(d2, d1, k2, M2, M1)
        term = -1*(coef1*kz/sqrt(k2) + coef2*kz**3/(k2*sqrt(k2))) #Minus sign since num_()_pow calculates with kpq and not kmq

      if qzexp == 4:
        coef1, coef2, coef3 = self.num_four_pow(d2, d1, k2, M2, M1)
        term = coef1 + kz**2*coef2/k2 + coef3*(kz)**4/(k2)**2

      result += term
      return result 
    
    if (kmq2exp == 0 and d1 >= 0) and (q2exp != 0 and d2 == 0):  
      #This is integral of (q.z)^(qzexp) * q2^(q2exp)*((k-q)^2 + M1)^(-d1)
      #After u-sub, integral of ((k-q).z)^(qzexp) * (k-q)^(2*q2exp) * (q^2 + M1)^(-d1)
      
      if q2exp > 0: #Expand brackets and use dim_gen
        list_length_qz = qzexp + 1
        list_length_q2 = (1 + q2exp)*(2 + q2exp)/2

        term_list1, exp_list1 = self.expand_massive_num(qzexp,True)
        term_list2, exp_list2 = self.num_terms(q2exp,False)

        for i in range(list_length_qz):
          for j in range(list_length_q2):
            #Defining exponents
            kzexp = exp_list1[i][0]
            qzexpN = exp_list1[i][1]

            k2expN = exp_list2[j][0]
            q2expN = exp_list2[j][1]
            kqexpN = exp_list2[j][2]

            dproductexp = qzexpN + kqexpN 

            if dproductexp %2 != 0:
              term = 0 

            #cases where dproductexp == 2
            if kqexpN == 2 and qzexpN == 0:
              term = self.dim_gen(q2expN+1, d1, M1)*k2**(kqexp/2)/3 
            elif kqexpN == 0 and qzexpN == 2:
              term = self.dim_gen(q2expN+1, d1, M1)*(1/3) 
            elif kqexpN == 1 and qzexpN == 1:
              term = self.dim_gen(q2expN+1, d1, M1)*(kz/3)

            # cases where dproductexp == 4
            elif kqexpN == 0 and qzexpN == 4:
              term = self.dim_gen(q2expN+2, d1, M1)*(1/5)
            elif kqexpN == 4 and qzexpN == 0:
              term = self.dim_gen(q2expN+2, d1, M1)*(k2)**(2)/5
            elif kqexpN == 1 and qzexpN == 3:
              term = self.dim_gen(q2exp_2+2, d1, M1)*kz*(1/5)
            elif kqexpN == 3 and qzexpN == 1:
              term = self.dim_gen(q2exp_2+2, d1, M1)*kz*(k2)*(1/5)
            elif kqexpN == 2 and qzexpN == 2:
              term = self.dim_gen(q2expN+2, d1, M1)*(k2 + 2*(kz)**2)/15

            # cases where kqexp == 6
            elif kqexpN == 6 and qzexpN == 0:
              term = self.dim_gen(q2expN + 3, d1, M1)*k2**3/7
            elif kqexpN == 0 and qzexpN == 6:
              term = self.dim_gen(q2expN + 3, d1, M1)**3/7
            elif kqexpN == 5 and qzexpN == 1:
              term = self.dim_gen(q2expN + 3, d1, M1)*k2**2*kz
            elif kqexpN == 1 and qzexpN == 5:
              term = self.dim_gen(q2expN + 3, d1, M1)**2*kz/7
            elif kqexpN == 4 and qzexpN == 2:
              term = self.dim_gen(q2expN + 3, d1, M1)*(k2**2 + 4*(kz)**2*k2)/35
            elif kqexpN == 3 and qzexpN == 3:
              term = self.dim_gen(q2expN + 3, d1, M1)*(3*k2*kz + 2*(kz)**3)/35
            elif kqexpN == 2 and qzexpN == 4:
              term = self.dim_gen(q2expN + 3, d1, M1)*(k2 + 4*(kz)**2)/35

            # cases where kqexp == 8
            elif kqexpN == 4 and qzexpN == 4:
              term = self.dim_gen(q2expN + 4, d1, M1)*(3*k2**2 + 24*k2*kz**2 + 8*(kz)**4)/315
            else:
              print('Exceeded computing power', kqexpN, qzexpN)
        
            result += kz**(kzexp)*term_list1[i]*term_list2[j]*(k2)**(k2expN)*term

        return result

      if q2exp < 0: #Use tensor contraction 
                    #Numerically divergent for -q2exp>1. 

        if qzexp == 1:
          coef = self.num_one_pow(-q2exp, d1, k2, 0, M1)
          term = -1*(kz*coef)

        if qzexp == 2:
          coef1, coef2 = self.num_two_pow(-q2exp, d1, k2, 0, M1)
          term = kz**2*(coef2/k2) + coef1

        if qzexp == 3:
          coef1, coef2 = self.num_three_pow(-q2exp, d1, k2, 0, M1)
          term = -1*(coef1*kz/sqrt(k2) + coef2*kz**3/(k2*sqrt(k2)))

        if qzexp == 4:
          coef1, coef2, coef3 = self.num_four_pow(-q2exp, d1, k2, 0, M1)
          term = coef1 + kz**2*coef2/k2 + coef3*(kz)**4/(k2)**2
        
        result += term 
      return result 
      
    if (kmq2exp !=0  and d1 == 0) and (q2exp == 0 and d2 >= 0) : 
      #This is integral of (q.z)^(qzexp)*(k-q)^(2*kmq2exp)*(q2 + M2)^(-d2)
    
      if kmq2exp >= 0:

        #expand (k-q)^(2*kmq2exp) bracket
        term_list, exp_list = self.num_terms(kmq2exp, kmq = True)
        list_length = int((1 + kmq2exp)*(2 + kmq2exp)/2)

        for i in range(list_length):
          #print('loop')
          k2exp = exp_list[i][0]
          q2exp_2 = exp_list[i][1] 
          kqexp = exp_list[i][2]

          dproductexp = kqexp + qzexp 

          if dproductexp != 0:
            #Integrals over odd powers of q vanish by symmetry (q -> -q)
            if dproductexp %2 != 0:
              term = 0
            
            #cases where dproductexp == 2
            elif kqexp == 2 and qzexp == 0:
              term = self.dim_gen(q2exp_2+1, d2, M2)*k2**(kqexp/2)/3 
            elif kqexp == 0 and qzexp == 2:
              term = self.dim_gen(q2exp_2+1, d2, M2)*(1/3) 
            elif kqexp == 1 and qzexp == 1:
              term = self.dim_gen(q2exp_2+1, d2, M2)*(kz/3)

            # cases where dproductexp == 4
            elif kqexp == 0 and qzexp == 4:
              term = self.dim_gen(q2exp_2+2, d2, M2)*(1/5)
            elif kqexp == 4 and qzexp == 0:
              term = self.dim_gen(q2exp_2+2, d2, M2)*(k2)**(2)/5
            elif kqexp == 1 and qzexp == 3:
              term = self.dim_gen(q2exp_2+2, d2, M2)*kz*(1/5)
            elif kqexp == 3 and qzexp == 1:
              term = self.dim_gen(q2exp_2+2, d2, M2)*kz*(k2)*(1/5)
            elif kqexp == 2 and qzexp == 2:
              term = self.dim_gen(q2exp_2+2, d2, M2)*(k2 + 2*(kz)**2)/15

            # cases where kqexp == 6
            elif kqexp == 6 and qzexp == 0:
              term = self.dim_gen(q2exp_2 + 3, d2, M2)*k2**3/7
            elif kqexp == 0 and qzexp == 6:
              term = self.dim_gen(q2exp_2 + 3, d2, M2)**3/7
            elif kqexp == 5 and qzexp == 1:
              term = self.dim_gen(q2exp_2 + 3, d2, M2)*k2**2*kz
            elif kqexp == 1 and qzexp == 5:
              term = self.dim_gen(q2exp + 3, d2, M2)**2*kz/7
            elif kqexp == 4 and qzexp == 2:
              term = self.dim_gen(q2exp_2 + 3,d2,M2)*(k2**2 + 4*(kz)**2*k2)/35
            elif kqexp == 3 and qzexp == 3:
              term = self.dim_gen(q2exp_2 + 3,d2,M2)*(3*k2*kz + 2*(kz)**3)/35
            elif kqexp == 2 and qzexp == 4:
              term = self.dim_gen(q2exp_2 + 3,d2,M2)*(k2 + 4*(kz)**2)/35

            # cases where kqexp == 8
            elif kqexp == 4 and qzexp == 4:
              term = self.dim_gen(q2exp_2 + 4,d2,M2)*(3*k2**2 + 24*k2*kz**2 + 8*(kz)**4)/315
            
            else:
              print('Exceeded computing power', kqexp, qzexp)
              
          else:
            # case where kqexp == 0
            term = self.dim_gen(q2exp,d2,M2)
          
          result += term_list[i]*term*(k2)**(k2exp)
        #print(result, term_list[i]*term*(k2)**(k2exp), kqexp, qzexp, 'result')
        
      if kmq2exp < 0: #These are numerically divergent
        #integral of (q.z)^(qzexp)*(1/(k-q)^2)^(-kmq2exp)*(q2 + M2)^(-d2) 
        if qzexp == 1:
          coef = self.num_one_pow(d2, -kmq2exp, k2, M2, 0)
          term = kz*coef
          result = term

        if qzexp == 2:
          coef1, coef2 = self.num_two_pow(d2, -kmq2exp, k2, M2,0)
          term = kz**2*(coef2/k2) + coef1
          result = term

        if qzexp == 3:
          coef1, coef2 = self.num_three_pow(d2, -kmq2exp, k2, M2,0)
          term = coef1*kz/sqrt(k2) + coef2*kz**3/(k2*sqrt(k2))
          result = term

        if qzexp == 4:
          coef1, coef2, coef3 = self.num_four_pow(d2, -kmq2exp, k2, M2,0)
          term = coef1 + kz**2*coef2/k2 + coef3*(kz)**4/(k2)**2
          result = term
        
        return result 
        
      return result 

    if (kmq2exp != 0 and d1 == 0) and (q2exp != 0 and d2 == 0):
      #This gives integral of (q.z)^(qzexp)*(k-q)^(2kmq2exp)*(q2)^(q2exp) (massless propagators)

      if kmq2exp < 0 and q2exp < 0:
        if qzexp == 1:
          coef = self.num_one_pow(-q2exp, -kmq2exp, k2, 0, 0)
          term = kz*coef
          result = term
        
        if qzexp == 2:
          coef1, coef2 = self.num_two_pow(-q2exp, -kmq2exp, k2, 0, 0)
          term = kz**2*(coef2/k2) + coef1
          result = term

        if qzexp == 3:
          coef1, coef2 = self.num_three_pow(-q2exp, -kmq2exp, k2, 0, 0)
          term = coef1*kz/sqrt(k2) + coef2*kz**3/(k2*sqrt(k2))
          result = term

        if qzexp == 4:
          coef1, coef2, coef3 = self.num_four_pow(-q2exp, -kmq2exp, k2, 0, 0)
          term = coef1 + kz**2*coef2/k2 + coef3*(kz)**4/(k2)**2
          result = term   

      else:
        result = 0

      return result
    

  def redshift_kernel(self, qzexp, kmq2explist, q2explist, k2explist, coeflist, d1, d2, M1, M2, k2, kz):
    #integral of (q.z)^(qzexp)*RSDkernel*((k-q)^2 + M1)^(-d1)*(q^2 + M2)^(-d2)
    #qzexplist, kmq2explist, q2explist, k2exp, coeflist, all specify the RSD kernel

    result = 0
    for i in range(len(coeflist)):

      kmq2exp = kmq2explist[i]
      q2exp = q2explist[i]
      k2exp = k2explist[i]
      coef = coeflist[i]

      result += coef*k2**(k2exp)*self.redshift_term(qzexp, kmq2exp, q2exp, d1, d2, M1, M2, k2, kz)
    
    return result

  def redshift_matrix(self, n, m, k2, kz, qzexp, kmq2explist, q2explist, k2explist, coeflist, nBW=12):
    # Function to compute matrix elements
    #Computes integral of (q.z)^(qzexp)*RSDkernel*f_n(q2)*f_m(|k-q|^2)
    #For us, 0 <= n,m, <= 15 since we have 16 fitting functions

    ini = 0
    I = [ini, 0,0,0,1,1,1,1,0,0,0,0,0,0,0,0] #15 babis functions
    J = [ini, 1,2,3,1,2,3,4,2,3,4,5,1,2,3,4]

    k0 = self.k0
    k2UVlist = self.k2UVlist
    Mlist = self.Mlist
    kappa = self.kappa 

    if nBW:
      out_BW_params = self.BW_fitting_params()
      k2UVlistBW = out_BW_params['k2UVlistBW']
      MlistBW = out_BW_params['MlistBW']
      IBW = out_BW_params['IBW']
      JBW = out_BW_params['JBW']
      for N in range(nBW):
        I.append(IBW[0])
        J.append(JBW[0])
        k2UVlist.append(k2UVlistBW[N])
        Mlist.append(MlistBW[N])

    # print(n,m)
    
    i_n = I[n]
    i_m = I[m]

    j_n = J[n]
    j_m = J[m]

    Mn = Mlist[n]
    Mm = Mlist[m]

    k2UVn = k2UVlist[n]
    k2UVm = k2UVlist[m]

    onesq2 = np.ones(len(q2explist))
    oneskmq2 = np.ones(len(kmq2explist))

    Q2explist = q2explist + i_n*onesq2
    Kmq2explist = kmq2explist + i_m*oneskmq2

    result = 0 + 0*1j
    if m is not False: #Matrix elements for the P22 integral
      if n==0 or m==0:
        if n==0 and m!=0:
          for l1 in range(j_m):
            kap_l1_jm = kappa[l1, j_m -1]

            prefac = (1/(k0)**2)**(i_m)*k2UVn*k2UVm**(l1+1)
            int1 = kap_l1_jm*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l1+1, 1, Mm, Mn, k2, kz) 
            int2 = np.conj(int1) 
            result += prefac*(int1 + int2)
          return result
        
        if n!=0 and m==0:
          for l2 in range(j_n):
            kap_l2_jn = kappa[l2, j_n -1]

            prefac = (1/(k0)**2)**(i_n)*k2UVm*k2UVn**(l2+1)
            int1 = kap_l2_jn*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, 1, l2+1, Mm, Mn, k2, kz) 
            int2 = np.conj(int1)
            result += prefac*(int1 + int2)
          return result
        
        if n==0 and m==0:
          result = k2UVn*k2UVm*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, 1, 1, Mm, Mn, k2, kz)
          return result 

      #Matrix elements with 0 < n,m < 16
      for l1 in range(j_n):
        for l2 in range(j_m):
          kap_l1_jn = kappa[l1, j_n - 1] 
          kap_l2_jm = kappa[l2, j_m - 1]

          prefac = (1/(k0)**2)**(i_n + i_m)*k2UVn**(l1+1)*k2UVm**(l2+1)
          int1 = kap_l1_jn*kap_l2_jm*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, Mn, k2, kz)
          int2 = np.conj(kap_l1_jn)*kap_l2_jm*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, np.conj(Mn), k2, kz)
          int3 = np.conj(int2)
          int4 = np.conj(int1)
          result += prefac*(int1 + int2 + int3 + int4)

      #matrix element with BW function f_nBW(q^2)*f_0(|k-q|^2)
      if n > 15 and m == 0:
        for l1 in range(j_n):
          print(j_n)
          kap_l1_jn = kappa[l1, j_n - 1] 

          prefac = k2UVm*k2UVn**(l1-1)
          int1 = kap_l1_jn*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, 1, l1+1, Mm, Mn, k2, kz)
          int2 = np.conj(int1)
          result += prefac(int1 + int2)

      #matrix element with BW function f_0(q^2)*f_mBW(|k-q|^2)
      if n == 0 and m > 15:
        for l2 in range(j_m):
          kap_l2_jm = kappa[l2, j_m - 1] 

          prefac = k2UVn*k2UVm**(l2-1)
          int1 = kap_l2_jn*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, 1, Mm, Mn, k2, kz)
          int2 = np.conj(int1)
          result += prefac*(int1 + int2)

      #matrix element with BW function f_nBW(q^2)*f_m(|k-q|^2)
      if n > 15 and 0 < m < 16: 
        for l1 in range(j_n): 
          for l2 in range(j_m):
            kap_l1_jBW = kappa[l1, j_n - 1]
            kap_l2_jm = kappa[l2, j_m - 1]

            prefac = (1/(k0)**2)**(i_m)*k2UVn**(l1-1)*k2UVm**(l2+1)
            int1 = kap_l1_jBW*kap_l2_jm*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, Mn, k2, kz)
            int2 = np.conj(kap_l1_jn)*kap_l2_jm*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, np.conj(Mn), k2, kz)
            int3 = np.conj(int2)
            int4 = np.conj(int1)
            result += prefac*(int1 + int2 + int3 + int4)

      #matrix element with BW function f_n(q^2)*f_mBW(|k-q|^2)
      if 0 < n < 16 and m > 15: 
        for l1 in range(j_n): 
          for l2 in range(j_m):
            kap_l1_jn = kappa[l1, j_n - 1]
            kap_l2_jBW = kappa[l2, j_m - 1]

            prefac = (1/(k0)**2)**(i_n)*k2UVn**(l1+1)*k2UVm**(l2-1)
            int1 = kap_l1_jn*kap_l2_jBW*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, Mn, k2, kz)
            int2 = np.conj(kap_l1_jn)*kap_l2_jBW*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, np.conj(Mn), k2, kz)
            int3 = np.conj(int2)
            int4 = np.conj(int1)
            result += prefac*(int1 + int2 + int3 + int4)

      #matrix element with BW function f_nBW(q^2)*f_mBW(|k-q|^2)
      if n > 15 and m > 15:
        for l1 in range(j_n): 
          for l2 in range(j_m):
            kap_l1_jBW = kappa[l1, j_n - 1]
            kap_l2_jBW = kappa[l2, j_m - 1]

            prefac = k2UVn**(l2-1)*k2UVm**(l1-1)
            int1 = kap_l1_jBW*kap_l2_jBW*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, Mn, k2, kz)
            int2 = np.conj(kap_l1_jBW)*kap_l2_jBW*self.redshift_kernel(qzexp, Kmq2explist, Q2explist, k2explist, coeflist, l2+1, l1+1, Mm, np.conj(Mn), k2, kz)
            int3 = np.conj(int2)
            int4 = np.conj(int1)
            result += prefac*(int1 + int2 + int3 + int4)
  
      return result
    

    if m is False: #For P13

      #Fit function matrix elements
      if n==0:
        result += k2UVn*self.redshift_kernel(qzexp, kmq2explist, q2explist, k2explist, coeflist, 0, 1, 0, Mn, k2, kz)
        return result
      
      if 0 < n < 16:
        for l1 in range(j_n):
          kap_l1_jn = kappa[l1, j_n - 1] 

          prefac = (1/(k0)**2)**(i_n)*k2UVn**(l1+1) 
          int1 = kap_l1_jn*self.redshift_kernel(qzexp, kmq2explist, Q2explist, k2explist, coeflist, 0, l1+1, 0, Mn, k2, kz)
          int2 = np.conj(int1)
          result += prefac*(int1 + int2)

      #BW function matrix elements
      if n > 15:
        for l1 in range(j_n):
          kap_l1_jn = kappa[l1, j_n - 1]

          prefac = k2UVn**(l1-1)
          int1 = kap_l1_jn*self.redshift_kernel(qzexp, kmq2explist, Q2explist, k2explist, coeflist, 0, l1+1, 0, Mn, k2, kz)
          int2 = np.conj(int1)
          result += prefac*(int1 + int2)

      return result

  def compute_tracer_intergrals(self, k_sample=np.logspace(-3,np.log10(2),150), save_folder=None):
    if save_folder is None:
      save_folder = self.save_folder

    #Lists for non-redshift power spectrum
    #[k2exp, kmq2exp, q2exp, coef]
    # P22Kernel (2*F2(q, k-q)^2)

    try:
      if self.tracer_matrices is None:
        F22Matrix = np.load(f"{save_folder}/matrices_2F22.npy")
        F3Matrix = np.load(f"{save_folder}/matrices_6F3.npy")
        Idelta2Matrix = np.load(f"{save_folder}/matrices_Idelta2.npy")
        IG2Matrix = np.load(f"{save_folder}/matrices_IG2.npy")
        FG2Matrix = np.load(f"{save_folder}/matrices_FG2.npy")
        Idelta2delta2Matrix = np.load(f"{save_folder}/matrices_Idelta2delta2.npy")
        IG2G2Matrix = np.load(f"{save_folder}/matrices_IG2G2.npy")
        Idelta2G2Matrix = np.load(f"{save_folder}/matrices_Idelta2G2.npy")
        tracer_matrices = F22Matrix, F3Matrix, Idelta2Matrix, IG2Matrix, FG2Matrix, Idelta2delta2Matrix, IG2G2Matrix, Idelta2G2Matrix
        self.tracer_matrices = tracer_matrices
      return self.tracer_matrices
    except:
      pass

    print('Computing the intergrals requied to model the tracer fields...')
    tstart = time() 

    P22Kernel_terms = np.array([
        [0, 0, 0, 75/196],
        [2, -2, 0, -11/392],
        [1, -1, 0, 15/196],
        [2, 0, -2, -11/392],
        [4, -2, -2, 1/98],
        [3, -1, -2, 3/98],
        [1, 1, -2, -15/196],
        [0, 2, -2, 25/392],
        [1, 0, -1, 15/196],
        [3, -2, -1, 3/98],
        [2, -1, -1, 29/196],
        [0, 1, -1, -25/98],
        [1, -2, 1, -15/196],
        [0, -1, 1, -25/98],
        [0, -2, 2, 25/392]
    ])
    # P13Kernel (6*F3(q, -q, k))
    P13Kernel_terms = np.array([
        [0, 0, 0, 85/252],
        [1, -1, 0, 5/28],
        [-1, 1, 0, 1/12],
        [2, 0, -2, -31/252],
        [3, -1, -2, -1/42],
        [1, 1, -2, 11/42],
        [0, 2, -2, -5/84],
        [-1, 3, -2, -1/18],
        [1, 0, -1, -5/252],
        [2, -1, -1, -1/84],
        [0, 1, -1, -13/252],
        [-1, 2, -1, 1/12],
        [-1, 0, 1, -7/36],
        [0, -1, 1, -19/84],
        [-1, -1, 2, 1/12]
    ])
    # Idelta2
    Idelta2_terms = np.array([
        [0, 0, 0, 5/7],
        [1, -1, 0, 3/14],
        [1, 0, -1, 3/14],
        [2, -1, -1, 1/7],
        [0, 1, -1, -5/14],
        [0, -1, 1, -5/14]
    ])
    # IG2
    IG2_terms = np.array([
        [0, 0, 0, -15/28],
        [2, -2, 0, -9/56],
        [1, -1, 0, -13/56],
        [2, 0, -2, -9/56],
        [4, -2, -2, 1/28],
        [3, -1, -2, -1/56],
        [1, 1, -2, 13/56],
        [0, 2, -2, -5/56],
        [1, 0, -1, -13/56],
        [3, -2, -1, -1/56],
        [2, -1, -1, -3/28],
        [0, 1, -1, 5/14],
        [1, -2, 1, 13/56],
        [0, -1, 1, 5/14],
        [0, -2, 2, -5/56]
    ])
    # FG2
    FG2_terms = np.array([
        [0, 0, 0, -13/28],
        [1, -1, 0, -15/14],
        [-1, 1, 0, -9/28],
        [2, 0, -2, 13/28],
        [3, -1, -2, -5/28],
        [1, 1, -2, -9/28],
        [0, 2, -2, -1/28],
        [-1, 3, -2, 1/14],
        [1, 0, -1, -13/28],
        [2, -1, -1, 5/7],
        [0, 1, -1, -3/14],
        [-1, 2, -1, -1/28],
        [-1, 0, 1, 13/28],
        [0, -1, 1, 5/7],
        [-1, -1, 2, -5/28]
    ])
    # IG2G2
    IG2G2_terms = np.array([
        [0, 0, 0, 3/4],
        [2, -2, 0, 3/4],
        [1, -1, 0, 1/2],
        [2, 0, -2, 3/4],
        [4, -2, -2, 1/8],
        [3, -1, -2, -1/2],
        [1, 1, -2, -1/2],
        [0, 2, -2, 1/8],
        [1, 0, -1, 1/2],
        [3, -2, -1, -1/2],
        [2, -1, -1, 1/2],
        [0, 1, -1, -1/2],
        [1, -2, 1, -1/2],
        [0, -1, 1, -1/2],
        [0, -2, 2, 1/8]
    ])
    # Idelta2G2
    Idelta2G2_terms = np.array([
        [0, 0, 0, -1],
        [1, -1, 0, -1],
        [1, 0, -1, -1],
        [2, -1, -1, 1/2],
        [0, -1, 1, 1/2],
        [0, 1, -1, 1/2]
    ])
    #Extracting lists for non-redshift kernel
    P22Kernelk2list = P22Kernel_terms.T[0]
    P22Kernelkmq2list = P22Kernel_terms.T[1]
    P22Kernelq2list = P22Kernel_terms.T[2]
    P22Kernelcoeflist = P22Kernel_terms.T[3]
    P13Kernelk2list = P13Kernel_terms.T[0]
    P13Kernelkmq2list = P13Kernel_terms.T[1]
    P13Kernelq2list = P13Kernel_terms.T[2]
    P13Kernelcoeflist = P13Kernel_terms.T[3]
    Idelta2k2list = Idelta2_terms.T[0]
    Idelta2kmq2list = Idelta2_terms.T[1]
    Idelta2q2list = Idelta2_terms.T[2]
    Idelta2coeflist = Idelta2_terms.T[3]
    IG2k2list = IG2_terms.T[0]
    IG2kmq2list = IG2_terms.T[1]
    IG2q2list = IG2_terms.T[2]
    IG2coeflist = IG2_terms.T[3]
    FG2k2list = FG2_terms.T[0]
    FG2kmq2list = FG2_terms.T[1]
    FG2q2list = FG2_terms.T[2]
    FG2coeflist = FG2_terms.T[3]
    IG2G2k2list = IG2G2_terms.T[0]
    IG2G2kmq2list = IG2G2_terms.T[1]
    IG2G2q2list = IG2G2_terms.T[2]
    IG2G2coeflist = IG2G2_terms.T[3]
    Idelta2G2k2list = Idelta2G2_terms.T[0]
    Idelta2G2kmq2list = Idelta2G2_terms.T[1]
    Idelta2G2q2list = Idelta2G2_terms.T[2]
    Idelta2G2coeflist = Idelta2G2_terms.T[3]

    n_rows = 16 + 12
    m_cols = 16 + 12

    matrices_2F22 = []
    matrices_6F3 = []
    matrices_Idelta2 = []
    matrices_IG2 = []
    matrices_FG2 = []
    matrices_Idelta2delta2 = []
    matrices_IG2G2 = []
    matrices_Idelta2G2 = []

    counter = 0 
    for k in tqdm(k_sample):
      k2 = k**2
      matrix2F22 = np.zeros((n_rows, m_cols), dtype = complex)
      matrix6F3 = np.zeros((n_rows), dtype = complex)
      matrixIdelta2 = np.zeros((n_rows, m_cols), dtype = complex)
      matrixIG2 = np.zeros((n_rows, m_cols), dtype = complex)
      matrixFG2 = np.zeros((n_rows), dtype = complex)
      matrixIdelta2delta2 = np.zeros((n_rows, m_cols), dtype = complex)
      matrixIG2G2 = np.zeros((n_rows, m_cols), dtype = complex)
      matrixIdelta2G2 = np.zeros((n_rows, m_cols), dtype = complex)

      for n in range(n_rows):
        kz = 0
        matrix6F3[n] = self.redshift_matrix(n, False, k2, kz, 0, P13Kernelkmq2list, P13Kernelq2list, P13Kernelk2list, P13Kernelcoeflist)
        matrixFG2[n] = self.redshift_matrix(n, False, k2, kz, 0, FG2kmq2list, FG2q2list, FG2k2list, FG2coeflist)
        for m in range(m_cols):
          matrix2F22[n, m] = self.redshift_matrix(n, m, k2, kz, 0, P22Kernelkmq2list, P22Kernelq2list, P22Kernelk2list, P22Kernelcoeflist)
          matrixIdelta2[n, m] = self.redshift_matrix(n, m, k2, kz, 0, Idelta2kmq2list, Idelta2q2list, Idelta2k2list, Idelta2coeflist)
          matrixIG2[n, m] = self.redshift_matrix(n, m, k2, kz, 0, IG2kmq2list, IG2q2list, IG2k2list, IG2coeflist)
          matrixIdelta2delta2[n, m] = 2*(self.redshift_matrix(n, m, k2, kz, 0, [0], [0], [0], [1]) - self.redshift_matrix(n, m, 0.001**2, kz, 0, [0], [0], [0], [1]))
          matrixIG2G2[n, m] = self.redshift_matrix(n, m, k2, kz, 0, IG2G2kmq2list, IG2G2q2list, IG2G2k2list, IG2G2coeflist)
          matrixIdelta2G2[n, m] = self.redshift_matrix(n, m, k2, kz, 0, Idelta2G2kmq2list, Idelta2G2q2list, Idelta2G2k2list, Idelta2G2coeflist)

      counter += 1
      print('matrix element computed', counter)

      matrices_2F22.append(matrix2F22)
      matrices_6F3.append(matrix6F3)
      matrices_Idelta2.append(matrixIdelta2)
      matrices_IG2.append(matrixIG2)
      matrices_FG2.append(matrixFG2)
      matrices_Idelta2delta2.append(matrixIdelta2delta2)
      matrices_IG2G2.append(matrixIG2G2)
      matrices_Idelta2G2.append(matrixIdelta2G2)

    print(f'...done in {time()-tstart} s')

    if save_folder:
      if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
      np.save(f'{save_folder}/matrices_2F22.npy', matrices_2F22)
      np.save(f'{save_folder}/matrices_6F3.npy', matrices_6F3)
      np.save(f'{save_folder}/matrices_Idelta2.npy', matrices_Idelta2)
      np.save(f'{save_folder}/matrices_IG2.npy', matrices_IG2)
      np.save(f'{save_folder}/matrices_FG2.npy', matrices_FG2)
      np.save(f'{save_folder}/matrices_Idelta2delta2.npy', matrices_Idelta2delta2)
      np.save(f'{save_folder}/matrices_IG2G2.npy', matrices_IG2G2)
      np.save(f'{save_folder}/matrices_Idelta2G2.npy', matrices_Idelta2G2)
    else:
      print('The intergrals were computed, but not saved. To save them for reuse, provide a folder link via save_folder variable.')
    
    self.tracer_matrices = matrices_2F22, matrices_6F3, matrices_Idelta2, matrices_IG2, matrices_FG2, matrices_Idelta2delta2, matrices_IG2G2, matrices_Idelta2G2
    return self.tracer_matrices
  
  def contract_matter_matrices(self, fitparams, PlinSpline, k_sample, BW, redshift):
    #Contracts matrices to compute integrals of linear power spectrums at a specified redshift.
    #IRsummation=True uses the IR-summed LPS instead of the regular one.

    F22Matrix, F3Matrix, Idelta2Matrix, IG2Matrix, FG2Matrix, Idelta2delta2Matrix, IG2G2Matrix, Idelta2G2Matrix = self.tracer_matrices

    F22vals = []
    F13vals = []
    Plinvals = []
    for i in range(len(k_sample)):
      resultF22 = 0
      resultF13 = 0

      NFitnoBW = 16
      NFitBW = 16 + 12
      if BW == False:
        NFit = NFitnoBW
      if BW == True:
        NFit = NFitBW

      for N in range(NFit):
        resultF13 +=  PlinSpline(k_sample[i]) * fitparams[N] * F3Matrix[i][N] #still have to multiply by Plin
        for M in range(NFit):
          resultF22 += fitparams[N]*fitparams[M] * F22Matrix[i][N][M]
          
      F22vals.append(resultF22) 
      F13vals.append(resultF13)

    return F22vals, F13vals
  
  def contract_matrices(self, fitparams, PlinSpline, k_sample):
    #Contracts matrices to compute integrals of linear power spectrums at a specified redshift

    #Import Matrices
    F22Matrix, F3Matrix, Idelta2Matrix, IG2Matrix, FG2Matrix, Idelta2delta2Matrix, IG2G2Matrix, Idelta2G2Matrix = self.tracer_matrices

    F22vals = []
    F13vals = []
    Idelta2vals = []
    IG2vals = []
    FG2vals = []
    Idelta2delta2vals = []
    IG2G2vals = []
    Idelta2G2vals = []
    Plinvals = []

    if self.verbose:
      print('Start contracting matrices...')
    for i in tqdm(range(len(k_sample)), disable=not self.verbose):
      resultF22 = 0
      resultF13 = 0
      resultdelta2 = 0 
      resultIG2 = 0
      resultFG2 = 0
      resultIdelta2delta2 = 0
      resultIG2G2 = 0
      resultIdelta2G2 = 0
      Plinval = PlinSpline(k_sample[i]) 
      NFit = 16
      for N in range(NFit):
        resultF13 += Plinval * fitparams[N] * F3Matrix[i][N]
        resultFG2 += Plinval * fitparams[N] * FG2Matrix[i][N]
        for M in range(NFit):
          resultF22 += fitparams[N]*fitparams[M] * F22Matrix[i][N][M]
          resultdelta2 += fitparams[N]*fitparams[M] * Idelta2Matrix[i][N][M]
          resultIG2 += fitparams[N]*fitparams[M] * IG2Matrix[i][N][M]
          resultIdelta2delta2 += fitparams[N]*fitparams[M] * Idelta2delta2Matrix[i][N][M]
          resultIG2G2 += fitparams[N]*fitparams[M] * IG2G2Matrix[i][N][M]
          resultIdelta2G2 += fitparams[N]*fitparams[M] * Idelta2G2Matrix[i][N][M]
      
      F13vals.append(resultF13)
      FG2vals.append(resultFG2)
      F22vals.append(resultF22) 
      Idelta2vals.append(resultdelta2)
      IG2vals.append(resultIG2)
      Idelta2delta2vals.append(resultIdelta2delta2)
      IG2G2vals.append(resultIG2G2)
      Idelta2G2vals.append(resultIdelta2G2)
      Plinvals.append(Plinval)
    if self.verbose:
      print('...done contracting matrices')
    return F22vals, F13vals, Idelta2vals, IG2vals, FG2vals, Idelta2delta2vals, IG2G2vals, Idelta2G2vals, Plinvals
  
  def _model_P_tracer(self, k_sample, b1, b2, bG2, bGamma3, R2, Pshot, cs2, redshift, plin=None):
    par = self.param

    if plin is None:
      plin = get_Plin(par)

    PlinSpline = interp1d(plin['k'], plin['P'], kind='cubic', fill_value='extrapolate')
    ps_fit = self.fit_P_linear(plin=plin)
    fparams, fparamsBW = ps_fit['alpha_n'], ps_fit['alpha_n_BW']
    design_matrix, design_matrixBW = ps_fit['design_matrix'], ps_fit['design_matrixBW']

    TotFitparams = np.append(fparams, fparamsBW)
    #Contract matrices to get final shapes
    P22vals, P13vals, Idelta2vals, IG2vals, FG2vals, Idelta2delta2vals, IG2G2vals, Idelta2G2vals, Plinvals = self.contract_matrices(TotFitparams, PlinSpline, k_sample)
    
    #Creating splines through calculated values. 
    P22spline = interp1d(k_sample, P22vals, kind='cubic', fill_value='extrapolate')
    P13spline = interp1d(k_sample, P13vals, kind='cubic', fill_value='extrapolate')
    Idelta2spline = interp1d(k_sample, Idelta2vals, kind='cubic', fill_value='extrapolate')
    IG2spline = interp1d(k_sample, IG2vals, kind='cubic', fill_value='extrapolate')
    FG2spline = interp1d(k_sample, FG2vals, kind='cubic', fill_value='extrapolate')
    Idelta2delta2spline = interp1d(k_sample, Idelta2delta2vals, kind='cubic', fill_value='extrapolate')
    IG2G2spline = interp1d(k_sample, IG2G2vals, kind='cubic', fill_value='extrapolate')
    Idelta2G2spline = interp1d(k_sample, Idelta2G2vals, kind='cubic', fill_value='extrapolate')

    Dz = growth_factor(redshift, par)

    pl = PlinSpline(k_sample) * Dz**2
    k2 = k_sample**2 

    P13int = P13spline(k_sample) * Dz**2
    FG2int = FG2spline(k_sample) * Dz**2
    P22int = P22spline(k_sample) * Dz**4
    Idelta2int = Idelta2spline(k_sample) * Dz**4
    IG2int = IG2spline(k_sample) * Dz**4
    Idelta2delta2int = Idelta2delta2spline(k_sample) * Dz**4
    IG2G2int = IG2G2spline(k_sample) * Dz**4
    Idelta2G2int = Idelta2G2spline(k_sample) * Dz**4

    counterterm_noRSD = - 2 * cs2 * k2 * pl
    P1loop = P22int + P13int

    Term1 = (pl +P1loop)
    Term2 = Idelta2int
    Term3 = IG2int
    Term4 = FG2int
    Term5 = Idelta2delta2int
    Term6 = IG2G2int
    Term7 = Idelta2G2int

    counterterm = -2 * b1 * (R2 + cs2 * b1) * k2 * pl
    Pstoch = Pshot 
    result = b1**2 * Term1 + b1*b2 * Term2 + 2*b1*bG2*Term3 + b1*(2*bG2* + (4/5)*bGamma3) * Term4 + (1/4)*b2**2 * Term5 + bG2**2 * Term6 + b2*bG2 * Term7 + counterterm + Pstoch

    return np.array(result).astype(np.float64)#, Term1, P1loop, plin
  
  def model_P_tracer(self, k_sample, b1, b2, bG2, bGamma3, R2, Pshot, cs2, redshift, plin=None):
    return self._model_P_tracer(k_sample, b1, b2, bG2, bGamma3, R2, Pshot, cs2, redshift, plin=plin)

class EFTformalism_Qin2022(EFTformalism_Anastasiou2024):
  def __init__(self, param, CHOP_TOL=1e-30, k_min=0.02, k_max=0.4,
               nBW=12,  # Number of Breit-Wigner terms
               verbose=True,
               save_folder='IntegralMatrices',
               ):
    super().__init__(param, CHOP_TOL=CHOP_TOL, k_min=k_min, k_max=k_max,
               nBW=nBW,  # Number of Breit-Wigner terms
               verbose=verbose,
               save_folder=save_folder,
               )

  def model_P_tracer(self, k_sample, b1, b2, bG2, R2, redshift, plin=None):
    bGamma3 = 0.0
    Pshot = 0.0
    cs2 = 0.0
    return self._model_P_tracer(k_sample, b1, b2, bG2, bGamma3, R2, Pshot, cs2, redshift, plin=plin)


# ---------------------------------------------------------------------------
# Grid-based EFT classes (Issue #6)
# ---------------------------------------------------------------------------

class EFTBase(ABC):
    """
    Abstract base class for grid-based EFT (Effective Field Theory) models.

    Provides shared infrastructure: cosmological parameter storage, linear
    power spectrum access, growth factor, k-grid setup, MAS window function,
    and EFT counterterm helpers.

    Subclasses must implement
    -------------------------
    compute_bias_fields(delta_lin)
    model_P_tracer(k, bias_params, z)
    """

    def __init__(self, param, grid_size=None, box_size=None, MAS='CIC', verbose=True):
        """
        Parameters
        ----------
        param : object
            Cosmological parameter object (as used throughout toolscosmo).
        grid_size : int, optional
            Number of grid cells per dimension.
        box_size : float, optional
            Physical box size in Mpc/h.
        MAS : str, optional
            Mass assignment scheme for deconvolution: 'NGP', 'CIC', 'TSC', 'PCS'.
        verbose : bool
            Print progress messages.
        """
        self.param = param
        self.grid_size = grid_size
        self.box_size = box_size
        self.MAS = MAS
        self.verbose = verbose
        self._plin_cache = None

    def _get_plin(self):
        """Return cached linear power spectrum dict with keys 'k' and 'P'."""
        if self._plin_cache is None:
            self._plin_cache = get_Plin(self.param)
        return self._plin_cache

    def _get_D(self, z):
        """Growth factor D(z), normalised so that D(z→0) = 1. Handles scalar z."""
        return float(growth_factor(np.atleast_1d(float(z)), self.param)[0])

    def _get_f(self, z):
        """Logarithmic growth rate f(z) = d ln D / d ln a. Handles scalar z."""
        return float(growth_rate(np.atleast_1d(float(z)), self.param)[0])

    def _setup_k_grid(self):
        """
        Build rFFT k-grid arrays for (grid_size, box_size).

        Returns
        -------
        KX, KY, KZ : np.ndarray, shape (N, N, N//2+1)
        K2         : np.ndarray, shape (N, N, N//2+1); K2[0,0,0] = 1.
        """
        N, L = self.grid_size, self.box_size
        kF = 2 * np.pi / L
        kx = np.fft.fftfreq(N) * N * kF
        ky = np.fft.fftfreq(N) * N * kF
        kz = np.fft.rfftfreq(N) * N * kF
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0
        return KX, KY, KZ, K2

    def _eft_counterterm(self, delta_k, R2, K2):
        """
        EFT counterterm −R² ∇²δ in Fourier space.

        Returns +R² K² delta_k (the sign flip from ∇²→−k² is already
        absorbed, so this is the *additive* contribution to delta_k).
        """
        return R2 * K2 * delta_k

    def _mas_window(self, K, MAS=None):
        """
        Scalar MAS window function W(k) = sinc(k Δx/2)^p.

        Parameters
        ----------
        K : np.ndarray
            Wavenumber magnitudes (h/Mpc).
        MAS : str, optional
            Override self.MAS.

        Returns
        -------
        W : np.ndarray, same shape as K.
        """
        scheme = (MAS or self.MAS).upper()
        orders = {'NGP': 1, 'CIC': 2, 'TSC': 3, 'PCS': 4}
        p = orders.get(scheme, 2)
        dx = self.box_size / self.grid_size
        x = K * dx / 2.0
        return np.where(x == 0, 1.0, (np.sin(x) / x) ** p)

    def _T_bar_21(self, z):
        """
        Mean 21-cm brightness temperature T̄_b(z) in mK.

        Uses the approximation (Furlanetto et al. 2006):
            T̄_b(z) ≈ 189 h (1+z)² [H₀/H(z)] Ω_HI  mK
        with fiducial Ω_HI = 6.25 × 10⁻⁴ (Crighton et al. 2015).
        """
        h   = self.param.cosmo.h0
        H0  = 100.0 * h          # km/s/Mpc
        Hz  = hubble(z, self.param)
        Omega_HI = 6.25e-4
        return 189.0 * h * (1 + z)**2 * (H0 / Hz) * Omega_HI

    @abstractmethod
    def compute_bias_fields(self, delta_lin):
        """
        Evaluate all bias operator fields on the input 3-D density mesh.

        Parameters
        ----------
        delta_lin : np.ndarray, shape (N, N, N)
            Linear density contrast field.

        Returns
        -------
        dict
            Dictionary of bias operator arrays on the same grid.
        """

    @abstractmethod
    def model_P_tracer(self, k, bias_params, z):
        """
        Compute the (1-loop) power spectrum of the biased tracer.

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc.
        bias_params : dict
            Bias parameters (keys depend on the subclass).
        z : float
            Redshift.

        Returns
        -------
        P : np.ndarray
            Power spectrum in (Mpc/h)³.
        """


class EFTmodel(EFTBase):
    """
    EFT model for the 21-cm brightness temperature power spectrum.

    Implements the Lagrangian bias expansion formalism described in:
        Qin et al. 2022, Phys. Rev. D 106, 123506 (arXiv:2205.06270).

    Bias basis — 5 Lagrangian operators
    ------------------------------------
    Operator             Parameter   Description
    1                    —           mean background
    δₗ                   b₁          linear density
    δₗ² − ⟨δₗ²⟩         b₂          quadratic density
    sₗ² − ⟨sₗ²⟩         bₛ          tidal-field squared
    ∇²δₗ                 b∇          derivative / bubble-scale bias

    EFT counterterm:  −bₑ k² δ  (renormalises UV sensitivity).

    Parameters
    ----------
    param : object
        Cosmological parameters.
    grid_size : int, optional
        Grid resolution (required for compute_bias_fields).
    box_size : float, optional
        Box size in Mpc/h (required for compute_bias_fields).
    MAS : str, optional
        Mass assignment scheme. Default: 'CIC'.
    n_q : int, optional
        Number of log-spaced q points for loop integrals. Default: 256.
    n_mu : int, optional
        Number of μ points for loop integrals. Default: 128.
    verbose : bool, optional
        Print progress messages. Default: True.
    """

    def __init__(self, param, grid_size=None, box_size=None, MAS='CIC',
                 n_q=256, n_mu=128, q_UV=2.0, verbose=True):
        super().__init__(param, grid_size=grid_size, box_size=box_size,
                         MAS=MAS, verbose=verbose)
        self.n_q = n_q
        self.n_mu = n_mu
        self.q_UV = q_UV   # UV cutoff for P13 (h/Mpc); P22 uses full k range
        self._loop_cache = {}

    # ------------------------------------------------------------------
    # Grid-based bias operators
    # ------------------------------------------------------------------

    def compute_bias_fields(self, delta_lin):
        """
        Compute the 4 independent Lagrangian bias operator fields on a 3-D mesh.

        Parameters
        ----------
        delta_lin : np.ndarray, shape (N, N, N)
            Zero-mean linear density contrast.

        Returns
        -------
        dict with keys:
            'delta'       : δₗ
            'delta2'      : δₗ² − ⟨δₗ²⟩
            's2'          : sₗ² − ⟨sₗ²⟩  (tidal field squared)
            'nabla2delta' : ∇²δₗ
        """
        if self.box_size is None:
            raise ValueError("box_size must be set before calling compute_bias_fields.")
        N = delta_lin.shape[0]
        L = self.box_size
        kF = 2 * np.pi / L

        kx = np.fft.fftfreq(N) * N * kF
        ky = np.fft.fftfreq(N) * N * kF
        kz = np.fft.rfftfreq(N) * N * kF
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0

        delta_k = np.fft.rfftn(delta_lin)

        # O2: δₗ² − ⟨δₗ²⟩
        delta2 = delta_lin**2 - np.mean(delta_lin**2)

        # O3: sₗ² − ⟨sₗ²⟩
        # s_ij(k) = (k_i k_j / k² − δ_ij/3) δ(k)
        axes = [KX, KY, KZ]
        s2 = np.zeros(delta_lin.shape, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                sij_k = (axes[i] * axes[j] / K2 - float(i == j) / 3.0) * delta_k
                s2 += np.fft.irfftn(sij_k, s=delta_lin.shape) ** 2
        s2 -= np.mean(s2)

        # O4: ∇²δₗ  →  −k² δ(k) in Fourier space
        nabla2delta = np.fft.irfftn(-K2 * delta_k, s=delta_lin.shape)

        return {
            'delta':       delta_lin.copy(),
            'delta2':      delta2,
            's2':          s2,
            'nabla2delta': nabla2delta,
        }

    # ------------------------------------------------------------------
    # Loop integrals
    # ------------------------------------------------------------------

    def _compute_loop_integrals(self, k_arr, z):
        """
        Numerically compute 1-loop power spectrum integrals at redshift z.

        Integrals computed for each k in k_arr:

            P22(k)    = 2 ∫ d³q/(2π)³  F₂²(q,k−q) P(q) P(|k−q|)
            P13(k)    — Makino et al. 1992 angle-averaged kernel (q ≤ q_UV)
            I_F2(k)   = ∫ d³q/(2π)³  F₂(q,k−q)    P(q) P(|k−q|)
            I_G2(k)   = ∫ d³q/(2π)³  G₂(q,k−q)    P(q) P(|k−q|)
            I_1(k)    = ∫ d³q/(2π)³               P(q) P(|k−q|)
            I_G2sq(k) = ∫ d³q/(2π)³  G₂²(q,k−q)  P(q) P(|k−q|)

        Numerical method for P22, I_*
        --------------------------------
        Substituting x = |k−q| (so μ = (k²+q²−x²)/(2kq)) converts the
        double integral over (q, μ) into a 1-D integral over q with an inner
        1-D integral over x ∈ [|k−q|, k+q].  Parameterising the inner range
        via t ∈ [0, 1]:  x = |k−q| + 2 min(k,q) t eliminates boundary
        truncation artefacts.

        SPT kernels in terms of magnitudes k, q, x = |k−q|
        -------------------------------------------------------
        Δ ≡ k₁·k₂ = (k² − q² − x²) / 2
        F₂ = 5/7 + Δ/2 (1/q² + 1/x²) + 2/7 Δ²/(q²x²)
        G₂ = Δ²/(q²x²) − 1/3

        Results are cached by (k_arr, z).
        """
        cache_key = (tuple(np.round(k_arr, 8)), round(float(z), 6))
        if cache_key in self._loop_cache:
            return self._loop_cache[cache_key]

        plin_data = self._get_plin()
        Dz = self._get_D(z)
        Plin = interp1d(plin_data['k'], plin_data['P'] * Dz**2,
                        kind='cubic', bounds_error=False, fill_value=0.0)

        q_min    = float(plin_data['k'].min())
        q_max    = float(plin_data['k'].max())
        # P13 uses a UV cutoff to keep the result finite; the residual
        # UV sensitivity is absorbed into the EFT counterterm (be parameter).
        q_max_13 = min(q_max, float(self.q_UV)) if self.q_UV is not None else q_max

        # Outer q grid (log-spaced) shared by all integrals
        q_arr   = np.logspace(np.log10(q_min), np.log10(q_max),    self.n_q)
        q_arr13 = np.logspace(np.log10(q_min), np.log10(q_max_13), self.n_q)
        # Inner t grid in [0, 1] for the x = |k−q| + 2 min(k,q) t substitution
        t_arr   = np.linspace(0.0, 1.0, self.n_mu)

        Pq    = Plin(q_arr)
        Pq13  = Plin(q_arr13)

        keys = ['P22', 'P13', 'I_F2', 'I_G2', 'I_1', 'I_G2sq']
        results = {key: np.zeros(len(k_arr)) for key in keys}

        for ik, k in enumerate(k_arr):
            if self.verbose and len(k_arr) > 5 and ik % max(1, len(k_arr) // 5) == 0:
                print(f"  Loop integrals: {ik + 1}/{len(k_arr)}", flush=True)

            # ----------------------------------------------------------
            # P22 / bias-cross integrals via x = |k−q| substitution
            # ----------------------------------------------------------
            # Inner x range: x ∈ [|k−q|, k+q].  Parameterise:
            #   x(q, t) = |k−q| + 2 min(k,q) t,  t ∈ [0, 1]
            #   dx = 2 min(k,q) dt
            # Full 3-D integral:
            #   ∫ d³q/(2π)³ h P(q) P(x)
            #   = ∫ d(lnq) q²P(q)/(4π²k) × ∫_{x_lo}^{x_hi} x P(x) h dx
            #   ≈ ∫ d(lnq) q²P(q)/(4π²k) × [2 min(k,q)] × trapz_t(x P(x) h, t)

            q_2d   = q_arr[:, None]           # (n_q, 1)
            t_2d   = t_arr[None, :]           # (1, n_mu)
            x_lo   = np.abs(k - q_arr)        # (n_q,)
            min_kq = np.minimum(k, q_arr)     # (n_q,)

            x_2d   = x_lo[:, None] + 2.0 * min_kq[:, None] * t_2d  # (n_q, n_mu)
            x_2d   = np.maximum(x_2d, 1e-10)
            Px_2d  = Plin(x_2d)               # (n_q, n_mu)

            # SPT kernels via Δ = (k² − q² − x²)/2
            Delta   = (k**2 - q_2d**2 - x_2d**2) * 0.5
            q2_2d   = np.maximum(q_2d**2, 1e-60)
            x2_2d   = np.maximum(x_2d**2, 1e-60)

            inv_q2x2 = 1.0 / (q2_2d * x2_2d)
            F2 = (5.0/7.0
                  + 0.5 * Delta * (1.0/q2_2d + 1.0/x2_2d)
                  + 2.0/7.0 * Delta**2 * inv_q2x2)
            G2 = Delta**2 * inv_q2x2 - 1.0/3.0

            # Inner integral over t: ∫_0^1 x P(x) h dt × 2 min(k,q)
            wx = x_2d * Px_2d                                         # (n_q, n_mu)
            dt_factor = (2.0 * min_kq)                                # (n_q,)

            I_F2sq_  = np.trapz(wx * F2**2, t_arr, axis=1) * dt_factor
            I_F2_    = np.trapz(wx * F2,    t_arr, axis=1) * dt_factor
            I_G2_    = np.trapz(wx * G2,    t_arr, axis=1) * dt_factor
            I_1_     = np.trapz(wx,          t_arr, axis=1) * dt_factor
            I_G2sq_  = np.trapz(wx * G2**2, t_arr, axis=1) * dt_factor

            # Outer q integral: (4π²k)⁻¹ ∫ d(lnq) q² P(q) × I_inner
            norm_out = 1.0 / (4.0 * np.pi**2 * k)
            w_out    = q_arr**2 * Pq                                   # (n_q,)
            lnq      = np.log(q_arr)

            results['P22'][ik]    = 2.0 * norm_out * np.trapz(w_out * I_F2sq_,  lnq)
            results['I_F2'][ik]   =       norm_out * np.trapz(w_out * I_F2_,    lnq)
            results['I_G2'][ik]   =       norm_out * np.trapz(w_out * I_G2_,    lnq)
            results['I_1'][ik]    =       norm_out * np.trapz(w_out * I_1_,     lnq)
            results['I_G2sq'][ik] =       norm_out * np.trapz(w_out * I_G2sq_,  lnq)

            # ----------------------------------------------------------
            # P13 via Makino et al. 1992 angle-averaged kernel (q ≤ q_UV)
            # ----------------------------------------------------------
            # P13(k) = (3/π²) P_lin(k) ∫₀^{q_UV} d(lnq) q³ P_lin(q) g₁₃(q/k)
            # g₁₃(r) = [12/r² − 158 + 100r² − 42r⁴
            #            + (3/r³)(1−r²)³(7r²+2) ln|(1+r)/(1−r)|] / 504
            r13   = q_arr13 / k
            r2_13 = r13**2
            with np.errstate(divide='ignore', invalid='ignore'):
                log_term13 = np.where(
                    np.abs(r13 - 1) > 1e-3,
                    np.log(np.abs((1.0 + r13)
                                  / np.maximum(np.abs(1.0 - r13), 1e-200))),
                    0.0,
                )
            g13 = (12.0/r2_13 - 158.0 + 100.0*r2_13 - 42.0*r2_13**2
                   + 3.0*(r2_13 - 1.0)**3 * (7.0*r2_13 + 2.0)
                   / (r13**3 + 1e-200) * log_term13) / 504.0

            Plin_k = float(Plin(k))
            results['P13'][ik] = (3.0 / np.pi**2) * Plin_k * np.trapz(
                q_arr13**3 * Pq13 * g13, np.log(q_arr13)
            )

        self._loop_cache[cache_key] = results
        return results

    # ------------------------------------------------------------------
    # Public power spectrum methods
    # ------------------------------------------------------------------

    def P_1loop(self, k, z):
        """
        Standalone 1-loop matter power spectrum P_mm(k, z).

        P_mm(k) = P_lin(k, z) + P22(k, z) + P13(k, z)

        Parameters
        ----------
        k : array-like
        z : float

        Returns
        -------
        dict with keys 'k', 'P_lin', 'P22', 'P13', 'P_1loop'.
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        plin_data = self._get_plin()
        Dz = self._get_D(z)
        Plin_k = interp1d(plin_data['k'], plin_data['P'] * Dz**2,
                          kind='cubic', bounds_error=False, fill_value=0.0)(k)
        loops = self._compute_loop_integrals(k, z)
        return {
            'k':       k,
            'P_lin':   Plin_k,
            'P22':     loops['P22'],
            'P13':     loops['P13'],
            'P_1loop': Plin_k + loops['P22'] + loops['P13'],
        }

    def model_P_tracer(self, k, bias_params, z):
        """
        1-loop power spectrum of the Lagrangian-biased 21-cm tracer.

        P(k) = b₁² P_mm(k)
             + 2 b₁ b₂ I_F2(k)
             + 2 (b₁ bₛ + b₂ bₛ) I_G2(k)
             + b₂² I_1(k)
             + bₛ² I_G2sq(k)
             − 2 (b₁ b∇ + bₑ) k² P_lin(k)

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc.
        bias_params : dict
            Keys (all optional except 'b1'):
                'b1'    — linear bias (default 1)
                'b2'    — quadratic bias (default 0)
                'bs'    — tidal bias (default 0)
                'bgrad' — derivative bias b∇ (default 0)
                'be'    — EFT counterterm amplitude (default 0)
        z : float
            Redshift.

        Returns
        -------
        P : np.ndarray  — tracer power spectrum in (Mpc/h)³.
        """
        k     = np.atleast_1d(np.asarray(k, dtype=float))
        b1    = float(bias_params.get('b1',    1.0))
        b2    = float(bias_params.get('b2',    0.0))
        bs    = float(bias_params.get('bs',    0.0))
        bgrad = float(bias_params.get('bgrad', 0.0))
        be    = float(bias_params.get('be',    0.0))

        plin_data = self._get_plin()
        Dz = self._get_D(z)
        Plin_k = interp1d(plin_data['k'], plin_data['P'] * Dz**2,
                          kind='cubic', bounds_error=False, fill_value=0.0)(k)

        loops  = self._compute_loop_integrals(k, z)
        P_mm   = Plin_k + loops['P22'] + loops['P13']

        P = (b1**2         * P_mm
             + 2*b1*b2     * loops['I_F2']
             + 2*(b1*bs + b2*bs) * loops['I_G2']
             + b2**2        * loops['I_1']
             + bs**2        * loops['I_G2sq']
             - 2*(b1*bgrad + be) * k**2 * Plin_k)
        return P

    def model_P_tracer_rsd(self, k, mu, bias_params, z, fog_sigma=0.0):
        """
        Redshift-space tracer power spectrum P(k, μ) and Legendre multipoles.

        Uses a Kaiser + Gaussian FoG model:
            P_s(k,μ) = [(b₁ + f μ²)² P_mm(k) + bias loop terms]
                       × exp(−k² μ² σᵥ²)

        Parameters
        ----------
        k : array-like, shape (n_k,)
            Wavenumbers in h/Mpc.
        mu : array-like, shape (n_mu,)
            Cosine of angle to the line-of-sight.
        bias_params : dict
            Same keys as model_P_tracer, plus optional 'fog_sigma'.
        z : float
            Redshift.
        fog_sigma : float, optional
            FoG velocity dispersion in Mpc/h (overrides bias_params key).

        Returns
        -------
        dict with keys:
            'k'      : 1-D array of wavenumbers
            'mu'     : 1-D array of mu values
            'Pk_mu'  : 2-D array (n_k, n_mu) — P(k, μ)
            'P0'     : 1-D array (n_k) — monopole
            'P2'     : 1-D array (n_k) — quadrupole
            'P4'     : 1-D array (n_k) — hexadecapole
        """
        k  = np.atleast_1d(np.asarray(k,  dtype=float))
        mu = np.atleast_1d(np.asarray(mu, dtype=float))

        b1    = float(bias_params.get('b1',    1.0))
        b2    = float(bias_params.get('b2',    0.0))
        bs    = float(bias_params.get('bs',    0.0))
        bgrad = float(bias_params.get('bgrad', 0.0))
        be    = float(bias_params.get('be',    0.0))
        sigma_v = fog_sigma or float(bias_params.get('fog_sigma', 0.0))

        f = self._get_f(z)  # logarithmic growth rate d ln D / d ln a

        plin_data = self._get_plin()
        Dz = self._get_D(z)
        Plin_k = interp1d(plin_data['k'], plin_data['P'] * Dz**2,
                          kind='cubic', bounds_error=False, fill_value=0.0)(k)

        loops = self._compute_loop_integrals(k, z)
        P_mm  = Plin_k + loops['P22'] + loops['P13']

        # Broadcast to (n_k, n_mu)
        k_2d  = k[:, None]
        mu_2d = mu[None, :]

        # Kaiser factor (b₁ + f μ²)²
        kaiser = (b1 + f * mu_2d**2)**2

        # FoG damping
        fog = np.exp(-(k_2d * mu_2d * sigma_v)**2) if sigma_v > 0.0 else 1.0

        # Isotropic bias correction terms
        P_iso = (2*b1*b2     * loops['I_F2']
                 + 2*(b1*bs + b2*bs) * loops['I_G2']
                 + b2**2      * loops['I_1']
                 + bs**2      * loops['I_G2sq']
                 - 2*(b1*bgrad + be) * k**2 * Plin_k)  # (n_k,)

        Pk_mu = kaiser * P_mm[:, None] * fog + P_iso[:, None]  # (n_k, n_mu)

        # Legendre multipoles: Pₗ(k) = (2ℓ+1)/2 ∫_{-1}^{1} P(k,μ) Lₗ(μ) dμ
        L2 = (3*mu_2d**2 - 1) / 2.0
        L4 = (35*mu_2d**4 - 30*mu_2d**2 + 3) / 8.0

        P0 = 0.5 * np.trapz(Pk_mu,          mu, axis=1)
        P2 = 2.5 * np.trapz(Pk_mu * L2,     mu, axis=1)
        P4 = 4.5 * np.trapz(Pk_mu * L4,     mu, axis=1)

        return {'k': k, 'mu': mu, 'Pk_mu': Pk_mu, 'P0': P0, 'P2': P2, 'P4': P4}

    def fit_bias_params(self, k, Pk_measured, z,
                        p0=None, params_to_fit=('b1', 'b2', 'bs'),
                        method='Nelder-Mead', **minimize_kwargs):
        """
        Fit bias parameters to a measured power spectrum by minimising
        chi² in log-space.

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc.
        Pk_measured : array-like
            Measured power spectrum in (Mpc/h)³.
        z : float
            Redshift.
        p0 : dict, optional
            Initial guess. Keys must be a superset of params_to_fit.
            Defaults: b1=1, others=0.
        params_to_fit : tuple of str, optional
            Subset of {'b1','b2','bs','bgrad','be'} to optimise.
            Default: ('b1','b2','bs').
        method : str, optional
            scipy.optimize.minimize method. Default: 'Nelder-Mead'.
        **minimize_kwargs
            Forwarded to scipy.optimize.minimize (e.g. options={...}).

        Returns
        -------
        fitted_params : dict
            Full bias parameter dict (all 5 keys) with fitted values.
        result : OptimizeResult
            scipy minimisation result object.
        """
        k  = np.atleast_1d(np.asarray(k,           dtype=float))
        Pk = np.atleast_1d(np.asarray(Pk_measured,  dtype=float))

        _defaults = {'b1': 1.0, 'b2': 0.0, 'bs': 0.0, 'bgrad': 0.0, 'be': 0.0}
        p0_dict   = {p: (_defaults[p] if p0 is None else p0.get(p, _defaults[p]))
                     for p in params_to_fit}
        p0_arr    = np.array([p0_dict[p] for p in params_to_fit])

        mask   = Pk > 0
        k_fit  = k[mask]
        lnPk   = np.log(Pk[mask])

        def _residual(x):
            bp       = {**_defaults, **dict(zip(params_to_fit, x))}
            Pk_model = self.model_P_tracer(k_fit, bp, z)
            Pk_model = np.maximum(Pk_model, 1e-10)
            return float(np.sum((np.log(Pk_model) - lnPk)**2))

        result       = minimize(_residual, p0_arr, method=method, **minimize_kwargs)
        fitted       = dict(zip(params_to_fit, result.x))
        fitted_full  = {**_defaults, **fitted}
        return fitted_full, result

    def construct_tracer_field(self, bias_fields, bias_params, R2=0.0):
        """
        Construct the on-grid biased tracer field from pre-computed bias operators.

        delta_tracer = b1*delta + b2*(delta²-<delta²>) + bs*(s²-<s²>)
                       + bgrad*nabla²delta  [+ EFT counterterm if R2 > 0]

        Parameters
        ----------
        bias_fields : dict
            Output of compute_bias_fields: keys 'delta','delta2','s2','nabla2delta'.
        bias_params : dict
            Bias parameter dict (same keys as model_P_tracer).
        R2 : float, optional
            EFT counterterm scale R² (Mpc/h)². Adds −R² ∇²δ to the field.
            Default: 0 (no counterterm).

        Returns
        -------
        delta_tracer : np.ndarray, shape (N, N, N)
            On-grid dimensionless tracer density contrast.
        """
        b1    = float(bias_params.get('b1',    1.0))
        b2    = float(bias_params.get('b2',    0.0))
        bs    = float(bias_params.get('bs',    0.0))
        bgrad = float(bias_params.get('bgrad', 0.0))

        delta_tracer = (b1    * bias_fields['delta']
                      + b2    * bias_fields['delta2']
                      + bs    * bias_fields['s2']
                      + bgrad * bias_fields['nabla2delta'])

        if R2 > 0.0 and self.box_size is not None:
            N, L = delta_tracer.shape[0], self.box_size
            kF = 2 * np.pi / L
            kx = np.fft.fftfreq(N) * N * kF
            ky = np.fft.fftfreq(N) * N * kF
            kz = np.fft.rfftfreq(N) * N * kF
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            K2 = KX**2 + KY**2 + KZ**2
            K2[0, 0, 0] = 1.0
            delta_k = np.fft.rfftn(bias_fields['delta'])
            ct = np.fft.irfftn(self._eft_counterterm(delta_k, R2, K2),
                               s=delta_tracer.shape)
            delta_tracer = delta_tracer + ct

        return delta_tracer


if __name__ == "__main__":
  import toolscosmo

  par = toolscosmo.par()
  plin = toolscosmo.get_Plin(par)
  par.file.ps = plin
  eft = toolscosmo.EFTformalism_Anastasiou2024(par)
  eft.compute_tracer_intergrals()
  ps_fit = eft.fit_P_linear()

  fig, ax = plt.subplots(1,1,figsize=(7,5))
  k, P_L = plin['k'], plin['P']
  ax.loglog(k, P_L, ls='-', label='$P_\mathrm{lin}$')
  P_Lfit = ps_fit['Plin_fit']
  ax.loglog(k, P_Lfit, ls='--', label='fit')
  ax.set_xlabel(f'$k [h/\mathrm{{Mpc}}]$')
  ax.set_ylabel(f'$P(k)$')
  plt.tight_layout()
  plt.show()