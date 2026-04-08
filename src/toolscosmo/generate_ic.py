import numpy as np
from numpy.fft import fftn, ifftn, rfftn, irfftn, fftshift, ifftshift
from time import time

from .cosmo import growth_factor, get_Plin, hubble

def k_Nyquist(grid_size, box_size):
    kmax = 2*np.pi/box_size*grid_size
    knyq = kmax/2
    return knyq

def growth_rate(z, param, delta_a=1e-5):
    """
    Growth rate f(a)=(dlnD(a))/(dlna)

    z: array of redshifts from zmin to zmax
    """
    z2a = lambda z: 1/(z+1)
    a2z = lambda a: 1/a-1
    a = z2a(z)
    D1_a = growth_factor(a2z(a), param)
    D1_aplus = growth_factor(a2z(a+delta_a), param)
    f = (np.log(D1_aplus) - np.log(D1_a)) / (np.log(a + delta_a) - np.log(a))
    return f

def growth_factor_a(a, param):
    '''
    It calls the growth_factor(z,param) that needs redshift z.
    '''
    return growth_factor(1/a-1, param)

def compute_derivative(func, a, h=1e-5):
    """Compute the numerical derivative of a function using central difference."""
    return (func(a + h) - func(a - h)) / (2 * h)

def D1_growth_factor(a, param):
    """
    Linear growth factor dependent on scale factor a
    """
    return growth_factor_a(a, param)

def D2_growth_factor(a, param):
    """
    Second-order growth factor using D2(a) = -(3/7) * D1^2(a).
    """
    D1_val = D1_growth_factor(a, param)
    D2_val = - (3/7) * D1_val**2
    return D2_val

def D3a_growth_factor(a, param):
    """
    Third-order (3a) growth factor for the tidal-determinant term.
    EdS approximation: D3a ≈ (1/3) D1^3.
    """
    return (1/3) * D1_growth_factor(a, param)**3

def D3b_growth_factor(a, param):
    """
    Third-order (3b) growth factor for the mixed φ^(1)–φ^(2) term.
    EdS approximation: D3b ≈ -(10/21) D1 D2 ≈ (10/49) D1^3.
    """
    D1 = D1_growth_factor(a, param)
    D2 = D2_growth_factor(a, param)
    return -(10/21) * D1 * D2

def create_k_grid_fft(grid_size, box_size):
    k_grid = 2 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    kx, ky, kz = np.meshgrid(k_grid, k_grid, k_grid, indexing='ij')
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    return [kx, ky, kz], kmag

def create_k_grid_rfft(grid_size, box_size):
    k_grid = 2 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    k_gridz = 2 * np.pi * np.fft.rfftfreq(grid_size, d=box_size / grid_size)
    kx, ky, kz = np.meshgrid(k_grid, k_grid, k_gridz, indexing='ij')
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    return [kx, ky, kz], kmag

def generate_gaussian_random_field(grid_size, box_size, power_spectrum=None, param=None, random_seed=42,
                                   fixed_amplitude=False, verbose=True, return_white_noise=False):
    """
    Generate a Gaussian random field delta(x) with a given power spectrum P(k).

    This function creates a Gaussian random field in real space by first generating a random field
    in Fourier space, and then transforming it back to real space using an inverse Fourier transform.
    The field is constructed such that its power spectrum matches the provided input or is inferred
    from the cosmological parameters.

    Parameters
    ----------
    grid_size : int
        The number of grid points along each axis (grid resolution).
    box_size : float
        The physical size of the simulation box (in units of Mpc/h).
    power_spectrum : dict or function, optional
        The power spectrum P(k) to use for generating the field. If None, the function will infer it
        from the cosmological parameters `param` by calling `get_Plin`.
    param : dict, optional
        A dictionary of cosmological parameters, such as matter density parameter, Hubble constant, etc.
        Used only if `power_spectrum` is None.
    random_seed : int, optional
        Seed for the random number generator. Default is 42.
        A negative seed generates the **paired** field of ``abs(random_seed)``: same phases but
        with the overall sign of the density contrast flipped (Angulo & Pontzen 2016).
    fixed_amplitude : bool, optional
        If True, fix the amplitude of each Fourier mode to exactly sqrt(P(k)) (only the phase is
        random). This eliminates sample-variance scatter in the amplitude, so the measured P(k) of
        a single realisation matches the input P(k) exactly. Default is False.
    verbose : bool, optional
        If True, prints progress updates. Default is True.

    Returns
    -------
    dict with keys:
        ``delta_lin`` — Gaussian random field in real space, shape (grid_size, grid_size, grid_size).
        ``kx``, ``ky``, ``kz`` — wavevectors in Fourier space.

    Notes
    -----
    **Fixed-and-paired simulations** (Angulo & Pontzen 2016):
    Use ``fixed_amplitude=True`` to suppress sample-variance noise in P(k) measurements.
    Use a negative seed to obtain the partner simulation with the same phases but opposite sign,
    so that ``(P_fixed + P_paired) / 2`` converges to the ensemble mean faster.

    This function can be used to generate initial conditions for cosmological simulations.
    """
    if verbose:
        print('Generating Gaussian random field...')
        tstart = time()

    assert not (power_spectrum is None and param is None)
    if power_spectrum is None:
        power_spectrum = get_Plin(param)

    np.random.seed(abs(random_seed))
    [kx, ky, kz], k = create_k_grid_fft(grid_size, box_size)

    # Interpolating the power spectrum
    sqrtPgrid = np.sqrt(np.interp(k, power_spectrum['k'], power_spectrum['P']) / (box_size / grid_size)**3)
    sqrtPgrid[0, 0, 0] = 0.

    # Generate Gaussian random field in Fourier space
    delta_k_white = (np.random.normal(size=(grid_size, grid_size, grid_size)) +
               1j * np.random.normal(size=(grid_size, grid_size, grid_size)))

    if fixed_amplitude:
        # Fix amplitude to sqrt(2) (= E[|z|] for standard complex Gaussian) keeping only the phase.
        # Each mode then has exactly the right power; only phase variance remains.
        amp = np.abs(delta_k_white)
        amp[amp == 0] = 1.
        delta_k_white = delta_k_white / amp * np.sqrt(2)

    delta_k = delta_k_white * sqrtPgrid

    # Transform back to real space; negate for paired simulation (negative seed)
    sign = -1 if random_seed < 0 else 1
    delta_r = sign * ifftn(delta_k, norm='ortho').real

    if verbose:
        print(f'...done in {(time()-tstart):.3f} seconds')
    out = {'delta_lin': delta_r, 'kx': kx, 'ky': ky, 'kz': kz}
    if return_white_noise:
        delta_r_white = ifftn(delta_k_white, norm='ortho').real
        out['delta_white'] = delta_r_white
    return out

def create_gradient_kernel_rfft3(res, boxsize, oversampling_factor=1):
    """
    Create a modified gradient kernel in Fourier space for rfftn, setting Nyquist frequencies to zero.
    Now accounts for oversampling.
    """
    kF = 2 * np.pi / boxsize
    effective_res = res * oversampling_factor # Resolution for kernel creation

    kxy = np.array(np.fft.fftfreq(effective_res, 1/effective_res)) * kF
    kz = np.array(np.fft.rfftfreq(effective_res, 1/effective_res)) * kF
    kmesh = np.array(np.meshgrid(kxy, kxy, kz, indexing='ij'))

    grad_kernel = 1.j * kmesh
    grad_kernel[0, effective_res//2] = 0.  # Nyquist in x-direction
    grad_kernel[1, :, effective_res//2] = 0.  # Nyquist in y-direction
    grad_kernel[2, :, :, -1] = 0.  # Nyquist in z-direction

    # grad_kernel[np.abs(grad_kernel)==0] = np.abs(grad_kernel)[np.abs(grad_kernel)>0].min() # Prevent division by zero
    k_squared = kmesh[0]**2+kmesh[1]**2+kmesh[2]**2

    return grad_kernel, k_squared

def create_gradient_kernel_fft3(res, boxsize):
    """
    Create a modified gradient kernel in Fourier space for fftn, setting Nyquist frequencies to zero.
    """
    kF = 2 * np.pi / boxsize
    kxy = np.array(np.fft.fftfreq(res, 1/res)) * kF
    kz = np.array(np.fft.fftfreq(res, 1/res)) * kF  # Use fftfreq for z as well now
    kmesh = np.array(np.meshgrid(kxy, kxy, kz, indexing='ij'))

    grad_kernel = 1.j * kmesh

    # Nyquist frequencies for fftn are at index res//2 for all dimensions
    grad_kernel[0, res//2, :, :] = 0.  # Nyquist in x-direction
    grad_kernel[1, :, res//2, :] = 0.  # Nyquist in y-direction
    grad_kernel[2, :, :, res//2] = 0.  # Nyquist in z-direction

    # grad_kernel[np.abs(grad_kernel)==0] = np.abs(grad_kernel)[np.abs(grad_kernel)>0].min() # Prevent division by zero
    k_squared = kmesh[0]**2+kmesh[1]**2+kmesh[2]**2

    return grad_kernel, k_squared


def first_order_lpt(delta_k, D1, grad_kernel, k_squared, oversampling_factor):
    """
    Calculate the displacement field using the Zel'dovich approximation (1LPT).

    1LPT equation: Psi_i = - ∂phi/∂xi, where ∇^2phi = -δ.
    """
    grid_size_eff = delta_k.shape[0]
    grid_size = grid_size_eff/oversampling_factor

    phi_k = -delta_k / k_squared

    # Compute displacements Psi_i = -∂phi/∂ki
    disp_k = -grad_kernel * phi_k
    disp_r = np.array([irfftn(disp_k[i], norm='ortho').real for i in range(3)])[:,:int(grid_size),:int(grid_size),:int(grid_size)]
    disp_x1 = D1 * disp_r[0]
    disp_y1 = D1 * disp_r[1]
    disp_z1 = D1 * disp_r[2]

    return disp_x1, disp_y1, disp_z1

def second_order_lpt(delta_k, D2, grad_kernel, k_squared, oversampling_factor):
    """
    Calculate the second-order displacement field using 2LPT.

    2LPT equation: Psi_ij = ∂ij_phi_2LPT, where phi_2LPT = -S_ij S_kl (delta_k / k^2).
    """
    # Effective grid sizes
    grid_size_eff = delta_k.shape[0]  # High-res grid size
    grid_size = int(grid_size_eff // oversampling_factor)  # Low-res grid size

    # Solve Poisson equation in Fourier space for first-order potential
    phi_k = -delta_k / k_squared

    # Compute 1LPT displacements in Fourier space: Psi_ij
    Psi_1LPT_k = -grad_kernel * phi_k

    # Compute second derivatives of the potential (S_ij) in real space
    phi_dxdx = np.zeros((3, 3, grid_size_eff, grid_size_eff, grid_size_eff))
    for i in range(3):
        for j in range(3):
            phi_dxdx[i, j] = irfftn(grad_kernel[i] * Psi_1LPT_k[j], norm='ortho').real

    # Compute the 2LPT source term
    phi_2LPT = (phi_dxdx[0, 0] * phi_dxdx[1, 1] - phi_dxdx[0, 1]**2 +
                phi_dxdx[0, 0] * phi_dxdx[2, 2] - phi_dxdx[0, 2]**2 +
                phi_dxdx[2, 2] * phi_dxdx[1, 1] - phi_dxdx[2, 1]**2)

    # Compute Fourier transform of 2LPT potential
    phi_2LPT_k = rfftn(phi_2LPT, norm='ortho')

    # Compute 2LPT displacements in Fourier space: Psi_ij = -∇phi / k^2
    Psi_2LPT_k = -grad_kernel * phi_2LPT_k / k_squared
    Psi_2LPT_r = np.array([irfftn(Psi_2LPT_k[i], norm='ortho').real for i in range(3)])[:,:int(grid_size),:int(grid_size),:int(grid_size)]

    # Extract the central grid_size x grid_size x grid_size portion
    disp_x2 = D2 * Psi_2LPT_r[0]
    disp_y2 = D2 * Psi_2LPT_r[1]
    disp_z2 = D2 * Psi_2LPT_r[2]

    return disp_x2, disp_y2, disp_z2


def third_order_lpt(delta_k, D3a, D3b, grad_kernel, k_squared, oversampling_factor):
    """
    Calculate the third-order displacement field using 3LPT.

    Two contributions are summed:
    - 3a: Ψ^(3a) = D3a ∇φ^(3a),  sourced by det(∂_i∂_j φ^(1))  (tidal determinant).
    - 3b: Ψ^(3b) = D3b ∇φ^(3b),  sourced by G_2(φ^(1), φ^(2))  (mixed 1st/2nd order).

    Growth factors used (EdS approximations):
        D3a ≈  (1/3) D1^3
        D3b ≈ -(10/21) D1 D2  ≈ (10/49) D1^3

    References: Bouchet et al. 1995; Scoccimarro 2002; Michaux et al. 2021.

    Note on sign convention: phi_dxdx[i,j] = FT^{-1}[k_i k_j φ_k] = −∂_i∂_j φ,
    so det(phi_dxdx) = −det(Hessian φ).  The 3a source det(Hessian φ^(1)) is
    therefore −det(phi_dxdx_1), hence the leading minus sign below.
    For the 3b source, the minus signs cancel in every product, so no extra sign is needed.
    """
    grid_size_eff = delta_k.shape[0]
    grid_size = int(grid_size_eff // oversampling_factor)

    # 1LPT potential and displacement in Fourier space
    phi_k = -delta_k / k_squared
    Psi_1LPT_k = -grad_kernel * phi_k   # (3, N, N, N//2+1)

    # Hessian of φ^(1): phi_dxdx_1[i,j] = -∂_i∂_j φ^(1)
    phi_dxdx_1 = np.zeros((3, 3, grid_size_eff, grid_size_eff, grid_size_eff))
    for i in range(3):
        for j in range(3):
            phi_dxdx_1[i, j] = irfftn(grad_kernel[i] * Psi_1LPT_k[j], norm='ortho').real

    # 2LPT source (reuse same formula as second_order_lpt)
    phi_2LPT_src = (phi_dxdx_1[0,0]*phi_dxdx_1[1,1] - phi_dxdx_1[0,1]**2 +
                    phi_dxdx_1[0,0]*phi_dxdx_1[2,2] - phi_dxdx_1[0,2]**2 +
                    phi_dxdx_1[2,2]*phi_dxdx_1[1,1] - phi_dxdx_1[2,1]**2)
    phi_2LPT_k   = rfftn(phi_2LPT_src, norm='ortho')
    Psi_2LPT_k   = -grad_kernel * phi_2LPT_k / k_squared

    # Hessian of φ^(2): phi_dxdx_2[i,j] = -∂_i∂_j φ^(2)
    phi_dxdx_2 = np.zeros((3, 3, grid_size_eff, grid_size_eff, grid_size_eff))
    for i in range(3):
        for j in range(3):
            phi_dxdx_2[i, j] = irfftn(grad_kernel[i] * Psi_2LPT_k[j], norm='ortho').real

    # 3a source: det(Hessian φ^(1)) = -det(phi_dxdx_1)
    d = phi_dxdx_1
    phi_3a = -(
        d[0,0] * (d[1,1]*d[2,2] - d[1,2]**2)
      - d[0,1] * (d[0,1]*d[2,2] - d[1,2]*d[0,2])
      + d[0,2] * (d[0,1]*d[1,2] - d[1,1]*d[0,2])
    )

    # 3b source: mixed G2(φ^(1), φ^(2))  — signs cancel in each product
    e = phi_dxdx_2
    phi_3b = (
        d[0,0]*e[1,1] + e[0,0]*d[1,1] - 2*d[0,1]*e[0,1]
      + d[0,0]*e[2,2] + e[0,0]*d[2,2] - 2*d[0,2]*e[0,2]
      + d[1,1]*e[2,2] + e[1,1]*d[2,2] - 2*d[1,2]*e[1,2]
    )

    # Solve Poisson equations and compute displacements for both terms
    phi_3a_k  = rfftn(phi_3a, norm='ortho')
    Psi_3a_k  = -grad_kernel * phi_3a_k / k_squared
    Psi_3a_r  = np.array([irfftn(Psi_3a_k[i], norm='ortho').real
                          for i in range(3)])[:, :grid_size, :grid_size, :grid_size]

    phi_3b_k  = rfftn(phi_3b, norm='ortho')
    Psi_3b_k  = -grad_kernel * phi_3b_k / k_squared
    Psi_3b_r  = np.array([irfftn(Psi_3b_k[i], norm='ortho').real
                          for i in range(3)])[:, :grid_size, :grid_size, :grid_size]

    disp_x3 = D3a * Psi_3a_r[0] + D3b * Psi_3b_r[0]
    disp_y3 = D3a * Psi_3a_r[1] + D3b * Psi_3b_r[1]
    disp_z3 = D3a * Psi_3a_r[2] + D3b * Psi_3b_r[2]

    return disp_x3, disp_y3, disp_z3


# def first_order_lpt(delta_k, D1, kx, ky, kz):
#     """
#     Calculate the displacement field using the Zel'dovich approximation (1LPT).
    
#     1LPT equation: Psi_i = - ∂phi/∂xi, where ∇^2phi = -δ.
#     """
#     k_squared = kx**2 + ky**2 + kz**2
#     k_squared[0, 0, 0] = 1  # Prevent division by zero

#     # Solve Poisson equation in Fourier space: phi_k = -delta_k / k^2
#     phi_k = - delta_k / k_squared
#     phi_k[0, 0, 0] = 0  # Ensure no monopole contribution

#     # Compute displacements Psi_i = -∂phi/∂ki
#     disp_x1 = irfftn(-1j * kx * phi_k).real * D1
#     disp_y1 = irfftn(-1j * ky * phi_k).real * D1
#     disp_z1 = irfftn(-1j * kz * phi_k).real * D1

#     return disp_x1, disp_y1, disp_z1

# def second_order_lpt(delta_k, D2, kx, ky, kz):
#     """Calculate the second-order displacement field using 2LPT."""

#     k_squared = kx**2 + ky**2 + kz**2
#     k_squared[0, 0, 0] = 1

#     kmesh = np.array([kx, ky, kz])
#     # kgrid = np.sqrt(kmesh[0]**2 + kmesh[1]**2 + kmesh[2]**2)
#     # kgrid[0,0,0] = 1.

#     grid = delta_k.shape[0]
 
#     inv_k2 = 1/k_squared

#     phi_k = -delta_k / k_squared
#     phi_k[0, 0, 0] = 0

#     # 1LPT Displacement Field in Fourier space
#     Psi_1LPT_k = -1j * phi_k * kmesh

#     # Calculate the second-order potential
#     phi_dxdx = 1.j * np.einsum("ijkl,mjkl->imjkl", kmesh, Psi_1LPT_k)

#     # Pad to prevent aliasing
#     phi_dxdx = np.fft.fftshift(phi_dxdx, axes=(-3,-2))
#     phi_dxdx = np.pad(phi_dxdx, ((0,0), (0,0), (grid//4, grid//4), (grid//4, grid//4), (0, grid//4)),constant_values=0.+0.j) * (3/2)**3.
#     phi_dxdx = np.fft.ifftshift(phi_dxdx, axes=(-3,-2))
#     phi_dxdx = np.fft.irfftn(phi_dxdx, axes=(-3,-2,-1))
    
#     # Compute 2LPT potential
#     phi_2LPT = phi_dxdx[0,0]*phi_dxdx[1,1] - phi_dxdx[0,1]**2.
#     phi_2LPT+= phi_dxdx[0,0]*phi_dxdx[2,2] - phi_dxdx[0,2]**2.
#     phi_2LPT+= phi_dxdx[2,2]*phi_dxdx[1,1] - phi_dxdx[2,1]**2.
#     phi_2LPT = np.fft.rfftn(phi_2LPT, axes=(-3,-2,-1))
    
#         # Downsample after antialiased products
#     phi_2LPT = np.fft.fftshift(phi_2LPT, axes=(-3,-2))
#     phi_2LPT = phi_2LPT[grid//4:-grid//4, grid//4:-grid//4,:-grid//4] / (3/2)**3.
#     phi_2LPT = np.fft.ifftshift(phi_2LPT, axes=(-3,-2))    

#     Psi_2LPT_r = np.fft.irfftn(- 1.j * phi_2LPT * kmesh * inv_k2,axes=(-3,-2,-1))
#     disp_x2_r, disp_y2_r, disp_z2_r = D2 * Psi_2LPT_r

#     return disp_x2_r, disp_y2_r, disp_z2_r

def generate_initial_condition_positions(grid_size, box_size, z, param, LPT=2, power_spectrum=None, verbose=True, 
                                anti_aliasing='oversampling', anti_aliasing_kwargs={'oversampling_factor': 2}, 
                                model_velocity=False, delta_lin=None, random_seed=42):
    """
    Generate cosmological initial conditions using Lagrangian Pertubation Theory (LPT).

    This function generates the initial particle positions for a cosmological simulation by applying
    Lagrangian Perturbation Theory (LPT), either first-order (1LPT, Zel'dovich) or second-order (2LPT).
    It computes the displacement fields from the linear density field (Gaussian random field) using the 
    provided cosmological parameters and a given redshift.

    Parameters
    ----------
    grid_size : int
        The number of grid points along each axis (grid resolution).
    box_size : float
        The physical size of the simulation box (in units of Mpc/h).
    z : float
        The redshift at which to generate the initial conditions.
    param : dict
        A dictionary of cosmological parameters (e.g., matter density, Hubble constant, etc.).
    LPT : int, optional
        The order of Lagrangian Perturbation Theory to use (1 for 1LPT, 2 for 2LPT). Default is 2.
    power_spectrum : function, optional
        A function that computes the power spectrum given the wavevector (in units of Mpc/h).
    verbose : bool, optional
        If True, prints progress updates. Default is True.
    anti_aliasing : str, optional
        Method for anti-aliasing. Options are:
            - 'sharpk' or 'sharp-k': Sharp Fourier filter (default).
            - 'gaussian' or 'gaussian-filter' or 'gaussianfilter' or 'gaussian_filter': Gaussian filter.
            - 'tapered': Cosine tapered filter around the Nyquist frequency.
            - 'padding', 'zeropadding', or 'oversampling': Oversampling with zero-padding in Fourier space.
            - None or False: No anti-aliasing filter.
    anti_aliasing_kwargs : dict, optional
        Keyword arguments to control the behavior of the chosen anti-aliasing filter.
        - For 'gaussian':
            - 'gaussian_sigma_ratio' (float, optional): Standard deviation of the Gaussian filter
              in units of the Nyquist wavenumber. Default is 1.0.
        - For 'tapered':
            - 'taper_width_ratio' (float, optional): Width of the cosine taper region around the
              Nyquist wavenumber, as a fraction of the Nyquist wavenumber. Default is 0.2.
        - For 'padding', 'zeropadding', or 'oversampling':
            - 'oversampling_factor' (int, optional): The factor by which to over sample the field.
               Default is 1.0.
    model_velocity : bool, optional
        If True, compute and return velocities using linear theory. Default is False.
    delta_lin : np.ndarray
        A 3D array of gridded initial linear delta field. Default is None, which will ask the code to create 
        a Gaussian Random Field from the provided power_spectrum.
    random_seed : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    np.ndarray
        A 2D array of particle positions with shape (N_particles, 3), where N_particles is the 
        total number of particles in the grid.

    Notes
    -----
    This function generates a Gaussian random field (GRF) as the initial density field. Then it 
    computes the displacement fields for both 1LPT and 2LPT orders (if applicable). The resulting 
    particle positions are returned, adjusted by the LPT displacements and periodic boundary conditions.
    """
    a = 1/(1+z)
    D1  = D1_growth_factor(a, param)
    D2  = D2_growth_factor(a, param)
    D3a = D3a_growth_factor(a, param)
    D3b = D3b_growth_factor(a, param)

    if delta_lin is None:
        deltagrf  = generate_gaussian_random_field(grid_size, box_size, power_spectrum=power_spectrum, param=param, random_seed=random_seed, verbose=verbose)
        delta_lin = deltagrf['delta_lin']
    else:
        # [kx, ky, kz], _ = create_k_grid_fft(grid_size, box_size)
        deltagrf = {}

    # Create modified delta_k and gradient kernel, considering oversampling
    oversampling_factor = anti_aliasing_kwargs.get('oversampling_factor') if anti_aliasing.lower() in ['padding', 'zero-padding', 'zeropadding', 'zero_padding', 'oversampling'] else 1
    grid_size_eff = grid_size * oversampling_factor

    grad_kernel, k_squared = create_gradient_kernel_rfft3(grid_size_eff, box_size)
    # [kx, ky, kz], kmag = create_k_grid_rfft(grid_size, box_size)

    delta_k = rfftn(delta_lin, s=(grid_size_eff, grid_size_eff, grid_size_eff), norm='ortho')

    if verbose: 
        print(f'Displacing particles using {LPT}LPT...')
        tstart = time()

    # k_squared = np.sum(np.abs(grad_kernel)**2, axis=0) #kmag**2 # # k^2 = kx^2 + ky^2 + kz^2
    k_squared[k_squared==0] = k_squared[k_squared>0].min()  # Prevent division by zero
    kmax = k_Nyquist(grid_size_eff, box_size)  # Nyquist frequency

    # Apply aliasing filter by supressing high-frequency modes
    W_k = np.ones_like(delta_k)
    if anti_aliasing.lower() in ['sharpk', 'sharp-k']:
        W_k[k_squared > kmax**2] = 0
    elif anti_aliasing.lower() in ['gaussian', 'gaussian-filter', 'gaussianfilter', 'gaussian_filter']:
        gaussian_sigma_ratio = anti_aliasing_kwargs.get('gaussian_sigma_ratio', 1.0) if anti_aliasing_kwargs else 1.0
        W_k = np.exp(-0.5*k_squared/(gaussian_sigma_ratio*kmax)**2)
    elif anti_aliasing.lower() in ['tapered', 'cosine-tapered', 'cosinetapered', 'cosine_tapered']:
        taper_width_ratio = anti_aliasing_kwargs.get('taper_width_ratio', 0.2) if anti_aliasing_kwargs else 0.2
        k_taper_start = kmax * (1 - taper_width_ratio)
        k_taper_end = kmax
        taper_mask = (k_squared >= k_taper_start**2) & (k_squared <= k_taper_end**2)
        W_k[taper_mask] = W_k[taper_mask] * (0.5 * (1 + np.cos(np.pi * (np.sqrt(k_squared[taper_mask]) - k_taper_start) / (k_taper_end - k_taper_start))))
        W_k[k_squared > k_taper_end**2] = 0.
    elif anti_aliasing.lower() in ['padding', 'zero-padding', 'zeropadding', 'zero_padding', 'oversampling']:
        pass # No filter in Fourier space for zero-padding, oversampling handles it
    elif anti_aliasing is None or anti_aliasing is False:
        pass # No filter applied
    else:
        print(f'Warning: Anti-aliasing method "{anti_aliasing}" not recognized. No filter applied.')

    delta_k *= W_k  # Suppress high-frequency modes

    # 1LPT (Zel'dovich Approximation)
    disp_x1, disp_y1, disp_z1 = first_order_lpt(delta_k, D1, grad_kernel, k_squared, oversampling_factor)
    # 2LPT Displacement
    disp_x2, disp_y2, disp_z2 = (0, 0, 0)
    if LPT > 1:
        disp_x2, disp_y2, disp_z2 = second_order_lpt(delta_k, D2, grad_kernel, k_squared, oversampling_factor)
    # 3LPT Displacement (two contributions: 3a det-term + 3b mixed term)
    disp_x3, disp_y3, disp_z3 = (0, 0, 0)
    if LPT > 2:
        disp_x3, disp_y3, disp_z3 = third_order_lpt(delta_k, D3a, D3b, grad_kernel, k_squared, oversampling_factor)
    if verbose:
        print(f'...done in {time()-tstart:.3f} seconds')

    # Initial particle positions
    x = np.linspace(0, box_size, grid_size, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    particles_x = X + disp_x1 + disp_x2 + disp_x3
    particles_y = Y + disp_y1 + disp_y2 + disp_y3
    particles_z = Z + disp_z1 + disp_z2 + disp_z3

    # Apply periodic boundary conditions
    particles_x %= box_size
    particles_y %= box_size
    particles_z %= box_size

    # Flatten arrays for output
    positions = np.stack([particles_x.flatten(), particles_y.flatten(), particles_z.flatten()], axis=-1)
    out = {'delta_lin': delta_lin, 'positions': positions}
    if 'delta_white' in list(deltagrf.keys()):
        out['delta_white'] = deltagrf['delta_white']

    if model_velocity:
        if verbose: print('Computing velocities...')
        vx, vy, vz = compute_velocity(delta_lin, kx, ky, kz, a, param)
        if verbose: print('done')
        velocities = np.stack([vx.flatten(), vy.flatten(), vz.flatten()], axis=-1)
        out['velocities'] = velocities

    return out

def generate_initial_condition_grid(grid_size, box_size, z, param, LPT=2, power_spectrum=None, verbose=True, 
                                anti_aliasing='sharpk', filter_kwargs=None, model_velocity=False, delta_white_noise=None, random_seed=42, MAS='PCS'):
    """
    Generate cosmological initial conditions using Lagrangian Pertubation Theory (LPT).

    This function generates the initial particle positions for a cosmological simulation by applying
    Lagrangian Perturbation Theory (LPT), either first-order (1LPT, Zel'dovich) or second-order (2LPT).
    It computes the displacement fields from the linear density field (Gaussian random field) using the 
    provided cosmological parameters and a given redshift. The final particle distribution is put on a grid.

    Parameters
    ----------
    grid_size : int
        The number of grid points along each axis (grid resolution).
    box_size : float
        The physical size of the simulation box (in units of Mpc/h).
    z : float
        The redshift at which to generate the initial conditions.
    param : dict
        A dictionary of cosmological parameters (e.g., matter density, Hubble constant, etc.).
    LPT : int, optional
        The order of Lagrangian Perturbation Theory to use (1 for 1LPT, 2 for 2LPT). Default is 2.
    power_spectrum : function, optional
        A function that computes the power spectrum given the wavevector (in units of Mpc/h).
    verbose : bool, optional
        If True, prints progress updates. Default is True.
    anti_aliasing : str, optional
        Method for anti-aliasing. Options are:
            - 'sharpk' or 'sharp-k': Sharp Fourier filter (default).
            - 'gaussian' or 'gaussian-filter' or 'gaussianfilter' or 'gaussian_filter': Gaussian filter.
            - 'tapered': Cosine tapered filter around the Nyquist frequency.
            - None or False: No anti-aliasing filter.
    filter_kwargs : dict, optional
        Keyword arguments to control the behavior of the chosen anti-aliasing filter.
        - For 'gaussian':
            - 'gaussian_sigma_ratio' (float, optional): Standard deviation of the Gaussian filter
              in units of the Nyquist wavenumber. Default is 1.0.
        - For 'tapered':
            - 'taper_width_ratio' (float, optional): Width of the cosine taper region around the
              Nyquist wavenumber, as a fraction of the Nyquist wavenumber. Default is 0.2.
    model_velocity : bool, optional
        If True, compute and return velocities using linear theory. Default is False.
    delta_white_noise : np.ndarray
        A 3D array of gridded initial random delta field. Default is None, which will ask the code to create 
        a Gaussian Random Field from the provided power_spectrum.
    random_seed : int, optional
        Seed for the random number generator. Default is 42.
    MAS : str, optional
        The mass assignment scheme to use. Default is 'PCS' (Piecewise Constant Scheme).

    Returns
    -------
    np.ndarray
        A 2D array of particle positions with shape (N_particles, 3), where N_particles is the 
        total number of particles in the grid.

    Notes
    -----
    This function generates a Gaussian random field (GRF) as the initial density field. Then it 
    computes the displacement fields for both 1LPT and 2LPT orders (if applicable). The resulting 
    particle positions are returned, adjusted by the LPT displacements and periodic boundary conditions.
    """
    out = generate_initial_condition_positions(grid_size, box_size, z, param, LPT=LPT, power_spectrum=power_spectrum, verbose=verbose, 
                                anti_aliasing=anti_aliasing, filter_kwargs=filter_kwargs, model_velocity=model_velocity, delta_white_noise=delta_white_noise, random_seed=random_seed)
    grid = particles_on_grid(out['positions'], grid_size, box_size, MAS=MAS, verbose=verbose)
    out['delta_LPT'] = grid
    return out

def compute_velocity(delta, kx, ky, kz, a, param):
    """
    Compute the peculiar velocity field in real space.

    Parameters
    ----------
    delta : ndarray
        The density field in real space.
    kx, ky, kz : ndarray
        Wavevectors corresponding to the grid in Fourier space.
    a : float
        Scale factor corresponding to the desired redshift.
    param : dict
        Cosmological parameters dictionary, used to compute H(a) and f(a).

    Returns
    -------
    vx, vy, vz : ndarray
        Velocity components in real space.
    """
    H_of_a = lambda a, param: hubble(1/a - 1,param)

    # Growth rate f(a)
    f_a = growth_rate(1/a - 1, param)
    
    # Hubble parameter at scale factor a
    H_a = H_of_a(a, param)  # Define this function if not already present

    # Fourier transform the density field
    delta_k = fftn(delta, norm='ortho')

    # Compute velocity potential in Fourier space
    k_squared = kx**2 + ky**2 + kz**2
    k_squared[0, 0, 0] = 1  # Prevent division by zero
    phi_k = -delta_k / k_squared
    phi_k[0, 0, 0] = 0  # Avoid zero-mode issues

    # Velocity field components in Fourier space
    vx_k = 1j * kx * phi_k * a * H_a * f_a
    vy_k = 1j * ky * phi_k * a * H_a * f_a
    vz_k = 1j * kz * phi_k * a * H_a * f_a

    # Transform back to real space
    vx = ifftn(vx_k, norm='ortho').real
    vy = ifftn(vy_k, norm='ortho').real
    vz = ifftn(vz_k, norm='ortho').real

    return vx, vy, vz

def particles_on_grid(positions, grid_size, box_size, MAS='PCS', backend='auto', verbose=True):
    """
    Assign particles to a grid and compute the density field using a Mass Assignment Scheme (MAS).

    This function places particles from the input `positions` array onto a grid of size `grid_size` 
    and computes the corresponding density field using a Mass Assignment Scheme (MAS), such as the 
    Piecewise Constant Scheme (PCS). The resulting density field is returned after normalizing the 
    density contrast.

    Parameters
    ----------
    positions : np.ndarray
        A 2D array of particle positions with shape (N_particles, 3) for 3D or (N_particles, 2) for 2D.
    grid_size : int
        The number of grid points along each axis (grid resolution).
    box_size : float
        The physical size of the simulation box (in units of Mpc/h).
    MAS : str, optional
        The mass assignment scheme to use. Default is 'PCS' (Piecewise Constant Scheme).
    backend : str
        'auto'   — try numba (beorn) → numpy (built-in). Use 'pylians' explicitly if needed.
        'numpy'  — always use the built-in pure-NumPy implementation (default fallback)
        'numba'  — use beorn's Numba JIT backend (requires: pip install toolscosmo[numba])
        'pylians'— use pylians MAS_library    (requires: pip install toolscosmo[pylians])
    verbose : bool, optional
        If True, prints progress updates. Default is True.

    Returns
    -------
    np.ndarray
        A 3D or 2D density field depending on the input positions, with shape (grid_size, grid_size, grid_size) 
        for 3D or (grid_size, grid_size) for 2D. The density contrast is normalized to zero mean.
    
    Notes
    -----
    The Mass Assignment Scheme (MAS) used here assigns each particle's mass to its nearest grid points 
    based on the chosen scheme ('PCS' or others). The density contrast is computed as the overdensity 
    (delta = density - mean density).
    """
    from .mass_assignment import assign_mass

    if positions.shape[1] == 2:
        raise NotImplementedError("2D mass assignment is not yet supported by the built-in backend.")
    delta = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    if verbose:
        print(f'\nUsing {MAS} (backend={backend}) mass assignment scheme')
        tstart = time()

    assign_mass(delta, box_size, positions, scheme=MAS, backend=backend, verbose=verbose)

    if verbose:
        print(f'Time taken = {time()-tstart:.3f} seconds\n')

    # Compute overdensity and density contrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    return delta