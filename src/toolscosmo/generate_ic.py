import numpy as np
from numpy.fft import fftn, ifftn, rfftn, irfftn, fftshift, ifftshift

from .cosmo import growth_factor, get_Plin, hubble

def growth_rate(z, param, delta_a=1e-5):
    """
    Growth rate f(a)=(dlnD(a))/(dlna)

    z: array of redshifts from zmin to zmax
    """
    z2a = lambda a: 1/(1+z)
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
    # D1_prime = compute_derivative(lambda x: growth_factor_a(x,param), a)
    # Calculate second-order growth factor approximation
    D2_val = - (3/7) * D1_val**2
    return D2_val

def create_k_grid(grid_size, box_size):
    k_grid = 2 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    kx, ky, kz = np.meshgrid(k_grid, k_grid, k_grid, indexing='ij')
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    return [kx, ky, kz], kmag

def generate_gaussian_random_field(grid_size, box_size, power_spectrum=None, param=None, **kwargs):
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
    
    Returns
    -------
    delta_x : np.ndarray
        The generated Gaussian random field in real space, with shape (grid_size, grid_size, grid_size).
    kx, ky, kz : np.ndarray
        The wavevectors corresponding to the grid in Fourier space along each axis.
    
    Notes
    -----
    The generated field has the specified power spectrum in Fourier space, and its variance in real 
    space is set by the power spectrum normalization. The function uses the inverse Fourier transform 
    to convert the field back to real space after applying the correct amplitude in Fourier space.
    
    This function can be used to generate initial conditions for cosmological simulations.
    """
    assert not (power_spectrum is None and param is None)
    if power_spectrum is None:
        power_spectrum = get_Plin(param)

    random_seed = kwargs.get('random_seed', 42)
    np.random.seed(random_seed)
    [kx, ky, kz], k = create_k_grid(grid_size, box_size)

    # Interpolating the power spectrum
    sqrtPgrid = np.sqrt(np.interp(k, power_spectrum['k'], power_spectrum['P']) / (box_size / grid_size)**3)
    sqrtPgrid[0, 0, 0] = 0.

    # Generate Gaussian random field in Fourier space
    delta_k = (np.random.normal(size=(grid_size, grid_size, grid_size)) +
               1j * np.random.normal(size=(grid_size, grid_size, grid_size))) * sqrtPgrid

    # Transform back to real space
    delta_x = ifftn(delta_k, norm='ortho').real
    return delta_x, kx, ky, kz

def zeldovich_approximation(delta, kx, ky, kz, D1, grid_size, filter_aliasing=True):
    """
    Calculate the displacement field using the Zel'dovich approximation (1LPT).
    
    1LPT equation: Psi_i = - ∂phi/∂xi, where ∇^2phi = -δ.
    """
    k_squared = kx**2 + ky**2 + kz**2
    k_squared[0, 0, 0] = 1  # Prevent division by zero

    # Solve Poisson equation in Fourier space: phi_k = -delta_k / k^2
    delta_k = fftn(delta, norm='ortho')
    
    phi_k = - delta_k / k_squared
    phi_k[0, 0, 0] = 0  # Ensure no monopole contribution

    # Compute displacements Psi_i = -∂phi/∂ki
    disp_x1 = ifftn(-1j * kx * phi_k, norm='ortho').real * D1
    disp_y1 = ifftn(-1j * ky * phi_k, norm='ortho').real * D1
    disp_z1 = ifftn(-1j * kz * phi_k, norm='ortho').real * D1

    return disp_x1, disp_y1, disp_z1

def second_order_displacement(delta, kx, ky, kz, D2, grid_size, filter_aliasing=True):
    """
    Calculate the second-order displacement field using 2LPT.
    
    2LPT equation: Psi_ij = ∂ij_phi_2LPT, where phi_2LPT = -S_ij S_kl (delta_k / k^2).
    """
    k_squared = kx**2 + ky**2 + kz**2
    k_squared[0, 0, 0] = 1  # Prevent division by zero

    delta_k = fftn(delta, norm='ortho')
    
    phi_k = -delta_k / k_squared
    phi_k[0, 0, 0] = 0

    # Compute 1LPT Psi_ij
    Psi_1LPT_k = -1j * phi_k * np.array([kx, ky, kz])
    Psi_1LPT_r = np.array([ifftn(Psi_1LPT_k[i], norm='ortho').real for i in range(3)])

    # 2LPT term S_ij S_kl in real space
    phi_dxdx = 1j * np.einsum("ijkl,mjkl->imjkl", np.array([kx, ky, kz]), Psi_1LPT_k)
    phi_2LPT = (phi_dxdx[0, 0] * phi_dxdx[1, 1] - phi_dxdx[0, 1]**2 +
                phi_dxdx[0, 0] * phi_dxdx[2, 2] - phi_dxdx[0, 2]**2 +
                phi_dxdx[2, 2] * phi_dxdx[1, 1] - phi_dxdx[2, 1]**2)

    # Compute Fourier transform of 2LPT potential
    phi_2LPT_k = rfftn(phi_2LPT, norm='ortho')

    # Adjust shapes of kx, ky, kz for real FFT space
    kx_r, ky_r, kz_r = np.meshgrid(
        2 * np.pi * np.fft.fftfreq(delta.shape[0], d=1.0),
        2 * np.pi * np.fft.fftfreq(delta.shape[1], d=1.0),
        2 * np.pi * np.fft.rfftfreq(delta.shape[2], d=1.0),
        indexing='ij'
    )
    k_squared_r = kx_r**2 + ky_r**2 + kz_r**2
    k_squared_r[0, 0, 0] = 1  # Prevent division by zero

    # Compute 2LPT Psi_ij = -∇phi / k^2
    Psi_2LPT_k = -1j * phi_2LPT_k * np.array([kx_r, ky_r, kz_r]) / k_squared_r
    Psi_2LPT_r = np.array([irfftn(Psi_2LPT_k[i], norm='ortho').real for i in range(3)])

    disp_x2, disp_y2, disp_z2 = D2 * Psi_2LPT_r[0], D2 * Psi_2LPT_r[1], D2 * Psi_2LPT_r[2]
    return disp_x2, disp_y2, disp_z2

def generate_initial_conditions(grid_size, box_size, z, param, LPT=2, power_spectrum=None, verbose=True, 
                                filter_aliasing=True, model_velocity=False, **kwargs):
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
    **kwargs : dict, optional
        Additional arguments that can be passed to the `generate_gaussian_random_field` function.

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
    D1 = D1_growth_factor(a, param)
    D2 = D2_growth_factor(a, param)

    if verbose: print('Generating Gaussian random field...')
    delta = kwargs.get('delta_grf', None)
    if delta is None:
        delta, kx, ky, kz = generate_gaussian_random_field(grid_size, box_size, power_spectrum=power_spectrum, param=param, **kwargs)
    else:
        [kx, ky, kz], _ = create_k_grid(grid_size, box_size)
    if verbose: print('done')

    if verbose: print(f'Displacing particles using {LPT}LPT...')
    # 1LPT (Zel'dovich Approximation)
    disp_x1, disp_y1, disp_z1 = zeldovich_approximation(delta, kx, ky, kz, D1, grid_size, filter_aliasing=filter_aliasing)
    # 2LPT Displacement
    disp_x2, disp_y2, disp_z2 = (0, 0, 0)
    if LPT > 1:
        disp_x2, disp_y2, disp_z2 = second_order_displacement(delta, kx, ky, kz, D2, grid_size, filter_aliasing=filter_aliasing)
    if verbose: print('done')

    # Initial particle positions
    x = np.linspace(0, box_size, grid_size, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    particles_x = X + disp_x1 + disp_x2
    particles_y = Y + disp_y1 + disp_y2
    particles_z = Z + disp_z1 + disp_z2

    # Apply periodic boundary conditions
    particles_x %= box_size
    particles_y %= box_size
    particles_z %= box_size

    # Flatten arrays for output
    positions = np.stack([particles_x.flatten(), particles_y.flatten(), particles_z.flatten()], axis=-1)
    out = {'positions': positions}

    if model_velocity:
        if verbose: print('Computing velocities...')
        vx, vy, vz = compute_velocity(delta, kx, ky, kz, a, param)
        if verbose: print('done')
        velocities = np.stack([vx.flatten(), vy.flatten(), vz.flatten()], axis=-1)
        out['velocities'] = velocities

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

def particles_on_grid(positions, grid_size, box_size, MAS='PCS', verbose=True):
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
    import MAS_library as MASL

    if positions.shape[1]==2:
        # define 2D density field
        delta = np.zeros((grid_size,grid_size), dtype=np.float32)
    elif positions.shape[1]==3:
        # define 3D density field
        delta = np.zeros((grid_size,grid_size,grid_size), dtype=np.float32)

    # construct 2D density field
    MASL.MA(positions.astype(np.float32), delta, box_size, MAS, verbose=verbose)

    # Compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

    return delta
