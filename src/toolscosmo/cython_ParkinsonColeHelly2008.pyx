import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
from scipy.special import gamma
import logging
from time import time

from cython.parallel import prange

cimport numpy as np
cimport cython
from libc.math cimport sqrt, log, exp

# Declare floating-point arrays for performance
ctypedef np.double_t DTYPE_t

# Define a custom struct for cosmological parameters
cdef class CosmoParams:
    cdef double Om
    cdef double Ob
    cdef double Or
    cdef double Ok
    cdef double Ode
    cdef double h0

    def __init__(self, double Om, double Ob, double Or, double Ok, double Ode, double h0):
        self.Om = Om
        self.Ob = Ob
        self.Or = Or
        self.Ok = Ok
        self.Ode = Ode
        self.h0 = h0

# Define the growth factor D(z) as a standalone cdef function
@cython.cfunc
cdef double D(double z, double Om0, double Olam0):
    cdef double Otot = Olam0 + Om0 * ((1.0 + z) ** 3)
    cdef double Om = Om0 * ((1.0 + z) ** 3) / Otot
    cdef double Olam = Olam0 / Otot
    cdef double FirstFactor = 5.0 * Om / (2.0 * (1.0 + z))
    cdef double SecondFactor = Om**(4.0 / 7.0) - Olam + (1.0 + 0.5 * Om) * (1.0 + Olam / 70.0)
    return FirstFactor / SecondFactor

# Define the GrowthFactor function using the standalone D function
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double GrowthFactor(double z, CosmoParams cosmo):
    """
    Growth factor for Flat LCDM cosmology.
    """
    cdef double Om0 = cosmo.Om
    cdef double Olam0 = cosmo.Ode

    # Normalize the growth factor to D(z)/D(0)
    cdef double Dz = D(z, Om0, Olam0) / D(0.0, Om0, Olam0)
    return Dz

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple prepare_sigmaM(np.ndarray[DTYPE_t, ndim=1] mbin, np.ndarray[DTYPE_t, ndim=1] sbin):
    """
    This function computes sigma(M) and its derivative, returning splines.
    """
    cdef np.ndarray[DTYPE_t, ndim=1] log_mbin = np.log(mbin)
    cdef np.ndarray[DTYPE_t, ndim=1] log_sbin = np.log(sbin)

    # Compute spline of ln(sigma) vs ln(M)
    lnsigma_tck = splrep(log_mbin, log_sbin)
    
    # Compute derivative of ln(sigma) with respect to ln(M)
    dlnsbin_dlnMbin = np.gradient(log_sbin, log_mbin)
    
    # Compute alpha as -dln(sigma)/dln(M) and make spline
    alpha_tck = splrep(log_mbin, -dlnsbin_dlnMbin)

    return lnsigma_tck, alpha_tck


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sigma(double M, tuple lnsigma_tck):
    """
    This function computes sigma(M) by evaluating the spline `lnsigma_tck`.
    """
    return exp(splev(log(M), lnsigma_tck))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double alpha(double M, tuple alpha_tck):
    """
    This function computes alpha(M) = -dln(sigma)/dln(M) by evaluating the spline `alpha_tck`.
    """
    return splev(log(M), alpha_tck)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double J(double u_res, tuple J_tck=None, double g1=0.38):
    """
    Computes the J(u_res) function.
    """
    cdef double result

    if J_tck is None:
        # Define a C function instead of a lambda for performance
        result = quad(J_integral_function, 0, u_res, args=(g1,))[0]
    else:
        result = splev(u_res, J_tck)
    
    return result


cdef double J_integral_function(double u, double g1):
    """
    This is the function to integrate for the J(u) calculation.
    """
    return pow(1 + 1/u**2, g1 / 2)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple prepare_Jfit(double g1=0.38):
    """
    Precomputes and returns a spline table for J(u_res).
    """
    cdef np.ndarray[DTYPE_t, ndim=1] uu = np.linspace(1e-5, 300.0, 600)
    cdef np.ndarray[DTYPE_t, ndim=1] JJ = np.empty_like(uu)
    
    cdef int i
    cdef double u0

    # Compute J(u) for each value in `uu`
    for i in range(600):
        u0 = uu[i]
        JJ[i] = J(u0, g1=g1)

    # Create spline for J(u_res)
    J_tck = splrep(uu, JJ)

    return J_tck

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double V(double q, double M2, tuple lnsigma_tck):
    cdef double s1 = sigma(q * M2, lnsigma_tck)
    cdef double s2 = sigma(M2, lnsigma_tck)
    return s1**2 / (s1**2 - s2**2)**1.5

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double beta(double q_res, double M2, tuple lnsigma_tck):
    return log(V(q_res, M2, lnsigma_tck) / V(0.5, M2, lnsigma_tck)) / log(2 * q_res)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double B(double q_res, double M2, tuple lnsigma_tck):
    return V(q_res, M2, lnsigma_tck) / q_res**beta(q_res, M2, lnsigma_tck)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mu(double M2, tuple lnsigma_tck, tuple alpha_tck, double g1, double M_res):
    cdef double sigma_res, sigma_h, q_res, result
    if g1 >= 0:
        result = alpha(M2 / 2, alpha_tck)
    else:
        sigma_res = sigma(M_res, lnsigma_tck)
        sigma_h = sigma(M2 / 2, lnsigma_tck)
        q_res = M_res / M2
        result = -log(sigma_res / sigma_h) / log(2 * q_res)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double eta(double q_res, double M2, tuple lnsigma_tck, tuple alpha_tck, double g1):
    cdef double M_res = q_res * M2
    return beta(q_res, M2, lnsigma_tck) - 1 - g1 * mu(M2, lnsigma_tck, alpha_tck, g1, M_res)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double delta(double z, CosmoParams cosmo, double dc = 1.686):
    cdef double d = dc / GrowthFactor(z, cosmo)
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ddelta_dz(double z, CosmoParams cosmo, double dz=1e-5):
    return (delta(z + dz, cosmo) - delta(z - dz, cosmo)) / (2 * dz)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double S(double q, double z, double M2, double q_res, tuple lnsigma_tck, tuple alpha_tck, CosmoParams cosmo, double G0, double g1, double g2):
    cdef double M_res = q_res * M2
    cdef double s2 = sigma(M2, lnsigma_tck)
    cdef double s_h = sigma(M2 / 2, lnsigma_tck)
    cdef double alpha_h = alpha(M2 / 2, alpha_tck)

    cdef double f1 = q**(eta(q_res, M2, lnsigma_tck, alpha_tck, g1) - 1)
    cdef double f2 = (delta(z, cosmo) / s2)**g2
    cdef double f3 = (s_h / s2)**g1

    cdef double Sq = sqrt(2 / np.pi) * B(q_res, M2, lnsigma_tck) * alpha_h * f1 * (G0 / 2**(mu(M2, lnsigma_tck, alpha_tck, g1, M_res) * g1)) * f2 * f3 * ddelta_dz(z, cosmo)
    return Sq

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double R(double q, double M2, double q_res, tuple lnsigma_tck, tuple alpha_tck, double G0, double g1, double g2):
    cdef double M_res = q_res * M2
    cdef double s1 = sigma(q * M2, lnsigma_tck)
    cdef double s2 = sigma(M2, lnsigma_tck)
    cdef double s_h = sigma(M2 / 2, lnsigma_tck)
    cdef double alpha_h = alpha(M2 / 2, alpha_tck)
    cdef double alpha_1 = alpha(q * M2, alpha_tck)

    cdef double f1 = alpha_1 / alpha_h
    cdef double f2 = V(q, M2, lnsigma_tck) / (B(q_res, M2, lnsigma_tck) * q**beta(q_res, M2, lnsigma_tck))
    cdef double f3 = ((2 * q)**mu(M2, lnsigma_tck, alpha_tck, g1, M_res) * s1 / s_h)**g1

    return f1 * f2 * f3

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double N_upper_Dz1(double z, double M2, double q_res, tuple lnsigma_tck, tuple alpha_tck, CosmoParams param, double G0, double g1, double g2):
    cdef double et = eta(q_res, M2, lnsigma_tck, alpha_tck, g1)
    cdef double f1 = S(1, z, M2, q_res, lnsigma_tck, alpha_tck, param, G0, g1, g2) / et
    cdef double f2 = 0.5**et - q_res**et
    return f1*f2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double N_upper(double z, double M2, double q_res, tuple lnsigma_tck, tuple alpha_tck, CosmoParams cosmo, double G0, double g1, double g2):
    return N_upper_Dz1(z, M2, q_res, lnsigma_tck, alpha_tck, cosmo, G0, g1, g2) * Dz(z, M2, q_res, lnsigma_tck, alpha_tck, cosmo, G0, g1, g2)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double Dz(double z, double M2, double q_res, tuple lnsigma_tck, tuple alpha_tck, CosmoParams param, double e1=0.1, double e2=0.1, double G0=0.57, double g1=0.38, double g2=-0.01):
    cdef double M_res = q_res * M2
    cdef double Integral = N_upper_Dz1(z, M2, q_res, lnsigma_tck, alpha_tck, param, G0, g1, g2)
    # print('I = ', Integral)

    cdef double s2 = sigma(M2, lnsigma_tck)
    cdef double s_h = sigma(M2 / 2, lnsigma_tck)
    cdef double dz1 = sqrt(2) * sqrt(s_h**2 - s2**2) / ddelta_dz(z, param)

    return min(e1 * dz1, e2 / Integral)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double F(double z, double M2, double q_res, tuple lnsigma_tck, tuple alpha_tck, tuple J_tck, CosmoParams param, double e1=0.1, double e2=0.1, double G0=0.57, double g1=0.38, double g2=-0.01):
    cdef double M_res = q_res * M2
    cdef double s_res = sigma(M_res, lnsigma_tck)
    cdef double s2 = sigma(M2, lnsigma_tck)
    cdef double u_res = s2 / sqrt(s_res**2 - s2**2)
    cdef double d2 = delta(z, param)
    cdef double dz = Dz(z, M2, q_res, lnsigma_tck, alpha_tck, param, e1, e2, G0, g1, g2)

    return sqrt(2 / np.pi) * J(u_res, J_tck) * (G0 / s2) * ((d2 / s2)**g2) * ddelta_dz(z, param) * dz



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple ParkinsonColeHelly2008_run(double M0, 
                                       double z0, 
                                       double z_max, 
                                       CosmoParams param, 
                                       np.ndarray[DTYPE_t, ndim=1] mbin, 
                                       np.ndarray[DTYPE_t, ndim=1] sbin,
                                       double M_res=1e4, 
                                       double e1=0.1, 
                                       double e2=0.1, 
                                       double G0=0.57, 
                                       double g1=0.38, 
                                       double g2=-0.01,
                                       int max_tree_length=1000,
                                       ):
    """
    Simulates the halo splitting over redshift starting from an initial halo of mass M0 at redshift z0.
    """
    # Prepare the sigma(M) and J(u_res) splines
    cdef tuple lnsigma_tck, alpha_tck
    lnsigma_tck, alpha_tck = prepare_sigmaM(mbin, sbin)
    cdef tuple J_tck = prepare_Jfit(g1=g1)

    # Declare the tree structures as typed memoryviews for faster access
    cdef np.ndarray[DTYPE_t, ndim=1] z_tree = np.zeros(max_tree_length, dtype=np.double)
    cdef np.ndarray[DTYPE_t, ndim=1] M_tree = np.zeros(max_tree_length, dtype=np.double)
    cdef np.ndarray[DTYPE_t, ndim=1] z_subh = np.zeros(max_tree_length, dtype=np.double)
    cdef np.ndarray[DTYPE_t, ndim=1] M_subh = np.zeros(max_tree_length, dtype=np.double)

    # Start initial values
    z_tree[0] = z0
    M_tree[0] = M0

    cdef int tree_index = 1
    cdef int sub_index = 0

    cdef double current_M, current_z
    cdef double dz, q_res, q, z_new, M_new

    # Start the loop to evolve the halo tree
    tstart = time()

    while z_tree[tree_index - 1] < z_max and M_tree[tree_index - 1] > 2 * M_res and tree_index < max_tree_length:
        print(f'z = {z_tree[tree_index - 1]:.6f} | main M = {M_tree[tree_index - 1]:.3e} | counter = {tree_index} | t = {(time()-tstart)/60:.2f} mins')
        current_z = z_tree[tree_index - 1]
        current_M = M_tree[tree_index - 1]
        q_res = M_res / current_M

        # Determine if a split occurs or not
        r1 = np.random.uniform()
        
        if r1 > N_upper(current_z, current_M, q_res, lnsigma_tck, alpha_tck, param, G0, g1, g2):
            # No split occurs, reduce mass M2
            
            M_new = current_M * (1 - F(current_z, current_M, q_res, lnsigma_tck, alpha_tck, J_tck, param, e1, e2, G0, g1, g2))
            dz = Dz(current_z, current_M, q_res, lnsigma_tck, alpha_tck, param, e1, e2, G0, g1, g2)
            z_new = current_z + dz

            # Store the new halo properties
            z_tree[tree_index] = z_new
            M_tree[tree_index] = M_new
            tree_index += 1

        else:
            # Split occurs, determine the mass fractions and update the tree
            r2 = np.random.uniform()

            et = eta(q_res, current_M, lnsigma_tck, alpha_tck, g1)
            q = (q_res**et + (2**(-et)-q_res**et)*r2)**(1/et)

            # Store the two halo masses in subhalo arrays
            r3 = np.random.uniform()
            
            if r3 < R(q, current_M, q_res, lnsigma_tck, alpha_tck, G0, g1, g2):
                # Two progenitors are created with 
                # mass q*M2 and M2*(1 - F - q)

                M_new1 = current_M * q
                M_new2 = current_M * (1 - F(current_z, current_M, q_res, lnsigma_tck, alpha_tck, J_tck, param, e1, e2, G0, g1, g2) - q)

                # Add the larger mass to the tree and the smaller to the subhalo array
                if M_new1 > M_new2:
                    M_tree[tree_index] = M_new1
                    M_subh[sub_index] = M_new2
                else:
                    M_tree[tree_index] = M_new2
                    M_subh[sub_index] = M_new1

                # Update the redshift step
                dz = Dz(current_z, current_M, q_res, lnsigma_tck, alpha_tck, param, e1, e2, G0, g1, g2)
                z_tree[tree_index] = current_z + dz
                z_subh[sub_index] = current_z

                tree_index += 1
                sub_index += 1

            else:
                # No split occurs, but the halo
                # mass M2 is reduced to M2(1 - F).
                M_new = current_M * (1 - F(current_z, current_M, q_res, lnsigma_tck, alpha_tck, J_tck, param, e1, e2, G0, g1, g2))
                dz = Dz(current_z, current_M, q_res, lnsigma_tck, alpha_tck, param, e1, e2, G0, g1, g2)
                z_new = current_z + dz

                # Store the new halo properties
                z_tree[tree_index] = z_new
                M_tree[tree_index] = M_new
                tree_index += 1

    # Resize the arrays to remove any unused space
    z_tree = z_tree[:tree_index]
    M_tree = M_tree[:tree_index]
    z_subh = z_subh[:sub_index]
    M_subh = M_subh[:sub_index]

    return z_tree, M_tree, z_subh, M_subh


