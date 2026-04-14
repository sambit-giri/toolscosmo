import numpy as np
from libc.math cimport sqrt, log
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
from scipy.special import gamma
import logging
from time import time

cimport cython

# Define the class with appropriate cdef declarations
cdef class ParkinsonColeHelly2008:
    cdef object param
    cdef double e1, e2, G0, g1, g2, M_res
    cdef object J_tck, lnsigma_tck, alpha_tck
    cdef double *sigmaM, *alpha

    def __init__(self, param, double e1=0.1, double e2=0.1, double G0=0.57, double g1=0.38, double g2=-0.01, double M_res=1e4):
        self.param = param 
        self.e1 = e1
        self.e2 = e2
        self.G0 = G0 
        self.g1 = g1 
        self.g2 = g2 
        self.M_res = M_res 

        self.prepare_sigmaM()
        self.J_tck = None
        self.prepare_Jfit()

    cpdef double J(self, double u_res):
        if self.J_tck is None:
            cdef double fun(double u):
                return (1 + 1/u**2)**(self.g1 / 2)
            return quad(fun, 0, u_res)[0]
        else:
            return splev(u_res, self.J_tck)
    
    cpdef void prepare_Jfit(self):
        print('Creating the table for J(u_res) function...')
        cdef int i
        cdef double u, *uu, *JJ
        uu = np.linspace(1e-5, 300.0, 600)
        JJ = np.array([self.J(u0) for u0 in uu])
        self.J_tck = splrep(uu, JJ)
        print('...done')
    
    cpdef void prepare_sigmaM(self):
        print('Computing the sigma(M) and required derivatives...')
        cdef double *sig, *mbin
        param = self.param
        param.file.ps = calc_Plin(param)
        rbin, mbin, var = sigma_squared_table(param)
        sig = np.sqrt(var)
        self.lnsigma_tck = splrep(np.log(mbin), np.log(sig))
        dlnsbin_dlnMbin = np.gradient(np.log(sig), np.log(mbin))
        self.alpha_tck = splrep(np.log(mbin), -dlnsbin_dlnMbin)
        print('...done')

    cpdef double sigma(self, double M):
        return np.exp(splev(np.log(M), self.lnsigma_tck))
    
    cpdef double alpha(self, double M):
        return splev(np.log(M), self.alpha_tck)
    
    cpdef double V(self, double q, double M2):
        cdef double s1 = self.sigma(q * M2)
        cdef double s2 = self.sigma(M2)
        return s1**2 / (s1**2 - s2**2)**1.5
    
    cpdef double beta(self, double q_res, double M2):
        return log(self.V(q_res, M2) / self.V(0.5, M2)) / log(2 * q_res)
    
    cpdef double B(self, double q_res, double M2):
        return self.V(q_res, M2) / (q_res**self.beta(q_res, M2))
    
    cpdef double mu(self, double M2):
        cdef double sigma_res, sigma_h, q_res
        if self.g1 >= 0:
            return self.alpha(M2 / 2)
        else:
            sigma_res = self.sigma(self.M_res)
            sigma_h = self.sigma(M2 / 2)
            q_res = self.M_res / M2
            return -log(sigma_res / sigma_h) / log(2 * q_res)
    
    cpdef double eta(self, double q_res, double M2):
        return self.beta(q_res, M2) - 1 - self.g1 * self.mu(M2)
    
    cpdef double delta(self, double z):
        cdef double d = self.param.mf.dc / growth_factor(z, self.param)
        return d
    
    cpdef double ddelta_dz(self, double z, double dz=1e-5):
        return (self.delta(z + dz) - self.delta(z - dz)) / (2 * dz)
    
    cpdef double S(self, double q, double z, double M2, double q_res):
        cdef double s2 = self.sigma(M2)
        cdef double s_h = self.sigma(M2 / 2)
        cdef double alpha_h = self.alpha(M2 / 2)
        cdef double f1 = q**(self.eta(q_res, M2) - 1)
        cdef double f2 = (self.delta(z) / s2)**self.g2
        cdef double f3 = (s_h / s2)**self.g1
        cdef double Sq = sqrt(2 / np.pi) * self.B(q_res, M2) * alpha_h * f1 * (self.G0 / 2**(self.mu(M2) * self.g1)) * f2 * f3 * self.ddelta_dz(z)
        return Sq
    
    cpdef double R(self, double q, double M2, double q_res):
        cdef double s1 = self.sigma(q * M2)
        cdef double s2 = self.sigma(M2)
        cdef double s_h = self.sigma(M2 / 2)
        cdef double alpha_h = self.alpha(M2 / 2)
        cdef double alpha_1 = self.alpha(q * M2)
        cdef double f1 = alpha_1 / alpha_h
        cdef double f2 = self.V(q, M2) / (self.B(q_res, M2) * (q**self.beta(q_res, M2)))
        cdef double f3 = ((2 * q)**self.mu(M2) * s1 / s_h)**self.g1
        return f1 * f2 * f3
    
    cpdef double N_upper_Dz1(self, double z, double M2, double q_res):
        cdef double Nupp = (self.S(1, z, M2, q_res) / self.eta(q_res, M2)) * (0.5**self.eta(q_res, M2) - q_res**self.eta(q_res, M2))
        return Nupp
    
    cpdef double N_upper(self, double z, double M2, double q_res, double Dz=1):
        return self.N_upper_Dz1(z, M2, q_res) * self.Dz(z, M2, q_res)
    
    cpdef double Dz(self, double z, double M2, double q_res):
        cdef double Integral = self.N_upper_Dz1(z, M2, q_res)
        cdef double s2 = self.sigma(M2)
        cdef double s_h = self.sigma(M2 / 2)
        cdef double dz1 = sqrt(2) * sqrt(s_h**2 - s2**2) / self.ddelta_dz(z)
        return min(self.e1 * dz1, self.e2 / Integral)
    
    cpdef double F(self, double z, double M2, double q_res):
        cdef double M_res = q_res * M2
        cdef double s_res = self.sigma(M_res)
        cdef double s2 = self.sigma(M2)
        cdef double u_res = s2 / sqrt(s_res**2 - s2**2)
        cdef double d2 = self.delta(z)
        cdef double dz = self.Dz(z, M2, q_res)
        return sqrt(2 / np.pi) * self.J(u_res) * (self.G0 / s2) * (d2 / s2)**self.g2 * self.ddelta_dz(z) * dz
    
    cpdef dict run(self, double M0, double z0, double z_max, double M_res=None):
        if M_res is None: 
            M_res = self.M_res
        else: 
            self.M_res = M_res

        print(f'Simulation setup:')
        print(f'Starting halo mass = {M0:.2e}')
        print(f'Starting redshift  = {z0:.5f}')
        print(f'Mass resolution    = {M_res:.2e}')
        print(f'Maximum redshift   = {z_max:.5f}')

        self.z_tree = [z0]
        self.M_tree = [M0]
        self.z_subh = []
        self.M_subh = []

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        tstart = time()
        cdef int count = 0
        cdef double r1, r2, r3, q
        cdef double M2, q_res, z_new, M_new, M_new1, M_new2
        cdef double dz

        while self.z_tree[-1] < z_max and self.M_tree[-1] > 2 * M_res:
            logging.info(f'z = {self.z_tree[-1]:.6f} | main M = {self.M_tree[-1]:.3e} | counter = {count} | t = {(time() - tstart) / 60:.2f} mins')
            count += 1
            M2 = self.M_tree[-1]
            q_res = M_res / M2

            r1 = np.random.uniform()
            if r1 > self.N_upper(self.z_tree[-1], M2, q_res):
                M_new = M2 * (1 - self.F(self.z_tree[-1], M2, q_res))
                self.M_tree.append(M_new)
                z_new = self.z_tree[-1] + self.Dz(self.z_tree[-1], M2, q_res)
                self.z_tree.append(z_new)
            else:
                r2 = np.random.uniform()
                et = self.eta(q_res, M2)
                q = (q_res**et + (2**(-et) - q_res**et) * r2)**(1 / et)

                r3 = np.random.uniform()
                if r3 < self.R(q, M2, q_res):
                    M_new1 = M2 * q
                    M_new2 = M2 * (1 - self.F(self.z_tree[-1], M2, q_res) - q)
                    self.M_tree.append(max(M_new1, M_new2))
                    self.M_subh.append(min(M_new1, M_new2))
                    self.z_subh.append(self.z_tree[-1])
                    z_new = self.z_tree[-1] + self.Dz(self.z_tree[-1], M2, q_res)
                    self.z_tree.append(z_new)
                else:
                    M_new = M2 * (1 - self.F(self.z_tree[-1], M2, q_res))
                    self.M_tree.append(M_new)
                    z_new = self.z_tree[-1] + self.Dz(self.z_tree[-1], M2, q_res)
                    self.z_tree.append(z_new)

        return {
            'z_main': np.array(self.z_tree), 
            'M_main': np.array(self.M_tree), 
            'z_subh': np.array(self.z_subh), 
            'M_subh': np.array(self.M_subh),
        }
