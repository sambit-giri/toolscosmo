import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import logging
from .scipy_func import *
from .cosmo import variance, rbin_to_mbin, growth_factor, calc_Plin

# def GrowthFactor_FlatLCDM(z, param):
#     """ 
#     Growth factor. 
#     """
#     Om0 = param.cosmo.Om
#     Olam0 = 1-Om0  #Flat LCDM cosmology 
#     def D(z):
#         Otot = Olam0 + Om0*((1+z)**3)
#         Om   = Om0*((1+z)**3)/Otot
#         Olam = Olam0/Otot
#         FirstFactor  = 5*Om/(2*(1+z))
#         SecondFactor = Om**(4/7) - Olam + (1 + 0.5*Om)*(1 + Olam/70)
#         GrowthFactor = FirstFactor/SecondFactor
#         return GrowthFactor

#     # Normalize the growth factor:
#     Dz = D(z)/D(0.0)
#     return Dz

def sigma_squared_table(param, **kwargs):
    '''
    Function to calculate the density variance (sigma^2(M)) table.
    '''
    mbin = kwargs.get('mbin')
    var  = kwargs.get('var')
    if mbin is None or var is None:
        rbin, var, dlnvardlnr = variance(param)
        mbin = rbin_to_mbin(rbin, param)
        var = var[mbin>param.code.Mdark]
        rbin = rbin[mbin>param.code.Mdark]
        mbin = mbin[mbin>param.code.Mdark]
    return rbin, mbin, var

def sigma_approx(M, param, **kwargs):
    '''
    Approximate fit for sigma(M) developed in Nebrin (2022).
    '''
    Om = param.cosmo.Om
    Or = param.cosmo.Or
    h0 = param.cosmo.h0
    s8 = param.cosmo.s8
    assert s8 is not None, "sigma_8 is required"
    zeq = kwargs.get('zeq', Om/Or - 1)
    Meq = 2.4e17*(Om*h0**2/0.14)**(-0.5)*((1+zeq)/3400)**(-1.5) #Msun
    Mbar = 8*M/Meq 
    sigN = 0.0845*s8 
    sigmaM = sigN*np.sqrt(36/(1+3*Mbar)-np.log(Mbar/(1+Mbar))**3)
    return sigmaM

def dlnsigma_dlnM_approx(M, param, **kwargs):
    '''
    Approximate fit for dln(sigma(M))/dln(M) derived from sigma(M) fit in Nebrin (2022).
    '''
    dM = 1e-6*M
    dsigdM = 1/(2*dM)*(sigma_approx(M+dM, param, **kwargs)-sigma_approx(M-dM, param, **kwargs))
    dlnsigmaM_dlnM = (M/sigma_approx(M, param, **kwargs))*dsigdM
    return dlnsigmaM_dlnM

def sigma_squared(M, param, **kwargs):
    '''
    Function to calculate the density variance (sigma^2(M)).
    '''
    var_tck = kwargs.get('var_tck')
    if var_tck is None: 
        rbin, mbin, var = sigma_squared_table(param, **kwargs)
        var_tck = splrep(mbin, var)
    return splev(M, var_tck)

def dlnsigma_dlnM(M, param, **kwargs):
    '''
    Function to calculate dlnsigma_dlnM, where sigma and M are mass standard deviation and mass, respectively.
    '''
    rbin, mbin, var = sigma_squared_table(param)
    dlnsbin_dlnMbin = np.gradient(np.log(np.sqrt(var)), np.log(mbin))
    tck = splrep(np.log(mbin), dlnsbin_dlnMbin)
    return splev(np.log(M), tck)

def mass_derivative_of_sigma_squared(M, param, **kwargs):
    '''
    Function to calculate the mass derivative of density variance.
    '''
    Mmin = param.code.Mmin
    Mmax = param.code.Mmax 
    NM = param.code.NM
    Mbin = 10**np.linspace(np.log10(Mmin),np.log10(Mmax),NM)
    Sbin = sigma_squared(Mbin, param, **kwargs)
    dSbin_dMbin = Sbin * np.gradient(np.log(Sbin)) / (Mbin * np.gradient(np.log(Mbin)))
    tck = splrep(Mbin, dSbin_dMbin)
    return splev(M, tck)

def delcrit_SC(z, param):
    return 1.686/growth_factor(z,param)

def progenitor_conditional_probability_function(M1, z1, M0, z0, param, collapse_model='SC', **kwargs):
    '''
    Function to compute the conditional probability function for progenitors (Eq. 1 in Jiang & van den Bosch 2014).
    '''
    dX = lambda X1,X2: (X1-X2)
    if collapse_model.upper() in ['SC', 'SPHERICAL COLLAPSE', 'SPHERICAL_COLLAPSE']:
        fcoll = lambda S1,w1,S0,w0: 1/np.sqrt(2*np.pi)*dX(w1,w0)/dX(S1,S0)**(1.5)*np.exp(-dX(w1,w0)**2/(2*dX(S1,S0)))
    elif collapse_model.upper() in ['EC', 'ELLIPSOIDAL COLLAPSE', 'ELLIPSOIDAL_COLLAPSE']:
        nu = lambda S,w: w**2/S
        Stilde = lambda S1,S0: dX(S1,S0)/S0
        A0 = lambda S0,w0: 0.8661*(1-0.133*nu(S0,w0)**(-0.615))
        A1 = lambda S0,w0: 0.308*nu(S0,w0)**(-0.115)
        A2 = lambda S0,w0: 0.0373*nu(S0,w0)**(-0.115)
        A3 = lambda S1,w1,S0,w0: A0(S0,w0)**2+2*A0(S0,w0)*A1(S0,w0)*np.sqrt(dX(S1,S0)*Stilde(S1,S0))/dX(w1,w0)
        fcoll = lambda S1,w1,S0,w0: A0(S0,w0)/np.sqrt(2*np.pi)*dX(w1,w0)/dX(S1,S0)**(1.5)*np.exp(-A1(S0,w0)**2/2*Stilde(S1,S0))\
                                    *(np.exp(-A3(S1,w1,S0,w0)*dX(w1,w0)**2/2/dX(S1,S0))+A2(S0,w0)*Stilde(S1,S0)**1.5*(1+2*A1(S0,w0)*np.sqrt(Stilde(S1,S0)/np.pi)))
    else:
        assert False, f"{collapse_model} is not implemented in progenitor_conditional_probability_function module."
    S1 = sigma_squared(M1, param, **kwargs) 
    S0 = sigma_squared(M0, param, **kwargs)
    w1 = delcrit_SC(z1, param)
    w0 = delcrit_SC(z0, param)
    dlnsigmaM1_dlnM1 = dlnsigma_dlnM(M1, param)
    dS1dM1_abs = -dlnsigmaM1_dlnM1*2*S1/M1
    p_CPF = fcoll(S1,w1,S0,w0)*dS1dM1_abs
    # print(M0, M1, S0, S1, p_CPF)
    return p_CPF

def progenitor_mass_function(M1, z1, M0, z0, param, collapse_model='SC', **kwargs):
    '''
    Function to compute the mass function for progenitors (Eq. 3 in Jiang & van den Bosch 2014).
    '''
    var_tck = kwargs.get('var_tck')
    if var_tck is None: 
        rbin, mbin, var = sigma_squared_table(param, **kwargs)
        var_tck = splrep(mbin, var)
    kwargs['var_tck'] = var_tck
    p_CPF = progenitor_conditional_probability_function(M1, z1, M0, z0, param, collapse_model=collapse_model, **kwargs)
    n_EPS = (M0/M1)*p_CPF
    return n_EPS

class ParkinsonColeHelly2008:
    def __init__(self, param, e1=0.1, e2=0.1, G0=0.57, g1=0.38, g2=- 0.01, M_res=1e4):
        self.param = param 
        self.e1 = e1
        self.e2 = e2
        self.G0 = G0 
        self.g1 = g1 
        self.g2 = g2 
        self.M_res = M_res 

    def J(self, u_res):
        if self.J_tck is None:
            fun = lambda u: (1+1/u**2)**(self.g1/2)
            return quad(fun, 0, u_res)[0]
        else:
            return splev(u_res, self.J_tck)
    
    def prepare_Jfit(self):
        if self.param.code.verbose:
            print('Creating the table for J(u_res) function...')
        uu = np.linspace(1e-5, 300.0, 600)
        JJ = np.array([self.J(u0) for u0 in tqdm(uu)])
        self.J_tck = splrep(uu, JJ)
        if self.param.code.verbose:
            print('...done')
    
    def prepare_sigmaM(self):
        if self.param.code.verbose:
            print('Computing the sigma(M) and required derivatives...')
        param = self.param
        param.file.ps = calc_Plin(param)
        rbin, mbin, var = sigma_squared_table(param)
        sig = np.sqrt(var)
        lnsigma_tck = splrep(np.log(mbin), np.log(sig))
        dlnsbin_dlnMbin = np.gradient(np.log(sig), np.log(mbin))
        alpha_tck = splrep(np.log(mbin), -dlnsbin_dlnMbin)
        self.lnsigma_tck = lnsigma_tck
        self.alpha_tck = alpha_tck
        if self.param.code.verbose:
            print('...done')

    def sigma(self, M):
        return np.exp(splev(np.log(M), self.lnsigma_tck))
    
    def alpha(self, M):
        return splev(np.log(M), self.alpha_tck)
    
    def V(self, q, M2):
        s1 = self.sigma(q*M2)
        s2 = self.sigma(M2)
        return s1**2/(s1**2-s2**2)**(1.5)
    
    def beta(self, q_res, M2):
        return np.log(self.V(q_res,M2)/self.V(0.5, M2))/np.log(2*q_res)
    
    def B(self, q_res, M2): 
        return self.V(q_res,M2)/(q_res**self.beta(q_res,M2))
    
    def mu(self, M2): 
        if self.g1>=0:
            mu_ = self.alpha(M2/2)
        else:
            sigma_res = self.sigma(self.M_res)
            sigma_h = self.sigma(M2/2)
            q_res = self.M_res/M2
            mu_ = -np.log(sigma_res/sigma_h)/np.log(2*q_res)
        return mu_
    
    def eta(self, q_res, M2):
        return self.beta(q_res,M2) - 1 - self.g1*self.mu(M2)
    
    def delta(self, z):
        param = self.param
        d = param.mf.dc/growth_factor(z, param)
        return d
    
    def ddelta_dz(self, z, dz=1e-5):
        return (self.delta(z+dz)-self.delta(z-dz))/(2*dz)
    
    def S(self, q, z, M2, q_res):
        G0 = self.G0 
        g1 = self.g1 
        g2 = self.g2
        s2 = self.sigma(M2)
        s_h = self.sigma(M2/2)
        alpha_h = self.alpha(M2/2)
        
        f1 = q**(self.eta(q_res,M2)-1)
        f2 = (self.delta(z)/s2)**g2
        f3 = (s_h/s2)**g1
        Sq = np.sqrt(2/np.pi)*self.B(q_res,M2)*alpha_h*f1*(G0/2**(self.mu(M2)*g1))*f2*f3*self.ddelta_dz(z)

        return Sq
    
    def R(self, q, M2, q_res):
        G0 = self.G0 
        g1 = self.g1 
        g2 = self.g2
        s1 = self.sigma(q*M2)
        s2 = self.sigma(M2)
        s_h = self.sigma(M2/2)
        alpha_h = self.alpha(M2/2)
        alpha_1 = self.alpha(q*M2)

        f1 = alpha_1/alpha_h
        f2 = self.V(q,M2)/(self.B(q_res,M2)*(q**self.beta(q_res,M2))) 
        f3 = ((2*q)**self.mu(M2) * s1/s_h)**g1
        Rq = f1*f2*f3
        
        return Rq
    
    def N_upper_Dz1(self, z, M2, q_res):
        Nupp = ( self.S(1,z,M2,q_res)/self.eta(q_res,M2) ) \
                    *( 0.5**self.eta(q_res,M2) - q_res**self.eta(q_res,M2) )
        return Nupp
    
    def N_upper(self, z, M2, q_res):
        """ 
        The upper limit to the expected number of
        resolved fragments produced in a redshift
        step Δz. 
        """
        Nupp = self.N_upper_Dz1(z, M2, q_res) * self.Dz(z, M2, q_res)
        return Nupp
    
    def Dz(self, z, M2, q_res):
        """ The redshift step size for the merger
            tree algorithm. """

        Integral = self.N_upper_Dz1(z, M2, q_res)
        # print('I = ', Integral)

        s2 = self.sigma(M2)
        s_h = self.sigma(M2/2)
        dz1 = np.sqrt(2)*np.sqrt( s_h**2 - s2**2 )/self.ddelta_dz(z)

        dz = min( self.e1*dz1, self.e2/Integral )
        return dz
    
    def F(self, z, M2, q_res):
        """ 
        The fraction of mass in progenitors to M2
            below the resolution limit. 
        """
        M_res = q_res*M2
        s_res = self.sigma(M_res)
        s2    = self.sigma(M2)
        u_res = s2/np.sqrt( s_res**2 - s2**2 )
        d2    = self.delta(z)

        G0 = self.G0 
        g2 = self.g2
        dz = self.Dz(z,M2,q_res)
        FF = np.sqrt(2/np.pi)*self.J(u_res)*(G0/s2)*((d2/s2)**g2)*self.ddelta_dz(z)*dz
        return FF
    
    def run(self, M0, z0, z_max, M_res=None, use_cython=True, max_tree_length=1000):
        if M_res is None: M_res = self.M_res
        else: self.M_res = M_res

        print(f'Simulation setup:')
        print(f'Starting halo mass = {M0:.2e}')
        print(f'Starting redshift  = {z0:.5f}')
        print(f'Mass resolution    = {M_res:.2e}')
        print(f'Maximum redshift   = {z_max:.5f}')

        if use_cython:
            from . import cython_ParkinsonColeHelly2008

            param = self.param
            cosmo = cython_ParkinsonColeHelly2008.CosmoParams(
                                        Om = param.cosmo.Om, 
                                        Ob = param.cosmo.Ob, 
                                        Or = param.cosmo.Or, 
                                        Ok = param.cosmo.Ok, 
                                        Ode = 1-param.cosmo.Om, 
                                        h0  = param.cosmo.h0, 
                                        )
            
            param.file.ps = calc_Plin(param)
            rbin, mbin, var = sigma_squared_table(param)

            z_tree, M_tree, z_subh, M_subh = cython_ParkinsonColeHelly2008.ParkinsonColeHelly2008_run(
                                    M0, z0, z_max, 
                                    cosmo, 
                                    mbin, 
                                    np.sqrt(var),
                                    M_res = M_res, 
                                    e1 = self.e1, 
                                    e2 = self.e2, 
                                    G0 = self.G0, 
                                    g1 = self.g1, 
                                    g2 = self.g2,
                                    max_tree_length=max_tree_length,
                                    )
            self.z_tree, self.M_tree, self.z_subh, self.M_subh = z_tree, M_tree, z_subh, M_subh
            
        else:
            self.prepare_sigmaM()
            self.J_tck = None
            self.prepare_Jfit()

            self.z_tree = [z0]
            self.M_tree = [M0]
            self.z_subh = []
            self.M_subh = []

            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            tstart = time()
            count  = 0
            while self.z_tree[-1] < z_max and self.M_tree[-1] > 2*M_res and count < max_tree_length:
                logging.info(f'z = {self.z_tree[-1]:.6f} | main M = {self.M_tree[-1]:.3e} | counter = {count} | t = {(time()-tstart)/60:.2f} mins')
                count += 1
                # print(f'Modeling splitting of halo with mass {self.M_tree[-1]:.2e} at z={self.z_tree[-1]:.5f}...')
                M2 = self.M_tree[-1]
                q_res = M_res/M2

                r1 = np.random.uniform()
                if r1 > self.N_upper(self.z_tree[-1], M2, q_res):
                    # No split occurs, but the halo
                    # mass M2 is reduced to M2(1 - F).
                    M_new  = M2*( 1 - self.F(self.z_tree[-1], M2, q_res) )
                    self.M_tree.append(M_new)
                    z_new  = self.z_tree[-1] + self.Dz(self.z_tree[-1], M2, q_res)
                    self.z_tree.append(z_new)
                    # print(f'Halo reduced to mass {self.M_tree[-1]:.2e}.')
                else:
                    r2 = np.random.uniform()
                    # We use inverse transform techqniue to 
                    # draw a random number from our distrbution.
                    et = self.eta(q_res, M2)
                    q = (q_res**et + (2**(-et)-q_res**et)*r2)**(1/et)

                    r3 = np.random.uniform()
                    if r3 < self.R(q, M2, q_res):
                        # Two progenitors are created with 
                        # mass q*M2 and M2*(1 - F - q)

                        M_new1 = M2*q
                        M_new2 = M2*(1 - self.F(self.z_tree[-1], M2, q_res) - q)

                        self.M_tree.append(max(M_new1,M_new2))
                        self.M_subh.append(min(M_new1,M_new2))

                        self.z_subh.append(self.z_tree[-1])
                        z_new  = self.z_tree[-1] + self.Dz(self.z_tree[-1], M2, q_res)
                        self.z_tree.append(z_new)
                        # print(f'Halo fragmented into two haloes of mass {self.M_tree[-1]:.2e} and {self.M_subh[-1]:.2e}.')
                    else:
                        # No split occurs, but the halo
                        # mass M2 is reduced to M2(1 - F).
                        M_new  = M2*( 1 - self.F(self.z_tree[-1], M2, q_res) )
                        self.M_tree.append(M_new)
                        z_new  = self.z_tree[-1] + self.Dz(self.z_tree[-1], M2, q_res)
                        self.z_tree.append(z_new)
                        # print(f'Halo reduced to mass {self.M_tree[-1]:.2e}.')

        return {
            'z_main': np.array(self.z_tree), 
            'M_main': np.array(self.M_tree), 
            'z_subh': np.array(self.z_subh), 
            'M_subh': np.array(self.M_subh),
            }


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import splrep, splev

    import toolscosmo
    from toolscosmo import merger_trees

    EH1998_data = np.array([[1.09648e+2, 1.32650e+1],
                            [1.58489e+3, 1.16878e+1],
                            [2.88403e+5, 8.86487e+0],
                            [4.36516e+7, 6.41818e+0],
                            [2.08930e+10, 3.64317e+0],
                            [1.04713e+13, 1.58339e+0],
                            [1.20226e+15, 5.71158e-1],
                            [1.99526e+16, 2.42192e-1],
                            [1.51356e+17, 1.07545e-1],
                            ])

    param = toolscosmo.par('lcdm')
    param.code.zmin = 0.0
    param.code.kmax = 500
    param.code.rmin = 1e-4 #0.002
    param.code.rmax = 50 #25
    param.code.Nrbin = 300
    param.cosmo.Om = 0.315
    param.cosmo.Or = 9.1e-5
    param.cosmo.Ob = 0.045
    param.cosmo.h0 = 0.67
    param.cosmo.ns = 0.965
    param.cosmo.As = None #2.089e-09
    param.cosmo.s8 = 0.811
    param.cosmo.solver = 'toolscosmo' #'astropy'
    param.mf.dc = 1.686
    param.file.ps = 'CLASS'
    param.file.ps = toolscosmo.calc_Plin(param)

    Ms = 10**np.linspace(6,15,120)
    sigM_approx = merger_trees.sigma_approx(Ms, param)
    sigM = merger_trees.sigma_squared(Ms*param.cosmo.h0, param)**0.5
    rbin, mbin, var = merger_trees.sigma_squared_table(param)

    dlnsigmaM_dlnM_approx = merger_trees.dlnsigma_dlnM_approx(Ms, param)
    dlnsigmaM_dlnM = merger_trees.dlnsigma_dlnM(Ms*param.cosmo.h0, param)

    fig, axs = plt.subplots(1,2,figsize=(14,6))
    ax = axs[0]
    ax.loglog(Ms, sigM_approx, lw=3, c='C1', label='sigma(M) fit')
    ax.loglog(Ms, 10**splev(np.log10(Ms), splrep(np.log10(EH1998_data[:,0]),np.log10(EH1998_data[:,1]))), 
                               lw=3, c='k', ls='--', label='Eisenstein & Hu (1998)')
    ax.loglog(mbin/param.cosmo.h0, var**0.5, lw=3, c='C0', ls=':', label='toolscosmo')
    ax.set_xlabel('M $[M_\odot]$', fontsize=16)
    ax.set_ylabel('$\sigma (M)$', fontsize=16)
    ax.axis([1e6,1e15,0.5,15])
    ax.legend()
    ax = axs[1]
    ax.semilogx(Ms, -dlnsigmaM_dlnM_approx, lw=3, c='C1', label='derived from sigma(M) fit')
    ax.semilogx(Ms, -dlnsigmaM_dlnM, lw=3, c='C0', ls=':', label='toolscosmo')
    ax.set_xlabel('M $[M_\odot]$', fontsize=16)
    ax.set_ylabel('-dlog$\sigma(M)$/dlog$M$', fontsize=16)
    ax.axis([1e6,1e15,0.01,0.35])
    plt.tight_layout()
    plt.show()

    