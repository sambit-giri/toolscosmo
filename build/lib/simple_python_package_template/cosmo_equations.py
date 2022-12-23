import numpy as np
from scipy.integrate import quad

from .basic_functions import *
from . import constants as const

class FriedmannEquation:

    def __init__(self, param):
        self.param = param
        self.create_functions()

    def create_functions(self):
        self.Ea = lambda a: np.sqrt(self.param.cosmo.Or/a**4+self.param.cosmo.Om/a**3+\
                            self.param.cosmo.Ode+self.param.cosmo.Ok/a**2)
        self.Ez = lambda z: self.Ea(z_to_a(z))
        self.Ha = lambda a: (self.param.cosmo.h*100)*self.Ea(a)
        self.Hz = lambda z: self.Ha(z_to_a(z))

    def H(self, z=None, a=None):
        assert z is not None or a is not None 
        if z is not None: return self.Hz(z)
        else: return self.Ha(a)

class CosmoDistances(FriedmannEquation):

    def __init__(self, param):
        self.param = param
        self.create_functions()

    def _comoving_dist(self, z):
        I = lambda z: const.c_kmps/self.H(z=z)
        return quad(I, 0, z)[0] # Mpc

    def comoving_dist(self, z=None, a=None):
        if a is not None: z = a_to_z(a)
        return np.vectorize(self._comoving_dist)(z)

    def proper_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc/(1+z)

    def angular_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc/(1+z)
    
    def luminosity_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc*(1+z)