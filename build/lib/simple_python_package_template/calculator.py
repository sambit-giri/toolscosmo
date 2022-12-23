import numpy as np 
from scipy.integrate import quad

from .basic_functions import *
from .cosmo_equations import * 
from . import constants as const

def age_estimator(param, z):
    Feq = FriedmannEquation(param)
    a = z_to_a(z)
    I = lambda a: 1/a/Feq.H(a=a)
    t = lambda a: quad(I, 0, a)[0]*const.Mpc_to_km/const.Gyr_to_s
    return np.vectorize(t)(a) 