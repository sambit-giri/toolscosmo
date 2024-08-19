from scipy.integrate import quad, odeint
try:
    from scipy.integrate import trapz
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import trapezoid as trapz
    from scipy.integrate import cumulative_trapezoid as cumtrapz

from scipy.interpolate import splrep, splev, interp1d
from scipy.special import gamma, erf
from scipy.signal import savgol_filter
from scipy.optimize import fsolve