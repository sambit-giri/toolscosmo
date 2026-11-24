from scipy.integrate import quad, odeint
try:
    from scipy.integrate import trapz, cumtrapz, simps
except:
    from scipy.integrate import trapezoid as trapz
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    from scipy.integrate import simpson as simps

try:
    from scipy.integrate import cumulative_simpson as cumsimpson  # scipy >= 1.12
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumsimpson

from scipy.interpolate import splrep, splev, interp1d, CubicSpline
from scipy.special import gamma, erf
from scipy.signal import savgol_filter
from scipy.optimize import fsolve