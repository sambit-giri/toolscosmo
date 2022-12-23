"""

CONSTANTS AND UNIT CONVERSIONS

"""

#Constants
c        = 2.99792e5                # Speed of light [km/s]
hP       = 4.1357e-15               # Planck constant [eV/Hz]
m_p      = 8.4119e-58               # Proton mass in [Msun]
kB       = 1.380649e-16             # Boltzmann constsant [erg/K]
rhoc0    = 2.755e11                 # Critical density at z=0 [h^2 Msun/Mpc^3]

nu_LL = 3.2898e15                   # Lyman-limit frequency [Hz]
nu_al = 2.4674e15                   # Lyman-alpha frequency [Hz]
nu_be = 2.9243e15                   # Lyman-beta frequency [Hz]

f_He_bymass = 0.2453                # Helium fraction by mass from BBN
f_He_bynumb = 1/(1/f_He_bymass-1)/4 # Helium fraction by number

#Unit conversions
sec_per_yr = 3600*24*365.25         # Seconds per year
cm_per_Mpc = 3.08568e24             # Centimetres per Mpc
km_per_Mpc = cm_per_Mpc/1e5         # km per Mpc
eV_per_erg = 6.24151e11             # eV per erg      


#Unsorted
c_kmps = 299792.
Mpc_to_km = 3.0860e19
Gyr_to_s  = 3.1536e16