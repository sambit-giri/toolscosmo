"""
Accuracy of the native 'toolscosmo' solver against astropy.

For every background cosmology that astropy implements (flat LambdaCDM, wCDM
and CPL/w0wa), the native solver must reproduce astropy's background
quantities and distances. The cosmologies are tested with radiation switched
off (astropy Tcmb0=0) and switched on. With radiation on, astropy uses massless
neutrinos so that its radiation density scales exactly as (1+z)^4, and
toolscosmo's Or is set to astropy's Ogamma0 + Onu0.

Tolerances:
  - E(z), H(z), w(z), Omega_DE(z), T_cmb(z) are analytic: rtol 1e-8.
  - Distances need a numerical integral: |dmu| < 1e-5 mag
    (relative distance error < ~5e-6) over 1e-4 <= z <= 1100.
"""
import numpy as np
import pytest
import toolscosmo
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM

H0 = 67.3
TCMB = 2.7255
Z_BG = np.concatenate([[0.0], np.logspace(-4, np.log10(1100), 400)])
Z_DIST = Z_BG[1:]

DMU_TOL = 1e-5     # mag
RTOL_ANALYTIC = 1e-8

COSMOLOGIES = [
    # id, DE, Om, DE params
    ('lcdm_Om0.315', 'LCDM', 0.315, {}),
    ('lcdm_Om0.15',  'LCDM', 0.15,  {}),
    ('lcdm_Om0.6',   'LCDM', 0.6,   {}),
    ('wcdm_w-0.8',   'wCDM', 0.3,   {'w': -0.8}),
    ('wcdm_w-1.2',   'wCDM', 0.3,   {'w': -1.2}),
    ('cpl_-0.7_-1',  'CPL',  0.3,   {'w0': -0.7, 'wa': -1.0}),
    ('cpl_-1.2_0.8', 'CPL',  0.3,   {'w0': -1.2, 'wa': 0.8}),
    ('cpl_-0.9_0.3', 'CPL',  0.35,  {'w0': -0.9, 'wa': 0.3}),
]


def _astropy(DE, Om, de, radiation):
    kw = dict(H0=H0, Om0=Om)
    kw.update(dict(Tcmb0=TCMB, Neff=3.046, m_nu=0.0) if radiation else dict(Tcmb0=0.0))
    if DE == 'LCDM':
        return FlatLambdaCDM(**kw)
    if DE == 'wCDM':
        return FlatwCDM(w0=de['w'], **kw)
    return Flatw0waCDM(w0=de['w0'], wa=de['wa'], **kw)


def _toolscosmo(DE, Om, de, ref):
    p = toolscosmo.par(DE=DE) if DE != 'LCDM' else toolscosmo.par()
    p.cosmo.solver = 'toolscosmo'
    p.cosmo.h0 = H0 / 100
    p.cosmo.Om = Om
    p.cosmo.Or = float(ref.Ogamma0 + ref.Onu0)
    p.cosmo.Tcmb = float(ref.Tcmb0.value)
    for k, v in de.items():
        setattr(p.DE, k, v)
    return p


@pytest.fixture(
    params=[(c, rad) for c in COSMOLOGIES for rad in (False, True)],
    ids=[f"{c[0]}-{'rad' if rad else 'norad'}" for c in COSMOLOGIES for rad in (False, True)],
)
def models(request):
    (_, DE, Om, de), radiation = request.param
    ref = _astropy(DE, Om, de, radiation)
    return _toolscosmo(DE, Om, de, ref), ref


# --- Background quantities ---

def test_Ez(models):
    p, ref = models
    np.testing.assert_allclose(toolscosmo.Ez_model(p)(Z_BG), ref.efunc(Z_BG), rtol=RTOL_ANALYTIC)

def test_hubble(models):
    p, ref = models
    np.testing.assert_allclose(toolscosmo.hubble(Z_BG, p), ref.H(Z_BG).value, rtol=RTOL_ANALYTIC)

def test_w_DE(models):
    p, ref = models
    np.testing.assert_allclose(toolscosmo.w_DE(Z_BG, p) * np.ones_like(Z_BG), ref.w(Z_BG), rtol=RTOL_ANALYTIC)

def test_Omega_DE(models):
    # toolscosmo.Omega_DE(z) is rho_DE(z)/rho_crit,0 = Ode0 * rho_DE(z)/rho_DE(0)
    p, ref = models
    expected = ref.Ode0 * ref.de_density_scale(Z_BG)
    got = np.array([toolscosmo.Omega_DE(z, p) for z in Z_BG])
    np.testing.assert_allclose(got, expected, rtol=RTOL_ANALYTIC)

def test_Tcmb(models):
    p, ref = models
    if ref.Tcmb0.value == 0:
        pytest.skip('no CMB temperature without radiation')
    np.testing.assert_allclose(toolscosmo.T_cmb(Z_BG, p), ref.Tcmb(Z_BG).value, rtol=RTOL_ANALYTIC)


# --- Distances ---

def _dmu(d, d_ref):
    return np.max(np.abs(5 * np.log10(d / d_ref)))

@pytest.mark.parametrize('method', ['grid', 'quad'])
def test_comoving_distance(models, method):
    p, ref = models
    d = toolscosmo.comoving_distance(Z_DIST, p, method=method)
    assert _dmu(d, ref.comoving_distance(Z_DIST).value) < DMU_TOL

def test_luminosity_distance(models):
    p, ref = models
    d = toolscosmo.luminosity_distance(Z_DIST, p)
    assert _dmu(d, ref.luminosity_distance(Z_DIST).value) < DMU_TOL

def test_distance_modulus(models):
    p, ref = models
    mu = toolscosmo.distance_modulus(Z_DIST, p)
    assert np.max(np.abs(mu - ref.distmod(Z_DIST).value)) < DMU_TOL

def test_distance_modulus_random_unsorted_sne_like(models):
    # Unsorted, repeated SNe-like redshifts, as passed by a likelihood.
    p, ref = models
    z = np.random.default_rng(42).uniform(0.001, 2.3, 1500)
    z[::7] = z[3]
    mu = toolscosmo.distance_modulus(z, p)
    assert np.max(np.abs(mu - ref.distmod(z).value)) < DMU_TOL


# --- The astropy solver path inside toolscosmo must agree with native ---

def test_native_vs_astropy_solver_distance_modulus():
    # The 'astropy' solver wrapper builds a radiation-free cosmology, so
    # compare with Or=0 on the native side.
    for DE, de in [('LCDM', {}), ('wCDM', {'w': -0.8}), ('CPL', {'w0': -0.7, 'wa': -1.0})]:
        p_nat = toolscosmo.par(DE=DE) if DE != 'LCDM' else toolscosmo.par()
        p_ast = toolscosmo.par(DE=DE) if DE != 'LCDM' else toolscosmo.par()
        for p, solver in [(p_nat, 'toolscosmo'), (p_ast, 'astropy')]:
            p.cosmo.solver = solver
            p.cosmo.Or = 0.0
            for k, v in de.items():
                setattr(p.DE, k, v)
        z = np.logspace(-3, np.log10(3), 200)
        dmu = toolscosmo.distance_modulus(z, p_nat) - toolscosmo.distance_modulus(z, p_ast)
        assert np.max(np.abs(dmu)) < DMU_TOL, DE
