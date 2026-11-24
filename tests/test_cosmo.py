import numpy as np
import pytest
import toolscosmo


@pytest.fixture
def param():
    p = toolscosmo.par()
    p.cosmo.solver = 'toolscosmo'
    return p


@pytest.fixture
def param_cpl():
    p = toolscosmo.par(DE='CPL')
    p.cosmo.solver = 'toolscosmo'
    p.DE.w0 = -0.9
    p.DE.wa = 0.1
    return p


# --- T_cmb ---

def test_Tcmb():
    param = toolscosmo.par()
    assert np.isclose(toolscosmo.T_cmb(1, param), 2.72 * 2, atol=0.01)

def test_Tcmb_z0(param):
    assert np.isclose(toolscosmo.T_cmb(0, param), param.cosmo.Tcmb)

def test_Tcmb_scales_as_1pz(param):
    T1 = toolscosmo.T_cmb(1, param)
    T2 = toolscosmo.T_cmb(2, param)
    assert np.isclose(T2 / T1, 3/2, rtol=1e-6)


# --- Hubble ---

def test_hubble_z0_equals_H0(param):
    H0 = toolscosmo.hubble(0, param)
    assert np.isclose(H0, 100 * param.cosmo.h0, rtol=1e-4)

def test_hubble_increases_with_z(param):
    H = [toolscosmo.hubble(z, param) for z in [0, 1, 5, 10]]
    assert all(H[i] < H[i+1] for i in range(len(H)-1))

def test_Ez_unity_at_z0(param):
    Ez = toolscosmo.Ez_model(param)
    assert np.isclose(Ez(0), 1.0, rtol=1e-4)

def test_Ez_positive(param):
    Ez = toolscosmo.Ez_model(param)
    for z in [0, 1, 5, 10, 20]:
        assert Ez(z) > 0


# --- Omega_DE ---

def test_Omega_DE_lcdm_z0_flat(param):
    # Flat universe: Omega_DE(z=0) = 1 - Om - Or
    expected = 1 - param.cosmo.Om - param.cosmo.Or
    assert np.isclose(toolscosmo.Omega_DE(0, param), expected, rtol=1e-4)

def test_Omega_DE_lcdm_constant(param):
    # ΛCDM: Omega_DE is constant (w=-1, a^0)
    Ode0 = toolscosmo.Omega_DE(0, param)
    Ode1 = toolscosmo.Omega_DE(1, param)
    assert np.isclose(Ode0, Ode1, rtol=1e-6)

def test_Omega_DE_cpl_z0(param_cpl):
    # At z=0 (a=1), CPL reduces to Omega_L regardless of w0, wa
    expected = 1 - param_cpl.cosmo.Om - param_cpl.cosmo.Or
    assert np.isclose(toolscosmo.Omega_DE(0, param_cpl), expected, rtol=1e-4)

def test_Omega_DE_cpl_evolves(param_cpl):
    # For w0 != -1, Omega_DE should differ at z=0 vs z=1
    Ode0 = toolscosmo.Omega_DE(0, param_cpl)
    Ode1 = toolscosmo.Omega_DE(1, param_cpl)
    assert not np.isclose(Ode0, Ode1, rtol=1e-3)


# --- w_DE ---

def test_w_DE_lcdm(param):
    assert toolscosmo.w_DE(0, param) == -1

def test_w_DE_cpl_at_z0(param_cpl):
    # w(z=0) = w0 + wa * 0/(1+0) = w0
    assert np.isclose(toolscosmo.w_DE(0, param_cpl), param_cpl.DE.w0)

def test_w_DE_cpl_at_highz(param_cpl):
    # w(z→∞) → w0 + wa
    w_highz = toolscosmo.w_DE(100, param_cpl)
    assert np.isclose(w_highz, param_cpl.DE.w0 + param_cpl.DE.wa, rtol=1e-2)


# --- Distances ---

def test_comoving_distance_z0(param):
    assert np.isclose(toolscosmo.comoving_distance(0, param), 0.0, atol=1e-3)

def test_comoving_distance_positive(param):
    assert toolscosmo.comoving_distance(1.0, param) > 0

def test_comoving_distance_monotone(param):
    d = [toolscosmo.comoving_distance(z, param) for z in [0.5, 1.0, 2.0, 5.0]]
    assert all(d[i] < d[i+1] for i in range(len(d)-1))

def test_luminosity_distance_z0(param):
    assert np.isclose(toolscosmo.luminosity_distance(0, param), 0.0, atol=1e-3)

def test_luminosity_distance_geq_comoving(param):
    for z in [0.5, 1.0, 2.0]:
        dl = toolscosmo.luminosity_distance(z, param)
        dc = toolscosmo.comoving_distance(z, param)
        assert dl >= dc - 1e-6  # dl = dc*(1+z) >= dc

def test_luminosity_distance_relation(param):
    z = 1.0
    dc = toolscosmo.comoving_distance(z, param)
    dl = toolscosmo.luminosity_distance(z, param)
    assert np.isclose(dl, dc * (1 + z), rtol=1e-3)


# --- Distances: regression against astropy (issue #16) ---
# Radiation is switched off so the toolscosmo model matches astropy's
# Tcmb0=0 flat cosmologies exactly.

Z_TEST = np.concatenate([np.logspace(-4, np.log10(3), 200), [10.0, 100.0, 1100.0]])

def _par_norad(DE='LCDM', Om=0.315, w0=-1.0, wa=0.0):
    p = toolscosmo.par(DE=DE) if DE != 'LCDM' else toolscosmo.par()
    p.cosmo.solver = 'toolscosmo'
    p.cosmo.Om = Om
    p.cosmo.Or = 0.0
    if DE == 'CPL':
        p.DE.w0, p.DE.wa = w0, wa
    return p

def _astropy_ref(DE, Om, h0, w0=-1.0, wa=0.0):
    from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
    if DE == 'CPL':
        return Flatw0waCDM(H0=100*h0, Om0=Om, w0=w0, wa=wa, Tcmb0=0)
    return FlatLambdaCDM(H0=100*h0, Om0=Om, Tcmb0=0)

@pytest.mark.parametrize('method', ['grid', 'quad'])
@pytest.mark.parametrize('DE,Om,w0,wa', [
    ('LCDM', 0.315, -1.0,  0.0),
    ('LCDM', 0.15,  -1.0,  0.0),
    ('LCDM', 0.6,   -1.0,  0.0),
    ('CPL',  0.3,   -0.7, -1.0),
    ('CPL',  0.3,   -1.2,  0.8),
])
def test_distance_modulus_matches_astropy(method, DE, Om, w0, wa):
    p = _par_norad(DE, Om, w0, wa)
    ref = _astropy_ref(DE, Om, p.cosmo.h0, w0, wa)
    dc = toolscosmo.comoving_distance(Z_TEST, p, method=method)
    dmu = 5*np.log10(dc / ref.comoving_distance(Z_TEST).value)
    assert np.max(np.abs(dmu)) < 1e-4  # mag

def test_distance_modulus_low_z_no_offset():
    # The old integrator started at z=1e-3 and was 0.1 mag low at z=0.023.
    p = _par_norad()
    ref = _astropy_ref('LCDM', 0.315, p.cosmo.h0)
    z = np.array([0.001, 0.005, 0.01, 0.023])
    assert np.allclose(toolscosmo.distance_modulus(z, p), ref.distmod(z).value, atol=1e-4)

def test_comoving_distance_tiny_z_positive_and_hubble_law(param):
    # For z << 1, D_C -> c z / H0. The old code returned negative values for z < 1e-3.
    from toolscosmo.constants import c
    z = np.array([1e-6, 1e-5, 1e-4])
    dc = toolscosmo.comoving_distance(z, param)
    assert np.all(dc > 0)
    assert np.allclose(dc, c*z/(100*param.cosmo.h0), rtol=1e-3)

def test_comoving_distance_shapes(param):
    assert np.isscalar(toolscosmo.comoving_distance(1.0, param))
    assert toolscosmo.comoving_distance(0.0, param) == 0.0
    out = toolscosmo.comoving_distance([0.0, 0.5, 1.0], param)
    assert out.shape == (3,) and out[0] == 0.0
    z2d = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert toolscosmo.comoving_distance(z2d, param).shape == (2, 2)

def test_comoving_distance_order_independent(param):
    # The grid method interpolates, so output must not depend on input ordering.
    z = np.array([2.0, 0.01, 1.0, 0.1])
    d = toolscosmo.comoving_distance(z, param)
    d_sorted = toolscosmo.comoving_distance(np.sort(z), param)
    assert np.allclose(d[np.argsort(z)], d_sorted, rtol=1e-12)

def test_comoving_distance_grid_vs_quad_with_radiation(param):
    z = np.logspace(-3, np.log10(1100), 50)
    d_grid = toolscosmo.comoving_distance(z, param, method='grid')
    d_quad = toolscosmo.comoving_distance(z, param, method='quad')
    assert np.allclose(d_grid, d_quad, rtol=1e-5)

def test_comoving_distance_bad_method(param):
    with pytest.raises(ValueError):
        toolscosmo.comoving_distance(1.0, param, method='trapz')
