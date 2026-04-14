import numpy as np
import pytest
import toolscosmo


@pytest.fixture
def param_lcdm():
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


Z = np.array([0.5, 1.0, 5.0, 10.0, 20.0])  # z <= 30 (cosmic dawn range)


# --- solveODE ---

def test_ode_unity_at_z0(param_lcdm):
    D = toolscosmo.growth_factor_solveODE(np.array([0.0]), param_lcdm)
    assert np.isclose(D[0], 1.0, rtol=1e-3)

def test_ode_less_than_unity(param_lcdm):
    D = toolscosmo.growth_factor_solveODE(Z, param_lcdm)
    assert np.all(D < 1.0)

def test_ode_positive(param_lcdm):
    D = toolscosmo.growth_factor_solveODE(Z, param_lcdm)
    assert np.all(D > 0)

def test_ode_decreasing_with_z(param_lcdm):
    D = toolscosmo.growth_factor_solveODE(Z, param_lcdm)
    assert np.all(np.diff(D) < 0)


# --- Linder2005 ---

def test_linder_unity_at_z0(param_lcdm):
    D = toolscosmo.growth_factor_Linder2005(np.array([0.0]), param_lcdm)
    assert np.isclose(D[0], 1.0, rtol=1e-3)

def test_linder_less_than_unity(param_lcdm):
    D = toolscosmo.growth_factor_Linder2005(Z, param_lcdm)
    assert np.all(D < 1.0)

def test_linder_positive(param_lcdm):
    D = toolscosmo.growth_factor_Linder2005(Z, param_lcdm)
    assert np.all(D > 0)

def test_linder_decreasing_with_z(param_lcdm):
    D = toolscosmo.growth_factor_Linder2005(Z, param_lcdm)
    assert np.all(np.diff(D) < 0)


# --- CPT (Carroll, Press & Turner 1992) ---

def test_cpt_unity_at_z0(param_lcdm):
    D = toolscosmo.growth_factor_CPT(np.array([0.0]), param_lcdm)
    assert np.isclose(D[0], 1.0, rtol=1e-3)

def test_cpt_less_than_unity(param_lcdm):
    D = toolscosmo.growth_factor_CPT(Z, param_lcdm)
    assert np.all(D < 1.0)

def test_cpt_positive(param_lcdm):
    D = toolscosmo.growth_factor_CPT(Z, param_lcdm)
    assert np.all(D > 0)

def test_cpt_decreasing_with_z(param_lcdm):
    D = toolscosmo.growth_factor_CPT(Z, param_lcdm)
    assert np.all(np.diff(D) < 0)


# --- Cross-solver agreement for ΛCDM (z <= 30) ---

def test_linder_vs_ode_lcdm(param_lcdm):
    D_ode = toolscosmo.growth_factor_solveODE(Z, param_lcdm)
    D_lin = toolscosmo.growth_factor_Linder2005(Z, param_lcdm)
    rel_err = np.abs(D_lin - D_ode) / D_ode
    assert np.all(rel_err < 0.01), \
        f"Linder vs ODE (ΛCDM) max error: {rel_err.max():.4f} at z={Z[rel_err.argmax()]}"

def test_cpt_vs_ode_lcdm(param_lcdm):
    z = np.array([0.5, 1.0, 5.0, 10.0])
    D_ode = toolscosmo.growth_factor_solveODE(z, param_lcdm)
    D_cpt = toolscosmo.growth_factor_CPT(z, param_lcdm)
    rel_err = np.abs(D_cpt - D_ode) / D_ode
    assert np.all(rel_err < 0.02), \
        f"CPT vs ODE (ΛCDM) max error: {rel_err.max():.4f} at z={z[rel_err.argmax()]}"

def test_hamilton_vs_ode_lcdm(param_lcdm):
    # Hamilton (2000) should be accurate for ΛCDM (<0.5%)
    z = np.array([0.5, 1.0, 5.0])
    param_lcdm.code.Dz_solver = 'Hamilton2000'
    D_ham = toolscosmo.growth_factor(z, param_lcdm)
    param_lcdm.code.Dz_solver = 'solveODE'
    D_ode = toolscosmo.growth_factor(z, param_lcdm)
    rel_err = np.abs(D_ham - D_ode) / D_ode
    assert np.all(rel_err < 0.005), \
        f"Hamilton vs ODE (ΛCDM) max error: {rel_err.max():.4f}"


# --- CPL: ODE and Linder agree (<1%) ---

def test_linder_vs_ode_cpl(param_cpl):
    D_ode = toolscosmo.growth_factor_solveODE(Z, param_cpl)
    D_lin = toolscosmo.growth_factor_Linder2005(Z, param_cpl)
    rel_err = np.abs(D_lin - D_ode) / D_ode
    assert np.all(rel_err < 0.01), \
        f"Linder vs ODE (CPL) max error: {rel_err.max():.4f} at z={Z[rel_err.argmax()]}"

def test_hamilton_fails_for_cpl(param_cpl):
    # Hamilton (2000) has >1% error for w != -1; document this in test
    z = np.array([0.5, 1.0, 5.0])
    param_cpl.code.Dz_solver = 'Hamilton2000'
    D_ham = toolscosmo.growth_factor(z, param_cpl)
    param_cpl.code.Dz_solver = 'solveODE'
    D_ode = toolscosmo.growth_factor(z, param_cpl)
    rel_err = np.abs(D_ham - D_ode) / D_ode
    # Hamilton is known to deviate for CPL; assert it exceeds the ΛCDM-level accuracy
    assert np.any(rel_err > 0.005), \
        "Expected Hamilton to have >0.5% error for CPL — known limitation"


# --- WDM: linear growth factor identical to CDM ---

def test_wdm_growth_same_as_cdm():
    param_cdm = toolscosmo.par()
    param_cdm.cosmo.solver = 'toolscosmo'
    param_wdm = toolscosmo.par(DM='WDM')
    param_wdm.cosmo.solver = 'toolscosmo'
    param_wdm.DM.m_wdm = 3.0
    D_cdm = toolscosmo.growth_factor_solveODE(Z, param_cdm)
    D_wdm = toolscosmo.growth_factor_solveODE(Z, param_wdm)
    assert np.allclose(D_cdm, D_wdm, rtol=1e-6), \
        "WDM and CDM linear growth factors should be identical"


# --- Dispatcher (growth_factor) ---

@pytest.mark.parametrize("solver", ['Linder2005', 'LinderCahn2007', 'solveODE', 'CPT', 'Hamilton2000'])
def test_dispatcher_returns_valid(solver, param_lcdm):
    param_lcdm.code.Dz_solver = solver
    D = toolscosmo.growth_factor(np.array([1.0]), param_lcdm)
    assert D.shape == (1,)
    assert 0 < D[0] < 1.0

def test_dispatcher_default_is_linder(param_lcdm):
    # Default Dz_solver should be Linder2005
    assert param_lcdm.code.Dz_solver == 'Linder2005'


# --- LinderCahn2007 ---

def test_lc2007_unity_at_z0(param_lcdm):
    D = toolscosmo.growth_factor_LinderCahn2007(np.array([0.0]), param_lcdm)
    assert np.isclose(D[0], 1.0, rtol=1e-3)

def test_lc2007_less_than_unity(param_lcdm):
    D = toolscosmo.growth_factor_LinderCahn2007(Z, param_lcdm)
    assert np.all(D < 1.0)

def test_lc2007_decreasing_with_z(param_lcdm):
    D = toolscosmo.growth_factor_LinderCahn2007(Z, param_lcdm)
    assert np.all(np.diff(D) < 0)

def test_lc2007_equals_linder2005_for_lcdm(param_lcdm):
    # For ΛCDM w=-1 everywhere, γ(a) is constant → identical to Linder2005
    D_l05 = toolscosmo.growth_factor_Linder2005(Z, param_lcdm)
    D_lc7 = toolscosmo.growth_factor_LinderCahn2007(Z, param_lcdm)
    assert np.allclose(D_l05, D_lc7, rtol=1e-4), \
        "LinderCahn2007 should equal Linder2005 for ΛCDM (constant w=-1)"

def test_lc2007_equals_linder2005_for_wcdm():
    # For constant w (wa=0), γ(a) is constant → identical to Linder2005
    param = toolscosmo.par(DE='CPL')
    param.cosmo.solver = 'toolscosmo'
    param.DE.w0 = -0.9
    param.DE.wa = 0.0   # constant w → same as Linder2005
    D_l05 = toolscosmo.growth_factor_Linder2005(Z, param)
    D_lc7 = toolscosmo.growth_factor_LinderCahn2007(Z, param)
    assert np.allclose(D_l05, D_lc7, rtol=1e-4), \
        "LinderCahn2007 should equal Linder2005 when wa=0 (constant w)"

def test_lc2007_differs_from_linder2005_for_large_wa():
    # For large wa, γ(a) varies → LinderCahn2007 should differ from Linder2005
    param = toolscosmo.par(DE='CPL')
    param.cosmo.solver = 'toolscosmo'
    param.DE.w0 = -0.9
    param.DE.wa = 0.6   # large wa → γ(a) varies significantly
    D_l05 = toolscosmo.growth_factor_Linder2005(Z, param)
    D_lc7 = toolscosmo.growth_factor_LinderCahn2007(Z, param)
    assert not np.allclose(D_l05, D_lc7, rtol=1e-4), \
        "LinderCahn2007 should differ from Linder2005 for large wa"

def test_lc2007_vs_ode_cpl(param_cpl):
    # Both should agree with ODE to < 1% for typical CPL
    D_ode = toolscosmo.growth_factor_solveODE(Z, param_cpl)
    D_lc7 = toolscosmo.growth_factor_LinderCahn2007(Z, param_cpl)
    rel_err = np.abs(D_lc7 - D_ode) / D_ode
    assert np.all(rel_err < 0.01), \
        f"LinderCahn2007 vs ODE (CPL) max error: {rel_err.max():.4f}"
