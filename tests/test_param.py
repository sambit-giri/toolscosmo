import numpy as np
import pytest
import toolscosmo
from toolscosmo.param import Bunch, cosmo_par, code_par, de_par, dm_par, mf_par


# --- Bunch ---

def test_bunch_attribute_access():
    b = Bunch({'x': 1, 'y': 2})
    assert b.x == 1 and b.y == 2

def test_bunch_update_scalar():
    b = Bunch({'a': 1, 'b': 2})
    b.update({'b': 99, 'c': 3})
    assert b.b == 99
    assert b.c == 3
    assert b.a == 1  # unchanged

def test_bunch_nested_update():
    inner = Bunch({'x': 1, 'y': 2})
    outer = Bunch({'sub': inner, 'z': 0})
    outer.update({'sub': {'x': 99}})
    assert outer.sub.x == 99
    assert outer.sub.y == 2  # untouched

def test_bunch_update_new_key():
    b = Bunch({'a': 1})
    b.update({'new': 42})
    assert b.new == 42


# --- cosmo_par defaults ---

def test_cosmo_par_Om():
    p = cosmo_par()
    assert np.isclose(p.Om, 0.315)

def test_cosmo_par_h0():
    p = cosmo_par()
    assert np.isclose(p.h0, 0.673)

def test_cosmo_par_Ode_none():
    # None signals flat universe assumption
    p = cosmo_par()
    assert p.Ode is None

def test_cosmo_par_solver():
    p = cosmo_par()
    assert p.solver == 'toolscosmo'


# --- code_par defaults ---

def test_code_par_default_solver():
    p = code_par()
    assert p.Dz_solver == 'Linder2005'

def test_code_par_solver_options_comment():
    # Verify the default is one of the documented options
    p = code_par()
    valid = ['solveODE', 'Hamilton2000', 'Linder2005', 'CPT']
    assert p.Dz_solver in valid


# --- de_par ---

def test_de_par_lcdm():
    p = de_par('LCDM')
    assert p.name == 'LCDM'

def test_de_par_lcdm_case_insensitive():
    p = de_par('lcdm')
    assert p.name == 'lcdm'

def test_de_par_wcdm_has_w():
    p = de_par('WCDM')
    assert hasattr(p, 'w')
    assert np.isclose(p.w, -1.0)

def test_de_par_cpl_has_w0_wa():
    p = de_par('CPL')
    assert hasattr(p, 'w0') and hasattr(p, 'wa')
    assert np.isclose(p.w0, -1.0)
    assert np.isclose(p.wa, 0.0)

def test_de_par_unknown_has_wDE():
    p = de_par('some_model')
    assert hasattr(p, 'wDE')


# --- dm_par ---

def test_dm_par_lcdm():
    p = dm_par('LCDM')
    assert p.name == 'LCDM'

def test_dm_par_wdm_has_mass():
    p = dm_par('WDM')
    assert hasattr(p, 'm_wdm')

def test_dm_par_cwdm_has_mass_and_fraction():
    p = dm_par('CWDM')
    assert hasattr(p, 'm_wdm') and hasattr(p, 'f_wdm')


# --- mf_par ---

def test_mf_par_window():
    p = mf_par()
    assert p.window == 'tophat'

def test_mf_par_dc():
    p = mf_par()
    assert np.isclose(p.dc, 1.675)


# --- par() top-level ---

def test_par_has_all_keys():
    p = toolscosmo.par()
    for key in ['cosmo', 'code', 'DE', 'DM', 'mf', 'file', 'sfe', 'lyal', 'xray', 'reio']:
        assert hasattr(p, key), f"Missing key: {key}"

def test_par_de_lcdm():
    p = toolscosmo.par(DE='LCDM')
    assert p.DE.name == 'LCDM'

def test_par_de_cpl():
    p = toolscosmo.par(DE='CPL')
    assert hasattr(p.DE, 'w0') and hasattr(p.DE, 'wa')

def test_par_dm_wdm():
    p = toolscosmo.par(DM='WDM')
    assert hasattr(p.DM, 'm_wdm')

def test_par_flat_universe():
    # Default par() has Ode=None, meaning flat universe
    p = toolscosmo.par()
    assert p.cosmo.Ode is None

def test_par_cosmo_solver_default():
    p = toolscosmo.par()
    assert p.cosmo.solver == 'toolscosmo'
