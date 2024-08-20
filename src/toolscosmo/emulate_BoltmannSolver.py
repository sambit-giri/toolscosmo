import numpy as np 
from scipy.interpolate import splrep,splev
from time import time 
import warnings
import os, pickle, pkg_resources

from .neural_networks import NNRegressor

def emulate_class(param, **info):
    tstart = time()
    if param.code.verbose: print('Using CLASSemu to estimate linear power spectrum.')

    # Cosmological parameters
    h0  = param.cosmo.h0
    Omc = param.cosmo.Om - param.cosmo.Ob
    Omb = param.cosmo.Ob 
    Omk = 0.0 #param.cosmo.Ok # Assuming flat cosmology
    mnu = param.cosmo.mnu
    ns = param.cosmo.ns 
    As = param.cosmo.As 
    log10As = np.log10(As)
    if As is None and param.cosmo.s8:
        print(f'Provide As as normalisation with sigma8 not implemented.')
        return None

    # Wavenumbers [h/Mpc]
    k_max = param.code.kmax

    fn_wa = lambda w0,wb: -w0-wb**4
    fn_wb = lambda w0,wa: (-w0-wa)**(1/4)

    if param.DE.name.lower() in ['lcdm']:
        pass
    elif param.DE.name.lower() in ['cpl', 'w0wa']:
        w0 = param.DE.w0 
        wa = param.DE.wa 
        path_to_class_cpl_emu = pkg_resources.resource_filename('toolscosmo', 'input_data/cpl_class_emulator.pkl')
        emu = NNRegressor(layers=[7, 128, 256, 128, 32])
        emu.load_model(path_to_class_cpl_emu)
        pca = emu.extra_data['pca_y']
        k = emu.extra_data['k']
        X = np.array([Omc, Omb, h0, log10As, ns, w0, fn_wb(w0,wa)])
        Y_pred = emu.predict(X)
        y_pred = 10**emu.PCA_inverse_transform_data(Y_pred, pca)
        if param.code.verbose: print(f'{param.DE.name}: w0,wa={w0},{wa}')
    else:
        print(f'{param.DE.name} is an unknown dark energy model for CAMB.')
    
    if param.code.verbose: 
        # print(f'sigma_8={r.get_sigma8_0():.3f}')
        print('CLASSemu runtime: {:.2f} s'.format(time()-tstart))
    out = {'k': k.squeeze(), 'P': y_pred.squeeze()}
    return out