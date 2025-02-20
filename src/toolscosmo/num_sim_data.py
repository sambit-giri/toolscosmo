import numpy as np
import os, pickle, pkg_resources

def hmf_wdm_data(name='Schneider2018'):
    if name.lower()=='schneider2018':
        filename = 'input_data/Schneider2018_wdm_hmf.pkl'
        path_to_file = pkg_resources.resource_filename('toolscosmo', filename)
        data = pickle.load(open(path_to_file,'rb'))
        print('The data contains HMF for cosmology with WDM mass of 6 keV.')
    else:
        print(f'{name} data is not available.')
        data = None
    return data