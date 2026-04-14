import numpy as np
import tools21cm as t2c
import os, pickle

def hmf_wdm_data(name='Schneider2018'):
    if name.lower()=='schneider2018':
        filename = 'input_data/Schneider2018_wdm_hmf.pkl'
        # path_to_file = pkg_resources.resource_filename('toolscosmo', filename)
        path_to_file = t2c.get_package_resource_path('toolscosmo', filename)
        data = pickle.load(open(path_to_file,'rb'))
        print('The data contains HMF for cosmology with WDM mass of 6 keV.')
    else:
        print(f'{name} data is not available.')
        data = None
    return data