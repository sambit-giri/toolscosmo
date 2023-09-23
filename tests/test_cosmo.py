import numpy as np 
import tools_cosmo

def test_Tcmb():
	param = tools_cosmo.par()
	Tcmb = tools_cosmo.T_cmb(1,param)
	assert np.abs(Tcmb-2.72*2)<0.01