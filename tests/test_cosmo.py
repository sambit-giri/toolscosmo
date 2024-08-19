import numpy as np 
import toolscosmo

def test_Tcmb():
	param = toolscosmo.par()
	Tcmb = toolscosmo.T_cmb(1,param)
	assert np.abs(Tcmb-2.72*2)<0.01