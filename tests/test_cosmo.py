import numpy as np 
import GalaxyTools

def test_Tcmb():
	param = GalaxyTools.par()
	Tcmb = GalaxyTools.T_cmb(1,param)
	assert np.abs(Tcmb-2.72*2)<0.01