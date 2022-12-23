import numpy as np 
import GalaxyTools

def test_age_estimator():
	param = GalaxyTools.param()
	t0 = GalaxyTools.age_estimator(param, 0)
	assert np.abs(t0-13.74)<0.01