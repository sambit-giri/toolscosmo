import os
import numpy as np
import pytest
import toolscosmo
from toolscosmo.data_utils import read_sne_covariance, _DATA_DIR, _DATASETS


def _write_cov(path, C, n_header=None):
    n = C.shape[0] if n_header is None else n_header
    with open(path, 'w') as f:
        f.write(f"{n}\n")
        for v in C.ravel():
            f.write(f"{v:.8e}\n")


@pytest.fixture
def small_cov():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    return A @ A.T + 5 * np.eye(5)


def test_read_sne_covariance_roundtrip(tmp_path, small_cov):
    p = tmp_path / 'c.cov'
    _write_cov(p, small_cov)
    C = read_sne_covariance(str(p), n_expected=5)
    np.testing.assert_allclose(C, small_cov, rtol=1e-7)


def test_read_sne_covariance_truncated(tmp_path, small_cov):
    p = tmp_path / 'c.cov'
    _write_cov(p, small_cov, n_header=6)
    with pytest.raises(ValueError, match='truncated'):
        read_sne_covariance(str(p))


def test_read_sne_covariance_size_mismatch(tmp_path, small_cov):
    p = tmp_path / 'c.cov'
    _write_cov(p, small_cov)
    with pytest.raises(ValueError, match='data file has 4'):
        read_sne_covariance(str(p), n_expected=4)


def test_read_sne_covariance_asymmetric(tmp_path, small_cov):
    C = small_cov.copy()
    C[0, 1] += 1.0
    p = tmp_path / 'c.cov'
    _write_cov(p, C)
    with pytest.raises(ValueError, match='symmetric'):
        read_sne_covariance(str(p))


def test_cov_only_for_pantheon_plus():
    with pytest.raises(NotImplementedError):
        toolscosmo.load_sne_data('pantheon', cov=True)


def test_pantheon_plus_loads_without_cov():
    sne = toolscosmo.load_sne_data('pantheon+')
    assert len(sne['z']) == 1701
    assert 'cov' not in sne


_COV_PATH = os.path.join(_DATA_DIR, _DATASETS['pantheon+_cov']['filename'])

@pytest.mark.skipif(not os.path.exists(_COV_PATH),
                    reason='Pantheon+ covariance not downloaded (run load_sne_data("pantheon+", cov=True) once)')
def test_pantheon_plus_cov():
    sne = toolscosmo.load_sne_data('pantheon+', cov=True)
    C = sne['cov']
    assert C.shape == (1701, 1701)
    assert np.all(np.diag(C) > 0)
    # Positive definite (needed for the likelihood).
    np.linalg.cholesky(C)
