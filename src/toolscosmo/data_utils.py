"""
Utilities for downloading and loading observational datasets.

Usage
-----
    from toolscosmo.data_utils import download_data, load_sne_data

    download_data('pantheon')          # one-time download
    sne = load_sne_data('pantheon')    # loads (downloads if needed)
"""

import os
import urllib.request
import numpy as np

# Directory where data files are stored alongside the package
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'input_data')

# ---------------------------------------------------------------------------
# Dataset registry
# Each entry: local_filename, list of candidate URLs (tried in order),
#             brief description, and column metadata.
# ---------------------------------------------------------------------------
_DATASETS = {
    # ---- SNe Ia ------------------------------------------------------------
    'pantheon': {
        'filename': 'pantheon_sne.txt',
        'urls': [
            'https://raw.githubusercontent.com/dscolnic/Pantheon/master/'
            'lcparam_full_long_zhel.txt',
        ],
        'description': 'Pantheon 2018 SNe Ia compilation (Scolnic et al. 2018) — 1048 SNe',
        'type': 'sne',
    },
    'pantheon+': {
        'filename': 'pantheon_plus_sne.dat',
        'urls': [
            'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/'
            'main/Pantheon%2B_Data/1_DATA/Pantheon+SH0ES.dat',
        ],
        'description': 'Pantheon+ 2022 SNe Ia compilation (Brout et al. 2022) — 1701 SNe',
        'type': 'sne',
    },
    'union2.1': {
        'filename': 'union2p1_sne.txt',
        'urls': [
            'https://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt',
        ],
        'description': 'Union 2.1 SNe Ia compilation (Suzuki et al. 2012) — 580 SNe',
        'type': 'sne',
    },
    
    # ---- UV Luminosity Functions (HST / JWST) ------------------------------
    'bouwens2015': {
        'filename': 'bouwens2015_uvlf.txt',
        'urls': [
            # Example placeholder URL -- replace with an actual raw repository link!
            'https://raw.githubusercontent.com/username/repo/main/bouwens2015_uvlf.txt',
        ],
        'description': 'HST UV Luminosity Functions z~4-10 (Bouwens+ 2015)',
        'type': 'lf',
    },
    'harikane2023': {
        'filename': 'harikane2023_jwst_uvlf.txt',
        'urls': [
            # Example placeholder URL -- replace with an actual raw repository link!
            'https://raw.githubusercontent.com/username/repo/main/harikane2023_jwst_uvlf.txt',
        ],
        'description': 'JWST Early UV Luminosity Functions z~9-17 (Harikane+ 2023)',
        'type': 'lf',
    },
}


def list_datasets():
    """Print all available datasets with descriptions."""
    print(f"{'Name':<14}  {'Type':<6}  Description")
    print('-' * 70)
    for name, meta in _DATASETS.items():
        local = os.path.join(_DATA_DIR, meta['filename'])
        cached = ' [cached]' if os.path.exists(local) else ''
        print(f"{name:<14}  {meta['type']:<6}  {meta['description']}{cached}")


def download_data(name, overwrite=False, dest_dir=None):
    """
    Download a known dataset into the package data directory.

    Parameters
    ----------
    name : str
        Dataset name (see ``list_datasets()``).
    overwrite : bool
        Re-download even if the file already exists.
    dest_dir : str or None
        Override destination directory (default: package input_data folder).

    Returns
    -------
    str
        Path to the downloaded file.
    """
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_DATASETS.keys())}"
        )
    meta = _DATASETS[name]
    dest = dest_dir or _DATA_DIR
    os.makedirs(dest, exist_ok=True)
    local_path = os.path.join(dest, meta['filename'])

    if os.path.exists(local_path) and not overwrite:
        print(f"'{name}' already cached at {local_path}. Use overwrite=True to re-download.")
        return local_path

    last_err = None
    for url in meta['urls']:
        try:
            print(f"Downloading '{name}' from {url} ...")
            urllib.request.urlretrieve(url, local_path)
            print(f"Saved to {local_path}")
            return local_path
        except Exception as e:
            last_err = e
            print(f"  Failed ({e}), trying next URL...")

    raise RuntimeError(
        f"Could not download '{name}'. Last error: {last_err}\n"
        f"You can manually place the file at {local_path}"
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sne_data(name='pantheon', **kwargs):
    """
    Load an SNe Ia dataset, downloading it first if not already cached.

    Parameters
    ----------
    name : str
        Dataset name: ``'pantheon'``, ``'pantheon+'``, or ``'union2.1'``.
    **kwargs
        Forwarded to ``numpy.genfromtxt``.

    Returns
    -------
    dict with keys:
        ``z``    — CMB-frame redshifts
        ``mu``   — distance moduli
        ``mu_err`` — 1-sigma uncertainties on mu
        ``raw``  — full structured array from the file
    """
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown SNe dataset '{name}'. Available: {list(_DATASETS.keys())}"
        )
    meta = _DATASETS[name]
    if meta['type'] != 'sne':
        raise ValueError(f"'{name}' is not an SNe dataset.")

    local_path = os.path.join(_DATA_DIR, meta['filename'])
    if not os.path.exists(local_path):
        download_data(name)

    if name == 'pantheon':
        return _load_pantheon(local_path, **kwargs)
    elif name == 'pantheon+':
        return _load_pantheon_plus(local_path, **kwargs)
    elif name == 'union2.1':
        return _load_union21(local_path, **kwargs)

def load_lf_data(name='bouwens2015', **kwargs):
    """
    Load a UV Luminosity Function dataset.

    Parameters
    ----------
    name : str
        Dataset name: e.g. ``'bouwens2015'`` or ``'harikane2023'``.
    **kwargs
        Forwarded to ``numpy.genfromtxt``.

    Returns
    -------
    dict with keys:
        ``z``      — Redshift bin
        ``Muv``    — Absolute UV magnitude
        ``phi``    — Volume density (Mpc^-3 mag^-1)
        ``err_up`` — Upper uncertainty on phi
        ``err_dn`` — Lower uncertainty on phi
    """
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown LF dataset '{name}'. Available: {[k for k, v in _DATASETS.items() if v['type'] == 'lf']}"
        )
    meta = _DATASETS[name]
    if meta['type'] != 'lf':
        raise ValueError(f"'{name}' is not an LF dataset.")

    local_path = os.path.join(_DATA_DIR, meta['filename'])
    if not os.path.exists(local_path):
        # We try to download, but if the URL is a placeholder it will crash.
        # This will instruct the user to correctly place the file locally.
        download_data(name)

    kwargs.setdefault('names', ('z', 'Muv', 'phi', 'err_up', 'err_dn'))
    kwargs.setdefault('comments', '#')
    kwargs.setdefault('dtype', float)
    
    data = np.genfromtxt(local_path, **kwargs)
    return {
        'z': data['z'],
        'Muv': data['Muv'],
        'phi': data['phi'],
        'err_up': data['err_up'],
        'err_dn': data['err_dn'],
        'raw': data
    }


# --- format-specific loaders -----------------------------------------------

def _load_pantheon(path, **kwargs):
    """
    Pantheon 2018 (lcparam_full_long_zhel.txt).
    Columns: name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar
             cov_m_s cov_m_c cov_s_c set ra dec
    mu is not given directly; mb is the corrected apparent B magnitude.
    We return mb as mu (absolute calibration is left to the user).
    """
    kwargs.setdefault('names', True)
    kwargs.setdefault('comments', '#')
    kwargs.setdefault('usecols', range(18))
    data = np.genfromtxt(path, **kwargs)
    return {
        'z'      : data['zcmb'],
        'z_hel'  : data['zhel'],
        'mu'     : data['mb'],       # corrected apparent magnitude (add -M_B)
        'mu_err' : data['dmb'],
        'raw'    : data,
    }


def _load_pantheon_plus(path, **kwargs):
    """
    Pantheon+ 2022 (Pantheon+SH0ES.dat).
    Relevant columns: zHD (CMB redshift), MU_SH0ES, MU_SH0ES_ERR_DIAG
    """
    kwargs.setdefault('names', True)
    kwargs.setdefault('comments', '#')
    data = np.genfromtxt(path, **kwargs)
    return {
        'z'      : data['zHD'],
        'mu'     : data['MU_SH0ES'],
        'mu_err' : data['MU_SH0ES_ERR_DIAG'],
        'raw'    : data,
    }


def _load_union21(path, **kwargs):
    """
    Union 2.1 (SCPUnion2.1_mu_vs_z.txt).
    Columns: name z mu sigma_mu P_stretch P_color
    """
    kwargs.setdefault('names', ('name', 'z', 'mu', 'mu_err', 'P_stretch', 'P_color'))
    kwargs.setdefault('comments', '#')
    kwargs.setdefault('dtype', None)
    data = np.genfromtxt(path, **kwargs)
    return {
        'z'      : data['z'].astype(float),
        'mu'     : data['mu'].astype(float),
        'mu_err' : data['mu_err'].astype(float),
        'raw'    : data,
    }
