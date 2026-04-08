"""
Particle-to-mesh mass assignment schemes.

Implements NGP, CIC, TSC, PCS in pure NumPy.  No compiled extensions required —
works on any platform including Apple Silicon without patching.

Optional acceleration (tried in order, install via pip install toolscosmo[...]):
  1. toolscosmo[numba]   → beorn[numba]  — Numba-JIT loops (10–50× faster than NumPy)
  2. toolscosmo[pylians] → pylians3      — Fortran/OpenMP backend (fastest for very large N,
                                           but can be tricky to install on some platforms)
  3. built-in NumPy fallback             — always available, no extra dependencies

Kernel weights (1-D, applied separably in x, y, z)
---------------------------------------------------
Let d = fractional distance from particle to cell centre (grid units).

NGP  (stencil 1):  W = 1
CIC  (stencil 2):  W(d) = 1 - |d|                              for |d| < 1
TSC  (stencil 3):  W(d) = 3/4 - d²                             for |d| < 1/2
                   W(d) = 1/2 (3/2 - |d|)²                     for 1/2 ≤ |d| < 3/2
PCS  (stencil 4):  W(d) = (4 - 6d² + 3|d|³) / 6               for |d| < 1
                   W(d) = (2 - |d|)³ / 6                        for 1 ≤ |d| < 2
"""

import numpy as np

_SCHEMES = ('NGP', 'CIC', 'TSC', 'PCS')


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _w_tsc(d):
    ad = np.abs(d)
    return np.where(ad < 0.5, 0.75 - d * d,
           np.where(ad < 1.5, 0.5 * (1.5 - ad) ** 2, 0.0)).astype(np.float32)


def _w_pcs(d):
    ad = np.abs(d)
    return np.where(ad < 1.0, (4.0 - 6.0 * d * d + 3.0 * ad ** 3) / 6.0,
           np.where(ad < 2.0, (2.0 - ad) ** 3 / 6.0, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-scheme painters
# ---------------------------------------------------------------------------

def _ngp(mesh, N, pos, w):
    ix = np.round(pos[:, 0]).astype(np.int32) % N
    iy = np.round(pos[:, 1]).astype(np.int32) % N
    iz = np.round(pos[:, 2]).astype(np.int32) % N
    np.add.at(mesh, (ix, iy, iz), w)


def _cic(mesh, N, pos, w):
    i0 = np.floor(pos).astype(np.int32)
    d1 = (pos - i0).astype(np.float32)
    d0 = 1.0 - d1
    i0 %= N
    i1  = (i0 + 1) % N
    for cx, wx in ((i0[:, 0], d0[:, 0]), (i1[:, 0], d1[:, 0])):
        for cy, wy in ((i0[:, 1], d0[:, 1]), (i1[:, 1], d1[:, 1])):
            for cz, wz in ((i0[:, 2], d0[:, 2]), (i1[:, 2], d1[:, 2])):
                np.add.at(mesh, (cx, cy, cz), w * wx * wy * wz)


def _tsc(mesh, N, pos, w):
    i_cen = np.floor(pos).astype(np.int32)
    for kx in (-1, 0, 1):
        ix = (i_cen[:, 0] + kx) % N
        wx = _w_tsc(pos[:, 0] - (i_cen[:, 0] + kx))
        for ky in (-1, 0, 1):
            iy = (i_cen[:, 1] + ky) % N
            wy = _w_tsc(pos[:, 1] - (i_cen[:, 1] + ky))
            for kz in (-1, 0, 1):
                iz = (i_cen[:, 2] + kz) % N
                wz = _w_tsc(pos[:, 2] - (i_cen[:, 2] + kz))
                np.add.at(mesh, (ix, iy, iz), w * wx * wy * wz)


def _pcs(mesh, N, pos, w):
    i_cen = np.floor(pos).astype(np.int32)
    for kx in (-1, 0, 1, 2):
        ix = (i_cen[:, 0] + kx) % N
        wx = _w_pcs(pos[:, 0] - (i_cen[:, 0] + kx))
        for ky in (-1, 0, 1, 2):
            iy = (i_cen[:, 1] + ky) % N
            wy = _w_pcs(pos[:, 1] - (i_cen[:, 1] + ky))
            for kz in (-1, 0, 1, 2):
                iz = (i_cen[:, 2] + kz) % N
                wz = _w_pcs(pos[:, 2] - (i_cen[:, 2] + kz))
                np.add.at(mesh, (ix, iy, iz), w * wx * wy * wz)


_NUMPY_FN = {'NGP': _ngp, 'CIC': _cic, 'TSC': _tsc, 'PCS': _pcs}

# Batch sizes to keep intermediate arrays below ~1 GB
_BATCH = {'NGP': 10_000_000, 'CIC': 10_000_000, 'TSC': 5_000_000, 'PCS': 3_000_000}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def assign_mass(mesh, box_size, positions, scheme='PCS', backend='auto', verbose=False):
    """
    Assign particle masses to a 3-D mesh in place.

    Parameters
    ----------
    mesh : np.ndarray, shape (N, N, N), dtype float32
        Target grid. Modified in place.
    box_size : float
        Side length of the simulation box (same units as positions).
    positions : np.ndarray, shape (n_parts, 3), dtype float32
        Particle positions. Must lie in [0, box_size).
    scheme : str
        Mass assignment kernel: 'NGP', 'CIC', 'TSC', or 'PCS'. Default 'PCS'.
    backend : str
        'auto'   — try numba (beorn) → numpy (built-in). Use 'pylians' explicitly if needed.
        'numpy'  — always use the built-in pure-NumPy implementation (default fallback)
        'numba'  — use beorn's Numba JIT backend (requires: pip install toolscosmo[numba])
        'pylians'— use pylians MAS_library    (requires: pip install toolscosmo[pylians])
    verbose : bool
        Print progress (passed through to pylians backend only).
    """
    scheme = scheme.upper()
    if scheme not in _SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {_SCHEMES}.")

    positions = np.ascontiguousarray(positions, dtype=np.float32)

    if backend == 'pylians':
        _assign_pylians(mesh, box_size, positions, scheme, verbose)
        return

    if backend in ('numba', 'auto'):
        try:
            from beorn.particle_mapping.core import map_particles_to_mesh
            map_particles_to_mesh(mesh, box_size, positions, mass_assignment=scheme, backend='numba')
            return
        except Exception:
            if backend == 'numba':
                raise

    if backend in ('auto',):
        try:
            from beorn.particle_mapping.core import map_particles_to_mesh
            map_particles_to_mesh(mesh, box_size, positions, mass_assignment=scheme, backend='numpy')
            return
        except Exception:
            pass

    # Pure-NumPy fallback (always works)
    _assign_numpy(mesh, box_size, positions, scheme)


def _assign_numpy(mesh, box_size, positions, scheme):
    N      = mesh.shape[0]
    scale  = N / box_size
    fn     = _NUMPY_FN[scheme]
    batch  = _BATCH[scheme]
    n_part = len(positions)
    w_ones = np.ones(min(batch, n_part), dtype=np.float32)
    for start in range(0, n_part, batch):
        end = min(start + batch, n_part)
        pos = positions[start:end] * scale
        w   = w_ones[:end - start]
        fn(mesh, N, pos, w)


def _assign_pylians(mesh, box_size, positions, scheme, verbose):
    try:
        import MAS_library as MASL
        MASL.MA(positions, mesh, box_size, scheme, verbose=verbose)
    except ImportError as e:
        raise ImportError(
            "pylians (MAS_library) is not installed. "
            "Install it with: pip install pylians3  or use backend='numpy'."
        ) from e
