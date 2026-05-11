"""Helpers for reading the per-rank HDF5 output produced by output_*_gf.

Each rank writes one file `rank_<R>.h5` containing:
  /metadata           (group with grid attributes — see io.c)
  /Var0, /Var1, ...   (per-variable datasets, ny*nx in 2D or nz*ny*nx in 3D)

`load_rank` returns a dict whose keys mirror the legacy JSON field names
the verifier scripts already use (`nx`, `dx`, `local_i0`, `lower_x_ghost`,
`data`, etc.) so each verifier needs only a one-line change at the I/O
boundary.  Scalar attributes are de-arrayed (HDF5 stores them as
1-element arrays); booleans come back as Python `bool` so the verifier's
existing `if lower_x_ghost:` reads work unchanged.
"""
from __future__ import annotations

import h5py
import numpy as np


def _scalar(attr):
    """Unwrap a single-element HDF5 attribute to a plain Python scalar."""
    a = np.asarray(attr)
    if a.shape == (1,):
        return a[0].item() if hasattr(a[0], "item") else a[0]
    return a.item() if hasattr(a, "item") else a


def load_rank(rank: int, varname: str = "Var0", *, directory: str = ".") -> dict:
    """Read the per-rank file and return a dict with the legacy field names.

    `varname` selects which dataset in the file to load into `data`.
    Missing metadata fields (e.g. z-axis fields on a 2D file) are
    represented exactly the way the old `d.get(name, default)` calls
    expected: `nz`, `dz`, `z0`, `local_k0`, and the global z fields
    are absent rather than zero, so verifiers can still use `.get()`
    to test 2D vs 3D.
    """
    path = f"{directory}/rank_{rank}.h5"
    with h5py.File(path, "r") as f:
        m = f["metadata"].attrs
        d: dict = {name: _scalar(m[name]) for name in m.keys()}
        # h5py returns ghost / Neumann flags as numpy ints (0/1); turn
        # them into Python bools so existing `if d["lower_x_ghost"]:`
        # branches read the same way they did under JSON.
        for k, v in list(d.items()):
            if k.endswith("_ghost") or k.startswith("neumann_"):
                d[k] = bool(v)
        d["data"] = f[varname][:]
    return d
