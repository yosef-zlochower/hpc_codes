#!/bin/bash
# End-to-end smoke test for scripts/make_xdmf.py.
#
# Runs the driver on manufactured_mixed at np=2 (1x1x2 topology, so the
# z-axis is split across two ranks), invokes make_xdmf.py against the
# resulting rank_*.h5 files, and validates the XDMF sidecar:
#
#   1. The XMF file exists and parses as well-formed XML.
#   2. Every HDF5 dataset referenced by the XMF actually exists.
#   3. Reconstructing the global field from the per-rank HyperSlabs
#      yields the same total grid extent the driver was configured for
#      (no missing or duplicated points along internal MPI seams).

set -euo pipefail

# Resolve absolute paths to the driver and the top-level XMF generator
# BEFORE cd'ing into the per-test scratch directory.  The test script
# itself runs from <build>/src/tests/; driver_multigrid is one level up
# (in <build>/src/), and scripts/make_xdmf.py is at the project root
# (three levels up from <build>/src/tests/).
DRIVER="$(cd .. && pwd)/driver_multigrid"
MAKE_XDMF="$(cd ../../.. && pwd)/scripts/make_xdmf.py"

if [ ! -x "$DRIVER" ]; then
    echo "FAILURE: driver '$DRIVER' is missing or not executable." >&2
    exit 1
fi
if [ ! -f "$MAKE_XDMF" ]; then
    echo "FAILURE: '$MAKE_XDMF' not found." >&2
    exit 1
fi

WORKDIR="xdmf_smoke_run"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

cat > params.toml <<EOF
[grid]
nx_cells = 16
ny_cells = 16
nz_cells = 16

[solver]
multigrid = true
omega     = 1.5
n_smooth  = 2
n_iters   = 20
tol       = 1.0e-10
subcycles = 1
min_cells = 2

[problem]
name = "manufactured_mixed"

[output]
dir = "."
EOF

echo "--- driver (np=2) ---"
mpirun --map-by :OVERSUBSCRIBE -np 2 "$DRIVER" params.toml > driver.log 2>&1
tail -3 driver.log

echo "--- make_xdmf.py ---"
python3 "$MAKE_XDMF" --dir .
test -f multigrid.xmf
echo "wrote multigrid.xmf"

echo "--- validate XMF ---"
python3 <<'PY'
import sys
from pathlib import Path
from xml.etree.ElementTree import parse
import h5py

xmf = parse("multigrid.xmf").getroot()
assert xmf.tag == "Xdmf", f"unexpected root tag: {xmf.tag}"

# Every HyperSlab DataItem with Format="HDF" should resolve to an
# existing dataset in the named file.
hdf_refs = [
    di.text.strip()
    for di in xmf.iter("DataItem")
    if di.get("Format") == "HDF"
]
assert hdf_refs, "no HDF DataItems found in XMF"

for ref in hdf_refs:
    fname, dpath = ref.split(":")
    assert Path(fname).exists(), f"missing HDF5 file referenced by XMF: {fname}"
    with h5py.File(fname, "r") as f:
        assert dpath.lstrip("/") in f, f"missing dataset {dpath} in {fname}"

# Sanity-check the total number of unique (x,y,z) vertex coords across
# all rank tiles.  manufactured_mixed has all-Neumann upper faces (cell-
# centred, no upper vertex slot) and a homogeneous Dirichlet on lower-x
# (hybrid D-N: D vertex included on the x=0 end).  Likewise lower-y and
# lower-z are pure Neumann at homog q=0 -- actually mixed_neumann gives
# q=0 there, but the layout flags say neumann_lower_y/z = True.  So:
#   x-axis: 1 D vertex + N cells = N+1 visible points     (N=16 -> 17)
#   y-axis: N cell centres                                 (16)
#   z-axis: N cell centres                                 (16)
# The XMF concatenates per-rank slabs along the partitioned z-axis,
# with neighbouring ranks sharing exactly one point.  For np=2 with
# topology 1x1x2 the global counts after de-duplication are 17 x 16 x 16.
def _coord_list(elem):
    return [float(v) for v in elem.text.split()]

x_unique, y_unique, z_unique = set(), set(), set()
for grid in xmf.iter("Grid"):
    if grid.get("GridType") != "Uniform":
        continue
    geom = grid.find("Geometry")
    cs = geom.findall("DataItem")
    assert len(cs) == 3, "expected VXVYVZ with 3 DataItems"
    for v in _coord_list(cs[0]): x_unique.add(round(v, 12))
    for v in _coord_list(cs[1]): y_unique.add(round(v, 12))
    for v in _coord_list(cs[2]): z_unique.add(round(v, 12))

if (len(x_unique), len(y_unique), len(z_unique)) != (17, 16, 16):
    print(f"FAIL: unique vertex counts (x,y,z) = "
          f"({len(x_unique)}, {len(y_unique)}, {len(z_unique)}); "
          f"expected (17, 16, 16)", file=sys.stderr)
    sys.exit(1)

print("XMF references valid, vertex counts match expected layout")
PY

cd ..
rm -rf "$WORKDIR"
echo "make_xdmf smoke test passed"
