#!/usr/bin/env python3
"""Generate an XDMF sidecar for Multigrid HDF5 output.

Reads the per-rank `rank_*.h5` files in a run directory and writes a
single `multigrid.xmf` that describes the assembled global grid and the
per-rank data slabs.  ParaView, VisIt, and PyVista open the `.xmf`
directly and show the assembled field without any data conversion.

The HDF5 files are unchanged.  The XMF is purely a manifest: each
rank's owned region appears as a Uniform sub-grid whose `Geometry`
gives explicit per-axis vertex coordinates (`VXVYVZ`) and whose
`Attribute` points at a HyperSlab of the rank's local dataset.

Usage:
    python make_xdmf.py                    # writes multigrid.xmf in cwd
    python make_xdmf.py --dir RUN_DIR      # specify run directory
    python make_xdmf.py --output PATH      # explicit output path

For the Multigrid cell-centred layout the script does the right thing
on all four axis configurations:
  * pure D-D  : N+1 vertex-centred slots, all rendered.
  * pure N-N  : N cell-centred slots, both N ghosts skipped.
  * hybrid D-N: D vertex (1 slot) + N cell centres rendered;
                upper-N ghost skipped.  The first inter-vertex gap is
                h/2 (vertex to first cell center), all subsequent gaps
                are h (cell to cell).
  * hybrid N-D: mirror of D-N (N ghost on lower, D vertex on upper).

Internal MPI boundaries: the lower side starts strictly inside the
owned region (skips `gs` ghost slots).  The upper side extends one
slot past the owned region so adjacent ranks share exactly one point
at each internal interface (no visual gap between rank tiles).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

import h5py
import numpy as np


def _scalar(attr):
    """Unwrap a 1-element HDF5 attribute to a plain Python scalar."""
    a = np.asarray(attr)
    if a.shape == (1,):
        return a[0].item() if hasattr(a[0], "item") else a[0]
    return a.item() if hasattr(a, "item") else a


def read_rank_metadata(path: Path) -> dict:
    """Read the /metadata group attributes and list dataset names."""
    with h5py.File(path, "r") as f:
        m = f["metadata"].attrs
        md = {name: _scalar(m[name]) for name in m.keys()}
        # boolean flags as Python bool for clean if-tests below
        for k in list(md):
            if k.endswith("_ghost") or k.startswith("neumann_"):
                md[k] = bool(md[k])
        # collect dataset names at the file root (everything except /metadata)
        md["var_names"] = sorted(
            k for k in f.keys() if k != "metadata" and isinstance(f[k], h5py.Dataset)
        )
        md["local_dim"] = tuple(int(f[md["var_names"][0]].shape[d])
                                for d in range(f[md["var_names"][0]].ndim))
    return md


def _axis_slab(local_n: int, local_0: int, global_n_cells: int, gs: int,
               x0_shifted: float, dx: float,
               neumann_lower: bool, neumann_upper: bool,
               has_neighbour_lower: bool, has_neighbour_upper: bool):
    """Return (slot_lo, slot_hi, coords) for one axis on one rank.

    `slot_lo:slot_hi` is the half-open range of LOCAL array indices to
    include in the visualisation slab.  `coords` is the matching 1-D
    list of physical vertex coordinates (same length as the slab).

    Decisions per side:
      * MPI-internal side  -> skip `gs` slots; on the upper side also
        extend one slot past the owned region so neighbours share a
        point exactly.
      * Pure-D physical    -> include the boundary slot (it's a real
        Dirichlet vertex value).
      * Pure-N physical    -> skip the boundary slot (it's a mirror
        ghost cell whose coordinate is h/2 outside the domain).
      * Hybrid D end       -> include the boundary slot (it's a D
        vertex placed AT the boundary; its physical coord is offset
        by h/2 from what the standard cell-centred formula predicts).
      * Hybrid N end       -> skip the boundary slot (mirror ghost).
    """
    is_cc = neumann_lower or neumann_upper      # axis has any Neumann face
    x0_user = x0_shifted + (0.5 * dx if is_cc else 0.0)
    global_upper = x0_user + global_n_cells * dx

    # Lower side
    if has_neighbour_lower:
        slot_lo = gs
    elif neumann_lower:                 # pure-N or hybrid lower-N: skip ghost
        slot_lo = 1
    else:                               # pure-D vertex or hybrid lower-D: include
        slot_lo = 0

    # Upper side (slot_hi is exclusive)
    if has_neighbour_upper:
        slot_hi = local_n - gs + 1      # share one point with the neighbour
    elif neumann_upper:                 # pure-N or hybrid upper-N: skip ghost
        slot_hi = local_n - 1
    else:                               # pure-D vertex or hybrid upper-D: include
        slot_hi = local_n

    # Coordinates for each included slot.  The standard mapping
    # x = x0_shifted + i*dx is exact for every regular cell centre and
    # for every pure-D vertex; the two special cases are the hybrid D
    # boundary slots, where the formula is off by h/2 because the slot
    # holds the value AT the boundary (not at the formula coordinate).
    coords = []
    for i in range(slot_lo, slot_hi):
        is_lower_d_vertex = (i == 0 and not has_neighbour_lower
                             and not neumann_lower and is_cc)
        is_upper_d_vertex = (i == local_n - 1 and not has_neighbour_upper
                             and not neumann_upper and is_cc)
        if is_lower_d_vertex:
            coords.append(x0_user)
        elif is_upper_d_vertex:
            coords.append(global_upper)
        else:
            coords.append(x0_shifted + i * dx)

    return slot_lo, slot_hi, coords


def build_xdmf(rank_files: dict[int, Path]) -> Element:
    """Build the XDMF XML tree for a run whose per-rank files are given."""
    rank_meta = {r: read_rank_metadata(p) for r, p in rank_files.items()}
    first = next(iter(rank_meta.values()))
    var_names = first["var_names"]

    xdmf = Element("Xdmf", Version="3.0")
    domain = SubElement(xdmf, "Domain")
    step = SubElement(domain, "Grid",
                      Name="MultigridSolution",
                      GridType="Collection",
                      CollectionType="Spatial")

    for rank in sorted(rank_meta):
        rmd = rank_meta[rank]
        gs = int(rmd["gs"])

        lo_i, hi_i, x_coords = _axis_slab(
            int(rmd["nx"]), int(rmd["local_i0"]),
            int(rmd["global_cells_x"]), gs,
            float(rmd["x0"]), float(rmd["dx"]),
            rmd["neumann_lower_x"], rmd["neumann_upper_x"],
            rmd["lower_x_ghost"], rmd["upper_x_ghost"])
        lo_j, hi_j, y_coords = _axis_slab(
            int(rmd["ny"]), int(rmd["local_j0"]),
            int(rmd["global_cells_y"]), gs,
            float(rmd["y0"]), float(rmd["dy"]),
            rmd["neumann_lower_y"], rmd["neumann_upper_y"],
            rmd["lower_y_ghost"], rmd["upper_y_ghost"])
        lo_k, hi_k, z_coords = _axis_slab(
            int(rmd["nz"]), int(rmd["local_k0"]),
            int(rmd["global_cells_z"]), gs,
            float(rmd["z0"]), float(rmd["dz"]),
            rmd["neumann_lower_z"], rmd["neumann_upper_z"],
            rmd["lower_z_ghost"], rmd["upper_z_ghost"])

        sz_i, sz_j, sz_k = hi_i - lo_i, hi_j - lo_j, hi_k - lo_k
        if sz_i <= 0 or sz_j <= 0 or sz_k <= 0:
            continue

        rank_grid = SubElement(step, "Grid",
                               Name=f"rank{rank}",
                               GridType="Uniform")

        # 3DRectMesh + VXVYVZ: explicit per-axis vertex coordinates;
        # ParaView/VisIt both parse this without geometry ambiguities.
        SubElement(rank_grid, "Topology",
                   TopologyType="3DRectMesh",
                   Dimensions=f"{sz_k} {sz_j} {sz_i}")
        geom = SubElement(rank_grid, "Geometry", GeometryType="VXVYVZ")
        for coords, n in ((x_coords, sz_i), (y_coords, sz_j), (z_coords, sz_k)):
            SubElement(geom, "DataItem",
                       Dimensions=str(n),
                       NumberType="Float", Precision="8",
                       Format="XML").text = " ".join(repr(v) for v in coords)

        h5_ref = rank_files[rank].name      # relative to the XMF
        local_nk = int(rmd["local_dim"][0])
        local_nj = int(rmd["local_dim"][1])
        local_ni = int(rmd["local_dim"][2])
        for vname in var_names:
            attr = SubElement(rank_grid, "Attribute",
                              Name=vname,
                              AttributeType="Scalar",
                              Center="Node")
            hs = SubElement(attr, "DataItem",
                            ItemType="HyperSlab",
                            Dimensions=f"{sz_k} {sz_j} {sz_i}",
                            Type="HyperSlab")
            # start / stride / count  in (k, j, i) order
            SubElement(hs, "DataItem",
                       Dimensions="3 3", Format="XML").text = (
                f"{lo_k} {lo_j} {lo_i} "
                f"1 1 1 "
                f"{sz_k} {sz_j} {sz_i}")
            SubElement(hs, "DataItem",
                       Format="HDF",
                       Dimensions=f"{local_nk} {local_nj} {local_ni}",
                       NumberType="Float", Precision="8").text = (
                f"{h5_ref}:/{vname}")

    return xdmf


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write an XDMF sidecar for Multigrid HDF5 output.")
    ap.add_argument("--dir", default=".",
                    help="Directory containing rank_*.h5 files "
                         "(default: current directory)")
    ap.add_argument("--output", default=None,
                    help="Output XMF path (default: <dir>/multigrid.xmf)")
    args = ap.parse_args()

    run_dir = Path(args.dir).resolve()
    out_path = Path(args.output) if args.output else run_dir / "multigrid.xmf"

    rank_files: dict[int, Path] = {}
    for p in sorted(run_dir.glob("rank_*.h5")):
        try:
            rank = int(p.stem.split("_", 1)[-1])
        except ValueError:
            continue
        rank_files[rank] = p
    if not rank_files:
        print(f"no rank_*.h5 files found in {run_dir}", file=sys.stderr)
        return 1

    xdmf = build_xdmf(rank_files)

    tree = ElementTree(xdmf)
    indent(tree, space="  ")
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out_path} ({len(rank_files)} rank"
          f"{'s' if len(rank_files) != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
