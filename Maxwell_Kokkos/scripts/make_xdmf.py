#!/usr/bin/env python3
"""Generate an XDMF sidecar for Maxwell_Penalty HDF5 output.

Reads the per-rank ``3D_rank_*.h5`` files in a run directory and writes a
single ``.xmf`` file that describes the time series, the global uniform
grid geometry, and the per-rank data slabs. ParaView, VisIt, and PyVista
(via meshio or the VTK Python bindings) can open the ``.xmf`` directly
and show the assembled simulation without any data conversion.

The HDF5 files are unchanged. The XDMF is purely metadata — a manifest
that describes where each rank's owned data block lives in the global
grid and which slab of the local HDF5 dataset to read.

Usage:
    python make_xdmf.py                    # writes maxwell.xmf in cwd
    python make_xdmf.py --dir RUN_DIR
    python make_xdmf.py --dt-per-output 1.6

The ``--dt-per-output`` flag gives the physical time between successive
HDF5 output groups (i.e., ``output_every * cfl * min(dx,dy,dz)``). If
omitted, the XMF uses the integer output index as its time value, which
is still enough for ParaView's time-slider to step through the run.

Open the result in ParaView: File -> Open -> maxwell.xmf. The Xdmf3
reader is the preferred one (the legacy Xdmf2 reader also works). The
same file opens in VisIt and in PyVista via ``pyvista.XdmfReader``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

import h5py


def read_rank_metadata(path: Path) -> dict:
    """Read the /metadata attributes and iteration group keys from one file."""
    with h5py.File(path, "r") as f:
        m = f["metadata"].attrs
        md = {
            "gs":        int(m["ghost_zones"][0]),
            "mpi_size":  int(m["mpi_size"][0]),
            "n_evol":    int(m["n_evol_vars"][0]),
            "n_aux":     int(m["n_aux_vars"][0]),
            "local_ni":  int(m["local_ni"][0]),
            "local_nj":  int(m["local_nj"][0]),
            "local_nk":  int(m["local_nk"][0]),
            "local_i0":  int(m["local_i0"][0]),
            "local_j0":  int(m["local_j0"][0]),
            "local_k0":  int(m["local_k0"][0]),
            "global_ni": int(m["global_ni"][0]),
            "global_nj": int(m["global_nj"][0]),
            "global_nk": int(m["global_nk"][0]),
            "global_x0": float(m["global_x0"][0]),
            "global_y0": float(m["global_y0"][0]),
            "global_z0": float(m["global_z0"][0]),
            "dx":        float(m["dx"][0]),
            "dy":        float(m["dy"][0]),
            "dz":        float(m["dz"][0]),
        }
        names = []
        for i in range(md["n_evol"]):
            names.append(m[f"var_{i}"][0].decode())
        for i in range(md["n_aux"]):
            names.append(m[f"aux_{i}"][0].decode())
        md["var_names"] = names
        # Timestep groups are integer-named at the top level.
        md["iterations"] = sorted(int(k) for k in f.keys() if k.isdigit())
    return md


def visual_slab_1d(local_0: int, local_n: int, global_n: int, gs: int):
    """Return (start_local, count_local, start_global) for the visual block.

    Includes one ghost point past the owned region on the *upper* side of
    every MPI boundary; strict owned start on the lower side. This makes
    adjacent ranks share exactly one point at each internal boundary, so
    the rendered cells tile the global grid without a one-cell gap.
    Sides that coincide with the global domain edge are not extended.

    A side is a physical-boundary side iff that side of this rank's
    coverage coincides with the global domain edge:
      - lower physical: local_0 == 0
      - upper physical: local_0 + local_n == global_n
    All other sides are MPI (or periodic-wrap) neighbours.
    """
    lower_is_mpi = (local_0 != 0)
    upper_is_mpi = (local_0 + local_n) != global_n

    start_local  = gs if lower_is_mpi else 0
    end_local    = (local_n - gs + 1) if upper_is_mpi else local_n
    count_local  = end_local - start_local
    start_global = local_0 + start_local
    return start_local, count_local, start_global


def build_xdmf(rank_files: dict[int, Path], dt: float | None) -> Element:
    """Build the XDMF XML tree for a run whose per-rank HDF5 files are given
    by ``rank_files = {rank: Path(...)}``. ``dt`` is the physical time
    between successive HDF5 output groups (or ``None`` to use the output
    index as the time value)."""
    rank_meta = {r: read_rank_metadata(p) for r, p in rank_files.items()}
    first     = next(iter(rank_meta.values()))
    var_names = first["var_names"]
    # Use the intersection of iteration sets — ranks should all have the
    # same output cadence but be robust against partial runs.
    iter_sets = [set(md["iterations"]) for md in rank_meta.values()]
    iterations = sorted(set.intersection(*iter_sets)) if iter_sets else []

    xdmf   = Element("Xdmf", Version="3.0")
    domain = SubElement(xdmf, "Domain")
    series = SubElement(
        domain, "Grid",
        Name="MaxwellTimeSeries",
        GridType="Collection",
        CollectionType="Temporal",
    )

    for it in iterations:
        t_value = it * dt if dt is not None else float(it)
        step = SubElement(
            series, "Grid",
            Name=f"step_{it}",
            GridType="Collection",
            CollectionType="Spatial",
        )
        SubElement(step, "Time", Value=repr(t_value))

        for rank in sorted(rank_meta):
            rmd = rank_meta[rank]
            gs  = rmd["gs"]
            lo_i, sz_i, gi0 = visual_slab_1d(rmd["local_i0"], rmd["local_ni"],
                                            rmd["global_ni"], gs)
            lo_j, sz_j, gj0 = visual_slab_1d(rmd["local_j0"], rmd["local_nj"],
                                            rmd["global_nj"], gs)
            lo_k, sz_k, gk0 = visual_slab_1d(rmd["local_k0"], rmd["local_nk"],
                                            rmd["global_nk"], gs)
            if sz_i <= 0 or sz_j <= 0 or sz_k <= 0:
                continue

            origin_x = rmd["global_x0"] + gi0 * rmd["dx"]
            origin_y = rmd["global_y0"] + gj0 * rmd["dy"]
            origin_z = rmd["global_z0"] + gk0 * rmd["dz"]

            rank_grid = SubElement(
                step, "Grid",
                Name=f"rank{rank}",
                GridType="Uniform",
            )
            # 3DRectMesh + VXVYVZ is the unambiguous way to describe a
            # rectilinear grid: each axis gets an explicit coordinate list
            # whose role (x, y, or z) is fixed by the DataItem order. This
            # avoids the ORIGIN_DXDYDZ ambiguity between VisIt (which reads
            # spatial X-Y-Z) and the ParaView Xdmf3 reader (which reads the
            # same tuple as matching Topology Dimensions, Z-Y-X).
            SubElement(
                rank_grid, "Topology",
                TopologyType="3DRectMesh",
                Dimensions=f"{sz_k} {sz_j} {sz_i}",
            )
            x_coords = [origin_x + ii * rmd["dx"] for ii in range(sz_i)]
            y_coords = [origin_y + jj * rmd["dy"] for jj in range(sz_j)]
            z_coords = [origin_z + kk * rmd["dz"] for kk in range(sz_k)]

            geom = SubElement(
                rank_grid, "Geometry",
                GeometryType="VXVYVZ",
            )
            SubElement(
                geom, "DataItem",
                Dimensions=str(sz_i), NumberType="Float",
                Precision="8", Format="XML",
            ).text = " ".join(repr(v) for v in x_coords)
            SubElement(
                geom, "DataItem",
                Dimensions=str(sz_j), NumberType="Float",
                Precision="8", Format="XML",
            ).text = " ".join(repr(v) for v in y_coords)
            SubElement(
                geom, "DataItem",
                Dimensions=str(sz_k), NumberType="Float",
                Precision="8", Format="XML",
            ).text = " ".join(repr(v) for v in z_coords)

            h5_ref = rank_files[rank].name  # path relative to the XMF
            for vname in var_names:
                attr = SubElement(
                    rank_grid, "Attribute",
                    Name=vname,
                    AttributeType="Scalar",
                    Center="Node",
                )
                hs = SubElement(
                    attr, "DataItem",
                    ItemType="HyperSlab",
                    Dimensions=f"{sz_k} {sz_j} {sz_i}",
                    Type="HyperSlab",
                )
                # start / stride / count, in (k, j, i) order
                SubElement(
                    hs, "DataItem",
                    Dimensions="3 3", Format="XML",
                ).text = (f"{lo_k} {lo_j} {lo_i} "
                          f"1 1 1 "
                          f"{sz_k} {sz_j} {sz_i}")
                SubElement(
                    hs, "DataItem",
                    Format="HDF",
                    Dimensions=f"{rmd['local_nk']} {rmd['local_nj']} "
                               f"{rmd['local_ni']}",
                    NumberType="Float", Precision="8",
                ).text = f"{h5_ref}:/{it}/{vname}"

    return xdmf


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write an XDMF sidecar for Maxwell_Penalty HDF5 output.")
    ap.add_argument("--dir", default=".",
                    help="Directory containing 3D_rank_*.h5 files "
                         "(default: current directory)")
    ap.add_argument("--output", default=None,
                    help="Output XMF path (default: <dir>/maxwell.xmf)")
    ap.add_argument("--dt-per-output", "--dt", dest="dt",
                    type=float, default=None,
                    help="Physical time between successive HDF5 output "
                         "groups (output_every * cfl * min(dx,dy,dz)). If "
                         "given, XMF time values are output_index * dt; "
                         "otherwise the output index itself is used as "
                         "the time.")
    args = ap.parse_args()

    run_dir = Path(args.dir).resolve()
    out_path = Path(args.output) if args.output else run_dir / "maxwell.xmf"

    rank_files: dict[int, Path] = {}
    for p in sorted(run_dir.glob("3D_rank_*.h5")):
        try:
            rank = int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        rank_files[rank] = p
    if not rank_files:
        print(f"no 3D_rank_*.h5 files found in {run_dir}", file=sys.stderr)
        return 1

    xdmf = build_xdmf(rank_files, args.dt)

    # Write with the XMF file in the same directory as the HDF5 files so the
    # relative paths in the XMF resolve correctly.
    tree = ElementTree(xdmf)
    indent(tree, space="  ")
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out_path} "
          f"({len(rank_files)} rank{'s' if len(rank_files) != 1 else ''}, "
          f"{len(xdmf[0][0])} timestep"
          f"{'s' if len(xdmf[0][0]) != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
