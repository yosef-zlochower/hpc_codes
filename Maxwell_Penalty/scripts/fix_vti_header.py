#!/usr/bin/env python3
"""Patch np.float64(...) wrappers out of the header of existing VTI files.

The bug (now fixed in hdf5_to_vtk.py) wrote things like
    Origin="np.float64(-3.0) np.float64(-3.0) np.float64(0.0) "
because numpy >= 2.0 changed str(np.float64(x)) to include the type tag.
ParaView/VisIt reject those attributes.

This script edits only the first HEADER_BYTES of each file (the XML header,
typically a few hundred bytes). Each np.float64(x) is replaced with x padded
to the original match length, so the byte length is preserved and the
subsequent DataArray payload (potentially gigabytes) is never touched or
rewritten. VTK tolerates extra whitespace inside attribute values.

Usage:
    python fix_vti_header.py image*.vti
"""
import re
import sys
from pathlib import Path

HEADER_BYTES = 16384  # generous buffer; real headers are a few hundred bytes

PAT = re.compile(rb"np\.float64\(([^)]*)\)")


def fix(path: Path) -> str:
    with path.open("r+b") as f:
        head = f.read(HEADER_BYTES)
        if b"np.float64" not in head:
            return "unchanged"

        def sub(m: re.Match) -> bytes:
            orig = m.group(0)
            inner = m.group(1)
            return inner + b" " * (len(orig) - len(inner))

        new_head = PAT.sub(sub, head)
        if len(new_head) != len(head):
            raise RuntimeError(f"{path}: byte length changed (internal bug)")
        if new_head == head:
            return "unchanged"

        # Sanity: a leftover np.float64 near the buffer end would mean the
        # header is larger than HEADER_BYTES and a match was truncated.
        if b"np.float64" in new_head[-200:]:
            raise RuntimeError(
                f"{path}: np.float64 near end of {HEADER_BYTES}-byte window — "
                "header may be larger than the buffer; raise HEADER_BYTES"
            )

        f.seek(0)
        f.write(new_head)
        return "fixed"


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: fix_vti_header.py <file.vti> [file2.vti ...]",
              file=sys.stderr)
        return 1
    for arg in args:
        path = Path(arg)
        try:
            status = fix(path)
        except OSError as e:
            print(f"error {path}: {e}", file=sys.stderr)
            continue
        print(f"{status:10s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
