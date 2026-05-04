#!/usr/bin/env python3
"""Verify second-order discretisation convergence of driver_multigrid.

Reads two or more driver_multigrid log files, parses the
``|u - u_exact|_inf = ...`` line from each, and asserts that the
empirical convergence rate on the finest pair lies in [LO, HI].

Each log filename must end in ``_<N>.txt`` so the cell count per
direction can be inferred from the name; e.g. ``log_np1_64.txt``
corresponds to N = 64 cells per direction (h = 1/64).

For the model problem (manufactured solution
u = sin(pi x) sin(pi y) sin(pi z), 7-point Laplacian, homogeneous
Dirichlet BCs on the unit cube) the standard truncation-error analysis
predicts an asymptotic rate of exactly 2.  A pass-band of [1.8, 2.3]
gives comfortable margin for finite-N preasymptotic effects without
masking real regressions.

Exit code: 0 on PASS, 1 on FAIL, 2 on usage / parse error.
"""
import math
import re
import sys

ERROR_RE = re.compile(r"\|u - u_exact\|_inf\s*=\s*([+\-0-9.eE]+)")
SIZE_RE  = re.compile(r"_(\d+)\.txt$")

LO, HI = 1.8, 2.3


def parse_log(path):
    """Return (n_cells, error) for one log file."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"verify_convergence: cannot read '{path}': {e}")

    matches = ERROR_RE.findall(text)
    if not matches:
        sys.exit(f"verify_convergence: '{path}' contains no "
                 f"'|u - u_exact|_inf = ...' line")
    err = float(matches[-1])

    m = SIZE_RE.search(path)
    if not m:
        sys.exit(f"verify_convergence: cannot infer cell count from "
                 f"'{path}' (expected a name ending '_<N>.txt')")
    return int(m.group(1)), err


def main(paths):
    if len(paths) < 2:
        sys.stderr.write(__doc__)
        return 2

    runs = sorted(parse_log(p) for p in paths)  # ascending in n

    print(f"    {'cells/dir':>9}  {'h':>11}  {'|err|_inf':>13}  "
          f"{'ratio':>7}  {'rate':>6}")
    for i, (n, err) in enumerate(runs):
        h = 1.0 / n
        if i == 0:
            print(f"    {n:>9}  {h:>11.4e}  {err:>13.6e}  "
                  f"{'-':>7}  {'-':>6}")
        else:
            n_prev, err_prev = runs[i-1]
            ratio = err_prev / err
            rate  = math.log(ratio) / math.log(n / n_prev)
            print(f"    {n:>9}  {h:>11.4e}  {err:>13.6e}  "
                  f"{ratio:>7.3f}  {rate:>6.3f}")

    # Assert on the finest pair: that's the cleanest indicator of the
    # asymptotic rate.  Coarser pairs may still be in the preasymptotic
    # regime where higher-order truncation terms contribute.
    (n_prev, err_prev), (n_fine, err_fine) = runs[-2], runs[-1]
    if err_fine <= 0.0 or err_prev <= 0.0:
        print(f"\nFAIL: non-positive error encountered "
              f"({err_prev=}, {err_fine=})")
        return 1

    ratio = err_prev / err_fine
    rate  = math.log(ratio) / math.log(n_fine / n_prev)

    if LO <= rate <= HI:
        print(f"\nPASS: rate {rate:.3f} on finest pair "
              f"({n_prev} -> {n_fine}) is in [{LO}, {HI}]")
        return 0

    print(f"\nFAIL: rate {rate:.3f} on finest pair "
          f"({n_prev} -> {n_fine}) is NOT in [{LO}, {HI}]")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
