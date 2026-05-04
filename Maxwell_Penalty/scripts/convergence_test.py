#!/usr/bin/env python3
"""Convergence study for the SBP-SAT Maxwell scheme.

Drives the solver across four boundary configurations and a sequence of
resolutions, and reports the empirical L2 convergence rate for each
configuration. The four configurations are chosen so that each new row of
the table exercises one additional class of boundary structure:

    Configuration   | Physical faces | Edges | Corners | Expected L2 rate
    ----------------+----------------+-------+---------+------------------
    fully_periodic  |      0         |   0   |    0    |        4
    z_physical      |      2         |   0   |    0    |        3
    yz_physical     |      4         |   4   |    0    |        3
    all_physical    |      6         |  12   |    8    |        3

The test source is the analytic ``te_waveguide_mode`` from
analytic_solutions.c, prescribed as SBP-SAT boundary data on every
physical face. Since the analytic is an exact vacuum Maxwell solution,
the numerical error at the final time is dominated by the scheme's own
truncation error, and the L2 convergence rate should approach the
values above as ``dx -> 0``.

Usage:
    python convergence_test.py --solver /path/to/maxwell_system
                                [--np N]
                                [--resolutions 16 24 32 48 64]
                                [--final-time 0.2]
                                [--work-dir /tmp/maxwell_conv]

If a rate is *not* at the expected value, the test prints a warning and
returns a non-zero exit status.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Parameters for a single solver run. Anything not set is the test default.
class Config:
    def __init__(
        self,
        name: str,
        periodic: tuple[bool, bool, bool],
        expected_rate: float,
    ):
        self.name = name
        self.periodic = periodic
        self.expected_rate = expected_rate


CONFIGS: list[Config] = [
    Config("fully_periodic",  (True,  True,  True),  4.0),
    Config("z_physical",      (True,  True,  False), 3.0),
    Config("yz_physical",     (True,  False, False), 3.0),
    Config("all_physical",    (False, False, False), 3.0),
]


TOML_TEMPLATE = """[grid]
nx = {n}
ny = {n}
nz = {n}
x0 = 0.0
y0 = 0.0
z0 = 0.0
xn = 1.0
yn = 1.0
zn = 1.0
periodic_x = {px}
periodic_y = {py}
periodic_z = {pz}

[solver]
ghost_size = 3
cfl_factor = {cfl}
max_iterations = {maxit}
output_every = {outevery}
checkpoint_every = 0
max_checkpoints = 2
recover = false
use_dissipation = false
diss_coeff = 0.1
tau = 1.0

[physics]
kappa_D = 0.1
kappa_B = 0.1

[source]
type = "te_waveguide_mode"

[source.plane_wave]
ax = 1.0
ay = 1.0
k = 4.0
bump_a = 0.0
bump_b = 1.0

[source.gaussian_beam]
w0 = 0.3
z_waist = 0.5
k = 4.0
amplitude = 1.0
ramp_a = 0.0
ramp_b = 1.0

[source.te_waveguide_mode]
l = 2
m = 2
n = 2

[material]
epsilon_type = "uniform"

[material.background]
epsilon = 1.0
mu = 1.0
sigma = 0.0

[material.elliptical]
max = 1.0
x0 = 0.5
z0 = 0.5
s = 2.0
a = 0.5
b = 0.25
"""


def build_toml(
    *, n: int, periodic: tuple[bool, bool, bool],
    cfl: float, final_time: float,
) -> str:
    dx = 1.0 / n
    dt = cfl * dx
    # Round maxit up to the next multiple of output_every so we always get
    # an output at the final time.
    target_steps = int(math.ceil(final_time / dt))
    output_every = target_steps  # single output at the final step
    maxit = output_every + 1  # max_iterations is exclusive upper bound
    pbool = lambda b: "true" if b else "false"
    return TOML_TEMPLATE.format(
        n=n, cfl=cfl, maxit=maxit, outevery=output_every,
        px=pbool(periodic[0]),
        py=pbool(periodic[1]),
        pz=pbool(periodic[2]),
    )


def final_l2_error(run_dir: Path) -> float:
    """Read the last line of l2_norm.dat and return its error column."""
    path = run_dir / "l2_norm.dat"
    if not path.exists():
        raise FileNotFoundError(f"{path} not produced by the solver")
    with path.open() as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    return float(lines[-1].split()[1])


def run_solver(
    solver: Path, run_dir: Path, toml_path: Path, nproc: int,
) -> None:
    cmd = [
        "mpirun", "--map-by", ":OVERSUBSCRIBE", "-np", str(nproc),
        str(solver.resolve()), str(toml_path.name),
    ]
    result = subprocess.run(
        cmd, cwd=run_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"solver failed (rc={result.returncode})")


def run_case(
    *, solver: Path, work_root: Path, nproc: int, cfg: Config,
    n: int, cfl: float, final_time: float,
) -> float:
    run_dir = work_root / cfg.name / f"N{n:03d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    toml = run_dir / "maxwell.toml"
    toml.write_text(build_toml(
        n=n, periodic=cfg.periodic, cfl=cfl, final_time=final_time,
    ))
    run_solver(solver, run_dir, toml, nproc)
    return final_l2_error(run_dir)


def fit_rates(resolutions: list[int], errors: list[float]) -> list[float]:
    """Per-pair convergence rates log2(err_N / err_{2N-ish})."""
    rates: list[float] = []
    for i in range(1, len(resolutions)):
        n0, n1 = resolutions[i - 1], resolutions[i]
        e0, e1 = errors[i - 1], errors[i]
        if e0 <= 0 or e1 <= 0:
            rates.append(float("nan"))
        else:
            rates.append(math.log(e0 / e1) / math.log(n1 / n0))
    return rates


def fmt_row(label: str, values: list, width: int = 12) -> str:
    return label.ljust(18) + "".join(
        (f"{v:>{width}.4g}" if isinstance(v, float) else str(v).rjust(width))
        for v in values
    )


def run_study(
    *, solver: Path, work_root: Path, nproc: int,
    resolutions: list[int], cfl: float, final_time: float,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    all_errors: dict[str, list[float]] = {}
    all_rates:  dict[str, list[float]] = {}
    for cfg in CONFIGS:
        print(f"\n=== {cfg.name}  (expected L2 rate {cfg.expected_rate}) ===")
        errors: list[float] = []
        for n in resolutions:
            print(f"  N={n:3d} ... ", end="", flush=True)
            try:
                err = run_case(
                    solver=solver, work_root=work_root, nproc=nproc,
                    cfg=cfg, n=n, cfl=cfl, final_time=final_time,
                )
            except Exception as e:
                print(f"FAILED: {e}")
                err = float("nan")
            errors.append(err)
            print(f"L2 = {err:.4g}")
        rates = fit_rates(resolutions, errors)
        print(fmt_row("  resolution",    list(resolutions)))
        print(fmt_row("  L2 error",      errors))
        print(fmt_row("  rate N_k,N_k+1",[None] + rates))
        all_errors[cfg.name] = errors
        all_rates[cfg.name]  = rates
    return all_errors, all_rates


def print_summary(
    resolutions: list[int],
    all_errors: dict[str, list[float]],
    all_rates:  dict[str, list[float]],
) -> int:
    print("\n\n=== SUMMARY ===")
    print(fmt_row("config",        [f"N={n}" for n in resolutions]
                                   + ["final rate"]))
    problems = 0
    for cfg in CONFIGS:
        errs  = all_errors.get(cfg.name, [])
        rates = all_rates.get(cfg.name, [])
        final_rate = rates[-1] if rates else float("nan")
        status = "OK"
        if math.isnan(final_rate):
            status = "FAIL (run errored)"
            problems += 1
        elif abs(final_rate - cfg.expected_rate) > 1.0:
            status = f"SUSPICIOUS (expected {cfg.expected_rate})"
            problems += 1
        print(fmt_row(
            cfg.name,
            errs + [final_rate],
        ) + f"   {status}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solver", type=Path,
                    default=Path("src/build/maxwell_system"),
                    help="Path to the maxwell_system executable")
    ap.add_argument("--np", type=int, default=2,
                    help="MPI ranks (default: 2)")
    ap.add_argument(
        "--resolutions", nargs="+", type=int, default=[16, 24, 32, 48],
        help="Grid sizes to test (per axis). Default: 16 24 32 48",
    )
    ap.add_argument("--cfl", type=float, default=0.2,
                    help="CFL factor for all runs (default: 0.2)")
    ap.add_argument("--final-time", type=float, default=0.2,
                    help="Physical simulation time per run (default: 0.2)")
    ap.add_argument("--work-dir", type=Path,
                    default=Path("/tmp/maxwell_conv"),
                    help="Scratch directory for the runs")
    args = ap.parse_args()

    if not args.solver.exists():
        print(f"solver not found: {args.solver}", file=sys.stderr)
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    resolutions = sorted(set(args.resolutions))

    all_errors, all_rates = run_study(
        solver=args.solver, work_root=args.work_dir, nproc=args.np,
        resolutions=resolutions, cfl=args.cfl,
        final_time=args.final_time,
    )
    problems = print_summary(resolutions, all_errors, all_rates)
    if problems:
        print(f"\n{problems} configuration(s) did not meet the expected rate.",
              file=sys.stderr)
        return 1
    print("\nAll configurations at the expected convergence rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
