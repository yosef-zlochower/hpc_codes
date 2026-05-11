#!/bin/bash
#
# End-to-end discretisation-convergence test.
#
# Runs driver_multigrid against a named manufactured-solution preset at
# three resolutions (32^3, 64^3, 128^3) and asserts that the printed
# |u - u_exact|_inf shrinks at the second-order rate predicted by the
# 7-point Laplacian.
#
# The test is performed twice: first sequentially (np=1) to isolate
# algorithmic behaviour, then in parallel (np=8) to ensure the parallel
# path does not degrade the discretisation error.
#
# Usage:
#     run_test_convergence.sh [PRESET]
#
# PRESET is the name of an entry in g_problems[] (problem_registry.c).
# Defaults to "manufactured_dirichlet_homog" (the original test problem).

PRESET="${1:-manufactured_dirichlet_homog}"

check()
{
    if [ $? -ne 0 ]
    then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit 1
    fi
    echo "Passed"
}

# Resolve absolute paths before we cd into the work directory.
DRIVER="$(cd .. && pwd)/driver_multigrid"
VERIFIER="$(pwd)/verify_convergence.py"

if [ ! -x "$DRIVER" ]
then
    echo "FAILURE: driver '$DRIVER' is missing or not executable." 1>&2
    echo "         Run 'make driver_multigrid' first." 1>&2
    exit 1
fi

# All driver outputs (logs, TOMLs, per-rank JSON spam) land in a per-
# preset subdirectory and are wiped at the end.  Per-preset isolation
# means CTest's -j parallel mode does not collide between presets.
WORKDIR="convergence_run_${PRESET}"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# gen_toml <n_cells_per_dir> <output_path>
gen_toml()
{
    local n="$1" out="$2"
    cat >"$out" <<EOF
[grid]
nx_cells = $n
ny_cells = $n
nz_cells = $n

[solver]
multigrid = true
omega     = 1.5
n_smooth  = 2
n_iters   = 40
tol       = 1.0e-12
subcycles = 1
# min_cells = 2 (not 4): at np=8 the auto topology decomposes the
# global grid into 2x2x2 per-rank tiles, so for N=32 each rank starts
# with only 16 cells/dir.  min_cells=4 caps the hierarchy at 3 levels
# (coarsest 4 cells/dir per rank = 8 cells/dir globally), which leaves
# the smooth-mode error under-resolved and the V-cycle plateaus at a
# defect reduction factor of ~0.8 per cycle.  min_cells=2 lets the
# hierarchy go 5+ levels deep on the same N and recovers the
# h-independent convergence rate the convergence test expects.
min_cells = 2

[problem]
name = "${PRESET}"
EOF
}

# run_one <nproc> <n_cells>
# Runs one driver invocation, captures its stdout/stderr to a log, and
# wipes the per-rank JSON output it produces.
run_one()
{
    local nproc="$1" n="$2"
    local toml="params_${n}.toml"
    local log="log_np${nproc}_${n}.txt"

    gen_toml "$n" "$toml"
    echo -n "  np=${nproc}, ${n} cells/dir: "
    mpirun --map-by :OVERSUBSCRIBE -np "$nproc" "$DRIVER" "$toml" \
        >"$log" 2>&1
    check
    rm -f rank_*.h5
}

# verify <log1> <log2> <log3>
verify()
{
    echo "  rate check:"
    python3 "$VERIFIER" "$@"
    check
}

# -----------------------------------------------------------------------------
# Sequential pass
# -----------------------------------------------------------------------------
echo "=== convergence test [${PRESET}] (sequential, np=1) ==="
run_one 1 32
run_one 1 64
run_one 1 128
verify log_np1_32.txt log_np1_64.txt log_np1_128.txt

# -----------------------------------------------------------------------------
# Parallel pass (8 ranks, 2 x 2 x 2 process grid)
# -----------------------------------------------------------------------------
echo "=== convergence test [${PRESET}] (parallel, np=8) ==="
run_one 8 32
run_one 8 64
run_one 8 128
verify log_np8_32.txt log_np8_64.txt log_np8_128.txt

cd ..
echo "All convergence tests for ${PRESET} passed"
