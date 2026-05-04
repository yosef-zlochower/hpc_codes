#!/bin/bash
#
# End-to-end discretisation-convergence test.
#
# Runs driver_multigrid against the manufactured-solution Poisson problem
# at three resolutions (32^3, 64^3, 128^3) and asserts that the printed
# |u - u_exact|_inf shrinks at the second-order rate predicted by the
# 7-point Laplacian.
#
# The test is performed twice: first sequentially (np=1) to isolate
# algorithmic behaviour, then in parallel (np=8) to ensure the
# parallel path does not degrade the discretisation error.

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

# All driver outputs (logs, TOMLs, per-rank JSON spam) land in this
# subdirectory and are wiped at the end.
WORKDIR=convergence_run
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
n_smooth  = 50
n_iters   = 20
tol       = 1.0e-12
subcycles = 1
min_cells = 4
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
    rm -f Var*_rank_*.json VAR_*_rank_*.json
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
echo "=== convergence test (sequential, np=1) ==="
run_one 1 32
run_one 1 64
run_one 1 128
verify log_np1_32.txt log_np1_64.txt log_np1_128.txt

# -----------------------------------------------------------------------------
# Parallel pass (8 ranks, 2 x 2 x 2 process grid)
# -----------------------------------------------------------------------------
echo "=== convergence test (parallel, np=8) ==="
run_one 8 32
run_one 8 64
run_one 8 128
verify log_np8_32.txt log_np8_64.txt log_np8_128.txt

cd ..
echo "All convergence tests passed"
