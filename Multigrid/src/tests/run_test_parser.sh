#!/bin/bash
#
# Parser-hardening regression test.
#
# Drives the multigrid driver against several intentionally-malformed
# TOML files and confirms that each is rejected with the expected
# diagnostic.  Covers Plan.md items 4.2 (unknown-key detection) and
# 4.3 (out-of-range integer narrowing).

check()
{
    if [ $? -ne 0 ]
    then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit 1
    fi
    echo "Passed"
}

DRIVER="$(cd .. && pwd)/driver_multigrid"

if [ ! -x "$DRIVER" ]
then
    echo "FAILURE: driver '$DRIVER' is missing or not executable." 1>&2
    echo "         Run 'make driver_multigrid' first." 1>&2
    exit 1
fi

WORKDIR=parser_run
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# A baseline TOML the per-test files will start from and locally edit.
write_baseline()
{
    local out="$1"
    cat >"$out" <<'EOF'
[grid]
nx_cells = 32
ny_cells = 32
nz_cells = 32

[solver]
multigrid = true
omega     = 1.5
n_smooth  = 5
n_iters   = 5
tol       = 1.0e-6
subcycles = 1
min_cells = 4
EOF
}

# run_should_fail <description> <toml-path> <expected-substring>
#
# Runs the driver against <toml-path>, expects a non-zero exit code,
# and expects <expected-substring> to appear in the captured output.
run_should_fail()
{
    local desc="$1" toml="$2" expect="$3"
    local log="${toml%.toml}.log"
    echo -n "  $desc: "

    # Disable shell-level error propagation so we can inspect $? ourselves.
    if mpirun --map-by :OVERSUBSCRIBE -np 1 "$DRIVER" "$toml" >"$log" 2>&1
    then
        echo "FAILURE: driver exited 0 but the TOML should have been rejected"
        echo "         (expected to see: $expect)"
        echo "--- driver output ---"
        cat "$log"
        exit 1
    fi

    if ! grep -qF "$expect" "$log"
    then
        echo "FAILURE: missing expected diagnostic"
        echo "         (looking for: $expect)"
        echo "--- driver output ---"
        cat "$log"
        exit 1
    fi
    echo "Passed"
}

# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------

echo "=== parser hardening tests ==="

# Case 1 (4.2): unknown key in [solver] -- typo'd "mulitgrid"
write_baseline typo_key.toml
sed -i 's/^multigrid = true/mulitgrid = true/' typo_key.toml
run_should_fail "unknown key in [solver] (typo: mulitgrid)" \
    typo_key.toml \
    "unknown key 'solver.mulitgrid'"

# Case 2 (4.2): unknown key in [grid] -- typo'd "nx_celss"
write_baseline typo_grid_key.toml
sed -i 's/^nx_cells = 32/nx_celss = 32/' typo_grid_key.toml
run_should_fail "unknown key in [grid] (typo: nx_celss)" \
    typo_grid_key.toml \
    "unknown key 'grid.nx_celss'"

# Case 3 (4.2): unknown top-level section -- typo'd "[gird]"
write_baseline typo_section.toml
sed -i 's/^\[grid\]/[gird]/' typo_section.toml
run_should_fail "unknown top-level section (typo: gird)" \
    typo_section.toml \
    "unknown top-level entry 'gird'"

# Case 4 (4.3): integer above INT_MAX must be rejected, not silently truncated
write_baseline oversize.toml
sed -i 's/^n_smooth  = 5/n_smooth  = 9999999999/' oversize.toml
run_should_fail "out-of-range n_smooth (32-bit overflow)" \
    oversize.toml \
    "outside the 32-bit int range"

cd ..
echo "All parser hardening tests passed"
