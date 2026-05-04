#!/bin/bash
set -e

check_py()
{
    python verify.py > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "FAIL (Python verification)"
        FAIL=$((FAIL + 1))
        return 1
    fi
    rm -f Var0_rank_*.json
    return 0
}

PASS=0
FAIL=0

run()
{
    local label="$1"
    shift
    echo -n "  $label "
    if "$@" > /dev/null 2>&1; then
        check_py
        if [ $? -eq 0 ]; then
            PASS=$((PASS + 1))
            echo "ok"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL (C test aborted)"
        rm -f Var0_rank_*.json
    fi
}

run_nocheck()
{
    local label="$1"
    shift
    echo -n "  $label "
    if "$@" > /dev/null 2>&1; then
        PASS=$((PASS + 1))
        echo "ok"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL"
    fi
}

MPI="mpirun --map-by :OVERSUBSCRIBE"

echo "===== test_topology ====="
run_nocheck "topology:" $MPI -np 1 ./test_topology

echo ""
echo "===== test_rk4 ====="
run_nocheck "rk4 convergence:" $MPI -np 1 ./test_rk4

echo ""
echo "===== test_sync (non-periodic — all directions) ====="
run "1 proc:  " $MPI -np 1  ./test_sync 32 32 32 0 0 0
run "2 procs: " $MPI -np 2  ./test_sync 64 32 32 0 0 0
run "4 procs: " $MPI -np 4  ./test_sync 32 32 32 0 0 0
run "8 procs: " $MPI -np 8  ./test_sync 32 32 32 0 0 0

echo ""
echo "===== test_sync (periodic x,y; non-periodic z — Maxwell default) ====="
# Use 2+ procs to avoid 1-proc periodic self-communication test artifact
run "2 procs: " $MPI -np 2  ./test_sync 32 32 64 1 1 0
run "4 procs: " $MPI -np 4  ./test_sync 32 32 32 1 1 0
run "6 procs: " $MPI -np 6  ./test_sync 64 64 64 1 1 0
run "8 procs: " $MPI -np 8  ./test_sync 32 32 32 1 1 0

echo ""
echo "===== test_sync (fully periodic) ====="
run "2 procs: " $MPI -np 2  ./test_sync 32 32 64 1 1 1
run "4 procs: " $MPI -np 4  ./test_sync 32 32 32 1 1 1
run "8 procs: " $MPI -np 8  ./test_sync 32 32 32 1 1 1
run "27 procs:" $MPI -np 27 ./test_sync 32 32 32 1 1 1

echo ""
echo "===== Summary ====="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
