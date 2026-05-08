#!/bin/bash
# Operator-level test for restrict_var_cc_3d, the cell-centred
# (8-cell box-average) restriction.  Runs the test binary at a few
# resolutions and rank counts; the binary itself does the pass/fail
# check internally and exits with the appropriate status.

set -euo pipefail

check() {
    if [ $? -ne 0 ]; then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit 1
    fi
    echo "Passed"
}

echo "=== restrict_var_cc_3d (np=1, 16x16x16) ==="
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_restrict_cc_3d 16 16 16
check

echo "=== restrict_var_cc_3d (np=2, 32x16x16) ==="
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_restrict_cc_3d 32 16 16
check

echo "=== restrict_var_cc_3d (np=8, 32x32x32) ==="
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_restrict_cc_3d 32 32 32
check

echo "=== restrict_var_cc_3d (np=8, 64x64x64) ==="
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_restrict_cc_3d 64 64 64
check

echo "All restrict_var_cc_3d tests passed"
