#!/bin/bash
# Operator-level test for boundary_truncation_3d, the
# deferred-correction diagnostic that estimates the boundary-cell
# truncation error tau_h(u_h).  The test fills u with a known
# cubic and checks tau_buf entries against the analytic formula
# to round-off in C; pass/fail status is the binary's exit code.

set -euo pipefail

check() {
    if [ $? -ne 0 ]; then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit 1
    fi
    echo "Passed"
}

echo "=== boundary_truncation_3d (np=1, 16x16x16) ==="
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_deferred_correction_3d 16 16 16
check

echo "=== boundary_truncation_3d (np=2, 32x16x16) ==="
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_deferred_correction_3d 32 16 16
check

echo "=== boundary_truncation_3d (np=8, 32x32x32) ==="
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_deferred_correction_3d 32 32 32
check

echo "All boundary_truncation_3d tests passed"
