#!/bin/bash

check()
{
    if [ $? -ne 0 ]
    then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit 1
    fi
    echo "Passed"
}

run_verify()
{
    python verify_nl_restrict.py
    check
    rm -f rank_*.h5
}

echo "=== 2D nonlinear restriction tests ==="

echo -n "2D 1x1 (33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_restrict_nl_2d 32 32
check
run_verify

echo -n "2D 2x1 (65x33): "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_restrict_nl_2d 64 32
check
run_verify

echo -n "2D 2x2 (33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 4 ./test_restrict_nl_2d 32 32
check
run_verify

echo -n "2D 2x2 (63x63): "
mpirun --map-by :OVERSUBSCRIBE -np 4 ./test_restrict_nl_2d 62 62
check
run_verify

echo -n "2D 3x3 (33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 9 ./test_restrict_nl_2d 32 32
check
run_verify

echo "=== 3D nonlinear restriction tests ==="

echo -n "3D 1x1x1 (33x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_restrict_nl_3d 32 32 32
check
run_verify

echo -n "3D 2x1x1 (65x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_restrict_nl_3d 64 32 32
check
run_verify

echo -n "3D 2x2x2 (33x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_restrict_nl_3d 32 32 32
check
run_verify

echo -n "3D 2x2x2 (63x63x63): "
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_restrict_nl_3d 62 62 62
check
run_verify

echo -n "3D 3x3x3 (33x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 27 ./test_restrict_nl_3d 32 32 32
check
run_verify

echo "All nonlinear restriction tests passed"
