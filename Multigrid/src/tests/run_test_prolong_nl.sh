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
    python verify_nl_prolong.py
    check
    rm -f Var0_rank_*.json
}

echo "=== 2D nonlinear prolongation tests ==="

echo -n "2D 1x1 (33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_prolong_nl_2d 33 33
check
run_verify

echo -n "2D 2x1 (65x33): "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_prolong_nl_2d 65 33
check
run_verify

echo -n "2D 2x2 (33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 4 ./test_prolong_nl_2d 33 33
check
run_verify

echo -n "2D 2x2 (63x63): "
mpirun --map-by :OVERSUBSCRIBE -np 4 ./test_prolong_nl_2d 63 63
check
run_verify

echo -n "2D 3x3 (33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 9 ./test_prolong_nl_2d 33 33
check
run_verify

echo "=== 3D nonlinear prolongation tests ==="

echo -n "3D 1x1x1 (33x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_prolong_nl_3d 33 33 33
check
run_verify

echo -n "3D 2x1x1 (65x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_prolong_nl_3d 65 33 33
check
run_verify

echo -n "3D 2x2x2 (33x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_prolong_nl_3d 33 33 33
check
run_verify

echo -n "3D 2x2x2 (63x63x63): "
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_prolong_nl_3d 63 63 63
check
run_verify

echo -n "3D 3x3x3 (33x33x33): "
mpirun --map-by :OVERSUBSCRIBE -np 27 ./test_prolong_nl_3d 33 33 33
check
run_verify

echo "All nonlinear prolongation tests passed"
