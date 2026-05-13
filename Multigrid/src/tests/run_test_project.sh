#!/bin/bash

check()
{
    if [ $? -ne 0 ]
    then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit 1
    fi
}

run_verify()
{
    python verify.py
    check
    rm -f rank_*.h5
}

for mode in inject restrict; do
    echo "=== Mode: $mode ==="

    echo -n "3D 1x1x1 (33x33x33): "
    mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_project_3d 32 32 32 $mode
    check
    run_verify
    echo "Passed"

    echo -n "3D 2x1x1 (65x33x33): "
    mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_project_3d 64 32 32 $mode
    check
    run_verify
    echo "Passed"

    echo -n "3D 2x2x2 (33x33x33): "
    mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_project_3d 32 32 32 $mode
    check
    run_verify
    echo "Passed"

    echo -n "3D 2x2x2 (63x63x63): "
    mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_project_3d 62 62 62 $mode
    check
    run_verify
    echo "Passed"

    echo -n "3D 3x3x3 (33x33x33): "
    mpirun --map-by :OVERSUBSCRIBE -np 27 ./test_project_3d 32 32 32 $mode
    check
    run_verify
    echo "Passed"

done

echo "All project tests passed"
