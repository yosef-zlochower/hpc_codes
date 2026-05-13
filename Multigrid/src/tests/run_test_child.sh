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


echo -n "3D 1x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_child_3d 32 32 32
check

echo -n "3D 2x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_3d 64 32 32
check

echo -n "3D 1x2x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_3d 32 64 32
check

echo -n "3D 1x1x2: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_3d 32 32 64
check

echo -n "3D 2x2x2: "
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_child_3d 32 32 32
check

echo -n "3D 3x3x3: "
mpirun --map-by :OVERSUBSCRIBE -np 27 ./test_child_3d 32 32 32
check
