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


echo -n "2D 1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_child_2d 33 33
check

echo -n "2D 2x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_2d 65 33
check

echo -n "2D 1x2: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_2d 33 65
check

echo -n "2D 3x1: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_child_2d 33 97
check

echo -n "2D 1x3: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_child_2d 97 33
check

echo -n "2D 2x2: "
mpirun --map-by :OVERSUBSCRIBE -np 4 ./test_child_2d 65 65
check

echo -n "2D 3x3: "
mpirun --map-by :OVERSUBSCRIBE -np 9 ./test_child_2d 33 33
check

echo -n "3D 1x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_child_3d 33 33 33
check

echo -n "3D 2x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_3d 65 33 33
check

echo -n "3D 1x2x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_3d 33 65 33
check

echo -n "3D 1x1x2: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_child_3d 33 33 65
check

echo -n "3D 2x2x2: "
mpirun --map-by :OVERSUBSCRIBE -np 8 ./test_child_3d 33 33 33
check

echo -n "3D 3x3x3: "
mpirun --map-by :OVERSUBSCRIBE -np 27 ./test_child_3d 33 33 33
check
