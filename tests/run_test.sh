#!/bin/bash

check()
{
    python verify.py >& /dev/null
    if [ $? -ne 0 ]
    then
        echo "FAILURE OF TEST. STOPPING" 1>&2
        exit -1
    fi
    echo "Passed"
}


echo -n  "1x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_domain_3d 32 32 32
check
rm Var0_rank_*.json

echo -n  "2x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_domain_3d 64 32 32
check
rm Var0_rank_*.json

echo -n  "1x2x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_domain_3d 32 64 32
check
rm Var0_rank_*.json

echo -n  "1x1x2: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_domain_3d 32 32 64
check
rm Var0_rank_*.json

echo -n  "3x1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_domain_3d 96 32 32
check
rm Var0_rank_*.json

echo -n  "1x3x1: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_domain_3d 32 96 32
check
rm Var0_rank_*.json

echo -n  "1x1x3: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_domain_3d 32 32 96
check
rm Var0_rank_*.json

echo -n  "3x3x3: "
mpirun --map-by :OVERSUBSCRIBE -np 27 ./test_domain_3d 32 32 32
check
rm Var0_rank_*.json

echo -n  "4x4x4: "
mpirun --map-by :OVERSUBSCRIBE -np 64 ./test_domain_3d 32 32 32
check
rm Var0_rank_*.json


echo -n  "1x1: "
mpirun --map-by :OVERSUBSCRIBE -np 1 ./test_domain_2d 32 32
check
rm Var0_rank_*.json

echo -n  "2x1: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_domain_2d 64 32
check
rm Var0_rank_*.json

echo -n  "1x2: "
mpirun --map-by :OVERSUBSCRIBE -np 2 ./test_domain_2d 32 64
check
rm Var0_rank_*.json

echo -n  "3x1: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_domain_2d 96 32
check
rm Var0_rank_*.json

echo -n  "1x3: "
mpirun --map-by :OVERSUBSCRIBE -np 3 ./test_domain_2d 32 96
check
rm Var0_rank_*.json

echo -n  "3x3: "
mpirun --map-by :OVERSUBSCRIBE -np 9 ./test_domain_2d 32 32
check
rm Var0_rank_*.json

echo -n  "4x4: "
mpirun --map-by :OVERSUBSCRIBE -np 16 ./test_domain_2d 32 32
check
rm Var0_rank_*.json
