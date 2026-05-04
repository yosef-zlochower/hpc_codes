import numpy as np
import json
import sys
import argparse

def check_rank(n, expect_failure=False):
    try:
        with open(f"Var0_rank_{n}.json", "r") as f:
            d = json.load(f)
    except FileNotFoundError:
        if not expect_failure:
            sys.stderr.write(f"Rank {n} not found\n")
            sys.exit(-1)
        return None

    if expect_failure:
        sys.stderr.write(f"Found extra rank {n}\n")
        sys.exit(-1)
        return None

    grid = np.array(d['data'])

    if len(grid.shape) == 3:
        X = np.zeros_like(grid)
        Y = np.zeros_like(grid)
        Z = np.zeros_like(grid)

        for i in range(d['nx']):
            for j in range(d['ny']):
                for k in range(d['nz']):
                    X[k, j, i] = d['x0'] + i * d['dx']
                    Y[k, j, i] = d['y0'] + j * d['dy']
                    Z[k, j, i] = d['z0'] + k * d['dz']
 
        err = np.abs(grid - (1.5 * X - 2 * Y + Z))
    elif len(grid.shape) == 2:
        X = np.zeros_like(grid)
        Y = np.zeros_like(grid)
        for i in range(d['nx']):
            for j in range(d['ny']):
                    X[j, i] = d['x0'] + i * d['dx']
                    Y[j, i] = d['y0'] + j * d['dy']
        err = np.abs(grid - (1.5 * X - 2 * Y))
    else:
        print("Unsupported")
        sys.exit(-1)

    return np.max(err)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-procs", type=int, default=1, help="number of mpi processes used")

    args = parser.parse_args()
    num_procs=args.num_procs
    assert num_procs > 0
    gerr = -np.inf
    for rank in range(num_procs):
        lerr= check_rank(rank)
        if lerr is not None:
            if lerr  > gerr:
                gerr = lerr
            print(f'Error in rank {rank} is {lerr}')
        else:
            gerr = np.inf

    check_rank(num_procs, expect_failure=True)
    print(f'Infinity norm over all gridpoints is {gerr}')
    if (gerr < 1.0e-12):
        print("PASSED")
    else:
        print("FAILED")
        sys.exit(-1)  # send fail exit code

if __name__=="__main__":
    main()
