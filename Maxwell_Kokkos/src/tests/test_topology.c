/* test_topology.c — Test automatic_topology() for various process counts.
 *
 * Usage: mpirun -np 1 ./test_topology
 *
 * Runs on a single process; tests that automatic_topology produces valid
 * factorisations for a range of process counts and grid configurations.
 */
#include "domain.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int check(const char *label, int ndim, const size_t dims[],
                 size_t nproc)
{
    size_t topology[3] = { 0, 0, 0 };
    int rc = automatic_topology(ndim, dims, nproc, topology);
    if (rc != 0)
    {
        fprintf(stderr, "FAIL [%s]: automatic_topology returned %d\n", label, rc);
        return -1;
    }

    /* Product must equal nproc */
    size_t product = 1;
    for (int i = 0; i < ndim; i++)
        product *= topology[i];

    if (product != nproc)
    {
        fprintf(stderr,
                "FAIL [%s]: product of topology (%lu) != nproc (%lu)\n",
                label, product, nproc);
        return -1;
    }

    /* Each dimension must have at least 1 process */
    for (int i = 0; i < ndim; i++)
    {
        if (topology[i] < 1)
        {
            fprintf(stderr, "FAIL [%s]: topology[%d] = %lu < 1\n",
                    label, i, topology[i]);
            return -1;
        }
    }

    fprintf(stderr, "  %-20s nproc=%3lu -> %lu x %lu x %lu\n",
            label, nproc,
            ndim >= 1 ? topology[0] : 0,
            ndim >= 2 ? topology[1] : 0,
            ndim >= 3 ? topology[2] : 0);
    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int failures = 0;

    if (rank == 0)
    {
        fprintf(stderr, "Testing automatic_topology()...\n");

        /* Cubic grid 64^3 with various process counts */
        size_t cubic[] = { 64, 64, 64 };
        failures += (check("cubic 1p", 3, cubic, 1) != 0);
        failures += (check("cubic 2p", 3, cubic, 2) != 0);
        failures += (check("cubic 4p", 3, cubic, 4) != 0);
        failures += (check("cubic 6p", 3, cubic, 6) != 0);
        failures += (check("cubic 8p", 3, cubic, 8) != 0);
        failures += (check("cubic 12p", 3, cubic, 12) != 0);
        failures += (check("cubic 16p", 3, cubic, 16) != 0);
        failures += (check("cubic 27p", 3, cubic, 27) != 0);
        failures += (check("cubic 64p", 3, cubic, 64) != 0);

        /* For a cubic grid with 8 procs, expect 2x2x2 */
        {
            size_t topo[3];
            automatic_topology(3, cubic, 8, topo);
            if (topo[0] != 2 || topo[1] != 2 || topo[2] != 2)
            {
                fprintf(stderr, "FAIL: cubic 8p should be 2x2x2, got %lux%lux%lu\n",
                        topo[0], topo[1], topo[2]);
                failures++;
            }
        }

        /* Elongated grid 256x32x32: should prefer splitting x */
        size_t elongated[] = { 256, 32, 32 };
        failures += (check("elongated 2p", 3, elongated, 2) != 0);
        failures += (check("elongated 4p", 3, elongated, 4) != 0);
        failures += (check("elongated 8p", 3, elongated, 8) != 0);

        /* Prime number of processes */
        failures += (check("cubic 7p", 3, cubic, 7) != 0);
        failures += (check("cubic 13p", 3, cubic, 13) != 0);

        /* Single process */
        failures += (check("cubic 1p", 3, cubic, 1) != 0);

        if (failures > 0)
        {
            fprintf(stderr, "FAILED: %d test(s) failed\n", failures);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(stderr, "PASSED\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
