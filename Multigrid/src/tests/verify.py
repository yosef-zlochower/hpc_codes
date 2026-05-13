import numpy as np
import sys

from h5read import load_rank


ROUNDOFF_TOLERANCE = 1.0e-12

"""
Verify domain decomposition and syncing tests in 3D.

Reads per-rank HDF5 output (rank_<R>.h5) written by output_3d_gf.  Each
file contains a /metadata group with grid attributes and one or more
/<vname> datasets; load_rank returns those in a dict whose keys mirror
the legacy JSON field names.

Ghost metadata fields:
  gs            : ghost zone width (integer)
  lower_x_ghost : true if this rank has an MPI neighbor on the lower-x face
  upper_x_ghost : true if this rank has an MPI neighbor on the upper-x face
  lower_y_ghost : true if this rank has an MPI neighbor on the lower-y face
  upper_y_ghost : true if this rank has an MPI neighbor on the upper-y face
  lower_z_ghost : true if this rank has an MPI neighbor on the lower-z face
  upper_z_ghost : true if this rank has an MPI neighbor on the upper-z face
"""


def verify_domain_and_sync(tol=ROUNDOFF_TOLERANCE):
    d = load_rank(0)

    # Cell counts -> grid-point counts for the vertex-centred Dirichlet
    # layout (points = cells + 1).
    global_ni = d["global_cells_x"] + 1
    global_nj = d["global_cells_y"] + 1
    global_nk = d["global_cells_z"] + 1
    global_x0 = d["global_x0"]
    global_y0 = d["global_y0"]
    global_z0 = d["global_z0"]
    mpi_size = d["mpi_size"]

    test_array = np.full((global_nk, global_nj, global_ni), np.nan)
    max_error = -np.inf

    for rank in range(mpi_size):
        d = load_rank(rank)

        nz = d['nz']
        ny = d['ny']
        nx = d['nx']
        x0 = d['x0']
        y0 = d['y0']
        z0 = d['z0']
        dx = d['dx']
        dy = d['dy']
        dz = d['dz']

        local_k0 = d["local_k0"]
        local_j0 = d["local_j0"]
        local_i0 = d["local_i0"]

        gs = d["gs"]
        lower_x_ghost = d["lower_x_ghost"]
        upper_x_ghost = d["upper_x_ghost"]
        lower_y_ghost = d["lower_y_ghost"]
        upper_y_ghost = d["upper_y_ghost"]
        lower_z_ghost = d["lower_z_ghost"]
        upper_z_ghost = d["upper_z_ghost"]

        local_data = d["data"]

        # Build coordinate arrays for ALL local points (including ghost zones)
        local_x = x0 + np.arange(nx) * dx          # shape (nx,)
        local_y = y0 + np.arange(ny) * dy          # shape (ny,)
        local_z = z0 + np.arange(nz) * dz          # shape (nz,)
        # Broadcast to (nz, ny, nx)
        expected = (1.5 * local_x[np.newaxis, np.newaxis, :]
                    - 2.0 * local_y[np.newaxis, :, np.newaxis]
                    + local_z[:, np.newaxis, np.newaxis])

        # Check ALL data (including ghost zones) against the test function.
        # After sync, ghost zones must hold the correct values from neighbors.
        local_error = np.max(np.abs(local_data - expected))
        if local_error > max_error:
            max_error = local_error

        if local_error > tol:
            print(f"Test Failure for rank {rank}: max error = {local_error:.3e}")
            return False

        # Owned (non-ghost) index ranges for assembling the global array
        i_lo = gs if lower_x_ghost else 0
        i_hi = nx - (gs if upper_x_ghost else 0)
        j_lo = gs if lower_y_ghost else 0
        j_hi = ny - (gs if upper_y_ghost else 0)
        k_lo = gs if lower_z_ghost else 0
        k_hi = nz - (gs if upper_z_ghost else 0)

        gi_lo = local_i0 + i_lo
        gi_hi = local_i0 + i_hi
        gj_lo = local_j0 + j_lo
        gj_hi = local_j0 + j_hi
        gk_lo = local_k0 + k_lo
        gk_hi = local_k0 + k_hi

        test_array[gk_lo:gk_hi, gj_lo:gj_hi, gi_lo:gi_hi] = \
            local_data[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]

    print(f"Max local error (all data including ghost zones): {max_error:.3e}")

    # Build global expected array
    gx = global_x0 + np.arange(global_ni) * d["dx"]
    gy = global_y0 + np.arange(global_nj) * d["dy"]
    gz = global_z0 + np.arange(global_nk) * d["dz"]
    global_expected = (1.5 * gx[np.newaxis, np.newaxis, :]
                       - 2.0 * gy[np.newaxis, :, np.newaxis]
                       + gz[:, np.newaxis, np.newaxis])

    if np.isnan(test_array).any():
        print("Domain Test Failure: global array has unfilled (NaN) points")
        return False

    global_error = np.max(np.abs(test_array - global_expected))
    print(f"Global error (owned data assembled): {global_error:.3e}")

    if global_error > tol or max_error > tol:
        print("Sync Test Failure")
        return False

    return True


def main():
    res = verify_domain_and_sync(tol=1.0e-14)
    if res == False:
        sys.exit(-1)
    print("Test passed")


if __name__ == "__main__":
    main()
