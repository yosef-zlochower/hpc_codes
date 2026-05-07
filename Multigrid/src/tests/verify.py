import numpy as np
import json
import sys


ROUNDOFF_TOLERANCE = 1.0e-12

"""
Verify domain decomposition and syncing tests for both 2d and 3d cases. 2d cases are mapped to 3d grids
  data[j,i] ->   data[0,j,i].
  Note that the data are stored in fortran order.

Ghost metadata fields (added to JSON output from test programs):
  gs            : ghost zone width (integer)
  lower_x_ghost : true if this rank has an MPI neighbor on the lower-x face
  upper_x_ghost : true if this rank has an MPI neighbor on the upper-x face
  lower_y_ghost : true if this rank has an MPI neighbor on the lower-y face
  upper_y_ghost : true if this rank has an MPI neighbor on the upper-y face
  lower_z_ghost : true if this rank has an MPI neighbor on the lower-z face (3D only)
  upper_z_ghost : true if this rank has an MPI neighbor on the upper-z face (3D only)

If ghost metadata is absent (old-style JSON), all data is treated as owned.
"""


def verify_domain_and_sync(tol=ROUNDOFF_TOLERANCE):
    with open("Var0_rank_0.json", "r") as f:
        d = json.load(f)

    # JSON now stores cell counts; convert to grid-point counts for the
    # vertex-centred Dirichlet layout (points = cells + 1).
    global_ni = d["global_cells_x"] + 1
    global_nj = d["global_cells_y"] + 1
    global_nk = d.get("global_cells_z", 0) + 1  # 2D: 1 plane → 0+1 cells
    global_x0 = d["global_x0"]
    global_y0 = d["global_y0"]
    global_z0 = d.get("global_z0", 0)  # in 2d case, set z=0
    dx = d["dx"]
    dy = d["dy"]
    dz = d.get("dz", 0)
    mpi_size = d["mpi_size"]

    test_array = np.full((global_nk, global_nj, global_ni), np.nan)
    max_error = -np.inf

    for rank in range(mpi_size):
        with open(f"Var0_rank_{rank}.json", "r") as f:
            d = json.load(f)

        nz = d.get('nz', 1)  # in 2d case, set nz=1
        ny = d['ny']
        nx = d['nx']
        x0 = d['x0']
        y0 = d['y0']
        z0 = d.get('z0', 0)
        dx = d['dx']
        dy = d['dy']
        dz = d.get('dz', 0)

        local_k0 = d.get("local_k0", 0)
        local_j0 = d["local_j0"]
        local_i0 = d["local_i0"]

        # Ghost metadata (defaults: no ghost zones = backward compatible)
        gs = d.get("gs", 0)
        lower_x_ghost = d.get("lower_x_ghost", False)
        upper_x_ghost = d.get("upper_x_ghost", False)
        lower_y_ghost = d.get("lower_y_ghost", False)
        upper_y_ghost = d.get("upper_y_ghost", False)
        lower_z_ghost = d.get("lower_z_ghost", False)
        upper_z_ghost = d.get("upper_z_ghost", False)

        local_data = np.array(d["data"])
        if len(local_data.shape) == 2:
            local_data = local_data.reshape((1,) + local_data.shape)

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
    gz = global_z0 + np.arange(global_nk) * d.get("dz", 0)
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
