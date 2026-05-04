"""
Verify domain decomposition and ghost-zone synchronisation for the Maxwell
test suite.  Reads per-rank JSON files output by test_sync, reconstructs
the global grid from owned (non-ghost) points, and checks all data
(including ghost zones) against the expected test function (var 0):

    f(x, y, z) = sin(2*pi*1*x) + cos(2*pi*2*y) + sin(2*pi*3*z)

This function is periodic with period 1 in all directions, so ghost zone
values are correct at their local coordinates regardless of boundary type.
"""
import numpy as np
import json
import sys

ROUNDOFF_TOLERANCE = 1.0e-12


def verify_domain_and_sync(tol=ROUNDOFF_TOLERANCE):
    with open("Var0_rank_0.json", "r") as f:
        d = json.load(f)

    global_ni = d["global_ni"]
    global_nj = d["global_nj"]
    global_nk = d["global_nk"]
    global_x0 = d["global_x0"]
    global_y0 = d["global_y0"]
    global_z0 = d["global_z0"]
    dx = d["dx"]
    dy = d["dy"]
    dz = d["dz"]
    mpi_size = d["mpi_size"]

    test_array = np.full((global_nk, global_nj, global_ni), np.nan)
    max_error = -np.inf

    for rank in range(mpi_size):
        with open(f"Var0_rank_{rank}.json", "r") as f:
            d = json.load(f)

        nz = d["nz"]
        ny = d["ny"]
        nx = d["nx"]
        x0 = d["x0"]
        y0 = d["y0"]
        z0 = d["z0"]
        dx = d["dx"]
        dy = d["dy"]
        dz = d["dz"]

        local_k0 = d["local_k0"]
        local_j0 = d["local_j0"]
        local_i0 = d["local_i0"]

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

        # Build expected array: f_0(x,y,z) = sin(2π·1·x) + cos(2π·2·y) + sin(2π·3·z)
        # (var 0: kx=1, ky=2, kz=3)
        local_x = x0 + np.arange(nx) * dx
        local_y = y0 + np.arange(ny) * dy
        local_z = z0 + np.arange(nz) * dz
        expected = (np.sin(2*np.pi * 1 * local_x[np.newaxis, np.newaxis, :])
                    + np.cos(2*np.pi * 2 * local_y[np.newaxis, :, np.newaxis])
                    + np.sin(2*np.pi * 3 * local_z[:, np.newaxis, np.newaxis]))

        # Check ALL data including ghost zones
        local_error = np.max(np.abs(local_data - expected))
        if local_error > max_error:
            max_error = local_error

        if local_error > tol:
            print(f"Test Failure for rank {rank}: max error = {local_error:.3e}")
            return False

        # Owned (non-ghost) region for global assembly
        i_lo = gs if lower_x_ghost else 0
        i_hi = nx - (gs if upper_x_ghost else 0)
        j_lo = gs if lower_y_ghost else 0
        j_hi = ny - (gs if upper_y_ghost else 0)
        k_lo = gs if lower_z_ghost else 0
        k_hi = nz - (gs if upper_z_ghost else 0)

        # Map to global indices (handle periodic wrapping)
        gi_lo = local_i0 + i_lo
        gj_lo = local_j0 + j_lo
        gk_lo = local_k0 + k_lo

        gi_hi = local_i0 + i_hi
        gj_hi = local_j0 + j_hi
        gk_hi = local_k0 + k_hi

        # For periodic boundaries, local_i0 can be negative
        # Clip to global array (points outside are wrapped ghosts)
        gi_lo_c = max(0, gi_lo)
        gj_lo_c = max(0, gj_lo)
        gk_lo_c = max(0, gk_lo)
        gi_hi_c = min(global_ni, gi_hi)
        gj_hi_c = min(global_nj, gj_hi)
        gk_hi_c = min(global_nk, gk_hi)

        li_lo = gi_lo_c - local_i0
        lj_lo = gj_lo_c - local_j0
        lk_lo = gk_lo_c - local_k0
        li_hi = gi_hi_c - local_i0
        lj_hi = gj_hi_c - local_j0
        lk_hi = gk_hi_c - local_k0

        if gi_hi_c > gi_lo_c and gj_hi_c > gj_lo_c and gk_hi_c > gk_lo_c:
            test_array[gk_lo_c:gk_hi_c, gj_lo_c:gj_hi_c, gi_lo_c:gi_hi_c] = \
                local_data[lk_lo:lk_hi, lj_lo:lj_hi, li_lo:li_hi]

    print(f"Max local error (all data including ghost zones): {max_error:.3e}")

    # Check global assembly
    if np.isnan(test_array).any():
        nan_count = np.isnan(test_array).sum()
        print(f"Domain Test Failure: global array has {nan_count} unfilled (NaN) points")
        return False

    # Build global expected
    gx = global_x0 + np.arange(global_ni) * d["dx"]
    gy = global_y0 + np.arange(global_nj) * d["dy"]
    gz = global_z0 + np.arange(global_nk) * d["dz"]
    global_expected = (np.sin(2*np.pi * 1 * gx[np.newaxis, np.newaxis, :])
                       + np.cos(2*np.pi * 2 * gy[np.newaxis, :, np.newaxis])
                       + np.sin(2*np.pi * 3 * gz[:, np.newaxis, np.newaxis]))

    global_error = np.max(np.abs(test_array - global_expected))
    print(f"Global error (owned data assembled): {global_error:.3e}")

    if global_error > tol or max_error > tol:
        print("Sync Test Failure")
        return False

    return True


def main():
    res = verify_domain_and_sync(tol=1.0e-12)
    if not res:
        sys.exit(-1)
    print("Test passed")


if __name__ == "__main__":
    main()
