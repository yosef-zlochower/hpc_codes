import numpy as np
import json
import sys

ROUNDOFF_TOLERANCE = 1.0e-12

"""
Verify the nonlinear prolongation test.

The child (coarse) grid is filled with F = X*(1-X)*Y*(1-Y)*Z*(1-Z).
The parent (fine) grid starts at zero.  After prolong_var (which subtracts
P[F] from the parent), the parent holds  -P[F].

Expected value at a fine-grid point with global indices (global_ip, global_jp,
global_kp) and coordinates (x, y, z):

  Physical boundary points (global_ip == 0 or global_ip == global_ni-1,
  or global_jp == 0 or global_jp == global_nj-1,
  or (3D) global_kp == 0 or global_kp == global_nk-1):
    0.0   (prolongation skips all physical boundary points)

  All other points:
    -P[F](x,y,z)  where
      h  = dx_parent  (fine grid spacing)
      gx = x*(1-x),  gy = y*(1-y),  gz = z*(1-z)
      dx = global_ip % 2,  dy = global_jp % 2,  dz = global_kp % 2
      P[F] = (gx - dx*h^2) * (gy - dy*h^2) * (gz - dz*h^2)   [3D]
      P[F] = (gx - dx*h^2) * (gy - dy*h^2)                    [2D]

Only owned (non-ghost) parent points are checked.
"""


def _prolong1d(t, delta, h2):
    """Analytic 1D prolongation of G(T) = T*(1-T) at fine-grid point t.
    delta = 0 for even (coincident) index, 1 for odd (midpoint) index."""
    return t * (1.0 - t) - delta * h2


def verify_nl_prolong(tol=ROUNDOFF_TOLERANCE):
    with open("Var0_rank_0.json") as f:
        d0 = json.load(f)

    mpi_size = d0["mpi_size"]
    is_3d = "nz" in d0

    max_error = 0.0

    for rank in range(mpi_size):
        with open(f"Var0_rank_{rank}.json") as f:
            d = json.load(f)

        nx = d["nx"]
        ny = d["ny"]
        nz = d.get("nz", 1)
        dx = d["dx"]   # fine grid spacing = h
        dy = d["dy"]
        dz = d.get("dz", 1.0)   # unused in 2D
        x0 = d["x0"]
        y0 = d["y0"]
        z0 = d.get("z0", 0.0)
        local_i0 = d["local_i0"]
        local_j0 = d["local_j0"]
        local_k0 = d.get("local_k0", 0)
        # JSON now stores cell counts; convert to grid-point counts for the
        # vertex-centred Dirichlet layout (points = cells + 1).
        global_ni = d["global_cells_x"] + 1
        global_nj = d["global_cells_y"] + 1
        global_nk = d.get("global_cells_z", 0) + 1

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

        # Fine grid spacing squared per axis
        h2_x = dx * dx
        h2_y = dy * dy
        h2_z = dz * dz

        # Owned index ranges
        i_lo = gs if lower_x_ghost else 0
        i_hi = nx - (gs if upper_x_ghost else 0)
        j_lo = gs if lower_y_ghost else 0
        j_hi = ny - (gs if upper_y_ghost else 0)
        k_lo = gs if lower_z_ghost else 0
        k_hi = nz - (gs if upper_z_ghost else 0)

        owned = local_data[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]

        # Build coordinate and global-index arrays for owned region
        local_i = np.arange(i_lo, i_hi)
        local_j = np.arange(j_lo, j_hi)
        local_k = np.arange(k_lo, k_hi)

        X = x0 + local_i * dx   # (ni_owned,)
        Y = y0 + local_j * dy   # (nj_owned,)
        GI = local_i0 + local_i
        GJ = local_j0 + local_j

        if is_3d:
            Z  = z0 + local_k * dz  # (nk_owned,)
            GK = local_k0 + local_k

            XX  = X[np.newaxis, np.newaxis, :]
            YY  = Y[np.newaxis, :, np.newaxis]
            ZZ  = Z[:, np.newaxis, np.newaxis]
            GII = GI[np.newaxis, np.newaxis, :]
            GJJ = GJ[np.newaxis, :, np.newaxis]
            GKK = GK[:, np.newaxis, np.newaxis]

            delta_x = (GII % 2).astype(float)
            delta_y = (GJJ % 2).astype(float)
            delta_z = (GKK % 2).astype(float)

            P_F = (_prolong1d(XX, delta_x, h2_x) *
                   _prolong1d(YY, delta_y, h2_y) *
                   _prolong1d(ZZ, delta_z, h2_z))

            on_bdy = ((GII == 0) | (GII == global_ni - 1) |
                      (GJJ == 0) | (GJJ == global_nj - 1) |
                      (GKK == 0) | (GKK == global_nk - 1))
        else:
            XX  = X[np.newaxis, :]
            YY  = Y[:, np.newaxis]
            GII = GI[np.newaxis, :]
            GJJ = GJ[:, np.newaxis]

            delta_x = (GII % 2).astype(float)
            delta_y = (GJJ % 2).astype(float)

            P_F = (_prolong1d(XX, delta_x, h2_x) *
                   _prolong1d(YY, delta_y, h2_y))

            on_bdy = ((GII == 0) | (GII == global_ni - 1) |
                      (GJJ == 0) | (GJJ == global_nj - 1))

        # prolong skips all physical boundaries (leaves parent at 0);
        # all other points: parent = 0 - P[F] = -P[F]
        expected = np.where(on_bdy, 0.0, -P_F)

        err = np.abs(owned - expected)
        local_max = float(np.max(err))
        if local_max > max_error:
            max_error = local_max

        if local_max > tol:
            idx = np.unravel_index(np.argmax(err), err.shape)
            ki, ji, ii = idx
            print(f"Rank {rank}: max error = {local_max:.3e} at "
                  f"owned local ({ii},{ji},{ki})")
            return False

    print(f"Max error (nonlinear prolongation): {max_error:.3e}")
    return True


def main():
    if not verify_nl_prolong():
        sys.exit(-1)
    print("Test passed")


if __name__ == "__main__":
    main()
