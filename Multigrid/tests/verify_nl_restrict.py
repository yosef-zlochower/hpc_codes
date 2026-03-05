import numpy as np
import json
import sys

ROUNDOFF_TOLERANCE = 1.0e-12

"""
Verify the nonlinear restriction test.

The parent (fine) grid is filled with f = x*(1-x)*y*(1-y)*z*(1-z).
After inject + restrict, the child (coarse) grid should hold:

  Physical boundary child points (global index 0 or global_n-1 in any axis):
    F(X,Y,Z) = X*(1-X)*Y*(1-Y)*Z*(1-Z)  (= 0, since f vanishes on boundaries)

  Interior child points (full-weighting restriction):
    R[f](X,Y,Z) = [X*(1-X) - h^2/2] * [Y*(1-Y) - h^2/2] * [Z*(1-Z) - h^2/2]

where h = dx_child / 2  (the parent fine-grid spacing).

The 2D formula omits the Z factor.

Only owned (non-ghost) child points are checked.
"""


def _restrict1d(t, h2):
    """Analytic 1D restriction of g(t) = t*(1-t)."""
    return t * (1.0 - t) - h2 / 2.0


def verify_nl_restrict(tol=ROUNDOFF_TOLERANCE):
    with open("Var0_rank_0.json") as f:
        d0 = json.load(f)

    global_ni = d0["global_ni"]
    global_nj = d0["global_nj"]
    global_nk = d0.get("global_nk", 1)
    mpi_size = d0["mpi_size"]
    is_3d = "global_nk" in d0

    max_error = 0.0

    for rank in range(mpi_size):
        with open(f"Var0_rank_{rank}.json") as f:
            d = json.load(f)

        nx = d["nx"]
        ny = d["ny"]
        nz = d.get("nz", 1)
        dx = d["dx"]   # coarse grid spacing (= 2h)
        dy = d["dy"]
        dz = d.get("dz", 1.0)   # unused in 2D
        x0 = d["x0"]
        y0 = d["y0"]
        z0 = d.get("z0", 0.0)
        local_i0 = d["local_i0"]
        local_j0 = d["local_j0"]
        local_k0 = d.get("local_k0", 0)

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

        # Parent (fine) grid spacing per axis: h = dx_child / 2
        h2_x = (dx / 2.0) ** 2
        h2_y = (dy / 2.0) ** 2
        h2_z = (dz / 2.0) ** 2

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

            on_bdy = ((GII == 0) | (GII == global_ni - 1) |
                      (GJJ == 0) | (GJJ == global_nj - 1) |
                      (GKK == 0) | (GKK == global_nk - 1))
            restrict_val = (_restrict1d(XX, h2_x) *
                            _restrict1d(YY, h2_y) *
                            _restrict1d(ZZ, h2_z))
        else:
            XX  = X[np.newaxis, :]
            YY  = Y[:, np.newaxis]
            GII = GI[np.newaxis, :]
            GJJ = GJ[:, np.newaxis]

            on_bdy = ((GII == 0) | (GII == global_ni - 1) |
                      (GJJ == 0) | (GJJ == global_nj - 1))
            restrict_val = _restrict1d(XX, h2_x) * _restrict1d(YY, h2_y)

        # f = 0 on all physical boundaries (homogeneous Dirichlet)
        expected = np.where(on_bdy, 0.0, restrict_val)

        err = np.abs(owned - expected)
        local_max = float(np.max(err))
        if local_max > max_error:
            max_error = local_max

        if local_max > tol:
            idx = np.unravel_index(np.argmax(err), err.shape)
            ki, ji, ii = idx
            gi = int(GI[i_lo + ii]) if not is_3d else int(
                (local_i0 + local_i)[ii])
            print(f"Rank {rank}: max error = {local_max:.3e} at "
                  f"owned local ({ii},{ji},{ki})")
            return False

    print(f"Max error (nonlinear restriction): {max_error:.3e}")
    return True


def main():
    if not verify_nl_restrict():
        sys.exit(-1)
    print("Test passed")


if __name__ == "__main__":
    main()
