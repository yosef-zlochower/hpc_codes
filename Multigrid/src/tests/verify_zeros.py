import numpy as np
import json
import sys

ROUNDOFF_TOLERANCE = 1.0e-12

"""
Verify the prolongation test.

After fill_parent(f) -> inject+restrict -> child -> prolong -> parent, the
parent array should be zero everywhere EXCEPT at physical boundary points,
which are never updated by the prolongation stencil (prolong preserves all
Dirichlet boundary values on both the lower and upper faces of every axis).

For each owned (non-ghost) point, the check is:
  - If the point lies on any global physical boundary
    (global_ix == 0 or gni-1, global_jy == 0 or gnj-1,
     3D: global_kz == 0 or gnk-1): skip
  - Otherwise: |value| must be < tol
"""


def verify_zeros(tol=ROUNDOFF_TOLERANCE):
    with open("Var0_rank_0.json", "r") as f:
        d0 = json.load(f)
    mpi_size = d0["mpi_size"]

    max_error = 0.0

    for rank in range(mpi_size):
        with open(f"Var0_rank_{rank}.json", "r") as f:
            d = json.load(f)

        is_3d = "nz" in d
        nz = d.get("nz", 1)
        ny = d["ny"]
        nx = d["nx"]

        local_k0 = d.get("local_k0", 0)
        local_j0 = d["local_j0"]
        local_i0 = d["local_i0"]

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

        # Owned (non-ghost) index ranges
        i_lo = gs if lower_x_ghost else 0
        i_hi = nx - (gs if upper_x_ghost else 0)
        j_lo = gs if lower_y_ghost else 0
        j_hi = ny - (gs if upper_y_ghost else 0)
        k_lo = gs if lower_z_ghost else 0
        k_hi = nz - (gs if upper_z_ghost else 0)

        for k in range(k_lo, k_hi):
            global_kz = local_k0 + k
            for j in range(j_lo, j_hi):
                global_jy = local_j0 + j
                for i in range(i_lo, i_hi):
                    global_ix = local_i0 + i
                    # Skip all physical boundary points (lo and hi on every axis):
                    # prolong never updates these, so they retain their original f values.
                    if global_ix == 0 or global_ix == global_ni - 1:
                        continue
                    if global_jy == 0 or global_jy == global_nj - 1:
                        continue
                    if is_3d and (global_kz == 0 or global_kz == global_nk - 1):
                        continue
                    val = local_data[k, j, i]
                    err = abs(val)
                    if err > max_error:
                        max_error = err
                    if err > tol:
                        print(f"Rank {rank}: non-zero at "
                              f"(global {global_ix},{global_jy},{global_kz}) "
                              f"local ({i},{j},{k}): value={val:.6e}")
                        return False

    print(f"Max error (interior after prolongation): {max_error:.3e}")
    return True


def main():
    if not verify_zeros(tol=1.0e-14):
        sys.exit(-1)
    print("Test passed")


if __name__ == "__main__":
    main()
