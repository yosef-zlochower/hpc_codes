import numpy as np
import h5py
import matplotlib.pyplot as plt
import evtk

MPI_SIZE = 100
all_vars = []
found = False
for rank in range(MPI_SIZE):
    try:
        with h5py.File(f"3D_rank_{rank}.h5") as file:
            attrs = file["metadata"].attrs
            # Cast to native Python types at read time. HDF5 returns numpy
            # scalars, and numpy >= 2.0 renders str(np.float64(x)) as
            # "np.float64(x)", which confuses the VTK XML writer.
            global_ni = int(attrs["global_ni"][:][0])
            global_nj = int(attrs["global_nj"][:][0])
            global_nk = int(attrs["global_nk"][:][0])

            mpi_size  = int(attrs["mpi_size"][:][0])
            ghost_size = int(attrs["ghost_zones"][:][0])

            dx = float(attrs["dx"][:][0])
            dy = float(attrs["dy"][:][0])
            dz = float(attrs["dz"][:][0])

            global_x0 = float(attrs["global_x0"][:][0])
            global_y0 = float(attrs["global_y0"][:][0])
            global_z0 = float(attrs["global_z0"][:][0])

            n_evol_vars = int(attrs["n_evol_vars"][:][0])
            n_aux_vars  = int(attrs["n_aux_vars"][:][0])

            for i in range(n_evol_vars):
                all_vars.append(attrs[f"var_{i}"][:][0].decode())
            for i in range(n_aux_vars):
                all_vars.append(attrs[f"aux_{i}"][:][0].decode())
            found = True
            Var = [None] * (n_evol_vars + n_aux_vars)
            for i in range(n_evol_vars + n_aux_vars):
                Var[i] = np.zeros((global_nk, global_nj, global_ni))
            break
    except OSError:
        continue
assert found

# loop over all iterations.
iteration = 0
while True:
    iteration_found = False

    for rank in range(mpi_size):
        try:
            with h5py.File(f"3D_rank_{rank}.h5") as file:
                attrs = file["metadata"].attrs
                local_ni = int(attrs["local_ni"][:][0])
                local_nj = int(attrs["local_nj"][:][0])
                local_nk = int(attrs["local_nk"][:][0])

                local_i0 = int(attrs["local_i0"][:][0])
                local_j0 = int(attrs["local_j0"][:][0])
                local_k0 = int(attrs["local_k0"][:][0])

                for i in range(n_evol_vars + n_aux_vars):
                    # Load processor local data from HDF5 file
                    lVar = file[f"{iteration}/" + all_vars[i]][:]

                    kkl = 0
                    kku = local_nk

                    jjl = 0
                    jju = local_nj

                    iil = 0
                    iiu = local_ni


                    kl = local_k0
                    ku = local_nk + local_k0

                    jl = local_j0
                    ju = local_nj + local_j0

                    il = local_i0
                    iu = local_ni + local_i0

                    if il < 0:
                        iil += -il
                        il = 0

                    if jl < 0:
                        jjl += -jl
                        jl = 0

                    if kl < 0:
                        kkl += -kl
                        kl = 0

                    if iu > global_ni:
                        iiu -= (iu - global_ni)
                        iu = global_ni

                    if ju > global_nj:
                        jju -= (ju - global_nj)
                        ju = global_nj
                    if ku > global_nk:
                        kku -= (ku - global_nk)
                        ku = global_nk




                    # store processor local data into a global grid
                    Var[i][kl:ku, jl:ju, il:iu] = lVar[kkl:kku,jjl:jju,iil:iiu]

            iteration_found = True

        except OSError:
            continue
        except KeyError:
            break

    if iteration_found == False:
        break

    # create dictionary of gridfunctions. Swap X and Z axes (fortran ordering to c-ordering)
    vars = {}
    for i, name in enumerate(all_vars):
        vars[name] = np.swapaxes(Var[i], axis1=0, axis2=2)

    # Output VTK files for use by, e.g., Visit
    evtk.hl.imageToVTK(
        f"./image{iteration:07d}",
        origin=(global_x0, global_y0, global_z0),
        spacing=(dx, dy, dz),
        pointData=vars,
    )
    iteration += 1
