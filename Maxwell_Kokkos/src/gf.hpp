#ifndef GF_HPP
#define GF_HPP
/* View-owning grid-function structures.
 *
 * Each Field3D is a Kokkos::View<double***, LayoutLeft>, sized
 *   (nx, ny, nz)   with i fastest.
 * That layout matches the flat offset used by the C-version stencils:
 *   ijk = i + j*nx + k*nx*ny  ==  &V(i,j,k) - V.data().
 * So the existing pointer-and-stride stencils in derivatives.hpp and the
 * SAT macros in simple_maxwell.h work without modification — every
 * lambda captures   double *Bx = ev.state.data()   and indexes by [ijk].
 *
 * Tradeoff vs. multi-dim Kokkos accessors: keeps the SymPy-generated
 * SAT macros in simple_maxwell.h verbatim; loses some generality
 * (the layout is fixed to LayoutLeft).  We pin LayoutLeft so the i-fast
 * coalescing pattern is the same on CPU OpenMP and CUDA.
 */

#include <Kokkos_Core.hpp>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

extern "C" {
#include "domain.h"
}

using Field3D = Kokkos::View<double***, Kokkos::LayoutLeft>;
using Field1D = Kokkos::View<double*,   Kokkos::LayoutLeft>;

/* The RHS kernels and SAT macros capture raw pointers from these Views
 * and index them by hand as  ijk = i + j*nx + k*nx*ny  (i fastest).
 * That arithmetic is only correct for LayoutLeft; a layout change would
 * silently corrupt every stencil with no compile error.  Fail loudly
 * instead. */
static_assert(std::is_same<Field3D::array_layout,
                           Kokkos::LayoutLeft>::value,
              "Field3D must be LayoutLeft: stencil pointer arithmetic "
              "assumes i-fastest (ijk = i + j*nx + k*nx*ny).");
static_assert(std::is_same<Field1D::array_layout,
                           Kokkos::LayoutLeft>::value,
              "Field1D must be LayoutLeft.");

/* Evolved variable: state + RK4 stage buffers + previous-step snapshot. */
struct EvolField
{
    Field3D state;   /* current "new" */
    Field3D old_;    /* pre-stage snapshot used for new = old + c*dt*K */
    Field3D K[4];    /* K1, K2, K3, K4 */
    std::string vname;
};

struct AuxField
{
    Field3D state;
    std::string vname;
};

/* Per-axis ghost-exchange buffers.  Device buffers are always allocated;
 * the host mirrors are cheap aliases on CPU backends and explicit host
 * memory on CUDA backends (only consulted when not using CUDA-aware MPI). */
struct CommAxis
{
    size_t face_size = 0;        /* doubles per face per variable */
    Field1D src_lo_dev, dst_lo_dev;
    Field1D src_up_dev, dst_up_dev;
    Field1D::host_mirror_type src_lo_host, dst_lo_host;
    Field1D::host_mirror_type src_up_host, dst_up_host;
};

struct NGFS
{
    int n_evol_vars = 0;
    int n_aux_vars  = 0;
    double x0 = 0.0, y0 = 0.0, z0 = 0.0;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    int64_t n_tot = 0;
    int64_t nx = 0, ny = 0, nz = 0;
    int gs = 0;
    EvolField *evol = nullptr;     /* length n_evol_vars (slot-indexed) */
    AuxField  *aux  = nullptr;     /* length n_aux_vars  (slot-indexed) */
    struct domain3d_st domain = {};
    CommAxis comm_x, comm_y, comm_z;
};

/* Allocate Views for n_evol evolved + n_aux auxiliary fields, plus
 * communication buffers for the largest face × max(n_evol, n_aux). */
int  ngfs_allocate(int n_evol, int n_aux, NGFS *gfs);
int  ngfs_deallocate(NGFS *gfs);
void gf_rename_evol(NGFS *gfs, int slot, const char *name);
void gf_rename_aux (NGFS *gfs, int slot, const char *name);

/* Linear index helper that matches LayoutLeft offset arithmetic.
 * Identical to the C version's ijk_indx so the existing macros and
 * stencils keep working. */
KOKKOS_INLINE_FUNCTION
int64_t ijk_indx(int64_t i, int64_t j, int64_t k, int64_t nx, int64_t ny)
{
    return i + j * nx + k * nx * ny;
}

#endif /* GF_HPP */
