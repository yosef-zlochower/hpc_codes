#include "rk4.hpp"
#include "timer.h"
#include <Kokkos_Core.hpp>

/* Classical RK4 across all evolved fields.
 *
 *   stage 1:  K1 = f(t0,        u_n)         u_a = u_n + dt/2 * K1
 *   stage 2:  K2 = f(t0+dt/2,   u_a)         u_b = u_n + dt/2 * K2
 *   stage 3:  K3 = f(t0+dt/2,   u_b)         u_c = u_n + dt   * K3
 *   stage 4:  K4 = f(t0+dt,     u_c)
 *   final:    u_{n+1} = u_n + dt/6 * (K1 + 2 K2 + 2 K3 + K4)
 *
 * The C code reused `state` as the per-stage input buffer (writing
 * intermediate u_a / u_b / u_c into the same `new` array as where K_i
 * was read from). We do the same here: between stages, `state` holds
 * the stage-input value and `old_` holds the unchanged u_n.
 */

using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

int RK4_Step(NGFS *gfs, double t0, double dt, dotfunction time_deriv)
{
    static int t_copy   = -1;
    static int t_post1  = -1, t_post2 = -1, t_post3 = -1, t_post4 = -1;
    if (t_copy < 0)
    {
        t_copy  = register_timer("/evol/rk/copy");
        t_post1 = register_timer("/evol/rk/post_k1");
        t_post2 = register_timer("/evol/rk/post_k2");
        t_post3 = register_timer("/evol/rk/post_k3");
        t_post4 = register_timer("/evol/rk/post_k4");
    }

    const int     nvars = gfs->n_evol_vars;
    const int64_t nx = gfs->nx, ny = gfs->ny, nz = gfs->nz;
    Range3D pol({0, 0, 0}, {nx, ny, nz});

    /* Snapshot state into old_. */
    BEGIN_TIMER(t_copy)
    {
        for (int v = 0; v < nvars; v++)
        {
            Kokkos::deep_copy(gfs->evol[v].old_, gfs->evol[v].state);
        }
    }
    END_TIMER(t_copy)

    /* ── stage 1 ─────────────────────────────────────────────────── */
    time_deriv(gfs, t0, /*kidx=*/0);

    BEGIN_TIMER(t_post1)
    {
        const double half_dt = 0.5 * dt;
        for (int v = 0; v < nvars; v++)
        {
            auto S  = gfs->evol[v].state;
            auto O  = gfs->evol[v].old_;
            auto K1 = gfs->evol[v].K[0];
            Kokkos::parallel_for("rk4_post1", pol,
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    S(i, j, k) = O(i, j, k) + half_dt * K1(i, j, k);
                });
        }
    }
    END_TIMER(t_post1)

    /* ── stage 2 ─────────────────────────────────────────────────── */
    time_deriv(gfs, t0 + 0.5 * dt, /*kidx=*/1);

    BEGIN_TIMER(t_post2)
    {
        const double half_dt = 0.5 * dt;
        for (int v = 0; v < nvars; v++)
        {
            auto S  = gfs->evol[v].state;
            auto O  = gfs->evol[v].old_;
            auto K2 = gfs->evol[v].K[1];
            Kokkos::parallel_for("rk4_post2", pol,
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    S(i, j, k) = O(i, j, k) + half_dt * K2(i, j, k);
                });
        }
    }
    END_TIMER(t_post2)

    /* ── stage 3 ─────────────────────────────────────────────────── */
    time_deriv(gfs, t0 + 0.5 * dt, /*kidx=*/2);

    BEGIN_TIMER(t_post3)
    {
        for (int v = 0; v < nvars; v++)
        {
            auto S  = gfs->evol[v].state;
            auto O  = gfs->evol[v].old_;
            auto K3 = gfs->evol[v].K[2];
            Kokkos::parallel_for("rk4_post3", pol,
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    S(i, j, k) = O(i, j, k) + dt * K3(i, j, k);
                });
        }
    }
    END_TIMER(t_post3)

    /* ── stage 4 + final combine ────────────────────────────────── */
    time_deriv(gfs, t0 + dt, /*kidx=*/3);

    BEGIN_TIMER(t_post4)
    {
        const double sixth_dt = dt / 6.0;
        for (int v = 0; v < nvars; v++)
        {
            auto S  = gfs->evol[v].state;
            auto O  = gfs->evol[v].old_;
            auto K1 = gfs->evol[v].K[0];
            auto K2 = gfs->evol[v].K[1];
            auto K3 = gfs->evol[v].K[2];
            auto K4 = gfs->evol[v].K[3];
            Kokkos::parallel_for("rk4_final", pol,
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    S(i, j, k) = O(i, j, k)
                               + sixth_dt * (K1(i, j, k) + 2.0 * K2(i, j, k)
                                            + 2.0 * K3(i, j, k) + K4(i, j, k));
                });
        }
    }
    END_TIMER(t_post4)
    return 0;
}
