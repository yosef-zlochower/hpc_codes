#include "rk4.h"
#include "gf.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Generic RK4. Updates nvars different variables at once
 * Arguments : Pointer to struct ngfs
 * time value (generally unused)
 * timestep size
 * pointer to function that calculates time derivatives
 *
 * Return value is always zero
 *
 * Each step below consists if two nested loops
 * Loop over variables:
 *   Loop over individual points
 *     update variable at individual point
 *
 * This routine sets var[*]->dot pointer  to point to the k1, k2, k3, or k4 arrays
 * (depending on which of the k's are about the be calculated).
 */
int RK4_Step(struct ngfs *gfs, const double t0, const double dt,
             dotfunction time_deriv)
{
    const int nvars = gfs->n_evol_vars;
    const int64_t n_pts = gfs->n_tot;

    static int timer_rk_copy = -1;
    static int timer_rk_post_1 = -1;
    static int timer_rk_post_2 = -1;
    static int timer_rk_post_3 = -1;
    static int timer_rk_post_4 = -1;
    if (timer_rk_copy < 0)
    {
        timer_rk_copy = register_timer("/evol/rk/copy");
        timer_rk_post_1 = register_timer("/evol/rk/post_k1");
        timer_rk_post_2 = register_timer("/evol/rk/post_k2");
        timer_rk_post_3 = register_timer("/evol/rk/post_k3");
        timer_rk_post_4 = register_timer("/evol/rk/post_k4");
    }

    /* At entry,  'new'  holds the current state. Copy it to  old
     * so each of the four RK stages can do new = old + c·dt·K_i,
     * then rebind  dot  to  K1  so  time_deriv() writes into  K1.
     */
    BEGIN_TIMER(timer_rk_copy)
    {
        for (int var = 0; var < nvars; var++)
        {
            gfs->vars[var]->dot = gfs->vars[var]->K1;
            for (int64_t idx = 0; idx < n_pts; idx++)
            {
                gfs->vars[var]->old[idx] = gfs->vars[var]->new[idx];
            }
        }
    }
    END_TIMER(timer_rk_copy)

    time_deriv(gfs, t0); /* this will fill in K1 */
    /* Step 1 of RK4. Also make sure dot points to K2 in preparation for
     * step 2
     */
    BEGIN_TIMER(timer_rk_post_1)
    {
        for (int var = 0; var < nvars; var++)
        {
            gfs->vars[var]->dot = gfs->vars[var]->K2;
            for (int64_t idx = 0; idx < n_pts; idx++)
            {
                gfs->vars[var]->new[idx] = gfs->vars[var]->old[idx] +
                                           0.5 * dt * gfs->vars[var]->K1[idx];
            }
        }
    }
    END_TIMER(timer_rk_post_1)

    time_deriv(gfs, t0 + 0.5 * dt); /* this will fill in K2 */
    /* Step 2 of RK4. Also make sure dot points to K3 in preparation for
     * step 3
     */
    BEGIN_TIMER(timer_rk_post_2)
    {
        for (int var = 0; var < nvars; var++)
        {
            gfs->vars[var]->dot = gfs->vars[var]->K3;
            for (int64_t idx = 0; idx < n_pts; idx++)
            {
                gfs->vars[var]->new[idx] = gfs->vars[var]->old[idx] +
                                           0.5 * dt * gfs->vars[var]->K2[idx];
            }
        }
    }
    END_TIMER(timer_rk_post_2)

    time_deriv(gfs, t0 + 0.5 * dt); /* this will fill in K3 */

    /* Step 3 of RK4. Also make sure dot points to K4 in preparation for
     * step 4
     */
    BEGIN_TIMER(timer_rk_post_3)
    {
        for (int var = 0; var < nvars; var++)
        {
            gfs->vars[var]->dot = gfs->vars[var]->K4;
            for (int64_t idx = 0; idx < n_pts; idx++)
            {
                gfs->vars[var]->new[idx] =
                    gfs->vars[var]->old[idx] + dt * gfs->vars[var]->K3[idx];
            }
        }
    }
    END_TIMER(timer_rk_post_3)

    time_deriv(gfs, t0 + dt); /* this will fill in K4 */
    /* final step */
    BEGIN_TIMER(timer_rk_post_4)
    {
        for (int var = 0; var < nvars; var++)
        {
            for (int64_t idx = 0; idx < n_pts; idx++)
            {
                gfs->vars[var]->new[idx] =
                    gfs->vars[var]->old[idx] +
                    1.0 / 6.0 *
                        dt *(gfs->vars[var]->K1[idx] +
                             2.0 * gfs->vars[var]->K2[idx] +
                             2.0 * gfs->vars[var]->K3[idx] +
                             gfs->vars[var]->K4[idx]);
            }
        }
    }
    END_TIMER(timer_rk_post_4)

    /* 'new' now contains the new value */

    return 0;
}
