#include "bc.h"
#include <stddef.h>

void bc_spec_homogenize(const struct bc_spec_t *src,
                        struct bc_spec_t *dst)
{
    if (!src || !dst) return;

    for (int f = 0; f < NUM_FACES; f++)
    {
        dst->face[f].kind        = src->face[f].kind;
        dst->face[f].homogeneous = true;
        dst->face[f].value       = NULL;
    }
}
