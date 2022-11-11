/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "plm_math.h"
#include "ray_trace.h"
#include "volume.h"
#include "volume_limit.h"

void
ray_trace_exact_init_loopvars (
    plm_long* ai,      /* Output: index */
    int* aidir,        /* Output: are indices moving up or down? */
    double* ao,        /* Output: absolute length to next voxel crossing */
    double* al,        /* Output: length between voxel crossings */
    double pt,         /* Input:  initial intersection of ray with volume */
    double ry,         /* Input:  normalized direction of ray */
    double origin,     /* Input:  origin of volume */
    plm_long dim,      /* Input:  dimension of volume */
    double samp        /* Input:  pixel spacing of volume */
)
{
#if DRR_VERBOSE
    printf ("pt/ry/off/samp: %g %g %g %g\n", pt, ry, origin, samp);
#endif

    *aidir = SIGN (ry) * SIGN (samp);
    *ai = ROUND_INT ((pt - origin) / samp);
    *ai = clamp<int> (*ai, 0, (int) dim - 1);
    *ao = SIGN (ry)
	* (((*ai) * samp + origin) + (SIGN (ry) * 0.5 * fabs (samp)) - pt);

    if (fabs(ry) > DRR_STRIDE_TOLERANCE) {
	*ao = *ao / fabs(ry);
	*al = fabs(samp) / fabs(ry);
    } else {
	*ao = DRR_HUGE_DOUBLE;
	*al = DRR_HUGE_DOUBLE;
    }
}

/* Initialize loop variables.  Returns 1 if the segment intersects
   the volume, and 0 if the segment does not intersect. */
int
ray_trace_exact_init (
    plm_long current_idx[3],
    int travel_dir[3],
    double next_crossing[3],
    double crossing_dist[3],
    Volume* vol, 
    Volume_limit *vol_limit,
    double* p1, 
    double* p2 
)
{
    double ray[3];
    double ip1[3];
    double ip2[3];

    /* Test if ray intersects volume */
    if (!vol_limit->clip_segment (ip1, ip2, p1, p2)) {
	return 0;
    }

    /* Create the volume intersection points */
    vec3_sub3 (ray, p2, p1);
    vec3_normalize1 (ray);

#if defined (DRR_VERBOSE)
    printf ("ip1 = %g %g %g\n", ip1[0], ip1[1], ip1[2]);
    printf ("ip2 = %g %g %g\n", ip2[0], ip2[1], ip2[2]);
    printf ("ray = %g %g %g\n", ray[0], ray[1], ray[2]);
#endif

    /* We'll go from p1 to p2 */
    /* GCS FIX: This doesn't respect direction cosines */
    for (int d = 0; d < 3; d++) {
        ray_trace_exact_init_loopvars (
            &current_idx[d],
            &travel_dir[d],
            &next_crossing[d],
            &crossing_dist[d],
            ip1[d],
            ray[d],
            vol->origin[d],
            vol->dim[d],
            vol->spacing[d]);
    }
    return 1;
}

void
ray_trace_exact (
    Volume *vol,                  /* Input: volume */
    Volume_limit *vol_limit,      /* Input: min/max coordinates of volume */
    Ray_trace_callback callback,  /* Input: callback function */
    void *callback_data,          /* Input: callback function private data */
    double *p1in,                 /* Input: start point for ray */
    double *p2in                  /* Input: end point for ray */
)
{
    plm_long current_idx[3];
    int travel_dir[3];
    double next_crossing[3];      /* Length to next voxel crossing */
    double crossing_dist[3];      /* Spacing between crossings for this angle */
    float* img = (float*) vol->img;

#if defined (DRR_VERBOSE)
    printf ("p1in: %f %f %f\n", p1in[0], p1in[1], p1in[2]);
    printf ("p2in: %f %f %f\n", p2in[0], p2in[1], p2in[2]);
#endif

    if (!ray_trace_exact_init (
	    current_idx,
	    travel_dir,
	    next_crossing,
	    crossing_dist,
	    vol, 
	    vol_limit, 
	    p1in, 
	    p2in))
    {
	return;
    }

    int travel_limit[3] = { -1, -1, -1 };
    if (travel_dir[0] == 1) {
        travel_limit[0] = vol->dim[0];
    }
    if (travel_dir[1] == 1) {
        travel_limit[1] = vol->dim[1];
    }
    if (travel_dir[2] == 1) {
        travel_limit[2] = vol->dim[2];
    }
    
    /* We'll go from p1 to p2 */
    do {
        float pix_density;
        double pix_len;
        int index = vol->index (current_idx);

#if defined (DRR_ULTRA_VERBOSE)
        printf ("(%d %d %d) (%g,%g,%g)\n",current_idx[0],current_idx[1],current_idx[2],next_crossing[0],next_crossing[1],next_crossing[2]);
        fflush (stdout);
#endif
        pix_density = img[index];
        if ((next_crossing[0] < next_crossing[1]) && (next_crossing[0] < next_crossing[2])) {
            pix_len = next_crossing[0];
            next_crossing[1] -= next_crossing[0];
            next_crossing[2] -= next_crossing[0];
            next_crossing[0] = crossing_dist[0];
            current_idx[0] += travel_dir[0];
            if (current_idx[0] == travel_limit[0]) {
                break;
            }
        } else if ((next_crossing[1] < next_crossing[2])) {
            pix_len = next_crossing[1];
            next_crossing[0] -= next_crossing[1];
            next_crossing[2] -= next_crossing[1];
            next_crossing[1] = crossing_dist[1];
            current_idx[1] += travel_dir[1];
            if (current_idx[1] == travel_limit[1]) {
                break;
            }
        } else {
            pix_len = next_crossing[2];
            next_crossing[0] -= next_crossing[2];
            next_crossing[1] -= next_crossing[2];
            next_crossing[2] = crossing_dist[2];
            current_idx[2] += travel_dir[2];
            if (current_idx[2] == travel_limit[2]) {
                break;
            }
        }
        (*callback) (callback_data, index, pix_len, pix_density);

    } while (1);
}
