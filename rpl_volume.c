/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math_util.h"
#include "mha_io.h"
#include "proj_matrix.h"
#include "proton_dose.h"
#include "ray_trace_exact.h"
#include "ray_trace_uniform.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_limit.h"

//#define UNIFIED_DEPTH_OFFSET 1

#define VERBOSE 1

typedef struct callback_data Callback_data;
struct callback_data {
    Rpl_volume* rpl_vol;    /* Radiographic depth volume */
    int* ires;              /* Aperture Dimensions */
    int ap_idx;             /* Current Aperture Coord */
#if UNIFIED_DEPTH_OFFSET
    int step_offset;        /* Number of steps before first ray sample */
#endif
    double accum;           /* Accumulated intensity */
};

static double
lookup_rgdepth (
    Rpl_volume *rpl_vol, 
    int ap_x,
    int ap_y,
    double dist,
    float ray_step
)
{
    int idx1, idx2;
    int ijk[3];
    double rg1, rg2, rgdepth;
    float* d_img = (float*) rpl_vol->vol->img;

    if (dist < 0) {
        return d_img[0];
    }

    ijk[0] = ap_x;
    ijk[1] = ap_y;
    ijk[2] = (int)floorf(dist / ray_step);

    /* Depth behind point */
    idx1 = INDEX_OF (ijk, rpl_vol->vol->dim);
    if (idx1 < rpl_vol->vol->npix) {
        rg1 = d_img[idx1];
    } else {
        return 0.0f;
    }

    /* Depth in front of point */
    ijk[2]++;
    idx2 = INDEX_OF (ijk, rpl_vol->vol->dim);
    if (idx2 < rpl_vol->vol->npix) {
        rg2 = d_img[idx2];
    } else {
        rg2 = 0.0f;
    }

    dist = dist - floorf (dist);
    
    rgdepth = rg1 + dist * (rg2 - rg1);

    return rgdepth;
}

/* Lookup radiological path length from depth_vol */
double
rpl_volume_get_rgdepth (
    Rpl_volume *rpl_vol,   /* I: volume of radiological depths */
    double* ct_xyz,        /* I: location of voxel in world space */
    Proj_matrix *pmat
)
{
    int ap_x, ap_y, ap_idx;
    double ap_xy[3], ap_xyz[3], tmp[3];
    double dist, rgdepth;
    int ires[2];

    /* A couple of abbreviations */
    ires[0] = rpl_vol->vol->dim[0];
    ires[1] = rpl_vol->vol->dim[1];

    /* Back project the voxel to the aperture plane */
    mat43_mult_vec3 (ap_xy, pmat->matrix, ct_xyz);
    ap_xy[0] = pmat->ic[0] + ap_xy[0] / ap_xy[2];
    ap_xy[1] = pmat->ic[1] + ap_xy[1] / ap_xy[2];

    ap_x = ROUND_INT (ap_xy[0]);
    ap_y = ROUND_INT (ap_xy[1]);

    /* Only handle voxels inside the (square) aperture */
    if (ap_x < 0 || ap_x >= ires[0] ||
        ap_y < 0 || ap_y >= ires[1]) {
        return -1;
    }

    ap_idx = ap_y * ires[0] + ap_x;

    /* Convert aperture indices into space coords */
    vec3_copy (ap_xyz, rpl_vol->ap_ul_room);
    vec3_scale3 (tmp, rpl_vol->incr_r, ap_x);
    vec3_add2 (ap_xyz, tmp);
    vec3_scale3 (tmp, rpl_vol->incr_c, ap_y);
    vec3_add2 (ap_xyz, tmp);

    /* Compute distance from aperture to voxel */
    dist = vec3_dist (ap_xyz, ct_xyz);
#if UNIFIED_DEPTH_OFFSET
    dist -= rpl_vol->depth_offset[0];
#else
    dist -= rpl_vol->depth_offset[ap_idx];
#endif

    /* Retrieve the radiographic depth */
    rgdepth = lookup_rgdepth (
    rpl_vol, 
    ap_x, ap_y,
    dist,
    rpl_vol->ray_step);

    return rgdepth;
}

/* Lookup radiological path length from depth_vol.  Note, we do not 
   clip to the aperture.  This lets scatter locations find the nearest 
   pencil beams. */
static void
get_depth_vol_idx (
    double *dv_ijk,        /* O: The 3d index of the voxel within depth vol */
    double* ct_xyz,        /* I: location of voxel in world space */
    double* depth_offset,  /* I: distance to first slice of depth_vol */
    Volume* depth_vol,     /* I: the volume of RPL values */
    Proj_matrix *pmat,
    int* ires,
    double* ap_ul,
    double* incr_r,
    double* incr_c,
    Proton_dose_options *options
)
{
    int ap_x, ap_y, ap_idx;
    double ap_xy[3], ap_xyz[3], tmp[3];
    double dist;
    //double rgdepth;

    /* Back project the voxel to the aperture plane */
    mat43_mult_vec3 (ap_xy, pmat->matrix, ct_xyz);

    /* Compute x & y coordinates */
    dv_ijk[0] = pmat->ic[0] + ap_xy[0] / ap_xy[2];
    dv_ijk[1] = pmat->ic[1] + ap_xy[1] / ap_xy[2];

    /* Find real-world position on aperture plane */
    vec3_copy (ap_xyz, ap_ul);
    vec3_scale3 (tmp, incr_r, dv_ijk[0]);
    vec3_add2 (ap_xyz, tmp);
    vec3_scale3 (tmp, incr_c, dv_ijk[1]);
    vec3_add2 (ap_xyz, tmp);

    /* Compute distance from aperture to voxel */
    dist = vec3_dist (ap_xyz, ct_xyz);

    /* GCS FIX:::: 
       The depth_volume has a different depth_offset for each ray.
       This makes hong algorithm implementation more complex.
       Therefore, at least we should move depth_volume to a separate 
       structure to simplify coding, and perhaps also make a second 
       implementation of depth_volume with uniform depth_offset.
    */

    /* Find aperture index of the ray */
    ap_x = ROUND_INT (dv_ijk[0]);
    ap_y = ROUND_INT (dv_ijk[1]);

#if defined (commentout)
    /* Only handle voxels that were hit by the beam */
    if (ap_x < 0 || ap_x >= ires[0] ||
        ap_y < 0 || ap_y >= ires[1]) {
        return -1;
    }
#endif

    ap_idx = ap_y * ires[0] + ap_x;

    /* Subtract distance from aperture to depth volume */
    dist -= depth_offset[ap_idx];

    /* Convert mm to index */
    dv_ijk[2] = dist / options->ray_step;
}

static float
lookup_attenuation_weq (float density)
{
    const double min_hu = -1000.0;
    if (density <= min_hu) {
        return 0.0;
    } else {
        return ((density + 1000.0)/1000.0);
    }
}

static float
lookup_attenuation (float density)
{
    return lookup_attenuation_weq (density);
}

Rpl_volume*
rpl_volume_create (
    Volume* ct_vol,       // ct volume
    int ires[2],          // aperture dimensions
    double cam[3],        // position of source
    double ap_ul_room[3], // position of aperture in room coords
    double incr_r[3],     // change in room coordinates for each ap pixel
    double incr_c[3],     // change in room coordinates for each ap pixel
    float ray_step        // uniform ray step size
)
{
    int dv_dims[3];
    float dv_off[3] = {0.0f, 0.0f, 0.0f};   // arbitrary
    float dv_ps[3] = {1.0f, 1.0f, 1.0f};    //
    float ct_diag;
    float ct_dims_mm[3];
    Rpl_volume *rvol;

    rvol = (Rpl_volume*) malloc (sizeof (Rpl_volume));
    memset (rvol, 0, sizeof (Rpl_volume));

    /* Copy over input fields */
    memcpy (rvol->cam, cam, 3 * sizeof(double));
    memcpy (rvol->ap_ul_room, ap_ul_room, 3 * sizeof(double));
    memcpy (rvol->incr_r, incr_r, 3 * sizeof(double));
    memcpy (rvol->incr_c, incr_c, 3 * sizeof(double));
    rvol->ray_step = (double) ray_step;

    /* Holds distance from aperture to CT_vol entry point for each ray */
#if UNIFIED_DEPTH_OFFSET
    rvol->depth_offset = (double*) malloc (sizeof(double));
    rvol->depth_offset[0] = 0.0;
#else
    rvol->depth_offset = (double*) malloc (ires[0] * ires[1] * sizeof(double));
    memset (rvol->depth_offset, 0, ires[0] * ires[1] * sizeof(double));
#endif

    ct_dims_mm[0] = ct_vol->dim[0] * ct_vol->pix_spacing[0];
    ct_dims_mm[1] = ct_vol->dim[1] * ct_vol->pix_spacing[1];
    ct_dims_mm[2] = ct_vol->dim[2] * ct_vol->pix_spacing[2];

    ct_diag =  ct_dims_mm[0]*ct_dims_mm[0];
    ct_diag += ct_dims_mm[1]*ct_dims_mm[1];
    ct_diag += ct_dims_mm[2]*ct_dims_mm[2];
    ct_diag = sqrt (ct_diag);

    dv_dims[0] = ires[0];   // rows = aperture rows
    dv_dims[1] = ires[1];   // cols = aperture cols
    dv_dims[2] = (int) floorf (ct_diag + 0.5) / ray_step;

    rvol->vol = volume_create (dv_dims, dv_off, dv_ps, PT_FLOAT, NULL, 0);

    return rvol;
}

void
rpl_volume_destroy (Rpl_volume *rpl_vol)
{
    free (rpl_vol->depth_offset);
    volume_destroy (rpl_vol->vol);
    free (rpl_vol);
}

void
rpl_volume_save (Rpl_volume *rpl_vol, char *filename)
{
    write_mha (filename, rpl_vol->vol);
}

static
void
proton_dose_ray_trace_callback (
    void *callback_data, 
    int vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    float *depth_img = (float*) rpl_vol->vol->img;
    int ap_idx = cd->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
#if UNIFIED_DEPTH_OFFSET
    int step_num = vox_index + cd->step_offset;
#else
    int step_num = vox_index;
#endif

    cd->accum += vox_len * lookup_attenuation (vox_value);

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

static
void
proton_dose_ray_trace (
    Rpl_volume *rpl_vol,         /* O: radiographic depths */
    Volume *ct_vol,              /* I: CT volume */
    Volume_limit *vol_limit,     /* I: CT bounding region */
    double *p1,                  /* I: @ source */
    double *p2,                  /* I: @ aperture */
    int* ires,                   /* I: ray cast resolution */
    int ap_idx                   /* I: linear index of ray in ap */
)
{
    Callback_data cd;
    double ray[3];
    double ip1[3];
    double ip2[3];

    /* Define unit vector in ray direction */
    vec3_sub3 (ray, p2, p1);
    vec3_normalize1 (ray);

    /* Test if ray intersects volume and create intersection points */
    if (!volume_limit_clip_ray (vol_limit, ip1, ip2, p1, ray)) {
        return;
    }

#if (!UNIFIED_DEPTH_OFFSET)
    /* store the distance from aperture to CT_vol for later */
    rpl_vol->depth_offset[ap_idx] = vec3_dist (p2, ip1);
#endif

#if VERBOSE
    printf ("ap_idx: %d\n", ap_idx);
    printf ("P1: %g %g %g\n", p1[0], p1[1], p1[2]);
    printf ("P2: %g %g %g\n", p2[0], p2[1], p2[2]);

    printf ("ip1 = %g %g %g\n", ip1[0], ip1[1], ip1[2]);
    printf ("ip2 = %g %g %g\n", ip2[0], ip2[1], ip2[2]);
    printf ("ray = %g %g %g\n", ray[0], ray[1], ray[2]);
    printf ("off = %g\n", rpl_vol->depth_offset[ap_idx]);
#endif

    /* init callback data for this ray */
    cd.accum = 0.0f;
    cd.ires = ires;
    cd.rpl_vol = rpl_vol;
    cd.ap_idx = ap_idx;

#if (UNIFIED_DEPTH_OFFSET)
    /* account for distance between depth_offset and intersection with 
       volume boundary */
    {
	double tmp[3];
	double dist;
	int num_steps;

	/* Compute distance from depth_offset to volume boundary */
	dist = vec3_dist (ip1, p2);
	printf ("dist = %g, depth_off = %g\n", 
	    dist, rpl_vol->depth_offset[0]);
	dist = dist - rpl_vol->depth_offset[0];

	/* Figure out how many steps to first step within volume */
	cd.step_offset = (int) ceil (dist / rpl_vol->ray_step);
	printf ("step_offset = %d\n", cd.step_offset);
	
	/* Find location of first step within volume */
	vec3_scale3 (tmp, ray, num_steps * (double) rpl_vol->ray_step);
	vec3_add2 (ip1, tmp);
    }
#endif

    /* get radiographic depth along ray */
    printf ("Tracing ray...\n");
    ray_trace_uniform (
        ct_vol,                             // INPUT: CT volume
        vol_limit,                          // INPUT: CT volume bounding box
        &proton_dose_ray_trace_callback,    // INPUT: step action cbFunction
        &cd,                                // INPUT: callback data
        ip1,                                // INPUT: ray starting point
        ip2,                                // INPUT: ray ending point
        rpl_vol->ray_step);                 // INPUT: uniform ray step size
}

void
rpl_volume_compute_unified (
    Rpl_volume *rpl_vol,   /* I/O: this gets filled in with depth info */
    Volume *ct_vol         /* I:   the ct volume */
)
{
    int r;
    int ires[2];
    Volume_limit ct_limit;

    /* A couple of abbreviations */
    ires[0] = rpl_vol->vol->dim[0];
    ires[1] = rpl_vol->vol->dim[1];

    /* Compute volume boundary box */
    volume_limit_set (&ct_limit, ct_vol);

    /* Make two passes through the aperture grid.  The first pass 
       is used to find the offset_depth.  The second pass actually 
       traces the rays. */

    /* Scan through the aperture -- first pass */
    rpl_vol->depth_offset[0] = DBL_MAX;
    for (r = 0; r < ires[0]; r++) {
        int c;
        double r_tgt[3];
        double tmp[3];
        double p2[3];

        //if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);
        vec3_copy (r_tgt, rpl_vol->ap_ul_room);
        vec3_scale3 (tmp, rpl_vol->incr_r, (double) r);
        vec3_add2 (r_tgt, tmp);

        for (c = 0; c < ires[1]; c++) {
            int ap_idx;
	    double ray[3];
	    double ip1[3];
	    double ip2[3];
	    double dist;

            /* Compute index of aperture pixel */
            ap_idx = c * ires[0] + r;

            /* Compute p2 = 3d coordinates of point on aperture */
            vec3_scale3 (tmp, rpl_vol->incr_c, (double) c);
            vec3_add3 (p2, r_tgt, tmp);

	    /* Define unit vector in ray direction */
	    vec3_sub3 (ray, p2, rpl_vol->cam);
	    vec3_normalize1 (ray);

	    /* Test if ray intersects volume and create intersection points */
	    if (!volume_limit_clip_ray (&ct_limit, ip1, ip2, 
		    rpl_vol->cam, ray))
	    {
		return;
	    }

	    /* store the distance from aperture to CT_vol for later */
	    dist = vec3_dist (p2, ip1);
	    if (dist < rpl_vol->depth_offset[0]) {
		rpl_vol->depth_offset[0] = dist;
	    }
        }
    }

    /* Scan through the aperture -- second pass */
    for (r = 0; r < ires[0]; r++) {
        int c;
        double r_tgt[3];
        double tmp[3];
        double p2[3];

        //if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);
        vec3_copy (r_tgt, rpl_vol->ap_ul_room);
        vec3_scale3 (tmp, rpl_vol->incr_r, (double) r);
        vec3_add2 (r_tgt, tmp);

        for (c = 0; c < ires[1]; c++) {
            int ap_idx;

            /* Compute index of aperture pixel */
            ap_idx = c * ires[0] + r;

            /* Compute p2 = 3d coordinates of point on aperture */
            vec3_scale3 (tmp, rpl_vol->incr_c, (double) c);
            vec3_add3 (p2, r_tgt, tmp);

            proton_dose_ray_trace (
                rpl_vol,      /* O: radiographic depths */
                ct_vol,       /* I: CT volume */
                &ct_limit,    /* I: CT bounding region */
                rpl_vol->cam, /* I: @ source */
                p2,           /* I: @ aperture */
                ires,         /* I: ray cast resolution */
                ap_idx        /* I: linear index of ray in ap */
                );
        }
    }
}

void
rpl_volume_compute_separate (
    Rpl_volume *rpl_vol,   /* I/O: this gets filled in with depth info */
    Volume *ct_vol         /* I:   the ct volume */
)
{
    int r;
    int ires[2];
    Volume_limit ct_limit;

    /* A couple of abbreviations */
    ires[0] = rpl_vol->vol->dim[0];
    ires[1] = rpl_vol->vol->dim[1];

    /* Compute volume boundary box */
    volume_limit_set (&ct_limit, ct_vol);

    /* Scan through the aperture */
    for (r = 0; r < ires[0]; r++) {
        int c;
        double r_tgt[3];
        double tmp[3];
        double p2[3];

        //if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);
        vec3_copy (r_tgt, rpl_vol->ap_ul_room);
        vec3_scale3 (tmp, rpl_vol->incr_r, (double) r);
        vec3_add2 (r_tgt, tmp);

        for (c = 0; c < ires[1]; c++) {
            int ap_idx;

            /* Compute index of aperture pixel */
            ap_idx = c * ires[0] + r;

            /* Compute p2 = 3d coordinates of point on aperture */
            vec3_scale3 (tmp, rpl_vol->incr_c, (double) c);
            vec3_add3 (p2, r_tgt, tmp);

            proton_dose_ray_trace (
                rpl_vol,      /* O: radiographic depths */
                ct_vol,       /* I: CT volume */
                &ct_limit,    /* I: CT bounding region */
                rpl_vol->cam, /* I: @ source */
                p2,           /* I: @ aperture */
                ires,         /* I: ray cast resolution */
                ap_idx        /* I: linear index of ray in ap */
                );
        }
    }
}

void
rpl_volume_compute (
    Rpl_volume *rpl_vol,   /* I/O: this gets filled in with depth info */
    Volume *ct_vol         /* I:   the ct volume */
)
{
#if UNIFIED_DEPTH_OFFSET
    rpl_volume_compute_unified (rpl_vol, ct_vol);
#else
    rpl_volume_compute_separate (rpl_vol, ct_vol);
#endif
}
