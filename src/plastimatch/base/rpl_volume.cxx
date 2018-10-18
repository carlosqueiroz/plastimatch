/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>

#include "aperture.h"
#include "interpolate.h"
#include "compiler_warnings.h"
#include "file_util.h"
#include "logfile.h"
#include "mha_io.h"
#include "path_util.h"
#include "plm_int.h"
#include "plm_math.h"
#include "plm_image_header.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "ray_data.h"
#include "ray_trace.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_fill.h"
#include "volume_limit.h"
#include "volume_macros.h"
#include "print_and_exit.h"

//#define VERBOSE 1

#if VERBOSE
static bool global_debug = false;
#endif

//20140827_YKP
//col0 = HU, col1 = Relative stopping power
//Table: XiO, ctedproton 2007 provided by Yoost
extern const double lookup_PrSTPR_XiO_MGH[][2] ={
    -1000.0,    0.01,
    0.0,        1.0,
    40.0,       1.04,
    1000.0,     1.52,
    2000.0,     2.02,
    3000.0,     2.55,
};

extern const double lookup_PrSTPR_XiO_MGH[][2];

static void rpl_callback_accum (
    void *callback_data, size_t vox_index, 
    double vox_len, float vox_value);
static void rpl_callback_sample (
    void *callback_data, size_t vox_index, 
    double vox_len, float vox_value);

static void rpl_ray_trace_callback_ct_density (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value);
static void rpl_ray_trace_callback_ct_HU (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value);
static void rpl_ray_trace_callback_PrSTPR (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value);
static void rpl_ray_trace_callback_PrSTPR_XiO_MGH (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value);
static void rpl_ray_trace_callback_range_length (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value);

typedef struct callback_data Callback_data;
struct callback_data {
    Rpl_volume *rpl_vol;         /* Radiographic depth volume */
    Ray_data *ray_data;          /* Data specific to the ray */
    int* ires;                   /* Aperture Dimensions */
    int step_offset;             /* Number of steps before first ray sample */
    double accum;                /* Accumulated intensity */
    int last_step_completed;     /* Last step written to output image */
};

class Rpl_volume_private {
public:
    Proj_volume *proj_vol;
    Plm_image::Pointer ct;
    Volume_limit ct_limit;
    Ray_data *ray_data;
    // Clipping distances are computed from aperture plane, not from source
    double front_clipping_dist;
    double back_clipping_dist;
    Aperture::Pointer aperture;
    Rpl_volume_ray_trace_start rvrts;
    
public:
    Rpl_volume_private () {
        proj_vol = new Proj_volume;
        ct = Plm_image::New ();
        ray_data = 0;
        front_clipping_dist = DBL_MAX;
        back_clipping_dist = -DBL_MAX;
        aperture = Aperture::New ();
        rvrts = RAY_TRACE_START_AT_RAY_VOLUME_INTERSECTION;
    }
    ~Rpl_volume_private () {
        delete proj_vol;
        if (ray_data) {
            delete[] ray_data;
        }
    }
};

Rpl_volume::Rpl_volume () {
    d_ptr = new Rpl_volume_private;
}

Rpl_volume::~Rpl_volume () {
    delete d_ptr;
}

void 
Rpl_volume::set_geometry (
    const double src[3],           // position of source (mm)
    const double iso[3],           // position of isocenter (mm)
    const double vup[3],           // dir to "top" of projection plane
    double sid,                    // dist from proj plane to source (mm)
    const plm_long image_dim[2],   // resolution of image
    const double image_center[2],  // image center (pixels)
    const double image_spacing[2], // pixel size (mm)
    const double step_length       // spacing between planes
)
{
    double clipping_dist[2] = {sid, sid};

#if defined (commentout)
    printf ("> src = %f %f %f\n", src[0], src[1], src[2]);
    printf ("> iso = %f %f %f\n", iso[0], iso[1], iso[2]);
    printf ("> vup = %f %f %f\n", vup[0], vup[1], vup[2]);
    printf ("> sid = %f\n", sid);
    printf ("> idim = %d %d\n", image_dim[0], image_dim[1]);
    printf ("> ictr = %f %f\n", image_center[0], image_center[1]);
    printf ("> isp = %f %f\n", image_spacing[0], image_spacing[1]);
    printf ("> stp = %f\n", step_length);
#endif

    /* This sets everything except the clipping planes.  We don't know 
       these until caller tells us the CT volume to compute against. */
    d_ptr->proj_vol->set_geometry (
        src, iso, vup, sid, image_dim, image_center, image_spacing,
        clipping_dist, step_length);
}

void 
Rpl_volume::clone_geometry (const Rpl_volume *rv)
{
    this->set_geometry (
        rv->d_ptr->proj_vol->get_src(),
        rv->d_ptr->proj_vol->get_iso(),
        rv->get_aperture()->vup,
        rv->get_aperture()->get_distance(),
        rv->get_aperture()->get_dim(),
        rv->get_aperture()->get_center(),
        rv->get_aperture()->get_spacing(),
        rv->d_ptr->proj_vol->get_step_length());
}

void
Rpl_volume::set_ray_trace_start (Rpl_volume_ray_trace_start rvrts)
{
    printf ("Setting RVRTS = %d\n", (int) rvrts);
    d_ptr->rvrts = rvrts;
}

void 
Rpl_volume::set_ct_volume (Plm_image::Pointer& ct_volume)
{
    d_ptr->ct = ct_volume;

    /* Compute volume boundary box */
    d_ptr->ct_limit.find_limit (ct_volume->get_volume_float());
}

Aperture::Pointer& 
Rpl_volume::get_aperture ()
{
    return d_ptr->aperture;
}

const Aperture::Pointer& 
Rpl_volume::get_aperture () const
{
    return d_ptr->aperture;
}

void 
Rpl_volume::set_aperture (Aperture::Pointer& ap)
{
    d_ptr->aperture = ap;
}

const plm_long*
Rpl_volume::get_image_dim ()
{
    return d_ptr->proj_vol->get_image_dim();
}

plm_long
Rpl_volume::get_num_steps ()
{
    return d_ptr->proj_vol->get_num_steps();
}

/* 1D interpolation */
double
Rpl_volume::get_value (
    plm_long ap_ij[2],       /* I: aperture index */
    double dist              /* I: distance from aperture in mm */
) const
{
    plm_long idx1, idx2;
    plm_long ijk[3];
    double rg1, rg2, rgdepth, frac;
    const Proj_volume *proj_vol = this->get_proj_volume ();
    const Volume *vol = this->get_vol();
    float* d_img = (float*) vol->img;

    if (dist < 0) {
        return 0.0;
    }

    ijk[0] = ap_ij[0];
    ijk[1] = ap_ij[1];
    ijk[2] = (int) floorf (dist / proj_vol->get_step_length());

    /* Depth to step before point */
    idx1 = volume_index (vol->dim, ijk);
    if (idx1 < vol->npix) {
        rg1 = d_img[idx1];
    } else {
        return 0.0f;
    }

    /* Fraction from step before point to point */
    frac = (dist - ijk[2] * proj_vol->get_step_length()) 
        / proj_vol->get_step_length();
    
    /* Depth to step after point */
    ijk[2]++;
    idx2 = volume_index (vol->dim, ijk);
    if (idx2 < vol->npix) {
        rg2 = d_img[idx2];
    } else {
        rg2 = d_img[idx1];
    }

    /* Radiographic depth, interpolated in depth only */
    rgdepth = rg1 + frac * (rg2 - rg1);

    return rgdepth;
}

/* 3D interpolation */
double
Rpl_volume::get_value (
    double ap_ij[2],    /* I: aperture index */
    double dist         /* I: distance from aperture in mm */
) const
{
    const Proj_volume *proj_vol = this->get_proj_volume ();
    const Volume *vol = this->get_vol();

    if (dist < 0) {
        return 0.0;
    }

    float ijk[3] = {
        (float) ap_ij[0], 
        (float) ap_ij[1], 
        (float) (dist / proj_vol->get_step_length())
    };
    float val = vol->get_ijk_value (ijk);
    return val;
}

/* Lookup radiological path length to a voxel in world space */
double
Rpl_volume::get_value (
    const double* ct_xyz         /* I: location of voxel in world space */
) const
{
    int ap_ij[2], ap_idx;
    double ap_xy[3];
    double dist, rgdepth = 0.;
    bool debug = false;

    /* A couple of abbreviations */
    const plm_long *ires = d_ptr->proj_vol->get_image_dim();
    Proj_matrix *pmat = d_ptr->proj_vol->get_proj_matrix();

    if (debug) {
        pmat->debug ();
    }

    /* Back project the voxel to the aperture plane */
    d_ptr->proj_vol->project (ap_xy, ct_xyz);

    /* Make sure value is not inf or NaN */
    if (!is_number (ap_xy[0]) || !is_number (ap_xy[1])) {
    	return -1;
    }

    /* Round to nearest aperture index */
    ap_ij[0] = ROUND_INT (ap_xy[0]);
    ap_ij[1] = ROUND_INT (ap_xy[1]);

    if (debug) {
	printf ("ap_xy = %g %g\n", ap_xy[0], ap_xy[1]);
    }

    /* Only handle voxels inside the (square) aperture */
    if (ap_ij[0] < 0 || ap_ij[0] >= ires[0] ||
        ap_ij[1] < 0 || ap_ij[1] >= ires[1]) {
        return -1;
    }

    /* Look up pre-computed data for this ray */
    ap_idx = ap_ij[1] * ires[0] + ap_ij[0];
    Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
    double *ap_xyz = ray_data->p2;

    if (debug) {
	printf ("ap_xyz = %g %g %g\n", ap_xyz[0], ap_xyz[1], ap_xyz[2]);
    }

    /* Compute distance from aperture to voxel */
    dist = vec3_dist (ap_xyz, ct_xyz);

    // Subtract off standoff distance.  Nearest neighbor aperture index
    // is used for this calculation.
    if (d_ptr->rvrts == RAY_TRACE_START_AT_RAY_VOLUME_INTERSECTION) {
        dist -= ray_data->front_dist;
    } else {
        dist -= d_ptr->front_clipping_dist;
    }

    /* Retrieve the radiographic depth */
    rgdepth = this->get_value (ap_xy, dist);

    return rgdepth;
}

void Rpl_volume::set_ct (const Plm_image::Pointer& ct_volume)
{
    d_ptr->ct = ct_volume;
}

Plm_image::Pointer Rpl_volume::get_ct()
{
    return d_ptr->ct;
}

void Rpl_volume::set_ct_limit (Volume_limit* ct_limit)
{
    d_ptr->ct_limit.lower_limit[0] = ct_limit->lower_limit[0];
    d_ptr->ct_limit.lower_limit[1] = ct_limit->lower_limit[1];
    d_ptr->ct_limit.lower_limit[2] = ct_limit->lower_limit[2];
    d_ptr->ct_limit.upper_limit[0] = ct_limit->upper_limit[0];
    d_ptr->ct_limit.upper_limit[1] = ct_limit->upper_limit[1];
    d_ptr->ct_limit.upper_limit[2] = ct_limit->upper_limit[2];
}

Volume_limit* Rpl_volume::get_ct_limit()
{
    return &d_ptr->ct_limit;
}

void Rpl_volume::set_ray_data (Ray_data *ray)
{
    d_ptr->ray_data = ray;
}

Ray_data* Rpl_volume::get_ray_data()
{
    return d_ptr->ray_data;
}

void Rpl_volume::set_front_clipping_plane (double front_clip)
{
    d_ptr->front_clipping_dist = front_clip;
}

double Rpl_volume::get_front_clipping_plane() const
{
    return d_ptr->front_clipping_dist;
}

void Rpl_volume::set_back_clipping_plane(double back_clip)
{
    d_ptr->back_clipping_dist = back_clip;
}

double Rpl_volume::get_back_clipping_plane() const
{
    return d_ptr->back_clipping_dist;
}

double
Rpl_volume::get_step_length () const
{
    return d_ptr->proj_vol->get_step_length ();
}

void 
Rpl_volume::compute_ray_data ()
{
    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    const double *nrm = proj_vol->get_nrm();
    const plm_long *ires = d_ptr->proj_vol->get_image_dim();
    Volume *ct_vol = d_ptr->ct->get_vol();

    /* Allocate data for each ray */
    if (d_ptr->ray_data) delete[] d_ptr->ray_data;
    d_ptr->ray_data = new Ray_data[ires[0]*ires[1]];

    /* Scan through the aperture plane */
    for (int r = 0; r < ires[1]; r++) {
        double r_tgt[3];
        double tmp[3];

        /* Compute r_tgt = 3d coordinates of first pixel in this row
           on aperture */
        vec3_copy (r_tgt, proj_vol->get_ul_room());
        vec3_scale3 (tmp, proj_vol->get_incr_r(), (double) r);
        vec3_add2 (r_tgt, tmp);

        for (int c = 0; c < ires[0]; c++) {
            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;
            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            double *ip1 = ray_data->ip1;
            double *ip2 = ray_data->ip2;
            double *p2 = ray_data->p2;
            double *ray = ray_data->ray;

#if VERBOSE
            global_debug = false;
            if (r == 49 && (c == 49 || c == 50)) {
                global_debug = true;
            }
#endif

            /* Save the aperture index */
            ray_data->ap_idx = ap_idx;

            /* Compute p2 = 3d coordinates of point on aperture */
            vec3_scale3 (tmp, proj_vol->get_incr_c(), (double) c);
            vec3_add3 (p2, r_tgt, tmp);

	    /* Define unit vector in ray direction */
	    vec3_sub3 (ray, p2, src);
	    vec3_normalize1 (ray);

	    /* Test if ray intersects volume and create intersection points */
            ray_data->intersects_volume = false;

	    if (!d_ptr->ct_limit.clip_ray (ip1, ip2, src, ray))
            {
		continue;
	    }

	    /* If intersect points are before or after aperture. 
               If before, clip them at aperture plane. */

            /* First, check the second point */
            double tmp[3];
            vec3_sub3 (tmp, ip2, p2);
            if (vec3_dot (tmp, nrm) > 0) {
                /* If second point is behind aperture, then so is 
                   first point, and therefore the ray doesn't intersect 
                   the volume. */
                continue;
            }

            /* OK, by now we know this ray does intersect the volume */
            ray_data->intersects_volume = true;

#if VERBOSE
            if (global_debug) {
                printf ("(%d,%d)\n", r, c);
                printf ("%d\n", (int) ap_idx);
                printf ("ap  = %f %f %f\n", p2[0], p2[1], p2[2]);
                printf ("ip1 = %f %f %f\n", ip1[0], ip1[1], ip1[2]);
                printf ("ip2 = %f %f %f\n", ip2[0], ip2[1], ip2[2]);
            }
#endif

            /* Compute distance to front intersection point, and set 
               front clipping plane if indicated */
            vec3_sub3 (tmp, ip1, p2);
            if (vec3_dot (tmp, nrm) > 0) {
                ray_data->front_dist = 0;
            } else {
                ray_data->front_dist = vec3_dist (p2, ip1);
            }
            if (ray_data->front_dist < d_ptr->front_clipping_dist) {
#if defined (commentout)
                /* GCS FIX.  This should not be here.  */
                // - 0.001 mm to avoid the closest ray to intersect
                // the volume with a step inferior to its neighbours.
                // The minimal ray will be the only one to touch
                // the volume when offset_step = 0.
                d_ptr->front_clipping_dist = ray_data->front_dist - 0.001;
#endif
                d_ptr->front_clipping_dist = ray_data->front_dist;
            }

            /* Compute distance to back intersection point, and set 
               back clipping plane if indicated */
	    ray_data->back_dist = vec3_dist (p2, ip2);
            if (ray_data->back_dist > d_ptr->back_clipping_dist) {
                d_ptr->back_clipping_dist = ray_data->back_dist;
            }
#if VERBOSE
            if (global_debug) {
                printf ("fd/bd = %f %f\n", ray_data->front_dist,
                    ray_data->back_dist);
            }
#endif
        }
    }
}

void 
Rpl_volume::compute_rpl (
    bool use_aperture, Ray_trace_callback callback)
{
    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    int ires[2];
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);

    unsigned char *ap_img = 0;
    if (use_aperture && d_ptr->aperture->have_aperture_image()) {
        Volume::Pointer ap_vol = d_ptr->aperture->get_aperture_volume ();
        ap_img = (unsigned char*) ap_vol->img;
    }
    Volume *ct_vol = d_ptr->ct->get_vol();

    /* Preprocess data by clipping against volume */
    this->compute_ray_data ();

    if (d_ptr->front_clipping_dist == DBL_MAX) {
        lprintf ("Sorry, total failure intersecting volume\n");
        return;
    }

    lprintf ("FPD = %f, BPD = %f\n", 
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist);

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();
 
    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];

            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);

#if VERBOSE
            global_debug = false;
            if (r == 49 && (c == 49 || c == 50)) {
                global_debug = true;
            }
            if (global_debug) {
                printf ("Tracing ray (%d,%d)\n", r, c);
            }
#endif

            /* Check if beamlet is inside aperture, if not 
               we skip ray tracing */
            if (ap_img && ap_img[r*ires[0]+c] == 0) {
                continue;
            }

            this->rpl_ray_trace (
                ct_vol,
                ray_data,
                callback,
                &d_ptr->ct_limit,
                src,
                0,
                ires
            );
        }
    }
}

void 
Rpl_volume::compute_rpl_sample (bool use_aperture)
{
    this->compute_rpl (use_aperture, rpl_callback_sample);
}

void 
Rpl_volume::compute_rpl_accum (bool use_aperture)
{
    this->compute_rpl (use_aperture, rpl_callback_accum);
}

/* This function samples the CT into a RPL equivalent geometry.
   The rpl_volume should be in proj_wed format, not in proj_ct format. */
/* GCS Note: Why do others call compute_ray_data(), but this one does not?*/
void 
Rpl_volume::compute_rpl_ct_density ()
{
    int ires[2];

    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);
    unsigned char *ap_img = 0;
    if (d_ptr->aperture->have_aperture_image()) {
        Volume::Pointer ap_vol = d_ptr->aperture->get_aperture_volume ();
        ap_img = (unsigned char*) ap_vol->img;
    }
    Volume *ct_vol = d_ptr->ct->get_vol();

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();
    
    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);

#if VERBOSE
            global_debug = false;
            if (r == 49 && (c == 49 || c == 50)) {
                global_debug = true;
            }
            if (global_debug) {
                printf ("Tracing ray (%d,%d)\n", r, c);
            }
#endif

            /* Check if beamlet is inside aperture, if not 
               we skip ray tracing */
            if (ap_img && ap_img[r*ires[0]+c] == 0) {
                continue;
            }

            this->rpl_ray_trace (
                ct_vol,            /* I: CT volume */
                ray_data,          /* I: Pre-computed data for this ray */
                rpl_ray_trace_callback_ct_density, /* I: callback */
                &d_ptr->ct_limit,  /* I: CT bounding region */
                src,               /* I: @ source */
                0,           /* I: range compensator thickness */
                ires               /* I: ray cast resolution */
            );

        }
    }
}

/* This function samples the CT in rpl geometry with HU units */
/* GCS NOTE: It does not check the aperture */
void 
Rpl_volume::compute_rpl_HU ()
{
    int ires[2];

    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);
    Volume *ct_vol = d_ptr->ct->get_vol();

    /* We don't need to do the first pass, as it was already done for the real rpl_volume */

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();
 
    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);

#if VERBOSE
            global_debug = false;
            if (r == 49 && (c == 49 || c == 50)) {
                global_debug = true;
            }
            if (global_debug) {
                printf ("Tracing ray (%d,%d)\n", r, c);
            }
#endif

            this->rpl_ray_trace (
                ct_vol,            /* I: CT volume */
                ray_data,          /* I: Pre-computed data for this ray */
                rpl_ray_trace_callback_ct_HU, /* I: callback */
                &d_ptr->ct_limit,  /* I: CT bounding region */
                src,               /* I: @ source */
                0,            /* I: range compensator thickness */
                ires               /* I: ray cast resolution */
            );
        }
    }
}

/* GCS Note: This is different from compute_rpl() functions.
   It calls compute_ray_data(), allocates, and computes ray_data->cp.  */
void 
Rpl_volume::compute_rpl_void ()
{
    /* A couple of abbreviations */
    int ires[2];
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);

    /* Preprocess data by clipping against volume */
    this->compute_ray_data ();

    if (d_ptr->front_clipping_dist == DBL_MAX) {
        print_and_exit ("Sorry, total failure intersecting volume (compute_rpl_void)\n");
    }

    lprintf ("FPD = %f, BPD = %f\n", 
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist);

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();
    
    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);
        }
    }
}

/* GCS Note: The rgc should be added after ray tracing */
void 
Rpl_volume::compute_rpl_range_length_rgc ()
{
    int ires[2];

    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);
    float *rc_img = 0;
    if (d_ptr->aperture->have_range_compensator_image()) {
        Volume::Pointer rc_vol 
            = d_ptr->aperture->get_range_compensator_volume ();
        rc_img = (float*) rc_vol->img;
    }
    Volume *ct_vol = d_ptr->ct->get_vol();

    /* Preprocess data by clipping against volume */
    this->compute_ray_data ();

    if (d_ptr->front_clipping_dist == DBL_MAX) {
        print_and_exit ("Sorry, total failure intersecting volume\n");
    }

    lprintf ("FPD = %f, BPD = %f\n", 
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist);

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
      d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();
 
    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);

#if VERBOSE
            global_debug = false;
            if (r == 49 && (c == 49 || c == 50)) {
                global_debug = true;
            }
            if (global_debug) {
                printf ("Tracing ray (%d,%d)\n", r, c);
            }
#endif

            /* Initialize ray trace accum to range compensator thickness */
            double rc_thk = 0.;
            if (rc_img) {
                rc_thk = rc_img[r*ires[0]+c];
            }

            this->rpl_ray_trace (
                ct_vol,            /* I: CT volume */
                ray_data,          /* I: Pre-computed data for this ray */
                rpl_ray_trace_callback_range_length, /* I: callback */
                &d_ptr->ct_limit,  /* I: CT bounding region */
                src,               /* I: @ source */
                rc_thk,            /* I: range compensator thickness */
                ires               /* I: ray cast resolution */
            );
        }
    }
}

/* GCS NOTE: This also does not check the aperture */
void 
Rpl_volume::compute_rpl_PrSTRP_no_rgc ()
{
    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    int ires[2];
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);

    Volume *ct_vol = d_ptr->ct->get_vol();
    
    /* Preprocess data by clipping against volume */
    this->compute_ray_data ();

    if (d_ptr->front_clipping_dist == DBL_MAX) {
        print_and_exit ("Sorry, total failure intersecting volume (compute_rpl_rglength_wo_rg_compensator)\n");
    }

    lprintf ("FPD = %f, BPD = %f\n", 
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist);

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
      d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();

    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];

            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);

#if VERBOSE
            global_debug = false;
            if (r == 49 && (c == 49 || c == 50)) {
                global_debug = true;
            }
            if (global_debug) {
                printf ("Tracing ray (%d,%d)\n", r, c);
            }
#endif

            this->rpl_ray_trace (
                ct_vol,            /* I: CT volume */
                ray_data,          /* I: Pre-computed data for this ray */
                rpl_ray_trace_callback_PrSTPR, /* I: callback */
                &d_ptr->ct_limit,  /* I: CT bounding region */
                src,               /* I: @ source */
                0,            /* I: range compensator thickness */
                ires               /* I: ray cast resolution */
            );
        }
    }
    /* Now we only have a rpl_volume without compensator, from which we need to compute the sigma along this ray */
}

double Rpl_volume::compute_farthest_penetrating_ray_on_nrm(float range)
{
    //int dim[3] = { this->get_vol()->dim[0], this->get_vol()->dim[1], this->get_vol()->dim[2]};
    const plm_long *dim = this->get_vol()->dim;
    int idx = 0;
    double POI[3] = {0.0, 0.0, 0.0};
    double tmp[3] = {0.0, 0.0, 0.0};

    double dist = 0;
    double max_dist = 0;
    double offset = vec3_dist(this->get_proj_volume()->get_src(), this->get_proj_volume()->get_iso()) - this->get_aperture()->get_distance();

    float* img = (float*) this->get_vol()->img;

    for (int apert_idx = 0; apert_idx < dim[0] * dim[1]; apert_idx++)
    {
        Ray_data* ray_data = (Ray_data*) &this->get_ray_data()[apert_idx];
        for (int s = 0; s < dim[2]; s++)
        {
            idx = s * dim[0] * dim[1] + apert_idx;
            if (s == dim[2]-1 || dim[2] == 0)
            {
                max_dist = offset + (double) dim[2] * this->get_vol()->spacing[2];
                printf("Warning: Range > ray_length in volume => Some rays might stop outside of the volume image.\n");
                printf("position of the maximal range on the z axis: z = %lg\n", max_dist);
                return max_dist;
            }

            if (img[idx] > range)
            {
                /* calculation of the length projected on the nrm axis, from the aperture */
                vec3_copy(POI, ray_data->cp);
                vec3_copy(tmp, ray_data->ray);
                vec3_scale2(tmp, (double)s * this->get_vol()->spacing[2]);
                vec3_add2(POI, tmp);
                
                dist = -vec3_dot(POI, this->get_proj_volume()->get_nrm());
                dist = offset + dist;
                if (dist > max_dist)
                {
                    max_dist = dist;
                }
                break;
            }
        }
    }

    printf("position of the maximal range on the z axis: z = %lg\n", max_dist);
    return max_dist;
}

void 
Rpl_volume::compute_proj_wed_volume (
    Volume *proj_wed_vol, float background)
{

    //A few abbreviations
    Proj_volume *proj_vol = d_ptr->proj_vol;
    float *proj_wed_vol_img = (float*) proj_wed_vol->img;

    /* Get some parameters from the proj volume, 
       calculate src to isocenter distance */
    const double *src = proj_vol->get_src();
    const double *iso = proj_vol->get_iso();
    const double sid_length = proj_vol->get_proj_matrix()->sid; //distance from source to aperture
    double src_iso_vec[3];  
    vec3_sub3(src_iso_vec,src,iso);
    const double src_iso_distance = vec3_len(src_iso_vec);
    const double ap_iso_distance = src_iso_distance - sid_length;

    /* Subtract off standoff distance */
    //This is the perpendicular "base" distance that we calculate all rgdepths from.
    double base_rg_dist = ap_iso_distance - d_ptr->front_clipping_dist;

    //This is the perpendicular "base" distance that we calculate how much
    //each geometric distance should increase, due to divergence.
    const double base_dist = proj_vol->get_proj_matrix()->sid; //distance from source to aperture
  
    const plm_long *ires = proj_vol->get_image_dim();

    plm_long ap_ij[2]; //ray index of rvol
    plm_long ap_idx = 0;  //ray number
    Ray_data *ray_data;
    double ray_ap[3]; //vector from src to ray intersection with ap plane
    double ray_ap_length; //length of vector from src to ray intersection with ap plane
    double rglength; //length that we insert into get_value for each ray

    for (ap_ij[1] = 0; ap_ij[1] < ires[1]; ap_ij[1]++) {
        for (ap_ij[0] = 0; ap_ij[0] < ires[0]; ap_ij[0]++) {

            /* Ray number */
            ap_idx = ap_ij[1] * ires[0] + ap_ij[0];
            ray_data = &d_ptr->ray_data[ap_idx];

            /* Set each ray to "background", defined in wed_main (default 0) */
            proj_wed_vol_img[ap_idx] = background;

            /* Coordinate of ray intersection with aperture plane */
            double *ap_xyz = ray_data->p2;
            vec3_sub3 (ray_ap, ap_xyz, src);
            ray_ap_length = vec3_len(ray_ap);

            rglength = base_rg_dist*(ray_ap_length/base_dist);

            proj_wed_vol_img[ap_idx] = (float) (this->get_value(ap_ij,rglength));
      
        }
    }
}

/* Resample a ct or dose volume into a wed_ct or wed_dose volume */
void 
Rpl_volume::compute_wed_volume (
    Volume *wed_vol, Volume *in_vol, float background)
{
    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    Volume *rvol = proj_vol->get_vol();
    float *rvol_img = (float*) rvol->img;
    float *in_vol_img = (float*) in_vol->img;
    float *wed_vol_img = (float*) wed_vol->img;
    const plm_long *ires = proj_vol->get_image_dim();

    plm_long wijk[3];  /* Index within wed_volume */

    /* Fill the wed_vol with background values */
    volume_fill (wed_vol, background);
   
    for (wijk[1] = 0; wijk[1] < ires[1]; wijk[1]++) {
        for (wijk[0] = 0; wijk[0] < ires[0]; wijk[0]++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = wijk[1] * ires[0] + wijk[0];

            bool debug = false;
            if (ap_idx == (ires[1]/2) * ires[0] + (ires[0] / 2)) {
                // printf ("DEBUGGING %d %d\n", ires[1], ires[0]);
                // debug = true;
            }

	    /* Nothing to do if ray misses volume */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            if (!ray_data->intersects_volume) {
                continue;
            }

            /* Keep track of index within rpl_volume */
            plm_long rijk[3] = { wijk[0], wijk[1], 0 };

	    /*Perform the clipping, so the projection volume 
              start points are the same */
	    double ray_start[3];
	    double ray_end[3];

            /* GCS FIX: Why do this?  Is the ray data not already valid? */
	    if (!d_ptr->ct_limit.clip_segment (ray_start, ray_end, ray_data->p2, ray_data->ip2)) {
                printf("Error in ray clipping, exiting...\n");
                return;
	    }

            /* Loop, looking for each output voxel */
            for (wijk[2] = 0; wijk[2] < rvol->dim[2]; wijk[2]++) {
                plm_long widx = volume_index (rvol->dim, wijk);

                /* Compute the currently required rpl for this step */
                double req_rpl = wijk[2] * 1.0;

                if (debug) printf ("--- (%d,%f)\n", (int) wijk[2], req_rpl);

                /* Loop through input voxels looking for appropriate value */
		double prev_rpl = 0.;

                while (rijk[2] < rvol->dim[2]) {
                    plm_long ridx = volume_index (rvol->dim, rijk);
                    double curr_rpl = rvol_img[ridx];

                    if (debug) printf ("(%d,%f)\n", (int) rijk[2], curr_rpl);

                    /* Test if the current input voxel is suitable */
                    if (curr_rpl > req_rpl) {
                        /* Compute coordinate of matching voxel */
                        double xyz_init[3];
                        double xyz[3];

			/* Get the distance relative to the reqired rad. length.  */
			double dist = rijk[2]*proj_vol->get_step_length() - ( (curr_rpl - req_rpl)/(curr_rpl-prev_rpl) ) * proj_vol->get_step_length();

                        vec3_scale3 (xyz_init, ray_data->ray, dist);
			vec3_add3 (xyz, xyz_init, ray_start);
                        
			float in_ijk_f[3];
			in_ijk_f[0] = (xyz[0] - in_vol->origin[0]) / in_vol->spacing[0];
			in_ijk_f[1] = (xyz[1] - in_vol->origin[1]) / in_vol->spacing[1];
			in_ijk_f[2] = (xyz[2] - in_vol->origin[2]) / in_vol->spacing[2];

			if (ROUND_PLM_LONG(in_ijk_f[0]) < 0 || ROUND_PLM_LONG(in_ijk_f[0]) >= in_vol->dim[0]) {break;}
			if (ROUND_PLM_LONG(in_ijk_f[1]) < 0 || ROUND_PLM_LONG(in_ijk_f[1]) >= in_vol->dim[1]) {break;}
			if (ROUND_PLM_LONG(in_ijk_f[2]) < 0 || ROUND_PLM_LONG(in_ijk_f[2]) >= in_vol->dim[2]) {break;}
	
			plm_long ijk_floor[3];
			plm_long ijk_round[3];
			float li_1[3], li_2[3];

			// Compute linear interpolation fractions
			li_clamp_3d (in_ijk_f, ijk_floor, ijk_round,li_1,li_2,in_vol);

			plm_long idx_floor;

			// Find linear indices for moving image
			idx_floor = volume_index (in_vol->dim, ijk_floor);

			float value = li_value(li_1[0], li_2[0],li_1[1], li_2[1],li_1[2], li_2[2],idx_floor,in_vol_img,in_vol);

			/* Write value to output image */
			wed_vol_img[widx] = value;

                        /* Suitable voxel found and processed, so move on 
                           to the next output voxel */
                        break;
                    }
                    /* Otherwise, current voxel has insufficient 
                       rpl, so move on to the next */
		    prev_rpl = curr_rpl;
                    rijk[2] ++;
                }
            }
        }
    }
}

void 
Rpl_volume::compute_dew_volume (
    Volume *wed_vol, Volume *dew_vol, float background)
{
  
    double dummy_vec[3] = {0., 0., 0.};
    double dummy_length = 0.;

    double master_coord[2]; //coordinate within a unit box that determines weighting of the final trilinear interpolation
    double master_square[2][2]; //"box" containing the 4 values used for the final bilinear interpolation

    //A couple of abbreviations
    Proj_volume *proj_vol = d_ptr->proj_vol;
    //  Volume *rvol = proj_vol->get_volume();
    float *dew_vol_img = (float*) dew_vol->img;
    float *wed_vol_img = (float*) wed_vol->img;
    const plm_long *dew_dim = dew_vol->dim; 
  
    //Get some parameters from the proj volume
    const plm_long *ires = proj_vol->get_image_dim();
    const double *src = proj_vol->get_src();
    const double dist = proj_vol->get_proj_matrix()->sid; //distance from source to aperture
    double src_iso_vec[3];   //vector from source to isocenter
    proj_vol->get_proj_matrix()->get_nrm(src_iso_vec); 
    vec3_invert(src_iso_vec);
    //  const double *center = proj_vol->get_proj_matrix()->ic;

    //Contruct aperture "box", in which each voxel's respective
    //ray intersection must be within.
    Ray_data *ray_box[4];
    ray_box[0] = &d_ptr->ray_data[ 0 ];
    ray_box[1] = &d_ptr->ray_data[ ires[0]-1 ];
    ray_box[2] = &d_ptr->ray_data[ ires[0]*(ires[1]-1) ];
    ray_box[3] = &d_ptr->ray_data[ ires[0]*ires[1]-1 ];

    //Compute aperture dimension lengths and normalized axes
    double ap_axis1[3]; //unit vector of ap. axis 1
    double ap_axis2[3];
    double  ap_res[2]; //resolution of aperture grid

    vec3_sub3(ap_axis1,ray_box[1]->p2,ray_box[0]->p2);
    ap_res[0] = vec3_len(ap_axis1)/(ires[0]-1);
    vec3_normalize1(ap_axis1);
    vec3_sub3(ap_axis2,ray_box[2]->p2,ray_box[0]->p2);
    ap_res[1] = vec3_len(ap_axis2)/(ires[1]-1);
    vec3_normalize1(ap_axis2);
  
    Ray_data *ray_adj[4]; //the 4 rays in rpl space that border each coordinate
    double ray_adj_len; //calculated length each adjacent ray to the voxel
    double rad_depth_input; //input length to calculate rgdepth

    double ray_start[3];
    double ray_end[3];

    plm_long wijk[3]; //index within wed_volume
    plm_long ap_ij[2]; //ray indox of rvol
    plm_long dijk[3]; //Index within dew_volume
    plm_long didx; //image index within dew_volume

    bool skipflag;

    double coord[3];   //coordinate within dew_volume
    double ap_coord[3]; //coordinate in aperture plane from source
    double ap_coord_plane[2]; //transformed, 2-d aperture coordinate of each voxel ray

    double coord_vec[3]; //vector along source to coordinate
    double unit_coord_vec[3]; //unit vector along source to coordinate
    double ap_coord_vec[3]; //vector along source to coordinate, terminated at ap. plane
  
    double adj_ray_coord[3]; //adjacent ray vector, used to compute rad. length
    double dummy_adj_ray[3];
  
    double coord_ap_len; //distance from coordinate to aperture
    double dummy_lin_ex;
    plm_long dummy_index1;
    plm_long dummy_index2;

    plm_long ray_lookup[4][2]; //quick lookup table for ray coordinates for rijk input
    double ray_rad_len[4]; //radiation length of each ray

    for (dijk[0] = 0; dijk[0] != dew_dim[0]; ++dijk[0])  {
        coord[0] = dijk[0]*dew_vol->spacing[0]+dew_vol->origin[0];
        for (dijk[1] = 0; dijk[1] != dew_dim[1]; ++dijk[1])  {
            coord[1] = dijk[1]*dew_vol->spacing[1]+dew_vol->origin[1];
            for (dijk[2] = 0; dijk[2] != dew_dim[2]; ++dijk[2])  {
                coord[2] = dijk[2]*dew_vol->spacing[2]+dew_vol->origin[2];

                didx = volume_index (dew_dim, dijk);

                //Set the default to background.
                dew_vol_img[didx] = background;
	
                vec3_sub3(coord_vec,coord,src); //Determine the vector to this voxel from the source
                vec3_copy(unit_coord_vec,coord_vec);
                vec3_normalize1(unit_coord_vec); //Get unit vector from source to voxel
                coord_ap_len = dist/vec3_dot(unit_coord_vec,src_iso_vec); //trig + dot product for distance
                vec3_copy(ap_coord_vec,unit_coord_vec);
                vec3_scale2(ap_coord_vec, coord_ap_len); //calculate vector from source to aperture plane
                vec3_add3(ap_coord,ap_coord_vec,src);  //calculate vector from origin to aperture plane

                //Some math will fail if we try to compute nonsensical values of the volume
                //between the source and aperture.
                if (coord_ap_len>=vec3_len(coord_vec))  {continue;}

                //As ap_coord is on the proj. plane, then check the 6 coord. boundaries
                //We also don't know which coordinates are larger depending on the orientation of
                //the projection plane, so account for that.
                skipflag = false;
                for (int i=0;i!=3;++i)  {
                    if(ray_box[0]->p2[i] >= ray_box[3]->p2[i])  {
                        if ( !( (ap_coord[i] <= ray_box[0]->p2[i]) && (ap_coord[i] >= ray_box[3]->p2[i]) ) )  {skipflag = true; break;}
                    }
                    else  {
                        if ( !( (ap_coord[i] >= ray_box[0]->p2[i]) && (ap_coord[i] <= ray_box[3]->p2[i]) ) )  {skipflag = true; break;}
                    }
                }
                if (skipflag) {continue;}

                for (int i=0;i!=4;++i)  {master_square[i/2][i%2] = background;}
	
                //Now we must find the projection of the point on the two aperture axes
                //Do this by calculating the closest point along both
                vec3_sub3(dummy_vec,ap_coord,ray_box[0]->p2);
                dummy_length = vec3_len(dummy_vec);
                vec3_normalize1(dummy_vec);
                ap_coord_plane[0] = vec3_dot(dummy_vec,ap_axis1)*dummy_length;
                ap_coord_plane[1] = vec3_dot(dummy_vec,ap_axis2)*dummy_length;
	
                master_coord[0] = ap_coord_plane[0]/ap_res[0] - floor(ap_coord_plane[0]/ap_res[0]);
                master_coord[1] = ap_coord_plane[1]/ap_res[1] - floor(ap_coord_plane[1]/ap_res[1]);

                //Get the 4 adjacent rays relative to the aperature coordinates
                int base_ap_coord = (int) (floor(ap_coord_plane[1]/ap_res[1])*ires[0] + floor(ap_coord_plane[0]/ap_res[0])); 

                ray_adj[0] = &d_ptr->ray_data[ base_ap_coord ];
                ray_adj[1] = &d_ptr->ray_data[ base_ap_coord + 1 ];
                ray_adj[2] = &d_ptr->ray_data[ base_ap_coord + ires[0] ];
                ray_adj[3] = &d_ptr->ray_data[ base_ap_coord + ires[0] + 1 ];

                //Compute ray indices for later rpl calculations.

                ray_lookup[0][0] = floor(ap_coord_plane[0]/ap_res[0]);
                ray_lookup[0][1] = floor(ap_coord_plane[1]/ap_res[1]);
                ray_lookup[1][0] = floor(ap_coord_plane[0]/ap_res[0]);
                ray_lookup[1][1] = floor(ap_coord_plane[1]/ap_res[1]) + 1;
                ray_lookup[2][0] = floor(ap_coord_plane[0]/ap_res[0]) + 1;
                ray_lookup[2][1] = floor(ap_coord_plane[1]/ap_res[1]);
                ray_lookup[3][0] = floor(ap_coord_plane[0]/ap_res[0]) + 1;
                ray_lookup[3][1] = floor(ap_coord_plane[1]/ap_res[1]) + 1;

                //Now compute the distance along each of the 4 rays
                //Distance chosen to be the intersection of each ray with the plane that both
                //contains the voxel and is normal to the aperture plane.

                for (int i=0;i!=4;++i)  {
                    //Compute distance along each ray.
                    //Vector along ray from source to aperture
                    vec3_sub3(dummy_adj_ray,ray_adj[i]->p2,src);

                    //Compute length, then ray from "ray start" to target position, using
                    //ratio of coordinate-aperture to coordinate lengths.
                    ray_adj_len = (vec3_len(coord_vec)/coord_ap_len)*vec3_len(dummy_adj_ray);
                    vec3_scale3(adj_ray_coord,ray_adj[i]->ray,ray_adj_len);

                    if (!d_ptr->ct_limit.clip_segment (ray_start, ray_end, ray_adj[i]->p2, ray_adj[i]->ip2)) {
                        printf("Error in ray clipping, exiting...\n");
                        return;
                    }

                    vec3_add2(adj_ray_coord,src);
                    vec3_sub2(adj_ray_coord,ray_start);
		    
                    rad_depth_input = vec3_len(adj_ray_coord);	    

                    //Now look up the radiation length, using the provided function,
                    //knowing the ray and the length along it.
                    ap_ij[0] = ray_lookup[i][0];
                    ap_ij[1] = ray_lookup[i][1];
                    /* GCS FIX: I think the ray_lookup stuff is 
                       3D interpolation, should reduce this code to 
                       use the 3D interpolated version of get_value() */

                    ray_rad_len[i] = this->get_value (ap_ij, rad_depth_input);

                    //Set each corner to background.
                    master_square[i/2][i%2] = background;

                    //Now, with the radiation length, extract the two dose values on either side
	  
                    //Check the borders - rvol should have an extra "border" for this purpose.
                    //If any rays are these added borders, it is outside dose and is background
                    if ( (ray_lookup[i][0]==0) || (ray_lookup[i][0]==ires[0]-1) ||
                        (ray_lookup[i][1]==0) || (ray_lookup[i][1]==ires[1]-1) )  {continue;}

                    //Set radiation lengths of 0 to background.
                    //While this is a boundary, keeps dose values from being assigned
                    //everywhere that rad depth is 0 (for example, the air before the volume).
                    if (ray_rad_len[i]<=0.)  {continue;}
	  
                    else {
                        dummy_lin_ex = ray_rad_len[i]-floor(ray_rad_len[i]);

                        wijk[0] = (ray_lookup[i][0] - 1)/wed_vol->spacing[0];
                        wijk[1] = (ray_lookup[i][1] - 1)/wed_vol->spacing[1];

                        //wijk[0] = ray_lookup[i][0] - 1;
                        //wijk[1] = ray_lookup[i][1] - 1;

                        //Needed if dew dimensions are not automatically set by wed in wed_main.
                        //wijk[0] = ((ray_lookup[i][0] - 1) - wed_vol->origin[0])/wed_vol->spacing[0];
                        //wijk[1] = ((ray_lookup[i][1] - 1) - wed_vol->origin[1])/wed_vol->spacing[1];

                        if (wijk[0] < 0 || wijk[0] >= wed_vol->dim[0]) {break;}
                        if (wijk[1] < 0 || wijk[1] >= wed_vol->dim[1]) {break;}
	    
                        wijk[2] = (int) ((floor(ray_rad_len[i])) - wed_vol->origin[2])/wed_vol->spacing[2];
                        if (wijk[2] < 0) {break;}
                        dummy_index1 = volume_index ( wed_vol->dim, wijk );

                        wijk[2] = (int) ((ceil(ray_rad_len[i])) - wed_vol->origin[2])/wed_vol->spacing[2];
                        if (wijk[2] >= wed_vol->dim[2]) {break;}
                        dummy_index2 = volume_index ( wed_vol->dim, wijk );

                        master_square[i/2][i%2] = wed_vol_img[dummy_index1] * (1-dummy_lin_ex) + wed_vol_img[dummy_index2] * dummy_lin_ex;
                    }
                }
                //Bilear interpolation from the square of wed dose ray values
                dew_vol_img[didx] = (float)
                    ( master_square[0][0]*(1-master_coord[0])*(1-master_coord[1]) +
                        master_square[0][1]*(1-master_coord[0])*(master_coord[1]) + 
                        master_square[1][0]*(master_coord[0])*(1-master_coord[1]) + 
                        master_square[1][1]*(master_coord[0])*(master_coord[1]) );
            }
        }
    }
}

void
Rpl_volume::compute_volume_aperture(Aperture::Pointer ap)
{
    int dim[3] = {(int) this->get_vol()->dim[0], (int) this->get_vol()->dim[1], (int) this->get_vol()->dim[2]};
	
    float* ap_vol_img = (float*) this->get_vol()->img;

    Volume::Pointer ap_vol = ap->get_aperture_volume ();
    unsigned char *ap_img = (unsigned char*) ap_vol->img;

    int idx = 0;

    for(int i = 0; i < dim[0] * dim[1]; i++)
    {
        for(int j = 0; j < dim[2]; j++)
        {
            idx = j * dim[0] * dim[1] + i;
            if ((float) ap_img[i] == 1)
            {
                ap_vol_img[idx] = 1;
            }
            else
            {
                ap_vol_img[idx] = 0;
            }
        }
    }
}

// In this new version the range compensator is added later in the code depending on the algorithm
void 
Rpl_volume::apply_beam_modifiers ()
{
    Volume::Pointer ap_vol = d_ptr->aperture->get_aperture_volume ();
    unsigned char *ap_img = (unsigned char*) ap_vol->img;
    Volume *proj_vol = d_ptr->proj_vol->get_vol();
    float *proj_img = (float*) proj_vol->img;

    /* For each ray in aperture */
    const plm_long *ires = d_ptr->proj_vol->get_image_dim();

    printf ("ires = %ld %ld\n", ires[0], ires[1]);
    printf ("proj_vol dim = %d %d %d\n", (int) proj_vol->dim[0], 
        (int) proj_vol->dim[1], (int) proj_vol->dim[2]);

    int ap_nvox = ires[0] * ires[1];
    for (int r = 0; r < ires[1]; r++) {
        for (int c = 0; c < ires[0]; c++) {

            /* Get aperture and rc values */
            plm_long ap_idx = r * ires[0] + c;
            float ap_val = (float) ap_img[ap_idx];

            /* For each voxel in ray */
            for (int z = 0; z < proj_vol->dim[2]; z++) {

                /* Adjust value of voxel */
                plm_long vol_idx = z * ap_nvox + ap_idx;
                proj_img[vol_idx] = ap_val * (proj_img[vol_idx]);
            }
        }
    }
}

Volume* 
Rpl_volume::get_vol ()
{
    return d_ptr->proj_vol->get_vol ();
}

const Volume*
Rpl_volume::get_vol () const
{
    return d_ptr->proj_vol->get_vol ();
}

Proj_volume* 
Rpl_volume::get_proj_volume ()
{
    return d_ptr->proj_vol;
}

const Proj_volume* 
Rpl_volume::get_proj_volume () const
{
    return d_ptr->proj_vol;
}

void
Rpl_volume::save (const char *filename)
{
    std::string fn_base = strip_extension_if (filename, "rpl");
    std::string rpl_vol_fn = fn_base + ".rpl";

    /* Save proj volume */
    std::string proj_vol_fn = fn_base + ".projv";
    d_ptr->proj_vol->save_projv (proj_vol_fn);

    /* Save ray data */
    if (d_ptr->ray_data) {
        std::string raydata_fn = fn_base + ".raydata";
        FILE *fp = plm_fopen (raydata_fn, "wb");
        const plm_long *ires = d_ptr->proj_vol->get_image_dim();
        for (int r = 0; r < ires[1]; r++) {
            for (int c = 0; c < ires[0]; c++) {
                int ap_idx = r * ires[0] + c;
                Ray_data *rd = &d_ptr->ray_data[ap_idx];
                fprintf (fp, 
                    "%d %g %g %g %g %g %g %g %g %g "
                    "%g %g %g %g %g %g %g %g %d\n",
                    rd->ap_idx, 
                    rd->ip1[0], rd->ip1[1], rd->ip1[2], 
                    rd->ip2[0], rd->ip2[1], rd->ip2[2], 
                    rd->p2[0], rd->p2[1], rd->p2[2], 
                    rd->ray[0], rd->ray[1], rd->ray[2], 
                    rd->front_dist, rd->back_dist, 
                    rd->cp[0], rd->cp[1], rd->cp[2], 
                    rd->step_offset);
            }
        }
        fclose (fp);
    }

    /* Don't save aperture (right?) */

    /* Don't save CT image (right?) */

    /* Save remaining private data */
    FILE *fp = plm_fopen (rpl_vol_fn, "wb");
    fprintf (fp, "front_clipping_dist = %g\n", d_ptr->front_clipping_dist);
    fprintf (fp, "back_clipping_dist = %g\n", d_ptr->back_clipping_dist);
    fprintf (fp, "volume_limit = %g %g %g %g %g %g %d %d %d\n",
        d_ptr->ct_limit.lower_limit[0],
        d_ptr->ct_limit.lower_limit[1],
        d_ptr->ct_limit.lower_limit[2],
        d_ptr->ct_limit.upper_limit[0],
        d_ptr->ct_limit.upper_limit[1],
        d_ptr->ct_limit.upper_limit[2],
        d_ptr->ct_limit.dir[0],
        d_ptr->ct_limit.dir[1],
        d_ptr->ct_limit.dir[2]);
    fclose (fp);
}

void
Rpl_volume::save (const std::string& filename)
{
    this->save (filename.c_str());
}

void
Rpl_volume::save_img (const char *filename)
{
    d_ptr->proj_vol->save_img (filename);
}

void
Rpl_volume::save_img (const std::string& filename)
{
    this->save_img (filename.c_str());
}

void
Rpl_volume::load_rpl (const char *filename)
{
    printf ("Loading rpl\n");
    std::string fn_base = strip_extension_if (filename, "rpl");
    std::string proj_vol_fn = fn_base + ".projv";
    printf ("-> %s\n-> %s-> %s\n", 
        filename, fn_base.c_str(), proj_vol_fn.c_str());
    d_ptr->proj_vol->load_projv (proj_vol_fn);
    printf ("Done.\n");
}

void
Rpl_volume::load_rpl (const std::string& filename)
{
    this->load_rpl (filename.c_str());
}

void
Rpl_volume::load_img (const char *filename)
{
    d_ptr->proj_vol->load_img (filename);
}

void
Rpl_volume::load_img (const std::string& filename)
{
    this->load_img (filename.c_str());
}

float compute_PrSTPR_from_HU(float CT_HU)
{
    return compute_PrSTPR_Schneider_weq_from_HU(CT_HU);
}

float 
compute_PrSTPR_Schneider_weq_from_HU (float CT_HU) // From Schneider's paper: Phys. Med. Biol.41 (1996) 111-124
{
    if (CT_HU <= -1000)
    {
        return 0.00106;
    }
    else if (CT_HU > -1000 && CT_HU <= 0)
    {
        return (1 - 0.00106) / 1000 * CT_HU + 1;
    }
    else if (CT_HU > 0 && CT_HU <= 41.46)
    {
        return .001174 * CT_HU + 1;
    }
    else
    {
        return .0005011 * CT_HU + 1.0279;
    }
}

float
compute_PrSTRP_XiO_MGH_weq_from_HU (float CT_HU) //YKP, Linear interpolation
{
    double minHU = -1000.0;
    double maxHU = 3000.0;

    if (CT_HU <= minHU)
	CT_HU = minHU;
    else if (CT_HU >= maxHU)
	CT_HU = maxHU;

    double CT_HU1 = minHU;
    double CT_HU2 = minHU;
    double RSP1 = 0;
    double RSP2 = 0;

    double interpolated_RSP = 0.0;

    int i=0;

    if (CT_HU >minHU)
    {
	while (CT_HU >= CT_HU1)
	{
            CT_HU1 = lookup_PrSTPR_XiO_MGH[i][0];
            RSP1 = lookup_PrSTPR_XiO_MGH[i][1];

            if (CT_HU >= CT_HU1)
            {
		CT_HU2 = CT_HU1;
		RSP2 = RSP1;
            }
            i++;
	}

	if ((CT_HU1-CT_HU2) == 0)
            interpolated_RSP = 0.0;
	else
            interpolated_RSP = (RSP2+(CT_HU-CT_HU2)*(RSP1-RSP2)/(CT_HU1-CT_HU2));	
    }
    else
    {
	interpolated_RSP = 0.0;
    }

    return interpolated_RSP;
}

float compute_PrWER_from_HU(float CT_HU)
{
    return compute_PrSTPR_from_HU(CT_HU) / compute_density_from_HU(CT_HU);
}

float compute_density_from_HU (float CT_HU) // from Schneider's paper: Phys. Med. Biol.41 (1996) 111-124
{
    if(CT_HU <= -1000)
    {
        return 0.001205;
    }
    else if (CT_HU > -1000 && CT_HU <= 65.64)
    {
        return (1-.001205)/1000 * CT_HU + 1;
    }
    else
    {
        return .0006481 * CT_HU + 1.0231;
    }
}

void
Rpl_volume::rpl_ray_trace (
    Volume *ct_vol,              /* I: CT volume */
    Ray_data *ray_data,          /* I: Pre-computed data for this ray */
    Ray_trace_callback callback, /* I: Callback function */
    Volume_limit *vol_limit,     /* I: CT bounding region */
    const double *src,           /* I: @ source */
    double rc_thk,               /* I: range compensator thickness */
    int* ires                    /* I: ray cast resolution */
)
{
    Callback_data cd;

    if (!ray_data->intersects_volume) {
        return;
    }

    /* Initialize callback data for this ray */
    cd.rpl_vol = this;
    cd.ray_data = ray_data;
    cd.accum = rc_thk;
    cd.ires = ires;

    // Figure out location of, and number of steps to first step within volume
    double first_loc[3];
    if (d_ptr->rvrts == RAY_TRACE_START_AT_RAY_VOLUME_INTERSECTION) {
        ray_data->step_offset = cd.step_offset = 0;
    } else {
        ray_data->step_offset
            = cd.step_offset
            = (int) floor (ray_data->front_dist - d_ptr->front_clipping_dist)
            / d_ptr->proj_vol->get_step_length ();
    }
    double tmp[3];
    vec3_scale3 (tmp, ray_data->ray, 
        cd.step_offset * d_ptr->proj_vol->get_step_length ());
    vec3_add3 (first_loc, ray_data->p2, tmp);

#if VERBOSE
    if (global_debug) {
        printf ("front_dist = %f\n", ray_data->front_dist);
        printf ("front_clip = %f\n", d_ptr->front_clipping_dist);
        printf ("dist = %f\n", dist);
        printf ("step_offset = %d\n", cd.step_offset);
        printf ("first_loc = (%f, %f, %f)\n", 
            first_loc[0], first_loc[1], first_loc[2]);
        printf ("ip2 = (%f, %f, %f)\n", 
            ray_data->ip2[0], ray_data->ip2[1], ray_data->ip2[2]);
        printf ("rlen = %g\n", vec3_dist (first_loc, ray_data->ip2));
    }
#endif

    /* get radiographic depth along ray */
    ray_trace_uniform (
        ct_vol,                     // INPUT: CT volume
        vol_limit,                  // INPUT: CT volume bounding box
        callback,                   // INPUT: step action Function
        &cd,                        // INPUT: callback data
        first_loc,                  // INPUT: ray starting point
        ray_data->ip2,              // INPUT: ray ending point
        d_ptr->proj_vol->get_step_length()); // INPUT: uniform ray step size

    /* Ray tracer will stop short of rpl volume for central rays. 
       We should continue padding the remaining voxels */
    float *depth_img = (float*) this->get_vol()->img;
    for (int s = cd.last_step_completed+1; 
         s < this->get_vol()->dim[2];
         s++)
    {
        int ap_nvox = cd.ires[0] * cd.ires[1];
        //printf ("Extending step %d\n", s);
        depth_img[ap_nvox*s + ray_data->ap_idx] = cd.accum;
    }
}

static
void
rpl_callback_accum (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum += vox_len * vox_value;

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

static
void
rpl_callback_sample (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum = 0;

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = vox_value;
}

static
void
rpl_ray_trace_callback_ct_HU (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum = 0;

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = vox_value;
}

static
void
rpl_ray_trace_callback_ct_density (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum = 0;

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = compute_density_from_HU(vox_value);
}

static
void
rpl_ray_trace_callback_PrSTPR (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum += vox_len * compute_PrSTPR_from_HU (vox_value); //vox_value = CT_HU

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

static
void
rpl_ray_trace_callback_range_length (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum += vox_len * compute_density_from_HU (vox_value); //vox_value = CT_HU

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

//Added by YKPark. Relative stopping power-based water eq. path length. Valid for 20 MeV ~ 240 MeV proton beam
//t_w = t_m * rel. density * SP ratio(m to w) = t_m*RSP
static
void
rpl_ray_trace_callback_RSP (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_vol()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    size_t step_num = vox_index + cd->step_offset;

    cd->accum += vox_len * compute_PrSTRP_XiO_MGH_weq_from_HU (vox_value); //vox_value = CT_HU

#if VERBOSE
    if (global_debug) {
	printf ("%d %4d: %20g %20g\n", ap_idx, (int) step_num, 
	    vox_value, cd->accum);
        printf ("dim = %d %d %d\n", 
            (int) rpl_vol->get_vol()->dim[0],
            (int) rpl_vol->get_vol()->dim[1],
            (int) rpl_vol->get_vol()->dim[2]);
        printf ("ap_area = %d, ap_idx = %d, vox_len = %g\n", 
            ap_area, (int) ap_idx, vox_len);
    }
#endif

    cd->last_step_completed = step_num;

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if ((plm_long) step_num >= rpl_vol->get_vol()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

