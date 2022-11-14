/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmreconstruct_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "delayload.h"
#include "drr.h"
#if CUDA_FOUND
#include "drr_cuda.h"
#endif
#include "drr_opencl.h"
#include "drr_options.h"
#include "drr_trilin.h"
#include "file_util.h"
#include "plm_image.h"
#include "plm_int.h"
#include "plm_math.h"
#include "plm_timer.h"
#include "proj_image.h"
#include "proj_matrix.h"
#include "print_and_exit.h"
#include "ray_trace.h"
#include "string_util.h"
#include "threading.h"
#include "volume.h"
#include "volume_limit.h"

class Callback_data {
public:
    Volume *vol;                 /* Volume being traced */
    int r;                       /* Row of ray */
    int c;                       /* Column of ray */
    FILE *details_fp;            /* Write ray trace details to this file */
    double accum;                /* Accumulated intensity */
    int num_pix;                 /* Number of pixels traversed */
    Hu_conversion hu_conversion; /* Should input voxels be mapped from 
                                    HU to attenuation?  How? */
public:
    Callback_data () {
        vol = 0;
        r = 0;
        c = 0;
        details_fp = 0;
        accum = 0.;
        num_pix = 0;
        hu_conversion = PREPROCESS_CONVERSION;
    }
};


//#define DRR_DEBUG_CALLBACK 1
//#define DEBUG_INTENSITIES 1

/* According to NIST, the mass attenuation coefficient of H2O at 50 keV
   is 0.22 cm^2 per gram.  Thus, we scale by 0.022 per mm
   http://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html  */
static float
attenuation_lookup_hu (float pix_density)
{
    const double min_hu = -800.0;
    const double mu_h2o = 0.022;
    if (pix_density <= min_hu) {
	return 0.0;
    } else {
	return (pix_density/1000.0) * mu_h2o + mu_h2o;
    }
}

static float
attenuation_lookup (float pix_density)
{
    return attenuation_lookup_hu (pix_density);
}

void
autoscale_image (Proj_image *proj, float range[2])
{
    float *img = proj->img;
    plm_long nvox = (plm_long) proj->dim[0] * (plm_long) proj->dim[1];
    float img_min = FLT_MAX;
    float img_max = -FLT_MAX;
    for (plm_long i = 0; i < nvox; i++) {
        if (img[i] > img_max) { img_max = img[i]; }
        if (img[i] < img_min) { img_min = img[i]; }
    }
    float offset = range[0] - img_min;
    float slope = 0;
    if (img_max - img_min > 1e-6) {
        slope = (range[1] - range[0]) / (img_max - img_min);
    }
    printf ("Src range = %f %f\n", img_min, img_max);
    printf ("Dst range = %f %f\n", range[0], range[1]);
    printf ("Slope = %f, Offset = %f\n", slope, offset);
    for (plm_long i = 0; i < nvox; i++) {
        img[i] = offset + slope * (img[i] - img_min);
    }
}


void
drr_preprocess_attenuation (Volume* vol)
{
    plm_long i;
    float* new_img;
    float* old_img;

    old_img = (float*) vol->img;
    new_img = (float*) malloc (vol->npix*sizeof(float));
    
    for (i = 0; i < vol->npix; i++) {
	new_img[i] = attenuation_lookup (old_img[i]);
    }
    vol->pix_type = PT_FLOAT;
    free (vol->img);
    vol->img = new_img;
}

void
drr_ray_trace_callback (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;

    if (cd->hu_conversion == INLINE_CONVERSION) {
        cd->accum += vox_len * attenuation_lookup (vox_value);
    } else {
        cd->accum += vox_len * vox_value;
    }

#if defined (DRR_DEBUG_CALLBACK)
    printf ("idx: %d len: %10g dens: %10g acc: %10g\n", 
	(int) vox_index, vox_len, vox_value, cd->accum);
#endif

    if (cd->details_fp) {
        plm_long ijk[3];
        COORDS_FROM_INDEX (ijk, vox_index, cd->vol->dim);
        fprintf (cd->details_fp,
            "%d,%d,%d,%d,%d,%g,%g,%g\n",
            cd->r, cd->c, (int) ijk[0], (int) ijk[1], (int) ijk[2],
            vox_len, vox_value, cd->accum);
    }

    cd->num_pix++;
}

double                            /* Return value: intensity of ray */
drr_ray_trace_exact (
    Callback_data *cd,            /* Input: callback data */
    Volume *vol,                  /* Input: volume */
    Volume_limit *vol_limit,      /* Input: min/max coordinates of volume */
    double *p1in,                 /* Input: start point for ray */
    double *p2in                  /* Input: end point for ray */
)
{
    ray_trace_exact (vol, vol_limit, &drr_ray_trace_callback, cd, 
	p1in, p2in);
    return cd->accum;
}

double                            /* Return value: intensity of ray */
drr_ray_trace_uniform (
    Callback_data *cd,            /* Input: callback data */
    Volume *vol,                  /* Input: volume */
    Volume_limit *vol_limit,      /* Input: min/max coordinates of volume */
    double *p1in,                 /* Input: start point for ray */
    double *p2in                  /* Input: end point for ray */
)
{
    float ray_step;

    /* Set ray_step proportional to voxel size */
    ray_step = vol->spacing[0];
    if (vol->dim[1] < ray_step) ray_step = vol->spacing[1];
    if (vol->dim[2] < ray_step) ray_step = vol->spacing[2];
    ray_step *= 0.75;

#if defined (commentout)
    printf ("p1 = %f %f %f\n", p1in[0], p1in[1], p1in[2]);
    printf ("p2 = %f %f %f\n", p2in[0], p2in[1], p2in[2]);
#endif

    ray_trace_uniform (vol, vol_limit, &drr_ray_trace_callback, cd, 
	p1in, p2in, ray_step);
    return cd->accum;
}

void
drr_ray_trace_image (
    Proj_image *proj, 
    Volume *vol, 
    Volume_limit *vol_limit, 
    double p1[3], 
    double ul_room[3], 
    double incr_r[3], 
    double incr_c[3], 
    Drr_options *options
)
{
    int r;
#if defined (DRR_VERBOSE)
    int rows = options->image_resolution[1];
#endif
    int cols = options->image_resolution[0];

    FILE *details_fp = 0;
    if (options->output_details_fn != "") {
        details_fp = plm_fopen (options->output_details_fn.c_str(), "w");
        if (!details_fp) {
            print_and_exit ("Failed to open %s for write\n",
                options->output_details_fn.c_str());
        }
    }

    /* Compute the drr pixels */
#pragma omp parallel for
    for (r = 0; r < options->image_resolution[1]; r++) {
	int c;
	double r_tgt[3];
	double tmp[3];
	double p2[3];

	//if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);
	vec3_copy (r_tgt, ul_room);
	vec3_scale3 (tmp, incr_r, (double) r);
	vec3_add2 (r_tgt, tmp);

	for (c = 0; c < options->image_resolution[0]; c++) {
	    double value = 0.0;
	    int idx = c + r * cols;

#if defined (DRR_VERBOSE)
	    printf ("Row: %4d/%d  Col:%4d/%d\n", r, rows, c, cols);
#endif
	    vec3_scale3 (tmp, incr_c, (double) c);
	    vec3_add3 (p2, r_tgt, tmp);

            Callback_data cd;
            cd.vol = vol;
            cd.r = r;
            cd.c = c;
            cd.details_fp = details_fp;
            cd.hu_conversion = options->hu_conversion;
	    switch (options->algorithm) {
	    case DRR_ALGORITHM_EXACT:
		value = drr_ray_trace_exact (&cd, vol, vol_limit, p1, p2);
		break;
	    case DRR_ALGORITHM_TRILINEAR_EXACT:
		value = drr_trace_ray_trilin_exact (vol, p1, p2);
		break;
	    case DRR_ALGORITHM_TRILINEAR_APPROX:
		value = drr_trace_ray_trilin_approx (vol, p1, p2);
		break;
	    case DRR_ALGORITHM_UNIFORM:
		value = drr_ray_trace_uniform (&cd, vol, vol_limit, p1, p2);
		break;
	    default:
		print_and_exit ("Error, unknown drr algorithm\n");
		break;
	    }
	    value = value / 10;     /* Translate from mm pixels to cm*gm */
	    if (options->exponential_mapping) {
		value = exp(-value);
	    }
	    value = value * options->manual_scale;   /* User requested scaling */

	    proj->img[idx] = (float) value;
	}
    }
    if (options->output_details_fn != "") {
        fclose (details_fp);
    }
}

void
drr_render_volume_perspective (
    Proj_image *proj,
    Volume *vol, 
    double ps[2], 
    void *dev_state, 
    Drr_options *options
)
{
    double p1[3];
    double ic_room[3];
    double ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double tmp[3];
    Volume_limit vol_limit;
    double nrm[3], pdn[3], prt[3];
    Proj_matrix *pmat = proj->pmat;

    pmat->get_nrm (nrm);
    pmat->get_pdn (pdn);
    pmat->get_prt (prt);

    /* Compute position of image center in room coordinates */
    vec3_scale3 (tmp, nrm, - pmat->sid);
    vec3_add3 (ic_room, pmat->cam, tmp);

    /* Compute incremental change in 3d position for each change 
       in panel row/column. */
    vec3_scale3 (incr_c, prt, ps[0]);
    vec3_scale3 (incr_r, pdn, ps[1]);

    /* Get position of upper left pixel on panel */
    vec3_copy (ul_room, ic_room);
    vec3_scale3 (tmp, incr_c, - pmat->ic[0]);
    vec3_add2 (ul_room, tmp);
    vec3_scale3 (tmp, incr_r, - pmat->ic[1]);
    vec3_add2 (ul_room, tmp);

    /* drr_ray_trace uses p1 & p2, p1 is the camera, p2 is in the 
       direction of the ray */
    vec3_copy (p1, pmat->cam);

#if defined (DRR_VERBOSE)
    printf ("NRM: %g %g %g\n", nrm[0], nrm[1], nrm[2]);
    printf ("PDN: %g %g %g\n", pdn[0], pdn[1], pdn[2]);
    printf ("PRT: %g %g %g\n", prt[0], prt[1], prt[2]);
    printf ("CAM: %g %g %g\n", pmat->cam[0], pmat->cam[1], pmat->cam[2]);
    printf ("ICR: %g %g %g\n", ic_room[0], ic_room[1], ic_room[2]);
    printf ("INCR_C: %g %g %g\n", incr_c[0], incr_c[1], incr_c[2]);
    printf ("INCR_R: %g %g %g\n", incr_r[0], incr_r[1], incr_r[2]);
    printf ("UL_ROOM: %g %g %g\n", ul_room[0], ul_room[1], ul_room[2]);
    printf ("IMG WDW: %d %d %d %d\n", 
	options->image_window[0], options->image_window[1], 
	options->image_window[2], options->image_window[3]);
#endif

    /* Compute volume boundary box */
    vol_limit.find_limit (vol);

    /* Trace the set of rays */
    switch (options->threading) {
    case THREADING_CUDA: {
#if CUDA_FOUND
        LOAD_LIBRARY_SAFE (libplmreconstructcuda);
        LOAD_SYMBOL (drr_cuda_ray_trace_image, libplmreconstructcuda);
	drr_cuda_ray_trace_image (proj, vol, &vol_limit, 
	    p1, ul_room, incr_r, incr_c, dev_state, options);
        UNLOAD_LIBRARY (libplmreconstructcuda);
	break;
#else
	/* Fall through */
#endif
    }

#if OPENCL_FOUND
    case THREADING_OPENCL:
	drr_opencl_ray_trace_image (proj, vol, &vol_limit, 
	    p1, ul_room, incr_r, incr_c, dev_state, options);
	break;
#else
	/* Fall through */
#endif

    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
	drr_ray_trace_image (proj, vol, &vol_limit, 
	    p1, ul_room, incr_r, incr_c, options);
	break;
    }
}

static void*
allocate_gpu_memory (
    Proj_image *proj,
    Volume *vol,
    Drr_options *options
)
{
#if CUDA_FOUND || OPENCL_FOUND
    void* tmp;
#endif

    switch (options->threading) {
#if CUDA_FOUND
    case THREADING_CUDA: {
        LOAD_LIBRARY_SAFE (libplmreconstructcuda);
        LOAD_SYMBOL (drr_cuda_state_create, libplmreconstructcuda);
        tmp = drr_cuda_state_create (proj, vol, options);
        UNLOAD_LIBRARY (libplmreconstructcuda);
        return tmp;
    }
#endif
#if OPENCL_FOUND
    case THREADING_OPENCL:
        tmp = drr_opencl_state_create (proj, vol, options);
        return tmp;
#endif
    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
    default:
        return 0;
    }
}

static void
free_gpu_memory (
    void *dev_state,
    Drr_options *options
)
{

    switch (options->threading) {
#if CUDA_FOUND
    case THREADING_CUDA: {
        LOAD_LIBRARY_SAFE (libplmreconstructcuda);
        LOAD_SYMBOL (drr_cuda_state_destroy, libplmreconstructcuda);
        if (dev_state) {
            drr_cuda_state_destroy (dev_state);
        }
        UNLOAD_LIBRARY (libplmreconstructcuda);
        return;
    }
#endif
#if OPENCL_FOUND
    case THREADING_OPENCL:
        if (dev_state) {
            drr_opencl_state_destroy (dev_state);
        }
        return;
#endif
    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
    default:
        return;
    }
}

static void
create_matrix_and_drr (
    Volume* vol,
    Proj_image *proj,
    double cam[3],
    double tgt[3],
    double nrm[3],
    int a,
    void *dev_state,
    Drr_options* options
)
{
    char mat_fn[256];
    char img_fn[256];
    std::string details_fn;
    Proj_matrix *pmat = proj->pmat;
    double vup[3] = {
        options->vup[0],
        options->vup[1],
        options->vup[2] };
    double sid = options->sid;
    Plm_timer* timer = new Plm_timer;

    /* Set ic = image center (in pixels), and ps = pixel size (in mm)
       Note: pixel is defined relative to the entire detector, not
       the image window, numbered from 0 to detector_resolution - 1 */
    double ic[2] = {
        options->image_center[0] - options->image_window[0],
        options->image_center[1] - options->image_window[2]
    };

    /* Set physical size of imager in mm */
    float isize[2] = {
        options->image_size[0] * ((float) options->image_resolution[0]
            / (float) options->detector_resolution[0]),
        options->image_size[1] * ((float) options->image_resolution[1]
            / (float) options->detector_resolution[1]),
    };

    /* Set pixel size in mm */
    double ps[2] = {
        (double)isize[0] / (double)options->image_resolution[0],
        (double)isize[1] / (double)options->image_resolution[1],
    };

    /* Create projection matrix */
    sprintf (mat_fn, "%s%04d.txt", options->output_prefix.c_str(), a);
    pmat->set (cam, tgt, vup, sid, ic, ps);

    if (options->output_format == OUTPUT_FORMAT_PFM) {
        sprintf (img_fn, "%s%04d.pfm", options->output_prefix.c_str(), a);
    } else if (options->output_format == OUTPUT_FORMAT_PGM) {
        sprintf (img_fn, "%s%04d.pgm", options->output_prefix.c_str(), a);
    } else {
        sprintf (img_fn, "%s%04d.raw", options->output_prefix.c_str(), a);
    }

    if (options->output_details_prefix != "") {
        options->output_details_fn = string_format ("%s%04d.txt",
            options->output_details_prefix.c_str(), a);
    }

    if (options->geometry_only) {
        proj->save (0, mat_fn);
    } else {
        drr_render_volume_perspective (proj, vol, ps, dev_state, options);
        if (options->autoscale) {
            autoscale_image (proj, options->autoscale_range);
        }
        timer->start ();
        // If defined output file, then use it, otherwise an old method
        if (options->output_file != "") {
          proj->save (options->output_file.c_str(), nullptr);
        }
        else {
          proj->save (img_fn, mat_fn);
        }
        printf ("I/O time: %f sec\n", timer->report ());
    }

    delete timer;
}

/* All distances in mm */
void
drr_render_volume (Volume* vol, Drr_options* options)
{
    Proj_image *proj;
    int a;
    void *dev_state = 0;

    /* tgt is isocenter */
    double tgt[3] = {
        options->isocenter[0],
        options->isocenter[1],
        options->isocenter[2] };

    Plm_timer* timer = new Plm_timer;
    timer->start ();

    /* Allocate data for image and matrix */
    proj = new Proj_image;
    proj_image_create_pmat (proj);
    proj_image_create_img (proj, options->image_resolution);

    /* Allocate memory on the gpu device */
    dev_state = allocate_gpu_memory (proj, vol, options);

    /* If nrm was specified, only create a single image */
    if (options->have_nrm) {
        double cam[3];
        double nrm[3] = {
            options->nrm[0],
            options->nrm[1],
            options->nrm[2] };

        /* Make sure nrm is normal */
        vec3_normalize1 (nrm);

        /* Place camera at distance "sad" from the volume isocenter */
        cam[0] = tgt[0] + options->sad * nrm[0];
        cam[1] = tgt[1] + options->sad * nrm[1];
        cam[2] = tgt[2] + options->sad * nrm[2];

        create_matrix_and_drr (vol, proj, cam, tgt, nrm, 0,
            dev_state, options);
    }

    /* Otherwise, loop through camera angles */
    else {
        for (a = 0; a < options->num_angles; a++) {
            double angle = options->start_angle + a * options->angle_diff;
            double cam[3];
            double nrm[3];

            printf ("Rendering DRR %d\n", a);

            /* Place camera at distance "sad" from the volume isocenter */
            cam[0] = tgt[0] + options->sad * cos(angle);
            cam[1] = tgt[1] - options->sad * sin(angle);
            cam[2] = tgt[2];

            /* Compute normal vector */
            vec3_sub3 (nrm, tgt, cam);
            vec3_normalize1 (nrm);

            create_matrix_and_drr (vol, proj, cam, tgt, nrm, a,
                dev_state, options);
        }
    }
    delete proj;

    free_gpu_memory (dev_state, options);

    printf ("Total time: %g secs\n", timer->report ());

    delete timer;
}

void
set_isocenter (Volume* vol, Drr_options* options)
{
    vol->origin[0] -= options->isocenter[0];
    vol->origin[1] -= options->isocenter[1];
    vol->origin[2] -= options->isocenter[2];
}

void
drr_compute (Drr_options *options)
{
    Plm_image::Pointer plm_image = Plm_image::New();
    Volume* vol = 0;

    if (options->geometry_only) {
        options->threading = THREADING_CPU_SINGLE;
    }
    else {
        plm_image->load_native (options->input_file);
        if (!plm_image->have_image()) {
            /* GCS FIX: Error handling */
            return;
        }
        plm_image->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
        vol = plm_image->get_vol ();
    }

    if (options->hu_conversion == PREPROCESS_CONVERSION
        && !options->geometry_only)
    {
        drr_preprocess_attenuation (vol);
    }

    drr_render_volume (vol, options);
}
