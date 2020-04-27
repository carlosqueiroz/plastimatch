/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
    Proposed variable naming guide:
        Fixed image voxel                (f[3]), fidx <currently (fi,fj,fk),fv>
        Moving image voxel               (m[3]), midx < ditto >
            - what about ROI's              ?
            - what about physical coords    ?
        Tile (fixed)                        (t[3]), tidx <currently p[3]>
        Offset within tile (fixed)          (o[3]), oidx <currently q[3]>
        Control point                       (c[3]), cidx <currently (i,j,k), m>
        Coefficient array                   ?                <currently cidx>
        Multiplier LUT                      qidx
        Index LUT                           pidx

    qlut = Multiplier LUT
    clut = Index LUT
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif

#include "bspline.h"
#include "bspline_gm.h"
#include "bspline_interpolate.h"
#include "bspline_landmarks.h"
#include "bspline_macros.h"
#include "bspline_mi.h"
#include "bspline_mse.h"
#include "bspline_optimize.h"
#include "bspline_parms.h"
#include "bspline_regularize.h"
#include "bspline_state.h"
#include "bspline_xform.h"
#include "delayload.h"
#include "file_util.h"
#include "interpolate.h"
#include "interpolate_macros.h"
#include "joint_histogram.h"
#include "logfile.h"
#include "plm_math.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "volume.h"
#include "volume_header.h"
#include "volume_macros.h"


/* Stub */
void bspline_score_pd (Bspline_optimize *bod)
{
    //printf ("Hello from bspline_score_pd!\n");
    
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();
    Bspline_score *ssd = &bst->ssd;
    
    //Volume *fixed_image  = bst->fixed;
    Volume *moving_image = bst->moving;
    Volume *moving_grad = bst->moving_grad;

    float* m_image = (float*) moving_image->img;
    float* m_grad = (float*) moving_grad->img;
    Labeled_pointset* fixed_pointset = bst->fixed_pointset;
    
    //printf ("fixed = %p\n", bst->fixed);
    //printf ("moving = %p\n", bst->moving);
    //printf ("fixed_pointset = %p\n", bst->fixed_pointset);
    /*if (bst->fixed_pointset) {
      printf ("fixed_pointset has %zd points\n", bst->fixed_pointset->get_count());
      }*/
    
    plm_long fijk[3];
    float mijk[3];
    float fxyz[3];
    float mxyz[3];
    plm_long mijk_f[3], mvf;
    plm_long mijk_r[3], mvr;
    plm_long p[3], pidx;
    plm_long q[3], qidx;

    float dc_dv[3];
    float li_1[3];
    float li_2[3];
    float m_x, m_y, m_z;
    float inv_rx, inv_ry, inv_rz;
    inv_rx = 1.0/moving_image->spacing[0];
    inv_ry = 1.0/moving_image->spacing[1];
    inv_rz = 1.0/moving_image->spacing[2];
    double score_acc =0.;
    int num_points = bst->fixed_pointset->get_count();
    int idx;
    float *moving_value;

    Volume_header fixed_vh;
    fixed_vh.set_from_bxf (bxf);
    Volume::Pointer fixed_image = Volume::New (fixed_vh, PT_FLOAT, 1);
    
    /* Compute interpolation fractions */
    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving_image);
    mvf = volume_index(moving_image->dim, mijk_f);	

    int points_used = 0;
    for (int i = 0; i < num_points; i++) {
        const Labeled_point& fp = bst->fixed_pointset->point_list[i];
        plm_long landmark_ijk[3];
        float landmark_xyz[3];
        landmark_xyz[0] = fp.p[0];
        landmark_xyz[1] = fp.p[1];
        landmark_xyz[2] = fp.p[2];
        bool is_inside = false;
        fixed_image->get_ijk_from_xyz (landmark_ijk, landmark_xyz, &is_inside);
        if (!is_inside) {
            continue;
        }

        plm_long *fijk = landmark_ijk;
        p[2] = REGION_INDEX_Z (fijk, bxf);
        q[2] = REGION_OFFSET_Z (fijk, bxf);
        p[1] = REGION_INDEX_Y (fijk, bxf);
        q[1] = REGION_OFFSET_Y (fijk, bxf);
        p[0] = REGION_INDEX_X (fijk, bxf);
        q[0] = REGION_OFFSET_X (fijk, bxf);
        pidx = volume_index (bxf->rdims, p);
        qidx = volume_index (bxf->vox_per_rgn, q);
        
        /*printf("%g,%g,%g\n",landmark_ijk[0],landmark_ijk[1],landmark_ijk[2]);*/
        
        idx = moving_image->get_idx_from_xyz (landmark_xyz, &is_inside);
        if (!is_inside) {
            continue;
        }
        score_acc += m_image[idx];
        //printf("%f",score_acc);
	    
        /*
          m_x = li_value_dx ( 
          li_1, li_2, inv_rx, 
          mvf,
          m_image, moving_image,
          );
		
          m_y = li_value_dy ( 
          li_1, li_2, inv_ry, 
          mvf,
          m_image, moving_image
          );
          m_z = li_value_dz ( 
          li_1, li_2, inv_rz, 
          mvf,
          m_image, moving_image
          );*/

        /* Compute spatial gradient using nearest neighbors */
        //mvr = volume_index (moving->dim, mijk_r);
        dc_dv[0] = -m_grad[3*idx + 0];  /* x component */
        dc_dv[1] = -m_grad[3*idx + 1];  /* y component */
        dc_dv[2] = -m_grad[3*idx + 2];  /* z component */

        bst->ssd.update_smetric_grad_b (bxf, pidx, qidx, dc_dv);

        points_used++;
    }

    /* Normalize score for MSE */
    if (points_used > 0) {
        ssd->curr_smetric = score_acc / points_used;
        for (int i = 0; i < num_points; i++) {
            ssd->curr_smetric_grad[i] = ssd->curr_smetric_grad[i] / points_used;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: calc_offsets()
//
// This function accepts the number or voxels per control region
// and the dimensions of the control grid to generate where the linear
// memory offsets lie for the beginning of each tile in a 32-byte
// aligned tile-major data organization scheme (such as that which
// is produced by kernel_row_to_tile_major().
//
// Author: James Shackleford
// Data: July 30th, 2009
////////////////////////////////////////////////////////////////////////////////
int*
calc_offsets (int* tile_dims, int* cdims)
{
    int vox_per_tile = (tile_dims[0] * tile_dims[1] * tile_dims[2]);
    int pad = 32 - (vox_per_tile % 32);
    int num_tiles = (cdims[0]-3)*(cdims[1]-3)*(cdims[2]-3);

    int* output = (int*)malloc(num_tiles*sizeof(int));

    int i;
    for(i = 0; i < num_tiles; i++)
        output[i] = (vox_per_tile + pad) * i;

    return output;
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: find_knots()
//
// This function takes a tile index as an input and generates
// the indicies of the 64 control knots that it affects.
//
// Returns:
//    knots[idx] - idx is [0,63] and knots[idx] = the linear knot index
//                 of affected control knot # idx within the entire control
//                 knot grid.
//
//    tile_pos[
//
// Author: James Shackleford
// Data: July 13th, 2009
////////////////////////////////////////////////////////////////////////////////
void
find_knots (
    plm_long* knots, 
    plm_long tile_num, 
    plm_long* rdims, 
    plm_long* cdims
) {
    int tile_loc[3];
    int i, j, k;
    int idx = 0;
    int num_tiles_x = cdims[0] - 3;
    int num_tiles_y = cdims[1] - 3;
    int num_tiles_z = cdims[2] - 3;
        
    // First get the [x,y,z] coordinate of
    // the tile in the control grid.
    tile_loc[0] = tile_num % num_tiles_x;
    tile_loc[1] = ((tile_num - tile_loc[0]) / num_tiles_x) % num_tiles_y;
    tile_loc[2] = ((((tile_num - tile_loc[0]) / num_tiles_x) / num_tiles_y) % num_tiles_z);

    // Tiles do not start on the edges of the grid, so we
    // push them to the center of the control grid.
    tile_loc[0]++;
    tile_loc[1]++;
    tile_loc[2]++;

    // Find 64 knots' [x,y,z] coordinates
    // and convert into a linear knot index
    for (k = -1; k < 3; k++) {
        for (j = -1; j < 3; j++) {
            for (i = -1; i < 3; i++)
            {
                knots[idx++] = (cdims[0]*cdims[1]*(tile_loc[2]+k)) + (cdims[0]*(tile_loc[1]+j)) + (tile_loc[0]+i);
            }
        }
    }
}


/* -----------------------------------------------------------------------
   Debugging routines
   ----------------------------------------------------------------------- */
void
dump_total_gradient (Bspline_xform* bxf, Bspline_score* ssd, const char* fn)
{
    int i;
    FILE* fp;

    make_parent_directories (fn);
    fp = fopen (fn, "wb");
    for (i = 0; i < bxf->num_coeff; i++) {
        fprintf (fp, "%20.20f\n", ssd->total_grad[i]);
    }
    fclose (fp);
}

void
bspline_display_coeff_stats (Bspline_xform* bxf)
{
    float cf_min, cf_avg, cf_max;
    int i;

    cf_avg = 0.0;
    cf_min = cf_max = bxf->coeff[0];
    for (i = 0; i < bxf->num_coeff; i++) {
        cf_avg += bxf->coeff[i];
        if (cf_min > bxf->coeff[i]) cf_min = bxf->coeff[i];
        if (cf_max < bxf->coeff[i]) cf_max = bxf->coeff[i];
    }
    logfile_printf ("         "
                    "CMIN %6.2f CAVG %6.2f CMAX %6.2f\n", 
                    cf_min, cf_avg / bxf->num_coeff, cf_max);
}

void
bspline_save_debug_state (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform *bxf
)
{
    if (parms->debug) {
        std::string fn;
        char buf[1024];

        sprintf (buf, "%02d_grad_%03d_%03d.txt", 
            parms->debug_stage, bst->it, bst->feval);
        fn = parms->debug_dir + "/" + buf;
        dump_total_gradient (bxf, &bst->ssd, fn.c_str());

        sprintf (buf, "%02d_coeff_%03d_%03d.txt", 
            parms->debug_stage, bst->it, bst->feval);
        fn = parms->debug_dir + "/" + buf;
        bxf->save (fn.c_str());

        if (bst->has_metric_type (SIMILARITY_METRIC_MI_MATTES)) {
            sprintf (buf, "%02d_", parms->debug_stage);
            fn = parms->debug_dir + "/" + buf;
            bst->get_mi_hist()->dump_hist (bst->feval, fn);
        }
    }
}

Volume*
bspline_compute_vf (const Bspline_xform* bxf)
{
    Volume* vf = new Volume (
        bxf->img_dim, bxf->img_origin, 
        bxf->img_spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 3
    );
    bspline_interpolate_vf (vf, bxf);

    return vf;
}

void
bspline_update_sets (float* sets_x, float* sets_y, float* sets_z,
    int qidx, float* dc_dv, Bspline_xform* bxf)
{
    int sidx;   // set index

    /* Initialize q_lut */
    float* q_lut = &bxf->q_lut[64*qidx];

    /* Condense dc_dv & commit to sets for tile */
    for (sidx=0; sidx<64; sidx++) {
        sets_x[sidx] += dc_dv[0] * q_lut[sidx];
        sets_y[sidx] += dc_dv[1] * q_lut[sidx];
        sets_z[sidx] += dc_dv[2] * q_lut[sidx];
    }
}

void
bspline_update_sets_b (float* sets_x, float* sets_y, float* sets_z,
    plm_long *q, float* dc_dv, Bspline_xform* bxf)
{
    int i,j,k,m;
    float A,B,C;

    /* Initialize b_luts */
    float* bx_lut = &bxf->bx_lut[q[0]*4];
    float* by_lut = &bxf->by_lut[q[1]*4];
    float* bz_lut = &bxf->bz_lut[q[2]*4];

    /* Condense dc_dv & commit to sets for tile */
    m=0;
    for (k=0; k<4; k++) {
        C = bz_lut[k];
        for (j=0; j<4; j++) {
            B = by_lut[j] * C;
            for (i=0; i<4; i++) {
                A = bx_lut[i] * B;
                sets_x[m] += dc_dv[0] * A;
                sets_y[m] += dc_dv[1] * A;
                sets_z[m] += dc_dv[2] * A;
                m++;
            }
        }
    }
}


void
bspline_sort_sets (float* cond_x, float* cond_y, float* cond_z,
    float* sets_x, float* sets_y, float* sets_z,
    plm_long pidx, Bspline_xform* bxf)
{
    int sidx, kidx;
    plm_long* k_lut = (plm_long*) malloc (64*sizeof(plm_long));

    /* Generate the knot index lut */
    find_knots (k_lut, pidx, bxf->rdims, bxf->cdims);

    /* Rackem' Up */
    for (sidx=0; sidx<64; sidx++) {
        kidx = k_lut[sidx];

        cond_x[ (64*kidx) + sidx ] = sets_x[sidx];
        cond_y[ (64*kidx) + sidx ] = sets_y[sidx];
        cond_z[ (64*kidx) + sidx ] = sets_z[sidx];
    }

    free (k_lut);
}

void
bspline_condense_smetric_grad (float* cond_x, float* cond_y, float* cond_z,
    Bspline_xform* bxf, Bspline_score* ssd)
{
    plm_long kidx, sidx;

    for (kidx=0; kidx < bxf->num_knots; kidx++) {
        for (sidx=0; sidx<64; sidx++) {
            ssd->curr_smetric_grad[3*kidx + 0] += cond_x[64*kidx + sidx];
            ssd->curr_smetric_grad[3*kidx + 1] += cond_y[64*kidx + sidx];
            ssd->curr_smetric_grad[3*kidx + 2] += cond_z[64*kidx + sidx];
        }
    }
}

static void
logfile_print_score (float score)
{
    if (score < 10. && score > -10.) {
        logfile_printf (" %1.7f ", score);
    } else {
        logfile_printf (" %9.3f ", score);
    }
}

void
report_score (
    Bspline_parms *parms,
    Bspline_xform *bxf, 
    Bspline_state *bst
) 
{
    Bspline_score* ssd = &bst->ssd;
    const Regularization_parms* rparms = parms->regularization_parms;
    Bspline_landmarks* blm = parms->blm;

    int i;
    double ssd_grad_norm, ssd_grad_mean;

    /* Compute gradient statistics */
    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
        ssd_grad_mean += bst->ssd.total_grad[i];
        ssd_grad_norm += (double) bst->ssd.total_grad[i]
            * (double) bst->ssd.total_grad[i];
    }

    /* Compute total time */
    double total_smetric_time = 0;
    double total_time = 0;
    plm_long hack_num_vox = 0;
    std::vector<Metric_score>::const_iterator it_mr 
        = ssd->metric_record.begin();
    while (it_mr != ssd->metric_record.end()) {
        total_time += it_mr->time;
        if (hack_num_vox == 0) {
            hack_num_vox = it_mr->num_vox;
        }
        ++it_mr;
    }
    total_smetric_time = total_time;
    total_time += ssd->time_rmetric;
    
    /* First line, iterations, score, misc stats */
    logfile_printf ("[%2d,%3d] ", bst->it, bst->feval);
    if (blm->num_landmarks > 0 
        || bst->similarity_data.size() > 1
        || rparms->total_displacement_penalty > 0 
        || rparms->diffusion_penalty > 0 
        || rparms->curvature_penalty > 0 
        || rparms->lame_coefficient_1 > 0 
        || rparms->lame_coefficient_2 > 0 
        || rparms->third_order_penalty > 0)
    {
        logfile_printf ("SCORE ");
    } else {
        logfile_printf ("%-6s", 
            bst->similarity_data.front()->metric_string());
    }
    logfile_print_score (ssd->total_score);
    logfile_printf (
        "NV %6d GM %9.3f GN %9.3g [ %9.3f s ]\n",
        hack_num_vox, ssd_grad_mean, sqrt (ssd_grad_norm), total_time);
    
    /* Second line */
    if (blm->num_landmarks > 0 
        || bst->similarity_data.size() > 1
        || rparms->total_displacement_penalty > 0 
        || rparms->diffusion_penalty > 0 
        || rparms->curvature_penalty > 0 
        || rparms->lame_coefficient_1 > 0 
        || rparms->lame_coefficient_2 > 0 
        || rparms->third_order_penalty > 0)
    {
        logfile_printf ("         ");
        /* Part 1 - smetric(s) */
        /* GCS FIX: It should not be that one of these is a list 
           and the other is a vector. */
        std::vector<Metric_score>::const_iterator it_mr 
            = ssd->metric_record.begin();
        std::list<Metric_state::Pointer>::const_iterator it_st
            = bst->similarity_data.begin();
        while (it_mr != ssd->metric_record.end()) {
            logfile_printf ("%-6s", (*it_st)->metric_string());
            logfile_print_score (it_mr->score);
            ++it_mr, ++it_st;
        }
    }
    if (ssd->metric_record.size() > 1 
        && (blm->num_landmarks > 0 
            || rparms->total_displacement_penalty > 0 
            || rparms->diffusion_penalty > 0 
            || rparms->curvature_penalty > 0 
            || rparms->lame_coefficient_1 > 0 
            || rparms->lame_coefficient_2 > 0 
            || rparms->third_order_penalty > 0))

    {
        logfile_printf ("\n");
        logfile_printf ("         ");
    }
    if (rparms->total_displacement_penalty > 0 
        || rparms->diffusion_penalty > 0 
        || rparms->curvature_penalty > 0 
        || rparms->lame_coefficient_1 > 0 
        || rparms->lame_coefficient_2 > 0 
        || rparms->third_order_penalty > 0) 
    {	
        logfile_printf ("RM %9.3f ", bst->ssd.rmetric);
    }
    /* Part 3 - landmark metric */
    if (blm->num_landmarks > 0) {
        logfile_printf ("LM %9.3f ", 
            blm->landmark_stiffness * bst->ssd.lmetric);
    }
    /* Part 4 - timing */
    if (rparms->total_displacement_penalty > 0 
        || rparms->diffusion_penalty > 0 
        || rparms->curvature_penalty > 0 
        || rparms->lame_coefficient_1 > 0 
        || rparms->lame_coefficient_2 > 0 
        || rparms->third_order_penalty > 0) 
    {
        logfile_printf ("[ %9.3f | %9.3f ]", 
            total_smetric_time, ssd->time_rmetric);
    }
    if (blm->num_landmarks > 0 
        || rparms->total_displacement_penalty > 0 
        || rparms->diffusion_penalty > 0 
        || rparms->curvature_penalty > 0 
        || rparms->lame_coefficient_1 > 0 
        || rparms->lame_coefficient_2 > 0 
        || rparms->third_order_penalty > 0)
    {
        logfile_printf ("\n");
    }
}
void
bspline_score (Bspline_optimize *bod)
{
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();

    const Regularization_parms* rparms = parms->regularization_parms;
    Bspline_landmarks* blm = parms->blm;

    /* Zero out the score for this iteration */
    bst->ssd.reset_score ();

    /* Compute similarity metric.  This is done for each metric 
       and each similarity metric within each image plane. */
    std::list<Metric_state::Pointer>::const_iterator it_sd;
    bst->sm = 0;
    for (it_sd = bst->similarity_data.begin();
         it_sd != bst->similarity_data.end(); ++it_sd)
    {
        bst->set_metric_state (*it_sd);
        bst->initialize_similarity_images ();
        Plm_timer timer;
        timer.start ();

        switch ((*it_sd)->metric_type) {
        case SIMILARITY_METRIC_DMAP_DMAP:
        case SIMILARITY_METRIC_MSE:
            bspline_score_mse (bod);
            break;
        case SIMILARITY_METRIC_MI_MATTES:
            bspline_score_mi (bod);
            break;
        case SIMILARITY_METRIC_GM:
            bspline_score_gm (bod);
            break;
        case SIMILARITY_METRIC_POINT_DMAP:
            bspline_score_pd (bod);
            break;
        default:
            print_and_exit (
                "Unknown similarity metric in bspline_score()\n");
            break;
        }

        bst->ssd.metric_record.push_back (
            Metric_score (bst->ssd.curr_smetric, timer.report (),
                bst->ssd.curr_num_vox));
#if defined (commentout)
        printf (">> %f + %f * %f ->",
            bst->ssd.total_score, (*it_sd)->metric_lambda, 
            bst->ssd.curr_smetric);
#endif
        bst->ssd.accumulate ((*it_sd)->metric_lambda);
#if defined (commentout)
        printf (" %f\n", bst->ssd.total_score);
#endif
        bst->sm ++;
    }

    /* Compute regularization */
    if (rparms->implementation != '\0') {
        bst->rst.compute_score (&bst->ssd, rparms, bxf);
        bst->ssd.total_score += bst->ssd.rmetric; 
    }

    /* Compute landmark score/gradient to image score/gradient */
    if (blm->num_landmarks > 0) {
        bspline_landmarks_score (parms, bst, bxf);
        bst->ssd.total_score += blm->landmark_stiffness * bst->ssd.lmetric;
    }

    /* Report results of this iteration */
    report_score (parms, bxf, bst);
}
