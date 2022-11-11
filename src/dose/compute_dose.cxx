/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aperture.h"
#include "beam_calc.h"
#include "compute_dose.h"
#include "dose_volume_functions.h"
#include "interpolate.h"
#include "plm_image.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "ray_data.h"
#include "ray_trace.h"
#include "rpl_volume.h"
#include "rpl_volume_lut.h"
#include "rt_depth_dose.h"
#include "rt_dij.h"
#include "rt_lut.h"
#include "rt_mebs.h"
#include "rt_sigma.h"
#include "string_util.h"
#include "threading.h"
#include "volume.h"

/* Ray Tracer */
double
energy_direct (
    float rgdepth,          /* voxel to dose */
    Beam_calc* beam,
    int beam_idx
)
{
    /* The voxel was not hit directly by the beam */
    if (rgdepth <= 0.0f) {
        return 0.0f;
    }

    /* return the dose at this radiographic depth */
    return (double) beam->get_mebs()->get_depth_dose()[beam_idx]->lookup_energy(rgdepth);
}

void
compute_dose_a (
    Volume::Pointer dose_vol, 
    Beam_calc* beam, 
    const Volume::Pointer ct_vol
)
{
    float* dose_img = (float*) dose_vol->img;

    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }

    /* Dose D(POI) = Dose(z_POI) but z_POI =  rg_comp + depth in CT, 
       if there is a range compensator */
    if (ap->have_range_compensator_image()) {
        beam->add_rcomp_length_to_rpl_volume ();
    }

    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double idx_ap[2] = {0,0};
    plm_long idx_ap_int[2] = {0,0};
    double rest[2] = {0,0};
    double particle_number = 0;
    float WER = 0;
    float rgdepth = 0;

    for (ct_ijk[2] = 0; ct_ijk[2] < ct_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < ct_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < ct_vol->dim[0]; ct_ijk[0]++) {
                double dose = 0.0;

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->origin[0] + ct_ijk[0] * ct_vol->spacing[0]);
                ct_xyz[1] = (double) (ct_vol->origin[1] + ct_ijk[1] * ct_vol->spacing[1]);
                ct_xyz[2] = (double) (ct_vol->origin[2] + ct_ijk[2] * ct_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;

                if (beam->get_intersection_with_aperture(idx_ap, idx_ap_int, rest, ct_xyz) == false)
                {
                    continue;
                }

                /* Check that the ray cross the aperture */
                if (idx_ap[0] < 0 || idx_ap[0] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(0)-1
                    || idx_ap[1] < 0 || idx_ap[1] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(1)-1)
                {
                    continue;
                }

                /* Check that the ray cross the active part of the aperture */
                if (ap_img && beam->is_ray_in_the_aperture(idx_ap_int, ap_img) == false)
                {
                    continue;
                }

                dose = 0;
                rgdepth = beam->rsp_accum_vol->get_value (ct_xyz);
                WER = compute_PrWER_from_HU (beam->hu_samp_vol->get_value(ct_xyz));

                const Rt_mebs::Pointer& mebs = beam->get_mebs();
                for (size_t dd_idx = 0; dd_idx < mebs->get_depth_dose().size(); dd_idx++)
                {
                    particle_number = mebs->get_particle_number_xyz (idx_ap_int, rest, dd_idx, beam->get_aperture()->get_dim());
                    if (particle_number != 0 && rgdepth >=0 && rgdepth < mebs->get_depth_dose()[dd_idx]->dend) 
                    {
                        dose += particle_number * WER * energy_direct (rgdepth, beam, dd_idx);
                    }
                }

                /* Insert the dose into the dose volume */
                idx = volume_index (dose_vol->dim, ct_ijk);
                dose_img[idx] = dose;
            }
        }
    }
}

void
compute_dose_b (
    Beam_calc* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol
)
{
    Rpl_volume *wepl_rv = beam->rsp_accum_vol;
    Volume *wepl_vol = wepl_rv->get_vol();
    float *wepl_img = wepl_vol->get_raw<float> ();

    Rpl_volume *dose_rv = beam->dose_rv;
    Volume *dose_rv_vol = dose_rv->get_vol();
    float *dose_rv_img = dose_rv_vol->get_raw<float> ();

    Rt_mebs::Pointer mebs = beam->get_mebs();
    const Rt_depth_dose *depth_dose = mebs->get_depth_dose()[energy_index];
    std::vector<float>& num_part = mebs->get_num_particles();

    /* scan through rpl volume */
    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }
    const plm_long *dim = wepl_rv->get_image_dim();
    int num_steps = wepl_rv->get_num_steps();
    plm_long ij[2] = {0,0};
    for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
        for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
            if (ap_img && ap_img[ap_vol->index(ij[0],ij[1],0)] == 0) {
                continue;
            }
            size_t np_index = energy_index * dim[0] * dim[1]
                + ij[1] * dim[0] + ij[0];
            float np = num_part[np_index];
            if (np == 0.f) {
                continue;
            }
            for (int s = 0; s < num_steps; s++) {
                int dose_index = ap_vol->index(ij[0],ij[1],s);
                float wepl = wepl_img[dose_index];
                dose_rv_img[dose_index] += np * depth_dose->lookup_energy(wepl);
            }
        }
    }
}

void
compute_dose_ray_trace_dij_a (
    Beam_calc* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol,
    Volume::Pointer& dose_vol
)
{
    float* dose_img = (float*) dose_vol->img;

    /* Dose D(POI) = Dose(z_POI) but z_POI =  rg_comp + depth in CT, 
       if there is a range compensator */
    if (beam->get_aperture()->have_range_compensator_image())
    {
        beam->add_rcomp_length_to_rpl_volume ();
    }

    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double idx_ap[2] = {0,0};
    plm_long idx_ap_int[2] = {0,0};
    double rest[2] = {0,0};
    unsigned char* ap_img = (unsigned char*) beam->get_aperture()->get_aperture_volume()->img;
    double particle_number = 0;
    float WER = 0;
    float rgdepth = 0;

    for (ct_ijk[2] = 0; ct_ijk[2] < ct_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < ct_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < ct_vol->dim[0]; ct_ijk[0]++) {
                double dose = 0.0;

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->origin[0] + ct_ijk[0] * ct_vol->spacing[0]);
                ct_xyz[1] = (double) (ct_vol->origin[1] + ct_ijk[1] * ct_vol->spacing[1]);
                ct_xyz[2] = (double) (ct_vol->origin[2] + ct_ijk[2] * ct_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;

                if (beam->get_intersection_with_aperture (idx_ap, idx_ap_int, rest, ct_xyz) == false)
                {
                    continue;
                }

                /* Check that the ray cross the aperture */
                if (idx_ap[0] < 0 || idx_ap[0] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(0)-1
                    || idx_ap[1] < 0 || idx_ap[1] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(1)-1)
                {
                    continue;
                }

                /* Check that the ray cross the active part of the aperture */
                if (beam->get_aperture()->have_aperture_image() && beam->is_ray_in_the_aperture (idx_ap_int, ap_img) == false)
                {
                    continue;
                }

                dose = 0;
                rgdepth = beam->rsp_accum_vol->get_value (ct_xyz);
                WER = compute_PrWER_from_HU (beam->hu_samp_vol->get_value(ct_xyz));

                const Rt_mebs::Pointer& mebs = beam->get_mebs();
                for (size_t dd_idx = 0; dd_idx < mebs->get_depth_dose().size(); dd_idx++)
                {
                    particle_number = mebs->get_particle_number_xyz (idx_ap_int, rest, dd_idx, beam->get_aperture()->get_dim());
                    if (particle_number != 0 && rgdepth >=0 && rgdepth < mebs->get_depth_dose()[dd_idx]->dend) 
                    {
                        dose += particle_number * WER * energy_direct (rgdepth, beam, dd_idx);
                    }
                }

                /* Insert the dose into the dose volume */
                idx = volume_index (dose_vol->dim, ct_ijk);
                dose_img[idx] = dose;
            }
        }
    }
}

void
compute_dose_ray_trace_dij_b (
    Beam_calc* beam,
    const Volume::Pointer ct_vol,
    Volume::Pointer& dose_vol
)
{
    Rpl_volume *wepl_rv = beam->rsp_accum_vol;
    Volume *wepl_vol = wepl_rv->get_vol();
    float *wepl_img = wepl_vol->get_raw<float> ();

    Rpl_volume *dose_rv = beam->dose_rv;
    Volume *dose_rv_vol = dose_rv->get_vol();
    float *dose_rv_img = dose_rv_vol->get_raw<float> ();

    Rt_mebs::Pointer mebs = beam->get_mebs();
    const std::vector<Rt_depth_dose*> depth_dose = mebs->get_depth_dose();
    std::vector<float>& num_part = mebs->get_num_particles();

    /* Create the beamlet dij matrix */
    Rt_dij rt_dij;

    /* Create geometry map from volume to rpl_volume */
    Rpl_volume_lut rpl_volume_lut (dose_rv, dose_vol.get());
    rpl_volume_lut.build_lut ();

    /* scan through rpl volume */
    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }
    const plm_long *dim = wepl_rv->get_image_dim();
    int num_steps = wepl_rv->get_num_steps();
    plm_long ij[2] = {0,0};
    for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
        for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
            if (ap_img && ap_img[ap_vol->index(ij[0],ij[1],0)] == 0) {
                continue;
            }
            for (size_t energy_index = 0; 
                 energy_index < depth_dose.size(); 
                 energy_index++)
            {
                // Get beamlet weight
                size_t np_index = energy_index * dim[0] * dim[1]
                    + ij[1] * dim[0] + ij[0];
                float np = num_part[np_index];
                if (np == 0.f) {
                    continue;
                }

                // Fill in dose
                const Rt_depth_dose *dd = depth_dose[energy_index];
                for (int s = 0; s < num_steps; s++) {
                    int dose_index = ap_vol->index(ij[0],ij[1],s);
                    float wepl = wepl_img[dose_index];
                    dose_rv_img[dose_index] = np * dd->lookup_energy(wepl);
                }

                // Create beamlet dij
                rt_dij.set_from_dose_rv (
                    ij, energy_index, dose_rv, ct_vol);

                // Zero out again
                for (int s = 0; s < num_steps; s++) {
                    int dose_index = ap_vol->index(ij[0],ij[1],s);
                    dose_rv_img[dose_index] = 0.f;
                }
            }
        }
    }

    // Write beamlet dij
    if (beam->get_dij_out() != "") {
        rt_dij.dump (beam->get_dij_out());
    }
}

void
compute_dose_d (
    Beam_calc* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol
)
{
    beam->get_rt_dose_timing()->timer_dose_calc.resume ();
    Rpl_volume *wepl_rv = beam->rsp_accum_vol;
    Volume *wepl_vol = wepl_rv->get_vol();
    float *wepl_img = wepl_vol->get_raw<float> ();

    Rpl_volume *dose_rv = beam->dose_rv;
    Volume *dose_rv_vol = dose_rv->get_vol();
    float *dose_rv_img = dose_rv_vol->get_raw<float> ();

    Rt_mebs::Pointer mebs = beam->get_mebs();
    const Rt_depth_dose *depth_dose = mebs->get_depth_dose()[energy_index];
    std::vector<float>& num_part = mebs->get_num_particles();
    beam->get_rt_dose_timing()->timer_dose_calc.stop ();

    // Compute sigma for this energy
    beam->get_rt_dose_timing()->timer_sigma.resume ();
    int margins[2] = {0,0};
    float sigma_max = 0;
    compute_sigmas (beam, depth_dose->E0, &sigma_max, "small", margins);
    beam->get_rt_dose_timing()->timer_sigma.stop ();

    beam->get_rt_dose_timing()->timer_dose_calc.resume ();
    Rpl_volume *sigma_rv = beam->sigma_vol;
    Volume *sigma_vol = sigma_rv->get_vol();
    float *sigma_img = sigma_vol->get_raw<float> ();
    const plm_long *sigma_dim = sigma_vol->get_dim();

    // Get the variable magnification at each step
    std::vector <double> lateral_spacing_0 (sigma_dim[2],0);
    std::vector <double> lateral_spacing_1 (sigma_dim[2],0);
    double sid = sigma_rv->get_aperture()->get_distance();
    const double *ap_spacing = sigma_rv->get_aperture()->get_spacing();
    float clipping_dist = sigma_rv->get_front_clipping_plane();
    float step_length = sigma_rv->get_step_length ();
    for (int k = 0; k < sigma_dim[2]; k++) {
        float mag = (sid + clipping_dist + k * step_length) / sid;
        lateral_spacing_0[k] = ap_spacing[0] * mag;
        lateral_spacing_1[k] = ap_spacing[1] * mag;
    }

    // Compute lateral search distance (2.5 max sigma) for each depth
    // (Only needed if pulling dose, not needed if pushing)
    // GCS FIX: This need only be computed once per beam, not once per energy
    std::vector <int> lateral_step_0 (sigma_dim[2],0);
    std::vector <int> lateral_step_1 (sigma_dim[2],0);
    for (int k = 0; k < sigma_dim[2]; k++) {
        float sigma_max = 0.f;
        for (int i = 0; i < sigma_dim[0]*sigma_dim[1]; i++) {
            plm_long idx = k*sigma_dim[0]*sigma_dim[1] + i;
            if (sigma_img[idx] > sigma_max) {
                sigma_max = sigma_img[idx];
            }
        }
        lateral_step_0[k] = ceil (2.5 * sigma_max / lateral_spacing_0[k]);
        lateral_step_1[k] = ceil (2.5 * sigma_max / lateral_spacing_1[k]);
    }

    // Create central axis dose volume
    Rpl_volume *cax_dose_rv = new Rpl_volume;
    if (!cax_dose_rv) return;
    cax_dose_rv->clone_geometry (wepl_rv);
    cax_dose_rv->set_ray_trace_start (RAY_TRACE_START_AT_CLIPPING_PLANE);
    cax_dose_rv->set_aperture (beam->get_aperture());
    cax_dose_rv->set_ct (wepl_rv->get_ct());
    cax_dose_rv->set_ct_limit (wepl_rv->get_ct_limit());
    cax_dose_rv->compute_ray_data();
    cax_dose_rv->set_front_clipping_plane (wepl_rv->get_front_clipping_plane());
    cax_dose_rv->set_back_clipping_plane (wepl_rv->get_back_clipping_plane());
    cax_dose_rv->compute_rpl_void ();
    Volume *cax_dose_vol = cax_dose_rv->get_vol ();
    float *cax_dose_img = cax_dose_vol->get_raw<float> ();
    
    // Compute central axis dose
    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }
    const plm_long *dim = wepl_rv->get_image_dim();
    plm_long num_steps = wepl_rv->get_num_steps();
    plm_long ij[2] = {0,0};
    for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
        for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
            if (ap_img && ap_img[ap_vol->index(ij[0],ij[1],0)] == 0) {
                continue;
            }
            size_t np_index = energy_index * dim[0] * dim[1]
                + ij[1] * dim[0] + ij[0];
            float np = num_part[np_index];
            if (np == 0.f) {
                continue;
            }
            for (int s = 0; s < num_steps; s++) {
                int dose_index = ap_vol->index(ij[0],ij[1],s);
                float wepl = wepl_img[dose_index];
                cax_dose_img[dose_index] += np * depth_dose->lookup_energy(wepl);
            }
        }
    }

    /* Save sigma volume */
    if (beam->get_sigma_out() != "") {
        std::string fn;
        fn = string_format ("%s/cax-%02d",
            beam->get_sigma_out().c_str(), energy_index);
        cax_dose_rv->save (fn);
        fn = string_format ("%s/sig-%02d", 
            beam->get_sigma_out().c_str(), energy_index);
        sigma_rv->save (fn);
    }
    
    // Smear dose by specified sigma
    for (int s = 0; s < num_steps; s++) {
        double pixel_spacing[2] = {
            lateral_spacing_0[s],
            lateral_spacing_1[s]
        };
        for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
            for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
                plm_long idx = s*sigma_dim[0]*sigma_dim[1] + ij[1]*dim[0] + ij[0];
                float cax_dose = cax_dose_img[idx];
                if (cax_dose == 0.f) {
                    continue;
                }
                double sigma = (double) sigma_img[idx];
                double sigma_x3 = sigma * 2.5;
                
                // finding the rpl_volume pixels that are contained in the
                // the 3 sigma range
                plm_long ij_min[2], ij_max[2];
                ij_min[0] = ij[0] - ceil (sigma_x3 / lateral_spacing_0[s]);
                if (ij_min[0] < 0) ij_min[0] = 0;
                ij_min[1] = ij[1] - ceil (sigma_x3 / lateral_spacing_1[s]);
                if (ij_min[1] < 0) ij_min[1] = 0;
                ij_max[0] = ij[0] + ceil (sigma_x3 / lateral_spacing_0[s]);
                if (ij_max[0] > dim[0]-1) ij_max[0] = dim[0]-1;
                ij_max[1] = ij[1] + ceil (sigma_x3 / lateral_spacing_1[s]);
                if (ij_max[1] > dim[1]-1) ij_max[1] = dim[1]-1;

                float tot_off_axis = 0.f;
                plm_long ij1[2];
                for (ij1[1] = ij_min[1]; ij1[1] <= ij_max[1]; ij1[1]++) {
                    for (ij1[0] = ij_min[0]; ij1[0] <= ij_max[0]; ij1[0]++) {
                        plm_long idxs = s*sigma_dim[0]*sigma_dim[1]
                            + ij1[1]*dim[0] + ij1[0];
                        double gaussian_center[2] = { 0., 0. };
                        double pixel_center[2] = {
                            (double) ij1[0]-ij[0],
                            (double) ij1[1]-ij[1]
                        };
                        double off_axis_factor;
                        if (sigma == 0)
                        {
                            off_axis_factor = 1;
                        }
                        else
                        {
                            off_axis_factor = double_gaussian_interpolation (
                                gaussian_center, pixel_center,
                                sigma, pixel_spacing);
                        }

                        dose_rv_img[idxs] += cax_dose * off_axis_factor;
#if defined (commentout)
                        // GCS FIX: The below correction would give the
                        // option for dose to tissue
                        / ct_density / STPR;
#endif
                        tot_off_axis += off_axis_factor;
                    }
                }
            }
        }
    }

    // Free temporary memory
    delete cax_dose_rv;
    beam->get_rt_dose_timing()->timer_dose_calc.stop ();
}
