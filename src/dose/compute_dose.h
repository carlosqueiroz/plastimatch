/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_dose_h_
#define _rt_dose_h_

#include "aperture.h"
#include "rt_depth_dose.h"
#include "plmdose_config.h"
#include "plm_image.h"

class Plan_calc;

double
energy_direct (
    float rgdepth,             /* voxel to dose */
    Beam_calc* beam,
    int beam_idx
);

void compute_dose_a (
    Volume::Pointer dose_vol, 
    Beam_calc* beam, 
    const Volume::Pointer ct_vol
);
void compute_dose_b (
    Beam_calc* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol
);
void compute_dose_ray_trace_dij_a (
    Beam_calc* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol,
    Volume::Pointer& dose_vol
);
void compute_dose_ray_trace_dij_b (
    Beam_calc* beam,
    const Volume::Pointer ct_vol,
    Volume::Pointer& dose_vol
);
void compute_dose_d (
    Beam_calc* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol
);
void compute_dose_ray_desplanques (
    Volume* dose_volume, 
    Volume::Pointer ct_vol, 
    Beam_calc* beam, 
    Volume::Pointer final_dose_volume, 
    int beam_index
);
void compute_dose_ray_sharp (
    const Volume::Pointer ct_vol,  
    Beam_calc* beam,
    Rpl_volume* rpl_dose_volume,
    int beam_index,
    const int* margins
);
void compute_dose_ray_shackleford (
    Volume::Pointer dose_volume, 
    Plan_calc* plan_calc,
    Beam_calc *beam,
    int beam_index, 
    std::vector<double>* area, 
    std::vector<double>* xy_grid, 
    int radius_sample, 
    int theta_sample
);

#endif
