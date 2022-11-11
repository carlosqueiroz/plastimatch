/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "interpolate.h"
#include "proj_volume.h"
#include "ray_data.h"
#include "rpl_volume.h"
#include "rpl_volume_lut.h"
#include "volume.h"

class Lut_entry
{
public:
    Lut_entry() {
        for (int i = 0; i < 8; i++) {
            idx[i] = -i;
            weight[i] = 0.f;
        }
    }
public:
    plm_long idx[8];
    float weight[8];
};

class PLMBASE_API Rpl_volume_lut_private
{
public:
    Rpl_volume_lut_private (Rpl_volume *rv, Volume *vol)
        : rv(rv), vol(vol), lut(0)
    {
    }
    ~Rpl_volume_lut_private ()
    {
        delete[] lut;
    }
public:
    Rpl_volume *rv;
    Volume *vol;
    Lut_entry *lut;
};

Rpl_volume_lut::Rpl_volume_lut ()
{
    d_ptr = new Rpl_volume_lut_private (0, 0);
}

Rpl_volume_lut::Rpl_volume_lut (Rpl_volume *rv, Volume *vol)
{
    d_ptr = new Rpl_volume_lut_private (rv, vol);
}

Rpl_volume_lut::~Rpl_volume_lut ()
{
    delete d_ptr;
}

void
Rpl_volume_lut::set_lut_entry (
    const Ray_data* ray_data,
    plm_long vox_idx,
    const float *vox_ray,
    plm_long ap_idx,
    float li_frac,
    float step_length,
    int lut_entry_idx
)
{
    // Make sure this ray has positive weight
    if (li_frac <= 0.f) {
        return;
    }

    // Project voxel vector onto unit vector of aperture ray
    // This assumes that
    // d_ptr->rvrts == RAY_TRACE_START_AT_RAY_VOLUME_INTERSECTION
    // We omit the check for speed.
    const double *ap_ray = ray_data[ap_idx].ray;
    float dist = vec3_dot (vox_ray, ap_ray);
    dist -= ray_data->front_dist;
    if (dist < 0) {
        return;
    }

    // Compute number of steps
    plm_long steps_f = (plm_long) floorf (dist / step_length);
    float dist_frac = (dist - steps_f * step_length) / step_length;
    if (steps_f >= d_ptr->rv->get_num_steps()) {
        return;
    }

    // Compute lut entries
    const Aperture::Pointer ap = d_ptr->rv->get_aperture ();
    plm_long lut_idx = ap_idx + steps_f * ap->get_dim(0) * ap->get_dim(1);
    d_ptr->lut[lut_idx].idx[lut_entry_idx] = lut_idx;
    d_ptr->lut[lut_idx].weight[lut_entry_idx] = dist_frac * li_frac;

    if (steps_f >= d_ptr->rv->get_num_steps() - 1) {
        return;
    }
    lut_idx = lut_idx + ap->get_dim(0) * ap->get_dim(1);
    d_ptr->lut[lut_idx].idx[4+lut_entry_idx] = lut_idx;
    d_ptr->lut[lut_idx].weight[4+lut_entry_idx] = (1. - dist_frac) * li_frac;
}

void
Rpl_volume_lut::build_lut ()
{
    const Proj_volume *pv = d_ptr->rv->get_proj_volume ();
    const double *src = pv->get_src ();
    const Aperture::Pointer ap = d_ptr->rv->get_aperture ();
    const plm_long *ap_dim = ap->get_dim ();
    const Ray_data* ray_data = d_ptr->rv->get_ray_data();

    /* Allocate memory for lut */
    d_ptr->lut = new Lut_entry[d_ptr->vol->npix];

    plm_long ijk[3];
    double xyz[3];
    LOOP_Z (ijk, xyz, d_ptr->vol) {
        LOOP_Y (ijk, xyz, d_ptr->vol) {
            LOOP_X (ijk, xyz, d_ptr->vol) {
                plm_long idx = d_ptr->vol->index (ijk);

                /* Project the voxel to the aperture plane */
                double ap_xy[2];
                pv->project (ap_xy, xyz);
                if (!is_number (ap_xy[0]) || !is_number (ap_xy[1])) {
                    continue;
                }

                /* Check if voxel is completely outside aperture boundary */
                if (ap_xy[0] <= -1.f || ap_xy[0] >= ap_dim[0]
                    || ap_xy[1] <= -1.f || ap_xy[1] >= ap_dim[1])
                {
                    continue;
                }

                /* Get vector from source to voxel */
                float vox_ray[3];
                vec3_sub3 (vox_ray, xyz, src);

                /* Solve for interpolation fractions on aperture planes */
                plm_long ijk_f[3];
                float li_frac_1[3], li_frac_2[3];
                float ap_xy_float[2] = { static_cast<float>(ap_xy[0]), static_cast<float>(ap_xy[1]) };
                li_2d (ijk_f, li_frac_1, li_frac_2, ap_xy_float, ap_dim);

                /* Inspect four interpolant aperture pixels.
                   For each pixel, calculate distance to point
                   on ray closest to voxel center */
                plm_long ap_ij[2], ap_idx;
                ap_ij[0] = ijk_f[0], ap_ij[1] = ijk_f[1];
                ap_idx = ap_ij[0] + ap_ij[1] * ap_dim[0];

                set_lut_entry (ray_data, idx, vox_ray,
                    ap_idx, li_frac_1[0], li_frac_2[0], 0);

            }
        }
    }
}
