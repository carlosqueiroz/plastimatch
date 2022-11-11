/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "bspline_header.h"
#include "plm_image_header.h"
#include "logfile.h"

Bspline_header::Bspline_header ()
{
    for (int d = 0; d < 3; d++) {
        this->img_origin[d] = 0.0f;
        this->img_spacing[d] = 1.0f;
        this->img_dim[d] = 0;
        this->roi_offset[d] = 0;
        this->roi_dim[d] = 0;
        this->vox_per_rgn[d] = 30;
        this->grid_spac[d] = 30.0f;
        this->rdims[d] = 0;
        this->cdims[d] = 0;
    }
    this->dc.set_identity ();
    this->num_knots = 0;
    this->num_coeff = 0;
}

void
Bspline_header::set (
    const float img_origin[3],
    const float img_spacing[3],
    const plm_long img_dim[3],
    const plm_long roi_offset[3],
    const plm_long roi_dim[3],
    const plm_long vox_per_rgn[3],
    const float direction_cosines[9]
)
{
    this->dc.set (direction_cosines);
    for (int d = 0; d < 3; d++) {
        /* copy input parameters over */
        this->img_origin[d] = img_origin[d];
        this->img_spacing[d] = img_spacing[d];
        this->img_dim[d] = img_dim[d];
        this->roi_offset[d] = roi_offset[d];
        this->roi_dim[d] = roi_dim[d];
        this->vox_per_rgn[d] = vox_per_rgn[d];

        /* grid spacing is in mm */
        this->grid_spac[d] = this->vox_per_rgn[d] * fabs (this->img_spacing[d]);

        /* rdims is the number of regions */
        this->rdims[d] = 1 + (this->roi_dim[d] - 1) / this->vox_per_rgn[d];

        /* cdims is the number of control points */
        this->cdims[d] = 3 + this->rdims[d];
    }

    /* total number of control points & coefficients */
    this->num_knots = this->cdims[0] * this->cdims[1] * this->cdims[2];
    this->num_coeff = this->cdims[0] * this->cdims[1] * this->cdims[2] * 3;
}

void
Bspline_header::set (
    const Plm_image_header *pih,
    const float grid_spac[3]
)
{
    float img_origin[3];
    float img_spacing[3];
    plm_long img_dim[3];
    plm_long roi_offset[3];
    plm_long roi_dim[3];
    plm_long vox_per_rgn[3];
    float direction_cosines[9];

    pih->get_origin (img_origin);
    pih->get_dim (img_dim);
    pih->get_spacing (img_spacing);
    pih->get_direction_cosines (direction_cosines);

    for (int d = 0; d < 3; d++) {
        /* Old ROI was whole image */
        roi_offset[d] = 0;
        roi_dim[d] = img_dim[d];
        /* Compute vox_per_rgn */
        vox_per_rgn[d] = ROUND_INT (grid_spac[d] / fabs(img_spacing[d]));
        if (vox_per_rgn[d] < 4) {
            lprintf ("Warning: vox_per_rgn was less than 4.\n");
            vox_per_rgn[d] = 4;
        }
    }

    this->set (img_origin, img_spacing, img_dim,
        roi_offset, roi_dim, vox_per_rgn, direction_cosines);
}

void
Bspline_header::set_unaligned (
    const float img_origin[3],
    const float img_spacing[3],
    const plm_long img_dim[3],
    const plm_long roi_offset[3],
    const plm_long roi_dim[3],
    const float grid_spac[3],
    const float direction_cosines[9]
)
{
    this->dc.set (direction_cosines);
    for (int d = 0; d < 3; d++) {
        /* copy input parameters over */
        this->img_origin[d] = img_origin[d];
        this->img_spacing[d] = img_spacing[d];
        this->img_dim[d] = img_dim[d];
        this->roi_offset[d] = roi_offset[d];
        this->roi_dim[d] = roi_dim[d];

        /* vox_per_rgn is unused for unaligned grids */
        this->vox_per_rgn[d] = 0;

        /* rdims is the number of regions */
        float img_ext = (img_dim[d] - 1) * fabs (img_spacing[d]);
        this->rdims[d] = 1 + (int) floor (img_ext / grid_spac[d]);

        /* cdims is the number of control points */
        this->cdims[d] = 3 + this->rdims[d];
        
    }

    /* total number of control points & coefficients */
    this->num_knots = this->cdims[0] * this->cdims[1] * this->cdims[2];
    this->num_coeff = this->cdims[0] * this->cdims[1] * this->cdims[2] * 3;
}

void
Bspline_header::set_unaligned (
    const Plm_image_header *pih,
    const float grid_spac[3]
)
{
    float img_origin[3];
    float img_spacing[3];
    plm_long img_dim[3];
    plm_long roi_offset[3];
    plm_long roi_dim[3];
    plm_long vox_per_rgn[3];
    float direction_cosines[9];

    pih->get_origin (img_origin);
    pih->get_dim (img_dim);
    pih->get_spacing (img_spacing);
    pih->get_direction_cosines (direction_cosines);

    for (int d = 0; d < 3; d++) {
        /* Old ROI was whole image */
        roi_offset[d] = 0;
        roi_dim[d] = img_dim[d];
    }
    
    this->set_unaligned (
        img_origin, img_spacing, img_dim, roi_offset, roi_dim,
        grid_spac, direction_cosines);
}
