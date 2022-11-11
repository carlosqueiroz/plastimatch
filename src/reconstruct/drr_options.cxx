/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmreconstruct_config.h"
#include "drr_options.h"

Drr_options::Drr_options ()
{
    this->threading = THREADING_CPU_OPENMP;
    this->detector_resolution[0] = 128;
    this->detector_resolution[1] = 128;
    this->image_size[0] = 600;
    this->image_size[1] = 600;
    this->have_image_center = 0;
    this->have_image_window = 0;
    this->isocenter[0] = 0.0f;
    this->isocenter[1] = 0.0f;
    this->isocenter[2] = 0.0f;

    this->start_angle = 0.f;
    this->num_angles = 1;
    this->have_angle_diff = 0;
    this->angle_diff = 1.0f;

    this->have_nrm = 0;
    this->nrm[0] = 1.0f;
    this->nrm[1] = 0.0f;
    this->nrm[2] = 0.0f;
    this->vup[0] = 0.0f;
    this->vup[1] = 0.0f;
    this->vup[2] = 1.0f;

    this->sad = 1000.0f;
    this->sid = 1630.0f;

    this->exponential_mapping = 0;
    this->output_format= OUTPUT_FORMAT_PFM;
    this->hu_conversion = PREPROCESS_CONVERSION;
    this->algorithm = DRR_ALGORITHM_EXACT;
    this->geometry_only = 0;
    this->input_file = "";
    this->output_file = "";
    this->output_prefix = "out_";

    this->autoscale = false;
    this->autoscale_range[0] = 0.f;
    this->autoscale_range[1] = 255.f;
    this->manual_scale = 1.0f;

    this->output_details_prefix = "";
    this->output_details_fn = "";
}
