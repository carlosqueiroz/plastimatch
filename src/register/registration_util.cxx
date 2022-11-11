/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "registration_util.h"
#include "shared_parms.h"

plm_long
count_fixed_voxels (
    Registration_data *regd,
    Stage_parms* stage, 
    FloatImageType::Pointer& fixed_ss)
{
    // Do simple calculation if there is no ROI
    const Shared_parms *shared = stage->get_shared_parms();
    if (!shared->fixed_roi_enable || !regd->get_fixed_roi()) {
        plm_long dim[3];
        get_image_header (dim, 0, 0, fixed_ss);
        return dim[0] * dim[1] * dim[2];
    }

    // Else, iterate through image to find voxels where ROI not zero
    Plm_image::Pointer& fixed_roi = regd->get_fixed_roi ();
    const UCharImageType::Pointer itk_fixed_roi = fixed_roi->itk_uchar ();
    typedef itk::ImageRegionConstIteratorWithIndex < FloatImageType 
        > IteratorType;
    FloatImageType::RegionType region = fixed_ss->GetLargestPossibleRegion();
    IteratorType it (fixed_ss, region);
    plm_long num_voxels = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        FloatImageType::PointType phys_loc;
        fixed_ss->TransformIndexToPhysicalPoint (it.GetIndex(), phys_loc);
        UCharImageType::IndexType roi_idx;
        bool is_inside = itk_fixed_roi->TransformPhysicalPointToIndex (
            phys_loc, roi_idx);
        if (is_inside && itk_fixed_roi->GetPixel (roi_idx)) {
            num_voxels ++;
        }
    }
    return num_voxels;
}

