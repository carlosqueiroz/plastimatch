/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "itk_image.h"
#include "itk_bbox.h"
#include "itk_crop.h"
#include "zcrop.h"

void
zcrop (
    UCharImageType::Pointer& ref,
    UCharImageType::Pointer& cmp,
    float zcrop[2])
{
    float bbox_coordinates[6];
    int bbox_indices[6];
    itk_bbox(ref, bbox_coordinates, bbox_indices);
    bbox_coordinates[4] += zcrop[1];
    bbox_coordinates[5] -= zcrop[0];
    ref = itk_crop_by_coord(ref, bbox_coordinates);
    cmp = itk_crop_by_coord(cmp, bbox_coordinates);
}
