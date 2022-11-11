/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_histogram_matching_h_
#define _itk_histogram_matching_h_

#include "plmutil_config.h"
#include <list>
#include <utility>
#include "itk_image_type.h"

PLMUTIL_API FloatImageType::Pointer itk_histogram_matching (const FloatImageType::Pointer source_image,
        const FloatImageType::Pointer reference_image, const bool threshold, const int levels, const int match_points);

#endif
