/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>
#include "itkImageRegionIterator.h"
#include "itkHistogramMatchingImageFilter.h"

#include "itk_histogram_matching.h"
#include "itk_image_clone.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "pwlut.h"

FloatImageType::Pointer
itk_histogram_matching (
    const FloatImageType::Pointer source_image,
    const FloatImageType::Pointer reference_image,
    const bool threshold,
    const int levels,
    const int match_points)
{
    typedef itk::HistogramMatchingImageFilter<FloatImageType, FloatImageType> MatchingFilterType;

    // Initialize filter settings
    MatchingFilterType::Pointer matcher = MatchingFilterType::New();
    if (threshold)
        matcher->ThresholdAtMeanIntensityOn();
    else
        matcher->ThresholdAtMeanIntensityOff();
    matcher->SetNumberOfHistogramLevels(levels);
    matcher->SetNumberOfMatchPoints(match_points);

    matcher->SetSourceImage(source_image);
    matcher->SetReferenceImage(reference_image);

    matcher->Update();

    return matcher->GetOutput();
}
