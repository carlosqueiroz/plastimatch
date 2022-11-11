/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>
#include "itk_image.h"
#include "itk_image_header_compare.h"
#include "itk_resample.h"
#include "itkImageRegionIterator.h"
#include "itk_local_intensity_correction.h"
#include "itk_image_clone.h"
#include "itk_image_save.h"
#include "itkResampleImageFilter.h"
#include "itkTranslationTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itk_mask.h"
#include "itkMedianImageFilter.h"


UCharImageType::Pointer GetFullMask(FloatImageType::Pointer img) {
    UCharImageType::Pointer mask = UCharImageType::New();
    mask->SetRegions(img->GetLargestPossibleRegion());
    mask->SetOrigin(img->GetOrigin());
    mask->SetSpacing(img->GetSpacing());
    mask->SetDirection(img->GetDirection());
    mask->Allocate();
    mask->FillBuffer(1);

    return mask;
}

void GetIntensityCorrectionField (
    FloatImageType::Pointer& source_image,
    FloatImageType::Pointer& reference_image,
    SizeType patch_size,
    UCharImageType::Pointer& source_mask,
    UCharImageType::Pointer& reference_mask,
    FloatImageType::Pointer& shift_field,
    FloatImageType::Pointer& scale_field)
{
    typedef itk::ImageRegionIterator<FloatImageType> RegionIteratorType;
    typedef itk::ImageRegionIterator<UCharImageType> MaskRegionIteratorType;

    SizeType size = source_image->GetLargestPossibleRegion().GetSize();

    scale_field = FloatImageType::New();

    scale_field->SetDirection(source_image->GetDirection());
    IndexType index;
    index.Fill(0);
    SizeType field_size;
    SpacingType sp;
    itk::ContinuousIndex<float, 3> ci;
    for (int i = 0; i < 3; ++i) {
        ci[i] = (patch_size[i] - 1) / 2;
        field_size[i] = (size[i] + patch_size[i] - 1) / patch_size[i];
        sp[i] = source_image->GetSpacing()[i] * patch_size[i];
    }
    scale_field->SetSpacing(sp);
    FloatPoint3DType og;
    source_image->TransformContinuousIndexToPhysicalPoint (ci, og);
    scale_field->SetOrigin(og);
    RegionType reg;
    reg.SetIndex(index);
    reg.SetSize(field_size);
    scale_field->SetRegions(reg);
    scale_field->Allocate();
    scale_field->FillBuffer(1);

    shift_field = itk_image_clone(scale_field);
    shift_field->FillBuffer(0);

    RegionIteratorType it_scale(scale_field, scale_field->GetLargestPossibleRegion());
    RegionIteratorType it_shift(shift_field, shift_field->GetLargestPossibleRegion());

    for (it_scale.GoToBegin(), it_shift.GoToBegin();
         !it_scale.IsAtEnd(); ++it_scale, ++it_shift)
    {
        SizeType current_patch_size = patch_size;
        IndexType current_index = it_scale.GetIndex();
        for (int i = 0; i < 3; ++i) {
            current_index[i] = current_index[i] * patch_size[i];
            if (current_index[i] + current_patch_size[i] >= size[i])
                current_patch_size[i] = size[i] - current_index[i];
        }
        RegionType patch(current_index, current_patch_size);
        RegionIteratorType it_src(source_image, patch);
        RegionIteratorType it_ref(reference_image, patch);
        MaskRegionIteratorType it_src_mask(source_mask, patch);
        MaskRegionIteratorType it_ref_mask(reference_mask, patch);

        double avg_src = 0, avg_ref = 0, std_src = 0, std_ref = 0;
        int num_src = 0, num_ref = 0;
        for (it_src.GoToBegin(), it_ref.GoToBegin(),
                 it_src_mask.GoToBegin(), it_ref_mask.GoToBegin();
             !it_src.IsAtEnd();
             ++it_src, ++it_ref, ++it_src_mask, ++it_ref_mask)
        {
            if (it_src_mask.Get() > 0) {
                avg_src += double(it_src.Get());
                ++num_src;
            }
            if (it_ref_mask.Get() > 0) {
                avg_ref += double(it_ref.Get());
                ++num_ref;
            }
        }
        if (num_src > 0) avg_src /= num_src;  // if not, avg = 0
        if (num_ref > 0) avg_ref /= num_ref;

        for (it_src.GoToBegin(), it_ref.GoToBegin(),
                 it_src_mask.GoToBegin(), it_ref_mask.GoToBegin();
             !it_src.IsAtEnd();
             ++it_src, ++it_ref, ++it_src_mask, ++it_ref_mask)
        {
            if (it_src_mask.Get() > 0) {
                double d1 = it_src.Get() - avg_src;
                std_src += (double) sqrt(d1 * d1);
            }
            if (it_ref_mask.Get() > 0) {
                double d2 = it_ref.Get() - avg_ref;
                std_ref += (double) sqrt(d2 * d2);
            }
        }
        if (num_src > 0)
            std_src /= num_src;
        else
            std_src = 0;
        if (num_ref > 0)
            std_ref /= num_ref;
        else
            std_src = 0;
        float scale = (float) (std_ref / std_src);
        if (std_ref == 0 || std_src == 0) { scale = 1; }

        float shift = (float) (avg_ref - avg_src * scale);

        it_scale.Set(scale);
        it_shift.Set(shift);
    }
}

FloatImageType::Pointer BlendField(
    FloatImageType::Pointer field,
    FloatImageType::Pointer source,
    bool trilinear)
{
    typedef itk::TranslationTransform<double, 3> TranslationTransformType;
    typedef itk::ResampleImageFilter<FloatImageType, FloatImageType> ResampleImageFilterType;
    typedef itk::NearestNeighborInterpolateImageFunction<FloatImageType, double> NearestNeighborInterpolatorType;
    typedef itk::LinearInterpolateImageFunction<FloatImageType, double> LinearInterpolatorType;

    TranslationTransformType::Pointer transform = TranslationTransformType::New();
    TranslationTransformType::OutputVectorType translation;
    for (int i = 0; i < 3; ++i) {
        //translation[i] = -0.5 * field->GetSpacing()[i];
        translation[i] = 0;
    }
    transform->Translate(translation);

    ResampleImageFilterType::Pointer filter = ResampleImageFilterType::New();
//    filter->SetTransform(transform.GetPointer());
    filter->SetInput(field);
    filter->SetReferenceImage(source);
    filter->UseReferenceImageOn();
    if (trilinear)
        filter->SetInterpolator(LinearInterpolatorType::New());
    else
        filter->SetInterpolator(NearestNeighborInterpolatorType::New());

    filter->UpdateLargestPossibleRegion();
    return filter->GetOutput();
}

void BlendIntensityCorrectionField(
    FloatImageType::Pointer& shift_field,
    FloatImageType::Pointer& scale_field,
    const FloatImageType::Pointer& source,
    const UCharImageType::Pointer& mask,
    bool trilinear)
{
    shift_field = BlendField(shift_field, source, trilinear);
    scale_field = BlendField(scale_field, source, trilinear);

    shift_field = mask_image(shift_field, mask, MASK_OPERATION_MASK, 0);
    scale_field = mask_image(scale_field, mask, MASK_OPERATION_MASK, 1);
}

void ApplyIntensityCorrectionField(
    FloatImageType::Pointer& img,
    const FloatImageType::Pointer& shift,
    const FloatImageType::Pointer& scale)
{
    typedef itk::ImageRegionIterator<FloatImageType> RegionIteratorType;
    RegionType region = img->GetLargestPossibleRegion();
    RegionIteratorType it_img (img, region);
    RegionIteratorType it_shift (shift, region);
    RegionIteratorType it_scale (scale, region);

    for (it_img.GoToBegin(), it_shift.GoToBegin(), it_scale.GoToBegin();
         !it_img.IsAtEnd(); ++it_img, ++it_shift, ++it_scale) {
        float val = it_img.Get();
        it_img.Set(val * it_scale.Get() + it_shift.Get());
    }
}

FloatImageType::Pointer ApplyMedianFilter (
    FloatImageType::Pointer img, SizeType mediansize) {
    typedef itk::MedianImageFilter<FloatImageType, FloatImageType> MedianFilterType;
    MedianFilterType::RadiusType radius = mediansize;
    MedianFilterType::Pointer filter = MedianFilterType::New();
    filter->SetRadius(radius);
    filter->SetInput(img);
    filter->Update();

    return filter->GetOutput();
}

FloatImageType::Pointer
itk_local_intensity_correction (
    FloatImageType::Pointer& source_image,
    FloatImageType::Pointer& reference_image,
    SizeType patch_size, bool blend, SizeType mediansize)
{
    FloatImageType::Pointer shift_field, scale_field;
    return itk_local_intensity_correction(source_image, reference_image,
            patch_size, shift_field, scale_field, blend, mediansize);
}

FloatImageType::Pointer
itk_local_intensity_correction (
    FloatImageType::Pointer& source_image,
    FloatImageType::Pointer& reference_image,
    SizeType patch_size,
    FloatImageType::Pointer& shift_field,
    FloatImageType::Pointer& scale_field,
    bool blend,
    SizeType mediansize)
{
    UCharImageType::Pointer source_mask = GetFullMask(source_image);

    /* reference image will be resampled to source geometry, we can use source
       geometry here so that reference_mask will not be resampled */
    UCharImageType::Pointer reference_mask = GetFullMask(source_image);

    return itk_masked_local_intensity_correction (
        source_image, reference_image,
        patch_size, source_mask, reference_mask,
        shift_field, scale_field, blend, mediansize);
}

FloatImageType::Pointer
itk_masked_local_intensity_correction(
    FloatImageType::Pointer& source_image,
    FloatImageType::Pointer& reference_image,
    SizeType patch_size,
    UCharImageType::Pointer& source_mask,
    UCharImageType::Pointer& reference_mask,
    FloatImageType::Pointer& shift_field,
    FloatImageType::Pointer& scale_field,
    bool blend,
    SizeType mediansize)
{
    typedef itk::ImageRegionIterator<FloatImageType> RegionIteratorType;
    typedef itk::ImageRegionIterator<UCharImageType> MaskRegionIteratorType;

    if (!itk_image_header_compare (source_image, reference_image)) {
        reference_image = itk_resample_image (
            reference_image, source_image, 0, 0);
    }
    if (!itk_image_header_compare (source_image, source_mask)) {
        source_mask = itk_resample_image (source_mask, source_image, 0, 0);
    }
    if (!itk_image_header_compare (source_image, reference_mask)) {
        reference_mask = itk_resample_image(reference_mask, source_image, 0, 0);
    }

    SizeType size = source_image->GetLargestPossibleRegion().GetSize();
    FloatImageType::Pointer out_image = itk_image_clone(source_image);

    GetIntensityCorrectionField (
        source_image, reference_image, patch_size, source_mask, reference_mask,
        shift_field, scale_field);

    if (mediansize[0] > 0) {
        shift_field = ApplyMedianFilter(shift_field, mediansize);
        scale_field = ApplyMedianFilter(scale_field, mediansize);
    }

    BlendIntensityCorrectionField (shift_field, scale_field,
        source_image, source_mask, blend);

    ApplyIntensityCorrectionField(out_image, shift_field, scale_field);

    return out_image;
}
