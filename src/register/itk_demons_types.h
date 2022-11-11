/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef itk_demons_types_h
#define itk_demons_types_h

#include "plmregister_config.h"
#include "itk_image_type.h"

#if PLM_USE_NEW_ITK_DEMONS
#include "itkLogDomainDemonsRegistrationWithMaskFilter.h"
#include "itkSymmetricLogDomainDemonsRegistrationWithMaskFilter.h"
#else
#include "itkLogDomainDemonsRegistrationFilterWithMaskExtension.h"
#include "itkSymmetricLogDomainDemonsRegistrationFilterWithMaskExtension.h"
#endif
#include "itkPDEDeformableRegistrationWithMaskFilter.h"

typedef itk::PDEDeformableRegistrationWithMaskFilter<
    FloatImageType,
    FloatImageType,DeformationFieldType>  PDEDeformableRegistrationFilterType;
#if PLM_USE_NEW_ITK_DEMONS
typedef itk::LogDomainDemonsRegistrationWithMaskFilter<
    FloatImageType,
    FloatImageType,
    DeformationFieldType> LogDomainDemonsFilterType;
typedef itk::SymmetricLogDomainDemonsRegistrationWithMaskFilter<
    FloatImageType,
    FloatImageType,
    DeformationFieldType> SymmetricLogDomainDemonsFilterType;
#else
typedef itk::LogDomainDemonsRegistrationFilterWithMaskExtension<
    FloatImageType,
    FloatImageType,
    DeformationFieldType> LogDomainDemonsFilterType;
typedef itk::SymmetricLogDomainDemonsRegistrationFilterWithMaskExtension<
    FloatImageType,
    FloatImageType,
    DeformationFieldType> SymmetricLogDomainDemonsFilterType;
#endif

#endif
