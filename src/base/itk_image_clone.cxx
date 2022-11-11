/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "itkImage.h"
#include "itkImageDuplicator.h"

#include "itk_image_clone.h"
#include "itk_image_create.h"
#include "itk_image_type.h"
#include "plm_image_header.h"

template<class T> 
T
itk_image_clone (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageDuplicator < ImageType > DuplicatorType;

    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage (image);
    duplicator->Update();
    return duplicator->GetOutput();
}

template<class T> 
T
itk_image_clone_empty (T image)
{
    T img = T::ObjectType::New ();
    img->SetOrigin (image->GetOrigin());
    img->SetSpacing (image->GetSpacing());
    img->SetDirection (image->GetDirection());
    img->SetRegions (image->GetLargestPossibleRegion());
    img->Allocate ();
    img->FillBuffer (static_cast<typename T::ObjectType::PixelType>(0));

    return img;
}


/* Explicit instantiations */
template PLMBASE_API FloatImageType::Pointer itk_image_clone (FloatImageType::Pointer);

template PLMBASE_API UCharImageType::Pointer itk_image_clone_empty (UCharImageType::Pointer);
