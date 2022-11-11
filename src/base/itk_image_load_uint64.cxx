/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "itk_image_load.txx"

UInt64ImageType::Pointer
itk_image_load_uint64 (const char* fname, Plm_image_type* original_type)
{
    UInt64ImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
#if defined (commentout)
	img = load_dicom_uint64 (fname);
#endif
        print_and_exit ("Error: attempt to load DICOM using ITK reader.\n");
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<uint64_t>(0));
    }
    return itk_image_load_postprocess (img);
}
