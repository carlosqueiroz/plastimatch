/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _image_boundary_h_
#define _image_boundary_h_

#include "plmutil_config.h"
#include "itk_image.h"
#include "volume_boundary_behavior.h"
#include "volume_boundary_type.h"

class Plm_image;
class Image_boundary_private;

/*! \brief 
 * The Image_boundary class takes an input image (binary) and computes
 * an output image.  Voxels of the output image will be one 
 * if they are (1) non-zero in the input image, and (2) have a zero 
 * voxel in their six-neighborhood.  Other output image voxels will have 
 * value zero.
 */
class PLMUTIL_API Image_boundary {
public:
    Image_boundary ();
    ~Image_boundary ();
public:
    Image_boundary_private *d_ptr;

public:

    /*! \name Inputs */
    ///@{
    /*! \brief Set the input image.  The image will be loaded
      from the specified filename. */
    void set_input_image (const char* image_fn);
    /*! \brief Set the input image as an ITK image. */
    void set_input_image (const UCharImageType::Pointer image);
    /*! \brief Set the volume boundary type, either 
      INTERIOR_EDGE or INTERIOR_FACE */
    void set_volume_boundary_type (Volume_boundary_type vbt);
    /*! \brief Set the volume boundary behavior, either 
      ZERO_PADDING, EDGE_PADDING, or ADAPTIVE_PADDING */
    void set_volume_boundary_behavior (Volume_boundary_behavior vbb);
    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute image boundary */
    void run ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Return the boundary image as an ITK image.  */
    UCharImageType::Pointer get_output_image ();
    ///@}
};

PLMUTIL_API
UCharImageType::Pointer do_image_boundary (UCharImageType::Pointer image);

#endif
