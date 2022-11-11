/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_warp_h_
#define _plm_warp_h_

#include "plmbase_config.h"
#include "itkBSplineDeformableTransform.h"
#include "xform.h"

class Plm_image;

/*! \brief 
 * The plm_warp function creates a new image from an input image 
 * and a transform.  It also, optionally, creates a vector field.
 */
PLMBASE_API void
plm_warp (
    /*! Output: Output image (optional) */
    Plm_image::Pointer& im_warped,
    /*! Output: Output vf (optional) */
    DeformationFieldType::Pointer *vf,
    /*! Input:  Input image warped by this xform */
    const Xform::Pointer& xf_in,
    /*! Input:  Size of output image */
    Plm_image_header *pih,
    /*! Input:  Input image */
    const Plm_image::Pointer& im_in,
    /*! Input:  Value for pixels without match */
    float default_val,
    /*! Input:  Force resample of image for linear transforms */
    bool resample_linear_xf,
    /*! Input:  Force use of itk (1) or not (0) */
    bool use_itk,
    /*! Input:  Trilinear (1) or nn (0) */
    bool interp_lin
);

#endif
