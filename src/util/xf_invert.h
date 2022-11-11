/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xf_invert_h_
#define _xf_invert_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

class Xf_invert_private;
class Xform;

class PLMUTIL_API Xf_invert {
public:
    Xf_invert ();
    ~Xf_invert ();
public:
    Xf_invert_private *d_ptr;
public:

    /*! \name Inputs */
    ///@{
    /*! \brief Set the transform to be inverted.  
      The transform will be loaded from the specified filename. */
    void set_input_xf (const char* xf_fn);
    /*! \brief Set the input vector field to be inverted
      as an ITK image. */
    void set_input_vf (const DeformationFieldType::Pointer vf);
    /*! \brief Set the geometry of the output vector field to match 
      another image.
      The image will be loaded from the specified filename. */
    void set_fixed_image (const char* image_fn);
    /*! \brief Set the geometry of the output vector field to match 
      another image, specified as an ITK image. */
    void set_fixed_image (const FloatImageType::Pointer image);
    /*! \brief Set the image dimension (number of voxels) 
      of the output vector field. */
    void set_dim (const plm_long dim[3]);
    /*! \brief Set the image origin 
      of the output vector field. */
    void set_origin (const float origin[3]);
    /*! \brief Set the image spacing 
      of the output vector field. */
    void set_spacing (const float spacing[3]);
    /*! \brief Set the image direction cosines 
      of the output vector field. */
    void set_direction_cosines (const float direction_cosines[9]);
    /*! \brief Set the number of iterations to run the inversion 
      routine (default is 20 iterations). */
    void set_iterations (int iterations);
    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute inverse vector field */
    void run ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Return the inverse xform */
    const Xform* get_output ();
    ///@}

protected:
    void run_invert_itk ();
    void run_invert_vf ();
};

#endif
