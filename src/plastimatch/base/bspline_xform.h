/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_xform_h_
#define _bspline_xform_h_

#include "plmbase_config.h"
#include "bspline_header.h"
#include "direction_cosines.h"
#include "smart_pointer.h"
#include "plm_int.h"

class Volume;
class Volume_header;

/*! \brief 
 * The Bspline_xform class encapsulates the B-spline coefficients 
 * used by native registration and warping algorithms.  Information 
 * describing the B-spline 
 * geometry is held in the base class: Bspline_header.
 */
class PLMBASE_API Bspline_xform
    : public Bspline_header
{
public:
    SMART_POINTER_SUPPORT (Bspline_xform);
public:
    Bspline_xform ();
    ~Bspline_xform ();
public:
    enum Lut_type {
        LUT_3D_ALIGNED,
        LUT_1D_ALIGNED,
        LUT_1D_UNALIGNED
    };
public:
    /** Array of B-spline coefficients. */
    float* coeff;

    /** Choose which kind of LUT is used */
    Lut_type lut_type;

    /* Aligned grid (3D) LUTs */
    plm_long* cidx_lut;          /* Lookup volume for region number */
    plm_long* c_lut;             /* Lookup table for control point indices */
    plm_long* qidx_lut;          /* Lookup volume for region offset */
    float* q_lut;                /* Lookup table for influence multipliers */

    /* Aligned grid (1D) LUTs */
    float *bx_lut;               /* LUT for influence multiplier in x dir */
    float *by_lut;               /* LUT for influence multiplier in y dir */
    float *bz_lut;               /* LUT for influence multiplier in z dir */

    /* Unaligned grid (1D) LUTs */
    float *ux_lut;               /* LUT for influence multiplier in x dir */
    float *uy_lut;               /* LUT for influence multiplier in y dir */
    float *uz_lut;               /* LUT for influence multiplier in z dir */

public:
    /** Initialize B-spline geometry and allocate memory for coefficients.
        This version of the function gets used when loading a B-Spline 
        from file. */
    void initialize (
        float img_origin[3],          /* Image origin (in mm) */
        float img_spacing[3],         /* Image spacing (in mm) */
        plm_long img_dim[3],          /* Image size (in vox) */
        plm_long roi_offset[3],       /* Position of first vox in ROI (in vox) */
        plm_long roi_dim[3],	      /* Dimension of ROI (in vox) */
        plm_long vox_per_rgn[3],      /* Knot spacing (in vox) */
        float direction_cosines[9]    /* Direction cosines */
    );
    /** Initialize B-spline geometry and allocate memory for coefficients.
        This version of the function gets used when creating a B-Spline 
        with a specified grid spacing. 
        \param pih The image geometry associated with B-spline
        \param grid_spac The B-Spline grid spacing (in mm)
    */
    void initialize (
        const Plm_image_header *pih,
        const float grid_spac[3]
    );
    void save (const char* filename);
    void fill_coefficients (float val);
    /*! \brief This function jitters the coefficients if they are all zero. 
     *  It is used to prevent local minima artifact when optimizing an MI cost 
     *  function for images with the same geometry.
     */
    void jitter_if_zero ();
    void get_volume_header (Volume_header *vh);
    Plm_image_header get_plm_image_header ();
    void log_header ();
protected:
    /** Allocate and initialize coefficients and LUTs
     */
    void allocate ();
};

PLMBASE_C_API Bspline_xform* bspline_xform_load (const char* filename);

/* Debugging routines */
PLMBASE_C_API void bspline_xform_dump_coeff (Bspline_xform* bxf, const char* fn);
PLMBASE_C_API void bspline_xform_dump_luts (Bspline_xform* bxf);


#endif
