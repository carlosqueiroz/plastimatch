/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_header_h_
#define _bspline_header_h_

#include "plmbase_config.h"
#include "direction_cosines.h"
#include "plm_int.h"

//TODO: Change type of dc to Direction_cosines*

//class Direction_cosines;
class Plm_image_header;
class Volume;
class Volume_header;

/*! \brief 
 * The Bspline_header class encapsulates the B-spline grid geometry
 */
class PLMBASE_API Bspline_header {
public:
    Bspline_header ();
public:
    float img_origin[3];        /* Image origin (in mm) */
    float img_spacing[3];       /* Image spacing (in mm) */
    plm_long img_dim[3];        /* Image size (in vox) */
    Direction_cosines dc;       /* Image direction cosines */
    plm_long roi_offset[3];     /* Position of first vox in ROI (in vox) */
    plm_long roi_dim[3];        /* Dimension of ROI (in vox) */
    plm_long vox_per_rgn[3];	/* Knot spacing (in vox) */
    float grid_spac[3];         /* Knot spacing (in mm) */
    plm_long rdims[3];          /* # of regions in (x,y,z) */
    plm_long cdims[3];          /* # of knots in (x,y,z) */
    /** Total number of knots (= product(cdims)) */
    int num_knots;
    /** Total number of coefficents (= product(cdims) * 3) */
    int num_coeff;

public:
    /** Set the internal geometry.  
        This version of the function gets used when loading a B-Spline 
        from file.
        \param img_origin      Image origin (in mm)
        \param img_spacing     Image spacing (in mm) 
        \param img_dim         Image size (in vox) 
        \param roi_offset      Position of first vox in ROI (in vox) 
        \param roi_dim	       Dimension of ROI (in vox) 
        \param vox_per_rgn     Knot spacing (in vox) 
        \param direction_cosines Direction cosines 
    */
    void set (
        const float img_origin[3],
        const float img_spacing[3],
        const plm_long img_dim[3],
        const plm_long roi_offset[3],
        const plm_long roi_dim[3],
        const plm_long vox_per_rgn[3],
        const float direction_cosines[9]
    );

    /** Set the internal geometry.  
        This version of the function gets used to calculate the 
        geometry of an ALIGNED B-Spline with a specified grid spacing. 
        \param pih The image geometry associated with B-spline
        \param grid_spac The B-Spline grid spacing (in mm)
    */
    void set (
        const Plm_image_header *pih,
        const float grid_spac[3]
    );

    /** Set the internal geometry.  
        This version of the function gets used to calculate the 
        geometry of an UNALIGNED B-Spline with a specified grid spacing. 
        \param pih The image geometry associated with B-spline
        \param grid_spac The B-Spline grid spacing (in mm)
    */
    void set_unaligned (
        const float img_origin[3],
        const float img_spacing[3],
        const plm_long img_dim[3],
        const plm_long roi_offset[3],
        const plm_long roi_dim[3],
        const float grid_spac[3],
        const float direction_cosines[9]
    );

    /** Set the internal geometry.  
        This version of the function gets used to calculate the 
        geometry of an UNALIGNED B-Spline with a specified grid spacing. 
        \param pih The image geometry associated with B-spline
        \param grid_spac The B-Spline grid spacing (in mm)
    */
    void set_unaligned (
        const Plm_image_header *pih,
        const float grid_spac[3]
    );
    void get_volume_header (Volume_header *vh);
};

#endif
