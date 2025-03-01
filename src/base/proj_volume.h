/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_volume_h_
#define _proj_volume_h_

#include "plmbase_config.h"
#include <string>
#include "plm_int.h"

class Proj_matrix;
class Proj_volume_private;
class Volume;

/*! \brief 
 * The Proj_volume class represents a three-dimensional volume 
 * on a uniform non-orthogonal grid.  The grid is regular within 
 * a rectangular frustum, the geometry of which is specified by 
 * a projection matrix.
 */
class PLMBASE_API Proj_volume 
{
public:
    Proj_volume ();
    ~Proj_volume ();
public:
    Proj_volume_private *d_ptr;
public:
    void set_geometry (
        const double src[3],           // position of source (mm)
        const double iso[3],           // position of isocenter (mm)
        const double vup[3],           // dir to "top" of projection plane
        double sid,                    // dist from proj plane to source (mm)
        const plm_long image_dim[2],   // resolution of image
        const double image_center[2],  // image center (pixels)
        const double image_spacing[2], // pixel size (mm)
        const double clipping_dist[2], // dist from src to clipping planes (mm)
        const double step_length       // spacing between planes
    );
    void set_clipping_dist (const double clipping_dist[2]);
    const plm_long* get_image_dim ();
    plm_long get_image_dim (int dim);
    plm_long get_num_steps ();
    const double* get_incr_c ();
    const double* get_incr_r ();
    Proj_matrix *get_proj_matrix ();
    const double* get_nrm ();
    const double* get_src () const;
    const double* get_iso ();
    const double* get_clipping_dist();
    double get_step_length () const;
    const double* get_ul_room ();
    Volume *get_vol ();
    const Volume *get_vol () const;

    void allocate ();

    void save_img (const char* filename);
    void save_img (const std::string& filename);
    void save_header (const char* filename);
    void save_header (const std::string& filename);
    void save_projv (const char* filename);
    void save_projv (const std::string& filename);
    void load_img (const char* filename);
    void load_img (const std::string& filename);
    void load_header (const char* filename);
    void load_header (const std::string& filename);
    void load_projv (const char* filename);
    void load_projv (const std::string& filename);

    /* Project 3D coordinate xyz of cartesian space 
       into 2D coordinate ij coordinate on projection plane.  
       In this version, the inputs and outputs are homogenous, 
       not cartesian. */
    void project_h (double* ij, const double* xyz) const;
    /* Project 3D coordinate xyz of cartesian space 
       into 2D coordinate ij coordinate on projection plane.  
       In this version, the inputs and outputs are cartesian, 
       not homogenous. */
    void project (double* ij, const double* xyz) const;

    void debug ();
};

#endif
