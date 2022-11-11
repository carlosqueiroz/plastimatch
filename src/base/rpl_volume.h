/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plmbase_config.h"
#include <string>
#include "aperture.h"
#include "plm_image.h"
#include "ray_trace_callback.h"
#include "smart_pointer.h"

#define PMMA_DENSITY 1.19		// PMMA density in g
#define PMMA_STPR 0.98		// PMMA Stopping Power Ratio, no dim

PLMBASE_API float compute_PrSTPR_from_HU(float);
PLMBASE_API float compute_PrSTPR_Schneider_weq_from_HU (float CT_HU); // Stopping Power Ratio - Schneider's model
PLMBASE_API float compute_PrWER_from_HU(float CT_HU); // WER = STRP / density
PLMBASE_API float compute_density_from_HU (float CT_HU); // density VS HU - Schneider's model: broken curve

class Proj_volume;
class Ray_data;
class Rpl_volume_private;
class Volume;
class Volume_limit;

enum Rpl_volume_ray_trace_start {
    RAY_TRACE_START_AT_RAY_VOLUME_INTERSECTION,
    RAY_TRACE_START_AT_CLIPPING_PLANE
};

class PLMBASE_API Rpl_volume 
{
public:
    SMART_POINTER_SUPPORT (Rpl_volume);
    Rpl_volume_private *d_ptr;
public:
    Rpl_volume ();
    ~Rpl_volume ();
public:
    void set_geometry (
        const double src[3],           // position of source (mm)
        const double iso[3],           // position of isocenter (mm)
        const double vup[3],           // dir to "top" of projection plane
        double sid,                    // dist from proj plane to source (mm)
        const plm_long image_dim[2],   // resolution of image
        const double image_center[2],  // image center (pixels)
        const double image_spacing[2], // pixel size (mm)
        const double step_length       // spacing between planes
    );
    void clone_geometry (const Rpl_volume *rv);

    void set_ray_trace_start (Rpl_volume_ray_trace_start rvrtt);

    Aperture::Pointer& get_aperture ();
    const Aperture::Pointer& get_aperture () const;
    void set_aperture (Aperture::Pointer& ap);

    Volume* get_vol ();
    const Volume* get_vol () const;
    Proj_volume* get_proj_volume ();
    void set_ct_volume (Plm_image::Pointer& ct_volume);

    const Proj_volume* get_proj_volume () const;

    const plm_long *get_image_dim ();
    plm_long get_num_steps ();

    double get_value (plm_long ap_ij[2], double dist) const;
    double get_value (double ap_ij[2], double dist) const;
    double get_value (const double *xyz) const;

    void set_ct (const Plm_image::Pointer& ct_volume);
    Plm_image::Pointer get_ct();

    void set_ct_limit(Volume_limit* ct_limit);
    Volume_limit* get_ct_limit();

    Ray_data* get_ray_data();
    const Ray_data* get_ray_data() const;
    void set_ray_data (Ray_data *ray);

    void set_front_clipping_plane(double front_clip);
    double get_front_clipping_plane () const;
    void set_back_clipping_plane(double back_clip);
    double get_back_clipping_plane () const;
    double get_step_length () const;

    void compute_rpl_ct_density (); // compute density volume
    void compute_rpl_HU ();	// compute HU volume
    void compute_rpl_void ();	// compute void volume

    void compute_rpl (bool use_aperture, Ray_trace_callback callback);
    void compute_rpl_sample (bool use_aperture);
    void compute_rpl_accum (bool use_aperture);
    
    void compute_rpl_range_length_rgc(); // range length volume creation taking into account the range compensator
    void compute_rpl_PrSTRP_no_rgc (); // compute Proton Stopping Power Ratio volume without considering the range compensator

    double compute_farthest_penetrating_ray_on_nrm(float range); // return the distance from aperture to the farthest which rg_lenght > range

    void compute_wed_volume (Volume *wed_vol, Volume *in_vol, float background);
    void compute_dew_volume (Volume *wed_vol, Volume *dew_vol, float background);
    void compute_proj_wed_volume (Volume *proj_wed_vol, float background);
    
    void compute_volume_aperture(Aperture::Pointer ap);

    void apply_beam_modifiers ();

    void save (const char* filename);
    void save (const std::string& filename);
    void load_rpl (const char* filename);
    void load_rpl (const std::string& filename);
    void load_img (const char *filename);
    void load_img (const std::string& filename);

    void compute_ray_data ();

protected:
    void rpl_ray_trace (
        Volume *ct_vol,              /* I: CT volume */
        Ray_data *ray_data,          /* I: Pre-computed data for this ray */
        Ray_trace_callback callback, /* I: Ray trace callback function */
        Volume_limit *vol_limit,     /* I: CT bounding region */
        const double *src,           /* I: @ source */
        double rc_thk,               /* I: range compensator thickness */
        int* ires                    /* I: ray cast resolution */
    );
protected:
    void save_img (const char* filename);
    void save_img (const std::string& filename);
};

#endif
