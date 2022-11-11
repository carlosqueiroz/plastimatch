/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _beam_calc_h_
#define _beam_calc_h_

#include "plmdose_config.h"
#include <string>
#include <vector>

#include "aperture.h"
#include "particle_type.h"
#include "rpl_volume.h"
#include "rtplan_beam.h"
#include "rt_dose_timing.h"
#include "rt_mebs.h"
#include "smart_pointer.h"

class Beam_calc_private;
class Rt_mebs;

/*! \brief 
 * The Beam_calc class encapsulates all data needed to calculate dose for 
 * a single Rt beam.
 */
class PLMDOSE_API Beam_calc {
public:
    SMART_POINTER_SUPPORT (Beam_calc);
    Beam_calc_private *d_ptr;
public:
    Beam_calc ();
    Beam_calc (const Beam_calc* beam_calc);
    ~Beam_calc ();

public:
    /*! \name Inputs */
    /*! \brief Load PDD from XiO or txt file */
    bool load (const char* fn);

    /*! \brief Copy information from Rtplan_beam object into Beam_calc object */
    void set_rtplan_beam (const Rtplan_beam *rtplan_beam);

    /*! \brief Get the position of the beam source in world coordinates. */
    const double* get_source_position () const;
    /*! \brief Get the x, y, or z coordinate of the beam source 
      in world coordinates. */
    double get_source_position (int dim) const;
    /*! \brief Set the position of the beam source in world coordinates. */
    void set_source_position (const float position[3]);
    /*! \brief Set the position of the beam source in world coordinates. */
    void set_source_position (const double position[3]);
    /*! \brief Compute the position of the beam source in world coordinates. */
    void compute_source_position (
        float gantry_angle,
        float patient_support_angle,
        const float *virtual_source_axis_distances);

    /*! \brief Get the position of the beam isocenter in world coordinates. */
    const double* get_isocenter_position () const;
    /*! \brief Get the x, y, or z coordinate of the beam source 
      in world coordinates. */
    double get_isocenter_position (int dim) const;
    /*! \brief Set the position of the beam isocenter in world coordinates. */
    void set_isocenter_position (const float position[3]);
    /*! \brief Set the position of the beam isocenter in world coordinates. */
    void set_isocenter_position (const double position[3]);

    /*! \brief Get the source distance. */
    double get_source_distance () const;
    
    /*! \brief Get "flavor" parameter of dose calculation algorithm */
    const std::string& get_flavor () const;
    /*! \brief Set "flavor" parameter of dose calculation algorithm */
    void set_flavor (const std::string& flavor);

    /*! \brief Get "homo_approx" parameter of dose calculation algorithm */
    char get_homo_approx () const;
    /*! \brief Set "homo_approx" parameter of dose calculation algorithm */
    void set_homo_approx (char homo_approx);

    /*! \brief Get mebs */
    Rt_mebs::Pointer get_mebs();

    /*! \brief Get "beam_weight" parameter of dose calculation algorithm */
    float get_beam_weight () const;
    /*! \brief Set "beam_weight" parameter of dose calculation algorithm */
    void set_beam_weight (float beam_weight);

    /*! \brief Get "rc_MC_model" for the model of the range compensator, y = Monte Carlo, n = Highland */
    char get_rc_MC_model () const;
    /*! \brief Set "rc_MC_model" for the model of the range compensator, y = Monte Carlo, n = Highland */
    void set_rc_MC_model (char rc_MC_model);

    /* Set source size in mm */
    void set_source_size(float source_size);

    /* Get source size in mm */
    float get_source_size() const;

    /*! \brief Request debugging information to be written to directory */
    void set_debug (const std::string& dir);

    /*! \name Outputs */
    void dump (const char* dir);
    void dump (const std::string& dir);

    /* Spot scanning */
    void add_spot (
        float xpos, float ypos, float energy, float sigma, float weight);

    /* Compute beam modifiers, SOBP etc. according to the teatment strategy */
    bool prepare_for_calc (
        Plm_image::Pointer& ct_hu,
        Plm_image::Pointer& ct_psp,
        Plm_image::Pointer& target);

    /* Different strategies preparation */
    void compute_beam_data_from_beamlet_map();
    void compute_beam_data_from_spot_map();
    void compute_beam_data_from_manual_peaks(Plm_image::Pointer& target);
    void compute_beam_data_from_prescription(Plm_image::Pointer& target);
    void compute_beam_data_from_target(Plm_image::Pointer& target);
    void compute_default_beam();

    /* This computes the aperture and range compensator */
    void compute_beam_modifiers_active_scanning_a (
        Volume *seg_vol, float smearing, float proximal_margin,
        float distal_margin);
    void compute_beam_modifiers_passive_scattering_a (
        Volume *seg_vol, float smearing, float proximal_margin, 
        float distal_margin);
    void compute_beam_modifiers_active_scanning_b (
        Volume *seg_vol, float smearing, float proximal_margin,
        float distal_margin, std::vector<double>& map_wed_min,
        std::vector<double>& map_wed_max);
    void compute_beam_modifiers_passive_scattering_b (
        Volume *seg_vol, float smearing, float proximal_margin, 
        float distal_margin, std::vector<double>& map_wed_min, 
        std::vector<double>& map_wed_max);
    void compute_beam_modifiers (
        Volume *seg_vol,
        bool active,
        float smearing,
        float proximal_margin,
        float distal_margin,
        std::vector<double>& map_wed_min,
        std::vector<double>& map_wed_max);
    void apply_smearing_to_target (
        float smearing,
        std::vector <double>& map_min_distance,
        std::vector <double>& map_max_distance);
    void compute_target_wepl_min_max (
        std::vector<double>& map_wed_min,
        std::vector<double>& map_wed_max);
    void add_rcomp_length_to_rpl_volume ();

    /* Set/ Get ct_psp */
    Plm_image::Pointer& get_ct_psp ();
    const Plm_image::Pointer& get_ct_psp () const;
    void set_ct_psp(Plm_image::Pointer& ct_psp);

    /* Set/ Get target */
    Plm_image::Pointer& get_target ();
    const Plm_image::Pointer& get_target () const;
    void set_target(Plm_image::Pointer& target);

    /* Set/ Get timer */
    Rt_dose_timing::Pointer& get_rt_dose_timing ();
    void set_rt_dose_timing (Rt_dose_timing::Pointer& rt_dose_timing);

    /* Set/ Get dose_volume*/
    Plm_image::Pointer& get_dose ();
    const Plm_image::Pointer& get_dose () const;
    void set_dose(Plm_image::Pointer& dose);

    /* Get aperture and range compensator */
    Aperture::Pointer& get_aperture ();
    const Aperture::Pointer& get_aperture () const;
    Plm_image::Pointer& get_aperture_image ();
    const Plm_image::Pointer& get_aperture_image () const;
    const plm_long* get_aperture_dim () const;
    Plm_image::Pointer& get_range_compensator_image ();
    const Plm_image::Pointer& get_range_compensator_image () const;
    void set_aperture_vup (const float[]);
    void set_aperture_distance (float);
    void set_aperture_origin (const float[]);
    void set_aperture_resolution (const plm_long[]);
    void set_aperture_spacing (const float[]);

    void set_step_length(float step);
    float get_step_length();

    /* Set smearing */
    void set_smearing (float smearing);
    float get_smearing();

    /* Set/Get intput file names */
    void set_aperture_in (const std::string& str);
    std::string get_aperture_in();

    void set_range_compensator_in (const std::string& str);
    std::string get_range_compensator_in();

    /* Set/Get output file names */
    void set_aperture_out(std::string str);
    std::string get_aperture_out();

    void set_proj_dose_out(std::string str);
    std::string get_proj_dose_out();

    void set_proj_img_out(std::string str);
    std::string get_proj_img_out();

    void set_range_compensator_out(std::string str);
    std::string get_range_compensator_out();

    void set_sigma_out(std::string str);
    std::string get_sigma_out();

    void set_mebs_out (const std::string& str);
    const std::string& get_mebs_out ();

    void set_beam_dump_out(std::string str);
    std::string get_beam_dump_out();

    void set_dij_out (const std::string& str);
    const std::string& get_dij_out();

    void set_wed_out(std::string str);
    std::string get_wed_out();

    void set_proj_target_out(std::string str);
    std::string get_proj_target_out();

    void set_beam_line_type(std::string str);
    std::string get_beam_line_type();

    bool get_intersection_with_aperture (
        double* idx_ap, plm_long* idx, double* rest, double* ct_xyz);
    bool is_ray_in_the_aperture (
        const plm_long* idx, const unsigned char* ap_img);

    /* computes the minimal geometric distance of the target for this beam
       -- used for smearing */
    float compute_minimal_target_distance(Volume* target_vol, float background);

    /* functions that pass through to mebs object */
    void set_energy_resolution (float eres);
    float get_energy_resolution () const;
    void set_proximal_margin (float proximal_margin);
    float get_proximal_margin() const;
    void set_distal_margin (float distal_margin);
    float get_distal_margin() const;
    void set_prescription (float prescription_min, float prescription_max);

    /* Save beam-specific output files to disk */
    void save_beam_output ();
    
public: 

    /*** Volumes useful for dose calculation */
    /* contains the target */
    Rpl_volume::Pointer target_rv;
    /* contains the radiologic path length along a ray, according to 
       stopping power */
    Rpl_volume* rsp_accum_vol;
    /* contains HU, sampled at each point on the ray */
    Rpl_volume* hu_samp_vol;
    // contains the sigma (lateral spread of the pencil beam, 
    // used to calculate the off-axis term) along the ray 
    Rpl_volume* sigma_vol;

    /* larger volumes for Hong and divergent geometry algorithms */
    Rpl_volume* rpl_vol_lg;
    Rpl_volume* rpl_vol_samp_lg;
    Rpl_volume* sigma_vol_lg;
    Rpl_volume* dose_rv;

private:
    bool load_xio (const char* fn);
    bool load_txt (const char* fn);
};

#endif
