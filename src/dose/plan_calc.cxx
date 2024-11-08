/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "aperture.h"
#include "beam_calc.h"
#include "compute_dose.h"
#include "dose_volume_functions.h"
#include "float_pair_list.h"
#include "plan_calc.h"
#include "plm_image.h"
#include "plm_exception.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "ray_data.h"
#include "rpl_volume.h"
#include "rt_beam_model.h"
#include "rt_depth_dose.h"
#include "rt_dose_timing.h"
#include "rt_lut.h"
#include "rt_parms.h"
#include "rt_sigma.h"
#include "rt_mebs.h"
#include "rt_study.h"
#include "rtplan.h"
#include "rtplan_beam.h"
#include "string_util.h"
#include "volume.h"
#include "volume_adjust.h"
#include "volume_header.h"
#include "volume_macros.h"
#include "volume_resample.h"

class Plan_calc_private {

public:
    bool debug;
    float rdp[3];
    bool have_rdp;
    bool have_dose_norm;
    float normalization_dose; // dose prescription
    char non_norm_dose;
    float depth_dose_max;

    /* Filenames for input and output images */
    std::string patient_fn;
    std::string target_fn;
    std::string output_dose_fn;
    std::string output_dicom;
    std::string output_psp_fn;
    std::string debug_dir;

    /* Patient (hu), patient (ed or sp) , target, output dose volume */
    Plm_image::Pointer patient_hu;
    Plm_image::Pointer patient_psp;
    Plm_image::Pointer target;
    Plm_image::Pointer dose;

    Rt_parms::Pointer rt_parms;
    Rt_study::Pointer rt_study;

    Rt_dose_timing::Pointer rt_dose_timing;

    Rt_beam_model beam_model;

    /* Storage of beams */
    std::vector<Beam_calc*> beam_storage;

public: 
    Plan_calc_private ()
    {
        debug = false;
        this->rdp[0] = 0.f;
        this->rdp[1] = 0.f;
        this->rdp[2] = 0.f;
        this->have_rdp = false;
        this->have_dose_norm = false;
        this->normalization_dose = 100.f;
        this->non_norm_dose = 'n';
        this->depth_dose_max = 1.f;
        
        patient_hu = Plm_image::New();
        patient_psp = Plm_image::Pointer();
        target = Plm_image::Pointer();
        dose = Plm_image::New();
        rt_parms = Rt_parms::New ();
        rt_study = Rt_study::New ();
        rt_dose_timing = Rt_dose_timing::New ();
    }

    ~Plan_calc_private ()
    {
    }
};

Plan_calc::Plan_calc ()
{
    this->d_ptr = new Plan_calc_private;
    d_ptr->rt_parms->set_plan_calc (this);
}

Plan_calc::~Plan_calc ()
{
    delete d_ptr;
}

Plm_return_code
Plan_calc::load_command_file (const char *command_file)
{
    return d_ptr->rt_parms->load_command_file (command_file);
}

Plm_return_code
Plan_calc::load_beam_model (const char *beam_model)
{
    return PLM_SUCCESS;
}

Plm_return_code
Plan_calc::load_dicom_plan (const char *dicom_input_dir)
{
    d_ptr->rt_dose_timing->timer_io.resume ();
    d_ptr->rt_study = Rt_study::New ();
    d_ptr->rt_study->load (dicom_input_dir);
    if (d_ptr->rt_study->have_image()) {
        this->set_patient (d_ptr->rt_study->get_image());
    }
    if (d_ptr->rt_study->have_plan()) {
        this->set_rtplan (d_ptr->rt_study->get_plan());
    }
    d_ptr->rt_dose_timing->timer_io.stop ();
    return PLM_SUCCESS;
}

void
Plan_calc::set_patient (const std::string& patient_fn)
{
    d_ptr->patient_fn = patient_fn;
}

void
Plan_calc::set_patient (Plm_image::Pointer& ct_vol)
{
    d_ptr->patient_hu = ct_vol;
    d_ptr->patient_hu->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
    d_ptr->patient_psp = Plm_image::Pointer ();
}

void
Plan_calc::set_patient (ShortImageType::Pointer& ct_vol)
{
    /* compute_segdepth_volume assumes float, so convert here */
    d_ptr->patient_hu->set_itk (ct_vol);
    d_ptr->patient_hu->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
    d_ptr->patient_psp = Plm_image::Pointer ();
}

void
Plan_calc::set_patient (FloatImageType::Pointer& ct_vol)
{
    d_ptr->patient_hu->set_itk (ct_vol);
    d_ptr->patient_hu->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
    d_ptr->patient_psp = Plm_image::Pointer ();
}

void
Plan_calc::set_patient (Volume* ct_vol)
{
    d_ptr->patient_hu->set_volume (ct_vol);
}

Volume::Pointer
Plan_calc::get_patient_volume ()
{
    return d_ptr->patient_hu->get_volume_float ();
}

Plm_image *
Plan_calc::get_patient ()
{
    return d_ptr->patient_hu.get();
}

void
Plan_calc::set_target (const std::string& target_fn)
{
    d_ptr->target_fn = target_fn;
}

void
Plan_calc::set_target (UCharImageType::Pointer& target_vol)
{
    d_ptr->target = Plm_image::New (target_vol);

    /* compute_segdepth_volume assumes float */
    d_ptr->target->convert (PLM_IMG_TYPE_GPUIT_FLOAT);

    this->propagate_target_to_beams ();
}

void
Plan_calc::set_target (FloatImageType::Pointer& target_vol)
{
    d_ptr->target = Plm_image::New (target_vol);

    /* compute_segdepth_volume assumes float */
    d_ptr->target->convert (PLM_IMG_TYPE_GPUIT_FLOAT);

    this->propagate_target_to_beams ();
}

void
Plan_calc::load_target ()
{
    if (d_ptr->target_fn == "") {
        return;
    }
    d_ptr->target = Plm_image::New (new Plm_image (d_ptr->target_fn));

    /* Need float, because compute_segdepth_volume assumes float */
    d_ptr->target->convert (PLM_IMG_TYPE_GPUIT_FLOAT);

    this->propagate_target_to_beams ();
}

Plm_image::Pointer&
Plan_calc::get_target ()
{
    return d_ptr->target;
}

void
Plan_calc::set_rtplan (const Rtplan::Pointer& rtplan)
{
    /* Copy information from Rtplan object into Plan_calc object */
    std::vector<Rtplan_beam*>::const_iterator it = rtplan->beamlist.begin();
    while (it != rtplan->beamlist.end()) {
        printf ("Beam name = %s\n", (*it)->name.c_str());
        Beam_calc *beam_calc = append_beam ();
        beam_calc->set_rtplan_beam (*it);
        ++it;
    }
}

Beam_calc*
Plan_calc::append_beam ()
{
    Beam_calc* last_beam = get_last_rt_beam ();
    Beam_calc* new_beam;
    if (last_beam) {
        new_beam = new Beam_calc (last_beam);
    } else {
        new_beam = new Beam_calc;
    }
    d_ptr->beam_storage.push_back (new_beam);
    new_beam->set_rt_dose_timing (d_ptr->rt_dose_timing);
    new_beam->set_target (d_ptr->target);
    return new_beam;
}

Beam_calc*
Plan_calc::get_last_rt_beam ()
{
    if (d_ptr->beam_storage.empty()) {
        return 0;
    }
    return d_ptr->beam_storage.back();
}

bool
Plan_calc::get_debug (void) const
{
    return d_ptr->debug;
}

void
Plan_calc::set_debug (bool debug)
{
    d_ptr->debug = debug;
}

void
Plan_calc::set_threading (Threading threading)
{
    /* Not used yet */
}

void
Plan_calc::set_normalization_dose (float normalization_dose)
{
    d_ptr->normalization_dose = normalization_dose;
}

float
Plan_calc::get_normalization_dose()
{
    return d_ptr->normalization_dose;
}

const float*
Plan_calc::get_ref_dose_point () const
{
    return d_ptr->rdp;
}

float
Plan_calc::get_ref_dose_point (int dim) const
{
    return d_ptr->rdp[dim];
}

void
Plan_calc::set_ref_dose_point (const float* rdp)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->rdp[d] = rdp[d];
    }
}

void
Plan_calc::set_ref_dose_point (const double* rdp)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->rdp[d] = rdp[d];
    }
}

void 
Plan_calc::set_have_ref_dose_point(bool have_rdp)
{
    d_ptr->have_rdp = have_rdp;
}

bool
Plan_calc::get_have_ref_dose_point()
{
    return d_ptr->have_rdp;
}

void 
Plan_calc::set_have_dose_norm (bool have_dose_norm)
{
    d_ptr->have_dose_norm = have_dose_norm;
}

bool
Plan_calc::get_have_dose_norm()
{
    return d_ptr->have_dose_norm;
}

char 
Plan_calc::get_non_norm_dose () const
{
    return d_ptr->non_norm_dose;
}
    
void 
Plan_calc::set_non_norm_dose (char non_norm_dose)
{
    d_ptr->non_norm_dose = non_norm_dose;
}

void
Plan_calc::propagate_target_to_beams ()
{
    /* Loop through beams, and reset target on them */
    for (size_t i = 0; i < d_ptr->beam_storage.size(); i++) {
        d_ptr->beam_storage[i]->set_target(d_ptr->target);
    }
}

void
Plan_calc::create_patient_psp ()
{
    Float_pair_list lookup;
    lookup.push_back (std::pair<float,float> (NLMIN(float), 0));
    lookup.push_back (std::pair<float,float> (-1000, 0.00106));
    lookup.push_back (std::pair<float,float> (0, 1.0));
    lookup.push_back (std::pair<float,float> (41.46, 1.048674));
    lookup.push_back (std::pair<float,float> (NLMAX(float), 0.005011));

    Volume::Pointer psp = volume_adjust (
        d_ptr->patient_hu->get_volume(), lookup);
    d_ptr->patient_psp = Plm_image::New (psp);
}

void
Plan_calc::normalize_beam_dose (Beam_calc *beam)
{
    Plm_image::Pointer dose = beam->get_dose ();
    Volume::Pointer dose_vol = dose->get_volume ();
    float* dose_img = (float*) dose_vol->img;
    
    /* Dose normalization process*/
    if (this->get_non_norm_dose() != 'y')
    {
        if (this->get_have_ref_dose_point()) // case 1: ref dose point defined
        {
            float rdp_ijk[3] = {0,0,0};
            float rdp[3] = {this->get_ref_dose_point(0), this->get_ref_dose_point(1), this->get_ref_dose_point(2)};
            rdp_ijk[0] = (rdp[0] - dose_vol->origin[0]) / dose_vol->spacing[0];
            rdp_ijk[1] = (rdp[1] - dose_vol->origin[1]) / dose_vol->spacing[1];
            rdp_ijk[2] = (rdp[2] - dose_vol->origin[2]) / dose_vol->spacing[2];
			
            if (rdp_ijk[0] >=0 && rdp_ijk[1] >=0 && rdp_ijk[2] >=0 && rdp_ijk[0] < dose_vol->dim[0] && rdp_ijk[1] < dose_vol->dim[1] && rdp_ijk[2] < dose_vol->dim[2])
            {
                printf("Dose normalized to the dose reference point.\n");
                dose_normalization_to_dose_and_point(dose_vol, beam->get_beam_weight() * this->get_normalization_dose(), rdp_ijk, rdp, beam); // if no normalization dose, norm_dose = 1 by default
                if (this->get_have_dose_norm())
                {
                    printf("%lg x %lg Gy.\n", beam->get_beam_weight(), this->get_normalization_dose());
                }
                else
                {
                    printf("%lg x 100%%.\n", beam->get_beam_weight());
                }
                printf("Primary PB num. x, y: %ld, %ld, primary PB res. x, y: %lg PB/mm, %lg PB/mm\n", beam->get_aperture()->get_dim(0), beam->get_aperture()->get_dim(1), 1.0 / (double) beam->get_aperture()->get_spacing(0), 1.0 / (double) beam->get_aperture()->get_spacing(1));
            }
            else
            {
                printf("***WARNING***\nThe reference dose point is not in the image volume.\n");
                dose_normalization_to_dose(dose_vol, beam->get_beam_weight() * this->get_normalization_dose(), beam);
                if (this->get_have_dose_norm())
                {
                    printf("%lg x %lg Gy.\n", beam->get_beam_weight(), this->get_normalization_dose());
                }
                else
                {
                    printf("%lg x 100%%.\n", beam->get_beam_weight());
                }
                printf("Primary PB num. x, y: %ld, %ld, primary PB res. x, y: %lg PB/mm, %lg PB/mm\n", beam->get_aperture()->get_dim(0), beam->get_aperture()->get_dim(1), 1.0 / (double) beam->get_aperture()->get_spacing(0), 1.0 / (double) beam->get_aperture()->get_spacing(1));
            }
        }
        else // case 2: no red dose point defined
        {				
            dose_normalization_to_dose(dose_vol, beam->get_beam_weight() * this->get_normalization_dose(), beam); // normalization_dose = 1 if no dose_prescription is set
            if (this->get_have_dose_norm())
            {
                printf("%lg x %lg Gy.\n", beam->get_beam_weight(), this->get_normalization_dose());
            }
            else
            {
                printf("%lg x 100%%.\n", beam->get_beam_weight());
            }
            printf("Primary PB num. x, y: %ld, %ld, primary PB res. x, y: %lg PB/mm, %lg PB/mm\n", beam->get_aperture()->get_dim(0), beam->get_aperture()->get_dim(1), 1.0 / (double) beam->get_aperture()->get_spacing(0), 1.0 / (double) beam->get_aperture()->get_spacing(1));
        }
    }
    else // raw dose, dose not normalized
    {
        for (int i = 0; i < dose_vol->dim[0] * dose_vol->dim[1] * dose_vol->dim[2]; i++)
        {
            dose_img[i] *= beam->get_beam_weight();
        }
    }
}

void
Plan_calc::compute_beam_dose (Beam_calc *beam)
{
    printf ("Plan_calc::compute_beam_dose(beam) entry --\n");
    d_ptr->rt_dose_timing->timer_misc.resume ();
    Volume::Pointer ct_vol = this->get_patient_volume ();
    Volume::Pointer dose_vol = ct_vol->clone_empty ();

    Volume* dose_volume_tmp = new Volume;

    float margin = 0;
    int margins[2] = {0,0};
    double range = 0;
    int new_dim[2]={0,0};
    double new_center[2]={0,0};
    double biggest_sigma_ever = 0;

    /* Convert from HU to stopping power, if not already done */
    if (!d_ptr->patient_psp) {
        this->create_patient_psp ();
    }

    /* Resample target to match CT resolution, if not already done */
    if (d_ptr->target) {
        Volume_header vh (d_ptr->patient_hu);
        d_ptr->target->set_volume (
            volume_resample (d_ptr->target->get_volume(), &vh));
        this->propagate_target_to_beams ();
    }
    d_ptr->rt_dose_timing->timer_misc.stop ();
    
    /* Create rpl images, compute beam modifiers, SOBP etc. according 
       to the teatment strategy */
    d_ptr->rt_dose_timing->timer_dose_calc.resume ();
    if (!beam->prepare_for_calc (d_ptr->patient_hu,
            d_ptr->patient_psp, d_ptr->target))
    {
        print_and_exit ("ERROR: Unable to initialize plan.\n");
    }
    d_ptr->rt_dose_timing->timer_dose_calc.stop ();

#if defined (commentout)
    printf ("Computing rpl_ct\n");
    beam->hu_samp_vol->compute_rpl_HU ();
#endif
    
    if (beam->get_flavor() == "a") {
        compute_dose_a (dose_vol, beam, ct_vol);
    }
    else if (beam->get_flavor() == "b") {

        d_ptr->rt_dose_timing->timer_dose_calc.resume ();

        // Add range compensator to rpl volume
        if (beam->rsp_accum_vol->get_aperture()->have_range_compensator_image()) {
            beam->add_rcomp_length_to_rpl_volume ();
        }
        
        // Loop through energies
        Rt_mebs::Pointer mebs = beam->get_mebs();
        std::vector<Rt_depth_dose*> depth_dose = mebs->get_depth_dose();
        for (size_t i = 0; i < depth_dose.size(); i++) {
            compute_dose_b (beam, i, ct_vol);
        }
        d_ptr->rt_dose_timing->timer_dose_calc.stop ();
        d_ptr->rt_dose_timing->timer_reformat.resume ();
        dose_volume_reconstruction (beam->dose_rv, dose_vol);
        d_ptr->rt_dose_timing->timer_reformat.stop ();
    }
    else if (beam->get_flavor() == "ray_trace_dij_a")
    {
        /* This is the same as alg 'a', except that it computes 
           and exports Dij matrices */
        d_ptr->rt_dose_timing->timer_dose_calc.resume ();
        // Loop through energies
        Rt_mebs::Pointer mebs = beam->get_mebs();
        std::vector<Rt_depth_dose*> depth_dose = mebs->get_depth_dose();
        for (size_t i = 0; i < depth_dose.size(); i++) {
            compute_dose_ray_trace_dij_a (beam, i, ct_vol, dose_vol);
        }
        d_ptr->rt_dose_timing->timer_dose_calc.resume ();
    }
    else if (beam->get_flavor() == "ray_trace_dij_b")
    {
        /* This is the same as alg 'b', except that it computes 
           and exports Dij matrices */

        d_ptr->rt_dose_timing->timer_dose_calc.resume ();

        // Add range compensator to rpl volume
        if (beam->rsp_accum_vol->get_aperture()->have_range_compensator_image()) {
            beam->add_rcomp_length_to_rpl_volume ();
        }
        
        // Loop through energies
        compute_dose_ray_trace_dij_b (beam, ct_vol, dose_vol);
        d_ptr->rt_dose_timing->timer_dose_calc.stop ();

        d_ptr->rt_dose_timing->timer_reformat.resume ();
        dose_volume_reconstruction (beam->dose_rv, dose_vol);
        d_ptr->rt_dose_timing->timer_reformat.stop ();
    }
    else if (beam->get_flavor() == "d") {
        // Loop through energies
        Rt_mebs::Pointer mebs = beam->get_mebs();
        std::vector<Rt_depth_dose*> depth_dose = mebs->get_depth_dose();
        for (size_t i = 0; i < depth_dose.size(); i++) {
            compute_dose_d (beam, i, ct_vol);
        }
        d_ptr->rt_dose_timing->timer_reformat.resume ();
        dose_volume_reconstruction (beam->dose_rv, dose_vol);
        d_ptr->rt_dose_timing->timer_reformat.stop ();
    }

    d_ptr->rt_dose_timing->timer_misc.resume ();
    Plm_image::Pointer dose = Plm_image::New();
    dose->set_volume (dose_vol);
    beam->set_dose (dose);
    this->normalize_beam_dose (beam);
    d_ptr->rt_dose_timing->timer_misc.stop ();
}

Plm_return_code
Plan_calc::compute_plan ()
{
    d_ptr->rt_dose_timing->reset ();
    
    if (!d_ptr->rt_parms) {
        print_and_exit ("Error: cannot compute_plan without an Rt_parms\n");
    }

    if (d_ptr->output_dose_fn == "" && d_ptr->output_dicom == "") {
        print_and_exit ("Error: No output dose file specified\n");
    }
    if (d_ptr->patient_fn == "" && !d_ptr->rt_study->have_image()) {
        print_and_exit ("Error: No input CT image specified\n");
    }

    /* Load the patient CT image */
    d_ptr->rt_dose_timing->timer_io.resume ();
    if (d_ptr->patient_fn != "") {
        Plm_image::Pointer ct = Plm_image::New (d_ptr->patient_fn,
            PLM_IMG_TYPE_ITK_FLOAT);
        if (!ct) {
            print_and_exit ("Error: Unable to load patient volume.\n");
        }
        this->set_patient (ct);
    }

    /* Load the patient target structure */
    this->load_target ();
    d_ptr->rt_dose_timing->timer_io.stop ();

    /* Display debugging information */
    d_ptr->rt_dose_timing->timer_misc.resume ();
    this->print_verif ();
    
    Volume::Pointer ct_vol = this->get_patient_volume ();
    Volume::Pointer dose_vol = ct_vol->clone_empty ();
    plm_long dim[3] = {dose_vol->dim[0], dose_vol->dim[1], dose_vol->dim[2]};
    float* total_dose_img = (float*) dose_vol->img;
    d_ptr->rt_dose_timing->timer_misc.stop ();
    
    for (size_t i = 0; i < d_ptr->beam_storage.size(); i++)
    {
        printf ("\nStart dose calculation Beam %d\n", (int) i + 1);
        Beam_calc *beam = d_ptr->beam_storage[i];

        /* Generate dose */
        this->set_debug (true);
        this->compute_beam_dose (beam);

        /* Save beam data */
        d_ptr->rt_dose_timing->timer_io.resume ();
        if (d_ptr->debug_dir != "") {
            beam->set_proj_img_out (string_format ("%s/%02d_proj_img",
                    d_ptr->debug_dir.c_str(), i));
            beam->set_proj_dose_out (string_format ("%s/%02d_proj_dose.nrrd",
                    d_ptr->debug_dir.c_str(), i));
            beam->set_wed_out (string_format ("%s/%02d_wed.nrrd",
                    d_ptr->debug_dir.c_str(), i));
            beam->set_mebs_out (string_format ("%s/%02d_mebs.txt",
                    d_ptr->debug_dir.c_str(), i));
        }
        beam->save_beam_output ();
        d_ptr->rt_dose_timing->timer_io.stop ();

        /* Dose cumulation to the plan dose volume */
        d_ptr->rt_dose_timing->timer_misc.resume ();
        float* beam_dose_img = (float*) d_ptr->beam_storage[i]->get_dose()->get_volume()->img;
        for (int j = 0; j < dim[0] * dim[1] * dim[2]; j++)
        {
            total_dose_img[j] += beam_dose_img[j];
        }
        d_ptr->rt_dose_timing->timer_misc.stop ();
    }

    /* Save stopping power image */
    d_ptr->rt_dose_timing->timer_io.resume ();
    if (d_ptr->output_psp_fn != "") {
        d_ptr->patient_psp->save_image (d_ptr->output_psp_fn);
    }
    
    /* Save dose output */
    if (d_ptr->output_dose_fn != "" || d_ptr->output_dicom != "") {
        Plm_image::Pointer dose = Plm_image::New();
        dose->set_volume (dose_vol);
        this->set_dose(dose);

        if (d_ptr->output_dose_fn != "") {
            this->get_dose()->save_image (d_ptr->output_dose_fn.c_str());
        }
        if (d_ptr->output_dicom != "") {
            d_ptr->rt_study->set_dose (dose);
            d_ptr->rt_study->save_dicom_dose (d_ptr->output_dicom.c_str());
        }
    }
    d_ptr->rt_dose_timing->timer_io.stop ();
    d_ptr->rt_dose_timing->report ();
    
    printf ("done.  \n\n");
    return PLM_SUCCESS;
}

Plm_image::Pointer
Plan_calc::get_dose ()
{
    return d_ptr->dose;
}

FloatImageType::Pointer
Plan_calc::get_dose_itk ()
{
    return d_ptr->dose->itk_float();
}

void 
Plan_calc::set_output_dose_fn (const std::string& output_dose_fn)
{
    d_ptr->output_dose_fn = output_dose_fn;
}

void 
Plan_calc::set_output_dicom (const std::string& output_dicom)
{
    d_ptr->output_dicom = output_dicom;
}

void 
Plan_calc::set_output_psp_fn (const std::string& output_psp_fn)
{
    d_ptr->output_psp_fn = output_psp_fn;
}

void 
Plan_calc::set_debug_directory (const std::string& debug_dir)
{
    d_ptr->debug_dir = debug_dir;
}

void 
Plan_calc::set_dose (Plm_image::Pointer& dose)
{
    d_ptr->dose = dose;
}

void
Plan_calc::print_verif ()
{
    printf("\n [PLAN]");
    printf("\n patient : %s", d_ptr->patient_fn.c_str());
    printf("\n target : %s", d_ptr->target_fn.c_str());
    printf("\n dose_out : %s", d_ptr->output_dose_fn.c_str());
    printf("\n debug : %d", this->get_debug());
    printf("\n dose norm : %lg", this->get_normalization_dose());

    printf("\n \n [SETTINGS]");
    int num_beams = d_ptr->beam_storage.size();
    printf("\n flavor: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_flavor().c_str());}
    printf("\n homo_approx: "); for (int i = 0; i < num_beams; i++) {printf("%c ** ", d_ptr->beam_storage[i]->get_homo_approx());}
    printf("\n ray_step: "); for (int i = 0; i < num_beams; i++) {printf("%lg ** ", d_ptr->beam_storage[i]->get_step_length());}
    printf("\n aperture_out: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_aperture_out().c_str());}
    printf("\n proj_dose_out: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_proj_dose_out().c_str());}
    printf("\n proj_img_out: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_proj_img_out().c_str());}
    printf("\n range_comp_out: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_range_compensator_out().c_str());}
    printf("\n sigma_out: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_sigma_out().c_str());}
    printf("\n wed_out: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_wed_out().c_str());}
    printf("\n beam_weight: "); for (int i = 0; i < num_beams; i++) {printf("%g ** ", d_ptr->beam_storage[i]->get_beam_weight());}

    printf("\n \n [GEOMETRY & APERTURE]");
    printf("\n source: "); for (int i = 0; i < num_beams; i++) {printf("%lg %lg %lg ** ", d_ptr->beam_storage[i]->get_source_position()[0], d_ptr->beam_storage[i]->get_source_position()[1], d_ptr->beam_storage[i]->get_source_position()[2]);}
    printf("\n isocenter: "); for (int i = 0; i < num_beams; i++) {printf("%lg %lg %lg ** ", d_ptr->beam_storage[i]->get_isocenter_position()[0], d_ptr->beam_storage[i]->get_isocenter_position()[1], d_ptr->beam_storage[i]->get_isocenter_position()[2]);}
    printf("\n vup: "); for (int i = 0; i < num_beams; i++) {printf("%lg %lg %lg ** ", d_ptr->beam_storage[i]->get_aperture()->vup[0], d_ptr->beam_storage[i]->get_aperture()->vup[1], d_ptr->beam_storage[i]->get_aperture()->vup[2]);}
    printf("\n offset: "); for (int i = 0; i < num_beams; i++) {printf("%lg ** ", d_ptr->beam_storage[i]->get_aperture()->get_distance());}
    printf("\n ap_center in pixels: "); for (int i = 0; i < num_beams; i++) {printf("%lg %lg ** ", d_ptr->beam_storage[i]->get_aperture()->get_center()[0], d_ptr->beam_storage[i]->get_aperture()->get_center()[1]);}
    printf("\n i_res: "); for (int i = 0; i < num_beams; i++) {printf("%ld %ld ** ", d_ptr->beam_storage[i]->get_aperture()->get_dim()[0], d_ptr->beam_storage[i]->get_aperture()->get_dim()[1]);}
    printf("\n spacing: "); for (int i = 0; i < num_beams; i++) {printf("%lg %lg ** ", d_ptr->beam_storage[i]->get_aperture()->get_spacing()[0], d_ptr->beam_storage[i]->get_aperture()->get_spacing()[1]);}
    printf("\n source_size: "); for (int i = 0; i < num_beams; i++) {printf("%lg ** ", d_ptr->beam_storage[i]->get_source_size());}
    printf("\n ap_file_in: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_aperture_in().c_str());}
    printf("\n rc_file_in: "); for (int i = 0; i < num_beams; i++) {printf("%s ** ", d_ptr->beam_storage[i]->get_range_compensator_in().c_str());}
    printf("\n smearing: "); for (int i = 0; i < num_beams; i++) {printf("%lg ** ", d_ptr->beam_storage[i]->get_smearing());}

    printf("\n \n [PEAK]");
    printf("\n E0: ");
    for (int i = 0; i < num_beams; i++) {
        printf("P%d ",i);
        for (size_t j = 0; j < d_ptr->beam_storage[i]->get_mebs()->get_depth_dose().size(); j++)
        {
            printf("%lg ** ", d_ptr->beam_storage[i]->get_mebs()->get_depth_dose()[j]->E0);
        }
    }
    printf("\n spread: ");
    for (int i = 0; i < num_beams; i++) {
        printf("P%d ",i);
        for (size_t j = 0; j < d_ptr->beam_storage[i]->get_mebs()->get_depth_dose().size(); j++)
        {
            printf("%lg ** ", d_ptr->beam_storage[i]->get_mebs()->get_depth_dose()[j]->spread);
        }
    }
    printf("\n weight: ");
    for (int i = 0; i < num_beams; i++) {
        printf("P%d ",i);
        for (size_t j = 0; j < d_ptr->beam_storage[i]->get_mebs()->get_depth_dose().size(); j++) {
            printf("%lg ** ", d_ptr->beam_storage[i]->get_mebs()->get_weight()[j]);
        }
    }
    printf ("\n");
}
