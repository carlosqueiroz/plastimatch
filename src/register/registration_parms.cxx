/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <fstream>
#include <list>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "groupwise_parms.h"
#include "logfile.h"
#include "parameter_parser.h"
#include "plm_return_code.h"
#include "registration_parms.h"
#include "shared_parms.h"
#include "stage_parms.h"
#include "string_util.h"

#undef CUDA_FOUND   // GCS 2023-04-24.  Defeat CUDA. 

class Registration_parms_private
{
public:
    std::list<Stage_parms*> stages;
    Shared_parms *shared;
    Groupwise_parms *gw_parms;

public:
    Registration_parms_private () {
        shared = new Shared_parms;
        gw_parms = 0;
    }
    ~Registration_parms_private () {
        delete_all_stages ();
        delete shared;
        delete gw_parms;
    }
    void delete_all_stages () {
        std::list<Stage_parms*>::iterator it;
        for (it = stages.begin(); it != stages.end(); it++) {
            delete *it;
        }
        stages.clear ();
    }
};

class Registration_parms_parser : public Parameter_parser
{
public:
    Registration_parms *rp;
public:
    Registration_parms_parser (Registration_parms *rp)
    {
        this->rp = rp;
        this->enable_key_regularization (true);
        this->set_default_index (DEFAULT_IMAGE_KEY);
    }
public:
    virtual Plm_return_code begin_section (
        const std::string& section)
    {
        if (section == "GLOBAL") {
            return PLM_SUCCESS;
        }
        if (section == "STAGE") {
            rp->append_stage ();
            return PLM_SUCCESS;
        }
        if (section == "COMMENT") {
            return PLM_SUCCESS;
        }
        if (section == "PROCESS") {
            rp->append_process_stage ();
            return PLM_SUCCESS;
        }

        /* else, unknown section */
        return PLM_ERROR;
    }
    virtual Plm_return_code end_section (
        const std::string& section)
    {
        /* Do nothing */
        return PLM_SUCCESS;
    }
    virtual Plm_return_code set_key_value (
        const std::string& section,
        const std::string& key, 
        const std::string& index, 
        const std::string& val)
    {
        return this->rp->set_key_value (section, key, index, val);
    }
};

Registration_parms::Registration_parms()
{
    d_ptr = new Registration_parms_private;

    init_type = STAGE_TRANSFORM_NONE;
    default_value = 0.0;
    num_stages = 0;
}

Registration_parms::~Registration_parms()
{
    delete d_ptr;
}

Plm_return_code
Registration_parms::set_key_value (
    const std::string& section,
    const std::string& key,
    const std::string& index,
    const std::string& val)
{
    int rc;
    Stage_parms *stage = 0;
    Shared_parms *shared = 0;
    Process_parms::Pointer process;
    bool section_global = false;
    bool section_stage = false;
    bool section_process = false;

    if (section == "COMMENT") {
        return PLM_SUCCESS;
    }

    if (section == "GLOBAL") {
        shared = d_ptr->shared;
        section_global = true;
    }
    else if (section == "STAGE") {
        stage = d_ptr->stages.back();
        shared = stage->get_shared_parms();
        section_stage = true;
    }
    else if (section == "PROCESS") {
        stage = d_ptr->stages.back();
        process = stage->get_process_parms();
        section_process = true;
    }

    /* The following keywords are only allowed globally */
    if (key == "xf_in"
        || key == "xform_in"
        || key == "vf_in")
    {
        if (!section_global) goto key_only_allowed_in_section_global;
        this->xf_in_fn = val;
    }
    else if (key == "log" || key == "logfile") {
        if (!section_global) goto key_only_allowed_in_section_global;
        this->log_fn = val;
    }
    else if (key == "group_dir") {
        if (!section_global) goto key_only_allowed_in_section_global;
        this->group_dir = val;
    }

    /* The following keywords are allowed either globally or in stages */
    else if (key == "fixed") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->metric[index].fixed_fn = val;
    }
    else if (key == "moving") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->metric[index].moving_fn = val;
    }
    else if (key == "background_val"
        || key == "default_value")
    {
        float f;
        if (sscanf (val.c_str(), "%g", &f) != 1) {
            goto error_exit;
        }
        if (section_global) {
            this->default_value = f;
        } else if (section_stage) {
            stage->default_value = f;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "fixed_mask" || key == "fixed_roi") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->metric[index].fixed_roi_fn = val;
    }
    else if (key == "moving_mask" || key == "moving_roi") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->metric[index].moving_roi_fn = val;
    }
    else if (key == "fixed_roi_enable") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_roi_enable = string_value_true (val);
    }
    else if (key == "moving_roi_enable")
    {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->moving_roi_enable = string_value_true (val);
    }
    else if (key == "fixed_stiffness") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_stiffness_fn = val;
    }
    else if (key == "fixed_stiffness_enable") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_stiffness_enable = string_value_true (val);
    }
    else if (key == "legacy_subsampling") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->legacy_subsampling = string_value_true (val);
    }
    else if (key == "img_out" || key == "image_out") {
        if (section_global) {
            this->img_out_fn = val;
        } else if (section_stage) {
            stage->img_out_fn = val;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "img_out_fmt") {
        if (section_process) goto key_not_allowed_in_section_process;
        int fmt = IMG_OUT_FMT_AUTO;
        if (val == "dicom") {
            fmt = IMG_OUT_FMT_DICOM;
        } else {
            goto error_exit;
        }
        shared->img_out_fmt = fmt;
    }
    else if (key == "img_out_type") {
        if (section_process) goto key_not_allowed_in_section_process;
        Plm_image_type type = plm_image_type_parse (val.c_str());
        if (type == PLM_IMG_TYPE_UNDEFINED) {
            goto error_exit;
        }
        shared->img_out_type = type;
    }
    else if (key == "resample_when_linear") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->img_out_resample_linear_xf = string_value_true (val);
    }
    else if (key == "vf_out") {
        if (section_global) {
            this->vf_out_fn = val;
        } else if (section_stage) {
            stage->vf_out_fn = val;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "xf_out_itk") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->xf_out_itk = string_value_true (val);
    }
    else if (key == "xf_out" || key == "xform_out") {
        /* xf_out is special.  You can have more than one of these.  
           This capability is used by the slicer plugin. */
        if (section_global) {
            this->xf_out_fn.push_back (val.c_str());
        } else if (section_stage) {
            stage->xf_out_fn.push_back (val.c_str());
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "valid_roi_out") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->valid_roi_out_fn = val;
    }
    else if (key == "fixed_landmarks") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_landmarks_fn = val;
    }
    else if (key == "moving_landmarks") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->moving_landmarks_fn = val;
    }
    else if (key == "fixed_landmark_list") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_landmarks_list = val;
    }
    else if (key == "moving_landmark_list") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->moving_landmarks_list = val;
    }
    else if (key == "warped_landmarks") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->warped_landmarks_fn = val;
    }

    /* The following keywords are only allowed in stages */
    else if (key == "num_substages") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_substages) != 1) {
            goto error_exit;
        }
    }
    else if (key == "flavor" || key == "alg_flavor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val.length() >= 1) {
            stage->alg_flavor = val[0];
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "resume") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->resume_stage = string_value_true (val);
    }
    else if (key == "xform") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "translation") {
            stage->xform_type = STAGE_TRANSFORM_TRANSLATION;
        }
        else if (val == "rigid" || val == "versor") {
            stage->xform_type = STAGE_TRANSFORM_VERSOR;
        }
        else if (val == "quaternion") {
            stage->xform_type = STAGE_TRANSFORM_QUATERNION;
        }
        else if (val == "affine") {
            stage->xform_type = STAGE_TRANSFORM_AFFINE;
        }
        else if (val == "similarity") {
            stage->xform_type = STAGE_TRANSFORM_SIMILARITY;
        }
        else if (val == "bspline") {
            stage->xform_type = STAGE_TRANSFORM_BSPLINE;
        }
        else if (val == "vf") {
            stage->xform_type = STAGE_TRANSFORM_VECTOR_FIELD;
        }
        else if (val == "align_center") {
            stage->xform_type = STAGE_TRANSFORM_ALIGN_CENTER;
        }
        else if (val == "align_center_of_gravity") {
            stage->xform_type = STAGE_TRANSFORM_ALIGN_CENTER_OF_GRAVITY;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "optim") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "none") {
            stage->optim_type = OPTIMIZATION_NO_REGISTRATION;
        }
        else if (val == "amoeba") {
            stage->optim_type = OPTIMIZATION_AMOEBA;
        }
        else if (val == "demons") {
            stage->optim_type = OPTIMIZATION_DEMONS;
        }
        else if (val == "frpr") {
            stage->optim_type = OPTIMIZATION_FRPR;
        }
        else if (val == "grid" || val == "grid_search"
            || val == "gridsearch") {
            stage->optim_type = OPTIMIZATION_GRID_SEARCH;
        }
        else if (val == "lbfgs") {
            stage->optim_type = OPTIMIZATION_LBFGS;
        }
        else if (val == "lbfgsb") {
            stage->optim_type = OPTIMIZATION_LBFGSB;
        }
        else if (val == "liblbfgs") {
            stage->optim_type = OPTIMIZATION_LIBLBFGS;
        }
        else if (val == "nocedal") {
            stage->optim_type = OPTIMIZATION_LBFGSB;
        }
        else if (val == "oneplusone") {
            stage->optim_type = OPTIMIZATION_ONEPLUSONE;
        }
        else if (val == "rsg") {
            stage->optim_type = OPTIMIZATION_RSG;
        }
        else if (val == "steepest") {
            stage->optim_type = OPTIMIZATION_STEEPEST;
        }
        else if (val == "versor") {
            stage->optim_type = OPTIMIZATION_VERSOR;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "impl") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "none") {
            stage->impl_type = IMPLEMENTATION_NONE;
        }
        else if (val == "itk") {
            stage->impl_type = IMPLEMENTATION_ITK;
        }
        else if (val == "plastimatch") {
            stage->impl_type = IMPLEMENTATION_PLASTIMATCH;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "optim_subtype") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "fsf") {
            stage->optim_subtype = OPTIMIZATION_SUB_FSF;
        }
        else if (val == "diffeomorphic") {
            stage->optim_subtype = OPTIMIZATION_SUB_DIFF_ITK;
        }
        else if (val == "log_domain") {
            stage->optim_subtype = OPTIMIZATION_SUB_LOGDOM_ITK;
        }
        else if (val == "sym_log_domain") {
            stage->optim_subtype = OPTIMIZATION_SUB_SYM_LOGDOM_ITK;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "threading") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "single") {
            stage->threading_type = THREADING_CPU_SINGLE;
        }
        else if (val == "openmp") {
#if (OPENMP_FOUND)
            stage->threading_type = THREADING_CPU_OPENMP;
#else
            stage->threading_type = THREADING_CPU_SINGLE;
#endif
        }
        else if (val == "cuda") {
#if (CUDA_FOUND)
            stage->threading_type = THREADING_CUDA;
#elif (OPENMP_FOUND)
            stage->threading_type = THREADING_CPU_OPENMP;
#else
            stage->threading_type = THREADING_CPU_SINGLE;
#endif
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "gpuid") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->gpuid) != 1) {
            goto error_exit;
        }
    }
    else if (key == "metric" || key == "smetric") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (shared->metric[index].set_metric_type (val) != PLM_SUCCESS) {
            goto error_exit;
        }
    }
    else if (key == "metric_lambda" || key == "smetric_lambda") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        float f;
        if (sscanf (val.c_str(), "%f", &f) != 1) {
            goto error_exit;
        }
        shared->metric[index].metric_lambda = f;
    }
    else if (key == "histogram_type") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "eqsp" || val == "EQSP") {
            stage->mi_hist_type = HIST_EQSP;
        }
        else if (val == "vopt" || val == "VOPT") {
            stage->mi_hist_type = HIST_VOPT;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "regularization")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Regularization_type& rtype
            = stage->regularization_parms.regularization_type;
        if (val == "none") {
            rtype = REGULARIZATION_NONE;
        }
        else if (val == "analytic") {
            rtype = REGULARIZATION_BSPLINE_ANALYTIC;
        }
        else if (val == "semi_analytic") {
            rtype = REGULARIZATION_BSPLINE_SEMI_ANALYTIC;
        }
        else if (val == "numeric") {
            rtype = REGULARIZATION_BSPLINE_NUMERIC;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "total_displacement_penalty") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.total_displacement_penalty) != 1) {
            goto error_exit;
        }
    }
    else if (key == "diffusion_penalty") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.diffusion_penalty) != 1) {
            goto error_exit;
        }
    }
    else if (key == "curvature_penalty"
        || key == "regularization_lambda"
        || key == "young_modulus") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.curvature_penalty) != 1) {
            goto error_exit;
        }
#if PLM_CONFIG_LEGACY_SQUARED_REGULARIZER
        stage->regularization_parms.curvature_penalty
            *= stage->regularization_parms.curvature_penalty;
#endif
    }
    else if (key == "curvature_mixed_weight") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.curvature_mixed_weight) != 1) {
            goto error_exit;
        }
    }
    else if (key == "lame_coefficient_1") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.lame_coefficient_1) != 1) {
            goto error_exit;
        }
    }
    else if (key == "lame_coefficient_2") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.lame_coefficient_2) != 1) {
            goto error_exit;
        }
    }
    else if (key == "linear_elastic_multiplier") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.linear_elastic_multiplier) != 1) {
            goto error_exit;
        }
    }
    else if (key == "third_order_penalty") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f",
                &stage->regularization_parms.third_order_penalty) != 1) {
            goto error_exit;
        }
    }
    else if (key == "background_max") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->background_max) != 1) {
            goto error_exit;
        }
    }
    else if (key == "min_its") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->min_its) != 1) {
            goto error_exit;
        }
    }
    else if (key == "iterations" 
        || key == "max_iterations"
        || key == "max_its"
        || key == "its")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->max_its) != 1) {
            goto error_exit;
        }
    }
    else if (key == "learn_rate") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->learn_rate) != 1) {
            goto error_exit;
        }
    }
    else if (key == "grad_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->grad_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "pgtol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f", &stage->pgtol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "lbfgsb_mmax") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->lbfgsb_mmax) != 1) {
            goto error_exit;
        }
    }
    else if (key == "max_step") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->max_step) != 1) {
            goto error_exit;
        }
    }
    else if (key == "min_step") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->min_step) != 1) {
            goto error_exit;
        }
    }
    else if (key == "rsg_grad_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->rsg_grad_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "translation_scale_factor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->translation_scale_factor) != 1) {
            goto error_exit;
        }
    }
    else if (key == "rotation_scale_factor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->rotation_scale_factor) != 1) {
            goto error_exit;
        }
    }
    else if (key == "scaling_scale_factor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f", &stage->scaling_scale_factor) != 1) {
            goto error_exit;
        }
    }
    else if (key == "convergence_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->convergence_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "opo_epsilon") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->opo_epsilon) != 1) {
            goto error_exit;
        }
    }
    else if (key == "opo_initial_search_rad") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->opo_initial_search_rad) != 1) {
            goto error_exit;
        }
    }
    else if (key == "frpr_step_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->frpr_step_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "frpr_step_length") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->frpr_step_length) != 1) {
            goto error_exit;
        }
    }
    else if (key == "frpr_max_line_its") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->frpr_max_line_its) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_histogram_bins" 
        || key == "mi_histogram_bins") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        rc = sscanf (val.c_str(), "%d %d", &stage->mi_hist_fixed_bins,
            &stage->mi_hist_moving_bins);
        if (rc == 1) {
            stage->mi_hist_moving_bins = stage->mi_hist_fixed_bins;
        } else if (rc != 2) {
            goto error_exit;
        }
    }
    else if (key == "mattes_fixed_minVal"
        || key == "mi_fixed_minVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_fixed_image_minVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_fixed_maxVal"
        || key == "mi_fixed_maxVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_fixed_image_maxVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_moving_minVal"
        || key == "mi_moving_minVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_moving_image_minVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_moving_maxVal"
        || key == "mi_moving_maxVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_moving_image_maxVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "num_samples"
        || key == "mattes_num_spatial_samples"
        || key == "mi_num_spatial_samples") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->mi_num_spatial_samples) != 1) {
            goto error_exit;
        }
    }
    else if (key == "num_samples_pct") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f", &stage->mi_num_spatial_samples_pct) != 1) {
            goto error_exit;
        }
    }
    else if ((key == "demons_std_deformation_field") || (key == "demons_std")) {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_std) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_std_update_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_std_update_field) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_step_length") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_step_length) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_smooth_deformation_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->demons_smooth_deformation_field = string_value_true (val);
    }
    else if (key == "demons_smooth_update_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->demons_smooth_update_field = string_value_true (val);
    }
    else if (key == "demons_gradient_type")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "symmetric") {
            stage->demons_gradient_type = SYMMETRIC;
        }
        else if (val == "fixed") {
            stage->demons_gradient_type = FIXED_IMAGE;
        }
        else if (val == "warped_moving") {
            stage->demons_gradient_type = WARPED_MOVING;
        }
        else if (val == "mapped_moving") {
            stage->demons_gradient_type = MAPPED_MOVING;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "num_approx_terms_log_demons") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_approx_terms_log_demons) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_acceleration") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_acceleration) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_homogenization") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_homogenization) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_filter_width") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d %d %d", 
                &(stage->demons_filter_width[0]), 
                &(stage->demons_filter_width[1]), 
                &(stage->demons_filter_width[2])) != 3) {
            goto error_exit;
        }
    }
    else if (key == "amoeba_parameter_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &(stage->amoeba_parameter_tol)) != 1) {
            goto error_exit;
        }
    }
    else if (key == "gridsearch_min_overlap") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->gridsearch_min_overlap[0]), 
                &(stage->gridsearch_min_overlap[1]), 
                &(stage->gridsearch_min_overlap[2])) != 3) {
            goto error_exit;
        }
    }
    else if (key == "gridsearch_min_steps") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d %d %d", 
                &(stage->gridsearch_min_steps[0]), 
                &(stage->gridsearch_min_steps[1]), 
                &(stage->gridsearch_min_steps[2])) != 3) {
            goto error_exit;
        }
    }
    else if (key == "gridsearch_strategy") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "global") {
            stage->gridsearch_strategy = GRIDSEARCH_STRATEGY_GLOBAL;
        }
        else if (val == "local") {
            stage->gridsearch_strategy = GRIDSEARCH_STRATEGY_LOCAL;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "stages") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "global") {
        }
    }
    else if (key == "landmark_stiffness") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->landmark_stiffness) != 1) {
            goto error_exit;
        }
    }   
    else if (key == "landmark_flavor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%c", &stage->landmark_flavor) != 1) {
            goto error_exit;
        }
    }   
    else if (key == "overlap_penalty_lambda") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->overlap_penalty_lambda) != 1) {
            goto error_exit;
        }
    }   
    else if (key == "overlap_penalty_fraction") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->overlap_penalty_fraction) != 1) {
            goto error_exit;
        }
    }   
    else if (key == "res_vox" || key == "res" || key == "ss") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_VOXEL_RATE;
    }
    else if (key == "res_vox_fixed" 
        || key == "ss_fixed" || key == "fixed_ss")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample_fixed (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_VOXEL_RATE;
    }
    else if (key == "res_vox_moving" 
        || key == "ss_moving" || key == "moving_ss")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample_moving (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_VOXEL_RATE;
    }
    else if (key == "res_mm") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_MM;
    }
    else if (key == "res_mm_fixed") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample_fixed (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_MM;
    }
    else if (key == "res_mm_moving") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample_moving (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_MM;
    }
    else if (key == "res_pct") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_PCT;
    }
    else if (key == "res_pct_fixed") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample_fixed (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_PCT;
    }
    else if (key == "res_pct_moving") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = stage->set_resample_moving (val);
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
        stage->resample_type = RESAMPLE_PCT;
    }
    else if (key == "grid_spac"
        || key == "grid_spacing")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        Plm_return_code rc = parse_float13 (stage->grid_spac, val.c_str());
        if (rc != PLM_SUCCESS) {
            goto error_exit;
        }
    }
    else if (key == "lut_type") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "3d aligned") {
            stage->lut_type = Bspline_xform::LUT_3D_ALIGNED;
        } else if (val == "1d aligned") {
            stage->lut_type = Bspline_xform::LUT_1D_ALIGNED;
        } else if (val == "1d unaligned") {
            stage->lut_type = Bspline_xform::LUT_1D_UNALIGNED;
        } else {
            goto error_exit;
        }
    }
    else if (key == "histo_equ") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->histoeq = string_value_true (val);
    }
    else if (key == "thresh_mean_intensity") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->thresh_mean_intensity = string_value_true (val);
    }
    else if (key == "num_hist_levels") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_hist_levels) != 1) {
            goto error_exit;
        }
    }
    else if (key == "num_matching_points") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_matching_points) != 1) {
            goto error_exit;
        }
    }
    else if (key == "debug_dir") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->debug_dir = val;
    }

    /* The following keywords are only allowed in process section */
    else if (section_process) {
        Process_parms::Pointer pp = stage->get_process_parms ();
        if (key == "action") {
            pp->set_action (val);
        } else {
            pp->set_key_value (key, val);
        }
    }

    else {
        goto error_exit;
    }
    return PLM_SUCCESS;

key_only_allowed_in_section_global:
    lprintf (
        "This key (%s) is only allowed in a global section\n", key.c_str());
    return PLM_ERROR;

key_only_allowed_in_section_stage:
    lprintf (
        "This key (%s) is only allowed in a stage section\n", key.c_str());
    return PLM_ERROR;

key_not_allowed_in_section_process:
    lprintf (
        "This key (%s) not is allowed in a process section\n", key.c_str());
    return PLM_ERROR;

error_exit:
    lprintf (
        "Unknown (key,val) combination: (%s,%s)\n", key.c_str(), val.c_str());
    return PLM_ERROR;
}

Plm_return_code
Registration_parms::set_command_string (
    const std::string& command_string
)
{
    this->delete_all_stages ();
    Registration_parms_parser rpp (this);
    return rpp.parse_config_string (command_string);
}

Plm_return_code
Registration_parms::parse_command_file (const char* options_fn)
{
    /* Read file into string */
    std::ifstream t (options_fn);
    std::stringstream buffer;
    buffer << t.rdbuf();

    /* Parse the string */
    return this->set_command_string (buffer.str());
}

Shared_parms*
Registration_parms::get_shared_parms ()
{
    return d_ptr->shared;
}

void
Registration_parms::delete_all_stages ()
{
    d_ptr->delete_all_stages ();
    this->num_stages = 0;
}

std::list<Stage_parms*>& 
Registration_parms::get_stages ()
{
    return d_ptr->stages;
}

Stage_parms* 
Registration_parms::append_stage ()
{
    Stage_parms *sp;

    this->num_stages ++;
    if (this->num_stages == 1) {
        sp = new Stage_parms();
    } else {
        sp = new Stage_parms(*d_ptr->stages.back());
    }
    d_ptr->stages.push_back (sp);

    /* Some parameters that should be copied from global 
       to the first stage. */
    if (this->num_stages == 1) {
        sp->default_value = this->default_value;
    }

    sp->stage_no = this->num_stages;

    return sp;
}

Stage_parms* 
Registration_parms::append_process_stage ()
{
    Stage_parms *sp = this->append_stage ();

    Process_parms::Pointer pp = Process_parms::New ();
    sp->set_process_parms (pp);
    return sp;
}
