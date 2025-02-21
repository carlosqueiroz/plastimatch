/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdlib.h>

#include "bspline_xform.h"
#include "dcmtk_sro.h"
#include "geometry_chooser.h"
#include "logfile.h"
#include "pcmd_xf_convert.h"
#include "plm_clp.h"
#include "plm_image_header.h"
#include "print_and_exit.h"
#include "rt_study.h"
#include "rt_study_metadata.h"
#include "volume_header.h"
#include "xform.h"
#include "xform_convert.h"

class Xf_convert_parms {
public:
    std::string input_fn;
    std::string output_type;
    std::string output_fn;
    std::string output_dicom_dir;

    std::string fixed_fn;
    std::string fixed_image;
    std::string moving_image;
    std::string fixed_rcs;
    std::string moving_rcs;

    std::string series_description;

    bool dicom_filenames_with_uids;

    /* Geometry options */
    Geometry_chooser geometry_chooser;
    bool m_have_dim;
    bool m_have_origin;
    bool m_have_spacing;
    Volume_header m_vh;

    bool m_have_grid_spacing;
    float m_grid_spacing[3];

    Xform_convert xfc;
public:
    Xf_convert_parms () {
        dicom_filenames_with_uids = true;
        m_have_dim = false;
        m_have_origin = false;
        m_have_spacing = false;
        m_have_grid_spacing = false;
    }
};

void
set_output_xform_type (Xform_convert *xfc, const std::string& output_type)
{
    if (output_type == "vf") {
        xfc->m_xf_out_type = XFORM_ITK_VECTOR_FIELD;
    }
    else if (output_type == "bspline") {
        xfc->m_xf_out_type = XFORM_GPUIT_BSPLINE;
    }
    else if (output_type == "itk_bsp" || output_type == "itk_bspline") {
        xfc->m_xf_out_type = XFORM_ITK_BSPLINE;
    }
    else if (output_type == "affine") {
        xfc->m_xf_out_type = XFORM_ITK_AFFINE;
    }
    else if (output_type == "none") {
        xfc->m_xf_out_type = XFORM_NONE;
    }
    else {
        print_and_exit ("Sorry, can't convert output type\n");
    }
}

static void
do_xf_convert (Xf_convert_parms *parms)
{
    Xform_convert *xfc = &parms->xfc;

    /* Set up inputs */
    Xform::Pointer xf_in = xform_load (parms->input_fn);
    Xform::Pointer xf_out = Xform::New ();
    set_output_xform_type (xfc, parms->output_type);

    /* Set grid spacing as needed */
    xf_in->get_grid_spacing (xfc->m_grid_spac);
    if (parms->m_have_grid_spacing) {
        for (int d = 0; d < 3; d++) {
            xfc->m_grid_spac[d] = parms->m_grid_spacing[d];
        }
    }

    /* Set volume header as needed */
    if (parms->fixed_fn != "") {
        parms->geometry_chooser.set_fixed_image (parms->fixed_fn);
    } else if (xf_in->get_type() == XFORM_GPUIT_BSPLINE) {
        Bspline_xform* bxf = xf_in->get_gpuit_bsp();
        parms->geometry_chooser.set_fixed_image (
            bxf->get_plm_image_header());
    }
    if (parms->m_have_dim) {
        parms->geometry_chooser.set_dim (parms->m_vh.get_dim());
    }
    if (parms->m_have_origin) {
        parms->geometry_chooser.set_origin (parms->m_vh.get_origin());
    }
    if (parms->m_have_spacing) {
        parms->geometry_chooser.set_spacing (parms->m_vh.get_spacing());
    }
    xfc->set_geometry (parms->geometry_chooser.get_geometry());
    
    /* Do conversion */
    /* GCS FIX: This is not quite right.  Probably one should be 
       allowed to run xf-convert e.g. on a B-spline, and regrid 
       the b-spline, with the expectation that the xform 
       type will be read from the input file.  */
    if (xfc->m_xf_out_type == XFORM_NONE) {
        /* Copy input to output */
        xf_out = xf_in;
    } else {
        printf ("about to xform_convert\n");
        xfc->set_input_xform (xf_in);
        xfc->run ();
        xf_out = xfc->get_output_xform ();
        printf ("did xform_convert\n");
    }

    /* Save output file */
    if (!parms->output_fn.empty()) {
        xf_out->save (parms->output_fn);
    }

    /* Save output dicom */
    if (!parms->output_dicom_dir.empty()) {
        /* Load referenced image sets */
#if PLM_DCM_USE_DCMTK
        Rt_study_metadata::Pointer rtm_fixed;
        Rt_study_metadata::Pointer rtm_moving;

        /* Fixed image */
        if (!parms->fixed_image.empty()) {
            lprintf ("Loading fixed...\n");
            Rt_study::Pointer rtds = Rt_study::New ();
            rtds->load_image (parms->fixed_image);
            std::string fixed_path = parms->output_dicom_dir + "/fixed";
            rtds->save_dicom (fixed_path);
            rtm_fixed = rtds->get_rt_study_metadata();
        }
        else if (!parms->moving_rcs.empty()) {
            lprintf ("Loading fixed...\n");
            rtm_fixed = Rt_study_metadata::load (parms->fixed_rcs);
        }

        /* Moving image */
        if (!parms->moving_image.empty()) {
            lprintf ("Loading moving...\n");
            Rt_study::Pointer rtds = Rt_study::New ();
            rtds->load_image (parms->moving_image);
            std::string moving_path = parms->output_dicom_dir + "/moving";
            rtds->save_dicom (moving_path);
            rtm_moving = rtds->get_rt_study_metadata();
        }
        else if (!parms->fixed_rcs.empty()) {
            lprintf ("Loading moving...\n");
            rtm_moving = Rt_study_metadata::load (parms->moving_rcs);
        }

        lprintf ("Saving sro\n");
        Dcmtk_sro::save (
            xf_out, rtm_fixed, rtm_moving, parms->output_dicom_dir,
            parms->dicom_filenames_with_uids);
        lprintf ("Done saving sro\n");
#endif
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch xf-convert [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Xf_convert_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "input", 
        "Input xform filename (required)", 1, "");
    parser->add_long_option ("", "output", 
        "Output xform filename", 1, "");
    parser->add_long_option ("", "output-type", 
        "Type of xform to create; optional, choose from "
        "{bspline, itk_bspline, vf}", 1, "none");
    parser->add_long_option ("", "output-dicom", 
        "Directory for output of dicom spatial registration IOD, and "
        "optionally, fixed and/or moving images", 1, "");

    parser->add_long_option ("", "fixed",
        "Match vector field output size and geometry to this image", 1, "");
    parser->add_long_option ("", "dim", 
        "Size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "origin", 
        "Location of first image voxel in mm \"x y z\"", 1, "");
    parser->add_long_option ("", "spacing", 
        "Voxel spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "grid-spacing", 
        "B-spline grid spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "nobulk", 
        "Omit bulk transform for itk_bspline", 0);

    /* DICOM spatial registration */
    parser->add_long_option ("", "fixed-image",
        "Fixed image, to be converted to dicom", 1, "");
    parser->add_long_option ("", "moving-image",
        "Moving image, to be converted to dicom", 1, "");
    parser->add_long_option ("", "fixed-rcs", 
        "Directory containing DICOM image with reference "
        "coordinate system of fixed image", 
        1, "");
    parser->add_long_option ("", "moving-rcs", 
        "Directory containing DICOM image with reference "
        "coordinate system of moving image", 
        1, "");
    parser->add_long_option ("", "series-description",
        "DICOM Series Description",
        1, "");
    parser->add_long_option ("", "filenames-without-uids", 
        "create simple dicom filenames that don't include the uid", 0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Input files */
    parser->check_required ("input");
    parms->input_fn = parser->get_string("input");

    /* Output files */
    if (parser->option ("output")) {
        parms->output_fn = parser->get_string("output");
    }
    if (parser->option ("output-dicom")) {
        parms->output_dicom_dir = parser->get_string("output-dicom");
    }
    if (parms->output_dicom_dir.empty() && parms->output_fn.empty()) {
        throw (dlib::error ("Error.  You must specify either the --output "
                "or the --output-dicom option"));
    }

    /* DICOM spatial registration */
    parms->fixed_image = parser->get_string("fixed-image");
    parms->moving_image = parser->get_string("moving-image");
    parms->fixed_rcs = parser->get_string("fixed-rcs");
    parms->moving_rcs = parser->get_string("moving-rcs");
    if (parser->option ("filenames-without-uids")) {
        parms->dicom_filenames_with_uids = false;
    }

    Xform_convert *xfc = &parms->xfc;

    /* Copy values into output struct */
    parms->output_type = parser->get_string("output-type").c_str();

    /* Geometry options */
    parms->fixed_fn = parser->get_string("fixed");
    if (parser->option ("dim")) {
        parms->m_have_dim = true;
        parser->assign_plm_long_13 (parms->m_vh.get_dim(), "dim");
    }
    if (parser->option ("origin")) {
        parms->m_have_origin = true;
        parser->assign_float_13 (parms->m_vh.get_origin(), "origin");
    }
    if (parser->option ("spacing")) {
        parms->m_have_spacing = true;
        parser->assign_float_13 (parms->m_vh.get_spacing(), "spacing");
    }
    if (parser->option ("grid-spacing")) {
        parms->m_have_grid_spacing = true;
        parser->assign_float_13 (parms->m_grid_spacing, "grid-spacing");
    }

    if (parser->option ("nobulk")) {
        xfc->m_nobulk = true;
    }
}

void
do_command_xf_convert (int argc, char *argv[])
{
    Xf_convert_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_xf_convert (&parms);
}
