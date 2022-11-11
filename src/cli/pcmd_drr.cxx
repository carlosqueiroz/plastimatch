/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdlib.h>

#include "drr.h"
#include "drr_options.h"
#include "pcmd_drr.h"
#include "plm_clp.h"
#include "string_util.h"

class Drr_parms {
public:
    std::string input_fn;
    std::string output_fn;
    std::string output_dicom;
};

static void
do_drr (Drr_options *drr_options)
{
    drr_compute (drr_options);
}

void
set_image_parms (Drr_options* options)
{
    if (!options->have_image_center) {
	options->image_center[0] = (options->detector_resolution[0]-1)/2.0;
	options->image_center[1] = (options->detector_resolution[1]-1)/2.0;
    }
    if (options->have_image_window) {
	options->image_window[0] = plm_max (0, options->image_window[0]);
	options->image_window[1] = plm_min (options->image_window[1],
            options->detector_resolution[0] - 1);
	options->image_window[2] = plm_max (0, options->image_window[2]);
	options->image_window[3] = plm_min (options->image_window[3],
            options->detector_resolution[1] - 1);
    } else {
	options->image_window[0] = 0;
	options->image_window[1] = options->detector_resolution[0] - 1;
	options->image_window[2] = 0;
	options->image_window[3] = options->detector_resolution[1] - 1;
    }
    options->image_resolution[0] = options->image_window[1]
	- options->image_window[0] + 1;
    options->image_resolution[1] = options->image_window[3]
	- options->image_window[2] + 1;
    if (options->have_angle_diff) {
	options->angle_diff *= (float) (M_TWOPI / 360.0);
    } else {
	options->angle_diff = M_TWOPI / options->num_angles;
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch drr [options] [infile]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Drr_options* options,
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    parser->add_long_option ("A", "threading",
	"Threading option {cpu,cuda,opencl} (default: cpu)", 1, "cpu");
    parser->add_long_option ("", "autoscale",
	"Automatically rescale intensity", 0);
    parser->add_long_option ("", "autoscale-range",
	"Range used for autoscale in form \"min max\" "
        "(default: \"0 255\")", 1, "0 255");
    parser->add_long_option ("a", "num-angles",
	"Generate this many images at equal gantry spacing", 1, "1");
    parser->add_long_option ("N", "gantry-angle-spacing",
	"Difference in gantry angle spacing in degrees", 1, "");
    parser->add_long_option ("y", "gantry-angle",
	"Gantry angle for image source in degrees", 1, "0");
    parser->add_long_option ("n", "nrm",
	"Normal vector of detector in format \"x y z\"", 1, "");
    parser->add_long_option ("", "vup",
	"The vector pointing from the detector center to the top row of "
        "the detector in format \"x y z\"", 1, "0 0 1");
    parser->add_long_option ("", "sad",
	"The SAD (source-axis-distance) in mm (default: 1000)", 1, "1000");
    parser->add_long_option ("", "sid",
	"The SID (source-image-distance) in mm (default: 1500)", 1, "1500");
    parser->add_long_option ("r", "dim",
	"The detector resolution in format \"row col\" (in mm)", 1, "128 128");
    parser->add_long_option ("s", "intensity-scale",
	"Scaling factor for output image intensity", 1, "1.0");
    parser->add_long_option ("e", "exponential",
	"Do exponential mapping of output values", 0);
    parser->add_long_option ("c", "image-center",
	"The image center in the format \"row col\", in pixels", 1, "");
    parser->add_long_option ("z", "detector-size",
	"The physical size of the detector in format \"row col\", in mm",
        1, "600 600");
    parser->add_long_option ("w", "subwindow",
	"Limit DRR output to a subwindow in format \"r1 r2 c1 c2\","
        "in pixels", 1, "");
    parser->add_long_option ("t", "output-format",
	"Select output format {pgm, pfm, raw}", 1, "pfm");
    parser->add_long_option ("S", "raytrace-details",
	"Create output file with complete ray trace details", 1, "");
    parser->add_long_option ("i", "algorithm",
	"Choose algorithm {exact,uniform}", 1, "exact");
    parser->add_long_option ("o", "isocenter", 
	"Isocenter position \"x y z\" in DICOM coordinates (mm)", 1, "0 0 0");
    parser->add_long_option ("G", "geometry-only",
	"Create geometry files only", 0);
    parser->add_long_option ("P", "hu-conversion",
	"Choose HU conversion type {preprocess,inline,none}", 1, "preprocess");
    parser->add_long_option ("I", "input", 
	"Input file", 1, "");
    parser->add_long_option ("O", "output",
	"Prefix for output file(s)", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that input was given */
    parser->check_required ("output");

    /* Insert command line options into options array */
    std::string s;
    s = make_lowercase (parser->get_string("threading"));
    if (s == "cpu" || s == "openmp") {
        options->threading = THREADING_CPU_OPENMP;
    } else if (s == "cuda" || s == "gpu") {
        options->threading = THREADING_CUDA;
    } else if (s == "opencl") {
        options->threading = THREADING_OPENCL;
    } else {
        throw (dlib::error ("Error.  Option \"threading\" should be one of "
                "{cpu,cuda,opencl}"));
    }

    options->num_angles = parser->get_int ("num-angles");
    if (parser->have_option ("gantry-angle-spacing")) {
        options->have_angle_diff = true;
        options->angle_diff = parser->get_float ("gantry-angle-spacing");
    }
    options->start_angle = parser->get_float ("gantry-angle");
    if (parser->have_option ("nrm")) {
        options->have_nrm = true;
        parser->assign_float_3 (options->nrm, "nrm");
    }
    parser->assign_float_3 (options->vup, "vup");
    options->sad = parser->get_float ("sad");
    options->sid = parser->get_float ("sid");
    parser->assign_int_2 (options->detector_resolution, "dim");
    options->manual_scale = parser->get_float ("intensity-scale");

    if (parser->have_option ("exponential")) {
        options->exponential_mapping = true;
    }
    if (parser->have_option ("image-center")) {
        options->have_image_center = true;
        parser->assign_float_2 (options->image_center, "image-center");
    }
    parser->assign_float_2 (options->image_size, "detector-size");
    if (parser->have_option ("subwindow")) {
        options->have_image_window = true;
        parser->assign_int_4 (options->image_window, "subwindow");
    }
    s = make_lowercase (parser->get_string("output-format"));
    if (s == "pfm") {
        options->output_format = OUTPUT_FORMAT_PFM;
    } else if (s == "pgm") {
        options->output_format = OUTPUT_FORMAT_PGM;
    } else if (s == "raw") {
        options->output_format = OUTPUT_FORMAT_RAW;
    }
    else {
        throw (dlib::error ("Error.  Option \"output-format\" should be one of "
                "{pfm,pgm,raw}"));
    }
    if (parser->have_option ("raytrace-details")) {
        options->output_details_prefix = parser->get_string("raytrace-details");
    }
    s = make_lowercase (parser->get_string("algorithm"));
    if (s == "exact") {
        options->algorithm = DRR_ALGORITHM_EXACT;
    } else if (s == "uniform") {
        options->algorithm = DRR_ALGORITHM_UNIFORM;
    } else {
        throw (dlib::error ("Error.  Option \"algorithm\" should be one of "
                "{exact,uniform}"));
    }
    parser->assign_float_3 (options->isocenter, "isocenter");
    options->geometry_only = parser->have_option ("geometry-only");
    s = make_lowercase (parser->get_string("hu-conversion"));
    if (s == "preprocess") {
        options->hu_conversion = PREPROCESS_CONVERSION;
    } else if (s == "inline") {
        options->hu_conversion = INLINE_CONVERSION;
    } else if (s == "none") {
        options->hu_conversion = NO_CONVERSION;
    } else {
        throw (dlib::error ("Error.  Option \"hu-conversion\" should be one of "
                "{preprocess,inline,none}"));
    }

    options->autoscale = parser->have_option ("autoscale");
    parser->assign_float_2 (options->autoscale_range, "autoscale-range");

    /* Verify that either an inputfile was given, or --geometry-only was 
       requested */
    if (parser->have_option ("input")) {
        options->input_file = parser->get_string("input");
    } else if (parser->number_of_arguments() == 1) {
        options->input_file = (*parser)[0];
    } else if (!options->geometry_only) {
        throw (dlib::error ("Error.  Specify either a single input file "
                "or option \"geometry-only\""));
    }

    options->output_prefix = parser->get_string("output");

    set_image_parms (options);
}

void
do_command_drr (int argc, char *argv[])
{
    Drr_options parms;
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);
    do_drr (&parms);
}
