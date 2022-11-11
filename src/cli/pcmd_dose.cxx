/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdlib.h>

#include "logfile.h"
#include "pcmd_dose.h"
#include "plan_calc.h"
#include "plm_clp.h"
#include "plm_return_code.h"
#include "plm_timer.h"

class Dose_parms {
public:
    std::string command_fn;
    std::string input_fn;
    std::string beam_model_fn;
    std::string output_dose_image_fn;
    std::string debug_dir;
    std::string output_dicom;
};

static void
do_dose (Dose_parms *parms)
{
    Plm_timer timer;
    Plan_calc plan_calc;
    timer.start ();
    if (parms->command_fn != ""
        && plan_calc.load_command_file (parms->command_fn.c_str()) != PLM_SUCCESS)
    {
        lprintf ("Error parsing command file.\n");
        return;
    }
    if (parms->beam_model_fn != ""
        && plan_calc.load_beam_model (parms->beam_model_fn.c_str()) != PLM_SUCCESS)
    {
        lprintf ("Error parsing beam model file.\n");
        return;
    }
    if (parms->input_fn != ""
        && plan_calc.load_dicom_plan (parms->input_fn.c_str()) != PLM_SUCCESS)
    {
        lprintf ("Error parsing input dicom files.\n");
        return;
    }
    if (parms->output_dose_image_fn != "") {
        plan_calc.set_output_dose_fn (parms->output_dose_image_fn);
    }
    if (parms->output_dicom != "") {
        plan_calc.set_output_dicom (parms->output_dicom);
    }
    if (parms->debug_dir != "") {
        plan_calc.set_debug_directory (parms->debug_dir);
    }
    plan_calc.compute_plan ();
    lprintf ("Total execution time : %f seconds.\n", timer.report ());
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch dose [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Dose_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input and output filenames */
    parser->add_long_option ("", "beam-model", 
	"Input beam model filename", 1, "");
    parser->add_long_option ("", "command-file", 
	"Input command file filename", 1, "");
    parser->add_long_option ("", "debug", 
	"Directory for saving debugging files", 1, "");
    parser->add_long_option ("", "input", 
	"Input dicom directory containing CT and RTPLAN", 1, "");
    parser->add_long_option ("", "output-dose-img", 
	"Output dose as image with filename", 1, "");
    parser->add_long_option ("", "output-dicom", 
	"Output dicom directory (for writing dose)", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an input file was given */
    if (!parser->option("input") && !parser->option("command-file")) {
        throw dlib::error (
            "Error, you must specify either --input or --command-file.\n"
        );
    }

    /* Copy values into output struct */
    parms->beam_model_fn = parser->get_string("beam-model");
    parms->command_fn = parser->get_string("command-file");
    parms->debug_dir = parser->get_string("debug");
    parms->input_fn = parser->get_string("input");
    parms->output_dose_image_fn = parser->get_string("output-dose-img");
    parms->output_dicom = parser->get_string("output-dicom");
}

void
do_command_dose (int argc, char *argv[])
{
    Dose_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_dose (&parms);
}
