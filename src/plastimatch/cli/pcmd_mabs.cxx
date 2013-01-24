/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "mabs.h"
#include "mabs_parms.h"
#include "pcmd_mabs.h"
#include "plm_clp.h"
#include "pstring.h"

class Mabs_parms_pcmd {
public:
    Pstring cmd_file_fn;
    bool prep;
    bool train;
    bool train_registration;
public:
    Mabs_parms_pcmd () {
        prep = false;
        train = false;
        train_registration = false;
    }
};

static void
usage_fn (dlib::Plm_clp *parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch mabs [options] command_file\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Mabs_parms_pcmd *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char *argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Parameters */
    parser->add_long_option ("", "prep", 
        "pre-process atlas", 0);
    parser->add_long_option ("", "train", 
        "perform full training to find the best registration "
        " and segmentation parameters", 0);
    parser->add_long_option ("", "train-registration", 
        "perform limited training to find the best registration "
        "parameters only", 0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that only one argument was given */
    if (parser->number_of_arguments() != 1) {
	throw (dlib::error (
                "Error.  Only one configuration file can be used."));
    }

    /* Get filename of command file */
    parms->cmd_file_fn = (*parser)[0].c_str();

    /* Parameters */
    if (parser->have_option ("prep")) {
        parms->prep = true;
    }
    if (parser->have_option ("train")) {
        parms->train = true;
    }
    if (parser->have_option ("train-registration")) {
        parms->train_registration = true;
    }
}

void
do_command_mabs (int argc, char *argv[])
{
    Mabs_parms_pcmd parms;
    
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    Mabs_parms mabs_parms;
    mabs_parms.parse_config (parms.cmd_file_fn.c_str());

    Mabs mabs;
    mabs.set_parms (&mabs_parms);
    if (parms.prep) {
        mabs.atlas_prep ();
    }
    else if (parms.train) {
        mabs.train ();
    }
    else if (parms.train_registration) {
        mabs.train_registration ();
    }
    else {
        mabs.run ();
    }
}
