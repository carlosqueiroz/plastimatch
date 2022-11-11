/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "itk_intersect.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_file_format.h"
#include "xform.h"

class Intersect_parms {
public:
    std::vector<std::string> inputs_fn;
    std::string output_fn;
};

void
intersect_main (Intersect_parms *parms)
{
    /* Load first input and iterate over the others */
    Plm_image img_1 (parms->inputs_fn.at(0), PLM_IMG_TYPE_ITK_UCHAR);
    Plm_image temp (parms->inputs_fn.at(1), PLM_IMG_TYPE_ITK_UCHAR);
    UCharImageType::Pointer itk_out = itk_intersect (
                img_1.itk_uchar(), temp.itk_uchar());
    for (size_t i = 2; i < parms->inputs_fn.size(); ++i) {
            Plm_image temp (parms->inputs_fn.at(i), PLM_IMG_TYPE_ITK_UCHAR);
            /* Make the intersect */
            itk_out = itk_intersect (itk_out, temp.itk_uchar());
    }
    
    /* Save it */
    Plm_image img_out (itk_out);
    img_out.save_image (parms->output_fn);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] input_1 input_2 ...\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Intersect_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "output", 
        "filename for output image", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Copy input filenames to parms struct.  Two input files 
       must be specified. */
    if (parser->number_of_arguments() < 2) {
	throw (dlib::error ("Error.  You must specify at least two input files"));
    }
    for (size_t i = 0; i < parser->number_of_arguments(); ++i) {
            parms->inputs_fn.push_back((*parser)[i]);
    }

    /* Output files */
    parms->output_fn = parser->get_string("output").c_str();
}

void
do_command_intersect (int argc, char *argv[])
{
    Intersect_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    intersect_main (&parms);
}
