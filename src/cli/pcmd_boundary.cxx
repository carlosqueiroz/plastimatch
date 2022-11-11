/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <list>

#include "image_boundary.h"
#include "itk_image_load.h"
#include "itk_image_save.h"
#include "plm_clp.h"

class Boundary_parms {
public:
    Volume_boundary_behavior vbb;
    Volume_boundary_type vbt;
    std::string output_fn;
    std::string input_fn;
    Boundary_parms () {
        vbb = ADAPTIVE_PADDING;
        vbt = INTERIOR_EDGE;
    }
};

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] input_file\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Boundary_parms* parms, 
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

    /* Algorithm options */
    parser->add_long_option ("", "boundary-behavior",
        "algorithm behavior at the image boundary: {zero-pad, edge-pad,"
        " adaptive}, default is adaptive; specify zero-pad if voxels"
        " outside image are zero, edge-pad if voxels outside image"
        " are equal to closest edge voxel, adaptive for zero-pad"
        " except for dimensions of a single slice",
        1, "adaptive");
    parser->add_long_option ("", "boundary-type",
        "algorithm behavior controlling boundary detection: {interior-edge,"
        " interior-face}, default is interior-edge; specify interior-edge"
        " to create an image that has value 1 for segment boundary voxels "
        " or interior-face to create an image that "
        " encodes the presence of face boundaries for segment boundary voxels",
        1, "interior-edge");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Check that one, and only one, input file was given */
    if (parser->number_of_arguments() == 0) {
	throw (dlib::error ("Error.  You must specify an input file"));
	
    } else if (parser->number_of_arguments() > 1) {
	std::string extra_arg = (*parser)[1];
	throw (dlib::error ("Error.  Unknown option " + extra_arg));
    }

    /* Copy input filenames to parms struct */
    parms->input_fn = (*parser)[0];

    /* Output files */
    parms->output_fn = parser->get_string("output");

    /* Algorithm options */
    parms->vbb = volume_boundary_behavior_parse(parser->get_string(
            "boundary-behavior"));
    parms->vbt = volume_boundary_type_parse(parser->get_string(
            "boundary-type"));
}

void
do_command_boundary (int argc, char *argv[])
{
    Boundary_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    /* Load input image */
    UCharImageType::Pointer input_image = itk_image_load_uchar (
        parms.input_fn, 0);

    /* Find image boundary */
    Image_boundary ib;
    ib.set_input_image (input_image);
    ib.set_volume_boundary_type (parms.vbt);
    ib.set_volume_boundary_behavior (parms.vbb);
    ib.run ();
    
    /* Save output image */
    UCharImageType::Pointer output_image = ib.get_output_image ();
    itk_image_save (output_image, parms.output_fn);
}
