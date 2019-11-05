/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "dice_statistics.h"
#include "hausdorff_distance.h"
#include "itk_bbox.h"
#include "itk_crop.h"
#include "itk_image_load.h"
#include "itk_resample.h"
#include "plm_clp.h"
#include "plm_image_header.h"

class Pcmd_dice_parms {
public:
    bool commands_were_requested;
    bool have_dice_option;
    bool have_hausdorff_option;
    std::string reference_image_fn;
    std::string test_image_fn;
    float z_crop_low;
    float z_crop_high;
public:
    Pcmd_dice_parms () {
        commands_were_requested = false;
        have_dice_option = false;
        have_hausdorff_option = false;
	z_crop_low = 0.0;
	z_crop_high = 0.0;
    }
};

static void
usage_fn (dlib::Plm_clp *parser, int argc, char *argv[])
{
    std::cout << 
        "Usage: plastimatch dice [options] reference-image test-image\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Pcmd_dice_parms *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char *argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Commands to execute */
    parser->add_long_option ("", "all", 
        "Compute both Dice and Hausdorff distances (equivalent"
        " to --dice --hausdorff)", 0);
    parser->add_long_option ("", "dice", 
        "Compute Dice coefficient (default)", 0);
    parser->add_long_option ("", "hausdorff", 
        "Compute Hausdorff distances (max, average, boundary, etc.)", 0);
    parser->add_long_option("","z_crop_low","crop the reference and the test image in the negative z dimension by this value (float)",1,"0.0");
    parser->add_long_option("","z_crop_high","crop the reference and the test image in the positive z dimension by this value (float)",1,"0.0");
    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    if (parser->option("dice")) {
        parms->commands_were_requested = true;
        parms->have_dice_option = true;
    }
    if (parser->option("hausdorff")) {
        parms->commands_were_requested = true;
        parms->have_hausdorff_option = true;
    }
    if (parser->option("all")) {
        parms->commands_were_requested = true;
        parms->have_dice_option = true;
        parms->have_hausdorff_option = true;
    }

    if (parser->option("z_crop_low")) {
	    parms->z_crop_low = parser->get_float("z_crop_low");
    }
    if (parser->option("z_crop_high")) {
	    parms->z_crop_high = parser->get_float("z_crop_high");
    }
    if (!parms->commands_were_requested) {
        parms->have_dice_option = true;
    }
    /* Check that two input files were given */
    if (parser->number_of_arguments() < 2) {
	throw (dlib::error ("Error.  You must specify two input files"));
	
    }   else if (parser->number_of_arguments() > 2) {
	    std::string extra_arg = (*parser)[1];
	    throw (dlib::error ("Error.  Extra argument " + extra_arg));
    }


    /* Copy values into output struct */
    parms->reference_image_fn = (*parser)[0];
    parms->test_image_fn = (*parser)[1];
}

void
do_command_dice (int argc, char *argv[])
{
    Pcmd_dice_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    UCharImageType::Pointer image_1 = itk_image_load_uchar (
        parms.reference_image_fn, 0);
    UCharImageType::Pointer image_2 = itk_image_load_uchar (
        parms.test_image_fn, 0);
    int bbox_indices_1[6];
    int bbox_indices_2[6];
    float bbox_coordinates_1[6];
    float bbox_coordinates_2[6];
    itk_bbox(image_1,bbox_coordinates_1,bbox_indices_1);
    itk_bbox(image_2,bbox_coordinates_2,bbox_indices_2);
    /* Apply z-crop to the coordinates, assuming the array is [x-min,x-max,y-min,y-max,z-min,z-max] */
    bbox_coordinates_1[4] -= parms.z_crop_low;
    bbox_coordinates_2[4] -= parms.z_crop_low;
    bbox_coordinates_1[5] += parms.z_crop_high;
    bbox_coordinates_2[5] += parms.z_crop_high;
    
    image_1 = itk_crop_by_coord(image_1,bbox_coordinates_1);
    image_2 = itk_crop_by_coord(image_2,bbox_coordinates_2);
    
    if (parms.have_dice_option) {
        Dice_statistics ds;
        ds.set_reference_image (image_1);
        ds.set_compare_image (image_2);
        ds.run ();
        ds.debug ();
    }
    if (parms.have_hausdorff_option) {
        do_hausdorff (image_1, image_2);
    }
}
