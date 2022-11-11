/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <string>

#include "itk_crop.h"
#include "pcmd_crop.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "print_and_exit.h"

class Crop_parms {
public:
    std::string img_in_fn;
    std::string structure_in_fn;
    std::string img_out_fn;
    bool have_coordinates;
    bool have_vox;
    bool have_zcrop;
    int crop_vox[6];
    float crop_coords[6];
    float zcrop[2];
public:
    Crop_parms () {
        img_in_fn = "";
        img_out_fn = "";
        for (int i = 0; i < 2; i++) {
            zcrop[i] = 0.f;
        }
        for (int i = 0; i < 6; i++) {
            crop_vox[i] = 0;
            crop_coords[i] = 0.f;
        }
        have_coordinates = false;
        have_vox = false;
        have_zcrop = false;
    }
};

template <class T>
void
crop_image (Plm_image& plm_image, T& image,
    const Plm_image& structure_image, const Crop_parms* parms)
{
    if (parms->have_coordinates) {
        plm_image.set_itk (itk_crop_by_coord (image, parms->crop_coords));
    } else if (parms->have_vox) {
        plm_image.set_itk (itk_crop_by_index (image, parms->crop_vox));
    } else if (parms->structure_in_fn != "") {
        plm_image.set_itk (itk_crop_by_image (
                image, structure_image.m_itk_uchar));
    }
}

static void
crop_main (Crop_parms* parms)
{
    Plm_image plm_image;
    Plm_image structure_image;

    plm_image.load_native (parms->img_in_fn);
    if (parms->structure_in_fn != "") {
        structure_image.load (parms->structure_in_fn, PLM_IMG_TYPE_ITK_UCHAR);
    }

    switch (plm_image.m_type) {
    case PLM_IMG_TYPE_ITK_CHAR:
        crop_image (plm_image, plm_image.m_itk_char, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_UCHAR:
        crop_image (plm_image, plm_image.m_itk_uchar, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
        crop_image (plm_image, plm_image.m_itk_short, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
        crop_image (plm_image, plm_image.m_itk_ushort, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_LONG:
        crop_image (plm_image, plm_image.m_itk_int32, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
        crop_image (plm_image, plm_image.m_itk_uint32, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
        crop_image (plm_image, plm_image.m_itk_float, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
        crop_image (plm_image, plm_image.m_itk_double, structure_image, parms);
	break;
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
        crop_image (plm_image, plm_image.m_itk_uchar_vec,
            structure_image, parms);
	break;
    default:
	print_and_exit ("Unhandled image type in crop_main()\n");
	break;
    }
    
    plm_image.convert_and_save (parms->img_out_fn, plm_image.m_type);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options]\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Crop_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files */
    parser->add_long_option ("", "input", 
        "input directory or filename", 1, "");
    parser->add_long_option ("", "structure", 
        "filename of segmentation image; the input image "
        "will be cropped to the bounding box of the structure", 1, "");

    /* Output files */
    parser->add_long_option ("", "output", 
        "output image", 1, "");

    /* Adjustment string */
    parser->add_long_option ("", "voxels", 
        "a string that specifies the voxel indices of the six corners "
        "of the region to be cropped, in the form "
        "\"x1 x2 y1 y2 z1 z2\"", 
        1, "");

    /* Adjustment string */
    parser->add_long_option ("", "coordinates", 
        "a string that specifies the coordinates of the six corners "
        "of the region to be cropped, in the form "
        "\"x1 x2 y1 y2 z1 z2\"", 
        1, "");

    /* Adjustment string */
    parser->add_long_option ("", "zcrop",
        "crop the superior and inferior z directions by an additional "
        "amount, specified in the form \"zsup zinf\"", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("input")) {
	throw (dlib::error ("Error.  Please specify an input file "
		"using the --input option"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Input files */
    parms->img_in_fn = parser->get_string("input").c_str();
    parms->structure_in_fn = parser->get_string("structure").c_str();

    /* Output files */
    parms->img_out_fn = parser->get_string("output").c_str();

    /* Voxels option */
    int cropping_options_specified = 0;
    if (parser->option("voxels")) {
        parms->have_vox = true;
        parser->assign_int_6 (parms->crop_vox, "voxels");
        cropping_options_specified++;
    }
    
    /* Coordinates option */
    if (parser->option("coordinates")) {
        parms->have_coordinates = true;
        parser->assign_float_6 (parms->crop_coords, "coordinates");
        cropping_options_specified++;
    }

    if (parser->option("structure")) {
        cropping_options_specified++;
    }

    /* Zcrop option */
    if (parser->option("zcrop")) {
        parms->have_zcrop = true;
        parser->assign_float_2 (parms->zcrop, "zcrop");
    }

    if (cropping_options_specified != 1) {
	throw (dlib::error ("Error.  Please specify the crop region "
		"using one and only one of the options "
                "--coordinates, --structure, or --voxels"));
    }
}

void
do_command_crop (int argc, char *argv[])
{
    Crop_parms parms;
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    crop_main (&parms);
}
