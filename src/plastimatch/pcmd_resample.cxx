/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkCastImageFilter.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"

#include "plmsys.h"

#include "itk_image.h"
#include "itk_image_load.h"
#include "itk_image_save.h"
#include "itk_resample.h"
#include "pcmd_resample.h"
#include "plm_clp.h"
#include "plm_file_format.h"
#include "plm_image_header.h"

/* Return true if geometry was deduced, else false */
static bool
deduce_geometry (Plm_image_header *pih, const Resample_parms* parms)
{
    /* use the spacing of user-supplied fixed image */
    if (parms->fixed_fn.not_empty()) {
	Plm_file_format ff = plm_file_format_deduce (
	    (const char*) parms->fixed_fn);
	if (ff == PLM_FILE_FMT_VF) {
	    DeformationFieldType::Pointer fixed	= itk_image_load_float_field (
		(const char*) parms->fixed_fn);
	    pih->set_from_itk_image (fixed);
	}
	else {
	    /* Hope for the best... */
	    FloatImageType::Pointer fixed = itk_image_load_float (
		(const char*) parms->fixed_fn, 0);
	    pih->set_from_itk_image (fixed);
	}
	return true;
    }
    /* use user specified geometry */
    else if (parms->m_have_dim && parms->m_have_origin 
	&& parms->m_have_spacing)
    {
	if (parms->m_have_direction_cosines) {
	    pih->set_from_gpuit (parms->dim, parms->origin, parms->spacing, 
		parms->m_dc);
	} else {
	    pih->set_from_gpuit (parms->dim, parms->origin, parms->spacing, 0);
	}
	return true;
    }

    /* else, we failed */
    return false;
}

template<class T>
T
do_resample_itk (Resample_parms* parms, T img)
{
    if (parms->m_have_subsample) {
	return subsample_image (img, parms->subsample[0], parms->subsample[1], 
	    parms->subsample[2], parms->default_val);
    }

    Plm_image_header pih;
    if (deduce_geometry (&pih, parms)) {
	/* Return resampled image */
	return resample_image (img, &pih, parms->default_val, 
	    parms->interp_lin);
    } else {
	/* Return original image */
	return img;
    }
}

void
resample_main_itk_vf (Resample_parms* parms)
{
    DeformationFieldType::Pointer vector_field 
	= itk_image_load_float_field ((const char*) parms->input_fn);

    if (parms->m_have_subsample) {
	print_and_exit ("Error. Subsample not supported for vector field.\n");
	exit (-1);
    }

    Plm_image_header pih;
    if (deduce_geometry (&pih, parms)) {
	/* Resample image */
	vector_field = vector_resample_image (vector_field, &pih);
    }
    itk_image_save (vector_field, (const char*) parms->output_fn);
}

void
resample_main (Resample_parms* parms)
{
    Plm_image plm_image;

    Plm_file_format file_format;

    file_format = plm_file_format_deduce ((const char*) parms->input_fn);

    /* Vector fields are templated differently, so do them separately */
    if (file_format == PLM_FILE_FMT_VF) {
	resample_main_itk_vf (parms);
	return;
    }

    plm_image.load_native ((const char*) parms->input_fn);

    if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
	parms->output_type = plm_image.m_type;
    }

    switch (plm_image.m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	plm_image.m_itk_uchar 
	    = do_resample_itk (parms, plm_image.m_itk_uchar);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	plm_image.m_itk_short 
	    = do_resample_itk (parms, plm_image.m_itk_short);
	break;
    case PLM_IMG_TYPE_ITK_LONG:
	plm_image.m_itk_int32 
	    = do_resample_itk (parms, plm_image.m_itk_int32);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	plm_image.m_itk_uint32 
	    = do_resample_itk (parms, plm_image.m_itk_uint32);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	plm_image.m_itk_float 
	    = do_resample_itk (parms, plm_image.m_itk_float);
	break;
    default:
	print_and_exit ("Unhandled image type in resample_main()\n");
	break;
    }

    plm_image.convert_and_save (
	(const char*) parms->output_fn, 
	parms->output_type);
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
    Resample_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files */
    parser->add_long_option ("", "input", 
	"input directory or filename; "
	"can be an image or vector field", 1, "");
    
    /* Output files */
    parser->add_long_option ("", "output", 
	"output image or vector field", 1, "");

    /* Output options */
    parser->add_long_option ("", "output-type", 
	"type of output image, one of {uchar, short, float, ...}", 1, "");

    /* Algorithm options */
    parser->add_long_option ("", "default-value", 
	"value to set for pixels with unknown value, default is 0", 1, "");
    parser->add_long_option ("", "interpolation", 
	"interpolation type, either \"nn\" or \"linear\", "
	"default is linear", 1, "linear");

    /* Geometry options */
    parser->add_long_option ("F", "fixed", 
	"fixed image (match output size to this image)", 1, "");
    parser->add_long_option ("", "origin", 
	"location of first image voxel in mm \"x y z\"", 1, "");
    parser->add_long_option ("", "dim", 
	"size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "spacing", 
	"voxel spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "direction-cosines", 
	"oriention of x, y, and z axes; Specify either preset value,"
	" {identity,rotated-{1,2,3},sheared},"
	" or 9 digit matrix string \"a b c d e f g h i\"", 1, "");
    parser->add_long_option ("", "subsample", 
	"bin voxels together at integer subsampling rate \"x [y z]\"", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an input file was given */
    if (!parser->option ("input"))
    {
	throw (dlib::error ("Error.  Please specify an input file "
		"using the --input option"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output"))
    {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Check that no extraneous options were given */
    if (parser->number_of_arguments() != 0) {
	std::string extra_arg = (*parser)[0];
	throw (dlib::error ("Error.  Unknown option " + extra_arg));
    }

    /* Input files */
    parms->input_fn = parser->get_string("input").c_str();

    /* Output files */
    parms->output_fn = parser->get_string("output").c_str();

    /* Output options */
    if (parser->option("output-type")) {
	std::string arg = parser->get_string ("output-type");
	parms->output_type = plm_image_type_parse (arg.c_str());
	if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
	    throw (dlib::error ("Error. Unknown --output-type argument: " 
		    + parser->get_string("output-type")));
	}
    }

    /* Algorithm options */
    if (parser->option("default-value")) {
	parms->default_val = parser->get_float("default-value");
    }
    std::string arg = parser->get_string ("interpolation");
    if (arg == "nn") {
	parms->interp_lin = 0;
    }
    else if (arg == "linear") {
	parms->interp_lin = 1;
    }
    else {
	throw (dlib::error ("Error. Unknown --interpolation argument: " 
		+ arg));
    }

    /* Geometry options */
    if (parser->option ("dim")) {
	parms->m_have_dim = 1;
	parser->assign_plm_long_13 (parms->dim, "dim");
    }
    if (parser->option ("origin")) {
	parms->m_have_origin = 1;
	parser->assign_float13 (parms->origin, "origin");
    }
    if (parser->option ("spacing")) {
	parms->m_have_spacing = 1;
	parser->assign_float13 (parms->spacing, "spacing");
    }
    if (parser->option ("subsample")) {
	parms->m_have_subsample = 1;
	parser->assign_int13 (parms->subsample, "subsample");
    }
    /* Direction cosines */
    if (parser->option ("direction-cosines")) {
	parms->m_have_direction_cosines = true;
	std::string arg = parser->get_string("direction-cosines");
	if (!parms->m_dc.set_from_string (arg)) {
	    throw (dlib::error ("Error parsing --direction-cosines "
		    "(should have nine numbers)\n"));
	}
    }

    parms->fixed_fn = parser->get_string("fixed").c_str();
}

void
do_command_resample (int argc, char *argv[])
{
    Resample_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    resample_main (&parms);
}
