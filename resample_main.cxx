/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "resample_main.h"
#include "itk_image.h"
#include "plm_file_format.h"
#include "resample_mha.h"
#include "getopt.h"

void
print_usage (void)
{
    printf ("Usage: plastimatch resample [options]\n"
	    "Required:   --input=file\n"
	    "            --output=file\n"
	    "Optional:   --subsample=\"x y z\"\n"
	    "            --origin=\"x y z\"\n"
	    "            --spacing=\"x y z\"\n"
	    "            --size=\"x y z\"\n"
	    "            --output_type={uchar,short,ushort,float,vf}\n"
	    "            --interpolation={nn, linear}\n"
	    "            --default_val=val\n");
    exit (1);
}

void
show_stats (ShortImageType::Pointer image)
{
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    const ShortImageType::IndexType& st = region.GetIndex();
    const ShortImageType::SizeType& sz = image->GetLargestPossibleRegion().GetSize();
    const ShortImageType::PointType& og = image->GetOrigin();
    const ShortImageType::SpacingType& sp = image->GetSpacing();

    printf ("Origin = %g %g %g\n", og[0], og[1], og[2]);
    printf ("Spacing = %g %g %g\n", sp[0], sp[1], sp[2]);
    std::cout << "Start = " << st[0] << " " << st[1] << " " << st[2] << std::endl;
    std::cout << "Size = " << sz[0] << " " << sz[1] << " " << sz[2] << std::endl;
}

void
shift_pet_values (FloatImageType::Pointer image)
{
    printf ("Shifting values for pet...\n");
    typedef itk::ImageRegionIterator< FloatImageType > IteratorType;
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	float f = it.Get();
	f = f - 100;
	if (f < 0) {
	    f = f * 10;
	} else if (f > 0x4000) {
	    f = 0x4000 + (f - 0x4000) / 2;
	}
	if (f > 0x07FFF) {
	    f = 0x07FFF;
	}
	it.Set(f);
    }
}

void
fix_invalid_pixels_with_shift (ShortImageType::Pointer image)
{
    typedef itk::ImageRegionIterator< ShortImageType > IteratorType;
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	short c = it.Get();
	if (c < -1000) {
	    c = -1000;
	}
	it.Set (c + 1000);
    }
}

void
fix_invalid_pixels (ShortImageType::Pointer image)
{
    typedef itk::ImageRegionIterator< ShortImageType > IteratorType;
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	short c = it.Get();
	if (c < -1000) {
	    it.Set (-1000);
	}
    }
}

static void
parse_args (Resample_parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "output",         required_argument,      NULL,           1 },
	{ "output_type",    required_argument,      NULL,           2 },
	{ "output-type",    required_argument,      NULL,           2 },
	{ "input",          required_argument,      NULL,           3 },
	{ "subsample",      required_argument,      NULL,           5 },
	{ "origin",         required_argument,      NULL,           6 },
	{ "spacing",        required_argument,      NULL,           7 },
	{ "size",           required_argument,      NULL,           8 },
	{ "interpolation",  required_argument,      NULL,           9 },
	{ "default_val",    required_argument,      NULL,           10 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->mha_out_fn, optarg, _MAX_PATH);
	    break;
	case 2:
	    parms->output_type = plm_image_type_parse (optarg);
	    if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
		print_usage();
	    }
	    break;
	case 3:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    break;
	case 5:
	    rc = sscanf (optarg, "%d %d %d", &(parms->subsample[0]), 
			 &(parms->subsample[1]), &(parms->subsample[2]));
	    if (rc != 3) {
		printf ("Subsampling option must have three arguments\n");
		exit (1);
	    }
	    parms->have_subsample = 1;
	    break;
	case 6:
	    rc = sscanf (optarg, "%g %g %g", &(parms->origin[0]), 
			 &(parms->origin[1]), &(parms->origin[2]));
	    if (rc != 3) {
		printf ("Origin option must have three arguments\n");
		exit (1);
	    }
	    parms->have_origin = 1;
	    break;
	case 7:
	    rc = sscanf (optarg, "%g %g %g", &(parms->spacing[0]), 
			 &(parms->spacing[1]), &(parms->spacing[2]));
	    if (rc != 3) {
		printf ("Spacing option must have three arguments\n");
		exit (1);
	    }
	    parms->have_spacing = 1;
	    break;
	case 8:
	    rc = sscanf (optarg, "%d %d %d", &(parms->size[0]), 
			 &(parms->size[1]), &(parms->size[2]));
	    if (rc != 3) {
		printf ("Size option must have three arguments\n");
		exit (1);
	    }
	    parms->have_size = 1;
	    break;
	case 9:
	    if (!strcmp (optarg, "nn")) {
		parms->interp_lin = 0;
	    } else if (!strcmp (optarg, "linear")) {
		parms->interp_lin = 1;
	    } else {
		fprintf (stderr, 
			 "Interpolation must be either nn or linear.\n");
		print_usage ();
	    }
	    break;
	case 10:
	    rc = sscanf (optarg, "%g", &(parms->default_val));
	    if (rc != 1) {
		printf ("Default value option must have one arguments\n");
		exit (1);
	    }
	    parms->have_default_val = 1;
	    break;
	default:
	    break;
	}
    }
    if (!parms->mha_in_fn[0] || !parms->mha_out_fn[0]) {
	printf ("Error: must specify --input and --output\n");
	print_usage();
    }
}

template<class T>
T
do_resample_itk (Resample_parms* parms, T img)
{
    if (parms->have_subsample) {
	return subsample_image (img, parms->subsample[0], parms->subsample[1], 
			       parms->subsample[2], parms->default_val);
    }
    else if (parms->have_origin && parms->have_spacing && parms->have_size) {
	return resample_image (img, parms->origin, parms->spacing, 
			      parms->size, parms->default_val, 
			      parms->interp_lin);
    } else {
	/* Do nothing */
	return img;
    }
}

void
resample_main_itk_vf (Resample_parms* parms)
{
    DeformationFieldType::Pointer input_field 
	    = load_float_field (parms->mha_in_fn);

    if (parms->have_subsample) {
	print_and_exit ("Error. Subsample not supported for vector field.\n");
	exit (-1);
    }
    else if (parms->have_origin && parms->have_spacing && parms->have_size) {
	printf ("Resampling...\n");
	input_field = vector_resample_image (input_field, parms->origin, 
					     parms->spacing, parms->size);
    }
    itk_image_save (input_field, parms->mha_out_fn);
}

void
resample_main (Resample_parms* parms)
{
    PlmImage plm_image;

    Plm_file_format file_format;

    file_format = plm_file_format_deduce (parms->mha_in_fn);

    /* Vector fields are templated differently, so do them separately */
    if (file_format == PLM_FILE_FMT_VF) {
	resample_main_itk_vf (parms);
	return;
    }

    plm_image.load_native (parms->mha_in_fn);

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

    plm_image.convert_and_save (parms->mha_out_fn, parms->output_type);
}

void
do_command_resample (int argc, char *argv[])
{
    Resample_parms parms;

    parse_args (&parms, argc, argv);

    resample_main (&parms);
}
