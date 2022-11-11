/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"

#include "diff.h"
#include "itk_image_header_compare.h"
#include "itk_image_type.h"
#include "mha_io.h"
#include "pcmd_compare.h"
#include "plm_clp.h"
#include "plm_file_format.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "print_and_exit.h"
#include "vf_stats.h"
#include "volume.h"
#include "xform.h"

class Compare_parms {
public:
    std::string input_1;
    std::string input_2;
    std::string mask_fn;
};

static void
vf_compare (Compare_parms* parms)
{
    Xform xf1, xf2;

    xf1.load (parms->input_1);
    if (xf1.m_type != XFORM_ITK_VECTOR_FIELD) {
        print_and_exit ("Error: %s not loaded as a vector field\n",
            parms->input_1.c_str());
    }

    xf2.load (parms->input_2);
    if (xf2.m_type != XFORM_ITK_VECTOR_FIELD) {
        print_and_exit ("Error: %s not loaded as a vector field\n",
            parms->input_2.c_str());
    }

    DeformationFieldType::Pointer vf1 = xf1.get_itk_vf();
    DeformationFieldType::Pointer vf2 = xf2.get_itk_vf();

    if (!itk_image_header_compare (vf1, vf2)) {
	print_and_exit ("Error: vector field sizes do not match\n");
    }

    DeformationFieldType::Pointer vf_diff = diff_vf (vf1, vf2);

    /* GCS FIX: This logic should be moved inside of xform class */
    Xform xf_diff;
    xf_diff.set_itk_vf (vf_diff);
    Plm_image_header pih;
    pih.set_from_itk_image (vf_diff);
    xform_to_gpuit_vf (&xf2, &xf_diff, &pih);
    Volume *vol = xf2.get_gpuit_vf().get();

    /* GCS FIX: This should be moved to base library; it is replicated in pcmd_stats */
    if (parms->mask_fn == "") {
    	vf_analyze (vol, 0);
    }
    else {
        Plm_image::Pointer pli = Plm_image::New (new Plm_image(parms->mask_fn));
        pli->convert (PLM_IMG_TYPE_GPUIT_UCHAR);
        Volume* mask = pli->get_vol();
	vf_analyze (vol, mask);
    }
}

static void
img_compare (Compare_parms* parms)
{
    Plm_image::Pointer img1, img2;

    img1 = plm_image_load_native (parms->input_1);
    if (!img1) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    parms->input_1.c_str());
    }
    img2 = plm_image_load_native (parms->input_2);
    if (!img2) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    parms->input_2.c_str());
    }

    if (!Plm_image::compare_headers (img1, img2)) {
	print_and_exit ("Error: image sizes do not match\n");
    }

    FloatImageType::Pointer fi1 = img1->itk_float ();
    FloatImageType::Pointer fi2 = img2->itk_float ();

    typedef itk::SubtractImageFilter< FloatImageType, FloatImageType, 
				      FloatImageType > SubtractFilterType;
    SubtractFilterType::Pointer sub_filter = SubtractFilterType::New();

    sub_filter->SetInput1 (fi1);
    sub_filter->SetInput2 (fi2);

    try {
	sub_filter->Update();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "ITK exception caught: " << excep << std::endl;
	exit (-1);
    }
    FloatImageType::Pointer diff = sub_filter->GetOutput ();

    typedef itk::ImageRegionConstIterator < FloatImageType > FloatIteratorType;
    FloatIteratorType it (diff, diff->GetRequestedRegion ());

    int first = 1;
    float min_val = 0.0;
    float max_val = 0.0;
    int num = 0, num_dif = 0;
    double ave = 0.0;
    double mae = 0.0;
    double mse = 0.0;

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	float v = it.Get();
	if (first) {
	    min_val = v;
	    max_val = v;
	    first = 0;
	}
	if (min_val > v)     min_val = v;
	if (max_val < v)     max_val = v;
	if (v != 0.0)        num_dif ++;
	ave += v;
	mae += fabs (v);
	mse += (v * v);
	num ++;
    }

    printf ("MIN %f AVE %f MAX %f\n"
	"MAE %f MSE %f\n"
	"DIF %d NUM %d\n",
	min_val, 
	(float) (ave / num), 
	max_val, 
	(float) (mae / num), 
	(float) (mse / num), 
	num_dif, 
	num);
}

static void
compare_main (Compare_parms* parms)
{
    Plm_file_format file_type_1, file_type_2;

    /* What is the input file type? */
    file_type_1 = plm_file_format_deduce (parms->input_1);
    file_type_2 = plm_file_format_deduce (parms->input_2);

    if (file_type_1 == PLM_FILE_FMT_VF 
	&& file_type_2 == PLM_FILE_FMT_VF)
    {
	vf_compare (parms);
    }
    else
    {
	img_compare (parms);
    }
}

static void
compare_print_usage (void)
{
    printf ("Usage: plastimatch compare file_1 file_2\n"
	    );
    exit (-1);
}

static void
compare_parse_args (Compare_parms* parms, int argc, char* argv[])
{
    if (argc != 4) {
	compare_print_usage ();
    }
    
    parms->input_1 = argv[2];
    parms->input_2 = argv[3];
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] input_file input_file\n", 
        argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}
static void
parse_fn (
    Compare_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "mask", "Compare inputs within this ROI", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that two input files were specified */
    if (parser->number_of_arguments() != 2) {
	throw (dlib::error ("Error.  You must specify at two input files."));
    }

    /* Copy input filenames to parms struct */
    parms->input_1 = (*parser)[0];
    parms->input_2 = (*parser)[1];

    /* ROI */
    parms->mask_fn = parser->get_string("mask");
}

void
do_command_compare (int argc, char *argv[])
{
    Compare_parms parms;
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    compare_main (&parms);
}
