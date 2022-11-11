/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdio.h>
#include <stdlib.h>

#include "diff.h"
#include "itk_image_header_compare.h"
#include "itk_image_save.h"
#include "image_stats.h"
#include "pcmd_diff.h"
#include "plm_file_format.h"
#include "print_and_exit.h"
#include "xform.h"

class Diff_parms {
public:
    std::string in_fn_1;
    std::string in_fn_2;
    std::string out_fn;
};

void
diff_vf (Diff_parms* parms)
{
    Xform xf1, xf2;

    xf1.load (parms->in_fn_1);
    if (xf1.m_type != XFORM_ITK_VECTOR_FIELD) {
        print_and_exit ("Error: %s not loaded as a vector field\n",
            parms->in_fn_1.c_str());
    }

    xf2.load (parms->in_fn_2);
    if (xf2.m_type != XFORM_ITK_VECTOR_FIELD) {
        print_and_exit ("Error: %s not loaded as a vector field\n",
            parms->in_fn_2.c_str());
    }

    DeformationFieldType::Pointer vf1 = xf1.get_itk_vf();
    DeformationFieldType::Pointer vf2 = xf2.get_itk_vf();

    if (!itk_image_header_compare (vf1, vf2)) {
	print_and_exit ("Error: vector field sizes do not match\n");
    }

    DeformationFieldType::Pointer vf_diff = diff_vf (vf1, vf2);
    itk_image_save (vf_diff, parms->out_fn.c_str());
}

void
diff_image (Diff_parms* parms)
{
    Plm_image::Pointer img1, img2;

    img1 = plm_image_load_native (parms->in_fn_1);
    if (!img1) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    parms->in_fn_1.c_str());
    }
    img2 = plm_image_load_native (parms->in_fn_2);
    if (!img2) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    parms->in_fn_2.c_str());
    }

    if (!Plm_image::compare_headers (img1, img2)) {
	print_and_exit ("Error: image sizes do not match\n");
    }

    Plm_image::Pointer pi_diff = diff_image (img1, img2);
    pi_diff->save_image (parms->out_fn.c_str());
}

void
diff_main (Diff_parms* parms)
{
    Plm_file_format file_type_1, file_type_2;

    /* What is the input file type? */
    file_type_1 = plm_file_format_deduce (parms->in_fn_1);
    file_type_2 = plm_file_format_deduce (parms->in_fn_2);

    if (file_type_1 == PLM_FILE_FMT_VF 
	&& file_type_2 == PLM_FILE_FMT_VF)
    {
	diff_vf (parms);
    }
    else
    {
	diff_image (parms);
    }
}

static void
diff_print_usage (void)
{
    printf ("Usage: plastimatch diff image_in_1 image_in_2 image_out\n"
	    );
    exit (-1);
}

static void
diff_parse_args (Diff_parms* parms, int argc, char* argv[])
{
    if (argc != 5) {
	diff_print_usage ();
    }
    
    parms->in_fn_1 = argv[2];
    parms->in_fn_2 = argv[3];
    parms->out_fn = argv[4];
}

void
do_command_diff (int argc, char *argv[])
{
    Diff_parms parms;
    
    diff_parse_args (&parms, argc, argv);

    diff_main (&parms);
}
