/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "itk_point.h"
#include "itk_pointset.h"
#include "pcmd_warp.h"
#include "pointset.h"
#include "warp_parms.h"
#include "xform.h"

void
warp_pointset_main (Warp_parms* parms)
{
    Unlabeled_pointset ps;
    ps.load (parms->input_fn.c_str());

    FloatPointSetType::Pointer itk_ps_in 
	= itk_float_pointset_from_pointset (&ps);

    FloatPointSetType::Pointer itk_ps_out;
    if (parms->xf_in_fn != "") {
        Xform xf;
        xform_load (&xf, parms->xf_in_fn);
        itk_ps_out = itk_pointset_warp (itk_ps_in, &xf);
    } else {
        itk_ps_out = itk_ps_in;
    }

    if (parms->output_pointset_fn != "") {
	Unlabeled_pointset *ps_out =
	    unlabeled_pointset_from_itk_float_pointset (itk_ps_out);
	ps_out->save (parms->output_pointset_fn.c_str());
	delete ps_out;
    }
}
