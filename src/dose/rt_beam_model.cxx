/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"

#include "rt_beam_model.h"

class Rt_beam_model_private {
public:
    int i;
};

Rt_beam_model::Rt_beam_model ()
{
    d_ptr = new Rt_beam_model_private;
}

Rt_beam_model::~Rt_beam_model ()
{
    delete d_ptr;
}
