/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plastimatch_startup.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmjpeg/djdecode.h"
#include "dcmtk/dcmjpls/djdecode.h"
#include "dcmtk/dcmdata/dcrledrg.h"

void
plastimatch_startup ()
{
    /* Initialize DCMTK codecs */
    DJDecoderRegistration::registerCodecs();
    DcmRLEDecoderRegistration::registerCodecs();
}
