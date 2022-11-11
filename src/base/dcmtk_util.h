/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_util_h_
#define _dcmtk_util_h_

#include "plmbase_config.h"
#include <vector>
#include "dcmtk/dcmdata/dcitem.h"
#include "dcmtk/dcmdata/dctag.h"
#include "dcmtk/ofstd/ofcond.h"

PLMBASE_API void
dcmtk_get_date_time (
    std::string *date,
    std::string *time
);

template<class T> PLMBASE_API OFCondition
dcmtk_put (DcmItem*, const DcmTag &, T);
PLMBASE_API OFCondition
dcmtk_put (DcmItem*, const DcmTag &, const std::string&);
PLMBASE_API OFCondition
dcmtk_put (DcmItem*, const DcmTag &, const char*);
PLMBASE_API OFCondition
dcmtk_get_ds_float (DcmItem* item, const DcmTag &tag, float *s);
PLMBASE_API OFCondition
dcmtk_get_ds_float2 (DcmItem* item, const DcmTag &tag, float *s);
PLMBASE_API OFCondition
dcmtk_get_ds_float3 (DcmItem* item, const DcmTag &tag, float *s);
PLMBASE_API OFCondition
dcmtk_get_ds_float_vec (DcmItem* item, const DcmTag &tag,
    std::vector<float> *f);

#endif
