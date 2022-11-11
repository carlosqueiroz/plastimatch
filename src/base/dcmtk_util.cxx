/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dcmtk_config.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_util.h"
#include "print_and_exit.h"
#include "string_util.h"

/* Workaround for older versions of DCMTK */
#if !DCMTK_HAS_EC_INVALIDVALUE
#define EC_InvalidValue EC_InvalidTag
#endif

void
dcmtk_get_date_time (
    std::string *current_date,
    std::string *current_time
)
{
    OFString date_string;
    OFString time_string;
    DcmDate::getCurrentDate (date_string);
    DcmTime::getCurrentTime (time_string);
    *current_date = date_string.c_str();
    *current_time = time_string.c_str();
    
    //*date = "20110101";
    //*time = "120000";
            
}

template<class T>
OFCondition
dcmtk_put (DcmItem* item, const DcmTag &tag, T t)
{
    std::string s;
    s = PLM_to_string (t);
    return item->putAndInsertString (tag, s.c_str());
}

OFCondition
dcmtk_put (DcmItem* item, const DcmTag &tag, const std::string& s)
{
    return item->putAndInsertString (tag, s.c_str());
}

OFCondition
dcmtk_put (DcmItem* item, const DcmTag &tag, const char* s)
{
    return item->putAndInsertString (tag, s);
}

OFCondition
dcmtk_get_ds_float (DcmItem* item, const DcmTag &tag, float *s)
{
    const char *c;
    OFCondition ofc = item->findAndGetString (tag, c);
    if (!ofc.good()) {
        return ofc;
    }

    int rc = sscanf (c, "%f", s);
    if (rc != 1) {
        return EC_InvalidValue;
    }

    return EC_Normal;
}

OFCondition
dcmtk_get_ds_float2 (DcmItem* item, const DcmTag &tag, float *s)
{
    const char *c;
    OFCondition ofc = item->findAndGetString (tag, c);
    if (!ofc.good()) {
        return ofc;
    }

    Plm_return_code rc = parse_dicom_float2 (s, c);
    if (rc == PLM_SUCCESS) {
        return EC_Normal;
    } else {
        return EC_InvalidValue;
    }
}

OFCondition
dcmtk_get_ds_float3 (DcmItem* item, const DcmTag &tag, float *s)
{
    const char *c;
    OFCondition ofc = item->findAndGetString (tag, c);
    if (!ofc.good()) {
        return ofc;
    }

    Plm_return_code rc = parse_dicom_float3 (s, c);
    if (rc == PLM_SUCCESS) {
        return EC_Normal;
    } else {
        return EC_InvalidValue;
    }
}

OFCondition
dcmtk_get_ds_float_vec (DcmItem* item, const DcmTag &tag,
    std::vector<float> *f)
{
    const char *c;
    OFCondition ofc = item->findAndGetString (tag, c);
    if (!ofc.good()) {
        return ofc;
    }

    *f = parse_dicom_float_vec (c);
    return EC_Normal;
}

template PLMBASE_API OFCondition dcmtk_put (DcmItem*, const DcmTag &, int);
template PLMBASE_API OFCondition dcmtk_put (DcmItem*, const DcmTag &, size_t);
template PLMBASE_API OFCondition dcmtk_put (DcmItem*, const DcmTag &, float);
