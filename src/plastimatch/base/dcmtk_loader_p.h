/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_loader_p_h_
#define _dcmtk_loader_p_h_

#include "plmbase_config.h"
#include <map>
#include <string>
#include "dcmtk_series.h"
#include "dicom_rt_study.h"
#include "plm_image.h"
#include "rtss_structure_set.h"

class Dcmtk_rt_study;

/* Map from SeriesInstanceUID to Dcmtk_series */
typedef std::map<std::string, Dcmtk_series*> Dcmtk_series_map;
typedef std::pair<std::string, Dcmtk_series*> Dcmtk_series_map_pair;

class Dcmtk_loader_private {
public:
    Dicom_rt_study::Pointer m_drs;

    Plm_image::Pointer img;
    Plm_image::Pointer dose;
    Rtss_structure_set::Pointer cxt;

public:
    Dcmtk_series_map m_smap;

    Dcmtk_series *ds_image;
    Dcmtk_series *ds_rtss;
    Dcmtk_series *ds_rtdose;

public:
    Dcmtk_loader_private () {
    }
    ~Dcmtk_loader_private () {
        /* Delete Dicom_series objects in map */
        Dcmtk_series_map::iterator it;
        for (it = m_smap.begin(); it != m_smap.end(); ++it) {
            delete (*it).second;
        }

        /* Don't delete m_drs.  It belongs to caller. */
    }
};

#endif
