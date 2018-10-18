/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <list>

#include "file_util.h"
#include "string_util.h"
#include "rpl_volume.h"
#include "rt_dij.h"
#include "volume_macros.h"

void 
Rt_dij::set_from_dose_rv (
    const plm_long ij[2],
    size_t energy_index,
    const Rpl_volume *dose_rv, 
    const Volume::Pointer& dose_vol)
{
    this->rows.push_back (Rt_dij_row (
            float (ij[0]), float (ij[1]), float (energy_index)));
    Rt_dij_row& rt_dij_row = this->rows.back ();

    const Volume *dose = dose_vol.get();
    plm_long ijk[3];
    double xyz[3];
    LOOP_Z (ijk, xyz, dose) {
        LOOP_Y (ijk, xyz, dose) {
            LOOP_X (ijk, xyz, dose) {
                plm_long idx = dose->index (ijk);
                float val = dose_rv->get_value (xyz);
                if (val > 0.f) {
                    rt_dij_row.dose.push_back (Rt_dij_dose (idx, val));
                }
            }
        }
    }
}

void 
Rt_dij::dump (const std::string& dir) const
{
    int i = 0;
    std::list<Rt_dij_row>::const_iterator r = rows.begin();
    while (r != rows.end()) {
        std::string fn = string_format ("%s/dij_%04d.txt", dir.c_str(), i++);
        FILE *fp = plm_fopen (fn, "w");
        fprintf (fp, "%f %f %f\n", r->xpos, r->ypos, r->energy);
        std::list<Rt_dij_dose>::const_iterator c = r->dose.begin();
        while (c != r->dose.end()) {
            fprintf (fp, "%lu %f\n", c->index, c->dose);
            c++;
        }
        fclose (fp);
        ++r;
    }
}
