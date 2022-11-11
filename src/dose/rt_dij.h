/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_dij_h_
#define _rt_dij_h_

#include "plmdose_config.h"
#include "smart_pointer.h"
#include "volume.h"

class Rpl_volume;
class Rt_dij_private;

class PLMDOSE_API Rt_dij_dose {
public:
    Rt_dij_dose (size_t index, float dose) : index(index), dose(dose) 
    {}
public:
    size_t index;
    float dose;
};

class PLMDOSE_API Rt_dij_row {
public:
    Rt_dij_row (
        float xpos, float ypos, float energy)
        : xpos(xpos), ypos(ypos), energy(energy)
    {}
public:
    float xpos;
    float ypos;
    float energy;
    std::list<Rt_dij_dose> dose;
};

class PLMDOSE_API Rt_dij {
public:
    std::list<Rt_dij_row> rows;
public:
    void set_from_dose_rv (
        const plm_long ij[2],
        size_t energy_index,
        const Rpl_volume *dose_rv, 
        const Volume::Pointer& dose_vol);
    void dump (const std::string& dir) const;
};

#endif
