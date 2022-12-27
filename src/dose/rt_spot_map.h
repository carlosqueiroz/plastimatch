/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_spot_map_h_
#define _rt_spot_map_h_

#include "plmdose_config.h"
#include <string>
#include "smart_pointer.h"

class Rt_spot_map_private;

class PLMDOSE_API Rt_spot {
public:
    Rt_spot (
        float xpos, float ypos, float energy, float sigma, float weight)
        : xpos(xpos), ypos(ypos), energy(energy), sigma(sigma), weight(weight)
    {}
public:
    float xpos;
    float ypos;
    float energy;
    float sigma;
    float weight;
};

class PLMDOSE_API Rt_spot_map {
public:
    SMART_POINTER_SUPPORT (Rt_spot_map);
    Rt_spot_map_private *d_ptr;
public:
    Rt_spot_map ();
    ~Rt_spot_map ();
public:
    void add_spot (
        float xpos, float ypos, float energy, float sigma, float weight);
    size_t num_spots () const;
    const std::list<Rt_spot>& get_spot_list () const;

    void load_spot_map (const std::string& fn);
    void save_spot_map (const std::string& fn) const;
};

#endif
