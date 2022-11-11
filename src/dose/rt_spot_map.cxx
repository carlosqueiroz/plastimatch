/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <list>
#include "rt_spot_map.h"

class Rt_spot_map_private {
public:
    std::list<Rt_spot> spot_list;
};

Rt_spot_map::Rt_spot_map ()
{
    d_ptr = new Rt_spot_map_private;
}

Rt_spot_map::~Rt_spot_map ()
{
    delete d_ptr;
}

void 
Rt_spot_map::add_spot (
    float xpos,
    float ypos,
    float energy,
    float sigma,
    float weight)
{
    d_ptr->spot_list.push_back (
        Rt_spot (xpos, ypos, energy, sigma, weight));
}

size_t
Rt_spot_map::num_spots () const
{
    return d_ptr->spot_list.size ();
}

const std::list<Rt_spot>&
Rt_spot_map::get_spot_list () const
{
    return d_ptr->spot_list;
}

void
Rt_spot_map::load_spot_map (const std::string& fn)
{
}

void
Rt_spot_map::save_spot_map (const std::string& fn) const
{
}
