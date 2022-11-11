/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_limit_h_
#define _volume_limit_h_

#include "plmbase_config.h"

enum point_location {
    POINTLOC_LEFT,
    POINTLOC_INSIDE,
    POINTLOC_RIGHT,
};
typedef enum point_location Point_location;

class PLMBASE_API Volume_limit {
public:
    /* upper and lower limits of volume, including tolerances */
    double lower_limit[3];
    double upper_limit[3];

    /* dir == 0 if lower_limit corresponds to lower index */
    int dir[3];
public:
    int clip_ray (
        double *ip1,                /* OUTPUT: Intersection point 1 */
        double *ip2,                /* OUTPUT: Intersection point 2 */
        const double *p1,           /* INPUT:  Starting point of ray */
        const double *ray           /* INPUT:  Direction of ray */
    );
    int clip_segment (
        double *ip1,                /* OUTPUT: Intersection point 1 */
        double *ip2,                /* OUTPUT: Intersection point 2 */
        const double *p1,           /* INPUT:  Line segment point 1 */
        const double *p2            /* INPUT:  Line segment point 2 */
    );
    void find_limit (const Volume *vol);
    void find_limit (const Volume::Pointer& vol);
    Point_location test_boundary (int d, double x);
    void print ();
};

#endif
