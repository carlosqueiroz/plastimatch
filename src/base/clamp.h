/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _clamp_h_
#define _clamp_h_

#define CLAMP(value, min_value, max_value)        \
    do {                                          \
        if (value < min_value) {                  \
            value = min_value;                    \
        } else if (value > max_value) {           \
            value = max_value;                    \
        }                                         \
    } while (0)

#define CLAMP2(v1, v2, min_value, max_value)      \
    do {                                          \
        if (v1 < min_value) {                     \
            v1 = min_value;                       \
        }                                         \
        if (v2 > max_value) {                     \
            v2 = max_value;                       \
        }                                         \
        if (v2 < v1) {                            \
            v2 = v1;                              \
        }                                         \
    } while (0)

#endif
