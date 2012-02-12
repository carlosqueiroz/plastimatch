/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_endian_h_
#define _plm_endian_h_

#include "plm_config.h"

gpuit_EXPORT
void 
endian2_big_to_native (void* buf, unsigned long len);

gpuit_EXPORT
void
endian2_native_to_big (void* buf, unsigned long len);

gpuit_EXPORT
void
endian2_little_to_native (void* buf, unsigned long len);

gpuit_EXPORT
void
endian2_native_to_little (void* buf, unsigned long len);

gpuit_EXPORT
void 
endian4_big_to_native (void* buf, unsigned long len);

gpuit_EXPORT
void
endian4_native_to_big (void* buf, unsigned long len);

gpuit_EXPORT
void
endian4_little_to_native (void* buf, unsigned long len);

gpuit_EXPORT
void
endian4_native_to_little (void* buf, unsigned long len);

#endif
