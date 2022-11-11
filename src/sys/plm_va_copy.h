/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __plm_va_copy_h__
#define __plm_va_copy_h__

#if defined(__BORLANDC__) || defined(_MSC_VER)
#ifndef va_copy
#define va_copy(d,s) ((d) = (s))
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#ifndef va_copy
#define va_copy(dest, src) __builtin_va_copy(dest, src)
#endif
#endif

#endif
