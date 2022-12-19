/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __plmutil_config_h__
#define __plmutil_config_h__

#include "plm_config.h"

#if ((defined(_WIN32) || defined(WIN32)) && (defined (PLM_BUILD_SHARED_LIBS)))
# ifdef plmutil_EXPORTS
#   define PLMUTIL_C_API EXTERNC __declspec(dllexport)
#   define PLMUTIL_API __declspec(dllexport)
# else
#   define PLMUTIL_C_API EXTERNC __declspec(dllimport)
#   define PLMUTIL_API __declspec(dllimport)
# endif
# ifdef plmutilcuda_EXPORTS
#   define PLMUTILCUDA_C_API EXTERNC __declspec(dllexport)
#   define PLMUTILCUDA_API __declspec(dllexport)
# else
#   define PLMUTILCUDA_C_API EXTERNC __declspec(dllimport)
#   define PLMUTILCUDA_API __declspec(dllimport)
# endif
#else
# define PLMUTIL_C_API EXTERNC 
# define PLMUTIL_API
# define PLMUTILCUDA_C_API EXTERNC 
# define PLMUTILCUDA_API
#endif
#ifdef SWIG
#define PLMUTIL_API 
#endif

#endif
