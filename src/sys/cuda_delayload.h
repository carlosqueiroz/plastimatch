/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_delayload_h_
#define _cuda_delayload_h_

#include "plmsys_config.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <stdlib.h>

// Needed for delay loading windows DLLs
#if _WIN32
    #pragma comment(lib, "delayimp")
    #pragma comment(lib, "user32")
#endif

// Note: if lib contains a null pointer here
//       (see above note), this will return a
//       null function pointer.  Be careful pls.
#if _WIN32
    #define LOAD_SYMBOL(sym, lib)                  \
        ;
#else
    #define LOAD_SYMBOL(sym, lib)                  \
        sym##_##t* sym = (sym##_##t*) dlsym (lib, #sym);
#endif

// JAS 2012.03.29
// ------------------------------------------------------------
// Now that plastimatch is officially C++, we can now safely
// define this macro, which reduces programmer error.  This
// should be used instead of LOAD_LIBRARY
#if _WIN32
    #define LOAD_LIBRARY_SAFE(lib)                 \
        if (!delayload_##lib()) { exit (0); }      \
        ;
#else
    #define LOAD_LIBRARY_SAFE(lib)                 \
        if (!delayload_##lib()) { exit (0); }      \
        void* lib = dlopen_ex (#lib".so");
#endif

// JAS 2010.12.09
// Despite what the man pages say, dlclose()ing NULL
// was resulting in segfaults!  So, now we check 1st.
#if !defined(_WIN32) && defined(PLM_USE_GPU_PLUGINS)
    #define UNLOAD_LIBRARY(lib)                    \
        if (lib != NULL) {                         \
            dlclose (lib);                         \
        }
#else
    #define UNLOAD_LIBRARY(lib)                    \
        ;
#endif

#define DELAYLOAD_WRAP(f, ...)                     \
    f (__VA_ARGS__); typedef f##_t(__VA_ARGS__);

PLMSYS_C_API int delayload_libplmcuda (void);
PLMSYS_C_API int delayload_libplmreconstructcuda (void);
PLMSYS_C_API int delayload_libplmregistercuda (void);
PLMSYS_C_API int delayload_libplmopencl (void);
PLMSYS_C_API void* dlopen_ex (const char* lib);

#endif
