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

#define DELAYLOAD_WRAP(f, ...)                             \
    f (__VA_ARGS__); typedef f##_t(__VA_ARGS__);

// Now that plastimatch is officially C++, we can now safely
// define this macro, which reduces programmer error.  This
// should be used instead of LOAD_LIBRARY
#if _WIN32
    #define LOAD_LIBRARY_SAFE(lib)                              \
        void* lib = delayload_cuda (#lib ".dll");
#else
    #define LOAD_LIBRARY_SAFE(lib)                              \
        void* lib = delayload_cuda ("lib" #lib ".so");
#endif

// Note: if lib contains a null pointer here
//       (see above note), this will return a
//       null function pointer.  Be careful pls.
#if _WIN32
    #define LOAD_SYMBOL(f, lib)                  \
        ;
#else
    #define LOAD_SYMBOL(f, lib)                  \
        f##_t* sym = 0;                          \
        if (lib) sym = dlsym (lib, #sym);
#endif

// Despite what the man pages say, dlclose()ing NULL
// was resulting in segfaults!  So, now we check 1st.
#if _WIN32
    #define UNLOAD_LIBRARY(lib)                    \
        ;
#else
    #define UNLOAD_LIBRARY(lib)                    \
        if (lib) {                                 \
            dlclose (lib);                         \
        }
#endif

PLMSYS_C_API bool check_library (const char *);
PLMSYS_C_API void* delayload_cuda (const char *);

#endif
