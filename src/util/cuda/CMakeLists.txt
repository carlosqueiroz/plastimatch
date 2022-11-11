##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_util_cuda)

set_directory_properties (PROPERTIES INCLUDE_DIRECTORIES "")
set_directory_properties (PROPERTIES COMPILE_DEFINITIONS "")

include_directories (BEFORE ${PLM_BINARY_DIR})
include_directories (BEFORE ${CMAKE_SOURCE_DIR}/src/base)
include_directories (BEFORE ${CMAKE_SOURCE_DIR}/src/cuda)
include_directories (BEFORE ${CMAKE_SOURCE_DIR}/src/sys)
include_directories (BEFORE ${CMAKE_SOURCE_DIR}/src/util)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (AFTER ${DLIB_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})

include_directories (AFTER ${CUDA_INCLUDE_DIRS})

set (PLMUTILCUDA_SRC
  distance_map_cuda.cu distance_map_cuda.h
  )

set (PLMUTILCUDA_IFACE_SRC
  # If ordinary cxx file is needed, add here
  )

set (PLMUTILCUDA_LIBRARY_SRC
  ${PLMUTILCUDA_IFACE_SRC}
  ${PLMUTILCUDA_SRC}
  )

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMUTILCUDA_LIBRARY_DEPENDENCIES
  plmbase plmcuda ${PLM_CUDA_LIBRARIES})

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: CUDA
##-----------------------------------------------------------------------------
if (CUDA_FOUND)
  if (PLM_USE_GPU_PLUGINS)

    # build CUDA as separate plmutilcuda shared library
    plm_add_gpu_plugin_library (plmutilcuda 
      "${PLMUTILCUDA_LIBRARY_SRC}")

    target_link_libraries (plmutilcuda 
      ${PLMUTILCUDA_LIBRARY_DEPENDENCIES})

    # set dependencies that plmutil library needs 
    if (WIN32 AND NOT CYGWIN AND NOT MINGW)
      # windows needs to delayload it
      set (PLMUTIL_LIBRARY_DEPENDENCIES
	${PLMUTIL_LIBRARY_DEPENDENCIES}
	plmutilcuda)
      set (PLMUTIL_LIBRARY_LDFLAGS
	"${PLMUTIL_LIBRARY_LDFLAGS} /DELAYLOAD:plmutilcuda.dll")
    endif ()
    if (NOT WIN32 AND LIBDL_FOUND)
      # and unix needs dlopen()
      set (PLMUTIL_LIBRARY_DEPENDENCIES
	${PLMUTIL_LIBRARY_DEPENDENCIES}
	-ldl)
    endif ()
  else ()
    # building CUDA into plmutil
    cuda_compile (CUDA_WRAPPERS ${PLMUTILCUDA_SRC})
    set (PLMUTILCUDA_LIBRARY_SRC
      ${CUDA_WRAPPERS}
      ${PLMUTILCUDA_IFACE_SRC}
      )
    plm_add_library (plmutilcuda
      "${PLMUTILCUDA_LIBRARY_SRC}"
      "${PLMUTILCUDA_LIBRARY_DEPENDENCIES}"
      ""
      "${PLASTIMATCH_INCLUDE_DIRECTORIES}"
      "")
  endif ()
endif ()
