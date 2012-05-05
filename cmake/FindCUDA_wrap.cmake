# - Wrapper around FindCUDA

if (MINGW)
  # Cuda doesn't work with mingw at all
  set (CUDA_FOUND FALSE)
elseif (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  # FindCuda is included with CMake 2.8
  set (CUDA_FOUND FALSE)
else ()
  # GCS 2011.03.16
  # Make nvcc less whiny
  if (CMAKE_COMPILER_IS_GNUCC)
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    set (CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS} ${CMAKE_CXX_FLAGS})
  endif ()
  find_package (CUDA QUIET)
endif ()

# ITK headers cannot be processed by nvcc, so we define
# PLM_CUDA_COMPILE for the purpose of guarding
# (see base/plmbase.h)
set (CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS};-DPLM_CUDA_COMPILE)

if (CUDA_CXX_FLAGS)
    set (CUDA_NVCC_FLAGS --compiler-options ${CUDA_CXX_FLAGS})
endif ()

set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

if (CUDA_FOUND)
  cuda_include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif ()

# JAS 08.25.2010
#   Check to make sure nvcc has gcc-4.3 for compiling.
#   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
include (nvcc-check)

# JAS 2010.12.09
#   Build code for all known compute capabilities by default.
#   When developing, it is sometimes nice to turn this off in order
#   to speed up the build processes (since you only have 1 GPU in your machine).
set (PLM_CUDA_ALL_DEVICES ON CACHE BOOL 
  "Generate GPU code for all compute capabilities?")
if (PLM_CUDA_ALL_DEVICES)
  message (STATUS "CUDA Build Level: ALL Compute Capabilities")

  message (STATUS "  >> Generation 1: [X]")
  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_10,code=sm_10
        -gencode arch=compute_11,code=sm_11
        -gencode arch=compute_12,code=sm_12
        -gencode arch=compute_13,code=sm_13
    )

if(CUDA_VERSION_MAJOR GREATER "2")
  message (STATUS "  >> Generation 2: [X]")
    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_20,code=sm_20
    )
else()
  message (STATUS "  >> Generation 2: [ ]")
endif()

  #MESSAGE(STATUS "<<-->>: CUDA_NVCC_FLAGS set to \"${CUDA_NVCC_FLAGS}\"")
else ()
  message (STATUS "CUDA Build Level: Build system Compute Capability ONLY!")
endif ()

