# - Wrapper around FindCUDA

macro (set_compute_capabilities)
  # JAS 2010.12.09
  #   Build code for all known compute capabilities by default.
  #   When developing, it is sometimes nice to turn this off in order
  #   to speed up the build processes (since you only have 1 GPU in your
  #   machine).
  # GCS 2024-09-17
  #   Remove support for compute capabilities less than 5.
  if (PLM_CUDA_ALL_DEVICES)
    message (STATUS "CUDA Build Level: ALL Compute Capabilities")

    if (CUDA_VERSION VERSION_LESS "6.5")
      message (FATAL "CUDA version is too old")
    endif ()

    if (CUDA_VERSION VERSION_EQUAL "6.5" OR CUDA_MAJOR_VERSION EQUAL "7")
      message (STATUS " Compute Cap 5: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_53,code=compute_53
	)
    endif()

    if (CUDA_VERSION_MAJOR EQUAL "8")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_60,code=compute_60
	)
    endif()

    if (CUDA_VERSION_MAJOR EQUAL "9")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_72,code=compute_72
	)
    endif()

    if (CUDA_VERSION_MAJOR EQUAL "10")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_75,code=compute_75
      )
    endif()

    if (CUDA_VERSION VERSION_EQUAL "11.0")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      message (STATUS " Compute Cap 8: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-gencode arch=compute_80,code=compute_80
      )
    endif()

    if (CUDA_VERSION VERSION_GREATER "11.0"
	AND CUDA_VERSION VERSION_LESS_EQUAL "11.4")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      message (STATUS " Compute Cap 8: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-gencode arch=compute_86,code=sm_86
	-gencode arch=compute_86,code=compute_86
      )
    endif()

    if (CUDA_VERSION VERSION_GREATER "11.4"
	AND CUDA_VERSION VERSION_LESS_EQUAL "11.7.1")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      message (STATUS " Compute Cap 8: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-gencode arch=compute_86,code=sm_86
	-gencode arch=compute_87,code=sm_87
	-gencode arch=compute_87,code=compute_87
      )
    endif()

    if (CUDA_VERSION VERSION_EQUAL "11.8")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      message (STATUS " Compute Cap 8: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-gencode arch=compute_86,code=sm_86
	-gencode arch=compute_87,code=sm_87
	-gencode arch=compute_88,code=sm_88
	-gencode arch=compute_88,code=compute_88
      )
    endif()

    if (CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
      message (STATUS " Compute Cap 5: [X]")
      message (STATUS " Compute Cap 6: [X]")
      message (STATUS " Compute Cap 7: [X]")
      message (STATUS " Compute Cap 8: [X]")
      set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_70,code=sm_70
	-gencode arch=compute_72,code=sm_72
	-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-gencode arch=compute_86,code=sm_86
	-gencode arch=compute_87,code=sm_87
	-gencode arch=compute_88,code=sm_88
	-gencode arch=compute_90,code=sm_90
	-gencode arch=compute_90,code=compute_90
      )
    endif()
  else ()
    message (STATUS "CUDA Build Level: Build system Compute Capability ONLY!")
  endif ()
endmacro ()

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
  endif ()

  # GCS 2012-05-11:  We need to propagate cxx flags to nvcc, but
  # the flag -ftest-coverage causes nvcc to barf, so exclude that one
  if (CMAKE_COMPILER_IS_GNUCC)
    string (REPLACE "-ftest-coverage" "" TMP "${CMAKE_CXX_FLAGS}")
    string (REPLACE " " "," TMP "${TMP}")
    set (CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS} ${TMP})
  endif ()

  # GCS 2012-05-07: Workaround for poor, troubled FindCUDA
  set (CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)

  find_package (CUDA QUIET)

  # GCS 2016-12-23: Tell FindCUDA to tell nvcc to use the c++ compiler,
  # which it doesn't do even if CUDA_HOST_COMPILATION_CPP is true.
  # This has to be done after calling FindCUDA, because FindCUDA overwrites
  # the variable.  PS: Merry Christmas!
  if (NOT MSVC)
    set (CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif ()
endif ()



if (CUDA_FOUND AND CUDA_VERSION VERSION_GREATER_EQUAL "6.5")
  message (STATUS "CUDA version ${CUDA_VERSION}")
elseif (CUDA_FOUND)
  message (STATUS "CUDA version ${CUDA_VERSION} is too old.")
  set (CUDA_FOUND OFF)
else ()
  message (STATUS "CUDA not found")
endif ()

if (CUDA_FOUND)
  # 14-5-2016 PAOLO: WORKAROUND GCC 6.1 AND CUDA 7.5 INCOMPATIBILITY
  if (CMAKE_COMPILER_IS_GNUCC
      AND (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
      AND (NOT CUDA_VERSION_MAJOR VERSION_GREATER 7))
    set (CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS},-std=c++98")
  endif ()

  # ITK headers cannot be processed by nvcc, so we define
  # PLM_CUDA_COMPILE for the purpose of guarding
  # (see base/plmbase.h)
  if (CUDA_CXX_FLAGS)
    set (CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS},-DPLM_CUDA_COMPILE=1")
  else ()
    set (CUDA_CXX_FLAGS "-DPLM_CUDA_COMPILE=1")
  endif ()

  # GCS 2012-09-25 - Seems this is needed too
  if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set (CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS},-fPIC")
  endif ()

  if (CUDA_CXX_FLAGS)
    list (APPEND CUDA_NVCC_FLAGS --compiler-options ${CUDA_CXX_FLAGS})
  endif ()

  cuda_include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

  # GCS 2012-05-22 -- viscous fluid registration requires CUDA SDK.
  if (NOT CUDA_SDK_ROOT_DIR)
    # Try some obvious paths not searched by stock FindCUDA
    find_path(CUDA_SDK_ROOT_DIR common/inc/cutil.h
      "$ENV{HOME}/NVIDIA_GPU_Computing_SDK/C"
      "/usr/local/NVIDIA_GPU_Computing_SDK/C"
      )
  endif ()
  if (CUDA_SDK_ROOT_DIR)
    find_path (CUDA_CUT_INCLUDE_DIR
      cutil.h
      PATHS ${CUDA_SDK_SEARCH_PATH}
      PATH_SUFFIXES "common/inc" "C/common/inc"
      DOC "Location of cutil.h"
      NO_DEFAULT_PATH
      )
    if (CUDA_CUT_INCLUDE_DIR)
      cuda_include_directories (
	${CUDA_CUT_INCLUDE_DIR}
	)
    endif ()
  endif ()

  # JAS 08.25.2010
  #   Check to make sure nvcc has gcc-4.3 for compiling.
  #   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
  include (nvcc-check)

  # Make nvcc less whiny
  if (CUDA_VERSION_MAJOR GREATER "5")
    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      --Wno-deprecated-gpu-targets)
  endif ()

  # Let CUDA work with unsupported (too new) gcc versions
  # Note: CUDA 10 cannot be used with gcc 10 or greater.
  if (CUDA_VERSION_MAJOR EQUAL "8"
      AND CMAKE_COMPILER_IS_GNUCC
      AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    list (APPEND CUDA_NVCC_FLAGS
      --compiler-options -D__GNUC__=5)
  elseif (CUDA_VERSION_MAJOR EQUAL "9"
      AND CMAKE_COMPILER_IS_GNUCC
      AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    list (APPEND CUDA_NVCC_FLAGS
      --compiler-options -D__GNUC__=6)
  elseif (CUDA_VERSION_MAJOR EQUAL "10"
      AND CMAKE_COMPILER_IS_GNUCC)
    if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
      list (APPEND CUDA_NVCC_FLAGS
	--compiler-options -D__GNUC__=8)
      # For gcc 10, let it fail.
    endif()
  endif ()

  # Choose compute capabilities
  set_compute_capabilities ()

endif ()
