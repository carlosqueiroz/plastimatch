##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (plastimatch)

# The version here should be equal to the "most recent release"
set (PLM_VERSION_MAJOR "1")
set (PLM_VERSION_MINOR "9")
set (PLM_VERSION_PATCH "4")

##-----------------------------------------------------------------------------
##  Set up CMake defaults
##-----------------------------------------------------------------------------
# CMAKE_CXX_STANDARD was introduced in 3.1.3
cmake_minimum_required (VERSION 3.1.3)

# CMP0003: Libraries linked via full path no longer produce linker search
#  paths.
if (POLICY CMP0003)
  cmake_policy (SET CMP0003 NEW)
endif ()
# CMP0012: if() recognizes numbers and boolean constants.
# GCS 2012-03-04: This is added to suppress ITK warning
if (POLICY CMP0012)
  cmake_policy (SET CMP0012 NEW)
endif ()
# CMP0017: Prefer files from the CMake module directory when including
# from there.
if (POLICY CMP0017)
  cmake_policy (SET CMP0017 NEW)
endif ()
# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()
# CMP0054: Quoted variables no longer dereferenced
# GCS 2017-11-02: This is added to suppress Qt warnings
if (POLICY CMP0054)
  cmake_policy (SET CMP0054 NEW)
endif ()
# Allow plastimatch to set CUDA option variables
if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif ()
# UseSWIG generates standard target names
if (POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif ()
# UseSWIG honors SWIG_MODULE_NAME
if (POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif ()


##-----------------------------------------------------------------------------
##  CMake path
##-----------------------------------------------------------------------------
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

##-----------------------------------------------------------------------------
##  Prevent in-source build
##-----------------------------------------------------------------------------
include (PreventInSourceBuilds)

##-----------------------------------------------------------------------------
##  Misc macros and functions used in this script
##-----------------------------------------------------------------------------
include (PlmMacros)
include (SuperbuildOptions)

##-----------------------------------------------------------------------------
##  Define default build type as Release
##-----------------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
endif ()
sb_variable (CMAKE_BUILD_TYPE)

##-----------------------------------------------------------------------------
##  Define plastimatch configuration variables
##-----------------------------------------------------------------------------
sb_option (PLM_CONFIG_ENABLE_CUDA "Set to ON to allow build with CUDA" ON)
option (PLM_CONFIG_ENABLE_DCMTK "Set to ON to allow build with DCMTK" ON)
option (PLM_CONFIG_ENABLE_FATM "Set to ON to allow building fatm" OFF)
option (PLM_CONFIG_ENABLE_MONDOSHOT "Set to ON to allow building mondoshot" OFF)
sb_option (PLM_CONFIG_ENABLE_OPENCL "Set to ON to allow build with OpenCL" ON)
sb_option (PLM_CONFIG_ENABLE_OPENMP "Set to ON to allow build with OpenMP" ON)
sb_option (PLM_CONFIG_ENABLE_OPT4D "Set to ON to allow build with Opt4D" OFF)
option (PLM_CONFIG_ENABLE_PLASTIMATCH "Set to ON to allow building plastimatch" ON)
option (PLM_CONFIG_ENABLE_QT "Set to ON to allow build with QT" ON)
sb_option (PLM_CONFIG_ENABLE_SSE2 "Set to ON to allow build with SSE" ON)
option (PLM_CONFIG_ENABLE_MATLAB "Set to ON to allow build of Matlab plugins" OFF)
sb_option (PLM_CONFIG_ENABLE_VISCOUS
  "Set to ON to allow build of experimental viscous fluid registration algorithm" OFF)
sb_option (PLM_CONFIG_ENABLE_PLASTIMATCH_QT
  "Set to ON to allow build of experimental plastimatch_qt executable" OFF)
sb_option (PLM_CONFIG_ENABLE_CSHARP "Set to ON to allow build CSharp wrapping" OFF)
sb_option (PLM_CONFIG_ENABLE_PYTHON "Set to ON to allow build Python wrapping" OFF)

option (PLM_CONFIG_BUILD_QT_PLUGINS "Set to ON to allow build of QT4 Designer Plugins" OFF)
sb_option (PLM_CONFIG_DEBIAN_BUILD "Set to ON to configure build for debian" OFF)
option (PLM_CONFIG_LIBRARY_BUILD "Set to ON to build only libraries" OFF)
option (PLM_CONFIG_NOMANIFEST
  "Set to ON to build windows DLLs without manifests" OFF)
option (PLM_CONFIG_ENABLE_SUPERBUILD
  "Set to ON to allow src/plastimatch to be built using nested superbuild" OFF)

# Plastimatch software configuration options
option (PLM_CONFIG_CLANG_COMPLETE
  "Generate .clang_complete for hipster Vim-ers" OFF)
sb_option (PLM_PREFER_NEW_ITK_DEMONS
  "Prefer using the newer version of patched ITK demons files" OFF)
sb_option (PLM_CONFIG_ITKV4_REGISTRATION
  "Set to ON to use the ITKV4 registration framework" OFF)
sb_option (PLM_CONFIG_VOL_LIST
  "Native support for volumes with irregular slice thicknesses" OFF)
sb_option (PLM_CONFIG_HARDEN_XFORM_BY_DEFAULT
  "By default, harden linear transforms rather than resample" OFF)

# Plastimatch legacy options
sb_option (PLM_CONFIG_LEGACY_BSPLINE_EXTEND
  "Use legacy code for extending b-spline domain" OFF)
sb_option (PLM_CONFIG_LEGACY_BSPLINE_XFORM_IO
  "Use legacy code for reading and writing b-spline xform files" OFF)
sb_option (PLM_CONFIG_LEGACY_CUDA_DELAYLOAD
  "Use legacy code for verifying CUDA runtime" ON)
sb_option (PLM_CONFIG_LEGACY_DRR_CUDA_TEXTURE
  "Use legacy 1D texture for CUDA DRR generation" OFF)
sb_option (PLM_CONFIG_LEGACY_MI_METRIC
  "For ITK metrics, the legacy implementation of the mi metric is Viola-Wells to Mattes" OFF)
sb_option (PLM_CONFIG_LEGACY_PROJ_GEO
  "Use legacy method for specifying projection geometry" ON)
sb_option (PLM_CONFIG_LEGACY_SQUARED_REGULARIZER
  "Use legacy regularization lambda value scheme where values are squared" OFF)
sb_option (PLM_CONFIG_LEGACY_RAY_TRACE_EXACT
  "Use legacy algorithm for exact ray tracing" OFF)

# Compile and link options
sb_option (PLM_CUDA_ALL_DEVICES
  "Generate GPU code for all compute capabilities?" OFF)
sb_option (BUILD_SHARED_LIBS "Build plastimatch as shared library" OFF)

# Choose whether to build against included and/or superbuild libraries
# instead of system libraries
sb_option_enum (PLM_SYSTEM_DCMTK "Prefer DCMTK provided by operating system"
  PREFERRED
  NO PREFERRED YES)
sb_option (PLM_PREFER_SYSTEM_DLIB
    "Prefer the system dlib over the included dlib" ON)
sb_option_enum (PLM_SYSTEM_ITK "Prefer ITK provided by operating system"
  PREFERRED
  NO PREFERRED YES)

# Testing
sb_option (PLM_BUILD_TESTING "Enable regression testing" ON)

# Installer Options
option (PLM_INSTALL_RPATH "Add full RPATH to install" OFF)
sb_option (PLM_CONFIG_INSTALL_LIBRARIES "Include libraries in install" ON)

# Packaging
option (PLM_PACKAGE_32BIT
    "Set this when building 32-bit packages on a 64-bit machine" OFF)
option (PLM_PACKAGE_NSIS "Set to ON when packaging binaries with NSIS" OFF)
option (PLM_PACKAGE_WIX "Set to ON when packaging binaries with WIX" ON)
sb_option (PLM_PACKAGE_LEGACY_CMAKE_CONFIG
    "Use the old code for creating PlastimatchConfig.cmake and friends" OFF)

# Use legacy packaging if cmake version is < 3.0
if (CMAKE_VERSION VERSION_LESS "3.0")
  set (PLM_PACKAGE_LEGACY_CMAKE_CONFIG ON CACHE BOOL "Use legacy packaging routines")
endif ()

# Override some options if library build is selected
if (PLM_CONFIG_LIBRARY_BUILD)
  set (PLM_CONFIG_ENABLE_FATM OFF)
  set (PLM_CONFIG_ENABLE_MONDOSHOT OFF)
endif ()

##-----------------------------------------------------------------------------
##  Setup important locations
##-----------------------------------------------------------------------------
sb_set (PLM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
sb_set (PLM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Offer the user the choice of overriding the installation directories
set (PLM_INSTALL_LIB_DIR lib CACHE PATH
  "Installation directory for libraries")
set (PLM_INSTALL_BIN_DIR bin CACHE PATH
  "Installation directory for executables")
set (PLM_INSTALL_INCLUDE_DIR include/plastimatch CACHE PATH
  "Installation directory for header files")
if (WIN32 AND NOT CYGWIN)
  set (DEF_INSTALL_CMAKE_DIR CMake)
else()
  set (DEF_INSTALL_CMAKE_DIR lib/cmake/plastimatch)
endif()
set (PLM_INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
mark_as_advanced (
  PLM_INSTALL_LIB_DIR
  PLM_INSTALL_BIN_DIR
  PLM_INSTALL_INCLUDE_DIR
  PLM_INSTALL_CMAKE_DIR)


# GCS 2015-07-08.  The below workflow is described here:
#   http://www.cmake.org/Wiki/CMake/Tutorials/How_to_create_a_ProjectConfig.cmake_file
# However, it creates install targets with absolute paths, which is
# not allowed for NSIS targets.
# It is possible this is resolved by newer documentation, which is referred
# by the above link:
#   http://www.cmake.org/cmake/help/git-master/manual/cmake-packages.7.html#creating-packages
# Needs further investigation

# Make relative install paths absolute (needed later on)
if (PLM_PACKAGE_LEGACY_CMAKE_CONFIG)
foreach (p LIB BIN INCLUDE CMAKE)
  set (var PLM_INSTALL_${p}_DIR)
  if (NOT IS_ABSOLUTE "${${var}}")
    set (${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif ()
endforeach ()
endif ()

##-----------------------------------------------------------------------------
##  CMake include files
##-----------------------------------------------------------------------------
include (CTest)
include (CheckFunctionExists)

# http://www.cmake.org/pipermail/cmake/2008-December/025886.html
# http://www.cmake.org/Bug/view.php?id=15117
set (CMAKE_INSTALL_OPENMP_LIBRARIES ON)
if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 3.1)
    include (InstallRequiredSystemLibrariesPatched)
else ()
    include (InstallRequiredSystemLibraries)
endif ()

# Superbuild
include (ExternalProject)

##-----------------------------------------------------------------------------
##  Disable spurious warnings on MSVC (version 8 & higher)
##-----------------------------------------------------------------------------
if (WIN32 AND NOT CYGWIN AND NOT MINGW)
  add_definitions(
    -D_CRT_FAR_MAPPINGS_NO_DEPRECATE
    -D_CRT_IS_WCTYPE_NO_DEPRECATE
    -D_CRT_MANAGED_FP_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_DEPRECATE_GLOBALS
    -D_CRT_SETERRORMODE_BEEP_SLEEP_NO_DEPRECATE
    -D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
    -D_CRT_VCCLRIT_NO_DEPRECATE
    -D_SCL_SECURE_NO_DEPRECATE
    # _SCL_SECURE_NO_DEPRECATE became _SCL_SECURE_NO_WARNINGS in VC9
    -D_SCL_SECURE_NO_WARNINGS
    )
endif ()

##-----------------------------------------------------------------------------
##  Disable spurious warnings on GCC
##-----------------------------------------------------------------------------
if (CMAKE_COMPILER_IS_GNUCC)
  add_compile_options (-Wno-cpp)
endif ()

##-----------------------------------------------------------------------------
##  Compile with -fPIC is needed for amd64 and other platforms
##-----------------------------------------------------------------------------
plm_set_pic_flags ()

##-----------------------------------------------------------------------------
##  Extra paths
##-----------------------------------------------------------------------------
set (PLM_TESTING_SOURCE_DIR "${PLM_SOURCE_DIR}/Testing")
set (PLM_TESTING_DATA_DIR "${PLM_SOURCE_DIR}/Testing/Data")
set (PLM_TESTING_DOWNLOAD_DATA_DIR "${PLM_BINARY_DIR}/TestData")
set (PLM_BUILD_TESTING_DIR "${PLM_BINARY_DIR}/Testing")

##-----------------------------------------------------------------------------
##  Set the math library
##-----------------------------------------------------------------------------
if (UNIX)
  set (MATH_LIB -lm)
else ()
  set (MATH_LIB)
endif ()

##-----------------------------------------------------------------------------
##  Libdl
##-----------------------------------------------------------------------------
# Some systems such as pcBSD don't have libdl
find_library (LIBDL_FOUND dl)

##-----------------------------------------------------------------------------
##  DCMTK
##-----------------------------------------------------------------------------
## If it is a modern version of DCMTK, such as found in Slicer build,
## it will have a working version of DCMTKConfig.cmake.
## In this case, we can use the modern find_package variant.
## Otherwise we use the old hacked version of FindDCMTK.cmake.
if (PLM_CONFIG_ENABLE_DCMTK)
  if (PLM_SYSTEM_DCMTK STREQUAL "YES")
    find_package (DCMTK REQUIRED)
  elseif (PLM_SYSTEM_DCMTK STREQUAL "PREFERRED")
    #find_package (DCMTK QUIET)
    find_package (DCMTK)
  endif ()

  if (NOT DCMTK_FOUND AND NOT PLM_SYSTEM_DCMTK STREQUAL "YES")
    include (SuperBuild/External_DCMTK.cmake)
    message (STATUS "DCMTK will be downloaded and built.")
  else ()
    if (DCMTK_FOUND)
      message (STATUS "DCMTK version ${DCMTK_VERSION} found (1).")
    else ()
      message (STATUS "DCMTK not found.")
    endif ()
  endif ()
else ()
  set (DCMTK_FOUND FALSE)
endif ()

# Workaround for CMake bug in FindDCMTK.  This was fixed some time between
# 3.4.3 and 3.7.2
# if (CMAKE_VERSION VERSION_LESS "3.7.2" AND DCMTK_ofstd_INCLUDE_DIR)
#   get_filename_component(_tmp ${DCMTK_ofstd_INCLUDE_DIR} PATH)
#   get_filename_component(_tmp ${_tmp} PATH)
#   list (APPEND DCMTK_INCLUDE_DIRS ${_tmp})
# endif ()

##-----------------------------------------------------------------------------
##  Search for libraries
##-----------------------------------------------------------------------------
find_package (Etags)
find_package (FFTW)
find_package (Git)
find_package (libLBFGS)
find_package (nlopt)
if (PLM_CONFIG_ENABLE_MATLAB)
  find_package (Matlab)
else ()
  set (MATLAB_FOUND false)
endif ()
find_package (Octave)
if (PLM_CONFIG_ENABLE_OPENCL)
  find_package (OpenCL)
endif ()
if (PLM_CONFIG_ENABLE_OPENMP)
  find_package (OpenMP)
endif ()
if (PLM_CONFIG_ENABLE_SSE2)
  find_package (SSE)
else ()
  set (SSE2_FOUND false)
endif ()
find_package (wxWidgets)

##-----------------------------------------------------------------------------
##  Use local libsbfgs if not found
##-----------------------------------------------------------------------------
if (NOT libLBFGS_FOUND)
  add_subdirectory (libs/liblbfgs-1.9)
  sb_set (libLBFGS_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/liblbfgs-1.9/include")
  sb_set (libLBFGS_LIBRARY lbfgs)
  link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/liblbfgs-1.9)
  sb_set (libLBFGS_FOUND TRUE)
endif ()

##-----------------------------------------------------------------------------
##  SQLite3
##-----------------------------------------------------------------------------
if (PLM_CONFIG_DEBIAN_BUILD)
  set (SQLite3_FOUND false)
else ()
  if (CMAKE_VERSION VERSION_LESS "3.14")
    # Local implementation
    find_package (SQLite)
  else ()
    find_package (SQLite3)
  endif ()
endif ()
if (NOT SQLite3_FOUND AND NOT PLM_CONFIG_DEBIAN_BUILD)
  add_subdirectory (libs/sqlite-3.6.21)
  set (SQLite3_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/sqlite-3.6.21")
  set (SQLite3_LIBRARIES sqlite3)
  set (SQLite3_FOUND TRUE)
endif ()

#-----------------------------------------------------------------------------
##  Only use local devillard
##-----------------------------------------------------------------------------
sb_set (DEVILLARD_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/devillard")
add_subdirectory (libs/devillard)

##-----------------------------------------------------------------------------
##  Only use local inih
##-----------------------------------------------------------------------------
sb_set (INIH_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/inih-r29")
add_subdirectory (libs/inih-r29)

##-----------------------------------------------------------------------------
##  LUA 5.1.4
##-----------------------------------------------------------------------------
# We can't use debian's lua 5.1 because it doesn't contain lobject.h
#add_subdirectory (libs/lua-5.1.4)
#set (LUA_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/lua-5.1.4/src")
#link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/lua-5.1.4)

# for history, etc in plastimatch tty mode
#find_package (Readline)
#find_library (TERMCAP_LIBRARY termcap)

##-----------------------------------------------------------------------------
##  Only use local specfun
##-----------------------------------------------------------------------------
add_subdirectory (libs/specfun)
link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/specfun)
sb_set (SPECFUN_FOUND TRUE)

##-----------------------------------------------------------------------------
##  Only use local msinttypes
##-----------------------------------------------------------------------------
sb_set (MSINTTYPES_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs")

##-----------------------------------------------------------------------------
##  Only use local nkidecompress
##-----------------------------------------------------------------------------
sb_set (NKIDECOMPRESS_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/nkidecompress")
add_subdirectory (libs/nkidecompress)

##-----------------------------------------------------------------------------
##  Special CUDA processing
##-----------------------------------------------------------------------------
if (PLM_CONFIG_ENABLE_CUDA)
  find_package (CUDA_wrap)
  set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")
  if (CUDA_FOUND)
    cuda_include_directories (${CMAKE_CURRENT_SOURCE_DIR})
  endif ()
else ()
  set (CUDA_FOUND false)
endif ()

## GCS 2020-12-28.  Not sure what to do with this.  Because Thrust is now included
## in CUDA, we should be using that one directly.
# find_package (Thrust)

##-----------------------------------------------------------------------------
## JAS 10.28.2010
## nvcc automatically passed the flag -malign-double to gcc when compiling .cu
## files.  This can be a problem when structs are shared between
## .c & .cu files on 32-bit machines.  So, we pass -malign-double to gcc
## for .c files as well so that everybody is using the same alignment.
## -malign-double is automatically passed to gcc for 64-bit architectures.
##
## GCS 2010-10-30
## -malign-double should only be enabled for 32-bit machines.  gcc 4.1.2
## gives an error if it is used on x86_64.
##
## GCS 2011-07-21
## -malign-double is not compatible with ITK.  Need a fix soon...
##
## JAS 2011-07-23
## I have disabled -malign-double passing to gcc/g++ compiled objects and
## have taken to manually byte aligning structures that are passed to
## nvcc compiled objects.  (See double_align8 typedef in bspline.h)
## Hopefully this will solve the problem.
##-----------------------------------------------------------------------------
#if (CUDA_FOUND AND CMAKE_COMPILER_IS_GNUCC AND MACHINE_IS_32_BIT)
#  if (CMAKE_C_FLAGS)
#    set (CMAKE_C_FLAGS "-malign-double")
#  else ()
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -malign-double")
#  endif ()
#  if (CMAKE_CXX_FLAGS)
#    set (CMAKE_CXX_FLAGS "-malign-double")
#  else ()
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -malign-double")
#  endif ()
#endif ()

##-----------------------------------------------------------------------------
##  Doxygen
##-----------------------------------------------------------------------------
find_package (Doxygen)

##-----------------------------------------------------------------------------
##  SWIG
##-----------------------------------------------------------------------------
set (SWIG_FOUND OFF)
if (PLM_CONFIG_ENABLE_CSHARP OR PLM_CONFIG_ENABLE_PYTHON)
  find_package (SWIG 4.0 COMPONENTS csharp python)
  if (SWIG_FOUND)
    message (STATUS "SWIG found: ${SWIG_EXECUTABLE}")
    if (NOT SWIG_csharp_FOUND)
      message(WARNING "SWIG CSharp bindings cannot be generated")
    endif()
    if (NOT SWIG_python_FOUND)
      message(WARNING "SWIG Python bindings cannot be generated")
    endif()
  endif()
endif()

##-----------------------------------------------------------------------------
##  For shared libraries, we enable dynamic loading of cuda, opencl
##  Note: we can't use BUILD_SHARED_LIBS directly, because it conflicts
##  with the ITK header files
##-----------------------------------------------------------------------------
set (PLM_USE_GPU_PLUGINS OFF)
if (BUILD_SHARED_LIBS)
  set (PLM_USE_GPU_PLUGINS ON)
endif ()

##-----------------------------------------------------------------------------
##  Special linking instructions on unix
##  http://www.cmake.org/Wiki/CMake_RPATH_handling
##-----------------------------------------------------------------------------
if (PLM_INSTALL_RPATH)
  # use, i.e. don't skip the full RPATH for the build tree
  set (CMAKE_SKIP_BUILD_RPATH  FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  set (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  # the RPATH to be used when installing
  set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif ()

##-----------------------------------------------------------------------------
##  User feedback on which languages and libraries were found
##-----------------------------------------------------------------------------
if (wxWidgets_FOUND)
  message (STATUS "Looking for wxWidgets - found.")
else ()
  message (STATUS "Looking for wxWidgets - not found.")
endif ()
if (Matlab_FOUND)
  message (STATUS "Looking for Matlab - found.")
else ()
  message (STATUS "Looking for Matlab - not found.")
endif ()
if (OCTAVE_FOUND)
  message (STATUS "Looking for Octave - found.")
else ()
  message (STATUS "Looking for Octave - not found.")
endif ()

##-----------------------------------------------------------------------------
##  Set version string
##-----------------------------------------------------------------------------
set (PLM_BASIC_VERSION_STRING
  "${PLM_VERSION_MAJOR}.${PLM_VERSION_MINOR}.${PLM_VERSION_PATCH}")
set (PLASTIMATCH_VERSION_STRING
  "${PLM_VERSION_MAJOR}.${PLM_VERSION_MINOR}.${PLM_VERSION_PATCH}")
if (EXISTS "${PLM_SOURCE_DIR}/.git" AND GIT_FOUND)
  if (GIT_EXECUTABLE MATCHES "[wW][sS][lL].*")
    set (GIT_EXECUTABLE "wsl;git")
  endif ()
  execute_process (
    COMMAND
    #${GIT_EXECUTABLE} rev-list --first-parent --count HEAD
    ${GIT_EXECUTABLE} ${GIT_EXECUTABLE_EXTRA} describe --always
    WORKING_DIRECTORY "${PLM_SOURCE_DIR}"
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if (${git_result} EQUAL 0)
    string (REGEX MATCH "[^-]*-([0-9]+)-.*" junk ${git_output})
    set (PLM_VERSION_TWEAK "${CMAKE_MATCH_1}")
    set (PLASTIMATCH_VERSION_STRING "${git_output}")
  endif ()
endif ()
message (STATUS "Plastimatch version is ${PLASTIMATCH_VERSION_STRING}")

##-----------------------------------------------------------------------------
##  ITK
##-----------------------------------------------------------------------------
## ITK clobbers DCMTK_DIR, which is used in the superbuild
set (DCMTK_DIR_CACHE "${DCMTK_DIR}")
set (FIND_LIBRARY_USE_LIB64 true)
if (PLM_SYSTEM_ITK STREQUAL "YES")
  find_package (ITK REQUIRED
    PATHS "/usr/lib64/cmake/InsightToolkit")
elseif (PLM_SYSTEM_ITK STREQUAL "PREFERRED")
  find_package (ITK QUIET
    PATHS "/usr/lib64/cmake/InsightToolkit")
endif ()
if (NOT ITK_FOUND AND NOT PLM_SYSTEM_ITK STREQUAL "YES")
  message (STATUS "ITK will be downloaded and built.")
  include (SuperBuild/External_ITK.cmake)
endif ()

if (ITK_FOUND)
  include (HandleITK)
  include (HandleVXL)
endif ()
set (DCMTK_DIR "${DCMTK_DIR_CACHE}")

##-----------------------------------------------------------------------------
##  QT
##-----------------------------------------------------------------------------
if (PLM_CONFIG_ENABLE_QT)
  find_package (Qt4 4.6.3 QUIET COMPONENTS QtCore QtGui QtDesigner)
  find_package (Qt5 QUIET COMPONENTS Core Gui Designer Network Widgets)
else ()
  set (QT4_FOUND false)
  set (QT5_FOUND false)
endif ()

if (QT4_FOUND)
  # Test Qt install to make sure it can build and run a test program
  include (CheckQt)
  check_qt (QT_TEST_COMPILE_SUCCEEDED)
  if (NOT QT_TEST_COMPILE_SUCCEEDED)
    message (STATUS "Qt failed to compile a test program")
    set (QT4_FOUND false)
  endif ()
endif ()

if (QT4_FOUND)
    # Debian qtchooser is broken, so we disable.
    # http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=730423
    execute_process (
	COMMAND
	rcc
	RESULT_VARIABLE rcc_result
	OUTPUT_VARIABLE rcc_output
	ERROR_VARIABLE rcc_error)
    if (${rcc_error} MATCHES "could not find a Qt installation")
	message (STATUS "Sorry, your Qt build environment is broken")
	set (QT4_FOUND FALSE)
    else ()
	message (STATUS "Looking for Qt4 - found")
	#include (${QT_USE_FILE})
    endif ()
else ()
    message (STATUS "Looking for Qt4 - not found")
endif ()

if (Qt5_FOUND)
  message (STATUS "Looking for Qt5 - found")
else ()
  message (STATUS "Looking for Qt5 - not found")
endif ()

##-----------------------------------------------------------------------------
##  Only use local rapidjson
##-----------------------------------------------------------------------------
set (RAPIDJSON_DIR "${PLM_SOURCE_DIR}/libs/rapidjson-2015-03-22")
if (EXISTS "${RAPIDJSON_DIR}" AND IS_DIRECTORY "${RAPIDJSON_DIR}")
  set (RAPIDJSON_INCLUDE_DIR
    "${RAPIDJSON_DIR}/include"
    )
  set (RAPIDJSON_FOUND true)
endif ()

##-----------------------------------------------------------------------------
##  libyaml
##-----------------------------------------------------------------------------
find_package (yaml)

##-----------------------------------------------------------------------------
##  Opt4D
##-----------------------------------------------------------------------------
if (PLM_CONFIG_ENABLE_OPT4D)
  set (opt4D_DIR "${opt4D_DIR}" CACHE PATH "Path to opt4D build directory")
  find_package (opt4D)
  if (NOT opt4D_FOUND)
    message (STATUS "Opt4D will be downloaded and built.")
    include (SuperBuild/External_opt4D.cmake)
  endif ()
else ()
  set (opt4D_FOUND FALSE)
endif ()

##-----------------------------------------------------------------------------
##  Configure include files
##-----------------------------------------------------------------------------
sb_set (PLM_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

##-----------------------------------------------------------------------------
##  Subdirectories
##-----------------------------------------------------------------------------
add_subdirectory (src)

##-----------------------------------------------------------------------------
##  Additional install files
##-----------------------------------------------------------------------------
if (ITK_FOUND)
  set (DLL_PATTERNS "ITKCommon*.dll" "ITKIO*.dll" "ITKOptimizers*.dll" "ITKStatistics*.dll" "ITKTransform*.dll" "ITKSpatial*.dll" "ITKSmoothing*.dll" "itkhdf5*.dll")
  foreach (p ${DLL_PATTERNS})
    # .../bin/release/*.dll for build tree, .../bin/*.dll for install tree
    file (GLOB DLLFILES "${ITK_DLL_DIR}/${p}")
    if (DLLFILES)
      install (FILES ${DLLFILES} DESTINATION bin)
    endif ()
  endforeach ()
endif ()

if (DCMTK_FOUND)
  set (DLL_PATTERNS "oflog.dll" "ofstd.dll" "dcmdata.dll" "dcmjpeg.dll"
    "dcmimgle.dll" "ijg8.dll" "ijg12.dll" "ijg16.dll")
  foreach (p ${DLL_PATTERNS})
    # .../bin/release/*.dll for build tree, .../bin/*.dll for install tree
    file (GLOB DLLFILES "${DCMTK_DIR}/bin/release/${p}")
    if (DLLFILES)
      install (FILES ${DLLFILES} DESTINATION bin)
    endif ()
  endforeach ()
endif ()

#Add QT dlls to Install
if (QT4_FOUND)
  if (UNIX)
    # YKP: should be implemented soon
  else ()
    set (QT4_CORE_DLL_WIN "${QT_LIBRARY_DIR}/QtCore4.dll")
    set (QT4_GUI_DLL_WIN "${QT_LIBRARY_DIR}/QtGui4.dll")
    if (EXISTS "${QT4_CORE_DLL_WIN}")
      install (FILES "${QT4_CORE_DLL_WIN}" DESTINATION bin)
    endif ()
    if (EXISTS "${QT4_GUI_DLL_WIN}")
      install (FILES "${QT4_GUI_DLL_WIN}" DESTINATION bin)
    endif ()
  endif ()
endif ()

#Add FFT dlls to Install
if (FFTW_FOUND)
  if (EXISTS "${FFTW_DIR}/libfftw3-3.dll")
    install (FILES "${FFTW_DIR}/libfftw3-3.dll" DESTINATION bin)
  endif ()
  #YKP 05/27/2016: no need of libfftw3f-3.dll and libfftw3l-3.dll?
endif ()

# Add sample directory/files to Install Only for windows users
if (WIN32 OR WIN64)
  set (PLM_WINDOWS_INSTALL_DIR "${PLM_SOURCE_DIR}/extra/windows-install")
  install (DIRECTORY "${PLM_WINDOWS_INSTALL_DIR}/sample" DESTINATION bin)
  install (FILES "${PLM_WINDOWS_INSTALL_DIR}/launch_cmd_prompt.bat" DESTINATION bin)
endif ()

# JAS 2011.01.24
# I have moved the documentation DESTINATION from
# /usr/doc to /usr/share/doc/plastimatch/ for Linux/UNIX
if (UNIX)
  # FHS compliant path for Linux/UNIX
  set (DOC_DESTINATION "share/doc/plastimatch")
else ()
  # Just throw TXTs into a doc folder for Windows
  set (DOC_DESTINATION "doc")
endif ()

install (FILES
  README.TXT
  src/plastimatch/COPYRIGHT.TXT
  DESTINATION ${DOC_DESTINATION}
  )

# Debian doesn't like INSTALL, LICENSE included in documentation
if (NOT PLM_CONFIG_DEBIAN_BUILD)
  install (FILES
    INSTALL.TXT
    src/plastimatch/LICENSE.TXT
    DESTINATION ${DOC_DESTINATION}
    )
endif ()

##-----------------------------------------------------------------------------
##  MAKE UNINSTALL
##-----------------------------------------------------------------------------
if (UNIX)
  configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target (uninstall COMMAND
    ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()

##-----------------------------------------------------------------------------
##  Regression testing
##-----------------------------------------------------------------------------
##  * Tests need to be launched from a script, because CTest is not able
##    to add the ITK DLL directory to the path.  We use CMAKE as scripting
##    language to avoid requirement of outside package (e.g. python)
##  * "Release" is hard coded.  There is (apparently) no easy way to
##    avoid this (i.e. no good suggestions on CMake/CTest email list.
##    This is the purpose of the PATH_HACK code below.
##-----------------------------------------------------------------------------
if (WIN32 AND NOT CYGWIN AND NOT MINGW)
  set (PLM_PLASTIMATCH_PATH
    ${PLM_BINARY_DIR}/Release)
  set (PLM_PLASTIMATCH_TESTING_PATH
    ${PLM_BINARY_DIR}/Testing/Release)
  set (PLM_FFTW_PATH ${FFTW_DIR})

  if (NOT "${ITK_DLL_DIR}" STREQUAL "")
    set (PLM_ITK_LIBRARY_PATH_HACK -DITK_LIBRARY_PATH=${ITK_DLL_DIR})
  else ()
    set (PLM_ITK_LIBRARY_PATH_HACK "")
  endif ()
  set (PLM_PLASTIMATCH_PATH_HACK
     -DPLM_PLASTIMATCH_PATH=${PLM_PLASTIMATCH_PATH})
  set (PLM_DCMTK_PATH_HACK -DPLM_DCMTK_PATH=${DCMTK_DIR}/bin/Release)
  set (PLM_FFTW_PATH_HACK
     -DPLM_FFTW_PATH=${PLM_FFTW_PATH})
else ()
  set (PLM_PLASTIMATCH_PATH ${PLM_BINARY_DIR})
  set (PLM_PLASTIMATCH_TESTING_PATH ${PLM_BINARY_DIR}/Testing)
  set (PLM_FFTW_PATH "${FFTW_DIR}")
  set (PLM_ITK_LIBRARY_PATH_HACK "")
  set (PLM_PLASTIMATCH_PATH_HACK "")
  set (PLM_DCMTK_PATH_HACK "")
  set (PLM_FFTW_PATH_HACK "")
endif ()

macro (PLM_ADD_TEST PLM_TEST_NAME PLM_TEST_COMMAND PARMS)
  # Optional extra parameters are passed through ${ARGN}
  # Allowed EXTRA_PARMS are:
  # "-DWORKING_DIR=XXX;-DEXPECTED_ERRNO=XXX"
  set (EXTRA_PARMS ${ARGV3})

  # CMake doesn't allow "=" to be passed in a -D parameter.  So we substitute
  # with replacement string, which will get substituted back within the
  # cmake script itself
  string (REPLACE "=" "&equal&" TMP_PARMS "${PARMS}")
  add_test (${PLM_TEST_NAME} ${CMAKE_COMMAND}
    -DPLM_TEST_NAME=${PLM_TEST_NAME}
    ${PLM_ITK_LIBRARY_PATH_HACK}
    ${PLM_PLASTIMATCH_PATH_HACK}
    ${PLM_DCMTK_PATH_HACK}
    ${PLM_FFTW_PATH_HACK}
    -DPLM_TEST_COMMAND=${PLM_TEST_COMMAND}
    -DPLM_TESTING_SOURCE_DIR=${PLM_TESTING_SOURCE_DIR}
    -DPLM_BUILD_TESTING_DIR=${PLM_BUILD_TESTING_DIR}
    "-DPARMS=${TMP_PARMS}"
    ${EXTRA_PARMS}
    -P ${PLM_SOURCE_DIR}/cmake/RUN_CTEST.cmake
    )
# message (STATUS "${CMAKE_COMMAND} -DPLM_TEST_NAME=${PLM_TEST_NAME} ${PLM_ITK_LIBRARY_PATH_HACK} ${PLM_PLASTIMATCH_PATH_HACK} ${PLM_FFTW_PATH_HACK} -DPLM_TEST_COMMAND=${PLM_TEST_COMMAND} -DPLM_TESTING_SOURCE_DIR=${PLM_TESTING_SOURCE_DIR} -DPLM_BUILD_TESTING_DIR=${PLM_BUILD_TESTING_DIR} \"-DPARMS=${TMP_PARMS}\" ${EXTRA_PARMS} -P ${PLM_SOURCE_DIR}/cmake/RUN_CTEST.cmake")
endmacro ()

# Figure out which tests to ignore
# Note: we need two copies, because unix users might run "make test"
# in either directory.  MSVC uses the one in ${CMAKE_BINARY_DIR}.
configure_file (
  ${PLM_TESTING_SOURCE_DIR}/CTestCustom.cmake.in
  ${PLM_BINARY_DIR}/CTestCustom.cmake
  @ONLY)
configure_file (
  ${PLM_TESTING_SOURCE_DIR}/CTestCustom.cmake.in
  ${PLM_BINARY_DIR}/Testing/CTestCustom.cmake
  @ONLY)

if (PLM_BUILD_TESTING)
  enable_testing ()
  add_subdirectory (Testing)

  # Copy the lconv script
  if (EXISTS "${PLM_SOURCE_DIR}/extra/devtools/run_lcov.sh")
    configure_file (
      "${PLM_SOURCE_DIR}/extra/devtools/run_lcov.sh"
      "${PLM_BINARY_DIR}/run_lcov.sh"
      COPYONLY)
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  Packaging
##-----------------------------------------------------------------------------
##  Simple version: "make package_source" to create tarball
##
##  Advanced version: to generate...
##   * zip    : cpack -G ZIP <source tree>
##   * tar.bz2: cpack -G TBZ2 <source tree>
##
##    ...from within your build directory
##-----------------------------------------------------------------------------
# Choose generator
if (WIN32 AND PLM_PACKAGE_WIX)
  set (CPACK_GENERATOR "WIX")
elseif (PLM_PACKAGE_NSIS)
  set (CPACK_GENERATOR "NSIS")
else ()
  set (CPACK_GENERATOR "ZIP")
endif ()
set (CPACK_SOURCE_GENERATOR "TBZ2")

# For WiX install, and possibly for debian -dev targets,
# we should properly set up the install components.
# Until then, let's just kill it
set (CPACK_MONOLITHIC_INSTALL 1)

# General stuff
set (CPACK_INSTALL_PREFIX "")
set (CPACK_PACKAGE_CONTACT "plastimatch@googlegroups.com")
set (CPACK_PACKAGE_NAME "plastimatch")
set (CPACK_PACKAGE_INSTALL_DIRECTORY "Plastimatch")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Plastimatch - A Medical Imaging Application")
set (CPACK_PACKAGE_VENDOR "Plastimatch Dev Team")
set (CPACK_PACKAGE_VERSION_MAJOR "${PLM_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${PLM_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${PLM_VERSION_PATCH}")
set (CPACK_PACKAGE_VERSION_TWEAK "${PLM_VERSION_TWEAK}")
set (CPACK_SOURCE_IGNORE_FILES
  "/\\\\.git"
  "/\\\\.svn"
  "~$"
)

# NSIS stuff
#set(CPACK_NSIS_INSTALLED_ICON_NAME "${APP_LOW_NAME}.ico")
set (CPACK_NSIS_HELP_LINK "http://plastimatch.org")
set (CPACK_NSIS_URL_INFO_ABOUT "http://plastimatch.org")
set (CPACK_NSIS_CONTACT "plastimatch@googlegroups.com")
set (PLM_NSIS_VERSION_STRING "${PLM_VERSION_MAJOR}.${PLM_VERSION_MINOR}.${PLM_VERSION_PATCH} (${PLASTIMATCH_REPOSITORY_VERSION})")

# WIX stuff
set (CPACK_WIX_HELP_LINK "http://plastimatch.org")
set (CPACK_WIX_UPGRADE_GUID "AA7C7964-14D7-4890-9CD1-EA1D80E4DC8C")
set (CPACK_WIX_LICENSE_RTF  "${PLM_WINDOWS_INSTALL_DIR}/License.rtf")
set (CPACK_WIX_UI_BANNER  "${PLM_WINDOWS_INSTALL_DIR}/plm_logo.jpg")
set (CPACK_WIX_UI_DIALOG  "${PLM_WINDOWS_INSTALL_DIR}/plm_logo_GUI.jpg")

set (CPACK_WIX_EXTRA_SOURCES
    "${PLM_SOURCE_DIR}/extra/wix/cmake_extra_dialog.wxs"
    "${PLM_SOURCE_DIR}/extra/wix/install_dir.wxs"
    "${PLM_SOURCE_DIR}/extra/wix/start_menu.wxs"
    )
set (CPACK_WIX_PATCH_FILE
    "${PLM_SOURCE_DIR}/extra/wix/patch_path_env.xml"
#    "${PLM_SOURCE_DIR}/extra/wix/patch_start_menu.xml"
    )
set (CPACK_WIX_UI_REF "CMakeUI_InstallDir")
if (CPACK_GENERATOR STREQUAL "WIX")
  set (CPACK_PACKAGE_NAME "Plastimatch")
  if (CPACK_PACKAGE_VERSION_TWEAK AND CPACK_PACKAGE_VERSION_TWEAK GREATER "0")
    set (CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${CPACK_PACKAGE_VERSION_TWEAK}")
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      set (CPACK_SYSTEM_NAME "win64")
    else ()
      set (CPACK_SYSTEM_NAME "win32")
    endif ()
    set (CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}")
  endif ()
  set_property (INSTALL "bin/launch_cmd_prompt.bat"
    PROPERTY CPACK_START_MENU_SHORTCUTS "Plastimatch Command Prompt")
endif ()

# ZIP stuff
if (CPACK_GENERATOR STREQUAL "ZIP")
  set (CPACK_SET_DESTDIR ON)
endif ()

include (CPack)
