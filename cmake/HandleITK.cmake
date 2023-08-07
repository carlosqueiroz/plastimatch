##-----------------------------------------------------------------------------
##  HandleITK.cmake
##    Check ITK version and optional components
##    Include use file (for registering IO factories)
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
include (CheckIncludeFileCXX)

# 2023-08-07.  On Debian, ITK 5.3.1, ITK sets the CMAKE_CXX_STANDARD to 14
# which conflicts with DCMTK requirement that CMAKE_CXX_STANDARD be 17
push_var (CMAKE_CXX_STANDARD)

# GCS 2017-12-14 On older ITK version, the use file sets variables such
# as DCMTK_FOUND, DCMTK_DIR.  Needs more investigation.
include (${ITK_USE_FILE})

pop_var (CMAKE_CXX_STANDARD)

if (NOT ITK_VERSION)
  set (ITK_VERSION
    "${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR}.${ITK_VERSION_PATCH}")
endif ()

if (${ITK_VERSION} VERSION_LESS "4.1")
  message (FATAL_ERROR 
    "Fatal Error. ITK must be 4.1 or greater")
endif ()

message (STATUS "ITK_VERSION = ${ITK_VERSION} found")

# Find ITK DLL directory.  This is used on Windows for both regression testing
# and packaging.
if (NOT ITK_FOUND)
  set (ITK_BASE "${PLM_BINARY_DIR}/ITK-build")
elseif (${ITK_VERSION} VERSION_LESS "4.1")
  set (ITK_BASE "${ITK_LIBRARY_DIRS}")
else ()
  # At some point in time (presumably around ITK 4.1), ITK stopped
  # creating the variable ITK_LIBRARY_DIRS.  Therefore, we infer from the 
  # configuration filename.
  # Remove filename
  string (REGEX REPLACE "/[^/]*$" "" ITK_LIBRARY_DIRS_41
    "${ITK_CONFIG_TARGETS_FILE}")
  # If configuring against installation directory, walk up to base directory
  string (REGEX REPLACE "/lib/cmake/ITK-.*]*$" "" ITK_LIBRARY_DIRS_41
    "${ITK_LIBRARY_DIRS_41}")
  set (ITK_BASE "${ITK_LIBRARY_DIRS_41}")
endif ()

# Fix VXL problem on newer GCC compilers
if (NOT ${ITK_VERSION} VERSION_GREATER "4.13.2"
    AND CMAKE_COMPILER_IS_GNUCC
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "8.3")
  add_definitions (-D__GNUC__=8 -D__GNUC_MINOR__=3)
endif ()

# Look for itkVectorCentralDifferenceImageFunction.h
# It is not enabled by default in ITK 5.1.0
push_vars ("CMAKE_REQUIRED_INCLUDES")
unset (HAVE_ITK_VECTOR_CD CACHE)
set (CMAKE_REQUIRED_INCLUDES "${ITK_INCLUDE_DIRS}")
check_include_file_cxx ("itkVectorCentralDifferenceImageFunction.h" HAVE_ITK_VECTOR_CD)
pop_vars ("CMAKE_REQUIRED_INCLUDES")

if (${ITK_VERSION} VERSION_LESS "5")
  # See if itkArray.h can be included.  ITK 4 is broken on Fedora with gcc 10 compiler.
  # There is no way to fix this.
  set (ITK_TEST_SOURCE
    "#include <itkArray.h>
     int main (int argc, char* argv[]) {return 0;}")
  push_vars ("CMAKE_REQUIRED_INCLUDES")
  set (CMAKE_REQUIRED_INCLUDES ${ITK_INCLUDE_DIRS})
      set (CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_compiles ("${ITK_TEST_SOURCE}" ITK_IS_OK)
  if (NOT ITK_IS_OK)
    message (STATUS "ITK is broken. Sorry.")
    set (ITK_FOUND OFF)
  endif ()
  pop_vars ("CMAKE_REQUIRED_INCLUDES")
endif()

push_vars ("CMAKE_REQUIRED_INCLUDES")
unset (HAVE_ITK_VECTOR_CD CACHE)
set (CMAKE_REQUIRED_INCLUDES "${ITK_INCLUDE_DIRS}")
check_include_file_cxx ("itkVectorCentralDifferenceImageFunction.h" HAVE_ITK_VECTOR_CD)
pop_vars ("CMAKE_REQUIRED_INCLUDES")

# For ITK 4, we need to override the default itkContourExtractor2DImageFilter
# because it doesn't compile with newer gcc versions
if ("${ITK_VERSION}" VERSION_LESS "5")
  set (PLM_USE_PATCHED_CONTOUR_EXTRACTOR ON)
else ()
  set (PLM_USE_PATCHED_CONTOUR_EXTRACTOR OFF)
endif ()

message (STATUS "ITK_BASE = ${ITK_BASE}")
if (NOT WIN32)
  set (ITK_DLL_DIR "")
elseif (IS_DIRECTORY "${ITK_BASE}/bin/Release")
  set (ITK_DLL_DIR "${ITK_BASE}/bin/Release")
elseif (IS_DIRECTORY "${ITK_BASE}/Release")
  set (ITK_DLL_DIR "${ITK_BASE}/Release")
elseif (IS_DIRECTORY "${ITK_BASE}/bin")
  set (ITK_DLL_DIR "${ITK_BASE}/bin")
else ()
  set (ITK_DLL_DIR "")
endif ()
