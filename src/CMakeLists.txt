##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src)

##-----------------------------------------------------------------------------
##  Add subdirectories
##-----------------------------------------------------------------------------
if (PLM_CONFIG_DEBIAN_BUILD)
  set (PLM_CONFIG_ENABLE_FATM OFF)
endif ()

if (PLM_CONFIG_ENABLE_FATM)
  add_subdirectory (fatm)
endif ()

##-----------------------------------------------------------------------------
##  CMake include files
##-----------------------------------------------------------------------------
set (CMAKE_MODULE_PATH "${PLM_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include (PlmMacros)

##-----------------------------------------------------------------------------
##  Processor and OS characteristics
##    32-bit or 64-bit machine
##    Endian-ness
##    Machine precision
##    Processor type
##-----------------------------------------------------------------------------
include (CheckCharSign)
include (CheckEpsilon)
# No need to check this yet.  Doing so may force C++11 compiler flag,
# which causes build error on earlier version of ITK.
#include (CheckNullptr)
include (CheckTypeSize)
include (TestBigEndian)

check_type_size ("unsigned int" CMAKE_SIZEOF_UINT)
check_type_size ("unsigned long" CMAKE_SIZEOF_ULONG)
check_type_size ("unsigned long long" CMAKE_SIZEOF_ULONGLONG)
check_type_size ("size_t" CMAKE_SIZEOF_SIZE_T)
if (NOT APPLE)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set (MACHINE_IS_32_BIT TRUE)
    set (MACHINE_IS_64_BIT FALSE)
    message (STATUS "Machine is 32-bit")
  else ()
    set (MACHINE_IS_32_BIT FALSE)
    set (MACHINE_IS_64_BIT TRUE)
    message (STATUS "Machine is 64-bit")
  endif ()
endif ()

test_big_endian (PLM_BIG_ENDIAN)
check_epsilon (MACHINE_EPS)
check_char_sign (CHAR_SIGN)
message (STATUS "Checking host processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message (STATUS "Checking target processor: ${CMAKE_SYSTEM_PROCESSOR}")
message (STATUS "Checking epsilon: ${MACHINE_EPS}")
message (STATUS "Checking sign of char: ${CHAR_SIGN}")

include (CheckIncludeFiles)
check_include_files ("stdint.h" HAVE_STDINT_H)
check_include_files ("sys/stat.h" HAVE_SYS_STAT_H)

##-----------------------------------------------------------------------------
##  DCMTK
##-----------------------------------------------------------------------------
find_package (DCMTK)
set (PLM_DCM_USE_DCMTK 0)
if (DCMTK_FOUND)
  set (PLM_DCM_USE_DCMTK 1)
endif ()

##-----------------------------------------------------------------------------
##  dlib
##-----------------------------------------------------------------------------
if (PLM_PREFER_SYSTEM_DLIB)
  # Use this version if you want to use the internal cmake find.
  # The internal version has issue finding the correct BLAS library
  # for Debian, so we prefer Debian version.
  # find_package (Dlib)

  # A previous version of the Debian cmake find was broken in a way that 
  # could be worked around by setting dlib_BINARY_DIR.  However, modern
  # dlibConfig.cmake scripts do not have this problem.
  ##  set (dlib_BINARY_DIR 1)
  find_package (dlib QUIET)
endif ()
set (DLIB_HAVE_LIBRARY FALSE)
if (dlib_FOUND)
  set (DLIB_INCLUDE_DIR ${dlib_INCLUDE_DIR})
  #set (DLIB_LIBRARIES ${dlib_LIBRARIES})
  set (DLIB_LIBRARIES dlib::dlib)
  set (DLIB_FOUND TRUE)
  if (DLIB_LIBRARIES)
    set (DLIB_HAVE_LIBRARY TRUE)
  endif ()
else ()
  set (DLIB_INCLUDE_DIR "${PLM_SOURCE_DIR}/libs/dlib-19.1")
  set (DLIB_LIBRARIES "")
endif ()

##-----------------------------------------------------------------------------
##  ransac
##-----------------------------------------------------------------------------
set (RANSAC_INCLUDE_DIRS
  "${PLM_SOURCE_DIR}/libs/ransac"
  "${PLM_SOURCE_DIR}/libs/ransac/Common")

##-----------------------------------------------------------------------------
##  ITK
##-----------------------------------------------------------------------------
if (PLM_USING_SUPERBUILD)
  find_package (ITK REQUIRED
    PATHS "/usr/lib64/cmake/InsightToolkit")
  include (HandleITK)
  include (HandleVXL)
endif ()

##-----------------------------------------------------------------------------
##  Hack for superbuild
##-----------------------------------------------------------------------------
#link_directories (${PLM_BINARY_DIR})

##-----------------------------------------------------------------------------
##  Libraries
##-----------------------------------------------------------------------------
set (PLASTIMATCH_LIBS
  plmclp
#  plmscript
  plmsegment
  plmregister
  plmreconstruct
  plmdose
  plmutil
  plmbase
  plmsys
  ${ITK_LIBRARIES}
  devillard
  nkidecompress
#  lua
  ${MATH_LIB}
  )

#if (CUDA_FOUND AND NOT PLM_USE_GPU_PLUGINS)
if (CUDA_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} plmcuda)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} plmutilcuda)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} plmreconstructcuda)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} plmregistercuda)
endif ()

if (DCMTK_FOUND)
  if (TARGET DCMTK::dcmdata)
    set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} DCMTK::dcmdata)
  else ()
    set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${DCMTK_LIBRARIES})
  endif ()
endif ()

if (DLIB_LIBRARIES)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${DLIB_LIBRARIES})
endif ()

if (FFTW_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${FFTW_LIBRARIES})
endif ()

if (LIBDL_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} dl)
endif ()

if (nlopt_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${nlopt_LIBRARIES})
endif ()

if (OpenCL_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} plmopencl)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${OpenCL_LIBRARIES})
endif ()

if (OPENMP_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${OPENMP_LIBRARIES})
endif ()

if (SPECFUN_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} specfun)
endif ()

### Let QT applications add this themselves
#if (QT4_FOUND)
#  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${QT_LIBRARIES})
#endif ()

##-----------------------------------------------------------------------------
##  Linker flags
##-----------------------------------------------------------------------------
set (PLASTIMATCH_LDFLAGS "${OPENMP_LDFLAGS}")
if (PLM_USE_GPU_PLUGINS AND WIN32 AND NOT CYGWIN AND NOT MINGW)
  set (PLASTIMATCH_LDFLAGS
 "${PLASTIMATCH_LDFLAGS} /DELAYLOAD:plmutilcuda.dll /DELAYLOAD:plmregistercuda.dll /DELAYLOAD:plmreconstructcuda.dll")
endif ()

if (PLM_CONFIG_NOMANIFEST AND WIN32)
  set (PLASTIMATCH_LDFLAGS "${PLASTIMATCH_LDFLAGS} /MANIFEST:NO")
endif ()

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------

# ITK compatibility shims
if (NOT HAVE_ITK_VECTOR_CD)
  message (STATUS "Appending ${CMAKE_SOURCE_DIR}/libs/itk-5.1.0")
  include_directories (AFTER "${CMAKE_SOURCE_DIR}/libs/itk-5.1.0")
else ()
  message (STATUS "Not appending ${CMAKE_SOURCE_DIR}/libs/itk-5.1.0")
endif ()
if (("${ITK_VERSION}" VERSION_LESS "4.13") OR (NOT ${PLM_PREFER_NEW_ITK_DEMONS}))
  set (PLM_USE_NEW_ITK_DEMONS OFF)
  include_directories (AFTER "${PLM_SOURCE_DIR}/libs/demons_itk_insight")
  include_directories (AFTER ${PLM_SOURCE_DIR}/libs/demons_itk_insight/DiffeomorphicDemons)
  include_directories (AFTER ${PLM_SOURCE_DIR}/libs/demons_itk_insight/FastSymmetricForces)
  include_directories (AFTER ${PLM_SOURCE_DIR}/libs/demons_itk_insight/LOGDomainDemons)
else ()
  set (PLM_USE_NEW_ITK_DEMONS ON)
  include_directories (AFTER "${PLM_SOURCE_DIR}/libs/itk-4.13.2-demons")
  include_directories (AFTER "${PLM_SOURCE_DIR}/libs/LogSymmetricDemons")
endif ()
if ("${ITK_VERSION}" VERSION_LESS "5")
  message (STATUS "Appending ${CMAKE_SOURCE_DIR}/libs/itk-5.1.2")
  include_directories (BEFORE "${PLM_SOURCE_DIR}/libs/itk-5.1.2")
else ()
  message (STATUS "Not appending ${CMAKE_SOURCE_DIR}/libs/itk-5.1.2")
endif ()

include_directories (AFTER ${PLM_SOURCE_DIR}/libs/nSIFT)
include_directories (AFTER ${DLIB_INCLUDE_DIR})
include_directories (AFTER ${libLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
include_directories (AFTER ${RANSAC_INCLUDE_DIRS})

if (DCMTK_FOUND)
  include_directories (BEFORE ${DCMTK_INCLUDE_DIRS})
  if (DEFINED DCMTK_CMAKE_INSTALL_LIBDIR)
    link_directories ("${DCMTK_DIR}/${DCMTK_CMAKE_INSTALL_LIBDIR}")
  endif ()
endif ()
if (FFTW_FOUND)
  include_directories (BEFORE ${FFTW_INCLUDE_DIR})
  link_directories (${FFTW_DIR})
endif ()
if (MATLAB_FOUND)
  include_directories (AFTER ${MATLAB_INCLUDE_DIRS})
endif ()
if (nlopt_FOUND)
  include_directories (AFTER ${nlopt_INCLUDE_DIR})
endif ()
if (OpenCL_FOUND)
  include_directories (BEFORE ${OpenCL_INCLUDE_DIRS})
endif ()
#if (QT4_FOUND)
#  if (QT_QTGUI_FOUND)
#    include_directories (AFTER ${QT_QTGUI_INCLUDE_DIR})
#  endif ()
#  if (QT_QTSQL_FOUND)
#    include_directories (AFTER ${QT_QTSQL_INCLUDE_DIR})
#  endif ()
#endif ()
if (RAPIDJSON_FOUND)
    include_directories (AFTER ${RAPIDJSON_INCLUDE_DIR})
endif ()
if (SQLite3_FOUND)
  include_directories (AFTER ${SQLite3_INCLUDE_DIR})
endif ()

##-----------------------------------------------------------------------------
##  CONFIGURE INCLUDE FILES
##-----------------------------------------------------------------------------
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/sys/plm_config.h.in
  ${PLM_BINARY_DIR}/plm_config.h)
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/sys/plm_version.h.in
  ${PLM_BINARY_DIR}/plm_version.h)

##-----------------------------------------------------------------------------
## Option to generate .clang_complete for hip Vim users using clang_complete
##-----------------------------------------------------------------------------
if (PLM_CONFIG_CLANG_COMPLETE)
  get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  set (file_clang_complete "${CMAKE_SOURCE_DIR}/src/.clang_complete")
  foreach (arg ${inc_dirs})
    set (inc_args "${inc_args} -I${arg}\n")
  endforeach ()
  file (WRITE "${file_clang_complete}" "${inc_args}")
  message (STATUS "Generated ${file_clang_complete}")
endif ()

##-----------------------------------------------------------------------------
##  Do we have prerequisites for viscous code?
##-----------------------------------------------------------------------------
set (PLM_BUILD_VISCOUS FALSE)
if (PLM_CONFIG_ENABLE_VISCOUS
    AND CUDA_FOUND AND CUDA_SDK_ROOT_DIR AND THRUST_FOUND)
  set (PLM_BUILD_VISCOUS TRUE)
endif ()


##  Specify which libraries of plastimatch should be built
##  Support libraries (base, util, opencl, cuda, etc. are not optional)
if (PLM_CONFIG_LIBRARY_BUILD)
else ()
  set (PLMLIB_CONFIG_ENABLE_CLI true)
  set (PLMLIB_CONFIG_ENABLE_CLP true)
  set (PLMLIB_CONFIG_ENABLE_DOSE true)
  set (PLMLIB_CONFIG_ENABLE_QT true)
  set (PLMLIB_CONFIG_ENABLE_RECONSTRUCT true)
  set (PLMLIB_CONFIG_ENABLE_REGISTER true)
#  set (PLMLIB_CONFIG_ENABLE_SCRIPT true)
  set (PLMLIB_CONFIG_ENABLE_SEGMENT true)
  set (PLMLIB_CONFIG_ENABLE_STANDALONE true)
  set (PLMLIB_CONFIG_ENABLE_TEST true)
endif ()

##-----------------------------------------------------------------------------
##  SETUP IMPORTANT LOCATIONS
##-----------------------------------------------------------------------------
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


## Specify which include directories are needed when
## compiling code that links to the libraries
set (PLASTIMATCH_INCLUDE_DIRECTORIES "")
list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
  $<INSTALL_INTERFACE:${PLM_INSTALL_INCLUDE_DIR}>
  )
list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${PLM_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/sys>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/base>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/util>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/libs/devillard>
  $<BUILD_INTERFACE:${DLIB_INCLUDE_DIR}>
  )
if (CUDA_FOUND)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cuda>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/util/cuda>
    )
endif ()
if (OPENCL_FOUND)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/opencl>
    )
endif ()
if (PLMLIB_CONFIG_ENABLE_REGISTER)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/register>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/libs/liblbfgs-1.9/include>
    )
  if (CUDA_FOUND)
    list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/register/cuda>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/libs/liblbfgs-1.9/include>
      )
  endif ()
endif ()
if (PLMLIB_CONFIG_ENABLE_SEGMENT)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/segment>
    )
endif ()
if (PLMLIB_CONFIG_ENABLE_DOSE)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/dose>
    )
endif ()
if (PLMLIB_CONFIG_ENABLE_RECONSTRUCT)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/reconstruct>
    )
  if (CUDA_FOUND)
    list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/reconstruct/cuda>
      )
  endif ()
endif ()

if (NOT PLM_CONFIG_LIBRARY_BUILD)
  list (APPEND PLASTIMATCH_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/clp>
    )
endif ()

# Core
add_subdirectory(base)
add_subdirectory(sys)
add_subdirectory(util)

# Optional
if (PLMLIB_CONFIG_ENABLE_CLI)
  add_subdirectory(cli)
endif ()
if (PLMLIB_CONFIG_ENABLE_CLP)
  add_subdirectory(clp)
endif ()
if (CUDA_FOUND)
  add_subdirectory(cuda)
endif ()
if (PLMLIB_CONFIG_ENABLE_DOSE)
  add_subdirectory(dose)
endif ()
if (OpenCL_FOUND)
  add_subdirectory(opencl)
endif ()
if (QT_FOUND AND PLMLIB_CONFIG_ENABLE_QT)
  add_subdirectory(qt)
endif ()
if (PLMLIB_CONFIG_ENABLE_RECONSTRUCT)
  add_subdirectory(reconstruct)
endif ()
if (PLMLIB_CONFIG_ENABLE_REGISTER)
  add_subdirectory(register)
endif ()
#if (PLMLIB_CONFIG_ENABLE_SCRIPT)
#  add_subdirectory(script)
#endif ()
if (PLMLIB_CONFIG_ENABLE_SEGMENT)
  add_subdirectory(segment)
endif ()
if (PLMLIB_CONFIG_ENABLE_STANDALONE)
  add_subdirectory(standalone)
endif ()
if (PLMLIB_CONFIG_ENABLE_TEST)
  add_subdirectory(test)
endif ()

# Create a list of targets to be exported.  These are used by applications
# which link to plastimatch libraries.
set (EXPORT_TARGET_LIST plmsys plmbase devillard nkidecompress plmutil)
if (PLMLIB_CONFIG_ENABLE_DOSE)
  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} specfun plmdose)
endif ()
if (PLMLIB_CONFIG_ENABLE_REGISTER)
  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} lbfgs plmregister)
endif ()
if (PLMLIB_CONFIG_ENABLE_RECONSTRUCT)
  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmreconstruct)
endif ()
if (CUDA_FOUND)
  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmcuda)
  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmutilcuda)
  if (PLMLIB_CONFIG_ENABLE_REGISTER)
    set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmregistercuda)
  endif ()
  if (PLMLIB_CONFIG_ENABLE_RECONSTRUCT)
    set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmreconstructcuda)
  endif ()
endif ()
if (OpenCL_FOUND)
  if (PLMLIB_CONFIG_ENABLE_REGISTER OR PLMLIB_CONFIG_ENABLE_RECONSTRUCT)
    set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmopencl)
  endif ()
endif ()
if (PLMLIB_CONFIG_ENABLE_SEGMENT)
  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} plmsegment)
endif ()
#if (PLMLIB_CONFIG_ENABLE_SCRIPT)
#  set (EXPORT_TARGET_LIST ${EXPORT_TARGET_LIST} lua plmscript)
#endif ()

# Create the configuration files used by client applications
## GCS: This is not working
include (CMakePackageConfigHelpers)

# Make the version file
write_basic_package_version_file (
  "${CMAKE_CURRENT_BINARY_DIR}/PlastimatchConfigVersion.cmake"
  VERSION ${PLASTIMATCH_VERSION_STRING}
  COMPATIBILITY AnyNewerVersion
  )

# Make the targets file
if (NOT ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 3.0)
  export (EXPORT PlastimatchLibraryDepends
    FILE "${PLM_BINARY_DIR}/PlastimatchLibraryDepends.cmake"
    )
endif ()

# Make the config file
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/PlastimatchConfig.cmake.in"
  "${PLM_BINARY_DIR}/PlastimatchConfig.cmake"
  @ONLY)

# Install the files
install (EXPORT PlastimatchLibraryDepends
  DESTINATION "${PLM_INSTALL_CMAKE_DIR}"
  )
install (FILES
  "${PLM_BINARY_DIR}/PlastimatchConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/PlastimatchConfigVersion.cmake"
  DESTINATION "${PLM_INSTALL_CMAKE_DIR}"
  COMPONENT Devel
  )

##-----------------------------------------------------------------------------
##  DOXYGEN
##-----------------------------------------------------------------------------
if (DOXYGEN_FOUND)
  file (MAKE_DIRECTORY "${PLM_BINARY_DIR}/doc")
  configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY)
  add_custom_target (doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY "${PLM_BINARY_DIR}/doc"
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif ()


##-----------------------------------------------------------------------------
##  Broken superbuild stuff
##-----------------------------------------------------------------------------
if (false)
sb_set (PLM_USING_SUPERBUILD OFF)
if (PLM_CONFIG_ENABLE_PLASTIMATCH)
  if (NOT ITK_FOUND OR NOT DCMTK_FOUND)
    if (PLM_CONFIG_ENABLE_SUPERBUILD)
      set (proj plastimatch)
      set (PLM_USING_SUPERBUILD ON)
      if(CMAKE_EXTRA_GENERATOR)
	set(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
      else()
	set(gen "${CMAKE_GENERATOR}")
      endif()

      set (sb_option_list "")
      foreach (_var ${sb_cmake_vars})
	list (APPEND sb_option_list "-D${_var}=${${_var}}")
      endforeach ()
      if (opt4D_FOUND)
	list (APPEND sb_option_list "-Dopt4D_DIR=${opt4D_DIR}")
      endif ()

      set (sb_dependencies devillard inih lbfgs nkidecompress specfun)
      if (NOT ITK_FOUND)
	list (APPEND sb_dependencies ITK)
      endif ()
      if (NOT DCMTK_FOUND AND PLM_CONFIG_ENABLE_DCMTK)
	list (APPEND sb_dependencies DCMTK)
      endif ()
      if (NOT SQLite3_FOUND)
	list (APPEND sb_dependencies sqlite3)
      endif ()

      ExternalProject_Add (${proj}
	DOWNLOAD_COMMAND ""
	INSTALL_COMMAND ""
	SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/plastimatch"
	BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/plastimatch"
	CMAKE_GENERATOR ${gen}
	CMAKE_ARGS
	-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
	-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
	-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
	-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
	-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
	-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
	-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
	-DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}
	-DITK_DIR:PATH=${ITK_DIR}
	-DDCMTK_DIR:PATH=${DCMTK_DIR}
	-DPLASTIMATCH_VERSION_STRING:STRING=${PLASTIMATCH_VERSION_STRING}
	${sb_option_list}
	DEPENDS
	${sb_dependencies}
	)
    else ()
      message (STATUS "Plastimatch will not be built (ITK or DCMTK not found; superbuild disabled)")
    endif ()
  else ()
    add_subdirectory (plastimatch)
  endif ()
endif ()
endif (false)
