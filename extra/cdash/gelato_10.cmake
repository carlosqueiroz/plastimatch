set (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/build/nightly/src/plastimatch")
set (CTEST_BINARY_DIRECTORY "$ENV{HOME}/build/nightly/gelato_10/plastimatch")

set (CTEST_SITE "gelato-gcc-4.4.5")
set (CTEST_BUILD_NAME "g01-lin64-Pir-cd-static-memcheck")
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_CMAKE_GENERATOR "Unix Makefiles")
set (CTEST_BUILD_OPTIONS "-DCMAKE_BUILD_TYPE:STRING=Debug -DITK_DIR:PATH=$ENV{HOME}/build/itk-3.20.1-debug -DPLM_CONFIG_DISABLE_REG23:BOOL=ON")

#set (CTEST_MEMORYCHECK_COMMAND valgrind)
#set (CTEST_MEMORYCHECK_COMMAND_OPTIONS --trace-children=yes)
set (WITH_MEMCHECK ON)

##########

#ctest_empty_binary_directory (${CTEST_BINARY_DIRECTORY})

find_program (CTEST_SVN_COMMAND NAMES svn)
#find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
#set (CTEST_MEMORYCHECK_SUPPRESSIONS_FILE ${CTEST_SOURCE_DIRECTORY}/tests/valgrind.supp)

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  set (CTEST_CHECKOUT_COMMAND "${CTEST_SVN_COMMAND} --username anonymous --password \\\"\\\" https://forge.abcd.harvard.edu/svn/plastimatch/plastimatch/trunk ${CTEST_SOURCE_DIRECTORY}")
endif ()

set (CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}")
set (SVN_UPDATE_OPTIONS "--username anonymous --password \"\"")

set (CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}")
set (CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} -DWITH_TESTING:BOOL=ON ${CTEST_BUILD_OPTIONS}")
set (CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"-G${CTEST_CMAKE_GENERATOR}\"")
set (CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"${CTEST_SOURCE_DIRECTORY}\"")

ctest_start("Nightly")
#ctest_update()
#ctest_configure()

## This reputedly fixes bug where CTestCustom.cmake is ignored
## Cf. http://www.cmake.org/pipermail/cmake/2009-November/033320.html
ctest_read_custom_files ("${CTEST_BINARY_DIRECTORY}")

ctest_build()
ctest_test()
if (WITH_MEMCHECK AND CTEST_COVERAGE_COMMAND)
  ctest_coverage()
endif (WITH_MEMCHECK AND CTEST_COVERAGE_COMMAND)
if (WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND)
  ctest_memcheck()
endif (WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND)
#ctest_submit()
