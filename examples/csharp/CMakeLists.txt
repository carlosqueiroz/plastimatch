##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_csharp LANGUAGES CSharp)
cmake_minimum_required(VERSION 3.25)

message (STATUS "CSharp example!")

# Grab all the .cs files created by SWIG
set (PLM_CS_DIR "C:/Users/gcs6/build/x64/plastimatch/csharp")
file(GLOB SWIG_GENERATED_CS_FILES ${PLM_CS_DIR}/*.cs)

add_executable(csharp-example
  csharp-example.cs
  ${SWIG_GENERATED_CS_FILES}
)

find_package (DCMTK REQUIRED)
find_package (ITK REQUIRED)
link_directories ("C:/Users/gcs6/build/x64/plastimatch/Release")
target_link_libraries (csharp-example PUBLIC 
  plmutil-cs plmutil plmbase plmsys)
if (TARGET DCMTK::dcmdata)
  target_link_libraries (csharp-example PUBLIC DCMTK::dcmdata)
else ()
  target_link_libraries (csharp-example PUBLIC dcmdata)
endif ()
target_link_libraries (csharp-example PUBLIC "${ITK_LIBRARIES}")
