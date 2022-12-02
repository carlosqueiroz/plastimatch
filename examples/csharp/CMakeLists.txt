##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_csharp LANGUAGES CSharp)
cmake_minimum_required(VERSION 3.25)

message (STATUS "CSharp example!")
add_executable(csharp-example csharp-example.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/Gamma_dose_comparison.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_FloatImageType__Pointer.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_Gamma_dose_comparison_private.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_Plm_image.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_Plm_image__Pointer.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_UCharImageType__Pointer.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_f_float__void.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/SWIGTYPE_p_std__string.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/plastimatch.cs
  C:/Users/gcs6/build/x64/plastimatch/csharp/plastimatchPINVOKE.cs
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
