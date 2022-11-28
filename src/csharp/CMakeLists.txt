##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_csharp LANGUAGES CSharp)

message (STATUS "CSharp example!")
add_executable(csharp-example csharp-example.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_FloatImageType__Pointer.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_Gamma_dose_comparison_private.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_Plm_image.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_Plm_image__Pointer.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_UCharImageType__Pointer.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_f_float__void.cs
  ${CMAKE_BINARY_DIR}/csharp/SWIGTYPE_p_std__string.cs
  ${CMAKE_BINARY_DIR}/csharp/gamma_dose_comparisonCSHARP_wrap.cxx
  ${CMAKE_BINARY_DIR}/csharp/plastimatch.cs
  ${CMAKE_BINARY_DIR}/csharp/plastimatchPINVOKE.cs
)

target_link_libraries (csharp-example PUBLIC 
  plmutil-cs plmutil plmbase plmsys)
if (TARGET DCMTK::dcmdata)
  target_link_libraries (csharp-example PUBLIC DCMTK::dcmdata)
else ()
  target_link_libraries (csharp-example PUBLIC dcmdata)
endif ()
target_link_libraries (csharp-example PUBLIC "${ITK_LIBRARIES}")

