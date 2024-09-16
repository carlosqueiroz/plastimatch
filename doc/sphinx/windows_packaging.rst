Windows packaging
=================
This section describes the recommended build configuration for 
building official plastimatch windows binaries.

You should have a working virtual machine, as described in :ref:`windows_vm`.

As of plastimatch 1.10.0, the following build tools are used::

  QEMU/KVM        3.1               (Debian host)
  MS Windows      10                (Windows 10 Pro)
  Visual Studio   14.0.25431.01     (Aka MSVC Community 2015 Update 3)
  CMake           3.24.1
  WiX             3.11
  
And the following third party libraries are used::

  CUDA            8.0 GA2           (64 bit only)
  DCMTK           3.6.8             (Build separate 32 & 64 bit)
  FFTW            3.3.5             (Download separate 32 & 64 bit)
  ITK             5.3.0             (Build separate 32 & 64 bit)

NVIDIA deprecated support for 32-bit CUDA as of version 7.0.
Therefore, plastimatch does not offer CUDA in its 32-bit
binary download.

#. Build DCMTK

   #. Use fresh build directory
   #. Set BUILD_APPS to OFF
   #. Set BUILD_SHARED_LIBS to ON
   #. Set DCMTK_OVERWRITE_WIN32_COMPILE_FLAGS to OFF
   #. Make

#. Build ITK
   
   #. Use fresh build directory
   #. Set BUILD_SHARED_LIBS to ON
   #. Set BUILD_STATIC_LIBS to OFF
   #. Set Module_ITKReview to ON
   #. Add and set ITK_SKIP_PATH_LENGTH_CHECKS to ON
   #. Set CMAKE_INSTALL_PREFIX %USERPROFILE%/install/{32,64}/ITK-{version}
   #. Make and install

#. Build plastimatch

   #. Use fresh build directory
   #. Set GIT_EXECUTABLE to "git" or "wsl git" as appropriate
   #. Set BUILD_SHARED to ON
   #. For 32-bit build, set PLM_CONFIG_ENABLE_CUDA to OFF
   #. For 64-bit build, set PLM_CUDA_ALL_DEVICES to ON

#. Run test cases, make sure all pass
#. Build PACKAGE
#. Test package on fresh VM
#. Upload to sourceforge::

     sftp gregsharp@frs.sourceforge.net
     cd /home/pfs/project/plastimatch/Windows\ Binaries
     lcd ~/build/32/plastimatch-1.8.0
     put Plastimatch-1.8.0-win32.msi
     lcd ~/build/64/plastimatch-1.8.0
     put Plastimatch-1.8.0-win64.msi
