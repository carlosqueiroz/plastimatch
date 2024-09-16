Welcome to plastimatch
======================

Plastimatch is an open source software for image computation.  
Our main focus is high-performance 
volumetric registration of medical images, such as 
X-ray computed tomography (CT), magnetic resonance imaging (MRI), 
and positron emission tomography (PET). 

:ref:`getting_started`

Software features include:

* B-spline method for deformable image registration 
  (GPU and multicore accelerated)
* Analytic regularization for B-spline registration
* Landmark penalty function for B-spline registration
* Demons method for deformable image registration 
  (GPU accelerated)
* ITK-based algorithms for translation, rigid, affine,
  demons, and B-spline registration
* Pipelined, multi-stage registration framework with seamless conversion 
  between most algorithms and transform types
* Landmark-based deformable registration using thin-plate splines for 
  global registration
* Landmark-based deformable registration using radial basis functions 
  for local corrections
* Tools for converting and manipulating vector fields and other 
  geometric transforms
* Broad support for 3D image file formats (using ITK), including 
  DICOM, Nifti, NRRD, MetaImage, and Analyze
* DICOM and DICOM-RT import and export
* XiO import and export
* Plugin to 3D Slicer

Plastimatch also features other handy utilities which are not
directly related to image registration:

* FDK cone-beam CT reconstruction (GPU and multicore accelerated)
* Digitally reconstructed radiograph (DRR) generation
  (GPU and multicore accelerated)
* A DICOM screen capture utility "Mondoshot"

Plastimatch lacks the following:

* Landmark-based rigid registration
* Viscous fluid registration
* FEM registration
* Surface matching registration

Reg-2-3 is no longer included in the plastimatch download.
Please see 
<https://www.open-radart.org/cms/index.php/sorry>
or
<https://gitlab.com/plastimatch/reg-2-3>

Acknowledgments:

* An Ira J Spiro translational research grant (2009)
* NIH / NCI 6-P01CA21239
* The Federal share of program income earned by MGH on C06CA059267
* Progetto Rocca Foundation -- 
  A collaboration between MIT and Politecnico di Milano 
* The National Alliance for Medical Image
  Computing (NAMIC), funded by the National Institutes of Health
  through the NIH Roadmap for Medical Research, Grant 2-U54-EB005149
* NSF ERC Innovation Award EEC-0946463
* NSF / CISE 1642345
* NSF / SMOR 1662852
* Hardware grant from NVIDIA Corporation

For more information:

* `Gitlab project page (for git) <https://gitlab.com/plastimatch/plastimatch>`_
* `Sourceforge project page (for files) <https://sourceforge.net/projects/plastimatch/>`_
* `Plastimatch license (BSD-style) <https://gitlab.com/plastimatch/plastimatch/blob/master/src/plastimatch/LICENSE.TXT>`_
* `Doxygen </doxygen>`_
* `E-mail list <http://groups.google.com/group/plastimatch>`_

Documentation:

.. toctree::
   :maxdepth: 1

   contents

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`search`
