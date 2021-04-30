.. _batch_registration:

Batch registration
==================
In order to run registration in batch, you must write a script.
Because I am a perl guy, this example is written in perl.::

  #!/usr/bin/perl
  # This example shows how to register a group of images in batch

  # This is a template for the command file.  Each registration will
  # substitute the strings FIXED, MOVING, WARPED, and VF.
  $template = <<EODATA
  [GLOBAL]
  fixed=FIXED
  moving=MOVING
  img_out=WARPED
  vf_out=VF
  [STAGE]
  xform=bspline
  res=1 1 1
  grid_spac=4 4 4
  regularization_lambda=0.1
  max_iterations=11
  [STAGE]
  max_iterations=11
  EODATA
    ;
  
  # Create an array with the filenames of the moving and fixed images
  # In this case, there is a single moving image (00001.nrrd) and a bunch
  # of fixed images (00001.nrrd through 00500.nrrd)
  @moving_images = ( "00001.nrrd" );
  @fixed_images = <000*.nrrd>;

  # Loop through both lists
  for $f (@fixed_images) {
    for $m (@moving_images) {

      # Get the "number" portion of the filenames
      $f =~ m/([0-9]*)/;
      $f_image_no = $1;
      $m =~ m/([0-9]*)/;
      $m_image_no = $1;
  
      # Create filenames for the output files
      $warped = "${f_image_no}_${m_image_no}.nrrd";
      $vf = "${f_image_no}_${m_image_no}_vf.nrrd";
  
      # Substitute the image filenames into the template
      $t = $template;
      $t =~ s/FIXED/$f/;
      $t =~ s/MOVING/$m/;
      $t =~ s/WARPED/$warped/;
      $t =~ s/VF/$vf/;
  
      # Write the command file
      open FH, ">", "parameters.txt";
      print FH $t;
      close FH;
  
      # Run plastimatch on the command file
      $cmd = "plastimatch register ${pfn}";
      system ($cmd);
    }
  }
