amp = 2.0;  # amplitude in cm

delta = (2 * pi) / 10;
x = 0:delta:2*pi;
x = x(1:end-1);
x = 10 * amp * (cos(x)-1) / 2;

[status, for_uid] = system("/home/gcs6/build/plastimatch/dicom_uid");
[status, study_uid] = system("/home/gcs6/build/plastimatch/dicom_uid");
for_uid = for_uid(1:end-1);
study_uid = study_uid(1:end-1);

template = {
    '#Insight Transform File V1.0'
    '#Transform 0'
    'Transform: TranslationTransform_double_3_3'
    'Parameters: 0 0 %f'
    'FixedParameters: '
  };
template = sprintf ("%s\n", template{:});

system ("rm -rf dicom-*cm*");

for i = 1:length(x)
  xf_fn = sprintf ("xlat-%d.txt", (i-1))
  out_dir = sprintf ("dicom-%3.1fcm-%02d", amp, (i-1)*10)
  xlat_string = sprintf (template, x(i));
  description = sprintf ("\"Matrixx %3.1fcm %02d %%\"", amp, (i-1)*10);

  fp = fopen (xf_fn, "w");
  fprintf (fp, template, x(i));
  fclose (fp);

  command = sprintf ("plastimatch warp --input GBDAY_02 --output-dicom %s --xf %s --patient-name \"GBTEST_01^PBS\" --patient-id GBTEST_01 --series-description %s --study-description %s --default-value \"-1000\"", out_dir, xf_fn, description, description)
  system (command);

  # Apparently this needs to be done in two steps.  The private creator needs
  # to be added before adding the private tags
  command = sprintf ("dcmodify --no-backup -m \"(0020,0052)=%s\" -m \"(0020,000D)=%s\" -m \"(0010,0030)=20200101\" -m \"(0010,0040)=O\" -i \"(3773,0060)=MIM\"  %s/*", for_uid, study_uid, out_dir)
  system (command);

  command = sprintf ("dcmodify --no-backup -i \"(3773,6000)=8675389\" -i \"(3773,6001)=%d\" -i \"(3773,6002)=10\" -i \"(3773,6003)=%d%%\"  %s/*", i, (i-1)*10, out_dir)
  system (command);
end
