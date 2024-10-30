amp = 2.0;     # amplitude in cm
#period = 3.0;  
period = 5.0;  

delta = 0.02997;
rpm.time = 0:delta:20*pi;
rpm.phase = rpm.time * 2 * pi / period;
rpm.phase = rpm.phase - floor (rpm.phase / 2 / pi) * 2 * pi;
rpm.phase(find([0, rpm.phase(1:end-1)-rpm.phase(2:end) > 0])) = 0;
rpm.amp = 10 * amp * (cos(rpm.phase)-1) / 2;
rpm.header = {
  'CRC=57065'
  'Version=1.6'
  'Data_layout=amplitude,phase,timestamp,validflag,ttlin,mark,ttlout'
  'Patient_ID=GBTEST_01'
  'Date=01-01-2020'
  'Total_study_time=180.565'
  'Samples_per_second=30'
  'Scale_factor=10.0'
  '[Data]'
  };
rpm.header = sprintf ("%s\n", rpm.header{:});
rpm.version = 'VXP 1.6';
rpm.valid = zeros(length(rpm.time),1);
rpm.ttlin = zeros(length(rpm.time),1);
rpm.mark = zeros(length(rpm.time),1);
rpm.ttlout = zeros(length(rpm.time),1);

#writerpm ('GBTEST_01_3_seconds.vxp', rpm);
writerpm ('GBTEST_01_5_seconds.vxp', rpm);
