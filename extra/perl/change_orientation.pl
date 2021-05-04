#!/usr/bin/perl

use File::Copy qw(copy move);
use File::Find;

$push_to_mim = 0;
$overwrite_for = 1;
$overwrite_study = 1;
$dob_empty = 0;
$sex_empty = 0;

# Set to 1 if you are only converting a plan object, to keep the
# existing referenced structure set
$keep_referenced_uid = 0;
#$keep_referenced_uid = 1;

#$dicom_dir = "/PHShome/gcs6/shared/ben-1/GBMON_SOBP1-3";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcom_H2Oart";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcommis_mornqa";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/GBDAY_01";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/TNW-01-02";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcommissSRS_BB";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcom_pg_hfs";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcom_01";
$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcom_05";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/019-01-16";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcom_08";
#$dicom_dir = "/PHShome/gcs6/conquest-1.4.17/data/LPcom_tt000_v2";

$new_name = "";
$new_id = "";
$new_birth_date = "";
$new_sex = "";
$new_series_description = "";

# $new_name = "LPcom_01^PBS";
# $new_id = "LPcom_01";
# $new_birth_date = "20180101";
# $new_sex = "X";
# $new_name = "LPcom_01^PBS";
# $new_id = "LPcom_01";
# $new_birth_date = "20180101";
# $new_sex = "X";
#$new_name = "LPcom_02^PBS";
#$new_id = "LPcom_02";
#$new_birth_date = "20180725";
#$new_sex = "O";
#$new_name = "Mobius^DoseLab";
#$new_id = "LN5-MB-IUS";
#$new_birth_date = "20190112";
#$new_sex = "O";
# $new_name = "LPcom_03^PBS";
# $new_id = "LPcom_03";
# $new_birth_date = "20180101";
# $new_sex = "O";
# $new_name = "LPcom_04^PBS";
# $new_id = "LPcom_04";
# $new_birth_date = "20180101";
# $new_sex = "O";
# $new_name = "LPcom_05^PBS";
# $new_id = "LPcom_05";
# $new_birth_date = "20180101";
# $new_sex = "O";
# $new_name = "LPcom_06^PBS";
# $new_id = "LPcom_06";
# $new_birth_date = "20180101";
# $new_sex = "O";
# $new_name = "LPcom_07^PBS";
# $new_id = "LPcom_07";
# $new_birth_date = "20180101";
# $new_sex = "O";
#$new_name = "LPcom_08^PBS";
#$new_id = "LPcom_08";
#$new_birth_date = "20180101";
#$new_sex = "O";
# $new_name = "LPcom_09^PBS";
# $new_id = "LPcom_09";
# $new_birth_date = "20180101";
# $new_sex = "O";
# $new_name = "GBDAY_01^PBS";
# $new_id = "GBDAY_01";
# $new_birth_date = "20200101";
# $new_sex = "M";
# $new_name = "GBDAY_02^PBS";
# $new_id = "GBDAY_02";
# $new_birth_date = "20200101";
# $new_sex = "M";
# $new_name = "LPcommis_tt_000_v2^PBS";
# $new_id = "LPcom_tt000_v2";
# $new_birth_date = "20190101";
# $new_sex = "O";
# $new_name = "GBMON_SOBP1^PBS";
# $new_id = "GBMON_SOBP1";
# $new_birth_date = "20200101";
# $new_sex = "M";
# $new_name = "GBMON_FF1^PBS";
# $new_id = "GBMON_FF1";
# $new_birth_date = "20200101";
# $new_sex = "M";
# $new_name = "GBMON_RANGE1^PBS";
# $new_id = "GBMON_RANGE1";
# $new_birth_date = "20200101";
# $new_sex = "M";
# $new_name = "GBMON_GAMMA1^PBS";
# $new_id = "GBMON_GAMMA1";
# $new_birth_date = "20200101";
# $new_sex = "M";
# $new_name = "GBMON_REF1^PBS";
# $new_id = "GBMON_REF1";
# $new_birth_date = "20200101";
# $new_sex = "M";


#$new_series_description = "Monthly Ref";
#$new_series_description = "Monthly Gamma";
#$new_series_description = "Monthly RU";
#$new_series_description = "Monthly DU";
#$new_series_description = "Monthly Zebra";
#$new_series_description = "HFP nz ISO 101.5";
#$new_series_description = "Empty Cantilev";
#$new_series_description = "Non Cantilev";
#$new_series_description = "Cantilev";
#$new_series_description = "SACRUM COPY";
#$new_series_description = "HFP Non-zero ISO";
#$new_series_description = "2020-06-29 Mock CSI";
#$new_series_description = "Medcom phantom";
#$new_series_description = "Morning QA 60 Fx";
#$new_series_description = "Morning QA 30 Fx";
#$new_series_description = "HFP SRS 270-360";
#$new_series_description = "HFP SRS 180-270";
#$new_series_description = "2020-04-28 MUd";
#$new_series_description = "Rewrite UIDs";

$new_patient_position = "";
$new_image_orientation = "";

#$new_patient_position = "FFS";
#$new_image_orientation = "-1\\0\\0\\0\\-1\\0";

#$new_patient_position = "HFP";
#$new_image_orientation = "-1\\0\\0\\0\\-1\\0";

if ($dob_empty) {
    $new_birth_date = "";
}
if ($sex_empty) {
    $new_sex = "";
}

#######################################################################################

$dcmdir = "/PHShome/gcs6/shared/ben-1/out";
$txtdir = "/PHShome/gcs6/shared/ben-1/txt";
(not -d $dcmdir) and mkdir "$dcmdir";
(not -d $txtdir) and mkdir "$txtdir";

unlink glob "$dcmdir/*";
unlink glob "$txtdir/*";

$overwrite_for_uid = `dicom_uid`;
chomp($overwrite_for_uid);
$overwrite_study_uid = `dicom_uid`;
chomp($overwrite_study_uid);

%uid_map = {};

sub get_value {
    my ($value) = @_;
    if ($value =~ /.*\[(.*)\]/) {
	$value = $1;
    } else {
	die "Parse error.";
    }
    return $value
}

sub reorient {
    my ($x, $y, $z) = @_;

#    return ($x + 0.06, $y - 1.5, $z + 355.4);
#    return ($x - 0.91, $y + 61.17 - 6.8, $z - 9.88);
#    return ($x + 0.79, $y + 70.97, $z + 499.39);
#    return (-$x + 10, -$y + 20, $z + 30);
#    return ($x, $y, $z-500);
#    return ($x + 0.800000, $y - 3.900000, $z + 499.399994);
#    return ($x + 0.34, $y - 1.24, $z + 500.61);
#    return ($x - 0.38, $y - 3.88, $z + 90);
    return ($x, $y, $z);
#    return (-$x + 30, -$y + 60, $z + 90);
}

sub change_isocenter {
    my ($x, $y, $z) = @_;

#    return (10, -14, 485);
#    return (0, 0, 1000);
#    return (10, 20, 30 - 10);
#    return (-$x, -$y, $z);
#    return (0, 0, 0);
#    return (30, 60, 90);
    return ($x, $y, $z);
}

sub process_file {

    my ($dcm_file) = @_;

    $cmd = "dcmdump -q +W raw +L $dcm_file > dump.txt";
    print ("$cmd\n");
    system ($cmd);

    $dump_in = "dump.txt";
    $dump_out = "dump_out.txt";

    open FIN, "<$dump_in";
    open FOUT, ">$dump_out";
    $sop_instance_uid = "";
    $coalescing_image_comments = 0;

    print FOUT "(0008,0005) CS [ISO_IR 100]\n";
    while (<FIN>) {
	if (/^\s*\(([^)]*)\)/) {
	    $key = $1;
	} else {
	    if ($coalescing_image_comments) {
		if (/\]/) {
		    $coalescing_image_comments = 0;
		} else {
		    $_ = s/\a\r/ /g;
		}
		print FOUT;
	    }
	    next;
	}
	if ($key eq "0002,0003" or $key eq "0008,0018" or $key eq "0020,000d"
	    or $key eq "0020,000e" or $key eq "0020,0052" or $key eq "0008,1155"
	    or $key eq "3006,0024") {
	    $value = get_value ($_);
	    if ($key eq "0008,1155" and $keep_referenced_uid) {
		print FOUT $_;
		next;
	    }
	    if (($key eq "0020,0052" or $key eq "3006,0024") and $overwrite_for) {
		print FOUT "($key) UI [$overwrite_for_uid]\n";
		next;
	    }
	    if ($key eq "0020,000d" and $overwrite_study) {
		print FOUT "($key) UI [$overwrite_study_uid]\n";
		next;
	    }
	    if (not exists ($uid_map {$value})) {
		$new_uid = `dicom_uid`;
		chomp ($new_uid);
		$uid_map{$value} = $new_uid;
	    }
	    if ($key eq "0008,0018") {
		$sop_instance_uid = $uid_map{$value};
	    }
	    print FOUT "($key) UI [$uid_map{$value}]\n";
	    next;
	}
	if ($key eq "0008,0005") {
	    #print FOUT "($key) CS [ISO_IR 100]\n";
	    next;
	}
	if ($key eq "0008,103e" and $new_series_description ne "") {
	    print FOUT "($key) LO [$new_series_description]\n";
	    next;
	}
	if ($key eq "0010,0010" and $new_name ne "") {
	    print FOUT "($key) PN [$new_name] # PatientName\n";
	    next;
	}
	if ($key eq "0010,0020" and $new_id ne "") {
	    print FOUT "($key) LO [$new_id] # PatientID\n";
	    next;
	}
	if ($key eq "0010,0030" and ($new_birth_date ne "" or $dob_empty)) {
	    print FOUT "($key) DA [$new_birth_date] # PatientBirthDate\n";
	    next;
	}
	if ($key eq "0010,0040" and ($new_sex ne "" or $sex_empty)) {
	    print FOUT "($key) CS [$new_sex] # PatientSex\n";
	    next;
	}
	if ($key eq "0018,5100" and $new_patient_position ne "") {
	    print FOUT "($key) CS [$new_patient_position] # PatientPosition\n";
	    next;
	}
	if ($key eq "0020,0037" and $new_image_orientation ne "") {
	    print FOUT "($key) DS [$new_image_orientation] # ImageOrientationPatient\n";
	    next;
	}
	if ($key eq "300a,0002" and $new_series_description ne "") {
	    print FOUT "($key) SH [$new_series_description]\n";
	    next;
	}
	if ($key eq "0020,0032") {
	    $value = get_value ($_);
	    my ($x,$y,$z,$rest) = split (/\\/,$value,3);
	    ($x,$y,$z) = reorient ($x,$y,$z);
	    print FOUT "($key) DS [$x\\$y\\$z]\n";
	    next;
	}
	if ($key eq "300a,0082" or $key eq "300a,012c") {
	    $value = get_value ($_);
	    my ($x,$y,$z,$rest) = split (/\\/,$value,3);
	    ($x,$y,$z) = change_isocenter ($x,$y,$z);
	    print FOUT "($key) DS [$x\\$y\\$z]\n";
	    next;
	}
	# ContourPoints
	if ($key eq "3006,0050") {
	    $value = get_value ($_);
	    $outvalue = "";
	    while (true) {
		my ($x,$y,$z,$rest) = split (/\\/,$value,4);
		($x,$y,$z) = reorient ($x,$y,$z);
		if ($outvalue ne "") {
		    $outvalue = "$outvalue\\";
		}
		$outvalue = "${outvalue}$x\\$y\\$z";
		if ($rest eq "") {
		    last;
		}
		$value = $rest;
	    }
	    print FOUT "($key) DS [$outvalue]\n";
	    next;
	}
	# ImageComments
	if ($key eq "0020,4000") {
	    if (/\[/ and not /\]/) {
		$coalescing_image_comments = 1;
		$_ = s/\a\r/ /g;
	    }
	    print FOUT $_;
	    next;
	}
	# else...
	print FOUT $_;
    }
    close FIN;
    close FOUT;

    if (${sop_instance_uid} eq "") {
	die "Error, no SOP Instance UID";
    }

    $new_dump_out = "${txtdir}/${sop_instance_uid}.txt";
    move ("${dump_out}", "$new_dump_out");
    $cmd = "dump2dcm +l 500000 ${new_dump_out} ${dcmdir}/${sop_instance_uid}.dcm";
    system ($cmd);
}

@dcm_files = ();

sub wanted {
    next if -d $File::Find::name;
    push @dcm_files, $File::Find::name;
}

find(\&wanted, $dicom_dir);

for my $dcm_file (@dcm_files) {
    process_file ($dcm_file);
}

if ($push_to_mim) {
    $cmd = "push_mim.sh ${dcmdir}";
    system ($cmd);
}
