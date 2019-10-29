#! /usr/bin/perl
use Getopt::Long;
use File::Basename;
use List::Util;
use File::Copy::Recursive;
use File::Path;
use File::Copy;

my $input_base = "/PHShome/gcs6/conquest-1.4.17/data";
my $scratch_base = "/PHShome/gcs6/scratch";

sub find_newest_dir {
    my ($DIR) = @_;
    opendir my $dh, $DIR or die "Error opening $DIR: $!";
    my @files = sort { $b->[10] <=> $a->[10] }
      map {[ $_, CORE::stat "$DIR/$_" ]}
      grep !m/^\.\.?$/, readdir $dh;
    closedir $dh;
    return @{$files[0]};
}

##my $newest_dir = find_newest_dir ($input_base);
my $newest_dir = shift;
if ($newest_dir eq "") {
    die "Usage: extend_inf.pl input-dir";
}

my $ori_copy_dir = "$input_base/$newest_dir";
my $scratch_dir = "$scratch_base/$newest_dir/scratch";
my $merged_dir = "$scratch_base/$newest_dir/merged";
my $fixed_dir = "$scratch_base/$newest_dir/fixed";

(-d $ori_copy_dir) or die "Directory $ori_copy_dir does not exist!";

opendir (DIR, $ori_copy_dir);
my @list = sort readdir (DIR);
my $lowest_position;
my $lowest_position_file = "";
print "Searching directory for most inferior slice\n";
for (@list){
    next if (($_ eq ".") or ($_ eq ".."));
    my $cmd = "dcmdump -q $ori_copy_dir/$_ | grep ImagePositionPatient | sed -e 's/.*\\\\//g' | sed -e 's/\\].*//g'";
    my $cmd_output = `$cmd`;
    chomp ($cmd_output);
    if ($lowest_position_file eq "" or $lowest_position > $cmd_output) {
	$lowest_position_file = $_;
	$lowest_position = $cmd_output;
    }
}
closedir (DIR);

print "$lowest_position | $lowest_position_file\n";

File::Path::make_path ($scratch_dir);
$cmd = "dcmdump -q +L +E +W $scratch_dir \"$ori_copy_dir/$lowest_position_file\"";
print "$cmd\n";
open (my $dcmdump_stream, "$cmd|");
open (my $dcmdump_output, ">$scratch_dir/1.dump");
while (<$dcmdump_stream>) {
    if (/.*\\([^]]*).*ImagePositionPatient/) {
	print ">> $_\n";
	$new_z = $lowest_position - 20;
	s/\\[^]\\]*\]/\\$new_z\]/;
	print ">> $_\n";
    }
    print $dcmdump_output $_;
}
close $dcmdump_stream;
close $dcmdump_output;

$cmd = "dump2dcm +E $scratch_dir/1.dump $scratch_dir/1.dcm";
print `$cmd`;

## Move all necessary files to "final" dir
if (-d $merged_dir) {
    File::Path::rmtree ($merged_dir);
}
File::Copy::Recursive::dircopy ("$input_base/$newest_dir", $merged_dir) or die $!;
File::Copy::copy ("$scratch_dir/1.dcm", $merged_dir);

## Interpolate into "fixed" dir
$cmd = "plastimatch convert --input \"$merged_dir\" --output-dicom \"$fixed_dir\" --series-description \"Extended Inferiorly\"";
print "$cmd\n";
system ($cmd);
