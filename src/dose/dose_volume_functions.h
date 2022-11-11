/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dose_volume_functions_h_
#define _dose_volume_functions_h_

#include "rpl_volume.h"
#include "volume.h"

class Beam_calc;

void dose_volume_create(Volume* dose_volume, float* sigma_max, Rpl_volume* volume, double range);

void calculate_rpl_coordinates_xyz(std::vector<std::vector<double> >* xyz_coordinates_volume, Rpl_volume* rpl_volume);
void dose_volume_reconstruction(Rpl_volume* dose_rv, Volume::Pointer dose_vol);

void build_hong_grid(std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample);

void find_ijk_pixel(int* ijk_idx, double* xyz_ray_center, Volume* dose_volume);
void find_ijk_pixel(int* ijk_idx, double* xyz_ray_center, Volume::Pointer dose_volume);
void find_xyz_center_entrance(double* xyz_ray_center, double* ray, float z_axis_offset);
void find_xyz_center(double* xyz_ray_center, double* ray, float z_axis_offset, int k, float z_spacing);
void find_xyz_from_ijk(double* xyz, Volume* volume, int* ijk);

double erf_gauss(double x);
double double_gaussian_interpolation(double* gaussian_center, double* pixel_center, double sigma, double* spacing);

double get_off_axis(double radius, double dr, double sigma);

void dose_normalization_to_dose(Volume::Pointer dose_volume, double dose, Beam_calc* beam);
void dose_normalization_to_dose_and_point(Volume::Pointer dose_volume, double dose, const float* rdp_ijk, const float* rdp, Beam_calc* beam);

void
save_vector_as_image (
    const std::vector<double>& v,
    const int dim2d[2],
    const std::string& filename);

#endif
