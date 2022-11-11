/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <algorithm>
#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dir_list.h"
#include "path_util.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "xio_studyset.h"

int Xio_studyset::gcd(int a, int b) {

    // Euclidean algorithm

    while (b != 0) {
	int t = b;
	b = a % b;
	a = t;
    }

    return a;
}

Xio_studyset::Xio_studyset (const std::string& input_dir)
{
    this->studyset_dir = input_dir;

    // Open index.dat file in input_dir
    std::string indexdat = compose_filename (input_dir, "index.dat");
    std::ifstream index (indexdat.c_str());

    int all_number_slices = 0;
    std::vector<Xio_studyset_slice> all_slices;

    if (index.is_open()) {
	// Get total number of slices
	index >> all_number_slices;

	// Loop through slices getting filename and location
	std::string slice_filename_scan;
	std::string slice_name;
	float slice_location;

	for (int i = 0; i < all_number_slices; i++) {
	    index >> slice_filename_scan;
	    index >> slice_location;

	    Xio_studyset_slice slice (slice_filename_scan, slice_location);
	    all_slices.push_back (slice);
	}
        index.close ();
    } else {
        // Older data has no index.dat file.  Use the filenames instead.
        Dir_list d (input_dir);
        for (int i = 0; i < d.num_entries; i++) {
            std::string entry = d.entry(i);
            if (extension_is (entry, "CT")) {
                std::string fn = basename (entry);
                std::string loc = fn.substr (2, fn.length()-4);
                //printf ("loc = %s\n", loc.c_str());
                float slice_location = std::stof (loc);
                //printf ("%f %s\n", slice_location, fn.c_str());
                Xio_studyset_slice slice (fn, slice_location);
                all_slices.push_back (slice);
            }
        }
    }
    
    // Sort slices in positive direction
    std::sort (all_slices.begin(), all_slices.end());

    for (auto it = all_slices.begin(); it != all_slices.end(); it++) {
        printf ("%f\n", it->location);
    }
    
    // Workaround for multiple slice thickness
    // Create volume with uniform voxel sizes by finding most common
    // slice thickness and duplicating slices to obtain a uniform Z axis.
    std::map<float,int> slice_thicknesses;
    for (auto it = all_slices.begin(); it != all_slices.end(); it++) {
        if (it == all_slices.begin()) {
            continue;
        }
        float spacing = it->location - prev(it)->location;
        spacing = ROUND_INT (spacing * 100) / 100.f;
        // Workaround for Xio position rounding.  Xio rounds positions to the
        // nearest 0.1 mm.
        if (spacing >= 1.1 && spacing <= 1.4) {
            spacing = 1.25;
        }
        if (spacing >= 3.6 && spacing <= 3.9) {
            spacing = 3.75;
        }
        if (slice_thicknesses.find(spacing) != slice_thicknesses.end()) {
            slice_thicknesses[spacing]++;
        } else {
            slice_thicknesses[spacing] = 1;
        }
    }

    int best_count = 0;
    float best_spacing = 2.5;
    for (auto it = slice_thicknesses.begin(); it != slice_thicknesses.end(); it++) {
        printf ("(%f) -> %d\n", it->first, it->second);
        if (it->second > best_count) {
            best_spacing = it->first;
            best_count = it->second;
        }
    }
    this->thickness = best_spacing;

    this->number_slices = 0;
    if (all_slices.size() > 0) {
        this->number_slices = ROUND_INT (
            (all_slices.back().location - all_slices.front().location) / best_spacing) + 1;
        printf ("Number of slices: (%f - %f) / %f = %d\n",
            all_slices.back().location, all_slices.front().location, best_spacing,
            this->number_slices);
    }

    auto it = all_slices.begin();
    for (int i = 0; i < this->number_slices; i++) {
        float location = all_slices[0].location + i * best_spacing;
        while (next(it) != all_slices.end()) {
            float curr_slice_dist = fabs(location - it->location);
            float next_slice_dist = fabs(location - next(it)->location);
            if (next_slice_dist < curr_slice_dist) {
                it++;
            } else {
                break;
            }
        }
        Xio_studyset_slice slice(it->filename_scan, location);
        slices.push_back(slice);
        printf ("%3d: %f, %f\n", i, location, it->location);
    }

    // Initialize pixel spacing to zero.  This get set when the 
    // CT is loaded
    this->ct_pixel_spacing[0] = this->ct_pixel_spacing[1] = 0.f;
}

Xio_studyset::~Xio_studyset ()
{
}


Xio_studyset_slice::Xio_studyset_slice (std::string slice_filename_scan, const float slice_location)
{
    filename_scan = slice_filename_scan;
    location = slice_location;

    // Get name from slice filename
    size_t extension_dot = filename_scan.find_last_of("."); 
    name = filename_scan.substr(0, extension_dot);

    filename_contours = name + ".WC";
}

Xio_studyset_slice::~Xio_studyset_slice ()
{
}
