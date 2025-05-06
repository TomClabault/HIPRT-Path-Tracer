/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_SHADING_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_SHADING_SETTINGS_H

struct ReGIRShadingSettings
{
	// At path tracing time, how many reservoirs of the grid cell of the point we're trying to shade
	// are going to be resampled (with the BRDF term) to produce the final light sample used for NEE
	int cell_reservoir_resample_per_shading_point = 1;
	// Whether or not to jitter the world space position used when looking up the ReGIR grid
	// This helps eliminate grid discretization  artifacts
	bool do_cell_jittering = false;

	// For each grid cell, indicates whether that grid cell has been used at least once during shading in the last frame
	// 
	// The unsigned char contains 1 if the grid cell was alive, meaning that the grid cell will be populated during the grid fill
	// as well as during the spatial reuse and that grid cell is also able to be used during shading
	//
	// If the unsigned char is 0, that grid cell hasn't been used last frame and will be filled by the grid fill/temporal/spatial reuse
	// passes
	// const unsigned char* grid_cells_alive = nullptr;

	// The staging buffer is used to store the grid cells that are alive during shading: for each grid cell that a ray falls into during shading,
	// we position the unsigned char to 1
	//
	// We need a staging buffer to do that because modifying the 'grid_cells_alive' buffer directly would be a race condition since other threads
	// may be reading from that buffer at the same time to see if a cell is alive or not
	//
	// That staging buffer is then copied to the 'grid_cells_alive' buffer at the end of the frame
	AtomicType<unsigned int>* grid_cells_alive = nullptr;
	unsigned int* grid_cells_alive_list = nullptr;
	// unsigned int grid_cells_alive_count;
	AtomicType<unsigned int>* grid_cells_alive_count = nullptr;
};

#endif
