/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SETTINGS_H
#define DEVICE_KERNELS_REGIR_SETTINGS_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

#include "HostDeviceCommon/Xorshift.h"

struct ReGIRSettings
{
	HIPRT_HOST_DEVICE float3 get_cell_size() const
	{
		float3 grid_resolution_float = make_float3(grid_resolution.x, grid_resolution.y, grid_resolution.z);
		float3 cell_size = extents / grid_resolution_float;

		return cell_size;
	}

	HIPRT_HOST_DEVICE float3 get_cell_center(unsigned int linear_cell_index) const
	{
		float3 cell_size = get_cell_size();

		int index_x = linear_cell_index % grid_resolution.x;
		int index_y = (linear_cell_index % (grid_resolution.x * grid_resolution.y)) / grid_resolution.x;
		int index_z = linear_cell_index % (grid_resolution.x * grid_resolution.y);

		int3 cell_index_xyz = make_int3(index_x, index_y, index_z);
		float3 cell_index_xyz_float = make_float3(static_cast<float>(index_x), static_cast<float>(index_y), static_cast<float>(index_z));

		return grid_origin + cell_size * cell_index_xyz_float + cell_size / 2.0f;
	}

	HIPRT_HOST_DEVICE int get_cell_index(float3 world_position, Xorshift32Generator* rng = nullptr, bool jitter = false) const
	{
		if (jitter)
			world_position += (make_float3(rng->operator()(), rng->operator()(), rng->operator()()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * get_cell_size() / 2.0f;

		float3 position_in_grid = world_position - grid_origin;
		float3 position_in_grid_cell_unit = position_in_grid / get_cell_size();

		int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
		// If a point is on the very edge of the grid, we're going to have one of the coordinates be 'grid_resolution.XXX'
		// exactly, 16 for a grid resolution of 16 for example. 
		// 
		// But that's out of bounds because our grid cells are in [0, 15] so we're sub
		cell_xyz = hippt::min(cell_xyz, grid_resolution - make_int3(1, 1, 1));

		return cell_xyz.x + cell_xyz.y * grid_resolution.x + cell_xyz.z * grid_resolution.x * grid_resolution.y;
	}

	/**
	 * If 'out_point_outside_of_grid' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_HOST_DEVICE ReGIRReservoir get_cell_reservoir(float3 shading_point, bool& out_point_outside_of_grid, Xorshift32Generator& rng, bool jitter = false) const
	{
		int cell_linear_index = get_cell_index(shading_point, &rng, jitter);
		if (cell_linear_index < 0 || cell_linear_index >= grid_resolution.x * grid_resolution.y * grid_resolution.z)
		{
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		out_point_outside_of_grid = false;

		int random_reservoir_index_in_cell = 0;
		if (reservoirs_count_per_grid_cell > 1)
			random_reservoir_index_in_cell = rng.random_index(reservoirs_count_per_grid_cell);

		// Returning the reservoir number 'random_reservoir_index_in_cell' in the cell number 'cell_linear_index'
		return grid_buffer[cell_linear_index * reservoirs_count_per_grid_cell + random_reservoir_index_in_cell];
	}

	HIPRT_HOST_DEVICE ColorRGB32F get_random_cell_color(float3 position, Xorshift32Generator* rng = nullptr, bool jitter = true) const
	{
		int cell_index = get_cell_index(position, rng, jitter);

		return ColorRGB32F::random_color(cell_index);
	}

	HIPRT_HOST_DEVICE int get_total_number_of_reservoirs() const
	{
		return grid_resolution.x * grid_resolution.y * grid_resolution.z * reservoirs_count_per_grid_cell;
	}

	float3 grid_origin;
	// "Length" of the grid in each X, Y, Z axis directions
	float3 extents;
	int3 grid_resolution = make_int3(16, 16, 16);

	// How many light samples are resampled into each reservoir of the grid cell
	int sample_count_per_cell_reservoir = 32;
	// How many reservoirs are going to be produced per each cell of the grid
	int reservoirs_count_per_grid_cell = 64;
	// At path tracing time, how many reservoirs of the grid cell of the point we're trying to shade
	// are going to be resampled (with the BRDF term) to produce the final light sample used for NEE
	int cell_reservoir_resample_per_shading_point = 8;
	// Whether or not to jitter the world space position used when looking up the ReGIR grid
	// This helps eliminate grid discretization  artifacts
	bool do_jittering = true;

	ReGIRReservoir* grid_buffer = nullptr;
};

#endif
