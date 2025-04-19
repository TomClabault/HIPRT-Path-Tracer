/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SETTINGS_H
#define DEVICE_KERNELS_REGIR_SETTINGS_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

#include "HostDeviceCommon/Xorshift.h"

struct ReGIRGridSettings
{
	float3 grid_origin;
	// "Length" of the grid in each X, Y, Z axis directions
	float3 extents;

	static constexpr int DEFAULT_GRID_SIZE = 16;
	int3 grid_resolution = make_int3(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);
};

struct ReGIRGridFillSettings
{
	// How many light samples are resampled into each reservoir of the grid cell
	int sample_count_per_cell_reservoir = 32;
	// How many reservoirs are going to be produced per each cell of the grid
	int reservoirs_count_per_grid_cell = 48;

	// This is a linear buffer that contains enough space for 'get_total_number_of_reservoirs_ReGIR()' reservoirs
	ReGIRReservoir* grid_buffers = nullptr;
};

struct ReGIRTemporalReuseSettings
{
	// Whether or not to reuse the reservoirs from the last frame as well as current frame
	bool do_temporal_reuse = false;

	int m_cap = 50;
	// How many grids to keep in memory to help with sample quality
	int temporal_history_length = 8;
	// Index of the grid of the current frame. In [0, temporal_history_length - 1]
	int current_grid_index = 0;
};

struct ReGIRSpatialReuseSettings
{
	HIPRT_HOST_DEVICE void store_reservoir(const ReGIRReservoir& reservoir, int linear_reservoir_index_in_grid)
	{
		output_grid[linear_reservoir_index_in_grid] = reservoir;
	}

	bool do_spatial_reuse = false;

	int spatial_neighbor_reuse_count = 128;
	int spatial_reuse_radius = 3;

	// Grid that contains the output of the spatial reuse pass
	ReGIRReservoir* output_grid = nullptr;
};

struct ReGIRShadingSettings
{
	// At path tracing time, how many reservoirs of the grid cell of the point we're trying to shade
	// are going to be resampled (with the BRDF term) to produce the final light sample used for NEE
	int cell_reservoir_resample_per_shading_point = 1;
	// Whether or not to jitter the world space position used when looking up the ReGIR grid
	// This helps eliminate grid discretization  artifacts
	bool do_cell_jittering = false;
};

struct ReGIRSettings
{
	HIPRT_HOST_DEVICE float3 get_cell_size() const
	{
		float3 grid_resolution_float = make_float3(grid.grid_resolution.x, grid.grid_resolution.y, grid.grid_resolution.z);
		float3 cell_size = grid.extents / grid_resolution_float;

		return cell_size;
	}

	HIPRT_HOST_DEVICE int3 get_xyz_cell_index_from_linear(int linear_cell_index) const
	{
		int index_x = linear_cell_index % grid.grid_resolution.x;
		int index_y = (linear_cell_index % (grid.grid_resolution.x * grid.grid_resolution.y)) / grid.grid_resolution.x;
		int index_z = linear_cell_index / (grid.grid_resolution.x * grid.grid_resolution.y);
		
		return make_int3(index_x, index_y, index_z);
	}

	HIPRT_HOST_DEVICE float3 get_cell_center_from_world_pos(float3 world_point) const
	{
		return get_cell_center(get_cell_linear_index_from_world_pos(world_point));
	}

	HIPRT_HOST_DEVICE float3 get_cell_center(unsigned int linear_cell_index) const
	{
		float3 cell_size = get_cell_size();

		int3 cell_index_xyz = get_xyz_cell_index_from_linear(linear_cell_index);
		float3 cell_index_xyz_float = make_float3(static_cast<float>(cell_index_xyz.x), static_cast<float>(cell_index_xyz.y), static_cast<float>(cell_index_xyz.z));

		return grid.grid_origin + cell_size * cell_index_xyz_float + cell_size / 2.0f;
	}

	HIPRT_HOST_DEVICE int get_cell_linear_index_from_world_pos(float3 world_position, Xorshift32Generator* rng = nullptr, bool jitter = false) const
	{
		if (jitter)
			world_position += (make_float3(rng->operator()(), rng->operator()(), rng->operator()()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * get_cell_size() / 2.0f;

		float3 position_in_grid = world_position - grid.grid_origin;
		float3 position_in_grid_cell_unit = position_in_grid / get_cell_size();

		int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
		// If a point is on the very edge of the grid, we're going to have one of the coordinates be 'grid_resolution.XXX'
		// exactly, 16 for a grid resolution of 16 for example. 
		// 
		// But that's out of bounds because our grid cells are in [0, 15] so we're sub
		cell_xyz = hippt::min(cell_xyz, grid.grid_resolution - make_int3(1, 1, 1));

		return cell_xyz.x + cell_xyz.y * grid.grid_resolution.x + cell_xyz.z * grid.grid_resolution.x * grid.grid_resolution.y;
	}

	HIPRT_HOST_DEVICE int get_cell_linear_index_from_xyz(int3 xyz_cell_index) const
	{
		if (xyz_cell_index.x < 0 || xyz_cell_index.x >= grid.grid_resolution.x 
			|| xyz_cell_index.y < 0 || xyz_cell_index.y >= grid.grid_resolution.y
			|| xyz_cell_index.z < 0 || xyz_cell_index.z >= grid.grid_resolution.z)
			// Outside of the grid
			return -1;

		return xyz_cell_index.x + xyz_cell_index.y * grid.grid_resolution.x + xyz_cell_index.z * grid.grid_resolution.x * grid.grid_resolution.y;
	}

	/**
	 * If 'out_point_outside_of_grid' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_HOST_DEVICE ReGIRReservoir get_reservoir_for_shading_from_linear_index(int reservoir_index_in_grid) const
	{
		if (spatial_reuse.do_spatial_reuse)
			// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
			return spatial_reuse.output_grid[reservoir_index_in_grid];
		else if (temporal_reuse.do_temporal_reuse)
			// If only doing temporal reuse, reading from the output of the spatial reuse pass
			return get_temporal_reservoir(reservoir_index_in_grid);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return grid_fill.grid_buffers[reservoir_index_in_grid];
	}

	/**
	 * If 'out_point_outside_of_grid' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_HOST_DEVICE ReGIRReservoir get_reservoir_for_shading_from_world_pos(float3 shading_point, bool& out_point_outside_of_grid, Xorshift32Generator& rng, bool jitter = false) const
	{
		int cell_linear_index = get_cell_linear_index_from_world_pos(shading_point, &rng, jitter);
		if (cell_linear_index < 0 || cell_linear_index >= grid.grid_resolution.x * grid.grid_resolution.y * grid.grid_resolution.z)
		{
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		out_point_outside_of_grid = false;

		int random_reservoir_index_in_cell = 0;
		if (grid_fill.reservoirs_count_per_grid_cell > 1)
			random_reservoir_index_in_cell = rng.random_index(grid_fill.reservoirs_count_per_grid_cell);

		return get_reservoir_for_shading_from_linear_index(cell_linear_index * grid_fill.reservoirs_count_per_grid_cell + random_reservoir_index_in_cell);
	}

	/**
	 * Returns the reservoir indicated by lienar_reservoir_index_in_grid but in the grid_index given
	 * 
	 * This function only makes sense with temporal reuse where we have more than 1 grid and so a single reservoir index
	 * isn't enough to fetch the reservoir in the reservoir buffer
	 * 
	 * The 'grid_index' parameter allows reading from a specific grid of past frames. 
	 * This is index should be in [0, temporal_reuse.temporal_history_length - 1].
	 * 
	 * If not specified, this function reads from the grid of the current frame
	 */
	HIPRT_HOST_DEVICE ReGIRReservoir get_temporal_reservoir(int linear_reservoir_index_in_grid, int grid_index = -1) const
	{
		if (grid_index != -1)
			return grid_fill.grid_buffers[grid_index * get_number_of_reservoirs_per_grid() + linear_reservoir_index_in_grid];
		else
			return grid_fill.grid_buffers[temporal_reuse.current_grid_index * get_number_of_reservoirs_per_grid() + linear_reservoir_index_in_grid];
	}

	HIPRT_HOST_DEVICE ReGIRReservoir get_grid_fill_output_reservoir(int linear_reservoir_index_in_grid) const
	{
		// The output of the grid fill pass is in the current frame grid so we can call the temporal method with
		// index -1
		return get_temporal_reservoir(linear_reservoir_index_in_grid, -1);
	}

	HIPRT_HOST_DEVICE void store_reservoir(ReGIRReservoir reservoir, int linear_reservoir_index_in_grid, int grid_index = -1)
	{
		if (grid_index != -1)
			grid_fill.grid_buffers[grid_index * get_number_of_reservoirs_per_grid() + linear_reservoir_index_in_grid] = reservoir;
		else
			grid_fill.grid_buffers[temporal_reuse.current_grid_index * get_number_of_reservoirs_per_grid() + linear_reservoir_index_in_grid] = reservoir;
	}

	HIPRT_HOST_DEVICE ColorRGB32F get_random_cell_color(float3 position) const
	{
		int cell_index = get_cell_linear_index_from_world_pos(position);

		return ColorRGB32F::random_color(cell_index);
	}

	HIPRT_HOST_DEVICE int get_number_of_reservoirs_per_grid() const
	{
		int number_of_cells = grid.grid_resolution.x * grid.grid_resolution.y * grid.grid_resolution.z;

		return number_of_cells * grid_fill.reservoirs_count_per_grid_cell;
	}

	HIPRT_HOST_DEVICE int get_total_number_of_reservoirs_ReGIR() const
	{
		int temporal_grid_count = temporal_reuse.do_temporal_reuse ? temporal_reuse.temporal_history_length : 1;

		return get_number_of_reservoirs_per_grid() * temporal_grid_count;
	}

	/**
	 * Resets all the reservoirs of all the grids at the given 'reservoir_index'
	 */
	HIPRT_HOST_DEVICE void reset_reservoirs(int reservoir_index)
	{
		int temporal_grid_count = temporal_reuse.do_temporal_reuse ? temporal_reuse.temporal_history_length : 1;

		for (int grid_index = 0; grid_index < temporal_grid_count; grid_index++)
            store_reservoir(ReGIRReservoir(), reservoir_index, grid_index);

		// Also clearing the spatial reuse output buffers (grid) if spatial reuse is enabled
		if (spatial_reuse.do_spatial_reuse)
			spatial_reuse.output_grid[reservoir_index] = ReGIRReservoir();
	}

	ReGIRGridSettings grid;
	ReGIRGridFillSettings grid_fill;
	ReGIRTemporalReuseSettings temporal_reuse;
	ReGIRSpatialReuseSettings spatial_reuse;
	ReGIRShadingSettings shading;

	// Multiplicative factor to multiply the output of some debug views
	float debug_view_scale_factor = 1.0f;
};

#endif
