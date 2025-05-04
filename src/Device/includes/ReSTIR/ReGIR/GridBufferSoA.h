/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_GRID_BUFFER_SOA_H
#define DEVICE_INCLUDES_REGIR_GRID_BUFFER_SOA_H

#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"
#include "Device/includes/ReSTIR/ReGIR/GridSettings.h"

struct ReGIRGridBufferSoADevice
{
	// TODO pack this to 4 bytes
	ReGIRReservoirSoADevice reservoirs;
	ReGIRSampleSoADevice samples;

	HIPRT_DEVICE static float3 get_cell_size(const ReGIRGridSettings& grid_settings)
	{
		return grid_settings.m_cell_size;
	}

	HIPRT_DEVICE static float get_cell_diagonal_length(const ReGIRGridSettings& grid_settings)
	{
		return grid_settings.m_cell_diagonal_length;
	}

	HIPRT_DEVICE static int3 get_xyz_cell_index_from_linear(const ReGIRGridSettings& grid_settings, int linear_cell_index)
	{
		int index_x = linear_cell_index % grid_settings.grid_resolution.x;
		int index_y = (linear_cell_index % (grid_settings.grid_resolution.x * grid_settings.grid_resolution.y)) / grid_settings.grid_resolution.x;
		int index_z = linear_cell_index / (grid_settings.grid_resolution.x * grid_settings.grid_resolution.y);

		return make_int3(index_x, index_y, index_z);
	}

	HIPRT_DEVICE static float3 get_cell_origin_from_linear_cell_index(const ReGIRGridSettings& grid_settings, int linear_cell_index)
	{
		float3 cell_size = get_cell_size(grid_settings);

		int3 cell_index_xyz = get_xyz_cell_index_from_linear(grid_settings, linear_cell_index);
		float3 cell_index_xyz_float = make_float3(static_cast<float>(cell_index_xyz.x), static_cast<float>(cell_index_xyz.y), static_cast<float>(cell_index_xyz.z));

		return grid_settings.grid_origin + cell_size * cell_index_xyz_float;
	}

	HIPRT_DEVICE static float3 get_cell_center_from_linear_cell_index(const ReGIRGridSettings& grid_settings, unsigned int linear_cell_index)
	{
		float3 cell_size = get_cell_size(grid_settings);

		return get_cell_origin_from_linear_cell_index(grid_settings, linear_cell_index) + cell_size * 0.5f;
	}

	HIPRT_DEVICE static int get_linear_cell_index_from_world_pos(const ReGIRGridSettings& grid_settings, float3 world_position, Xorshift32Generator* rng = nullptr, bool jitter = false)
	{
		if (jitter)
			world_position += (make_float3(rng->operator()(), rng->operator()(), rng->operator()()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * get_cell_size(grid_settings) * 0.5f;

		float3 position_in_grid = world_position - grid_settings.grid_origin;
		float3 position_in_grid_cell_unit = position_in_grid / get_cell_size(grid_settings);

		int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
		// If a point is on the very edge of the grid, we're going to have one of the coordinates be 'grid_resolution.XXX'
		// exactly, 16 for a grid resolution of 16 for example. 
		// 
		// But that's out of bounds because our grid cells are in [0, 15] so we're subing 
		cell_xyz = hippt::min(cell_xyz, grid_settings.grid_resolution - make_int3(1, 1, 1));

		return cell_xyz.x + cell_xyz.y * grid_settings.grid_resolution.x + cell_xyz.z * grid_settings.grid_resolution.x * grid_settings.grid_resolution.y;
	}

	HIPRT_DEVICE static float3 get_cell_center_from_world_pos(const ReGIRGridSettings& grid_settings, float3 world_point)
	{
		return get_cell_center_from_linear_cell_index(grid_settings, get_linear_cell_index_from_world_pos(grid_settings, world_point));
	}

	HIPRT_DEVICE static int get_linear_cell_index_from_xyz(const ReGIRGridSettings& grid_settings, int3 xyz_cell_index)
	{
		if (xyz_cell_index.x < 0 || xyz_cell_index.x >= grid_settings.grid_resolution.x
			|| xyz_cell_index.y < 0 || xyz_cell_index.y >= grid_settings.grid_resolution.y
			|| xyz_cell_index.z < 0 || xyz_cell_index.z >= grid_settings.grid_resolution.z)
			// Outside of the grid
			return -1;

		return xyz_cell_index.x + xyz_cell_index.y * grid_settings.grid_resolution.x + xyz_cell_index.z * grid_settings.grid_resolution.x * grid_settings.grid_resolution.y;
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, int reservoir_index_in_grid)
	{
		if (reservoir.UCW <= 0.0f)
		{
			// No need to store the rest if the UCW is invalid
			reservoirs.UCW[reservoir_index_in_grid] = reservoir.UCW;

			return;
		}

		reservoirs.store_reservoir_opt(reservoir_index_in_grid, reservoir);
		samples.store_sample(reservoir_index_in_grid, reservoir.sample);
	}

	/**
	 * Just some overload to adhere to the API of the hash grid
	 */
	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_grid, int grid_index = -1)
	{ 
		return store_reservoir_and_sample_opt(reservoir, reservoir_index_in_grid);
	}

	/**
	 * Just some overload to adhere to the API of the hash grid
	 */
	HIPRT_DEVICE void store_reservoir_and_sample_opt_from_index_in_grid(const ReGIRReservoir& reservoir, int reservoir_index_in_grid, int grid_index = -1)
	{
		store_reservoir_and_sample_opt(reservoir, reservoir_index_in_grid);
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(int reservoir_index_in_grid) const
	{
		ReGIRReservoir reservoir;

		float UCW = reservoirs.UCW[reservoir_index_in_grid];
		if (UCW <= 0.0f)
			// If the reservoir doesn't have a valid sample, not even reading the rest of it
			return ReGIRReservoir();

		reservoir = reservoirs.read_reservoir<false>(reservoir_index_in_grid);
		reservoir.UCW = UCW;
		reservoir.sample = samples.read_sample(reservoir_index_in_grid);

		return reservoir;
	}

	/**
	 * Just some overload to adhere to the API of the hash grid
	 */
	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(const ReGIRGridSettings& grid_settings, int linear_cell_index, int reservoir_index_in_cell, int grid_index = -1) const 
	{
		int reservoir_index_in_grid = linear_cell_index * grid_settings.m_number_of_reservoirs_per_cell + reservoir_index_in_cell;

		return read_full_reservoir_opt(reservoir_index_in_grid); 
	}

	/**
	 * Just some overload to adhere to the API of the hash grid
	 */
	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(const ReGIRGridSettings& grid_settings, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1) const 
	{
		int linear_cell_index = get_linear_cell_index_from_world_pos(grid_settings, world_position);

		return read_full_reservoir_opt(grid_settings, linear_cell_index, reservoir_index_in_cell, grid_index);
	}
};

#endif
