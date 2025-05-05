/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_GRID_BUFFER_SOA_H
#define DEVICE_INCLUDES_REGIR_GRID_BUFFER_SOA_H

#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"
#include "Device/includes/ReSTIR/ReGIR/HashGrid.h"

struct ReGIRGridBufferSoADevice
{
	// TODO pack this to 4 bytes
	ReGIRReservoirSoADevice reservoirs;
	ReGIRSampleSoADevice samples;

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
	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(const ReGIRHashGrid& grid_settings, int hash_grid_cell_index, int reservoir_index_in_cell, int grid_index = -1) const 
	{
		int reservoir_index_in_grid = hash_grid_cell_index * grid_settings.m_number_of_reservoirs_per_cell + reservoir_index_in_cell;

		return read_full_reservoir_opt(reservoir_index_in_grid); 
	}

	/**
	 * Just some overload to adhere to the API of the hash grid
	 */
	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(const ReGIRHashGrid& grid_settings, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1) const 
	{
		int hash_grid_cell_index = grid_settings.get_hash_grid_cell_index_from_world_pos(world_position, camera_position);

		return read_full_reservoir_opt(grid_settings, hash_grid_cell_index, reservoir_index_in_cell, grid_index);
	}
};

#endif
