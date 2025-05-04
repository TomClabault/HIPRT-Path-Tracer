/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_H

#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

struct ReGIRHashGridSoADevice
{
	// TODO pack this to 4 bytes
	ReGIRReservoirSoADevice reservoirs;
	ReGIRSampleSoADevice samples;

	HIPRT_DEVICE unsigned int hash(float3 world_position, float3 camera_position) const
	{
		float3 relative_to_camera = world_position - camera_position;

		constexpr unsigned int p1 = 73856093;
		constexpr unsigned int p2 = 19349663;
		constexpr unsigned int p3 = 83492791;

		unsigned int x = relative_to_camera.x / 1;
		unsigned int y = relative_to_camera.y / 1;
		unsigned int z = relative_to_camera.z / 1;

		return (x * p1) ^ (y * p2) ^ (z * p3);
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt_from_index_in_grid(int reservoir_index_in_grid, const ReGIRReservoir& reservoir, int grid_index = -1)
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return;

		if (reservoir.UCW <= 0.0f)
		{
			// No need to store the rest if the UCW is invalid
			reservoirs.UCW[reservoir_index_in_grid] = reservoir.UCW;

			return;
		}

		reservoirs.store_reservoir_opt(reservoir_index_in_grid, reservoir);
		samples.store_sample(reservoir_index_in_grid, reservoir.sample);
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell, const ReGIRReservoir& reservoir, int grid_index = -1)
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return;

		int linear_cell_index = hash(world_position, camera_position);
		int reservoir_index_in_grid = linear_cell_index + reservoir_index_in_cell;

		store_reservoir_and_sample_opt_from_index_in_grid(reservoir_index_in_grid, reservoir, grid_index);
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1) const
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return ReGIRReservoir();

		ReGIRReservoir reservoir;

		int linear_cell_index = hash(world_position, camera_position);
		int reservoir_index_in_grid = linear_cell_index + reservoir_index_in_cell;

		float UCW = reservoirs.UCW[reservoir_index_in_grid];
		if (UCW <= 0.0f)
			// If the reservoir doesn't have a valid sample, not even reading the rest of it
			return ReGIRReservoir();

		reservoir = reservoirs.read_reservoir<false>(reservoir_index_in_grid);
		reservoir.UCW = UCW;
		reservoir.sample = samples.read_sample(reservoir_index_in_grid);

		return reservoir;
	}
};

#endif
