/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_H

#include "Device/includes/ReSTIR/ReGIR/HashGrid.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"
#include "Device/includes/ReSTIR/ReGIR/ShadingSettings.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"

struct ReGIRHashGridSoADevice
{
	HIPRT_DEVICE void reset_reservoir(unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index == -1)
			grid_index = 0;

		unsigned int reservoirs_per_grid = m_total_number_of_cells * reservoirs.number_of_reservoirs_per_cell;
		int reservoir_index_in_grid = reservoirs_per_grid * grid_index + hash_grid_cell_index * reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		reservoirs.store_reservoir_opt(reservoir_index_in_grid, ReGIRReservoir());
		samples.store_sample(reservoir_index_in_grid, ReGIRReservoir().sample);
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return;

		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);
		if (!resolve_collision(hash_grid_cell_index, hash_key))
			return;

		int reservoir_index_in_grid = hash_grid_cell_index * reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		store_reservoir_and_sample_opt_from_index_in_grid(reservoir_index_in_grid, reservoir, grid_index);
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1) const
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return ReGIRReservoir();

		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);
		if (!resolve_collision(hash_grid_cell_index, hash_key))
			return ReGIRReservoir();

		ReGIRReservoir reservoir;

		int reservoir_index_in_grid = hash_grid_cell_index * reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		float UCW = reservoirs.UCW[reservoir_index_in_grid];
		if (UCW <= 0.0f)
			// If the reservoir doesn't have a valid sample, not even reading the rest of it
			return ReGIRReservoir();

		reservoir = reservoirs.read_reservoir<false>(reservoir_index_in_grid);
		reservoir.UCW = UCW;
		reservoir.sample = samples.read_sample(reservoir_index_in_grid);

		return reservoir;
	}

	HIPRT_DEVICE void update_hash_cell_data(ReGIRShadingSettings& shading_settings, float3 world_position, float3 camera_position, float3 shading_normal, int primitive_index)
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);
		/*if (shading_normal.z > 0.8f && hash_grid_cell_index == 694)
			printf("\n");*/

		unsigned int current_hash_key = hash_cell_data.hash_keys[hash_grid_cell_index];
		if (current_hash_key != ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
		{
			// We already have something in that cell
			if (current_hash_key != hash_key)
			{
				// And it's not our hash so this is a collision

				unsigned int new_hash_cell_index = hash_grid_cell_index;
				if (!resolve_collision(new_hash_cell_index, hash_key))
					// Could not resolve the collision
					return;
				else if (new_hash_cell_index == hash_grid_cell_index)
					// We resolved the collision by finding our own cell
					// 
					// This means that we already have something in the cell, nothing to do
					return;
				else
					// We resolved the collision with a new cell
					hash_grid_cell_index = new_hash_cell_index;
			}
			else
				// Already have something in the cell and that's our hash, no collision, nothing to do
				return;
		}

		if (hippt::atomic_compare_exchange(&hash_cell_data.representative_primitive[hash_grid_cell_index], ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE, primitive_index) == ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE)
		{
			hash_cell_data.representative_points[hash_grid_cell_index] = world_position;
			hash_cell_data.representative_normals[hash_grid_cell_index].pack(shading_normal);
			hash_cell_data.hash_keys[hash_grid_cell_index] = hash_key;
		}

		// Because we just updated that grid cell, it is now alive
		// Only go through all that atomic stuff if the cell hasn't been staged already
		if (shading_settings.grid_cells_alive[hash_grid_cell_index] == 0)
		{
			if (hippt::atomic_compare_exchange(&shading_settings.grid_cells_alive[hash_grid_cell_index], 0u, 1u) == 0u)
			{
				unsigned int cell_alive_index = hippt::atomic_fetch_add(shading_settings.grid_cells_alive_count, 1u);

				shading_settings.grid_cells_alive_list[cell_alive_index] = hash_grid_cell_index;
			}
		}
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_no_collision_resolve(float3 world_position, float3 camera_position) const
	{
		unsigned int hash_key;
		unsigned int hash_cell_index = hash(world_position, camera_position, hash_key);

		return hash_cell_index;
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(float3 world_position, float3 camera_position) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);
		if (hash_key == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
			// This is refering to a grid cell that hasn't been filled yet
			return ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY;

		/*if (DEBUG_SHADING_NORMAL.z > 0.8f && hash_grid_cell_index == 694)
			printf("\n");*/

		if (!resolve_collision(hash_grid_cell_index, hash_key))
			return ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY;
		else
			return hash_grid_cell_index;
	}

	HIPRT_DEVICE float3 get_cell_size(float3 world_position = make_float3(0, 0, 0), float3 camera_position = make_float3(0, 0, 0)) const
	{
		return make_float3(1.0f / hash_grid.grid_resolution.x, 1.0f / hash_grid.grid_resolution.y, 1.0f / hash_grid.grid_resolution.z);
	}

	// HIPRT_DEVICE float get_cell_diagonal_length() const
	// {
	// 	return hash_grid.m_cell_diagonal_length;
	// }

	HIPRT_DEVICE float3 jitter_world_position(float3 original_world_position, Xorshift32Generator& rng) const
	{
		return original_world_position + (make_float3(rng(), rng(), rng()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * get_cell_size() * 0.5f;
	}

	ReGIRHashGrid hash_grid;

	// These two SoAs are allocated to hold 'number_cells * number_reservoirs_per_cell'
	// So for a given 'hash_grid_cell_index', the cell contains reservoirs and samples going from 
	// reservoirs[hash_grid_cell_index * number_reservoirs_per_cell] to reservoirs[cell_index * number_reservoirs_per_cell + number_reservoirs_per_cell[
	ReGIRReservoirSoADevice reservoirs;
	ReGIRSampleSoADevice samples;
	// This SoA is allocated to hold 'number_cells' only
	ReGIRHashCellDataSoADevice hash_cell_data;

	unsigned int m_total_number_of_cells = 0;

private:
	/**
	 * Returns the hash cell index of the given world position and camera position. Does not resolve collisions.
	 * The hash key for resolving collision is given in 'out_hash_key'
	 */
	HIPRT_DEVICE unsigned int hash(float3 world_position, float3 camera_position, unsigned int& out_hash_key) const
	{
		float3 relative_to_camera = world_position;// -camera_position;

		constexpr unsigned int p1 = 73856093;
		constexpr unsigned int p2 = 19349663;
		constexpr unsigned int p3 = 83492791;

		unsigned int grid_coord_x = static_cast<int>(relative_to_camera.x * hash_grid.grid_resolution.x);
		unsigned int grid_coord_y = static_cast<int>(relative_to_camera.y * hash_grid.grid_resolution.y);
		unsigned int grid_coord_z = static_cast<int>(relative_to_camera.z * hash_grid.grid_resolution.z);

		unsigned int x = grid_coord_x % (1 << 10);
		unsigned int y = grid_coord_y % (1 << 10);
		unsigned int z = grid_coord_z % (1 << 10);

		out_hash_key = x | (y << 10) | (z << 20);

		return ((x * p1) ^ (y * p2) ^ (z * p3)) % m_total_number_of_cells;
	}

	/**
	 * Returns true if the collision was resolved with success and the new hash
	 * (or unchanged if there was no collision) is set in 'in_out_base_hash'
	 */
	HIPRT_DEVICE bool resolve_collision(unsigned int& in_out_hash_cell_index, unsigned int hash_key) const
	{
		if (hash_cell_data.hash_keys[in_out_hash_cell_index] != hash_key)
		{
			// This is a collision

			unsigned int base_hash_cell = in_out_hash_cell_index;

			// Linear probing
			for (int i = 1; i <= 32; i++)
			{
				unsigned int next_hash_cell_index = (base_hash_cell + i) % m_total_number_of_cells;
				if (next_hash_cell_index == base_hash_cell)
					// We looped on the whole hash table. Couldn't find a better hash
					return false;

				unsigned int next_cell_hash_key = hash_cell_data.hash_keys[next_hash_cell_index];
				if (next_cell_hash_key == hash_key || next_cell_hash_key == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
				{
					// Stopping if we found our proper cell (with our hash) or if we found an empty cell

					in_out_hash_cell_index = next_hash_cell_index;

					// We found an empty cell
					return true;
				}
			}

			// Linear probing couldn't find a better position in the hash grid
			return false;
		}
		else
			// This is already our hash, no collision
			return true;
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
};

#endif
