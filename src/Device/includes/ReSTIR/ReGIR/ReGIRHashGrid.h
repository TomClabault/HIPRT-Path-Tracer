/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_H

#include "Device/includes/HashGrid.h"
#include "Device/includes/HashGridHash.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridSoADevice.h"
#include "Device/includes/ReSTIR/ReGIR/ShadingSettings.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"

struct ReGIRHashGrid
{
	HIPRT_DEVICE void reset_reservoir(ReGIRHashGridSoADevice& soa, unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell)
	{
		int reservoir_index_in_grid = hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		soa.reservoirs.store_reservoir_opt(reservoir_index_in_grid, ReGIRReservoir());
		soa.samples.store_sample(reservoir_index_in_grid, ReGIRReservoir().sample);
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, ReGIRHashGridSoADevice& soa, ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell)
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash_position_camera(soa.m_total_number_of_cells, world_position, current_camera, m_grid_cell_target_projected_size, m_grid_cell_min_size, hash_key);
		if (!HashGrid::resolve_collision<ReGIR_LinearProbingSteps>(hash_cell_data.hash_keys, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key))
			return;

		int reservoir_index_in_grid = hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		store_full_reservoir(soa, reservoir, reservoir_index_in_grid);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash_position_camera(soa.m_total_number_of_cells, world_position, current_camera, m_grid_cell_target_projected_size, m_grid_cell_min_size, hash_key);
		if (!HashGrid::resolve_collision<ReGIR_LinearProbingSteps>(hash_cell_data.hash_keys, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key) || hash_cell_data.grid_cell_alive[hash_grid_cell_index] == 0u)
			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

		return hash_grid_cell_index;
	}

	HIPRT_DEVICE unsigned int get_reservoir_index_in_grid(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell) const
	{
		unsigned int hash_grid_cell_index = get_hash_grid_cell_index(soa, hash_cell_data, world_position, current_camera);
		if (hash_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

		return hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;
	}

	HIPRT_DEVICE void store_full_reservoir(ReGIRHashGridSoADevice& soa, const ReGIRReservoir& reservoir, int reservoir_index_in_grid)
	{
		if (reservoir.UCW <= 0.0f)
		{
			soa.reservoirs.UCW[reservoir_index_in_grid] = reservoir.UCW;
			
			// No need to store the rest if the UCW is invalid, we can already return
			return;
		}

		soa.reservoirs.store_reservoir_opt(reservoir_index_in_grid, reservoir);
		soa.samples.store_sample(reservoir_index_in_grid, reservoir.sample);
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir(const ReGIRHashGridSoADevice& soa, unsigned int reservoir_index_in_grid) const
	{
		if (reservoir_index_in_grid == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			return ReGIRReservoir();

		ReGIRReservoir reservoir;

		float UCW = soa.reservoirs.UCW[reservoir_index_in_grid];
		if (UCW <= 0.0f)
		{
			// If the reservoir doesn't have a valid sample, not even reading the rest of it
			ReGIRReservoir out;
			out.UCW = UCW;

			return out;
		}

		reservoir = soa.reservoirs.read_reservoir<false>(reservoir_index_in_grid);
		reservoir.UCW = UCW;
		reservoir.sample = soa.samples.read_sample(reservoir_index_in_grid);

		return reservoir;
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data,
		float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		unsigned int reservoir_index_in_grid = get_reservoir_index_in_grid(soa, hash_cell_data, world_position, current_camera, reservoir_index_in_cell);

		if (out_invalid_sample)
		{
			if (reservoir_index_in_grid == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				*out_invalid_sample = true;
			else
				*out_invalid_sample = false;
		}

		return read_full_reservoir(soa, reservoir_index_in_grid);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_no_collision_resolve(const ReGIRHashGridSoADevice& soa, 
		float3 world_position, const HIPRTCamera& current_camera) const
	{
		unsigned int hash_key;
		unsigned int hash_cell_index = hash_position_camera(soa.m_total_number_of_cells, world_position, current_camera, m_grid_cell_target_projected_size, m_grid_cell_min_size, hash_key);

		return hash_cell_index;
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash_position_camera(soa.m_total_number_of_cells, world_position, current_camera, m_grid_cell_target_projected_size, m_grid_cell_min_size, hash_key);

		if (!HashGrid::resolve_collision<ReGIR_LinearProbingSteps>(hash_cell_data.hash_keys, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key))
			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
		else
			return hash_grid_cell_index;
	}

	HIPRT_DEVICE float3 jitter_world_position(float3 original_world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng, float jittering_radius = 0.5f) const
	{
		float3 random_offset = make_float3(rng(), rng(), rng()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);

		return original_world_position + random_offset * compute_adaptive_cell_size(original_world_position, current_camera, m_grid_cell_target_projected_size, m_grid_cell_min_size) * jittering_radius;
	}

	HashGrid m_hash_grid;

	float m_grid_cell_min_size = 0.2f;
	float m_grid_cell_target_projected_size = 25.0f;
};

#endif
