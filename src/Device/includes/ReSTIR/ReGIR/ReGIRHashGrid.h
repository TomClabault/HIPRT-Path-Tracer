/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_H

#include "Device/includes/HashGrid.h"
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
		unsigned int hash_grid_cell_index = hash(soa.m_total_number_of_cells, world_position, current_camera, hash_key);
		if (!m_hash_grid.resolve_collision<ReGIR_LinearProbingSteps>(hash_cell_data.hash_keys, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key))
			return;

		int reservoir_index_in_grid = hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		store_full_reservoir(soa, reservoir, reservoir_index_in_grid);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(soa.m_total_number_of_cells, world_position, current_camera, hash_key);
		if (!m_hash_grid.resolve_collision<ReGIR_LinearProbingSteps>(hash_cell_data.hash_keys, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key) || hash_cell_data.grid_cells_alive[hash_grid_cell_index] == 0u)
			return HashGrid::UNDEFINED_HASH_KEY;

		return hash_grid_cell_index;
	}

	HIPRT_DEVICE unsigned int get_reservoir_index_in_grid(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell) const
	{
		unsigned int hash_grid_cell_index = get_hash_grid_cell_index(soa, hash_cell_data, world_position, current_camera);
		if (hash_grid_cell_index == HashGrid::UNDEFINED_HASH_KEY)
			return HashGrid::UNDEFINED_HASH_KEY;

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
		if (reservoir_index_in_grid == HashGrid::UNDEFINED_HASH_KEY)
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
			if (reservoir_index_in_grid == HashGrid::UNDEFINED_HASH_KEY)
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
		unsigned int hash_cell_index = hash(soa.m_total_number_of_cells, world_position, current_camera, hash_key);

		return hash_cell_index;
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, const HIPRTCamera& current_camera) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(soa.m_total_number_of_cells, world_position, current_camera, hash_key);

		if (!m_hash_grid.resolve_collision<ReGIR_LinearProbingSteps>(hash_cell_data.hash_keys, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key))
			return HashGrid::UNDEFINED_HASH_KEY;
		else
			return hash_grid_cell_index;
	}

	HIPRT_DEVICE float3 jitter_world_position(float3 original_world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng, float jittering_radius = 0.5f) const
	{
		float3 random_offset = make_float3(rng(), rng(), rng()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);

		return original_world_position + random_offset * compute_adaptive_cell_size(original_world_position, current_camera) * jittering_radius;
	}

	/**
	 * Reference: [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boissé, 2021]
	 */
	HIPRT_DEVICE float compute_adaptive_cell_size(float3 world_position, const HIPRTCamera& current_camera) const
	{
		float3 camera_position = current_camera.position;
		float target_projected_size = m_grid_cell_target_projected_size_ratio;// current_camera.sensor_width* m_grid_cell_target_projected_size_ratio;
		float min_cell_size = m_grid_cell_min_size;
		float vertical_fov = current_camera.vertical_fov;
		int width = current_camera.sensor_width;
		int height = current_camera.sensor_height;

		float cell_size_step = hippt::length(world_position - camera_position) * tanf(target_projected_size * vertical_fov * hippt::max(1.0f / height, (float)height / hippt::square(width)));
		float log_step = floorf(log2f(cell_size_step / min_cell_size));

		return hippt::max(min_cell_size, min_cell_size * exp2f(log_step));
	}

	/**
	 * Returns the hash cell index of the given world position and camera position. Does not resolve collisions.
	 * The hash key for resolving collision is given in 'out_hash_key'
	 */
	HIPRT_DEVICE unsigned int hash(unsigned int total_number_of_cells, float3 world_position, const HIPRTCamera& current_camera, unsigned int& out_hash_key) const
	{
		float cell_size = compute_adaptive_cell_size(world_position, current_camera);

		unsigned int grid_coord_x = static_cast<int>(floorf(world_position.x / cell_size));
		unsigned int grid_coord_y = static_cast<int>(floorf(world_position.y / cell_size));
		unsigned int grid_coord_z = static_cast<int>(floorf(world_position.z / cell_size));

		// Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boissé, 2021]
		out_hash_key = hippt::max(1u, h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x)))));
		
		unsigned int cell_hash = h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x)))) % total_number_of_cells;

		return cell_hash;
	}

	HashGrid m_hash_grid;

	static constexpr float DEFAULT_GRID_SIZE = 2.5f;

	float m_grid_cell_min_size = 0.3f;
	float m_grid_cell_target_projected_size_ratio = 25.0f;

private:
	/**
	 * PCG for the first hash function
	 */
	HIPRT_DEVICE unsigned int h1_pcg(unsigned int seed) const
	{
		unsigned int state = seed * 747796405u + 2891336453u;
		unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		
		return (word >> 22u) ^ word;
	}

	HIPRT_HOST_DEVICE unsigned int h1_pcg(float seed) const
	{
		return h1_pcg(hippt::float_as_uint(seed));
	}

	/**
	 * xxhash32 for the second hash function
	 */
	HIPRT_DEVICE unsigned int h2_xxhash32(unsigned int seed) const
	{
		constexpr unsigned int PRIME32_2 = 2246822519U;
		constexpr unsigned int PRIME32_3 = 3266489917U;
		constexpr unsigned int PRIME32_4 = 668265263U;
		constexpr unsigned int PRIME32_5 = 374761393U;

		unsigned int h32 = seed + PRIME32_5;

		h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
		h32 = PRIME32_2*(h32^(h32 >> 15));
		h32 = PRIME32_3*(h32^(h32 >> 13));

		return h32^(h32 >> 16);
	}

	HIPRT_HOST_DEVICE unsigned int h2_xxhash32(float seed) const
	{
		return h2_xxhash32(hippt::float_as_uint(seed));
	}
};

#endif
