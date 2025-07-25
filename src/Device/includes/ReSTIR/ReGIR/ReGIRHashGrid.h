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
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"

struct ReGIRHashGrid
{
	HIPRT_DEVICE static float compute_adaptive_cell_size_roughness(float3 world_position, const HIPRTCamera& current_camera, float roughness, float target_projected_size, float grid_cell_min_size)
	{
		int width = current_camera.sensor_width;
		int height = current_camera.sensor_height;
		
#if ReGIR_AdaptiveRoughnessGridPrecision == KERNEL_OPTION_TRUE
		if (roughness >= 0.08f && roughness < 0.2f)
		{
			float t = hippt::inverse_lerp(roughness, 0.08f, 0.2f);
			float res_increase_factor = hippt::lerp(2.0f, 5.0f, 1.0f - t);

			target_projected_size /= res_increase_factor;
			grid_cell_min_size /= res_increase_factor;
		}
		else if (roughness >= 0.2f && roughness < 0.35f)
		{
			float t = hippt::inverse_lerp(roughness, 0.2f, 0.35f);
			float res_increase_factor = hippt::lerp(1.0f, 2.0f, 1.0f - t);

			target_projected_size /= res_increase_factor;
			grid_cell_min_size /= res_increase_factor;
		}
		// 0.1 roughness: 4.5 & 0.05
		// 0.2 roughness: 12.5 & 0.2
		// 0.35 roughness: 25.0 & 0.3
#endif

		float cell_size_step = hippt::length(world_position - current_camera.position) * tanf(target_projected_size * current_camera.vertical_fov * hippt::max(1.0f / height, (float)height / hippt::square(width)));
		float log_step = floorf(log2f(cell_size_step / grid_cell_min_size));

		return hippt::max(grid_cell_min_size, grid_cell_min_size * exp2f(log_step));
	}

	HIPRT_DEVICE unsigned int custom_regir_hash(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, unsigned int total_number_of_cells, unsigned int& out_checksum) const
	{
		float cell_size = ReGIRHashGrid::compute_adaptive_cell_size_roughness(world_position, current_camera, roughness, m_grid_cell_target_projected_size, m_grid_cell_min_size);

		// Reference: SIGGRAPH 2022 - Advances in Spatial Hashing
		world_position = hash_periodic_shifting(world_position, cell_size);

		unsigned int grid_coord_x = static_cast<int>(floorf(world_position.x / cell_size));
		unsigned int grid_coord_y = static_cast<int>(floorf(world_position.y / cell_size));
		unsigned int grid_coord_z = static_cast<int>(floorf(world_position.z / cell_size));

		// Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boisse, 2021]
#if ReGIR_HashGridHashSurfaceNormal == KERNEL_OPTION_TRUE
		// And adding normal hasing from [World-Space Spatiotemporal Path Resampling for Path Tracing, 2023]
		unsigned int quantized_normal = hash_quantize_normal(surface_normal, m_normal_quantization_steps);
		unsigned int checksum = h2_xxhash32(quantized_normal + h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x)))));
		unsigned int cell_hash = h1_pcg(quantized_normal + h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x))))) % total_number_of_cells;
#else
		unsigned int checksum = h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x))));
		unsigned int cell_hash = h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x)))) % total_number_of_cells;
#endif

		out_checksum = checksum;
		return cell_hash;
	}

	HIPRT_DEVICE void reset_reservoir(ReGIRHashGridSoADevice& soa, unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell)
	{
		int reservoir_index_in_grid = hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		soa.reservoirs.store_reservoir_opt(reservoir_index_in_grid, ReGIRReservoir());
		soa.samples.store_sample(reservoir_index_in_grid, ReGIRReservoir().sample);
	}

	/**
	 * Overload if you already the hash grid cell index
	 */
	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, ReGIRHashGridSoADevice& soa, unsigned int hash_grid_cell_index, int reservoir_index_in_cell)
	{
		int reservoir_index_in_grid = hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		store_full_reservoir(soa, reservoir, reservoir_index_in_grid);
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, ReGIRHashGridSoADevice& soa, ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, int reservoir_index_in_cell)
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = custom_regir_hash(world_position, surface_normal, current_camera, roughness, soa.m_total_number_of_cells, hash_key);
		if (!HashGrid::resolve_collision<ReGIR_HashGridCollisionResolutionMaxSteps>(hash_cell_data.checksums, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key))
			return;

		store_reservoir_and_sample_opt(reservoir, soa, hash_grid_cell_index, reservoir_index_in_cell);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = custom_regir_hash(world_position, surface_normal, current_camera, roughness, soa.m_total_number_of_cells, hash_key);
		unsigned int original = hash_grid_cell_index;
		if (!HashGrid::resolve_collision<ReGIR_HashGridCollisionResolutionMaxSteps>(hash_cell_data.checksums, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key) || hash_cell_data.grid_cell_alive[hash_grid_cell_index] == 0u)
		{
			if (hippt::is_pixel_index(1000, 699 - 1 - 354))
			{
		        printf("collision not resolved: ");
				for (int i = 0; i < 32; i++)
					printf("%u | ", hippt::atomic_load(&hash_cell_data.grid_cell_alive[original + i]));

				printf("\n");
			}


			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
		}

		return hash_grid_cell_index;
	}

	/**
	 * Overload if you already the hash grid cell index
	 */
	HIPRT_DEVICE unsigned int get_reservoir_index_in_grid(const ReGIRHashGridSoADevice& soa, unsigned int hash_grid_cell_index, int reservoir_index_in_cell) const
	{
		return hash_grid_cell_index * soa.reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;
	}

	HIPRT_DEVICE unsigned int get_reservoir_index_in_grid(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, int reservoir_index_in_cell) const
	{
		unsigned int hash_grid_cell_index = get_hash_grid_cell_index(soa, hash_cell_data, world_position, surface_normal, current_camera, roughness);
		if (hash_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

		return get_reservoir_index_in_grid(soa, hash_grid_cell_index, reservoir_index_in_cell);
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

	/**
	 * Override if you already have the hash grid cell index
	 */
	HIPRT_DEVICE ReGIRReservoir read_full_reservoir(const ReGIRHashGridSoADevice& soa, unsigned int hash_grid_cell_index, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		unsigned int reservoir_index_in_grid = get_reservoir_index_in_grid(soa, hash_grid_cell_index, reservoir_index_in_cell);

		if (out_invalid_sample)
		{
			if (reservoir_index_in_grid == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				*out_invalid_sample = true;
			else
				*out_invalid_sample = false;
		}

		return read_full_reservoir(soa, reservoir_index_in_grid);
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data,
		float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		unsigned int reservoir_index_in_grid = get_reservoir_index_in_grid(soa, hash_cell_data, world_position, surface_normal, current_camera, roughness, reservoir_index_in_cell);

		if (out_invalid_sample)
		{
			if (reservoir_index_in_grid == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				*out_invalid_sample = true;
			else
				*out_invalid_sample = false;
		}

		return read_full_reservoir(soa, reservoir_index_in_grid);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos(const ReGIRHashGridSoADevice& soa, const ReGIRHashCellDataSoADevice& hash_cell_data, 
		float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = custom_regir_hash(world_position, surface_normal, current_camera, roughness, soa.m_total_number_of_cells, hash_key);

		if (!HashGrid::resolve_collision<ReGIR_HashGridCollisionResolutionMaxSteps>(hash_cell_data.checksums, soa.m_total_number_of_cells, hash_grid_cell_index, hash_key))
			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
		else
			return hash_grid_cell_index;
	}

	HIPRT_DEVICE float3 jitter_world_position(float3 original_world_position, const HIPRTCamera& current_camera, float roughness, Xorshift32Generator& rng, float jittering_radius = 0.5f) const
	{
		float3 random_offset = make_float3(rng(), rng(), rng()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);

		return original_world_position + random_offset * ReGIRHashGrid::compute_adaptive_cell_size_roughness(original_world_position, current_camera, roughness, m_grid_cell_target_projected_size, m_grid_cell_min_size) * jittering_radius;
	}

	HashGrid m_hash_grid;

	float m_grid_cell_min_size = 0.4f;
	float m_grid_cell_target_projected_size = 20.0f;
	int m_normal_quantization_steps = 2;
};

#endif
