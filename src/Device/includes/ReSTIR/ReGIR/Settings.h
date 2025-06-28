/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_SETTINGS_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/ReGIRHashGrid.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridSoADevice.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Xorshift.h"

struct ReGIRGridFillSettings
{
	HIPRT_DEVICE ReGIRGridFillSettings() : ReGIRGridFillSettings(true) {}
		
	HIPRT_DEVICE ReGIRGridFillSettings(bool primary_hit)
	{
		light_sample_count_per_cell_reservoir = 32;

		reservoirs_count_per_grid_cell_non_canonical = primary_hit ? 48 : 8;
		reservoirs_count_per_grid_cell_canonical = primary_hit ? 24 : 4;
	}

	// How many light samples are resampled into each reservoir of the grid cell
	int light_sample_count_per_cell_reservoir;

	HIPRT_DEVICE int get_non_canonical_reservoir_count_per_cell() const { return reservoirs_count_per_grid_cell_non_canonical; }
	HIPRT_DEVICE int get_canonical_reservoir_count_per_cell() const { return reservoirs_count_per_grid_cell_canonical; }
	HIPRT_DEVICE int get_total_reservoir_count_per_cell() const { return reservoirs_count_per_grid_cell_canonical + reservoirs_count_per_grid_cell_non_canonical; }

	HIPRT_DEVICE int* get_non_canonical_reservoir_count_per_cell_ptr() { return &reservoirs_count_per_grid_cell_non_canonical; }
	HIPRT_DEVICE int* get_canonical_reservoir_count_per_cell_ptr() { return &reservoirs_count_per_grid_cell_canonical; }

	HIPRT_DEVICE bool reservoir_index_in_cell_is_canonical(int reservoir_index_in_cell) const { return reservoir_index_in_cell >= get_non_canonical_reservoir_count_per_cell(); }

private:
	// How many reservoirs are going to be produced per each cell of the grid.
	// 
	// These reservoirs are "non-canonical" as they can include visibility/cosine terms
	// if visibility reuse is used
	// 
	// Because these visibility/cosine terms are approximate, using these reservoirs alone
	// is going to be biased and so we need to combine them with "canonical" reservoirs during
	// shading for unbiasedness
	//
	// In the grid buffers, these reservoirs are stored first, i.e., for a grid cell with 3 non-canonical reservoirs
	// and 1 canonical reservoir:
	//
	// [non-canon, non-canon, non-canon, canonical]
	int reservoirs_count_per_grid_cell_non_canonical;

	// Number of canonical reservoirs per cell
	// 
	// In the grid buffers, these reservoirs are stored last, i.e., for a grid cell with 3 non-canonical reservoirs
	// and 1 canonical reservoir:
	// 
	// [non-canon, non-canon, non-canon, canonical]
	int reservoirs_count_per_grid_cell_canonical;
};

struct ReGIRSpatialReuseSettings
{
	bool do_spatial_reuse = true;
 	// If true, the same random seed will be used by all grid cells during the spatial reuse for a given frame
 	// This has the effect of coalescing neighbors memory accesses which improves performance
	bool do_coalesced_spatial_reuse = false;

	int spatial_reuse_pass_count = 2;
	int spatial_reuse_pass_index = 0;

	int spatial_neighbor_count = 4;
	int reuse_per_neighbor_count = 4;
	// When picking a random cell in the neighborhood for reuse, if that
	// cell is out of the grid or if that cell is not alive etc..., we're
	// going to retry another cell this many times
	//
	// This improves the chances that we're actually going to have a good
	// neighbor to reuse from --> more reuse --> less variance
	int retries_per_neighbor = 4;
	int spatial_reuse_radius = 1;
};

struct ReGIRCorrelationReductionSettings
{
	bool do_correlation_reduction = true;

	int correlation_reduction_factor = 4;
	int correl_frames_available = 0;
	unsigned int correl_reduction_current_grid = 0;

	ReGIRHashGridSoADevice correlation_reduction_grid;
};

struct ReGIRSettings
{
	HIPRT_DEVICE const ReGIRHashGridSoADevice& get_initial_reservoirs_grid(bool primary_hit) const { return primary_hit ? initial_reservoirs_primary_hits_grid : initial_reservoirs_secondary_hits_grid; }
	HIPRT_DEVICE ReGIRHashGridSoADevice& get_initial_reservoirs_grid(bool primary_hit) { return primary_hit ? initial_reservoirs_primary_hits_grid : initial_reservoirs_secondary_hits_grid; }

	HIPRT_DEVICE const ReGIRHashGridSoADevice& get_spatial_output_reservoirs_grid(bool primary_hit) const 
	{
		if (spatial_reuse.spatial_reuse_pass_count & 1)
			return primary_hit ? spatial_output_primary_hits_grid : spatial_output_secondary_hits_grid; 
		else
			return primary_hit ? initial_reservoirs_primary_hits_grid : initial_reservoirs_secondary_hits_grid;
	}

	HIPRT_DEVICE ReGIRHashGridSoADevice& get_spatial_output_reservoirs_grid(bool primary_hit) 
	{ 
		if (spatial_reuse.spatial_reuse_pass_count & 1)
			return primary_hit ? spatial_output_primary_hits_grid : spatial_output_secondary_hits_grid;
		else
			return primary_hit ? initial_reservoirs_primary_hits_grid : initial_reservoirs_secondary_hits_grid;
	}

	HIPRT_DEVICE const ReGIRHashCellDataSoADevice& get_hash_cell_data_soa(bool primary_hit) const { return primary_hit ? hash_cell_data_primary_hits : hash_cell_data_secondary_hits; }
	HIPRT_DEVICE ReGIRHashCellDataSoADevice& get_hash_cell_data_soa(bool primary_hit) { return primary_hit ? hash_cell_data_primary_hits : hash_cell_data_secondary_hits; }

	HIPRT_DEVICE const ReGIRGridFillSettings& get_grid_fill_settings(bool primary_hit) const { return primary_hit ? grid_fill_primary_hits : grid_fill_secondary_hits; }

	HIPRT_DEVICE const AtomicType<float>* get_non_canonical_pre_integration_factor_buffer(bool primary_hit) const { return primary_hit ? non_canonical_pre_integration_factors_primary_hits : non_canonical_pre_integration_factors_secondary_hits; }
	HIPRT_DEVICE AtomicType<float>* get_non_canonical_pre_integration_factor_buffer(bool primary_hit) { return primary_hit ? non_canonical_pre_integration_factors_primary_hits : non_canonical_pre_integration_factors_secondary_hits; }

	HIPRT_DEVICE const AtomicType<float>* get_canonical_pre_integration_factor_buffer(bool primary_hit) const { return primary_hit ? canonical_pre_integration_factors_primary_hits : canonical_pre_integration_factors_secondary_hits; }
	HIPRT_DEVICE AtomicType<float>* get_canonical_pre_integration_factor_buffer(bool primary_hit) { return primary_hit ? canonical_pre_integration_factors_primary_hits : canonical_pre_integration_factors_secondary_hits; }

	HIPRT_DEVICE float get_non_canonical_pre_integration_factor(unsigned hash_grid_cell_index, bool primary_hit) const { return get_non_canonical_pre_integration_factor_buffer(primary_hit)[hash_grid_cell_index]; }
	HIPRT_DEVICE float get_canonical_pre_integration_factor(unsigned hash_grid_cell_index, bool primary_hit) const { return get_canonical_pre_integration_factor_buffer(primary_hit)[hash_grid_cell_index]; }

	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE float3 get_cell_size(float3 world_position, const HIPRTCamera& current_camera, float roughness) const
	{
		float cell_size = ReGIRHashGrid::compute_adaptive_cell_size_roughness(world_position, current_camera, roughness, hash_grid.m_grid_cell_target_projected_size, hash_grid.m_grid_cell_min_size);

		return make_float3(cell_size, cell_size, cell_size);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit) const
	{
		return hash_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, surface_normal, current_camera, roughness);
	}

	///////////////////// Delegating to the grid for these functions /////////////////////

	/**
	 * Returns the given reservoir index in the given grid cell index in the given grid of reservoirs
	 */
	HIPRT_DEVICE ReGIRReservoir get_reservoir_from_grid_cell_index(ReGIRHashGridSoADevice reservoir_grid, unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell)
	{
		return hash_grid.read_full_reservoir(reservoir_grid, hash_grid.get_reservoir_index_in_grid(reservoir_grid, hash_grid_cell_index, reservoir_index_in_cell));
	}

	/**
	 * Returns a reservoir from the grid cell that corresponds to the given world position, surface normal.
	 * The returned reservoir is a non-canonical reservoir given by the non_canonical_reservoir_number.
	 *
	 * That number must be in the range [0, get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell()[.
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_non_canonical_reservoir_from_index(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, unsigned int non_canonical_reservoir_number, bool* out_invalid_sample = nullptr) const
	{
		return get_reservoir_for_shading_from_cell_indices(world_position, surface_normal, current_camera, roughness, primary_hit, non_canonical_reservoir_number, out_invalid_sample);
	}

	/**
	 * Overlaod if you already the hash grid cell index
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_non_canonical_reservoir_from_index(unsigned int hash_grid_cell_index, bool primary_hit, unsigned int non_canonical_reservoir_number, bool* out_invalid_sample = nullptr) const
	{
		return get_reservoir_for_shading_from_cell_indices(hash_grid_cell_index, primary_hit, non_canonical_reservoir_number, out_invalid_sample);
	}

	/**
	 * Same as get_cell_non_canonical_reservoir_from_index() but for canonical reservoirs.
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_canonical_reservoir_from_index(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, unsigned int canonical_reservoir_number, bool* out_invalid_sample = nullptr) const
	{
		unsigned int non_canonical_reservoir_count = get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();

		return get_reservoir_for_shading_from_cell_indices(world_position, surface_normal, current_camera, roughness, primary_hit, non_canonical_reservoir_count + canonical_reservoir_number, out_invalid_sample);
	}

	/**
	 * Overlaod if you already the hash grid cell index
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_canonical_reservoir_from_index(unsigned int hash_grid_cell_index, bool primary_hit, unsigned int canonical_reservoir_number, bool* out_invalid_sample = nullptr) const
	{
		unsigned int non_canonical_reservoir_count = get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();

		return get_reservoir_for_shading_from_cell_indices(hash_grid_cell_index, primary_hit, non_canonical_reservoir_count + canonical_reservoir_number, out_invalid_sample);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_non_canonical_reservoir(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, Xorshift32Generator& rng, bool* out_invalid_sample = nullptr) const
	{
		int random_non_canonical_reservoir_index_in_cell = rng.random_index(get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell());

		return get_reservoir_for_shading_from_cell_indices(world_position, surface_normal, current_camera, roughness, primary_hit, random_non_canonical_reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_canonical_reservoir(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, Xorshift32Generator& rng, bool* out_invalid_sample = nullptr) const
	{
		int random_canonical_reservoir_index_in_cell = rng.random_index(get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell());

		unsigned int non_canonical_reservoir_count = get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();
		return get_reservoir_for_shading_from_cell_indices(world_position, surface_normal, current_camera, roughness, primary_hit, non_canonical_reservoir_count + random_canonical_reservoir_index_in_cell, out_invalid_sample);
	}

	/**
	 * Overload if you already have the hash grid cell index
	 */
	HIPRT_DEVICE ReGIRReservoir get_reservoir_for_shading_from_cell_indices(unsigned int hash_grid_cell_index, bool primary_hit, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		if (spatial_reuse.do_spatial_reuse)
			// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
			return hash_grid.read_full_reservoir(get_spatial_output_reservoirs_grid(primary_hit), hash_grid_cell_index, reservoir_index_in_cell, out_invalid_sample);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return hash_grid.read_full_reservoir(get_initial_reservoirs_grid(primary_hit), hash_grid_cell_index, reservoir_index_in_cell, out_invalid_sample);
	}

	/**
	 * If 'out_invalid_sample' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_reservoir_for_shading_from_cell_indices(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		unsigned int hash_grid_cell_index = hash_grid.get_hash_grid_cell_index(get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, surface_normal, current_camera, roughness);

		return get_reservoir_for_shading_from_cell_indices(hash_grid_cell_index, primary_hit, reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE unsigned int get_neighbor_replay_hash_grid_cell_index_for_shading(float3 shading_point, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, bool replay_canonical, bool do_jittering, float jittering_radius, Xorshift32Generator& rng) const
	{
		unsigned int neighbor_cell_index;
		if (replay_canonical)
			neighbor_cell_index = find_valid_jittered_neighbor_cell_index<true>(shading_point, surface_normal, current_camera, roughness, primary_hit, do_jittering, jittering_radius, rng);
        else
            neighbor_cell_index = find_valid_jittered_neighbor_cell_index<false>(shading_point, surface_normal, current_camera, roughness, primary_hit, do_jittering, jittering_radius, rng);

		if (neighbor_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		{
			// Advancing the RNG simulating the random reservoir pick within the grid cell
			if (replay_canonical)
				rng.random_index(get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell());
			else
				rng.random_index(get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell());
		}

		return neighbor_cell_index;
	}

	template <bool fallbackOnCenterCell>
	HIPRT_DEVICE unsigned int find_valid_jittered_neighbor_cell_index(float3 world_position, float3 shading_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, bool do_jittering, float jittering_radius, Xorshift32Generator& rng) const
	{
		unsigned int retry = 0;
		unsigned int neighbor_grid_cell_index;
		
		do
		{
			float3 jittered;
			if (do_jittering)
				jittered = hash_grid.jitter_world_position(world_position, current_camera, roughness, rng, jittering_radius);
			else
				jittered = world_position;

			neighbor_grid_cell_index = hash_grid.get_hash_grid_cell_index(get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), jittered, shading_normal, current_camera, roughness);
			if (neighbor_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			{
				// This part here is to avoid race concurrency issues from the Megakernel shader:
				//
				// In the megakernel, rays that bounce around the scene may hit cells that have never been hit
				// before. This will cause these cells to become alive.
				//
				// When a cell is alive, it may be picked during the megakernel shading with ReGIR.
				// However, the cells are only filled during the grid fill pass/spatial reuse pass of ReGIR
				//
				// What can happen is that the Megakernel sets some grid cells alive and some other threads of the Megakernel then
				// tries to use that grid cell for shading (since that grid cell is now alive). This though is that the grid fill pass
				// hasn't been launched yet (it will be launched at the next frame) and so the grid cell, even though it's alive, doesn't
				// contain valid data --> reading invalid reservoir data for shading
				//
				// So we're checking here if the cell contains valid data and if it doesn't, we're going to position the cell
				// as being invalid with UNDEFINED_HASH_KEY

				float UCW;
				if (spatial_reuse.do_spatial_reuse)
					UCW = get_spatial_output_reservoirs_grid(primary_hit).reservoirs.UCW[neighbor_grid_cell_index * get_number_of_reservoirs_per_cell(primary_hit)];
				else
					UCW = get_initial_reservoirs_grid(primary_hit).reservoirs.UCW[neighbor_grid_cell_index * get_number_of_reservoirs_per_cell(primary_hit)];

				if (UCW == ReGIRReservoir::UNDEFINED_UCW)
					neighbor_grid_cell_index = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
			}

			retry++;
		} while (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX && retry < ReGIR_ShadingJitterTries);

		if (fallbackOnCenterCell && neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX && retry == ReGIR_ShadingJitterTries)
			// We couldn't find a valid neighbor and the fallback on center cell is enabled: we're going to return the index of the center cell
			neighbor_grid_cell_index = hash_grid.get_hash_grid_cell_index(get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, shading_normal, current_camera, roughness);

		return neighbor_grid_cell_index;
	}

	template <bool getCanonicalReservoir>
	HIPRT_DEVICE ReGIRReservoir get_random_reservoir_in_grid_cell_for_shading(unsigned int grid_cell_index, bool primary_hit, Xorshift32Generator& rng) const
	{
		unsigned int reservoir_index_in_cell;
		// If this stays to 0, this means that we're going to read the reservoirs from
		// either the regular initial candidates grid or spatial reuse grid
		// 
		// If this is > 0, then we're going to read the reservoirs from the supersampling grid
		unsigned int grid_index = 0;

		if constexpr (getCanonicalReservoir)
		{
			if (supersampling.do_correlation_reduction)
			{
				// If correlation reduction is enabled, we want to pick a reservoir from the whole pool of (regular reservoirs + correlation reduction reservoirs)
				reservoir_index_in_cell = rng.random_index(get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell() * (supersampling.correl_frames_available + 1));
			}
			else
				reservoir_index_in_cell = rng.random_index(get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell());
		}
		else
		{
			if (supersampling.do_correlation_reduction)
				// If correlation reduction is enabled, we want to pick a reservoir from the whole pool of (regular reservoirs + correlation reduction reservoirs)
				reservoir_index_in_cell = rng.random_index(get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell() * (supersampling.correl_frames_available + 1));
			else
				reservoir_index_in_cell = rng.random_index(get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell());
		}

		if constexpr (getCanonicalReservoir)
		{
			grid_index = reservoir_index_in_cell / get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell();
			reservoir_index_in_cell %= get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell();
		}
		else
		{
			grid_index = reservoir_index_in_cell / get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();
			reservoir_index_in_cell %= get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();
		}

		unsigned int canonical_offset = getCanonicalReservoir ? get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell() : 0;
		unsigned int reservoir_index_in_grid = grid_cell_index * get_number_of_reservoirs_per_cell(primary_hit) + canonical_offset + reservoir_index_in_cell;

		if (grid_index == 0 || !primary_hit)
		{
			// Reading from the regular grids because the grid index is 0 or we're reading
			// secondary hits because we're not doing correlation reduction for secondary hits

			if (spatial_reuse.do_spatial_reuse)
				// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
				return hash_grid.read_full_reservoir(get_spatial_output_reservoirs_grid(primary_hit), reservoir_index_in_grid);
			else
				// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
				return hash_grid.read_full_reservoir(get_initial_reservoirs_grid(primary_hit), reservoir_index_in_grid);
		}
		else
		{
			// If we have grid_index == 1 here for example, this is going to be grid index 0 of the supersampling grid
			// so we have grid_index - 1
			unsigned int reservoir_index_in_supersample_grid = reservoir_index_in_grid + (grid_index - 1) * get_number_of_reservoirs_per_grid(primary_hit);

			return hash_grid.read_full_reservoir(supersampling.correlation_reduction_grid, reservoir_index_in_supersample_grid);
		}
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
	 * 
	 * The 'opt' suffix of the function means that the UCW of the reservoir will be read first and the rest of the reservoir
	 * will only be read if the UCW is > 0.0f.
	 * If the UCW is <= 0.0f, the returned reservoir will have uninitialized values in all of its fields
	 */
	HIPRT_DEVICE ReGIRReservoir get_temporal_reservoir_opt(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		return hash_grid.read_full_reservoir(get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, surface_normal, current_camera, roughness, reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE ReGIRReservoir get_grid_fill_output_reservoir_opt(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		// The output of the grid fill pass is in the current frame grid so we can call the temporal method with
		// index -1
		return get_temporal_reservoir_opt(world_position, surface_normal, current_camera, roughness, primary_hit, reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt(ReGIRHashGridSoADevice& output_reservoirs_grid, ReGIRHashCellDataSoADevice& output_reservoirs_cell_data, const ReGIRReservoir& reservoir, float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, int reservoir_index_in_cell)
	{
		hash_grid.store_reservoir_and_sample_opt(reservoir, output_reservoirs_grid, output_reservoirs_cell_data, world_position, surface_normal, current_camera, roughness, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt(const ReGIRReservoir& reservoir, float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, int reservoir_index_in_cell)
	{
		store_spatial_reservoir_opt(get_spatial_output_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), reservoir, world_position, surface_normal, current_camera, roughness, reservoir_index_in_cell);
	}

	/**
	 * Overload if you already have the hash grid cell index
	 */
	HIPRT_DEVICE void store_reservoir_opt(ReGIRReservoir reservoir, unsigned int hash_grid_cell_index, bool primary_hit, int reservoir_index_in_cell)
	{
		hash_grid.store_reservoir_and_sample_opt(reservoir, get_initial_reservoirs_grid(primary_hit), hash_grid_cell_index, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void store_reservoir_opt(ReGIRReservoir reservoir, float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit, int reservoir_index_in_cell)
	{
		hash_grid.store_reservoir_and_sample_opt(reservoir, get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, surface_normal, current_camera, roughness, reservoir_index_in_cell);
	}

	HIPRT_DEVICE ColorRGB32F get_random_cell_color(float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, float roughness, bool primary_hit) const
	{
		unsigned int cell_index = hash_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, surface_normal, current_camera, roughness);
		if (cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			return ColorRGB32F(0.0f);

		return ColorRGB32F::random_color(cell_index);
	}

	HIPRT_DEVICE unsigned int get_total_number_of_cells_per_grid(bool primary_hit_cells) const
	{
		return get_initial_reservoirs_grid(primary_hit_cells).m_total_number_of_cells;
	}

	HIPRT_DEVICE unsigned int get_number_of_reservoirs_per_grid(bool primary_hit_cells) const
	{
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return get_total_number_of_cells_per_grid(primary_hit_cells) * get_grid_fill_settings(primary_hit_cells).get_total_reservoir_count_per_cell();
	}

	HIPRT_DEVICE unsigned int get_number_of_reservoirs_per_cell(bool primary_hit_cells) const
	{
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return get_grid_fill_settings(primary_hit_cells).get_total_reservoir_count_per_cell();
	}

	/**
	 * Resets all the reservoirs of all the grids at the given 'reservoir_index'
	 */
	HIPRT_DEVICE void reset_reservoirs(unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell, bool primary_hit)
	{
		hash_grid.reset_reservoir(get_initial_reservoirs_grid(primary_hit), hash_grid_cell_index, reservoir_index_in_cell);

		// Also clearing the spatial reuse output buffers (grid) if spatial reuse is enabled
		if (spatial_reuse.do_spatial_reuse)
			hash_grid.reset_reservoir(get_spatial_output_reservoirs_grid(primary_hit), hash_grid_cell_index, reservoir_index_in_cell);
	}

	HIPRT_DEVICE static void insert_hash_cell_point_normal(ReGIRHashCellDataSoADevice& hash_cell_data_to_update,
		unsigned int hash_grid_cell_index, float3 world_position, float3 shading_normal, int primitive_index, const DeviceUnpackedEffectiveMaterial& material)
	{
		// TODO is this atomic needed since we can only be here if the cell was unoccupied?
		if (hippt::atomic_compare_exchange(&hash_cell_data_to_update.hit_primitive[hash_grid_cell_index], ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE, primitive_index) == ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE)
		{
			hash_cell_data_to_update.world_points[hash_grid_cell_index] = world_position;
			hash_cell_data_to_update.world_normals[hash_grid_cell_index].pack(shading_normal);
			hash_cell_data_to_update.roughness[hash_grid_cell_index] = material.roughness * 255.0f;
			hash_cell_data_to_update.metallic[hash_grid_cell_index] = material.metallic * 255.0f;
			hash_cell_data_to_update.specular[hash_grid_cell_index] = material.specular * 255.0f;

			hash_cell_data_to_update.sum_points[hash_grid_cell_index] = world_position;
			hash_cell_data_to_update.num_points[hash_grid_cell_index] = 1;
		}

		// Because we just inserted into that grid cell, it is now alive
		// Only go through all that atomic stuff if the cell isn't alive
		if (hash_cell_data_to_update.grid_cell_alive[hash_grid_cell_index] == 0)
		{
			// TODO is this atomic needed since we can only be here if the cell was unoccoupied?

			if (hippt::atomic_compare_exchange(&hash_cell_data_to_update.grid_cell_alive[hash_grid_cell_index], 0u, 1u) == 0u)
			{
				unsigned int cell_alive_index = hippt::atomic_fetch_add(hash_cell_data_to_update.grid_cells_alive_count, 1u);

				hash_cell_data_to_update.grid_cells_alive_list[cell_alive_index] = hash_grid_cell_index;
			}
		}
	}

	HIPRT_DEVICE static void update_hash_cell_representative_data(ReGIRHashCellDataSoADevice& hash_cell_data_to_update,
		unsigned int hash_grid_cell_index, float3 world_position, float3 shading_normal, int primitive_index, const DeviceUnpackedEffectiveMaterial& material)
	{
		unsigned int current_num_points = hash_cell_data_to_update.num_points[hash_grid_cell_index];

		if (current_num_points >= 255)
			// We've already accumulated enough points for that grid cell, not doing more to save on perf
			return;
		
		// We're going to add our point to the sum of points for that grid cell.
		// 
		// We need to do that atomically in case many threads want to add to the same grid cell at the same time
		// so we're going to lock that grid cell by setting the existing distance to CELL_LOCKED_DISTANCE, indicating that a thread is already
		// incrementing the sum of points for that grid cell

		unsigned int previous_num_points = hippt::atomic_compare_exchange(&hash_cell_data_to_update.num_points[hash_grid_cell_index], current_num_points, ReGIRHashCellDataSoADevice::CELL_LOCKED_SENTINEL_VALUE);

		if (previous_num_points == current_num_points && previous_num_points != ReGIRHashCellDataSoADevice::CELL_LOCKED_SENTINEL_VALUE)
		{
			// We have access to the cell if the value of the distance wasn't ReGIRHashCellDataSoADevice::CELL_LOCKED_DISTANCE
			// and if we are the one thread that swapped its value with the distance (previous_distance == existing_distance)

			// We can increment everything atomically here

			float3 current_sum_points = hash_cell_data_to_update.sum_points[hash_grid_cell_index];
			// Adding our point
			current_sum_points += world_position;

			// Computing the average of the points that have been added to that grid cell so far
			float3 average_cell_point = current_sum_points / (current_num_points + 1);

			float existing_distance = hippt::length(hash_cell_data_to_update.world_points[hash_grid_cell_index] - average_cell_point);
			float new_distance_to_average_point = hippt::length(world_position - average_cell_point);

			if (new_distance_to_average_point < existing_distance)
			{
				// If our point is closer to the center of the cell (approximated by the average of all hit points in the cell)
				// than the existing point, then our current hit (world pos, normal, primitive, ...) becomes the new
				// representative hit for that grid cell

				hash_cell_data_to_update.world_points[hash_grid_cell_index] = world_position;
				hash_cell_data_to_update.world_normals[hash_grid_cell_index].pack(shading_normal);
				hash_cell_data_to_update.hit_primitive[hash_grid_cell_index] = primitive_index;
				hash_cell_data_to_update.roughness[hash_grid_cell_index] = material.roughness * 255.0f;
				hash_cell_data_to_update.metallic[hash_grid_cell_index] = material.metallic * 255.0f;
				hash_cell_data_to_update.specular[hash_grid_cell_index] = material.specular * 255.0f;
			}

			// Writing back the new sum of points
			hash_cell_data_to_update.sum_points[hash_grid_cell_index] = current_sum_points;
			// Incrementing the number of points
			hash_cell_data_to_update.num_points[hash_grid_cell_index] = current_num_points + 1;
		}
	}

	HIPRT_DEVICE static void insert_hash_cell_data_static(
		const ReGIRHashGrid& hash_grid, ReGIRHashGridSoADevice& hash_grid_to_update, ReGIRHashCellDataSoADevice& hash_cell_data_to_update,
		float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, int primitive_index, const DeviceUnpackedEffectiveMaterial& material)
	{
		unsigned int checksum;
		unsigned int hash_grid_cell_index = hash_grid.custom_regir_hash(world_position, surface_normal, current_camera, material.roughness, hash_grid_to_update.m_total_number_of_cells, checksum);
		/*if (DEBUG)
		{
			printf("Debug grid cell index: %u\n\n", hash_grid_cell_index);

			printf("  world_position: %f %f %f\n", world_position.x, world_position.y, world_position.z);
			printf("  shading_normal: %f %f %f\n", surface_normal.x, surface_normal.y, surface_normal.z);
			printf("  roughness: %f\n", material.roughness);
		}*/
		// TODO we can have a if (current_hash_key != undefined_key) here to skip some atomic operations
		
		// Trying to insert the new key atomically 
		unsigned int existing_checksum = hippt::atomic_compare_exchange(&hash_cell_data_to_update.checksums[hash_grid_cell_index], HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX, checksum);
		if (existing_checksum != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		{
			// We tried inserting in our cell but there is something else there already
			
			if (existing_checksum != checksum)
			{
				// And it's not our hash so this is a collision

				unsigned int new_hash_cell_index = hash_grid_cell_index;
				if (!HashGrid::resolve_collision<ReGIR_HashGridLinearProbingSteps, true>(hash_cell_data_to_update.checksums, hash_grid_to_update.m_total_number_of_cells, new_hash_cell_index, checksum, existing_checksum))
				{
					// Could not resolve the collision

					return;
				}
				else 
				{
					// We resolved the collision by finding an empty cell
					hash_grid_cell_index = new_hash_cell_index;

					insert_hash_cell_point_normal(hash_cell_data_to_update, hash_grid_cell_index, world_position, surface_normal, primitive_index, material);
				}
			}
			else
			{
				// We're trying to insert in a cell that has the same hash as us so we're going to update
				// that cell with our data
				update_hash_cell_representative_data(hash_cell_data_to_update, hash_grid_cell_index, world_position, surface_normal, primitive_index, material);
			}
		}
		else
		{
			// We just succeeded the insertion of our key in an empty cell
			
			insert_hash_cell_point_normal(hash_cell_data_to_update, hash_grid_cell_index, world_position, surface_normal, primitive_index, material);
		}

	}

	HIPRT_DEVICE void insert_hash_cell_data(ReGIRShadingSettings& shading_settings, float3 world_position, float3 surface_normal, const HIPRTCamera& current_camera, bool primary_hit, int primitive_index, const DeviceUnpackedEffectiveMaterial& material)
	{
		ReGIRSettings::insert_hash_cell_data_static(hash_grid, get_initial_reservoirs_grid(primary_hit), get_hash_cell_data_soa(primary_hit), world_position, surface_normal, current_camera, primitive_index, material);
	}

	bool DEBUG_INCLUDE_CANONICAL = true;
	bool DEBUG_FORCE_REGIR8CANONICAL = false;
	bool DEBUG_CORRELATE_rEGIR = true;
	bool DEBUG_DO_RIS_INTEGRAL_NORMALIZATION = true;
	bool DEBUG_DO_BSDF_PDF_AS_SHADING_TG = false;
	bool DEBUG_DO_BALANCE_HEURISTIC = false;
	bool DEBUG_DO_PAIRWISE_MIS = true;
	int DEBUG_CORRELATE_rEGIR_SIZE = 32;

	// How many frames to skip before running the grid fill and spatial reuse passes again
	// 
	// A value of 1 for example means that the grid fill and spatial reuse will be ran at frame 0
	// but not at frame 1. And ran at frame 2 but not at frame 3. ...
	//
	// This amortizes the overhead of ReGIR grid fill / spatial reuse by using the fact that each cell
	// contains many reservoirs so the same cell can be used multiple times before all reservoirs have been used
	// and new samples are necessary
	int frame_skip_primary_hit_grid = 0;
	int frame_skip_secondary_hit_grid = 2;

	ReGIRHashGrid hash_grid;

	// Grid that contains the output reservoirs of the grid fill pass for the primary hits grid cells
	ReGIRHashGridSoADevice initial_reservoirs_primary_hits_grid;
	ReGIRHashGridSoADevice initial_reservoirs_secondary_hits_grid;
	// Grid that contains the output reservoirs of the spatial reuse pass for the primary hits grid cells
	ReGIRHashGridSoADevice spatial_output_primary_hits_grid;
	ReGIRHashGridSoADevice spatial_output_secondary_hits_grid;

	// Contains data associated with the primary hits grid cells
	ReGIRHashCellDataSoADevice hash_cell_data_primary_hits;
	ReGIRHashCellDataSoADevice hash_cell_data_secondary_hits;

	ReGIRGridFillSettings grid_fill_primary_hits = ReGIRGridFillSettings(true);
	ReGIRGridFillSettings grid_fill_secondary_hits = ReGIRGridFillSettings(false);

	ReGIRSpatialReuseSettings spatial_reuse;
	ReGIRShadingSettings shading;
	ReGIRCorrelationReductionSettings supersampling;

	AtomicType<float>* non_canonical_pre_integration_factors_primary_hits = nullptr;
	AtomicType<float>* canonical_pre_integration_factors_primary_hits = nullptr;

	AtomicType<float>* non_canonical_pre_integration_factors_secondary_hits = nullptr;
	AtomicType<float>* canonical_pre_integration_factors_secondary_hits = nullptr;

	// Multiplicative factor to multiply the output of some debug views
	float debug_view_scale_factor = 0.05f;
};

#endif
