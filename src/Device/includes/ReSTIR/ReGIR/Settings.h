/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_SETTINGS_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/GridBufferSoA.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridSoA.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/Xorshift.h"

struct ReGIRGridFillSettings
{
	// How many light samples are resampled into each reservoir of the grid cell
	int sample_count_per_cell_reservoir = 32;

	HIPRT_DEVICE int get_non_canonical_reservoir_count_per_cell() const { return reservoirs_count_per_grid_cell_non_canonical; }
	HIPRT_DEVICE int get_canonical_reservoir_count_per_cell() const { return reservoirs_count_per_grid_cell_canonical; }
	HIPRT_DEVICE int get_total_reservoir_count_per_cell() const { return reservoirs_count_per_grid_cell_canonical + reservoirs_count_per_grid_cell_non_canonical; }

	HIPRT_DEVICE int* get_non_canonical_reservoir_count_per_cell_ptr() { return &reservoirs_count_per_grid_cell_non_canonical; }
	HIPRT_DEVICE int* get_canonical_reservoir_count_per_cell_ptr() { return &reservoirs_count_per_grid_cell_canonical; }

	HIPRT_DEVICE bool reservoir_index_in_cell_is_canonical(int reservoir_index_in_cell) { return reservoir_index_in_cell >= get_non_canonical_reservoir_count_per_cell(); }

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
	int reservoirs_count_per_grid_cell_non_canonical = 16;

	// Number of canonical reservoirs per cell
	// 
	// In the grid buffers, these reservoirs are stored last, i.e., for a grid cell with 3 non-canonical reservoirs
	// and 1 canonical reservoir:
	// 
	// [non-canon, non-canon, non-canon, canonical]
	int reservoirs_count_per_grid_cell_canonical = 8;
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
	bool do_spatial_reuse = true;
 	// If true, the same random seed will be used by all grid cells during the spatial reuse for a given frame
 	// This has the effect of coalescing neighbors memory accesses which improves performance
	bool do_coalesced_spatial_reuse = true;

	int spatial_neighbor_count = 8;
	int reuse_per_neighbor_count = 1;
	// When picking a random cell in the neighborhood for reuse, if that
	// cell is out of the grid or if that cell is not alive etc..., we're
	// going to retry another cell this many times
	//
	// This improves the chances that we're actually going to have a good
	// neighbor to reuse from --> more reuse --> less variance
	int retries_per_neighbor = 16;
	int spatial_reuse_radius = 1;

	bool DEBUG_oONLY_ONE_CENTER_CELL = true;
};

struct ReGIRSettings
{
	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE float3 get_cell_size(float3 world_position = make_float3(0, 0, 0), float3 camera_position = make_float3(0, 0, 0)) const
	{
		return grid_fill_grid.get_cell_size(world_position, camera_position);
	}

	/*HIPRT_DEVICE float get_cell_diagonal_length() const
	{
		return grid_fill_grid.get_cell_diagonal_length();
	}*/

	/*HIPRT_DEVICE float3 get_cell_center_from_hash_grid_cell_index(unsigned int hash_grid_cell_index) const
	{
		return grid_fill_grid.hash_grid.get_cell_center_from_hash_grid_cell_index(hash_grid_cell_index);
	}*/

	/*HIPRT_DEVICE float3 get_cell_origin_from_hash_grid_cell_index(int hash_grid_cell_index) const
	{
		return grid_fill_grid.hash_grid.get_cell_origin_from_hash_grid_cell_index(hash_grid_cell_index);
	}*/

	/*HIPRT_DEVICE int get_hash_grid_cell_index_from_xyz(int3 xyz_cell_index) const
	{
		return grid_fill_grid.hash_grid.get_hash_grid_cell_index_from_xyz(xyz_cell_index);
	}*/

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_no_collision_resolve(float3 world_position, float3 camera_position, Xorshift32Generator* rng = nullptr, bool jitter = false) const
	{
		if (jitter)
			world_position = grid_fill_grid.jitter_world_position(world_position, *rng);

		return grid_fill_grid.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(world_position, camera_position);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(float3 world_position, float3 camera_position) const
	{
		return grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(hash_cell_data, world_position, camera_position);
	}

	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE unsigned int get_reservoir_index_in_grid_from_world_pos(float3 world_position, float3 camera_position, int reservoir_index_in_cell) const
	{
		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(world_position, camera_position);

		return hash_grid_cell_index * grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;
	}

	/**
	 * Here, 'non_canonical_reservoir_index_in_cell' should be in [0, grid_fill.get_non_canonical_reservoir_count_per_cell() - 1]
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_non_canonical_reservoir_from_cell_reservoir_index(float3 world_position, float3 camera_position, int non_canonical_reservoir_index_in_cell) const
	{
		return get_reservoir_for_shading_from_cell_indices(world_position, camera_position, non_canonical_reservoir_index_in_cell);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_non_canonical_reservoir(float3 world_position, float3 camera_position, Xorshift32Generator& rng) const
	{
		int random_non_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
			random_non_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

		return get_cell_non_canonical_reservoir_from_cell_reservoir_index(world_position, camera_position, random_non_canonical_reservoir_index_in_cell);
	}

	/**
	 * Here, 'canonical_reservoir_index_in_cell' should be in [0, grid_fill.get_canonical_reservoir_count_per_cell() - 1]
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_canonical_reservoir_from_cell_reservoir_index(float3 world_position, float3 camera_position, int canonical_reservoir_index_in_cell) const
	{
		return get_reservoir_for_shading_from_cell_indices(world_position, camera_position, canonical_reservoir_index_in_cell);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_canonical_reservoir(float3 world_position, float3 camera_position, Xorshift32Generator& rng) const
	{
		int random_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_canonical_reservoir_count_per_cell() > 1)
			random_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_canonical_reservoir_count_per_cell());

		return get_cell_canonical_reservoir_from_cell_reservoir_index(world_position, camera_position, grid_fill.get_non_canonical_reservoir_count_per_cell() + random_canonical_reservoir_index_in_cell);
	}

	/**
	 * If 'out_invalid_sample' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_reservoir_for_shading_from_cell_indices(float3 world_position, float3 camera_position, int reservoir_index_in_cell) const
	{
		if (spatial_reuse.do_spatial_reuse)
			// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
			return spatial_grid.read_full_reservoir_opt(hash_cell_data, world_position, camera_position, reservoir_index_in_cell);
		else if (temporal_reuse.do_temporal_reuse)
			// If only doing temporal reuse, reading from the output of the spatial reuse pass
			return get_temporal_reservoir_opt(world_position, camera_position, reservoir_index_in_cell);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return grid_fill_grid.read_full_reservoir_opt(hash_cell_data, world_position, camera_position, reservoir_index_in_cell);
	}

	/**
	 * If 'out_invalid_sample' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_non_canonical_reservoir_for_shading_from_world_pos(float3 world_position, float3 camera_position, bool& out_invalid_sample, Xorshift32Generator& rng, bool jitter = false, float3 DEBUG_SHADING_NORMAL = make_float3(0, 0, 0)) const
	{	
		if (jitter)
			world_position = grid_fill_grid.jitter_world_position(world_position, rng);

		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(hash_cell_data, world_position, camera_position);

		 if (hash_grid_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY || shading.grid_cells_alive[hash_grid_cell_index] == 0)
		 {
		 	// The grid cell is inside the grid but not alive
		 	// We're indicating that this cell should not be used by setting the 'out_invalid_sample' to true
		 	out_invalid_sample = true;

		 	return ReGIRReservoir();
		 }

		out_invalid_sample = false;

		return get_random_cell_non_canonical_reservoir(world_position, camera_position, rng);
	}

	HIPRT_DEVICE ReGIRReservoir get_canonical_reservoir_for_shading_from_world_pos(float3 world_position, float3 camera_position, bool& out_point_outside_of_grid, Xorshift32Generator& rng, bool jitter = false) const
	{
		if (jitter)
			world_position = grid_fill_grid.jitter_world_position(world_position, rng);

		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(hash_cell_data, world_position, camera_position);

		if (hash_grid_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY || shading.grid_cells_alive[hash_grid_cell_index] == 0)
		{
			// The grid cell is inside the grid but not alive
			// We're indicating that this cell should not be used by setting the 'out_invalid_sample' to true
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		out_point_outside_of_grid = false;

		return get_random_cell_canonical_reservoir(world_position, camera_position, rng);
	}

	HIPRT_DEVICE unsigned int get_neighbor_replay_hash_grid_cell_index_for_shading(float3 shading_point, float3 camera_position, Xorshift32Generator& rng, bool jitter = false) const
	{
		if (jitter)
			shading_point = grid_fill_grid.jitter_world_position(shading_point, rng);

		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(hash_cell_data, shading_point, camera_position);

		if (hash_grid_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY || shading.grid_cells_alive[hash_grid_cell_index] == 0)
			// The grid cell is inside the grid but not alive
			return ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY;

		// Advancing the RNG to mimic 'get_non_canonical_reservoir_for_shading_from_world_pos'
		if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
			rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

		return hash_grid_cell_index;
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
	HIPRT_DEVICE ReGIRReservoir get_temporal_reservoir_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1) const
	{
		if (grid_index != -1)
			return grid_fill_grid.read_full_reservoir_opt(hash_cell_data, world_position, camera_position, reservoir_index_in_cell, grid_index);
		else
			return grid_fill_grid.read_full_reservoir_opt(hash_cell_data, world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE ReGIRReservoir get_grid_fill_output_reservoir_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell) const
	{
		// The output of the grid fill pass is in the current frame grid so we can call the temporal method with
		// index -1
		return get_temporal_reservoir_opt(world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt(const ReGIRReservoir& reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell)
	{
		spatial_grid.store_reservoir_and_sample_opt(reservoir, hash_cell_data, world_position, camera_position, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void store_reservoir_opt(ReGIRReservoir reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index != -1)
			grid_fill_grid.store_reservoir_and_sample_opt(reservoir, hash_cell_data, world_position, camera_position, reservoir_index_in_cell, grid_index);
		else
			grid_fill_grid.store_reservoir_and_sample_opt(reservoir, hash_cell_data, world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE ColorRGB32F get_random_cell_color(float3 position, float3 camera_position) const
	{
		if( hippt::thread_idx_global() < 5)
			printf("Grid res: %d\n", grid_fill_grid.hash_grid.grid_resolution.x);
		unsigned int cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(hash_cell_data, position, camera_position);

		return ColorRGB32F::random_color(cell_index);
	}

	HIPRT_DEVICE unsigned int get_total_number_of_cells_per_grid() const
	{
		return grid_fill_grid.m_total_number_of_cells;
	}

	HIPRT_DEVICE unsigned int get_number_of_reservoirs_per_grid() const
	{
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return get_total_number_of_cells_per_grid() * grid_fill.get_total_reservoir_count_per_cell();
	}

	HIPRT_DEVICE unsigned int get_number_of_reservoirs_per_cell() const
	{
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return grid_fill.get_total_reservoir_count_per_cell();
	}

	HIPRT_DEVICE unsigned int get_total_number_of_reservoirs_ReGIR() const
	{
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		int temporal_grid_count = temporal_reuse.do_temporal_reuse ? temporal_reuse.temporal_history_length : 1;

		return get_number_of_reservoirs_per_grid() * temporal_grid_count;
	}

	/**
	 * Resets all the reservoirs of all the grids at the given 'reservoir_index'
	 */
	HIPRT_DEVICE void reset_reservoirs(unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell)
	{
		int temporal_grid_count = temporal_reuse.do_temporal_reuse ? temporal_reuse.temporal_history_length : 1;

		for (int grid_index = 0; grid_index < temporal_grid_count; grid_index++)
			grid_fill_grid.reset_reservoir(hash_grid_cell_index, reservoir_index_in_cell, grid_index);

		// Also clearing the spatial reuse output buffers (grid) if spatial reuse is enabled
		if (spatial_reuse.do_spatial_reuse)
			spatial_grid.reset_reservoir(hash_grid_cell_index, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void update_hash_cell_data(ReGIRShadingSettings& shading_settings, float3 world_position, float3 camera_position, float3 shading_normal, int primitive_index)
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = grid_fill_grid.hash(world_position, camera_position, hash_key);

		unsigned int current_hash_key = hash_cell_data.hash_keys[hash_grid_cell_index];
		if (current_hash_key != ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
		{
			// We already have something in that cell
			if (current_hash_key != hash_key)
			{
				// And it's not our hash so this is a collision

				unsigned int new_hash_cell_index = hash_grid_cell_index;
				if (!grid_fill_grid.resolve_collision(hash_cell_data, new_hash_cell_index, hash_key))
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

	bool DEBUG_INCLUDE_CANONICAL = false;

	// Grid that contains the output reservoirs of the grid fill pass
	ReGIRHashGridSoADevice grid_fill_grid;
	// Grid that contains the output reservoirs of the spatial reuse pass
	ReGIRHashGridSoADevice spatial_grid;
	// This SoA is allocated to hold 'number_cells' only.

	// It contains data associated with the grid cells themselves
	ReGIRHashCellDataSoADevice hash_cell_data;

	ReGIRGridFillSettings grid_fill;
	ReGIRTemporalReuseSettings temporal_reuse;
	ReGIRSpatialReuseSettings spatial_reuse;
	ReGIRShadingSettings shading;

	// Multiplicative factor to multiply the output of some debug views
	float debug_view_scale_factor = 0.05f;
};

#endif
