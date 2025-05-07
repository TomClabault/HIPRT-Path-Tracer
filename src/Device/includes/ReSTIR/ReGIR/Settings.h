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
	bool do_spatial_reuse = false;
 	// If true, the same random seed will be used by all grid cells during the spatial reuse for a given frame
 	// This has the effect of coalescing neighbors memory accesses which improves performance
	bool do_coalesced_spatial_reuse = true;

	int spatial_neighbor_count = 8;
	int reuse_per_neighbor_count = 8;
	// When picking a random cell in the neighborhood for reuse, if that
	// cell is out of the grid or if that cell is not alive etc..., we're
	// going to retry another cell this many times
	//
	// This improves the chances that we're actually going to have a good
	// neighbor to reuse from --> more reuse --> less variance
	int retries_per_neighbor = 8;
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

	/*HIPRT_DEVICE int3 get_xyz_cell_index_from_linear(int hash_grid_cell_index) const
	{
		unsigned int packed_rep_point = grid_fill_grid.hash_cell_data.representative_points[hash_grid_cell_index];
		float3 rep_point = ReGIR_unpack_representative_point(get_cell_size(), , packed_rep_point, hash_grid_cell_index);

		int index_x = hash_grid_cell_index % grid_resolution.x;
		int index_y = (hash_grid_cell_index % (grid_resolution.x * grid_resolution.y)) / grid_resolution.x;
		int index_z = hash_grid_cell_index / (grid_resolution.x * grid_resolution.y);

		return make_int3(index_x, index_y, index_z);
	}*/

	/*HIPRT_DEVICE float3 get_cell_origin_from_hash_grid_cell_index(int hash_grid_cell_index) const
	{
		float3 cell_size = get_cell_size();

		int3 cell_index_xyz = get_xyz_cell_index_from_linear(hash_grid_cell_index);
		float3 cell_index_xyz_float = make_float3(static_cast<float>(cell_index_xyz.x), static_cast<float>(cell_index_xyz.y), static_cast<float>(cell_index_xyz.z));

		return grid_origin + cell_size * cell_index_xyz_float;
	}*/

	/*HIPRT_DEVICE float3 get_cell_center_from_hash_grid_cell_index(unsigned int hash_grid_cell_index) const
	{
		float3 cell_size = get_cell_size();

		return get_cell_origin_from_hash_grid_cell_index(hash_grid_cell_index) + cell_size * 0.5f;
	}*/

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
			return spatial_grid.read_full_reservoir_opt(world_position, camera_position, reservoir_index_in_cell);
		else if (temporal_reuse.do_temporal_reuse)
			// If only doing temporal reuse, reading from the output of the spatial reuse pass
			return get_temporal_reservoir_opt(world_position, camera_position, reservoir_index_in_cell);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return grid_fill_grid.read_full_reservoir_opt(world_position, camera_position, reservoir_index_in_cell);
	}

	/**
	 * If 'out_invalid_sample' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_non_canonical_reservoir_for_shading_from_world_pos(float3 world_position, float3 camera_position, bool& out_invalid_sample, Xorshift32Generator& rng, bool jitter = false, float3 DEBUG_SHADING_NORMAL = make_float3(0, 0, 0)) const
	{	
		if (jitter)
			world_position = grid_fill_grid.jitter_world_position(world_position, rng);

		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(world_position, camera_position);

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

		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(world_position, camera_position);

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

		unsigned int hash_grid_cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(shading_point, camera_position);

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
			return grid_fill_grid.read_full_reservoir_opt(world_position, camera_position, reservoir_index_in_cell, grid_index);
		else
			return grid_fill_grid.read_full_reservoir_opt(world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE ReGIRReservoir get_grid_fill_output_reservoir_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell) const
	{
		// The output of the grid fill pass is in the current frame grid so we can call the temporal method with
		// index -1
		return get_temporal_reservoir_opt(world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt(const ReGIRReservoir& reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell)
	{
		spatial_grid.store_reservoir_and_sample_opt(reservoir, world_position, camera_position, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void store_reservoir_opt(ReGIRReservoir reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index != -1)
			grid_fill_grid.store_reservoir_and_sample_opt(reservoir, world_position, camera_position, reservoir_index_in_cell, grid_index);
		else
			grid_fill_grid.store_reservoir_and_sample_opt(reservoir, world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE ColorRGB32F get_random_cell_color(float3 position, float3 camera_position) const
	{
		unsigned int cell_index = grid_fill_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(position, camera_position);

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

	bool DEBUG_INCLUDE_CANONICAL = false;

	ReGIRHashGridSoADevice grid_fill_grid;
	// Grid that contains the output of the spatial reuse pass
	ReGIRHashGridSoADevice spatial_grid;

	ReGIRGridFillSettings grid_fill;
	ReGIRTemporalReuseSettings temporal_reuse;
	ReGIRSpatialReuseSettings spatial_reuse;
	ReGIRShadingSettings shading;

	// If true, the center of the cells will not be used anymore to compute the ReGIR target functions
	// but rather, points on the surface of the scene within the cells will be used
	//
	// This feature is non-deterministic
	bool use_representative_points = true;
	
	// Multiplicative factor to multiply the output of some debug views
	float debug_view_scale_factor = 0.05f;
};

#endif
