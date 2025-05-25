/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_SETTINGS_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/HashGrid.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridSoADevice.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/HIPRTCamera.h"
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

struct ReGIRSpatialReuseSettings
{
	bool do_spatial_reuse = true;
 	// If true, the same random seed will be used by all grid cells during the spatial reuse for a given frame
 	// This has the effect of coalescing neighbors memory accesses which improves performance
	bool do_coalesced_spatial_reuse = true;

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

	bool DEBUG_oONLY_ONE_CENTER_CELL = true;
};

struct ReGIRSettings
{
	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE float3 get_cell_size(float3 world_position, const HIPRTCamera& current_camera) const
	{
		float cell_size = hash_grid.compute_adaptive_cell_size(world_position, current_camera);

		return make_float3(cell_size, cell_size, cell_size);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_no_collision_resolve(float3 world_position, const HIPRTCamera& current_camera, Xorshift32Generator* rng = nullptr, bool jitter = false) const
	{
		if (jitter)
			world_position = hash_grid.jitter_world_position(world_position, current_camera, *rng);

		return hash_grid.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(initial_reservoirs_grid, world_position, current_camera);
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(float3 world_position, const HIPRTCamera& current_camera) const
	{
		return hash_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(initial_reservoirs_grid, hash_cell_data, world_position, current_camera);
	}

	
	HIPRT_DEVICE unsigned int get_reservoir_index_in_grid_from_world_pos(float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell) const
	{
		unsigned int hash_grid_cell_index = hash_grid.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(initial_reservoirs_grid, world_position, current_camera);
		
		return hash_grid_cell_index * grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;
	}

	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE ReGIRReservoir get_random_cell_non_canonical_reservoir(float3 world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng, bool* out_invalid_sample = nullptr) const
	{
		int random_non_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
			random_non_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

		return get_reservoir_for_shading_from_cell_indices(world_position, current_camera, random_non_canonical_reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_canonical_reservoir(float3 world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng, bool* out_invalid_sample = nullptr) const
	{
		int random_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_canonical_reservoir_count_per_cell() > 1)
			random_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_canonical_reservoir_count_per_cell());

		return get_reservoir_for_shading_from_cell_indices(world_position, current_camera, grid_fill.get_non_canonical_reservoir_count_per_cell() + random_canonical_reservoir_index_in_cell, out_invalid_sample);
	}

	/**
	 * If 'out_invalid_sample' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_reservoir_for_shading_from_cell_indices(float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		if (spatial_reuse.do_spatial_reuse)
			// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
			return hash_grid.read_full_reservoir(spatial_output_grid, hash_cell_data, world_position, current_camera, reservoir_index_in_cell, out_invalid_sample);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return hash_grid.read_full_reservoir(initial_reservoirs_grid, hash_cell_data, world_position, current_camera, reservoir_index_in_cell, out_invalid_sample);
	}

	/**
	 * If 'out_invalid_sample' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	template <bool getCanonicalReservoir>
	HIPRT_DEVICE ReGIRReservoir get_reservoir_for_shading_from_world_pos(float3 world_position, const HIPRTCamera& current_camera, bool& out_invalid_sample, Xorshift32Generator& rng, bool jitter = false) const
	{	
		if constexpr (getCanonicalReservoir)
		{
			// This is just constructing a function pointer to pass to the function below for
			// code factorization
			auto get_reservoir_lambda = [this](float3 world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng, bool* out_invalid_sample)
			{
				return this->get_random_cell_canonical_reservoir(world_position, current_camera, rng, out_invalid_sample);
			};
	
			ReGIRReservoir reservoir = internal_get_reservoirs_with_retries<ReGIR_ShadingJitterTries>(world_position, current_camera, out_invalid_sample, rng, jitter, get_reservoir_lambda);
			if (out_invalid_sample)
				// We couldn't find a canonical neighbor with jittering, directly returning the center cell instead
				//
				// Only 1 retry + no jittering is guaranteed to just return a reservoir from the center cell
				return internal_get_reservoirs_with_retries<1>(world_position, current_camera, out_invalid_sample, rng, false, get_reservoir_lambda);
			else
				// We could find a canonical reservoir in the neighborhood, all good
				return reservoir;
		}
		else
		{
			// This is just constructing a function pointer to pass to the function below for
			// code factorization
			auto get_reservoir_lambda = [this](float3 world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng, bool* out_invalid_sample)
			{
				return this->get_random_cell_non_canonical_reservoir(world_position, current_camera, rng, out_invalid_sample);
			};

			return internal_get_reservoirs_with_retries<ReGIR_ShadingJitterTries>(world_position, current_camera, out_invalid_sample, rng, jitter, get_reservoir_lambda);
		}
	}

	HIPRT_DEVICE unsigned int get_neighbor_replay_hash_grid_cell_index_for_shading(float3 shading_point, const HIPRTCamera& current_camera, bool replay_canonical, Xorshift32Generator& rng, bool jitter = false) const
	{
		unsigned int neighbor_cell_index;
		if (replay_canonical)
            neighbor_cell_index = find_valid_jittered_neighbor_cell_index<true>(shading_point, current_camera, rng);
        else
            neighbor_cell_index = find_valid_jittered_neighbor_cell_index<false>(shading_point, current_camera, rng);

		if (neighbor_cell_index != ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
		{
			// Advancing the RNG simulating the random reservoir pick within the grid cell
			if (replay_canonical)
			{
				if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
					rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());
			}
			else
			{
				if (grid_fill.get_canonical_reservoir_count_per_cell() > 1)
					rng.random_index(grid_fill.get_canonical_reservoir_count_per_cell());
			}
		}

		return neighbor_cell_index;
	}

	template <bool fallbackOnCenterCell>
	HIPRT_DEVICE unsigned int find_valid_jittered_neighbor_cell_index(float3 world_position, const HIPRTCamera& current_camera, Xorshift32Generator& rng) const
	{
		unsigned int retry = 0;
		unsigned int neighbor_grid_cell_index;
		
		do
		{
			float3 jittered = hash_grid.jitter_world_position(world_position, current_camera, rng);

			neighbor_grid_cell_index = hash_grid.get_hash_grid_cell_index(initial_reservoirs_grid, hash_cell_data, jittered, current_camera);
			if (neighbor_grid_cell_index != ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
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
					UCW = spatial_output_grid.reservoirs.UCW[neighbor_grid_cell_index * get_number_of_reservoirs_per_cell()];
				else
					UCW = initial_reservoirs_grid.reservoirs.UCW[neighbor_grid_cell_index * get_number_of_reservoirs_per_cell()];

				if (UCW == ReGIRReservoir::UNDEFINED_UCW)
					neighbor_grid_cell_index = ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY;
			}

			retry++;
		} while (neighbor_grid_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY && retry < ReGIR_ShadingJitterTries);

		if (fallbackOnCenterCell && neighbor_grid_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY && retry == ReGIR_ShadingJitterTries)
			// We couldn't find a valid neighbor and the fallback on center cell is enabled: we're going to return the index of the center cell
			neighbor_grid_cell_index = hash_grid.get_hash_grid_cell_index(initial_reservoirs_grid, hash_cell_data, world_position, current_camera);

		return neighbor_grid_cell_index;
	}

	template <bool getCanonicalReservoir>
	HIPRT_DEVICE ReGIRReservoir get_random_reservoir_in_grid_cell_for_shading(unsigned int grid_cell_index, Xorshift32Generator& rng) const
	{
		unsigned int reservoir_index_in_cell;

		if constexpr (getCanonicalReservoir)
		{
			int random_canonical_reservoir_index_in_cell = 0;
			if (grid_fill.get_canonical_reservoir_count_per_cell() > 1)
				random_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_canonical_reservoir_count_per_cell());

			reservoir_index_in_cell = random_canonical_reservoir_index_in_cell;
		}
		else
		{
			int random_non_canonical_reservoir_index_in_cell = 0;
			if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
				random_non_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

			reservoir_index_in_cell = random_non_canonical_reservoir_index_in_cell;
		}

		unsigned int canonical_offset = getCanonicalReservoir ? grid_fill.get_non_canonical_reservoir_count_per_cell() : 0;
		unsigned int reservoir_index_in_grid = grid_cell_index * get_number_of_reservoirs_per_cell() + canonical_offset + reservoir_index_in_cell;

		if (spatial_reuse.do_spatial_reuse)
			// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
			return hash_grid.read_full_reservoir(spatial_output_grid, hash_cell_data, reservoir_index_in_grid);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return hash_grid.read_full_reservoir(initial_reservoirs_grid, hash_cell_data, reservoir_index_in_grid);
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
	HIPRT_DEVICE ReGIRReservoir get_temporal_reservoir_opt(float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		return hash_grid.read_full_reservoir(initial_reservoirs_grid, hash_cell_data, world_position, current_camera, reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE ReGIRReservoir get_grid_fill_output_reservoir_opt(float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell, bool* out_invalid_sample = nullptr) const
	{
		// The output of the grid fill pass is in the current frame grid so we can call the temporal method with
		// index -1
		return get_temporal_reservoir_opt(world_position, current_camera, reservoir_index_in_cell, out_invalid_sample);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt(const ReGIRReservoir& reservoir, float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell)
	{
		hash_grid.store_reservoir_and_sample_opt(reservoir, spatial_output_grid, hash_cell_data, world_position, current_camera, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void store_reservoir_opt(ReGIRReservoir reservoir, float3 world_position, const HIPRTCamera& current_camera, int reservoir_index_in_cell)
	{
		hash_grid.store_reservoir_and_sample_opt(reservoir, initial_reservoirs_grid, hash_cell_data, world_position, current_camera, reservoir_index_in_cell);
	}

	HIPRT_DEVICE ColorRGB32F get_random_cell_color(float3 position, const HIPRTCamera& current_camera) const
	{
		unsigned int cell_index = hash_grid.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(initial_reservoirs_grid, hash_cell_data, position, current_camera);
		if (cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
			return ColorRGB32F(0.0f);

		return ColorRGB32F::random_color(cell_index);
	}

	HIPRT_DEVICE unsigned int get_total_number_of_cells_per_grid() const
	{
		return initial_reservoirs_grid.m_total_number_of_cells;
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
		return get_number_of_reservoirs_per_grid();
	}

	/**
	 * Resets all the reservoirs of all the grids at the given 'reservoir_index'
	 */
	HIPRT_DEVICE void reset_reservoirs(unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell)
	{
		hash_grid.reset_reservoir(initial_reservoirs_grid, hash_grid_cell_index, reservoir_index_in_cell);

		// Also clearing the spatial reuse output buffers (grid) if spatial reuse is enabled
		if (spatial_reuse.do_spatial_reuse)
			hash_grid.reset_reservoir(spatial_output_grid, hash_grid_cell_index, reservoir_index_in_cell);
	}

	HIPRT_DEVICE static void insert_hash_cell_point_normal(ReGIRHashCellDataSoADevice& hash_cell_data_to_update,
		unsigned int hash_grid_cell_index, float3 world_position, float3 shading_normal, int primitive_index)
	{
		// TODO is this atomic needed since we can only be here if the cell was unoccupied?
		if (hippt::atomic_compare_exchange(&hash_cell_data_to_update.hit_primitive[hash_grid_cell_index], ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE, primitive_index) == ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE)
		{
			hash_cell_data_to_update.world_points[hash_grid_cell_index] = world_position;
			hash_cell_data_to_update.world_normals[hash_grid_cell_index].pack(shading_normal);

			hash_cell_data_to_update.sum_points[hash_grid_cell_index] = world_position;
			hash_cell_data_to_update.num_points[hash_grid_cell_index] = 1;
		}

		// Because we just inserted into that grid cell, it is now alive
		// Only go through all that atomic stuff if the cell isn't alive
		if (hash_cell_data_to_update.grid_cells_alive[hash_grid_cell_index] == 0)
		{
			// TODO is this atomic needed since we can only be here if the cell was unoccoupied?

			if (hippt::atomic_compare_exchange(&hash_cell_data_to_update.grid_cells_alive[hash_grid_cell_index], 0u, 1u) == 0u)
			{
				unsigned int cell_alive_index = hippt::atomic_fetch_add(hash_cell_data_to_update.grid_cells_alive_count, 1u);

				hash_cell_data_to_update.grid_cells_alive_list[cell_alive_index] = hash_grid_cell_index;
			}
		}
	}

	HIPRT_DEVICE static void update_hash_cell_point_normal(ReGIRHashCellDataSoADevice& hash_cell_data_to_update,
		unsigned int hash_grid_cell_index, float3 world_position, float3 shading_normal, int primitive_index)
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
			}

			// Writing back the new sum of points
			hash_cell_data_to_update.sum_points[hash_grid_cell_index] = current_sum_points;
			// Incrementing the number of points
			hash_cell_data_to_update.num_points[hash_grid_cell_index] = current_num_points + 1;
		}
	}

	HIPRT_DEVICE static void insert_hash_cell_data_static(
		const ReGIRHashGrid& hash_grid, ReGIRHashGridSoADevice& hash_grid_to_update, ReGIRHashCellDataSoADevice& hash_cell_data_to_update,
		float3 world_position, const HIPRTCamera& current_camera, float3 shading_normal, int primitive_index)
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash_grid.hash(hash_grid_to_update.m_total_number_of_cells, world_position, current_camera, hash_key);
		
		// TODO we can have a if (current_hash_key != undefined_key) here to skip some atomic operations
		
		// Trying to insert the new key atomically 
		unsigned int before = hippt::atomic_compare_exchange(&hash_cell_data_to_update.hash_keys[hash_grid_cell_index], ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY, hash_key);
		if (before != ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
		{
			// We tried inserting in our cell but there is something else there already
			
			unsigned int existing_hash_key = hash_cell_data_to_update.hash_keys[hash_grid_cell_index];
			if (existing_hash_key != hash_key)
			{
				// And it's not our hash so this is a collision

				unsigned int new_hash_cell_index = hash_grid_cell_index;
				if (!hash_grid.resolve_collision<true>(hash_cell_data_to_update, hash_grid_to_update.m_total_number_of_cells, new_hash_cell_index, hash_key))
				{
					// Could not resolve the collision

					return;
				}
				else 
				{
					if (new_hash_cell_index == hash_grid_cell_index)
					{
						// We resolved the collision by finding our own cell hash with probing
						// 
						// This means that we already have something in our grid cell
						// We're going to update the average 

						// TODO never getting here?
						update_hash_cell_point_normal(hash_cell_data_to_update,
							hash_grid_cell_index, world_position, shading_normal, primitive_index);

						return;
					}
					else
					{
						// We resolved the collision by finding an empty cell
						hash_grid_cell_index = new_hash_cell_index;

						insert_hash_cell_point_normal(hash_cell_data_to_update,
							hash_grid_cell_index, world_position, shading_normal, primitive_index);
					}
				}
			}
			else
			{
				// We're trying to insert in a cell that has the same hash as us so we're going to update
				// that cell with our data
				update_hash_cell_point_normal(hash_cell_data_to_update,
					hash_grid_cell_index, world_position, shading_normal, primitive_index);
			}
		}
		else
		{
			// We just succeeded the insertion of our key in an empty cell
			
			insert_hash_cell_point_normal(hash_cell_data_to_update,
				hash_grid_cell_index, world_position, shading_normal, primitive_index);
		}

	}

	HIPRT_DEVICE void insert_hash_cell_data(ReGIRShadingSettings& shading_settings, float3 world_position, const HIPRTCamera& current_camera, float3 shading_normal, int primitive_index)
	{
		ReGIRSettings::insert_hash_cell_data_static(hash_grid, initial_reservoirs_grid, hash_cell_data, world_position, current_camera, shading_normal, primitive_index);
	}

	bool DEBUG_INCLUDE_CANONICAL = true;
	bool DEBUG_CORRELATE_rEGIR = true;
	int DEBUG_CORRELATE_rEGIR_SIZE = 32;

	ReGIRHashGrid hash_grid;

	// Grid that contains the output reservoirs of the grid fill pass
	ReGIRHashGridSoADevice initial_reservoirs_grid;
	// Grid that contains the output reservoirs of the spatial reuse pass
	ReGIRHashGridSoADevice spatial_output_grid;
	// This SoA is allocated to hold 'number_cells' only.

	// It contains data associated with the grid cells themselves
	ReGIRHashCellDataSoADevice hash_cell_data;

	ReGIRGridFillSettings grid_fill;
	ReGIRSpatialReuseSettings spatial_reuse;
	ReGIRShadingSettings shading;

	// Multiplicative factor to multiply the output of some debug views
	float debug_view_scale_factor = 0.05f;

private:
	template <int retryCount, typename FunctionType>
	HIPRT_DEVICE ReGIRReservoir internal_get_reservoirs_with_retries(float3 world_position, const HIPRTCamera& current_camera, bool& out_invalid_sample, Xorshift32Generator& rng, bool jitter, FunctionType get_reservoir_function) const
	{
		unsigned char retry = 0;
		bool local_invalid_sample = false;

		ReGIRReservoir reservoir;

		float3 jittered = world_position;
		do
		{
			if (jitter)
				jittered = hash_grid.jitter_world_position(world_position, current_camera, rng);

			reservoir = get_reservoir_function(jittered, current_camera, rng, &local_invalid_sample);
			retry++;
		} while (local_invalid_sample && retry < retryCount);

		if (retry == retryCount && local_invalid_sample == true)
			// We ran out of retries
			out_invalid_sample = true;
		else
			out_invalid_sample = false;

		return reservoir;
	}
};

#endif
