/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_SETTINGS_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/GridBufferSoAA.h"
#include "Device/includes/ReSTIR/ReGIR/GridSettings.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridSoAA.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/Xorshift.h"

struct ReGIRGridFillSettings
{
	// How many light samples are resampled into each reservoir of the grid cell
	int sample_count_per_cell_reservoir = 32;

	ReGIRGridBufferSoADevice grid_buffers;

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

	// Grid that contains the output of the spatial reuse pass
	ReGIRGridBufferSoADevice output_grid;
};

struct ReGIRShadingSettings
{
	// At path tracing time, how many reservoirs of the grid cell of the point we're trying to shade
	// are going to be resampled (with the BRDF term) to produce the final light sample used for NEE
	int cell_reservoir_resample_per_shading_point = 4;
	// Whether or not to jitter the world space position used when looking up the ReGIR grid
	// This helps eliminate grid discretization  artifacts
	bool do_cell_jittering = true;

	// For each grid cell, indicates whether that grid cell has been used at least once during shading in the last frame
	// 
	// The unsigned char contains 1 if the grid cell was alive, meaning that the grid cell will be populated during the grid fill
	// as well as during the spatial reuse and that grid cell is also able to be used during shading
	//
	// If the unsigned char is 0, that grid cell hasn't been used last frame and will be filled by the grid fill/temporal/spatial reuse
	// passes
	const unsigned char* grid_cells_alive = nullptr;

	// The staging buffer is used to store the grid cells that are alive during shading: for each grid cell that a ray falls into during shading,
	// we position the unsigned char to 1
	//
	// We need a staging buffer to do that because modifying the 'grid_cells_alive' buffer directly would be a race condition since other threads
	// may be reading from that buffer at the same time to see if a cell is alive or not
	//
	// That staging buffer is then copied to the 'grid_cells_alive' buffer at the end of the frame
	AtomicType<unsigned int>* grid_cells_alive_staging = nullptr;
	unsigned int* grid_cells_alive_list = nullptr;
	unsigned int grid_cells_alive_count;
	AtomicType<unsigned int>* grid_cells_alive_count_staging = nullptr;
};

struct ReGIRRepresentative
{
	// If the current representative data of a cell is at a distance 'OK_DISTANCE_TO_CENTER_FACTOR * get_cell_diagonal_length()' or lower,
	// then we assume that the representative data is good enough and we do not update it anymore
	static constexpr float OK_DISTANCE_TO_CENTER_FACTOR = 0.4f;

	static constexpr float UNDEFINED_DISTANCE = -42.0f;
	static constexpr unsigned int UNDEFINED_POINT = 0xFFFFFFFF;
	static constexpr float3 UNDEFINED_NORMAL = { 0.0f, 0.0f, 0.0f };
	static constexpr int UNDEFINED_PRIMITIVE = -1;

	// TODO Pack distance to unsigned char?
	AtomicType<float>* distance_to_center = nullptr;
	AtomicType<int>* representative_primitive = nullptr;
	// TODO test quantize these guys but we may end up with the point below the
	// actual surface because of the quantization and this may be an issue for shadow rays
	// on very finely tesselated geometry because the shadow rays may end up hitting another primitive
	unsigned int* representative_points = nullptr;
	// TODO Pack to octahedral
	Octahedral24BitNormalPadded32b* representative_normals = nullptr;
};

struct ReGIRSettings
{

	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE float3 get_cell_size() const
	{
		return ReGIRGridBufferSoADevice::get_cell_size(grid);
	}

	HIPRT_DEVICE float get_cell_diagonal_length() const
	{
		return ReGIRGridBufferSoADevice::get_cell_diagonal_length(grid);
	}

	HIPRT_DEVICE float3 get_cell_center_from_linear_cell_index(unsigned int linear_cell_index) const
	{
		return ReGIRGridBufferSoADevice::get_cell_center_from_linear_cell_index(grid, linear_cell_index);
	}

	HIPRT_DEVICE float3 get_cell_origin_from_linear_cell_index(int linear_cell_index) const
	{
		return ReGIRGridBufferSoADevice::get_cell_origin_from_linear_cell_index(grid, linear_cell_index);
	}

	HIPRT_DEVICE int get_linear_cell_index_from_xyz(int3 xyz_cell_index) const
	{
		return ReGIRGridBufferSoADevice::get_linear_cell_index_from_xyz(grid, xyz_cell_index);
	}

	HIPRT_DEVICE int get_linear_cell_index_from_world_pos(float3 world_position, Xorshift32Generator* rng = nullptr, bool jitter = false) const
	{
		return ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, world_position, rng, jitter);
	}

	HIPRT_DEVICE int3 get_xyz_cell_index_from_linear(int linear_cell_index) const
	{
		return ReGIRGridBufferSoADevice::get_xyz_cell_index_from_linear(grid, linear_cell_index);
	}

	///////////////////// Delegating to the grid for these functions /////////////////////

	HIPRT_DEVICE int get_reservoir_index_in_grid_from_world_pos(float3 world_position, int reservoir_index_in_cell) const
	{
		int linear_cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, world_position);

		return linear_cell_index * grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;
	}

	/**
	 * Here, 'non_canonical_reservoir_index_in_cell' should be in [0, grid_fill.get_non_canonical_reservoir_count_per_cell() - 1]
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_non_canonical_reservoir_from_cell_reservoir_index(int linear_cell_index, int non_canonical_reservoir_index_in_cell) const
	{
		return get_reservoir_for_shading_from_cell_indices(linear_cell_index, non_canonical_reservoir_index_in_cell);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_non_canonical_reservoir(int linear_cell_index, Xorshift32Generator& rng) const
	{
		int random_non_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
			random_non_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

		return get_cell_non_canonical_reservoir_from_cell_reservoir_index(linear_cell_index, random_non_canonical_reservoir_index_in_cell);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_non_canonical_reservoir(float3 world_position, float3 camera_position, Xorshift32Generator& rng) const
	{
		int random_non_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
			random_non_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

		int linear_cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, world_position);
		return get_cell_non_canonical_reservoir_from_cell_reservoir_index(linear_cell_index, random_non_canonical_reservoir_index_in_cell);
	}

	/**
	 * Here, 'canonical_reservoir_index_in_cell' should be in [0, grid_fill.get_canonical_reservoir_count_per_cell() - 1]
	 */
	HIPRT_DEVICE ReGIRReservoir get_cell_canonical_reservoir_from_cell_reservoir_index(float3 world_position, float3 camera_position, int canonical_reservoir_index_in_cell) const
	{
		int linear_cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, world_position);

		return get_reservoir_for_shading_from_cell_indices(linear_cell_index, canonical_reservoir_index_in_cell);
	}

	HIPRT_DEVICE ReGIRReservoir get_random_cell_canonical_reservoir(float3 world_position, float3 camera_position, Xorshift32Generator& rng) const
	{
		int random_canonical_reservoir_index_in_cell = 0;
		if (grid_fill.get_canonical_reservoir_count_per_cell() > 1)
			random_canonical_reservoir_index_in_cell = rng.random_index(grid_fill.get_canonical_reservoir_count_per_cell());

		return get_cell_canonical_reservoir_from_cell_reservoir_index(world_position, camera_position, random_canonical_reservoir_index_in_cell);
	}

	/**
	 * If 'out_point_outside_of_grid' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_reservoir_for_shading_from_cell_indices(int linear_cell_index, int reservoir_index_in_cell) const
	{
		if (spatial_reuse.do_spatial_reuse)
			// If spatial reuse is enabled, we're shading with the reservoirs from the output of the spatial reuse
			return spatial_reuse.output_grid.read_full_reservoir_opt(grid, linear_cell_index, reservoir_index_in_cell);
		else if (temporal_reuse.do_temporal_reuse)
			// If only doing temporal reuse, reading from the output of the spatial reuse pass
			return get_temporal_reservoir_opt(linear_cell_index, reservoir_index_in_cell);
		else
			// No temporal reuse and no spatial reuse, reading from the output of the grid fill pass
			return grid_fill.grid_buffers.read_full_reservoir_opt(grid, linear_cell_index, reservoir_index_in_cell);
	}

	/**
	 * If 'out_point_outside_of_grid' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_DEVICE ReGIRReservoir get_non_canonical_reservoir_for_shading_from_world_pos(float3 world_position, float3 camera_position, bool& out_point_outside_of_grid, Xorshift32Generator& rng, bool jitter = false, int sample_number = -1) const
	{	
		int linear_cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, world_position, &rng, jitter);
		if (linear_cell_index < 0 || linear_cell_index >= grid.grid_resolution.x * grid.grid_resolution.y * grid.grid_resolution.z)
		{
			// The cell index is physically outside of the grid
			// We're indicating that this cell should not be used by setting the 'out_point_outside_of_grid' to true
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		// TODO try an if (shading.grid_cells_alive_staging[linear_cell_index] == 0) to avoid the atomic operation and see if perf is better
		// Someone just wanted to use that grid cell so it's going to be alive in the next frame so we're indicating that in the staging buffer
#if ReGIR_DoDispatchCompaction == KERNEL_OPTION_TRUE
		// Only go through all that atomic stuff if the cell hasn't been staged already
		if (shading.grid_cells_alive_staging[linear_cell_index] == 0)
		{
			if (hippt::atomic_compare_exchange(&shading.grid_cells_alive_staging[linear_cell_index], 0u, 1u) == 0u)
			{
				unsigned int cell_alive_index = hippt::atomic_fetch_add(shading.grid_cells_alive_count_staging, 1u);

				shading.grid_cells_alive_list[cell_alive_index] = linear_cell_index;
			}
		}
#else
		shading.grid_cells_alive_staging[linear_cell_index] = 1;
#endif

		 if (shading.grid_cells_alive[linear_cell_index] == 0)
		 {
		 	// The grid cell is inside the grid but not alive
		 	// We're indicating that this cell should not be used by setting the 'out_point_outside_of_grid' to true
		 	out_point_outside_of_grid = true;

		 	return ReGIRReservoir();
		 }

		out_point_outside_of_grid = false;

		return get_random_cell_non_canonical_reservoir(linear_cell_index, rng);
	}

	HIPRT_DEVICE ReGIRReservoir get_canonical_reservoir_for_shading_from_world_pos(float3 world_position, float3 camera_position, bool& out_point_outside_of_grid, Xorshift32Generator& rng, bool jitter = false, int sample_number = -1) const
	{
		int linear_cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, world_position, &rng, jitter);
		if (linear_cell_index < 0 || linear_cell_index >= grid.grid_resolution.x * grid.grid_resolution.y * grid.grid_resolution.z)
		{
			// The cell index is physically outside of the grid
			// We're indicating that this cell should not be used by setting the 'out_point_outside_of_grid' to true
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		// Someone just wanted to use that grid cell so it's going to be alive in the next frame so we're indicating that in the staging buffer
#if ReGIR_DoDispatchCompaction == KERNEL_OPTION_TRUE
		// Only go through all that atomic stuff if the cell hasn't been staged already
		if (shading.grid_cells_alive_staging[linear_cell_index] == 0)
		{
			if (hippt::atomic_compare_exchange(&shading.grid_cells_alive_staging[linear_cell_index], 0u, 1u) == 0u)
			{
				unsigned int cell_alive_index = hippt::atomic_fetch_add(shading.grid_cells_alive_count_staging, 1u);

				shading.grid_cells_alive_list[cell_alive_index] = linear_cell_index;
			}
		}
#else
		shading.grid_cells_alive_staging[linear_cell_index] = 1;
#endif

		if (shading.grid_cells_alive[linear_cell_index] == 0)
		{
			// The grid cell is inside the grid but not alive
			// We're indicating that this cell should not be used by setting the 'out_point_outside_of_grid' to true
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		out_point_outside_of_grid = false;

		return get_random_cell_non_canonical_reservoir(linear_cell_index, rng);
	}

	HIPRT_DEVICE int get_neighbor_replay_linear_cell_index_for_shading(float3 shading_point, Xorshift32Generator& rng, bool jitter = false) const
	{
		int linear_cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, shading_point, &rng, jitter);
		if (linear_cell_index < 0 || linear_cell_index >= grid.grid_resolution.x * grid.grid_resolution.y * grid.grid_resolution.z)
			return -1;

		if (shading.grid_cells_alive[linear_cell_index] == 0)
			// The grid cell is inside the grid but not alive
			return -1;

		// Advancing the RNG to mimic 'get_non_canonical_reservoir_for_shading_from_world_pos'
		if (grid_fill.get_non_canonical_reservoir_count_per_cell() > 1)
			rng.random_index(grid_fill.get_non_canonical_reservoir_count_per_cell());

		return linear_cell_index;
	}

	HIPRT_DEVICE ReGIRReservoir get_temporal_reservoir_opt(int linear_cell_index, int reservoir_index_in_cell, int grid_index = -1) const
	{
		if (grid_index != -1)
			return grid_fill.grid_buffers.read_full_reservoir_opt(grid, linear_cell_index, reservoir_index_in_cell, grid_index);
		else
			return grid_fill.grid_buffers.read_full_reservoir_opt(grid, linear_cell_index, reservoir_index_in_cell, temporal_reuse.current_grid_index);
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
			return grid_fill.grid_buffers.read_full_reservoir_opt(grid, world_position, camera_position, reservoir_index_in_cell, grid_index);
		else
			return grid_fill.grid_buffers.read_full_reservoir_opt(grid, world_position, camera_position, reservoir_index_in_cell, temporal_reuse.current_grid_index);
	}

	HIPRT_DEVICE ReGIRReservoir get_grid_fill_output_reservoir_opt(float3 world_position, float3 camera_position, int reservoir_index_in_cell) const
	{
		// The output of the grid fill pass is in the current frame grid so we can call the temporal method with
		// index -1
		return get_temporal_reservoir_opt(world_position, camera_position, reservoir_index_in_cell, -1);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt(const ReGIRReservoir& reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell)
	{
		spatial_reuse.output_grid.store_reservoir_and_sample_opt(reservoir, world_position, camera_position, reservoir_index_in_cell);
	}

	HIPRT_DEVICE void store_spatial_reservoir_opt_from_index_in_grid(const ReGIRReservoir& reservoir, int reservoir_index_in_grid)
	{
		spatial_reuse.output_grid.store_reservoir_and_sample_opt_from_index_in_grid(reservoir, reservoir_index_in_grid);
	}

	HIPRT_DEVICE void store_reservoir_opt(ReGIRReservoir reservoir, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index != -1)
			grid_fill.grid_buffers.store_reservoir_and_sample_opt(reservoir, world_position, camera_position, reservoir_index_in_cell, grid_index);
		else
			grid_fill.grid_buffers.store_reservoir_and_sample_opt(reservoir, world_position, camera_position, reservoir_index_in_cell, temporal_reuse.current_grid_index);
	}

	HIPRT_DEVICE void store_reservoir_opt_from_index_in_grid(const ReGIRReservoir& reservoir, int reservoir_index_in_grid, int grid_index = -1)
	{
		grid_fill.grid_buffers.store_reservoir_and_sample_opt_from_index_in_grid(reservoir, reservoir_index_in_grid, grid_index);
	}

	HIPRT_DEVICE ColorRGB32F get_random_cell_color(float3 position) const
	{
		int cell_index = ReGIRGridBufferSoADevice::get_linear_cell_index_from_world_pos(grid, position);

		return ColorRGB32F::random_color(cell_index);
	}

	HIPRT_DEVICE unsigned int get_total_number_of_cells_per_grid() const
	{
#ifdef __KERNELCC__
		return grid.m_total_number_of_cells;
#else
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return grid.grid_resolution.x * grid.grid_resolution.y * grid.grid_resolution.z;
#endif
	}

	HIPRT_DEVICE unsigned int get_number_of_reservoirs_per_grid() const
	{
#ifdef __KERNELCC__
		return grid.m_number_of_reservoirs_per_grid;
#else
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return get_total_number_of_cells_per_grid() * grid_fill.get_total_reservoir_count_per_cell();
#endif
	}

	HIPRT_DEVICE unsigned int get_number_of_reservoirs_per_cell() const
	{
#ifdef __KERNELCC__
		return grid.m_number_of_reservoirs_per_cell;
#else
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		return grid_fill.get_total_reservoir_count_per_cell();
#endif
	}

	HIPRT_DEVICE unsigned int get_total_number_of_reservoirs_ReGIR() const
	{
#ifdef __KERNELCC__
		return grid.m_total_number_of_reservoirs;
#else
		// We need to keep this dynamic on the CPU so not using the precomputed variable
		int temporal_grid_count = temporal_reuse.do_temporal_reuse ? temporal_reuse.temporal_history_length : 1;

		return get_number_of_reservoirs_per_grid() * temporal_grid_count;
#endif
	}

	/**
	 * Resets all the reservoirs of all the grids at the given 'reservoir_index'
	 */
	HIPRT_DEVICE void reset_reservoirs(int reservoir_index_in_grid)
	{
		int temporal_grid_count = temporal_reuse.do_temporal_reuse ? temporal_reuse.temporal_history_length : 1;

		for (int grid_index = 0; grid_index < temporal_grid_count; grid_index++)
            store_reservoir_opt_from_index_in_grid(ReGIRReservoir(), reservoir_index_in_grid, grid_index);

		// Also clearing the spatial reuse output buffers (grid) if spatial reuse is enabled
		if (spatial_reuse.do_spatial_reuse)
			spatial_reuse.output_grid.store_reservoir_and_sample_opt_from_index_in_grid(ReGIRReservoir(), reservoir_index_in_grid);
	}

	bool DEBUG_INCLUDE_CANONICAL = true;

	ReGIRGridSettings grid;
	ReGIRGridFillSettings grid_fill;
	ReGIRTemporalReuseSettings temporal_reuse;
	ReGIRSpatialReuseSettings spatial_reuse;
	ReGIRShadingSettings shading;
	ReGIRRepresentative representative;

	// If true, the center of the cells will not be used anymore to compute the ReGIR target functions
	// but rather, points on the surface of the scene within the cells will be used
	//
	// This feature is non-deterministic
	bool use_representative_points = true;
	
	// Multiplicative factor to multiply the output of some debug views
	float debug_view_scale_factor = 0.05f;
};

#endif
