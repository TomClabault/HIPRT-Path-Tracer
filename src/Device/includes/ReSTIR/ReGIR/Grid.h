/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_GRID_H
#define DEVICE_KERNELS_REGIR_GRID_H

#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

#include "HostDeviceCommon/Xorshift.h"

struct ReGIRGrid
{
	HIPRT_HOST_DEVICE float3 get_cell_size() const
	{
		float3 grid_resolution_float = make_float3(grid_resolution.x, grid_resolution.y, grid_resolution.z);
		float3 cell_size = extents / grid_resolution_float;

		return cell_size;
	}

	HIPRT_HOST_DEVICE float3 get_cell_center(unsigned int linear_cell_index) const
	{
		float3 cell_size = get_cell_size();

		int index_x = linear_cell_index % grid_resolution.x;
		int index_y = (linear_cell_index % (grid_resolution.x * grid_resolution.y)) / grid_resolution.x;
		int index_z = linear_cell_index % (grid_resolution.x * grid_resolution.y);

		int3 cell_index_xyz = make_int3(index_x, index_y, index_z);
		float3 cell_index_xyz_float = make_float3(static_cast<float>(index_x), static_cast<float>(index_y), static_cast<float>(index_z));

		return origin + cell_size * cell_index_xyz_float + cell_size / 2.0f;
	}

	HIPRT_HOST_DEVICE int get_cell_index(float3 world_position, bool jitter = true) const
	{
		if (jitter)
		{
			constexpr unsigned int UNSIGNED_INT_MAX = 0xffffffff;

			unsigned int x = *reinterpret_cast<unsigned int*>(&world_position.x);
			unsigned int y = *reinterpret_cast<unsigned int*>(&world_position.y);
			unsigned int z = *reinterpret_cast<unsigned int*>(&world_position.z);

			unsigned int seed = wang_hash(x ^ y ^ z * 0x27d4eb2d);

			Xorshift32Generator rng_local(seed);
			world_position += (make_float3(rng_local(), rng_local(), rng_local()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * get_cell_size();
		}

		float3 position_in_grid = world_position - origin;
		float3 position_in_grid_cell_unit = position_in_grid / get_cell_size();

		int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
		// If a point is on the very edge of the grid, we're going to have one of the coordinates be 'grid_resolution.XXX'
		// exactly, 16 for a grid resolution of 16 for example. 
		// 
		// But that's out of bounds because our grid cells are in [0, 15] so we're sub
		cell_xyz = hippt::min(cell_xyz, grid_resolution - make_int3(1, 1, 1));

		return cell_xyz.x + cell_xyz.y * grid_resolution.x + cell_xyz.z * grid_resolution.x * grid_resolution.y;
	}

	/**
	 * If 'out_point_outside_of_grid' is set to true, then the given shading point (+ the jittering) was outside of the grid
	 * and no reservoir has been gathered
	 */
	HIPRT_HOST_DEVICE ReGIRReservoir get_cell_reservoir(float3 shading_point, bool& out_point_outside_of_grid, bool jitter = true) const
	{
		int cell_linear_index = get_cell_index(shading_point, jitter);
		if (cell_linear_index < 0 || cell_linear_index >= grid_resolution.x * grid_resolution.y * grid_resolution.z)
		{
			out_point_outside_of_grid = true;

			return ReGIRReservoir();
		}

		out_point_outside_of_grid = false;
		return grid_buffer[cell_linear_index];
	}

	HIPRT_HOST_DEVICE ColorRGB32F get_random_cell_color(float3 position, bool jitter = true) const
	{
		int cell_index = get_cell_index(position, jitter);

		return ColorRGB32F::random_color(cell_index);
	}

	float3 origin;

	// "Length" of the grid in each X, Y, Z axis directions
	float3 extents;

	int3 grid_resolution = make_int3(16, 16, 16);

	ReGIRReservoir* grid_buffer = nullptr;
};

#endif
