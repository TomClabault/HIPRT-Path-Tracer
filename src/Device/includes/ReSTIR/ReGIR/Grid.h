/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_GRID_H
#define DEVICE_KERNELS_REGIR_GRID_H

#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

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

	HIPRT_HOST_DEVICE ReGIRReservoir get_cell_reservoir(float3 shading_point) const
	{
		float3 position_in_grid = shading_point - origin;
		float3 position_in_grid_cell_unit = position_in_grid / get_cell_size();
		int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
		int cell_linear_index = cell_xyz.x + cell_xyz.y * grid_resolution.x + cell_xyz.z * grid_resolution.x * grid_resolution.y;

		return grid_buffer[cell_linear_index];
	}

	float3 origin;

	// "Length" of the grid in each X, Y, Z axis directions
	float3 extents;

	int3 grid_resolution = make_int3(16, 16, 16);

	ReGIRReservoir* grid_buffer = nullptr;
};

#endif
