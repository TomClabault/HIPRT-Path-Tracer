/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_HASH_GRID_CELL_DATA_H
#define DEVICE_KERNELS_REGIR_HASH_GRID_CELL_DATA_H

#include "HostDeviceCommon/Packing.h"

struct ReGIRHashCellDataSoADevice
{
	static constexpr float UNDEFINED_DISTANCE = -42.0f;

	static constexpr unsigned int UNDEFINED_POINT = 0xFFFFFFFF;
	static constexpr float3 UNDEFINED_NORMAL = { 0.0f, 0.0f, 0.0f };
	static constexpr int UNDEFINED_PRIMITIVE = -1;

	static constexpr unsigned int CELL_LOCKED_SENTINEL_VALUE = 0u;

	// These three buffers are only allocated per each cell, not per each reservoir so they are
	// 'number_cells' in size

	// Two buffers sum points and num points that we keep to compute the average of the points
	// that hit each cell
	float3* sum_points = nullptr;
	// Can we have this as an unsigned char? Maybe even less: 4 bits per cell
	AtomicType<unsigned int>* num_points = nullptr;

	AtomicType<int>* hit_primitive = nullptr;
	float3* world_points = nullptr;
	Octahedral24BitNormalPadded32b* world_normals = nullptr;

	// TODO pack these to unsigned char
	float* roughness = nullptr;
	float* specular = nullptr;
	float* metallic = nullptr;
	// The has for each entry of the table to check for collisions
	AtomicType<unsigned int>* hash_keys = nullptr;

	// The staging buffer is used to store the grid cells that are alive during shading: for each grid cell that a ray falls into during shading,
	// we position the unsigned char to 1
	//
	// We need a staging buffer to do that because modifying the 'grid_cell_alive' buffer directly would be a race condition since other threads
	// may be reading from that buffer at the same time to see if a cell is alive or not
	//
	// That staging buffer is then copied to the 'grid_cell_alive' buffer at the end of the frame
	AtomicType<unsigned int>* grid_cell_alive = nullptr;
	unsigned int* grid_cells_alive_list = nullptr;

	AtomicType<unsigned int>* grid_cells_alive_count = nullptr;
};

#endif
