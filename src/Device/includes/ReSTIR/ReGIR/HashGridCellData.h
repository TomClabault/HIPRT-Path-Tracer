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

	// These three buffers are only allocated per each cell, not per each reservoir so they are
	// 'number_cells' in size

	// Buffer that holds the index of the thread that inserted into that grid cell
	AtomicType<int>* hit_primitive = nullptr;
	float3* world_points = nullptr;
	Octahedral24BitNormalPadded32b* world_normals = nullptr;

	// TODO these guys in a single buffer to have only one memory access
	// TODO don't need these, we can fetch the material with the hit primitive index. Some more indirection however. How is going to be the perf?
	unsigned char* roughness = nullptr;
	unsigned char* specular = nullptr;
	unsigned char* metallic = nullptr;
	// The checksum for each entry of the table to check for collisions
	AtomicType<unsigned int>* checksums = nullptr;

	// The staging buffer is used to store the grid cells that are alive during shading: for each grid cell that a ray falls into during shading,
	// we position the unsigned char to 1
	//
	// We need a staging buffer to do that because modifying the 'grid_cell_alive' buffer directly would be a race condition since other threads
	// may be reading from that buffer at the same time to see if a cell is alive or not
	//
	// That staging buffer is then copied to the 'grid_cell_alive' buffer at the end of the frame
	//
	// TODO Do we need this buffer? Can we not just check the hit primitive index for example? If it's -1, the cell is not alive
	AtomicType<unsigned int>* grid_cell_alive = nullptr;
	unsigned int* grid_cells_alive_list = nullptr;

	AtomicType<unsigned int>* grid_cells_alive_count = nullptr;
};

#endif
