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
	static constexpr unsigned int UNDEFINED_HASH_KEY = 0xFFFFFFFF;

	static constexpr unsigned int CELL_LOCKED_SENTINEL_VALUE = 0u;

	// These three buffers are only allocated per each cell, not per each reservoir so they are
	// 'number_cells' in size

	// TODO Pack distance to unsigned char? Yep but then we need a way to atomic on that
	//
	// These 3 buffers are used to "optimize" the representative points of the cells
	// closer to the center of the cells
	AtomicType<float>* distance_to_center = nullptr;
	float3* sum_points = nullptr;
	AtomicType<unsigned int>* num_points = nullptr;

	AtomicType<int>* hit_primitive = nullptr;
	// TODO test quantize these guys but we may end up with the point below the
	// actual surface because of the quantization and this may be an issue for shadow rays
	// on very finely tesselated geometry because the shadow rays may end up hitting another primitive
	float3* world_points = nullptr;
	// TODO Pack to octahedral
	Octahedral24BitNormalPadded32b* world_normals = nullptr;
	// The has for each entry of the table to check for collisions
	AtomicType<unsigned int>* hash_keys = nullptr;
};

#endif
