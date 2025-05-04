/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_HASH_GRID_CELL_DATA_H
#define DEVICE_KERNELS_REGIR_HASH_GRID_CELL_DATA_H

#include "HostDeviceCommon/Packing.h"

struct ReGIRHashCellDataSoADevice
{
	// If the current representative data of a cell is at a distance 'OK_DISTANCE_TO_CENTER_FACTOR * get_cell_diagonal_length()' or lower,
	// then we assume that the representative data is good enough and we do not update it anymore
	static constexpr float OK_DISTANCE_TO_CENTER_FACTOR = 0.4f;

	static constexpr float UNDEFINED_DISTANCE = -42.0f;
	static constexpr unsigned int UNDEFINED_POINT = 0xFFFFFFFF;
	static constexpr float3 UNDEFINED_NORMAL = { 0.0f, 0.0f, 0.0f };
	static constexpr int UNDEFINED_PRIMITIVE = -1;
	static constexpr unsigned int UNDEFINED_HASH_KEY = 0xFFFFFFFF;

	// These three buffers are only allocated per each cell, not per each reservoir so they are
	// 'number_cells' in size

	// TODO Pack distance to unsigned char? Yep but then we need a way to atomic on that
	//AtomicType<float>* distance_to_center = nullptr;
	AtomicType<int>* representative_primitive = nullptr;
	// TODO test quantize these guys but we may end up with the point below the
	// actual surface because of the quantization and this may be an issue for shadow rays
	// on very finely tesselated geometry because the shadow rays may end up hitting another primitive
	//unsigned int* representative_points = nullptr;
	float3* representative_points = nullptr;
	// TODO Pack to octahedral
	Octahedral24BitNormalPadded32b* representative_normals = nullptr;
	// The has for each entry of the table to check for collisions
	unsigned int* hash_keys = nullptr;
};

#endif
