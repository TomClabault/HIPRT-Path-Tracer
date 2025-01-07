/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_CPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_CPU_DATA_H

// For int3 and AtomicType
#include "HostDeviceCommon/Math.h"

#include <vector>

struct NEEPlusPlusCPUData
{
	// Dimensions of the visibility map
	int3 map_dimensions = make_int3(NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE);

	std::vector<AtomicType<int>> visibility_map;
	std::vector<AtomicType<int>> visibility_map_count;
};

#endif