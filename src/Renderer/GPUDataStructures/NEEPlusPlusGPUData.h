/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_GPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_GPU_DATA_H

#include "HIPRT-Orochi/OrochiBuffer.h"

struct NEEPlusPlusGPUData
{
	// Dimensions of the visibility map
	int3 map_dimensions = make_int3(NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE);

	OrochiBuffer<int> visibility_map;
	OrochiBuffer<int> visibility_map_count;
};

#endif