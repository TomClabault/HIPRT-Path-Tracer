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
	unsigned int get_visibility_matrix_element_count() const
	{
		unsigned int grid_elements_count = map_dimensions.x * map_dimensions.y * map_dimensions.z;

		// Dividing by 2 because the visibility map is symmetrical so we only need half of the matrix
		unsigned half_matrix_size = grid_elements_count * (grid_elements_count + 1) / 2;

		return half_matrix_size;
	}

	// Dimensions of the visibility map
	int3 map_dimensions = make_int3(NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE);

	int frame_timer_before_visibility_map_update = 8;

	std::vector<unsigned int> visibility_map;
	std::vector<unsigned int> visibility_map_count;
	std::vector<AtomicType<unsigned int>> accumulation_buffer;
	std::vector<AtomicType<unsigned int>> accumulation_buffer_count;
};

#endif