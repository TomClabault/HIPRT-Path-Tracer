/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_GPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_GPU_DATA_H

#include "HIPRT-Orochi/OrochiBuffer.h"

struct NEEPlusPlusGPUData
{
	static constexpr float FINALIZE_ACCUMULATION_TIMER = 2000.0f;

	unsigned int get_visibility_matrix_element_count() const
	{
		unsigned int grid_elements_count = map_dimensions.x * map_dimensions.y * map_dimensions.z;

		// Dividing by 2 because the visibility map is symmetrical so we only need half of the matrix
		unsigned half_matrix_size = grid_elements_count * (grid_elements_count + 1) / 2;

		return half_matrix_size;
	}

	// Dimensions of the visibility map
	int3 map_dimensions = make_int3(NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEEPlusPlus::NEE_PLUS_PLUS_DEFAULT_GRID_SIZE);

	// How many seconds to render before copying the
	// visibility accumulation buffers to the visibility map
	//
	// See the comments of 'accumulation_buffer' and 'accumulation_buffer_count'
	// in the NEE++ device structure for more details
	//
	// Note that this parameter is dynamically updated by the application so even though
	// it is initialized at 2000.0f, it will actually decrease until it reaches 0ms. At 0ms, the buffers
	// are copied and this variable (which is essentially a timer) is reset back to its default counter value
	float milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_TIMER;

	OrochiBuffer<unsigned int> visibility_map;
	OrochiBuffer<unsigned int> visibility_map_count;
	OrochiBuffer<unsigned int> accumulation_buffer;
	OrochiBuffer<unsigned int> accumulation_buffer_count;
};

#endif