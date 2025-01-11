/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_GPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_GPU_DATA_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/NEEPlusPlusCPUGPUCommonData.h"

struct NEEPlusPlusGPUData : public NEEPlusPlusCPUGPUCommonData
{
	// This is the timer value 
	static constexpr float FINALIZE_ACCUMULATION_TIMER = 2000.0f;
	static constexpr float FINALIZE_ACCUMULATION_START_TIMER = 500.0f;

	// How many seconds to render before copying the
	// visibility accumulation buffers to the visibility map
	//
	// See the comments of 'accumulation_buffer' and 'accumulation_buffer_count'
	// in the NEE++ device structure for more details
	//
	// Note that this parameter is dynamically updated by the application so even though
	// it is initialized at 2000.0f, it will actually decrease until it reaches 0ms. At 0ms, the buffers
	// are copied and this variable (which is essentially a timer) is reset back to its default counter value
	float milliseconds_before_finalizing_accumulation = FINALIZE_ACCUMULATION_START_TIMER;

	OrochiBuffer<unsigned int> visibility_map;
	OrochiBuffer<unsigned int> visibility_map_count;
	OrochiBuffer<unsigned int> accumulation_buffer;
	OrochiBuffer<unsigned int> accumulation_buffer_count;
};

#endif