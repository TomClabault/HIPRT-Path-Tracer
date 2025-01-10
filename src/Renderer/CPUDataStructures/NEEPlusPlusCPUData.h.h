/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_CPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_CPU_DATA_H

// For int3 and AtomicType
#include "HostDeviceCommon/Math.h"
#include "Renderer/CPUGPUCommonDataStructures/NEEPlusPlusCPUGPUCommonData.h"

#include <vector>

struct NEEPlusPlusCPUData : public NEEPlusPlusCPUGPUCommonData
{
	int frame_timer_before_visibility_map_update = 1;

	std::vector<unsigned int> visibility_map;
	std::vector<unsigned int> visibility_map_count;
	std::vector<AtomicType<unsigned int>> accumulation_buffer;
	std::vector<AtomicType<unsigned int>> accumulation_buffer_count;
};

#endif