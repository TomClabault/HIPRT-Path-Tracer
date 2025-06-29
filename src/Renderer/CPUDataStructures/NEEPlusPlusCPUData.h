/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_CPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_CPU_DATA_H

// For int3 and AtomicType
#include "HostDeviceCommon/Math.h"

#include <vector>

struct NEEPlusPlusCPUData
{
	int frame_timer_before_visibility_map_update = 1;

	std::vector<AtomicType<unsigned int>> total_unoccluded_rays;
	std::vector<AtomicType<unsigned int>> total_num_rays;

	std::vector<AtomicType<unsigned int>> num_rays_staging;
	std::vector<AtomicType<unsigned int>> unoccluded_rays_staging;

	std::vector<AtomicType<unsigned int>> checksum_buffer;

	AtomicType<unsigned long long int> total_shadow_ray_queries;
	AtomicType<unsigned long long int> shadow_rays_actually_traced;
	AtomicType<unsigned int> total_cell_alive_count;
};

#endif