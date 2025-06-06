/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_GPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_GPU_DATA_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"

struct NEEPlusPlusGPUData
{
	OrochiBuffer<unsigned int> total_unoccluded_rays;
	OrochiBuffer<unsigned int> total_num_rays;

	OrochiBuffer<unsigned int> checksum_buffer;
	
	// Counters on the GPU for tracking 
	OrochiBuffer<unsigned long long int> total_shadow_ray_queries;
	OrochiBuffer<unsigned long long int> shadow_rays_actually_traced;

	// Same counters but on the CPU for displaying the stats in ImGui.
	// These counters are updated 
	unsigned long long int total_shadow_ray_queries_cpu = 1;
	unsigned long long int shadow_rays_actually_traced_cpu = 1;
};

#endif