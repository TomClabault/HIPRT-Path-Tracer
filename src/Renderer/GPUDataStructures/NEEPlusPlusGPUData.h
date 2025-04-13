/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_GPU_DATA_H
#define RENDERER_NEE_PLUS_PLUS_GPU_DATA_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "Renderer/CPUGPUCommonDataStructures/NEEPlusPlusCPUGPUCommonData.h"

struct NEEPlusPlusGPUData : public NEEPlusPlusCPUGPUCommonData
{
	NEEPlusPlusGPUData();

	void compile_finalize_accumulation_kernel(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx);
	void recompile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, bool silent = false, bool use_cache = true);

	// This is the timer value 
	static constexpr float FINALIZE_ACCUMULATION_TIMER = 2000.0f;
	static constexpr float FINALIZE_ACCUMULATION_START_TIMER = 500.0f;

	static constexpr float STATISTICS_REFRESH_TIMER = 1000.0f;

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

	OrochiBuffer<unsigned int> packed_buffer;
	
	// Counters on the GPU for tracking 
	OrochiBuffer<unsigned long long int> total_shadow_ray_queries;
	OrochiBuffer<unsigned long long int> shadow_rays_actually_traced;
	// Same counters but on the CPU for displaying the stats in ImGui.
	// These counters are updated 
	unsigned long long int total_shadow_ray_queries_cpu = 1;
	unsigned long long int shadow_rays_actually_traced_cpu = 1;
	float statistics_refresh_timer = STATISTICS_REFRESH_TIMER;

	std::shared_ptr<GPUKernel> finalize_accumulation_kernel;
};

#endif