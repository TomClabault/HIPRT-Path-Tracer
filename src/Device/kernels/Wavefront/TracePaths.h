/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_TRACE_PATHS_H
#define KERNELS_WAVEFRONT_TRACE_PATHS_H

#include "Device/includes/Wavefront/WavefrontCommon.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_TRACE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) WavefrontTracePaths()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline WavefrontTracePaths(HIPRTRenderData render_data, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_TRACE_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	unsigned int input_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[1], 0u);
	if (queue_slot >= input_count)
		return;

	unsigned int pixel_index = render_data.wavefront_data.path_queues[1][queue_slot];
	if (pixel_index >= render_data.wavefront_data.path_capacity)
		return;

	WavefrontTracePayload trace_payload;
	hiprtRay ray;
	HitInfo closest_hit_info;
	wavefront_load_trace_ray(render_data, pixel_index, trace_payload, ray, closest_hit_info);

	Xorshift32Generator random_number_generator(render_data.wavefront_data.path_rng_states[pixel_index]);
	bool intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, trace_payload, closest_hit_info, random_number_generator);

	wavefront_store_trace_result(render_data, pixel_index, trace_payload.volume_state, closest_hit_info, intersection_found,
								 random_number_generator.m_state.seed);
	wavefront_enqueue_path(render_data, 0, pixel_index);
}

#endif // #ifndef KERNELS_WAVEFRONT_TRACE_PATHS_H
