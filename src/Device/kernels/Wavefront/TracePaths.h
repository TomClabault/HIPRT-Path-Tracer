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

	RayPayload ray_payload;
	hiprtRay ray;
	HitInfo closest_hit_info;
	wavefront_load_trace_ray(render_data, pixel_index, ray_payload, ray, closest_hit_info);

	Xorshift32Generator random_number_generator(render_data.wavefront_data.path_rng_states[pixel_index]);
	bool intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);

	wavefront_load_trace_path_bookkeeping(render_data, pixel_index, ray_payload);
	NEEDeferredMISContext nee_deferred_MIS_context;
	wavefront_load_nee_deferred_mis_context(render_data, pixel_index, nee_deferred_MIS_context);

	ray_payload.ray_color +=
		do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, closest_hit_info, nee_deferred_MIS_context, random_number_generator);

	if (!intersection_found)
	{
		int x = static_cast<int>(pixel_index % render_data.render_settings.render_resolution.x);
		int y = static_cast<int>(pixel_index / render_data.render_settings.render_resolution.x);

		ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, pixel_index);
		ray_payload.next_ray_state = RayState::MISSED;

		wavefront_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
											 nee_deferred_MIS_context, true);
		return;
	}

	wavefront_store_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);
	render_data.wavefront_data.path_rng_states[pixel_index] = random_number_generator.m_state.seed;
	wavefront_enqueue_path(render_data, 0, pixel_index);
}

#endif // #ifndef KERNELS_WAVEFRONT_TRACE_PATHS_H
