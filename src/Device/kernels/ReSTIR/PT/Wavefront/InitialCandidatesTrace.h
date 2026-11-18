/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_TRACE_H
#define KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_TRACE_H

#include "Device/includes/ReSTIR/PT/Wavefront/WavefrontInitialCandidates.h"

/**
 * Traces the rays queued by the shading kernels of the wavefront ReSTIR PT initial candidates.
 *
 * Same as WavefrontTracePaths except for the direction of the ray: it is read in full precision from the ReSTIR PT wavefront data, see
 * ReSTIRPTWavefrontDataDevice::path_ray_directions. The generic wavefront data only holds an octahedral encoding of it
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_WAVEFRONT_TRACE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) ReSTIR_PT_WavefrontInitialCandidatesTrace()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_PT_WavefrontInitialCandidatesTrace(HIPRTRenderData render_data, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_WAVEFRONT_TRACE_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	unsigned int input_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[1], 0u);
	if (input_count > render_data.wavefront_data.path_capacity)
		input_count = render_data.wavefront_data.path_capacity;
	if (queue_slot >= input_count)
		return;

	unsigned int pixel_index = render_data.wavefront_data.path_queues[1][queue_slot];
	if (pixel_index >= render_data.wavefront_data.path_capacity)
		return;

	WavefrontTracePayload trace_payload;
	hiprtRay ray;
	HitInfo closest_hit_info;
	wavefront_load_trace_ray(render_data, pixel_index, trace_payload, ray, closest_hit_info);
	ray.direction = render_data.restir_pt_wavefront_data.path_ray_directions[pixel_index];

	Xorshift32Generator random_number_generator(render_data.wavefront_data.path_rng_states[pixel_index]);
	bool intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, trace_payload, closest_hit_info, random_number_generator);

	wavefront_store_trace_result(render_data, pixel_index, trace_payload.volume_state, closest_hit_info, intersection_found,
								 random_number_generator.m_state.seed);

	bool terminal_path = (render_data.wavefront_data.path_state_flags[pixel_index] & WAVEFRONT_PATH_STATE_TERMINAL) != 0;
	if (!terminal_path && intersection_found)
		// The hit is still in registers: classifying it here avoids a separate classification pass reloading the trace results.
		// The path goes straight to the queue of the material family of its hit
		wavefront_classify_secondary_hit(render_data, pixel_index, closest_hit_info, trace_payload.volume_state.inside_material);
	else
		// The completion kernel adds the last deferred BSDF candidate of the path
		wavefront_enqueue_path(render_data, WAVEFRONT_COMPLETION_QUEUE_INDEX, pixel_index);
}

#endif // #ifndef KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_TRACE_H
