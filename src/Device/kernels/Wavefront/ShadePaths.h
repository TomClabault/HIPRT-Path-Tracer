/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_SHADE_PATHS_H
#define KERNELS_WAVEFRONT_SHADE_PATHS_H

#include "Device/includes/Wavefront/WavefrontCommon.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_SHADE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) WavefrontShadePaths(unsigned int bounce_count)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline WavefrontShadePaths(HIPRTRenderData render_data, unsigned int bounce_count, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SHADE_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	unsigned int input_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[0], 0u);
	if (queue_slot >= input_count)
		return;

	unsigned int pixel_index = render_data.wavefront_data.path_queues[0][queue_slot];
	if (pixel_index >= render_data.wavefront_data.path_capacity)
		return;

	int x = static_cast<int>(pixel_index % render_data.render_settings.render_resolution.x);
	int y = static_cast<int>(pixel_index / render_data.render_settings.render_resolution.x);

	RayPayload ray_payload;
	hiprtRay ray;
	HitInfo closest_hit_info;
	bool intersection_found;
	wavefront_load_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);

	Xorshift32Generator random_number_generator(render_data.wavefront_data.path_rng_states[pixel_index]);
	NEEDeferredMISContext nee_deferred_MIS_context;

	if (ray_payload.next_ray_state == RayState::MISSED)
	{
		wavefront_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
											 nee_deferred_MIS_context, false);
		return;
	}

	if (!intersection_found)
	{
		ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, pixel_index);
		ray_payload.next_ray_state = RayState::MISSED;

		wavefront_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
											 nee_deferred_MIS_context, true);
		return;
	}

	if (ray_payload.bounce == 0)
		store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);
	else
	{
		bool ReGIR_primary_hit = render_data.render_settings.regir_settings.compute_is_primary_hit(ray_payload);

		// Storing data for ReGIR representative points
		ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, closest_hit_info.geometric_normal, render_data.current_camera,
										 closest_hit_info.primitive_index, ReGIR_primary_hit, ray_payload.material);
	}

	if (ray_payload.bounce > 0 || render_data.render_settings.enable_direct_lighting)
	{
		ray_payload.ray_color +=
			estimate_direct_lighting(render_data, ray_payload, closest_hit_info, -ray.direction, x, y, nee_deferred_MIS_context, random_number_generator);

		sanity_check<true>(render_data, ray_payload.ray_color, x, y);
	}

	BSDFIncidentLightInfo sampled_light_info = BSDFIncidentLightInfo::NO_INFO; // This variable is never used, this is just for debugging on the CPU
																			   // so that we know what the BSDF sampled
	bool valid_indirect_bounce = path_tracing_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray,
																		   random_number_generator, sampled_light_info, nee_deferred_MIS_context);

	if (!valid_indirect_bounce)
	{
		wavefront_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
											 nee_deferred_MIS_context, true);
		return;
	}

	if (ray_payload.bounce >= static_cast<int>(bounce_count))
	{
		// The original megakernel increments the loop counter before the final deferred NEE pass.
		ray_payload.bounce++;
		wavefront_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
											 nee_deferred_MIS_context, false);
		return;
	}

	ray_payload.bounce++;
	wavefront_store_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);
	wavefront_store_nee_deferred_mis_context(render_data, pixel_index, nee_deferred_MIS_context);
	render_data.wavefront_data.path_rng_states[pixel_index] = random_number_generator.m_state.seed;
	wavefront_enqueue_path(render_data, 1, pixel_index);
}

#endif // #ifndef KERNELS_WAVEFRONT_SHADE_PATHS_H
