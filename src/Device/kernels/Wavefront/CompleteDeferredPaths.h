/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_COMPLETE_DEFERRED_PATHS_H
#define KERNELS_WAVEFRONT_COMPLETE_DEFERRED_PATHS_H

#include "Device/includes/Wavefront/WavefrontCommon.h"

// Continuing hits complete deferred lighting at the start of shading; this queue only finalizes terminal paths and misses.
template <typename PreviousBSDFMaterialType>
HIPRT_DEVICE static void wavefront_complete_terminated_path(HIPRTRenderData& render_data, unsigned int path_index)
{
	RayPayloadCommon ray_payload(NoInitTag{});
	hiprtRay ray;
	HitInfo closest_hit_info;
	bool intersection_found;
	wavefront_load_secondary_material_state(render_data, path_index, ray_payload, closest_hit_info, intersection_found);

	bool terminal_path = (render_data.wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_TERMINAL) != 0;
	if (!terminal_path && intersection_found)
		return;

	wavefront_load_secondary_shading_state(render_data, path_index, ray_payload, ray, closest_hit_info);

	Xorshift32Generator random_number_generator(render_data.wavefront_data.path_rng_states[path_index]);
	NEEDeferredMISContext nee_deferred_MIS_context;
	wavefront_load_nee_deferred_mis_context(render_data, path_index, nee_deferred_MIS_context);

	EffectiveMaterialEmission current_hit_emission;
	current_hit_emission.emission		= ColorRGB32F(0.0f);
	current_hit_emission.emission_flags = 0u;
	if (intersection_found)
	{
		int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		load_effective_emission(render_data, material_index, closest_hit_info.texcoords, current_hit_emission);
	}

#if DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
	ResolvedMaterialUserControlsCache previous_resolved_user_controls;
	previous_resolved_user_controls.validity_mask = 0;
	// Stored light candidates are evaluated at the previous vertex even when the continuation ray misses or hits nonemissive geometry.
	if (nee_deferred_MIS_context.last_primary_gbuffer_path_index == NEE_DEFERRED_INVALID_PATH_INDEX)
		previous_resolved_user_controls = wavefront_load_resolved_material_user_controls(render_data, path_index, previous_resolved_user_controls);

	ray_payload.ray_color += do_deferred_NEE_MIS<PreviousBSDFMaterialType>(render_data, intersection_found, ray_payload, closest_hit_info, current_hit_emission,
																		   nee_deferred_MIS_context, previous_resolved_user_controls, random_number_generator);
#else
	ray_payload.ray_color += do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, closest_hit_info, current_hit_emission, nee_deferred_MIS_context,
												 random_number_generator);
#endif

	int x = static_cast<int>(path_index % render_data.render_settings.render_resolution.x);
	int y = static_cast<int>(path_index / render_data.render_settings.render_resolution.x);
	if (terminal_path)
	{
		wavefront_finalize_path(render_data, path_index, x, y, ray_payload, random_number_generator);
		return;
	}

	ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, path_index);
	ray_payload.next_ray_state = RayState::MISSED;
	wavefront_finalize_path(render_data, path_index, x, y, ray_payload, random_number_generator);
}

#ifdef __KERNELCC__
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) WavefrontCompleteDeferredPaths(bool route_by_family)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline WavefrontCompleteDeferredPaths(HIPRTRenderData render_data, unsigned int queue_slot, bool route_by_family = false)
#endif // #ifdef __KERNELCC__
{
	using KernelMaterial =
		typename EffectiveMaterialFor<static_cast<BSDFModel>(BSDF_MODEL), static_cast<KernelMaterialSpecialization>(KERNEL_MATERIAL_SPECIALIZATION)>::Type;

#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride	 = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int queue_stride = render_data.wavefront_data.path_capacity;
#endif // #ifdef __KERNELCC__

	unsigned int queue_start = 0;
	unsigned int queue_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[WAVEFRONT_COMPLETION_QUEUE_INDEX], 0u);
	if (route_by_family)
	{
		if (render_data.wavefront_data.material_family_routing_enabled == 0)
			return;

		unsigned int material_family = KERNEL_MATERIAL_SPECIALIZATION;
		queue_start					 = render_data.wavefront_data.material_family_offsets[material_family];
		queue_count					 = render_data.wavefront_data.material_family_counts[material_family];
	}
	else if (queue_count > render_data.wavefront_data.path_capacity)
		queue_count = render_data.wavefront_data.path_capacity;

	for (; queue_slot < queue_count; queue_slot += queue_stride)
	{
		unsigned int path_index = route_by_family ? render_data.wavefront_data.material_family_indices[queue_start + queue_slot]
												  : render_data.wavefront_data.path_queues[WAVEFRONT_COMPLETION_QUEUE_INDEX][queue_slot];
		if (path_index >= render_data.wavefront_data.path_capacity)
			continue;

		wavefront_complete_terminated_path<KernelMaterial>(render_data, path_index);
	}
}

#endif // #ifndef KERNELS_WAVEFRONT_COMPLETE_DEFERRED_PATHS_H
