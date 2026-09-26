/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_COMMON_H
#define DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_COMMON_H

#include "Device/includes/AdaptiveSampling.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/Material.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/PathTracingDebugViews.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/SanityCheck.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE void wavefront_store_resolved_material_user_controls(HIPRTRenderData& render_data,
																  unsigned int path_index,
																  const ResolvedMaterialUserControlsCache& resolved_user_controls)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	if (wavefront_data.path_resolved_material_control_validity_masks == nullptr)
		return;

	unsigned int validity_mask												 = resolved_user_controls.validity_mask;
	wavefront_data.path_resolved_material_control_validity_masks[path_index] = validity_mask;
	if (validity_mask & ResolvedMaterialUserControlRoughness)
		wavefront_data.path_resolved_material_roughness[path_index] = resolved_user_controls.roughness;
	if (validity_mask & ResolvedMaterialUserControlMetallic)
		wavefront_data.path_resolved_material_metallic[path_index] = resolved_user_controls.metallic;
	if (validity_mask & ResolvedMaterialUserControlSpecular)
		wavefront_data.path_resolved_material_specular[path_index] = resolved_user_controls.specular;
	if (validity_mask & ResolvedMaterialUserControlCoat)
		wavefront_data.path_resolved_material_coat[path_index] = resolved_user_controls.coat;
	if (validity_mask & ResolvedMaterialUserControlSheen)
		wavefront_data.path_resolved_material_sheen[path_index] = resolved_user_controls.sheen;
	if (validity_mask & ResolvedMaterialUserControlSpecularTransmission)
		wavefront_data.path_resolved_material_specular_transmission[path_index] = resolved_user_controls.specular_transmission;
}

HIPRT_DEVICE ResolvedMaterialUserControlsCache wavefront_load_resolved_material_user_controls(const HIPRTRenderData& render_data,
																							  unsigned int path_index,
																							  const ResolvedMaterialUserControlsCache& fallback_user_controls)
{
	const WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	if (wavefront_data.path_resolved_material_control_validity_masks == nullptr)
		return fallback_user_controls;

	ResolvedMaterialUserControlsCache resolved_user_controls;
	resolved_user_controls.validity_mask = wavefront_data.path_resolved_material_control_validity_masks[path_index];
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness)
		resolved_user_controls.roughness = wavefront_data.path_resolved_material_roughness[path_index];
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlMetallic)
		resolved_user_controls.metallic = wavefront_data.path_resolved_material_metallic[path_index];
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlSpecular)
		resolved_user_controls.specular = wavefront_data.path_resolved_material_specular[path_index];
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlCoat)
		resolved_user_controls.coat = wavefront_data.path_resolved_material_coat[path_index];
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlSheen)
		resolved_user_controls.sheen = wavefront_data.path_resolved_material_sheen[path_index];
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlSpecularTransmission)
		resolved_user_controls.specular_transmission = wavefront_data.path_resolved_material_specular_transmission[path_index];

	return resolved_user_controls;
}

HIPRT_DEVICE void wavefront_load_secondary_material_state(
	HIPRTRenderData& render_data, unsigned int path_index, RayPayloadCommon& ray_payload, HitInfo& closest_hit_info, bool& intersection_found)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	intersection_found		 = (wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_INTERSECTION_FOUND) != 0;
	ray_payload.volume_state = wavefront_data.path_volume_states[path_index];

	if (intersection_found)
	{
		HitInfo& stored_closest_hit_info = wavefront_data.path_closest_hit_infos[path_index];
		closest_hit_info.primitive_index = stored_closest_hit_info.primitive_index;
		closest_hit_info.texcoords		 = stored_closest_hit_info.texcoords;
		closest_hit_info.t				 = stored_closest_hit_info.t;
	}
}

HIPRT_DEVICE void wavefront_load_secondary_shading_state(
	HIPRTRenderData& render_data, unsigned int path_index, RayPayloadCommon& ray_payload, hiprtRay& ray, HitInfo& closest_hit_info)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	ray_payload.throughput			  = wavefront_data.path_throughputs[path_index];
	ray_payload.ray_color			  = wavefront_data.path_ray_colors[path_index];
	ray_payload.next_ray_state		  = RayState::BOUNCE;
	ray_payload.bounce				  = wavefront_data.path_bounces[path_index];
	ray_payload.accumulated_roughness = wavefront_data.path_accumulated_roughnesses[path_index];

	HitInfo& stored_closest_hit_info	 = wavefront_data.path_closest_hit_infos[path_index];
	closest_hit_info.inter_point		 = stored_closest_hit_info.inter_point;
	closest_hit_info.shading_normal		 = stored_closest_hit_info.shading_normal;
	closest_hit_info.geometric_normal	 = stored_closest_hit_info.geometric_normal;
	closest_hit_info.geometry_backfacing = stored_closest_hit_info.geometry_backfacing;

	ray.origin	  = closest_hit_info.inter_point;
	ray.direction = wavefront_data.path_ray_directions[path_index].unpack();
}

HIPRT_DEVICE void wavefront_store_trace_result(HIPRTRenderData& render_data,
											   unsigned int path_index,
											   const RayVolumeState& volume_state,
											   const HitInfo& closest_hit_info,
											   bool intersection_found,
											   unsigned int rng_seed)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	wavefront_data.path_volume_states[path_index]	  = volume_state;
	wavefront_data.path_closest_hit_infos[path_index] = closest_hit_info;
	unsigned int terminal_flag						  = wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_TERMINAL;
	wavefront_data.path_state_flags[path_index]		  = terminal_flag | (intersection_found ? WAVEFRONT_PATH_STATE_INTERSECTION_FOUND : 0u);
	wavefront_data.path_rng_states[path_index]		  = rng_seed;
}

// Queue 1 contains rays that need traversal, including terminal rays retained for deferred
// MIS completion. Traversal replaces the material and hit attributes, so only the origin and
// previous primitive are needed from the shaded surface. Previous surface data needed by
// deferred MIS remains in its separately persisted context.
HIPRT_DEVICE void wavefront_store_trace_ray(HIPRTRenderData& render_data,
											unsigned int path_index,
											const RayPayloadCommon& ray_payload,
											const hiprtRay& ray,
											const HitInfo& closest_hit_info,
											bool is_terminal_path)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	wavefront_data.path_throughputs[path_index]				= ray_payload.throughput;
	wavefront_data.path_ray_colors[path_index]				= ray_payload.ray_color;
	wavefront_data.path_bounces[path_index]					= ray_payload.bounce;
	wavefront_data.path_accumulated_roughnesses[path_index] = ray_payload.accumulated_roughness;
	wavefront_data.path_volume_states[path_index]			= ray_payload.volume_state;
	wavefront_data.path_state_flags[path_index]				= is_terminal_path ? WAVEFRONT_PATH_STATE_TERMINAL : 0u;

	wavefront_data.path_closest_hit_infos[path_index].inter_point	  = closest_hit_info.inter_point;
	wavefront_data.path_closest_hit_infos[path_index].primitive_index = closest_hit_info.primitive_index;
	wavefront_data.path_ray_directions[path_index]					  = Octahedral24BitNormalPadded32b(ray.direction);
}

HIPRT_DEVICE void wavefront_load_trace_ray(
	HIPRTRenderData& render_data, unsigned int path_index, WavefrontTracePayload& trace_payload, hiprtRay& ray, HitInfo& closest_hit_info)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	trace_payload.volume_state		 = wavefront_data.path_volume_states[path_index];
	closest_hit_info.inter_point	 = wavefront_data.path_closest_hit_infos[path_index].inter_point;
	closest_hit_info.primitive_index = wavefront_data.path_closest_hit_infos[path_index].primitive_index;
	ray.origin						 = closest_hit_info.inter_point;
	ray.direction					 = wavefront_data.path_ray_directions[path_index].unpack();
}

// Restore contribution bookkeeping only after traversal, without overwriting its new material
// or its volume-state updates (including skipped nested-dielectric boundaries).
HIPRT_DEVICE void wavefront_store_nee_deferred_mis_context(HIPRTRenderData& render_data,
														   unsigned int path_index,
														   const NEEDeferredMISContext& nee_deferred_MIS_context)
{
	NEEDeferredMISContext* contexts = reinterpret_cast<NEEDeferredMISContext*>(render_data.wavefront_data.path_nee_deferred_mis_contexts);
	contexts[path_index]			= nee_deferred_MIS_context;
}

HIPRT_DEVICE void wavefront_load_nee_deferred_mis_context(HIPRTRenderData& render_data,
														  unsigned int path_index,
														  NEEDeferredMISContext& nee_deferred_MIS_context)
{
	NEEDeferredMISContext* contexts = reinterpret_cast<NEEDeferredMISContext*>(render_data.wavefront_data.path_nee_deferred_mis_contexts);
	nee_deferred_MIS_context		= contexts[path_index];
}

HIPRT_DEVICE void wavefront_initialize_path(
	HIPRTRenderData& render_data, unsigned int path_index, RayPayloadCommon& ray_payload, hiprtRay& ray, HitInfo& closest_hit_info, bool& intersection_found)
{
	closest_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[path_index];
	closest_hit_info.geometric_normal = hippt::normalize(render_data.g_buffer.geometric_normals[path_index].unpack());
	closest_hit_info.shading_normal	  = hippt::normalize(render_data.g_buffer.shading_normals[path_index].unpack());
	closest_hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[path_index];

	ray.origin	  = closest_hit_info.inter_point;
	ray.direction = hippt::normalize(-render_data.g_buffer.get_view_direction(render_data.current_camera.position, path_index));

	ray_payload.next_ray_state = RayState::BOUNCE;

	// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
	// the ray volume state after the camera hit is just an empty interior stack but with
	// the material index that we hit pushed onto the stack. That's it. Because it is that
	// simple, we don't have the ray volume state in the GBuffer but rather we can
	intersection_found = closest_hit_info.primitive_index != -1;

	// Preserve the direction rounding of the former Initialize -> store -> Shade load boundary.
	ray.direction = Octahedral24BitNormalPadded32b(ray.direction).unpack();
}

HIPRT_DEVICE void wavefront_enqueue_path(HIPRTRenderData& render_data, unsigned int queue_index, unsigned int path_index)
{
	unsigned int output_index = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[queue_index], 1u);
	if (output_index >= render_data.wavefront_data.path_capacity)
		return;

	render_data.wavefront_data.path_queues[queue_index][output_index] = path_index;
}

HIPRT_DEVICE void wavefront_queue_path_for_tracing(HIPRTRenderData& render_data,
												   unsigned int path_index,
												   const RayPayloadCommon& ray_payload,
												   const hiprtRay& ray,
												   const HitInfo& closest_hit_info,
												   const NEEDeferredMISContext& nee_deferred_MIS_context,
												   const Xorshift32Generator& random_number_generator,
												   bool is_terminal_path)
{
	wavefront_store_trace_ray(render_data, path_index, ray_payload, ray, closest_hit_info, is_terminal_path);
	wavefront_store_nee_deferred_mis_context(render_data, path_index, nee_deferred_MIS_context);
	render_data.wavefront_data.path_rng_states[path_index] = random_number_generator.m_state.seed;
	wavefront_enqueue_path(render_data, 1, path_index);
}

HIPRT_DEVICE void wavefront_finalize_path(
	HIPRTRenderData& render_data, unsigned int pixel_index, int x, int y, RayPayloadCommon& ray_payload, Xorshift32Generator& random_number_generator)
{
	render_data.wavefront_data.path_state_flags[pixel_index] |= WAVEFRONT_PATH_STATE_TERMINAL;

	render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);

	// Checking for NaNs / negative value samples. Output
	if (!sanity_check<true>(render_data, ray_payload.ray_color, x, y))
		return;

	ColorRGB32F debug_color;
	path_tracing_compute_debug_view_debug_color(render_data, ray_payload, pixel_index, random_number_generator, debug_color);

	// If we got here, this means that we still have at least one ray active
	// This is a concurrent write by the way but we don't really care, everyone is writing
	// the same value
	render_data.aux_buffers.still_one_ray_active[0] = 1;

	path_tracing_accumulate_color(render_data, pixel_index, ray_payload.ray_color, debug_color);
}

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_COMMON_H
