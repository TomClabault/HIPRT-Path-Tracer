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

HIPRT_DEVICE void wavefront_load_path(
	HIPRTRenderData& render_data, unsigned int path_index, RayPayload& ray_payload, hiprtRay& ray, HitInfo& closest_hit_info, bool& intersection_found)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	ray_payload.throughput			  = wavefront_data.path_throughputs[path_index];
	ray_payload.ray_color			  = wavefront_data.path_ray_colors[path_index];
	ray_payload.next_ray_state		  = static_cast<RayState>(wavefront_data.path_next_ray_states[path_index]);
	ray_payload.bounce				  = wavefront_data.path_bounces[path_index];
	ray_payload.accumulated_roughness = wavefront_data.path_accumulated_roughnesses[path_index];
	ray_payload.material			  = wavefront_data.path_materials[path_index];
	ray_payload.volume_state		  = wavefront_data.path_volume_states[path_index];

	closest_hit_info   = wavefront_data.path_closest_hit_infos[path_index];
	intersection_found = wavefront_data.path_intersections_found[path_index] != 0;
	ray.origin		   = closest_hit_info.inter_point;
	ray.direction	   = wavefront_data.path_ray_directions[path_index].unpack();
}

HIPRT_DEVICE void wavefront_store_path(HIPRTRenderData& render_data,
									   unsigned int path_index,
									   const RayPayload& ray_payload,
									   const hiprtRay& ray,
									   const HitInfo& closest_hit_info,
									   bool intersection_found)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	wavefront_data.path_throughputs[path_index]				= ray_payload.throughput;
	wavefront_data.path_ray_colors[path_index]				= ray_payload.ray_color;
	wavefront_data.path_next_ray_states[path_index]			= static_cast<unsigned int>(ray_payload.next_ray_state);
	wavefront_data.path_bounces[path_index]					= ray_payload.bounce;
	wavefront_data.path_accumulated_roughnesses[path_index] = ray_payload.accumulated_roughness;
	wavefront_data.path_materials[path_index]				= ray_payload.material;
	wavefront_data.path_volume_states[path_index]			= ray_payload.volume_state;
	wavefront_data.path_closest_hit_infos[path_index]		= closest_hit_info;
	wavefront_data.path_ray_directions[path_index]			= Octahedral24BitNormalPadded32b(ray.direction);
	wavefront_data.path_intersections_found[path_index]		= intersection_found ? 1u : 0u;
}

HIPRT_DEVICE bool wavefront_initialize_path(HIPRTRenderData& render_data, unsigned int path_index)
{
	if (!render_data.aux_buffers.pixel_active[path_index])
		return false;

	RayPayload ray_payload;

	HitInfo closest_hit_info;
	closest_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[path_index];
	closest_hit_info.geometric_normal = hippt::normalize(render_data.g_buffer.geometric_normals[path_index].unpack());
	closest_hit_info.shading_normal	  = hippt::normalize(render_data.g_buffer.shading_normals[path_index].unpack());
	closest_hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[path_index];

	hiprtRay ray;
	ray.origin	  = render_data.current_camera.position;
	ray.direction = hippt::normalize(-render_data.g_buffer.get_view_direction(render_data.current_camera.position, path_index));

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(path_index));
	ray_payload.next_ray_state = RayState::BOUNCE;
	ray_payload.material	   = render_data.g_buffer.materials[path_index].unpack();

	// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
	// the ray volume state after the camera hit is just an empty interior stack but with
	// the material index that we hit pushed onto the stack. That's it. Because it is that
	// simple, we don't have the ray volume state in the GBuffer but rather we can
	// reconstruct the ray volume state on the fly
	ray_payload.volume_state.reconstruct_first_hit(ray_payload.material, render_data.buffers.material_indices, closest_hit_info.primitive_index,
												   random_number_generator);

	bool intersection_found = closest_hit_info.primitive_index != -1;
	wavefront_store_path(render_data, path_index, ray_payload, ray, closest_hit_info, intersection_found);
	render_data.wavefront_data.path_rng_states[path_index] = random_number_generator.m_state.seed;

	return true;
}

HIPRT_DEVICE void wavefront_enqueue_path(HIPRTRenderData& render_data, unsigned int queue_index, unsigned int path_index)
{
	unsigned int output_index = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[queue_index], 1u);
	if (output_index >= render_data.wavefront_data.path_capacity)
		return;

	render_data.wavefront_data.path_queues[queue_index][output_index] = path_index;
}

HIPRT_DEVICE void wavefront_finalize_path_with_context(HIPRTRenderData& render_data,
													   unsigned int pixel_index,
													   int x,
													   int y,
													   RayPayload& ray_payload,
													   hiprtRay& ray,
													   HitInfo& closest_hit_info,
													   Xorshift32Generator& random_number_generator,
													   NEEDeferredMISContext& nee_deferred_MIS_context,
													   bool increment_bounce_before_last_deferred_nee)
{
	if (increment_bounce_before_last_deferred_nee)
		ray_payload.bounce++;

	// We do one last intersection after the last bounce to get a BSDF sample for NEE MIS
	ray_payload.ray_color += do_last_deferred_NEE_MIS(render_data, ray, ray_payload, closest_hit_info, random_number_generator, nee_deferred_MIS_context);

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
