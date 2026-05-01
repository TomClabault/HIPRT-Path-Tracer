/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_PT_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_PT_INITIAL_CANDIDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/ReSTIR/GI_PT/InitialCandidatesUtils.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/ReGIR/Representative.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE void ReSTIRPT_sample_point_fill(const HIPRTRenderData& render_data,
											 const RayPayload& ray_payload,
											 const HitInfo& closest_hit_info,
											 ReSTIRPTReservoirSample& restir_pt_initial_sample)
{
	restir_pt_initial_sample.sample_point = closest_hit_info.inter_point;
	restir_pt_initial_sample.sample_point_geometric_normal.pack(closest_hit_info.geometric_normal);
	restir_pt_initial_sample.sample_point_shading_normal.pack(closest_hit_info.shading_normal);
	restir_pt_initial_sample.sample_point_material		  = ray_payload.material;
	restir_pt_initial_sample.sample_point_primitive_index = closest_hit_info.primitive_index;
	restir_pt_initial_sample.sample_point_rough_enough =
		ray_payload.material.can_do_light_sampling(render_data.render_settings.restir_pt_settings.neighbor_sample_point_roughness_threshold);
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_InitialCandidates(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_InitialCandidates(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

	if (!render_data.aux_buffers.pixel_active[pixel_index])
		// Pixel isn't active because of adaptive sampling or render resolution scaling
		return;

	if (render_data.render_settings.do_render_low_resolution())
		// Reducing the number of bounces to 3 if rendering at low resolution
		// for better interactivity
		render_data.render_settings.nb_bounces = hippt::min(3, render_data.render_settings.nb_bounces);

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
	// Resetting splatting samples
	for (int bounce = 0; bounce < render_data.render_settings.nb_bounces; bounce++)
		render_data.render_settings.restir_pg_settings.invalidate_splatting_sample(render_data.render_settings.render_resolution, x, y, bounce);
#endif

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(pixel_index));

	// Initializing the closest hit info the information from the camera ray pass
	HitInfo closest_hit_info;
	closest_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[pixel_index];
	closest_hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	closest_hit_info.shading_normal	  = render_data.g_buffer.shading_normals[pixel_index].unpack();
	closest_hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[pixel_index];

	// Initializing the ray with the information from the camera ray pass
	hiprtRay ray;
	ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

	RayPayload ray_payload;
	ray_payload.next_ray_state = RayState::BOUNCE;
	ray_payload.material	   = render_data.g_buffer.materials[pixel_index].unpack();

	// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
	// the ray volume state after the camera hit is just an empty interior stack but with
	// the material index that we hit pushed onto the stack. That's it. Because it is that
	// simple, we don't have the ray volume state in the GBuffer but rather we can
	// reconstruct the ray volume state on the fly
	ray_payload.volume_state.reconstruct_first_hit(ray_payload.material, render_data.buffers.material_indices, closest_hit_info.primitive_index,
												   random_number_generator);

	bool intersection_found = closest_hit_info.primitive_index != -1;

	// TODO re-read this at the end instread of storing it at the beginning for registers?
	ReSTIRSurface initial_surface;
	initial_surface.geometric_normal = closest_hit_info.geometric_normal;
	initial_surface.shading_normal	 = closest_hit_info.shading_normal;
	initial_surface.primitive_index	 = closest_hit_info.primitive_index;
	initial_surface.material		 = ray_payload.material;
	initial_surface.ray_volume_state = ray_payload.volume_state;
	initial_surface.shading_point	 = closest_hit_info.inter_point;
	initial_surface.view_direction	 = -ray.direction;

	ReSTIRPTReservoirSample restir_pt_initial_sample;
	restir_pt_initial_sample.pixel_index = pixel_index;
	ReSTIRPTReservoir restir_pt_initial_reservoir;

	ColorRGB32F path_unweighted_throughput				   = ColorRGB32F(1.0f);
	ColorRGB32F path_unweighted_throughput_to_sample_point = ColorRGB32F(1.0f);
	// BSDF_visible_point * cos_theta_visible_point
	ColorRGB32F first_bsdf_throughput = ColorRGB32F(1.0f);

	// + 1 to nb_bounces here because we want "0" bounces to still act as one
	// hit and to return some color
	for (int& bounce = ray_payload.bounce; bounce < render_data.render_settings.nb_bounces + 1; bounce++)
	{
		if (ray_payload.next_ray_state != RayState::MISSED)
		{
			if (bounce > 0)
			{
				if (bounce == 1)
					// This is going to be tracing the ray from the visible point to the sample:
					// we're saving the random seed used during the BVH traversal to be able to reproduce
					// alpha tests
					restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;

				intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);
			}

			if (intersection_found)
			{
				if (bounce == 0)
					store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);

				if (bounce > 0)
				{
					ReGIR_representative_points_update(render_data, ray_payload, closest_hit_info);
					if (bounce == 1)
						ReSTIRPT_sample_point_fill(render_data, ray_payload, closest_hit_info, restir_pt_initial_sample);

					/**
					 * Next-event estimation
					 */
					// Estimating with a throughput of 1.0f here because we're going to apply the throughput ourselves
					ColorRGB32F direct_lighting_estimation =
						estimate_direct_lighting(render_data, ray_payload, ColorRGB32F(1.0f), closest_hit_info, -ray.direction, x, y, random_number_generator);

					restir_pt_initial_sample.unweighted_throughput_to_visible_point = path_unweighted_throughput / first_bsdf_throughput;
					restir_pt_initial_sample.unweighted_throughput_to_sample_point	= path_unweighted_throughput_to_sample_point;
					restir_pt_initial_sample.path_radiance							= direct_lighting_estimation;
					restir_pt_initial_sample.target_function						= (path_unweighted_throughput * direct_lighting_estimation).luminance();
					restir_pt_initial_sample.x3_is_NEE								= bounce == 1;
					restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, (ray_payload.throughput * direct_lighting_estimation).luminance(),
																  random_number_generator);
				}

				float bsdf_pdf;
				BSDFIncidentLightInfo incident_light_info;
				bool valid_indirect_bounce =
					restir_pt_compute_next_indirect_bounce(render_data, ray_payload, path_unweighted_throughput, path_unweighted_throughput_to_sample_point,
														   closest_hit_info, -ray.direction, ray, random_number_generator, incident_light_info, &bsdf_pdf);

				if (!valid_indirect_bounce)
					// Bad BSDF sample (under the surface), killed by russian roulette, ...
					break;

				if (bounce == 0)
				{
					restir_pt_initial_sample.incident_light_info_at_visible_point = incident_light_info;
					first_bsdf_throughput										  = path_unweighted_throughput;
				}
				else if (bounce == 1)
				{
					restir_pt_initial_sample.incident_light_info_at_sample_point   = incident_light_info;
					restir_pt_initial_sample.sample_point_incident_light_direction = ray.direction;
				}

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
				// Not the last bounce
				if (bounce != render_data.render_settings.nb_bounces)
				{
					ReSTIRPGSplattingSample sample;
					sample.position			  = closest_hit_info.inter_point;
					sample.normal			  = closest_hit_info.geometric_normal;
					sample.incident_direction = ray.direction;

					render_data.render_settings.restir_pg_settings.splatting_samples_soa.store_sample(sample, render_data.render_settings.render_resolution, x,
																									  y, bounce);
				}
#endif
			}
			else
			{
				if (bounce == 1)
				{
					// For envmap path, the direction is stored in the hit point
					restir_pt_initial_sample.sample_point = ray.direction;
					// -1 for the primitive index indicates that this is an envmap sample
					restir_pt_initial_sample.sample_point_primitive_index = -1;
				}

				ColorRGB32F envmap_emission = path_tracing_miss_gather_envmap(render_data, ColorRGB32F(1.0f), ray.direction, ray_payload.bounce, pixel_index);

				// This unweighted throughput to visible point is the full unweighted throughput but without the first BSDF contribution
				restir_pt_initial_sample.unweighted_throughput_to_visible_point = path_unweighted_throughput / first_bsdf_throughput;
				restir_pt_initial_sample.unweighted_throughput_to_sample_point	= path_unweighted_throughput_to_sample_point;
				restir_pt_initial_sample.path_radiance							= envmap_emission;
				restir_pt_initial_sample.target_function						= (path_unweighted_throughput * envmap_emission).luminance();
				restir_pt_initial_sample.x3_is_NEE								= false;
				restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, (ray_payload.throughput * envmap_emission).luminance(),
															  random_number_generator);

				ray_payload.next_ray_state = RayState::MISSED;
			}
		}
		else if (ray_payload.next_ray_state == RayState::MISSED)
			break;
	}

	render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);

	// If we got here, this means that we still have at least one ray active
	// This is a concurrent write by the way but we don't really care, everyone is writing
	// the same value
	render_data.aux_buffers.still_one_ray_active[0] = 1;

	restir_pt_initial_reservoir.end();
	restir_pt_initial_reservoir.sanity_check(make_int2(x, y));

	render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer[pixel_index] = restir_pt_initial_reservoir;

	if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_SHADE_ONLY_INITIAL_CANDIDATES &&
		ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled)
	{
		float3_t to_light_direction = restir_pt_initial_reservoir.sample.is_envmap_path()
										  ? restir_pt_initial_sample.sample_point
										  : hippt::normalize(restir_pt_initial_reservoir.sample.sample_point - initial_surface.shading_point);
		BSDFContext first_hit_eval_context(initial_surface.view_direction, initial_surface.shading_normal, initial_surface.geometric_normal, to_light_direction,
										   restir_pt_initial_reservoir.sample.incident_light_info_at_visible_point, initial_surface.ray_volume_state, false,
										   initial_surface.material, 0.0f);

		float trash_pdf;
		ColorRGB32F bsdf_first_hit = bsdf_dispatcher_eval(render_data, first_hit_eval_context, trash_pdf, random_number_generator) *
									 hippt::abs(hippt::dot(initial_surface.shading_normal, to_light_direction));

		ColorRGB32F radiance_to_camera;
		if (restir_pt_initial_reservoir.sample.is_envmap_path())
		{
			radiance_to_camera = bsdf_first_hit * hippt::abs(hippt::dot(initial_surface.shading_normal, to_light_direction)) *
								 restir_pt_initial_reservoir.sample.path_radiance * restir_pt_initial_reservoir.UCW;
		}
		else
		{
			// TODO the ray volume state should be updated here
			float3_t view_direction					 = hippt::normalize(initial_surface.shading_point - restir_pt_initial_reservoir.sample.sample_point);
			float3_t to_light_direction_sample_point = restir_pt_initial_reservoir.sample.sample_point_incident_light_direction;
			BSDFContext secondary_hit_eval_context(view_direction, initial_surface.shading_normal, initial_surface.geometric_normal,
												   to_light_direction_sample_point, restir_pt_initial_reservoir.sample.incident_light_info_at_sample_point,
												   initial_surface.ray_volume_state, false, restir_pt_initial_reservoir.sample.sample_point_material, 0.0f);

			ColorRGB32F bsdf_secondary_hit =
				bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator) *
				hippt::abs(hippt::dot(restir_pt_initial_reservoir.sample.sample_point_shading_normal.unpack(), to_light_direction_sample_point));
			radiance_to_camera = bsdf_first_hit * bsdf_secondary_hit * restir_pt_initial_reservoir.sample.path_radiance * restir_pt_initial_reservoir.UCW;
		}

		render_data.buffers.accumulated_ray_colors[pixel_index] = radiance_to_camera * restir_pt_initial_reservoir.UCW;
	}
}

#endif
