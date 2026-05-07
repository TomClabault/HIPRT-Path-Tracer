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
#include "Device/includes/ReSTIR/PT/InitialCandidatesUtils.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/ReGIR/Representative.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE void ReSTIR_PT_stream_NEE(HIPRTRenderData& render_data,
									   ReSTIRSurface& initial_surface,
									   float3_t view_direction,
									   RayPayload& ray_payload,
									   ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
									   ColorRGB32F path_unweighted_throughput_after_rc_vertex,
									   ReSTIRPTReservoir& restir_pt_initial_reservoir,
									   ReSTIRPTReservoirSample& restir_pt_initial_sample,
									   HitInfo& closest_hit_info,
									   NEEDeferredMISContext& nee_deferred_MIS_context,
									   Xorshift32Generator& random_number_generator,
									   int x,
									   int y)
{
	if (ray_payload.bounce == 0 && !render_data.render_settings.enable_direct_lighting)
		return;

	int nb_light_candidates = render_data.render_settings.restir_pt_settings.initial_candidates.nee_ris_number_of_light_candidates;
	int nb_bsdf_candidates	= render_data.render_settings.restir_pt_settings.initial_candidates.nee_ris_number_of_bsdf_candidates;
	for (int light_candidate = 0; light_candidate < nb_light_candidates; light_candidate++)
	{
		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
			sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
									  closest_hit_info.geometric_normal, closest_hit_info.primitive_index, ray_payload, random_number_generator);

		for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
		{
			LightSamplePointInformation& light_sample = light_samples[i];

			float nee_connection_pdf_solid_angle =
				area_to_solid_angle_pdf(light_sample.area_measure_pdf, hippt::length(light_sample.point_on_light - closest_hit_info.inter_point),
										compute_cosine_term_at_light_source(light_sample.light_source_normal,
																			hippt::normalize(closest_hit_info.inter_point - light_sample.point_on_light)));

			if (nee_connection_pdf_solid_angle <= 0.0f)
				continue;

			float3_t shadow_ray_origin				 = closest_hit_info.inter_point;
			float3_t shadow_ray_direction			 = light_sample.point_on_light - shadow_ray_origin;
			float distance_to_light					 = hippt::length(shadow_ray_direction);
			float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

			hiprtRay shadow_ray;
			shadow_ray.origin	 = shadow_ray_origin;
			shadow_ray.direction = shadow_ray_direction_normalized;

			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point	 = shadow_ray_origin;
			bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
															   nee_plus_plus_context, random_number_generator, ray_payload.bounce);

			if (in_shadow)
				continue;

			BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
			BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray_direction_normalized,
									 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
									 MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);

			float bsdf_pdf;
			ColorRGB32F bsdf_color		= bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
			ColorRGB32F bsdf_throughput = bsdf_color * hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray_direction_normalized));

			if (ray_payload.bounce == 0)
				ReSTIR_PT_rc_di_vertex_fill_information(light_sample.point_on_light, light_sample.light_source_normal,
														light_sample.emissive_triangle_global_index, BSDFIncidentLightInfo::NO_INFO, restir_pt_initial_sample);
			if (ray_payload.bounce == 1)
				restir_pt_initial_sample.incident_light_info_at_sample_point = BSDFIncidentLightInfo::NO_INFO;
			if (ray_payload.bounce <= 1)
				restir_pt_initial_sample.rc_vertex_incident_light_direction = shadow_ray_direction_normalized;
			restir_pt_initial_sample.di_sample					 = ray_payload.bounce == 0;
			restir_pt_initial_sample.rc_vertex_incident_radiance = light_sample.emission * path_unweighted_throughput_after_rc_vertex;
			if (ray_payload.bounce >= 2)
				restir_pt_initial_sample.rc_vertex_incident_radiance *= bsdf_throughput;
			restir_pt_initial_sample.target_function =
				(path_unweighted_throughput_up_to_rc_vertex * path_unweighted_throughput_after_rc_vertex * bsdf_throughput * light_sample.emission).luminance();
			constexpr float multi_light_sample_mis_weight = 1.0f / DirectLightIntegrationFactor<DirectLightSamplingStrategy>();
			float nee_mis_weight =
				balance_heuristic(nee_connection_pdf_solid_angle, nb_light_candidates * DirectLightIntegrationFactor<DirectLightSamplingStrategy>(), bsdf_pdf,
								  nb_bsdf_candidates);
			float mis_weight = multi_light_sample_mis_weight * nee_mis_weight;
			float weight	 = mis_weight * (ray_payload.throughput * bsdf_throughput / nee_connection_pdf_solid_angle * light_sample.emission).luminance();

			restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, weight, random_number_generator);
			restir_pt_initial_reservoir.sanity_check(make_int2(x, y));
		}
	}

	// NB BSDF candidates - 1 here because we're already doing 1 candidate thanks to the bounce of the main path
	for (int bsdf_candidate = 0; bsdf_candidate < nb_bsdf_candidates - 1; bsdf_candidate++)
	{
		float bsdf_sample_pdf;
		float3_t sampled_bsdf_direction;
		BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

		BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f),
								 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
								 MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
		ColorRGB32F bsdf_color		= bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);
		ColorRGB32F bsdf_throughput = bsdf_color * hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_bsdf_direction));

		if (bsdf_sample_pdf > 0.0f)
		{
			hiprtRay new_ray;
			new_ray.origin	  = closest_hit_info.inter_point;
			new_ray.direction = sampled_bsdf_direction;

			BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
			bool intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index,
																	 ray_payload.bounce, random_number_generator);

			// Checking that we did hit something and if we hit something,
			// it needs to be emissive
			if (!intersection_found || shadow_light_ray_hit_info.hit_emission.is_black() ||
				compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction) <= 0.0f)
				continue;

			float3_t point_on_light = closest_hit_info.inter_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;
			if (ray_payload.bounce == 0)
				ReSTIR_PT_rc_di_vertex_fill_information(point_on_light, shadow_light_ray_hit_info.hit_geometric_normal,
														shadow_light_ray_hit_info.hit_prim_index, incident_light_info, restir_pt_initial_sample);
			if (ray_payload.bounce == 1)
				restir_pt_initial_sample.incident_light_info_at_sample_point = incident_light_info;
			if (ray_payload.bounce <= 1)
				restir_pt_initial_sample.rc_vertex_incident_light_direction = sampled_bsdf_direction;
			restir_pt_initial_sample.di_sample					 = ray_payload.bounce == 0;
			restir_pt_initial_sample.rc_vertex_incident_radiance = shadow_light_ray_hit_info.hit_emission * path_unweighted_throughput_after_rc_vertex;
			if (ray_payload.bounce >= 2)
				restir_pt_initial_sample.rc_vertex_incident_radiance *= bsdf_throughput;
			restir_pt_initial_sample.target_function = (path_unweighted_throughput_up_to_rc_vertex * path_unweighted_throughput_after_rc_vertex *
														bsdf_throughput * shadow_light_ray_hit_info.hit_emission)
														   .luminance();

			float light_sampler_solid_angle_pdf =
				pdf_of_emissive_triangle_hit_solid_angle(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
														 ray_payload.material, shadow_light_ray_hit_info, sampled_bsdf_direction);
			float nee_mis_weight = balance_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, light_sampler_solid_angle_pdf,
													 nb_light_candidates * DirectLightIntegrationFactor<DirectLightSamplingStrategy>());
			float weight = nee_mis_weight * (ray_payload.throughput * bsdf_throughput / bsdf_sample_pdf * shadow_light_ray_hit_info.hit_emission).luminance();

			restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, weight, random_number_generator);
			restir_pt_initial_reservoir.sanity_check(make_int2(x, y));
		}
	}
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
	HitInfo primary_surface_hit_info;
	primary_surface_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[pixel_index];
	primary_surface_hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	primary_surface_hit_info.shading_normal	  = render_data.g_buffer.shading_normals[pixel_index].unpack();
	primary_surface_hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[pixel_index];

	// Initializing the ray with the information from the camera ray pass
	hiprtRay initial_ray;
	initial_ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

	RayPayload initial_ray_payload;
	initial_ray_payload.next_ray_state = RayState::BOUNCE;
	initial_ray_payload.material	   = render_data.g_buffer.materials[pixel_index].unpack();

	// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
	// the ray volume state after the camera hit is just an empty interior stack but with
	// the material index that we hit pushed onto the stack. That's it. Because it is that
	// simple, we don't have the ray volume state in the GBuffer but rather we can
	// reconstruct the ray volume state on the fly
	initial_ray_payload.volume_state.reconstruct_first_hit(initial_ray_payload.material, render_data.buffers.material_indices,
														   primary_surface_hit_info.primitive_index, random_number_generator);

	bool intersection_found = primary_surface_hit_info.primitive_index != -1;

	// TODO re-read this at the end instread of storing it at the beginning for registers?
	ReSTIRSurface initial_surface;
	initial_surface.geometric_normal = primary_surface_hit_info.geometric_normal;
	initial_surface.shading_normal	 = primary_surface_hit_info.shading_normal;
	initial_surface.primitive_index	 = primary_surface_hit_info.primitive_index;
	initial_surface.material		 = initial_ray_payload.material;
	initial_surface.ray_volume_state = initial_ray_payload.volume_state;
	initial_surface.shading_point	 = primary_surface_hit_info.inter_point;
	initial_surface.view_direction	 = -initial_ray.direction;

	ReSTIRPTReservoir restir_pt_initial_reservoir;

	for (int candidate = 0; candidate < render_data.render_settings.restir_pt_settings.initial_candidates.initial_path_trees_count; candidate++)
	{
		HitInfo closest_hit_info = primary_surface_hit_info;
		bool intersection_found	 = closest_hit_info.primitive_index != -1;

		hiprtRay ray		   = initial_ray;
		RayPayload ray_payload = initial_ray_payload;

		ReSTIRPTReservoirSample restir_pt_initial_sample;
		restir_pt_initial_sample.pixel_index = pixel_index;

		bool reconnection_vertex_chosen = false;
		// Bounce throughput from rc_vertex to the next vertex included
		ColorRGB32F path_unweighted_throughput_up_to_rc_vertex = ColorRGB32F(1.0f);
		ColorRGB32F path_unweighted_throughput_after_rc_vertex = ColorRGB32F(1.0f);

		ColorRGB32F path_unweighted_throughput_up_to_rc_vertex_for_deferred_nee = ColorRGB32F(1.0f);
		ColorRGB32F path_unweighted_throughput_after_rc_vertex_for_deferred_nee = ColorRGB32F(1.0f);

		// + 1 to nb_bounces here because we want "0" bounces to still act as one
		// hit and to return some color
		NEEDeferredMISContext nee_deferred_MIS_context;
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

					intersection_found =
						path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);

					ReSTIR_PT_do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, path_unweighted_throughput_up_to_rc_vertex_for_deferred_nee,
												  path_unweighted_throughput_after_rc_vertex_for_deferred_nee, restir_pt_initial_reservoir,
												  restir_pt_initial_sample, closest_hit_info, nee_deferred_MIS_context, random_number_generator);

					path_unweighted_throughput_up_to_rc_vertex_for_deferred_nee = path_unweighted_throughput_up_to_rc_vertex;
					path_unweighted_throughput_after_rc_vertex_for_deferred_nee = path_unweighted_throughput_after_rc_vertex;
				}

				if (intersection_found)
				{
					if (bounce == 0)
						store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);
					if (bounce > 0)
					{
						ReGIR_representative_points_update(render_data, ray_payload, closest_hit_info);
						if (bounce == 1)
							ReSTIR_PT_rc_vertex_fill_information(render_data, ray_payload, closest_hit_info, restir_pt_initial_sample);
					}

					ReSTIR_PT_stream_NEE(render_data, initial_surface, -ray.direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
										 path_unweighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample, closest_hit_info,
										 nee_deferred_MIS_context, random_number_generator, x, y);

					float bsdf_pdf;
					BSDFIncidentLightInfo incident_light_info;
					ColorRGB32F this_bounce_unweighted_throughput =
						ReSTIR_PT_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray, random_number_generator,
															   incident_light_info, bsdf_pdf, nee_deferred_MIS_context);

					if (this_bounce_unweighted_throughput == ReSTIR_PT_invalid_throughput)
					{
						// Bad BSDF sample (under the surface), killed by russian roulette, ...
						bounce++;
						break;
					}

					if (reconnection_vertex_chosen)
						path_unweighted_throughput_after_rc_vertex *= this_bounce_unweighted_throughput;
					else
						path_unweighted_throughput_up_to_rc_vertex *= this_bounce_unweighted_throughput;

					if (bounce == 0)
						restir_pt_initial_sample.incident_light_info_at_visible_point = incident_light_info;
					else if (bounce == 1)
					{
						restir_pt_initial_sample.incident_light_info_at_sample_point = incident_light_info;
						// TODO remove this line
						restir_pt_initial_sample.rc_vertex_incident_light_direction = ray.direction;

						// Hardcoded to bounce 2 for a simple reconnection shift at the first indirect vertex
						reconnection_vertex_chosen = true;
					}

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
					// Not the last bounce
					if (bounce != render_data.render_settings.nb_bounces)
					{
						ReSTIRPGSplattingSample sample;
						sample.position			  = closest_hit_info.inter_point;
						sample.normal			  = closest_hit_info.geometric_normal;
						sample.incident_direction = ray.direction;

						render_data.render_settings.restir_pg_settings.splatting_samples_soa.store_sample(sample, render_data.render_settings.render_resolution,
																										  x, y, bounce);
					}
#endif
				}
				else
				{
					if (bounce == 1)
					{
						// For envmap path, the direction is stored in the hit point
						restir_pt_initial_sample.rc_vertex = ray.direction;
						// -1 for the primitive index indicates that this is an envmap sample
						restir_pt_initial_sample.rc_vertex_primitive_index = -1;
					}

					ColorRGB32F envmap_emission =
						path_tracing_miss_gather_envmap(render_data, ColorRGB32F(1.0f), ray.direction, ray_payload.bounce, pixel_index);

					restir_pt_initial_sample.rc_vertex_incident_radiance = envmap_emission;
					restir_pt_initial_sample.target_function			 = (path_unweighted_throughput_up_to_rc_vertex * envmap_emission).luminance();
					restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, (ray_payload.throughput * envmap_emission).luminance(),
																  random_number_generator);

					ray_payload.next_ray_state = RayState::MISSED;
				}
			}
			else if (ray_payload.next_ray_state == RayState::MISSED)
				break;
		}

		ReSTIR_PT_do_last_deferred_NEE_MIS(render_data, ray, ray_payload, path_unweighted_throughput_up_to_rc_vertex_for_deferred_nee,
										   path_unweighted_throughput_after_rc_vertex_for_deferred_nee, restir_pt_initial_reservoir, restir_pt_initial_sample,
										   closest_hit_info, nee_deferred_MIS_context, random_number_generator);
	}

	render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);

	// If we got here, this means that we still have at least one ray active
	// This is a concurrent write by the way but we don't really care, everyone is writing
	// the same value
	render_data.aux_buffers.still_one_ray_active[0] = 1;

	restir_pt_initial_reservoir.M = 1;
	restir_pt_initial_reservoir.end_with_normalization(1.0f, render_data.render_settings.restir_pt_settings.initial_candidates.initial_path_trees_count);
	restir_pt_initial_reservoir.sanity_check(make_int2(x, y));

	/*restir_pt_initial_reservoir.sample.target_function =
		ReSTIR_PT_evaluate_target_function<false, false>(render_data, restir_pt_initial_reservoir.sample, initial_surface, random_number_generator);*/

	render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer[pixel_index] = restir_pt_initial_reservoir;

	if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_SHADE_ONLY_INITIAL_CANDIDATES &&
		ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled)
	{
		float3_t to_light_direction = restir_pt_initial_reservoir.sample.is_envmap_path()
										  ? restir_pt_initial_reservoir.sample.rc_vertex
										  : hippt::normalize(restir_pt_initial_reservoir.sample.rc_vertex - initial_surface.shading_point);
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
								 restir_pt_initial_reservoir.sample.rc_vertex_incident_radiance * restir_pt_initial_reservoir.UCW;
		}
		else
		{
			// TODO the ray volume state should be updated here
			float3_t view_direction					 = hippt::normalize(initial_surface.shading_point - restir_pt_initial_reservoir.sample.rc_vertex);
			float3_t to_light_direction_sample_point = restir_pt_initial_reservoir.sample.rc_vertex_incident_light_direction;
			BSDFContext secondary_hit_eval_context(view_direction, initial_surface.shading_normal, initial_surface.geometric_normal,
												   to_light_direction_sample_point, restir_pt_initial_reservoir.sample.incident_light_info_at_sample_point,
												   initial_surface.ray_volume_state, false, restir_pt_initial_reservoir.sample.rc_vertex_material, 0.0f);

			ColorRGB32F bsdf_secondary_hit =
				bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator) *
				hippt::abs(hippt::dot(restir_pt_initial_reservoir.sample.rc_vertex_shading_normal.unpack(), to_light_direction_sample_point));
			radiance_to_camera =
				bsdf_first_hit * bsdf_secondary_hit * restir_pt_initial_reservoir.sample.rc_vertex_incident_radiance * restir_pt_initial_reservoir.UCW;
		}

		render_data.buffers.accumulated_ray_colors[pixel_index] = radiance_to_camera * restir_pt_initial_reservoir.UCW;
	}
}

#endif
