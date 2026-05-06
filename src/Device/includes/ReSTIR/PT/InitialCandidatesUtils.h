/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H
#define DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H

#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/PG/SampleDistribution.h"

#define ReSTIR_PT_invalid_throughput ColorRGB32F(-1.0f, -1.0f, -1.0f)

HIPRT_HOST_DEVICE ColorRGB32F ReSTIR_PT_update_ray_throughputs(HIPRTRenderData& render_data,
															   RayPayload& ray_payload,
															   HitInfo& closest_hit_info,
															   ColorRGB32F bsdf_color,
															   const float3_t& bounce_direction,
															   float bsdf_pdf,
															   Xorshift32Generator& random_number_generator,
															   NEEDeferredMISContext& nee_deferred_MIS_context)
{
	ColorRGB32F bsdf_throughput		  = bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal));
	ColorRGB32F dispersion_throughput = get_dispersion_ray_color(ray_payload.volume_state.sampled_wavelength, ray_payload.material.dispersion_scale);
	ColorRGB32F unweighted_throughput = bsdf_throughput * dispersion_throughput;
	ColorRGB32F weighted_throughput	  = unweighted_throughput / bsdf_pdf;

	nee_deferred_MIS_context.fill_last_bsdf_information(unweighted_throughput, bsdf_pdf);

	// With ReSTIR GI, we want the outgoing radiance from the second hit to the camera hit
	// This means that we're basically not taking the first hit into account and so we're not
	// updating the throughput (or the ray_color either, see the main loop) on the bounce 0

	float rr_throughput_scaling = 1.0f;
	// Doing the russian roulette
	if (!do_russian_roulette(render_data.render_settings, ray_payload.bounce, ray_payload.throughput, rr_throughput_scaling, weighted_throughput,
							 random_number_generator))
	{
		// Killed by russian roulette
		ray_payload.throughput = ColorRGB32F(0.0f);

		return ReSTIR_PT_invalid_throughput;
	}
	else
		// Not killed by russian roulette so we're scaling the throughputs
		unweighted_throughput *= rr_throughput_scaling;

	// Clamp every component to a minimum of 1.0e-5f to avoid numerical instabilities that can
	// happen: with some material, the throughput can get so low that it becomes denormalized and
	// this can cause issues in some parts of the renderer (most notably the NaN detection)
	unweighted_throughput.max(ColorRGB32F(1.0e-5f, 1.0e-5f, 1.0e-5f));

	ray_payload.throughput *= dispersion_throughput;
	ray_payload.throughput *= weighted_throughput;
	// Clamp every component to a minimum of 1.0e-5f to avoid numerical instabilities that can
	// happen: with some material, the throughput can get so low that it becomes denormalized and
	// this can cause issues in some parts of the renderer (most notably the NaN detection)
	ray_payload.throughput.max(ColorRGB32F(1.0e-5f, 1.0e-5f, 1.0e-5f));

	return unweighted_throughput;
}

/**
 * Returns true if the bounce was sampled successfully,
 * false otherwise (is the BSDF sample failed, if russian roulette killed the sample, ...)
 */
HIPRT_HOST_DEVICE ColorRGB32F ReSTIR_PT_compute_next_indirect_bounce(HIPRTRenderData& render_data,
																	 RayPayload& ray_payload,
																	 HitInfo& closest_hit_info,
																	 float3_t view_direction,
																	 hiprtRay& out_ray,
																	 Xorshift32Generator& random_number_generator,
																	 BSDFIncidentLightInfo& incident_light_info,
																	 float& out_bsdf_pdf,
																	 NEEDeferredMISContext& nee_deferred_MIS_context)
{
	nee_deferred_MIS_context.fill_last_hit_information(closest_hit_info, view_direction, ray_payload.volume_state, ray_payload.material,
													   ray_payload.throughput);

	ColorRGB32F bsdf_color;
	float3_t bounce_direction;
	float bsdf_pdf;

#if ReSTIRPGEnable == KERNEL_OPTION_FALSE
	path_tracing_sample_bsdf_next_indirect_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf,
												  random_number_generator, incident_light_info);
#else
	restir_pg_sample_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator,
							incident_light_info);
#endif

	out_bsdf_pdf = bsdf_pdf;

	// Terminate ray if bad sampling
	if (bsdf_pdf <= 0.0f)
		return ReSTIR_PT_invalid_throughput;

	ColorRGB32F this_bounce_unweighted_throughput;
	if ((this_bounce_unweighted_throughput = ReSTIR_PT_update_ray_throughputs(render_data, ray_payload, closest_hit_info, bsdf_color, bounce_direction,
																			  bsdf_pdf, random_number_generator, nee_deferred_MIS_context)) ==
		ReSTIR_PT_invalid_throughput)
		return ReSTIR_PT_invalid_throughput;

	nee_deferred_MIS_context.last_bsdf_incident_light_info = incident_light_info;

	out_ray.origin	  = closest_hit_info.inter_point;
	out_ray.direction = bounce_direction;

	// Returning this bounce's unweighted throughput
	return this_bounce_unweighted_throughput;
}

HIPRT_DEVICE void ReSTIR_PT_rc_di_vertex_fill_information(float3_t point_on_light,
														  float3_t light_geometric_normal,
														  int light_primitive_index,
														  BSDFIncidentLightInfo incident_light_info,
														  ReSTIRPTReservoirSample& restir_pt_initial_sample)
{
	restir_pt_initial_sample.rc_vertex = point_on_light;
	restir_pt_initial_sample.rc_vertex_geometric_normal.pack(light_geometric_normal);
	restir_pt_initial_sample.rc_vertex_primitive_index			  = light_primitive_index;
	restir_pt_initial_sample.incident_light_info_at_visible_point = incident_light_info;
	restir_pt_initial_sample.sample_point_rough_enough			  = true;
}

HIPRT_DEVICE void ReSTIR_PT_rc_vertex_fill_information(const HIPRTRenderData& render_data,
													   const RayPayload& ray_payload,
													   const HitInfo& closest_hit_info,
													   ReSTIRPTReservoirSample& restir_pt_initial_sample)
{
	restir_pt_initial_sample.rc_vertex = closest_hit_info.inter_point;
	restir_pt_initial_sample.rc_vertex_geometric_normal.pack(closest_hit_info.geometric_normal);
	restir_pt_initial_sample.rc_vertex_shading_normal.pack(closest_hit_info.shading_normal);
	restir_pt_initial_sample.rc_vertex_material		   = ray_payload.material;
	restir_pt_initial_sample.rc_vertex_primitive_index = closest_hit_info.primitive_index;
	restir_pt_initial_sample.sample_point_rough_enough =
		ray_payload.material.can_do_light_sampling(render_data.render_settings.restir_pt_settings.neighbor_sample_point_roughness_threshold);
}

HIPRT_DEVICE void ReSTIR_PT_do_deferred_NEE_MIS(HIPRTRenderData& render_data,
												bool intersection_found,
												RayPayload& ray_payload,
												ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
												ColorRGB32F path_unweighted_throughput_after_rc_vertex,
												ReSTIRPTReservoir& restir_pt_initial_reservoir,
												ReSTIRPTReservoirSample& restir_pt_initial_sample,
												HitInfo& light_hit_info,
												NEEDeferredMISContext& nee_deferred_MIS_context,
												Xorshift32Generator& random_number_generator)
{
#if PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT && DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
	if (ray_payload.bounce == 1 && !render_data.render_settings.enable_direct_lighting)
		// Deferred NEE MIS for the primary hit but we're not doing direct lighting
		return;

	// Checking that we did hit something and if we hit something,
	// it needs to be emissive
	float3_t sampled_bsdf_direction = hippt::normalize(light_hit_info.inter_point - nee_deferred_MIS_context.last_shading_point);
	float3_t view_direction			= -sampled_bsdf_direction;
	ColorRGB32F hit_emission		= ray_payload.material.emission;
	if (!intersection_found || hit_emission.is_black() || compute_cosine_term_at_light_source(light_hit_info.geometric_normal, view_direction) <= 0.0f)
		return;

	if (render_data.render_settings.restir_pt_settings.initial_candidates.nee_ris_number_of_bsdf_candidates > 0)
	{
		int nb_light_candidates = render_data.render_settings.restir_pt_settings.initial_candidates.nee_ris_number_of_light_candidates;
		int nb_bsdf_candidates	= render_data.render_settings.restir_pt_settings.initial_candidates.nee_ris_number_of_bsdf_candidates;

		float bsdf_sample_pdf					  = nee_deferred_MIS_context.last_bsdf_sample_pdf;
		BSDFIncidentLightInfo incident_light_info = nee_deferred_MIS_context.last_bsdf_incident_light_info;
		ColorRGB32F bsdf_throughput				  = nee_deferred_MIS_context.last_bsdf_cos_theta;

		if (bsdf_sample_pdf > 0.0f)
		{
			float3_t point_on_light = light_hit_info.inter_point;
			int last_bounce			= ray_payload.bounce - 1;

			if (last_bounce == 0)
				ReSTIR_PT_rc_di_vertex_fill_information(point_on_light, nee_deferred_MIS_context.last_geometric_normal,
														nee_deferred_MIS_context.last_primitive_index, incident_light_info, restir_pt_initial_sample);
			if (last_bounce == 1)
				restir_pt_initial_sample.incident_light_info_at_sample_point = incident_light_info;
			if (last_bounce <= 1)
				restir_pt_initial_sample.rc_vertex_incident_light_direction = sampled_bsdf_direction;
			restir_pt_initial_sample.di_sample					 = last_bounce == 0;
			restir_pt_initial_sample.rc_vertex_incident_radiance = hit_emission * path_unweighted_throughput_after_rc_vertex;
			if (last_bounce >= 2)
				restir_pt_initial_sample.rc_vertex_incident_radiance *= bsdf_throughput;
			restir_pt_initial_sample.target_function =
				(path_unweighted_throughput_up_to_rc_vertex * path_unweighted_throughput_after_rc_vertex * bsdf_throughput * hit_emission).luminance();

			float hit_distance					= hippt::length(light_hit_info.inter_point - nee_deferred_MIS_context.last_shading_point);
			float light_sampler_solid_angle_pdf = pdf_of_emissive_triangle_hit_solid_angle(
				render_data, nee_deferred_MIS_context.last_shading_point, view_direction, nee_deferred_MIS_context.last_shading_normal, ray_payload.material,
				light_hit_info.primitive_index, hit_emission, light_hit_info.geometric_normal, hit_distance, sampled_bsdf_direction);
			float nee_mis_weight = balance_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, light_sampler_solid_angle_pdf,
													 nb_light_candidates * DirectLightIntegrationFactor<DirectLightSamplingStrategy>());
			float weight		 = nee_mis_weight * (nee_deferred_MIS_context.last_ray_throughput * bsdf_throughput / bsdf_sample_pdf * hit_emission).luminance();

			restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, weight, random_number_generator);
			restir_pt_initial_reservoir.sanity_check(make_int2(-1, -1));
		}
	}
#endif
}

HIPRT_DEVICE void ReSTIR_PT_do_last_deferred_NEE_MIS(HIPRTRenderData& render_data,
													 hiprtRay ray,
													 RayPayload& ray_payload,
													 ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
													 ColorRGB32F path_unweighted_throughput_after_rc_vertex,
													 ReSTIRPTReservoir& restir_pt_initial_reservoir,
													 ReSTIRPTReservoirSample& restir_pt_initial_sample,
													 HitInfo& light_hit_info,
													 NEEDeferredMISContext& nee_deferred_MIS_context,
													 Xorshift32Generator& random_number_generator)
{
#if PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT && DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
	bool intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, light_hit_info, random_number_generator);

	ReSTIR_PT_do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
								  path_unweighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample, light_hit_info,
								  nee_deferred_MIS_context, random_number_generator);
#endif
}

#endif
