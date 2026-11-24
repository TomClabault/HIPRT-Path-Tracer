/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H
#define DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H

#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/PG/SampleDistribution.h"
#include "Device/includes/ReSTIR/PT/SpatialTargetFunctionBSDFApproximation.h"

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

	// With ReSTIR PT, we want the outgoing radiance from the second hit to the camera hit
	// This means that we're basically not taking the first hit into account and so we're not
	// updating the throughput (or the ray_color either, see the main loop) on the bounce 0

	float rr_throughput_scaling = 1.0f;
	// Doing the russian roulette
	if (!do_russian_roulette(render_data.render_settings, ray_payload.bounce, ray_payload.throughput, rr_throughput_scaling, weighted_throughput,
							 random_number_generator))
	{
		// Killed by russian roulette
		ray_payload.throughput = ColorRGB32F(0.0f);
		unweighted_throughput  = ColorRGB32F(0.0f);

		return ReSTIR_PT_invalid_throughput;
	}
	// else
	//	// Not killed by russian roulette so we're scaling the throughputs
	//	unweighted_throughput *= rr_throughput_scaling;

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
 *
 * 'primary_gbuffer_path_index' is the G-buffer index of the pixel at the primary hit, NEE_DEFERRED_INVALID_PATH_INDEX at the other vertices:
 * the deferred MIS of the next vertex then reads the material of the primary hit from the G-buffer, with its textures resolved at the hit
 */
HIPRT_HOST_DEVICE ColorRGB32F ReSTIR_PT_compute_next_indirect_bounce(HIPRTRenderData& render_data,
																	 RayPayload& ray_payload,
																	 HitInfo& closest_hit_info,
																	 float3_t view_direction,
																	 hiprtRay& out_ray,
																	 Xorshift32Generator& random_number_generator,
																	 BSDFIncidentLightInfo& incident_light_info,
																	 float& out_bsdf_pdf,
																	 NEEDeferredMISContext& nee_deferred_MIS_context,
																	 unsigned int primary_gbuffer_path_index)
{
	nee_deferred_MIS_context.fill_last_hit_information(closest_hit_info, view_direction, ray_payload.volume_state, closest_hit_info.primitive_index,
													   closest_hit_info.texcoords, primary_gbuffer_path_index, ray_payload.throughput);

	ColorRGB32F bsdf_color;
	float3_t bounce_direction;
	float bsdf_pdf;

#if ReSTIRPGEnable == KERNEL_OPTION_FALSE
	path_tracing_sample_bsdf_next_indirect_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf,
												  random_number_generator, incident_light_info);
#else
	restir_pg_sample_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator,
							incident_light_info);
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_FALSE

	out_bsdf_pdf = bsdf_pdf;

	out_ray.origin	  = closest_hit_info.inter_point;
	out_ray.direction = bounce_direction;

	// Terminate ray if bad sampling
	if (bsdf_pdf <= 0.0f)
		return ReSTIR_PT_invalid_throughput;

	ColorRGB32F this_bounce_unweighted_throughput;
	if ((this_bounce_unweighted_throughput = ReSTIR_PT_update_ray_throughputs(render_data, ray_payload, closest_hit_info, bsdf_color, bounce_direction,
																			  bsdf_pdf, random_number_generator, nee_deferred_MIS_context)) ==
		ReSTIR_PT_invalid_throughput)
		return ReSTIR_PT_invalid_throughput;

#if PathSamplingStrategyIsReSTIRPT
	nee_deferred_MIS_context.last_bsdf_incident_light_info = incident_light_info;
#endif

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
}

HIPRT_DEVICE void ReSTIR_PT_rc_vertex_fill_information(const HIPRTRenderData& render_data,
													   const RayPayload& ray_payload,
													   const HitInfo& closest_hit_info,
													   ReSTIRPTReservoirSample& restir_pt_initial_sample)
{
	restir_pt_initial_sample.rc_vertex = closest_hit_info.inter_point;
	restir_pt_initial_sample.rc_vertex_geometric_normal.pack(closest_hit_info.geometric_normal);
	restir_pt_initial_sample.rc_vertex_shading_normal.pack(closest_hit_info.shading_normal);
	restir_pt_initial_sample.rc_vertex_texcoords_u	   = closest_hit_info.texcoords.x;
	restir_pt_initial_sample.rc_vertex_texcoords_v	   = closest_hit_info.texcoords.y;
	restir_pt_initial_sample.rc_vertex_primitive_index = closest_hit_info.primitive_index;

	if (ReSTIR_PT_target_function_bsdf_approximation_enabled(render_data))
		// Material part of the target function BSDF approximation flags and lobe of the sample. The material of the payload is the one that the target
		// function fetches for the reconnection vertex (get_intersection_material() with the same primitive and texcoords): no extra material fetch
		ReSTIR_PT_target_function_bsdf_approximation_fill_rc_vertex(render_data, ray_payload.material, restir_pt_initial_sample);
}

/**
 * Adds to the reservoir the candidate of a light sample or of a BSDF sample of the vertex at 'bounce' that reaches a light in 'incident_light_direction'.
 *
 * 'light_point', 'light_geometric_normal' and 'light_primitive_index' describe the point reached on emissive geometry. For a candidate that reaches
 * the envmap, 'light_point' is the direction towards the envmap, 'light_geometric_normal' is zero and 'light_primitive_index' is -1.
 *
 * 'bsdf_throughput' is the BSDF times the cosine term of 'incident_light_direction' at the vertex, 'incident_light_info' is the lobe information
 * of the BSDF sample (BSDFIncidentLightInfo::NO_INFO for a light sample) and 'weight' is the resampling weight of the candidate
 *
 * 'path_unweighted_throughput_up_to_rc_vertex' is the BSDF * cosine product of the vertices up to the reconnection vertex and
 * 'path_weighted_throughput_after_rc_vertex' the BSDF * cosine / PDF product of the vertices after it. The target function of the candidate is
 * therefore its path contribution divided by the PDF of the bounces after the reconnection vertex. 'weight' does not depend on that division
 */
HIPRT_DEVICE void ReSTIR_PT_add_candidate(int bounce,
										  float3_t incident_light_direction,
										  float3_t light_point,
										  float3_t light_geometric_normal,
										  int light_primitive_index,
										  ColorRGB32F light_emission,
										  ColorRGB32F bsdf_throughput,
										  BSDFIncidentLightInfo incident_light_info,
										  ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
										  ColorRGB32F path_weighted_throughput_after_rc_vertex,
										  float weight,
										  ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
										  ReSTIRPTReservoirSample& restir_pt_initial_sample,
										  Xorshift32Generator& random_number_generator,
										  int x,
										  int y)
{
	if (bounce == 0)
		ReSTIR_PT_rc_di_vertex_fill_information(light_point, light_geometric_normal, light_primitive_index, incident_light_info, restir_pt_initial_sample);

	if (bounce == 1)
	{
		restir_pt_initial_sample.incident_light_info_at_sample_point	   = incident_light_info;
		restir_pt_initial_sample.bsdf_throughput_luminance_at_sample_point = bsdf_throughput.luminance();
	}

	if (bounce <= 1)
		restir_pt_initial_sample.rc_vertex_incident_light_direction = incident_light_direction;

	restir_pt_initial_sample.di_sample					 = bounce == 0;
	restir_pt_initial_sample.rc_vertex_incident_radiance = light_emission * path_weighted_throughput_after_rc_vertex;

	if (bounce >= 2)
		restir_pt_initial_sample.rc_vertex_incident_radiance *= bsdf_throughput;
	restir_pt_initial_sample.target_function =
		(path_unweighted_throughput_up_to_rc_vertex * path_weighted_throughput_after_rc_vertex * bsdf_throughput * light_emission).luminance();

	restir_pt_initial_reservoir.add_one_candidate(restir_pt_initial_sample, weight, random_number_generator);
	restir_pt_initial_reservoir.sanity_check(make_int2(x, y));
}

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
/**
 * Solid angle PDF with which the light candidates of the previous vertex sample the emissive triangle hit by the deferred BSDF candidate.
 * 0 when the lightcut of the previous vertex is unresolved: no light candidate sampled emissive geometry there.
 *
 * The material reloaded by get_last_material() cannot decide that: for the primary hit it is read at default texture coordinates, so a
 * textured material that is perfectly smooth at the hit (no light sampling, lightcut left unresolved) can be rough there. The PDF would
 * then read the lightcut data at the unresolved index, far out of bounds
 */
// Not inlined: inlined into the initial candidates kernel, this PDF raises the register count of the kernel above an allocation step
HIPRT_DEVICE HIPRT_NOINLINE float ReSTIR_PT_get_deferred_light_sampler_solid_angle_pdf(const HIPRTRenderData& render_data,
																					   const NEEDeferredMISContext& nee_deferred_MIS_context,
																					   const HitInfo& light_hit_info)
{
	if (nee_deferred_MIS_context.last_lightcut_index == IlluminationAwareKDTreeNode::UNRESOLVED_LIGHTCUT_INDEX)
		return 0.0f;

	IlluminationAwareKDTreeSGShadingContext shading_context =
		build_light_clustering_shading_context(nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction,
											   nee_deferred_MIS_context.last_shading_normal, nee_deferred_MIS_context.get_last_material(render_data));
	// The lightcut was resolved by the NEE of the previous vertex, at this same shading point. The emissive lights may have been one of two
	// choices of that NEE, the other one being the envmap, and sampled from the lightcut without its envmap slot
	float lightcut_pdf = pdf_of_emissive_triangle_hit_solid_angle_learning_to_cluster(
		render_data, shading_context, nee_deferred_MIS_context.last_lightcut_index, nee_deferred_MIS_context.get_last_material(render_data),
		light_hit_info.primitive_index, light_hit_info.inter_point, light_hit_info.original_geometric_normal(),
		nee_deferred_MIS_context.last_exclude_envmap_slot);

	return lightcut_pdf * nee_deferred_MIS_context.last_emissive_selection_probability;
}

/**
 * Probability that the light candidates of the previous vertex select the envmap cluster, for the MIS weight of the deferred BSDF candidate that
 * misses the scene. 0 when the lightcut of the previous vertex is unresolved: its material cannot do light sampling and no light candidate
 * sampled the envmap there
 */
// Not inlined, for the same reason as ReSTIR_PT_get_deferred_light_sampler_solid_angle_pdf()
HIPRT_DEVICE HIPRT_NOINLINE float ReSTIR_PT_get_deferred_envmap_cluster_probability(const HIPRTRenderData& render_data,
																					const NEEDeferredMISContext& nee_deferred_MIS_context)
{
	if (nee_deferred_MIS_context.last_lightcut_index == IlluminationAwareKDTreeNode::UNRESOLVED_LIGHTCUT_INDEX)
		return 0.0f;

	// The NEE of the previous vertex sampled the envmap outside of the lightcut, with that probability
	if (nee_deferred_MIS_context.last_exclude_envmap_slot)
		return nee_deferred_MIS_context.last_envmap_selection_probability;

	IlluminationAwareKDTreeSGShadingContext shading_context =
		build_light_clustering_shading_context(nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction,
											   nee_deferred_MIS_context.last_shading_normal, nee_deferred_MIS_context.get_last_material(render_data));

	// The lightcut was resolved by the NEE of the previous vertex, at this same shading point
	return pdf_of_envmap_cluster_learning_to_cluster(render_data, shading_context, nee_deferred_MIS_context.last_lightcut_index);
}
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

/**
 * Adds to the reservoir the BSDF candidate of the previous vertex: the BSDF sample that continues the main path, MIS weighted against the light
 * candidates of the previous vertex. It is the only BSDF candidate of that vertex
 */
HIPRT_DEVICE void ReSTIR_PT_do_deferred_NEE_MIS(HIPRTRenderData& render_data,
												bool intersection_found,
												float3_t sampled_bsdf_direction,
												RayPayload& ray_payload,
												ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
												ColorRGB32F path_weighted_throughput_after_rc_vertex,
												ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
												ReSTIRPTReservoirSample& restir_pt_initial_sample,
												HitInfo& light_hit_info,
												NEEDeferredMISContext& nee_deferred_MIS_context,
												Xorshift32Generator& random_number_generator)
{
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	int last_bounce = ray_payload.bounce - 1;
	if (last_bounce == 0 && !render_data.render_settings.enable_direct_lighting)
		// Deferred NEE MIS for the primary hit but we're not doing direct lighting
		return;

	float bsdf_sample_pdf = nee_deferred_MIS_context.last_bsdf_sample_pdf;
	if (bsdf_sample_pdf <= 0.0f)
		return;

	BSDFIncidentLightInfo incident_light_info = nee_deferred_MIS_context.last_bsdf_incident_light_info;
	ColorRGB32F bsdf_throughput				  = nee_deferred_MIS_context.last_bsdf_x_cos_theta;

	// Checking that we did hit something and if we hit something,
	// it needs to be emissive
	if (!intersection_found)
	{
		float envmap_pdf_solid_angle;
		ColorRGB32F envmap_emission;

		// The white furnace mode replaces the envmap with the uniform sky, as path_tracing_miss_gather_envmap() does
		if (render_data.world_settings.ambient_light_type == AmbientLightType::UNIFORM || render_data.bsdfs_data.white_furnace_mode)
		{
			envmap_emission		   = render_data.world_settings.uniform_light_color;
			envmap_pdf_solid_angle = 1.0f;
		}
		else if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
			// Envmap hit
			envmap_emission = envmap_eval(render_data, sampled_bsdf_direction, envmap_pdf_solid_angle);

		float nee_mis_weight;

		if (render_data.kd_tree_device.learning_to_cluster.envmap_cluster_enabled)
		{
			// The envmap is a cluster of the lightcuts: the learning-to-cluster light candidate of the previous vertex samples it
			float light_sampler_solid_angle_pdf =
				ReSTIR_PT_get_deferred_envmap_cluster_probability(render_data, nee_deferred_MIS_context) * envmap_pdf_solid_angle;
			nee_mis_weight = balance_heuristic(bsdf_sample_pdf, light_sampler_solid_angle_pdf);
		}
		else
			// The uniform sky, or an envmap that is not a cluster of the lightcuts (no importance sampling, zero intensity, white furnace mode), isn't
			// explicitly sampled: BSDF samples only can gather emission from the sky: 1.0f MIS weight
			nee_mis_weight = 1.0f;
		float weight = nee_mis_weight * (nee_deferred_MIS_context.last_ray_throughput * bsdf_throughput / bsdf_sample_pdf * envmap_emission).luminance();

		ReSTIR_PT_add_candidate(last_bounce, sampled_bsdf_direction, sampled_bsdf_direction, make_float3(0.0f, 0.0f, 0.0f), -1, envmap_emission,
								bsdf_throughput, incident_light_info, path_unweighted_throughput_up_to_rc_vertex, path_weighted_throughput_after_rc_vertex,
								weight, restir_pt_initial_reservoir, restir_pt_initial_sample, random_number_generator, -1, -1);

		return;
	}

	float3_t view_direction	 = -sampled_bsdf_direction;
	ColorRGB32F hit_emission = ray_payload.material.get_emission();
	if (hit_emission.is_black() || compute_cosine_term_at_light_source(light_hit_info.original_geometric_normal(), view_direction) <= 0.0f)
		return;

	if (bsdf_sample_pdf > 0.0f)
	{
		float nee_mis_weight = 1.0f;

		if (render_data.buffers.emissive_triangles_count > 0)
		{
			float light_sampler_solid_angle_pdf = ReSTIR_PT_get_deferred_light_sampler_solid_angle_pdf(render_data, nee_deferred_MIS_context, light_hit_info);
			nee_mis_weight						= balance_heuristic(bsdf_sample_pdf, light_sampler_solid_angle_pdf);
		}
		// else: no light candidate samples emissive geometry, the BSDF candidate is the only strategy: 1.0f MIS weight

		float weight = nee_mis_weight * (nee_deferred_MIS_context.last_ray_throughput * bsdf_throughput / bsdf_sample_pdf * hit_emission).luminance();

		ReSTIR_PT_add_candidate(last_bounce, sampled_bsdf_direction, light_hit_info.inter_point, light_hit_info.geometric_normal,
								light_hit_info.primitive_index, hit_emission, bsdf_throughput, incident_light_info, path_unweighted_throughput_up_to_rc_vertex,
								path_weighted_throughput_after_rc_vertex, weight, restir_pt_initial_reservoir, restir_pt_initial_sample,
								random_number_generator, -1, -1);
	}
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
}

HIPRT_DEVICE void ReSTIR_PT_do_last_deferred_NEE_MIS(HIPRTRenderData& render_data,
													 hiprtRay ray,
													 RayPayload& ray_payload,
													 ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
													 ColorRGB32F path_weighted_throughput_after_rc_vertex,
													 ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
													 ReSTIRPTReservoirSample& restir_pt_initial_sample,
													 HitInfo& light_hit_info,
													 NEEDeferredMISContext& nee_deferred_MIS_context,
													 bool last_intersection_found,
													 Xorshift32Generator& random_number_generator)
{
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	// We will have one more bounce than necessary when getting here and this can throw off the 'max bounce' of alpha testing so we need to substract one bounce
	// here
	ray_payload.bounce--;

	bool intersection_found;
	if (last_intersection_found)
		// If the last ray was a hit and we're doing deferred NEE, this means that we sampled the ray in the main loop but we need to trace it here to see if it
		// hits anything interesting
		intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, light_hit_info, random_number_generator);
	else
		// This means that the last ray we shot was a miss so we're doing deferred NEE MIS here for envmap, no need to re-trace a ray, we know it's a miss
		// already
		intersection_found = false;
	// And add it back before deferred NEE MIS so that the code inside deferred NEE MIS receives the bounce index that it expects
	ray_payload.bounce++;

	ReSTIR_PT_do_deferred_NEE_MIS(render_data, intersection_found, ray.direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
								  path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample, light_hit_info,
								  nee_deferred_MIS_context, random_number_generator);
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
}

#endif // #ifndef DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H
