/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PT_WAVEFRONT_WAVEFRONT_INITIAL_CANDIDATES_H
#define DEVICE_INCLUDES_RESTIR_PT_WAVEFRONT_WAVEFRONT_INITIAL_CANDIDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/ReSTIR/PT/InitialCandidatesUtils.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/SPMIS/SPMISUtils.h"
#include "Device/includes/SanityCheck.h"
#include "Device/includes/Wavefront/WavefrontShading.h"

#include "HostDeviceCommon/Xorshift.h"

/**
 * Wavefront version of the ReSTIR PT initial candidates kernel (kernels/ReSTIR/PT/InitialCandidates.h).
 *
 * Each path tree of a pixel is generated one bounce at a time:
 *	- the shading kernels process the vertex of one bounce: deferred BSDF candidate of the previous vertex, light candidate of the vertex (its
 *	  shadow ray is traced inline) and BSDF sample of the next bounce. The path is then queued for the trace kernel,
 *	- the trace kernel finds the next vertex and appends the path to the queue of the material family of that vertex. Misses and the rays of
 *	  the paths that end at this vertex go to the completion queue,
 *	- the completion kernel adds the last deferred BSDF candidate of the path and, after the last path tree of the pixel, normalizes the
 *	  reservoir.
 *
 * Every kernel restores the random number generator of the path from the wavefront data and stores it back: the paths consume their random
 * numbers in the same order as in the initial candidates megakernel.
 *
 * The vertices are shaded with the material of their material family, given by KERNEL_MATERIAL_SPECIALIZATION
 */

#if PathSamplingStrategyIsReSTIRPT && DirectLightNEEEstimator != LSS_LEARNING_TO_CLUSTER_MIS
#error "ReSTIR PT only supports the LSS_LEARNING_TO_CLUSTER_MIS NEE estimator."
#endif

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
// The ReSTIR PG bounce sampling, restir_pg_sample_bounce(), only exists for the full material: every material family shades with it
using ReSTIRPTWavefrontKernelMaterial = DeviceUnpackedPrincipledFullMaterial;
#else
using ReSTIRPTWavefrontKernelMaterial =
	EffectiveMaterialFor<static_cast<BSDFModel>(BSDF_MODEL), static_cast<KernelMaterialSpecialization>(KERNEL_MATERIAL_SPECIALIZATION)>::Type;
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_TRUE

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
/**
 * Light candidate of learning to cluster when the selected cluster of the lightcut is the envmap: samples a direction from the envmap distribution,
 * traces the shadow ray towards the envmap, adds the candidate to the reservoir and appends the learning-to-cluster and spatial training samples
 * of that light sample.
 *
 * 'learning_to_cluster_training_sample' is already filled with the cluster selection of 'cluster_sample'
 */
template <typename MaterialType>
HIPRT_DEVICE void ReSTIR_PT_wavefront_add_learning_to_cluster_envmap_light_candidate(
	HIPRTRenderData& render_data,
	float3_t view_direction,
	RayPayloadT<MaterialType>& ray_payload,
	ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
	ColorRGB32F path_weighted_throughput_after_rc_vertex,
	ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
	ReSTIRPTReservoirSample& restir_pt_initial_sample,
	HitInfo& closest_hit_info,
	Xorshift32Generator& random_number_generator,
	const IlluminationAwareKDTreeLearningToClusterCutTriangleSample& cluster_sample,
	IlluminationAwareKDTreeLearningToClusterTrainingSample& learning_to_cluster_training_sample,
	int x,
	int y)
{
	float envmap_pdf_solid_angle;
	float3_t sampled_direction;
	ColorRGB32F envmap_emission = envmap_sample(render_data.world_settings, sampled_direction, envmap_pdf_solid_angle, random_number_generator);

	float solid_angle_pdf = envmap_pdf_solid_angle * cluster_sample.cluster_probability;
	if (!(solid_angle_pdf > 0.0f))
	{
		// Keep a failed direction proposal observable to learning-to-cluster as a zero-reward observation, as for the emissive clusters
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return;
	}

	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	spatial_training_sample.position				   = closest_hit_info.inter_point;
	spatial_training_sample.incoming_direction		   = sampled_direction;
	spatial_training_sample.valid_for_spatial_training = true;
	spatial_training_sample.cached_guiding_node_index  = cluster_sample.guiding_node_index;

	hiprtRay shadow_ray;
	shadow_ray.origin	 = closest_hit_info.inter_point;
	shadow_ray.direction = sampled_direction;

	NEEPlusPlusContext nee_plus_plus_context;
	nee_plus_plus_context.envmap		 = true;
	nee_plus_plus_context.point_on_light = sampled_direction;
	nee_plus_plus_context.shaded_point	 = closest_hit_info.inter_point;
	if (ray_payload.bounce == 0)
		restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;
	bool shadow_ray_occluded =
		evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, 1.0e35f, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator);

	if (!shadow_ray_occluded)
	{
		spatial_training_sample.spatial_radiance_weight = (envmap_emission / solid_angle_pdf).max_component();

		float bsdf_pdf							  = 0.0f;
		BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
		BSDFContextT<MaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction,
												incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
												MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
		ColorRGB32F bsdf_color		= bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
		ColorRGB32F bsdf_throughput = bsdf_color * hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_direction));

		if (bsdf_pdf > 0.0f)
		{
			// MIS weighted against the BSDF sample of the main path, the only BSDF candidate of the vertex
			float light_candidate_mis_weight = balance_heuristic(solid_angle_pdf, bsdf_pdf);

			float weight = light_candidate_mis_weight * (ray_payload.throughput * bsdf_throughput / solid_angle_pdf * envmap_emission).luminance();
			ReSTIR_PT_add_candidate(ray_payload.bounce, sampled_direction, sampled_direction, make_float3(0.0f, 0.0f, 0.0f), -1, envmap_emission,
									bsdf_throughput, BSDFIncidentLightInfo::NO_INFO, path_unweighted_throughput_up_to_rc_vertex,
									path_weighted_throughput_after_rc_vertex, weight, restir_pt_initial_reservoir, restir_pt_initial_sample,
									random_number_generator, x, y);

			ColorRGB32F learning_estimator							 = envmap_emission * bsdf_throughput / solid_angle_pdf * light_candidate_mis_weight;
			learning_to_cluster_training_sample.q_reward			 = learning_estimator.luminance() * cluster_sample.cluster_probability;
			learning_to_cluster_training_sample.variance_observation = learning_to_cluster_training_sample.q_reward;
		}
	}

	render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);
}

template <typename MaterialType>
HIPRT_DEVICE void ReSTIR_PT_wavefront_add_learning_to_cluster_light_candidate(HIPRTRenderData& render_data,
																			  float3_t view_direction,
																			  RayPayloadT<MaterialType>& ray_payload,
																			  ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
																			  ColorRGB32F path_weighted_throughput_after_rc_vertex,
																			  ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
																			  ReSTIRPTReservoirSample& restir_pt_initial_sample,
																			  HitInfo& closest_hit_info,
																			  Xorshift32Generator& random_number_generator,
																			  unsigned int lightcut_index,
																			  unsigned int guiding_node_index,
																			  int x,
																			  int y)
{
	if (!ray_payload.material.can_do_light_sampling())
		return;

	IlluminationAwareKDTreeSGShadingContext shading_context =
		build_light_clustering_shading_context(closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material);
	unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
	if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
		mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

	IlluminationAwareKDTreeLearningToClusterCutTriangleSample triangle_sample =
		sample_one_emissive_triangle_learning_to_cluster(render_data, shading_context, lightcut_index, guiding_node_index, random_number_generator);
	if (!triangle_sample.valid())
	{
		append_failed_light_clustering_training_sample(render_data, closest_hit_info.inter_point, shading_context, mesh_id, triangle_sample);

		return;
	}

	IlluminationAwareKDTreeLearningToClusterTrainingSample learning_to_cluster_training_sample{};
	learning_to_cluster_training_sample.position					   = closest_hit_info.inter_point;
	learning_to_cluster_training_sample.mesh_id						   = mesh_id;
	learning_to_cluster_training_sample.shading_context				   = shading_context;
	learning_to_cluster_training_sample.selected_cluster_node_index	   = triangle_sample.cluster_node_index;
	learning_to_cluster_training_sample.emissive_triangle_global_index = triangle_sample.emissive_triangle_global_index;
	learning_to_cluster_training_sample.cluster_probability			   = triangle_sample.cluster_probability;
	learning_to_cluster_training_sample.sampled_lightcut_index		   = triangle_sample.lightcut_index;
	learning_to_cluster_training_sample.selected_lightcut_slot		   = triangle_sample.lightcut_slot;
	learning_to_cluster_training_sample.sampled_lightcut_size		   = triangle_sample.lightcut_size_at_sampling;
	learning_to_cluster_training_sample.valid_for_lightcut			   = true;

	if (triangle_sample.is_envmap())
	{
		ReSTIR_PT_wavefront_add_learning_to_cluster_envmap_light_candidate(render_data, view_direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
																		   path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir,
																		   restir_pt_initial_sample, closest_hit_info, random_number_generator, triangle_sample,
																		   learning_to_cluster_training_sample, x, y);

		return;
	}

	LightSamplePointInformation light_sample =
		sample_point_on_light_and_fill_light_sample_information(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
																ray_payload.material, triangle_sample.emissive_triangle_global_index, random_number_generator);
	light_sample.area_measure_pdf *= triangle_sample.triangle_probability();
	if (!(light_sample.area_measure_pdf > 0.0f))
	{
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return;
	}

	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	spatial_training_sample.position				   = closest_hit_info.inter_point;
	spatial_training_sample.valid_for_spatial_training = true;
	spatial_training_sample.cached_guiding_node_index  = triangle_sample.guiding_node_index;

	float3_t shadow_direction = light_sample.point_on_light - closest_hit_info.inter_point;
	float distance_to_light	  = hippt::length(shadow_direction);
	if (!(distance_to_light > 0.0f))
	{
		render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return;
	}
	shadow_direction /= distance_to_light;
	spatial_training_sample.incoming_direction = shadow_direction;

	hiprtRay shadow_ray;
	shadow_ray.origin		 = closest_hit_info.inter_point;
	shadow_ray.direction	 = shadow_direction;
	bool shadow_ray_occluded = true;

	float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_direction);
	if (dot_light_source > 0.0f)
	{
		NEEPlusPlusContext nee_plus_plus_context;
		nee_plus_plus_context.point_on_light = light_sample.point_on_light;
		nee_plus_plus_context.shaded_point	 = closest_hit_info.inter_point;

		if (ray_payload.bounce == 0)
			restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;
		shadow_ray_occluded = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																nee_plus_plus_context, random_number_generator);

		if (!shadow_ray_occluded)
		{
			float solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
			if (solid_angle_pdf > 0.0f)
			{
				spatial_training_sample.spatial_radiance_weight = (light_sample.emission / solid_angle_pdf).max_component();

				float bsdf_pdf							  = 0.0f;
				BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
				BSDFContextT<MaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
														incident_light_info, ray_payload.volume_state, false, ray_payload.material,
														ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
				ColorRGB32F bsdf_color		= bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
				ColorRGB32F bsdf_throughput = bsdf_color * hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));

				if (bsdf_pdf > 0.0f)
				{
					// MIS weighted against the BSDF sample of the main path, the only BSDF candidate of the vertex
					float light_candidate_mis_weight = balance_heuristic(solid_angle_pdf, bsdf_pdf);

					float weight =
						light_candidate_mis_weight * (ray_payload.throughput * bsdf_throughput / solid_angle_pdf * light_sample.emission).luminance();
					ReSTIR_PT_add_candidate(ray_payload.bounce, shadow_direction, light_sample.point_on_light, light_sample.light_source_normal,
											triangle_sample.emissive_triangle_global_index, light_sample.emission, bsdf_throughput,
											BSDFIncidentLightInfo::NO_INFO, path_unweighted_throughput_up_to_rc_vertex,
											path_weighted_throughput_after_rc_vertex, weight, restir_pt_initial_reservoir, restir_pt_initial_sample,
											random_number_generator, x, y);

					ColorRGB32F learning_estimator				 = light_sample.emission * bsdf_throughput / solid_angle_pdf * light_candidate_mis_weight;
					learning_to_cluster_training_sample.q_reward = learning_estimator.luminance() * triangle_sample.cluster_probability;
					learning_to_cluster_training_sample.variance_observation = learning_to_cluster_training_sample.q_reward;
				}
			}
		}
	}

	render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);
}

/**
 * Light candidate of the current vertex: one learning-to-cluster light candidate, which also samples the envmap when the envmap is a cluster of
 * the lightcuts.
 *
 * The only BSDF candidate of the vertex is the BSDF sample that continues the main path: it is MIS weighted against this light candidate when
 * the next vertex is found, by ReSTIR_PT_wavefront_do_deferred_NEE_MIS()
 */
template <typename MaterialType>
HIPRT_DEVICE void ReSTIR_PT_wavefront_stream_NEE(HIPRTRenderData& render_data,
												 float3_t view_direction,
												 RayPayloadT<MaterialType>& ray_payload,
												 ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
												 ColorRGB32F path_weighted_throughput_after_rc_vertex,
												 ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
												 ReSTIRPTReservoirSample& restir_pt_initial_sample,
												 HitInfo& closest_hit_info,
												 NEEDeferredMISContext& nee_deferred_MIS_context,
												 Xorshift32Generator& random_number_generator,
												 int x,
												 int y)
{
	// The lightcut of this shading point is resolved once and used by the light candidate and by the deferred PDF of the next vertex, which are
	// both evaluated at this same shading point.
	//
	// It is resolved before the early exit for disabled direct lighting because the deferred PDF of the next vertex is evaluated even then. It
	// stays unresolved only when no PDF reads it: when the lightcuts hold no cluster (no emissive triangles and no envmap cluster) or for a
	// material that cannot do light sampling, the PDFs return before reading the lightcut
	unsigned int lightcut_index		= IlluminationAwareKDTreeNode::UNRESOLVED_LIGHTCUT_INDEX;
	unsigned int guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	bool has_light_to_sample		= learning_to_cluster_has_light_to_sample(render_data);
	if (has_light_to_sample && ray_payload.material.can_do_light_sampling())
	{
		IlluminationAwareKDTreeSGShadingContext shading_context =
			build_light_clustering_shading_context(closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material);
		unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
		if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
			mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

		lightcut_index = render_data.kd_tree_device.resolve_lightcut(shading_context, mesh_id, &guiding_node_index);
	}

	// Also written when unresolved so that the deferred PDF of the next vertex never reads the lightcut of an older vertex
	nee_deferred_MIS_context.set_last_lightcut_index(lightcut_index);

	if (ray_payload.bounce == 0 && !render_data.render_settings.enable_direct_lighting)
		return;

	// When the envmap is a cluster of the lightcuts, the learning-to-cluster light candidate also samples it
	if (has_light_to_sample)
		ReSTIR_PT_wavefront_add_learning_to_cluster_light_candidate(
			render_data, view_direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex, path_weighted_throughput_after_rc_vertex,
			restir_pt_initial_reservoir, restir_pt_initial_sample, closest_hit_info, random_number_generator, lightcut_index, guiding_node_index, x, y);
}
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

/**
 * Same as ReSTIR_PT_update_ray_throughputs() with the dispersion scale of the material given separately: the materials of the material
 * families do not all have one
 */
HIPRT_DEVICE ColorRGB32F ReSTIR_PT_wavefront_update_ray_throughputs(HIPRTRenderData& render_data,
																	RayPayloadCommon& ray_payload,
																	HitInfo& closest_hit_info,
																	ColorRGB32F bsdf_color,
																	const float3_t& bounce_direction,
																	float bsdf_pdf,
																	float dispersion_scale,
																	Xorshift32Generator& random_number_generator,
																	NEEDeferredMISContext& nee_deferred_MIS_context)
{
	ColorRGB32F bsdf_throughput		  = bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal));
	ColorRGB32F dispersion_throughput = get_dispersion_ray_color(ray_payload.volume_state.sampled_wavelength, dispersion_scale);
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

		return ReSTIR_PT_invalid_throughput;
	}

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
 * Samples the next bounce of the path with the material of the current vertex.
 *
 * Returns the unweighted throughput of the bounce or ReSTIR_PT_invalid_throughput if the bounce could not be sampled (the BSDF sample failed,
 * russian roulette killed the path, ...)
 */
template <typename MaterialType>
HIPRT_DEVICE ColorRGB32F ReSTIR_PT_wavefront_compute_next_indirect_bounce(HIPRTRenderData& render_data,
																		  RayPayloadT<MaterialType>& ray_payload,
																		  HitInfo& closest_hit_info,
																		  float3_t view_direction,
																		  float dispersion_scale,
																		  hiprtRay& out_ray,
																		  Xorshift32Generator& random_number_generator,
																		  BSDFIncidentLightInfo& incident_light_info,
																		  float& out_bsdf_pdf,
																		  NEEDeferredMISContext& nee_deferred_MIS_context)
{
	nee_deferred_MIS_context.fill_last_hit_information(closest_hit_info, view_direction, ray_payload.volume_state, closest_hit_info.primitive_index,
													   closest_hit_info.texcoords, NEE_DEFERRED_INVALID_PATH_INDEX, ray_payload.throughput);

	ColorRGB32F bsdf_color;
	float3_t bounce_direction;
	float bsdf_pdf;

#if ReSTIRPGEnable == KERNEL_OPTION_FALSE
	path_tracing_sample_bsdf_next_indirect_bounce(render_data, ray_payload, ray_payload.material, closest_hit_info, view_direction, bsdf_color,
												  bounce_direction, bsdf_pdf, random_number_generator, incident_light_info);
#else
	// The kernel material is the full material with ReSTIR PG, see ReSTIRPTWavefrontKernelMaterial
	restir_pg_sample_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator,
							incident_light_info);
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_FALSE

	out_bsdf_pdf = bsdf_pdf;

	out_ray.origin	  = closest_hit_info.inter_point;
	out_ray.direction = bounce_direction;

	// Terminate ray if bad sampling
	if (bsdf_pdf <= 0.0f)
		return ReSTIR_PT_invalid_throughput;

	ColorRGB32F this_bounce_unweighted_throughput =
		ReSTIR_PT_wavefront_update_ray_throughputs(render_data, ray_payload, closest_hit_info, bsdf_color, bounce_direction, bsdf_pdf, dispersion_scale,
												   random_number_generator, nee_deferred_MIS_context);
	if (this_bounce_unweighted_throughput == ReSTIR_PT_invalid_throughput)
		return ReSTIR_PT_invalid_throughput;

	nee_deferred_MIS_context.last_bsdf_incident_light_info = incident_light_info;

	// Returning this bounce's unweighted throughput
	return this_bounce_unweighted_throughput;
}

/**
 * Same as ReSTIR_PT_rc_vertex_fill_information() for the material of a material family
 */
template <typename MaterialType>
HIPRT_DEVICE void ReSTIR_PT_wavefront_rc_vertex_fill_information(const HIPRTRenderData& render_data,
																 const MaterialType& material,
																 const HitInfo& closest_hit_info,
																 ReSTIRPTReservoirSample& restir_pt_initial_sample)
{
	restir_pt_initial_sample.rc_vertex = closest_hit_info.inter_point;
	restir_pt_initial_sample.rc_vertex_geometric_normal.pack(closest_hit_info.geometric_normal);
	restir_pt_initial_sample.rc_vertex_shading_normal.pack(closest_hit_info.shading_normal);
	restir_pt_initial_sample.rc_vertex_texcoords_u	   = closest_hit_info.texcoords.x;
	restir_pt_initial_sample.rc_vertex_texcoords_v	   = closest_hit_info.texcoords.y;
	restir_pt_initial_sample.rc_vertex_primitive_index = closest_hit_info.primitive_index;

	if (!ReSTIR_PT_MIS_target_function_bsdf_approximation_enabled(render_data))
		return;

	// Material part of the MIS target function BSDF approximation flags and lobe of the sample. It is computed from the material that the target
	// function fetches for the reconnection vertex: get_intersection_material() with the same primitive and texcoords
	if constexpr (std::is_same_v<MaterialType, DeviceUnpackedPrincipledFullMaterial>)
		ReSTIR_PT_MIS_target_function_bsdf_approximation_fill_rc_vertex(render_data, material, restir_pt_initial_sample);
	else
	{
		// The material of a material family does not have all the parameters of the full material
		int material_index										= render_data.buffers.material_indices[closest_hit_info.primitive_index];
		DeviceUnpackedPrincipledFullMaterial rc_vertex_material = get_intersection_material(render_data, material_index, closest_hit_info.texcoords);

		ReSTIR_PT_MIS_target_function_bsdf_approximation_fill_rc_vertex(render_data, rc_vertex_material, restir_pt_initial_sample);
	}
}

/**
 * Adds to the reservoir the BSDF candidate of the previous vertex: the BSDF sample that continues the main path, MIS weighted against the light
 * candidates of the previous vertex. It is the only BSDF candidate of that vertex.
 *
 * Same as ReSTIR_PT_do_deferred_NEE_MIS() with the bounce index of the path and the emission of the vertex found by the BSDF sample given
 * separately: 'bounce' is the bounce of that vertex, one past the bounce of the vertex that sampled the BSDF direction
 */
HIPRT_DEVICE void ReSTIR_PT_wavefront_do_deferred_NEE_MIS(HIPRTRenderData& render_data,
														  bool intersection_found,
														  float3_t sampled_bsdf_direction,
														  int bounce,
														  ColorRGB32F hit_emission,
														  ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
														  ColorRGB32F path_weighted_throughput_after_rc_vertex,
														  ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
														  ReSTIRPTReservoirSample& restir_pt_initial_sample,
														  HitInfo& light_hit_info,
														  NEEDeferredMISContext& nee_deferred_MIS_context,
														  Xorshift32Generator& random_number_generator)
{
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	int last_bounce = bounce - 1;
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

	float3_t view_direction = -sampled_bsdf_direction;
	if (hit_emission.is_black() || compute_cosine_term_at_light_source(light_hit_info.original_geometric_normal(), view_direction) <= 0.0f)
		return;

	float nee_mis_weight = 1.0f;
	if (render_data.buffers.emissive_triangles_count > 0)
	{
		float light_sampler_solid_angle_pdf = ReSTIR_PT_get_deferred_light_sampler_solid_angle_pdf(render_data, nee_deferred_MIS_context, light_hit_info);
		nee_mis_weight						= balance_heuristic(bsdf_sample_pdf, light_sampler_solid_angle_pdf);
	}
	// else: no light candidate samples emissive geometry, the BSDF candidate is the only strategy: 1.0f MIS weight

	float weight = nee_mis_weight * (nee_deferred_MIS_context.last_ray_throughput * bsdf_throughput / bsdf_sample_pdf * hit_emission).luminance();

	ReSTIR_PT_add_candidate(last_bounce, sampled_bsdf_direction, light_hit_info.inter_point, light_hit_info.geometric_normal, light_hit_info.primitive_index,
							hit_emission, bsdf_throughput, incident_light_info, path_unweighted_throughput_up_to_rc_vertex,
							path_weighted_throughput_after_rc_vertex, weight, restir_pt_initial_reservoir, restir_pt_initial_sample, random_number_generator,
							-1, -1);
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
}

/**
 * PT_SHADE_ONLY_INITIAL_CANDIDATES debug view: writes the contribution of the final initial candidates reservoir of the pixel to the
 * accumulation buffer
 */
HIPRT_DEVICE void ReSTIR_PT_wavefront_store_shade_only_initial_candidates_debug_color(HIPRTRenderData& render_data,
																					  unsigned int pixel_index,
																					  ReSTIRPTReservoir* restir_pt_initial_reservoir_output,
																					  Xorshift32Generator& random_number_generator)
{
	ReSTIRPTSurface initial_surface;
	initial_surface.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	initial_surface.shading_normal	 = render_data.g_buffer.shading_normals[pixel_index].unpack();
	initial_surface.primitive_index	 = render_data.g_buffer.first_hit_prim_index[pixel_index];
	initial_surface.material		 = render_data.g_buffer.materials[pixel_index].unpack();
	initial_surface.ray_volume_state.reconstruct_first_hit(initial_surface.material, render_data.buffers.material_indices, initial_surface.primitive_index,
														   random_number_generator);
	initial_surface.shading_point  = render_data.g_buffer.primary_hit_position[pixel_index];
	initial_surface.view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

	float3_t to_light_direction = restir_pt_initial_reservoir_output->sample.is_envmap_path()
									  ? restir_pt_initial_reservoir_output->sample.rc_vertex
									  : hippt::normalize(restir_pt_initial_reservoir_output->sample.rc_vertex - initial_surface.shading_point);
	BSDFContext first_hit_eval_context(initial_surface.view_direction, initial_surface.shading_normal, initial_surface.geometric_normal, to_light_direction,
									   restir_pt_initial_reservoir_output->sample.incident_light_info_at_visible_point, initial_surface.ray_volume_state, false,
									   initial_surface.material, 0.0f);

	float trash_pdf;
	ColorRGB32F bsdf_first_hit = bsdf_dispatcher_eval(render_data, first_hit_eval_context, trash_pdf, random_number_generator) *
								 hippt::abs(hippt::dot(initial_surface.shading_normal, to_light_direction));

	ColorRGB32F radiance_to_camera;
	if (restir_pt_initial_reservoir_output->sample.is_envmap_path())
		radiance_to_camera = bsdf_first_hit * restir_pt_initial_reservoir_output->sample.rc_vertex_incident_radiance * restir_pt_initial_reservoir_output->UCW;
	else if (!restir_pt_initial_reservoir_output->sample.di_sample)
	{
		// TODO the ray volume state should be updated here
		float3_t view_direction					 = hippt::normalize(initial_surface.shading_point - restir_pt_initial_reservoir_output->sample.rc_vertex);
		float3_t to_light_direction_sample_point = restir_pt_initial_reservoir_output->sample.rc_vertex_incident_light_direction;

		int rc_vertex_material_index = render_data.buffers.material_indices[restir_pt_initial_reservoir_output->sample.rc_vertex_primitive_index];
		DeviceUnpackedPrincipledFullMaterial rc_vertex_material = get_intersection_material(
			render_data, rc_vertex_material_index,
			make_float2(restir_pt_initial_reservoir_output->sample.rc_vertex_texcoords_u, restir_pt_initial_reservoir_output->sample.rc_vertex_texcoords_v));
		float3_t rc_vertex_shading_normal	= restir_pt_initial_reservoir_output->sample.rc_vertex_shading_normal.unpack();
		float3_t rc_vertex_geometric_normal = restir_pt_initial_reservoir_output->sample.rc_vertex_geometric_normal.unpack();
		BSDFContext secondary_hit_eval_context(view_direction, rc_vertex_shading_normal, rc_vertex_geometric_normal, to_light_direction_sample_point,
											   restir_pt_initial_reservoir_output->sample.incident_light_info_at_sample_point, initial_surface.ray_volume_state,
											   false, rc_vertex_material, 0.0f);

		ColorRGB32F bsdf_secondary_hit =
			bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator) *
			hippt::abs(hippt::dot(restir_pt_initial_reservoir_output->sample.rc_vertex_shading_normal.unpack(), to_light_direction_sample_point));
		radiance_to_camera = bsdf_first_hit * bsdf_secondary_hit * restir_pt_initial_reservoir_output->sample.rc_vertex_incident_radiance *
							 restir_pt_initial_reservoir_output->UCW;
	}
	else
		radiance_to_camera = bsdf_first_hit * restir_pt_initial_reservoir_output->sample.rc_vertex_incident_radiance * restir_pt_initial_reservoir_output->UCW;

	render_data.buffers.accumulated_ray_colors[pixel_index] = radiance_to_camera;
}

/**
 * Normalizes the initial candidates reservoir of the pixel after its last path tree and writes the per-pixel outputs of the initial candidates
 */
HIPRT_DEVICE void ReSTIR_PT_wavefront_finalize_initial_candidates(HIPRTRenderData& render_data,
																  unsigned int pixel_index,
																  ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
																  Xorshift32Generator& random_number_generator)
{
	int pixel_y = static_cast<int>(pixel_index) / render_data.render_settings.render_resolution.x;
	int pixel_x = static_cast<int>(pixel_index) - pixel_y * render_data.render_settings.render_resolution.x;

	ReSTIRPTReservoir* restir_pt_initial_reservoir_output =
		&render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer[pixel_index];

	render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);

	// If we got here, this means that we still have at least one ray active
	// This is a concurrent write by the way but we don't really care, everyone is writing
	// the same value
	render_data.aux_buffers.still_one_ray_active[0] = 1;

	restir_pt_initial_reservoir.end_with_normalization(1.0f, ReSTIR_PT_InitialPathTreesCount);
	restir_pt_initial_reservoir_output->sanity_check(make_int2(pixel_x, pixel_y));

	if (ReSTIR_PT_MIS_target_function_bsdf_approximation_enabled(render_data))
	{
		// Final MIS target function BSDF approximation flags of the sample (the material part was written with the reconnection vertex). The flags depend only
		// on the reconnection data of the sample, they stay valid in every copy of the sample
		unsigned char mis_target_function_bsdf_approximation_flags = 0;
		if (restir_pt_initial_reservoir_output->UCW > 0.0f)
			mis_target_function_bsdf_approximation_flags =
				ReSTIR_PT_MIS_target_function_bsdf_approximation_sample_flags(restir_pt_initial_reservoir_output->sample);

		restir_pt_initial_reservoir_output->sample.mis_target_function_bsdf_approximation_flags = mis_target_function_bsdf_approximation_flags;
	}

#if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||                                                                          \
	ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE
	float3_t pixel_first_hit_point			  = render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t pixel_first_hit_geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	ReSTIR_PT_spmis_insert_pixel_hash(render_data, pixel_x, pixel_y, pixel_first_hit_point, pixel_first_hit_geometric_normal);
#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS || ReSTIR_PT_MISWeightsType ==
	   // RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE

	if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_SHADE_ONLY_INITIAL_CANDIDATES &&
		ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled)
		ReSTIR_PT_wavefront_store_shade_only_initial_candidates_debug_color(render_data, pixel_index, restir_pt_initial_reservoir_output,
																			random_number_generator);
}

/**
 * Called once the path tree 'path_tree_index' of the pixel has no vertex left to process: finalizes the initial candidates of the pixel after
 * its last path tree, keeps the state that the next path tree of the pixel continues from otherwise
 */
HIPRT_DEVICE void ReSTIR_PT_wavefront_end_path_tree(HIPRTRenderData& render_data,
													unsigned int pixel_index,
													unsigned int path_tree_index,
													ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
													Xorshift32Generator& random_number_generator)
{
	if (path_tree_index + 1 < ReSTIR_PT_InitialPathTreesCount)
	{
		render_data.wavefront_data.path_rng_states[pixel_index]						 = random_number_generator.m_state.seed;
		render_data.restir_pt_wavefront_data.path_reservoir_weight_sums[pixel_index] = restir_pt_initial_reservoir.weight_sum;

		return;
	}

	ReSTIR_PT_wavefront_finalize_initial_candidates(render_data, pixel_index, restir_pt_initial_reservoir, random_number_generator);
}

/**
 * Shades the vertex of the current bounce of the path of 'pixel_index' and queues the ray of its next bounce for the trace kernel.
 *
 * With 'initialize_primary_path', the vertex is the primary hit of the G-buffer and the path tree 'path_tree_index' of the pixel starts here.
 * Otherwise, the vertex is the hit found by the trace kernel for the ray that the previous vertex sampled
 */
template <bool initialize_primary_path>
HIPRT_DEVICE void ReSTIR_PT_wavefront_shade_path(HIPRTRenderData& render_data,
												 unsigned int bounce_count,
												 unsigned int path_tree_index,
												 unsigned int pixel_index)
{
	WavefrontDataDevice& wavefront_data					  = render_data.wavefront_data;
	ReSTIRPTWavefrontDataDevice& restir_pt_wavefront_data = render_data.restir_pt_wavefront_data;

	RayPayloadT<ReSTIRPTWavefrontKernelMaterial> ray_payload(NoInitTag{});
	hiprtRay ray;
	// At the primary hit, the texture coordinates of the hit are not read from the G-buffer and stay at their default value, as in the initial
	// candidates megakernel: the deferred light PDF of bounce 1 reads the material of the primary hit with them
	HitInfo closest_hit_info;

	ResolvedMaterialUserControlsCache resolved_user_controls;
	PrincipledMaterialClassificationInputs classification_inputs;
	SurfaceTransportMetadata surface_transport_metadata;

	ColorRGB32F path_unweighted_throughput_up_to_rc_vertex;
	ColorRGB32F path_weighted_throughput_after_rc_vertex;
	ColorRGB32F pending_throughput;

	ReSTIRPTInitialCandidatesReservoir restir_pt_initial_reservoir(render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer,
																   pixel_index);
	// The sample of the path lives in memory, not in registers, see the initial candidates megakernel
	ReSTIRPTReservoirSample& restir_pt_initial_sample =
		render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_working_samples_buffer[pixel_index];

	NEEDeferredMISContext nee_deferred_MIS_context;

	// The first path tree of the pixel starts from the seed of the pixel, the next path trees continue the random numbers of the previous one
	unsigned int random_seed;
	if constexpr (initialize_primary_path)
		random_seed = path_tree_index == 0 ? render_data.get_updated_random_seed(pixel_index) : wavefront_data.path_rng_states[pixel_index];
	else
		random_seed = wavefront_data.path_rng_states[pixel_index];
	Xorshift32Generator random_number_generator(random_seed);

	int x = static_cast<int>(pixel_index % render_data.render_settings.render_resolution.x);
	int y = static_cast<int>(pixel_index / render_data.render_settings.render_resolution.x);

	if constexpr (initialize_primary_path)
	{
		if (path_tree_index == 0)
		{
#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
			// Resetting splatting samples
			for (int bounce = 0; bounce < static_cast<int>(bounce_count); bounce++)
				render_data.render_settings.restir_pg_settings.invalidate_splatting_sample(render_data.render_settings.render_resolution, x, y, bounce);
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_TRUE

			restir_pt_initial_reservoir.weight_sum = 0.0f;
		}
		else
			restir_pt_initial_reservoir.weight_sum = restir_pt_wavefront_data.path_reservoir_weight_sums[pixel_index];

		static_cast<RayPayloadCommon&>(ray_payload) = RayPayloadCommon();

		closest_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[pixel_index];
		closest_hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		closest_hit_info.shading_normal	  = render_data.g_buffer.shading_normals[pixel_index].unpack();
		closest_hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[pixel_index];

		ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

		restir_pt_initial_sample			 = ReSTIRPTReservoirSample();
		restir_pt_initial_sample.pixel_index = pixel_index;

		path_unweighted_throughput_up_to_rc_vertex = ColorRGB32F(1.0f);
		path_weighted_throughput_after_rc_vertex   = ColorRGB32F(1.0f);
		pending_throughput						   = ColorRGB32F(1.0f);

		if (closest_hit_info.primitive_index == -1)
		{
			// The camera ray missed the scene: the path tree has no vertex and no candidate. The initial candidates megakernel still runs its last
			// deferred MIS for such a path, with a deferred MIS context that no vertex has filled: that candidate is not reproduced here, the
			// reservoir of the pixel stays empty. The pixels without a primary hit are not shaded by the passes that read the reservoirs
			ReSTIR_PT_wavefront_end_path_tree(render_data, pixel_index, path_tree_index, restir_pt_initial_reservoir, random_number_generator);

			return;
		}

		classification_inputs	   = load_material_classification_inputs(render_data.g_buffer.materials[pixel_index], resolved_user_controls);
		surface_transport_metadata = load_surface_transport_metadata(render_data.g_buffer.materials[pixel_index], classification_inputs);

		// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
		// the ray volume state after the camera hit is just an empty interior stack but with
		// the material index that we hit pushed onto the stack
		ray_payload.volume_state.reconstruct_first_hit(render_data.buffers.material_indices, closest_hit_info.primitive_index,
													   surface_transport_metadata.dielectric_priority, surface_transport_metadata.dispersion_scale,
													   surface_transport_metadata.specular_transmission, random_number_generator);

		store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, render_data.g_buffer.materials[pixel_index].get_base_color());

		wavefront_load_current_bsdf_material<true>(render_data, pixel_index, closest_hit_info, resolved_user_controls, ray_payload.material);
	}
	else
	{
		bool intersection_found;
		wavefront_load_secondary_material_state(render_data, pixel_index, ray_payload, closest_hit_info, intersection_found);
		wavefront_load_secondary_shading_state(render_data, pixel_index, ray_payload, ray, closest_hit_info);
		ray.direction = restir_pt_wavefront_data.path_ray_directions[pixel_index];

		restir_pt_initial_reservoir.weight_sum	   = restir_pt_wavefront_data.path_reservoir_weight_sums[pixel_index];
		path_unweighted_throughput_up_to_rc_vertex = restir_pt_wavefront_data.path_unweighted_throughputs_up_to_rc_vertex[pixel_index];
		path_weighted_throughput_after_rc_vertex   = restir_pt_wavefront_data.path_weighted_throughputs_after_rc_vertex[pixel_index];
		pending_throughput						   = restir_pt_wavefront_data.path_pending_throughputs[pixel_index];

		wavefront_load_nee_deferred_mis_context(render_data, pixel_index, nee_deferred_MIS_context);

		int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		bool classification_cache_loaded =
			wavefront_load_current_material_classification(render_data, pixel_index, resolved_user_controls, classification_inputs);
		if (!classification_cache_loaded)
			classification_inputs = load_material_classification_inputs(render_data, material_index, closest_hit_info.texcoords, resolved_user_controls);
		surface_transport_metadata = load_surface_transport_metadata(render_data, material_index, classification_inputs);

		wavefront_load_current_bsdf_material<false>(render_data, pixel_index, closest_hit_info, resolved_user_controls, ray_payload.material);
		// The initial candidates megakernel samples the wavelength of a dispersive hit when it traces the ray, before the deferred MIS below:
		// sampled here, before the deferred MIS too, so that the path consumes its random numbers in the same order
		wavefront_initialize_secondary_hit_material(ray_payload, surface_transport_metadata, random_number_generator);

		// The trace kernel only queues the hits for shading, misses go to the completion kernel
		ReSTIR_PT_wavefront_do_deferred_NEE_MIS(render_data, true, ray.direction, ray_payload.bounce, ray_payload.material.get_emission(),
												path_unweighted_throughput_up_to_rc_vertex, path_weighted_throughput_after_rc_vertex,
												restir_pt_initial_reservoir, restir_pt_initial_sample, closest_hit_info, nee_deferred_MIS_context,
												random_number_generator);

		// Hardcoded to bounce 2 for a simple reconnection shift at the first indirect vertex.
		// The pending factor belongs to bounce - 1, so the first two factors belong to the prefix.
		if (ray_payload.bounce > 2)
			path_weighted_throughput_after_rc_vertex *= pending_throughput;
		else
			path_unweighted_throughput_up_to_rc_vertex *= pending_throughput;
	}

	// A compile-time 0 for the primary hit: the per-bounce branches below are then removed from the primary shading kernels. As a runtime test,
	// 'bounce == 1' is a branch region whose join block hipRTC 7.2 can fill with VGPR spill stores emitted before the exec mask restore: the
	// values of the lanes outside the region are then not saved and are reloaded stale (seen faulting in the full material primary kernel)
	int bounce = initialize_primary_path ? 0 : ray_payload.bounce;
	if (!initialize_primary_path && bounce == 1)
		ReSTIR_PT_wavefront_rc_vertex_fill_information(render_data, ray_payload.material, closest_hit_info, restir_pt_initial_sample);

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	ReSTIR_PT_wavefront_stream_NEE(render_data, -ray.direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
								   path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample, closest_hit_info,
								   nee_deferred_MIS_context, random_number_generator, x, y);
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

	float bsdf_pdf;
	BSDFIncidentLightInfo incident_light_info;
	pending_throughput = ReSTIR_PT_wavefront_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction,
																		  surface_transport_metadata.dispersion_scale, ray, random_number_generator,
																		  incident_light_info, bsdf_pdf, nee_deferred_MIS_context);

	// Bad BSDF sample (under the surface), killed by russian roulette, ...: the ray is still traced by a terminal trace for the last deferred
	// MIS of the path, as the initial candidates megakernel traces it again for its last deferred MIS
	bool terminal_path = pending_throughput == ReSTIR_PT_invalid_throughput;
	if (!terminal_path)
	{
		// This factor goes into the suffix after the reconnection vertex (see the multiplication with the pending factor above)
		if (bounce >= 2)
			pending_throughput /= bsdf_pdf;

		if (bounce == 0)
			restir_pt_initial_sample.incident_light_info_at_visible_point = incident_light_info;
		else if (bounce == 1)
		{
			restir_pt_initial_sample.incident_light_info_at_sample_point = incident_light_info;
			// TODO remove this line
			restir_pt_initial_sample.rc_vertex_incident_light_direction = ray.direction;
			// Paths continuing past the reconnection vertex must store the BSDF * cosine of their continuation direction, not the value
			// left by the NEE sample of this bounce: the spatial-reuse neighbor selection target uses it in place of the reconnection
			// vertex throughput
			restir_pt_initial_sample.bsdf_throughput_luminance_at_sample_point = pending_throughput.luminance();
		}

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
		// Not the last bounce
		if (bounce != static_cast<int>(bounce_count))
		{
			ReSTIRPGSplattingSample sample;
			sample.position			  = closest_hit_info.inter_point;
			sample.normal			  = closest_hit_info.geometric_normal;
			sample.incident_direction = ray.direction;

			render_data.render_settings.restir_pg_settings.splatting_samples_soa.store_sample(sample, render_data.render_settings.render_resolution, x, y,
																							  bounce);
		}
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_TRUE

		// The ray of the last bounce is only traced for the last deferred MIS of the path
		terminal_path = bounce >= static_cast<int>(bounce_count);
	}

	ray_payload.bounce++;

	// The state of the random number generator when the ray of bounce 1 is traced
	if (bounce == 0 && !terminal_path)
		restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;

	restir_pt_wavefront_data.path_unweighted_throughputs_up_to_rc_vertex[pixel_index] = path_unweighted_throughput_up_to_rc_vertex;
	restir_pt_wavefront_data.path_weighted_throughputs_after_rc_vertex[pixel_index]	  = path_weighted_throughput_after_rc_vertex;
	restir_pt_wavefront_data.path_pending_throughputs[pixel_index]					  = pending_throughput;
	restir_pt_wavefront_data.path_reservoir_weight_sums[pixel_index]				  = restir_pt_initial_reservoir.weight_sum;
	restir_pt_wavefront_data.path_ray_directions[pixel_index]						  = ray.direction;

	wavefront_queue_path_for_tracing(render_data, pixel_index, ray_payload, ray, closest_hit_info, nee_deferred_MIS_context, random_number_generator,
									 terminal_path);
}

/**
 * Last deferred MIS of a path whose ray missed the scene or whose path ended at the previous vertex (terminal ray), then end of its path tree.
 *
 * The hit of a terminal ray was found by the trace kernel: the initial candidates megakernel traces that same ray again in its last deferred MIS
 */
HIPRT_DEVICE void ReSTIR_PT_wavefront_complete_path(HIPRTRenderData& render_data, unsigned int path_tree_index, unsigned int pixel_index)
{
	ReSTIRPTWavefrontDataDevice& restir_pt_wavefront_data = render_data.restir_pt_wavefront_data;

	RayPayloadCommon ray_payload(NoInitTag{});
	hiprtRay ray;
	HitInfo closest_hit_info;
	bool intersection_found;
	wavefront_load_secondary_material_state(render_data, pixel_index, ray_payload, closest_hit_info, intersection_found);
	wavefront_load_secondary_shading_state(render_data, pixel_index, ray_payload, ray, closest_hit_info);
	ray.direction = restir_pt_wavefront_data.path_ray_directions[pixel_index];

	Xorshift32Generator random_number_generator(render_data.wavefront_data.path_rng_states[pixel_index]);

	ReSTIRPTInitialCandidatesReservoir restir_pt_initial_reservoir(render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer,
																   pixel_index);
	restir_pt_initial_reservoir.weight_sum = restir_pt_wavefront_data.path_reservoir_weight_sums[pixel_index];
	ReSTIRPTReservoirSample& restir_pt_initial_sample =
		render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_working_samples_buffer[pixel_index];

	NEEDeferredMISContext nee_deferred_MIS_context;
	wavefront_load_nee_deferred_mis_context(render_data, pixel_index, nee_deferred_MIS_context);

	ColorRGB32F hit_emission = ColorRGB32F(0.0f);
	if (intersection_found)
	{
		int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];

		// The trace kernel does not classify the hits of the terminal rays: the classification cache does not hold this hit
		ResolvedMaterialUserControlsCache resolved_user_controls;
		PrincipledMaterialClassificationInputs classification_inputs =
			load_material_classification_inputs(render_data, material_index, closest_hit_info.texcoords, resolved_user_controls);
		SurfaceTransportMetadata surface_transport_metadata = load_surface_transport_metadata(render_data, material_index, classification_inputs);

		// The initial candidates megakernel samples the wavelength of a dispersive hit when it traces the ray again: sampled here too so that the
		// next path tree of the pixel continues from the same random numbers
		wavefront_initialize_secondary_hit_material(ray_payload, surface_transport_metadata, random_number_generator);

		EffectiveMaterialEmission current_hit_emission;
		load_effective_emission(render_data, material_index, closest_hit_info.texcoords, current_hit_emission);
		hit_emission = current_hit_emission.get_emission();
	}

	ReSTIR_PT_wavefront_do_deferred_NEE_MIS(render_data, intersection_found, ray.direction, ray_payload.bounce, hit_emission,
											restir_pt_wavefront_data.path_unweighted_throughputs_up_to_rc_vertex[pixel_index],
											restir_pt_wavefront_data.path_weighted_throughputs_after_rc_vertex[pixel_index], restir_pt_initial_reservoir,
											restir_pt_initial_sample, closest_hit_info, nee_deferred_MIS_context, random_number_generator);

	ReSTIR_PT_wavefront_end_path_tree(render_data, pixel_index, path_tree_index, restir_pt_initial_reservoir, random_number_generator);
}

#endif // #ifndef DEVICE_INCLUDES_RESTIR_PT_WAVEFRONT_WAVEFRONT_INITIAL_CANDIDATES_H
